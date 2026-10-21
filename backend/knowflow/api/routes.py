"""
M6 · 路由层（HTTP 接口）

【每个路由函数做什么？】
  1. 收请求（Pydantic 自动校验参数）
  2. 调业务层（RAGPipeline / Agent / Queue）
  3. 返回响应（Pydantic 自动序列化成 JSON）

【路由函数不应该做什么？】
  - 不写业务逻辑（那是 RAGPipeline/Agent 的活）
  - 不直接碰数据库/队列实现（走依赖注入）
  - 不吞异常（让 FastAPI 的异常处理统一返回 500/422）

【状态码约定】
  200  成功
  202  已接受，正在异步处理（文档入库）
  404  资源不存在
  415  文件类型不支持
  422  参数校验失败（Pydantic 自动 / 业务拒收）
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from knowflow.api.deps import QueueDep, RagDep
from knowflow.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentStatusResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
    HealthResponse,
    RetrievedSource,
    ToolCallTrace,
)
from knowflow.queue import TOPIC_DOC_INGEST, InMemoryQueue, enqueue_doc_ingest  # noqa: F401

# 路由分组：/api/v1 前缀，方便未来版本升级（v2 不影响 v1 的调用方）
router = APIRouter(prefix="/api/v1")


# ---------------------------------------------------------------------------
# 健康检查 —— 运维/监控/负载均衡都靠它
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse, tags=["运维"])
async def health() -> HealthResponse:
    """
    健康检查。K8s 的 liveness/readiness probe 打这个接口。

    返回各组件状态 —— 排障时一眼看出是哪个组件挂了。
    """
    components: dict[str, str] = {}

    # 数据库
    try:
        from sqlalchemy import text as _sql_text

        from knowflow.db.session import get_engine

        with get_engine().connect() as conn:
            conn.execute(_sql_text("SELECT 1"))
        components["database"] = "ok"
    except Exception as exc:
        components["database"] = f"error: {type(exc).__name__}"

    # 向量库：Qdrant 才有真连接；内存 store 直接视为 ok
    try:
        from knowflow.api.deps import get_rag

        store = get_rag().store
        ping = getattr(store, "ping", None)
        if ping is None or ping():
            components["vector"] = "ok"
        else:
            components["vector"] = "error: unreachable"
    except Exception as exc:
        components["vector"] = f"error: {type(exc).__name__}"

    # 队列：InMemory 不会挂；Kafka 记录后端类型便于排障
    from knowflow.api.deps import get_queue

    q = get_queue()
    components["queue"] = "memory" if isinstance(q, InMemoryQueue) else "kafka"

    status_str = "ok" if all(
        v == "ok" or v in ("memory", "kafka") for v in components.values()
    ) else "degraded"
    return HealthResponse(status=status_str, components=components)


# ---------------------------------------------------------------------------
# 问答（融合链路：Router 分流 → 直答 / Agent）
# ---------------------------------------------------------------------------
@router.post("/ask", response_model=AskResponse, tags=["问答"])
async def ask(req: AskRequest, rag: RagDep) -> AskResponse:
    """
    问答。同步接口 —— 结果就在响应体里（状态码 200）。

    【为什么是同步的？】
      问答必须给用户看到答案，没有"异步返回答案"的场景。
      而文档入库是异步的（202），因为入库要几十秒，用户不需要等。

    【chain 字段】
      "rag"   直接问答（检索 + 生成），tool_traces 恒为空
      "agent" 工具循环（search_knowledge_base / calculator），tool_traces 有轨迹
    """
    from knowflow.agent.chain import answer_with_routing

    history = [{"role": t.role, "content": t.content} for t in req.history]
    # 截断到最近 20 轮：超长历史只烧 token，对当前问题增益有限
    if len(history) > 20:
        history = history[-20:]

    result = answer_with_routing(rag, req.question, top_k=req.top_k, history=history)

    return AskResponse(
        answer=result.answer,
        question=result.question,
        chain=result.chain,
        router_reason=result.router_reason,
        sources=result.sources,
        retrieved=[RetrievedSource(**it) for it in result.retrieved],
        tool_traces=[
            ToolCallTrace(
                step=t.step,
                tool_name=t.tool_name,
                arguments=dict(t.arguments or {}),
                result=(t.result or "")[:600],
                ok=t.ok,
                error=t.error,
                latency_ms=float(t.latency_ms or 0.0),
            )
            for t in result.tool_traces
        ],
        citations_valid=result.citations_valid,
        citation_errors=result.citation_errors,
        n_retrieved=len(result.retrieved),
        steps=result.steps,
        cost_cny=result.cost_cny,
    )


# ---------------------------------------------------------------------------
# 文档：异步入库（202 + 轮询状态）
# ---------------------------------------------------------------------------
@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["文档"],
)
async def upload_document(
    req: DocumentUploadRequest,
    queue: QueueDep,
) -> DocumentUploadResponse:
    """
    粘贴文本入库 → 发消息到队列 → **立刻返回 202**。

    【为什么是 202 不是 200？】
      200 = 「完成了，结果在响应里」
      202 = 「收到了，正在处理，还没完成」
      入库要经过解析/分块/向量化，可能要几十秒（大 PDF）。
      同步做 HTTP 会超时。所以 API 只发消息就返回，Worker 后台慢慢做。
      前端拿 doc_id 轮询 GET /documents/{doc_id} 看 status 变成 ready。

    【为什么 key=doc_id？】
      Kafka 按 key 哈希分区 → 同一文档的两次更新永远在同一个分区 → 有序。
    """
    doc_id = str(uuid.uuid4())
    _insert_pending(doc_id, req.source, req.owner_id)

    message_id = await enqueue_doc_ingest(
        queue,
        doc_id=doc_id,
        source=req.source,
        text=req.text,
        owner_id=req.owner_id,
    )

    return DocumentUploadResponse(
        doc_id=doc_id,
        message_id=message_id,
        status="pending",
        detail="文档已接收，正在异步入库。请轮询 GET /documents/{doc_id} 查看进度。",
    )


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["文档"],
)
async def upload_file(
    queue: QueueDep,
    file: UploadFile = File(..., description="PDF / txt / md 文件"),
    owner_id: str | None = Form(None, description="上传者 ID"),
) -> DocumentUploadResponse:
    """
    文件上传入库（PDF / txt / md）。

    【解析为什么放 run_in_threadpool？】
      pypdf 抽文本层是 CPU 密集的纯 Python 循环。放事件循环里会卡住整个
      API 进程 —— 解析大 PDF 的几秒内所有其它请求都排队。
    """
    from knowflow.ingest.parser import ParseError, parse_upload

    data = await file.read()
    filename = file.filename or "unnamed.txt"

    try:
        parsed = await run_in_threadpool(parse_upload, filename, data)
    except ParseError as exc:
        # 业务拒收：类型不支持 415，其余（扫描版/过大/解析失败）422
        code = 415 if exc.code == "unsupported_type" else 422
        raise HTTPException(status_code=code, detail=str(exc)) from exc

    doc_id = str(uuid.uuid4())
    _insert_pending(doc_id, parsed.source, owner_id)

    message_id = await enqueue_doc_ingest(
        queue,
        doc_id=doc_id,
        source=parsed.source,
        text=parsed.text,
        owner_id=owner_id,
    )

    return DocumentUploadResponse(
        doc_id=doc_id,
        message_id=message_id,
        status="pending",
        detail=f"已解析 {parsed.format}（{len(parsed.text)} 字符），正在异步入库。",
    )


def _insert_pending(doc_id: str, source: str, owner_id: str | None) -> None:
    """落一条 pending 状态到 DB（让前端能立刻查到这个 doc_id）。"""
    try:
        from knowflow.db.models import Document
        from knowflow.db.session import session_scope

        with session_scope() as s:
            s.add(Document(
                id=doc_id,
                source=source,
                owner_id=owner_id,
                status="pending",
            ))
    except Exception:
        pass


def _doc_to_status(doc) -> DocumentStatusResponse:
    created = getattr(doc, "created_at", None)
    if isinstance(created, datetime):
        created = created.replace(tzinfo=timezone.utc).isoformat()
    elif created is not None:
        created = str(created)
    return DocumentStatusResponse(
        doc_id=doc.id,
        source=doc.source,
        status=doc.status,
        chunk_count=doc.chunk_count or 0,
        error=doc.error,
        created_at=created,
    )


@router.get("/documents", response_model=DocumentListResponse, tags=["文档"])
async def list_documents(limit: int = 100, offset: int = 0) -> DocumentListResponse:
    """文档列表（按创建时间倒序）。前端文档面板的主数据源。"""
    from sqlalchemy import func, select

    from knowflow.db.models import Document
    from knowflow.db.session import session_scope

    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    with session_scope() as s:
        total = s.execute(select(func.count()).select_from(Document)).scalar_one()
        rows = s.execute(
            select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
        ).scalars().all()
        return DocumentListResponse(
            items=[_doc_to_status(d) for d in rows],
            total=int(total),
        )


@router.get("/documents/{doc_id}", response_model=DocumentStatusResponse, tags=["文档"])
async def get_document(doc_id: str) -> DocumentStatusResponse:
    """
    查文档状态。前端轮询这个接口看进度。

    【状态机】pending → processing → ready / failed
    """
    from sqlalchemy import select

    from knowflow.db.models import Document
    from knowflow.db.session import session_scope

    with session_scope() as s:
        doc = s.execute(select(Document).where(Document.id == doc_id)).scalar_one_or_none()
        if doc is None:
            raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
        return _doc_to_status(doc)


@router.delete(
    "/documents/{doc_id}",
    response_model=DocumentDeleteResponse,
    tags=["文档"],
)
async def delete_document(doc_id: str, rag: RagDep) -> DocumentDeleteResponse:
    """
    删除文档：向量 + BM25 + DB 行一起清。

    【为什么顺序是先删向量再删行？】
      反过来的话，DB 行没了而向量还在 → 检索仍能命中幽灵文档，
      前端列表里却已经看不到它。宁可行还在、向量先没（下一次删幂等补齐）。
    """
    from sqlalchemy import select

    from knowflow.db.models import Document
    from knowflow.db.session import session_scope

    with session_scope() as s:
        doc = s.execute(select(Document).where(Document.id == doc_id)).scalar_one_or_none()
        if doc is None:
            raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
        source = doc.source

    try:
        rag.delete(source)
    except Exception:
        # 向量删除失败不阻断 DB 清理，但要留日志
        import structlog

        structlog.get_logger(__name__).warning(
            "vector_delete_failed", doc_id=doc_id, source=source
        )

    with session_scope() as s:
        doc = s.execute(select(Document).where(Document.id == doc_id)).scalar_one_or_none()
        if doc is not None:
            s.delete(doc)

    return DocumentDeleteResponse(
        doc_id=doc_id,
        deleted=True,
        detail="文档及其向量已删除",
    )


# ---------------------------------------------------------------------------
# 错误处理示例 —— 自定义异常 → 统一响应格式
# ---------------------------------------------------------------------------
class NotFoundError(HTTPException):
    """业务级 404。也可以用 @app.exception_handler 注册自定义异常。"""

    def __init__(self, resource: str, id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} 不存在: {id}",
        )
