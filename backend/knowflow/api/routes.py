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
  201  创建成功
  202  已接受，正在异步处理（文档入库）
  404  资源不存在
  422  参数校验失败（Pydantic 自动）
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, status

from knowflow.api.deps import QueueDep, RagDep
from knowflow.api.schemas import (
    AgentRequest,
    AgentResponse,
    AskRequest,
    AskResponse,
    DocumentStatusResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
    HealthResponse,
    ToolCallTrace,
)
from knowflow.queue import InMemoryQueue, TOPIC_DOC_INGEST, enqueue_doc_ingest

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
    components = {}
    try:
        from knowflow.db.session import get_engine
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        components["database"] = "ok"
    except Exception as exc:
        components["database"] = f"error: {type(exc).__name__}"

    components["queue"] = "ok"  # InMemoryQueue 不会挂；Kafka 的话在这里 ping

    status_str = "ok" if all(v == "ok" for v in components.values()) else "degraded"
    return HealthResponse(status=status_str, components=components)


# ---------------------------------------------------------------------------
# RAG 问答
# ---------------------------------------------------------------------------
@router.post("/ask", response_model=AskResponse, tags=["问答"])
async def ask(req: AskRequest, rag: RagDep) -> AskResponse:
    """
    RAG 问答。同步接口 —— 结果就在响应体里（状态码 200）。

    【为什么是同步的？】
      问答必须给用户看到答案，没有"异步返回答案"的场景。
      而文档入库是异步的（202），因为入库要几十秒，用户不需要等。
    """
    result = rag.ask(req.question, top_k=req.top_k)
    return AskResponse(
        answer=result.answer,
        question=result.question,
        sources=result.sources,
        citations_valid=result.citations_valid,
        citation_errors=result.citation_errors,
        n_retrieved=len(result.retrieved),
        cost_cny=result.llm.usage.cost_cny if result.llm else 0.0,
    )


# ---------------------------------------------------------------------------
# Agent 多工具编排
# ---------------------------------------------------------------------------
@router.post("/agent", response_model=AgentResponse, tags=["Agent"])
async def agent(req: AgentRequest) -> AgentResponse:
    """
    Agent 多工具编排（知识库检索 / Text2SQL / 计算器）。

    返回 traces（工具调用轨迹）—— 前端可以渲染"思考过程"。
    """
    # 延迟导入 Agent：它依赖 LLM + 工具链，只有用到才加载
    from knowflow.agent.agent import Agent

    a = Agent()
    result = a.run(req.question, max_steps=req.max_steps)

    return AgentResponse(
        answer=result.answer,
        traces=[
            ToolCallTrace(
                step=i + 1,
                tool_name=t.tool_name,
                args=t.args,
                ok=t.ok,
                error=t.error,
            )
            for i, t in enumerate(result.traces)
        ],
        steps=result.steps,
        total_tokens=result.total_tokens,
        cost_cny=result.cost_cny,
        finished_reason=result.finished_reason,
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
    上传文档 → 发消息到队列 → **立刻返回 202**。

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

    # 落一条 pending 状态到 DB（让前端能立刻查到这个 doc_id）
    try:
        from knowflow.db.models import Document
        from knowflow.db.session import session_scope

        with session_scope() as s:
            s.add(Document(
                id=doc_id,
                source=req.source,
                owner_id=req.owner_id,
                status="pending",
            ))
    except Exception:
        # DB 没建表也能跑（纯队列模式），不阻塞主流程
        pass

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


@router.get("/documents/{doc_id}", response_model=DocumentStatusResponse, tags=["文档"])
async def get_document(doc_id: str) -> DocumentStatusResponse:
    """
    查文档状态。前端轮询这个接口看进度。

    【状态机】pending → processing → ready / failed
    """
    try:
        from sqlalchemy import select

        from knowflow.db.models import Document
        from knowflow.db.session import session_scope

        with session_scope() as s:
            doc = s.execute(select(Document).where(Document.id == doc_id)).scalar_one_or_none()
            if doc is None:
                raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
            return DocumentStatusResponse(
                doc_id=doc.id,
                source=doc.source,
                status=doc.status,
                chunk_count=doc.chunk_count,
                error=doc.error,
            )
    except HTTPException:
        raise
    except Exception:
        # DB 不可用时返回 pending（容错，不 500）
        return DocumentStatusResponse(doc_id=doc_id, source="", status="pending")


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
