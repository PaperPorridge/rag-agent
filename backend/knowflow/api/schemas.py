"""
M6 · 请求/响应模型（Pydantic）

【为什么单独放一个文件？】
  1. 这些模型就是「API 契约」——前后端都按这个格式来
  2. FastAPI 用它自动生成 OpenAPI 文档（/docs 页面的字段说明就是从这里来的）
  3. 改一处全生效：校验、序列化、文档三件事共享同一份类型声明

【Pydantic 的「免费三件事」】
  1. 校验:   类型不对/缺字段 → 自动返回 422 + 具体错误
  2. 序列化:   Python 对象 ↔ JSON 互转
  3. 文档:   字段名、类型、默认值 → 自动出现在 /docs
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 问答
# ---------------------------------------------------------------------------
class ChatTurn(BaseModel):
    """多轮对话里的一轮。role ∈ {user, assistant}，不做 system（那是服务端的事）。"""

    role: str = Field(..., pattern="^(user|assistant)$", description="发言角色")
    content: str = Field(..., min_length=1, max_length=8000, description="发言内容")


class AskRequest(BaseModel):
    """POST /ask 的请求体。"""

    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: int = Field(5, ge=1, le=20, description="检索返回的片段数")
    history: list[ChatTurn] = Field(default_factory=list, description="多轮对话历史")
    conversation_id: str | None = Field(None, description="会话 ID（前端本地用，服务端不持久化）")


class RetrievedSource(BaseModel):
    """一条被检索到的证据块。前端渲染「出处」列表。"""

    citation: str = Field(..., description="引用标记，如 差旅报销制度.pdf#0")
    source: str = Field(..., description="来源文件名")
    chunk_index: int = Field(..., description="块序号")
    snippet: str = Field(..., description="片段摘要（截断后的正文）")
    score: float = Field(..., description="融合得分（重排后优先，否则 RRF）")


class ToolCallTrace(BaseModel):
    """Agent 一次工具调用的轨迹。直答链路返回空数组。"""

    step: int = Field(..., description="第几步（1 起）")
    tool_name: str = Field(..., description="工具名")
    arguments: dict = Field(default_factory=dict, description="调用参数")
    result: str = Field("", description="工具返回（截断）")
    ok: bool = True
    error: str | None = None
    latency_ms: float = 0.0


class AskResponse(BaseModel):
    """POST /ask 的响应体。"""

    answer: str
    question: str
    chain: str = Field("rag", description="实际走的链路：rag = 直接问答，agent = 工具循环")
    router_reason: str = Field("", description="分流理由（便于排查路由误判）")
    sources: list[str] = Field(default_factory=list, description="来源文件名列表")
    retrieved: list[RetrievedSource] = Field(default_factory=list, description="检索到的证据块")
    tool_traces: list[ToolCallTrace] = Field(default_factory=list, description="Agent 工具轨迹")
    citations_valid: bool = True
    citation_errors: list[str] = Field(default_factory=list)
    n_retrieved: int = 0
    steps: int = Field(0, description="Agent 步数（直答为 0）")
    cost_cny: float = 0.0


# ---------------------------------------------------------------------------
# 文档（异步入库）
# ---------------------------------------------------------------------------
class DocumentUploadRequest(BaseModel):
    """POST /documents 的请求体（JSON 方式上传小文档）。"""

    source: str = Field(..., max_length=512, description="文件名或来源标识")
    text: str = Field(..., min_length=1, description="文档全文")
    owner_id: str | None = Field(None, description="上传者 ID")


class DocumentStatusResponse(BaseModel):
    """GET /documents/{doc_id} 的响应体，也是列表项。"""

    doc_id: str
    source: str
    status: str = Field(..., description="pending / processing / ready / failed")
    chunk_count: int = 0
    error: str | None = None
    created_at: str | None = Field(None, description="创建时间 ISO8601")


class DocumentUploadResponse(BaseModel):
    """POST /documents 返回 202 时的响应体。"""

    doc_id: str
    message_id: str
    status: str = "pending"
    detail: str = "文档已接收，正在异步入库"


class DocumentListResponse(BaseModel):
    """GET /documents 的响应体。"""

    items: list[DocumentStatusResponse] = Field(default_factory=list)
    total: int = 0


class DocumentDeleteResponse(BaseModel):
    """DELETE /documents/{doc_id} 的响应体。"""

    doc_id: str
    deleted: bool = True
    detail: str = "文档及其向量已删除"


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    components: dict[str, str] = Field(default_factory=dict)
