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
class AskRequest(BaseModel):
    """POST /ask 的请求体。"""

    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: int = Field(5, ge=1, le=20, description="检索返回的片段数")
    conversation_id: str | None = Field(None, description="会话 ID（多轮对话）")


class Source(BaseModel):
    """一个引用来源。"""

    citation: str = Field(..., description="引用标记，如 差旅报销制度.pdf#0")
    score: float = Field(..., description="相关性得分")


class AskResponse(BaseModel):
    """POST /ask 的响应体。"""

    answer: str
    question: str
    sources: list[str] = Field(default_factory=list)
    citations_valid: bool = True
    citation_errors: list[str] = Field(default_factory=list)
    n_retrieved: int = 0
    cost_cny: float = 0.0


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class AgentRequest(BaseModel):
    """POST /agent 的请求体。"""

    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    max_steps: int = Field(6, ge=1, le=10, description="最大工具调用步数")


class ToolCallTrace(BaseModel):
    """一次工具调用的轨迹（可观测性）。"""

    step: int
    tool_name: str
    args: dict = Field(default_factory=dict)
    ok: bool = True
    error: str | None = None


class AgentResponse(BaseModel):
    answer: str
    traces: list[ToolCallTrace] = Field(default_factory=list)
    steps: int = 0
    total_tokens: int = 0
    cost_cny: float = 0.0
    finished_reason: str = "done"


# ---------------------------------------------------------------------------
# 文档（异步入库）
# ---------------------------------------------------------------------------
class DocumentUploadRequest(BaseModel):
    """POST /documents 的请求体（JSON 方式上传小文档）。"""

    source: str = Field(..., max_length=512, description="文件名或来源标识")
    text: str = Field(..., min_length=1, description="文档全文")
    owner_id: str | None = Field(None, description="上传者 ID")


class DocumentStatusResponse(BaseModel):
    """GET /documents/{doc_id} 的响应体。"""

    doc_id: str
    source: str
    status: str = Field(..., description="pending / processing / ready / failed")
    chunk_count: int = 0
    error: str | None = None


class DocumentUploadResponse(BaseModel):
    """POST /documents 返回 202 时的响应体。"""

    doc_id: str
    message_id: str
    status: str = "pending"
    detail: str = "文档已接收，正在异步入库"


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    components: dict[str, str] = Field(default_factory=dict)
