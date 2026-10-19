"""
M6 · 依赖注入（Dependency Injection）

【为什么要有这个文件？】
  如果路由函数内部直接写 `rag = RAGPipeline()`，有两个问题：
    1. 每次请求都新建一个 RAG（慢、浪费连接）
    2. 测试时换不掉 → 测试会真调 LLM（花钱、结果不确定、网络假失败）

  依赖注入的解法：路由只声明「我要一个 RAG」，不决定「给哪个 RAG」。
  测试时一行替换:
      app.dependency_overrides[get_rag] = lambda: FakeRAG()

【FastAPI 的 Depends 是什么？】
  `rag = Depends(get_rag)` 的意思：
    请求进来时，FastAPI 自动调用 get_rag()，把返回值塞进参数 rag。
  好处：
    - 自动管理生命周期（全局单例 / 每请求新建，你说了算）
    - 测试时可替换（dependency_overrides）
    - 权限/日志/限流横切逻辑统一放依赖里，不污染业务代码
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from knowflow.cache import RateLimiter
from knowflow.core.llm import LLMClient
from knowflow.queue import MessageQueue, create_queue
from knowflow.rag.pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# 全局单例缓存 —— 这些资源「建一次，全复用」
# ---------------------------------------------------------------------------
_rag_pipeline: RAGPipeline | None = None
_llm_client: LLMClient | None = None
_message_queue: MessageQueue | None = None
_rate_limiter: RateLimiter | None = None


def get_rag() -> RAGPipeline:
    """
    提供 RAGPipeline 单例。

    【为什么是单例？】
      RAGPipeline 内部持有向量库连接、embedding 模型、BM25 索引。
      每次请求新建一遍 = 每次都重建索引 = 又慢又浪费。
      进程启动时建一次，后续请求复用。

    【测试时怎么替换？】
      app.dependency_overrides[get_rag] = lambda: FakeRAGPipeline()
    """
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


def get_llm() -> LLMClient:
    """LLMClient 单例（成本追踪、重试策略挂在这里，必须复用）。"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def get_queue() -> MessageQueue:
    """消息队列单例（生产 Kafka / 开发 InMemoryQueue，由 create_queue 工厂决定）。"""
    global _message_queue
    if _message_queue is None:
        _message_queue = create_queue()
    return _message_queue


def get_rate_limiter() -> RateLimiter:
    """限流器单例（令牌桶，挂在依赖上，路由一行代码接入）。"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(per_minute=30)
    return _rate_limiter


# ---------------------------------------------------------------------------
# 类型别名 —— 路由签名更短更可读
# ---------------------------------------------------------------------------
RagDep = Annotated[RAGPipeline, Depends(get_rag)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]
QueueDep = Annotated[MessageQueue, Depends(get_queue)]
RateLimitDep = Annotated[RateLimiter, Depends(get_rate_limiter)]


def reset_singletons() -> None:
    """
    测试用：清空所有单例，保证测试间隔离。

    生产环境不要调用 —— 清掉连接池会导致后续请求变慢。
    """
    global _rag_pipeline, _llm_client, _message_queue, _rate_limiter
    _rag_pipeline = None
    _llm_client = None
    _message_queue = None
    _rate_limiter = None
