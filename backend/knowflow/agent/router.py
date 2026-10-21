"""
S4 · 问答分流器（Router）

【它解决什么问题？】
  同一个 /ask 入口要同时服务两类问题：
    1. 事实问答：「差旅住宿标准是多少」→ 直接 RAG，一次检索一次生成，快且便宜
    2. 执行类：  「按这个标准算 3 天报销上限」→ 需要计算器/查库的多步 Agent
  一律走 Agent 会把简单问题的延迟和成本放大好几倍；一律走 RAG 又算不了数。
  所以先分流，再执行。

【分流策略：LLM 优先，启发式兜底，默认直答】
  - LLM 结构化输出 {chain, reason}：语义判断准，但多一次调用（几十 token）
  - 关键词启发：LLM 失败时的降级，能抓住明显计算/查库意图
  - 两者皆败 → rag。宁可少路由也不误路由 —— 走错到 rag 的代价是「答不了计算」，
    走错到 agent 的代价是「每次都多花钱多等待」，前者更可接受。
"""

from __future__ import annotations

from typing import Literal

import structlog
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)


class RouteDecision(BaseModel):
    """Router 的结构化输出。"""

    chain: Literal["rag", "agent"] = Field(..., description="走哪条链路")
    reason: str = Field("", description="一句话分流理由")


# 启发式兜底：命中任一关键词 → agent。中英都收，因为问题语言不固定。
_AGENT_HINTS = (
    "计算", "算一下", "算出", "多少钱", "总计", "合计", "上限", "乘", "除", "百分",
    "统计", "查询数据库", "查一下库", "多少条", "多少个", "平均",
    "calculate", "compute", "sum", "total", "how much", "how many", "average",
)


def _keyword_fallback(question: str) -> RouteDecision:
    """LLM 不可用时的降级：只看明显计算/统计意图，其余一律直答。"""
    q = question.lower()
    hit = next((h for h in _AGENT_HINTS if h in q), None)
    if hit:
        return RouteDecision(chain="agent", reason=f"keyword:{hit}")
    return RouteDecision(chain="rag", reason="keyword:default")


def decide(llm, question: str, history: list[dict] | None = None) -> RouteDecision:
    """
    判定走 rag 还是 agent。永不抛异常 —— 最坏退回直答。

    llm  可传 None（测试/无网环境直接走关键词）
    """
    if llm is not None:
        try:
            # 只把最近几轮喂给 Router：分流只需要意图，不需要完整上下文
            recent = (history or [])[-4:]
            context = "\n".join(f"{t.get('role')}: {t.get('content')}" for t in recent)
            user_msg = (
                f"对话历史（可能为空）：\n{context or '（无）'}\n\n"
                f"当前问题：{question}\n\n"
                "判定该问题应走哪条链路：\n"
                "- rag：知识库事实问答、政策制度解释、找资料\n"
                "- agent：需要数学计算、查数据库、多步工具编排才能回答"
            )
            d = llm.chat_structured(
                user_msg,
                RouteDecision,
                system="你是问答系统的分流器，只输出 JSON。",
                max_tokens=200,
            )
            if d.chain not in ("rag", "agent"):
                raise ValueError(f"invalid chain: {d.chain}")
            log.info("router_decision", chain=d.chain, reason=d.reason, via="llm")
            return d
        except Exception as exc:
            log.warning("router_llm_failed", error=str(exc), action="keyword_fallback")

    d = _keyword_fallback(question)
    log.info("router_decision", chain=d.chain, reason=d.reason, via="keyword")
    return d
