"""
S4 · 融合问答链路（/ask 单入口背后的调度）

【为什么把 Agent 融进 /ask 而不是单独开 /agent？】
  对用户来说「提问」只有一件事。分两个入口会把「该走哪条链」的判断推给前端，
  而前端根本不了解问题语义。单入口 + 后端分流，契约不变、体验统一。
  Agent 的能力经 chain="agent" 暴露，工具轨迹随响应返回。

【证据聚合是这里最容易做错的一环】
  Agent 途中可能调 search_knowledge_base 若干次，命中的块**必须**收进证据集，
  最后对整段答案做 validate_citations —— 否则模型编个 [来源: xxx] 就成幻觉引用。
  反过来：全程没查知识库（纯计算题），retrieved 就该是**空**，
  并把 citations_valid 置真 —— 没有证据可校验时不要假装校验过。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from knowflow.agent.router import RouteDecision, decide

log = structlog.get_logger(__name__)

_SNIPPET_LEN = 220
_RESULT_TRUNCATE = 600


@dataclass
class ChainResult:
    """融合链路的统一返回。API 层负责再映射成 Pydantic 契约。"""

    answer: str
    question: str
    chain: str
    router_reason: str = ""
    sources: list[str] = field(default_factory=list)
    retrieved: list[Any] = field(default_factory=list)  # list[RetrievedChunk]
    tool_traces: list[Any] = field(default_factory=list)  # list[ToolCallTrace]
    citations_valid: bool = True
    citation_errors: list[str] = field(default_factory=list)
    steps: int = 0
    cost_cny: float = 0.0


def _snippet(text: str, n: int = _SNIPPET_LEN) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _to_retrieved_items(retrieved: list[Any]) -> list[dict[str, Any]]:
    """RetrievedChunk → API 的 RetrievedSource 字典。"""
    items = []
    for r in retrieved:
        c = r.chunk
        score = r.rerank_score if r.rerank_score is not None else r.rrf_score
        items.append({
            "citation": f"{c.source}#{c.chunk_index}",
            "source": c.source,
            "chunk_index": c.chunk_index,
            "snippet": _snippet(c.text),
            "score": float(score or 0.0),
        })
    return items


def _knowledge_fn(rag, collected: list[Any]):
    """
    把 RAGPipeline.search_only 包一层：返回给 Agent 可读文本，
    同时把命中的块攒进 collected —— 这就是「证据聚合」的采集点。
    """
    def _search(query: str) -> str:
        try:
            hits = rag.search_only(query, top_k=4)
        except Exception as exc:
            log.warning("agent_knowledge_search_failed", error=str(exc))
            return f"[知识库查询失败: {exc}]"
        collected.extend(hits)
        if not hits:
            return "知识库中没有找到相关资料。"
        return "\n\n".join(
            f"[来源: {h.chunk.source}#{h.chunk.chunk_index}] {_snippet(h.chunk.text, 400)}"
            for h in hits
        )
    return _search


def answer_with_routing(
    rag,
    question: str,
    *,
    top_k: int = 5,
    history: list[dict[str, Any]] | None = None,
) -> ChainResult:
    """
    /ask 的统一执行入口。rag 是 deps.get_rag() 单例（Worker/BM25 同一实例）。
    """
    from knowflow.agent.agent import Agent
    from knowflow.agent.tools import create_default_registry
    from knowflow.rag.pipeline import validate_citations

    decision: RouteDecision = decide(rag.llm, question, history)

    # ---- 直答：一次检索一次生成 ----
    if decision.chain == "rag":
        ans = rag.ask(question, top_k=top_k, history=history)
        retrieved = list(ans.retrieved)
        items = _to_retrieved_items(retrieved)
        return ChainResult(
            answer=ans.answer,
            question=question,
            chain="rag",
            router_reason=decision.reason,
            sources=list(ans.sources),
            retrieved=items,
            tool_traces=[],
            citations_valid=ans.citations_valid,
            citation_errors=list(ans.citation_errors),
            steps=0,
            cost_cny=ans.llm.usage.cost_cny if ans.llm else 0.0,
        )

    # ---- Agent：多步工具循环 ----
    collected: list[Any] = []
    registry = create_default_registry(knowledge_fn=_knowledge_fn(rag, collected))
    # 只注册只读工具（calculator + search_knowledge_base）。
    # query_database 不进 web 路径：它接的是 SQLite 开发库，生产 PG 的只读 SQL
    # 工具留作后续切片 —— 见计划「诚实边界」。

    agent = Agent(llm=rag.llm, tools=registry)
    result = agent.run(task=question, history=history)

    # 证据去重（同块可能被多次检索命中）
    seen: set[tuple[str, int]] = set()
    unique = []
    for h in collected:
        key = (h.chunk.source, h.chunk.chunk_index)
        if key in seen:
            continue
        seen.add(key)
        unique.append(h)
    items = _to_retrieved_items(unique)

    if items:
        valid_sources = [it["citation"] for it in items]
        ok, errors = validate_citations(result.answer, valid_sources)
    else:
        # 纯计算/工具题：没检索就没有可校验的引用，不制造假失败
        ok, errors = True, []

    return ChainResult(
        answer=result.answer,
        question=question,
        chain="agent",
        router_reason=decision.reason,
        sources=[it["source"] for it in items],
        retrieved=items,
        tool_traces=result.traces,
        citations_valid=ok,
        citation_errors=errors,
        steps=result.steps,
        cost_cny=result.cost_cny,
    )
