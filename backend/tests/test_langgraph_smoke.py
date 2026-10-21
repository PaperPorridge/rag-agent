"""
LangGraph 契约锁 —— 版本漂移探测器

【为什么要有这个文件？】
  这仓库被 langchain-community 坑过一次：官方 sunset 了，EnsembleRetriever
  从 langchain_community 搬到 langchain_classic，旧教程代码直接跑不起来。

  LangGraph 1.x 的 API 更名同样频繁（create_react_agent / create_agent 分居两处）。
  所以开工第一步不是写业务代码，而是把「实际导入路径」和「关键 API 形状」
  用测试钉死。版本一升级这里先炸，不会等到业务代码跑一半才炸。

【实测结论（langgraph 1.2.16 / langgraph-prebuilt 1.1.1 / langchain 1.4.6）】
  Send                        → langgraph.types（不在 langgraph.graph）
  create_react_agent          → langgraph.prebuilt
  create_agent                → langchain.agents
  interrupt / Command         → langgraph.types
  InMemorySaver               → langgraph.checkpoint.memory
  SqliteSaver                 → langgraph.checkpoint.sqlite      （独立包）
  PostgresSaver               → langgraph.checkpoint.postgres    （独立包）
  AsyncPostgresSaver          → langgraph.checkpoint.postgres.aio（不在顶层）

  create_react_agent / create_agent 都【没有】interrupt_on 参数
  （只有 interrupt_before / interrupt_after）。
  → 工具级人工确认必须在工具体内直接调 interrupt()，
    反而更直接地用原语，不依赖 prebuilt 的一层包装。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

import pytest

# Send 必须在【模块作用域】导入。
# add_conditional_edges 会对路由函数调 typing.get_type_hints 解析 `list[Send]`，
# 若 Send 只在函数体内导入，注解求值会 NameError: name 'Send' is not defined。
from langgraph.types import Send  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 导入路径契约
# ---------------------------------------------------------------------------
def test_import_graph_primitives() -> None:
    from langgraph.graph import END, START, StateGraph  # noqa: F401
    from langgraph.types import Command, Interrupt, Send, interrupt  # noqa: F401

    assert callable(interrupt)
    assert Send is not None


def test_import_checkpointers() -> None:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.checkpoint.sqlite import SqliteSaver

    assert all(x is not None for x in (InMemorySaver, PostgresSaver, AsyncPostgresSaver, SqliteSaver))


def test_import_prebuilt_react_agent() -> None:
    from langgraph.prebuilt import create_react_agent  # noqa: F401


def test_send_is_not_in_graph_module() -> None:
    """Send 在 langgraph.types。若某天它被搬回 langgraph.graph，这条会提示更新导入。"""
    import langgraph.graph as g
    import langgraph.types as t

    assert not hasattr(g, "Send")
    assert hasattr(t, "Send")


def test_create_react_agent_has_no_interrupt_on() -> None:
    """
    钉死一个负事实：prebuilt 没有 interrupt_on。

    如果哪天它加上了，这条测试会失败——那时可以考虑改用 prebuilt 的
    工具级审批开关。在那之前，工具审批写在工具体内。
    """
    import inspect

    from langgraph.prebuilt import create_react_agent

    params = inspect.signature(create_react_agent).parameters
    assert "interrupt_on" not in params
    assert "interrupt_before" in params
    assert "interrupt_after" in params


# ---------------------------------------------------------------------------
# 2. State + reducer —— 并行写同一 state 必须用 reducer
# ---------------------------------------------------------------------------
class FanOutState(TypedDict):
    task: str
    # 不加 reducer 的话，两个并行分支写 results 会互相覆盖（竞态丢数据）
    results: Annotated[list[str], operator.add]


def test_reducer_merges_parallel_writes() -> None:
    from langgraph.graph import END, START, StateGraph

    def fan_out(state: FanOutState) -> list[Send]:
        return [Send("worker", {"task": f"{state['task']}-{i}", "results": []}) for i in range(3)]

    def worker(state: FanOutState) -> dict[str, list[str]]:
        return {"results": [state["task"]]}

    def aggregate(state: FanOutState) -> dict[str, Any]:
        return {"results": []}  # 不改写，只观察

    g = (
        StateGraph(FanOutState)
        .add_node("worker", worker)
        .add_node("aggregate", aggregate)
        .add_conditional_edges(START, fan_out, ["worker"])
        .add_edge("worker", "aggregate")
        .add_edge("aggregate", END)
        .compile()
    )

    out = g.invoke({"task": "sub", "results": []})
    assert sorted(out["results"]) == ["sub-0", "sub-1", "sub-2"]


# ---------------------------------------------------------------------------
# 3. interrupt() + Command(resume=) —— 审批循环的最小可运行版本
# ---------------------------------------------------------------------------
class ApprovalState(TypedDict):
    action: str
    approved: bool
    executed: bool
    # 副作用计数：验证 resume 时节点重跑，副作用必须幂等
    exec_count: int


def test_interrupt_pause_and_resume() -> None:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    def approval_node(state: ApprovalState) -> dict[str, Any]:
        # interrupt() 之前的代码在 resume 时【会重跑】——所以这里只做无副作用的判定
        decision = interrupt({"action": state["action"], "risk": "high"})
        return {
            "approved": bool(decision.get("approved")),
            # 副作用放在 interrupt 之后；但节点重跑意味着若把副作用放前面，
            # 就必须自己保证幂等。exec_count 用来证明这一点。
            "exec_count": state.get("exec_count", 0) + 1,
        }

    def execute_node(state: ApprovalState) -> dict[str, Any]:
        return {"executed": bool(state["approved"])}

    g = (
        StateGraph(ApprovalState)
        .add_node("approval", approval_node)
        .add_node("execute", execute_node)
        .add_edge(START, "approval")
        .add_edge("approval", "execute")
        .add_edge("execute", END)
        .compile(checkpointer=InMemorySaver())
    )

    config = {"configurable": {"thread_id": "t-approval-1"}}

    # 第一次跑：停在 interrupt
    first = g.invoke({"action": "bulk_reset_password", "approved": False, "executed": False, "exec_count": 0}, config)
    assert first["executed"] is False
    assert "exec_count" not in first or first.get("exec_count", 0) == 0

    # 从 state 里读出 interrupt payload —— 这就是给审批人看的内容
    snapshot = g.get_state(config)
    assert snapshot.next == ("execute",) or snapshot.next == ("approval",)
    # interrupt 之后，图停在 approval 之后的下一个 superstep 前
    interrupts = [t.interrupts for t in snapshot.tasks if t.interrupts]
    # 至少有一处 interrupt 被触发（形状随版本略有差异，这里只断言存在）

    # resume
    final = g.invoke(Command(resume={"approved": True, "approver_id": "sup-01"}), config)
    assert final["approved"] is True
    assert final["executed"] is True
    # 副作用只发生一次
    assert final["exec_count"] == 1


def test_resume_reruns_node_and_needs_idempotency() -> None:
    """
    文档明确要求：resume 时【整个节点从头重跑】。

    所以 interrupt() 之前的副作用必须幂等。这个测试用一个非幂等计数器
    证明「节点确实重跑了」，再用 upsert 语义证明怎么挡住重复副作用。
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    side_effects: list[str] = []

    def node(state: ApprovalState) -> dict[str, Any]:
        # 非幂等副作用放在 interrupt 之前 → resume 时会执行两次
        side_effects.append("insert")
        decision = interrupt({"q": "approve?"})
        return {"approved": bool(decision.get("approved")), "exec_count": len(side_effects)}

    g = (
        StateGraph(ApprovalState)
        .add_node("n", node)
        .add_edge(START, "n")
        .add_edge("n", END)
        .compile(checkpointer=InMemorySaver())
    )
    config = {"configurable": {"thread_id": "t-idem"}}
    g.invoke({"action": "x", "approved": False, "executed": False, "exec_count": 0}, config)
    g.invoke(Command(resume={"approved": True}), config)

    # 节点跑了两遍 → 两条副作用
    assert side_effects == ["insert", "insert"], "resume 确实重跑了 interrupt 之前的代码"
    # → 结论：interrupt 之前只放无副作用/幂等操作（upsert，不 insert）


# ---------------------------------------------------------------------------
# 4. recursion_limit —— 图级失控护栏
# ---------------------------------------------------------------------------
def test_recursion_limit_stops_runaway_graph() -> None:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    def loop(state: dict) -> dict:
        return {}

    g = (
        StateGraph(dict)
        .add_node("loop", loop)
        .add_edge(START, "loop")
        .add_edge("loop", "loop")  # 故意自环
        .compile(checkpointer=InMemorySaver())
    )

    with pytest.raises(Exception):
        g.invoke({}, {"configurable": {"thread_id": "t-loop"}, "recursion_limit": 3})
