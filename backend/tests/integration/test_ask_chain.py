"""
S6 · 融合问答链路集成测试

【测什么？】
  1. 直答分流：事实问题 → chain=="rag" 且 tool_traces==[]
  2. Agent 分流：计算问题 → chain=="agent" 且轨迹非空、含工具名
  3. 证据聚合：纯计算 → retrieved==[] 且 citations_valid==True（不造假引用）

【为什么不 mock Router？】
  这三条守卫的是「整条融合链路」的契约。Router 是链路的一部分，
  mock 掉就等于没测。真调 LLM（成本几分钱）才能抓到结构化输出格式漂移。
  若 MIMO_API_KEY 缺失，Router 落关键词兜底，同样能测到分流逻辑。
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from knowflow.api.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def _ask(client: TestClient, question: str, **kw) -> dict:
    r = client.post("/api/v1/ask", json={"question": question, **kw})
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1. 直答分流
# ---------------------------------------------------------------------------
def test_factual_question_routes_to_rag(client: TestClient):
    """事实问答应走 rag，且不触发工具调用。"""
    body = _ask(client, "请介绍一下这个项目是做什么的？")
    assert body["chain"] == "rag", f"路由误判: {body.get('router_reason')}"
    assert body["tool_traces"] == []
    assert body["answer"], "answer 不能为空"
    assert body["steps"] == 0
    assert body["router_reason"], "应给出分流理由便于排查"


# ---------------------------------------------------------------------------
# 2. Agent 分流
# ---------------------------------------------------------------------------
def test_calculation_question_routes_to_agent_with_traces(client: TestClient):
    """计算题应走 agent，且轨迹里能看到 calculator。"""
    body = _ask(
        client,
        "按每天 100 元的标准，出差 3 天的餐饮补贴总计是多少？请计算。",
    )
    assert body["chain"] == "agent", f"路由误判: {body.get('router_reason')}"
    assert len(body["tool_traces"]) > 0, "Agent 链路必须有工具轨迹"
    names = {t["tool_name"] for t in body["tool_traces"]}
    assert "calculator" in names, f"轨迹里没看到 calculator，实际: {names}"
    for t in body["tool_traces"]:
        assert set(t) >= {"step", "tool_name", "arguments", "ok"}
        assert t["step"] >= 1
    assert body["steps"] >= 1


# ---------------------------------------------------------------------------
# 3. 证据聚合：纯计算不造假引用
# ---------------------------------------------------------------------------
def test_pure_calculation_has_no_fake_citations(client: TestClient):
    """纯计算题不查知识库 → retrieved 必须为空，citations_valid 为真。"""
    body = _ask(client, "请计算 (1280 + 500) * 1.13 等于多少？")
    assert body["chain"] == "agent", f"路由误判: {body.get('router_reason')}"
    assert body["retrieved"] == [], f"纯计算不应有检索证据: {body['retrieved']}"
    assert body["citations_valid"] is True
    assert body["citation_errors"] == []
