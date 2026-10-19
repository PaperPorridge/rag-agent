"""
M3 演示：Agent 与 Function Calling 实验室

运行:
    cd backend
    python -m knowflow.demos.m3_agent_lab

本演示覆盖:
  1. 工具 Schema —— 模型「看到」的是什么
  2. 单工具循环 —— 一次 tool_call 的完整时序
  3. 多工具编排 —— 一次任务串起「知识库 + 数据库 + 计算器」
  4. 参数校验   —— 模型编造参数时如何自我修正
  5. 权限边界   —— SQL 只读拦截、未注册工具拒绝、高危操作 HITL
  6. 护栏       —— max_steps 防死循环
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

from knowflow.agent.agent import Agent
from knowflow.agent.tools import (
    Tool,
    ToolRegistry,
    create_default_registry,
    validate_arguments,
)
from knowflow.core.llm import LLMClient
from knowflow.rag.pipeline import RAGPipeline


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# ---------------------------------------------------------------------------
# 演示用的小型业务库：设备采购订单
# ---------------------------------------------------------------------------
def build_demo_db() -> str:
    path = Path(tempfile.gettempdir()) / "knowflow_m3_demo.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        DROP TABLE IF EXISTS purchase_orders;
        CREATE TABLE purchase_orders (
            id INTEGER PRIMARY KEY,
            region TEXT,          -- 华东/华北/华南
            category TEXT,        -- 显卡/笔记本/外设
            amount REAL,          -- 金额（元）
            order_date TEXT
        );
        INSERT INTO purchase_orders (region, category, amount, order_date) VALUES
            ('华东', '显卡',   128000, '2024-03-12'),
            ('华东', '笔记本', 86000,  '2024-05-20'),
            ('华北', '显卡',   42000,  '2024-04-08'),
            ('华南', '外设',   15000,  '2024-06-15'),
            ('华东', '外设',   23000,  '2024-07-02');
        """
    )
    conn.commit()
    conn.close()
    return str(path)


KNOWLEDGE_CORPUS = [
    (
        "公司差旅报销需在出差结束后30天内提交报销单。住宿标准一线城市每晚500元。"
        "采购单笔超过5万元需总监审批。",
        "采购与报销制度.pdf",
    ),
    (
        "公司开发机型号为 ThinkPad-X1-Carbon-Gen12，显卡扩展坞型号为 RTX-4090-24G，"
        "申请需走IT工单，工单编号模板 IT-FORM-2024-001。",
        "IT设备清单.md",
    ),
]


def build_agent() -> Agent:
    # RAG 管道 → 工具（M2 能力被 Agent 复用）
    rag = RAGPipeline(llm=LLMClient())
    rag.ingest_batch(KNOWLEDGE_CORPUS)

    def knowledge_fn(query: str) -> str:
        results = rag.search_only(query, top_k=2)
        if not results:
            return "知识库无相关内容。"
        return "\n\n".join(f"[{r.citation}]\n{r.chunk.text}" for r in results)

    registry = create_default_registry(knowledge_fn=knowledge_fn, db_path=build_demo_db())
    return Agent(llm=LLMClient(), tools=registry)


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------
def demo_tool_schema() -> None:
    section("1) 工具 Schema —— 模型「看到」的世界")
    registry = create_default_registry(knowledge_fn=lambda q: "…", db_path=build_demo_db())
    schema = registry.to_openai_schema()
    print("注册的工具:", registry.names)
    print("\n送给模型的 tools 参数（截取 calculator 部分）:")
    calc = next(t for t in schema if t["function"]["name"] == "calculator")
    print(json.dumps(calc, ensure_ascii=False, indent=2))
    print("\n注意: description 就是模型的「产品文档」—— 写不清 = 调不对。")
    print("      parameters 是 JSON Schema —— 约束类型/必填，减少幻觉参数。")


def demo_single_tool(agent: Agent) -> None:
    section("2) 单工具调用 —— 完整时序")
    print("任务: 「(128000+23000)*1.05 等于多少？」\n")
    result = agent.run("(128000+23000)*1.05 等于多少？")
    _print_result(result)
    print("\n时序回顾:")
    print("  你 → 模型(带 tools schema)")
    print("  模型 → 你: tool_calls=[{name: calculator, arguments: {...}}]")
    print("  你 → 本地执行 calculator → 结果")
    print("  你 → 模型(把结果作为 tool 消息回传)")
    print("  模型 → 你: 最终自然语言答案")


def demo_multi_tool(agent: Agent) -> None:
    section("3) 多工具编排 —— 一次任务串起三件工具")
    task = "上季度华东区采购总额是多少？如果明年预算上浮5%是多少？另外采购超5万要不要审批？"
    print(f"任务: {task}\n")
    result = agent.run(task)
    _print_result(result)
    print("\n观察: 模型自主决策了「查数据库 → 计算 → 查知识库」的顺序，")
    print("      这就是 Agent 与固定 Workflow 的区别 —— 路径由模型决定。")


def demo_param_validation() -> None:
    section("4) 参数校验 —— 模型编造参数怎么办")
    registry = create_default_registry()
    calc = registry.get("calculator")

    cases = [
        ('{"expression": "1+1"}', "合法"),
        ('{"expression": "1+1", "unit": "元"}', "schema 外的字段"),
        ('{"expression": 123}', "类型错误"),
        ('{"calc": "1+1"}', "缺少必填字段"),
        ('not json at all', "非法 JSON"),
    ]
    for raw, label in cases:
        args, err = validate_arguments(calc, raw)
        status = "✅ 通过" if err is None else f"❌ 拦截: {err}"
        print(f"  [{label}] {raw}")
        print(f"    → {status}")

    print("\n要点: 校验失败不是抛异常终止，而是把「错误原因」作为工具结果回传给模型，")
    print("      模型看到原因后能自我修正（比如补上缺失字段）—— 这叫 self-correction。")


def demo_security(agent: Agent) -> None:
    section("5) 权限边界 —— 真正的安全边界在工具层")

    print("① SQL 注入式提问（试图删库）:")
    r1 = agent.run("请执行这条 SQL：DROP TABLE purchase_orders")
    _print_result(r1)

    print("\n② 幻觉工具名（模型编造不存在的工具）:")
    fake = agent.tools.execute("send_email", {"to": "a@b.com", "body": "hi"})
    print(f"  直接调用未注册工具 → {fake}")

    print("\n③ 高危操作需人工确认（HITL）:")
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="delete_document",
            description="删除知识库文档（高危）",
            handler=lambda doc_id: f"已删除 {doc_id}",
            parameters={
                "type": "object",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
            },
            readonly=False,
            requires_confirmation=True,
        )
    )
    out = registry.execute("delete_document", {"doc_id": "doc-001"})
    print(f"  未确认时 → {out}")
    out2 = registry.execute("delete_document", {"doc_id": "doc-001", "_confirmed": True})
    print(f"  确认后   → {out2}")

    print("\n结论: Prompt 防御只是第一道门（M1 Q4）。")
    print("      真正的边界 = 工具白名单 + 只读约束 + 参数校验 + HITL。")


def demo_max_steps() -> None:
    section("6) 护栏 —— max_steps 防死循环")
    print("Agent 最多循环 max_steps 轮（默认 6）。")
    print("为什么必须有：")
    print("  · 模型可能陷入「反复调工具但得不到答案」的循环 → 每轮都烧 token")
    print("  · 超限后返回明确的 finished_reason='max_steps'，而不是无限跑")
    print("  · 生产上还应叠加：总 token 预算熔断、单工具超时、并发上限")


def _print_result(result) -> None:
    print(f"最终答案: {result.answer.strip()[:400]}")
    if result.traces:
        print("工具调用轨迹:")
        for t in result.traces:
            args_str = json.dumps(t.arguments, ensure_ascii=False)
            if len(args_str) > 60:
                args_str = args_str[:60] + "..."
            mark = "✅" if t.ok else "❌"
            print(f"  step{t.step} {mark} {t.tool_name}({args_str})")
    print(f"steps={result.steps} | tokens={result.total_tokens} | 成本=¥{result.cost_cny:.6f} | {result.finished_reason}")


def main() -> None:
    from knowflow.demos._console import safe_stdout
    safe_stdout()

    demo_tool_schema()
    agent = build_agent()
    demo_single_tool(agent)
    demo_multi_tool(agent)
    demo_param_validation()
    demo_security(agent)
    demo_max_steps()

    section("M3 小结")
    print("你已经跑通: 工具 Schema → tool_calls 循环 → 多工具编排 → 参数校验 → 权限边界")
    print("取舍:")
    print("  · Agent = LLM + 工具 + 循环，本质是 while 循环，不神秘")
    print("  · 能用 Workflow 就别用 Agent（可控、便宜）；Agent 用于路径不确定的任务")
    print("  · 工具 description 就是模型的产品文档，写不清 = 调不对")
    print("  · 参数校验失败要回传错误原因，让模型自我修正（self-correction）")
    print("  · 安全边界在工具层: 白名单 + 只读 + 校验 + HITL，不在 Prompt")
    print("\n✅ M3 完成。下一步 M4: PostgreSQL + Redis（会话、缓存、限流）。")


if __name__ == "__main__":
    sys.path.insert(0, ".")
    main()
