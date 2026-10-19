"""
M4.5 对照实验室：手写 RAG/Agent vs LangChain

运行:
    cd backend
    python -m knowflow.demos.m45_langchain_lab

本演示覆盖:
  1. 生态现状   —— LangChain 1.x 的组件与「版本地狱」实证
  2. RAG 对照   —— 同一语料同一问题，手写链路 vs LCEL 链路
  3. Agent 对照 —— 同一工具集同一任务，手写 ReAct vs create_agent
  4. 能力对照   —— 手写版多做了什么
  5. 代码量对照 —— 「少写代码」和「多写代码」各自换来什么

设计原则: 两边用**同一个 embedder、同一个模型、同一份语料、同一个问题**，
差异只来自「框架封装」，这样对比才有意义。
"""

from __future__ import annotations

import inspect
import json
import time

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI

from knowflow.core.llm import LLMClient
from knowflow.rag.embeddings import HashingEmbedder
from knowflow.rag.pipeline import RAGPipeline
from knowflow.rag.retrieval import format_for_prompt


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# ---------------------------------------------------------------------------
# 0. 共享素材：语料 / 问题 / 工具
# ---------------------------------------------------------------------------
CORPUS: list[tuple[str, str]] = [
    (
        "员工差旅报销需在出差结束后30天内提交报销单，逾期需部门总监特批。"
        "住宿标准为一线城市每晚500元，二线城市每晚350元。",
        "差旅报销制度.pdf",
    ),
    (
        "费用核销流程：提交发票原件至财务部，财务在5个工作日内完成审核。"
        "单笔金额超过5000元需附上事前审批邮件截图。",
        "费用核销SOP.md",
    ),
    (
        "公司开发机型号为 ThinkPad-X1-Carbon-Gen12，显卡扩展坞型号为 RTX-4090-24G，"
        "申请需走IT工单，工单模板编号 IT-FORM-2024-001。",
        "IT设备清单.md",
    ),
    (
        "年假规则：入职满一年享有10天年假，满三年15天。"
        "年假需提前3个工作日在OA系统申请。",
        "休假管理办法.pdf",
    ),
]

QUESTION = "报销时限是多少天？住宿标准是什么？"


class LangChainEmbeddingsAdapter(Embeddings):
    """把我们自己的 embedder 适配到 LangChain 的 Embeddings 契约。

    这本身就是教学点：LangChain 的价值之一是**接口约定**——
    你实现它的协议，就能接入它的整个生态（vectorstore / retriever / chain）。
    """

    def __init__(self, provider: HashingEmbedder) -> None:
        self._p = provider

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._p.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._p.embed_query(text)


def build_chat_model(thinking: str = "disabled") -> ChatOpenAI:
    """构造指向 MiMo 的 LangChain ChatModel（OpenAI 兼容协议）。"""
    from knowflow.core.config import get_settings

    s = get_settings()
    return ChatOpenAI(
        model=s.mimo_model,
        base_url=s.mimo_base_url,
        api_key=s.mimo_api_key.get_secret_value(),
        timeout=s.mimo_timeout,
        extra_body={"thinking": {"type": thinking}},
    )


# ---------------------------------------------------------------------------
# 1. 生态现状
# ---------------------------------------------------------------------------
def demo_ecosystem() -> None:
    section("1. LangChain 1.x 生态现状 —— 顺带看一眼「版本地狱」")

    import langchain
    import langchain_core
    import langchain_openai

    print(f"  langchain       {langchain.__version__}")
    print(f"  langchain-core  {langchain_core.__version__}")
    print(f"  langchain-openai {langchain_openai.__version__}")

    print("\n  核心抽象（6 个词）:")
    print("    Model     模型封装 ChatOpenAI / ChatMiMo")
    print("    Prompt    ChatPromptTemplate（模板 + 变量）")
    print("    Retriever 「给我 top-k 相关文档」的统一接口")
    print("    Tool      模型能调用的函数")
    print("    Agent     模型自己决定调哪个工具的循环")
    print("    Chain     用 | 把上面串起来（LCEL）")

    print("\n  版本地狱实证（就在这个环境里发生）:")
    try:
        from langchain_community.retrievers import EnsembleRetriever  # noqa: F401

        print("    EnsembleRetriever: 可用")
    except Exception as exc:
        print(f"    EnsembleRetriever: ✗ 已从当前版本移除/迁移 —— {type(exc).__name__}")
        print("      → 0.3 之前在 langchain.retrievers，现在拆到独立包/改名")
        print("      → 这就是「升级框架 ≠ 免费」的活教材")

    print("\n  社区包 sunset 警告:")
    print("    langchain-community 已停止活跃维护，官方推独立集成包")
    print("    （langchain-qdrant / langchain-openai / langchain-postgres ...）")


# ---------------------------------------------------------------------------
# 2. RAG 对照
# ---------------------------------------------------------------------------
def demo_rag_compare() -> None:
    section("2. RAG 对照 —— 同一语料、同一问题、同一 embedder")

    embedder = HashingEmbedder(dim=256)
    lc_embeds = LangChainEmbeddingsAdapter(embedder)

    # --- 2A. 手写版 ---
    print("\n  ▶ 2A 手写版（knowflow.rag.pipeline.RAGPipeline）")
    t0 = time.perf_counter()
    pipeline = RAGPipeline(llm=LLMClient(), embedder=embedder, store_location=":memory:")
    for text, source in CORPUS:
        pipeline.ingest(text, source=source)
    hw_result = pipeline.ask(QUESTION, top_k=3, use_rerank=True)
    hw_ms = (time.perf_counter() - t0) * 1000

    print(f"     答案: {hw_result.answer[:200]}")
    print(f"     引用: {hw_result.sources}")
    print(f"     引用校验: {'通过' if hw_result.citations_valid else '失败'}"
          f"   ← 手写版独有")
    print(f"     耗时: {hw_ms:.0f} ms")

    # --- 2B. LangChain 版 ---
    print("\n  ▶ 2B LangChain 版（LCEL: retriever | prompt | llm | parser）")
    t0 = time.perf_counter()

    docs = [
        Document(page_content=text, metadata={"source": source})
        for text, source in CORPUS
    ]
    vs = InMemoryVectorStore.from_documents(docs, lc_embeds)
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs_: list[Document]) -> str:
        return "\n\n".join(f"[来源: {d.metadata['source']}]\n{d.page_content}" for d in docs_)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是企业知识助手。只依据提供的资料回答，并标注 [来源: ...]。"),
            ("human", "资料:\n{context}\n\n问题: {question}"),
        ]
    )
    llm = build_chat_model()
    chain = (
        {"context": retriever | format_docs, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )
    lc_answer = chain.invoke(QUESTION)
    lc_ms = (time.perf_counter() - t0) * 1000

    print(f"     答案: {lc_answer[:200]}")
    print("     引用: 未校验  ← LangChain 不管这个")
    print(f"     耗时: {lc_ms:.0f} ms")

    # --- 2C. 对照要点 ---
    print("\n  ▶ 2C 差异在哪")
    print("     | 维度          | 手写版               | LangChain 版")
    print("     | 分块/检索/重排  | 可逐行调（RRF k=60）  | 封装在 retriever 里")
    print("     | 引用校验       | validate_citations() | 无，要自己写")
    print("     | 成本追踪       | cost_cny 累计         | 要挂 Callback")
    print("     | 代码量         | 多                    | 少（LCEL 一行链）")
    print("     | 调试           | 堆栈清晰              | 抽象层多，要开 verbose")


# ---------------------------------------------------------------------------
# 3. Agent 对照
# ---------------------------------------------------------------------------
def demo_agent_compare() -> None:
    section("3. Agent 对照 —— 同一工具集、同一任务")

    TASK = "一台开发机 12800 元，按公司规定单笔超过 5000 元需要审批。超了多少？"

    # --- 3A. 手写版 ---
    print("\n  ▶ 3A 手写版（knowflow.agent.agent.Agent）")
    from knowflow.agent.agent import Agent
    from knowflow.agent.tools import create_default_registry

    registry = create_default_registry()
    agent = Agent(llm=LLMClient(), tools=registry, max_steps=6)
    t0 = time.perf_counter()
    hw = agent.run(TASK)
    hw_ms = (time.perf_counter() - t0) * 1000

    print(f"     答案: {hw.answer[:200]}")
    print(f"     步数: {hw.steps}   工具轨迹: {[t.tool_name for t in hw.traces]}")
    print(f"     成本: ¥{hw.cost_cny:.4f}   耗时: {hw_ms:.0f} ms   结束原因: {hw.finished_reason}")
    print("     安全: 工具 allowlist / 参数四重校验 / SQL 只读 / max_steps 护栏")

    # --- 3B. LangChain 版 ---
    print("\n  ▶ 3B LangChain 版（create_agent + @tool）")
    from langchain.agents import create_agent

    @tool
    def calculator(expression: str) -> str:
        """计算一个数学表达式，例如 12800 - 5000"""
        try:
            import ast
            import operator as op

            allowed = {
                ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
                ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg,
            }

            def _eval(node):
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    return node.value
                if isinstance(node, ast.BinOp) and type(node.op) in allowed:
                    return allowed[type(node.op)](_eval(node.left), _eval(node.right))
                if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
                    return allowed[type(node.op)](_eval(node.operand))
                raise ValueError(f"不允许的表达式: {ast.dump(node)}")

            return str(_eval(ast.parse(expression, mode="eval").body))
        except Exception as exc:
            return f"计算错误: {exc}"

    @tool
    def search_policy(query: str) -> str:
        """检索公司制度文档（报销、审批、设备）"""
        for text, source in CORPUS:
            if any(k in text for k in query if len(k) > 1) or query[:2] in text:
                return f"[{source}] {text}"
        return "未找到相关制度"

    agent_lc = create_agent(
        build_chat_model(),
        tools=[calculator, search_policy],
        system_prompt="你是企业助手。需要计算时调用 calculator，查制度时调用 search_policy。",
    )

    t0 = time.perf_counter()
    lc = agent_lc.invoke({"messages": [HumanMessage(content=TASK)]})
    lc_ms = (time.perf_counter() - t0) * 1000

    final = lc["messages"][-1].content
    tool_calls = []
    for m in lc["messages"]:
        tool_calls.extend(getattr(m, "tool_calls", []) or [])
    print(f"     答案: {str(final)[:200]}")
    print(f"     步数: {len(lc['messages'])} 条消息   工具调用: {[t['name'] for t in tool_calls]}")
    print(f"     耗时: {lc_ms:.0f} ms")
    print("     安全: @tool 直接暴露函数 —— 校验/只读/HITL 全要自己补")

    print("\n  ▶ 3C 差异在哪")
    print("     | 维度        | 手写版                | LangChain 版")
    print("     | 工具校验     | 四重校验 + 自修正      | 依赖 Pydantic args_schema")
    print("     | 安全边界     | allowlist/只读/HITL    | 自己写")
    print("     | max_steps    | 显式护栏              | create_agent 默认行为")
    print("     | reasoning    | 保留 MiMo reasoning   | 标准接口不透传")
    print("     | 可观测       | trace + cost_cny      | 挂 LangSmith / Callbacks")


# ---------------------------------------------------------------------------
# 4. 代码量与取舍
# ---------------------------------------------------------------------------
def demo_tradeoff() -> None:
    section("4. 取舍：少写代码 vs 多写代码")

    print("""
  ┌────────────────────────────────────────────────────────────────┐
  │  LangChain 给你的                                               │
  │    · LCEL 一行链串联 prompt/llm/parser                          │
  │    · 统一接口（Embeddings/Retriever/Tool）方便换实现              │
  │    · 生态：Loader / LangSmith 追踪 / 评估 / 向量库集成            │
  │    · 快速原型验证                                                │
  ├────────────────────────────────────────────────────────────────┤
  │  LangChain 收走的                                                │
  │    · 每一步的可见性（调试要开 verbose / 翻源码）                  │
  │    · 细粒度控制（RRF k、重排阈值、引用校验）                     │
  │    · 版本稳定（1.x 改名 / 包拆分 / community sunset）             │
  │    · 安全边界（工具校验、只读、HITL 要自己补）                    │
  └────────────────────────────────────────────────────────────────┘

  本项目的取舍：
    · 核心链路（分块 / 混合检索 / RRF / Rerank / 引用校验 / ReAct）→ 手写
      理由: 可控、可调试、能加 RRF 与引用校验这类框架不做的事
    · 周边集成（Loader / LangSmith 追踪 / 向量库适配）→ LangChain 生态
      理由: 不重复造轮子，且换实现只改接口不动业务
    · 对照版本（本 lab）→ 证明两者都用过，且清楚各自边界
""")


def main() -> None:
    print("=" * 64)
    print("M4.5 · 手写 vs LangChain 对照实验室")
    print("=" * 64)

    demo_ecosystem()
    demo_rag_compare()
    demo_agent_compare()
    demo_tradeoff()

    section("M4.5 演示完成")
    print("  结论: 「核心链路手写保证可控可调试，周边集成借 LangChain 生态，")
    print("            两者对照过，清楚框架在封装什么、什么时候该绕开。」")
    print("  下一步: 说「进 M5」进入 Kafka 文档异步入库流水线。")


if __name__ == "__main__":
    main()
