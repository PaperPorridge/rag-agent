"""
M0 演示：工程化调用 LLM

运行前:
    cd backend
    pip install -e ".[dev]"
    cp ../.env.example ../.env   # 填入 MIMO_API_KEY

运行:
    python -m knowflow.demos.m0_hello_llm

本演示覆盖:
  1. 配置校验 —— 缺 API Key 启动即报错
  2. 带重试/日志/成本统计的对话调用
  3. 流式输出
  4. 结构化输出（Pydantic 校验，模型输出不可信任）
"""

from __future__ import annotations

import sys

from pydantic import BaseModel, Field

from knowflow.core.llm import LLMClient
from knowflow.core.prompts import registry


class TicketSummary(BaseModel):
    """结构化输出示例：从客服对话中抽取工单要素。"""

    category: str = Field(description="问题类别: 网络/账务/账号/其他")
    severity: str = Field(description="严重程度: P0/P1/P2/P3")
    need_human: bool = Field(description="是否需要人工介入")
    summary: str = Field(description="一句话摘要，不超过 30 字")


def main() -> None:
    from knowflow.demos._console import safe_stdout
    safe_stdout()

    llm = LLMClient()

    # ---------- 1. 基础对话（使用注册表里的模板，而不是硬编码字符串）----------
    print("=" * 60)
    print("1) 模板化对话")
    print("=" * 60)
    question = "什么是 RAG？用两句话说明。"
    resp = llm.chat(question, system=registry.render("assistant", question=question))
    print(f"\n回答:\n{resp.content}")
    print(f"\n用量: {resp.usage.total_tokens} tokens | 成本: ¥{resp.usage.cost_cny:.6f}")
    print(f"思考模式返回 reasoning_content: {'有' if resp.reasoning_content else '无'}")

    # ---------- 2. 流式输出 ----------
    print("\n" + "=" * 60)
    print("2) 流式输出（逐段打印）")
    print("=" * 60)
    for piece in llm.chat_stream("用一句话解释什么是向量数据库。"):
        print(piece, end="", flush=True)
    print()

    # ---------- 3. 结构化输出：模型输出必须过 Pydantic 校验 ----------
    print("\n" + "=" * 60)
    print("3) 结构化输出（Pydantic 校验）")
    print("=" * 60)
    raw_ticket = "用户反映家里宽带今天早上开始断断续续，重启光猫也没用，明天要开视频会议，很着急。"
    try:
        ticket = llm.chat_structured(
            raw_ticket,
            TicketSummary,
            system="你是客服工单分类助手。",
        )
        print(f"  类别   : {ticket.category}")
        print(f"  严重度 : {ticket.severity}")
        print(f"  转人工 : {ticket.need_human}")
        print(f"  摘要   : {ticket.summary}")
    except ValueError as exc:
        print(f"  [结构化输出失败 —— 这正是为什么必须做校验] {exc}")

    # ---------- 4. 成本汇总 ----------
    print("\n" + "=" * 60)
    print("4) 成本汇总（生产中写入监控）")
    print("=" * 60)
    print(llm.tracker.summary())

    print("\n✅ M0 完成。你已经有了: 强类型配置 / 重试 / 结构化日志 / 成本追踪 / 结构化输出。")
    print("下一步 M1: 把提示词当代码管理 —— 见 knowflow/core/prompts.py")


if __name__ == "__main__":
    # 让脚本在 `python -m knowflow.demos.m0_hello_llm` 与直接运行时都能找到包
    sys.path.insert(0, ".")
    main()
