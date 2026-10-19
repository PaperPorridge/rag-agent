"""
M3 · Agent 核心 —— Function Calling 循环（ReAct 模式）

【Agent 是什么？】
  LLM + 工具 + 循环 = Agent。
  模型自己决定「要不要调工具、调哪个、参数是什么」，执行完把结果喂回去，
  直到它认为可以给出最终答案。这就是一个 while 循环 —— 不神秘。

【ReAct = Reason + Act（思考-行动-观察）】
  循环一轮:
    Thought  模型思考下一步该干什么（MiMo 的 reasoning_content 就是它）
    Action   模型产出 tool_calls（我要调 X 工具，参数是 Y）
    Observation 你执行工具，把结果作为 tool 消息回传
  重复直到模型不再要求调工具，输出最终答案。

【与 Workflow 的区别】
  Workflow: 固定流水线 A→B→C，路径由你写死，可控、便宜、可预测
  Agent   : 模型自己决定下一步，灵活、贵、不可完全预测
  → 能用 Workflow 就别用 Agent。Agent 用于「路径不确定」的任务（如多跳问答）。

【工程护栏（比「能跑」更重要）】
  1. max_steps      最多循环几轮 —— 防死循环烧钱
  2. 参数校验        validate_arguments 拦截非法参数，返回错误让模型自我修正
  3. 工具白名单      未注册的工具名直接拒绝（模型会幻觉出不存在的工具）
  4. 结果截断        工具输出超长会淹没上下文
  5. 全程 trace      每一步记下来 —— 排障 + 评估都要用
  6. MiMo 特有: 多轮工具调用需保留历史 reasoning_content，否则效果下降
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from knowflow.core.llm import LLMClient, LLMResponse
from knowflow.agent.tools import ToolRegistry, validate_arguments

log = structlog.get_logger(__name__)


@dataclass
class ToolCallTrace:
    """一次工具调用的完整记录 —— 排障和评估的依据。"""

    step: int
    tool_name: str
    arguments: dict[str, Any]
    result: str
    ok: bool
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class AgentResult:
    """Agent 运行产物。不只看最终答案，还要看它「怎么走到答案的」。"""

    answer: str
    traces: list[ToolCallTrace] = field(default_factory=list)
    steps: int = 0
    total_tokens: int = 0
    cost_cny: float = 0.0
    finished_reason: str = "completed"  # completed / max_steps / error

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "steps": self.steps,
            "tool_calls": [
                {"tool": t.tool_name, "args": t.arguments, "ok": t.ok} for t in self.traces
            ],
            "total_tokens": self.total_tokens,
            "cost_cny": self.cost_cny,
            "finished_reason": self.finished_reason,
        }


DEFAULT_SYSTEM_PROMPT = """你是 KnowFlow 企业智能助手，可以调用工具完成任务。

## 工作方式（ReAct）
1. 先判断：回答这个问题需要哪些信息/计算？需要外部工具吗？
2. 需要就调用工具，可以连续调用多个，也可以根据结果继续调用
3. 拿到足够信息后，给出最终答案

## 规则
- 只使用提供的工具，不要假装调用了不存在的工具
- 工具返回错误时，读懂错误原因，修正参数后重试，不要重复同样的错误
- 数据/制度类问题必须先检索知识库或查询数据库，**不要凭记忆回答**
- 计算必须用 calculator 工具，不要心算
- 最终答案用中文，条理清晰；涉及数据要注明来源
- 如果工具都查不到，明确说「根据现有工具无法回答」，不要编造
"""


class Agent:
    """
    Function Calling Agent。

    用法:
        registry = create_default_registry(knowledge_fn=..., db_path=...)
        agent = Agent(llm=LLMClient(), tools=registry)
        result = agent.run("上季度华东区设备采购总额是多少？")

    内部就是个 while 循环:
        while True:
            resp = llm.chat_with_tools(messages, tools_schema)
            if resp 没有 tool_calls: break        # 模型认为可以出最终答案了
            执行工具 → 结果追加到 messages → 继续下一轮
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        tools: ToolRegistry | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 6,
    ) -> None:
        self.llm = llm or LLMClient()
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    def run(self, task: str, history: list[dict[str, Any]] | None = None) -> AgentResult:
        """
        执行一个任务。

        task     用户任务描述
        history  可选的对话历史（多轮对话场景）
        """
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": task})

        tools_schema = self.tools.to_openai_schema()
        traces: list[ToolCallTrace] = []
        total_tokens = 0
        cost_cny = 0.0

        for step in range(1, self.max_steps + 1):
            resp = self.llm.chat_with_tools(messages, tools_schema)
            total_tokens += resp.usage.total_tokens
            cost_cny += resp.usage.cost_cny

            tool_calls = self._extract_tool_calls(resp)

            # -- 没有工具调用 = 模型认为可以出最终答案了 --
            if not tool_calls:
                return AgentResult(
                    answer=resp.content,
                    traces=traces,
                    steps=step,
                    total_tokens=total_tokens,
                    cost_cny=cost_cny,
                )

            # -- 有工具调用: 把 assistant 消息（含 tool_calls）加入历史 --
            # MiMo 思考模式下还要保留 reasoning_content，多轮工具调用效果更好
            messages.append(self._assistant_message(resp, tool_calls))

            # -- 逐个执行工具，结果作为 tool 消息回传 --
            for tc in tool_calls:
                name, args_raw, tc_id = self._parse_tool_call(tc)
                tool = self.tools.get(name)

                if tool is None:
                    # 模型会幻觉出不存在的工具名 —— 白名单外一律拒绝
                    result_text = f"[拒绝] 工具 {name} 不存在或未授权。可用: {self.tools.names}"
                    traces.append(ToolCallTrace(step=step, tool_name=name, arguments={}, result=result_text, ok=False, error="unregistered"))
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": result_text})
                    continue

                args, err = validate_arguments(tool, args_raw)
                if err is not None:
                    # 参数非法 → 返回错误原因，让模型自我修正（不要抛异常终止）
                    result_text = f"[参数错误] {err}"
                    traces.append(
                        ToolCallTrace(step=step, tool_name=name, arguments=args_raw if isinstance(args_raw, dict) else {}, result=result_text, ok=False, error=err)
                    )
                    log.warning("tool_args_invalid", step=step, tool=name, error=err)
                else:
                    t0 = time.perf_counter()
                    result_text = self.tools.execute(name, args)
                    latency = (time.perf_counter() - t0) * 1000
                    ok = not result_text.startswith("[工具执行失败]") and not result_text.startswith("[拒绝]") and not result_text.startswith("[需人工确认]")
                    traces.append(
                        ToolCallTrace(step=step, tool_name=name, arguments=args, result=result_text, ok=ok, latency_ms=latency)
                    )
                    log.info("tool_executed", step=step, tool=name, ok=ok, latency_ms=round(latency, 1))

                messages.append({"role": "tool", "tool_call_id": tc_id, "content": result_text})

        # -- 循环到上限还没出答案：强制终止 --
        log.warning("agent_max_steps", max_steps=self.max_steps)
        return AgentResult(
            answer=(
                f"任务在 {self.max_steps} 步内未完成，已停止。"
                "建议拆分为更小的任务，或检查工具返回是否总是失败。"
            ),
            traces=traces,
            steps=self.max_steps,
            total_tokens=total_tokens,
            cost_cny=cost_cny,
            finished_reason="max_steps",
        )

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_tool_calls(resp: LLMResponse) -> list[dict[str, Any]]:
        """从响应中取 tool_calls。SDK 结构: raw["choices"][0]["message"]["tool_calls"]。"""
        raw = resp.raw or {}
        choices = raw.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            calls = message.get("tool_calls") or []
            if calls:
                return list(calls)
        return []

    @staticmethod
    def _parse_tool_call(tc: dict[str, Any]) -> tuple[str, Any, str]:
        """解析单个 tool_call → (name, arguments_raw, id)。兼容 OpenAI 两种结构。"""
        fn = tc.get("function") or {}
        name = fn.get("name") or tc.get("name") or ""
        args_raw = fn.get("arguments", tc.get("arguments", "{}"))
        tc_id = tc.get("id") or f"call_{name}"
        return name, args_raw, tc_id

    @staticmethod
    def _assistant_message(resp: LLMResponse, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
        """
        构造 assistant 消息（含 tool_calls）。

        MiMo 注意事项：思考模式下要把 reasoning_content 一起保留进 messages，
        否则后续工具调用轮次的效果会下降（官方文档明确建议）。
        """
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": resp.content or "",
            "tool_calls": tool_calls,
        }
        if resp.reasoning_content:
            msg["reasoning_content"] = resp.reasoning_content
        return msg
