"""
M0 · LLM 客户端封装（工程化调用）

【为什么要封装一层，而不是 openai.OpenAI() 到处 new？】

裸调用的四个生产事故：
  1. 网络抖动 → 直接抛异常给用户。需要：重试 + 指数退避
  2. 账单爆炸 → 没人知道花了多少钱。需要：每次调用记录 token 与成本
  3. 线上排障 → 不知道哪次请求慢/失败。需要：结构化日志 + request_id 贯穿
  4. 换模型 → 改几十处代码。需要：统一入口，配置里改一行

本模块提供 LLMClient，后续 RAG / Agent 全部经由它调用模型。

【MiMo 2.6 要点】
- OpenAI Chat Completions 兼容，直接用 openai SDK
- 思考模式经 extra_body 传入: {"thinking": {"type": "enabled"|"disabled"}}
- 思考开启时 temperature/top_p 会被服务端强制为 1.0/0.95，不要浪费力气调参
- 多轮工具调用时，需在 messages 中保留历史 reasoning_content
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

import structlog
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from knowflow.core.config import get_settings

log = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# 成本追踪
# ---------------------------------------------------------------------------
# MiMo 定价（人民币 / 百万 token）—— 用于估算，非账单依据
# 见 https://mimo.mi.com 价格页。缓存命中输入价格极低，这里保守按未命中计。
PRICING_CNY_PER_MTOK: dict[str, dict[str, float]] = {
    "mimo-v2.6-pro": {"input": 3.0, "output": 6.0},
    "mimo-v2.6-flash": {"input": 1.0, "output": 2.0},
    "mimo-v2.6-pro-ultraspeed": {"input": 30.0, "output": 60.0},
}
DEFAULT_PRICING = {"input": 3.0, "output": 6.0}


@dataclass
class UsageRecord:
    """一次调用的用量与成本。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_cny: float = 0.0
    model: str = ""
    latency_ms: float = 0.0
    request_id: str = ""


@dataclass
class CostTracker:
    """进程级成本累计器。生产中应写入 Redis/DB，这里先做内存版。"""

    total_cost_cny: float = 0.0
    total_tokens: int = 0
    call_count: int = 0
    records: list[UsageRecord] = field(default_factory=list)

    def add(self, record: UsageRecord) -> None:
        self.total_cost_cny += record.cost_cny
        self.total_tokens += record.total_tokens
        self.call_count += 1
        self.records.append(record)

    def summary(self) -> dict[str, Any]:
        return {
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "total_cost_cny": round(self.total_cost_cny, 6),
            "avg_latency_ms": (
                round(sum(r.latency_ms for r in self.records) / len(self.records), 1)
                if self.records
                else 0.0
            ),
        }


def compute_cost_cny(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """按 token 用量估算成本（人民币）。"""
    price = PRICING_CNY_PER_MTOK.get(model, DEFAULT_PRICING)
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# 结构化输出
# ---------------------------------------------------------------------------
class LLMResponse(BaseModel):
    """统一响应结构。不裸返回 SDK 对象 —— 避免业务代码耦合 openai 类型。"""

    content: str
    model: str
    usage: UsageRecord
    raw: dict[str, Any] = Field(default_factory=dict)
    reasoning_content: str | None = None  # MiMo 思考链，多轮工具调用需回传


# ---------------------------------------------------------------------------
# 主客户端
# ---------------------------------------------------------------------------
class LLMClient:
    """
    统一 LLM 入口。

    用法:
        llm = LLMClient()
        resp = llm.chat("你好")
        print(resp.content, resp.usage.cost_cny)
    """

    def __init__(self, tracker: CostTracker | None = None) -> None:
        settings = get_settings()
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.mimo_api_key.get_secret_value(),
            base_url=settings.mimo_base_url,
            timeout=settings.mimo_timeout,
            max_retries=0,  # 关掉 SDK 自带重试，统一交给 tenacity，避免双重退避
        )
        self.tracker = tracker or CostTracker()

    # -- 内部: 构造请求参数 ---------------------------------------------
    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
        json_mode: bool,
    ) -> dict[str, Any]:
        extra_body: dict[str, Any] = {"thinking": {"type": self._settings.mimo_thinking}}

        kwargs: dict[str, Any] = {
            "model": self._settings.mimo_model,
            "messages": messages,
            "max_completion_tokens": max_tokens or self._settings.max_tokens_per_request,
            "extra_body": extra_body,
        }

        # 思考模式下 temperature/top_p 会被服务端强制，这里显式跳过，省得误导自己
        if self._settings.mimo_thinking == "disabled" and temperature is not None:
            kwargs["temperature"] = temperature

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # 结构化输出: 强制模型产出合法 JSON（MiMo 兼容 response_format）
        if json_mode and not tools:
            kwargs["response_format"] = {"type": "json_object"}

        return kwargs

    # -- 对外: 普通对话 ---------------------------------------------------
    def chat(
        self,
        user: str,
        *,
        system: str | None = None,
        history: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """单轮/多轮对话。history 形如 [{"role": "user", "content": "..."}]。"""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})
        return self._invoke(messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)

    # -- 对外: 带工具调用的原始接口（M3 Agent 会基于它做循环）--------------
    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """返回的 raw 中含 tool_calls，由调用方执行工具后继续对话。"""
        return self._invoke(messages, tools=tools, max_tokens=max_tokens, json_mode=False)

    # -- 对外: 流式 -------------------------------------------------------
    def chat_stream(
        self,
        user: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """流式输出，逐段 yield 文本。适合 SSE 推给前端（M6）。"""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        kwargs = self._build_kwargs(
            messages, temperature=None, max_tokens=max_tokens, tools=None, json_mode=False
        )
        kwargs["stream"] = True

        request_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        stream = self._client.chat.completions.create(**kwargs)
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
        log.info("llm_stream_done", request_id=request_id, latency_ms=(time.perf_counter() - start) * 1000)

    # -- 对外: 结构化输出（Pydantic 模型）--------------------------------
    def chat_structured(
        self,
        user: str,
        schema_model: type[T],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """
        让模型输出符合 Pydantic 模型的 JSON，再做校验。

        工程化要点：模型生成的 JSON **不可信任**，必须 validate + 失败重试。
        这里用 JSON Schema 注入 system prompt 的方式，兼容性最好；
        M3 会讲更稳的 tool-calling 结构化输出。
        """
        schema = schema_model.model_json_schema()
        base_system = system or "你是一个精确的结构化输出助手。"
        prompt = (
            f"{base_system}\n\n"
            f"你必须只输出一个 JSON 对象，严格符合以下 JSON Schema，不要输出任何解释文字：\n"
            f"```json\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n```"
        )
        resp = self.chat(user, system=prompt, max_tokens=max_tokens, json_mode=True)
        return self._parse_structured(resp.content, schema_model)

    @staticmethod
    def _parse_structured(content: str, schema_model: type[T]) -> T:
        """容忍模型输出 markdown 代码块包裹的情况。"""
        text = content.strip()
        if text.startswith("```"):
            # 去掉 ```json ... ```
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return schema_model.model_validate_json(text)
        except ValidationError as exc:
            log.error("structured_parse_failed", error=str(exc), content=text[:500])
            raise ValueError(f"模型输出不符合 {schema_model.__name__} schema: {exc}") from exc

    # -- 核心调用 ---------------------------------------------------------
    def _invoke(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        request_id = uuid.uuid4().hex[:12]
        kwargs = self._build_kwargs(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            json_mode=json_mode,
        )
        log.info(
            "llm_request",
            request_id=request_id,
            model=kwargs["model"],
            n_messages=len(messages),
            has_tools=bool(tools),
        )

        response = self._call_with_retry(kwargs, request_id=request_id)

        choice = response.choices[0]
        message = choice.message
        usage = UsageRecord(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            model=response.model,
            request_id=request_id,
        )
        usage.cost_cny = compute_cost_cny(usage.model, usage.prompt_tokens, usage.completion_tokens)
        self.tracker.add(usage)

        log.info(
            "llm_response",
            request_id=request_id,
            finish_reason=choice.finish_reason,
            tokens=usage.total_tokens,
            cost_cny=round(usage.cost_cny, 6),
        )

        return LLMResponse(
            content=message.content or "",
            model=response.model,
            usage=usage,
            reasoning_content=getattr(message, "reasoning_content", None),
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    def _call_with_retry(self, kwargs: dict[str, Any], *, request_id: str) -> Any:
        """带指数退避的重试。4xx 业务错误（如鉴权失败）不重试。"""

        @retry(
            retry=retry_if_exception_type(
                (APIConnectionError, APITimeoutError, RateLimitError)
            ),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            stop=stop_after_attempt(self._settings.mimo_max_retries),
            before_sleep=lambda retry_state: log.warning(
                "llm_retry",
                request_id=request_id,
                attempt=retry_state.attempt_number,
                wait=retry_state.next_action.sleep if retry_state.next_action else None,
            ),
            reraise=True,
        )
        def _do() -> Any:
            return self._client.chat.completions.create(**kwargs)

        try:
            return _do()
        except APIStatusError as exc:
            # 4xx 不重试：401 密钥错、400 参数错，重试也没用
            log.error("llm_api_error", request_id=request_id, status=exc.status_code, body=str(exc)[:300])
            raise
