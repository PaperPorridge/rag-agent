"""
M3 · 工具定义与执行 —— Function Calling 的「手和脚」

【Function Calling 到底是什么？】
  LLM 本身只会生成文本，不会「执行动作」。
  Function Calling 的本质：**让模型输出「我想调用哪个函数、参数是什么」的结构化声明**，
  真正的执行由你的代码完成。模型是「决策者」，你的运行时是「执行者」。

  一次工具调用的时序：
    你 ──(带 tools schema 的请求)──> 模型
    模型 ──(tool_calls: [{name, arguments}])──> 你
    你 ──(本地执行函数，得到结果)──> 你
    你 ──(tool 消息把结果回传)──> 模型
    模型 ──(最终自然语言答案)──> 你

  注意：模型从不直接执行任何东西。它只是「猜」参数。
  官方文档原话：「模型生成的参数并非总能保证有效，且可能虚构 schema 中未定义的参数」。

【工具设计四原则】
  1. 描述即接口: tools 的 description 就是模型的「产品文档」，写不清 = 调不对
  2. 参数要 Schema: JSON Schema 约束类型/必填/枚举，减少幻觉参数
  3. 返回要短而结构化: 工具结果会塞进上下文，长文本 = 烧 token + 淹没关键信息
  4. 失败要返回错误信息: 让模型看到「为什么失败」才能自我修正（比抛异常强）

【安全三件套 —— 真正的边界在这里】
  ① 权限最小化: 只暴露必须的工具。搜索工具永远不该有 delete 权限
  ② 参数校验: Pydantic 强校验 + 白名单/正则，不信模型给的任何参数
  ③ 高危操作人工确认（HITL）: 发邮件/删数据/付款必须二次确认
  Prompt 防御只是第一道门，工具权限才是安全边界（呼应 M1 Q4）。
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin

import structlog
from pydantic import BaseModel, ValidationError

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool: 一个可被模型调用的函数
# ---------------------------------------------------------------------------
@dataclass
class Tool:
    """
    工具封装。三个东西：
      schema     给模型看（它靠这个决定何时调用、参数填什么）
      handler    给运行时执行（真正干活的函数）
      policy     安全策略（只读？需要人工确认？参数约束？）
    """

    name: str
    description: str
    handler: Callable[..., Any]
    parameters: dict[str, Any]  # JSON Schema
    readonly: bool = True
    requires_confirmation: bool = False
    # 返回给模型时的最大字符数 —— 工具结果会进上下文，必须截断
    max_result_chars: int = 2000

    def to_openai_schema(self) -> dict[str, Any]:
        """转成 OpenAI/MiMo tools 参数格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs: Any) -> str:
        """执行工具并把结果规整成字符串。任何异常都转成错误文本（给模型看，不是抛给用户）。"""
        try:
            result = self.handler(**kwargs)
            text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            # 关键设计：失败也返回文本，模型才能看到原因并自我修正
            log.warning("tool_failed", tool=self.name, args=kwargs, error=str(exc))
            return f"[工具执行失败] {self.name}: {exc}"

        if len(text) > self.max_result_chars:
            text = text[: self.max_result_chars] + f"\n...[已截断，共 {len(text)} 字符]"
        return text


# ---------------------------------------------------------------------------
# ToolRegistry: 工具注册表（集中管理 = 权限可审计）
# ---------------------------------------------------------------------------
class ToolRegistry:
    """
    工具注册表。

    为什么要有注册表，而不是 tools=[f1,f2,f3] 传参？
      1. 权限审计：一处列出「模型能做什么」，安全评审有依据
      2. 按场景裁剪：对话 Agent 和后台批处理可以挂不同工具子集
      3. 统一 schema 转换与执行日志
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重名: {tool.name}")
        self._tools[tool.name] = tool
        log.info("tool_registered", name=tool.name, readonly=tool.readonly, hitl=tool.requires_confirmation)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def to_openai_schema(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        tools = self._tools.values() if names is None else (self._tools[n] for n in names if n in self._tools)
        return [t.to_openai_schema() for t in tools]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self.get(name)
        if tool is None:
            # 白名单外的工具名 —— 直接拒绝（模型可能幻觉出不存在的工具）
            log.warning("tool_not_allowed", name=name, allowed=list(self._tools))
            return f"[拒绝] 工具 {name} 不存在或未授权。可用工具: {list(self._tools)}"
        if tool.requires_confirmation and not arguments.pop("_confirmed", False):
            return (
                f"[需人工确认] 工具 {name} 为高危操作，已暂缓执行。"
                f"参数: {json.dumps(arguments, ensure_ascii=False)}。请提示用户确认后重试。"
            )
        return tool.execute(**arguments)

    @property
    def names(self) -> list[str]:
        return list(self._tools)


# ---------------------------------------------------------------------------
# 参数校验：不信模型给的任何参数
# ---------------------------------------------------------------------------
def validate_arguments(tool: Tool, raw_arguments: str | dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """
    解析并校验模型给的参数。返回 (合法参数, 错误信息)。

    工程化要点：
      - 模型生成的 JSON 可能非法、可能缺必填、可能编造未定义字段
      - 用 JSON Schema 做第一道校验，再做业务白名单（如 SQL 只读）
      - 校验失败要返回**可读的错误原因**，模型才能自我修正
    """
    # 1) 解析 JSON
    if isinstance(raw_arguments, str):
        try:
            args = json.loads(raw_arguments) if raw_arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return None, f"参数不是合法 JSON: {exc}。请输出严格的 JSON 对象。"
    else:
        args = dict(raw_arguments)

    if not isinstance(args, dict):
        return None, "参数必须是 JSON 对象"

    # 2) 必填字段检查
    required = tool.parameters.get("required", [])
    missing = [k for k in required if k not in args]
    if missing:
        return None, f"缺少必填参数: {missing}"

    # 3) 未知字段拒绝（模型爱虚构 schema 外的参数）
    defined = set(tool.parameters.get("properties", {}).keys())
    extra = [k for k in args if k not in defined]
    if extra:
        return None, f"包含未定义参数: {extra}，允许的参数: {sorted(defined)}"

    # 4) 类型粗校验（JSON Schema type 字段）
    props = tool.parameters.get("properties", {})
    for key, value in args.items():
        expected = props.get(key, {}).get("type")
        if expected and not _type_matches(value, expected):
            return None, f"参数 {key} 类型错误: 期望 {expected}，实际 {type(value).__name__}"

    return args, None


def _type_matches(value: Any, expected: str) -> bool:
    mapping: dict[str, tuple[type, ...]] = {
        "string": (str,),
        "number": (int, float),
        "integer": (int,),
        "boolean": (bool,),
        "object": (dict,),
        "array": (list,),
    }
    types = mapping.get(expected, (object,))
    if expected in {"number", "integer"} and isinstance(value, bool):
        return False  # bool 是 int 子类，但 JSON Schema 里它们不等价
    return isinstance(value, types)


# ---------------------------------------------------------------------------
# 内置工具
# ---------------------------------------------------------------------------
class CalculatorInput(BaseModel):
    expression: str


def calculator(expression: str) -> str:
    """
    安全计算器。**绝不用 eval()** —— 那是任意代码执行漏洞。

    用 ast.parse 只允许数字/四则运算/幂/括号/白名单数学函数。
    模型算错账是小事，eval 被注入是安全事故。
    """
    node = ast.parse(expression, mode="eval")
    return f"{expression} = {_safe_eval(node.body)}"


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "round": round,
    "abs": abs,
    "min": min,
    "max": max,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _ALLOWED_FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"不允许的函数: {node.func.id}，可用: {sorted(_ALLOWED_FUNCS)}")
        return fn(*[_safe_eval(a) for a in node.args])
    raise ValueError(f"不允许的表达式节点: {type(node).__name__}")


def make_sql_tool(db_path: str, readonly: bool = True) -> Tool:
    """
    查询内部业务数据库（演示 Text2SQL 场景）。

    安全要点：
      - readonly=True 时用 SQL 白名单只放行 SELECT/WITH，其余直接拒绝
      - 这就是「权限最小化」的落地：就算模型被注入带偏，也执行不了 DROP/UPDATE
      - 连接只读打开（file:...?mode=ro）双保险
    """

    def query_database(sql: str) -> str:
        if readonly and not re.match(r"^\s*(select|with)\b", sql, flags=re.IGNORECASE):
            return f"[拒绝] 只允许只读查询（SELECT/WITH），收到: {sql[:80]}..."

        import sqlite3

        uri = f"file:{db_path}?mode={'ro' if readonly else 'rw'}"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
        try:
            conn.execute("PRAGMA query_only = ON" if readonly else "PRAGMA query_only = OFF")
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(50)  # 限制行数，防止大结果淹没上下文
            if not rows:
                return "查询成功，无结果。"
            lines = [" | ".join(cols)] + [" | ".join(str(v) for v in row) for row in rows]
            return "\n".join(lines)
        finally:
            conn.close()

    return Tool(
        name="query_database",
        description=(
            "查询公司内部业务数据库（SQLite），返回表格文本。"
            "只能执行 SELECT 或 WITH 查询，不支持写操作。"
            "表结构见表清单，不确定表名时先用 SELECT name FROM sqlite_master 查询。"
        ),
        handler=query_database,
        parameters={
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "要执行的 SQL 查询语句，仅限 SELECT"}
            },
            "required": ["sql"],
        },
        readonly=True,
        requires_confirmation=False,
    )


def search_knowledge_base(knowledge_fn: Callable[[str], str]) -> Tool:
    """
    把 M2 的 RAG 管道包装成工具 —— Agent 能力复用的关键：
    「知识库检索」对 Agent 而言只是一个带 schema 的函数。
    """

    def search(query: str) -> str:
        return knowledge_fn(query)

    return Tool(
        name="search_knowledge_base",
        description=(
            "在企业知识库（制度、流程、设备清单等文档）中检索与问题相关的片段，"
            "返回带来源标注的文本。适合回答公司制度、流程、规定类问题。"
        ),
        handler=search,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词或自然语言问题"}
            },
            "required": ["query"],
        },
        readonly=True,
    )


def create_default_registry(
    knowledge_fn: Callable[[str], str] | None = None,
    db_path: str | None = None,
) -> ToolRegistry:
    """组装默认工具集。注意：只注册场景必需的工具 —— 这就是权限最小化。"""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description="计算数学表达式，支持四则运算、幂、括号和 sqrt/round/abs/min/max/log 等函数。",
            handler=calculator,
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 (1280+500)*1.13"}
                },
                "required": ["expression"],
            },
        )
    )
    if knowledge_fn is not None:
        registry.register(search_knowledge_base(knowledge_fn))
    if db_path is not None:
        registry.register(make_sql_tool(db_path))
    return registry
