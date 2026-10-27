"""
M7 · RAG 评估指标

【为什么要自己写一个简化版评估器？】
  1. 理解原理：RAGAS 做的事就是「拆陈述句 → 逐句判断 → 算比例」，自己写一遍就不会觉得它是黑盒
  2. 零依赖：不需要调 LLM，不花钱，1ms 跑完，适合 CI 回归测试
  3. 对比基准：和 RAGAS 跑同一批数据，看两者差距，自查可讲「我知道框架在封装什么」

【四个指标怎么算的？】
  Faithfulness      答案拆成陈述句，逐句能否从 context 推出 → 支持数/总数
  Answer Relevancy  答案和问题的语义相关度（关键词/重叠）
  Context Precision 相关 chunk 是否排在前面（rank 加权）
  Context Recall    ground_truth 的关键信息是否被检索到（fact 覆盖率）

【SimpleEvaluator 的局限】
  用关键词/字符串匹配代替 LLM 判断——快、免费，但不够准：
  语义等价的表述可能匹配不上（"30天" vs "三十天"）。
  生产评估用 RAGAS（LLM-as-judge），CI 回归用 SimpleEvaluator（快+免费）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass
class EvalResult:
    """一次评估的完整结果。"""

    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_correctness: float = 0.0
    n_cases: int = 0
    # 每条用例的明细（排障时看哪条差）
    details: list[dict[str, Any]] = field(default_factory=list)
    # -- 可追溯性：这四个数字到底是谁算出来的 -------------------------
    # 不标来源的结果没法用：ragas 静默降级成 Simple 之后，
    # 报告里写的「RAGAS 0.82」可能是关键词匹配的 0.82，那是假的。
    evaluator: str = "simple"              # "simple" | "ragas" | "ragas+simple"
    ragas_ran: bool = False
    fallback_reason: str = ""              # 降级原因（ragas 没跑起来时必填）
    metrics_source: dict[str, str] = field(default_factory=dict)  # 指标名 -> "ragas"|"simple"

    @property
    def overall(self) -> float:
        """综合分 = 五个指标的平均。"""
        scores = [
            self.faithfulness,
            self.answer_relevancy,
            self.context_precision,
            self.context_recall,
            self.answer_correctness,
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def summary(self) -> str:
        # 只有真的降级过才打 fallback 标 —— Simple 本来就该跑，不算降级
        tag = f"{self.evaluator}(fallback)" if self.fallback_reason else self.evaluator
        return (
            f"[{tag}] "
            f"Faithfulness={self.faithfulness:.2f}  "
            f"Relevancy={self.answer_relevancy:.2f}  "
            f"Prec={self.context_precision:.2f}  "
            f"Recall={self.context_recall:.2f}  "
            f"Correct={self.answer_correctness:.2f}  "
            f"Overall={self.overall:.2f}"
        )

    def to_dict(self) -> dict[str, Any]:
        def clean(v: float) -> float | None:
            # NaN 不能进 JSON（Postgres JSON 列会拒绝），转成 None，报告里显示「—」
            return None if v != v else round(v, 4)

        return {
            "faithfulness": clean(self.faithfulness),
            "answer_relevancy": clean(self.answer_relevancy),
            "context_precision": clean(self.context_precision),
            "context_recall": clean(self.context_recall),
            "answer_correctness": clean(self.answer_correctness),
            "overall": clean(self.overall),
            "n_cases": self.n_cases,
            "evaluator": self.evaluator,
            "ragas_ran": self.ragas_ran,
            "fallback_reason": self.fallback_reason,
            "metrics_source": dict(self.metrics_source),
        }


# ---------------------------------------------------------------------------
# 简化版评估器 —— 零依赖，适合 CI 回归测试
# ---------------------------------------------------------------------------
# 中文停用字：bigram 里只要含这些纯功能字就整体丢弃（「的住」「是多」这类组合不算关键词）。
# 刻意不收录数字（一/二/三…）和否定词（不/没）——「一线」「不超过」都是实义信息，不能误删。
_CJK_STOP_CHARS = frozenset(
    "的了是在和与及或为于对很都也又更最有能可以需要这那什么多少怎么吗呢吧啊哪请如何否个之而则"
)


class SimpleEvaluator:
    """
    关键词/字符串匹配版评估器。不调 LLM，不花钱，瞬间出结果。

    用法:
        eval = SimpleEvaluator()
        result = eval.evaluate(dataset)
        print(result.summary())
    """

    def evaluate(self, dataset) -> EvalResult:
        """对整个数据集评估，返回聚合结果。"""
        details = []
        f_scores, r_scores, p_scores, c_scores, ac_scores = [], [], [], [], []

        for case in dataset:
            f = self._faithfulness(case.answer, case.contexts)
            r = self._answer_relevancy(case.question, case.answer)
            p = self._context_precision(case.contexts, case.expected_facts)
            c = self._context_recall(case.contexts, case.expected_facts)
            ac = self._answer_correctness(case.ground_truth, case.answer)

            f_scores.append(f)
            r_scores.append(r)
            p_scores.append(p)
            c_scores.append(c)
            ac_scores.append(ac)
            details.append({
                "question": case.question,
                "faithfulness": round(f, 3),
                "answer_relevancy": round(r, 3),
                "context_precision": round(p, 3),
                "context_recall": round(c, 3),
                "answer_correctness": round(ac, 3),
            })

        n = len(dataset)
        result = EvalResult(
            faithfulness=sum(f_scores) / n if n else 0,
            answer_relevancy=sum(r_scores) / n if n else 0,
            context_precision=sum(p_scores) / n if n else 0,
            context_recall=sum(c_scores) / n if n else 0,
            answer_correctness=sum(ac_scores) / n if n else 0,
            n_cases=n,
            details=details,
            evaluator="simple",
            ragas_ran=False,
            metrics_source={
                "faithfulness": "simple",
                "answer_relevancy": "simple",
                "context_precision": "simple",
                "context_recall": "simple",
                "answer_correctness": "simple",
            },
        )
        log.info("eval_done", **result.to_dict())
        return result

    def _faithfulness(self, answer: str, contexts: list[str]) -> float:
        """
        Faithfulness = 答案中的信息有多少能在 context 中找到支撑。

        简化版做法：把答案按标点拆成"陈述片段"，
        每个片段的关键词是否出现在 context 中。
        （RAGAS 用 LLM 做同样的事，只是判断更准）
        """
        if not answer or not contexts:
            return 0.0

        statements = self._split_statements(answer)
        if not statements:
            return 0.0

        context_text = " ".join(contexts)
        supported = 0
        for stmt in statements:
            if self._is_supported(stmt, context_text):
                supported += 1

        return supported / len(statements)

    def _answer_relevancy(self, question: str, answer: str) -> float:
        """
        Answer Relevancy = 答案和问题的关键词重叠度。

        简化版做法：问题的关键词有多少出现在答案中。
        （RAGAS 用 LLM 生成反向问题来评估语义相关度）
        """
        if not question or not answer:
            return 0.0

        q_keywords = self._extract_keywords(question)
        if not q_keywords:
            return 0.0

        a_text = answer.lower()
        matched = sum(1 for kw in q_keywords if kw in a_text)
        return matched / len(q_keywords)

    def _answer_correctness(self, ground_truth: str, answer: str) -> float:
        """
        Answer Correctness = 生成答案是否覆盖了标准答案的关键信息。

        免费近似：标准答案的关键词（bigram）有多少出现在生成答案里。
        这是「答案到底对不对」的最弱近似——真正的正确性要靠 LLM-as-judge
        逐条判，见 RagasEvaluator / report.py 的 --ragas-on 对照。
        """
        return self._answer_relevancy(ground_truth, answer)

    def _context_precision(self, contexts: list[str], expected_facts: list[str]) -> float:
        """
        Context Precision = 相关的 chunk 是否排在前面。

        简化版做法：包含预期事实的 chunk 越靠前得分越高。
        （RAGAS 用 LLM 判断每个 chunk 是否相关，再算加权精度）

        【分母是「相关块数」不是「返回块数」】
          RAGAS 的定义是 sum(P@k · v_k) / sum(v_k)，v_k 表示第 k 块是否相关。
          除以 len(contexts) 的话，top_k=5、唯一相关块排第 1 位（完美排序）
          只能得 1/5 = 0.2 —— 数字看着像检索很差，其实排序没问题。
          这种公式错误会让整张对比表失去意义。
        """
        if not contexts or not expected_facts:
            return 0.0

        total = 0.0
        found = 0
        for rank, ctx in enumerate(contexts):
            if self._contains_any_fact(ctx, expected_facts):
                found += 1
                total += found / (rank + 1)   # P@k：前 k 个里相关的比例

        return total / found if found else 0.0

    def _context_recall(self, contexts: list[str], expected_facts: list[str]) -> float:
        """
        Context Recall = ground_truth 的关键信息是否被检索到。

        简化版做法：预期事实有多少出现在检索到的 context 中。
        """
        if not expected_facts or not contexts:
            return 0.0

        context_text = " ".join(contexts)
        found = sum(1 for fact in expected_facts if fact in context_text)
        return found / len(expected_facts)

    # -- 工具方法 ---------------------------------------------------------
    @staticmethod
    def _split_statements(text: str) -> list[str]:
        """把答案拆成陈述片段。"""
        import re
        # 按中文/英文标点拆分
        parts = re.split(r'[。；！？;!?，,\n]+', text)
        return [p.strip() for p in parts if len(p.strip()) > 2]

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """提取关键词：英文按词，中文按字符二元组 + 停用字过滤。"""
        import re
        # 中文没有空格分词，SimpleEvaluator 又刻意不引第三方分词库（零依赖、CI 秒级）。
        # 用 shingling 折中：连续汉字切成二字串，含停用字的丢弃。
        # 这样「一线城市的住宿标准是多少」能切出「一线/城市/住宿/标准」，
        # 而不是旧版那样把整句当成一个词 —— 那是 answer_relevancy 恒为 0 的根源。
        en_words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        bigrams: list[str] = []
        for seg in re.findall(r"[一-鿿]+", text):
            for i in range(len(seg) - 1):
                bg = seg[i : i + 2]
                if bg[0] not in _CJK_STOP_CHARS and bg[1] not in _CJK_STOP_CHARS:
                    bigrams.append(bg)
        stopwords = {"the", "and", "for"}
        return [w for w in en_words + bigrams if w not in stopwords]

    @staticmethod
    def _is_supported(statement: str, context_text: str) -> bool:
        """判断一个陈述是否被 context 支持（简化：关键词匹配）。"""
        import re
        keywords = re.findall(r'[一-鿿]{2,}|[a-zA-Z0-9]{3,}', statement)
        if not keywords:
            return False
        # 超过一半的关键词在 context 中出现就算支持
        matched = sum(1 for kw in keywords if kw in context_text)
        return matched / len(keywords) >= 0.5

    @staticmethod
    def _contains_any_fact(text: str, facts: list[str]) -> bool:
        """判断文本是否包含任一预期事实。"""
        return any(fact in text for fact in facts)


# ---------------------------------------------------------------------------
# RAGAS 评估器 —— LLM-as-judge，正式评估用
# ---------------------------------------------------------------------------
# ragas 里只有这三个是纯 LLM 判断；answer_relevancy / answer_correctness 额外要 embedding 服务
# （前者生成反问题算向量相似度，后者要算答案相似度）。本地跑 bge-m3 时没有 OpenAI embeddings，
# 所以这两项单独走 Simple 并如实标注来源 —— 不装作是 ragas 算的。
_RAGAS_LLM_METRICS = ("faithfulness", "context_precision", "context_recall")
_RAGAS_ALL_METRICS = _RAGAS_LLM_METRICS + ("answer_relevancy", "answer_correctness")


def _nanmean(values: list[float]) -> float:
    """聚合均值，跳过 NaN（ragas 个别 judge 超时、MiMo 慢时会返回 NaN）。"""
    vals = [v for v in values if v == v]  # NaN != NaN，靠这个技巧过滤
    return sum(vals) / len(vals) if vals else float("nan")


def _ensure_ragas_importable() -> None:
    """
    让 `import ragas` 在当前环境能通过。

    ragas 0.4.x 在 ragas/llms/base.py 顶层硬 import：
        from langchain_community.chat_models.vertexai import ChatVertexAI
    但配合 langchain-core 1.x 的 langchain_community 0.4.x 已把 chat 版 vertexai
    拆到独立包（legacy 的 llms 版还在，所以另一行 `from langchain_community.llms
    import VertexAI` 是通的）。这个顶层 import 一炸，整个 ragas 都进不来。

    本项目只用 MiMo（ChatOpenAI）当评估裁判，从不碰 VertexAI；ragas 里
    ChatVertexAI 也只出现在 MULTIPLE_COMPLETION_SUPPORTED 里做 isinstance 判断。
    所以注入一个占位类型即可，不下载 google 系重依赖、不降级 langchain 全家桶。
    """
    import sys
    import types

    mod_name = "langchain_community.chat_models.vertexai"
    if mod_name in sys.modules:
        return

    class ChatVertexAI:
        """占位类型：只为让 isinstance 判断有类型可查，项目从不实例化。"""

    mod = types.ModuleType(mod_name)
    mod.ChatVertexAI = ChatVertexAI
    sys.modules[mod_name] = mod


class RagasEvaluator:
    """
    RAGAS 封装。用 MiMo 当裁判（LLM-as-judge）。

    【LLM-as-judge 的原理】
      RAGAS 内部让 LLM 做判断题：
        "这句话能从以下文档中推出吗？→ 是/否"
      统计"是"的比例 → 得到指标分数。

    【为什么降级要留痕？】
      旧版这里捕获异常后默默 fallback 到 SimpleEvaluator，结果上看不出来
      ragas 到底跑没跑。报告里写「RAGAS 得分 0.82」而实际是关键词匹配的
      0.82，这是假数据 —— 简历项目经不起这么问。
      现在 EvalResult.ragas_ran / fallback_reason / metrics_source 必填。

    【什么时候用 RAGAS vs SimpleEvaluator？】
      SimpleEvaluator：CI 回归测试（免费、快、不用网络）
      RagasEvaluator：正式评估报告（准、要花钱、需要网络）
    """

    def __init__(self, llm_client: Any = None, embeddings: Any = None) -> None:
        self.llm_client = llm_client   # 用来包一层 ragas 认识的 judge
        self.embeddings = embeddings  # 传了才有 answer_relevancy 的 ragas 版

    def evaluate(self, dataset) -> EvalResult:
        """
        跑 RAGAS 评估。需要 ragas + LLM API（会产生 token 费用）。

        任何一环失败都会降级到 SimpleEvaluator，但结果里会写明
        ragas_ran=False 和失败原因，绝不冒充 ragas 分数。
        """
        try:
            return self._evaluate_with_ragas(dataset)
        except Exception as exc:
            log.warning("ragas_unavailable", error=str(exc), action="fallback_simple")
            result = SimpleEvaluator().evaluate(dataset)
            result.evaluator = "simple"
            result.ragas_ran = False
            result.fallback_reason = f"{type(exc).__name__}: {exc}"
            result.metrics_source = {k: "simple" for k in _RAGAS_ALL_METRICS}
            return result

    def _evaluate_with_ragas(self, dataset) -> EvalResult:
        _ensure_ragas_importable()  # 必须在 `import ragas` 之前打补丁
        from datasets import Dataset as HFDataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness,
        }

        # 传了 embeddings 才跑 answer_relevancy，否则这项回退 Simple
        names = list(_RAGAS_ALL_METRICS if self.embeddings else _RAGAS_LLM_METRICS)
        metrics = [metric_map[n] for n in names]

        data = {
            "question": [c.question for c in dataset],
            "answer": [c.answer for c in dataset],
            "contexts": [c.contexts for c in dataset],
            "ground_truth": [c.ground_truth for c in dataset],
        }
        hf_ds = HFDataset.from_dict(data)

        kwargs: dict[str, Any] = {}
        judge = self._build_judge()
        if judge is not None:
            kwargs["llm"] = judge
        if self.embeddings is not None:
            kwargs["embeddings"] = self.embeddings

        result = evaluate(hf_ds, metrics=metrics, **kwargs)

        source = {n: "ragas" for n in names}
        # ragas 0.4.x 的 result[n] 是「每行分数」的 List[float]，聚合取均值才是该指标得分。
        # 个别 judge 超时（MiMo 慢）会返回 NaN，取均值时跳过，否则整个指标被污染成 NaN。
        scores = {n: _nanmean(result[n]) for n in names}

        # answer_relevancy / answer_correctness 没进 ragas 时用 Simple 算，并如实标注
        if "answer_relevancy" not in scores or "answer_correctness" not in scores:
            simple = SimpleEvaluator().evaluate(dataset)
        if "answer_relevancy" not in scores:
            scores["answer_relevancy"] = simple.answer_relevancy
            source["answer_relevancy"] = "simple"
        if "answer_correctness" not in scores:
            scores["answer_correctness"] = simple.answer_correctness
            source["answer_correctness"] = "simple"

        return EvalResult(
            faithfulness=scores.get("faithfulness", 0.0),
            answer_relevancy=scores.get("answer_relevancy", 0.0),
            context_precision=scores.get("context_precision", 0.0),
            context_recall=scores.get("context_recall", 0.0),
            answer_correctness=scores.get("answer_correctness", 0.0),
            n_cases=len(dataset),
            evaluator="ragas" if source.get("answer_relevancy") == "ragas" else "ragas+simple",
            ragas_ran=True,
            metrics_source=source,
        )

    def _build_judge(self) -> Any | None:
        """
        把 MiMo 包成 ragas 认识的 judge。

        ragas 只吃 LangChain Runnable 包装过的 LLM（LangchainLLMWrapper），
        不吃裸的 OpenAI 客户端 —— 这是接线时最容易卡住的一步。
        """
        if self.llm_client is None:
            return None  # 不指定 judge，ragas 自己找默认（需 OPENAI_API_KEY）

        try:
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper
        except ImportError:
            return None

        from knowflow.core.config import get_settings

        settings = get_settings()
        chat = ChatOpenAI(
            model=settings.mimo_model,
            api_key=settings.mimo_api_key.get_secret_value(),
            base_url=settings.mimo_base_url,
            temperature=0,
            timeout=settings.mimo_timeout,
        )
        return LangchainLLMWrapper(chat)


def evaluate_rag(dataset, evaluator: SimpleEvaluator | RagasEvaluator | None = None) -> EvalResult:
    """
    便捷入口。

    【为什么不自动挑？】
      默认必须是 SimpleEvaluator：ragas 要装依赖、要调 LLM、要花钱。
      「自动选择」在没装 ragas 时会静默降级，结果不可比。
      要跑 ragas 就显式传 RagasEvaluator() —— 花钱的事不该隐式发生。
    """
    if evaluator is None:
        evaluator = SimpleEvaluator()
    return evaluator.evaluate(dataset)
