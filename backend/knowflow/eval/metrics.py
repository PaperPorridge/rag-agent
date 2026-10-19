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
    n_cases: int = 0
    # 每条用例的明细（排障时看哪条差）
    details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def overall(self) -> float:
        """综合分 = 四个指标的平均。"""
        scores = [self.faithfulness, self.answer_relevancy, self.context_precision, self.context_recall]
        return sum(scores) / len(scores) if scores else 0.0

    def summary(self) -> str:
        return (
            f"Faithfulness={self.faithfulness:.2f}  "
            f"Relevancy={self.answer_relevancy:.2f}  "
            f"Prec@{self.n_cases}={self.context_precision:.2f}  "
            f"Recall={self.context_recall:.2f}  "
            f"Overall={self.overall:.2f}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "overall": round(self.overall, 4),
            "n_cases": self.n_cases,
        }


# ---------------------------------------------------------------------------
# 简化版评估器 —— 零依赖，适合 CI 回归测试
# ---------------------------------------------------------------------------
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
        f_scores, r_scores, p_scores, c_scores = [], [], [], []

        for case in dataset:
            f = self._faithfulness(case.answer, case.contexts)
            r = self._answer_relevancy(case.question, case.answer)
            p = self._context_precision(case.contexts, case.expected_facts)
            c = self._context_recall(case.contexts, case.expected_facts)

            f_scores.append(f)
            r_scores.append(r)
            p_scores.append(p)
            c_scores.append(c)
            details.append({
                "question": case.question,
                "faithfulness": round(f, 3),
                "answer_relevancy": round(r, 3),
                "context_precision": round(p, 3),
                "context_recall": round(c, 3),
            })

        n = len(dataset)
        result = EvalResult(
            faithfulness=sum(f_scores) / n if n else 0,
            answer_relevancy=sum(r_scores) / n if n else 0,
            context_precision=sum(p_scores) / n if n else 0,
            context_recall=sum(c_scores) / n if n else 0,
            n_cases=n,
            details=details,
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

    def _context_precision(self, contexts: list[str], expected_facts: list[str]) -> float:
        """
        Context Precision = 相关的 chunk 是否排在前面。

        简化版做法：包含预期事实的 chunk 在列表中越靠前得分越高。
        （RAGAS 用 LLM 判断每个 chunk 是否相关，再算加权精度）
        """
        if not contexts or not expected_facts:
            return 0.0

        total = 0.0
        found = 0
        for rank, ctx in enumerate(contexts):
            if self._contains_any_fact(ctx, expected_facts):
                found += 1
                total += found / (rank + 1)   # 排名越靠前得分越高

        return total / len(contexts)

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
        """提取关键词（去掉停用词和标点）。"""
        import re
        # 中文按字/词切（简化：2字以上连续中文），英文按单词
        cn_words = re.findall(r'[一-鿿]{2,}', text)
        en_words = re.findall(r'[a-zA-Z]{3,}', text.lower())
        stopwords = {"什么", "多少", "是否", "可以", "需要", "这个", "那个", "the", "and", "for"}
        return [w for w in cn_words + en_words if w not in stopwords]

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
class RagasEvaluator:
    """
    RAGAS 封装。需要 LLM API（用 MiMo 当裁判）。

    【LLM-as-judge 的原理】
      RAGAS 内部让 LLM 做判断题：
        "这句话能从以下文档中推出吗？→ 是/否"
      统计"是"的比例 → 得到指标分数。

    【什么时候用 RAGAS vs SimpleEvaluator？】
      SimpleEvaluator：CI 回归测试（免费、快、不用网络）
      RagasEvaluator：正式评估报告（准、要花钱、需要网络）
    """

    def __init__(self, llm_client: Any = None) -> None:
        self.llm_client = llm_client

    def evaluate(self, dataset) -> EvalResult:
        """
        跑 RAGAS 评估。

        注意：需要 OpenAI 兼容 API（MiMo），会产生 token 费用。
        演示里如果没配 API key，会 fallback 到 SimpleEvaluator。
        """
        try:
            return self._evaluate_with_ragas(dataset)
        except Exception as exc:
            log.warning("ragas_unavailable", error=str(exc), action="fallback_simple")
            evaluator = SimpleEvaluator()
            return evaluator.evaluate(dataset)

    def _evaluate_with_ragas(self, dataset) -> EvalResult:
        """真正的 RAGAS 调用（需要安装 ragas + 配置 LLM）。"""
        from datasets import Dataset as HFDataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        # 转成 HuggingFace Dataset 格式
        data = {
            "question": [c.question for c in dataset],
            "answer": [c.answer for c in dataset],
            "contexts": [c.contexts for c in dataset],
            "ground_truth": [c.ground_truth for c in dataset],
        }
        hf_ds = HFDataset.from_dict(data)

        # 跑 RAGAS（这里会调 LLM 当裁判）
        result = evaluate(hf_ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])

        return EvalResult(
            faithfulness=float(result["faithfulness"]),
            answer_relevancy=float(result["answer_relevancy"]),
            context_precision=float(result["context_precision"]),
            context_recall=float(result["context_recall"]),
            n_cases=len(dataset),
        )


def evaluate_rag(dataset, evaluator: SimpleEvaluator | RagasEvaluator | None = None) -> EvalResult:
    """便捷入口：自动选择评估器。"""
    if evaluator is None:
        evaluator = SimpleEvaluator()
    return evaluator.evaluate(dataset)
