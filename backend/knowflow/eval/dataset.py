"""
M7 · 评估数据集

【为什么需要数据集？】
  没有测试集的 RAG 系统，就像没有单元测试的代码——跑起来能用，
  但改了参数你不知道效果变好还是变坏了。

【数据集的三个来源】
  1. 人工标注    最贵最准——领域专家写问题+标准答案
  2. 半自动生成  最常用——LLM 从文档生成 QA 对，人工审核
  3. 线上日志    最有价值——真实用户的问题，反映真实分布

【一条测试用例包含什么？】
  question       用户问题
  ground_truth   标准答案（人工写/审过的）
  context        应该被检索到的文档片段（可选，用于算 Context Recall）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# 金标准语料目录：随包发布，评估报告直接读这里
CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
DISTRACTOR_DIR = CORPUS_DIR / "distractors"
GOLDEN_PATH = CORPUS_DIR / "golden.jsonl"
GOLDEN_PARAPHRASE_PATH = CORPUS_DIR / "golden_paraphrase.jsonl"


@dataclass
class EvalCase:
    """一条评估用例。"""

    question: str
    ground_truth: str
    source: str = ""                  # 对应哪个文档
    # 理想情况下应该被检索到的关键信息（用于 Context Recall）
    expected_facts: list[str] = field(default_factory=list)
    # 运行时填充：RAG 系统的输出
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "source": self.source,
            "answer": self.answer,
            "contexts": self.contexts,
            "expected_facts": self.expected_facts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalCase:
        """从 JSONL 行还原。answer/contexts 是运行时产物，磁盘上通常为空。"""
        return cls(
            question=data["question"],
            ground_truth=data.get("ground_truth", ""),
            source=data.get("source", ""),
            expected_facts=list(data.get("expected_facts", [])),
            answer=data.get("answer", ""),
            contexts=list(data.get("contexts", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class EvalDataset:
    """一组评估用例。"""

    name: str
    cases: list[EvalCase] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def __getitem__(self, idx: int) -> EvalCase:
        return self.cases[idx]

    def add(self, case: EvalCase) -> None:
        self.cases.append(case)

    def summary(self) -> str:
        return f"数据集 '{self.name}': {len(self.cases)} 条用例"


def build_demo_dataset() -> EvalDataset:
    """
    构建演示数据集 —— 从 KnowFlow 的语料自动生成 QA 对。

    生产里这个函数应该是：
      1. 用 LLM 从文档自动生成 QA（半自动）
      2. 人工审核修正
      3. 存到文件/数据库

    演示里直接手写，保证每条都有明确的 ground_truth。
    """
    ds = EvalDataset(name="knowflow-demo-v1")

    ds.add(EvalCase(
        question="差旅报销有时间限制吗？",
        ground_truth="需在出差结束后30天内提交报销单，逾期需部门总监特批。",
        source="差旅报销制度.md",
        expected_facts=["30天内提交", "逾期需特批"],
    ))
    ds.add(EvalCase(
        question="一线城市住宿标准是多少？",
        ground_truth="一线城市每晚500元，二线城市每晚350元。",
        source="差旅报销制度.md",
        expected_facts=["一线城市500元", "二线城市350元"],
    ))
    ds.add(EvalCase(
        question="公司开发机型号是什么？",
        ground_truth="ThinkPad-X1-Carbon-Gen12，标配内存32GB。",
        source="IT设备清单.md",
        expected_facts=["ThinkPad-X1-Carbon-Gen12", "32GB"],
    ))
    ds.add(EvalCase(
        question="单笔费用超过多少需要事前审批？",
        ground_truth="单笔金额超过5000元需附上事前审批邮件截图。",
        source="费用核销SOP.md",
        expected_facts=["5000元", "事前审批"],
    ))
    ds.add(EvalCase(
        question="年假有多少天？",
        ground_truth="入职满一年享有10天年假，满三年15天。",
        source="年假规则.md",
        expected_facts=["满一年10天", "满三年15天"],
    ))

    log.info("demo_dataset_built", n=len(ds))
    return ds


# ---------------------------------------------------------------------------
# JSONL 持久化 —— 金标准要进版本控制，评估结果才可复现
# ---------------------------------------------------------------------------
def load_dataset(path: str | Path) -> EvalDataset:
    """
    从 JSONL 读数据集（一行一条用例）。

    【为什么用 JSONL 而不是 JSON 数组？】
      金标准是人工逐条写的，一行一条便于 diff 审阅、按行增删，
      也便于 git 合并（改一条只动一行）。
    """
    p = Path(path)
    cases: list[EvalCase] = []
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases.append(EvalCase.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"{p}:{lineno} 不是合法的用例行: {exc}") from exc

    ds = EvalDataset(name=p.stem, cases=cases)
    log.info("dataset_loaded", path=str(p), n=len(ds))
    return ds


def save_dataset(dataset: EvalDataset, path: str | Path) -> Path:
    """写出 JSONL。用于保存跑完后带 answer/contexts 的快照，便于排障复盘。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for case in dataset:
            f.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
    log.info("dataset_saved", path=str(p), n=len(dataset))
    return p


def load_golden() -> EvalDataset:
    """
    读随包发布的金标准数据集（25 条手写 QA，含 5 条计算题）。

    这是 M7 的评估基准：改了 embedding / rerank / top_k 之后跑同一份，
    数字才能横向比。
    """
    return load_dataset(GOLDEN_PATH)


def load_golden_paraphrase() -> EvalDataset:
    """
    读「近义改写版」金标准 —— 同一批问题换成口语化、不逐字复刻原文的措辞。

    逐字版的问题里含「住宿标准」「餐饮补贴」这类原文原词，BM25 直接命中，
    Context Recall 恒为 1.0，测不出语义检索的价值。改写后词法召回失效，
    Recall 才有区分度（hashing < bge-m3 < bge-m3+rerank）。
    """
    return load_dataset(GOLDEN_PARAPHRASE_PATH)
