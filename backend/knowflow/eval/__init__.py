"""M7 · RAG 评估体系：指标计算 + 数据集管理 + 评估执行 + 评估报告。"""

from knowflow.eval.dataset import (
    EvalCase,
    EvalDataset,
    build_demo_dataset,
    load_dataset,
    load_golden,
    load_golden_paraphrase,
    save_dataset,
)
from knowflow.eval.metrics import (
    EvalResult,
    RagasEvaluator,
    SimpleEvaluator,
    evaluate_rag,
)
from knowflow.eval.runner import (
    DEFAULT_CONFIGS,
    PRESET_CONFIGS,
    EvalConfig,
    EvalRunResult,
    EvalRunner,
    FakeLLM,
    load_corpus,
)

__all__ = [
    # 数据集
    "EvalCase",
    "EvalDataset",
    "build_demo_dataset",
    "load_dataset",
    "load_golden",
    "load_golden_paraphrase",
    "save_dataset",
    "load_corpus",
    # 指标
    "EvalResult",
    "SimpleEvaluator",
    "RagasEvaluator",
    "evaluate_rag",
    # 执行
    "EvalConfig",
    "EvalRunResult",
    "EvalRunner",
    "FakeLLM",
    "PRESET_CONFIGS",
    "DEFAULT_CONFIGS",
]
