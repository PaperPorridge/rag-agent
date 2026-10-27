"""
M7 · 评估执行器

【这个文件解决什么问题？】
  指标（metrics.py）只负责「有了 answer + contexts 怎么打分」。
  但一个真正能写进简历的评估，还得回答：
    · 语料从哪来、金标准谁写的
    · 换一套参数（embedding / rerank / top_k）指标怎么变
    · 跑完的结果存哪、能不能复现
  runner.py 就是把这些串起来的执行层。

【为什么每个参数组合要各建一套 RAGPipeline？】
  1. embedding 维度不同（hashing-256 是 256 维，bge-m3 是 1024 维），
     不能写进同一个 Qdrant collection —— 会被维度校验拒掉。
  2. 改 chunk_size 会改变分块结果，必须重新入库。
  3. 同一个 pipeline 上开开关关（先无 rerank 再有 rerank）会让两次
     检索共享同一份索引，看起来公平，其实 chunk 已经互相污染过。
  所以：一套配置 = 一个独立 collection = 一次干净入库。

【为什么用 rag.ask() 而不是 answer_with_routing()？】
  评估要喂给指标的是「完整 chunk 文本」。
  answer_with_routing() 返回的 retrieved 是截断到 220 字的摘要（给前端看的），
  拿它算 Context Recall 会把「事实确实检索到了但被截断」误判成漏检。
  融合链路（Router/Agent）是另一层的事，本模块评估的是 RAG 核心。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import structlog

from knowflow.core.llm import LLMClient, LLMResponse, UsageRecord
from knowflow.eval.dataset import CORPUS_DIR, EvalCase, EvalDataset
from knowflow.eval.metrics import EvalResult, RagasEvaluator, SimpleEvaluator

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    """
    一套评估参数。

    注意 rerank 默认是 False，而且必须显式注入 Reranker ——
    光把 use_rerank 设成 True 什么也不会发生（见 RAGPipeline 的边界）。
    """

    name: str
    embedder_model: str = "hashing-256"
    use_rerank: bool = False
    top_k: int = 5
    chunk_size: int = 512
    overlap: int = 64
    device: str = "cpu"
    # 评估默认用进程内存向量库：每轮重新入库，跑完即弃。
    # 一是保证隔离（不会看到上一轮残留），二是让评估不依赖 Qdrant 是否在跑 ——
    # 离线 CI / 笔记本上也能出报告。要连真库传 qdrant 地址即可。
    store_location: str = ":memory:"

    @property
    def collection(self) -> str:
        """独立 collection 名。不同维度的向量绝不能共用一个 collection。"""
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", self.name).strip("-").lower()
        return f"eval-{slug}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "embedder_model": self.embedder_model,
            "use_rerank": self.use_rerank,
            "top_k": self.top_k,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "collection": self.collection,
            "store_location": self.store_location,
        }


# 预置参数组合 —— 报告默认跑这几套，凑出「embedding / rerank / top_k」三个轴
PRESET_CONFIGS: dict[str, EvalConfig] = {
    # 零依赖基线：hashing-256 + 无 rerank。不下载任何模型，秒级出结果。
    # 作用是给生产配置一个「下限参照」——提升幅度才是真数字。
    "baseline": EvalConfig(name="baseline", embedder_model="hashing-256", use_rerank=False, top_k=5),
    # 只换 embedding：看语义向量相对哈希向量的提升
    "bge-m3": EvalConfig(name="bge-m3", embedder_model="BAAI/bge-m3", use_rerank=False, top_k=5),
    # 生产配置：bge-m3 + CrossEncoder 精排。与 bge-m3 的差值就是 rerank 的贡献
    "production": EvalConfig(name="production", embedder_model="BAAI/bge-m3", use_rerank=True, top_k=5),
    # top_k 轴
    "topk3": EvalConfig(name="topk3", embedder_model="BAAI/bge-m3", use_rerank=True, top_k=3),
    "topk10": EvalConfig(name="topk10", embedder_model="BAAI/bge-m3", use_rerank=True, top_k=10),
}
DEFAULT_CONFIGS = ["baseline", "bge-m3", "production", "topk3"]


# ---------------------------------------------------------------------------
# 离线假 LLM —— CI 冒烟测试用，不联网、不花钱、结果确定
# ---------------------------------------------------------------------------
class FakeLLM:
    """
    可预测的假 LLM。

    【为什么评估模块需要它？】
      评估链路本身要跑通（入库 → 检索 → 生成 → 打分）才算有测试守卫。
      但 CI 里真调 MiMo 又慢又花钱还可能网络抖动导致假失败。
      FakeLLM 把「生成」这一环换成查表，剩下入库/检索/打分全是真代码。

    用法:
        llm = FakeLLM({q: "标准答案"})
        runner = EvalRunner(llm=llm)
    """

    def __init__(self, responses: dict[str, str] | None = None, default: str = "") -> None:
        self.responses = responses or {}
        self.default = default
        self.calls: list[str] = []

    def chat(
        self,
        user: str,
        *,
        system: str | None = None,
        history: list | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls.append(user)
        content = self.responses.get(user) or self.default or f"（离线假答案）{user}"
        return LLMResponse(content=content, model="fake-llm", usage=UsageRecord(model="fake-llm"))

    def chat_structured(self, user, schema_model, *, system=None, max_tokens=None):
        raise NotImplementedError("FakeLLM 只支持 chat，评估链路用不到结构化输出")

    def chat_with_tools(self, *args, **kwargs):
        raise NotImplementedError("FakeLLM 不支持工具调用")


# ---------------------------------------------------------------------------
# 语料加载
# ---------------------------------------------------------------------------
def load_corpus(
    corpus_dir: str | Path | None = None,
    include_distractors: bool = False,
) -> list[tuple[str, str]]:
    """
    读种子语料，返回 [(文本, source 文件名), ...]。

    这 5 份文档是手写的企业制度文本，和 golden.jsonl 里的金标准一一对应 ——
    评估的可信度来自「问题、标准答案、语料」三者互相咬得住。

    include_distractors=True 时额外并入 <dir>/distractors/ 下的干扰文档：
    它们与种子共用词汇但数值不同，用来把评测从「8 个 chunk 的天花板」拉下来，
    让不同检索配置在 Precision 上拉开差距（默认关，CI 冒烟不引入噪声）。
    """
    d = Path(corpus_dir) if corpus_dir else CORPUS_DIR
    docs: list[tuple[str, str]] = []
    for p in sorted(d.glob("*.md")):
        docs.append((p.read_text(encoding="utf-8"), p.name))
    if include_distractors:
        dist_dir = d / "distractors"
        if dist_dir.is_dir():
            for p in sorted(dist_dir.glob("*.md")):
                docs.append((p.read_text(encoding="utf-8"), p.name))
    if not docs:
        raise FileNotFoundError(f"语料目录为空: {d}")
    log.info("corpus_loaded", dir=str(d), n=len(docs), distractors=include_distractors)
    return docs


# ---------------------------------------------------------------------------
# 执行器
# ---------------------------------------------------------------------------
@dataclass
class EvalRunResult:
    """一套参数跑完的完整产物 —— 报告和 EvalRun 表都从这里取数。"""

    config: EvalConfig
    result: EvalResult
    dataset: EvalDataset
    elapsed_s: float = 0.0
    n_ingested: int = 0

    @property
    def evaluator(self) -> str:
        return self.result.evaluator

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "metrics": self.result.to_dict(),
            "n_ingested": self.n_ingested,
            "elapsed_s": round(self.elapsed_s, 2),
        }


def clone_dataset(dataset: EvalDataset) -> EvalDataset:
    """
    深拷贝一份「干净」数据集（answer/contexts 留空）。

    【为什么要拷贝？】
      每跑一套参数都会往 case 里写 answer/contexts。
      直接复用同一个对象，第二套配置会看到第一套残留的结果，
      对比出来的数字是假的。
    """
    return EvalDataset(
        name=dataset.name,
        cases=[
            EvalCase(
                question=c.question,
                ground_truth=c.ground_truth,
                source=c.source,
                expected_facts=list(c.expected_facts),
                metadata=dict(c.metadata),
            )
            for c in dataset
        ],
    )


class EvalRunner:
    """
    评估执行器。

    用法:
        runner = EvalRunner()
        report = runner.run(PRESET_CONFIGS["production"], load_golden())
        print(report.result.summary())
    """

    def __init__(self, llm: Any | None = None, device: str | None = None) -> None:
        self.llm = llm or LLMClient()
        self.device = device

    # -- 建管道 ---------------------------------------------------------
    def build_pipeline(self, config: EvalConfig):
        """按配置建一套干净的 RAGPipeline。rerank 必须显式注入，否则等于没开。"""
        from knowflow.rag.chunking import RecursiveChunker
        from knowflow.rag.embeddings import create_embedder
        from knowflow.rag.pipeline import RAGPipeline
        from knowflow.rag.retrieval import Reranker

        embedder = create_embedder(config.embedder_model, config.device)
        # 【关键】Reranker 不注入 = use_rerank 形同虚设，这里显式建
        reranker = Reranker(device=config.device) if config.use_rerank else None

        return RAGPipeline(
            llm=self.llm,
            embedder=embedder,
            chunker=RecursiveChunker(chunk_size=config.chunk_size, overlap=config.overlap),
            reranker=reranker,
            collection=config.collection,
            store_location=config.store_location,
        )

    # -- 跑一轮 ---------------------------------------------------------
    def run(
        self,
        config: EvalConfig,
        dataset: EvalDataset,
        evaluator: Any | None = None,
        corpus_dir: str | Path | None = None,
        include_distractors: bool = False,
    ) -> EvalRunResult:
        """
        跑一套参数：入库 → 逐条 ask → 回填 answer/contexts → 打分。

        evaluator 默认 SimpleEvaluator（免费）。
        要跑 ragas 传 RagasEvaluator()。
        """
        evaluator = evaluator or SimpleEvaluator()
        cases = clone_dataset(dataset)

        t0 = time.perf_counter()
        rag = self.build_pipeline(config)
        n_ingested = rag.ingest_batch(load_corpus(corpus_dir, include_distractors))
        log.info("eval_ingested", config=config.name, n_chunks=n_ingested)

        for case in cases:
            result = rag.ask(case.question, top_k=config.top_k, use_rerank=config.use_rerank)
            case.answer = result.answer
            # 【关键】取完整 chunk 文本，不是 API 层截断的 snippet
            case.contexts = [r.chunk.text for r in result.retrieved]

        scored = evaluator.evaluate(cases)
        elapsed = time.perf_counter() - t0

        run = EvalRunResult(
            config=config,
            result=scored,
            dataset=cases,
            elapsed_s=elapsed,
            n_ingested=n_ingested,
        )
        log.info("eval_run_done", config=config.name, elapsed_s=round(elapsed, 1), **scored.to_dict())
        return run

    # -- 只测检索层 -----------------------------------------------------
    def run_retrieval_only(
        self,
        config: EvalConfig,
        dataset: EvalDataset,
        corpus_dir: str | Path | None = None,
        include_distractors: bool = False,
    ) -> EvalResult:
        """
        只跑检索不生成 —— 省 token，专看 Context Precision / Recall。
        调参时先用这个，定好 top_k 再跑完整评估。
        """
        cases = clone_dataset(dataset)
        rag = self.build_pipeline(config)
        rag.ingest_batch(load_corpus(corpus_dir, include_distractors))

        for case in cases:
            hits = rag.search_only(case.question, top_k=config.top_k, use_rerank=config.use_rerank)
            case.contexts = [r.chunk.text for r in hits]
            case.answer = case.ground_truth  # 检索层评估不需要真答案

        return SimpleEvaluator().evaluate(cases)
