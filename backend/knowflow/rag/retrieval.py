"""
M2 · 检索：向量 + BM25 混合 + Rerank

【为什么必须混合检索？】
  向量检索擅长：语义相近 —— "如何请假" 命中 "休假申请流程"
  向量检索不擅长：精确匹配 —— 型号 RTX-4090、错误码 E1234、合同号 HT-2024-001
                  这些符号串在语义空间里没有"相近"概念，会被同义词淹没

  BM25 擅长：词面精确匹配（TF-IDF 的改进版，考虑词频饱和与文档长度归一）
  BM25 不擅长：同义词 —— 检索 "报销" 命中不了 "费用核销"

  → 向量负责语义召回，BM25 负责精确命中，两边都要。

【怎么融合两路结果？】
  ✗ 直接分数加权求和：向量分（余弦 0~1）与 BM25 分（0~几十）量纲不同，
    归一化后还得调权重 α，换一批数据就要重调，很脆。
  ✓ RRF（Reciprocal Rank Fusion，倒数排名融合）：
        RRF_score(d) = Σ  1 / (k + rank_i(d))      k 通常取 60
                      i
    只用排名、不用分数绝对值 → 鲁棒、几乎免调参、两路分数不可比也无所谓。
    一个文档在两边都靠前 → 融合分高。

【Rerank：精度的关键一跃】
  检索（bi-encoder 双塔）：问题、文档**分别**编码再算相似度。
    快、可预计算文档向量；但问题和文档不互相看，精度有限。
  重排（cross-encoder 交叉编码）：[CLS] 问题 [SEP] 文档 [SEP] 一起喂入。
    问题和文档深度交互，精度比双塔高 20~40%；但必须在线算、慢一个数量级。

  实践模式：粗排混合检索取 top-50（要召回）→ 精排交叉编码取 top-5（要精度）
  为什么 top-5 不是 top-1：多源交叉验证；但块越多噪声越多（lost-in-the-middle）。

【BM25 公式（了解即可，会讲直觉就行）】
  score(q,d) = Σ IDF(t) · (f(t,d)·(k1+1)) / (f(t,d) + k1·(1 - b + b·|d|/avgdl))
  直觉：词在本篇出现越多越相关（TF），但在语料里越稀有越重要（IDF），
       并按文档长度归一化，惩罚注水长文。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import structlog

from knowflow.rag.chunking import Chunk
from knowflow.rag.embeddings import EmbeddingProvider
from knowflow.rag.vectorstore import Match, VectorStore

log = structlog.get_logger(__name__)


@dataclass
class RetrievedChunk:
    """融合后的检索结果。stage 标记它经过了哪些环节 —— 排障很有用。"""

    chunk: Chunk
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float | None = None
    stage: str = "retrieved"

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf_score

    @property
    def citation(self) -> str:
        """引用标签，直接给 LLM 用：[来源: 文件#片段]"""
        return f"{self.chunk.source}#{self.chunk.chunk_index}"


# ---------------------------------------------------------------------------
# 1) BM25（自实现，零依赖）
# ---------------------------------------------------------------------------
class BM25Index:
    """
    简易 BM25 实现。生产可用 rank_bm25 库或 Elasticsearch/OpenSearch 的 BM25。

    自实现的价值：自查能讲清公式直觉，而不是「调了个库」。
    中英混合分词：英文按词、中文按字（生产用 jieba 分词会更好）。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[Chunk] = []
        self._doc_freqs: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+|[一-鿿]", text.lower())

    def index(self, chunks: list[Chunk]) -> None:
        self._docs = chunks
        self._doc_freqs = [Counter(self.tokenize(c.text)) for c in chunks]
        self._doc_lens = [sum(freqs.values()) for freqs in self._doc_freqs]
        self._avgdl = sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 0.0

        # IDF: ln((N - df + 0.5)/(df + 0.5) + 1) —— BM25 用的平滑版，避免负值
        n = len(chunks)
        df: Counter[str] = Counter()
        for freqs in self._doc_freqs:
            df.update(freqs.keys())
        self._idf = {
            term: math.log((n - d + 0.5) / (d + 0.5) + 1.0) for term, d in df.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[Match]:
        q_tokens = self.tokenize(query)
        scores: list[tuple[int, float]] = []

        for i, freqs in enumerate(self._doc_freqs):
            score = 0.0
            dl = self._doc_lens[i]
            for t in q_tokens:
                if t not in freqs:
                    continue
                tf = freqs[t]
                idf = self._idf.get(t, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                score += idf * (tf * (self.k1 + 1)) / denom
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        max_score = scores[0][1] if scores else 1.0
        return [
            Match(chunk=self._docs[i], score=s / max_score)  # 归一化到 0~1 便于观察
            for i, s in scores[:top_k]
        ]


# ---------------------------------------------------------------------------
# 2) RRF 融合
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    倒数排名融合。

    输入多路排名（每路内部有序），输出按 RRF 分数降序的去重结果。
    k=60 是论文推荐值：让第一名和第十名的差距不至于碾压第二路的贡献。
    """
    acc: dict[str, RetrievedChunk] = {}
    rrf_scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            key = f"{item.chunk.source}#{item.chunk.chunk_index}"
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in acc:
                acc[key] = item
            else:
                # 两路都命中时合并分数 —— 否则融合对象只保留先入的那一路分数
                merged = acc[key]
                merged.vector_score = merged.vector_score or item.vector_score
                merged.bm25_score = merged.bm25_score or item.bm25_score

    merged = []
    for key, item in acc.items():
        item.rrf_score = rrf_scores[key]
        item.stage = "fused"
        merged.append(item)
    merged.sort(key=lambda x: x.rrf_score, reverse=True)
    return merged


# ---------------------------------------------------------------------------
# 3) Reranker（交叉编码）
# ---------------------------------------------------------------------------
class Reranker:
    """
    交叉编码重排。

    bge-reranker 把 (query, doc) 拼在一起喂模型，输出相关性分数。
    比双塔精度高 20~40%，但每个候选都要一次前向 → 只对 top-N 精排。

    若未安装 sentence_transformers，降级为「词面重叠度」启发式 ——
    保证链路可跑，但 demo 会明确提示这不是真 rerank。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu") -> None:
        self._model_name = model_name
        self._impl = None
        try:
            from sentence_transformers import CrossEncoder

            self._impl = CrossEncoder(model_name, device=device)
            log.info("reranker_loaded", model=model_name)
        except ImportError:
            log.warning("reranker_fallback", reason="sentence_transformers 未安装，降级为词面重叠度")

    @property
    def available(self) -> bool:
        return self._impl is not None

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        if not candidates:
            return []

        if self._impl is not None:
            pairs = [(query, c.chunk.text) for c in candidates]
            scores = self._impl.predict(pairs)
            for c, s in zip(candidates, scores, strict=True):
                c.rerank_score = float(s)
        else:
            # 降级：Jaccard 词面重叠（仅保证链路可跑，不代表语义能力）
            q_tokens = set(BM25Index.tokenize(query))
            for c in candidates:
                d_tokens = set(BM25Index.tokenize(c.chunk.text))
                inter = len(q_tokens & d_tokens)
                union = len(q_tokens | d_tokens) or 1
                c.rerank_score = inter / union

        for c in candidates:
            c.stage = "reranked"
        candidates.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# 4) 混合检索器 —— 对外统一入口
# ---------------------------------------------------------------------------
class HybridRetriever:
    """
    混合检索器：向量 + BM25 → RRF 融合 → 可选 Rerank。

    用法:
        retriever = HybridRetriever(embedder, store, bm25, reranker)
        results = retriever.retrieve("RTX-4090 显存多大", top_k=5)

    设计要点：
      - 检索阶段取 candidate_k（默认 50）：要召回率，不怕有噪声
      - 精排后取 top_k（默认 5）：要精度、控成本
      - 支持 metadata filters：多租户场景按 tenant/source 过滤（权限隔离在检索层做！）
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        bm25: BM25Index | None = None,
        reranker: Reranker | None = None,
        candidate_k: int = 50,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.reranker = reranker
        self.candidate_k = candidate_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        use_rerank: bool = True,
    ) -> list[RetrievedChunk]:
        # -- 路 1: 向量检索（语义召回）--
        q_vec = self.embedder.embed_query(query)
        vector_matches = self.store.search(q_vec, top_k=self.candidate_k, filters=filters)
        vector_rank = [
            RetrievedChunk(chunk=m.chunk, vector_score=m.score, stage="vector")
            for m in vector_matches
        ]

        # -- 路 2: BM25（精确匹配）--
        bm25_rank: list[RetrievedChunk] = []
        if self.bm25 is not None:
            # 注意：BM25 是全内存索引，多租户过滤要先筛 chunk 集合
            bm25_matches = self.bm25.search(query, top_k=self.candidate_k)
            for m in bm25_matches:
                if filters and any(
                    (m.chunk.source if k == "source" else m.chunk.metadata.get(k)) != v
                    for k, v in filters.items()
                ):
                    continue
                bm25_rank.append(
                    RetrievedChunk(chunk=m.chunk, bm25_score=m.score, stage="bm25")
                )

        # -- 融合: RRF（用排名，不用分数绝对值）--
        rankings = [r for r in (vector_rank, bm25_rank) if r]
        fused = reciprocal_rank_fusion(rankings)

        # -- 精排: 交叉编码 --
        if use_rerank and self.reranker is not None and fused:
            return self.reranker.rerank(query, fused, top_k=top_k)

        return fused[:top_k]


def format_for_prompt(results: list[RetrievedChunk]) -> str:
    """
    把检索结果组装成带序号和来源的上下文文本。

    工程化要点：没有来源标签，LLM 就无法做引用溯源 —— 引用格式在 Prompt 里
    约定为 [来源: 文件名#片段号]，这里必须提供同样的标识。
    """
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[片段 {i} | 来源: {r.citation}]\n{r.chunk.text}")
    return "\n\n".join(lines)
