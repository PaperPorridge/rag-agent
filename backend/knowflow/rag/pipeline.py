"""
M2 · RAG 编排管道

把「分块 → 向量化 → 入库 → 检索 → 重排 → 生成」串成一条可复用的管道。

【为什么管道要独立于 API 层？】
  1. 可测试：管道能被单测直接调用，不经过 HTTP
  2. 可复用：Web / CLI / 定时任务 / 评估脚本共用一条链路，行为一致
  3. 可替换：换 embedding 或换检索策略，不动业务入口

【答案可信度的三件套】
  1. Prompt 约束: 只依据上下文回答，不足就说「资料中未提及」
  2. 结构化上下文: 每个片段带序号+来源（format_for_prompt）
  3. 引用后处理: 校验答案里的引用编号真实存在，防止模型编造引用
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from knowflow.core.llm import LLMClient, LLMResponse
from knowflow.core.prompts import registry
from knowflow.rag.chunking import BaseChunker, Chunk, RecursiveChunker
from knowflow.rag.embeddings import EmbeddingProvider, create_embedder
from knowflow.rag.retrieval import (
    BM25Index,
    HybridRetriever,
    Reranker,
    RetrievedChunk,
    format_for_prompt,
)
from knowflow.rag.vectorstore import VectorStore, create_vector_store

log = structlog.get_logger(__name__)


@dataclass
class RAGAnswer:
    """一次 RAG 问答的完整产物 —— 不只是答案文本。"""

    answer: str
    question: str
    sources: list[str] = field(default_factory=list)  # [来源: x.pdf#0] 清单
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    llm: LLMResponse | None = None
    citations_valid: bool = True
    citation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "question": self.question,
            "sources": self.sources,
            "n_retrieved": len(self.retrieved),
            "citations_valid": self.citations_valid,
            "citation_errors": self.citation_errors,
            "tokens": self.llm.usage.total_tokens if self.llm else 0,
            "cost_cny": self.llm.usage.cost_cny if self.llm else 0.0,
        }


class RAGPipeline:
    """
    RAG 主管道。

    用法:
        rag = RAGPipeline(llm=LLMClient())
        rag.ingest("差旅报销需在30天内提交。", source="报销制度.pdf")
        ans = rag.ask("报销有时间限制吗？")
        print(ans.answer, ans.sources)
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        embedder: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
        chunker: BaseChunker | None = None,
        reranker: Reranker | None = None,
        collection: str = "knowflow",
        store_location: str = ":memory:",
    ) -> None:
        self.llm = llm or LLMClient()
        self.embedder = embedder or create_embedder("hashing-256")
        self.store = store or create_vector_store(
            dim=self.embedder.dimension, location=store_location, collection=collection
        )
        self.chunker = chunker or RecursiveChunker(chunk_size=512, overlap=64)
        self.bm25 = BM25Index()
        self.reranker = reranker
        self.retriever = HybridRetriever(
            embedder=self.embedder,
            store=self.store,
            bm25=self.bm25,
            reranker=self.reranker,
        )
        self._all_chunks: list[Chunk] = []  # BM25 是内存索引，需要自己维护全集

    # -- 索引侧 -----------------------------------------------------------
    def ingest(self, text: str, source: str, metadata: dict[str, Any] | None = None) -> int:
        """单文档入库：分块 → 向量化 → 写向量库 → 重建 BM25。"""
        return self.ingest_batch([(text, source)], metadata)

    def ingest_batch(
        self,
        docs: list[tuple[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        批量入库。docs: [(text, source), ...]

        工程化要点：
          - 同一 source 重复入库前先删除旧块（文档更新场景，幂等）
          - 批量向量化（一次 embed_documents 调用，比循环快一个数量级）
          - BM25 在生产应放在 Elasticsearch/OpenSearch，这里内存实现只适合演示
        """
        sources = {src for _, src in docs}
        for src in sources:
            self.store.delete_by_source(src)
            self._all_chunks = [c for c in self._all_chunks if c.source != src]

        chunks: list[Chunk] = []
        for text, source in docs:
            chunks.extend(self.chunker.split(text, source, metadata))

        if not chunks:
            return 0

        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.store.upsert(chunks, vectors)
        self._all_chunks.extend(chunks)
        self.bm25.index(self._all_chunks)

        log.info("ingest_done", n_chunks=len(chunks), n_docs=len(docs))
        return len(chunks)

    # -- 查询侧 -----------------------------------------------------------
    def ask(
        self,
        question: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        use_rerank: bool = True,
    ) -> RAGAnswer:
        """一次 RAG 问答。返回答案 + 检索证据 + 引用校验结果。"""
        retrieved = self.retriever.retrieve(
            question, top_k=top_k, filters=filters, use_rerank=use_rerank
        )

        if not retrieved:
            return RAGAnswer(
                answer="知识库中没有找到相关资料，无法回答。",
                question=question,
                retrieved=[],
            )

        context = format_for_prompt(retrieved)
        system = registry.render("rag_qa", question=question, context=context)
        # rag_qa 模板已含 question/context，这里把它整体作为 system，
        # user 再重复一次问题 —— 让模型明确「要回答的是这个问题」
        resp = self.llm.chat(question, system=system)

        sources = [r.citation for r in retrieved]
        valid, errors = validate_citations(resp.content, sources)

        return RAGAnswer(
            answer=resp.content,
            question=question,
            sources=sources,
            retrieved=retrieved,
            llm=resp,
            citations_valid=valid,
            citation_errors=errors,
        )

    def search_only(self, question: str, top_k: int = 5, use_rerank: bool = True) -> list[RetrievedChunk]:
        """只检索不生成 —— 调试检索质量、做检索层评估时用（省 token）。"""
        return self.retriever.retrieve(question, top_k=top_k, use_rerank=use_rerank)


def validate_citations(answer: str, valid_sources: list[str]) -> tuple[bool, list[str]]:
    """
    校验答案中的引用是否真实存在于检索结果里。

    模型有时会编造不存在的引用编号 —— 这是幻觉的一种，必须在出口拦截。
    返回 (是否全部合法, 错误清单)。
    """
    # 匹配 [来源: xxx] 或 [来源: xxx#n]
    cited = re.findall(r"\[来源:\s*([^\]]+)\]", answer)
    if not cited:
        return True, []  # 没写引用不算错（Prompt 要求写，但不强制断言）

    valid_set = {s.strip() for s in valid_sources}
    # 兼容只写文件名不写 #编号 的情况
    valid_files = {s.split("#")[0].strip() for s in valid_sources}

    errors = []
    for c in cited:
        c = c.strip()
        if c not in valid_set and c.split("#")[0].strip() not in valid_files:
            errors.append(f"引用不存在: {c}")
    return len(errors) == 0, errors
