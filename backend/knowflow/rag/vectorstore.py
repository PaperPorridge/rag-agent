"""
M2 · 向量库（Qdrant）

【为什么不用 numpy 暴力算相似度？】
  100 万向量全量点积 = 每次查询都扫全库，毫秒级变秒级。
  向量库用 ANN（近似最近邻）换一点精度，换数量级的速度。

【HNSW —— 必懂概念】
  Hierarchical Navigable Small World：多层近邻图。
  查询从稀疏顶层「大跳」定位区域，再在密集底层「小爬」精确定位。
  复杂度 O(log N)，百万级毫秒响应。
  关键参数：
    m               每个节点的连边数（越大越准、越吃内存）
    ef_construction 建图搜索宽度（越大建得越准、越慢）
    查询时 ef        搜索宽度（越大越准越慢）

【选型对比】
  Qdrant   : Rust 实现，单机 Docker 一条命令起，元数据过滤强，自带 Web UI → 本项目
  Milvus   : 十亿级分布式，生态全，但依赖 etcd/MinIO，运维重 → 超大规模
  pgvector : PG 扩展，已有 PG 时零新增组件，但索引能力弱于专用库 → 中小规模
  Chroma   : 嵌入式，开发原型快，不适合生产

【本模块设计】
  VectorStore 是接口。QdrantVectorStore 支持 :memory:（无 Docker 也能跑 demo）
  和 URL（生产连真实 Qdrant）。换 pgvector/Milvus 只需再实现一个子类。
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import structlog

from knowflow.rag.chunking import Chunk

log = structlog.get_logger(__name__)


@dataclass
class Match:
    """一次检索命中。score 越大越相似（余弦相似度）。"""

    chunk: Chunk
    score: float


class VectorStore(ABC):
    """向量库抽象。业务只依赖这四个方法。"""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """写入/更新向量。返回写入条数。"""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[Match]:
        """向量近邻检索。filters 支持按元数据过滤（source/tenant 等）。"""

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def delete_by_source(self, source: str) -> int:
        """按来源文件删除（文档更新时先删后插）。"""


class QdrantVectorStore(VectorStore):
    """
    Qdrant 实现。

    location:
      ":memory:"        进程内内存模式 —— demo/单测用，无需 Docker
      "http://localhost:6333" 连接真实 Qdrant 服务

    collection_name 一个知识库一个 collection。维度必须与 embedding 一致。
    """

    def __init__(
        self,
        collection_name: str = "knowflow",
        dim: int = 256,
        location: str = ":memory:",
        distance: str = "Cosine",
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise ImportError("需要安装 RAG 依赖: pip install -e '.[rag]'") from exc

        self._collection = collection_name
        self._dim = dim
        self._client = QdrantClient(location=location)

        # 幂等建库：存在就跳过
        existing = {c.name for c in self._client.get_collections().collections}
        if collection_name not in existing:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            log.info("collection_created", name=collection_name, dim=dim, location=location)

    @property
    def client(self) -> Any:
        """暴露底层 client —— 混合检索要用它的 query API 做 BM25/过滤。"""
        return self._client

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks({len(chunks)}) 与 vectors({len(vectors)}) 数量不一致")
        if not chunks:
            return 0

        from qdrant_client.models import PointStruct

        points = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            # 稳定 ID：source + chunk_index 做哈希 → 重复入库天然幂等
            stable_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.source}#{chunk.chunk_index}"))
            payload = {
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata,
            }
            points.append(PointStruct(id=stable_id, vector=vec, payload=payload))

        self._client.upsert(collection_name=self._collection, points=points, wait=True)
        log.info("vector_upserted", count=len(points), collection=self._collection)
        return len(points)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[Match]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        qdrant_filter = None
        if filters:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()
                ]
            )

        results = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
            with_payload=True,
        ).points

        matches = []
        for p in results:
            payload = p.payload or {}
            matches.append(
                Match(
                    chunk=Chunk(
                        text=payload.get("text", ""),
                        source=payload.get("source", "unknown"),
                        chunk_index=int(payload.get("chunk_index", 0)),
                        metadata={
                            k: v
                            for k, v in payload.items()
                            if k not in {"text", "source", "chunk_index"}
                        },
                    ),
                    score=float(p.score),
                )
            )
        return matches

    def count(self) -> int:
        return int(self._client.count(self._collection).count)

    def delete_by_source(self, source: str) -> int:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        # Qdrant 的 points_selector 必须是模型对象（FilterSelector），不能传裸 dict
        self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=source))]
                )
            ),
        )
        log.info("vector_deleted_by_source", source=source)
        return 0  # Qdrant 不返回删除数，需要可自行 count 前后差值


class InMemoryVectorStore(VectorStore):
    """
    纯 Python 暴力检索 —— 用来做对照实验：

      「为什么需要向量库？」

    小数据量（<1 万）时它完全够用，而且零依赖；
    数据量大了就是 O(N) 全扫，这时才需要 HNSW。
    在 demo 里我们用它对比检索速度，直观感受 ANN 的价值。
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim
        self._vectors: list[list[float]] = []
        self._chunks: list[Chunk] = []

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        for c, v in zip(chunks, vectors, strict=True):
            if len(v) != self._dim:
                raise ValueError(f"向量维度 {len(v)} != 声明维度 {self._dim}")
            self._chunks.append(c)
            self._vectors.append(v)
        return len(chunks)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[Match]:
        import math

        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb) if na and nb else 0.0

        scored: list[Match] = []
        for chunk, vec in zip(self._chunks, self._vectors, strict=True):
            if filters:
                if any(chunk.metadata.get(k) != v and getattr(chunk, k, None) != v for k, v in filters.items()):
                    # 简化过滤：允许匹配 metadata 或 dataclass 字段
                    if any(
                        (chunk.source if k == "source" else chunk.metadata.get(k)) != v
                        for k, v in filters.items()
                    ):
                        continue
            score = cosine(query_vector, vec)
            if score_threshold is not None and score < score_threshold:
                continue
            scored.append(Match(chunk=chunk, score=score))

        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._chunks)

    def delete_by_source(self, source: str) -> int:
        before = len(self._chunks)
        keep = [(c, v) for c, v in zip(self._chunks, self._vectors, strict=True) if c.source != source]
        self._chunks = [c for c, _ in keep]
        self._vectors = [v for _, v in keep]
        return before - len(self._chunks)


def create_vector_store(
    dim: int,
    location: str = ":memory:",
    collection: str = "knowflow",
) -> VectorStore:
    """location 以 'memory' 开头且不需要 Qdrant 时可用纯内存实现。"""
    if location == "python-memory":
        return InMemoryVectorStore(dim=dim)
    return QdrantVectorStore(collection_name=collection, dim=dim, location=location)
