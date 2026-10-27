"""
M2 · 文本向量化（Embedding）

【原理一句话】
把文本映射成高维向量（如 512/1024 维），语义相近的文本在向量空间里距离近。
相似度常用余弦：cos(a,b) = a·b / (|a||b|)，取值 [-1,1]，越大越像。

【选型（中文场景）】
  BAAI/bge-m3          : 1024 维，稠密+稀疏+多向量，中英皆强，8K 上下文 → 生产推荐
  BAAI/bge-small-zh-v1.5: 512 维，~100MB，CPU 秒级 → 开发/演示
  text-embedding-3     : OpenAI，效果好但数据出境 + 按量计费 → 备选

【工程铁律】
  1. 索引和查询必须用同一个 embedding 模型 —— 向量空间不同 = 检索完全失效
  2. 换模型 = 必须重建全部向量索引（没有增量迁移这回事）
  3. 向量维度决定存储：100 万文档 × 1024 维 × 4 字节 ≈ 4GB（不含图索引开销）

【本模块设计】
  EmbeddingProvider 是抽象接口 —— 业务代码只依赖接口。
  换 bge-m3 / OpenAI / 私有模型时，只改配置不改业务（呼应 M0 的封装思想）。

  HashingEmbedder  : 零依赖、确定性、可跑单测。不理解语义，只保证链路能跑通。
  HFEmbedder       : sentence-transformers 真实语义向量（需安装 [rag] 依赖）。
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from functools import lru_cache

import structlog

log = structlog.get_logger(__name__)


class EmbeddingProvider(ABC):
    """向量化接口。索引侧和查询侧必须用同一个实例（同一个模型）。"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度。建库时用来声明 schema。"""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型标识。换模型要重建索引 —— 所以必须能拿到这个名字去校验。"""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文档（索引侧）。默认应走批处理优化。"""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """向量化查询（查询侧）。某些模型（bge 系列）查询侧有专门指令前缀。"""

    def embed_query_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


# ---------------------------------------------------------------------------
# 1) 零依赖实现：哈希向量（单测 / 离线冒烟）
# ---------------------------------------------------------------------------
class HashingEmbedder(EmbeddingProvider):
    """
    基于 token 哈希的确定性向量。

    注意：这**不是**语义向量 —— 相同词袋才相似，不懂同义词。
    它的存在价值：
      - 单元测试不下载几 GB 的模型
      - 链路冒烟（分块→入库→检索）不依赖网络
      - 演示「词法相似」与「语义相似」的差别（M2 demo 会用到）

    实现：把每个 token 哈希到 dim 个桶，出现一次加一次，最后 L2 归一化。
    归一化后点积 = 余弦相似度，与真实 embedding 的度量一致。
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return f"hashing-{self._dim}"

    def _tokenize(self, text: str) -> list[str]:
        import re

        # 中文按字、英文按词 —— 简化版分词，够哈希用
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+|[一-鿿]", text)
        return tokens

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in self._tokenize(text):
            h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:8], "big")
            idx = h % self._dim
            sign = 1.0 if (h >> 8) & 1 == 0 else -1.0  # 有符号哈希，降低碰撞偏置
            vec[idx] += sign
        # L2 归一化
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


# ---------------------------------------------------------------------------
# 2) 真实实现：HuggingFace sentence-transformers
# ---------------------------------------------------------------------------
class HFEmbedder(EmbeddingProvider):
    """
    sentence-transformers 语义向量（bge / m3e 等）。

    要点：
      - bge 系列**查询侧**要加指令前缀（中文查询: "为这个句子生成表示以用于检索相关文章："）
        文档侧不加。这是 bge 论文的训练方式，忽略它检索效果会掉。
      - normalize_embeddings=True → 之后用点积即可等价余弦
      - 批量编码 embed_documents 一次调用，比循环 embed_query 快一个数量级

    首次使用会从 HuggingFace 下载模型（bge-small-zh 约 100MB）。
    """

    # bge 系列查询指令前缀（中文）
    BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cpu",
        query_prefix: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "需要安装 RAG 依赖: pip install -e '.[rag]'"
            ) from exc

        self._model_name = model_name
        self._model = self._load(model_name, device)
        # sentence-transformers 3.x 起 get_sentence_embedding_dimension() 标记废弃，
        # 新名是 get_embedding_dimension()。两个都试，避免新旧版本各炸一次。
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        if get_dim is None:
            get_dim = self._model.get_sentence_embedding_dimension
        self._dim = int(get_dim())
        self._query_prefix = query_prefix
        log.info("embedder_loaded", model=model_name, dim=self._dim, device=device)

    @property
    def dimension(self) -> int:
        return self._dim

    @staticmethod
    def _load(model_name: str, device: str):
        """
        加载 sentence-transformers 模型：优先本地缓存，没有才联网下载。

        【为什么先试 local_files_only？】
          按模型 ID 加载时，huggingface_hub 会先联网解析 revision/文件清单。
          断网、代理不通、Hub 超时（WinError 10060）都会让这步失败，
          之后 AutoProcessor 直接抛 "Unrecognized processing class" ——
          报错指向模型文件缺失，实际是缓存完好、只是解析没走通。
          缓存已热的机器不该因为上不了网就起不来；反过来，
          先走网络则每次都要干等几十秒超时，冷启动体验极差。
          所以顺序是「本地优先，网络兜底」。
        """
        from sentence_transformers import SentenceTransformer

        try:
            return SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception as exc:
            log.info("embedder_download", model=model_name, reason=str(exc)[:120])
            return SentenceTransformer(model_name, device=device)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _prepare_query(self, text: str) -> str:
        if self._query_prefix is None:
            # bge 系列自动加前缀；其他模型保持原样
            if "bge" in self._model_name.lower() and "zh" in self._model_name.lower():
                return f"{self.BGE_QUERY_PREFIX}{text}"
            return text
        return f"{self._query_prefix}{text}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode(self._prepare_query(text), normalize_embeddings=True)
        return vec.tolist()


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _cached_hf(model_name: str, device: str) -> HFEmbedder:
    """HF 模型加载慢（几秒到几十秒），按 (model, device) 缓存。"""
    return HFEmbedder(model_name=model_name, device=device)


def create_embedder(
    model_name: str = "hashing-256",
    device: str = "cpu",
) -> EmbeddingProvider:
    """
    统一构造入口。

    model_name 以 "hashing" 开头 → 零依赖实现
    否则 → sentence-transformers 加载
    """
    if model_name.startswith("hashing"):
        dim = int(model_name.split("-")[1]) if "-" in model_name else 256
        return HashingEmbedder(dim=dim)
    return _cached_hf(model_name, device)
