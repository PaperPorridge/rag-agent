"""
M2 · 文档分块（Chunking）—— RAG 效果的第一决定因素

【为什么分块是第一位的？】
1. Embedding 模型有输入长度限制（bge-m3 约 8192 token）
2. 更关键：长文本向量化会「平均语义」—— 一篇文章讲 3 个主题，
   算出来的向量是三个主题的平均值，哪个主题都检索不准
3. 分块粒度决定「检索回来的东西能不能直接用」

【三种策略的取舍】
  固定长度 FixedChunker    : 按字符硬切。快，但会把句子/语义切断。
                             只配当「基线」，生产不要用。
  递归分割 RecursiveChunker: 按分隔符层级切（段落→句子→词）。
                             尊重文本结构，通用首选。LangChain 同款思路。
  语义分块 SemanticChunker : 按相邻句子 embedding 相似度突变处切。
                             质量最好但要额外算向量，慢。（本文件给接口，M9 可补）

【两个必须叠加的技巧】
  重叠 overlap   : 相邻块重叠 10~20%，防止关键句被切在边界后两边都残缺
  元数据 metadata: 文件名/章节/页码必须跟块走，否则引用溯源无从谈起

【块大小怎么定？】
  经验起点 512 token + 64 overlap，然后用评估集调（M7）。
  块太小 → 语义碎片化；块太大 → 语义稀释 + 浪费上下文。
  没有万能值，只有评估出来的值。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """一个分块。text + 元数据，是后续所有环节的最小单元。"""

    text: str
    source: str  # 来源文件名 —— 引用溯源的根基
    chunk_index: int  # 在文档内的序号
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """粗估 token 数。中文按 1 字 ≈ 1 token，英文按 1 词 ≈ 1.3 token。
        精确计数要 tiktoken，但分块阶段粗估够用且快。"""
        zh = len(re.findall(r"[一-鿿]", self.text))
        other = len(self.text) - zh
        return zh + max(other // 4, 0)


class BaseChunker(ABC):
    """分块器接口。所有分块器产出 Chunk 列表并保留元数据。"""

    @abstractmethod
    def split(self, text: str, source: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        ...

    def split_documents(self, docs: list[tuple[str, str]]) -> list[Chunk]:
        """docs: [(text, source), ...]，批量分块。"""
        chunks: list[Chunk] = []
        for text, source in docs:
            chunks.extend(self.split(text, source))
        return chunks


class FixedChunker(BaseChunker):
    """
    固定长度分块（基线实现，演示「为什么它不够」）。

    问题：按字符数硬切，会把一句话切成两半，检索回来的片段残缺。
    用途：快速验证链路是否通；作为对照组评估递归分块的收益。
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if overlap >= chunk_size:
            raise ValueError(f"overlap({overlap}) 必须小于 chunk_size({chunk_size})")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str, source: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        text = text.strip()
        if not text:
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        for i, start in enumerate(range(0, len(text), step)):
            piece = text[start : start + self.chunk_size]
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    text=piece,
                    source=source,
                    chunk_index=i,
                    metadata={**(metadata or {}), "chunker": "fixed", "start": start},
                )
            )
            if start + self.chunk_size >= len(text):
                break
        return chunks


class RecursiveChunker(BaseChunker):
    """
    递归分割：按分隔符优先级从大到小切，尽量不破坏文本结构。

    思路（与 LangChain RecursiveCharacterTextSplitter 同构）：
      1. 尝试用「段落」切
      2. 段落仍超长 → 用「句子」切
      3. 句子仍超长 → 用「词/字符」切
    这样小片段会尽量黏在自然边界上。

    加了 overlap：相邻块共享一段文本，防止关键句被切在边界两边都不完整。
    """

    # 分隔符优先级：段落 → 换行 → 句号 → 逗号 → 空格
    DEFAULT_SEPARATORS = ["\n\n", "\n", "。", ". ", "！", "？", "；", "; ", "，", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError(f"overlap({overlap}) 必须小于 chunk_size({chunk_size})")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def split(self, text: str, source: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        text = text.strip()
        if not text:
            return []

        pieces = self._recursive_split(text)
        pieces = [p.strip() for p in pieces if p.strip()]
        merged = self._merge_small(pieces)

        return [
            Chunk(
                text=piece,
                source=source,
                chunk_index=i,
                metadata={**(metadata or {}), "chunker": "recursive"},
            )
            for i, piece in enumerate(merged)
        ]

    # -- 核心递归 ---------------------------------------------------------
    def _recursive_split(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        for sep in self.separators:
            if sep == "":
                # 兜底：字符硬切
                return self._hard_split(text)
            if sep not in text:
                continue

            parts = text.split(sep)
            # 合并小块直到接近 chunk_size，避免切出一堆碎片
            chunks: list[str] = []
            current = ""
            for part in parts:
                candidate = f"{current}{sep}{part}" if current else part
                if len(candidate) <= self.chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # 单个 part 就超长 → 递归用更小的分隔符切
                    if len(part) > self.chunk_size:
                        chunks.extend(self._recursive_split(part))
                        current = ""
                    else:
                        current = part
            if current:
                chunks.append(current)
            return chunks

        return self._hard_split(text)

    def _hard_split(self, text: str) -> list[str]:
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    # -- 小块合并 + 重叠 ---------------------------------------------------
    def _merge_small(self, pieces: list[str]) -> list[str]:
        """把递归切出来的小块按尺寸合并，相邻块之间加 overlap。"""
        if not pieces:
            return []

        merged: list[str] = []
        current = ""
        for piece in pieces:
            if current and len(current) + len(piece) + 1 > self.chunk_size:
                merged.append(current)
                # overlap: 从上一块尾部带一段过来
                tail = current[-self.overlap :] if self.overlap > 0 else ""
                current = f"{tail}{piece}" if tail and not piece.startswith(tail) else piece
            else:
                current = f"{current}{piece}" if current else piece
        if current:
            merged.append(current)
        return merged


# ---------------------------------------------------------------------------
# 语义分块（接口占位，M9 实现）
# ---------------------------------------------------------------------------
class SemanticChunker(BaseChunker):
    """
    语义分块：计算相邻句子 embedding 相似度，在相似度突变处切分。

    优点：块内语义高度一致
    缺点：需要额外算 embedding，索引阶段慢 5~10 倍

    接口预留，M9 实现（需要 EmbeddingProvider 注入）。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("SemanticChunker 在 M9 实现。当前请用 RecursiveChunker。")

    def split(self, text: str, source: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        raise NotImplementedError


def naive_split_sentences(text: str) -> list[str]:
    """简单中英混合断句，供 SemanticChunker / 评估用。"""
    parts = re.split(r"(?<=[。！？.!?])\s*", text)
    return [p.strip() for p in parts if p.strip()]
