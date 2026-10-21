"""
文档解析 —— 原始字节 → 纯文本。

【职责边界（诚实声明）】
  支持：PDF（文本层）、.txt、.md —— 三大类覆盖企业知识库 80% 的入库来源。
  不支持：扫描版 PDF 的 OCR、Word/Excel/PPT。
  碰到不支持的格式**明确报错拒收**，绝不静默丢弃或返回空文本 ——
  「入库失败」比「入库了但内容是空的」好排查一百倍。

【为什么 PDF 只抽文本层？】
  pypdf 读的是 PDF 里内嵌的文字流。扫描版 PDF 每页是一张图，
  文字流是空的 —— 这种要走 OCR（PaddleOCR/tesseract）流水线，
  准确率/成本/延迟都是另一个量级的事，不该和文本层解析混在一起。
  本模块对扫描版给出明确错误码，让上游把文档路由到 OCR 队列（后续切片）。
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)

# 允许的扩展名 → 解析器
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".text"}

# 单文件大小上限（字节）。超过直接拒收，不进解析流程。
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


class ParseError(Exception):
    """解析失败。detail 面向用户，可直接出现在 API 错误信息里。"""

    def __init__(self, detail: str, code: str = "parse_failed") -> None:
        super().__init__(detail)
        self.detail = detail
        self.code = code  # unsupported_type / scanned_pdf / too_large / empty_text


@dataclass
class ParsedDoc:
    """解析产物。text 交给 chunker，pages 用于审计（PDF 页数）。"""

    text: str
    source: str
    format: str  # pdf / txt / md
    pages: int = 0


def parse_bytes(data: bytes, filename: str) -> ParsedDoc:
    """
    按扩展名分派解析器。data 是文件原始字节，filename 用于判定类型与 source 记录。

    抛 ParseError —— 调用方（API/Worker）负责转成 422 或 DLQ。
    """
    name = filename.lower()
    dot = name.rfind(".")
    ext = name[dot:] if dot >= 0 else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise ParseError(
            f"不支持的文件类型「{ext or '无扩展名'}」，当前仅支持 .pdf / .txt / .md",
            code="unsupported_type",
        )

    if len(data) > MAX_FILE_BYTES:
        raise ParseError(
            f"文件大小 {len(data) / 1024 / 1024:.1f}MB 超过上限 {MAX_FILE_BYTES / 1024 / 1024:.0f}MB",
            code="too_large",
        )

    if ext == ".pdf":
        return _parse_pdf(data, filename)

    # txt / md：按 utf-8 解码，容错 BOM 与 latin-1 兜底
    text = data.decode("utf-8-sig", errors="replace").strip()
    if not text:
        raise ParseError("文本内容为空", code="empty_text")
    fmt = "md" if ext in (".md", ".markdown") else "txt"
    log.info("doc_parsed", source=filename, format=fmt, chars=len(text))
    return ParsedDoc(text=text, source=filename, format=fmt)


def _parse_pdf(data: bytes, filename: str) -> ParsedDoc:
    """PDF 文本层抽取。无文本层 → scanned_pdf 错误（留给 OCR 流水线）。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 依赖缺失属部署问题
        raise ParseError("PDF 解析依赖 pypdf 未安装，请 pip install pypdf", code="unsupported_type") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise ParseError(f"PDF 文件损坏或格式无法识别: {exc}", code="parse_failed") from exc

    if getattr(reader, "is_encrypted", False):
        # 密码保护的 PDF —— 尝试空密码解密，失败则明确拒收
        try:
            reader.decrypt("")
        except Exception:
            raise ParseError("PDF 已加密，请先解除密码保护后重新上传", code="parse_failed") from None

    pages_text: list[str] = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")

    text = "\n\n".join(t.strip() for t in pages_text if t.strip()).strip()

    if not text:
        # 典型于扫描版 / 纯图 PDF
        raise ParseError(
            "PDF 无文本层（可能是扫描版），当前不支持 OCR，请上传含文字层的 PDF 或改用 .txt/.md",
            code="scanned_pdf",
        )

    log.info("doc_parsed", source=filename, format="pdf", pages=len(reader.pages), chars=len(text))
    return ParsedDoc(text=text, source=filename, format="pdf", pages=len(reader.pages))


def parse_upload(filename: str, data: bytes) -> ParsedDoc:
    """入口别名 —— 语义化调用名（API 层用）。"""
    return parse_bytes(data, filename)
