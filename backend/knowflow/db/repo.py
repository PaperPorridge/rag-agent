"""
数据库写入助手 —— 与路由/Worker 解耦的状态写回。

【为什么单独一个 repo 模块？】
  Worker 不该懂 SQLAlchemy 的会话细节，路由也不该。
  这里收拢「写 Document 行」这一件事，将来加审计表也在这层扩。
"""

from __future__ import annotations

import structlog
from sqlalchemy import select

from knowflow.db.models import Document
from knowflow.db.session import session_scope

log = structlog.get_logger(__name__)


def db_on_status(doc_id: str, status: str, n_chunks: int = 0, error: str | None = None) -> None:
    """
    Worker 状态回调：把入库进度写回 documents 表。

    【为什么绝不抛异常？】
      Worker 正在处理消息，DB 抖动一下不应该让消费循环崩掉 ——
      向量已经写进 Qdrant 的话，消息重投会幂等覆盖；但 Worker 崩了就全停。
      所以这里吞异常 + 结构化告警日志，让 SRE 能查但不影响主链路。
    """
    try:
        from knowflow.core.config import get_settings

        embedding_model = get_settings().embedding_model
        with session_scope() as s:
            doc = s.execute(select(Document).where(Document.id == doc_id)).scalar_one_or_none()
            if doc is None:
                # 文档行还没落（或已被删）—— 用 upsert 语义补一行，不丢状态
                doc = Document(id=doc_id, source=f"unknown-{doc_id}", status=status)
                s.add(doc)
            doc.status = status
            doc.chunk_count = n_chunks
            doc.error = error
            doc.embedding_model = embedding_model
        log.info("doc_status_updated", doc_id=doc_id, status=status, n_chunks=n_chunks)
    except Exception as exc:  # noqa: BLE001 —— 有意吞掉，见 docstring
        log.error(
            "doc_status_write_failed",
            doc_id=doc_id,
            status=status,
            error=f"{type(exc).__name__}: {exc}",
        )
