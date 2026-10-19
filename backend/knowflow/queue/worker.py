"""
M5 · 文档异步入库 Worker

【为什么入库要走队列？】（本课修的坑）
  上传一个 50MB 的 PDF，解析+分块+embedding 可能要 30 秒。
  同步做：HTTP 连接挂 30 秒 → 网关超时 → 用户看到失败但其实后端在跑 → 重复上传。
  异步做：API 只发一条消息「doc_id=xxx 待入库」立刻返回 202，
          Worker 在后台慢慢啃，啃完把 Document.status 置为 ready。

【流水线 5 步】
  1. 取消息        receive()  →  拿到 doc_id
  2. 状态流转      pending → processing（让前端能查进度）
  3. 解析/分块/向量化/写索引  →  RAGPipeline.ingest_batch
  4. 落库          Document.status = ready, chunk_count = n
  5. ack           提交 offset，消息「消费完成」

【异常三分类】
  可重试（RetryableError）   网络抖动、下游暂时不可用 → 退避重试 N 次
  毒消息（PoisonMessageError） 解析不了的 PDF、schema 非法 → 直接 DLQ，不浪费重试
  未知异常                   当作可重试，重试耗尽 → DLQ

【幂等】（at-least-once 的生存法则）
  同一条消息可能被投递多次（崩溃后重启、网络重传）。
  处理必须幂等：RAGPipeline.ingest_batch 会先 delete_by_source 再写，
  重复执行结果一致。Document.status 用 upsert 而不是盲 insert。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import structlog

from knowflow.queue.base import (
    Message,
    MessageQueue,
    PoisonMessageError,
    RetryableError,
)

log = structlog.get_logger(__name__)

# 消息主题（Kafka topic / 内存队列 key，同一套名字）
TOPIC_DOC_INGEST = "doc.ingest"


class DocParser(Protocol):
    """解析器接口：把原始字节/路径变成纯文本。生产用 pypdf / unstructured。"""

    def parse(self, payload: dict[str, Any]) -> str: ...


@dataclass
class WorkerStats:
    """可观测性：没有它，Worker 就是个黑盒，出问题只能猜。"""

    processed: int = 0
    succeeded: int = 0
    retried: int = 0
    dead_lettered: int = 0
    skipped_dup: int = 0
    durations: list[float] = field(default_factory=list)

    @property
    def avg_duration(self) -> float:
        return sum(self.durations) / len(self.durations) if self.durations else 0.0

    def summary(self) -> str:
        return (
            f"processed={self.processed} ok={self.succeeded} "
            f"retried={self.retried} dlq={self.dead_lettered} "
            f"dup_skipped={self.skipped_dup} avg={self.avg_duration * 1000:.1f}ms"
        )


@dataclass
class IngestResult:
    """一次入库的结果 —— 给上层调用（API 轮询状态）用。"""

    doc_id: str
    status: str
    chunk_count: int = 0
    error: str | None = None


class IngestWorker:
    """
    文档入库消费者。用法：

        worker = IngestWorker(queue=create_queue(), ingest_fn=pipeline.ingest_batch)
        await worker.run_once()          # 消费一条（测试用）
        await worker.run_forever()       # 持续消费（生产用）

    关键参数：
        max_attempts   同一消息最多处理几次，超过送 DLQ
        retry_backoff  重试间隔基数（秒），指数退避：0.1, 0.2, 0.4 ...
    """

    def __init__(
        self,
        queue: MessageQueue,
        *,
        ingest_fn: Callable[[list[tuple[str, str]]], int] | None = None,
        on_status: Callable[[str, str, int, str | None], None] | None = None,
        max_attempts: int = 3,
        retry_backoff: float = 0.1,
    ) -> None:
        self.queue = queue
        self.ingest_fn = ingest_fn
        self.on_status = on_status
        self.max_attempts = max_attempts
        self.retry_backoff = retry_backoff
        self.stats = WorkerStats()
        # 幂等表：已成功处理的 message_id / doc_id（生产用 Redis SETNX）
        self._done_ids: set[str] = set()

    # -- 状态流转 ---------------------------------------------------------
    def _set_status(self, doc_id: str, status: str, n_chunks: int = 0, error: str | None = None) -> None:
        if self.on_status:
            self.on_status(doc_id, status, n_chunks, error)

    # -- 核心处理逻辑（独立出来，方便单测） -------------------------------
    def process(self, msg: Message) -> IngestResult:
        """
        处理一条消息。抛 RetryableError / PoisonMessageError 区分异常类型。

        这里刻意做成**同步纯函数**：不碰队列、不碰 DB 连接池，
        只做「文档内容 → 分块数」的业务逻辑，可被 pytest 直接断言。
        """
        doc_id = str(msg.value.get("doc_id", "")).strip()
        source = str(msg.value.get("source", "")).strip()
        text = str(msg.value.get("text", "")).strip()

        if not doc_id or not source:
            # 缺必填字段 = 毒消息，重试也不会变好
            raise PoisonMessageError(f"消息缺必填字段 doc_id/source: {msg.value}")

        if not text:
            raise PoisonMessageError(f"文档内容为空: doc_id={doc_id}")

        # 幂等：已处理过就跳过（at-least-once 下最省事的正确姿势）
        if doc_id in self._done_ids:
            self.stats.skipped_dup += 1
            log.info("worker_skip_dup", doc_id=doc_id, message_id=msg.message_id)
            return IngestResult(doc_id=doc_id, status="ready", chunk_count=0)

        self._set_status(doc_id, "processing")

        if self.ingest_fn is None:
            n_chunks = 1  # demo 模式：不真的入库
        else:
            # RAGPipeline.ingest_batch 内部会 delete_by_source → 重复执行幂等
            n_chunks = self.ingest_fn([(text, source)])

        self._done_ids.add(doc_id)
        self._set_status(doc_id, "ready", n_chunks)
        return IngestResult(doc_id=doc_id, status="ready", chunk_count=n_chunks)

    # -- 消费循环 ---------------------------------------------------------
    async def run_once(self, topic: str = TOPIC_DOC_INGEST, timeout: float = 1.0) -> bool:
        """
        消费一条消息（含重试 / DLQ）。返回 True 表示拿到了消息。

        【重试策略】
          - RetryableError 或未知异常 → 退避重试，次数耗尽 → DLQ
          - PoisonMessageError       → 立刻 DLQ，不浪费重试配额
          - 重试时**不 ack**（内存队列会重新可见；Kafka 重新拉取）
        """
        msg = await self.queue.receive(topic, timeout=timeout)
        if msg is None:
            return False

        self.stats.processed += 1
        log.info(
            "worker_received",
            topic=msg.topic,
            message_id=msg.message_id,
            doc_id=msg.value.get("doc_id"),
            attempt=msg.attempts,
        )

        last_err: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = self.process(msg)
                await self.queue.ack(msg)
                self.stats.succeeded += 1
                self.stats.durations.append(time.monotonic() - getattr(msg, "_t0", time.monotonic()))
                log.info(
                    "worker_done",
                    doc_id=result.doc_id,
                    chunks=result.chunk_count,
                    attempt=attempt,
                )
                return True
            except PoisonMessageError as exc:
                last_err = exc
                self.stats.dead_lettered += 1
                await self.queue.nack_to_dlq(msg, error=str(exc))
                log.error("worker_poison", error=str(exc), message_id=msg.message_id)
                return True
            except RetryableError as exc:
                last_err = exc
                self.stats.retried += 1
                wait = self.retry_backoff * (2 ** (attempt - 1))
                log.warning(
                    "worker_retry",
                    attempt=attempt,
                    max_attempts=self.max_attempts,
                    wait=wait,
                    error=str(exc),
                )
                if attempt < self.max_attempts:
                    await asyncio.sleep(wait)
            except Exception as exc:  # 未知异常当可重试处理
                last_err = exc
                self.stats.retried += 1
                wait = self.retry_backoff * (2 ** (attempt - 1))
                log.warning(
                    "worker_retry_unexpected",
                    attempt=attempt,
                    error=f"{type(exc).__name__}: {exc}",
                )
                if attempt < self.max_attempts:
                    await asyncio.sleep(wait)

        # 重试耗尽 → 死信
        self.stats.dead_lettered += 1
        await self.queue.nack_to_dlq(msg, error=f"retries_exhausted: {last_err}")
        log.error("worker_dlq_after_retries", message_id=msg.message_id, error=str(last_err))
        return True

    async def run_forever(self, topic: str = TOPIC_DOC_INGEST, idle_sleep: float = 0.05) -> None:
        """生产入口：一直消费，Ctrl+C 停止。"""
        log.info("worker_started", topic=topic, max_attempts=self.max_attempts)
        try:
            while True:
                got = await self.run_once(topic)
                if not got:
                    await asyncio.sleep(idle_sleep)
        finally:
            log.info("worker_stopped", stats=self.stats.summary())


# ---------------------------------------------------------------------------
# 生产侧 API：上传后立刻发消息，立刻返回 202
# ---------------------------------------------------------------------------
async def enqueue_doc_ingest(
    queue: MessageQueue,
    *,
    doc_id: str,
    source: str,
    text: str,
    owner_id: str | None = None,
) -> str:
    """
    API 层调用：入队一条「待入库」事件。

    【为什么 key=doc_id？】
      Kafka 按 key 哈希分区 → 同一 doc_id 的消息永远落同一 partition → 有序。
      如果不设 key，两次更新同一文档可能落到不同 partition，后更新的先到 → 乱序覆盖。
    """
    return await queue.send(
        TOPIC_DOC_INGEST,
        {"doc_id": doc_id, "source": source, "text": text, "owner_id": owner_id},
        key=doc_id,
        headers={"event": "doc.ingest.requested"},
    )


# ---------------------------------------------------------------------------
# 进程入口：python -m knowflow.queue.worker
#
# 【为什么单独一个入口，而不是塞进 docker-compose 的 command？】
#   compose 的 command 若用 YAML 折叠标量（>），多行 Python 会被折成一行 → SyntaxError，
#   容器永远起不来。把启动逻辑放这里，compose 只需一行
#   command: ["python", "-m", "knowflow.queue.worker"]
#   本地也能直接跑，好调试、好测。
# ---------------------------------------------------------------------------
def _build_ingest_fn() -> Callable[[list[tuple[str, str]]], int] | None:
    """构造真正的入库函数；依赖不全时返回 None（worker 走 demo 分支）。"""
    try:
        from knowflow.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline()
        log.info("worker_ingest_fn", mode="rag_pipeline")
        return pipeline.ingest_batch
    except Exception as exc:
        log.warning(
            "worker_ingest_fn_unavailable",
            error=str(exc),
            action="demo_mode_only",
            hint="缺 embedding/向量库依赖时消息只做状态流转，不真的写索引",
        )
        return None


def main() -> None:
    """Worker 进程主入口。"""
    from knowflow.core.config import get_settings
    from knowflow.queue import create_queue

    settings = get_settings()
    log.info(
        "worker_starting",
        env=settings.app_env,
        kafka=settings.kafka_bootstrap_servers or "(memory)",
    )

    queue = create_queue()
    worker = IngestWorker(queue=queue, ingest_fn=_build_ingest_fn())
    asyncio.run(worker.run_forever())


if __name__ == "__main__":
    main()
