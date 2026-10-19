"""
M5 · Kafka 实现（aiokafka）

【生产部署形态】
  Producer  →  Kafka (topic: doc.ingest, 3 partitions)
                 ↑
                 └── 3 个 Worker 实例组成 consumer-group，各消费一个 partition

【关键设计点】
  1. key = document_id → 同一文档的事件落同一 partition → 有序
  2. enable_auto_commit=False → **手动提交**，处理成功才 commit
     （自动提交会「还没处理完就提交 offset」→ 崩溃丢消息）
  3. ack = commit offset；nack_to_dlq = 发到 *.dlq topic + commit
     （必须 commit，否则同一条毒消息无限重投，卡死 partition）
  4. 幂等：消费端按 message_id/document_id upsert，重复消费无副作用

【为什么 enable_auto_commit 必须关？】
  自动提交是「后台定期提交最后拉到的 offset」，不是「处理成功才提交」。
  若 Worker 拉到消息 → 自动提交 → 处理时崩溃 → 重启后消息已被跳过 → **丢消息**。
  at-least-once 的正确姿势：处理完手动 commit，宁可重复不可丢失。
"""

from __future__ import annotations

from typing import Any

import structlog

from knowflow.queue.base import Message, MessageQueue, QueueError
from knowflow.queue.memory import InMemoryQueue

log = structlog.get_logger(__name__)

DLQ_SUFFIX = ".dlq"


class KafkaQueue(MessageQueue):
    """
    Kafka 实现。用法：

        q = KafkaQueue(bootstrap_servers="localhost:9092", group_id="ingest-worker")
        await q.send("doc.ingest", {"doc_id": "..."}, key="doc_id")
        msg = await q.receive("doc.ingest")
        ...处理...
        await q.ack(msg)
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "knowflow-workers",
    ) -> None:
        try:
            from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
        except ImportError as exc:
            raise ImportError("需要安装: pip install aiokafka") from exc

        self._bootstrap = bootstrap_servers
        self._group_id = group_id
        self._producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers)
        self._consumer = AIOKafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            enable_auto_commit=False,   # ← 手动提交，见模块 docstring
            auto_offset_reset="earliest",
            max_poll_records=10,
        )
        self._started = False

    async def _ensure_started(self) -> None:
        if self._started:
            return
        await self._producer.start()
        await self._consumer.start()
        self._started = True
        log.info("kafka_started", bootstrap=self._bootstrap, group=self._group_id)

    async def send(
        self,
        topic: str,
        value: dict[str, Any],
        *,
        key: str = "",
        headers: dict[str, str] | None = None,
    ) -> str:
        await self._ensure_started()
        import json
        import uuid

        message_id = str(uuid.uuid4())
        hdrs = [(k, v.encode()) for k, v in (headers or {}).items()]
        hdrs.append(("message_id", message_id.encode()))
        try:
            await self._producer.send_and_wait(
                topic,
                json.dumps(value, ensure_ascii=False).encode("utf-8"),
                key=key.encode("utf-8") if key else None,
                headers=hdrs,
            )
            log.info("kafka_sent", topic=topic, key=key, message_id=message_id)
            return message_id
        except Exception as exc:
            raise QueueError(f"发送失败: {exc}") from exc

    async def receive(self, topic: str, *, timeout: float = 1.0) -> Message | None:
        await self._ensure_started()
        import json

        self._consumer.subscribe([topic])
        batch = await self._consumer.getmany(timeout_ms=int(timeout * 1000), max_records=1)
        for _tp, records in batch.items():
            for record in records:
                headers = {k: v.decode() for k, v in (record.headers or [])}
                message_id = headers.pop("message_id", str(record.offset))
                return Message(
                    topic=record.topic,
                    value=json.loads(record.value.decode("utf-8")),
                    key=(record.key or b"").decode("utf-8"),
                    message_id=message_id,
                    headers=headers,
                    offset=record.offset,
                )
        return None

    async def ack(self, message: Message) -> None:
        """提交 offset = 确认处理完成。"""
        await self._ensure_started()
        try:
            await self._consumer.commit()
            log.info("kafka_acked", topic=message.topic, offset=message.offset)
        except Exception as exc:
            raise QueueError(f"commit 失败: {exc}") from exc

    async def nack_to_dlq(self, message: Message, error: str) -> None:
        """
        毒消息送死信队列，**然后必须 commit**。

        不 commit 会怎样：Kafka 会把这条消息重投，无限循环卡死 partition。
        正确顺序：转发到 *.dlq topic → commit 原 partition 的 offset。
        """
        await self._ensure_started()
        await self.send(
            message.topic + DLQ_SUFFIX,
            message.value,
            key=message.key,
            headers={**message.headers, "error": error[:200], "attempts": str(message.attempts)},
        )
        await self.ack(message)
        log.warning(
            "kafka_dlq",
            topic=message.topic,
            offset=message.offset,
            error=error,
            attempts=message.attempts,
        )

    async def close(self) -> None:
        if self._started:
            await self._producer.stop()
            await self._consumer.stop()
            self._started = False


def create_queue(url: str | None = None, group_id: str = "knowflow-workers") -> MessageQueue:
    """
    工厂：有 Kafka URL 就用 Kafka，否则内存。
    url 不传时从配置读 KAFKA_BOOTSTRAP_SERVERS；Kafka 连不上也降级 ——
    队列挂了不该让开发跑不动测试。
    """
    if url is None:
        from knowflow.core.config import get_settings

        url = get_settings().kafka_bootstrap_servers or None
    if url:
        try:
            q = KafkaQueue(bootstrap_servers=url, group_id=group_id)
            log.info("queue_backend", backend="kafka", url=url)
            return q
        except Exception as exc:
            log.warning("kafka_init_failed", error=str(exc), action="use_memory")
    log.info("queue_backend", backend="memory")
    return InMemoryQueue()
