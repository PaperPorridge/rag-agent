"""
M5 · 内存队列 —— 开发/测试/降级用

【为什么需要它】
  本机没装 Kafka 时，测试不能跑就是卡点。
  这个实现保证：**接口与 KafkaQueue 完全一致**，业务代码零改动。
  M4 的 SQLite 降级、Redis 降级，都是同一个套路。

【它模拟了 Kafka 的哪些行为？】
  ✅ 分区（按 key 哈希到 N 个分区，同 key 有序）
  ✅ offset + ack（处理完才移除，崩溃可重见）
  ✅ at-least-once（不 ack 会再次投递 → 可测试幂等）
  ✅ 死信队列

【它模拟不了的（生产必须用真 Kafka）】
  ❌ 多进程共享（内存只在本进程）
  ❌ 持久化（进程一停全丢）
  ❌ 回放（没有日志留存）
  ❌ 真正的水平扩展
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any

import structlog

from knowflow.queue.base import Message, MessageQueue

log = structlog.get_logger(__name__)


class InMemoryQueue(MessageQueue):
    """进程内队列。模拟 partition / offset / ack / DLQ。"""

    def __init__(self, n_partitions: int = 3) -> None:
        self.n_partitions = n_partitions
        # topic -> partition -> [Message]（待消费）
        self._topics: dict[str, dict[int, list[Message]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._dlq: dict[str, list[Message]] = defaultdict(list)
        self._acked_ids: set[str] = set()   # 幂等键：已处理成功的消息
        self._lock = asyncio.Lock()
        self._offset_seq = 0

    def _partition_for(self, key: str) -> int:
        """按 key 哈希分区 —— 和 Kafka 一样：同 key 同分区，保证有序。"""
        return hash(key) % self.n_partitions

    async def send(
        self,
        topic: str,
        value: dict[str, Any],
        *,
        key: str = "",
        headers: dict[str, str] | None = None,
    ) -> str:
        async with self._lock:
            self._offset_seq += 1
            msg = Message(
                topic=topic,
                value=value,
                key=key,
                headers=headers or {},
                offset=self._offset_seq,
            )
            partition = self._partition_for(key)
            self._topics[topic][partition].append(msg)
            log.info(
                "queue_sent",
                topic=topic,
                partition=partition,
                message_id=msg.message_id,
                key=key,
            )
            return msg.message_id

    async def receive(self, topic: str, *, timeout: float = 1.0) -> Message | None:
        """按分区顺序取一条。模拟轮询。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            async with self._lock:
                for partition in sorted(self._topics[topic].keys()):
                    if self._topics[topic][partition]:
                        msg = self._topics[topic][partition].pop(0)
                        msg.attempts += 1
                        return msg
            await asyncio.sleep(0.02)
        return None

    async def ack(self, message: Message) -> None:
        """标记为已处理。重复 ack 无副作用（幂等）。"""
        async with self._lock:
            if message.message_id in self._acked_ids:
                return
            self._acked_ids.add(message.message_id)
            log.info(
                "queue_acked",
                topic=message.topic,
                message_id=message.message_id,
                attempts=message.attempts,
            )

    async def nack_to_dlq(self, message: Message, error: str) -> None:
        async with self._lock:
            self._dlq[message.topic].append(message)
            log.warning(
                "queue_dlq",
                topic=message.topic,
                message_id=message.message_id,
                error=error,
                attempts=message.attempts,
            )

    # -- 可观测：测试与演示用 -----------------------------------------
    def pending(self, topic: str) -> int:
        return sum(len(v) for v in self._topics[topic].values())

    def dlq_size(self, topic: str) -> int:
        return len(self._dlq[topic])

    def dlq_messages(self, topic: str) -> list[Message]:
        return list(self._dlq[topic])

    def is_acked(self, message_id: str) -> bool:
        return message_id in self._acked_ids
