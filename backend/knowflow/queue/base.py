"""
M5 · 消息队列抽象层

【为什么要有抽象层？】
  和 M4 的 CacheStore 一个套路：接口稳定，实现可换。
    生产 → Kafka（高吞吐、可回放）
    开发/CI → InMemoryQueue（零依赖、可跑测试）
  业务代码（Worker）只认 MessageQueue，不关心底下是 Kafka 还是内存。

【Kafka 语义速记】
  Topic        消息分类（例：doc.ingest）
  Partition    并行消费的单位，同分区有序
  Consumer Group  同组内每条消息只被一个成员消费
  Offset       消息在分区中的位置
  Commit       提交 offset = 「我处理完了」

  语义：at-least-once（至少一次）
    → 消费者可能重复收到消息，**消费逻辑必须幂等**

【三种投递语义】
  at-most-once   先 commit 再处理，可能丢，不会重复
  at-least-once  先处理再 commit，不丢，可能重复  ← Kafka 默认
  exactly-once   事务性生产/消费，最强，成本最高
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class QueueError(Exception):
    """队列操作基础异常。"""


class RetryableError(QueueError):
    """可重试错误（网络抖动、下游暂不可用）→ 重试 N 次。"""


class PoisonMessageError(QueueError):
    """
    毒消息：重试也救不回来的错误（解析不了的 PDF、schema 非法）。

    处理方式：进死信队列（DLQ），阻塞 partition，人工介入。
    没有 DLQ 会怎样：一条毒消息卡住整个分区，后面所有消息堵死。
    """


@dataclass
class Message:
    """
    一条消息。设计要点：

    - message_id: 幂等键。消费端据此做「处理过就跳过」。
    - key: 分区键。同一 key 落同一 partition → 同一文档的事件有序。
    - headers: 不参与业务的元数据（trace_id、schema 版本）。
    - attempts: 已重试次数，用于指数退避与死信判定。
    """

    topic: str
    value: dict[str, Any]
    key: str = ""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    headers: dict[str, str] = field(default_factory=dict)
    attempts: int = 0
    # 仅内存实现使用（Kafka 用 offset 管理）
    offset: int = -1
    enqueued_at: float = field(default_factory=time.time)


class MessageQueue(ABC):
    """消息队列接口。四个方法覆盖生产/消费/ack/死信。"""

    @abstractmethod
    async def send(self, topic: str, value: dict[str, Any], *, key: str = "", headers: dict[str, str] | None = None) -> str:
        """发送消息。返回 message_id。"""

    @abstractmethod
    async def receive(self, topic: str, *, timeout: float = 1.0) -> Message | None:
        """拉取一条消息。超时返回 None。"""

    @abstractmethod
    async def ack(self, message: Message) -> None:
        """确认消费成功（提交 offset）。"""

    @abstractmethod
    async def nack_to_dlq(self, message: Message, error: str) -> None:
        """标记为毒消息，送进死信队列。"""

    async def close(self) -> None:
        """释放资源。"""
