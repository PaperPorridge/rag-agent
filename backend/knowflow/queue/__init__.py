"""M5 · 消息队列：生产者/消费者抽象 + Kafka 实现 + 内存降级 + 入库 Worker。"""

from knowflow.queue.base import (
    Message,
    MessageQueue,
    PoisonMessageError,
    QueueError,
    RetryableError,
)
from knowflow.queue.memory import InMemoryQueue
from knowflow.queue.kafka import KafkaQueue, create_queue
from knowflow.queue.worker import (
    TOPIC_DOC_INGEST,
    IngestResult,
    IngestWorker,
    WorkerStats,
    enqueue_doc_ingest,
)

__all__ = [
    "Message",
    "MessageQueue",
    "QueueError",
    "RetryableError",
    "PoisonMessageError",
    "InMemoryQueue",
    "KafkaQueue",
    "create_queue",
    "IngestWorker",
    "IngestResult",
    "WorkerStats",
    "enqueue_doc_ingest",
    "TOPIC_DOC_INGEST",
]
