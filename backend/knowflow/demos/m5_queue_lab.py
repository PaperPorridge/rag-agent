"""
M5 演示：消息队列实验室（Kafka 语义 / 异步入库 / 幂等 / 重试 / 死信）

运行:
    cd backend
    python -m knowflow.demos.m5_queue_lab

本演示覆盖:
  1. 发送/消费基本流 —— key 分区、offset、ack
  2. 异步入库流水线 —— 上传 → 发消息 → Worker 消费 → 向量化 → 置 ready
  3. 幂等           —— 同一消息投递两次，只处理一次（at-least-once 的生存法则）
  4. 可重试错误     —— 网络抖动类错误退避重试后成功
  5. 毒消息 → DLQ   —— 重试也救不回来的错误，立刻进死信，不卡分区
  6. 队列选型对比   —— Kafka / RabbitMQ / SQS 怎么选

环境说明: 本机未装 Kafka。生产用 Kafka（高吞吐、可回放、多消费者组），
开发用 InMemoryQueue（零依赖、接口一致）—— 和 M4 的 SQLite/Redis 降级同一个套路。
"""

from __future__ import annotations

import asyncio

from knowflow.queue import (
    IngestWorker,
    InMemoryQueue,
    Message,
    PoisonMessageError,
    RetryableError,
    TOPIC_DOC_INGEST,
    create_queue,
    enqueue_doc_ingest,
)
from knowflow.rag.pipeline import RAGPipeline


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# ---------------------------------------------------------------------------
# 1. 发送/消费基本流
# ---------------------------------------------------------------------------
async def demo_basic_flow() -> InMemoryQueue:
    section("1. 发送 / 消费 / ack —— key 决定分区，ack 才算完成")

    q = InMemoryQueue(n_partitions=3)

    # 同一个 key 必须落同一分区 → 同一文档的事件有序
    for i in range(3):
        await q.send(
            TOPIC_DOC_INGEST,
            {"doc_id": "doc-1", "text": f"第 {i} 版内容"},
            key="doc-1",           # ← 分区键
            headers={"event": "doc.ingest.requested"},
        )
    await q.send(TOPIC_DOC_INGEST, {"doc_id": "doc-2", "text": "另一份"}, key="doc-2")

    print(f"  发送 4 条（3 条 key=doc-1，1 条 key=doc-2）")
    print(f"  队列积压 pending = {q.pending(TOPIC_DOC_INGEST)}")

    # 消费并 ack —— ack 前崩溃的话消息会再次可见（at-least-once）
    n = 0
    while (msg := await q.receive(TOPIC_DOC_INGEST, timeout=0.2)) is not None:
        print(f"    收到: doc_id={msg.value['doc_id']} key={msg.key} offset={msg.offset} attempt={msg.attempts}")
        await q.ack(msg)
        n += 1
    print(f"  消费并 ack {n} 条，pending={q.pending(TOPIC_DOC_INGEST)}")
    print("  [OK] key 相同的消息来自同一分区，顺序保持")
    return q


# ---------------------------------------------------------------------------
# 2. 异步入库流水线
# ---------------------------------------------------------------------------
async def demo_async_ingest() -> None:
    section("2. 异步入库流水线 —— 上传秒回，重活交给 Worker")

    q = InMemoryQueue()
    rag = RAGPipeline()          # 用 HashingEmbedder，零依赖可跑

    # 真实业务：RAGPipeline.ingest_batch 做分块+向量化+写索引
    worker = IngestWorker(
        queue=q,
        ingest_fn=rag.ingest_batch,   # ← 流水线的核心：分块/向量化/入库
    )

    # 模拟 API 层：用户点上传 → 发消息 → 立刻返回 202
    docs = [
        {"doc_id": "doc-101", "source": "差旅报销制度.pdf", "text": "员工差旅报销需在出差结束后30天内提交报销单，逾期需部门总监特批。"},
        {"doc_id": "doc-102", "source": "IT设备清单.md", "text": "公司开发机型号为 ThinkPad-X1-Carbon-Gen12，标配内存32GB。"},
    ]
    for d in docs:
        mid = await enqueue_doc_ingest(q, **d)
        print(f"  [API] 上传 {d['source']} → 发消息 {mid[:8]}… 立刻返回 202 Accepted")

    # 模拟 Worker：后台慢慢啃
    for _ in range(len(docs)):
        await worker.run_once()
    print(f"  [Worker] {worker.stats.summary()}")

    # 检索验证真的进去了
    hits = rag.search_only("报销有时间限制吗", top_k=2)
    print(f"  [检索] 命中 {len(hits)} 条: {[h.citation for h in hits]}")
    print("  [OK] 上传不被解析阻塞，Worker 异步完成索引")


# ---------------------------------------------------------------------------
# 3. 幂等 —— at-least-once 的生存法则
# ---------------------------------------------------------------------------
async def demo_idempotent() -> None:
    section("3. 幂等 —— 同一条消息投递两次，只处理一次")

    q = InMemoryQueue()
    processed: list[str] = []

    def ingest(docs):
        for _, src in docs:
            processed.append(src)
        return len(docs)

    worker = IngestWorker(queue=q, ingest_fn=ingest)

    # 同一份文档，用同一个 doc_id 发两次（模拟生产中「重复投递」）
    await enqueue_doc_ingest(q, doc_id="doc-777", source="重复.pdf", text="内容A")
    await enqueue_doc_ingest(q, doc_id="doc-777", source="重复.pdf", text="内容A")

    await worker.run_once()
    await worker.run_once()

    print(f"  发送 2 条（同一 doc_id）")
    print(f"  实际入库次数 = {len(processed)}  （应该是 1）")
    print(f"  幂等跳过 skip_dup = {worker.stats.skipped_dup}")
    print("  [OK] 幂等键 doc_id 挡住了重复处理，at-least-once 不会写脏数据")


# ---------------------------------------------------------------------------
# 4. 可重试错误
# ---------------------------------------------------------------------------
async def demo_retryable() -> None:
    section("4. 可重试错误 —— 网络抖动退避重试，不丢消息")

    q = InMemoryQueue()
    calls = {"n": 0}

    def flaky_ingest(docs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableError(f"向量库连接超时（第 {calls['n']} 次）")
        return len(docs)

    worker = IngestWorker(queue=q, ingest_fn=flaky_ingest, max_attempts=3, retry_backoff=0.05)

    await enqueue_doc_ingest(q, doc_id="doc-retry", source="抖动.pdf", text="内容")
    await worker.run_once()

    print(f"  前 2 次抛 RetryableError，第 3 次成功")
    print(f"  调用次数 = {calls['n']}  （max_attempts=3）")
    print(f"  {worker.stats.summary()}")
    print("  [OK] 可重试错误退避后重试，不进 DLQ —— 区分「暂时故障」和「永久故障」")


# ---------------------------------------------------------------------------
# 5. 毒消息 → DLQ
# ---------------------------------------------------------------------------
async def demo_poison() -> None:
    section("5. 毒消息 → 死信队列 —— 不卡住分区")

    q = InMemoryQueue()

    def bad_ingest(docs):
        raise ValueError("PDF 解析失败: 不支持的编码")   # 未知异常

    worker = IngestWorker(queue=q, ingest_fn=bad_ingest, max_attempts=2, retry_backoff=0.02)

    # 毒消息：缺必填字段
    await q.send(TOPIC_DOC_INGEST, {"doc_id": "", "source": ""}, key="poison-1")
    # 重试也救不回来的消息
    await enqueue_doc_ingest(q, doc_id="doc-bad", source="坏文件.pdf", text="内容")

    await worker.run_once()
    await worker.run_once()

    print(f"  DLQ 里有 {q.dlq_size(TOPIC_DOC_INGEST)} 条毒消息")
    for m in q.dlq_messages(TOPIC_DOC_INGEST):
        print(f"    - doc_id={m.value.get('doc_id')!r}  attempts={m.attempts}")
    print(f"  {worker.stats.summary()}")
    print("  [OK] 毒消息进 DLQ 后 ack 掉，不阻塞后续消息；人工可从 DLQ 排查")


# ---------------------------------------------------------------------------
# 6. 队列选型对比
# ---------------------------------------------------------------------------
def demo_queue_choice() -> None:
    section("6. Kafka / RabbitMQ / SQS 怎么选")

    rows = [
        ("维度", "Kafka", "RabbitMQ", "SQS"),
        ("定位", "分布式日志 / 流平台", "传统消息代理", "云托管队列"),
        ("消息模型", "pull + 消费者组", "push + 交换机路由", "pull + 可见性超时"),
        ("消息保留", "按时间/容量保留，可回放", "消费即删（默认）", "最长 14 天"),
        ("吞吐", "极高（百万级/秒）", "中（万级/秒）", "高，按需扩展"),
        ("顺序保证", "分区内有序", "队列内有序", "默认不保证（FIFO 队列可选）"),
        ("典型场景", "日志/事件流/回放", "业务解耦/任务分发", "云原生/无运维"),
        ("运维成本", "高（要管 ZK/KRaft）", "中", "零（云托管）"),
    ]
    for r in rows:
        print(f"  {r[0]:<10} | {r[1]:<20} | {r[2]:<16} | {r[3]}")

    print("\n  本项目选 Kafka 的理由:")
    print("    1. 文档入库是「事件流」，后续还要加「审核事件」「权限变更事件」——topic 天然扩展")
    print("    2. 需要回放：索引坏了，reset offset 重新消费一遍就能重建（SQS 没有 offset）")
    print("    3. 分区内有序：同一 doc_id 的两次更新必须有序，Kafka 用 key 就解决")
    print("  什么情况换 RabbitMQ: 只要「发个任务给某个 worker」，不需要回放，团队没 Kafka 运维能力")
    print("  什么情况换 SQS:    全栈在 AWS，不想自己运维队列，消息量中等且不需要回放")


# ---------------------------------------------------------------------------
async def main() -> None:
    print(__doc__)
    print("队列后端:", type(create_queue()).__name__, "（Kafka URL 未配置时自动降级到内存）")

    await demo_basic_flow()
    await demo_async_ingest()
    await demo_idempotent()
    await demo_retryable()
    await demo_poison()
    demo_queue_choice()

    section("完成")
    print("  概念回顾:")
    print("    · 异步解耦  上传秒回，解析/向量化交给 Worker，网关不超时")
    print("    · at-least-once  会重复投递 → 消费端必须幂等（doc_id 去重）")
    print("    · 重试三分类  可重试 / 毒消息 / 未知 → 退避重试、立刻 DLQ、重试耗尽 DLQ")
    print("    · DLQ  毒消息不卡分区，人工从死信排查")
    print("    · 选型  事件流+回放 → Kafka；简单任务分发 → RabbitMQ；云原生免运维 → SQS")


if __name__ == "__main__":
    asyncio.run(main())
