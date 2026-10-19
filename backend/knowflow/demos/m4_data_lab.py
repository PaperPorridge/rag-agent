"""
M4 演示：数据层实验室（PostgreSQL 模型 / 会话管理 / Redis 缓存 / 限流）

运行:
    cd backend
    python -m knowflow.demos.m4_data_lab

本演示覆盖:
  1. ORM 模型与建表      —— UUID 主键、联合索引、状态机
  2. 会话边界            —— session_scope 提交/回滚、N+1 查询对比
  3. 缓存                —— 命中/未命中、TTL 过期、缓存穿透与雪崩抖动
  4. 限流                —— 令牌桶突发特性、每分钟配额
  5. 降级路径            —— 无 Redis 时自动落到内存缓存（生产 Redis 故障同理）

环境说明: 本机未装 PG/Redis。生产用 PostgreSQL + Redis，开发用 SQLite + 内存缓存，
接口完全一致 —— 这正是「配置决定实现」的工程价值。
"""

from __future__ import annotations

import time

from sqlalchemy import select

from knowflow.cache import InMemoryCache, RateLimiter, TokenBucket, create_cache, retrieval_cache_key
from knowflow.db.models import Base, Conversation, Document, EvalRun, Message, User
from knowflow.db.session import init_db, session_scope


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def reset_demo_db():
    """
    演示专用：清空重建，保证每次运行结果确定（幂等）。

    不这么做的话，第二次运行会撞 users.username 的 UNIQUE 约束
    （alice 还在库里），或让「回滚验证 bob 不存在」的断言失真。

    ⚠️ 生产绝对不能 drop_all —— 那是删库。
    生产的表结构变更走 Alembic 迁移，数据永远不整表清空。
    """
    engine = init_db()  # SQLite 开发模式：sqlite:///./knowflow.db
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return engine


# ---------------------------------------------------------------------------
# 1. ORM 模型与建表
# ---------------------------------------------------------------------------
def demo_models() -> str:
    section("1. ORM 模型与建表 —— UUID 主键 / 联合索引 / 状态机")

    engine = reset_demo_db()
    print(f"  引擎: {engine.url}")
    print(f"  建表: {sorted(engine.dialect.get_table_names(engine.connect()))}")

    print("\n  表设计要点:")
    print("    · UUID 主键      分布式生成无冲突，且不可枚举（自增 ID 会被爬）")
    print("    · ix_messages_conv_seq          (conversation_id, seq) 拉历史不全表扫")
    print("    · ix_conversations_user_updated (user_id, updated_at) 最近会话列表")
    print("    · Document.status 状态机: pending → processing → ready / failed")
    print("    · EvalRun.metrics JSON: 指标可扩展，不用改表结构")
    return engine


# ---------------------------------------------------------------------------
# 2. CRUD + 会话边界
# ---------------------------------------------------------------------------
def demo_session_crud() -> tuple[str, str]:
    section("2. 会话边界与 CRUD —— 一次请求一个 Session，用完即关")

    # --- 写入：一个事务里完成「建用户 + 建会话 + 两条消息」 ---
    with session_scope() as s:
        user = User(username="alice", rate_limit_per_min=30, daily_token_budget=500_000)
        s.add(user)
        s.flush()  # 拿到 user.id，但还不 commit

        conv = Conversation(user_id=user.id, title="报销制度咨询", meta={"model": "mimo-v2.6-pro"})
        s.add(conv)
        s.flush()

        s.add_all(
            [
                Message(conversation_id=conv.id, seq=1, role="user", content="报销时限是多少天？", tokens=12),
                Message(conversation_id=conv.id, seq=2, role="assistant", content="30 天内提交。", tokens=48, cost_cny=0.001),
            ]
        )
        conv_id, user_id = conv.id, user.id
        print(f"  ✅ 事务提交: user={user.username} conv={conv.title} + 2 条消息")

    # --- 回滚演示 ①：业务异常 ---
    try:
        with session_scope() as s:
            s.add(User(username="bob"))
            s.flush()
            raise RuntimeError("模拟业务异常")
    except Exception as exc:
        print(f"  ✅ 业务异常自动 rollback: {type(exc).__name__}")

    # --- 回滚演示 ②：UNIQUE 约束（DB 层校验，也会回滚整个事务）---
    try:
        with session_scope() as s:
            s.add(User(username="alice"))  # 上面已插入
            s.flush()  # 触发 IntegrityError
    except Exception as exc:
        print(f"  ✅ UNIQUE 约束拒绝并回滚: {type(exc).__name__}")

    with session_scope() as s:
        bob = s.scalar(select(User).where(User.username == "bob"))
        alice = s.scalar(select(User).where(User.username == "alice"))
        print(f"  ✅ 验证: bob 存在? {bob is not None}（应 False）  "
              f"alice 唯一? {alice is not None}（应 True）")

    # --- 读取：relationship 预排序（order_by=Message.seq）---
    with session_scope() as s:
        conv = s.get(Conversation, conv_id)
        print(f"\n  会话历史 [{conv.title}]（按 seq 稳定排序）:")
        for m in conv.messages:
            print(f"    #{m.seq} {m.role:>9}: {m.content}  (tokens={m.tokens})")

        # --- N+1 反模式对比 ---
        print("\n  N+1 查询反模式:")
        convs = s.scalars(select(Conversation)).all()
        print(f"    会话数={len(convs)}")
        print("    ❌ 反例: for c in convs: print(len(c.messages))  → 1 + N 次查询")
        print("    ✅ 正例: selectinload(Conversation.messages)   → 2 次查询，或直接用 relationship 已加载")

    return user_id, conv_id


# ---------------------------------------------------------------------------
# 3. 缓存
# ---------------------------------------------------------------------------
def demo_cache() -> None:
    section("3. 缓存 —— 命中 / TTL / 穿透 / 雪崩抖动")

    cache = InMemoryCache()

    # 基本读写
    print("  ① 基本读写")
    cache.set("greeting", {"text": "你好"}, ttl_seconds=2)
    print(f"     set('greeting') → get = {cache.get('greeting')}")

    # TTL 过期
    print("\n  ② TTL 过期")
    cache.set("ephemeral", "soon-gone", ttl_seconds=1)
    print(f"     立刻取: {cache.get('ephemeral')}")
    time.sleep(1.2)
    print(f"     1.2s 后取: {cache.get('ephemeral')}（None = 已过期）")

    # 缓存穿透：空值也缓存
    print("\n  ③ 缓存穿透防御: 空值也写入短 TTL")
    missing_key = "user:9999"
    val = cache.get(missing_key)
    if val is None:
        # 查库未命中 → 写空值哨兵，短 TTL，挡住重复穿透
        cache.set(missing_key, "__NULL__", ttl_seconds=30)
        print(f"     查库未命中 → 缓存空值哨兵 __NULL__ (ttl=30s)")
    print(f"     再次 get = {cache.get(missing_key)}（不再打到 DB）")

    # 雪崩：TTL 抖动
    print("\n  ④ 缓存雪崩防御: TTL 加 0~10% 随机抖动")
    for i in range(3):
        cache.set(f"burst:{i}", i, ttl_seconds=100)
    print("     3 个 key 的 TTL 不会精确同秒过期 → 不会同时打穿后端")

    # 检索缓存 key 设计
    print("\n  ⑤ 检索缓存 key 规范")
    k1 = retrieval_cache_key("报销时限是多久？", top_k=5)
    k2 = retrieval_cache_key("  报销时限是多久？ ", top_k=5)  # 规范化后应相同
    k3 = retrieval_cache_key("报销时限是多久？", top_k=10)  # top_k 不同 → key 不同
    print(f"     key(query,5)  = {k1}")
    print(f"     key(同 query 不同空白,5) = {k2}  → 规范化后命中: {k1 == k2}")
    print(f"     key(query,10) = {k3}  → 参数不同必须不同 key: {k1 != k3}")


# ---------------------------------------------------------------------------
# 4. 限流
# ---------------------------------------------------------------------------
def demo_rate_limit() -> None:
    section("4. 限流 —— 令牌桶（突发）与每分钟配额（公平）")

    # --- 令牌桶: capacity=5, rate=2/s ---
    print("  ① 令牌桶 capacity=5, rate=2/s（允许突发，稳态 2 QPS）")
    bucket = TokenBucket(capacity=5, rate=2.0)
    results = [bucket.try_acquire() for _ in range(7)]
    ok = results.count(True)
    print(f"     连打 7 次: 放行 {ok} 次, 拒绝 {results.count(False)} 次（前 5 次应放行）")

    print("     等 1.5s（补充约 3 个令牌）...")
    time.sleep(1.5)
    results2 = [bucket.try_acquire() for _ in range(3)]
    print(f"     再打 3 次: 放行 {results2.count(True)} 次")

    # --- 分布式友好限流: 每分钟配额 ---
    print("\n  ② RateLimiter per_minute=5（严格配额，可跨实例共享计数）")
    limiter = RateLimiter(InMemoryCache(), per_minute=5)
    outcomes = [limiter.allow("alice") for _ in range(7)]
    print(f"     连打 7 次: {['✓' if x else '✗' for x in outcomes]}")
    print(f"     剩余配额: {limiter.remaining('alice')}")
    print("     key 设计: rate:{user_id}:{当前分钟} —— 多实例共享 Redis 计数才准确")


# ---------------------------------------------------------------------------
# 5. 降级路径
# ---------------------------------------------------------------------------
def demo_fallback() -> None:
    section("5. 降级 —— 无 Redis 时自动落到内存缓存")

    cache = create_cache(url=None)  # 未配置 REDIS_URL
    print(f"  后端: {type(cache).__name__}")
    cache.set("key", "value", ttl_seconds=5)
    print(f"  get('key') = {cache.get('key')}")

    print("\n  生产语义:")
    print("    · Redis 可用   → 分布式缓存/限流/会话状态")
    print("    · Redis 故障   → 自动降级 InMemoryCache + 结构化告警日志")
    print("    · 原则: 服务可用性 > 缓存命中率（缓存只是加速器，不是依赖）")


def main() -> None:
    from knowflow.demos._console import safe_stdout
    safe_stdout()

    print("=" * 64)
    print("M4 · 数据层实验室 —— PostgreSQL 模型 / 会话 / Redis 缓存 / 限流")
    print("=" * 64)

    demo_models()
    demo_session_crud()
    demo_cache()
    demo_rate_limit()
    demo_fallback()

    section("M4 演示完成")
    print("  下一步 M5: 消息队列异步入库 —— python -m knowflow.demos.m5_queue_lab")


if __name__ == "__main__":
    main()
