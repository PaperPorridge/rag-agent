"""
M4 · 限流 —— 令牌桶算法

【为什么必须限流？】
  1. 成本护栏: LLM 按 token 计费，不限流 = 账单炸弹（被刷或自己代码死循环）
  2. 自我保护: 上游 LLM 有 QPS 限制，不限流会 429 连环雪崩
  3. 公平性: 防止单用户吃光所有配额

【三种限流算法对比】
  固定窗口: 简单，但窗口边界突发双倍流量（59s 与 1s 各打满 = 实际双倍）
  滑动窗口: 解决边界问题，但要存时间戳序列，内存开销大
  令牌桶  ✅: 平滑突发 + 允许合理突发（桶容量），实现简单，生产最常用
  漏桶   : 严格匀速，无法应对合理突发（用户按一下发3条消息就被拒）

【令牌桶原理】
  桶容量 capacity，按 rate 个/秒 放入令牌。
  每次请求取 1 个令牌；桶空 → 拒绝（429）。
  桶容量决定了「允许的最大突发」，rate 决定「稳态速率」。

  例: capacity=10, rate=5/s
    静置 2 秒后可连打 10 个（突发），之后稳定 5 QPS

【分布式限流】
  单机令牌桶在多实例下会放大放行倍数（3 实例 = 3 倍配额）。
  生产方案: Redis + Lua 脚本做原子令牌桶，或用网关（Kong/APISIX/SQS 配额）
  我们的 CacheStore.incr_with_ttl 就是给分布式计数用的基础设施。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

from knowflow.cache.store import CacheStore, InMemoryCache

log = structlog.get_logger(__name__)


@dataclass
class TokenBucket:
    """
    单机令牌桶（线程不安全的最小实现，讲清原理用）。

    生产多线程要加锁；多实例要用 Redis 版（见 RateLimiter）。
    """

    capacity: float  # 桶容量 = 允许的最大突发
    rate: float  # 令牌生成速率（个/秒）= 稳态 QPS
    tokens: float | None = None
    updated_at: float | None = None

    def __post_init__(self) -> None:
        if self.tokens is None:
            self.tokens = self.capacity  # 初始满桶，允许冷启动突发
        self.updated_at = time.monotonic()

    def try_acquire(self, n: int = 1) -> bool:
        """尝试取 n 个令牌。成功 True，不足 False。"""
        now = time.monotonic()
        elapsed = now - (self.updated_at or now)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)  # 补充令牌，不超容量
        self.updated_at = now

        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


class RateLimiter:
    """
    分布式友好的限流器（滑动窗口计数，基于 CacheStore）。

    比 TokenBucket 弱在「不允许突发」，强在「多实例共享计数」。
    生产按需求选：要突发用 Redis Lua 令牌桶；要严格配额用这个。

    用法:
        limiter = RateLimiter(cache, per_minute=30)
        if not limiter.allow(user_id):
            raise RateLimitExceeded(...)
    """

    def __init__(self, cache: CacheStore | None = None, per_minute: int = 30) -> None:
        self.cache = cache or InMemoryCache()
        self.per_minute = per_minute

    def allow(self, user_id: str) -> bool:
        """
        判断是否放行。key 设计: rate:{user_id}:{当前分钟} → 每分钟一个计数器。
        """
        window = int(time.time() // 60)
        key = f"rate:{user_id}:{window}"
        count = self.cache.incr_with_ttl(key, ttl_seconds=120)  # TTL 2 分钟，覆盖窗口+缓冲

        if count > self.per_minute:
            log.warning("rate_limited", user=user_id, count=count, limit=self.per_minute)
            return False
        return True

    def remaining(self, user_id: str) -> int:
        """剩余配额（给客户端返回 X-RateLimit-Remaining 用）。"""
        window = int(time.time() // 60)
        key = f"rate:{user_id}:{window}"
        count = self.cache.get(key) or 0
        return max(self.per_minute - int(count), 0)


# ---------------------------------------------------------------------------
# 检索缓存的 key 设计（配套工具）
# ---------------------------------------------------------------------------
def retrieval_cache_key(query: str, top_k: int, filters: dict | None = None) -> str:
    """
    检索缓存 key。要点:
      1. 必须包含**所有影响结果的参数**（query/top_k/filters），否则张冠李戴
      2. query 规范化（strip+lower）提高命中率
      3. 用哈希避免超长 key（Redis key 限制 512MB，但可读性差）
    """
    import hashlib

    normalized = query.strip().lower()
    payload = f"{normalized}|{top_k}|{sorted((filters or {}).items())}"
    return f"retrieval:{hashlib.sha256(payload.encode()).hexdigest()[:16]}"
