"""
M4 · 缓存层 —— Redis 与它的三个典型用途

【为什么需要缓存？LLM 应用的三个热点】
  ① 检索结果缓存: 同样的问题（"报销时限"）反复被问 → 向量检索结果可复用
  ② 会话状态: 多轮对话的中间状态（Agent 的工具轨迹）需要跨请求共享
  ③ 限流计数: 每用户每分钟的调用次数 —— 必须集中计数（多实例部署时内存计数会翻倍漏限）

【为什么用 Redis 而不是进程内 dict？】
  进程内 dict 的三个致命问题:
    1. 多实例部署时各自为政（3 个实例 = 3 份计数，限流形同虚设）
    2. 进程重启全丢
    3. 无法设置过期时间、无法原子操作
  Redis 是单线程原子操作 + TTL + 分布式共享 —— 限流/缓存/会话的标配。

【缓存三坑】
  1. 缓存穿透: 查不存在的 key，每次都打到 DB/LLM
     → 解法: 空值也缓存（短 TTL）或布隆过滤器
  2. 缓存击穿: 热点 key 过期瞬间，大量请求同时打到后端
     → 解法: 互斥锁（SETNX）或逻辑过期（异步刷新）
  3. 缓存雪崩: 大量 key 同时过期
     → 解法: TTL 加随机抖动（我们的实现就这么做）

【本模块设计】
  CacheStore 是接口。RedisCache 连真 Redis；InMemoryCache 用于开发/单测/降级。
  生产 Redis 挂了怎么办？我们的策略是**降级到 InMemoryCache 并告警**，
  服务可用性 > 缓存命中率（缓存本来就只是加速器）。
"""

from __future__ import annotations

import json
import random
import time
from abc import ABC, abstractmethod
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class CacheStore(ABC):
    """缓存接口。三个方法覆盖 90% 场景。"""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """取缓存。未命中返回 None。"""

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """写缓存。ttl 建议加随机抖动防雪崩。"""

    @abstractmethod
    def incr_with_ttl(self, key: str, ttl_seconds: int = 60) -> int:
        """原子自增并设置过期（限流计数专用）。返回自增后的值。"""

    @abstractmethod
    def delete(self, key: str) -> None:
        ...


class InMemoryCache(CacheStore):
    """
    进程内缓存 —— 开发/单测/Redis 故障降级用。

    注意它**不是分布式**的：多实例部署时每个进程一份，
    限流会漏、缓存不共享。所以生产必须上 Redis，这个实现只是保底。
    """

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, Any]] = {}  # key -> (expire_at, value)

    def _purge_if_expired(self, key: str) -> None:
        item = self._data.get(key)
        if item and item[0] < time.time():
            del self._data[key]

    def get(self, key: str) -> Any | None:
        self._purge_if_expired(key)
        item = self._data.get(key)
        return item[1] if item else None

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        # 加 0~10% 随机抖动 → 防止一批 key 同时过期（缓存雪崩）
        jitter = random.uniform(0, ttl_seconds * 0.1)
        self._data[key] = (time.time() + ttl_seconds + jitter, value)

    def incr_with_ttl(self, key: str, ttl_seconds: int = 60) -> int:
        self._purge_if_expired(key)
        item = self._data.get(key)
        current = (int(item[1]) + 1) if item else 1
        expire_at = item[0] if item else time.time() + ttl_seconds
        self._data[key] = (expire_at, current)
        return current

    def delete(self, key: str) -> None:
        self._data.pop(key, None)


class RedisCache(CacheStore):
    """
    Redis 实现。

    关键点:
      - value 统一 JSON 序列化: Redis 只存字节，直接存 dict 会炸
      - incr_with_ttl 用 pipeline: 保证 INCR 和 EXPIRE 的原子性
      - 失败降级: Redis 挂了不能把用户请求搞挂，降级到 InMemory + 告警
    """

    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        try:
            import redis
        except ImportError as exc:
            raise ImportError("需要安装: pip install redis") from exc

        self._redis = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2)
        self._fallback = InMemoryCache()
        self._degraded = False

    def _check_health(self) -> bool:
        """Redis 是否可用。不可用则标记降级。"""
        if self._degraded:
            return False
        try:
            self._redis.ping()
            return True
        except Exception as exc:
            log.error("redis_unhealthy", error=str(exc), action="fallback_to_memory")
            self._degraded = True
            return False

    def get(self, key: str) -> Any | None:
        if not self._check_health():
            return self._fallback.get(key)
        try:
            raw = self._redis.get(key)
            return json.loads(raw) if raw is not None else None
        except Exception as exc:
            log.warning("redis_get_failed", key=key, error=str(exc))
            return None

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        if not self._check_health():
            self._fallback.set(key, value, ttl_seconds)
            return
        try:
            jitter = random.randint(0, max(int(ttl_seconds * 0.1), 1))
            self._redis.setex(key, ttl_seconds + jitter, json.dumps(value, ensure_ascii=False))
        except Exception as exc:
            log.warning("redis_set_failed", key=key, error=str(exc))
            self._fallback.set(key, value, ttl_seconds)

    def incr_with_ttl(self, key: str, ttl_seconds: int = 60) -> int:
        if not self._check_health():
            return self._fallback.incr_with_ttl(key, ttl_seconds)
        try:
            # pipeline 保证 INCR + EXPIRE 原子执行
            pipe = self._redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, ttl_seconds, nx=True)  # nx: 已有 TTL 就不覆盖（滑动窗口语义）
            results = pipe.execute()
            return int(results[0])
        except Exception as exc:
            log.warning("redis_incr_failed", key=key, error=str(exc))
            return self._fallback.incr_with_ttl(key, ttl_seconds)

    def delete(self, key: str) -> None:
        if self._check_health():
            try:
                self._redis.delete(key)
            except Exception:
                pass
        self._fallback.delete(key)


def create_cache(url: str | None = None) -> CacheStore:
    """
    工厂：有 Redis URL 就用 Redis，否则内存。
    Redis 初始化失败也降级到内存 —— 缓存挂了不该影响主流程。
    """
    if url:
        try:
            cache = RedisCache(url)
            cache._check_health()
            if not cache._degraded:
                log.info("cache_backend", backend="redis", url=url)
                return cache
        except Exception as exc:
            log.warning("redis_init_failed", error=str(exc), action="use_memory")
    log.info("cache_backend", backend="memory")
    return InMemoryCache()
