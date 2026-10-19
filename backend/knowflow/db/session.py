"""
M4 · 数据库连接与会话管理

【三个必须做对的工程点】

1. 连接池（Pooling）
   每次请求新建 TCP 连接 → 延迟高、端口耗尽。
   SQLAlchemy 默认带 QueuePool：连接复用，连接数上限可控。
   生产要点：pool_size=并发常态，max_overflow=突发余量，pool_pre_ping=True
   （预检连接是否还活着，避免「MySQL server has gone away」类错误）。

2. 会话边界（Session Scope）
   反例: 全局单 Session —— 多线程/多请求互相污染，事务边界混乱
   正例: **一次请求/一次用例一个 Session**，用完即关（context manager）
   ORM 对象是 Session 缓存的快照，Session 关了还在用对象 → DetachedInstanceError

3. 事务边界
   一个业务操作 = 一个事务（unit of work）。提交在业务完成时，
   出错整体回滚。不要在循环里逐条 commit —— 慢且不原子。

【环境降级（本课的实际做法）】
   生产: PostgreSQL (postgresql+psycopg://...)
   开发/CI: SQLite (sqlite:///...) —— 零依赖跑通，接口完全一致
   这就是「配置决定实现」的价值，代码不用写两套。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import structlog
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from knowflow.core.config import get_settings
from knowflow.db.models import Base

log = structlog.get_logger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def get_engine(url: str | None = None) -> Engine:
    """
    获取全局 Engine（连接池挂在 Engine 上，必须复用）。

    url 不传时从配置读：DATABASE_URL，缺省 SQLite 文件（开发模式）。
    """
    global _engine
    if _engine is not None and url is None:
        return _engine

    settings = get_settings()
    db_url = url or getattr(settings, "database_url", None) or "sqlite:///./knowflow.db"

    kwargs: dict = {"pool_pre_ping": True, "echo": False}
    if db_url.startswith("sqlite"):
        # SQLite 是文件库，connect_args 必须关掉同线程检查（FastAPI 会跨线程）
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # PG/MySQL: 连接池参数 —— 并发的命根子
        kwargs.update(pool_size=10, max_overflow=20, pool_recycle=1800)

    engine = create_engine(db_url, **kwargs)
    log.info("db_engine_created", url=db_url.split("@")[-1][:60])  # 不打密码
    if url is None:
        _engine = engine
    return engine


def init_db(url: str | None = None) -> Engine:
    """建表。生产用 Alembic 迁移（见 migrations/），init_db 只用于开发/测试。"""
    engine = get_engine(url)
    Base.metadata.create_all(engine)
    log.info("db_tables_created", n_tables=len(Base.metadata.tables))
    return engine


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """
    事务性会话上下文。用法:
        with session_scope() as s:
            s.add(msg)
        # 正常退出自动 commit，异常自动 rollback
    """
    global _SessionFactory
    engine = get_engine(url)
    if _SessionFactory is None or url is not None:
        _SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)

    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
