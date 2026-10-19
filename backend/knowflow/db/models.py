"""
M4 · 数据模型 —— 用关系库存「结构化的真相」

【为什么需要数据库？Demo 的三个致命问题】
  1. 进程一重启，会话历史全丢
  2. 两个人同时提问，内存里互相踩
  3. 没法回答「用户上个月问了什么」这类运营问题

【为什么是关系库（PG）而不是直接上 MongoDB？】
  我们的数据特征：会话-消息是**强关联的一对多**，文档-评估结果要**事务性写入**，
  查询模式（按用户/时间范围/联表统计）高度结构化 → 关系模型完全匹配。
  MongoDB 适合：模式不固定（日志、用户行为事件）、文档内嵌、水平扩展优先。
  一句话：**结构稳定、要事务、要联表 → 关系库；模式自由、海量写入、文档聚合 → Mongo。**
  （本项目 MongoDB 的合理用途是存「原始文档解析产物」这类半结构化数据，见 doc。）

【表设计要点】
  - UUID 主键: 分布式生成无中心冲突，且不可枚举（自增 ID 会被爬）
  - 时间戳: created_at/updated_at 带时区，排序与审计的基础
  - 外键+索引: messages(conversation_id) 必须有索引，否则聊天历史查询全表扫
  - JSON 列: 用于「查询少、结构松」的扩展字段（metadata）——别过度范式化
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 风格基类。用 Mapped[] 声明类型，比老式 Column 更可读。"""


# ---------------------------------------------------------------------------
# 用户（简化版：为多租户与限流服务）
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # 限流配额（QPS / 日预算）放这里 → 配置即数据，改配额不用改代码
    rate_limit_per_min: Mapped[int] = mapped_column(Integer, default=30)
    daily_token_budget: Mapped[int] = mapped_column(Integer, default=500_000)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    conversations: Mapped[list[Conversation]] = relationship(back_populates="user")


# ---------------------------------------------------------------------------
# 会话与消息 —— 一对多，外键 + 索引
# ---------------------------------------------------------------------------
class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(256), default="")
    # 松散扩展字段：模型名、检索参数、标签等 —— 查询少，别单独建表
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    user: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        order_by="Message.seq",   # 消息必须按序返回，否则拼历史会乱
    )

    __table_args__ = (
        Index("ix_conversations_user_updated", "user_id", "updated_at"),  # 最近会话列表
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer)  # 会话内顺序号，保证稳定排序
    role: Mapped[str] = mapped_column(String(16))  # user / assistant / tool
    content: Mapped[str] = mapped_column(Text, default="")
    # Agent 工具调用轨迹放 JSON —— 查询少，但排障时要能整段取出来
    tool_calls: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_cny: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    __table_args__ = (
        # 联合索引：拉取某个会话的第 N 条之后的消息（分页拉历史）
        Index("ix_messages_conv_seq", "conversation_id", "seq"),
    )


# ---------------------------------------------------------------------------
# 文档索引元数据 —— 向量在 Qdrant，「真相」在 PG
# ---------------------------------------------------------------------------
class Document(Base):
    """
    文档元数据。注意职责划分：
      Qdrant  → 存向量与原文片段（检索用）
      这张表  → 存文档的「状态机」与归属（管理用）
    为什么？向量库能重建（从文档重新 embedding），但「谁上传的、审核了没、
    什么时候入库的」这类**业务事实**必须在关系库里，才能事务一致、可审计。
    """

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    source: Mapped[str] = mapped_column(String(512), index=True)  # 文件名/URL
    owner_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    # 状态机: pending → processing → ready / failed
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_model: Mapped[str] = mapped_column(String(64), default="")
    # 换 embedding 模型必须重建索引（M2 铁律）——这里记录用的哪个模型，可审计
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


# ---------------------------------------------------------------------------
# 评估结果 —— M7 的数据基础
# ---------------------------------------------------------------------------
class EvalRun(Base):
    """一次评估运行的结果。没有这张表，「效果提升 20%」就是一句空话。"""

    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_name: Mapped[str] = mapped_column(String(128), index=True)
    dataset: Mapped[str] = mapped_column(String(128), default="")
    # 指标放 JSON: faithfulness / answer_relevancy / context_precision ...
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # 本次评估的参数快照
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
