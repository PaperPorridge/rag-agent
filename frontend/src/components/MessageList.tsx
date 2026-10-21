/**
 * S7 · 消息列表：可滚动 + 自动滚到底。
 *
 * 【为什么要 scrollIntoView 而不是 scrollTop = scrollHeight？】
 *   前者在异步插入消息后浏览器会算好布局再滚，后者有时会差一行。
 * 空状态文案固定为「问我任何关于知识库的问题」。
 */

import { useEffect, useRef } from 'react'

import type { ChatMessage } from '../hooks/useChat'
import { AnswerCard } from './AnswerCard'

export interface MessageListProps {
  messages: ChatMessage[]
}

export function MessageList({ messages }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  // 每次消息变化滚到底 —— 新回答很长时用户不用手动翻
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (messages.length === 0) {
    return (
      <div className="message-list">
        <div className="empty-state chat-empty">问我任何关于知识库的问题</div>
      </div>
    )
  }

  return (
    <div className="message-list">
      {messages.map((m) => (
        <div key={m.id} className={`message-row message-${m.role}`}>
          {m.role === 'user' ? (
            <div className="bubble user-bubble">{m.content}</div>
          ) : m.result ? (
            <AnswerCard result={m.result} />
          ) : (
            // 有 error 就是请求失败的那条 assistant 占位消息
            <div className="bubble error-bubble">{m.error ?? m.content}</div>
          )}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
