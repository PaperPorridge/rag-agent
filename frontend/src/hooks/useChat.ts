/**
 * S7 · 问答消息流。
 *
 * 【为什么历史在这里组装而不是后端 Session？】
 *   后端不持久化会话（AskRequest.conversation_id 注释写明「前端本地用」）。
 *   每次 send 把「本次之前的」user/assistant 轮次塞进 history 发给 /ask，
 *   由后端自行截断到最近 20 轮。
 * 【为什么失败消息不进 history？】
 *   error 消息没有真实问答内容，传给模型只会污染上下文。
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { ask, formatError } from '../api/client'
import type { AskResponse, ChatTurn } from '../api/types'

/** 前端本地消息模型。result 只有 assistant 成功时才有。 */
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  /** assistant 消息的结构化结果（链路 / 来源 / 工具轨迹 / 成本） */
  result?: AskResponse
  /** 失败时的错误文案（展示在 assistant 侧） */
  error?: string
}

export interface UseChat {
  messages: ChatMessage[]
  loading: boolean
  /** 加载中已耗时（秒），给用户一个「还在跑」的反馈 */
  elapsed: number
  send: (text: string) => Promise<void>
}

/** 本项目是单屏 SPA，消息 id 只需本地唯一 —— crypto.randomUUID 足够。 */
function nextId(): string {
  return crypto.randomUUID()
}

export function useChat(): UseChat {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)
  const [elapsed, setElapsed] = useState(0)

  // 加载计时器：loading 变 true 时启动，变 false 时清理
  useEffect(() => {
    if (!loading) return
    setElapsed(0)
    const start = Date.now()
    const id = window.setInterval(() => {
      setElapsed((Date.now() - start) / 1000)
    }, 200)
    return () => window.clearInterval(id)
  }, [loading])

  // 用 ref 拿最新消息，避免 send 闭包捕获过期 state
  const messagesRef = useRef(messages)
  messagesRef.current = messages

  const send = useCallback(async (text: string) => {
    const question = text.trim()
    // loading 时忽略：连点 Enter 不该并行发多个 /ask
    if (!question || loading) return

    const history: ChatTurn[] = messagesRef.current
      .filter((m) => !m.error && m.content.length > 0)
      .map((m) => ({ role: m.role, content: m.content }))

    setMessages((prev) => [...prev, { id: nextId(), role: 'user', content: question }])
    setLoading(true)
    try {
      const result = await ask({ question, top_k: 5, history })
      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: 'assistant', content: result.answer, result },
      ])
    } catch (e) {
      // 网络/HTTP 失败：把错误贴在 assistant 侧，用户知道这条没有答案
      setMessages((prev) => [...prev, { id: nextId(), role: 'assistant', content: '', error: formatError(e) }])
    } finally {
      setLoading(false)
    }
  }, [loading])

  return { messages, loading, elapsed, send }
}
