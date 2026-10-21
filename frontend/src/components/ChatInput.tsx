/**
 * S7 · 输入框：Enter 发送，Shift+Enter 换行。
 *
 * 【为什么用 onKeyDown 而不是 form onSubmit？】
 *   textarea 里 Enter 的默认行为是换行，需要拦掉它；form 提交只能捕捉
 *   「点击按钮」这类提交动作，拦不住 textarea 的 Enter。两者一起用更啰嗦。
 * 加载中禁用并显示已耗时秒数 —— 比纯 disabled 更有反馈。
 */

import { useState } from 'react'

export interface ChatInputProps {
  loading: boolean
  /** 加载中已耗时（秒） */
  elapsed: number
  onSend: (text: string) => Promise<void>
}

export function ChatInput({ loading, elapsed, onSend }: ChatInputProps) {
  const [value, setValue] = useState('')

  const submit = () => {
    const text = value.trim()
    if (!text || loading) return
    // 先清空输入再发：体感更快，失败时错误会贴在消息流里
    setValue('')
    void onSend(text)
  }

  return (
    <div className="chat-input">
      <textarea
        className="input textarea"
        placeholder="问点什么…（Enter 发送，Shift+Enter 换行）"
        value={value}
        rows={3}
        disabled={loading}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
            // isComposing：中文输入法回车选词不应被当成发送
            e.preventDefault()
            submit()
          }
        }}
      />
      <div className="chat-input-actions">
        {loading ? (
          <span className="muted elapsed">思考中… {Math.floor(elapsed)}s</span>
        ) : null}
        <button type="button" className="btn primary" disabled={loading || !value.trim()} onClick={submit}>
          发送
        </button>
      </div>
    </div>
  )
}
