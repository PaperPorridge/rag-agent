/**
 * S7 · 右侧问答面板：MessageList + ChatInput。
 *
 * useChat 放在这里（而不是 App）：问答状态只服务右栏，
 * 和左侧文档列表互不影响，减少不必要的重渲染。
 */

import { useChat } from '../hooks/useChat'
import { ChatInput } from './ChatInput'
import { MessageList } from './MessageList'

export function ChatPanel() {
  const { messages, loading, elapsed, send } = useChat()

  return (
    <section className="panel chat-panel">
      <header className="panel-header">
        <h2>智能问答</h2>
      </header>
      <MessageList messages={messages} />
      <ChatInput loading={loading} elapsed={elapsed} onSend={send} />
    </section>
  )
}
