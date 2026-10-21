/**
 * S7 · Agent 工具调用轨迹（可折叠）。
 *
 * 只在 tool_traces 非空时被 AnswerCard 挂载 —— rag 链路根本不该看到这一栏。
 * arguments 用 JSON.stringify 展示：参数是任意 dict，结构化展示收益低。
 */

import type { ToolCallTrace } from '../api/types'

export interface ToolTraceListProps {
  traces: ToolCallTrace[]
}

export function ToolTraceList({ traces }: ToolTraceListProps) {
  return (
    // 原生 <details> 够用且免 JS：可折叠、可键盘操作
    <details className="tool-traces">
      <summary>
        工具调用轨迹
        <span className="muted">（{traces.length} 步）</span>
      </summary>
      <ol className="tool-trace-list">
        {traces.map((t) => (
          <li key={t.step} className={`tool-trace${t.ok ? '' : ' failed'}`}>
            <div className="tool-trace-head">
              <span className="tool-step">#{t.step}</span>
              <code className="tool-name">{t.tool_name}</code>
              <span className={`tool-ok ${t.ok ? 'ok' : 'err'}`}>{t.ok ? '成功' : '失败'}</span>
              <span className="muted">{t.latency_ms.toFixed(1)} ms</span>
            </div>
            <div className="tool-args">
              <span className="muted">参数</span> <code>{JSON.stringify(t.arguments)}</code>
            </div>
            {t.ok ? (
              <div className="tool-result">
                <span className="muted">结果</span> {t.result}
              </div>
            ) : (
              <div className="tool-error">
                <span className="muted">错误</span> {t.error ?? '未知错误'}
              </div>
            )}
          </li>
        ))}
      </ol>
    </details>
  )
}
