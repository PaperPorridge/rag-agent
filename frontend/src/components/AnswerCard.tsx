/**
 * S7 · 一条 assistant 回答的完整卡片。
 *
 * 展示层次从「结论」到「证据」：
 *   答案正文 → 链路徽章 + router_reason → 检索来源（可折叠）
 *   → 工具轨迹（可折叠）→ 成本/步数
 * 为什么 router_reason 放小字？它是排障信息（路由误判时看），不是用户主叙事。
 */

import type { AskResponse } from '../api/types'
import { ToolTraceList } from './ToolTraceList'

export interface AnswerCardProps {
  result: AskResponse
}

export function AnswerCard({ result }: AnswerCardProps) {
  const isAgent = result.chain === 'agent'

  return (
    <article className="answer-card">
      <div className="answer-text">{result.answer}</div>

      <div className="answer-meta">
        <span className={`chain-badge chain-${result.chain}`}>
          {isAgent ? 'Agent' : '直接问答'}
        </span>
        {result.router_reason ? (
          <span className="router-reason" title={result.router_reason}>
            {result.router_reason}
          </span>
        ) : null}
      </div>

      {result.retrieved.length > 0 ? (
        <details className="sources-accordion">
          <summary>
            检索来源
            <span className="muted">（{result.retrieved.length} 条）</span>
          </summary>
          <ul className="source-list">
            {result.retrieved.map((r) => (
              <li key={r.citation} className="source-item">
                <div className="source-head">
                  <code className="source-citation">{r.citation}</code>
                  {/* score 固定两位小数 —— RRF/重排得分本就只看相对大小 */}
                  <span className="source-score">{r.score.toFixed(2)}</span>
                </div>
                <div className="source-snippet">{r.snippet}</div>
              </li>
            ))}
          </ul>
        </details>
      ) : null}

      {result.tool_traces.length > 0 ? <ToolTraceList traces={result.tool_traces} /> : null}

      <div className="answer-stats muted">
        <span>成本 ¥{result.cost_cny.toFixed(4)}</span>
        <span>·</span>
        <span>{result.steps} 步</span>
      </div>
    </article>
  )
}
