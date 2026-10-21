/**
 * S7 · 文档列表行。
 *
 * 状态相关显示规则：
 *   - chunk_count 只在 ready 时展示（pending/processing 时是 0，展示 0 会误导）
 *   - failed 时展示 error 全文（title 再重复一份，长错误也能悬停看全）
 *   - 删除走 window.confirm —— 删除会连向量一起清掉，必须二次确认
 */

import type { DocumentStatusResponse } from '../api/types'
import { StatusBadge } from './StatusBadge'

export interface DocListProps {
  docs: DocumentStatusResponse[]
  total: number
  loading: boolean
  onDelete: (docId: string) => Promise<boolean>
}

/** ISO8601 → 本地可读时间；null/非法值原样不显示。 */
function formatTime(iso: string | null): string {
  if (!iso) return ''
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleString()
}

export function DocList({ docs, total, loading, onDelete }: DocListProps) {
  if (loading && docs.length === 0) {
    return <div className="empty-state">加载中…</div>
  }

  if (docs.length === 0) {
    return <div className="empty-state">还没有文档，先上传一份</div>
  }

  return (
    <div className="doc-list">
      <div className="doc-list-meta">共 {total} 份</div>
      {docs.map((doc) => (
        <article key={doc.doc_id} className="doc-row">
          <div className="doc-row-main">
            <div className="doc-source" title={doc.source}>
              {doc.source}
            </div>
            <div className="doc-row-sub">
              <StatusBadge status={doc.status} />
              {doc.status === 'ready' ? <span className="doc-chunks">{doc.chunk_count} 块</span> : null}
              {doc.created_at ? <span className="doc-time">{formatTime(doc.created_at)}</span> : null}
            </div>
            {doc.status === 'failed' && doc.error ? (
              // title 做简易 tooltip，不用额外组件
              <div className="doc-error" title={doc.error}>
                {doc.error}
              </div>
            ) : null}
          </div>
          <button
            type="button"
            className="btn danger ghost"
            title="删除文档及其向量"
            onClick={() => {
              if (window.confirm(`确定删除「${doc.source}」？文档及其向量将一并删除。`)) {
                void onDelete(doc.doc_id)
              }
            }}
          >
            删除
          </button>
        </article>
      ))}
    </div>
  )
}
