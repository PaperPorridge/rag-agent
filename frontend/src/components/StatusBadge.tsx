/**
 * S7 · 文档状态徽章。
 *
 * 颜色语义：pending=灰（排队）、processing=蓝+脉冲（在跑）、
 * ready=绿、failed=红。一眼扫列表就能看出哪些还没落定。
 */

import type { DocumentStatus } from '../api/types'

const LABEL: Record<DocumentStatus, string> = {
  pending: '排队中',
  processing: '处理中',
  ready: '就绪',
  failed: '失败',
}

export interface StatusBadgeProps {
  status: DocumentStatus
}

export function StatusBadge({ status }: StatusBadgeProps) {
  // 小写类名驱动 CSS；processing 额外带 pulse 动画
  const cls = `status-badge status-${status}${status === 'processing' ? ' pulse' : ''}`
  return <span className={cls}>{LABEL[status] ?? status}</span>
}
