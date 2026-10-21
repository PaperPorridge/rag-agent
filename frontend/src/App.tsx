/**
 * S7 · 单屏双栏布局 + 健康徽章。
 *
 * 【为什么不用路由？】
 *   产品形态就是一屏：左边文档、右边问答。加 router 是把单屏拆成多屏的
 *   自我复杂化 —— 技术栈已明确「无路由、无 Redux」。
 * 【为什么 health 放 App 而不是某个面板？】
 *   健康是整站状态，徽章在 header；下挂到左栏或右栏都会让「哪挂了」变模糊。
 */

import { useCallback, useEffect, useState } from 'react'

import { getHealth } from './api/client'
import type { HealthResponse } from './api/types'
import { ChatPanel } from './components/ChatPanel'
import { DocsPanel } from './components/DocsPanel'

/** 健康徽章的三态 + 未知（首次请求前）。fetch 失败统一归 down=红。 */
type HealthState = 'checking' | 'ok' | 'degraded' | 'down'

const HEALTH_POLL_MS = 30_000

const HEALTH_LABEL: Record<HealthState, string> = {
  checking: '检查中',
  ok: 'ok',
  degraded: 'degraded',
  down: '不可用',
}

function HealthBadge({ state, version }: { state: HealthState; version: string | null }) {
  return (
    <span className={`health-badge health-${state}`} title={version ? `version ${version}` : ''}>
      <span className="health-dot" />
      {HEALTH_LABEL[state]}
    </span>
  )
}

export default function App() {
  const [health, setHealth] = useState<HealthState>('checking')
  const [version, setVersion] = useState<string | null>(null)

  const pollHealth = useCallback(async () => {
    try {
      const data: HealthResponse = await getHealth()
      // status 只有 ok/degraded 两值（后端 schema 保证）
      setHealth(data.status === 'ok' ? 'ok' : 'degraded')
      setVersion(data.version)
    } catch {
      // 拉不到（后端没起 / 代理断了）→ 红色「不可用」；不弹 toast，避免每 30s 刷屏
      setHealth('down')
    }
  }, [])

  useEffect(() => {
    void pollHealth()
    const id = window.setInterval(() => {
      void pollHealth()
    }, HEALTH_POLL_MS)
    return () => window.clearInterval(id)
  }, [pollHealth])

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <h1>KnowFlow</h1>
          <span className="brand-sub">企业知识库 Agent</span>
        </div>
        <HealthBadge state={health} version={version} />
      </header>

      <main className="app-main">
        <DocsPanel />
        <ChatPanel />
      </main>

      {/* 错误文案统一在各面板 inline 展示，这里不做全局 toast（避免堆叠刷屏） */}
      <footer className="app-footer muted">KnowFlow · API /api/v1</footer>
    </div>
  )
}
