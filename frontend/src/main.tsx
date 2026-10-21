/**
 * S7 · 应用入口。
 *
 * StrictMode 的价值在开发期：双调用渲染会把「渲染期间写了外部状态」
 * 这类副作用问题提前暴露出来。
 */

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import App from './App'
import './styles.css'

const rootEl = document.getElementById('root')
if (!rootEl) {
  throw new Error('缺少 #root 挂载点 —— 检查 index.html')
}

createRoot(rootEl).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
