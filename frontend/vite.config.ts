import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 开发时前端跑 5173，API 跑 8000 —— 用 proxy 转发 /api，
// 浏览器侧永远是同源请求，不用配 CORS、也不用写死后端地址。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': { target: 'http://localhost:8000', changeOrigin: true } },
  },
})
