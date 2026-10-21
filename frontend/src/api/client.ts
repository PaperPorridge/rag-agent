/**
 * S7 · 薄封装 fetch —— 全部走 /api/v1 前缀（Vite proxy 转发到 :8000）。
 *
 * 【为什么不用 axios？】
 *   依赖越少越好。fetch 够用，错误处理自己写反而把 API 边界看得更清楚。
 * 【为什么每个函数都显式标注返回类型？】
 *   导出签名不允许 any —— 调用方拿到的类型就是契约本身（见 types.ts）。
 * 【为什么不自动重试？】
 *   问答是用户可见动作，失败就该如实报错；文档轮询本身有定时器，重试逻辑
 *   放在 hooks 层更合适，HTTP 层保持「一次请求一次结果」。
 */

import type {
  AskRequest,
  AskResponse,
  DocumentDeleteResponse,
  DocumentListResponse,
  DocumentStatusResponse,
  DocumentUploadRequest,
  DocumentUploadResponse,
  HealthResponse,
} from './types'

/** 所有客户端调用的基路径。相对路径 → 同源 → 被 Vite proxy 转到后端。 */
const BASE = '/api/v1'

/**
 * HTTP 错误。message 直接用后端 detail（人类可读），
 * status 保留下来方便调用方按 404/415/422 分支处理。
 */
export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/** 把任意 catch 到的东西转成可展示文案 —— 供 hooks / 组件统一用。 */
export function formatError(e: unknown): string {
  if (e instanceof ApiError) return e.message
  if (e instanceof Error) return e.message
  return String(e)
}

/**
 * 统一请求入口：拼 URL、解析 JSON、把非 2xx 转成 ApiError。
 * 错误体约定 { detail: string }（FastAPI HTTPException 的形状）。
 */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init)

  if (!res.ok) {
    // 后端错误体统一是 { detail }；万一是网关 HTML 等非 JSON，兜底用状态码文案
    let detail = `请求失败（HTTP ${res.status}）`
    try {
      const body = (await res.json()) as Partial<{ detail: unknown }>
      if (typeof body.detail === 'string' && body.detail.length > 0) {
        detail = body.detail
      }
    } catch {
      // 非 JSON 错误体：保留兜底文案
    }
    throw new ApiError(res.status, detail)
  }

  return (await res.json()) as T
}

// ---------------------------------------------------------------------------
// 运维
// ---------------------------------------------------------------------------

/** GET /api/v1/health —— 健康检查，头部徽章每 30s 调一次。 */
export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}

// ---------------------------------------------------------------------------
// 问答
// ---------------------------------------------------------------------------

/**
 * POST /api/v1/ask —— 同步问答。
 * history 只带本次 question 之前的轮次；question 单独放在请求体里。
 */
export function ask(body: AskRequest): Promise<AskResponse> {
  return request<AskResponse>('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

// ---------------------------------------------------------------------------
// 文档
// ---------------------------------------------------------------------------

/**
 * POST /api/v1/documents —— 粘贴文本入库（JSON）。
 * 返回 202：已接收，异步入库；调用方拿 doc_id 轮询状态。
 */
export function createDocument(body: DocumentUploadRequest): Promise<DocumentUploadResponse> {
  return request<DocumentUploadResponse>('/documents', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

/**
 * POST /api/v1/documents/upload —— multipart 文件上传（pdf/txt/md）。
 * 不手动设置 Content-Type —— 必须让浏览器自己带 multipart boundary。
 * 失败：415 不支持类型 / 422 扫描版、过大、空文本，错误体均为 { detail }。
 */
export function uploadFile(file: File, ownerId?: string | null): Promise<DocumentUploadResponse> {
  const form = new FormData()
  form.append('file', file)
  if (ownerId != null) form.append('owner_id', ownerId)
  return request<DocumentUploadResponse>('/documents/upload', {
    method: 'POST',
    body: form,
  })
}

/** GET /api/v1/documents?limit&offset —— 文档列表（创建时间倒序）。 */
export function listDocuments(limit = 100, offset = 0): Promise<DocumentListResponse> {
  return request<DocumentListResponse>(`/documents?limit=${limit}&offset=${offset}`)
}

/** GET /api/v1/documents/{doc_id} —— 单个文档状态。404 时抛 ApiError。 */
export function getDocument(docId: string): Promise<DocumentStatusResponse> {
  return request<DocumentStatusResponse>(`/documents/${encodeURIComponent(docId)}`)
}

/** DELETE /api/v1/documents/{doc_id} —— 删除文档及其向量。 */
export function deleteDocument(docId: string): Promise<DocumentDeleteResponse> {
  return request<DocumentDeleteResponse>(`/documents/${encodeURIComponent(docId)}`, {
    method: 'DELETE',
  })
}
