/**
 * S7 · API 契约类型 —— 与 backend/knowflow/api/schemas.py 一一对应。
 *
 * 【为什么前端也维护一份「契约文件」？】
 *   1. 后端的 Pydantic 模型是服务端事实源，这里是前端事实源，两边字段名必须同名同形
 *   2. client.ts / hooks / components 全部从这里导入 —— 改契约只改一处
 *   3. 导出签名里不允许 any：类型写错在 `npm run build` 阶段就炸，不用等运行时
 *
 * 字段说明与 FastAPI /docs 中的 description 保持一致，便于对照排查。
 */

// ---------------------------------------------------------------------------
// 通用
// ---------------------------------------------------------------------------

/**
 * 后端 4xx/5xx 的统一错误体：{ "detail": "human readable message" }。
 * FastAPI 的 HTTPException / 校验失败都是这个形状。
 */
export interface ApiErrorBody {
  detail: string
}

// ---------------------------------------------------------------------------
// GET /api/v1/health
// ---------------------------------------------------------------------------

/**
 * 组件健康值：正常为 "ok"，队列是 "memory"/"kafka"，异常为 "error: XxxError"。
 * 用模板字面量类型收窄 error 形态，同时保留字符串的可读性。
 */
export type ComponentHealth = 'ok' | 'memory' | 'kafka' | `error: ${string}`

export interface HealthResponse {
  /** 整体状态：ok = 全组件正常，degraded = 至少一个组件异常 */
  status: 'ok' | 'degraded'
  version: string
  components: Record<string, ComponentHealth>
}

// ---------------------------------------------------------------------------
// POST /api/v1/ask
// ---------------------------------------------------------------------------

/** 多轮对话里的一轮。role 只有 user / assistant（system 是服务端的事）。 */
export interface ChatTurn {
  role: 'user' | 'assistant'
  content: string
}

export interface AskRequest {
  question: string
  top_k: number
  history: ChatTurn[]
}

/** 一条被检索到的证据块。 */
export interface RetrievedSource {
  /** 引用标记，如 "差旅报销制度.pdf#0" */
  citation: string
  source: string
  chunk_index: number
  snippet: string
  /** 融合得分（重排后优先，否则 RRF） */
  score: number
}

/** Agent 一次工具调用的轨迹。chain === "rag" 时恒为空数组。 */
export interface ToolCallTrace {
  /** 第几步（1 起） */
  step: number
  tool_name: string
  arguments: Record<string, unknown>
  /** 工具返回（后端已截断到 600 字符） */
  result: string
  ok: boolean
  error: string | null
  latency_ms: number
}

export interface AskResponse {
  answer: string
  question: string
  /** rag = 直接问答，agent = 工具循环 */
  chain: 'rag' | 'agent'
  /** 路由分流理由，便于排查误判 */
  router_reason: string
  /** 来源文件名列表 */
  sources: string[]
  retrieved: RetrievedSource[]
  tool_traces: ToolCallTrace[]
  citations_valid: boolean
  citation_errors: string[]
  n_retrieved: number
  /** Agent 步数（直答为 0） */
  steps: number
  cost_cny: number
}

// ---------------------------------------------------------------------------
// 文档（异步入库）
// ---------------------------------------------------------------------------

/** POST /documents 的请求体（粘贴文本方式）。 */
export interface DocumentUploadRequest {
  source: string
  text: string
  owner_id: string | null
}

export type DocumentStatus = 'pending' | 'processing' | 'ready' | 'failed'

/** GET /documents/{doc_id} 的响应体，也是列表项形状。 */
export interface DocumentStatusResponse {
  doc_id: string
  source: string
  status: DocumentStatus
  chunk_count: number
  error: string | null
  /** ISO8601，后端可能返回 null */
  created_at: string | null
}

/** POST /documents 与 /documents/upload 返回 202 时的响应体。 */
export interface DocumentUploadResponse {
  doc_id: string
  message_id: string
  status: string
  detail: string
}

export interface DocumentListResponse {
  items: DocumentStatusResponse[]
  total: number
}

export interface DocumentDeleteResponse {
  doc_id: string
  deleted: boolean
  detail: string
}
