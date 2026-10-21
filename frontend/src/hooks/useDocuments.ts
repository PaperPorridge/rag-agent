/**
 * S7 · 文档列表 + 条件轮询 + 上传/删除。
 *
 * 【为什么轮询要「有条件」？】
 *   入库是异步的（202 + 轮询），但全部 settle（ready/failed）之后还每 2s
 *   打后端纯属浪费。只要有 pending/processing 就轮询，没有就停 ——
 *   用一个派生的 hasInflight 标志驱动 useEffect 开关定时器即可。
 * 【为什么删除后立刻 refresh 而不是本地过滤？】
 *   服务端才是列表事实源（还有 total 计数）。本地过滤快，但一旦失败就会
 *   出现「界面删了、后端还在」的幽灵行。宁可多一次请求保持一致。
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import {
  createDocument as createDocumentApi,
  deleteDocument as deleteDocumentApi,
  formatError,
  listDocuments,
  uploadFile as uploadFileApi,
} from '../api/client'
import type { DocumentStatusResponse } from '../api/types'

/** 列表里「仍在处理中」的状态 —— 命中任一就继续轮询。 */
const INFLIGHT: ReadonlySet<string> = new Set(['pending', 'processing'])

const POLL_MS = 2000

export interface UseDocuments {
  docs: DocumentStatusResponse[]
  total: number
  /** 首次加载中（避免闪空状态） */
  loading: boolean
  /** 上传/删除/拉取失败的可展示文案 */
  error: string | null
  /** 上传请求进行中（按钮禁用） */
  busy: boolean
  refresh: () => Promise<void>
  uploadFile: (file: File) => Promise<boolean>
  createDocument: (source: string, text: string) => Promise<boolean>
  remove: (docId: string) => Promise<boolean>
}

export function useDocuments(): UseDocuments {
  const [docs, setDocs] = useState<DocumentStatusResponse[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  // refresh 被 setInterval 回调引用，必须保持引用稳定，否则每次渲染都重置定时器
  const refresh = useCallback(async () => {
    try {
      const data = await listDocuments()
      setDocs(data.items)
      setTotal(data.total)
    } catch (e) {
      setError(formatError(e))
    } finally {
      setLoading(false)
    }
  }, [])

  // 首次加载
  useEffect(() => {
    void refresh()
  }, [refresh])

  // 条件轮询：只有存在未落定的文档时才跑 2s 定时器
  const hasInflight = docs.some((d) => INFLIGHT.has(d.status))
  const refreshRef = useRef(refresh)
  refreshRef.current = refresh

  useEffect(() => {
    if (!hasInflight) return
    const id = window.setInterval(() => {
      void refreshRef.current()
    }, POLL_MS)
    return () => window.clearInterval(id)
  }, [hasInflight])

  const uploadFile = useCallback(async (file: File): Promise<boolean> => {
    setBusy(true)
    setError(null)
    try {
      await uploadFileApi(file)
      await refresh()
      return true
    } catch (e) {
      setError(formatError(e))
      return false
    } finally {
      setBusy(false)
    }
  }, [refresh])

  const createDocument = useCallback(
    async (source: string, text: string): Promise<boolean> => {
      setBusy(true)
      setError(null)
      try {
        await createDocumentApi({ source, text, owner_id: null })
        await refresh()
        return true
      } catch (e) {
        setError(formatError(e))
        return false
      } finally {
        setBusy(false)
      }
    },
    [refresh],
  )

  const remove = useCallback(
    async (docId: string): Promise<boolean> => {
      setError(null)
      try {
        await deleteDocumentApi(docId)
        await refresh()
        return true
      } catch (e) {
        setError(formatError(e))
        return false
      }
    },
    [refresh],
  )

  return { docs, total, loading, error, busy, refresh, uploadFile, createDocument, remove }
}
