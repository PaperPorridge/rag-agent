/**
 * S7 · 上传表单：拖拽/点击选文件，或切换到粘贴文本模式。
 *
 * 【为什么文件模式和文本模式放一个组件？】
 *   两个入口产出同一件事（入库一份文档），只是载体不同。
 *   切换按钮只是本地 mode state，不需要路由。
 * 【为什么 accept 里同时写扩展名和 MIME？】
 *   Windows 的文件选择框认扩展名，拖拽路径走 MIME —— 两边都写才稳。
 */

import { useRef, useState } from 'react'

export interface UploadFormProps {
  /** 上传请求进行中 */
  busy: boolean
  /** 来自 useDocuments 的错误文案 */
  error: string | null
  onUploadFile: (file: File) => Promise<boolean>
  onCreateText: (source: string, text: string) => Promise<boolean>
}

type Mode = 'file' | 'text'

const ACCEPT = '.pdf,.txt,.md,application/pdf,text/plain,text/markdown'

export function UploadForm({ busy, error, onUploadFile, onCreateText }: UploadFormProps) {
  const [mode, setMode] = useState<Mode>('file')
  const [dragOver, setDragOver] = useState(false)
  const [source, setSource] = useState('')
  const [text, setText] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFile = (files: FileList | null) => {
    const file = files?.[0]
    if (!file || busy) return
    void onUploadFile(file)
  }

  const handleTextSubmit = () => {
    const s = source.trim()
    const t = text
    if (!s || !t.trim() || busy) return
    // 成功才清空，失败保留输入让用户改了重试
    void onCreateText(s, t).then((ok) => {
      if (ok) {
        setSource('')
        setText('')
      }
    })
  }

  return (
    <section className="upload-form">
      <div className="segmented">
        <button
          type="button"
          className={mode === 'file' ? 'active' : ''}
          onClick={() => setMode('file')}
        >
          上传文件
        </button>
        <button
          type="button"
          className={mode === 'text' ? 'active' : ''}
          onClick={() => setMode('text')}
        >
          粘贴文本
        </button>
      </div>

      {mode === 'file' ? (
        <div
          className={`dropzone${dragOver ? ' dragover' : ''}`}
          role="button"
          tabIndex={0}
          aria-label="拖拽或点击选择文件"
          // 整个区域可点击 —— 比「只能拖」和「只能点」都友好
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              fileInputRef.current?.click()
            }
          }}
          onDragOver={(e) => {
            // 不 preventDefault 浏览器会直接打开文件，不会触发 drop
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            handleFile(e.dataTransfer.files)
          }}
        >
          <div className="dropzone-icon">⬆</div>
          <div className="dropzone-text">
            拖拽文件到此处，或<b>点击选择</b>
          </div>
          <div className="dropzone-hint">支持 PDF / TXT / MD</div>
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPT}
            hidden
            onChange={(e) => {
              handleFile(e.target.files)
              // 清空 value：同一个文件连续选两次也能触发 change
              e.target.value = ''
            }}
          />
        </div>
      ) : (
        <div className="paste-form">
          <input
            type="text"
            className="input"
            placeholder="来源名称，如 notes.md"
            value={source}
            maxLength={512}
            onChange={(e) => setSource(e.target.value)}
          />
          <textarea
            className="input textarea"
            placeholder="粘贴文档全文…"
            value={text}
            rows={8}
            onChange={(e) => setText(e.target.value)}
          />
          <button
            type="button"
            className="btn primary"
            disabled={busy || !source.trim() || !text.trim()}
            onClick={handleTextSubmit}
          >
            {busy ? '提交中…' : '入库'}
          </button>
        </div>
      )}

      {error ? <div className="inline-error">{error}</div> : null}
    </section>
  )
}
