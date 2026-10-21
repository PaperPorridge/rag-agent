/**
 * S7 · 左侧文档面板：UploadForm + DocList。
 *
 * 数据全在 useDocuments 里 —— 组件只负责布局与事件转接，
 * 不自己发请求（保持组件无 IO，方便以后加「搜索文档」时不会挤在一起）。
 */

import { useDocuments } from '../hooks/useDocuments'
import { DocList } from './DocList'
import { UploadForm } from './UploadForm'

export function DocsPanel() {
  const { docs, total, loading, error, busy, uploadFile, createDocument, remove } = useDocuments()

  return (
    <aside className="panel docs-panel">
      <header className="panel-header">
        <h2>知识库文档</h2>
      </header>
      <UploadForm
        busy={busy}
        error={error}
        onUploadFile={uploadFile}
        onCreateText={createDocument}
      />
      <DocList docs={docs} total={total} loading={loading} onDelete={remove} />
    </aside>
  )
}
