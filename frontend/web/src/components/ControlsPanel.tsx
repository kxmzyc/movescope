import { useRef } from 'react'
import { Activity, AlertCircle, FileVideo, FlaskConical, Loader2, Server, Upload } from 'lucide-react'

import type { Health } from '../api/types'
import { API_BASE, actionLabel } from '../constants'
import type { AssessmentStatus } from '../hooks/useAssessment'

type Props = {
  file: File | null
  action: string
  availableActions: string[]
  health: Health | null
  busy: boolean
  status: AssessmentStatus
  onSelectFile: (file: File | null) => void
  onActionChange: (action: string) => void
  onSubmit: () => void
  onDemo: () => void
}

export function ControlsPanel({
  file,
  action,
  availableActions,
  health,
  busy,
  status,
  onSelectFile,
  onActionChange,
  onSubmit,
  onDemo,
}: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null)

  return (
    <aside className="panel controls">
      <div className="panelTitle">
        <Upload />
        <span>评估输入</span>
      </div>

      <button
        className="dropzone"
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault()
          onSelectFile(event.dataTransfer.files.item(0))
        }}
      >
        <FileVideo />
        <strong>{file ? file.name : '选择或拖入视频'}</strong>
        <span>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : 'MP4, MOV, AVI, WEBM'}</span>
      </button>

      <input
        ref={inputRef}
        className="hiddenInput"
        type="file"
        accept="video/*"
        onChange={(event) => onSelectFile(event.target.files?.item(0) ?? null)}
      />

      <label className="field">
        <span>动作类型</span>
        <select value={action} onChange={(event) => onActionChange(event.target.value)}>
          {(availableActions.length ? availableActions : ['squat']).map((name) => (
            <option key={name} value={name}>
              {actionLabel(name)}
            </option>
          ))}
        </select>
      </label>

      {!availableActions.length && health && (
        <div className="templateNotice">
          <AlertCircle />
          <span>未找到本地动作模板。请先构建模板后评估视频，或直接运行合成演示。</span>
        </div>
      )}

      <button className="primary" type="button" disabled={busy || !file} onClick={onSubmit}>
        {status === 'uploading' ? <Loader2 className="spin" /> : <Activity />}
        {status === 'uploading' ? '正在评估...' : '开始评估'}
      </button>

      <button className="secondary" type="button" disabled={busy || !health} onClick={onDemo}>
        {status === 'demo' ? <Loader2 className="spin" /> : <FlaskConical />}
        {status === 'demo' ? '正在生成...' : '运行合成演示'}
      </button>

      <div className="hint">
        <Server />
        <span>{health ? `${API_BASE} · 已连接` : API_BASE}</span>
      </div>
    </aside>
  )
}
