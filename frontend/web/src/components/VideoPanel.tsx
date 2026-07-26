import { AlertCircle, FileVideo, FlaskConical } from 'lucide-react'

import type { Diagnosis } from '../api/types'

type Props = {
  previewUrl: string | null
  diagnosis: Diagnosis | null
  error: string | null
}

export function VideoPanel({ previewUrl, diagnosis, error }: Props) {
  return (
    <section className="panel videoPanel">
      <div className="panelTitle">
        <FileVideo />
        <span>视频预览</span>
      </div>
      <div className="videoFrame">
        {previewUrl ? (
          <video src={previewUrl} controls />
        ) : diagnosis?.metadata?.source === 'synthetic' ? (
          <div className="emptyVideo demoVisual">
            <FlaskConical />
            <strong>合成角度序列</strong>
            <span>固定 {diagnosis.metadata.frames} 帧深蹲诊断</span>
          </div>
        ) : (
          <div className="emptyVideo">
            <FileVideo />
            <span>尚未选择视频</span>
          </div>
        )}
      </div>
      {error && (
        <div className="alert" role="alert" aria-live="polite">
          <AlertCircle />
          <span>{error}</span>
        </div>
      )}
    </section>
  )
}
