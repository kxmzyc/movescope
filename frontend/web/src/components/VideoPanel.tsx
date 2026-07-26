import { useEffect, useMemo, useRef } from 'react'
import type { RefObject } from 'react'
import { AlertCircle, FileVideo, FlaskConical } from 'lucide-react'

import type { Diagnosis, Skeleton } from '../api/types'

type Props = {
  previewUrl: string | null
  diagnosis: Diagnosis | null
  error: string | null
  videoRef?: RefObject<HTMLVideoElement | null>
  onSeek?: (timeSec: number) => void
}

type PhaseHighlight = {
  start: number
  end: number
  joints: Set<string>
}

export function VideoPanel({ previewUrl, diagnosis, error, videoRef, onSeek }: Props) {
  const fallbackRef = useRef<HTMLVideoElement | null>(null)
  const activeVideoRef = videoRef ?? fallbackRef
  const skeleton = diagnosis?.skeleton

  const highlights: PhaseHighlight[] = useMemo(() => {
    if (!diagnosis) return []
    return diagnosis.phases.map((phase) => ({
      start: phase.time_range[0],
      end: phase.time_range[1],
      joints: new Set(phase.anomalies.map((anomaly) => anomaly.joint)),
    }))
  }, [diagnosis])

  return (
    <section className="panel videoPanel">
      <div className="panelTitle">
        <FileVideo />
        <span>视频预览</span>
        {skeleton && previewUrl && <small className="panelTitleNote">骨架叠加中，红点为该阶段的异常关节</small>}
      </div>
      <div className="videoFrame">
        {previewUrl ? (
          <div className="videoWrap">
            <video src={previewUrl} controls ref={activeVideoRef} crossOrigin="anonymous" />
            {skeleton && (
              <SkeletonOverlay videoRef={activeVideoRef} skeleton={skeleton} highlights={highlights} />
            )}
          </div>
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

      {diagnosis && diagnosis.phases.length > 0 && (
        <PhaseBar diagnosis={diagnosis} onSeek={onSeek} />
      )}

      {error && (
        <div className="alert" role="alert" aria-live="polite">
          <AlertCircle />
          <span>{error}</span>
        </div>
      )}
    </section>
  )
}

function PhaseBar({ diagnosis, onSeek }: { diagnosis: Diagnosis; onSeek?: (timeSec: number) => void }) {
  const phases = diagnosis.phases
  const spanStart = phases[0].time_range[0]
  const spanEnd = phases[phases.length - 1].time_range[1]
  const total = Math.max(spanEnd - spanStart, 0.001)

  return (
    <div className="phaseBar" aria-label="动作阶段时间轴">
      {phases.map((phase) => {
        const width = ((phase.time_range[1] - phase.time_range[0]) / total) * 100
        const tone = phase.phase_score >= 90 ? 'good' : phase.phase_score >= 70 ? 'warn' : 'bad'
        return (
          <button
            key={phase.index}
            type="button"
            className={`phaseSegment ${tone}`}
            style={{ width: `${Math.max(width, 4)}%` }}
            title={`阶段 ${phase.index + 1}：${phase.time_range[0].toFixed(1)}-${phase.time_range[1].toFixed(1)} 秒，得分 ${phase.phase_score.toFixed(0)}`}
            onClick={() => onSeek?.(phase.time_range[0])}
          >
            <span>阶段 {phase.index + 1}</span>
            <strong>{phase.phase_score.toFixed(0)}</strong>
          </button>
        )
      })}
    </div>
  )
}

function SkeletonOverlay({
  videoRef,
  skeleton,
  highlights,
}: {
  videoRef: RefObject<HTMLVideoElement | null>
  skeleton: Skeleton
  highlights: PhaseHighlight[]
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    let raf = 0
    const draw = () => {
      raf = requestAnimationFrame(draw)
      const ctx = canvas.getContext('2d')
      if (!ctx || !video.videoWidth || !video.videoHeight) return

      const dpr = window.devicePixelRatio || 1
      const width = video.clientWidth
      const height = video.clientHeight
      if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
        canvas.width = Math.round(width * dpr)
        canvas.height = Math.round(height * dpr)
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, width, height)

      // <video> 使用 object-fit: contain，画面内容居中并按比例缩放；
      // 归一化关键点必须映射到实际画面区域，而不是元素矩形。
      const scale = Math.min(width / video.videoWidth, height / video.videoHeight)
      const drawWidth = video.videoWidth * scale
      const drawHeight = video.videoHeight * scale
      const offsetX = (width - drawWidth) / 2
      const offsetY = (height - drawHeight) / 2

      const t = video.currentTime
      const frame = Math.min(
        skeleton.keypoints.length - 1,
        Math.max(0, Math.round((t * skeleton.fps) / skeleton.frame_stride)),
      )
      const points = skeleton.keypoints[frame]
      const confidence = skeleton.confidence[frame] ?? []
      if (!points) return

      const highlighted = highlights.find((phase) => t >= phase.start && t <= phase.end + 0.05)?.joints

      ctx.lineWidth = 2
      ctx.strokeStyle = 'rgba(96, 190, 110, 0.9)'
      for (const [a, b] of skeleton.edges) {
        const pa = points[a]
        const pb = points[b]
        if (!pa || !pb) continue
        ctx.beginPath()
        ctx.moveTo(offsetX + pa[0] * drawWidth, offsetY + pa[1] * drawHeight)
        ctx.lineTo(offsetX + pb[0] * drawWidth, offsetY + pb[1] * drawHeight)
        ctx.stroke()
      }

      points.forEach((point, jointIdx) => {
        if (!point) return
        const x = offsetX + point[0] * drawWidth
        const y = offsetY + point[1] * drawHeight
        const jointName = skeleton.joint_names[jointIdx]
        const isHot = highlighted?.has(jointName) ?? false
        ctx.globalAlpha = (confidence[jointIdx] ?? 1) < 0.3 ? 0.35 : 1
        ctx.beginPath()
        ctx.arc(x, y, isHot ? 7 : 4, 0, Math.PI * 2)
        ctx.fillStyle = isHot ? '#e24830' : '#3fae57'
        ctx.fill()
        if (isHot) {
          ctx.lineWidth = 2
          ctx.strokeStyle = '#ffffff'
          ctx.stroke()
          ctx.lineWidth = 2
          ctx.strokeStyle = 'rgba(96, 190, 110, 0.9)'
        }
        ctx.globalAlpha = 1
      })
    }

    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [videoRef, skeleton, highlights])

  return <canvas ref={canvasRef} className="overlayCanvas" aria-hidden="true" data-testid="skeleton-overlay" />
}
