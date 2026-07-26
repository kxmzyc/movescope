import { useEffect, useState } from 'react'

import { assessVideo, fetchDemo, readError } from '../api/client'
import type { Diagnosis } from '../api/types'
import { FALLBACK_MAX_VIDEO_BYTES, FALLBACK_VIDEO_EXTENSIONS, isSupportedVideo } from '../constants'

export type AssessmentStatus = 'idle' | 'uploading' | 'demo'

type Limits = {
  maxVideoBytes?: number
  allowedExtensions?: string[]
}

export function useAssessment({ maxVideoBytes, allowedExtensions }: Limits = {}) {
  const maxBytes = maxVideoBytes ?? FALLBACK_MAX_VIDEO_BYTES
  const extensions = allowedExtensions ?? FALLBACK_VIDEO_EXTENSIONS

  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [diagnosis, setDiagnosis] = useState<Diagnosis | null>(null)
  const [status, setStatus] = useState<AssessmentStatus>('idle')
  const [error, setError] = useState<string | null>(null)

  // 统一在这里回收 object URL：previewUrl 变化时回收旧值，组件卸载时回收当前值。
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  function selectFile(nextFile: File | null) {
    if (nextFile && !isSupportedVideo(nextFile, extensions)) {
      setError('请选择 MP4、MOV、AVI、WEBM 或 MKV 格式的视频。')
      return
    }
    if (nextFile && nextFile.size > maxBytes) {
      setError(`视频超过 ${Math.floor(maxBytes / 1024 / 1024)} MB 上传上限。`)
      return
    }
    setFile(nextFile)
    setDiagnosis(null)
    setError(null)
    setPreviewUrl(nextFile ? URL.createObjectURL(nextFile) : null)
  }

  async function submit(action: string) {
    if (!file) {
      setError('请先选择视频，再开始评估。')
      return
    }

    setStatus('uploading')
    setError(null)
    setDiagnosis(null)
    try {
      setDiagnosis(await assessVideo(file, action.trim() || 'squat'))
    } catch (err) {
      setError(readError(err, '评估失败，请检查视频和动作模板。'))
    } finally {
      setStatus('idle')
    }
  }

  async function runDemo() {
    setFile(null)
    setPreviewUrl(null)
    setStatus('demo')
    setError(null)
    setDiagnosis(null)
    try {
      setDiagnosis(await fetchDemo())
    } catch (err) {
      setError(readError(err, '合成演示运行失败，请检查 API 连接。'))
    } finally {
      setStatus('idle')
    }
  }

  function downloadReport() {
    if (!diagnosis) return
    const blob = new Blob([JSON.stringify(diagnosis, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `movescope-${diagnosis.action}-report.json`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  return { file, previewUrl, diagnosis, status, error, selectFile, submit, runDemo, downloadReport }
}
