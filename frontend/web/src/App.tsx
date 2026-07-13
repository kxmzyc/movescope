import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'
import {
  Activity,
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Download,
  FileVideo,
  FlaskConical,
  Loader2,
  RefreshCw,
  Server,
  Upload,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import './App.css'

type JointSummary = Record<string, { mean_dev: number; anomaly_ratio: number }>

type Anomaly = {
  joint_name: string
  joint_idx: number
  direction: string
  mean_deviation_deg: number
  peak_deviation_deg: number
  peak_time_sec: number
  anomaly_ratio: number
}

type DiagnosisPhase = {
  name: string
  time_range: [number, number]
  phase_score: number
  anomalies: Anomaly[]
}

type Diagnosis = {
  action: string
  total_score: number
  phases: DiagnosisPhase[]
  per_joint_summary: JointSummary
  llm_advice?: string
  metadata?: {
    source: string
    label: string
    disclaimer: string
    frames: number
  }
  quality?: {
    frames: number
    fps: number
    valid_pose_ratio: number
    pose_source: string
  }
}

type Health = {
  status: string
  version: string
}

const API_BASE = import.meta.env.VITE_MOVESCOPE_API ?? 'http://127.0.0.1:8000'
const MAX_VIDEO_BYTES = 100 * 1024 * 1024
const VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.webm', '.mkv']
const ACTION_LABELS: Record<string, string> = { squat: '深蹲' }
const JOINT_LABELS: Record<string, string> = {
  pelvis: '骨盆',
  left_hip: '左髋',
  right_hip: '右髋',
  left_knee: '左膝',
  right_knee: '右膝',
  left_ankle: '左踝',
  right_ankle: '右踝',
  left_shoulder: '左肩',
  right_shoulder: '右肩',
  left_elbow: '左肘',
  right_elbow: '右肘',
  left_wrist: '左腕',
  right_wrist: '右腕',
  head: '头部',
  neck: '颈部',
  left_eye: '左眼',
  right_eye: '右眼',
}

function App() {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [action, setAction] = useState('squat')
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [diagnosis, setDiagnosis] = useState<Diagnosis | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [availableActions, setAvailableActions] = useState<string[]>([])
  const [status, setStatus] = useState<'idle' | 'checking' | 'uploading' | 'demo'>('checking')
  const [error, setError] = useState<string | null>(null)

  const jointRows = useMemo(() => {
    if (!diagnosis) return []
    return Object.entries(diagnosis.per_joint_summary)
      .map(([name, value]) => ({
        name: jointLabel(name),
        meanDev: Number(value.mean_dev.toFixed(2)),
        anomalyRate: Number((value.anomaly_ratio * 100).toFixed(1)),
      }))
      .sort((a, b) => b.meanDev - a.meanDev)
      .slice(0, 8)
  }, [diagnosis])

  const topAnomalies = useMemo(() => {
    return (
      diagnosis?.phases
        .flatMap((phase) => phase.anomalies.map((anomaly) => ({ ...anomaly, phase })))
        .sort((a, b) => b.mean_deviation_deg - a.mean_deviation_deg) ?? []
    )
  }, [diagnosis])

  const checkApi = useCallback(async () => {
    setStatus('checking')
    setError(null)
    try {
      const [healthResponse, actionsResponse] = await Promise.all([
        axios.get<Health>(`${API_BASE}/health`, { timeout: 5000 }),
        axios.get<{ actions: string[] }>(`${API_BASE}/actions`, { timeout: 5000 }),
      ])
      setHealth(healthResponse.data)
      setAvailableActions(actionsResponse.data.actions)
      setAction((current) =>
        actionsResponse.data.actions.length && !actionsResponse.data.actions.includes(current)
          ? actionsResponse.data.actions[0]
          : current,
      )
    } catch (err) {
      setError(readError(err, '无法连接 API，请确认 FastAPI 已在 8000 端口启动。'))
      setHealth(null)
      setAvailableActions([])
    } finally {
      setStatus('idle')
    }
  }, [])

  useEffect(() => {
    void checkApi()
  }, [checkApi])

  function selectFile(nextFile: File | null) {
    if (nextFile && !isSupportedVideo(nextFile)) {
      setError('请选择 MP4、MOV、AVI、WEBM 或 MKV 格式的视频。')
      return
    }
    if (nextFile && nextFile.size > MAX_VIDEO_BYTES) {
      setError('视频超过 100 MB 上传上限。')
      return
    }
    setFile(nextFile)
    setDiagnosis(null)
    setError(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(nextFile ? URL.createObjectURL(nextFile) : null)
  }

  async function submitAssessment() {
    if (!file) {
      setError('请先选择视频，再开始评估。')
      return
    }

    const body = new FormData()
    body.append('video', file)
    body.append('action', action.trim() || 'squat')

    setStatus('uploading')
    setError(null)
    setDiagnosis(null)
    try {
      const response = await axios.post<Diagnosis>(`${API_BASE}/assess`, body, {
        timeout: 300_000,
      })
      setDiagnosis(response.data)
    } catch (err) {
      setError(readError(err, '评估失败，请检查视频和动作模板。'))
    } finally {
      setStatus('idle')
    }
  }

  async function runDemo() {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setFile(null)
    setPreviewUrl(null)
    setStatus('demo')
    setError(null)
    setDiagnosis(null)
    try {
      const response = await axios.get<Diagnosis>(`${API_BASE}/demo`, { timeout: 15_000 })
      setDiagnosis(response.data)
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

  const busy = status !== 'idle'

  return (
    <main className="workspace">
      <header className="topbar">
        <div>
          <p className="eyebrow">MoveScope</p>
          <h1>单目深蹲动作评估工作台</h1>
        </div>
        <button className="statusButton" type="button" onClick={checkApi} disabled={busy}>
          {status === 'checking' ? <Loader2 className="spin" /> : health ? <CheckCircle2 /> : <RefreshCw />}
          {health ? `API v${health.version} 已连接` : '检查 API'}
        </button>
      </header>

      <section className="grid">
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
              selectFile(event.dataTransfer.files.item(0))
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
            onChange={(event) => selectFile(event.target.files?.item(0) ?? null)}
          />

          <label className="field">
            <span>动作类型</span>
            <select value={action} onChange={(event) => setAction(event.target.value)}>
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

          <button className="primary" type="button" disabled={busy || !file} onClick={submitAssessment}>
            {status === 'uploading' ? <Loader2 className="spin" /> : <Activity />}
            {status === 'uploading' ? '正在评估...' : '开始评估'}
          </button>

          <button className="secondary" type="button" disabled={busy || !health} onClick={runDemo}>
            {status === 'demo' ? <Loader2 className="spin" /> : <FlaskConical />}
            {status === 'demo' ? '正在生成...' : '运行合成演示'}
          </button>

          <div className="hint">
            <Server />
            <span>{health ? `${API_BASE} · 已连接` : API_BASE}</span>
          </div>
        </aside>

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
                <span>固定 72 帧深蹲诊断</span>
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

        <aside className="panel results">
          <div className="scoreHeader">
            <div>
              <p className="eyebrow">总分</p>
              <strong>{diagnosis ? diagnosis.total_score.toFixed(1) : '--'}</strong>
              <span className="scoreUnit">/ 100</span>
            </div>
            <div className={diagnosis ? 'scoreBadge active' : 'scoreBadge'}>
              {diagnosis ? <CheckCircle2 /> : <BarChart3 />}
            </div>
          </div>

          {diagnosis && (
            <div className="resultMeta">
              <span className={diagnosis.metadata ? 'sourceBadge synthetic' : 'sourceBadge'}>
                {diagnosis.metadata ? '合成验证' : '上传视频'}
              </span>
              <button className="iconCommand" type="button" onClick={downloadReport} title="下载 JSON 报告">
                <Download />
                导出 JSON
              </button>
            </div>
          )}

          <div className="chartBox">
            {jointRows.length ? (
              <ResponsiveContainer width="100%" height={230}>
                <BarChart data={jointRows} layout="vertical" margin={{ left: 8, right: 12, top: 8, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#d7dbe2" />
                  <XAxis type="number" unit="°" tick={{ fontSize: 11 }} stroke="#68707d" />
                  <YAxis dataKey="name" type="category" width={86} tick={{ fontSize: 11 }} stroke="#68707d" />
                  <Tooltip
                    cursor={{ fill: '#eef2f6' }}
                    formatter={(value) => [`${Number(value).toFixed(2)}°`, '平均偏差']}
                    labelFormatter={(label) => `关节：${label}`}
                  />
                  <Bar dataKey="meanDev" fill="#c14835" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="emptyChart">完成评估后将在这里显示各关节偏差。</div>
            )}
          </div>

          <section className="summary">
            <h2>主要问题</h2>
            {topAnomalies.length ? (
              <ul>
                {topAnomalies.slice(0, 4).map((item) => (
                  <li key={`${item.phase.name}-${item.joint_idx}`}>
                    <span>
                      {jointLabel(item.joint_name)}
                      <small>
                        {item.peak_time_sec.toFixed(2)} 秒峰值 · {(item.anomaly_ratio * 100).toFixed(0)}% 异常
                      </small>
                    </span>
                    <strong>{item.mean_deviation_deg.toFixed(1)}°</strong>
                  </li>
                ))}
              </ul>
            ) : (
              <p>暂无评估结果。</p>
            )}
          </section>

          <section className="advice">
            <h2>训练建议</h2>
            <p>{diagnosis?.llm_advice ?? '后端返回诊断结果后，将在这里显示动作纠正建议。'}</p>
            {diagnosis?.metadata && <small className="disclaimer">{diagnosis.metadata.disclaimer}</small>}
          </section>
        </aside>
      </section>
    </main>
  )
}

function jointLabel(name: string) {
  const key = name.split(':', 1)[0]
  return JOINT_LABELS[key] ?? key
}

function actionLabel(name: string) {
  return ACTION_LABELS[name] ?? name
}

function isSupportedVideo(file: File) {
  const lowerName = file.name.toLowerCase()
  return file.type.startsWith('video/') || VIDEO_EXTENSIONS.some((extension) => lowerName.endsWith(extension))
}

function readError(err: unknown, fallback: string) {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail
    if (typeof detail === 'string') return detail
  }
  return fallback
}

export default App
