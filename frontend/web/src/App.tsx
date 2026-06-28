import { useMemo, useRef, useState } from 'react'
import axios from 'axios'
import {
  Activity,
  AlertCircle,
  BarChart3,
  CheckCircle2,
  FileVideo,
  Loader2,
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
}

type Health = {
  status: string
  version: string
}

const API_BASE = import.meta.env.VITE_MOVESCOPE_API ?? 'http://127.0.0.1:8000'

function App() {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [action, setAction] = useState('squat')
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [diagnosis, setDiagnosis] = useState<Diagnosis | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [status, setStatus] = useState<'idle' | 'checking' | 'uploading'>('idle')
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
    return diagnosis?.phases.flatMap((phase) =>
      phase.anomalies.map((anomaly) => ({ ...anomaly, phase })),
    ) ?? []
  }, [diagnosis])

  async function checkApi() {
    setStatus('checking')
    setError(null)
    try {
      const response = await axios.get<Health>(`${API_BASE}/health`, { timeout: 5000 })
      setHealth(response.data)
    } catch (err) {
      setError(readError(err, 'API is not reachable. Start FastAPI on port 8000.'))
      setHealth(null)
    } finally {
      setStatus('idle')
    }
  }

  function selectFile(nextFile: File | null) {
    setFile(nextFile)
    setDiagnosis(null)
    setError(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(nextFile ? URL.createObjectURL(nextFile) : null)
  }

  async function submitAssessment() {
    if (!file) {
      setError('Select a video before running assessment.')
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
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setDiagnosis(response.data)
    } catch (err) {
      setError(readError(err, 'Assessment failed.'))
    } finally {
      setStatus('idle')
    }
  }

  const busy = status !== 'idle'

  return (
    <main className="workspace">
      <header className="topbar">
        <div>
          <p className="eyebrow">MoveScope</p>
          <h1>Monocular squat assessment console</h1>
        </div>
        <button className="statusButton" type="button" onClick={checkApi} disabled={busy}>
          {status === 'checking' ? <Loader2 className="spin" /> : <Server />}
          {health ? `API ${health.version}` : 'Check API'}
        </button>
      </header>

      <section className="grid">
        <aside className="panel controls">
          <div className="panelTitle">
            <Upload />
            <span>Input</span>
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
            <strong>{file ? file.name : 'Choose or drop a video'}</strong>
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
            <span>Action</span>
            <input value={action} onChange={(event) => setAction(event.target.value)} />
          </label>

          <button className="primary" type="button" disabled={busy || !file} onClick={submitAssessment}>
            {status === 'uploading' ? <Loader2 className="spin" /> : <Activity />}
            Run assessment
          </button>

          <div className="hint">
            <Server />
            <span>{API_BASE}</span>
          </div>
        </aside>

        <section className="panel videoPanel">
          <div className="panelTitle">
            <FileVideo />
            <span>Video Review</span>
          </div>
          <div className="videoFrame">
            {previewUrl ? (
              <video src={previewUrl} controls />
            ) : (
              <div className="emptyVideo">
                <FileVideo />
                <span>No video selected</span>
              </div>
            )}
          </div>
          {error && (
            <div className="alert">
              <AlertCircle />
              <span>{error}</span>
            </div>
          )}
        </section>

        <aside className="panel results">
          <div className="scoreHeader">
            <div>
              <p className="eyebrow">Score</p>
              <strong>{diagnosis ? diagnosis.total_score.toFixed(1) : '--'}</strong>
            </div>
            <div className={diagnosis ? 'scoreBadge active' : 'scoreBadge'}>
              {diagnosis ? <CheckCircle2 /> : <BarChart3 />}
            </div>
          </div>

          <div className="chartBox">
            {jointRows.length ? (
              <ResponsiveContainer width="100%" height={230}>
                <BarChart data={jointRows} layout="vertical" margin={{ left: 8, right: 12, top: 8, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#d7dbe2" />
                  <XAxis type="number" tick={{ fontSize: 11 }} stroke="#68707d" />
                  <YAxis dataKey="name" type="category" width={86} tick={{ fontSize: 11 }} stroke="#68707d" />
                  <Tooltip cursor={{ fill: '#eef2f6' }} />
                  <Bar dataKey="meanDev" fill="#c14835" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="emptyChart">Joint deviations appear here after assessment.</div>
            )}
          </div>

          <section className="summary">
            <h2>Findings</h2>
            {topAnomalies.length ? (
              <ul>
                {topAnomalies.slice(0, 4).map((item) => (
                  <li key={`${item.phase.name}-${item.joint_idx}`}>
                    <span>{jointLabel(item.joint_name)}</span>
                    <strong>{item.mean_deviation_deg.toFixed(1)} deg</strong>
                  </li>
                ))}
              </ul>
            ) : (
              <p>No assessment result yet.</p>
            )}
          </section>

          <section className="advice">
            <h2>Advice</h2>
            <p>{diagnosis?.llm_advice ?? 'Correction advice will appear after the backend returns a diagnosis.'}</p>
          </section>
        </aside>
      </section>
    </main>
  )
}

function jointLabel(name: string) {
  return name.split(':', 1)[0].replaceAll('_', ' ')
}

function readError(err: unknown, fallback: string) {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail
    if (typeof detail === 'string') return detail
    if (err.message) return err.message
  }
  return fallback
}

export default App
