import { useCallback, useEffect, useRef, useState } from 'react'

import './App.css'
import { ControlsPanel } from './components/ControlsPanel'
import { ResultsPanel } from './components/ResultsPanel'
import { TimelinePanel } from './components/TimelinePanel'
import { TopBar } from './components/TopBar'
import { VideoPanel } from './components/VideoPanel'
import { useApiStatus } from './hooks/useApiStatus'
import { useAssessment } from './hooks/useAssessment'

function App() {
  const [action, setAction] = useState('squat')
  const api = useApiStatus()
  const assessment = useAssessment({
    maxVideoBytes: api.health?.max_upload_bytes,
    allowedExtensions: api.health?.allowed_extensions,
  })
  const videoRef = useRef<HTMLVideoElement | null>(null)

  useEffect(() => {
    if (api.availableActions.length && !api.availableActions.includes(action)) {
      setAction(api.availableActions[0])
    }
  }, [api.availableActions, action])

  const seekTo = useCallback((timeSec: number) => {
    const video = videoRef.current
    if (!video || !Number.isFinite(timeSec)) return
    video.currentTime = Math.max(0, timeSec)
    video.pause()
  }, [])

  const busy = api.checking || assessment.status !== 'idle'
  const error = assessment.error ?? api.error

  return (
    <main className="workspace">
      <TopBar health={api.health} checking={api.checking} busy={busy} onRefresh={api.refresh} />

      <section className="grid">
        <ControlsPanel
          file={assessment.file}
          action={action}
          availableActions={api.availableActions}
          health={api.health}
          busy={busy}
          status={assessment.status}
          onSelectFile={assessment.selectFile}
          onActionChange={setAction}
          onSubmit={() => void assessment.submit(action)}
          onDemo={() => void assessment.runDemo()}
        />
        <VideoPanel
          previewUrl={assessment.previewUrl}
          diagnosis={assessment.diagnosis}
          error={error}
          videoRef={videoRef}
          onSeek={seekTo}
        />
        <ResultsPanel diagnosis={assessment.diagnosis} onDownload={assessment.downloadReport} onSeek={seekTo} />
      </section>

      {assessment.diagnosis?.timeline && (
        <section className="timelineRow">
          <TimelinePanel diagnosis={assessment.diagnosis} onSeek={seekTo} />
        </section>
      )}
    </main>
  )
}

export default App
