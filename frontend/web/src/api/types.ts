// 与 api/schemas.py 中的 Pydantic 模型一一对应。

export type Health = {
  status: string
  version: string
  max_upload_bytes: number
  allowed_extensions: string[]
}

export type FeatureSummary = {
  feature_index: number
  joint: string
  joint_display: string
  parent: string
  child: string
  mean_dev: number
  anomaly_ratio: number
}

export type Anomaly = {
  feature_index: number
  joint: string
  joint_display: string
  parent: string
  child: string
  direction: string
  mean_deviation_deg: number
  peak_deviation_deg: number
  peak_time_sec: number
  anomaly_ratio: number
}

export type DiagnosisPhase = {
  name: string
  index: number
  time_range: [number, number]
  phase_score: number
  anomalies: Anomaly[]
}

export type Diagnosis = {
  action: string
  total_score: number
  segmented: boolean
  phases: DiagnosisPhase[]
  per_feature_summary: FeatureSummary[]
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
