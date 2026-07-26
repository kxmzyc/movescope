// 与 api/schemas.py 中的 Pydantic 模型一一对应。

export type Health = {
  status: string
  version: string
  max_upload_bytes: number
  allowed_extensions: string[]
}

export type TemplateInfo = {
  action: string
  n_videos: number
  feature_dim: number
  frames: number
}

export type FeatureSummary = {
  feature_index: number
  joint: string
  joint_display: string
  parent: string
  child: string
  mean_dev: number
  anomaly_ratio: number
  tolerance_deg: number
  score_weight: number
}

export type ExcludedFeature = {
  feature_index: number
  joint: string
  joint_display: string
  parent: string
  child: string
  reason: string
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
  label: string
  index: number
  time_range: [number, number]
  phase_score: number
  anomalies: Anomaly[]
}

export type TimelineSeries = {
  feature_index: number
  joint: string
  joint_display: string
  parent: string
  child: string
  // 逐帧容差带，与 test_deg 等长（旧模板由后端广播为等值数组）。
  tolerance_deg: number[]
  test_deg: number[]
  reference_deg: number[]
  anomaly: boolean[]
}

export type Timeline = {
  fps: number
  frame_count: number
  frame_stride: number
  time_sec: number[]
  series: TimelineSeries[]
}

export type Skeleton = {
  fps: number
  frame_count: number
  frame_stride: number
  time_sec: number[]
  joint_names: string[]
  edges: [number, number][]
  keypoints: ([number, number] | null)[][]
  confidence: number[][]
}

export type Diagnosis = {
  action: string
  total_score: number
  segmented: boolean
  phases: DiagnosisPhase[]
  per_feature_summary: FeatureSummary[]
  excluded_features?: ExcludedFeature[]
  timeline?: Timeline
  skeleton?: Skeleton
  llm_advice?: string
  advice_source?: string
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
