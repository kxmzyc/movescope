"""API 请求/响应的 Pydantic 模型（响应 schema 唯一真源）。

前端 TypeScript 类型（frontend/web/src/api/types.ts）与此保持一一对应。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    max_upload_bytes: int
    allowed_extensions: list[str]


class TemplateInfoModel(BaseModel):
    action: str
    n_videos: int
    feature_dim: int
    frames: int


class ActionsResponse(BaseModel):
    actions: list[str]
    templates: list[TemplateInfoModel] = Field(default_factory=list)


class AnomalyModel(BaseModel):
    feature_index: int
    joint: str
    joint_display: str
    parent: str
    child: str
    direction: str
    mean_deviation_deg: float
    peak_deviation_deg: float
    peak_time_sec: float
    anomaly_ratio: float


class PhaseModel(BaseModel):
    name: str
    label: str
    index: int
    time_range: list[float] = Field(min_length=2, max_length=2)
    phase_score: float
    anomalies: list[AnomalyModel]


class FeatureSummaryModel(BaseModel):
    feature_index: int
    joint: str
    joint_display: str
    parent: str
    child: str
    mean_dev: float
    anomaly_ratio: float
    tolerance_deg: float
    score_weight: float


class ExcludedFeatureModel(BaseModel):
    feature_index: int
    joint: str
    joint_display: str
    parent: str
    child: str
    reason: str


class TimelineSeriesModel(BaseModel):
    feature_index: int
    joint: str
    joint_display: str
    parent: str
    child: str
    # 模板 v2 起为逐帧容差带（与 test_deg 等长）；旧模板由引擎广播为等值数组。
    tolerance_deg: list[float]
    test_deg: list[float]
    reference_deg: list[float]
    anomaly: list[bool]


class RepSummaryModel(BaseModel):
    """一次深蹲往复的摘要（多次往复视频逐次评分时返回）。"""

    index: int
    time_range: list[float] = Field(min_length=2, max_length=2)
    score: float
    knee_min_deg: float


class TimelineModel(BaseModel):
    fps: float
    frame_count: int
    frame_stride: int
    time_sec: list[float]
    series: list[TimelineSeriesModel]


class SkeletonModel(BaseModel):
    fps: float
    frame_count: int
    frame_stride: int
    time_sec: list[float]
    joint_names: list[str]
    edges: list[list[int]]
    keypoints: list[list[list[float] | None]]
    confidence: list[list[float]]


class QualityModel(BaseModel):
    frames: int
    fps: float
    valid_pose_ratio: float
    pose_source: str


class DemoMetadataModel(BaseModel):
    source: str
    label: str
    disclaimer: str
    frames: int


class DiagnosisResponse(BaseModel):
    action: str
    total_score: float
    segmented: bool
    phases: list[PhaseModel]
    per_feature_summary: list[FeatureSummaryModel]
    excluded_features: list[ExcludedFeatureModel] = Field(default_factory=list)
    # 检测到 ≥2 次往复时返回：total_score 为逐次均值，
    # phases/timeline 详情对应第 rep_detail_index 次（时间为原视频时刻）。
    reps: list[RepSummaryModel] | None = None
    rep_detail_index: int | None = None
    timeline: TimelineModel | None = None
    skeleton: SkeletonModel | None = None
    llm_advice: str | None = None
    advice_source: str | None = None
    metadata: DemoMetadataModel | None = None
    quality: QualityModel | None = None
