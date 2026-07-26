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


class ActionsResponse(BaseModel):
    actions: list[str]


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
    llm_advice: str | None = None
    metadata: DemoMetadataModel | None = None
    quality: QualityModel | None = None
