"""API 业务服务：上传保存与视频评估。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException, UploadFile

from movescope.advice import generate_advice
from movescope.alignment import WeightedSegmentedDTWAligner
from movescope.assessment import AssessmentEngine
from movescope.config import Settings
from movescope.constants import SKELETON_EDGES
from movescope.errors import InvalidInputError, PoseQualityError, TemplateNotFoundError
from movescope.features import CORE_FEATURE_INDICES, FeatureExtractor
from movescope.pose_extractor import PoseExtractor
from movescope.template import ActionTemplate
from movescope.types import PoseResult

# 骨架关键点通道的降采样上限：以 30 fps 计约 30 秒视频不降采样，
# 更长的视频按整数步长抽帧，控制响应体积。
SKELETON_MAX_FRAMES = 900


@dataclass
class AssessmentService:
    settings: Settings

    def load_template(self, action: str) -> ActionTemplate:
        try:
            return ActionTemplate.load(action, self.settings.templates_dir / f"{action}.npz")
        except TemplateNotFoundError as exc:
            raise TemplateNotFoundError(
                f"未找到动作“{action}”的模板，请先运行 movescope-build-template。"
            ) from exc

    async def save_upload(self, video: UploadFile, output_path: Path) -> None:
        max_bytes = self.settings.max_upload_bytes
        total = 0
        with output_path.open("wb") as handle:
            while chunk := await video.read(self.settings.upload_chunk_bytes):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"视频超过 {max_bytes // 1024 // 1024} MB 上传上限。",
                    )
                handle.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="上传的视频为空。")

    def assess_file(self, video_path: Path, template: ActionTemplate) -> dict[str, Any]:
        pose = PoseExtractor().extract(str(video_path))
        if pose.n_frames <= 0:
            raise PoseQualityError("无法从视频中解码出任何画面。")
        if pose.valid_ratio < 0.5:
            raise PoseQualityError("超过一半的视频帧未检测到有效人体姿态。")

        engine = AssessmentEngine(
            template=template,
            aligner=WeightedSegmentedDTWAligner(),
            feature_extractor=FeatureExtractor(),
            required_features=CORE_FEATURE_INDICES,
        )
        try:
            result = engine.assess_pose(pose)
        except InvalidInputError:
            raise
        except ValueError as exc:
            # 引擎内部的输入校验仍抛裸 ValueError；在 API 边界翻译成领域异常。
            raise InvalidInputError(str(exc)) from exc
        advice, advice_source = generate_advice(
            result,
            provider=self.settings.advice_provider,
            model=self.settings.openai_model,
            timeout_sec=self.settings.openai_timeout_sec,
        )
        result["llm_advice"] = advice
        result["advice_source"] = advice_source
        result["quality"] = {
            "frames": pose.n_frames,
            "fps": round(pose.fps, 3),
            "valid_pose_ratio": round(pose.valid_ratio, 3),
            "pose_source": pose.source,
        }
        result["skeleton"] = build_skeleton_payload(pose)
        return result


def build_skeleton_payload(pose: PoseResult, max_frames: int = SKELETON_MAX_FRAMES) -> dict[str, Any] | None:
    """把姿态提取的 2D 关键点整理成前端 canvas 叠加可直接使用的结构。

    坐标为 MediaPipe 归一化图像坐标（x/y ∈ [0,1]），非有限值输出 null，
    由前端跳过绘制；置信度原样下发，前端可按需淡化低置信关节。
    """
    coords = np.asarray(pose.coords_2d, dtype=float)
    if coords.ndim != 3 or len(coords) == 0:
        return None
    fps = float(pose.fps) if math.isfinite(pose.fps) and pose.fps > 0 else 30.0
    stride = max(1, math.ceil(len(coords) / max(1, max_frames)))
    sample = np.arange(0, len(coords), stride)
    joint_index = {name: idx for idx, name in enumerate(pose.joint_names)}
    edges = [
        [joint_index[start], joint_index[end]]
        for start, end in SKELETON_EDGES
        if start in joint_index and end in joint_index
    ]
    confidence = np.asarray(pose.confidence, dtype=float)

    keypoints: list[list[list[float] | None]] = []
    for frame_idx in sample:
        frame: list[list[float] | None] = []
        for joint_idx in range(coords.shape[1]):
            x, y = coords[frame_idx, joint_idx]
            frame.append([round(float(x), 4), round(float(y), 4)] if np.isfinite(x) and np.isfinite(y) else None)
        keypoints.append(frame)

    return {
        "fps": round(fps, 3),
        "frame_count": len(coords),
        "frame_stride": int(stride),
        "time_sec": [round(float(idx / fps), 3) for idx in sample],
        "joint_names": list(pose.joint_names),
        "edges": edges,
        "keypoints": keypoints,
        "confidence": [[round(float(v), 3) for v in confidence[frame_idx]] for frame_idx in sample],
    }
