"""API 业务服务：上传保存与视频评估。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

from movescope.advice import OpenAIAdvisor
from movescope.alignment import WeightedSegmentedDTWAligner
from movescope.assessment import AssessmentEngine
from movescope.config import Settings
from movescope.errors import InvalidInputError, PoseQualityError, TemplateNotFoundError
from movescope.features import FeatureExtractor
from movescope.pose_extractor import PoseExtractor
from movescope.template import ActionTemplate


@dataclass
class AssessmentService:
    settings: Settings

    def load_template(self, action: str) -> ActionTemplate:
        try:
            return ActionTemplate.load(action, self.settings.templates_dir / f"{action}.npz")
        except TemplateNotFoundError as exc:
            raise TemplateNotFoundError(
                f"未找到动作“{action}”的模板，请先运行 scripts/build_template.py。"
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
        )
        try:
            result = engine.assess_pose(pose)
        except InvalidInputError:
            raise
        except ValueError as exc:
            # 引擎内部的输入校验仍抛裸 ValueError；在 API 边界翻译成领域异常。
            raise InvalidInputError(str(exc)) from exc
        result["llm_advice"] = OpenAIAdvisor().generate_advice(result)
        result["quality"] = {
            "frames": pose.n_frames,
            "fps": round(pose.fps, 3),
            "valid_pose_ratio": round(pose.valid_ratio, 3),
            "pose_source": pose.source,
        }
        return result
