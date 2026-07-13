"""MoveScope 动作评估 FastAPI 后端。"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from movescope import __version__
from movescope.alignment import WeightedSegmentedDTWAligner
from movescope.assessment import AssessmentEngine
from movescope.demo import generate_synthetic_demo
from movescope.features import FeatureExtractor
from movescope.llm_advisor import LLMAdvisor
from movescope.pose_extractor import PoseExtractor
from movescope.template import ActionTemplate


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
ACTION_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_UPLOAD_BYTES = max(1, int(os.getenv("MOVESCOPE_MAX_UPLOAD_MB", "100"))) * 1024 * 1024
DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
LOCAL_CORS_ORIGIN_REGEX = r"^http://(localhost|127\.0\.0\.1):\d+$"


def _cors_origins() -> list[str]:
    raw = os.getenv("MOVESCOPE_CORS_ORIGINS", "")
    return [item.strip() for item in raw.split(",") if item.strip()] or DEFAULT_CORS_ORIGINS


app = FastAPI(
    title="MoveScope API",
    version=__version__,
    description="可解释的单目深蹲动作质量评估服务。",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=LOCAL_CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", summary="检查服务状态")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/actions", summary="获取可用动作模板")
def actions() -> dict[str, list[str]]:
    template_dir = Path("data/templates")
    names = sorted(path.stem for path in template_dir.glob("*.npz")) if template_dir.exists() else []
    return {"actions": names}


@app.get("/demo", summary="运行确定性合成验证")
def demo() -> dict[str, Any]:
    return generate_synthetic_demo()


@app.post("/assess", summary="评估上传的视频")
async def assess(
    video: UploadFile = File(..., description="待评估的视频文件"),
    action: str = Form("squat", description="动作标识，例如 squat"),
) -> dict[str, Any]:
    action = _validate_action(action)
    suffix = Path(video.filename or "upload.mp4").suffix.lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS))
        raise HTTPException(status_code=415, detail=f"不支持该视频格式，请使用以下格式之一：{supported}")

    try:
        template = ActionTemplate.load(action)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"未找到动作“{action}”的模板，请先运行 scripts/build_template.py。",
        ) from exc

    tmp_path = Path(tempfile.gettempdir()) / f"movescope_upload_{uuid.uuid4().hex}{suffix}"
    try:
        await _save_upload(video, tmp_path)
        try:
            return await asyncio.wait_for(
                run_in_threadpool(_assess_file, tmp_path, template),
                timeout=300.0,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="评估超时，请缩短视频后重试。") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await video.close()
        tmp_path.unlink(missing_ok=True)


def _validate_action(action: str) -> str:
    cleaned = action.strip()
    if not ACTION_PATTERN.fullmatch(cleaned):
        raise HTTPException(
            status_code=422,
            detail="动作标识只能包含英文字母、数字、下划线或连字符。",
        )
    return cleaned


async def _save_upload(video: UploadFile, output_path: Path) -> None:
    total = 0
    with output_path.open("wb") as handle:
        while chunk := await video.read(UPLOAD_CHUNK_BYTES):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"视频超过 {MAX_UPLOAD_BYTES // 1024 // 1024} MB 上传上限。",
                )
            handle.write(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="上传的视频为空。")


def _assess_file(video_path: Path, template: ActionTemplate) -> dict[str, Any]:
    pose = PoseExtractor().extract(str(video_path))
    n_frames = int(pose.get("n_frames", 0))
    if n_frames <= 0:
        raise ValueError("无法从视频中解码出任何画面。")
    valid_pose_ratio = 1.0 - (float(pose.get("skipped_frames", 0)) / n_frames)
    if valid_pose_ratio < 0.5:
        raise ValueError("超过一半的视频帧未检测到有效人体姿态。")

    coords_3d = pose.get("coords_3d")
    if coords_3d is None:
        coords_3d = pose["coords_3d_pseudo"]

    engine = AssessmentEngine(
        template=template,
        aligner=WeightedSegmentedDTWAligner(),
        feature_extractor=FeatureExtractor(),
        fps=float(pose.get("fps", 30.0)),
    )
    result = engine.assess(coords_3d)
    result["llm_advice"] = LLMAdvisor().generate_advice(result)
    result["quality"] = {
        "frames": n_frames,
        "fps": round(float(pose.get("fps", 30.0)), 3),
        "valid_pose_ratio": round(valid_pose_ratio, 3),
        "pose_source": "motionbert" if pose.get("coords_3d") is not None else "mediapipe_world",
    }
    return result
