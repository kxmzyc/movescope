"""FastAPI backend for MoveScope assessment."""

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
    description="Interpretable monocular squat assessment prototype.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=LOCAL_CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/actions")
def actions() -> dict[str, list[str]]:
    template_dir = Path("data/templates")
    names = sorted(path.stem for path in template_dir.glob("*.npz")) if template_dir.exists() else []
    return {"actions": names}


@app.get("/demo")
def demo() -> dict[str, Any]:
    return generate_synthetic_demo()


@app.post("/assess")
async def assess(
    video: UploadFile = File(...),
    action: str = Form("squat"),
) -> dict[str, Any]:
    action = _validate_action(action)
    suffix = Path(video.filename or "upload.mp4").suffix.lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS))
        raise HTTPException(status_code=415, detail=f"Unsupported video type. Use one of: {supported}")

    try:
        template = ActionTemplate.load(action)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Template for action '{action}' not found. Please run scripts/build_template.py first.",
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
            raise HTTPException(status_code=504, detail="Assessment timed out.") from exc
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
            detail="Action must contain only letters, numbers, underscores, or hyphens.",
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
                    detail=f"Video exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB upload limit.",
                )
            handle.write(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="Uploaded video is empty.")


def _assess_file(video_path: Path, template: ActionTemplate) -> dict[str, Any]:
    pose = PoseExtractor().extract(str(video_path))
    n_frames = int(pose.get("n_frames", 0))
    if n_frames <= 0:
        raise ValueError("No video frames could be decoded.")
    valid_pose_ratio = 1.0 - (float(pose.get("skipped_frames", 0)) / n_frames)
    if valid_pose_ratio < 0.5:
        raise ValueError("Pose detection failed on more than half of the video frames.")

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
