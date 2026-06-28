"""FastAPI backend for MoveScope assessment."""

from __future__ import annotations

import asyncio
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from movescope import __version__
from movescope.alignment import WeightedSegmentedDTWAligner
from movescope.assessment import AssessmentEngine
from movescope.features import FeatureExtractor
from movescope.llm_advisor import LLMAdvisor
from movescope.pose_extractor import PoseExtractor
from movescope.template import ActionTemplate


app = FastAPI(title="MoveScope API", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/actions")
def actions() -> dict[str, list[str]]:
    template_dir = Path("data/templates")
    names = sorted(path.stem for path in template_dir.glob("*.npz")) if template_dir.exists() else []
    return {"actions": names}


@app.post("/assess")
async def assess(
    video: UploadFile = File(...),
    action: str = Form("squat"),
) -> dict[str, Any]:
    try:
        template = ActionTemplate.load(action)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Template for action '{action}' not found. Please run scripts/build_template.py first.",
        ) from exc

    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    tmp_path = Path(tempfile.gettempdir()) / f"movescope_upload_{uuid.uuid4().hex}{suffix}"
    try:
        tmp_path.write_bytes(await video.read())
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
        tmp_path.unlink(missing_ok=True)


def _assess_file(video_path: Path, template: ActionTemplate) -> dict[str, Any]:
    pose = PoseExtractor().extract(str(video_path))
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
    return result
