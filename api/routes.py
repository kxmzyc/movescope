"""API 端点。"""

from __future__ import annotations

import asyncio
import re
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from api.schemas import ActionsResponse, DiagnosisResponse, HealthResponse
from api.services import AssessmentService
from api.settings import get_settings
from movescope import __version__
from movescope.config import Settings
from movescope.constants import VIDEO_EXTENSIONS
from movescope.demo import generate_synthetic_demo

router = APIRouter()

ACTION_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.get("/health", response_model=HealthResponse, summary="检查服务状态")
def health(settings: SettingsDep) -> Any:
    return {
        "status": "ok",
        "version": __version__,
        "max_upload_bytes": settings.max_upload_bytes,
        "allowed_extensions": sorted(VIDEO_EXTENSIONS),
    }


@router.get("/actions", response_model=ActionsResponse, summary="获取可用动作模板")
def actions(settings: SettingsDep) -> Any:
    template_dir = settings.templates_dir
    names = sorted(path.stem for path in template_dir.glob("*.npz")) if template_dir.exists() else []
    return {"actions": names}


@router.get(
    "/demo",
    response_model=DiagnosisResponse,
    response_model_exclude_none=True,
    summary="运行确定性合成验证",
)
def demo() -> Any:
    return generate_synthetic_demo()


@router.post(
    "/assess",
    response_model=DiagnosisResponse,
    response_model_exclude_none=True,
    summary="评估上传的视频",
)
async def assess(
    settings: SettingsDep,
    video: UploadFile = File(..., description="待评估的视频文件"),
    action: str = Form("squat", description="动作标识，例如 squat"),
) -> Any:
    action = _validate_action(action)
    suffix = Path(video.filename or "upload.mp4").suffix.lower() or ".mp4"
    if suffix not in VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise HTTPException(status_code=415, detail=f"不支持该视频格式，请使用以下格式之一：{supported}")

    service = AssessmentService(settings)
    template = service.load_template(action)

    tmp_path = Path(tempfile.gettempdir()) / f"movescope_upload_{uuid.uuid4().hex}{suffix}"
    try:
        await service.save_upload(video, tmp_path)
        try:
            return await asyncio.wait_for(
                run_in_threadpool(service.assess_file, tmp_path, template),
                timeout=settings.assess_timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="评估超时，请缩短视频后重试。") from exc
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
