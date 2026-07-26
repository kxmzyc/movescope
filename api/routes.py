"""API 端点。"""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from api.schemas import ActionsResponse, DiagnosisResponse, HealthResponse
from api.services import AssessmentService
from api.settings import get_settings
from movescope import __version__
from movescope.config import Settings
from movescope.constants import VIDEO_EXTENSIONS
from movescope.demo import generate_synthetic_demo
from movescope.template import ActionTemplate

logger = logging.getLogger(__name__)

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
    paths = sorted(template_dir.glob("*.npz")) if template_dir.exists() else []
    templates = []
    for path in paths:
        try:
            template = ActionTemplate.load(path.stem, path)
        except Exception:
            # 无法解析的模板文件仍出现在 actions 名单中，只是没有元数据。
            logger.warning("无法读取模板元数据：%s", path, exc_info=True)
            continue
        seq = template.representative_seq
        templates.append(
            {
                "action": template.action_name,
                "n_videos": template.n_videos,
                "feature_dim": int(seq.shape[1]) if seq is not None and seq.ndim == 2 else 0,
                "frames": len(seq) if seq is not None else 0,
            }
        )
    return {"actions": [path.stem for path in paths], "templates": templates}


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
    request: Request,
    settings: SettingsDep,
    video: UploadFile = File(..., description="待评估的视频文件"),
    action: str = Form("squat", description="动作标识，例如 squat"),
) -> Any:
    action = _validate_action(action)
    suffix = Path(video.filename or "upload.mp4").suffix.lower() or ".mp4"
    if suffix not in VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise HTTPException(status_code=415, detail=f"不支持该视频格式，请使用以下格式之一：{supported}")

    semaphore: asyncio.Semaphore = request.app.state.assess_semaphore
    if semaphore.locked():
        raise HTTPException(status_code=503, detail="评估服务繁忙，请稍后重试。")

    service = AssessmentService(settings)
    template = service.load_template(action)

    tmp_path = Path(tempfile.gettempdir()) / f"movescope_upload_{uuid.uuid4().hex}{suffix}"
    await semaphore.acquire()
    task: asyncio.Task[Any] | None = None
    try:
        await service.save_upload(video, tmp_path)
        # 评估线程无法被取消：超时或客户端断连后它仍会继续占用 CPU 与文件
        # 句柄。槽位与临时文件因此挂在任务完成回调上释放，而不是请求返回
        # 时——否则 504 后放行的新请求会与遗留线程叠加运行，并发上限形同
        # 虚设，Windows 上临时文件也会因句柄仍被占用而永久残留。
        task = asyncio.ensure_future(run_in_threadpool(service.assess_file, tmp_path, template))
        task.add_done_callback(lambda done: _release_after_assess(done, semaphore, tmp_path))
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=settings.assess_timeout_sec)
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="评估超时，请缩短视频后重试。") from exc
    finally:
        if task is None:
            semaphore.release()
            _safe_unlink(tmp_path)
        await video.close()


def _release_after_assess(task: asyncio.Task[Any], semaphore: asyncio.Semaphore, tmp_path: Path) -> None:
    if not task.cancelled():
        task.exception()  # 取回异常，避免事件循环记录 "exception was never retrieved"
    semaphore.release()
    _safe_unlink(tmp_path)


def _safe_unlink(path: Path) -> None:
    """删除临时文件；容忍 Windows 上文件句柄仍被占用的情况。

    删除挂在评估任务完成回调上，正常情况下线程已释放 cv2.VideoCapture
    的句柄；但杀毒软件等外部进程仍可能短暂占用文件，此时 unlink 抛
    PermissionError（missing_ok 只抑制 FileNotFoundError）。这里只记录
    告警，不让清理失败影响响应。
    """
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("临时视频文件暂时无法删除（可能仍被其他进程占用）：%s", path)


def _validate_action(action: str) -> str:
    cleaned = action.strip()
    if not ACTION_PATTERN.fullmatch(cleaned):
        raise HTTPException(
            status_code=422,
            detail="动作标识只能包含英文字母、数字、下划线或连字符。",
        )
    return cleaned
