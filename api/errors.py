"""领域异常到 HTTP 状态码的集中映射。

替代旧版路由里粗粒度的 except ValueError → 400（它会误吞任何无关的
ValueError）。领域异常在核心层抛出，这里统一翻译成 HTTP 语义。
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from movescope.errors import InvalidInputError, PoseQualityError, TemplateNotFoundError


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(TemplateNotFoundError)
    async def template_not_found(_request: Request, exc: TemplateNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(PoseQualityError)
    async def pose_quality(_request: Request, exc: PoseQualityError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(InvalidInputError)
    async def invalid_input(_request: Request, exc: InvalidInputError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
