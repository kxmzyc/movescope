"""集中式运行配置。

所有环境变量在 Settings.from_env() 中统一读取，调用方不再散落
os.getenv。data_dir 默认保持相对路径 "data"，与既有的
「以仓库根目录为工作目录」约定一致，也让测试可以用 chdir 隔离。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)
LOCAL_CORS_ORIGIN_REGEX = r"^http://(localhost|127\.0\.0\.1):\d+$"


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    max_upload_mb: int = 100
    cors_origins: tuple[str, ...] = field(default_factory=lambda: DEFAULT_CORS_ORIGINS)
    cors_origin_regex: str = LOCAL_CORS_ORIGIN_REGEX
    assess_timeout_sec: float = 300.0
    upload_chunk_bytes: int = 1024 * 1024

    @property
    def templates_dir(self) -> Path:
        return self.data_dir / "templates"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> Settings:
        data_dir = Path(os.getenv("MOVESCOPE_DATA_DIR", "data"))
        max_upload_mb = max(1, int(os.getenv("MOVESCOPE_MAX_UPLOAD_MB", "100")))
        raw_origins = os.getenv("MOVESCOPE_CORS_ORIGINS", "")
        origins = tuple(item.strip() for item in raw_origins.split(",") if item.strip()) or DEFAULT_CORS_ORIGINS
        return cls(
            data_dir=data_dir,
            max_upload_mb=max_upload_mb,
            cors_origins=origins,
        )
