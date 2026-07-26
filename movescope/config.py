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

# 建议来源："off" 不生成建议；"rule" 本地规则；"openai" 远程模型
# （失败时回退本地规则）。默认本地规则：诊断数据是否外发必须显式选择。
ADVICE_PROVIDERS = ("off", "rule", "openai")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    max_upload_mb: int = 100
    cors_origins: tuple[str, ...] = field(default_factory=lambda: DEFAULT_CORS_ORIGINS)
    cors_origin_regex: str = LOCAL_CORS_ORIGIN_REGEX
    assess_timeout_sec: float = 300.0
    upload_chunk_bytes: int = 1024 * 1024
    max_concurrent_assess: int = 2
    advice_provider: str = "rule"
    openai_model: str = "gpt-4o"
    openai_timeout_sec: float = 20.0

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
        cors_origin_regex = os.getenv("MOVESCOPE_CORS_ORIGIN_REGEX", LOCAL_CORS_ORIGIN_REGEX)
        assess_timeout_sec = max(1.0, float(os.getenv("MOVESCOPE_ASSESS_TIMEOUT_SEC", "300")))
        max_concurrent_assess = max(1, int(os.getenv("MOVESCOPE_MAX_CONCURRENT_ASSESS", "2")))
        advice_provider = os.getenv("MOVESCOPE_ADVICE_PROVIDER", "rule").strip().lower()
        if advice_provider not in ADVICE_PROVIDERS:
            supported = "、".join(ADVICE_PROVIDERS)
            raise ValueError(f"MOVESCOPE_ADVICE_PROVIDER 只支持 {supported}，当前为：{advice_provider}")
        return cls(
            data_dir=data_dir,
            max_upload_mb=max_upload_mb,
            cors_origins=origins,
            cors_origin_regex=cors_origin_regex,
            assess_timeout_sec=assess_timeout_sec,
            max_concurrent_assess=max_concurrent_assess,
            advice_provider=advice_provider,
            openai_model=os.getenv("MOVESCOPE_OPENAI_MODEL", "gpt-4o"),
            openai_timeout_sec=max(1.0, float(os.getenv("MOVESCOPE_OPENAI_TIMEOUT_SEC", "20"))),
        )
