"""API 层的配置依赖。

get_settings 用 lru_cache 保证每个进程只读一次环境变量；
测试通过 app.dependency_overrides[get_settings] 注入自定义配置。
"""

from __future__ import annotations

from functools import lru_cache

from movescope.config import Settings


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
