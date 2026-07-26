"""Pytest 共享配置。

仓库根目录的导入路径由 pyproject.toml 中的
[tool.pytest.ini_options].pythonpath 提供，`movescope`/`api` 包本身
通过可编辑安装（`pip install -e .`）可直接导入，因此这里不再需要手写
sys.path 注入。

测试对开发者 shell 环境的免疫分两层：

1. 模块导入期（下方的 os.environ 清理）：pytest 收集 tests/test_api.py 时
   api.main 顶层的 create_app() 立即执行，CORS 中间件与并发信号量在那一刻
   就按环境变量固化——夹具运行得太晚。conftest 先于测试模块导入，因此在
   这里先清掉相关变量。
2. 每个测试前后（_isolated_settings 夹具）：get_settings 的 lru_cache 会让
   首个测试为整个进程固化配置，这里统一清空环境变量、清缓存与依赖覆盖。
"""

from __future__ import annotations

import os

import pytest

_MOVESCOPE_ENV_VARS = (
    "MOVESCOPE_DATA_DIR",
    "MOVESCOPE_MAX_UPLOAD_MB",
    "MOVESCOPE_CORS_ORIGINS",
    "MOVESCOPE_CORS_ORIGIN_REGEX",
    "MOVESCOPE_ASSESS_TIMEOUT_SEC",
    "MOVESCOPE_MAX_CONCURRENT_ASSESS",
    "MOVESCOPE_ADVICE_PROVIDER",
    "MOVESCOPE_OPENAI_MODEL",
    "MOVESCOPE_OPENAI_TIMEOUT_SEC",
    "OPENAI_API_KEY",
)

for _name in _MOVESCOPE_ENV_VARS:
    os.environ.pop(_name, None)


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    from api.main import app
    from api.settings import get_settings

    for name in _MOVESCOPE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    app.dependency_overrides.clear()
