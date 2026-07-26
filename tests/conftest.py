"""Pytest 共享配置。

仓库根目录的导入路径由 pyproject.toml 中的
[tool.pytest.ini_options].pythonpath 提供，`movescope`/`api` 包本身
通过可编辑安装（`pip install -e .`）可直接导入，因此这里不再需要手写
sys.path 注入。
"""

from __future__ import annotations
