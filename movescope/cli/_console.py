"""CLI 共享的控制台工具。"""

from __future__ import annotations

import sys


def configure_utf8_stdio() -> None:
    """Windows 控制台默认代码页可能不是 UTF-8；统一重配以正确输出中文。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
