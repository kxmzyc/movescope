"""跨模块共享的常量（唯一真源）。"""

from __future__ import annotations

# 全项目统一的视频扩展名白名单：模板构建、实验目录扫描、
# API 上传校验和下载脚本都从这里读取。
VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".avi", ".webm", ".mkv"})
