"""跨模块共享的常量（唯一真源）。"""

from __future__ import annotations

# 全项目统一的视频扩展名白名单：模板构建、实验目录扫描、
# API 上传校验和下载脚本都从这里读取。
VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".avi", ".webm", ".mkv"})

# 17 关节骨架的连接拓扑（关节名与 features.JOINT_NAMES 一致）。
# Gradio 服务端叠加与 Web 端 canvas 叠加共用。
SKELETON_EDGES: tuple[tuple[str, str], ...] = (
    ("pelvis", "left_hip"),
    ("pelvis", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("neck", "head"),
)
