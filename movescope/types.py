"""跨模块共享的结构化结果类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Alignment:
    """一次 DTW 对齐的完整结果。

    query_segments 是待测序列上检测到的阶段边界（左闭右开）。
    分段检测失败或未启用时回退为覆盖全序列的单一分段，
    segmented=False 让下游能如实告知界面。
    """

    path: list[tuple[int, int]]
    query_segments: list[tuple[int, int]]
    segmented: bool


@dataclass(frozen=True)
class PoseResult:
    """一次视频姿态提取的完整输出。

    coords_3d 仅在 MotionBERT 提升成功时非空；日常路径使用
    MediaPipe world landmarks 得到的伪三维坐标 coords_3d_pseudo。
    """

    fps: float
    joint_names: list[str]
    coords_2d: np.ndarray  # (T, J, 2)
    confidence: np.ndarray  # (T, J)
    coords_3d_pseudo: np.ndarray  # (T, J, 3)
    coords_3d: np.ndarray | None = None
    skipped_frames: int = 0

    @property
    def n_frames(self) -> int:
        return len(self.coords_2d)

    @property
    def best_coords_3d(self) -> np.ndarray:
        """优先返回真三维坐标，否则回退到伪三维坐标。"""
        return self.coords_3d if self.coords_3d is not None else self.coords_3d_pseudo

    @property
    def source(self) -> str:
        return "motionbert" if self.coords_3d is not None else "mediapipe_world"

    @property
    def valid_ratio(self) -> float:
        """检测到有效姿态的帧占比；无帧时为 0。"""
        if self.n_frames == 0:
            return 0.0
        return 1.0 - self.skipped_frames / self.n_frames

    def to_npz_dict(self) -> dict[str, Any]:
        """np.savez_compressed 可直接使用的键值映射。"""
        return {
            "fps": self.fps,
            "n_frames": self.n_frames,
            "joint_names": np.asarray(self.joint_names),
            "coords_2d": self.coords_2d,
            "confidence": self.confidence,
            "coords_3d_pseudo": self.coords_3d_pseudo,
        }
