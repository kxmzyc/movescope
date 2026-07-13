"""面向视角变化的关节角特征提取。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "head",
    "neck",
    "left_eye",
    "right_eye",
]

JOINT_DISPLAY_NAMES = {
    "pelvis": "骨盆",
    "left_hip": "左髋",
    "right_hip": "右髋",
    "left_knee": "左膝",
    "right_knee": "右膝",
    "left_ankle": "左踝",
    "right_ankle": "右踝",
    "left_shoulder": "左肩",
    "right_shoulder": "右肩",
    "left_elbow": "左肘",
    "right_elbow": "右肘",
    "left_wrist": "左腕",
    "right_wrist": "右腕",
    "head": "头部",
    "neck": "颈部",
    "left_eye": "左眼",
    "right_eye": "右眼",
}

JOINT_TRIPLETS = [
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
    ("left_shoulder", "left_hip", "left_knee"),
    ("right_shoulder", "right_hip", "right_knee"),
    ("left_knee", "left_hip", "right_hip"),
    ("right_knee", "right_hip", "left_hip"),
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("neck", "left_shoulder", "left_elbow"),
    ("neck", "right_shoulder", "right_elbow"),
    ("pelvis", "left_hip", "left_knee"),
    ("pelvis", "right_hip", "right_knee"),
]


@dataclass
class FeatureExtractor:
    """从 17 关节三维骨架中提取逐帧关节角。"""

    joint_names: list[str] | None = None
    joint_triplets: list[tuple[str, str, str]] | None = None

    def __post_init__(self) -> None:
        self.joint_names = self.joint_names or JOINT_NAMES
        self.joint_triplets = self.joint_triplets or JOINT_TRIPLETS
        self._joint_index = {name: idx for idx, name in enumerate(self.joint_names)}

    def compute_angles(self, coords_3d: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords_3d, dtype=float)
        if coords.ndim != 3 or coords.shape[1:] != (len(self.joint_names), 3):
            raise ValueError(f"coords_3d 的形状必须为 (T, {len(self.joint_names)}, 3)")

        angles = np.empty((coords.shape[0], len(self.joint_triplets)), dtype=float)
        for feature_idx, (parent, joint, child) in enumerate(self.joint_triplets):
            a = coords[:, self._joint_index[parent], :]
            b = coords[:, self._joint_index[joint], :]
            c = coords[:, self._joint_index[child], :]
            v1 = a - b
            v2 = c - b
            denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
            valid = np.isfinite(denom) & (denom > 1e-8)
            cos_angle = np.full(coords.shape[0], np.nan, dtype=float)
            np.divide(
                np.sum(v1 * v2, axis=1),
                denom,
                out=cos_angle,
                where=valid,
            )
            angles[:, feature_idx] = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angles

    def normalize(self, angle_seq: np.ndarray) -> np.ndarray:
        angles = np.asarray(angle_seq, dtype=float)
        mean = angles.mean(axis=0, keepdims=True)
        std = angles.std(axis=0, keepdims=True)
        std = np.where(std == 0.0, 1.0, std)
        return (angles - mean) / std

    def extract(self, coords_3d: np.ndarray, normalize: bool = True) -> np.ndarray:
        angles = self.compute_angles(coords_3d)
        return self.normalize(angles) if normalize else angles
