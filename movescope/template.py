"""Expert-template statistics for MoveScope."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_K = 1.5
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}


@dataclass
class ActionTemplate:
    action_name: str
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    tolerance: np.ndarray | None = None
    representative_seq: np.ndarray | None = None
    n_videos: int = 0

    def build(self, expert_dir: str | Path, pose_extractor, feature_extractor, k: float = DEFAULT_K) -> None:
        expert_path = Path(expert_dir)
        if not expert_path.exists():
            raise FileNotFoundError(f"Expert directory not found: {expert_path}")

        video_files = sorted(path for path in expert_path.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)
        if not video_files:
            raise ValueError(f"No expert videos found in {expert_path}")

        feature_sequences = []
        for video_path in video_files:
            pose = pose_extractor.extract(str(video_path))
            coords_3d = pose.get("coords_3d")
            if coords_3d is None:
                coords_3d = pose["coords_3d_pseudo"]
            feature_sequences.append(feature_extractor.extract(coords_3d, normalize=False))

        self.build_from_features(feature_sequences, k=k)

    def build_from_features(self, feature_sequences: list[np.ndarray], k: float = DEFAULT_K) -> None:
        if not feature_sequences:
            raise ValueError("feature_sequences cannot be empty")

        vectors = np.vstack([np.asarray(seq, dtype=float).mean(axis=0) for seq in feature_sequences])
        self.mean = vectors.mean(axis=0)
        self.std = vectors.std(axis=0)
        self.tolerance = np.maximum(self.std * k, 1e-6)
        distances = np.linalg.norm(vectors - self.mean[None, :], axis=1)
        self.representative_seq = np.asarray(feature_sequences[int(np.argmin(distances))], dtype=float)
        self.n_videos = len(feature_sequences)

    def save(self, output_path: str | Path | None = None) -> Path:
        if self.mean is None or self.std is None or self.tolerance is None or self.representative_seq is None:
            raise ValueError("template has not been built")
        path = Path(output_path) if output_path else Path("data/templates") / f"{self.action_name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            action_name=self.action_name,
            mean=self.mean,
            std=self.std,
            tolerance=self.tolerance,
            representative_seq=self.representative_seq,
            n_videos=self.n_videos,
        )
        return path

    @classmethod
    def load(cls, action_name: str, path: str | Path | None = None) -> "ActionTemplate":
        template_path = Path(path) if path else Path("data/templates") / f"{action_name}.npz"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        data = np.load(template_path, allow_pickle=False)
        return cls(
            action_name=str(data["action_name"]),
            mean=data["mean"],
            std=data["std"],
            tolerance=data["tolerance"],
            representative_seq=data["representative_seq"],
            n_videos=int(data["n_videos"]),
        )
