"""MoveScope 专家动作模板统计。

模板统计的口径（v2，逐帧）：先按「每视频时间平均特征向量」选出离均值
最近的代表序列，再把每条专家序列用无权重 DTW 对齐到代表序列，按参考
帧聚合得到逐帧均值曲线（reference_seq）与逐帧标准差；容差带
tolerance_band 形状为 (T_ref, D)，随动作阶段变化——蹲底与站立位不再
共用同一个容差。

旧格式（v1，时间平均）模板仍可加载：缺少逐帧字段时，评估链路通过
reference_curve / tolerance_curve 属性回退为代表序列 + 全局容差广播。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from movescope.alignment.dtw import dtw_align
from movescope.config import Settings
from movescope.constants import VIDEO_EXTENSIONS
from movescope.errors import TemplateNotFoundError

DEFAULT_K = 1.5
MIN_TOLERANCE_DEG = 5.0

TEMPLATE_FORMAT_VERSION = 2


@dataclass
class ActionTemplate:
    action_name: str
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    tolerance: np.ndarray | None = None
    representative_seq: np.ndarray | None = None
    n_videos: int = 0
    reference_seq: np.ndarray | None = None
    tolerance_band: np.ndarray | None = None

    def build(self, expert_dir: str | Path, pose_extractor, feature_extractor, k: float = DEFAULT_K) -> None:
        expert_path = Path(expert_dir)
        if not expert_path.exists():
            raise FileNotFoundError(f"未找到专家视频目录：{expert_path}")

        video_files = sorted(path for path in expert_path.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)
        if not video_files:
            raise ValueError(f"目录中没有可用的专家视频：{expert_path}")

        feature_sequences = []
        for video_path in video_files:
            pose = pose_extractor.extract(str(video_path))
            feature_sequences.append(feature_extractor.extract(pose.best_coords_3d, normalize=False))

        self.build_from_features(feature_sequences, k=k)

    def build_from_features(
        self,
        feature_sequences: list[np.ndarray],
        k: float = DEFAULT_K,
        min_tolerance_deg: float = MIN_TOLERANCE_DEG,
    ) -> None:
        if not feature_sequences:
            raise ValueError("feature_sequences 不能为空")
        if min_tolerance_deg <= 0 or not np.isfinite(min_tolerance_deg):
            raise ValueError("min_tolerance_deg 必须是正有限值")

        sequences = [np.asarray(seq, dtype=float) for seq in feature_sequences]
        feature_dim = sequences[0].shape[1] if sequences[0].ndim == 2 else None
        for sequence in sequences:
            if sequence.ndim != 2 or len(sequence) == 0:
                raise ValueError("每个特征序列都必须是非空二维数组")
            if sequence.shape[1] != feature_dim:
                raise ValueError("所有特征序列的特征维度必须一致")
            if not np.isfinite(sequence).all():
                raise ValueError("特征序列只能包含有限值")

        vectors = np.vstack([sequence.mean(axis=0) for sequence in sequences])
        self.mean = vectors.mean(axis=0)
        distances = np.linalg.norm(vectors - self.mean[None, :], axis=1)
        representative = sequences[int(np.argmin(distances))]
        self.representative_seq = representative

        # 逐帧统计：把每条专家序列对齐到代表序列（无权重 DTW——此时还没有
        # std 可算权重），按参考帧聚合成 (T_ref, D) 曲线后跨视频取 mean/std。
        # 这样 std 度量的是「同一动作阶段上专家之间的离差」，而不是
        # 个体差异与阶段差异的混合。
        aligned = np.stack([_align_curve_to_reference(sequence, representative) for sequence in sequences])
        self.reference_seq = aligned.mean(axis=0)
        frame_std = aligned.std(axis=0)
        self.std = frame_std.mean(axis=0)
        self.tolerance = np.maximum(self.std * k, min_tolerance_deg)
        self.tolerance_band = np.maximum(frame_std * k, min_tolerance_deg)
        self.n_videos = len(sequences)

    @property
    def reference_curve(self) -> np.ndarray:
        """评估使用的参考曲线：逐帧均值曲线，旧格式回退为代表序列。"""
        if self.reference_seq is not None:
            return np.asarray(self.reference_seq, dtype=float)
        if self.representative_seq is None:
            raise ValueError("动作模板尚未构建")
        return np.asarray(self.representative_seq, dtype=float)

    @property
    def tolerance_curve(self) -> np.ndarray:
        """(T_ref, D) 逐帧容差；旧格式模板把全局容差广播到全部参考帧。"""
        if self.tolerance_band is not None:
            return np.asarray(self.tolerance_band, dtype=float)
        if self.tolerance is None:
            raise ValueError("动作模板尚未构建")
        reference = self.reference_curve
        return np.tile(np.asarray(self.tolerance, dtype=float)[None, :], (len(reference), 1))

    def save(self, output_path: str | Path | None = None) -> Path:
        if self.mean is None or self.std is None or self.tolerance is None or self.representative_seq is None:
            raise ValueError("动作模板尚未构建")
        path = Path(output_path) if output_path else Settings.from_env().templates_dir / f"{self.action_name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            action_name=self.action_name,
            mean=self.mean,
            std=self.std,
            tolerance=self.tolerance,
            representative_seq=self.representative_seq,
            n_videos=self.n_videos,
            format_version=TEMPLATE_FORMAT_VERSION,
            reference_seq=self.reference_curve,
            tolerance_band=self.tolerance_curve,
        )
        return path

    @classmethod
    def load(cls, action_name: str, path: str | Path | None = None) -> ActionTemplate:
        template_path = Path(path) if path else Settings.from_env().templates_dir / f"{action_name}.npz"
        if not template_path.exists():
            raise TemplateNotFoundError(f"未找到动作模板：{template_path}")
        data = np.load(template_path, allow_pickle=False)
        return cls(
            action_name=str(data["action_name"]),
            mean=data["mean"],
            std=data["std"],
            tolerance=np.maximum(data["tolerance"], MIN_TOLERANCE_DEG),
            representative_seq=data["representative_seq"],
            n_videos=int(data["n_videos"]),
            reference_seq=data["reference_seq"] if "reference_seq" in data.files else None,
            tolerance_band=(
                np.maximum(data["tolerance_band"], MIN_TOLERANCE_DEG) if "tolerance_band" in data.files else None
            ),
        )


def _align_curve_to_reference(sequence: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """把一条专家序列 DTW 对齐到代表序列，按参考帧聚合为 (T_ref, D)。

    DTW 路径在参考轴上单调且覆盖每个参考帧；一对多匹配时取该参考帧
    匹配到的全部专家帧均值。
    """
    path = dtw_align(sequence, reference)
    path_seq = np.fromiter((pair[0] for pair in path), dtype=int, count=len(path))
    path_ref = np.fromiter((pair[1] for pair in path), dtype=int, count=len(path))
    frame_starts = np.unique(path_ref, return_index=True)[1]
    counts = np.diff(np.append(frame_starts, len(path_ref)))
    return np.add.reduceat(sequence[path_seq], frame_starts, axis=0) / counts[:, None]
