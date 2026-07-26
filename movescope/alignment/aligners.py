"""DTW 对齐器：标准版与加权分段版。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from movescope.alignment import segmentation
from movescope.alignment.dtw import AlignmentPath, dtw_align
from movescope.types import Alignment


@dataclass
class DTWAligner:
    """标准（无权重）DTW 对齐器。"""

    def align(self, query: np.ndarray, reference: np.ndarray) -> AlignmentPath:
        query = np.asarray(query, dtype=float)
        reference = np.asarray(reference, dtype=float)
        if query.ndim != 2 or reference.ndim != 2:
            raise ValueError("待测序列与参考序列必须是二维数组")
        if query.shape[1] != reference.shape[1]:
            raise ValueError("待测序列与参考序列的特征维度必须一致")
        return dtw_align(query, reference)


@dataclass
class WeightedSegmentedDTWAligner(DTWAligner):
    """支持类 KMeans 阶段分割的加权 DTW 对齐器。

    use_segmented 是实例配置而非 align() 参数：调用方在构造时决定
    是否启用分段对齐，align() 的行为对所有调用方一致。
    """

    min_segment_frames: int = 3
    use_segmented: bool = True
    # 1/std 权重的极差上限。小样本模板中某特征的跨视频 std 偶然趋近于零
    # 时，原始 1/std 会到 1e6 量级，单特征即可主导对齐与总分；
    # 钳制到 min 权重的固定倍数保留「低方差更重要」的排序，同时避免退化。
    max_weight_ratio: float = 20.0

    def weighted_distance(self, query_frame: np.ndarray, reference_frame: np.ndarray, weights: np.ndarray) -> float:
        return float(np.sqrt(np.sum(weights * (query_frame - reference_frame) ** 2)))

    def compute_joint_weights(self, template) -> np.ndarray:
        std = np.asarray(template.std, dtype=float)
        if std.ndim != 1 or len(std) == 0 or not np.isfinite(std).all() or np.any(std < 0):
            raise ValueError("模板标准差必须由有限非负值组成")
        if not np.isfinite(self.max_weight_ratio) or self.max_weight_ratio < 1.0:
            raise ValueError("max_weight_ratio 必须是不小于 1 的有限值")
        weights = 1.0 / (std + 1e-6)
        weights = np.minimum(weights, weights.min() * self.max_weight_ratio)
        return self._normalize_weights(weights, len(std))

    def detect_phases(self, feature_seq: np.ndarray, n_phases: int = 4) -> list[tuple[int, int]]:
        return segmentation.detect_phases(feature_seq, n_phases, self.min_segment_frames)

    def weighted_dtw(self, query: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> AlignmentPath:
        query = np.asarray(query, dtype=float)
        reference = np.asarray(reference, dtype=float)
        weights = self._normalize_weights(weights, query.shape[1])
        return dtw_align(query, reference, weights)

    def align(
        self,
        query: np.ndarray,
        reference: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> AlignmentPath:
        return self.align_detailed(query, reference, weights).path

    def align_detailed(
        self,
        query: np.ndarray,
        reference: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> Alignment:
        """对齐并返回结构化结果（路径 + 待测序列分段 + 是否分段成功）。"""
        query = np.asarray(query, dtype=float)
        reference = np.asarray(reference, dtype=float)
        if query.ndim != 2 or reference.ndim != 2:
            raise ValueError("待测序列与参考序列必须是二维数组")
        if query.shape[1] != reference.shape[1]:
            raise ValueError("待测序列与参考序列的特征维度必须一致")
        if not np.isfinite(query).all() or not np.isfinite(reference).all():
            raise ValueError("待测序列与参考序列只能包含有限值")

        full_span = [(0, len(query))]
        default_weights = np.ones(query.shape[1], dtype=float) if weights is None else weights
        weights = self._normalize_weights(default_weights, query.shape[1])
        if not self.use_segmented:
            return Alignment(self.weighted_dtw(query, reference, weights), full_span, segmented=False)

        query_segments = self.detect_phases(query)
        ref_segments = self.detect_phases(reference)
        if not query_segments or not ref_segments:
            return Alignment([], [], segmented=False)
        if len(query_segments) != len(ref_segments):
            return Alignment(self.weighted_dtw(query, reference, weights), full_span, segmented=False)

        full_path: AlignmentPath = []
        for idx in range(len(query_segments)):
            q_start, q_end = query_segments[idx]
            r_start, r_end = ref_segments[idx]
            local_path = self.weighted_dtw(query[q_start:q_end], reference[r_start:r_end], weights)
            full_path.extend((q_start + i, r_start + j) for i, j in local_path)
        if not full_path or full_path[0] != (0, 0) or full_path[-1] != (len(query) - 1, len(reference) - 1):
            return Alignment(self.weighted_dtw(query, reference, weights), full_span, segmented=False)
        return Alignment(full_path, query_segments, segmented=True)

    @staticmethod
    def _normalize_weights(weights: np.ndarray, feature_dim: int) -> np.ndarray:
        normalized = np.asarray(weights, dtype=float)
        if normalized.ndim != 1 or normalized.shape[0] != feature_dim:
            raise ValueError("权重维度必须与特征维度一致")
        if not np.isfinite(normalized).all() or np.any(normalized < 0):
            raise ValueError("权重必须由有限非负值组成")
        total = float(normalized.sum())
        if total <= 0:
            raise ValueError("权重总和必须大于零")
        return normalized / total
