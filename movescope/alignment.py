"""基于 DTW 的时序对齐。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Path = list[tuple[int, int]]


@dataclass
class DTWAligner:
    """自包含的标准 DTW 对齐器。"""

    def distance(self, query_frame: np.ndarray, reference_frame: np.ndarray) -> float:
        return float(np.linalg.norm(query_frame - reference_frame))

    def align(self, query: np.ndarray, reference: np.ndarray) -> Path:
        query = np.asarray(query, dtype=float)
        reference = np.asarray(reference, dtype=float)
        if query.ndim != 2 or reference.ndim != 2:
            raise ValueError("待测序列与参考序列必须是二维数组")
        if query.shape[1] != reference.shape[1]:
            raise ValueError("待测序列与参考序列的特征维度必须一致")
        if len(query) == 0 or len(reference) == 0:
            return []

        n_query, n_ref = len(query), len(reference)
        costs = np.empty((n_query, n_ref), dtype=float)
        for i in range(n_query):
            for j in range(n_ref):
                costs[i, j] = self.distance(query[i], reference[j])

        dp = np.full((n_query, n_ref), np.inf, dtype=float)
        dp[0, 0] = costs[0, 0]
        for i in range(1, n_query):
            dp[i, 0] = costs[i, 0] + dp[i - 1, 0]
        for j in range(1, n_ref):
            dp[0, j] = costs[0, j] + dp[0, j - 1]
        for i in range(1, n_query):
            for j in range(1, n_ref):
                dp[i, j] = costs[i, j] + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

        path: Path = []
        i, j = n_query - 1, n_ref - 1
        while True:
            path.append((i, j))
            if i == 0 and j == 0:
                break
            candidates = []
            if i > 0:
                candidates.append((dp[i - 1, j], i - 1, j))
            if j > 0:
                candidates.append((dp[i, j - 1], i, j - 1))
            if i > 0 and j > 0:
                candidates.append((dp[i - 1, j - 1], i - 1, j - 1))
            _, i, j = min(candidates, key=lambda item: item[0])

        path.reverse()
        return path


@dataclass
class WeightedSegmentedDTWAligner(DTWAligner):
    """支持类 KMeans 阶段分割的加权 DTW 对齐器。"""

    min_segment_frames: int = 3

    def weighted_distance(self, query_frame: np.ndarray, reference_frame: np.ndarray, weights: np.ndarray) -> float:
        return float(np.sqrt(np.sum(weights * (query_frame - reference_frame) ** 2)))

    def compute_joint_weights(self, template) -> np.ndarray:
        std = np.asarray(template.std, dtype=float)
        if std.ndim != 1 or len(std) == 0 or not np.isfinite(std).all() or np.any(std < 0):
            raise ValueError("模板标准差必须由有限非负值组成")
        weights = 1.0 / (std + 1e-6)
        return self._normalize_weights(weights, len(std))

    def detect_phases(self, feature_seq: np.ndarray, n_phases: int = 4) -> list[tuple[int, int]]:
        seq = np.asarray(feature_seq, dtype=float)
        if len(seq) == 0:
            return []
        if len(seq) < n_phases * self.min_segment_frames:
            return [(0, len(seq))]

        labels = self._kmeans_labels(seq, n_clusters=n_phases)
        segments: list[tuple[int, int]] = []
        start = 0
        for idx in range(1, len(labels)):
            if labels[idx] != labels[start]:
                segments.append((start, idx))
                start = idx
        segments.append((start, len(labels)))
        return self._merge_short_segments(segments, len(seq))

    def _kmeans_labels(self, seq: np.ndarray, n_clusters: int, max_iter: int = 25) -> np.ndarray:
        n_clusters = max(1, min(n_clusters, len(seq)))
        if n_clusters == 1:
            return np.zeros(len(seq), dtype=int)

        center_indices = np.linspace(0, len(seq) - 1, n_clusters, dtype=int)
        centers = seq[center_indices].copy()
        labels = np.zeros(len(seq), dtype=int)

        for _ in range(max_iter):
            distances = np.sum((seq[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            next_labels = np.argmin(distances, axis=1)
            if np.array_equal(next_labels, labels):
                break
            labels = next_labels
            for cluster_idx in range(n_clusters):
                members = seq[labels == cluster_idx]
                if len(members) > 0:
                    centers[cluster_idx] = members.mean(axis=0)

        return labels

    def _merge_short_segments(self, segments: list[tuple[int, int]], total_len: int) -> list[tuple[int, int]]:
        if not segments:
            return [(0, total_len)] if total_len else []

        merged: list[tuple[int, int]] = []
        for start, end in segments:
            if end - start < self.min_segment_frames and merged:
                prev_start, _ = merged[-1]
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        if len(merged) > 1 and merged[0][1] - merged[0][0] < self.min_segment_frames:
            first_start, _ = merged.pop(0)
            _, next_end = merged[0]
            merged[0] = (first_start, next_end)

        return merged or [(0, total_len)]

    def weighted_dtw(self, query: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> Path:
        query = np.asarray(query, dtype=float)
        reference = np.asarray(reference, dtype=float)
        weights = self._normalize_weights(weights, query.shape[1])

        base = DTWAligner()
        base.distance = lambda q, r: self.weighted_distance(q, r, weights)  # type: ignore[method-assign]
        return base.align(query, reference)

    def align(
        self,
        query: np.ndarray,
        reference: np.ndarray,
        weights: np.ndarray | None = None,
        use_segmented: bool = True,
    ) -> Path:
        query = np.asarray(query, dtype=float)
        reference = np.asarray(reference, dtype=float)
        if query.ndim != 2 or reference.ndim != 2:
            raise ValueError("待测序列与参考序列必须是二维数组")
        if query.shape[1] != reference.shape[1]:
            raise ValueError("待测序列与参考序列的特征维度必须一致")
        if not np.isfinite(query).all() or not np.isfinite(reference).all():
            raise ValueError("待测序列与参考序列只能包含有限值")

        default_weights = np.ones(query.shape[1], dtype=float) if weights is None else weights
        weights = self._normalize_weights(default_weights, query.shape[1])
        if not use_segmented:
            return self.weighted_dtw(query, reference, weights)

        query_segments = self.detect_phases(query)
        ref_segments = self.detect_phases(reference)
        if not query_segments or not ref_segments:
            return []
        if len(query_segments) != len(ref_segments):
            return self.weighted_dtw(query, reference, weights)

        full_path: Path = []
        for idx in range(len(query_segments)):
            q_start, q_end = query_segments[idx]
            r_start, r_end = ref_segments[idx]
            local_path = self.weighted_dtw(query[q_start:q_end], reference[r_start:r_end], weights)
            full_path.extend((q_start + i, r_start + j) for i, j in local_path)
        if not full_path or full_path[0] != (0, 0) or full_path[-1] != (len(query) - 1, len(reference) - 1):
            return self.weighted_dtw(query, reference, weights)
        return full_path

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
