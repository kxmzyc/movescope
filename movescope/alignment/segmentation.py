"""基于类 KMeans 聚类的动作阶段分割。

从旧 alignment.py 原样迁出：标签遍历、短分段合并逻辑与原实现
逐行为等价，仅由方法改为自由函数。
"""

from __future__ import annotations

import numpy as np


def detect_phases(feature_seq: np.ndarray, n_phases: int = 4, min_segment_frames: int = 3) -> list[tuple[int, int]]:
    seq = np.asarray(feature_seq, dtype=float)
    if len(seq) == 0:
        return []
    if len(seq) < n_phases * min_segment_frames:
        return [(0, len(seq))]

    labels = kmeans_labels(seq, n_clusters=n_phases)
    segments: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, len(labels)):
        if labels[idx] != labels[start]:
            segments.append((start, idx))
            start = idx
    segments.append((start, len(labels)))
    return merge_short_segments(segments, len(seq), min_segment_frames)


def kmeans_labels(seq: np.ndarray, n_clusters: int, max_iter: int = 25) -> np.ndarray:
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


def merge_short_segments(
    segments: list[tuple[int, int]], total_len: int, min_segment_frames: int
) -> list[tuple[int, int]]:
    if not segments:
        return [(0, total_len)] if total_len else []

    merged: list[tuple[int, int]] = []
    for start, end in segments:
        if end - start < min_segment_frames and merged:
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    if len(merged) > 1 and merged[0][1] - merged[0][0] < min_segment_frames:
        first_start, _ = merged.pop(0)
        _, next_end = merged[0]
        merged[0] = (first_start, next_end)

    return merged or [(0, total_len)]
