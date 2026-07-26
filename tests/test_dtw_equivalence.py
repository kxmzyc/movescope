"""新向量化 DTW 与旧逐单元实现的对拍测试。

tests/legacy_alignment.py 是重构前 movescope/alignment.py 的逐字节副本，
作为参考实现。本文件在随机输入、平局密集输入和加权场景下断言新旧
路径逐点相同——这是 DTW 向量化重构的关键安全网。
"""

from __future__ import annotations

import legacy_alignment
import numpy as np
import pytest

from movescope.alignment import DTWAligner, WeightedSegmentedDTWAligner


def random_cases() -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(20260726)
    cases: list[tuple[np.ndarray, np.ndarray]] = []
    # 连续随机值
    for n_query, n_ref, dim in [(1, 1, 1), (1, 7, 3), (7, 1, 3), (5, 5, 2), (13, 9, 12), (40, 33, 12), (25, 60, 6)]:
        cases.append((rng.normal(size=(n_query, dim)), rng.normal(size=(n_ref, dim))))
    # 量化整数值：制造大量精确平局，专门考验回溯的打破平局顺序
    for n_query, n_ref, dim in [(8, 8, 2), (15, 12, 4), (30, 30, 3)]:
        cases.append(
            (
                rng.integers(0, 3, size=(n_query, dim)).astype(float),
                rng.integers(0, 3, size=(n_ref, dim)).astype(float),
            )
        )
    # 常数序列：所有代价相同的极端平局
    cases.append((np.ones((6, 3)), np.ones((9, 3))))
    # 序列对自身
    self_seq = rng.normal(size=(20, 5))
    cases.append((self_seq, self_seq.copy()))
    return cases


@pytest.mark.parametrize("query,reference", random_cases())
def test_standard_dtw_matches_legacy(query, reference):
    new_path = DTWAligner().align(query, reference)
    old_path = legacy_alignment.DTWAligner().align(query, reference)

    assert new_path == old_path


@pytest.mark.parametrize("query,reference", random_cases())
def test_weighted_segmented_matches_legacy(query, reference):
    rng = np.random.default_rng(hash((query.shape, reference.shape)) % (2**32))
    weights = rng.uniform(0.1, 5.0, size=query.shape[1])

    for use_segmented in (True, False):
        new_aligner = WeightedSegmentedDTWAligner(use_segmented=use_segmented)
        old_aligner = legacy_alignment.WeightedSegmentedDTWAligner()

        new_path = new_aligner.align(query, reference, weights=weights)
        old_path = old_aligner.align(query, reference, weights=weights, use_segmented=use_segmented)
        assert new_path == old_path, f"use_segmented={use_segmented} 时路径不一致"

        new_default = new_aligner.align(query, reference)
        old_default = old_aligner.align(query, reference, use_segmented=use_segmented)
        assert new_default == old_default, f"use_segmented={use_segmented} 默认权重路径不一致"


def test_weighted_dtw_direct_matches_legacy():
    rng = np.random.default_rng(7)
    query = rng.normal(size=(24, 12))
    reference = rng.normal(size=(31, 12))
    weights = rng.uniform(0.05, 3.0, size=12)

    new_path = WeightedSegmentedDTWAligner().weighted_dtw(query, reference, weights)
    old_path = legacy_alignment.WeightedSegmentedDTWAligner().weighted_dtw(query, reference, weights)

    assert new_path == old_path
