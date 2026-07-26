"""向量化的 DTW 代价矩阵、动态规划与回溯。

与最初的逐单元纯 Python 实现在数值上逐项一致：

- 代价采用「直接差值」形式 sqrt(sum(w * (q - r)**2))，与逐帧计算的
  运算次序相同。刻意不用 ||q||²+||r||²-2q·r 的 BLAS 展开式——它在
  q≈r 时存在灾难性抵消，可能以 1e-8 量级扰动代价值并翻转 DP 平局。
- DP 递推对每个单元执行 costs[i,j] + min(上, 左, 对角)，min 的结合
  顺序不影响取值，因此 DP 表与朴素实现完全相同。
- 回溯保留原实现的候选顺序 [上, 左, 对角] 与 min 语义（取第一个
  最小值），确保平局时路径逐点一致。
"""

from __future__ import annotations

import numpy as np

AlignmentPath = list[tuple[int, int]]

# 代价矩阵分块计算时，单块 (chunk, m, d) 差值张量的内存上限。
_CHUNK_BYTES = 64 * 1024 * 1024


def cost_matrix(
    query: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray | None = None,
    band_ratio: float | None = None,
) -> np.ndarray:
    """逐帧对的（加权）欧氏距离矩阵，形状 (len(query), len(reference))。

    band_ratio 非空时应用 Sakoe-Chiba 带约束：以斜率归一的对角线为中心，
    带宽为参考长度的 band_ratio 倍（并放宽到不小于两序列长度差与 1 帧，
    保证 (0,0)→(n-1,m-1) 始终存在可行路径），带外单元代价为 +inf。
    """
    query = np.asarray(query, dtype=float)
    reference = np.asarray(reference, dtype=float)
    n_query, feature_dim = query.shape
    n_ref = reference.shape[0]

    band_width = None
    if band_ratio is not None:
        if not np.isfinite(band_ratio) or not 0.0 < band_ratio <= 1.0:
            raise ValueError("band_ratio 必须在 (0, 1] 区间内")
        band_width = max(int(np.ceil(band_ratio * n_ref)), abs(n_ref - n_query), 1)

    costs = np.empty((n_query, n_ref), dtype=float)
    ref_index = np.arange(n_ref, dtype=float)
    slope = (n_ref - 1) / (n_query - 1) if n_query > 1 else 0.0
    chunk = max(1, _CHUNK_BYTES // (max(1, n_ref * feature_dim) * 8))
    for start in range(0, n_query, chunk):
        end = min(n_query, start + chunk)
        diff = query[start:end, None, :] - reference[None, :, :]
        np.multiply(diff, diff, out=diff)
        if weights is not None:
            diff *= weights
        np.sqrt(diff.sum(axis=2), out=costs[start:end])
        if band_width is not None:
            centers = np.arange(start, end, dtype=float) * slope
            outside = np.abs(centers[:, None] - ref_index[None, :]) > band_width
            costs[start:end][outside] = np.inf
    return costs


def dtw_dp(costs: np.ndarray) -> np.ndarray:
    """累积代价表。dp[i, j] = costs[i, j] + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])。"""
    n_query, n_ref = costs.shape
    dp = np.empty((n_query, n_ref), dtype=float)
    np.cumsum(costs[0], out=dp[0])

    for i in range(1, n_query):
        prev = dp[i - 1]
        # 每列 j 的 min(上, 对角) 可以整行向量化；只有对 dp[i, j-1] 的
        # 依赖必须顺序扫描。转成 Python 标量循环避免逐单元 ndarray 索引开销。
        up_or_diag = np.minimum(prev[1:], prev[:-1]).tolist()
        row_costs = costs[i].tolist()
        row = [0.0] * n_ref
        left = row_costs[0] + float(prev[0])
        row[0] = left
        for j in range(1, n_ref):
            best = up_or_diag[j - 1]
            if left < best:
                best = left
            left = row_costs[j] + best
            row[j] = left
        dp[i] = row
    return dp


def backtrack(dp: np.ndarray) -> AlignmentPath:
    """从 dp 右下角回溯路径。候选顺序与平局语义与朴素实现完全一致。"""
    n_query, n_ref = dp.shape
    path: AlignmentPath = []
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


def dtw_align(
    query: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray | None = None,
    band_ratio: float | None = None,
) -> AlignmentPath:
    """完整 DTW 对齐：代价矩阵 → DP → 回溯。空序列返回空路径。"""
    if len(query) == 0 or len(reference) == 0:
        return []
    return backtrack(dtw_dp(cost_matrix(query, reference, weights, band_ratio)))
