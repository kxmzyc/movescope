"""评分纯函数。

从 AssessmentEngine 中抽出的数学部分，与重构前数值完全一致。
"""

from __future__ import annotations

import numpy as np


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def total_score(anomaly_ratio_per_feature: np.ndarray, weights: np.ndarray | None = None) -> float:
    """由逐特征异常帧占比得到 0-100 总分。

    weights 非空时按权重平均（历史行为：直接复用 DTW 的 1/std 特征
    权重），否则等权平均。
    """
    if weights is not None:
        mean_ratio = float(np.average(anomaly_ratio_per_feature, weights=weights))
    else:
        mean_ratio = float(anomaly_ratio_per_feature.mean())
    return clamp(100.0 - mean_ratio * 100.0)
