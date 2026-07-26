"""把多次往复的深蹲序列按膝角曲线拆分为逐次片段并逐次评分。

模板按「单次完整深蹲」构建，整段评估多次往复的视频时，往复间歇、
站立与走动都会被 DTW 硬对齐成偏差，分数随拍摄起止点大幅波动。
这里在特征层检测每次往复（膝屈曲角滞回），逐次调用引擎评分：
总分为逐次均值，详情（阶段/时间轴）取分数最接近均值的代表次，
时间统一偏移回原视频时刻。

只有动作名含 squat 且检测到至少两次完整往复时才逐次评分，
否则保持整段评估的既有行为；非深蹲动作不套用膝角切分。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from movescope.assessment import AssessmentEngine
from movescope.features import FeatureExtractor
from movescope.types import PoseResult

# 判定存在往复动作的最小膝角活动范围（度），与语义阶段标签口径一致。
MIN_REP_RANGE_DEG = 15.0
# 每次往复的最小时长（秒），滤掉视频首尾的残缺小片段。
MIN_REP_SEC = 0.5
# 片段向两侧的余量帧数，保留起止的站立过渡供分段对齐使用。
REP_MARGIN_FRAMES = 6
# 滞回阈值：膝角降到活动范围下沿 30% 以内视为进入蹲底，
# 回升到上沿 20% 以内视为回到站立、当次往复结束。
_DESCEND_FRACTION = 0.3
_STAND_FRACTION = 0.2


def knee_curve(features: np.ndarray, angle_features: list) -> np.ndarray | None:
    """双膝屈曲角均值曲线；无膝特征或含非有限值时返回 None。"""
    knee_cols = [feature.index for feature in angle_features if "knee" in feature.joint]
    if not knee_cols:
        return None
    knee = features[:, knee_cols]
    if not np.isfinite(knee).all():
        return None
    return knee.mean(axis=1)


def detect_rep_segments(knee: np.ndarray, fps: float) -> list[tuple[int, int]]:
    """按膝角滞回检测每次往复的帧区间（含两侧余量，闭区间）。

    阈值取 5/95 百分位而不是 min/max，抵抗单帧尖刺；未回到站立位的
    末尾残段不计入。返回空列表表示未检测到有效往复。
    """
    if len(knee) < 3 or not np.isfinite(fps) or fps <= 0:
        return []
    curve = _smooth(knee)
    lo_v, hi_v = np.percentile(curve, [5, 95])
    span = hi_v - lo_v
    if span < MIN_REP_RANGE_DEG:
        return []
    descend = lo_v + _DESCEND_FRACTION * span
    stand = hi_v - _STAND_FRACTION * span
    min_frames = max(3, int(MIN_REP_SEC * fps))

    segments: list[tuple[int, int]] = []
    in_rep = False
    start = 0
    for idx, value in enumerate(curve):
        if not in_rep and value < descend:
            in_rep = True
            start = idx
            while start > 0 and curve[start - 1] < stand:
                start -= 1
        elif in_rep and value > stand:
            if idx - start + 1 >= min_frames:
                segments.append(
                    (max(0, start - REP_MARGIN_FRAMES), min(len(curve) - 1, idx + REP_MARGIN_FRAMES))
                )
            in_rep = False
    return segments


def assess_pose_by_rep(engine: AssessmentEngine, pose: PoseResult) -> dict[str, Any]:
    """assess_pose 的逐次往复版本：多次往复时逐次评分并汇总。"""
    if engine.feature_extractor is None:
        raise ValueError("assess_pose_by_rep 需要 feature_extractor")
    features = engine.feature_extractor.extract(pose.best_coords_3d, normalize=False)
    return assess_features_by_rep(engine, features, fps=pose.fps)


def assess_features_by_rep(
    engine: AssessmentEngine, features: np.ndarray, fps: float | None = None
) -> dict[str, Any]:
    """检测到 ≥2 次往复时逐次评分：总分为均值，详情取最接近均值的代表次。

    响应在整段评估的结构上追加 reps（逐次摘要）与 rep_detail_index
    （详情对应第几次）；单次/未检测到往复时与 assess_features 完全一致。
    """
    fps = engine.fps if fps is None else fps
    features = np.asarray(features, dtype=float)

    segments: list[tuple[int, int]] = []
    curve: np.ndarray | None = None
    if "squat" in engine.template.action_name.lower() and features.ndim == 2 and len(features) > 0:
        extractor = engine.feature_extractor or FeatureExtractor()
        angle_features = extractor.features
        if len(angle_features) == features.shape[1]:
            curve = knee_curve(features, angle_features)
            if curve is not None:
                segments = detect_rep_segments(curve, fps)

    if len(segments) < 2 or curve is None:
        return engine.assess_features(features, fps=fps)

    rep_results = [engine.assess_features(features[start : end + 1], fps=fps) for start, end in segments]
    scores = [float(rep["total_score"]) for rep in rep_results]
    mean_score = round(float(np.mean(scores)), 2)
    detail_index = min(range(len(scores)), key=lambda i: (abs(scores[i] - mean_score), i))

    result = rep_results[detail_index]
    _shift_times(result, segments[detail_index][0] / fps)
    result["total_score"] = mean_score
    result["reps"] = [
        {
            "index": idx,
            "time_range": [round(start / fps, 2), round(end / fps, 2)],
            "score": scores[idx],
            "knee_min_deg": round(float(np.min(curve[start : end + 1])), 1),
        }
        for idx, (start, end) in enumerate(segments)
    ]
    result["rep_detail_index"] = detail_index
    return result


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """边缘复制填充的滑动平均；零填充会在首尾制造假谷底。"""
    if len(values) < window:
        return values.astype(float, copy=True)
    pad = window // 2
    padded = np.pad(values.astype(float), pad, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _shift_times(result: dict[str, Any], offset_sec: float) -> None:
    """把代表次结果中的绝对时刻偏移回原视频坐标（原地修改）。"""
    if offset_sec <= 0:
        return
    for phase in result.get("phases", []):
        phase["time_range"] = [round(t + offset_sec, 2) for t in phase["time_range"]]
        for anomaly in phase.get("anomalies", []):
            anomaly["peak_time_sec"] = round(anomaly["peak_time_sec"] + offset_sec, 2)
    timeline = result.get("timeline")
    if timeline:
        timeline["time_sec"] = [round(t + offset_sec, 3) for t in timeline["time_sec"]]
