from itertools import pairwise

import numpy as np
import pytest

from movescope.alignment import DTWAligner
from movescope.assessment import AssessmentEngine
from movescope.features import FeatureExtractor
from movescope.reps import assess_features_by_rep, detect_rep_segments, knee_curve
from movescope.template import ActionTemplate

FPS = 30.0
STAND_DEG = 170.0
DEPTH_DEG = 80.0


def make_multi_rep_features(reps: int = 3, rep_frames: int = 30, rest_frames: int = 20) -> np.ndarray:
    """站立-深蹲交替的合成特征：膝角列为 sin² 往复，其余列恒定。"""
    rest = np.full(rest_frames, STAND_DEG)
    pieces = [rest]
    for _ in range(reps):
        t = np.linspace(0.0, np.pi, rep_frames)
        pieces.append(STAND_DEG - DEPTH_DEG * np.sin(t) ** 2)
        pieces.append(rest)
    knee = np.concatenate(pieces)
    features = np.full((len(knee), 12), 100.0)
    features[:, 0] = knee
    features[:, 1] = knee
    return features


def make_template(reference: np.ndarray, action: str = "squat") -> ActionTemplate:
    template = ActionTemplate(action)
    template.mean = reference.mean(axis=0)
    template.std = np.ones(reference.shape[1])
    template.tolerance = np.full(reference.shape[1], 8.0)
    template.representative_seq = reference
    template.n_videos = 1
    return template


def make_engine(features: np.ndarray, action: str = "squat") -> AssessmentEngine:
    """用首次往复片段做参考曲线，模拟单次深蹲模板。"""
    segments = detect_rep_segments(knee_curve(features, FeatureExtractor().features), FPS)
    start, end = segments[0]
    return AssessmentEngine(
        template=make_template(features[start : end + 1].copy(), action),
        aligner=DTWAligner(),
        feature_extractor=FeatureExtractor(),
        fps=FPS,
    )


def test_detect_rep_segments_finds_each_rep():
    features = make_multi_rep_features(reps=3)
    knee = knee_curve(features, FeatureExtractor().features)

    segments = detect_rep_segments(knee, FPS)

    assert len(segments) == 3
    assert all(start < end for start, end in segments)
    assert all(prev[1] <= cur[0] + 12 for prev, cur in pairwise(segments))
    # 每段都覆盖一个真实谷底
    for start, end in segments:
        assert knee[start : end + 1].min() < STAND_DEG - 0.8 * DEPTH_DEG


def test_detect_ignores_shallow_motion():
    t = np.linspace(0.0, 3 * np.pi, 120)
    knee = STAND_DEG - 8.0 * np.sin(t) ** 2  # 活动范围低于 15 度

    assert detect_rep_segments(knee, FPS) == []


def test_detect_filters_short_spike():
    """两次完整往复之间的瞬时深尖刺不算一次往复。"""
    features = make_multi_rep_features(reps=2)
    knee = knee_curve(features, FeatureExtractor().features)
    spike_at = len(knee) // 2
    knee = np.concatenate([knee[:spike_at], np.full(4, STAND_DEG - DEPTH_DEG), knee[spike_at:]])

    segments = detect_rep_segments(knee, FPS)

    assert len(segments) == 2


def test_detect_drops_trailing_incomplete_rep():
    features = make_multi_rep_features(reps=2)
    half_descend = np.linspace(STAND_DEG, STAND_DEG - DEPTH_DEG, 15)  # 视频在下蹲途中结束
    knee = np.concatenate([knee_curve(features, FeatureExtractor().features), half_descend])

    segments = detect_rep_segments(knee, FPS)

    assert len(segments) == 2


def test_by_rep_returns_mean_score_and_offset_detail():
    features = make_multi_rep_features(reps=3)
    engine = make_engine(features)

    result = assess_features_by_rep(engine, features, fps=FPS)

    assert len(result["reps"]) == 3
    detail_index = result["rep_detail_index"]
    assert 0 <= detail_index < 3
    mean_score = float(np.mean([rep["score"] for rep in result["reps"]]))
    assert result["total_score"] == pytest.approx(mean_score, abs=0.01)
    assert all(rep["knee_min_deg"] == pytest.approx(STAND_DEG - DEPTH_DEG, abs=2.0) for rep in result["reps"])
    # 详情（时间轴/阶段）的时刻已偏移回原视频坐标
    detail_start = result["reps"][detail_index]["time_range"][0]
    assert result["timeline"]["time_sec"][0] == pytest.approx(detail_start, abs=0.05)
    assert result["phases"][0]["time_range"][0] == pytest.approx(detail_start, abs=0.05)
    # 逐次片段时间单调排布
    starts = [rep["time_range"][0] for rep in result["reps"]]
    assert starts == sorted(starts)


def test_single_rep_input_keeps_plain_result():
    features = make_multi_rep_features(reps=1)
    engine = make_engine(features)

    result = assess_features_by_rep(engine, features, fps=FPS)

    assert "reps" not in result
    assert result["total_score"] == engine.assess_features(features, fps=FPS)["total_score"]


def test_non_squat_action_skips_rep_splitting():
    features = make_multi_rep_features(reps=3)
    engine = make_engine(features, action="jumping_jack")

    result = assess_features_by_rep(engine, features, fps=FPS)

    assert "reps" not in result


def test_nan_knee_falls_back_to_whole_sequence():
    features = make_multi_rep_features(reps=3)
    features[5, 0] = np.nan
    engine = make_engine(make_multi_rep_features(reps=3))

    result = assess_features_by_rep(engine, features, fps=FPS)

    assert "reps" not in result
    assert any(item["feature_index"] == 0 for item in result["excluded_features"])
