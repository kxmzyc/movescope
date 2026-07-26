import numpy as np
import pytest

from movescope.alignment import DTWAligner, WeightedSegmentedDTWAligner
from movescope.assessment import AssessmentEngine
from movescope.errors import InvalidInputError
from movescope.reporting import generate_text_summary
from movescope.template import ActionTemplate


def make_template(frames: int = 5, dims: int = 12, tolerance: float = 0.5):
    template = ActionTemplate("squat")
    template.mean = np.zeros(dims)
    template.std = np.ones(dims)
    template.tolerance = np.ones(dims) * tolerance
    template.representative_seq = np.zeros((frames, dims))
    template.n_videos = 1
    return template


def test_perfect_match_score_is_100():
    engine = AssessmentEngine(make_template(), DTWAligner())

    result = engine.assess_features(np.zeros((5, 12)))

    assert result["total_score"] == 100.0


def test_all_wrong_scores_below_50():
    engine = AssessmentEngine(make_template(), DTWAligner())

    result = engine.assess_features(np.ones((5, 12)) * 2.0)

    assert result["total_score"] < 50.0
    assert result["phases"][0]["anomalies"]
    anomaly = result["phases"][0]["anomalies"][0]
    assert {"feature_index", "joint", "joint_display", "parent", "child"} <= set(anomaly)


def test_structured_summary_and_phases():
    engine = AssessmentEngine(make_template(), DTWAligner())

    result = engine.assess_features(np.ones((5, 12)) * 2.0)

    assert result["segmented"] is False
    assert isinstance(result["per_feature_summary"], list)
    assert len(result["per_feature_summary"]) == 12
    first = result["per_feature_summary"][0]
    assert first["joint"] == "left_knee"
    assert first["joint_display"] == "左膝"
    assert first["parent"] == "left_hip"
    assert first["child"] == "left_ankle"
    assert first["tolerance_deg"] == 0.5
    assert first["score_weight"] == pytest.approx(1 / 12, abs=1e-4)
    for phase in result["phases"]:
        assert phase["name"] == f"phase_{phase['index']}"
    assert result["phases"][0]["label"] == "整段动作"


def test_anomaly_ratio_counts_unique_test_frames():
    """DTW 一对多匹配（如动作停顿）不应重复计入异常帧占比。"""
    engine = AssessmentEngine(make_template(frames=5), DTWAligner())
    test = np.zeros((20, 12))
    test[3] = 2.0  # 只有 1/20 的测试帧越界

    result = engine.assess_features(test)

    summary = result["per_feature_summary"][0]
    assert summary["anomaly_ratio"] == pytest.approx(1 / 20)
    assert result["total_score"] == pytest.approx(95.0)


def test_timeline_matches_frames_and_tolerance():
    engine = AssessmentEngine(make_template(frames=5), DTWAligner())
    test = np.zeros((8, 12))
    test[2] = 2.0

    timeline = engine.assess_features(test, fps=10.0)["timeline"]

    assert timeline["frame_count"] == 8
    assert timeline["frame_stride"] == 1
    assert timeline["time_sec"] == pytest.approx([i / 10.0 for i in range(8)])
    assert len(timeline["series"]) == 12
    series = timeline["series"][0]
    assert len(series["test_deg"]) == 8
    assert len(series["reference_deg"]) == 8
    assert series["anomaly"] == [False, False, True, False, False, False, False, False]
    # v1 模板无容差带，广播为等值的逐帧容差数组。
    assert series["tolerance_deg"] == [0.5] * 8


def test_timeline_downsamples_long_sequences():
    engine = AssessmentEngine(make_template(frames=5), DTWAligner(), timeline_max_points=10)

    timeline = engine.assess_features(np.zeros((25, 12)))["timeline"]

    assert timeline["frame_stride"] == 3
    assert len(timeline["time_sec"]) == 9
    assert all(len(series["test_deg"]) == 9 for series in timeline["series"])


def test_nan_feature_column_is_excluded_not_fatal():
    engine = AssessmentEngine(make_template(), DTWAligner())
    test = np.zeros((5, 12))
    test[2, 6] = np.nan  # left_elbow 特征列不完整

    result = engine.assess_features(test)

    assert [item["feature_index"] for item in result["excluded_features"]] == [6]
    assert "不完整" in result["excluded_features"][0]["reason"]
    assert len(result["per_feature_summary"]) == 11
    assert all(item["feature_index"] != 6 for item in result["per_feature_summary"])
    assert len(result["timeline"]["series"]) == 11
    assert result["total_score"] == 100.0


def test_missing_required_feature_is_rejected():
    engine = AssessmentEngine(make_template(), DTWAligner(), required_features=(0, 1))
    test = np.zeros((5, 12))
    test[:, 0] = np.nan

    with pytest.raises(InvalidInputError, match="左膝"):
        engine.assess_features(test)


def test_all_nan_features_are_rejected():
    engine = AssessmentEngine(make_template(), DTWAligner())

    with pytest.raises(ValueError, match="有限值"):
        engine.assess_features(np.full((5, 12), np.nan))


def test_segmented_alignment_reports_real_phases():
    """分段对齐成功时输出真实检测到的多阶段（0.3.0 核心行为的直接测试）。"""
    blocks = np.vstack(
        [
            np.zeros((6, 12)),
            np.ones((6, 12)) * 40.0,
            np.ones((6, 12)) * 80.0,
        ]
    )
    template = make_template(frames=18, tolerance=5.0)
    template.representative_seq = blocks.copy()
    engine = AssessmentEngine(template, WeightedSegmentedDTWAligner(min_segment_frames=2), fps=10.0)

    result = engine.assess_features(blocks.copy())

    assert result["segmented"] is True
    assert len(result["phases"]) >= 2
    starts = [phase["time_range"][0] for phase in result["phases"]]
    assert starts == sorted(starts)
    assert result["phases"][0]["time_range"][0] == 0.0
    assert result["phases"][-1]["time_range"][1] == pytest.approx(17 / 10.0, abs=0.01)
    assert result["total_score"] == 100.0


def test_generate_text_summary():
    engine = AssessmentEngine(make_template(), DTWAligner())
    result = engine.assess_features(np.ones((5, 12)) * 2.0)

    summary = generate_text_summary(result)

    assert "总分" in summary
    assert "主要问题" in summary


def test_assess_coords_requires_feature_extractor():
    engine = AssessmentEngine(make_template(), DTWAligner())

    with pytest.raises(ValueError, match="feature_extractor"):
        engine.assess_coords(np.zeros((5, 17, 3)))


def test_score_weights_default_uniform_decouples_alignment():
    """默认评分等权：低方差特征只主导对齐，不再同时主导总分。"""
    template = make_template()
    template.std = np.array([0.01] + [1.0] * 11)
    engine = AssessmentEngine(template, WeightedSegmentedDTWAligner())

    result = engine.assess_features(np.zeros((5, 12)))

    weights = [item["score_weight"] for item in result["per_feature_summary"]]
    assert all(w == pytest.approx(1 / 12, abs=1e-4) for w in weights)


def test_score_weights_aligner_reuses_alignment_weights():
    template = make_template()
    template.std = np.array([0.01] + [1.0] * 11)
    engine = AssessmentEngine(template, WeightedSegmentedDTWAligner(), score_weights="aligner")

    result = engine.assess_features(np.zeros((5, 12)))

    weights = [item["score_weight"] for item in result["per_feature_summary"]]
    assert weights[0] > weights[1]


def test_per_frame_tolerance_band_drives_anomaly():
    """同样的偏差量：容差宽的帧不越界，容差窄的帧越界。"""
    template = make_template(frames=6)
    reference = np.zeros((6, 12))
    reference[:, 11] = np.arange(6) * 10.0  # 锚定对角线对齐
    template.reference_seq = reference
    band = np.full((6, 12), 10.0)
    band[3:, :] = 2.0
    template.tolerance_band = band
    engine = AssessmentEngine(template, DTWAligner())

    test = reference.copy()
    test[:, 0] += 5.0  # 恒定 5 度偏差
    result = engine.assess_features(test)

    series = result["timeline"]["series"][0]
    assert series["tolerance_deg"] == [10.0, 10.0, 10.0, 2.0, 2.0, 2.0]
    assert series["anomaly"] == [False, False, False, True, True, True]


def test_squat_phase_labels_are_semantic():
    engine = AssessmentEngine(make_template(), DTWAligner())
    features = engine._angle_features(12)
    knee_cols = [idx for idx, feature in enumerate(features) if "knee" in feature.joint]
    test = np.full((25, 12), 170.0)
    ranges = [(i * 5, (i + 1) * 5) for i in range(5)]
    for (start, end), value in zip(ranges, [170.0, 140.0, 95.0, 140.0, 170.0], strict=True):
        test[start:end, knee_cols] = value

    labels = engine._phase_labels(test, features, ranges)

    assert labels == ["站立准备", "下蹲", "蹲底", "起立", "站立还原"]


def test_phase_labels_fall_back_for_non_squat_action():
    engine = AssessmentEngine(make_template(), DTWAligner())
    engine.template.action_name = "jumping_jack"
    features = engine._angle_features(12)
    test = np.full((10, 12), 170.0)
    test[5:, :] = 95.0

    labels = engine._phase_labels(test, features, [(0, 5), (5, 10)])

    assert labels == ["阶段 1", "阶段 2"]


def test_phase_labels_fall_back_for_shallow_motion():
    """膝角活动范围不足时不套用深蹲语义，避免误导性标签。"""
    engine = AssessmentEngine(make_template(), DTWAligner())
    features = engine._angle_features(12)
    test = np.full((10, 12), 170.0)
    test[5:, :] = 165.0

    labels = engine._phase_labels(test, features, [(0, 5), (5, 10)])

    assert labels == ["阶段 1", "阶段 2"]
