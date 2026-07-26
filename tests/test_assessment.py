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
    assert series["tolerance_deg"] == 0.5


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
