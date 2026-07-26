import numpy as np
import pytest

from movescope.alignment import DTWAligner
from movescope.assessment import AssessmentEngine
from movescope.reporting import generate_text_summary
from movescope.template import ActionTemplate


def make_template():
    template = ActionTemplate("squat")
    template.mean = np.zeros(12)
    template.std = np.ones(12)
    template.tolerance = np.ones(12) * 0.5
    template.representative_seq = np.zeros((5, 12))
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
    for phase in result["phases"]:
        assert phase["name"] == f"phase_{phase['index']}"


def test_generate_text_summary():
    engine = AssessmentEngine(make_template(), DTWAligner())
    result = engine.assess_features(np.ones((5, 12)) * 2.0)

    summary = generate_text_summary(result)

    assert "总分" in summary
    assert "主要问题" in summary


def test_non_finite_features_are_rejected():
    engine = AssessmentEngine(make_template(), DTWAligner())

    with pytest.raises(ValueError, match="有限值"):
        engine.assess_features(np.full((5, 12), np.nan))


def test_assess_coords_requires_feature_extractor():
    engine = AssessmentEngine(make_template(), DTWAligner())

    with pytest.raises(ValueError, match="feature_extractor"):
        engine.assess_coords(np.zeros((5, 17, 3)))
