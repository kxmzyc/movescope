import numpy as np
import pytest

from movescope.alignment import DTWAligner
from movescope.assessment import AssessmentEngine, generate_text_summary
from movescope.features import FeatureExtractor, JOINT_NAMES
from movescope.template import ActionTemplate


class PassthroughFeatureExtractor(FeatureExtractor):
    def extract(self, coords_3d, normalize=True):
        return np.asarray(coords_3d, dtype=float)


def make_template():
    template = ActionTemplate("squat")
    template.mean = np.zeros(12)
    template.std = np.ones(12)
    template.tolerance = np.ones(12) * 0.5
    template.representative_seq = np.zeros((5, 12))
    template.n_videos = 1
    return template


def test_perfect_match_score_is_100():
    engine = AssessmentEngine(make_template(), DTWAligner(), PassthroughFeatureExtractor())

    result = engine.assess(np.zeros((5, 12)))

    assert result["total_score"] == 100.0


def test_all_wrong_scores_below_50():
    engine = AssessmentEngine(make_template(), DTWAligner(), PassthroughFeatureExtractor())

    result = engine.assess(np.ones((5, 12)) * 2.0)

    assert result["total_score"] < 50.0
    assert result["phases"][0]["anomalies"]


def test_generate_text_summary():
    engine = AssessmentEngine(make_template(), DTWAligner(), PassthroughFeatureExtractor())
    result = engine.assess(np.ones((5, 12)) * 2.0)

    summary = generate_text_summary(result)

    assert "总分" in summary
    assert "主要问题" in summary


def test_non_finite_features_are_rejected():
    engine = AssessmentEngine(make_template(), DTWAligner(), PassthroughFeatureExtractor())

    with pytest.raises(ValueError, match="有限值"):
        engine.assess(np.full((5, 12), np.nan))
