"""Deterministic synthetic assessment for an out-of-box MoveScope demo."""

from __future__ import annotations

import numpy as np

from movescope.alignment import WeightedSegmentedDTWAligner
from movescope.assessment import AssessmentEngine
from movescope.features import FeatureExtractor
from movescope.llm_advisor import LLMAdvisor
from movescope.template import ActionTemplate


class _PassthroughFeatureExtractor(FeatureExtractor):
    def extract(self, coords_3d, normalize=True):
        return np.asarray(coords_3d, dtype=float)


def generate_synthetic_demo() -> dict:
    """Run the real template/alignment/scoring path on deterministic angle data."""
    frame_count = 72
    progress = np.linspace(0.0, 1.0, frame_count)
    squat_depth = np.sin(np.pi * progress) ** 2
    base = np.array([170, 170, 165, 165, 90, 90, 165, 165, 90, 90, 165, 165], dtype=float)
    amplitude = np.array([-78, -78, -48, -48, 12, 12, 0, 0, 8, 8, -42, -42], dtype=float)
    reference = base + squat_depth[:, None] * amplitude[None, :]

    rhythm = np.sin(2 * np.pi * progress)[:, None]
    experts = [reference - 1.5 * rhythm, reference, reference + 1.5 * rhythm]
    template = ActionTemplate("squat")
    template.build_from_features(experts)

    test = reference.copy()
    middle = (progress >= 0.25) & (progress <= 0.72)
    test[middle, 0] += 18.0
    test[middle, 2] -= 13.0
    test[middle, 8] += 9.0
    test[progress >= 0.55, 11] += 8.0

    engine = AssessmentEngine(
        template=template,
        aligner=WeightedSegmentedDTWAligner(),
        feature_extractor=_PassthroughFeatureExtractor(),
        fps=30.0,
    )
    result = engine.assess(test)
    result["llm_advice"] = LLMAdvisor().generate_advice(result, allow_remote=False)
    result["metadata"] = {
        "source": "synthetic",
        "label": "Deterministic squat angle demo",
        "disclaimer": "For UI and API verification only; this is not a real-video benchmark result.",
        "frames": frame_count,
    }
    return result
