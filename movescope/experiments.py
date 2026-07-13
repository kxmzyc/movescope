"""供 MoveScope notebook 复用的实验辅助函数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from movescope.alignment import DTWAligner, WeightedSegmentedDTWAligner
from movescope.assessment import AssessmentEngine
from movescope.features import FeatureExtractor
from movescope.pose_extractor import PoseExtractor
from movescope.template import ActionTemplate


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm", ".mkv")


@dataclass(frozen=True)
class Variant:
    name: str
    use_3d: bool
    weighted: bool
    segmented: bool


ABLATION_VARIANTS = [
    Variant("baseline_2d", use_3d=False, weighted=False, segmented=False),
    Variant("ours_3d", use_3d=True, weighted=False, segmented=False),
    Variant("ours_weighted", use_3d=True, weighted=True, segmented=False),
    Variant("ours_full", use_3d=True, weighted=True, segmented=True),
]


def load_feature_sequences(directory: str | Path) -> list[np.ndarray]:
    root = Path(directory)
    if not root.exists():
        return []

    sequences = []
    for path in sorted([*root.glob("*.npy"), *root.glob("*.npz")]):
        if path.suffix.lower() == ".npy":
            sequences.append(np.load(path))
            continue
        data = np.load(path)
        key = "features" if "features" in data else data.files[0]
        sequences.append(data[key])
    return sequences


def video_files(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)


def evaluate_video(video_path: str | Path, template: ActionTemplate, variant: Variant) -> dict:
    pose = PoseExtractor().extract(str(video_path))
    coords = _coords_for_variant(pose, variant)
    aligner = _aligner_for_variant(variant)
    engine = AssessmentEngine(template, aligner, FeatureExtractor(), fps=float(pose.get("fps", 30.0)))
    return engine.assess(coords)


def run_ablation(
    template: ActionTemplate,
    good_dir: str | Path = "data/test/good_squat",
    bad_dir: str | Path = "data/test/bad_squat",
    variants: Iterable[Variant] = ABLATION_VARIANTS,
) -> list[dict]:
    rows = []
    for label, directory in (("good", good_dir), ("bad", bad_dir)):
        for video_path in video_files(directory):
            for variant in variants:
                result = evaluate_video(video_path, template, variant)
                rows.append(
                    {
                        "filename": video_path.name,
                        "label": label,
                        "variant": variant.name,
                        "total_score": float(result["total_score"]),
                    }
                )
    return rows


def summarize_ablation(rows: list[dict]) -> list[dict]:
    summary = []
    variants = sorted({row["variant"] for row in rows})
    for variant in variants:
        good = np.array([row["total_score"] for row in rows if row["variant"] == variant and row["label"] == "good"])
        bad = np.array([row["total_score"] for row in rows if row["variant"] == variant and row["label"] == "bad"])
        if len(good) == 0 or len(bad) == 0:
            continue
        summary.append(
            {
                "variant": variant,
                "good_mean": float(good.mean()),
                "bad_mean": float(bad.mean()),
                "separation": float(good.mean() - bad.mean()),
                "n_good": int(len(good)),
                "n_bad": int(len(bad)),
            }
        )
    return summary


def run_viewpoint_consistency(
    template: ActionTemplate,
    multiview_dir: str | Path = "data/test/multiview",
) -> list[dict]:
    rows = []
    variants = [
        Variant("baseline_2d", use_3d=False, weighted=False, segmented=False),
        Variant("ours_full", use_3d=True, weighted=True, segmented=True),
    ]
    for video_path in video_files(multiview_dir):
        angle = _angle_label(video_path)
        for variant in variants:
            result = evaluate_video(video_path, template, variant)
            rows.append(
                {
                    "filename": video_path.name,
                    "angle": angle,
                    "variant": variant.name,
                    "total_score": float(result["total_score"]),
                }
            )
    return rows


def viewpoint_std(rows: list[dict]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        values.setdefault(row["variant"], []).append(float(row["total_score"]))
    return {variant: float(np.std(scores)) for variant, scores in values.items() if scores}


def run_template_sensitivity_from_features(
    expert_sequences: list[np.ndarray],
    test_sequences: list[np.ndarray],
    counts: Iterable[int] = (1, 3, 5, 10),
    action_name: str = "squat",
) -> list[dict]:
    rows = []
    if not expert_sequences or not test_sequences:
        return rows

    usable_counts = [count for count in counts if 0 < count <= len(expert_sequences)]
    for count in usable_counts:
        template = ActionTemplate(action_name)
        template.build_from_features(expert_sequences[:count])
        engine = AssessmentEngine(template, DTWAligner(), _PassthroughFeatureExtractor())
        scores = [float(engine.assess(seq)["total_score"]) for seq in test_sequences]
        rows.append(
            {
                "template_count": int(count),
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "n_tests": int(len(scores)),
            }
        )
    return rows


def _coords_for_variant(pose: dict, variant: Variant) -> np.ndarray:
    if variant.use_3d:
        coords = pose.get("coords_3d")
        return coords if coords is not None else pose["coords_3d_pseudo"]

    coords_2d = np.asarray(pose["coords_2d"], dtype=float)
    zeros = np.zeros((*coords_2d.shape[:2], 1), dtype=float)
    return np.concatenate([coords_2d, zeros], axis=2)


def _aligner_for_variant(variant: Variant):
    if not variant.weighted:
        return DTWAligner()
    return _ConfiguredWeightedAligner(use_segmented=variant.segmented)


class _ConfiguredWeightedAligner(WeightedSegmentedDTWAligner):
    def __init__(self, use_segmented: bool) -> None:
        super().__init__()
        self.use_segmented = use_segmented

    def align(self, query, reference, weights=None, use_segmented=True):
        return super().align(query, reference, weights=weights, use_segmented=self.use_segmented)


class _PassthroughFeatureExtractor(FeatureExtractor):
    def extract(self, coords_3d, normalize=True):
        return np.asarray(coords_3d, dtype=float)


def _angle_label(path: Path) -> str:
    name = path.stem.lower()
    for token in ("front", "side", "diagonal", "45"):
        if token in name:
            return token
    return path.stem
