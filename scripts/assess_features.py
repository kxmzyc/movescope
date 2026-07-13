"""Assess a precomputed feature sequence against an action template."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from movescope.alignment import WeightedSegmentedDTWAligner
from movescope.assessment import AssessmentEngine, generate_text_summary, save_diagnosis
from movescope.features import FeatureExtractor
from movescope.template import ActionTemplate


class PassthroughFeatureExtractor(FeatureExtractor):
    def extract(self, coords_3d, normalize=True):
        return np.asarray(coords_3d, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", default="squat")
    parser.add_argument("--features", required=True, help="Feature .npy or .npz file with shape (T, D).")
    parser.add_argument("--template", help="Template .npz path. Defaults to data/templates/<action>.npz")
    parser.add_argument("--output", help="Optional diagnosis JSON output path.")
    return parser.parse_args()


def load_features(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    data = np.load(path)
    return data["features"] if "features" in data else data[data.files[0]]


def main() -> None:
    args = parse_args()
    template = ActionTemplate.load(args.action, args.template)
    features = load_features(Path(args.features))
    engine = AssessmentEngine(template, WeightedSegmentedDTWAligner(), PassthroughFeatureExtractor())
    result = engine.assess(features)

    if args.output:
        save_diagnosis(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()
    print(generate_text_summary(result))


if __name__ == "__main__":
    main()
