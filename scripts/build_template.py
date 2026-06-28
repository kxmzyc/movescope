"""Build an action template from videos or precomputed feature sequences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from movescope.template import ActionTemplate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", default="squat")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--features-dir",
        help="Directory containing .npy or .npz feature arrays with shape (T, D).",
    )
    source.add_argument(
        "--expert-dir",
        help="Directory containing expert videos used to build the template.",
    )
    parser.add_argument("--output", help="Output .npz path. Defaults to data/templates/<action>.npz")
    return parser.parse_args()


def load_feature_file(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if "features" in data:
            return data["features"]
        first_key = data.files[0]
        return data[first_key]
    raise ValueError(f"Unsupported feature file: {path}")


def main() -> None:
    args = parse_args()
    template = ActionTemplate(args.action)
    if args.features_dir:
        features_dir = Path(args.features_dir)
        files = sorted([*features_dir.glob("*.npy"), *features_dir.glob("*.npz")])
        if not files:
            raise SystemExit(f"No .npy or .npz feature files found in {features_dir}")
        sequences = [load_feature_file(path) for path in files]
        template.build_from_features(sequences)
    else:
        from movescope.features import FeatureExtractor
        from movescope.pose_extractor import PoseExtractor

        template.build(Path(args.expert_dir), PoseExtractor(), FeatureExtractor())

    output_path = template.save(args.output)

    print(f"Saved template: {output_path}")
    print(f"Inputs: {template.n_videos}")
    print("Feature\tmean\ttolerance")
    for idx, (mean, tolerance) in enumerate(zip(template.mean, template.tolerance)):
        print(f"{idx}\t{mean:.3f}\t{tolerance:.3f}")


if __name__ == "__main__":
    main()
