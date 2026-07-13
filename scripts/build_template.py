"""从专家视频或预计算特征序列构建动作模板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from movescope.template import ActionTemplate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", default="squat", help="动作标识，默认为 squat")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--features-dir",
        help="包含形状为 (T, D) 的 .npy 或 .npz 特征数组的目录。",
    )
    source.add_argument(
        "--expert-dir",
        help="用于构建模板的专家视频目录。",
    )
    parser.add_argument("--output", help="输出 .npz 路径，默认写入 data/templates/<action>.npz。")
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
    raise ValueError(f"不支持该特征文件：{path}")


def main() -> None:
    args = parse_args()
    template = ActionTemplate(args.action)
    if args.features_dir:
        features_dir = Path(args.features_dir)
        files = sorted([*features_dir.glob("*.npy"), *features_dir.glob("*.npz")])
        if not files:
            raise SystemExit(f"目录中没有 .npy 或 .npz 特征文件：{features_dir}")
        sequences = [load_feature_file(path) for path in files]
        template.build_from_features(sequences)
    else:
        from movescope.features import FeatureExtractor
        from movescope.pose_extractor import PoseExtractor

        template.build(Path(args.expert_dir), PoseExtractor(), FeatureExtractor())

    output_path = template.save(args.output)

    print(f"模板已保存：{output_path}")
    print(f"输入序列数：{template.n_videos}")
    print("特征\t均值\t容差")
    for idx, (mean, tolerance) in enumerate(zip(template.mean, template.tolerance)):
        print(f"{idx}\t{mean:.3f}\t{tolerance:.3f}")


if __name__ == "__main__":
    main()
