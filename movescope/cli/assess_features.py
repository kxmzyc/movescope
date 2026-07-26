"""使用动作模板评估预计算特征序列。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from movescope.alignment import WeightedSegmentedDTWAligner
from movescope.assessment import AssessmentEngine
from movescope.cli._console import configure_utf8_stdio
from movescope.reporting import generate_text_summary, save_diagnosis
from movescope.reps import assess_features_by_rep
from movescope.template import ActionTemplate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", default="squat", help="动作标识，默认为 squat")
    parser.add_argument("--features", required=True, help="形状为 (T, D) 的 .npy 或 .npz 特征文件。")
    parser.add_argument("--template", help="模板 .npz 路径，默认读取 data/templates/<action>.npz。")
    parser.add_argument("--output", help="可选的诊断 JSON 输出路径。")
    return parser.parse_args()


def load_features(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    data = np.load(path)
    return data["features"] if "features" in data else data[data.files[0]]


def main() -> None:
    configure_utf8_stdio()
    args = parse_args()
    template = ActionTemplate.load(args.action, args.template)
    features = load_features(Path(args.features))
    engine = AssessmentEngine(template, WeightedSegmentedDTWAligner())
    result = assess_features_by_rep(engine, features)

    if args.output:
        save_diagnosis(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()
    print(generate_text_summary(result))


if __name__ == "__main__":
    main()
