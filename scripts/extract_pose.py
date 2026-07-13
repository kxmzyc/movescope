"""从视频中提取人体姿态数组。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from movescope.pose_extractor import PoseExtractor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="输入视频路径。")
    parser.add_argument("--output", required=True, help="姿态数组输出路径。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    result = PoseExtractor().extract(args.video)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        fps=result["fps"],
        n_frames=result["n_frames"],
        joint_names=np.asarray(result["joint_names"]),
        coords_2d=result["coords_2d"],
        confidence=result["confidence"],
        coords_3d_pseudo=result["coords_3d_pseudo"],
    )
    elapsed = time.perf_counter() - start
    print(
        f"已处理 {result['n_frames']} 帧，"
        f"跳过 {result['skipped_frames']} 帧，"
        f"耗时 {elapsed:.2f} 秒，输出：{output}"
    )


if __name__ == "__main__":
    main()
