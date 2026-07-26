"""从视频中提取人体姿态数组。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from movescope.cli._console import configure_utf8_stdio
from movescope.pose_extractor import PoseExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="输入视频路径。")
    parser.add_argument("--output", required=True, help="姿态数组输出路径。")
    return parser.parse_args()


def main() -> None:
    configure_utf8_stdio()
    args = parse_args()
    start = time.perf_counter()
    result = PoseExtractor().extract(args.video)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result.to_npz_dict())
    elapsed = time.perf_counter() - start
    print(
        f"已处理 {result.n_frames} 帧，"
        f"跳过 {result.skipped_frames} 帧，"
        f"耗时 {elapsed:.2f} 秒，输出：{output}"
    )


if __name__ == "__main__":
    main()
