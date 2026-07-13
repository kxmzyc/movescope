"""MoveScope 的 MediaPipe 姿态提取入门示例。"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


LEFT_HIP = 23
LEFT_KNEE = 25
LEFT_ANKLE = 27


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", help="输入视频路径；不传时使用 0 号摄像头。")
    parser.add_argument(
        "--output",
        default="data/test/hello_world_output.mp4",
        help="输出视频路径。",
    )
    return parser.parse_args()


def knee_angle_deg(landmarks: list, image_width: int, image_height: int) -> float | None:
    points = []
    for index in (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE):
        landmark = landmarks[index]
        if landmark.visibility < 0.3:
            return None
        points.append(np.array([landmark.x * image_width, landmark.y * image_height], dtype=np.float32))

    hip, knee, ankle = points
    v1 = hip - knee
    v2 = ankle - knee
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return None
    cos_angle = float(np.dot(v1, v2) / denom)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def create_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps or 30.0, (width, height))


def main() -> None:
    args = parse_args()
    source = args.video if args.video else 0
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"无法打开视频源：{source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = create_writer(Path(args.output), fps, width, height)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    frame_count = 0
    angles: list[float] = []
    start = time.perf_counter()

    with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            angle = None
            if result.pose_landmarks:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                angle = knee_angle_deg(result.pose_landmarks.landmark, width, height)
                if angle is not None:
                    angles.append(angle)

            text = "--" if angle is None else f"{angle:.1f}"
            cv2.putText(frame, text, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            writer.write(frame)
            frame_count += 1

    capture.release()
    writer.release()

    elapsed = time.perf_counter() - start
    if angles:
        print(
            f"已在 {elapsed:.2f} 秒内处理 {frame_count} 帧。"
            f"左膝平均角度={np.mean(angles):.1f}，"
            f"最小值={np.min(angles):.1f}，最大值={np.max(angles):.1f}。"
            f"输出：{args.output}"
        )
    else:
        print(f"已在 {elapsed:.2f} 秒内处理 {frame_count} 帧，未检测到可靠的左膝角度。")


if __name__ == "__main__":
    main()
