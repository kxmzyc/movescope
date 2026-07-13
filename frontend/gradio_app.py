"""MoveScope 深蹲动作评估 Gradio 调试界面。"""

from __future__ import annotations

import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from movescope.alignment import WeightedSegmentedDTWAligner
from movescope.assessment import AssessmentEngine, generate_text_summary
from movescope.features import FeatureExtractor, JOINT_DISPLAY_NAMES, JOINT_NAMES
from movescope.llm_advisor import LLMAdvisor
from movescope.pose_extractor import PoseExtractor
from movescope.template import ActionTemplate


SKELETON_EDGES = [
    ("pelvis", "left_hip"),
    ("pelvis", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("neck", "head"),
]


def assess_video(video_path: str | None, action: str = "squat") -> tuple[str | None, float, Any, str]:
    if not video_path:
        return None, 0.0, _empty_bar_data(), "请先上传深蹲视频。"

    try:
        template = ActionTemplate.load(action)
    except FileNotFoundError:
        return (
            None,
            0.0,
            _empty_bar_data(),
            f"未找到动作“{action}”的模板，请先运行 scripts/build_template.py。",
        )

    try:
        pose = PoseExtractor().extract(video_path)
        coords_3d = pose.get("coords_3d")
        if coords_3d is None:
            coords_3d = pose["coords_3d_pseudo"]

        engine = AssessmentEngine(
            template=template,
            aligner=WeightedSegmentedDTWAligner(),
            feature_extractor=FeatureExtractor(),
            fps=float(pose.get("fps", 30.0)),
        )
        result = engine.assess(coords_3d)
        advice = LLMAdvisor().generate_advice(result)
        overlay_path = render_overlay(video_path, pose, result)
        text = f"{generate_text_summary(result)}\n\n纠错建议：\n{advice}"
        return overlay_path, float(result["total_score"]), _bar_data(result), text
    except Exception as exc:
        return None, 0.0, _empty_bar_data(), f"评估失败：{exc}"


def render_overlay(video_path: str, pose: dict[str, Any], result: dict[str, Any]) -> str:
    import cv2

    output_path = Path(tempfile.gettempdir()) / f"movescope_overlay_{uuid.uuid4().hex}.mp4"

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"无法打开视频：{video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or pose.get("fps", 30.0) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("无法创建骨架叠加视频。")

    coords_2d = pose["coords_2d"]
    confidence = pose["confidence"]
    highlighted = _highlighted_joints(result)
    joint_index = {name: idx for idx, name in enumerate(JOINT_NAMES)}
    score = float(result.get("total_score", 0.0))

    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_idx < len(coords_2d):
            _draw_skeleton(frame, coords_2d[frame_idx], confidence[frame_idx], joint_index, highlighted)

        cv2.putText(
            frame,
            f"{score:.1f} / 100",
            (24, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (35, 35, 235),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_idx += 1

    capture.release()
    writer.release()
    return str(output_path)


def _draw_skeleton(frame: Any, coords: Any, confidence: Any, joint_index: dict[str, int], highlighted: set[str]) -> None:
    import cv2

    height, width = frame.shape[:2]
    points: dict[str, tuple[int, int]] = {}
    for name, idx in joint_index.items():
        if idx >= len(coords) or confidence[idx] < 0.2:
            continue
        x = int(float(coords[idx][0]) * width)
        y = int(float(coords[idx][1]) * height)
        points[name] = (x, y)

    for start, end in SKELETON_EDGES:
        if start in points and end in points:
            cv2.line(frame, points[start], points[end], (80, 170, 80), 2, cv2.LINE_AA)

    for name, point in points.items():
        if name in highlighted:
            cv2.circle(frame, point, 8, (35, 35, 235), -1, cv2.LINE_AA)
            cv2.circle(frame, point, 11, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, point, 5, (55, 190, 80), -1, cv2.LINE_AA)


def _highlighted_joints(result: dict[str, Any], limit: int = 3) -> set[str]:
    anomalies = []
    for phase in result.get("phases", []):
        anomalies.extend(phase.get("anomalies", []))
    anomalies.sort(key=lambda item: float(item.get("mean_deviation_deg", 0.0)), reverse=True)
    names = set()
    for item in anomalies[:limit]:
        joint = str(item.get("joint_name", "")).split(":", 1)[0]
        if joint in JOINT_NAMES:
            names.add(joint)
    return names


def _bar_data(result: dict[str, Any]) -> Any:
    import pandas as pd

    rows = []
    for joint, values in result.get("per_joint_summary", {}).items():
        rows.append(
            {
                "关节": JOINT_DISPLAY_NAMES.get(joint.split(":", 1)[0], joint.split(":", 1)[0]),
                "平均偏差（度）": round(float(values.get("mean_dev", 0.0)), 2),
                "异常帧占比（%）": round(float(values.get("anomaly_ratio", 0.0)) * 100.0, 1),
            }
        )
    rows.sort(key=lambda row: row["平均偏差（度）"], reverse=True)
    return pd.DataFrame(rows or [{"关节": "暂无", "平均偏差（度）": 0.0, "异常帧占比（%）": 0.0}])


def _empty_bar_data() -> Any:
    import pandas as pd

    return pd.DataFrame([{"关节": "暂无", "平均偏差（度）": 0.0, "异常帧占比（%）": 0.0}])


def create_demo():
    import gradio as gr

    with gr.Blocks(title="MoveScope", theme=gr.themes.Soft(primary_hue="red", neutral_hue="stone")) as demo:
        gr.Markdown("# MoveScope 深蹲动作评估")
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=280):
                video = gr.Video(label="上传待测视频")
                action = gr.Dropdown(label="动作类型", choices=[("深蹲", "squat")], value="squat")
                run = gr.Button("开始评估", variant="primary")
            with gr.Column(scale=1, min_width=320):
                output_video = gr.Video(label="骨架可视化")
            with gr.Column(scale=1, min_width=320):
                score = gr.Number(label="总分", precision=1)
                deviations = gr.BarPlot(
                    label="各关节平均偏差",
                    x="关节",
                    y="平均偏差（度）",
                    tooltip=["关节", "平均偏差（度）", "异常帧占比（%）"],
                    y_title="偏差（度）",
                )
                summary = gr.Textbox(label="诊断摘要与纠错建议", lines=10)

        run.click(assess_video, inputs=[video, action], outputs=[output_video, score, deviations, summary])

    return demo


if __name__ == "__main__":
    create_demo().launch(server_port=7860, share=False, prevent_thread_lock=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
