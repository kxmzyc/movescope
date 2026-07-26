"""诊断结果的文本摘要与 JSON 落盘。"""

from __future__ import annotations

import json
from pathlib import Path


def save_diagnosis(result: dict, output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_text_summary(result: dict, top_k: int = 3) -> str:
    lines = [f"总分：{result.get('total_score', 0):.1f}/100"]
    anomalies = []
    for phase in result.get("phases", []):
        for anomaly in phase.get("anomalies", []):
            anomalies.append((phase, anomaly))

    if not anomalies:
        lines.append("主要问题：未检测到明显关节偏差。")
    else:
        anomalies.sort(key=lambda item: float(item[1].get("mean_deviation_deg", 0.0)), reverse=True)
        lines.append(f"主要问题（按偏差排序前{top_k}）：")
        for idx, (phase, anomaly) in enumerate(anomalies[:top_k], start=1):
            start, end = phase["time_range"]
            display = anomaly.get("joint_display") or anomaly.get("joint", "")
            label = phase.get("label")
            stage = f"{label} " if label else ""
            deviation = float(anomaly["mean_deviation_deg"])
            lines.append(f"{idx}. [{stage}{start:.1f}-{end:.1f}秒] {display} 平均偏差 {deviation:.1f}度")

    excluded = result.get("excluded_features") or []
    if excluded:
        names = "、".join(
            str(item.get("joint_display") or item.get("joint", "")) for item in excluded
        )
        lines.append(f"未参与评分的关节（检测数据不完整）：{names}")
    return "\n".join(lines)
