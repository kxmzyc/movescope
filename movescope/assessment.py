"""动作质量评估与结构化诊断。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from movescope.features import FeatureExtractor, JOINT_DISPLAY_NAMES, JOINT_TRIPLETS


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


@dataclass
class AssessmentEngine:
    template: object
    aligner: object
    feature_extractor: FeatureExtractor
    fps: float = 30.0

    def assess(self, test_coords_3d: np.ndarray) -> dict:
        test_features = self.feature_extractor.extract(test_coords_3d, normalize=False)
        reference = np.asarray(self.template.representative_seq, dtype=float)
        tolerance = np.asarray(self.template.tolerance, dtype=float)
        if test_features.ndim != 2 or reference.ndim != 2 or len(test_features) == 0 or len(reference) == 0:
            raise ValueError("测试特征与参考特征必须是非空二维数组")
        if test_features.shape[1] != reference.shape[1]:
            raise ValueError("测试特征与参考特征的维度必须一致")
        if tolerance.shape != (test_features.shape[1],):
            raise ValueError("模板容差维度必须与特征维度一致")
        if not np.isfinite(test_features).all() or not np.isfinite(reference).all():
            raise ValueError("测试特征与参考特征只能包含有限值")
        if not np.isfinite(tolerance).all() or np.any(tolerance <= 0):
            raise ValueError("模板容差必须是正有限值")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps 必须是正有限值")

        weights = None
        if hasattr(self.aligner, "compute_joint_weights"):
            weights = self.aligner.compute_joint_weights(self.template)

        try:
            path = self.aligner.align(test_features, reference, weights=weights)
        except TypeError:
            path = self.aligner.align(test_features, reference)

        if not path:
            return self._empty_result()

        deviations = np.zeros((len(path), test_features.shape[1]), dtype=float)
        signed = np.zeros_like(deviations)
        test_indices = np.zeros(len(path), dtype=int)
        for row, (test_idx, ref_idx) in enumerate(path):
            diff = test_features[test_idx] - reference[ref_idx]
            signed[row] = diff
            deviations[row] = np.abs(diff)
            test_indices[row] = test_idx

        anomaly_mask = deviations > tolerance[None, :]
        per_joint_ratio = anomaly_mask.mean(axis=0)
        per_joint_mean = deviations.mean(axis=0)
        total_score = clamp(100.0 - float(np.average(per_joint_ratio, weights=weights)) * 100.0 if weights is not None else 100.0 - float(per_joint_ratio.mean()) * 100.0)

        phase = self._build_phase(test_indices, deviations, signed, anomaly_mask, total_score)
        summary = {
            self._joint_label(idx): {
                "mean_dev": float(per_joint_mean[idx]),
                "anomaly_ratio": float(per_joint_ratio[idx]),
            }
            for idx in range(len(per_joint_ratio))
        }

        return {
            "action": self.template.action_name,
            "total_score": round(total_score, 2),
            "phases": [phase],
            "per_joint_summary": summary,
        }

    def _empty_result(self) -> dict:
        return {
            "action": self.template.action_name,
            "total_score": 0.0,
            "phases": [],
            "per_joint_summary": {},
        }

    def _build_phase(
        self,
        test_indices: np.ndarray,
        deviations: np.ndarray,
        signed: np.ndarray,
        anomaly_mask: np.ndarray,
        score: float,
    ) -> dict:
        anomalies = []
        for joint_idx in range(anomaly_mask.shape[1]):
            rows = np.where(anomaly_mask[:, joint_idx])[0]
            if len(rows) == 0:
                continue
            peak_row = rows[int(np.argmax(deviations[rows, joint_idx]))]
            anomalies.append(
                {
                    "joint_name": self._joint_label(joint_idx),
                    "joint_idx": int(joint_idx),
                    "direction": "positive" if signed[rows, joint_idx].mean() >= 0 else "negative",
                    "mean_deviation_deg": round(float(deviations[rows, joint_idx].mean()), 2),
                    "peak_deviation_deg": round(float(deviations[peak_row, joint_idx]), 2),
                    "peak_time_sec": round(float(test_indices[peak_row] / self.fps), 2),
                    "anomaly_ratio": round(float(len(rows) / len(test_indices)), 3),
                }
            )

        anomalies.sort(key=lambda item: item["mean_deviation_deg"], reverse=True)
        return {
            "name": "phase_0",
            "time_range": [
                round(float(test_indices.min() / self.fps), 2),
                round(float(test_indices.max() / self.fps), 2),
            ],
            "phase_score": round(score, 2),
            "anomalies": anomalies,
        }

    @staticmethod
    def _joint_label(idx: int) -> str:
        parent, joint, child = JOINT_TRIPLETS[idx]
        return f"{joint}:{parent}-{child}"


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
        return "\n".join(lines)

    lines.append(f"主要问题（按偏差排序前{top_k}）：")
    for idx, (phase, anomaly) in enumerate(anomalies[:top_k], start=1):
        start, end = phase["time_range"]
        lines.append(
            f"{idx}. [{start:.1f}-{end:.1f}秒] "
            f"{JOINT_DISPLAY_NAMES.get(anomaly['joint_name'].split(':', 1)[0], anomaly['joint_name'])} "
            f"平均偏差 {anomaly['mean_deviation_deg']:.1f}度"
        )
    return "\n".join(lines)
