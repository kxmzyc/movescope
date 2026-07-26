"""动作质量评估与结构化诊断。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from movescope.alignment import DTWAligner, WeightedSegmentedDTWAligner
from movescope.features import AngleFeature, FeatureExtractor
from movescope.scoring import total_score
from movescope.template import ActionTemplate
from movescope.types import Alignment, PoseResult


@dataclass
class AssessmentEngine:
    """将待测序列与专家模板对齐并产出结构化诊断。

    三个入口按输入类型选择：
    - assess_features：调用方已持有 (T, D) 特征序列；
    - assess_coords：持有 (T, J, 3) 骨架坐标，需要 feature_extractor；
    - assess_pose：持有 PoseResult，自动取最优坐标并采用视频实际帧率。

    score_weights 控制评分权重来源：
    - "aligner"（默认，与历史行为一致）：复用 DTW 的 1/std 特征权重，
      低方差关节同时主导对齐与分数；
    - "uniform"：对齐仍用特征权重，评分等权。
    """

    template: ActionTemplate
    aligner: DTWAligner
    feature_extractor: FeatureExtractor | None = None
    fps: float = 30.0
    score_weights: Literal["aligner", "uniform"] = "aligner"

    def assess_pose(self, pose: PoseResult) -> dict:
        return self.assess_coords(pose.best_coords_3d, fps=pose.fps)

    def assess_coords(self, coords_3d: np.ndarray, fps: float | None = None) -> dict:
        if self.feature_extractor is None:
            raise ValueError("assess_coords 需要 feature_extractor；如已持有特征请使用 assess_features")
        features = self.feature_extractor.extract(coords_3d, normalize=False)
        return self.assess_features(features, fps=fps)

    def assess_features(self, test_features: np.ndarray, fps: float | None = None) -> dict:
        fps = self.fps if fps is None else fps
        test_features = np.asarray(test_features, dtype=float)
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
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps 必须是正有限值")

        weights = None
        if isinstance(self.aligner, WeightedSegmentedDTWAligner):
            weights = self.aligner.compute_joint_weights(self.template)

        alignment = self._align(test_features, reference, weights)
        if not alignment.path:
            return self._empty_result()

        path = alignment.path
        deviations = np.zeros((len(path), test_features.shape[1]), dtype=float)
        signed = np.zeros_like(deviations)
        test_indices = np.zeros(len(path), dtype=int)
        for row, (test_idx, ref_idx) in enumerate(path):
            diff = test_features[test_idx] - reference[ref_idx]
            signed[row] = diff
            deviations[row] = np.abs(diff)
            test_indices[row] = test_idx

        anomaly_mask = deviations > tolerance[None, :]
        per_feature_ratio = anomaly_mask.mean(axis=0)
        per_feature_mean = deviations.mean(axis=0)
        score_w = weights if self.score_weights == "aligner" else None
        score = total_score(per_feature_ratio, score_w)

        angle_features = self._angle_features(test_features.shape[1])
        phases = [
            self._build_phase(
                index=phase_idx,
                row_slice=row_slice,
                test_indices=test_indices,
                deviations=deviations,
                signed=signed,
                anomaly_mask=anomaly_mask,
                score_w=score_w,
                angle_features=angle_features,
                fps=fps,
            )
            for phase_idx, row_slice in enumerate(self._phase_row_slices(alignment, test_indices))
        ]

        summary = [
            {
                "feature_index": feature.index,
                "joint": feature.joint,
                "joint_display": feature.display_name,
                "parent": feature.parent,
                "child": feature.child,
                "mean_dev": float(per_feature_mean[feature.index]),
                "anomaly_ratio": float(per_feature_ratio[feature.index]),
            }
            for feature in angle_features
        ]

        return {
            "action": self.template.action_name,
            "total_score": round(score, 2),
            "segmented": alignment.segmented,
            "phases": phases,
            "per_feature_summary": summary,
        }

    def _align(self, test_features: np.ndarray, reference: np.ndarray, weights: np.ndarray | None) -> Alignment:
        if isinstance(self.aligner, WeightedSegmentedDTWAligner):
            return self.aligner.align_detailed(test_features, reference, weights=weights)
        path = self.aligner.align(test_features, reference)
        return Alignment(path=path, query_segments=[(0, len(test_features))], segmented=False)

    def _phase_row_slices(self, alignment: Alignment, test_indices: np.ndarray) -> list[slice]:
        """把路径行按待测序列分段切成连续区间（路径按 test_idx 单调不减）。"""
        slices = []
        for q_start, q_end in alignment.query_segments:
            row_start = int(np.searchsorted(test_indices, q_start, side="left"))
            row_end = int(np.searchsorted(test_indices, q_end, side="left"))
            if row_end > row_start:
                slices.append(slice(row_start, row_end))
        return slices or [slice(0, len(test_indices))]

    def _build_phase(
        self,
        index: int,
        row_slice: slice,
        test_indices: np.ndarray,
        deviations: np.ndarray,
        signed: np.ndarray,
        anomaly_mask: np.ndarray,
        score_w: np.ndarray | None,
        angle_features: list[AngleFeature],
        fps: float,
    ) -> dict:
        phase_indices = test_indices[row_slice]
        phase_devs = deviations[row_slice]
        phase_signed = signed[row_slice]
        phase_mask = anomaly_mask[row_slice]

        anomalies: list[dict[str, Any]] = []
        for feature in angle_features:
            rows = np.where(phase_mask[:, feature.index])[0]
            if len(rows) == 0:
                continue
            peak_row = rows[int(np.argmax(phase_devs[rows, feature.index]))]
            anomalies.append(
                {
                    "feature_index": feature.index,
                    "joint": feature.joint,
                    "joint_display": feature.display_name,
                    "parent": feature.parent,
                    "child": feature.child,
                    "direction": "positive" if phase_signed[rows, feature.index].mean() >= 0 else "negative",
                    "mean_deviation_deg": round(float(phase_devs[rows, feature.index].mean()), 2),
                    "peak_deviation_deg": round(float(phase_devs[peak_row, feature.index]), 2),
                    "peak_time_sec": round(float(phase_indices[peak_row] / fps), 2),
                    "anomaly_ratio": round(float(len(rows) / len(phase_indices)), 3),
                }
            )

        anomalies.sort(key=lambda item: float(item["mean_deviation_deg"]), reverse=True)
        return {
            "name": f"phase_{index}",
            "index": index,
            "time_range": [
                round(float(phase_indices.min() / fps), 2),
                round(float(phase_indices.max() / fps), 2),
            ],
            "phase_score": round(total_score(phase_mask.mean(axis=0), score_w), 2),
            "anomalies": anomalies,
        }

    def _angle_features(self, feature_dim: int) -> list[AngleFeature]:
        """特征标签：优先取注入的 extractor；无 extractor 时用默认三元组；
        维度对不上时退化为通用编号标签，而不是越界崩溃。"""
        extractor = self.feature_extractor or FeatureExtractor()
        features = extractor.features
        if len(features) == feature_dim:
            return features
        return [
            AngleFeature(index=idx, joint=f"feature_{idx}", parent="", child="", display_name=f"特征{idx}")
            for idx in range(feature_dim)
        ]

    def _empty_result(self) -> dict:
        return {
            "action": self.template.action_name,
            "total_score": 0.0,
            "segmented": False,
            "phases": [],
            "per_feature_summary": [],
        }
