"""动作质量评估与结构化诊断。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from movescope.alignment import DTWAligner, WeightedSegmentedDTWAligner
from movescope.errors import InvalidInputError
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

    异常帧占比与总分按「唯一测试帧」统计：DTW 一对多匹配时先对
    该测试帧匹配到的全部参考帧取均值得到对齐参考曲线，再与测试
    曲线比较。待测视频中的停顿不会因路径行重复而主导分数。

    含非有限值的特征列（通常来自未被可靠检测的关节）不再整体拒绝，
    而是从对齐、权重与评分中剔除并记入 excluded_features；
    required_features 中的特征列缺失时仍拒绝评估。
    """

    template: ActionTemplate
    aligner: DTWAligner
    feature_extractor: FeatureExtractor | None = None
    fps: float = 30.0
    score_weights: Literal["aligner", "uniform"] = "aligner"
    required_features: tuple[int, ...] = ()
    timeline_max_points: int = 600

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
        if not np.isfinite(reference).all():
            raise ValueError("测试特征与参考特征只能包含有限值")
        if not np.isfinite(tolerance).all() or np.any(tolerance <= 0):
            raise ValueError("模板容差必须是正有限值")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps 必须是正有限值")

        angle_features = self._angle_features(test_features.shape[1])
        col_valid = np.isfinite(test_features).all(axis=0)
        self._reject_missing_required(col_valid, angle_features)
        valid_idx = np.flatnonzero(col_valid)
        excluded = [
            self._feature_identity(angle_features[idx])
            | {"reason": "该关节角在待测视频中的数据不完整（对应关节未被可靠检测）"}
            for idx in np.flatnonzero(~col_valid)
        ]

        test_valid = test_features[:, valid_idx]
        ref_valid = reference[:, valid_idx]
        tol_valid = tolerance[valid_idx]
        valid_features = [angle_features[idx] for idx in valid_idx]

        weights = None
        if isinstance(self.aligner, WeightedSegmentedDTWAligner):
            weights = self.aligner.compute_joint_weights(self.template)[valid_idx]

        alignment = self._align(test_valid, ref_valid, weights)
        if not alignment.path:
            return self._empty_result(excluded)

        # 按唯一测试帧聚合：一对多匹配时取匹配参考帧的均值，
        # 得到与测试序列等长的对齐参考曲线。
        path_test = np.fromiter((pair[0] for pair in alignment.path), dtype=int, count=len(alignment.path))
        path_ref = np.fromiter((pair[1] for pair in alignment.path), dtype=int, count=len(alignment.path))
        frame_starts = np.unique(path_test, return_index=True)[1]
        if len(frame_starts) != len(test_valid):
            raise ValueError("对齐路径未覆盖全部测试帧，无法生成逐帧诊断")
        counts = np.diff(np.append(frame_starts, len(path_test)))
        ref_aligned = np.add.reduceat(ref_valid[path_ref], frame_starts, axis=0) / counts[:, None]

        signed = test_valid - ref_aligned
        deviations = np.abs(signed)
        anomaly_mask = deviations > tol_valid[None, :]
        per_feature_ratio = anomaly_mask.mean(axis=0)
        per_feature_mean = deviations.mean(axis=0)
        score_w = weights if self.score_weights == "aligner" else None
        score = total_score(per_feature_ratio, score_w)
        score_weight_norm = self._score_weight_norm(score_w, len(valid_idx))

        phases = [
            self._build_phase(
                index=phase_idx,
                frame_range=frame_range,
                deviations=deviations,
                signed=signed,
                anomaly_mask=anomaly_mask,
                score_w=score_w,
                angle_features=valid_features,
                fps=fps,
            )
            for phase_idx, frame_range in enumerate(self._phase_frame_ranges(alignment, len(test_valid)))
        ]

        summary = [
            self._feature_identity(feature)
            | {
                "mean_dev": float(per_feature_mean[col]),
                "anomaly_ratio": float(per_feature_ratio[col]),
                "tolerance_deg": round(float(tol_valid[col]), 2),
                "score_weight": round(float(score_weight_norm[col]), 4),
            }
            for col, feature in enumerate(valid_features)
        ]

        return {
            "action": self.template.action_name,
            "total_score": round(score, 2),
            "segmented": alignment.segmented,
            "phases": phases,
            "per_feature_summary": summary,
            "excluded_features": excluded,
            "timeline": self._build_timeline(
                test_valid=test_valid,
                ref_aligned=ref_aligned,
                anomaly_mask=anomaly_mask,
                tol_valid=tol_valid,
                angle_features=valid_features,
                fps=fps,
            ),
        }

    def _reject_missing_required(self, col_valid: np.ndarray, angle_features: list[AngleFeature]) -> None:
        if not col_valid.any():
            raise InvalidInputError("所有关节角特征都含有非有限值，无法评估；请检查姿态提取质量。")
        missing = [idx for idx in self.required_features if idx < len(col_valid) and not col_valid[idx]]
        if missing:
            names = "、".join(dict.fromkeys(angle_features[idx].display_name for idx in missing))
            raise InvalidInputError(
                f"以下关键关节角在视频中数据不完整，无法评估：{names}。请确保对应身体部位完整入镜。"
            )

    def _align(self, test_features: np.ndarray, reference: np.ndarray, weights: np.ndarray | None) -> Alignment:
        if isinstance(self.aligner, WeightedSegmentedDTWAligner):
            return self.aligner.align_detailed(test_features, reference, weights=weights)
        path = self.aligner.align(test_features, reference)
        return Alignment(path=path, query_segments=[(0, len(test_features))], segmented=False)

    def _phase_frame_ranges(self, alignment: Alignment, n_frames: int) -> list[tuple[int, int]]:
        """把待测序列分段裁剪到 [0, n_frames) 的连续帧区间。"""
        ranges = []
        for q_start, q_end in alignment.query_segments:
            start = max(0, int(q_start))
            end = min(n_frames, int(q_end))
            if end > start:
                ranges.append((start, end))
        return ranges or [(0, n_frames)]

    def _build_phase(
        self,
        index: int,
        frame_range: tuple[int, int],
        deviations: np.ndarray,
        signed: np.ndarray,
        anomaly_mask: np.ndarray,
        score_w: np.ndarray | None,
        angle_features: list[AngleFeature],
        fps: float,
    ) -> dict:
        start, end = frame_range
        phase_devs = deviations[start:end]
        phase_signed = signed[start:end]
        phase_mask = anomaly_mask[start:end]

        anomalies: list[dict[str, Any]] = []
        for col, feature in enumerate(angle_features):
            rows = np.flatnonzero(phase_mask[:, col])
            if len(rows) == 0:
                continue
            peak_row = rows[int(np.argmax(phase_devs[rows, col]))]
            anomalies.append(
                self._feature_identity(feature)
                | {
                    "direction": "positive" if phase_signed[rows, col].mean() >= 0 else "negative",
                    "mean_deviation_deg": round(float(phase_devs[rows, col].mean()), 2),
                    "peak_deviation_deg": round(float(phase_devs[peak_row, col]), 2),
                    "peak_time_sec": round(float((start + peak_row) / fps), 2),
                    "anomaly_ratio": round(float(len(rows) / (end - start)), 3),
                }
            )

        anomalies.sort(key=lambda item: float(item["mean_deviation_deg"]), reverse=True)
        return {
            "name": f"phase_{index}",
            "index": index,
            "time_range": [round(start / fps, 2), round((end - 1) / fps, 2)],
            "phase_score": round(total_score(phase_mask.mean(axis=0), score_w), 2),
            "anomalies": anomalies,
        }

    def _build_timeline(
        self,
        test_valid: np.ndarray,
        ref_aligned: np.ndarray,
        anomaly_mask: np.ndarray,
        tol_valid: np.ndarray,
        angle_features: list[AngleFeature],
        fps: float,
    ) -> dict:
        n_frames = len(test_valid)
        stride = max(1, math.ceil(n_frames / max(1, self.timeline_max_points)))
        sample = np.arange(0, n_frames, stride)
        series = [
            self._feature_identity(feature)
            | {
                "tolerance_deg": round(float(tol_valid[col]), 2),
                "test_deg": [round(float(v), 2) for v in test_valid[sample, col]],
                "reference_deg": [round(float(v), 2) for v in ref_aligned[sample, col]],
                "anomaly": [bool(v) for v in anomaly_mask[sample, col]],
            }
            for col, feature in enumerate(angle_features)
        ]
        return {
            "fps": round(float(fps), 3),
            "frame_count": int(n_frames),
            "frame_stride": int(stride),
            "time_sec": [round(float(idx / fps), 3) for idx in sample],
            "series": series,
        }

    @staticmethod
    def _score_weight_norm(score_w: np.ndarray | None, n_features: int) -> np.ndarray:
        """评分实际使用的归一化权重（等权时为 1/D）。"""
        if score_w is None:
            return np.full(n_features, 1.0 / n_features)
        return np.asarray(score_w, dtype=float) / float(np.sum(score_w))

    @staticmethod
    def _feature_identity(feature: AngleFeature) -> dict[str, Any]:
        return {
            "feature_index": feature.index,
            "joint": feature.joint,
            "joint_display": feature.display_name,
            "parent": feature.parent,
            "child": feature.child,
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

    def _empty_result(self, excluded: list[dict[str, Any]] | None = None) -> dict:
        return {
            "action": self.template.action_name,
            "total_score": 0.0,
            "segmented": False,
            "phases": [],
            "per_feature_summary": [],
            "excluded_features": excluded or [],
            "timeline": None,
        }
