"""MediaPipe-based pose extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from movescope.features import JOINT_NAMES


MEDIAPIPE_TO_COCO = {
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "head": 0,
    "neck": None,
    "pelvis": None,
    "left_eye": 2,
    "right_eye": 5,
}

MOTIONBERT_CHECKPOINT = Path("lib/MotionBERT/checkpoint/motionbert_lite.bin")


@dataclass
class PoseExtractor:
    min_confidence: float = 0.3
    motionbert_checkpoint: Path = MOTIONBERT_CHECKPOINT

    def extract(self, video_path: str) -> dict:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "Pose extraction requires opencv-python and mediapipe. "
                "Use a Python 3.10/3.11 environment and run: pip install -r requirements.txt"
            ) from exc

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        coords_2d = []
        coords_3d_pseudo = []
        confidence = []
        skipped_frames = 0

        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not result.pose_landmarks:
                    coords_2d.append(np.full((len(JOINT_NAMES), 2), np.nan))
                    coords_3d_pseudo.append(np.full((len(JOINT_NAMES), 3), np.nan))
                    confidence.append(np.zeros(len(JOINT_NAMES)))
                    skipped_frames += 1
                    continue

                landmarks = result.pose_landmarks.landmark
                world_landmarks = result.pose_world_landmarks.landmark if result.pose_world_landmarks else landmarks
                frame_2d, frame_3d, frame_conf = self._map_landmarks(landmarks, world_landmarks, width, height)
                coords_2d.append(frame_2d)
                coords_3d_pseudo.append(frame_3d)
                confidence.append(frame_conf)

        capture.release()

        coords_2d_arr = self._interpolate_low_confidence(np.asarray(coords_2d, dtype=float), np.asarray(confidence))
        coords_3d_arr = self._interpolate_low_confidence(np.asarray(coords_3d_pseudo, dtype=float), np.asarray(confidence))
        conf_arr = np.asarray(confidence, dtype=float)
        coords_3d = None
        if self.motionbert_checkpoint.exists():
            try:
                coords_3d = self.lift_to_3d(coords_2d_arr, fps)
            except Exception as exc:
                print(f"WARNING: MotionBERT lifting failed; falling back to pseudo-3D landmarks: {exc}")
        else:
            print(f"WARNING: MotionBERT checkpoint not found at {self.motionbert_checkpoint}; coords_3d=None")

        return {
            "fps": fps,
            "n_frames": int(len(coords_2d_arr)),
            "joint_names": JOINT_NAMES,
            "coords_2d": coords_2d_arr,
            "confidence": conf_arr,
            "coords_3d": coords_3d,
            "coords_3d_pseudo": coords_3d_arr,
            "skipped_frames": skipped_frames,
        }

    def lift_to_3d(self, coords_2d: np.ndarray, fps: float) -> np.ndarray:
        coords = np.asarray(coords_2d, dtype=float)
        if coords.ndim != 3 or coords.shape[1:] != (len(JOINT_NAMES), 2):
            raise ValueError(f"coords_2d must have shape (T, {len(JOINT_NAMES)}, 2)")
        if not self.motionbert_checkpoint.exists():
            raise FileNotFoundError(
                f"MotionBERT checkpoint not found: {self.motionbert_checkpoint}. "
                "See docs/motionbert_setup.md for setup steps."
            )

        # The upstream MotionBERT package has several model entry points across
        # releases. Until the local checkpoint and repository are installed, keep
        # the integration boundary explicit instead of silently inventing 3D data.
        raise NotImplementedError(
            "MotionBERT checkpoint detected, but the local MotionBERT inference adapter "
            "has not been configured for this checkout."
        )

    def _map_landmarks(self, landmarks, world_landmarks, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coords_2d = np.zeros((len(JOINT_NAMES), 2), dtype=float)
        coords_3d = np.zeros((len(JOINT_NAMES), 3), dtype=float)
        confidence = np.zeros(len(JOINT_NAMES), dtype=float)

        for idx, name in enumerate(JOINT_NAMES):
            mp_index = MEDIAPIPE_TO_COCO[name]
            if name == "pelvis":
                left, right = landmarks[23], landmarks[24]
                left_world, right_world = world_landmarks[23], world_landmarks[24]
                coords_2d[idx] = [(left.x + right.x) / 2.0, (left.y + right.y) / 2.0]
                coords_3d[idx] = [
                    (left_world.x + right_world.x) / 2.0,
                    (left_world.y + right_world.y) / 2.0,
                    (left_world.z + right_world.z) / 2.0,
                ]
                confidence[idx] = min(left.visibility, right.visibility)
            elif name == "neck":
                left, right = landmarks[11], landmarks[12]
                left_world, right_world = world_landmarks[11], world_landmarks[12]
                coords_2d[idx] = [(left.x + right.x) / 2.0, (left.y + right.y) / 2.0]
                coords_3d[idx] = [
                    (left_world.x + right_world.x) / 2.0,
                    (left_world.y + right_world.y) / 2.0,
                    (left_world.z + right_world.z) / 2.0,
                ]
                confidence[idx] = min(left.visibility, right.visibility)
            else:
                point = landmarks[mp_index]
                world_point = world_landmarks[mp_index]
                coords_2d[idx] = [point.x, point.y]
                coords_3d[idx] = [world_point.x, world_point.y, world_point.z]
                confidence[idx] = point.visibility

        return coords_2d, coords_3d, confidence

    def _interpolate_low_confidence(self, coords: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        if coords.size == 0:
            return coords

        filled = coords.copy()
        for joint_idx in range(coords.shape[1]):
            good = confidence[:, joint_idx] >= self.min_confidence
            if not good.any():
                print(f"WARNING: no reliable frames for joint {JOINT_NAMES[joint_idx]}")
                filled[:, joint_idx, :] = np.nan
                continue
            good_indices = np.where(good)[0]
            for dim in range(coords.shape[2]):
                values = filled[:, joint_idx, dim]
                filled[:, joint_idx, dim] = np.interp(np.arange(len(values)), good_indices, values[good_indices])
        return filled
