import numpy as np

from movescope.features import FeatureExtractor, JOINT_NAMES


def test_known_angle():
    coords = np.zeros((1, len(JOINT_NAMES), 3), dtype=float)
    idx = {name: i for i, name in enumerate(JOINT_NAMES)}
    coords[0, idx["left_hip"]] = [0, 1, 0]
    coords[0, idx["left_knee"]] = [0, 0, 0]
    coords[0, idx["left_ankle"]] = [1, 0, 0]

    angles = FeatureExtractor().compute_angles(coords)

    assert abs(angles[0, 0] - 90.0) < 1.0


def test_normalize_stats():
    angles = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])

    normalized = FeatureExtractor().normalize(angles)

    assert np.allclose(normalized.mean(axis=0), 0.0)
    assert np.allclose(normalized.std(axis=0), 1.0)


def test_degenerate_bones_are_marked_invalid():
    coords = np.zeros((2, len(JOINT_NAMES), 3), dtype=float)

    angles = FeatureExtractor().compute_angles(coords)

    assert np.isnan(angles).all()
