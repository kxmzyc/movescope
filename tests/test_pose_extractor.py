import numpy as np

from movescope.pose_extractor import PoseExtractor


def test_interpolate_low_confidence():
    extractor = PoseExtractor(min_confidence=0.3)
    coords = np.zeros((3, 1, 2), dtype=float)
    coords[0, 0] = [0.0, 0.0]
    coords[1, 0] = [99.0, 99.0]
    coords[2, 0] = [2.0, 2.0]
    confidence = np.array([[1.0], [0.0], [1.0]])

    filled = extractor._interpolate_low_confidence(coords, confidence)

    assert np.allclose(filled[1, 0], [1.0, 1.0])


def test_lift_to_3d_requires_checkpoint(tmp_path):
    extractor = PoseExtractor(motionbert_checkpoint=tmp_path / "missing.bin")
    coords_2d = np.zeros((4, 17, 2), dtype=float)

    try:
        extractor.lift_to_3d(coords_2d, fps=30.0)
    except FileNotFoundError as exc:
        assert "MotionBERT checkpoint not found" in str(exc)
    else:
        raise AssertionError("Expected missing MotionBERT checkpoint to raise FileNotFoundError")
