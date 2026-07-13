import numpy as np
import pytest

from movescope.template import ActionTemplate


def test_template_build_from_features():
    sequences = [np.ones((5, 12)), np.ones((5, 12)) * 2.0]
    template = ActionTemplate("squat")

    template.build_from_features(sequences)

    assert template.tolerance.shape == (12,)
    assert np.all(template.tolerance > 0)
    assert template.representative_seq.shape == (5, 12)


def test_template_build_from_expert_videos(tmp_path):
    expert_dir = tmp_path / "expert"
    expert_dir.mkdir()
    (expert_dir / "a.mp4").write_bytes(b"fake")
    (expert_dir / "b.webm").write_bytes(b"fake")
    (expert_dir / "ignore.txt").write_text("not a video", encoding="utf-8")

    class FakePoseExtractor:
        def extract(self, video_path):
            value = 1.0 if video_path.endswith("a.mp4") else 2.0
            return {"coords_3d": None, "coords_3d_pseudo": np.full((5, 17, 3), value)}

    class FakeFeatureExtractor:
        def extract(self, coords_3d, normalize=True):
            value = float(coords_3d[0, 0, 0])
            return np.full((5, 12), value)

    template = ActionTemplate("squat")
    template.build(expert_dir, FakePoseExtractor(), FakeFeatureExtractor())

    assert template.n_videos == 2
    assert np.allclose(template.mean, np.full(12, 1.5))
    assert np.all(template.tolerance > 0)


def test_single_expert_uses_practical_tolerance_floor():
    template = ActionTemplate("squat")

    template.build_from_features([np.zeros((8, 12))])

    assert np.all(template.tolerance >= 5.0)


def test_template_rejects_non_finite_features():
    template = ActionTemplate("squat")

    with pytest.raises(ValueError, match="有限值"):
        template.build_from_features([np.full((8, 12), np.nan)])
