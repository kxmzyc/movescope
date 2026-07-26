import numpy as np
import pytest

from movescope.template import ActionTemplate
from movescope.types import PoseResult


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
            return PoseResult(
                fps=30.0,
                joint_names=[f"joint_{idx}" for idx in range(17)],
                coords_2d=np.zeros((5, 17, 2)),
                confidence=np.ones((5, 17)),
                coords_3d=None,
                coords_3d_pseudo=np.full((5, 17, 3), value),
            )

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


def test_template_v2_builds_per_frame_curves():
    """构建后应产出与代表序列等长的逐帧参考曲线与容差带。"""
    rng = np.random.default_rng(7)
    sequences = [np.cumsum(rng.normal(size=(10, 12)), axis=0) for _ in range(3)]
    template = ActionTemplate("squat")

    template.build_from_features(sequences)

    assert template.reference_seq.shape == template.representative_seq.shape
    assert template.tolerance_band.shape == template.reference_seq.shape
    assert np.all(template.tolerance_band >= 5.0)
    # 全局 std 是逐帧 std 的时间平均，两种口径必须自洽。
    assert np.all(template.tolerance >= 5.0)
    assert template.tolerance.shape == (12,)


def test_template_v2_roundtrip_preserves_curves(tmp_path):
    sequences = [np.ones((6, 12)) * value for value in (1.0, 2.0, 3.0)]
    template = ActionTemplate("squat")
    template.build_from_features(sequences)
    path = template.save(tmp_path / "squat.npz")

    loaded = ActionTemplate.load("squat", path)

    assert np.allclose(loaded.reference_seq, template.reference_seq)
    assert np.allclose(loaded.tolerance_band, template.tolerance_band)


def test_template_v1_file_falls_back_to_broadcast(tmp_path):
    """旧格式 npz（无逐帧字段）应回退为代表序列 + 全局容差广播。"""
    path = tmp_path / "squat.npz"
    np.savez_compressed(
        path,
        action_name="squat",
        mean=np.zeros(12),
        std=np.ones(12),
        tolerance=np.full(12, 6.0),
        representative_seq=np.zeros((5, 12)),
        n_videos=2,
    )

    loaded = ActionTemplate.load("squat", path)

    assert loaded.reference_seq is None
    assert loaded.tolerance_band is None
    assert np.allclose(loaded.reference_curve, np.zeros((5, 12)))
    assert loaded.tolerance_curve.shape == (5, 12)
    assert np.allclose(loaded.tolerance_curve, 6.0)
