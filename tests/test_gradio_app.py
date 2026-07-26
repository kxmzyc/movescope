"""Gradio 调试界面评估链路的行为测试（不依赖 cv2/gradio 本体）。"""

import numpy as np

from frontend import gradio_app
from movescope.features import FeatureExtractor
from movescope.template import ActionTemplate
from movescope.types import PoseResult


class _FakePoseExtractor:
    def extract(self, _video_path):
        return PoseResult(
            fps=30.0,
            joint_names=[f"joint_{idx}" for idx in range(17)],
            coords_2d=np.zeros((6, 17, 2)),
            confidence=np.ones((6, 17)),
            coords_3d_pseudo=np.zeros((6, 17, 3)),
        )


class _CoreMissingFeatureExtractor:
    """双膝双髋（特征 0-3）全程缺失的特征序列。"""

    features = FeatureExtractor().features

    def extract(self, _coords_3d, normalize=True):
        result = np.ones((6, 12))
        result[:, :4] = np.nan
        return result


def test_gradio_assess_rejects_missing_core_features(tmp_path, monkeypatch):
    """核心关节数据不完整时应拒绝评估并给出可读报错，而不是静默给出高分。"""
    monkeypatch.chdir(tmp_path)
    template = ActionTemplate("squat")
    template.build_from_features([np.zeros((6, 12)), np.ones((6, 12))])
    template.save()

    monkeypatch.setattr(gradio_app, "PoseExtractor", _FakePoseExtractor)
    monkeypatch.setattr(gradio_app, "FeatureExtractor", _CoreMissingFeatureExtractor)

    overlay, score, _bar, text = gradio_app.assess_video("fake.mp4", "squat")

    assert overlay is None
    assert score == 0.0
    assert "评估失败" in text
    assert "左膝" in text
    assert "完整入镜" in text
