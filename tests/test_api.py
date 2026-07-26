import asyncio
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes import _safe_unlink
from api.services import AssessmentService, build_skeleton_payload
from api.settings import get_settings
from movescope.config import Settings
from movescope.template import ActionTemplate
from movescope.types import PoseResult


def test_health():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["max_upload_bytes"] > 0
    assert ".mp4" in payload["allowed_extensions"]


def test_local_frontend_origin_is_allowed():
    response = TestClient(app).get(
        "/health",
        headers={"Origin": "http://localhost:5173"},
    )

    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_actions_lists_templates(tmp_path, monkeypatch):
    template_dir = tmp_path / "data" / "templates"
    template_dir.mkdir(parents=True)
    (template_dir / "squat.npz").write_bytes(b"placeholder")
    monkeypatch.chdir(tmp_path)

    response = TestClient(app).get("/actions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["actions"] == ["squat"]
    # 无法解析的模板文件仍出现在 actions 名单中，但没有元数据。
    assert payload["templates"] == []


def test_actions_returns_template_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _save_template()

    response = TestClient(app).get("/actions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["actions"] == ["squat"]
    assert payload["templates"] == [
        {"action": "squat", "n_videos": 2, "feature_dim": 12, "frames": 6}
    ]


def test_assess_no_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    video_path = Path("sample.mp4")
    video_path.write_bytes(b"not-a-real-video")

    with video_path.open("rb") as handle:
        response = TestClient(app).post(
            "/assess",
            data={"action": "squat"},
            files={"video": ("sample.mp4", handle, "video/mp4")},
        )

    assert response.status_code == 422
    assert "movescope-build-template" in response.json()["detail"]


def test_demo_returns_reproducible_assessment():
    response = TestClient(app).get("/demo")

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["source"] == "synthetic"
    assert 0.0 <= payload["total_score"] <= 100.0
    assert payload["advice_source"] == "rule"
    assert any(phase["anomalies"] for phase in payload["phases"])
    anomaly = next(anomaly for phase in payload["phases"] for anomaly in phase["anomalies"])
    assert {"feature_index", "joint", "joint_display", "parent", "child"} <= set(anomaly)
    summary = payload["per_feature_summary"][0]
    assert {"tolerance_deg", "score_weight"} <= set(summary)
    timeline = payload["timeline"]
    assert timeline["frame_count"] == payload["metadata"]["frames"]
    assert len(timeline["series"]) == 12
    assert len(timeline["series"][0]["test_deg"]) == len(timeline["time_sec"])


def test_assess_rejects_invalid_action_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    response = TestClient(app).post(
        "/assess",
        data={"action": "../../secret"},
        files={"video": ("sample.mp4", b"video", "video/mp4")},
    )

    assert response.status_code == 422


def test_assess_rejects_unsupported_file_type(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    response = TestClient(app).post(
        "/assess",
        data={"action": "squat"},
        files={"video": ("sample.txt", b"not video", "text/plain")},
    )

    assert response.status_code == 415


def test_assess_enforces_upload_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _save_template()
    app.dependency_overrides[get_settings] = lambda: Settings(max_upload_mb=1)
    oversized = b"0" * (1024 * 1024 + 1)

    response = TestClient(app).post(
        "/assess",
        data={"action": "squat"},
        files={"video": ("sample.mp4", oversized, "video/mp4")},
    )

    assert response.status_code == 413


def test_assess_rejects_empty_upload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _save_template()

    response = TestClient(app).post(
        "/assess",
        data={"action": "squat"},
        files={"video": ("sample.mp4", b"", "video/mp4")},
    )

    assert response.status_code == 400
    assert "为空" in response.json()["detail"]


def test_assess_corrupt_video_returns_readable_400(tmp_path, monkeypatch):
    """伪造扩展名的损坏视频应得到可读的 400，而不是 500。"""
    monkeypatch.chdir(tmp_path)
    _save_template()

    response = TestClient(app).post(
        "/assess",
        data={"action": "squat"},
        files={"video": ("sample.mp4", b"not-a-real-video", "video/mp4")},
    )

    assert response.status_code == 400
    assert "视频" in response.json()["detail"]


def test_assess_timeout_returns_504(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _save_template()
    app.dependency_overrides[get_settings] = lambda: Settings(assess_timeout_sec=0.5)

    finish = threading.Event()

    def slow_assess(_self, _path, _template):
        finish.wait(10.0)
        return {}

    monkeypatch.setattr(AssessmentService, "assess_file", slow_assess)
    # 超时后槽位要等被遗弃的评估线程结束才释放；换用独立信号量，
    # 避免遗留线程占用模块级 app 的槽位影响其他测试。
    original = app.state.assess_semaphore
    app.state.assess_semaphore = asyncio.Semaphore(2)
    try:
        response = TestClient(app).post(
            "/assess",
            data={"action": "squat"},
            files={"video": ("sample.mp4", b"video", "video/mp4")},
        )
    finally:
        finish.set()
        app.state.assess_semaphore = original

    assert response.status_code == 504


def test_assess_slot_stays_held_until_abandoned_thread_finishes(tmp_path, monkeypatch):
    """504 后并发槽位仍被遗弃的评估线程占用：新请求应得到 503，而不是与其叠加运行。"""
    monkeypatch.chdir(tmp_path)
    _save_template()
    app.dependency_overrides[get_settings] = lambda: Settings(assess_timeout_sec=0.2)

    finish = threading.Event()

    def slow_assess(_self, _path, _template):
        finish.wait(10.0)
        return {}

    monkeypatch.setattr(AssessmentService, "assess_file", slow_assess)
    original = app.state.assess_semaphore
    app.state.assess_semaphore = asyncio.Semaphore(1)
    try:
        # 用同一个 TestClient（同一事件循环）发两个请求：独立 TestClient 在
        # 关闭事件循环时会取消被遗弃的任务并释放槽位，模拟不出常驻服务的行为。
        with TestClient(app) as client:
            first = client.post(
                "/assess",
                data={"action": "squat"},
                files={"video": ("sample.mp4", b"video", "video/mp4")},
            )
            assert first.status_code == 504

            second = client.post(
                "/assess",
                data={"action": "squat"},
                files={"video": ("sample.mp4", b"video", "video/mp4")},
            )
            assert second.status_code == 503
    finally:
        finish.set()
        app.state.assess_semaphore = original


def test_assess_busy_returns_503(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = app.state.assess_semaphore
    app.state.assess_semaphore = asyncio.Semaphore(0)
    try:
        response = TestClient(app).post(
            "/assess",
            data={"action": "squat"},
            files={"video": ("sample.mp4", b"video", "video/mp4")},
        )
    finally:
        app.state.assess_semaphore = original

    assert response.status_code == 503


def test_safe_unlink_tolerates_locked_file(tmp_path, monkeypatch):
    """Windows 上评估线程仍占用文件时，清理不应把 504 变成 500。"""
    target = tmp_path / "upload.mp4"
    target.write_bytes(b"video")
    monkeypatch.setattr(Path, "unlink", lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError()))

    _safe_unlink(target)  # 不应抛出


def test_assess_success_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _save_template()
    monkeypatch.setattr(
        AssessmentService,
        "assess_file",
        lambda _self, _path, _template: {
            "action": "squat",
            "total_score": 88.0,
            "segmented": False,
            "phases": [],
            "per_feature_summary": [],
            "llm_advice": "Keep the movement controlled.",
            "advice_source": "rule",
        },
    )

    response = TestClient(app).post(
        "/assess",
        data={"action": "squat"},
        files={"video": ("sample.mp4", b"video", "video/mp4")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_score"] == 88.0
    assert payload["advice_source"] == "rule"


def test_build_skeleton_payload_downsamples_and_masks_nan():
    coords = np.zeros((10, 17, 2))
    coords[:, :, 0] = 0.25
    coords[:, :, 1] = 0.75
    coords[1, 3, :] = np.nan
    pose = PoseResult(
        fps=30.0,
        joint_names=[f"j{i}" for i in range(17)],
        coords_2d=coords,
        confidence=np.ones((10, 17)) * 0.9,
        coords_3d_pseudo=np.zeros((10, 17, 3)),
    )

    payload = build_skeleton_payload(pose, max_frames=5)

    assert payload is not None
    assert payload["frame_stride"] == 2
    assert payload["frame_count"] == 10
    assert len(payload["keypoints"]) == 5
    assert payload["keypoints"][0][0] == [0.25, 0.75]
    assert len(payload["time_sec"]) == 5
    assert len(payload["confidence"][0]) == 17
    # 关节名与 SKELETON_EDGES 对不上时不输出边，而不是崩溃。
    assert payload["edges"] == []


def test_build_skeleton_payload_masks_nan_and_maps_edges():
    from movescope.constants import SKELETON_EDGES
    from movescope.features import JOINT_NAMES

    coords = np.full((4, 17, 2), 0.5)
    coords[0, 2, :] = np.nan
    pose = PoseResult(
        fps=30.0,
        joint_names=list(JOINT_NAMES),
        coords_2d=coords,
        confidence=np.ones((4, 17)),
        coords_3d_pseudo=np.zeros((4, 17, 3)),
    )

    payload = build_skeleton_payload(pose)

    assert payload is not None
    assert payload["keypoints"][0][2] is None
    assert payload["keypoints"][0][3] == [0.5, 0.5]
    assert len(payload["edges"]) == len(SKELETON_EDGES)
    assert all(len(edge) == 2 for edge in payload["edges"])


def _save_template() -> None:
    template = ActionTemplate("squat")
    template.build_from_features([np.zeros((6, 12)), np.ones((6, 12))])
    template.save()


def test_assess_multi_rep_video_returns_rep_summaries(tmp_path, monkeypatch):
    """多次往复的视频逐次评分：总分为均值，详情时刻偏移回原视频坐标。"""
    from movescope.features import JOINT_NAMES, FeatureExtractor

    monkeypatch.chdir(tmp_path)
    _save_template()

    stand, depth = 170.0, 80.0
    rest = np.full(20, stand)
    pieces = [rest]
    for _ in range(3):
        t = np.linspace(0.0, np.pi, 30)
        pieces.append(stand - depth * np.sin(t) ** 2)
        pieces.append(rest)
    knee = np.concatenate(pieces)
    features = np.full((len(knee), 12), 100.0)
    features[:, 0] = knee
    features[:, 1] = knee

    class FakePoseExtractor:
        def extract(self, _video_path):
            frames = len(knee)
            return PoseResult(
                fps=30.0,
                joint_names=list(JOINT_NAMES),
                coords_2d=np.full((frames, 17, 2), 0.5),
                confidence=np.ones((frames, 17)),
                coords_3d_pseudo=np.zeros((frames, 17, 3)),
            )

    class FakeFeatureExtractor:
        features = FeatureExtractor().features

        def extract(self, _coords, normalize=True):
            return features

    monkeypatch.setattr("api.services.PoseExtractor", FakePoseExtractor)
    monkeypatch.setattr("api.services.FeatureExtractor", FakeFeatureExtractor)

    response = TestClient(app).post(
        "/assess",
        data={"action": "squat"},
        files={"video": ("sample.mp4", b"video", "video/mp4")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["reps"]) == 3
    detail = payload["rep_detail_index"]
    assert 0 <= detail < 3
    scores = [rep["score"] for rep in payload["reps"]]
    assert payload["total_score"] == pytest.approx(sum(scores) / 3, abs=0.01)
    detail_start = payload["reps"][detail]["time_range"][0]
    assert payload["timeline"]["time_sec"][0] == pytest.approx(detail_start, abs=0.05)
    assert payload["phases"][0]["time_range"][0] == pytest.approx(detail_start, abs=0.05)
