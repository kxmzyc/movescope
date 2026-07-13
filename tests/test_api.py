from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app
from movescope.template import ActionTemplate


def test_health():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


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
    assert response.json() == {"actions": ["squat"]}


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
    assert "build_template.py" in response.json()["detail"]


def test_demo_returns_reproducible_assessment():
    response = TestClient(app).get("/demo")

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["source"] == "synthetic"
    assert 0.0 <= payload["total_score"] <= 100.0
    assert payload["phases"][0]["anomalies"]


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
    monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 4)

    response = TestClient(app).post(
        "/assess",
        data={"action": "squat"},
        files={"video": ("sample.mp4", b"12345", "video/mp4")},
    )

    assert response.status_code == 413


def test_assess_success_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _save_template()
    monkeypatch.setattr(
        api_main,
        "_assess_file",
        lambda _path, _template: {
            "action": "squat",
            "total_score": 88.0,
            "phases": [],
            "per_joint_summary": {},
            "llm_advice": "Keep the movement controlled.",
        },
    )

    response = TestClient(app).post(
        "/assess",
        data={"action": "squat"},
        files={"video": ("sample.mp4", b"video", "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json()["total_score"] == 88.0


def _save_template() -> None:
    template = ActionTemplate("squat")
    template.build_from_features([np.zeros((6, 12)), np.ones((6, 12))])
    template.save()
