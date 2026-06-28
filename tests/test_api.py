from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app


def test_health():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


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
