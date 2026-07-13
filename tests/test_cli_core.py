import subprocess
import sys

import numpy as np

from scripts import fetch_videos


def test_fetch_videos_dry_run_handles_unprintable_error(monkeypatch, capsys):
    def fake_run_yt_dlp(args):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="bad char: \ufffd")

    monkeypatch.setattr(fetch_videos, "run_yt_dlp", fake_run_yt_dlp)

    ok, bad = fetch_videos.dry_run("标准深蹲教学", 1)

    assert (ok, bad) == (0, 1)
    assert "[失败]" in capsys.readouterr().out


def test_feature_template_assessment_cli(tmp_path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    np.save(features_dir / "expert_a.npy", np.zeros((6, 12)))
    np.save(features_dir / "expert_b.npy", np.ones((6, 12)) * 0.2)
    test_features = tmp_path / "test.npy"
    np.save(test_features, np.ones((6, 12)) * 2.0)
    template_path = tmp_path / "squat.npz"

    build = subprocess.run(
        [
            sys.executable,
            "scripts/build_template.py",
            "--action",
            "squat",
            "--features-dir",
            str(features_dir),
            "--output",
            str(template_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert build.returncode == 0, build.stderr
    assert template_path.exists()

    assess = subprocess.run(
        [
            sys.executable,
            "scripts/assess_features.py",
            "--action",
            "squat",
            "--features",
            str(test_features),
            "--template",
            str(template_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert assess.returncode == 0, assess.stderr
    assert "total_score" in assess.stdout
