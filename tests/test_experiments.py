import numpy as np

from movescope.experiments import (
    load_feature_sequences,
    run_template_sensitivity_from_features,
    summarize_ablation,
    video_files,
    viewpoint_std,
)


def test_video_files_filters_supported_extensions(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "c.MOV").write_bytes(b"")

    files = video_files(tmp_path)

    assert [path.name for path in files] == ["a.mp4", "c.MOV"]


def test_summarize_ablation_separation():
    rows = [
        {"variant": "v1", "label": "good", "total_score": 90.0},
        {"variant": "v1", "label": "bad", "total_score": 60.0},
        {"variant": "v1", "label": "bad", "total_score": 70.0},
    ]

    summary = summarize_ablation(rows)

    assert summary[0]["separation"] == 25.0
    assert summary[0]["n_good"] == 1
    assert summary[0]["n_bad"] == 2


def test_viewpoint_std_by_variant():
    rows = [
        {"variant": "baseline_2d", "total_score": 80.0},
        {"variant": "baseline_2d", "total_score": 100.0},
        {"variant": "ours_full", "total_score": 90.0},
        {"variant": "ours_full", "total_score": 92.0},
    ]

    stats = viewpoint_std(rows)

    assert np.isclose(stats["baseline_2d"], 10.0)
    assert np.isclose(stats["ours_full"], 1.0)


def test_load_feature_sequences_npy_and_npz(tmp_path):
    np.save(tmp_path / "a.npy", np.zeros((3, 12)))
    np.savez_compressed(tmp_path / "b.npz", features=np.ones((4, 12)))

    sequences = load_feature_sequences(tmp_path)

    assert [seq.shape for seq in sequences] == [(3, 12), (4, 12)]


def test_template_sensitivity_from_features():
    experts = [np.ones((5, 12)) * offset for offset in (0.0, 0.2, 0.4)]
    tests = [np.ones((5, 12)) * offset for offset in (0.1, 0.3)]

    rows = run_template_sensitivity_from_features(experts, tests, counts=(1, 2, 5))

    assert [row["template_count"] for row in rows] == [1, 2]
    assert rows[0]["n_tests"] == 2
    assert all("std_score" in row for row in rows)
