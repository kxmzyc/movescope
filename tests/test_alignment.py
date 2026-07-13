import numpy as np
import pytest

from movescope.alignment import DTWAligner, WeightedSegmentedDTWAligner


def test_standard_dtw_path_is_monotonic():
    query = np.array([[0.0], [1.0], [2.0]])
    reference = np.array([[0.0], [2.0]])

    path = DTWAligner().align(query, reference)

    assert path[0] == (0, 0)
    assert path[-1] == (2, 1)
    assert all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(path, path[1:]))


def test_segmented_path_continuous():
    seq = np.arange(24, dtype=float).reshape(12, 2)
    path = WeightedSegmentedDTWAligner(min_segment_frames=2).align(seq, seq)

    assert path[0] == (0, 0)
    assert path[-1] == (11, 11)
    assert all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(path, path[1:]))


def test_detect_phases_tracks_feature_blocks():
    seq = np.vstack(
        [
            np.zeros((4, 2)),
            np.ones((4, 2)) * 10.0,
            np.ones((4, 2)) * 20.0,
        ]
    )

    phases = WeightedSegmentedDTWAligner(min_segment_frames=2).detect_phases(seq, n_phases=3)

    assert phases == [(0, 4), (4, 8), (8, 12)]


def test_weighted_distance_respects_feature_weights():
    aligner = WeightedSegmentedDTWAligner()
    query = np.array([10.0, 1.0])
    reference = np.zeros(2)

    high_first = aligner.weighted_distance(query, reference, np.array([0.99, 0.01]))
    high_second = aligner.weighted_distance(query, reference, np.array([0.01, 0.99]))

    assert high_first > high_second


def test_compute_joint_weights_prefers_low_variance_features():
    class Template:
        std = np.array([0.1, 10.0])

    weights = WeightedSegmentedDTWAligner().compute_joint_weights(Template())

    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > weights[1]


def test_segment_count_mismatch_falls_back_to_complete_alignment():
    class MismatchedSegmentsAligner(WeightedSegmentedDTWAligner):
        def detect_phases(self, feature_seq, n_phases=4):
            if len(feature_seq) == 6:
                return [(0, 3), (3, 6)]
            return [(0, len(feature_seq))]

    query = np.arange(12, dtype=float).reshape(6, 2)
    reference = np.arange(16, dtype=float).reshape(8, 2)

    path = MismatchedSegmentsAligner().align(query, reference)

    assert path[0] == (0, 0)
    assert path[-1] == (5, 7)


@pytest.mark.parametrize("weights", [np.zeros(2), np.array([1.0, -1.0])])
def test_invalid_weights_are_rejected(weights):
    seq = np.arange(8, dtype=float).reshape(4, 2)

    with pytest.raises(ValueError, match="weights"):
        WeightedSegmentedDTWAligner().align(seq, seq, weights=weights)
