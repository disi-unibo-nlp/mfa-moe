import pytest
from scripts.experimental_resume.summarize_motion_uncertainty import METRICS, scalar_intervals


def test_family_cluster_resampling_and_scalar_means():
    families = {str(i): {field: [float(i)] * 40 for field in METRICS.values()}
                for i in range(4)}
    result = scalar_intervals(families, n_boot=1000, seed=7)
    assert set(result) == set(METRICS)
    for row in result.values():
        assert row['estimate'] == 1.5
        assert row['simultaneous_interval'][0] <= 1.5 <= row['simultaneous_interval'][1]


def test_reject_duplicate_or_incomplete_family_profiles():
    field = next(iter(METRICS.values()))
    with pytest.raises(ValueError):
        scalar_intervals({'a': {field: [0.] * 40}})
