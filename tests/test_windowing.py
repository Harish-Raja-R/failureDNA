"""Comprehensive unit tests for temporal sliding window extraction."""

import numpy as np
import pandas as pd
import pytest

from failuredna.data.loader import CYCLE_COL, ID_COL
from failuredna.data.windowing import WindowGenerator


@pytest.fixture
def multi_engine_df():
    """Create a synthetic dataset with 3 engines of different lengths: 100, 75, and 30 cycles."""
    rows = []
    # Engine 1: 100 cycles
    for c in range(1, 101):
        rows.append([1, c, 10.0 + c, 20.0 + c])
    # Engine 2: 75 cycles
    for c in range(1, 76):
        rows.append([2, c, 50.0 + c, 60.0 + c])
    # Engine 3: 30 cycles (short trajectory for W=50)
    for c in range(1, 31):
        rows.append([3, c, 90.0 + c, 95.0 + c])

    return pd.DataFrame(rows, columns=[ID_COL, CYCLE_COL, "feat_1", "feat_2"])


def test_window_count_exact_formula(multi_engine_df):
    """Verify exact theoretical window counts for stride=1 and stride=2."""
    generator_s1 = WindowGenerator(window_size=50, stride=1, drop_short_trajectories=True)
    windows_s1, meta_s1, stats_s1 = generator_s1.create_windows(multi_engine_df)

    # Engine 1 (100 cycles): 100 - 50 + 1 = 51 windows
    # Engine 2 (75 cycles):  75 - 50 + 1 = 26 windows
    # Engine 3 (30 cycles):  short -> skipped
    # Total expected: 51 + 26 = 77 windows
    assert len(windows_s1) == 77
    assert len(meta_s1) == 77
    assert stats_s1["total_windows_generated"] == 77
    assert stats_s1["short_engines_skipped"] == [3]

    # Stride = 2
    generator_s2 = WindowGenerator(window_size=50, stride=2, drop_short_trajectories=True)
    windows_s2, meta_s2, stats_s2 = generator_s2.create_windows(multi_engine_df)
    # Engine 1: (100 - 50) // 2 + 1 = 26 windows
    # Engine 2: (75 - 50) // 2 + 1 = 13 windows
    # Total expected: 26 + 13 = 39 windows
    assert len(windows_s2) == 39
    assert len(meta_s2) == 39


def test_windows_single_engine_isolation(multi_engine_df):
    """Verify that no window ever mixes observations across multiple engines."""
    generator = WindowGenerator(window_size=50, stride=1, feature_cols=["feat_1", "feat_2"])
    windows, meta, _ = generator.create_windows(multi_engine_df)

    for idx in range(len(windows)):
        unit_id = meta.loc[idx, "unit_id"]
        start_c = meta.loc[idx, "start_cycle"]
        end_c = meta.loc[idx, "end_cycle"]

        # Ensure window metadata matches cycle spans
        assert end_c - start_c + 1 == 50

        # Verify values in window array correspond exactly to engine unit_id
        if unit_id == 1:
            expected_start_val = 10.0 + start_c
            actual_start_val = windows[idx, 0, 0]
            assert np.isclose(actual_start_val, expected_start_val)
        elif unit_id == 2:
            expected_start_val = 50.0 + start_c
            actual_start_val = windows[idx, 0, 0]
            assert np.isclose(actual_start_val, expected_start_val)


def test_windows_strict_temporal_ordering(multi_engine_df):
    """Verify that temporal ordering is strictly preserved and non-inverted."""
    generator = WindowGenerator(window_size=50, stride=1, feature_cols=["feat_1"])
    windows, meta, _ = generator.create_windows(multi_engine_df)

    for idx in range(len(windows)):
        feat_traj = windows[idx, :, 0]
        # In our fixture, feat_1 is monotonically increasing with cycle
        diffs = np.diff(feat_traj)
        assert (diffs > 0).all(), f"Temporal inversion detected in window {idx}."


def test_short_trajectory_rejection(multi_engine_df):
    """Verify ValueError is raised if drop_short_trajectories is False."""
    strict_generator = WindowGenerator(window_size=50, stride=1, drop_short_trajectories=False)
    with pytest.raises(ValueError, match="shorter than window_size"):
        strict_generator.create_windows(multi_engine_df)


def test_invalid_generator_parameters():
    """Verify ValueError for non-positive window_size or stride."""
    with pytest.raises(ValueError, match="window_size must be >= 1"):
        WindowGenerator(window_size=0)
    with pytest.raises(ValueError, match="stride must be >= 1"):
        WindowGenerator(window_size=50, stride=0)
