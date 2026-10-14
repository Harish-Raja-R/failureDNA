"""Unit tests for window padding and short trajectory handling strategies."""

import numpy as np
import pandas as pd
import pytest

from failuredna.data.loader import CYCLE_COL, ID_COL
from failuredna.data.windowing import WindowGenerator


@pytest.fixture
def short_and_long_engines_df():
    """Create a dataset with one long engine (60 cycles) and one short engine (20 cycles)."""
    rows = []
    # Engine 1: 60 cycles
    for c in range(1, 61):
        rows.append([1, c, 10.0 + c, 20.0 + c])
    # Engine 2: 20 cycles
    for c in range(1, 21):
        rows.append([2, c, 100.0 + c, 200.0 + c])

    return pd.DataFrame(rows, columns=[ID_COL, CYCLE_COL, "feat_1", "feat_2"])


def test_window_generator_invalid_padding_mode():
    """Verify ValueError on unsupported padding mode."""
    with pytest.raises(ValueError, match="padding_mode must be None, 'edge', or 'zero'"):
        WindowGenerator(window_size=50, padding_mode="reflection")


def test_edge_padding_short_trajectory(short_and_long_engines_df):
    """Verify that edge-padding repeats earliest observed state without future leakage."""
    generator = WindowGenerator(window_size=50, stride=1, padding_mode="edge")
    windows, meta, stats = generator.create_windows(short_and_long_engines_df)

    # Engine 1 (60 cycles): 60 - 50 + 1 = 11 windows (unpadded)
    # Engine 2 (20 cycles): 1 padded window (pad_len = 30)
    # Total windows: 12
    assert len(windows) == 12
    assert len(meta) == 12
    assert stats["padded_engines"] == [2]

    # Inspect the padded window for engine 2 (last record)
    padded_meta = meta[meta["unit_id"] == 2].iloc[0]
    assert bool(padded_meta["is_padded"]) is True
    assert padded_meta["pad_length"] == 30

    padded_window_idx = padded_meta["window_idx"]
    padded_window = windows[padded_window_idx]
    assert padded_window.shape == (50, 2)

    # First 30 steps must equal the first observed step of engine 2 (c=1: 101.0, 201.0)
    first_observed = np.array([101.0, 201.0], dtype=np.float32)
    for t in range(30):
        np.testing.assert_allclose(padded_window[t], first_observed)

    # Remaining 20 steps must equal the real trajectory (c=1..20)
    for step_i, c in enumerate(range(1, 21)):
        expected = np.array([100.0 + c, 200.0 + c], dtype=np.float32)
        np.testing.assert_allclose(padded_window[30 + step_i], expected)


def test_zero_padding_short_trajectory(short_and_long_engines_df):
    """Verify that zero-padding prepends exact zeros."""
    generator = WindowGenerator(window_size=50, stride=1, padding_mode="zero")
    windows, meta, _ = generator.create_windows(short_and_long_engines_df)

    padded_meta = meta[meta["unit_id"] == 2].iloc[0]
    padded_window = windows[padded_meta["window_idx"]]

    # First 30 steps must be zero
    np.testing.assert_allclose(padded_window[:30], 0.0)
    # Step 30 must be the first observed cycle (c=1)
    np.testing.assert_allclose(padded_window[30], [101.0, 201.0])


def test_extract_last_window_all_engines(short_and_long_engines_df):
    """Verify extract_last_window outputs exactly 1 window per engine, padding short ones."""
    generator = WindowGenerator(window_size=50)

    # With pad=True (edge mode)
    windows, meta = generator.extract_last_window(
        short_and_long_engines_df,
        pad=True,
        padding_mode="edge",
    )

    # Exactly 2 engines -> exactly 2 windows
    assert windows.shape == (2, 50, 2)
    assert len(meta) == 2

    # Engine 1 was unpadded
    eng1 = meta[meta["unit_id"] == 1].iloc[0]
    assert eng1["last_cycle"] == 60
    assert bool(eng1["is_padded"]) is False
    assert eng1["pad_length"] == 0

    # Engine 2 was padded by 30 cycles
    eng2 = meta[meta["unit_id"] == 2].iloc[0]
    assert eng2["last_cycle"] == 20
    assert bool(eng2["is_padded"]) is True
    assert eng2["pad_length"] == 30

    # With pad=False, raises ValueError on short engine
    with pytest.raises(ValueError, match="cycles < window_size"):
        generator.extract_last_window(short_and_long_engines_df, pad=False)
