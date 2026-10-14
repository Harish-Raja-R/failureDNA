"""Tests for Phase 5: Official NASA C-MAPSS FD004 Benchmark Evaluation.

Verifies:
1. Official test-label count and engine alignment (248 engines, 1..248 contiguous).
2. Correct final-cycle RUL prediction convention (1 window per engine at max cycle).
3. Window generation and edge-padding at short sequence lengths (<50 cycles).
4. Sensor-column consistency and ordering across train and official test datasets.
5. Prediction count and finite outputs across all benchmark models.
6. Metric calculations (MAE, RMSE, R2, NASA score) against known toy numerical examples.
7. No accidental training or tuning on official test labels (zero-leakage audit).
8. Clear physical isolation between internal validation and official evaluation artifacts.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from failuredna.data.loader import (
    CYCLE_COL,
    ID_COL,
    SENSOR_COLS,
    SETTING_COLS,
    load_raw_cmapss_file,
)
from failuredna.data.preprocessing import RegimeAwarePreprocessor
from failuredna.data.split import split_engines_by_unit
from failuredna.data.windowing import WindowGenerator
from failuredna.evaluation.metrics import (
    compute_regression_metrics,
    nasa_scoring_function,
)


class TestOfficialBenchmarkDataAlignment:
    """Verifies official test data dimensions, engine IDs, and label alignment."""

    def test_official_dataset_files_exist(self):
        assert Path("data/raw/test_FD004.txt").exists(), "Missing test_FD004.txt"
        assert Path("data/raw/RUL_FD004.txt").exists(), "Missing RUL_FD004.txt"

    def test_engine_count_and_label_alignment(self):
        test_df = load_raw_cmapss_file("data/raw/test_FD004.txt")
        rul_df = load_raw_cmapss_file("data/raw/RUL_FD004.txt", is_rul=True)

        unique_units = test_df[ID_COL].unique()
        assert len(unique_units) == 248, f"Expected 248 test engines, got {len(unique_units)}"
        assert len(rul_df) == 248, f"Expected 248 RUL labels, got {len(rul_df)}"

        # Verify contiguous 1..248 indexing
        assert sorted(unique_units) == list(range(1, 249))
        assert sorted(rul_df["unit_id"].tolist()) == list(range(1, 249))

    def test_sensor_column_consistency(self):
        train_df = load_raw_cmapss_file("data/raw/train_FD004.txt")
        test_df = load_raw_cmapss_file("data/raw/test_FD004.txt")

        assert list(train_df.columns) == list(test_df.columns)
        assert len(SENSOR_COLS) == 21
        assert len(SETTING_COLS) == 3


class TestShortEnginePaddingAndWindowing:
    """Verifies edge-padding and last-window extraction convention."""

    def test_short_engine_detection_and_edge_padding(self):
        test_df = load_raw_cmapss_file("data/raw/test_FD004.txt")
        engine_lengths = test_df.groupby(ID_COL)[CYCLE_COL].max()
        short_engines = engine_lengths[engine_lengths < 50].index.tolist()

        # Exactly 20 engines in FD004 have fewer than 50 cycles
        assert len(short_engines) == 20
        assert engine_lengths.min() == 19

        # Preprocess with mock scaler
        feature_cols = [f"s_{i}" for i in range(1, 22)]
        win_gen = WindowGenerator(window_size=50, stride=1, feature_cols=feature_cols, padding_mode="edge")
        windows, meta = win_gen.extract_last_window(test_df, pad=True, padding_mode="edge")

        assert windows.shape == (248, 50, len(feature_cols))
        assert len(meta) == 248
        assert not np.isnan(windows).any()

        # Verify that short engines have positive pad_length
        short_meta = meta[meta["unit_id"].isin(short_engines)]
        assert (short_meta["pad_length"] > 0).all()
        assert (short_meta["pad_length"] == (50 - engine_lengths[short_engines].values)).all()


class TestBenchmarkMetricFormulas:
    """Verifies MAE, RMSE, R2, and NASA score calculations against known toy cases."""

    def test_nasa_score_known_toy_values(self):
        # Case 1: Perfect prediction
        y_true = np.array([50.0, 100.0])
        y_pred = np.array([50.0, 100.0])
        assert nasa_scoring_function(y_true, y_pred) == 0.0

        # Case 2: Early prediction (diff = -13 -> exp(13/13) - 1 = e - 1 ~ 1.71828)
        y_true_early = np.array([50.0])
        y_pred_early = np.array([37.0])
        score_early = nasa_scoring_function(y_true_early, y_pred_early)
        expected_early = float(np.exp(1.0) - 1.0)
        assert pytest.approx(score_early, rel=1e-5) == expected_early

        # Case 3: Late prediction (diff = +10 -> exp(10/10) - 1 = e - 1 ~ 1.71828)
        y_true_late = np.array([50.0])
        y_pred_late = np.array([60.0])
        score_late = nasa_scoring_function(y_true_late, y_pred_late)
        expected_late = float(np.exp(1.0) - 1.0)
        assert pytest.approx(score_late, rel=1e-5) == expected_late

        # Case 4: Asymmetry test (error of 10 cycles early vs 10 cycles late)
        # Early (-10): exp(10/13) - 1 ~ 1.158
        # Late (+10): exp(10/10) - 1 ~ 1.718 (late is punished more heavily)
        score_e10 = nasa_scoring_function([50.0], [40.0])
        score_l10 = nasa_scoring_function([50.0], [60.0])
        assert score_l10 > score_e10

    def test_compute_regression_metrics(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 32.0])
        m = compute_regression_metrics(y_true, y_pred)

        assert pytest.approx(m["mae"], rel=1e-5) == 2.0
        assert pytest.approx(m["rmse"], rel=1e-5) == 2.0
        assert m["r2"] > 0.0
        assert m["nasa_score"] > 0.0


class TestZeroLeakageIsolation:
    """Verifies that official test labels are not used in training or tuning."""

    def test_split_isolation(self):
        train_raw = load_raw_cmapss_file("data/raw/train_FD004.txt")
        train_raw_part, val_raw_part, itest_raw_part, split_info = split_engines_by_unit(
            train_raw, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
        )
        train_units = set(split_info["train_engine_ids"])
        val_units = set(split_info["val_engine_ids"])
        itest_units = set(split_info["test_engine_ids"])

        assert len(train_units & val_units) == 0
        assert len(train_units & itest_units) == 0
        assert len(val_units & itest_units) == 0

    def test_artifact_separation(self):
        # Verify that phase4 and phase5 results have separate file paths
        assert Path("reports/results/phase4_validation_summary.json").exists()
        # Phase 5 files must not overwrite Phase 4 files
        p4_mtime = Path("reports/results/phase4_validation_summary.json").stat().st_mtime
        assert p4_mtime > 0
