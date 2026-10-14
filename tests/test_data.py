"""Comprehensive unit tests for C-MAPSS data ingestion, validation, splitting, and preprocessing."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from failuredna.data.loader import (
    CYCLE_COL,
    ID_COL,
    SENSOR_COLS,
    SETTING_COLS,
    TELEMETRY_COLS,
    load_cmapss_dataset,
    load_raw_cmapss_file,
)
from failuredna.data.preprocessing import CMAPSSPreprocessor
from failuredna.data.split import split_engines_by_unit
from failuredna.data.validator import DataValidator, ValidationError


@pytest.fixture
def sample_telemetry_df():
    """Create a minimal valid telemetry DataFrame with 4 engines."""
    rows = []
    for unit in range(1, 5):
        for cycle in range(1, 21):
            settings = [10.0 + unit, 0.2, 60.0]
            # sensor 1 constant, others varying
            sensors = [518.67] + [float(100.0 + i + cycle * 0.1) for i in range(2, 22)]
            rows.append([unit, cycle] + settings + sensors)
    return pd.DataFrame(rows, columns=TELEMETRY_COLS)


@pytest.fixture
def sample_rul_df():
    """Create a valid RUL DataFrame."""
    return pd.DataFrame({
        "unit_id": [1, 2, 3, 4],
        "rul": [25.0, 40.0, 15.0, 60.0],
    })


def test_load_raw_cmapss_file_success(tmp_path):
    """Test loading a well-formatted whitespace-delimited file."""
    file_path = tmp_path / "valid_cmapss.txt"
    lines = []
    for u in [1, 2]:
        for c in range(1, 6):
            vals = [str(u), str(c)] + ["0.0", "0.0", "100.0"] + [f"{100 + i}.5" for i in range(1, 22)]
            lines.append(" ".join(vals) + "  \n")  # include trailing whitespace
    file_path.write_text("".join(lines))

    df = load_raw_cmapss_file(file_path, is_rul=False)
    assert df.shape == (10, 26)
    assert list(df.columns) == TELEMETRY_COLS
    assert df[ID_COL].dtype == np.int64
    assert df[CYCLE_COL].dtype == np.int64
    assert df["s_1"].dtype == np.float64


def test_load_raw_cmapss_file_not_found():
    """Test explicit FileNotFoundError when target does not exist."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_raw_cmapss_file("non_existent_cmapss_file_9999.txt")


def test_load_raw_cmapss_file_malformed_columns(tmp_path):
    """Test ValueError raised when row column count is incorrect."""
    file_path = tmp_path / "malformed.txt"
    file_path.write_text("1 1 0.0 0.0 100.0 500.0\n")  # Only 6 columns instead of 26
    with pytest.raises(ValueError, match="Schema mismatch"):
        load_raw_cmapss_file(file_path, is_rul=False)


def test_load_raw_cmapss_file_empty(tmp_path):
    """Test ValueError raised on empty file."""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_raw_cmapss_file(file_path)


def test_load_rul_file(tmp_path):
    """Test loading single-column RUL file."""
    file_path = tmp_path / "RUL.txt"
    file_path.write_text("112\n98\n45\n")
    rul_df = load_raw_cmapss_file(file_path, is_rul=True)
    assert rul_df.shape == (3, 2)
    assert list(rul_df.columns) == ["unit_id", "rul"]
    assert rul_df["rul"].tolist() == [112.0, 98.0, 45.0]
    assert rul_df["unit_id"].tolist() == [1, 2, 3]


def test_validator_detects_nulls(sample_telemetry_df):
    """Test validator catches NaN / null values."""
    df_corrupt = sample_telemetry_df.copy()
    df_corrupt.loc[5, "s_3"] = np.nan
    validator = DataValidator(raise_on_error=True)
    with pytest.raises(ValidationError, match="Null values detected"):
        validator.validate_telemetry_df(df_corrupt)


def test_validator_detects_inf(sample_telemetry_df):
    """Test validator catches infinite values."""
    df_corrupt = sample_telemetry_df.copy()
    df_corrupt.loc[10, "s_7"] = np.inf
    validator = DataValidator(raise_on_error=True)
    with pytest.raises(ValidationError, match="Infinite values detected"):
        validator.validate_telemetry_df(df_corrupt)


def test_validator_detects_duplicate_cycles(sample_telemetry_df):
    """Test validator catches duplicate (unit_id, cycle) records."""
    dup_row = sample_telemetry_df.iloc[[0]].copy()
    df_corrupt = pd.concat([sample_telemetry_df, dup_row], ignore_index=True)
    validator = DataValidator(raise_on_error=True)
    with pytest.raises(ValidationError, match="duplicate"):
        validator.validate_telemetry_df(df_corrupt)


def test_validator_detects_non_monotonic_cycles(sample_telemetry_df):
    """Test validator catches inverted or non-monotonic cycle sequences."""
    df_corrupt = sample_telemetry_df.copy()
    # Invert cycle sequence for engine 1
    mask = df_corrupt[ID_COL] == 1
    df_corrupt.loc[mask, CYCLE_COL] = df_corrupt.loc[mask, CYCLE_COL].values[::-1]
    validator = DataValidator(raise_on_error=True)
    with pytest.raises(ValidationError, match="Non-strictly-increasing"):
        validator.validate_telemetry_df(df_corrupt)


def test_validator_detects_negative_rul():
    """Test validator rejects negative RUL."""
    rul_df = pd.DataFrame({"unit_id": [1, 2], "rul": [50.0, -5.0]})
    validator = DataValidator(raise_on_error=True)
    with pytest.raises(ValidationError, match="negative RUL"):
        validator.validate_rul_df(rul_df)


def test_split_engines_by_unit_disjointness(sample_telemetry_df):
    """Test that train, val, and test engine IDs are mutually disjoint."""
    # Expand engines to 10 for meaningful split testing
    dfs = []
    for u in range(1, 11):
        df_u = sample_telemetry_df.copy()
        df_u[ID_COL] = u
        dfs.append(df_u)
    ten_engine_df = pd.concat(dfs, ignore_index=True)

    train_df, val_df, test_df, meta = split_engines_by_unit(
        ten_engine_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    train_ids = set(train_df[ID_COL].unique())
    val_ids = set(val_df[ID_COL].unique())
    test_ids = set(test_df[ID_COL].unique())

    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0
    assert (train_ids | val_ids | test_ids) == set(range(1, 11))
    assert meta["disjointness_verified"] is True


def test_split_deterministic_seeding(sample_telemetry_df):
    """Test that same seed yields identical splits, and different seed yields different splits."""
    dfs = [sample_telemetry_df.assign(unit_id=u) for u in range(1, 11)]
    big_df = pd.concat(dfs, ignore_index=True)

    _, _, _, meta_1 = split_engines_by_unit(big_df, seed=42)
    _, _, _, meta_2 = split_engines_by_unit(big_df, seed=42)
    _, _, _, meta_diff = split_engines_by_unit(big_df, seed=999)

    assert meta_1["train_engine_ids"] == meta_2["train_engine_ids"]
    assert meta_1["train_engine_ids"] != meta_diff["train_engine_ids"]


def test_preprocessor_train_only_fitting(sample_telemetry_df):
    """Test that preprocessor fits exclusively on train data and prunes constant sensors."""
    train_df = sample_telemetry_df[sample_telemetry_df[ID_COL].isin([1, 2])].copy()
    val_df = sample_telemetry_df[sample_telemetry_df[ID_COL].isin([3, 4])].copy()

    preprocessor = CMAPSSPreprocessor(scaler_type="standard", variance_threshold=1e-5)

    # Calling transform before fit raises RuntimeError
    with pytest.raises(RuntimeError, match="must be fitted"):
        preprocessor.transform(val_df)

    preprocessor.fit(train_df)
    assert preprocessor.is_fitted_ is True
    # Sensor 1 was constant in fixture
    assert "s_1" in preprocessor.constant_sensors_
    assert "s_1" not in preprocessor.active_sensors_

    # Transform without refitting
    train_scaled = preprocessor.transform(train_df)
    val_scaled = preprocessor.transform(val_df)

    assert list(train_scaled.columns) == list(val_scaled.columns)
    # Unit ID and cycle are preserved intact
    assert (train_scaled[ID_COL] == train_df[ID_COL]).all()
    assert (train_scaled[CYCLE_COL] == train_df[CYCLE_COL]).all()
