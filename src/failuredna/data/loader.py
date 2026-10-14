"""Raw C-MAPSS data ingestion and parsing module.

Supports whitespace-delimited NASA C-MAPSS telemetry files with rigorous
schema enforcement and transparent error handling.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from failuredna.utils.logging import get_logger

logger = get_logger(__name__)

# Standard C-MAPSS Column Definitions
ID_COL = "unit_id"
CYCLE_COL = "cycle"
SETTING_COLS: List[str] = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS: List[str] = [f"s_{i}" for i in range(1, 22)]
TELEMETRY_COLS: List[str] = [ID_COL, CYCLE_COL] + SETTING_COLS + SENSOR_COLS
EXPECTED_TELEMETRY_COL_COUNT: int = len(TELEMETRY_COLS)  # 26


def load_raw_cmapss_file(
    filepath: Union[str, Path],
    is_rul: bool = False,
) -> pd.DataFrame:
    """Load and parse a single whitespace-delimited C-MAPSS text file.

    Args:
        filepath: Absolute or relative path to the text file.
        is_rul: If True, treats the file as an RUL ground-truth file (1 column).
            If False, treats it as a telemetry run file (26 columns).

    Returns:
        Cleaned, typed pandas DataFrame with standardized column names.

    Raises:
        FileNotFoundError: If the specified file does not exist on disk.
        ValueError: If file is empty or column counts deviate from the standard schema.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(
            f"C-MAPSS data file not found at: '{path.as_posix()}'. "
            f"Please verify that the raw file exists."
        )

    logger.debug(f"Loading raw file: {path.name}")

    try:
        # Load raw data handling arbitrary whitespace
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            engine="python",
        )
    except Exception as exc:
        raise ValueError(f"Failed to parse text file '{path.name}': {exc}") from exc

    if df.empty:
        raise ValueError(f"C-MAPSS file '{path.name}' is empty.")

    # Some C-MAPSS raw files have trailing spaces resulting in trailing NaN columns
    # Drop columns that are completely NaN
    df = df.dropna(axis=1, how="all")

    if is_rul:
        if df.shape[1] != 1:
            raise ValueError(
                f"Unexpected column count for RUL file '{path.name}': "
                f"expected 1 column, found {df.shape[1]} columns."
            )
        df.columns = ["rul"]
        # In C-MAPSS, RUL values are non-negative integers
        df["rul"] = pd.to_numeric(df["rul"], errors="raise").astype(float)
        # Unit ID for RUL corresponds 1-indexed to the row index
        df["unit_id"] = np.arange(1, len(df) + 1, dtype=int)
        df = df[["unit_id", "rul"]]
        return df

    if df.shape[1] != EXPECTED_TELEMETRY_COL_COUNT:
        raise ValueError(
            f"Schema mismatch in C-MAPSS telemetry file '{path.name}': "
            f"expected {EXPECTED_TELEMETRY_COL_COUNT} columns, found {df.shape[1]} columns."
        )

    df.columns = TELEMETRY_COLS

    # Cast engine unit_id and cycle to integer, settings and sensors to float
    df[ID_COL] = pd.to_numeric(df[ID_COL], errors="raise").astype(int)
    df[CYCLE_COL] = pd.to_numeric(df[CYCLE_COL], errors="raise").astype(int)

    for col in SETTING_COLS + SENSOR_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)

    logger.info(
        f"Successfully loaded '{path.name}': {len(df):,} rows, "
        f"{df[ID_COL].nunique()} unique engines."
    )
    return df


def load_cmapss_dataset(
    data_dir: Union[str, Path],
    variant: str = "FD004",
) -> Dict[str, pd.DataFrame]:
    """Load the complete C-MAPSS train, test, and RUL files for a given variant.

    Args:
        data_dir: Directory path containing the raw text files.
        variant: Dataset sub-identifier ('FD001', 'FD002', 'FD003', 'FD004').

    Returns:
        Dictionary with keys 'train', 'test', and 'rul'.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    dir_path = Path(data_dir)
    train_path = dir_path / f"train_{variant}.txt"
    test_path = dir_path / f"test_{variant}.txt"
    rul_path = dir_path / f"RUL_{variant}.txt"

    train_df = load_raw_cmapss_file(train_path, is_rul=False)
    test_df = load_raw_cmapss_file(test_path, is_rul=False)
    rul_df = load_raw_cmapss_file(rul_path, is_rul=True)

    return {
        "train": train_df,
        "test": test_df,
        "rul": rul_df,
    }
