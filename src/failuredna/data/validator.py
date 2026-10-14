"""C-MAPSS Data Validation and Statistical Audit Engine.

Enforces zero-tolerance validation on schema integrity, temporal monotonicity,
and absence of missing/corrupt values.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from failuredna.data.loader import CYCLE_COL, ID_COL, SENSOR_COLS, SETTING_COLS, TELEMETRY_COLS
from failuredna.utils.logging import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when telemetry or RUL data fails schema or integrity checks."""
    pass


class DataValidator:
    """Validates C-MAPSS telemetry datasets and computes comprehensive statistical audits."""

    def __init__(self, raise_on_error: bool = True):
        """Initialize validator.

        Args:
            raise_on_error: If True, raises ValidationError on first failure.
                If False, logs error and returns a diagnostic dictionary.
        """
        self.raise_on_error = raise_on_error

    def validate_telemetry_df(
        self,
        df: pd.DataFrame,
        dataset_name: str = "telemetry",
    ) -> Dict[str, Any]:
        """Perform comprehensive integrity checks on a C-MAPSS telemetry DataFrame.

        Checks:
        1. Required columns presence and count (26 standard columns).
        2. No NaN, null, or missing values.
        3. No infinite (inf, -inf) values.
        4. No duplicate (unit_id, cycle) records.
        5. Valid positive unit IDs.
        6. Strictly monotonic increasing cycle sequences per engine.

        Args:
            df: DataFrame to validate.
            dataset_name: Identifier for error reporting.

        Returns:
            Dictionary detailing validation results and statistics.

        Raises:
            ValidationError: If any check fails and raise_on_error is True.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Column Presence Check
        missing_cols = [c for c in TELEMETRY_COLS if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns in '{dataset_name}': {missing_cols}")

        if errors and self.raise_on_error:
            raise ValidationError("; ".join(errors))

        # 2. Missing Value Check
        null_counts = df[TELEMETRY_COLS].isna().sum()
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        if cols_with_nulls:
            errors.append(f"Null values detected in '{dataset_name}': {cols_with_nulls}")

        # 3. Non-Finite Value Check
        numeric_subset = df[SETTING_COLS + SENSOR_COLS]
        inf_mask = np.isinf(numeric_subset.to_numpy())
        if inf_mask.any():
            inf_col_indices = np.where(inf_mask.any(axis=0))[0]
            inf_cols = [numeric_subset.columns[i] for i in inf_col_indices]
            errors.append(f"Infinite values detected in '{dataset_name}' columns: {inf_cols}")

        # 4. Duplicate (unit_id, cycle) Check
        duplicates = df.duplicated(subset=[ID_COL, CYCLE_COL], keep=False)
        dup_count = int(duplicates.sum())
        if dup_count > 0:
            dup_examples = df.loc[duplicates, [ID_COL, CYCLE_COL]].head(5).to_dict(orient="records")
            errors.append(
                f"Found {dup_count} duplicate ({ID_COL}, {CYCLE_COL}) entries in '{dataset_name}'. "
                f"Samples: {dup_examples}"
            )

        # 5. Positive Unit ID Check
        if (df[ID_COL] <= 0).any():
            invalid_ids = df.loc[df[ID_COL] <= 0, ID_COL].unique().tolist()
            errors.append(f"Non-positive engine {ID_COL} found in '{dataset_name}': {invalid_ids}")

        # 6. Monotonic Cycle Progression Check per Engine
        non_monotonic_engines = []
        for unit_id, group in df.groupby(ID_COL):
            cycles = group[CYCLE_COL].to_numpy()
            diffs = np.diff(cycles)
            if (diffs <= 0).any():
                non_monotonic_engines.append(int(unit_id))

        if non_monotonic_engines:
            errors.append(
                f"Non-strictly-increasing cycle progression detected in '{dataset_name}' "
                f"for engines: {non_monotonic_engines[:10]}"
            )

        is_valid = len(errors) == 0

        if not is_valid and self.raise_on_error:
            error_msg = f"Validation failed for '{dataset_name}':\n - " + "\n - ".join(errors)
            logger.error(error_msg)
            raise ValidationError(error_msg)

        report = {
            "dataset_name": dataset_name,
            "is_valid": is_valid,
            "total_rows": len(df),
            "num_engines": int(df[ID_COL].nunique()),
            "errors": errors,
            "warnings": warnings,
        }

        logger.info(f"Validation passed for '{dataset_name}' ({len(df):,} rows, {report['num_engines']} engines).")
        return report

    def validate_rul_df(
        self,
        rul_df: pd.DataFrame,
        expected_engine_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Validate an RUL ground-truth DataFrame.

        Args:
            rul_df: DataFrame with ['unit_id', 'rul'].
            expected_engine_count: Expected number of test engines if known.

        Returns:
            Validation report dictionary.

        Raises:
            ValidationError: If RUL file has negative values, NaNs, or count mismatch.
        """
        errors: List[str] = []

        if "rul" not in rul_df.columns:
            errors.append("RUL DataFrame missing 'rul' column.")

        if errors and self.raise_on_error:
            raise ValidationError("; ".join(errors))

        if rul_df["rul"].isna().any():
            errors.append("NaN values found in RUL data.")

        if (rul_df["rul"] < 0).any():
            neg_count = int((rul_df["rul"] < 0).sum())
            errors.append(f"Found {neg_count} negative RUL values.")

        if expected_engine_count is not None and len(rul_df) != expected_engine_count:
            errors.append(
                f"RUL record count ({len(rul_df)}) does not match expected "
                f"engine count ({expected_engine_count})."
            )

        is_valid = len(errors) == 0
        if not is_valid and self.raise_on_error:
            raise ValidationError(f"RUL validation failed: {'; '.join(errors)}")

        return {
            "dataset_name": "RUL",
            "is_valid": is_valid,
            "total_records": len(rul_df),
            "min_rul": float(rul_df["rul"].min()),
            "max_rul": float(rul_df["rul"].max()),
            "median_rul": float(rul_df["rul"].median()),
            "errors": errors,
        }

    def audit_dataset(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        rul_df: pd.DataFrame,
        variance_threshold: float = 1e-5,
    ) -> Dict[str, Any]:
        """Perform a rigorous, publication-grade dataset audit across train, test, and RUL.

        Calculates exact trajectory length quantiles, near-constant sensor variance,
        operating-condition flight envelope distributions, and schema parity.

        Args:
            train_df: Validated training telemetry DataFrame.
            test_df: Validated test telemetry DataFrame.
            rul_df: Validated test RUL DataFrame.
            variance_threshold: Variance cutoff to flag near-constant sensors.

        Returns:
            Nested dictionary containing complete statistical audit metrics.
        """
        # Validate all sets first
        train_val = self.validate_telemetry_df(train_df, "train")
        test_val = self.validate_telemetry_df(test_df, "test")
        rul_val = self.validate_rul_df(rul_df, expected_engine_count=test_df[ID_COL].nunique())

        # Trajectory lengths
        train_lens = train_df.groupby(ID_COL)[CYCLE_COL].max()
        test_lens = test_df.groupby(ID_COL)[CYCLE_COL].max()

        def compute_length_stats(s: pd.Series) -> Dict[str, float]:
            return {
                "count": int(len(s)),
                "min": int(s.min()),
                "q25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "mean": float(s.mean()),
                "q75": float(s.quantile(0.75)),
                "max": int(s.max()),
                "std": float(s.std()),
            }

        train_len_stats = compute_length_stats(train_lens)
        test_len_stats = compute_length_stats(test_lens)

        # Sensor variance on training set
        sensor_variances = train_df[SENSOR_COLS].var().to_dict()
        constant_sensors = [
            sensor for sensor, var in sensor_variances.items() if var <= variance_threshold
        ]
        informative_sensors = [
            sensor for sensor, var in sensor_variances.items() if var > variance_threshold
        ]

        # Operating settings statistics
        settings_stats = {}
        for col in SETTING_COLS:
            settings_stats[col] = {
                "train_min": float(train_df[col].min()),
                "train_max": float(train_df[col].max()),
                "train_mean": float(train_df[col].mean()),
                "train_std": float(train_df[col].std()),
                "test_min": float(test_df[col].min()),
                "test_max": float(test_df[col].max()),
            }

        # Sensor summary statistics
        sensor_stats = {}
        for col in SENSOR_COLS:
            sensor_stats[col] = {
                "train_variance": float(sensor_variances[col]),
                "train_min": float(train_df[col].min()),
                "train_max": float(train_df[col].max()),
                "is_constant": col in constant_sensors,
            }

        # Test engine ID alignment with RUL
        test_engine_ids = sorted(test_df[ID_COL].unique().tolist())
        rul_engine_ids = sorted(rul_df[ID_COL].unique().tolist())
        id_alignment_pass = test_engine_ids == rul_engine_ids

        audit_report = {
            "schema": {
                "train_columns": list(train_df.columns),
                "test_columns": list(test_df.columns),
                "column_count": len(train_df.columns),
                "columns_match": list(train_df.columns) == list(test_df.columns),
            },
            "counts": {
                "train_rows": len(train_df),
                "train_engines": int(train_df[ID_COL].nunique()),
                "test_rows": len(test_df),
                "test_engines": int(test_df[ID_COL].nunique()),
                "rul_records": len(rul_df),
                "test_and_rul_ids_aligned": id_alignment_pass,
            },
            "trajectory_lengths": {
                "train": train_len_stats,
                "test": test_len_stats,
            },
            "sensor_analysis": {
                "total_sensors": len(SENSOR_COLS),
                "variance_threshold": variance_threshold,
                "constant_sensors": constant_sensors,
                "informative_sensors": informative_sensors,
                "informative_count": len(informative_sensors),
                "sensor_details": sensor_stats,
            },
            "operating_settings": settings_stats,
            "rul_statistics": {
                "min": float(rul_df["rul"].min()),
                "q25": float(rul_df["rul"].quantile(0.25)),
                "median": float(rul_df["rul"].median()),
                "q75": float(rul_df["rul"].quantile(0.75)),
                "max": float(rul_df["rul"].max()),
                "mean": float(rul_df["rul"].mean()),
                "std": float(rul_df["rul"].std()),
            },
        }

        logger.info(
            f"Dataset audit completed: {audit_report['counts']['train_engines']} train engines, "
            f"{audit_report['counts']['test_engines']} test engines, "
            f"{len(constant_sensors)} constant sensors flagged."
        )
        return audit_report
