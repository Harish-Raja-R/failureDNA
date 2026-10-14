"""Engine-level data partitioning module.

Strictly enforces zero data leakage by partitioning data at the engine trajectory
unit level rather than randomly sampling temporal windows.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from failuredna.data.loader import CYCLE_COL, ID_COL
from failuredna.utils.logging import get_logger

logger = get_logger(__name__)


def split_engines_by_unit(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Partition a C-MAPSS DataFrame into disjoint train, val, and test subsets by engine ID.

    Guarantees:
    1. Every engine's entire trajectory belongs to exactly one partition.
    2. Engine IDs across train, val, and test partitions are strictly pairwise disjoint.
    3. Trajectory cycle ordering is preserved.
    4. Deterministic and reproducible partition assignment governed by seed.

    Args:
        df: Input DataFrame containing telemetry data.
        train_ratio: Proportion of engines assigned to training (default: 0.70).
        val_ratio: Proportion of engines assigned to validation (default: 0.15).
        test_ratio: Proportion of engines assigned to internal test (default: 0.15).
        seed: Random seed for deterministic shuffling.

    Returns:
        Tuple containing:
            - train_df: DataFrame with training engine trajectories
            - val_df: DataFrame with validation engine trajectories
            - test_df: DataFrame with internal test engine trajectories
            - split_info: Dictionary documenting split sizes, engine IDs, and disjointness check

    Raises:
        ValueError: If partition ratios do not sum to 1.0 or if dataset has too few engines.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-5):
        raise ValueError(
            f"Split ratios must sum to 1.0 (got {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio})"
        )

    unique_units = np.array(sorted(df[ID_COL].unique()))
    n_engines = len(unique_units)

    if n_engines < 3:
        raise ValueError(f"At least 3 distinct engines required to perform 3-way split; found {n_engines}.")

    # Deterministic shuffling of unit IDs
    rng = np.random.RandomState(seed)
    shuffled_units = unique_units.copy()
    rng.shuffle(shuffled_units)

    # Compute partition counts
    n_train = int(np.round(n_engines * train_ratio))
    n_val = int(np.round(n_engines * val_ratio))
    # Ensure all splits have at least 1 engine
    n_train = max(1, n_train)
    n_val = max(1, n_val)
    n_test = n_engines - n_train - n_val

    if n_test <= 0:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train -= 1

    train_ids = set(shuffled_units[:n_train].tolist())
    val_ids = set(shuffled_units[n_train : n_train + n_val].tolist())
    test_ids = set(shuffled_units[n_train + n_val :].tolist())

    # Strict Mathematical Verification of Disjoint Sets
    assert len(train_ids & val_ids) == 0, "FATAL LEAKAGE: Overlap detected between train and val engine IDs."
    assert len(train_ids & test_ids) == 0, "FATAL LEAKAGE: Overlap detected between train and test engine IDs."
    assert len(val_ids & test_ids) == 0, "FATAL LEAKAGE: Overlap detected between val and test engine IDs."
    assert (train_ids | val_ids | test_ids) == set(unique_units), "FATAL: Engines lost during split partitioning."

    # Filter subsets preserving chronological cycle order
    train_df = df[df[ID_COL].isin(train_ids)].sort_values([ID_COL, CYCLE_COL]).reset_index(drop=True)
    val_df = df[df[ID_COL].isin(val_ids)].sort_values([ID_COL, CYCLE_COL]).reset_index(drop=True)
    test_df = df[df[ID_COL].isin(test_ids)].sort_values([ID_COL, CYCLE_COL]).reset_index(drop=True)

    split_info = {
        "seed": seed,
        "total_engines": n_engines,
        "n_train_engines": len(train_ids),
        "n_val_engines": len(val_ids),
        "n_test_engines": len(test_ids),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_engine_ids": sorted(list(train_ids)),
        "val_engine_ids": sorted(list(val_ids)),
        "test_engine_ids": sorted(list(test_ids)),
        "disjointness_verified": True,
    }

    logger.info(
        f"Seeded engine split completed (seed={seed}): "
        f"{len(train_ids)} train ({len(train_df):,} rows), "
        f"{len(val_ids)} val ({len(val_df):,} rows), "
        f"{len(test_ids)} test ({len(test_df):,} rows)."
    )

    return train_df, val_df, test_df, split_info
