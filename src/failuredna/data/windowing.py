"""Sliding temporal window extraction module.

Generates temporal sequences of shape (Batch, Window_Size, Feature_Dim) from
multivariate engine trajectories with strict single-engine boundary enforcement.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from failuredna.data.loader import CYCLE_COL, ID_COL
from failuredna.utils.logging import get_logger

logger = get_logger(__name__)


class WindowGenerator:
    """Extracts multivariate sliding windows strictly within engine trajectory boundaries."""

    def __init__(
        self,
        window_size: int = 50,
        stride: int = 1,
        feature_cols: Optional[List[str]] = None,
        drop_short_trajectories: bool = True,
        padding_mode: Optional[str] = None,
    ):
        """Initialize WindowGenerator.

        Args:
            window_size: Length of the temporal context window $W$ (default: 50).
            stride: Step size between consecutive sliding windows (default: 1).
            feature_cols: List of sensor/setting column names to extract.
            drop_short_trajectories: If True, skips trajectories with total length < window_size
                when padding_mode is None. If False, raises ValueError.
            padding_mode: Optional left-padding strategy for trajectories with length < window_size.
                Options: None (default), 'edge' (repeat initial observation), 'zero' (zero padding).
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}.")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}.")
        if padding_mode is not None and padding_mode not in ("edge", "zero"):
            raise ValueError(f"padding_mode must be None, 'edge', or 'zero', got '{padding_mode}'.")

        self.window_size = window_size
        self.stride = stride
        self.feature_cols = feature_cols
        self.drop_short_trajectories = drop_short_trajectories
        self.padding_mode = padding_mode

    def create_windows(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
        """Generate sliding windows from engine trajectories.

        Guarantees:
        1. Every window is composed exclusively of observations from a single engine.
        2. Strict chronological ordering is preserved within each window.
        3. No window crosses engine or split boundaries.
        4. Exhaustive metadata is tracked for every window (unit_id, start_cycle, end_cycle).
        5. Left-padding is applied strictly on the past without introducing future information.

        Args:
            df: DataFrame containing telemetry data (raw or preprocessed).

        Returns:
            Tuple containing:
                - windows: 3D float32 numpy array of shape (N_windows, window_size, N_features)
                - metadata: DataFrame of shape (N_windows, 5) mapping window index to (unit_id, start_cycle, end_cycle, is_padded, pad_length)
                - stats: Summary dictionary of the window extraction process

        Raises:
            ValueError: If a trajectory has length < window_size and drop_short_trajectories is False (with padding_mode=None).
        """
        if df.empty:
            raise ValueError("Cannot generate windows from an empty DataFrame.")

        # Determine feature columns if not provided
        cols = self.feature_cols
        if cols is None:
            # Exclude metadata columns
            cols = [c for c in df.columns if c not in [ID_COL, CYCLE_COL]]

        if not cols:
            raise ValueError("No feature columns available to extract temporal windows.")

        window_list: List[np.ndarray] = []
        meta_records: List[Dict[str, Any]] = []
        short_engines: List[int] = []
        padded_engines: List[int] = []
        total_engines = 0

        # Process each engine independently to guarantee boundary separation
        for unit_id, group in df.groupby(ID_COL, sort=True):
            total_engines += 1
            # Ensure chronological ordering
            sorted_group = group.sort_values(CYCLE_COL).reset_index(drop=True)
            n_cycles = len(sorted_group)

            if n_cycles < self.window_size:
                if self.padding_mode is None:
                    short_engines.append(int(unit_id))
                    if not self.drop_short_trajectories:
                        raise ValueError(
                            f"Engine {unit_id} trajectory length ({n_cycles}) is shorter "
                            f"than window_size ({self.window_size})."
                        )
                    continue
                else:
                    # Apply left-padding strategy
                    padded_engines.append(int(unit_id))
                    pad_len = self.window_size - n_cycles
                    raw_feats = sorted_group[cols].to_numpy(dtype=np.float32)

                    if self.padding_mode == "edge":
                        # Left-pad by repeating the earliest observed state
                        pad_block = np.repeat(raw_feats[[0]], pad_len, axis=0)
                    else:  # zero
                        pad_block = np.zeros((pad_len, len(cols)), dtype=np.float32)

                    padded_feats = np.vstack([pad_block, raw_feats])
                    window_list.append(padded_feats)
                    meta_records.append(
                        {
                            "unit_id": int(unit_id),
                            "start_cycle": int(sorted_group[CYCLE_COL].iloc[0]),
                            "end_cycle": int(sorted_group[CYCLE_COL].iloc[-1]),
                            "is_padded": True,
                            "pad_length": int(pad_len),
                        }
                    )
                    continue

            feature_matrix = sorted_group[cols].to_numpy(dtype=np.float32)
            cycle_array = sorted_group[CYCLE_COL].to_numpy(dtype=np.int64)

            # Slide window across the engine trajectory
            max_start = n_cycles - self.window_size
            for start_idx in range(0, max_start + 1, self.stride):
                end_idx = start_idx + self.window_size
                window = feature_matrix[start_idx:end_idx]

                window_list.append(window)
                meta_records.append(
                    {
                        "unit_id": int(unit_id),
                        "start_cycle": int(cycle_array[start_idx]),
                        "end_cycle": int(cycle_array[end_idx - 1]),
                        "is_padded": False,
                        "pad_length": 0,
                    }
                )

        if not window_list:
            logger.warning(
                f"No windows could be generated! Processed {total_engines} engines; "
                f"all were shorter than window_size={self.window_size}."
            )
            windows_arr = np.empty((0, self.window_size, len(cols)), dtype=np.float32)
            meta_df = pd.DataFrame(
                columns=["window_idx", "unit_id", "start_cycle", "end_cycle", "is_padded", "pad_length"]
            )
        else:
            windows_arr = np.stack(window_list, axis=0)
            meta_df = pd.DataFrame(meta_records)
            meta_df.insert(0, "window_idx", np.arange(len(meta_df), dtype=int))

        stats = {
            "window_size": self.window_size,
            "stride": self.stride,
            "padding_mode": self.padding_mode,
            "feature_dim": len(cols),
            "feature_names": cols,
            "total_engines_examined": total_engines,
            "engines_with_windows": total_engines - len(short_engines),
            "short_engines_skipped": short_engines,
            "padded_engines": padded_engines,
            "total_windows_generated": len(windows_arr),
        }

        logger.info(
            f"Extracted {len(windows_arr):,} temporal windows of shape "
            f"({self.window_size}, {len(cols)}) across {stats['engines_with_windows']} engines. "
            f"Skipped {len(short_engines)} short engines, padded {len(padded_engines)} engines."
        )

        return windows_arr, meta_df, stats

    def extract_last_window(
        self,
        df: pd.DataFrame,
        pad: bool = True,
        padding_mode: str = "edge",
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Extract exactly one final context window ending at cycle T_max for every engine.

        Specifically designed for official C-MAPSS test set evaluation where every
        test engine trajectory must produce an RUL prediction at its final observation.

        Args:
            df: DataFrame containing telemetry observations for test engines.
            pad: If True, left-pads engines with trajectory length < window_size.
            padding_mode: 'edge' (repeat initial cycle) or 'zero' (zero padding).

        Returns:
            Tuple containing:
                - windows: 3D float32 array of shape (N_engines, window_size, N_features).
                - meta: DataFrame with unit_id, last_cycle, is_padded, pad_length.
        """
        cols = self.feature_cols
        if cols is None:
            cols = [c for c in df.columns if c not in [ID_COL, CYCLE_COL]]

        windows = []
        meta_list = []

        for unit_id, group in df.groupby(ID_COL, sort=True):
            sorted_group = group.sort_values(CYCLE_COL).reset_index(drop=True)
            n_cycles = len(sorted_group)
            last_cycle = int(sorted_group[CYCLE_COL].iloc[-1])

            if n_cycles >= self.window_size:
                # Extract the last window_size observations
                sub = sorted_group.iloc[-self.window_size :][cols].to_numpy(dtype=np.float32)
                windows.append(sub)
                meta_list.append(
                    {
                        "unit_id": int(unit_id),
                        "last_cycle": last_cycle,
                        "is_padded": False,
                        "pad_length": 0,
                    }
                )
            else:
                if not pad:
                    raise ValueError(
                        f"Engine {unit_id} has {n_cycles} cycles < window_size={self.window_size} "
                        f"and pad=False."
                    )
                pad_len = self.window_size - n_cycles
                raw = sorted_group[cols].to_numpy(dtype=np.float32)
                if padding_mode == "edge":
                    pad_block = np.repeat(raw[[0]], pad_len, axis=0)
                else:
                    pad_block = np.zeros((pad_len, len(cols)), dtype=np.float32)

                padded = np.vstack([pad_block, raw])
                windows.append(padded)
                meta_list.append(
                    {
                        "unit_id": int(unit_id),
                        "last_cycle": last_cycle,
                        "is_padded": True,
                        "pad_length": int(pad_len),
                    }
                )

        windows_arr = np.stack(windows, axis=0)
        meta_df = pd.DataFrame(meta_list)
        return windows_arr, meta_df
