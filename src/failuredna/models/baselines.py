"""Supervised and non-neural prognostic baselines for C-MAPSS FD004.

Implements:
1. Operating-condition-only baseline (quantifies information from flight settings alone).
2. Linear / Ridge Regression on summary window statistics.
3. Random Forest on summary window statistics.
4. XGBoost on summary window statistics.
5. Deep Recurrent Sequential Baselines (LSTM and GRU).
6. Target RUL computation with configurable piecewise-linear capping.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import xgboost as xgb

from failuredna.data.loader import CYCLE_COL, ID_COL
from failuredna.evaluation.metrics import compute_regression_metrics
from failuredna.utils.logging import get_logger

logger = get_logger(__name__)


def compute_window_ruls(
    window_meta: pd.DataFrame,
    full_df: pd.DataFrame,
    cap_rul: Optional[float] = 125.0,
) -> np.ndarray:
    """Compute ground-truth RUL target for each temporal window.

    In run-to-failure trajectories (training, validation, internal test partitions),
    RUL at cycle t is defined as:
        RUL(t) = T_max - t
    where T_max is the final observed failure cycle of that engine, and t is
    end_cycle of the current context window.

    Piecewise-Linear Capping:
        Industrial machinery in early operational life shows virtually zero physical wear.
        Applying an upper cap (e.g., RUL_cap = 125) prevents models from predicting
        arbitrary large values in healthy states where no failure signal exists.

    Args:
        window_meta: Metadata DataFrame containing 'unit_id' and 'end_cycle'.
        full_df: DataFrame containing the complete trajectories to determine T_max per unit.
        cap_rul: Upper limit for piecewise-linear RUL clipping (default: 125.0).

    Returns:
        1D float32 numpy array of target RUL values.
    """
    # Compute maximum cycle per engine
    lifespans = full_df.groupby(ID_COL)[CYCLE_COL].max().to_dict()

    ruls = []
    for _, row in window_meta.iterrows():
        unit_id = int(row["unit_id"])
        end_cycle = int(row["end_cycle"])
        max_cycle = lifespans[unit_id]

        true_rul = float(max_cycle - end_cycle)
        if cap_rul is not None:
            true_rul = min(true_rul, float(cap_rul))
        ruls.append(true_rul)

    return np.array(ruls, dtype=np.float32)


def extract_window_summary_features(
    windows: np.ndarray,
    feature_names: Optional[List[str]] = None,
    include_min_max: bool = True,
) -> np.ndarray:
    """Extract summary tabular features from 3D window array of shape (N, W, D).

    Computes per-window summary statistics:
    - Mean across time: (N, D)
    - Std across time: (N, D)
    - Last observed step: (N, D)
    - Min and Max across time: (N, D), (N, D) (optional)

    Returns:
        2D float32 array of shape (N, 3D) or (N, 5D).
    """
    N, W, D = windows.shape

    mean_feats = np.mean(windows, axis=1)  # (N, D)
    std_feats = np.std(windows, axis=1)    # (N, D)
    last_feats = windows[:, -1, :]         # (N, D)

    feat_blocks = [mean_feats, std_feats, last_feats]

    if include_min_max:
        min_feats = np.min(windows, axis=1)
        max_feats = np.max(windows, axis=1)
        feat_blocks.extend([min_feats, max_feats])

    return np.concatenate(feat_blocks, axis=1).astype(np.float32)


class OperatingConditionOnlyBaseline:
    """Baseline using only operational settings to establish the environmental information floor."""

    def __init__(self, n_setting_cols: int = 3, alpha: float = 10.0):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.n_setting_cols = n_setting_cols
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, windows: np.ndarray, y: np.ndarray) -> "OperatingConditionOnlyBaseline":
        # Extract setting channels only (first n_setting_cols)
        settings_only = windows[:, :, : self.n_setting_cols]
        # Mean across window
        X = np.mean(settings_only, axis=1)
        self.model.fit(X, y)
        return self

    def predict(self, windows: np.ndarray) -> np.ndarray:
        settings_only = windows[:, :, : self.n_setting_cols]
        X = np.mean(settings_only, axis=1)
        return self.model.predict(X)


class LinearWindowBaseline:
    """Linear / Ridge Regression baseline on window summary statistics."""

    def __init__(self, alpha: float = 10.0):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, X_summary: np.ndarray, y: np.ndarray) -> "LinearWindowBaseline":
        self.model.fit(X_summary, y)
        return self

    def predict(self, X_summary: np.ndarray) -> np.ndarray:
        return self.model.predict(X_summary)


class RandomForestBaseline:
    """Random Forest regressor on window summary statistics."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_summary: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        self.model.fit(X_summary, y)
        return self

    def predict(self, X_summary: np.ndarray) -> np.ndarray:
        return self.model.predict(X_summary)


class XGBoostBaseline:
    """XGBoost regressor on window summary statistics."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        random_state: int = 42,
    ):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostBaseline":
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict(self, X_summary: np.ndarray) -> np.ndarray:
        return self.model.predict(X_summary)


class RecurrentRULBaseline(nn.Module):
    """Deep recurrent sequence baseline (LSTM or GRU) for supervised RUL prediction."""

    def __init__(
        self,
        input_dim: int = 24,
        hidden_dim: int = 128,
        num_layers: int = 2,
        cell_type: str = "lstm",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cell_type = cell_type.lower()
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if self.cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unknown cell_type '{cell_type}', choose 'lstm' or 'gru'.")

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass: maps (B, W, D) -> scalar RUL prediction (B,)."""
        h = self.input_proj(x)
        out, _ = self.rnn(h)  # (B, W, hidden_dim)

        if mask is not None:
            # Use masked average pooling
            mask_expanded = mask.unsqueeze(-1).to(x.device).float()
            pooled = torch.sum(out * mask_expanded, dim=1) / torch.clamp(torch.sum(mask_expanded, dim=1), min=1.0)
        else:
            pooled = out[:, -1, :]  # Last time step

        rul_pred = self.head(pooled).squeeze(-1)
        return rul_pred
