"""Prognostic evaluation metrics module for C-MAPSS RUL benchmarking.

Includes MAE, RMSE, R2, and the official NASA asymmetric scoring function.
"""

from typing import Dict, Union

import numpy as np


def nasa_scoring_function(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """Compute official NASA C-MAPSS asymmetric prognostic scoring function.

    Formula:
        d_i = y_pred_i - y_true_i
        s_i = exp(-d_i / 13) - 1  if d_i < 0 (late prediction: underestimating degradation, heavily penalized)
        s_i = exp( d_i / 10) - 1  if d_i >= 0 (early prediction: overestimating degradation)
        S = sum(s_i)

    Optimization direction: Lower is better (0.0 is perfect score).

    Args:
        y_true: Ground-truth target values (RUL).
        y_pred: Model predictions.

    Returns:
        Scalar NASA asymmetric score (float).
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()

    if len(y_t) != len(y_p):
        raise ValueError(f"Shape mismatch: y_true has {len(y_t)} elements, y_pred has {len(y_p)} elements.")

    diff = y_p - y_t
    # Vectorized piecewise evaluation
    penalties = np.where(
        diff < 0,
        np.exp(-diff / 13.0) - 1.0,
        np.exp(diff / 10.0) - 1.0,
    )

    return float(np.sum(penalties))


def compute_regression_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> Dict[str, float]:
    """Compute comprehensive prognostic regression evaluation metrics.

    Args:
        y_true: Ground-truth RUL targets.
        y_pred: Model RUL predictions.

    Returns:
        Dictionary with 'mae', 'rmse', 'r2', and 'nasa_score'.
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()

    if len(y_t) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")
    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: {len(y_t)} vs {len(y_p)}.")

    mae = float(np.mean(np.abs(y_p - y_t)))
    mse = float(np.mean((y_p - y_t) ** 2))
    rmse = float(np.sqrt(mse))

    # R2 score
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    ss_res = float(np.sum((y_t - y_p) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    nasa_score = nasa_scoring_function(y_t, y_p)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "nasa_score": nasa_score,
    }
