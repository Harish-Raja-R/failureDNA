"""Downstream prognostic evaluation probes on frozen FailureDNA embeddings.

Evaluates linear/ridge regression, Random Forest, and XGBoost on frozen embeddings.
Measures training time, inference time, and comprehensive metrics (MAE, RMSE, R2, NASA score).
"""

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

from failuredna.evaluation.metrics import compute_regression_metrics
from failuredna.utils.logging import get_logger

logger = get_logger(__name__)


def evaluate_probe(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    probe_name: str = "Probe",
) -> Dict[str, Any]:
    """Train downstream probe on frozen embeddings and evaluate performance.

    Args:
        model: Estimator implementing fit() and predict().
        X_train: Frozen embeddings for training windows (N_train, D_emb).
        y_train: Target RUL for training windows.
        X_val: Frozen embeddings for validation windows.
        y_val: Target RUL for validation windows.
        X_test: Optional internal test embeddings.
        y_test: Optional internal test target RUL.
        probe_name: Human-readable identifier.

    Returns:
        Dictionary containing validation and test metrics, fit time, and inference time.
    """
    logger.info(f"Fitting downstream probe '{probe_name}' on {X_train.shape[0]} training embeddings...")

    # Measure fit time
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time_sec = time.perf_counter() - t0

    # Measure inference time on validation set
    t1 = time.perf_counter()
    y_pred_val = model.predict(X_val)
    val_inf_time_sec = time.perf_counter() - t1

    val_metrics = compute_regression_metrics(y_val, y_pred_val)

    results = {
        "probe_name": probe_name,
        "fit_time_seconds": float(fit_time_sec),
        "inference_time_seconds": float(val_inf_time_sec),
        "val_metrics": val_metrics,
        "val_predictions": y_pred_val.tolist() if len(y_pred_val) < 200 else None,
    }

    if X_test is not None and y_test is not None:
        y_pred_test = model.predict(X_test)
        test_metrics = compute_regression_metrics(y_test, y_pred_test)
        results["internal_test_metrics"] = test_metrics

    logger.info(
        f"Probe '{probe_name}' Val Metrics: "
        f"RMSE={val_metrics['rmse']:.3f}, MAE={val_metrics['mae']:.3f}, "
        f"R2={val_metrics['r2']:.3f}, NASA Score={val_metrics['nasa_score']:.1f} "
        f"(fit: {fit_time_sec:.2f}s, inf: {val_inf_time_sec:.4f}s)"
    )

    return results


def run_all_embedding_probes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Run Ridge, Random Forest, and XGBoost probes on frozen embeddings."""
    probes_results = {}

    # 1. Ridge Linear Probe
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    probes_results["ridge"] = evaluate_probe(
        ridge_model, X_train, y_train, X_val, y_val, X_test, y_test, probe_name="Ridge Regression Probe"
    )

    # 2. Random Forest Probe
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=seed,
        n_jobs=-1,
    )
    probes_results["random_forest"] = evaluate_probe(
        rf_model, X_train, y_train, X_val, y_val, X_test, y_test, probe_name="Random Forest Probe"
    )

    # 3. XGBoost Probe
    xgb_model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    probes_results["xgboost"] = evaluate_probe(
        xgb_model, X_train, y_train, X_val, y_val, X_test, y_test, probe_name="XGBoost Probe"
    )

    return probes_results
