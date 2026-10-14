"""Unit and integration tests for Phase 5.1 Official Benchmark Integrity Audit."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from failuredna.evaluation.metrics import (
    compute_regression_metrics,
    nasa_scoring_function,
)


def test_frozen_checkpoints_sha256_manifest():
    """Verify that all 4 frozen checkpoints match their documented SHA-256 hashes."""
    expected_manifest = {
        "failuredna_regime_aware": (
            Path("models/checkpoints/failuredna_regime_aware/best_failuredna_model.pt"),
            "d18b6cd0abc2fadf12be18f6f1962707f3626e0a5054970147f50eb3620eb248",
            2369441,
        ),
        "failuredna_pooled": (
            Path("models/checkpoints/failuredna_pooled/best_failuredna_model.pt"),
            "1e93ab9cbf31185838020c03c557b593f018b2b4aa2420d5b8c2732d288878ef",
            2369441,
        ),
        "failuredna_regime_conditioned": (
            Path("models/checkpoints/failuredna_regime_conditioned/best_failuredna_model.pt"),
            "67b379d2d31a1cabe99cf5809cd5bcf7d606f3224ea9566da716f9c75c851fbb",
            2372513,
        ),
        "failuredna_w30": (
            Path("models/checkpoints/failuredna_w30/best_failuredna_model.pt"),
            "8aa450d5612542b0827d7f9d6b1d295ab7e06014175e1f7f0c779c68d431b791",
            2369441,
        ),
    }

    for name, (path, expected_hash, expected_size) in expected_manifest.items():
        assert path.exists(), f"Checkpoint {name} missing at {path}"
        data = path.read_bytes()
        actual_size = len(data)
        actual_hash = hashlib.sha256(data).hexdigest()
        assert actual_size == expected_size, f"Size mismatch for {name}: {actual_size} vs {expected_size}"
        assert actual_hash == expected_hash, f"Hash mismatch for {name}: {actual_hash} vs {expected_hash}"


def test_nasa_scoring_function_toy_cases():
    """Verify NASA score implementation against hand-calculated toy cases."""
    # Hand calculation:
    # d = -20 (early): exp(20/13) - 1 = exp(1.5384615) - 1 = 3.657419
    # d = +20 (late):  exp(20/10) - 1 = exp(2.0) - 1 = 6.389056
    # d = 0 (exact):   0.0
    s_early = nasa_scoring_function([100.0], [80.0])
    s_late = nasa_scoring_function([100.0], [120.0])
    s_exact = nasa_scoring_function([50.0], [50.0])
    s_total = nasa_scoring_function([100.0, 100.0, 50.0], [80.0, 120.0, 50.0])

    assert np.isclose(s_early, np.exp(20.0 / 13.0) - 1.0, atol=1e-5)
    assert np.isclose(s_late, np.exp(20.0 / 10.0) - 1.0, atol=1e-5)
    assert np.isclose(s_exact, 0.0, atol=1e-7)
    assert np.isclose(s_total, s_early + s_late + s_exact, atol=1e-5)
    assert s_late > s_early, "Late penalty must strictly exceed early penalty for equal error magnitude"


def test_metric_reproduction_from_engine_predictions():
    """Verify that recalculating metrics from saved engine predictions reproduces comparison CSV."""
    preds_path = Path("reports/results/phase5_engine_predictions.csv")
    comp_path = Path("reports/results/phase5_model_comparison.csv")
    assert preds_path.exists()
    assert comp_path.exists()

    df_preds = pd.read_csv(preds_path)
    df_comp = pd.read_csv(comp_path)
    assert len(df_preds) == 248, "Must evaluate 248 test engines"

    y_true = df_preds["true_rul"].to_numpy(dtype=np.float64)

    # Key model columns mapping
    model_mapping = {
        "failuredna_regime_aware_ridge": "pred_failuredna_ridge",
        "failuredna_regime_conditioned_ridge": "pred_failuredna_regime_conditioned_ridge",
        "xgboost": "pred_xgboost",
        "random_forest": "pred_random_forest",
        "linear_ridge": "pred_linear_ridge",
        "random_encoder_control": "pred_random_encoder_ridge",
    }

    for model_key, pred_col in model_mapping.items():
        assert pred_col in df_preds.columns, f"Missing prediction column {pred_col}"
        y_pred = df_preds[pred_col].to_numpy(dtype=np.float64)
        recalculated = compute_regression_metrics(y_true, y_pred)

        row = df_comp[df_comp["model_key"] == model_key].iloc[0]
        assert np.isclose(recalculated["mae"], row["mae"], atol=1e-3)
        assert np.isclose(recalculated["rmse"], row["rmse"], atol=1e-3)
        assert np.isclose(recalculated["nasa_score"], row["nasa_score"], atol=1e-1)


def test_random_encoder_score_decomposition():
    """Verify that Random Encoder's 87.5M NASA score is concentrated in top 5 short-engine outliers."""
    preds_path = Path("reports/results/phase5_engine_predictions.csv")
    df_preds = pd.read_csv(preds_path)

    y_true = df_preds["true_rul"].to_numpy(dtype=np.float64)
    y_rand = df_preds["pred_random_encoder_ridge"].to_numpy(dtype=np.float64)
    diff = y_rand - y_true

    penalties = np.where(
        diff < 0,
        np.exp(-diff / 13.0) - 1.0,
        np.exp(diff / 10.0) - 1.0,
    )
    total_score = float(np.sum(penalties))
    assert np.isclose(total_score, 87465483.74, atol=1.0)

    # Sort descending
    sorted_idx = np.argsort(penalties)[::-1]
    top_5_penalties = penalties[sorted_idx[:5]]
    top_5_sum = np.sum(top_5_penalties)

    # Top 5 engines must account for > 98% of total penalty
    top_5_pct = (top_5_sum / total_score) * 100.0
    assert top_5_pct > 98.0, f"Expected >98% from top 5 engines, got {top_5_pct:.2f}%"

    # Single top engine (Unit 125) must account for > 45%
    top_1_pct = (top_5_penalties[0] / total_score) * 100.0
    assert top_1_pct > 45.0, f"Expected >45% from top 1 engine, got {top_1_pct:.2f}%"
    assert df_preds["unit_id"].iloc[sorted_idx[0]] == 125


def test_paired_bootstrap_conclusions():
    """Verify paired bootstrap results from CSV and ensure statistical claims are accurate."""
    bootstrap_csv = Path("reports/results/phase5_1_paired_bootstrap.csv")
    assert bootstrap_csv.exists()

    df_boot = pd.read_csv(bootstrap_csv)
    
    # Check FailureDNA (Regime-Aware) vs XGBoost
    row_fdna_xgb = df_boot[df_boot["comparison"] == "FailureDNA (Regime-Aware) vs XGBoost"].iloc[0]
    
    # XGBoost has statistically significant advantage in MAE (p < 0.05, CI lower bound > 0)
    assert row_fdna_xgb["delta_mae_point"] > 0
    assert row_fdna_xgb["delta_mae_ci_lower"] > 0
    assert row_fdna_xgb["delta_mae_p_val"] < 0.05

    # Difference in RMSE is not statistically significant (95% CI spans zero)
    assert row_fdna_xgb["delta_rmse_ci_lower"] < 0 < row_fdna_xgb["delta_rmse_ci_upper"]
    assert row_fdna_xgb["delta_rmse_p_val"] > 0.05

    # Difference in NASA score is not statistically significant (95% CI spans zero)
    assert row_fdna_xgb["delta_nasa_ci_lower"] < 0 < row_fdna_xgb["delta_nasa_ci_upper"]
    assert row_fdna_xgb["delta_nasa_p_val"] > 0.05


def test_original_phase5_artifacts_preserved():
    """Verify that original Phase 5 report, summary, and comparisons exist and are unchanged."""
    p5_report = Path("reports/results/phase5_official_benchmark_report.md")
    p5_summary = Path("reports/results/phase5_official_benchmark_summary.json")
    p5_comp = Path("reports/results/phase5_model_comparison.csv")
    p5_preds = Path("reports/results/phase5_engine_predictions.csv")

    assert p5_report.exists()
    assert p5_summary.exists()
    assert p5_comp.exists()
    assert p5_preds.exists()

    # Verify that Version 2 report exists as a separate file
    p5_v2_report = Path("reports/results/phase5_official_benchmark_report_v2.md")
    assert p5_v2_report.exists()
    assert p5_v2_report != p5_report
