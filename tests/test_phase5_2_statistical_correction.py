"""Unit and integration tests for Phase 5.2 Statistical Inference and Reporting Correction."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


def holm_adjust_reference(p_values):
    """Reference implementation of Holm-Bonferroni step-down adjustment."""
    p_arr = np.asarray(p_values, dtype=float)
    m = len(p_arr)
    order = np.argsort(p_arr)
    sorted_p = p_arr[order]

    adj_sorted = np.zeros(m)
    current_max = 0.0
    for k in range(m):
        multiplier = m - k
        val = min(1.0, multiplier * sorted_p[k])
        current_max = max(current_max, val)
        adj_sorted[k] = current_max

    adj_p = np.zeros(m)
    adj_p[order] = adj_sorted
    return adj_p


def test_holm_step_down_procedure_logic():
    """Verify mathematical correctness of Holm-Bonferroni step-down adjustment."""
    # Synthetic test case
    raw_p = [0.01, 0.04, 0.03]
    # Sorted: 0.01 (m=3 -> 0.03), 0.03 (m=2 -> 0.06), 0.04 (m=1 -> 0.04 -> max(0.06, 0.04) = 0.06)
    expected_adj = [0.03, 0.06, 0.06]
    actual_adj = holm_adjust_reference(raw_p)
    assert np.allclose(actual_adj, expected_adj, atol=1e-7)

    # Monotonicity with respect to rank
    raw_p_complex = [0.001, 0.02, 0.025, 0.05, 0.10, 0.20]
    adj_complex = holm_adjust_reference(raw_p_complex)
    assert np.all(np.diff(adj_complex) >= 0), "Adjusted p-values must be non-decreasing with rank"
    assert np.all(adj_complex <= 1.0), "Adjusted p-values cannot exceed 1.0"
    assert np.all(adj_complex >= raw_p_complex), "Adjusted p-values must be >= raw p-values"


def test_phase5_2_adjusted_bootstrap_csv_integrity():
    """Verify Phase 5.2 adjusted bootstrap CSV values and statistical significance conclusions."""
    csv_path = Path("reports/results/phase5_2_adjusted_bootstrap.csv")
    assert csv_path.exists(), "Missing phase5_2_adjusted_bootstrap.csv"

    df = pd.read_csv(csv_path)
    assert len(df) == 15, "Must evaluate exactly 15 hypothesis tests (5 comparisons x 3 metrics)"

    # Verify FailureDNA (Regime-Aware) vs XGBoost in MAE
    row_ra_xgb_mae = df[(df["comparison"] == "FailureDNA (Regime-Aware) vs XGBoost") & (df["metric"] == "MAE")].iloc[0]
    assert np.isclose(row_ra_xgb_mae["delta_point"], 2.8571, atol=1e-3)
    assert row_ra_xgb_mae["ci_95_lower"] > 0
    assert row_ra_xgb_mae["ci_includes_zero"] == False
    assert row_ra_xgb_mae["p_val_raw"] == 0.0010
    assert np.isclose(row_ra_xgb_mae["p_val_holm_metric"], 0.0050, atol=1e-4)
    assert np.isclose(row_ra_xgb_mae["p_val_holm_omnibus"], 0.0150, atol=1e-4)
    assert row_ra_xgb_mae["fwer_significance_metric_05"] == True
    assert row_ra_xgb_mae["fwer_significance_omnibus_05"] == True

    # Verify that NO OTHER contrast is statistically significant under Holm metric adjustment
    other_rows = df[~((df["comparison"] == "FailureDNA (Regime-Aware) vs XGBoost") & (df["metric"] == "MAE"))]
    assert not other_rows["fwer_significance_metric_05"].any(), (
        "Only FailureDNA (Regime-Aware) vs XGBoost in MAE should be statistically significant under FWER control"
    )

    # Verify that all RMSE and NASA score tests have confidence intervals that include zero
    rmse_nasa_rows = df[df["metric"].isin(["RMSE", "NASA"])]
    assert rmse_nasa_rows["ci_includes_zero"].all(), (
        "All RMSE and NASA score comparisons must include zero in their 95% bootstrap CI"
    )


def test_nasa_score_uncertainty_and_heavy_tails():
    """Verify NASA score distribution diagnostics and per-engine penalty swings."""
    preds_path = Path("reports/results/phase5_engine_predictions.csv")
    df_preds = pd.read_csv(preds_path)

    y_true = df_preds["true_rul"].values
    y_fdna = df_preds["pred_failuredna_ridge"].values
    y_xgb = df_preds["pred_xgboost"].values

    diff_fdna = y_fdna - y_true
    pen_fdna = np.where(diff_fdna < 0, np.exp(-diff_fdna / 13.0) - 1.0, np.exp(diff_fdna / 10.0) - 1.0)
    diff_xgb = y_xgb - y_true
    pen_xgb = np.where(diff_xgb < 0, np.exp(-diff_xgb / 13.0) - 1.0, np.exp(diff_xgb / 10.0) - 1.0)
    diff_pen = pen_fdna - pen_xgb

    # Extreme outlier swings
    max_swing_pos = np.max(diff_pen)
    max_swing_neg = np.min(diff_pen)

    assert max_swing_pos > 800.0, f"Expected positive swing > 800 pts, got {max_swing_pos:.2f}"
    assert max_swing_neg < -800.0, f"Expected negative swing < -800 pts, got {max_swing_neg:.2f}"

    # Fleet totals
    total_fdna = np.sum(pen_fdna)
    total_xgb = np.sum(pen_xgb)
    assert np.isclose(total_fdna, 8049.31, atol=0.1)
    assert np.isclose(total_xgb, 8186.42, atol=0.1)
    assert np.isclose(total_fdna - total_xgb, -137.12, atol=0.1)


def test_phase5_2_addendum_report_compliance():
    """Verify that phase5_2_statistical_correction.md complies with phrasing requirements."""
    report_path = Path("reports/results/phase5_2_statistical_correction.md")
    assert report_path.exists(), "Missing phase5_2_statistical_correction.md"

    content = report_path.read_text(encoding="utf-8")

    # Must contain "the confidence interval includes zero"
    assert "the confidence interval includes zero" in content.lower(), (
        "Report must explicitly use 'the confidence interval includes zero'"
    )

    # Must verify that machine-readable JSON exists
    json_path = Path("reports/results/phase5_2_statistical_summary.json")
    assert json_path.exists()
    with open(json_path, "r") as f:
        summary_data = json.load(f)
    assert "multiple_comparisons_audit" in summary_data
    assert "nasa_score_uncertainty_analysis" in summary_data


def test_all_previous_phase_artifacts_preserved():
    """Verify that original Phase 5 and Phase 5.1 artifacts remain intact."""
    required_files = [
        Path("reports/results/phase5_official_benchmark_report.md"),
        Path("reports/results/phase5_official_benchmark_summary.json"),
        Path("reports/results/phase5_model_comparison.csv"),
        Path("reports/results/phase5_engine_predictions.csv"),
        Path("reports/results/phase5_1_integrity_audit.md"),
        Path("reports/results/phase5_1_integrity_summary.json"),
        Path("reports/results/phase5_1_paired_bootstrap.csv"),
        Path("reports/results/phase5_official_benchmark_report_v2.md"),
    ]
    for p in required_files:
        assert p.exists(), f"Preserved artifact {p} is missing!"


def test_reproducibility_audit_json_integrity():
    """Verify that Phase 5.2.1 reproducibility audit JSON exists and contains required statistical details."""
    audit_json_path = Path("reports/results/phase5_2_1_reproducibility_audit.json")
    assert audit_json_path.exists(), "Missing phase5_2_1_reproducibility_audit.json"

    with open(audit_json_path, "r") as f:
        data = json.load(f)

    assert data["audit_phase"] == "5.2.1"
    assert "H_0: Delta = Metric(Model A) - Metric(Model B) = 0" in data["null_hypothesis"]
    assert "two-sided" in data["tail_convention"].lower()
    assert "weak inequalities" in data["tie_convention"].lower()
    assert "RETROSPECTIVE_DESIGNATION" in data["family_preregistration_status"]["status"]
    assert "PASSED" in data["comparison_direction_verified"]["verification_result"]

    # Verify all 15 tests are recorded
    assert len(data["reproduced_tests"]) == 15


def test_direction_convention_matches_declared_delta():
    """Verify that the difference convention Delta = Metric(Model A) - Metric(Model B) is strictly obeyed."""
    preds_path = Path("reports/results/phase5_engine_predictions.csv")
    df_preds = pd.read_csv(preds_path)
    y_true = df_preds["true_rul"].values

    # Check FailureDNA vs XGBoost in MAE
    mae_fdna = np.mean(np.abs(df_preds["pred_failuredna_ridge"].values - y_true))
    mae_xgb = np.mean(np.abs(df_preds["pred_xgboost"].values - y_true))
    expected_delta_mae = mae_fdna - mae_xgb

    csv_path = Path("reports/results/phase5_2_adjusted_bootstrap.csv")
    df_adj = pd.read_csv(csv_path)
    row = df_adj[(df_adj["comparison"] == "FailureDNA (Regime-Aware) vs XGBoost") & (df_adj["metric"] == "MAE")].iloc[0]

    assert np.isclose(expected_delta_mae, row["delta_point"], atol=1e-4)
    assert row["delta_point"] > 0, "Positive delta for error metric indicates Model B (XGBoost) had lower error"


def test_davison_hinkley_pseudo_count_invariance():
    """Verify that using Davison-Hinkley pseudo-count correction yields identical statistical decisions."""
    audit_json_path = Path("reports/results/phase5_2_1_reproducibility_audit.json")
    with open(audit_json_path, "r") as f:
        data = json.load(f)

    tests = data["reproduced_tests"]
    for t in tests:
        p_emp = t["p_val_empirical"]
        p_dh = t["p_val_davison_hinkley"]
        
        # Davison-Hinkley should be >= empirical proportion
        assert p_dh >= p_emp - 1e-6
        
        # Decisions under Holm metric (m=5) must be identical
        sig_emp = t["holm_metric_emp"] < 0.05
        sig_dh = t["holm_metric_dh"] < 0.05
        assert sig_emp == sig_dh, f"Decision mismatch for {t['comparison']} {t['metric']}"


def test_preregistration_qualification_in_addendum():
    """Verify that phase5_2_statistical_correction.md qualifies the primary family as retrospective."""
    report_path = Path("reports/results/phase5_2_statistical_correction.md")
    content = report_path.read_text(encoding="utf-8")

    assert "retrospective" in content.lower(), "Report must explicitly qualify m=3 family as retrospective"
    assert "not preregistered" in content.lower(), "Report must declare that m=3 family was not preregistered"

