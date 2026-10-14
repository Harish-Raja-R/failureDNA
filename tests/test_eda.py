"""Unit tests for EDA output artifacts, summary JSON integrity, and generated figures."""

import json
from pathlib import Path
import pytest


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parent.parent


def test_eda_summary_json_exists_and_valid(repo_root):
    """Verify that eda_summary_FD004.json exists, is valid JSON, and has all expected keys."""
    summary_path = repo_root / "reports" / "results" / "eda_summary_FD004.json"
    assert summary_path.is_file(), f"Missing EDA summary file at {summary_path}"

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset_variant"] == "FD004"
    assert "counts" in data
    assert "trajectory_lengths" in data
    assert "sensor_distributions" in data
    assert "high_correlation_pairs" in data
    assert "operating_regimes" in data
    assert "short_test_trajectories" in data

    # Counts
    assert data["counts"]["train_engines"] == 249
    assert data["counts"]["train_observations"] == 61249
    assert data["counts"]["test_engines"] == 248
    assert data["counts"]["test_observations"] == 41214
    assert data["counts"]["rul_ground_truth_records"] == 248

    # Trajectory bounds
    assert data["trajectory_lengths"]["train"]["min"] == 128
    assert data["trajectory_lengths"]["train"]["max"] == 543
    assert data["trajectory_lengths"]["test"]["min"] == 19
    assert data["trajectory_lengths"]["test"]["max"] == 486


def test_eda_operating_regimes_diagnostics(repo_root):
    """Verify empirical clustering diagnostics in summary JSON."""
    summary_path = repo_root / "reports" / "results" / "eda_summary_FD004.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    diagnostics = data["operating_regimes"]["diagnostics_k2_to_k10"]
    assert "6" in diagnostics
    k6 = diagnostics["6"]
    assert k6["silhouette_score"] > 0.99
    assert k6["davies_bouldin_index"] < 0.01
    assert k6["kmeans_inertia"] < 1.0


def test_eda_short_trajectories_count(repo_root):
    """Verify exact count and bounds of short test trajectories in summary JSON."""
    summary_path = repo_root / "reports" / "results" / "eda_summary_FD004.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    short_info = data["short_test_trajectories"]
    assert short_info["engines_shorter_than_50_cycles_count"] == 20
    assert short_info["engines_shorter_than_30_cycles_count"] == 11
    assert short_info["minimum_test_trajectory_length"] == 19


def test_eda_generated_figures_exist_and_non_empty(repo_root):
    """Verify all 9 publication-grade figures were created and are non-empty."""
    figures_dir = repo_root / "reports" / "figures"
    expected_figures = [
        "01_trajectory_length_distribution.png",
        "02_sensor_variance_ranking.png",
        "03_sensor_correlation_matrix.png",
        "04_operating_settings_distributions.png",
        "05_operating_regime_clustering_diagnostics.png",
        "06_regime_wise_sensor_profiles.png",
        "07_sensor_evolution_normalized_life.png",
        "08_sensor_behavior_by_rul_interval.png",
        "09_short_test_trajectories_and_padding_analysis.png",
    ]

    for fig_name in expected_figures:
        fig_path = figures_dir / fig_name
        assert fig_path.is_file(), f"Missing expected figure: {fig_name}"
        # Ensure figure is substantial (greater than 50 KB)
        assert fig_path.stat().st_size > 50_000, f"Figure {fig_name} is unexpectedly small: {fig_path.stat().st_size} bytes"
