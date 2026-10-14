"""Unit tests for operating regime clustering and diagnostic evaluation."""

import numpy as np
import pandas as pd
import pytest

from failuredna.data.loader import ID_COL, CYCLE_COL, SENSOR_COLS, SETTING_COLS
from failuredna.data.regimes import OperatingRegimeClusterer


@pytest.fixture
def synthetic_settings_df():
    """Create a synthetic dataset with 3 distinct setting clusters."""
    rng = np.random.RandomState(42)
    rows = []
    # Cluster 1: Sea-level (alt=0, mach=0, TRA=100)
    for _ in range(50):
        rows.append([0.0 + rng.normal(0, 0.1), 0.0 + rng.normal(0, 0.01), 100.0])
    # Cluster 2: Mid-altitude (alt=20, mach=0.7, TRA=100)
    for _ in range(50):
        rows.append([20.0 + rng.normal(0, 0.1), 0.7 + rng.normal(0, 0.01), 100.0])
    # Cluster 3: High-altitude low TRA (alt=35, mach=0.84, TRA=60)
    for _ in range(50):
        rows.append([35.0 + rng.normal(0, 0.1), 0.84 + rng.normal(0, 0.01), 60.0])

    df = pd.DataFrame(rows, columns=SETTING_COLS)
    df[ID_COL] = np.repeat([1, 2, 3], 50)
    df[CYCLE_COL] = np.tile(np.arange(1, 51), 3)
    # Add dummy sensors
    for s in SENSOR_COLS:
        df[s] = rng.uniform(10.0, 100.0, size=len(df))
    return df


def test_clusterer_initialization_validation():
    """Verify ValueError on invalid cluster count or method."""
    with pytest.raises(ValueError, match="n_regimes must be >= 2"):
        OperatingRegimeClusterer(n_regimes=1)
    with pytest.raises(ValueError, match="method must be 'kmeans' or 'gmm'"):
        OperatingRegimeClusterer(method="dbscan")


def test_clusterer_fit_predict_kmeans(synthetic_settings_df):
    """Verify KMeans fitting, prediction shapes, and deterministic labels."""
    clusterer = OperatingRegimeClusterer(n_regimes=3, method="kmeans", random_state=42)
    assert not clusterer.is_fitted_

    with pytest.raises(RuntimeError, match="must be fitted"):
        clusterer.predict(synthetic_settings_df)

    clusterer.fit(synthetic_settings_df)
    assert clusterer.is_fitted_
    assert clusterer.regime_centers_.shape == (3, 3)

    labels = clusterer.predict(synthetic_settings_df)
    assert len(labels) == len(synthetic_settings_df)
    assert set(labels) == {0, 1, 2}

    # Transform adds column
    df_transformed = clusterer.transform(synthetic_settings_df, col_name="regime_id")
    assert "regime_id" in df_transformed.columns
    assert (df_transformed["regime_id"].to_numpy() == labels).all()


def test_clusterer_fit_predict_gmm(synthetic_settings_df):
    """Verify GMM fitting and prediction."""
    clusterer = OperatingRegimeClusterer(n_regimes=3, method="gmm", random_state=42)
    clusterer.fit(synthetic_settings_df)
    assert clusterer.is_fitted_
    labels = clusterer.predict(synthetic_settings_df)
    assert len(labels) == len(synthetic_settings_df)
    assert set(labels) == {0, 1, 2}


def test_clusterer_frozen_transform_on_unseen(synthetic_settings_df):
    """Verify that predicting on unseen data uses frozen training centers without shifting."""
    train_df = synthetic_settings_df.iloc[:100].copy()
    unseen_df = synthetic_settings_df.iloc[100:].copy()

    clusterer = OperatingRegimeClusterer(n_regimes=2, method="kmeans", random_state=42)
    clusterer.fit(train_df)
    frozen_centers = clusterer.regime_centers_.copy()

    # Predict on unseen
    _ = clusterer.predict(unseen_df)
    np.testing.assert_array_equal(clusterer.regime_centers_, frozen_centers)


def test_evaluate_cluster_diagnostics(synthetic_settings_df):
    """Verify diagnostic evaluation metrics across k range."""
    diagnostics = OperatingRegimeClusterer.evaluate_cluster_diagnostics(
        synthetic_settings_df,
        k_range=(2, 4),
        sample_size_for_silhouette=150,
        random_state=42,
    )

    assert set(diagnostics.keys()) == {2, 3, 4}
    for k, metrics in diagnostics.items():
        assert metrics["k"] == k
        assert 0.0 <= metrics["silhouette_score"] <= 1.0
        assert metrics["davies_bouldin_index"] >= 0.0
        assert metrics["calinski_harabasz_index"] >= 0.0
        assert metrics["cluster_size_balance_ratio"] > 0.0
        assert np.isfinite(metrics["gmm_bic"])
        assert np.isfinite(metrics["gmm_aic"])


def test_regime_profiles(synthetic_settings_df):
    """Verify profile extraction per regime."""
    clusterer = OperatingRegimeClusterer(n_regimes=3, method="kmeans", random_state=42)
    clusterer.fit(synthetic_settings_df)
    profiles = clusterer.get_regime_profiles(synthetic_settings_df)

    assert len(profiles) == 3
    for r_id in range(3):
        assert "settings_mean" in profiles[r_id]
        assert "sensors_mean" in profiles[r_id]
        assert "observation_count" in profiles[r_id]
        assert profiles[r_id]["observation_count"] > 0


def test_compute_within_regime_variances(synthetic_settings_df):
    """Verify within-regime variance vs raw variance calculations."""
    clusterer = OperatingRegimeClusterer(n_regimes=3, method="kmeans", random_state=42)
    clusterer.fit(synthetic_settings_df)

    var_dict = clusterer.compute_within_regime_variances(synthetic_settings_df)
    assert len(var_dict) == len(SENSOR_COLS)
    for s, metrics in var_dict.items():
        assert "raw_variance" in metrics
        assert "mean_within_variance" in metrics
        assert "within_to_raw_ratio" in metrics
        assert "coefficient_of_variation" in metrics
        assert metrics["raw_variance"] >= 0.0
        assert metrics["mean_within_variance"] >= 0.0


def test_compute_within_regime_correlations(synthetic_settings_df):
    """Verify pooled and mean within-regime correlation matrices."""
    clusterer = OperatingRegimeClusterer(n_regimes=3, method="kmeans", random_state=42)
    clusterer.fit(synthetic_settings_df)

    pooled_corr, within_corr = clusterer.compute_within_regime_correlations(synthetic_settings_df)
    assert pooled_corr.shape == (len(SENSOR_COLS), len(SENSOR_COLS))
    assert within_corr.shape == (len(SENSOR_COLS), len(SENSOR_COLS))
    # Diagonal of correlation matrices should be 1.0 (or close)
    for s in SENSOR_COLS:
        assert np.isclose(pooled_corr.loc[s, s], 1.0)
        assert np.isclose(within_corr.loc[s, s], 1.0)

