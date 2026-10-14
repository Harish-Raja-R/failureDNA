"""Operating Regime Discovery and Normalization Module.

Provides unsupervised clustering of flight conditions (altitude, Mach, TRA)
fitted strictly on training partitions, with comprehensive clustering diagnostics
and regime-conditional feature profiling.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from failuredna.data.loader import ID_COL, SENSOR_COLS, SETTING_COLS
from failuredna.utils.logging import get_logger

logger = get_logger(__name__)


class OperatingRegimeClusterer:
    """Discovers and assigns operating regimes from flight setting variables.

    Safeguards:
    1. Fitted exclusively on training settings (setting_1, setting_2, setting_3).
    2. Never uses RUL, cycle index, or failure labels for clustering.
    3. Frozen scaler and clusterer transform validation and test partitions without leakage.
    4. Evaluates empirical diagnostics (Silhouette, Davies-Bouldin, Calinski-Harabasz, BIC/AIC).
    """

    def __init__(
        self,
        n_regimes: int = 6,
        method: str = "kmeans",
        random_state: int = 42,
    ):
        """Initialize OperatingRegimeClusterer.

        Args:
            n_regimes: Number of operating regime clusters $k$ (default: 6).
            method: Clustering algorithm ('kmeans' or 'gmm').
            random_state: Random seed for reproducibility.
        """
        if n_regimes < 2:
            raise ValueError(f"n_regimes must be >= 2, got {n_regimes}.")
        if method not in ("kmeans", "gmm"):
            raise ValueError(f"method must be 'kmeans' or 'gmm', got '{method}'.")

        self.n_regimes = n_regimes
        self.method = method
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model: Optional[Union[KMeans, GaussianMixture]] = None
        self.is_fitted_ = False
        self.regime_centers_: Optional[np.ndarray] = None
        self.feature_cols_ = list(SETTING_COLS)

    def fit(self, train_df: pd.DataFrame) -> "OperatingRegimeClusterer":
        """Fit scaler and clustering model strictly on training operating settings.

        Args:
            train_df: Training telemetry DataFrame containing SETTING_COLS.

        Returns:
            Fitted instance of self.
        """
        logger.info(
            f"Fitting {self.method.upper()} operating-regime clusterer (k={self.n_regimes}) "
            f"strictly on training partition ({len(train_df):,} observations)..."
        )
        # Ensure only setting columns are extracted
        X_train = train_df[self.feature_cols_].to_numpy(dtype=np.float64)

        X_scaled = self.scaler.fit_transform(X_train)

        if self.method == "kmeans":
            self.model = KMeans(
                n_clusters=self.n_regimes,
                random_state=self.random_state,
                n_init=10,
                max_iter=300,
            )
            self.model.fit(X_scaled)
            self.regime_centers_ = self.scaler.inverse_transform(self.model.cluster_centers_)
        else:
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                random_state=self.random_state,
                n_init=3,
                max_iter=200,
            )
            self.model.fit(X_scaled)
            self.regime_centers_ = self.scaler.inverse_transform(self.model.means_)

        self.is_fitted_ = True
        logger.info(f"Operating regime clusterer fitted successfully with {self.n_regimes} regimes.")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict operating regime cluster IDs for telemetry observations.

        Args:
            df: DataFrame containing SETTING_COLS.

        Returns:
            1D integer array of cluster IDs in range [0, n_regimes - 1].
        """
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError("OperatingRegimeClusterer must be fitted before predict.")

        X = df[self.feature_cols_].to_numpy(dtype=np.float64)
        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)
        return labels.astype(int)

    def transform(self, df: pd.DataFrame, col_name: str = "regime_id") -> pd.DataFrame:
        """Add regime_id column to DataFrame using frozen model.

        Args:
            df: DataFrame to transform.
            col_name: Name of the assigned regime column (default: 'regime_id').

        Returns:
            Copy of df with the appended regime_id column.
        """
        labels = self.predict(df)
        df_out = df.copy()
        df_out[col_name] = labels
        return df_out

    @staticmethod
    def evaluate_cluster_diagnostics(
        train_df: pd.DataFrame,
        k_range: Tuple[int, int] = (2, 10),
        sample_size_for_silhouette: int = 10000,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Compute comprehensive clustering diagnostics across a range of cluster counts k.

        Evaluates:
        - Silhouette score
        - Davies-Bouldin index (lower is better)
        - Calinski-Harabasz index (higher is better)
        - GMM BIC & AIC
        - Cluster size balance (min_size / max_size and entropy)

        Args:
            train_df: Training telemetry DataFrame with SETTING_COLS.
            k_range: Tuple of (min_k, max_k) inclusive.
            sample_size_for_silhouette: Subsample size for fast exact silhouette computation.
            random_state: Random seed.

        Returns:
            Dictionary containing diagnostic metrics per k.
        """
        X = train_df[SETTING_COLS].to_numpy(dtype=np.float64)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_samples = len(X_scaled)
        rng = np.random.RandomState(random_state)
        if n_samples > sample_size_for_silhouette:
            sub_idx = rng.choice(n_samples, size=sample_size_for_silhouette, replace=False)
            X_sub = X_scaled[sub_idx]
        else:
            sub_idx = np.arange(n_samples)
            X_sub = X_scaled

        diagnostics: Dict[int, Dict[str, float]] = {}

        for k in range(k_range[0], k_range[1] + 1):
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
            km_labels = km.fit_predict(X_scaled)

            sil = float(silhouette_score(X_sub, km_labels[sub_idx]))
            db = float(davies_bouldin_score(X_scaled, km_labels))
            ch = float(calinski_harabasz_score(X_scaled, km_labels))
            inertia = float(km.inertia_)

            # Cluster size distribution
            counts = pd.Series(km_labels).value_counts().values
            min_count = int(counts.min())
            max_count = int(counts.max())
            balance_ratio = float(min_count / max_count)
            probs = counts / n_samples
            entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

            # GMM BIC/AIC
            gmm = GaussianMixture(n_components=k, random_state=random_state, n_init=2)
            gmm.fit(X_scaled)
            bic = float(gmm.bic(X_scaled))
            aic = float(gmm.aic(X_scaled))

            diagnostics[k] = {
                "k": k,
                "silhouette_score": sil,
                "davies_bouldin_index": db,
                "calinski_harabasz_index": ch,
                "kmeans_inertia": inertia,
                "cluster_min_size": min_count,
                "cluster_max_size": max_count,
                "cluster_size_balance_ratio": balance_ratio,
                "cluster_entropy": entropy,
                "gmm_bic": bic,
                "gmm_aic": aic,
            }

        return diagnostics

    def get_regime_profiles(
        self,
        df: pd.DataFrame,
    ) -> Dict[int, Dict[str, Any]]:
        """Compute comprehensive statistical profiles of settings and sensors per regime.

        Args:
            df: DataFrame containing SETTING_COLS and SENSOR_COLS.

        Returns:
            Dictionary mapping regime_id -> {counts, settings_summary, sensors_mean, etc.}
        """
        df_tagged = self.transform(df, col_name="regime_id")
        profiles: Dict[int, Dict[str, Any]] = {}

        for r_id in range(self.n_regimes):
            subset = df_tagged[df_tagged["regime_id"] == r_id]
            if subset.empty:
                continue

            settings_summary = {}
            for col in SETTING_COLS:
                settings_summary[col] = {
                    "mean": float(subset[col].mean()),
                    "std": float(subset[col].std()),
                    "min": float(subset[col].min()),
                    "max": float(subset[col].max()),
                }

            sensors_means = {k: float(v) for k, v in subset[SENSOR_COLS].mean().to_dict().items()}
            sensors_stds = {k: float(v) for k, v in subset[SENSOR_COLS].std().to_dict().items()}
            sensors_vars = {k: float(v) for k, v in subset[SENSOR_COLS].var().to_dict().items()}

            profiles[r_id] = {
                "regime_id": r_id,
                "observation_count": len(subset),
                "proportion": float(len(subset) / len(df_tagged)),
                "settings_summary": settings_summary,
                "settings_mean": {k: v["mean"] for k, v in settings_summary.items()},
                "sensors_mean": sensors_means,
                "sensors_std": sensors_stds,
                "sensors_var": sensors_vars,
            }

        return profiles

    def compute_within_regime_variances(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """Compute within-regime variance vs raw pooled variance for all sensors.

        Demonstrates that sensors like s_1, s_5, s_18, s_19 have essentially zero
        variance within individual regimes, showing their raw variance is purely
        an artifact of switching flight conditions.

        Args:
            df: DataFrame containing SENSOR_COLS.

        Returns:
            Dictionary mapping sensor_col -> {raw_variance, mean_within_variance, within_to_raw_ratio, cv}
        """
        df_tagged = self.transform(df, col_name="regime_id")
        results = {}

        for s in SENSOR_COLS:
            raw_var = float(df[s].var())
            raw_mean = float(df[s].mean())
            within_vars = [float(df_tagged[df_tagged["regime_id"] == r][s].var()) for r in range(self.n_regimes)]
            mean_within_var = float(np.mean(within_vars))
            ratio = float(mean_within_var / raw_var) if raw_var > 1e-12 else 0.0
            cv = float(np.sqrt(raw_var) / abs(raw_mean)) if abs(raw_mean) > 1e-12 else 0.0

            results[s] = {
                "raw_variance": raw_var,
                "mean_within_variance": mean_within_var,
                "within_to_raw_ratio": ratio,
                "coefficient_of_variation": cv,
            }

        return results

    def compute_within_regime_correlations(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute pooled sensor correlation vs mean within-regime sensor correlation.

        Allows detection of Simpson's paradox where altitude/Mach shifts create
        spurious cross-sensor correlations not present within any operational regime.

        Args:
            df: DataFrame containing SENSOR_COLS.

        Returns:
            Tuple of (pooled_correlation_df, mean_within_regime_correlation_df).
        """
        df_tagged = self.transform(df, col_name="regime_id")
        pooled_corr = df[SENSOR_COLS].corr(method="pearson")

        regime_corrs = []
        for r in range(self.n_regimes):
            sub = df_tagged[df_tagged["regime_id"] == r][SENSOR_COLS]
            # Replace NaNs (from zero-variance sensors like s_18 in fixed regimes) with 0 for averaging
            r_corr = sub.corr(method="pearson")
            regime_corrs.append(r_corr)

        # Average correlation across regimes
        stacked = np.stack([c.to_numpy() for c in regime_corrs], axis=0)
        with np.errstate(invalid="ignore"):
            mean_within_arr = np.nanmean(stacked, axis=0)
        # Fill any remaining NaNs (constant across all regimes) with 0.0
        mean_within_arr = np.nan_to_num(mean_within_arr, nan=0.0)

        mean_within_df = pd.DataFrame(mean_within_arr, index=SENSOR_COLS, columns=SENSOR_COLS)
        return pooled_corr, mean_within_df
