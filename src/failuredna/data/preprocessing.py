"""Data preprocessing and leakage-free feature scaling module.

Enforces strict isolation by fitting all normalization parameters, variance thresholds,
and regime clustering models exclusively on the training partition.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from failuredna.data.loader import CYCLE_COL, ID_COL, SENSOR_COLS, SETTING_COLS
from failuredna.utils.logging import get_logger

logger = get_logger(__name__)


class CMAPSSPreprocessor:
    """Leakage-safe preprocessor for C-MAPSS multivariate time-series.

    Guarantees:
    1. Fits all scalers and filters strictly on training engine data.
    2. Zero knowledge of validation or test sets during parameter estimation.
    3. Excludes unit IDs, cycle counts, and any downstream labels from feature matrices.
    4. Deterministic and serializable transformation state.
    """

    def __init__(
        self,
        scaler_type: str = "robust",
        variance_threshold: float = 1e-5,
        include_settings: bool = True,
    ):
        """Initialize preprocessor.

        Args:
            scaler_type: Scaling method ('robust', 'standard', 'minmax').
            variance_threshold: Variance floor to prune non-informative sensors in training set.
            include_settings: If True, includes normalized operating settings (1-3) as features.
        """
        self.scaler_type = scaler_type.lower()
        self.variance_threshold = variance_threshold
        self.include_settings = include_settings

        # Fitted state attributes
        self.is_fitted_: bool = False
        self.constant_sensors_: List[str] = []
        self.active_sensors_: List[str] = []
        self.feature_cols_: List[str] = []
        self._scaler: Optional[Union[RobustScaler, StandardScaler, MinMaxScaler]] = None

    def _init_scaler(self) -> Union[RobustScaler, StandardScaler, MinMaxScaler]:
        """Instantiate the underlying scikit-learn scaler."""
        if self.scaler_type == "robust":
            return RobustScaler(unit_variance=True)
        elif self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "minmax":
            return MinMaxScaler(feature_range=(-1.0, 1.0))
        else:
            raise ValueError(
                f"Unsupported scaler_type '{self.scaler_type}'. Choose from 'robust', 'standard', 'minmax'."
            )

    def fit(self, train_df: pd.DataFrame) -> "CMAPSSPreprocessor":
        """Fit feature selection and scaling parameters strictly on training engines.

        Args:
            train_df: DataFrame containing training engine trajectories.

        Returns:
            self: The fitted preprocessor instance.
        """
        logger.info("Fitting CMAPSSPreprocessor strictly on training engine partition...")

        # 1. Evaluate sensor variances on training data only
        sensor_vars = train_df[SENSOR_COLS].var()
        self.constant_sensors_ = sensor_vars[sensor_vars <= self.variance_threshold].index.tolist()
        self.active_sensors_ = sensor_vars[sensor_vars > self.variance_threshold].index.tolist()

        logger.info(
            f"Sensor pruning (threshold={self.variance_threshold}): "
            f"Retained {len(self.active_sensors_)} informative sensors; "
            f"Pruned {len(self.constant_sensors_)} constant sensors: {self.constant_sensors_}"
        )

        # 2. Define final active feature column set
        self.feature_cols_ = []
        if self.include_settings:
            self.feature_cols_.extend(SETTING_COLS)
        self.feature_cols_.extend(self.active_sensors_)

        # 3. Fit scaler exclusively on training feature matrix
        self._scaler = self._init_scaler()
        self._scaler.fit(train_df[self.feature_cols_].to_numpy())

        self.is_fitted_ = True
        logger.info(f"Preprocessor fitted successfully on {len(self.feature_cols_)} input features.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transformations to a dataset without refitting.

        Args:
            df: DataFrame to transform (train, val, or test).

        Returns:
            New DataFrame with scaled features and preserved metadata (unit_id, cycle).

        Raises:
            RuntimeError: If called before fit().
        """
        if not self.is_fitted_ or self._scaler is None:
            raise RuntimeError("CMAPSSPreprocessor must be fitted on training data before calling transform().")

        transformed_df = df[[ID_COL, CYCLE_COL]].copy()

        # Scale features using frozen training parameters
        feature_matrix = df[self.feature_cols_].to_numpy()
        scaled_matrix = self._scaler.transform(feature_matrix)

        scaled_feature_df = pd.DataFrame(
            scaled_matrix,
            columns=self.feature_cols_,
            index=df.index,
        )

        result = pd.concat([transformed_df, scaled_feature_df], axis=1)
        return result

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor on training data and return transformed training set."""
        return self.fit(train_df).transform(train_df)

    def get_feature_names(self) -> List[str]:
        """Return list of active feature names in the transformed representation."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor is not fitted.")
        return list(self.feature_cols_)

    def get_state(self) -> Dict[str, Any]:
        """Serialize preprocessor configuration and fitted parameters for audit logging."""
        if not self.is_fitted_ or self._scaler is None:
            return {"is_fitted": False}

        state: Dict[str, Any] = {
            "is_fitted": True,
            "scaler_type": self.scaler_type,
            "variance_threshold": self.variance_threshold,
            "include_settings": self.include_settings,
            "constant_sensors": self.constant_sensors_,
            "active_sensors": self.active_sensors_,
            "feature_cols": self.feature_cols_,
        }

        if hasattr(self._scaler, "center_"):
            state["center"] = self._scaler.center_.tolist()
        if hasattr(self._scaler, "scale_"):
            state["scale"] = self._scaler.scale_.tolist()
        if hasattr(self._scaler, "mean_"):
            state["mean"] = self._scaler.mean_.tolist()
        if hasattr(self._scaler, "var_"):
            state["var"] = self._scaler.var_.tolist()

        return state


class RegimeAwarePreprocessor:
    """Regime-aware preprocessor for C-MAPSS multivariate time-series.

    Evaluates Variant B (within-regime normalization) and Variant C (explicit regime conditioning).
    Guarantees:
    1. Operating regime clusterer is fitted strictly on training settings.
    2. Within-regime mean and std are estimated exclusively on training observations.
    3. Explicit division-by-zero protection for zero-variance channels (s_1, s_5, s_10, s_16, s_18, s_19).
    4. Zero leakage across engine splits.
    """

    def __init__(
        self,
        n_regimes: int = 6,
        include_settings: bool = True,
        include_regime_onehot: bool = False,
        seed: int = 42,
    ):
        """Initialize RegimeAwarePreprocessor.

        Args:
            n_regimes: Number of operating regimes to infer (default: 6).
            include_settings: If True, includes normalized setting channels.
            include_regime_onehot: If True (Variant C), appends one-hot regime indicator columns.
            seed: Random state for deterministic clustering.
        """
        self.n_regimes = n_regimes
        self.include_settings = include_settings
        self.include_regime_onehot = include_regime_onehot
        self.seed = seed

        self.is_fitted_: bool = False
        self.clusterer_: Any = None
        self.regime_stats_: Dict[int, Dict[str, Dict[str, float]]] = {}
        self.zero_variance_channels_: Dict[int, List[str]] = {}
        self.feature_cols_: List[str] = []
        self._settings_scaler: Optional[StandardScaler] = None

    def fit(self, train_df: pd.DataFrame) -> "RegimeAwarePreprocessor":
        """Fit regime clusterer and within-regime normalization statistics on train partition."""
        from failuredna.data.regimes import OperatingRegimeClusterer

        logger.info(f"Fitting RegimeAwarePreprocessor (k={self.n_regimes}) strictly on training engines...")
        self.clusterer_ = OperatingRegimeClusterer(
            n_regimes=self.n_regimes,
            method="kmeans",
            random_state=self.seed,
        )
        self.clusterer_.fit(train_df)

        train_tagged = self.clusterer_.transform(train_df, col_name="regime_id")
        self.regime_stats_ = {}
        self.zero_variance_channels_ = {}

        for r_id in range(self.n_regimes):
            subset = train_tagged[train_tagged["regime_id"] == r_id]
            stats_r = {}
            zero_vars = []

            for s in SENSOR_COLS:
                s_mean = float(subset[s].mean()) if len(subset) > 0 else 0.0
                s_std = float(subset[s].std(ddof=1)) if len(subset) > 1 else 0.0

                if s_std < 1e-6:
                    zero_vars.append(s)
                    s_std = 1.0  # Guard against division by zero

                stats_r[s] = {"mean": s_mean, "std": s_std}

            self.regime_stats_[r_id] = stats_r
            self.zero_variance_channels_[r_id] = zero_vars

        # Scale settings globally
        if self.include_settings:
            self._settings_scaler = StandardScaler()
            self._settings_scaler.fit(train_df[SETTING_COLS].to_numpy())

        # Define active feature columns
        self.feature_cols_ = []
        if self.include_settings:
            self.feature_cols_.extend(SETTING_COLS)
        self.feature_cols_.extend(SENSOR_COLS)
        if self.include_regime_onehot:
            self.feature_cols_.extend([f"regime_{i}" for i in range(self.n_regimes)])

        self.is_fitted_ = True
        logger.info(
            f"RegimeAwarePreprocessor fitted successfully. Total feature dim: {len(self.feature_cols_)}. "
            f"Zero within-regime channels detected in Regime 0: {len(self.zero_variance_channels_.get(0, []))}"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataset using frozen training regime statistics."""
        if not self.is_fitted_:
            raise RuntimeError("RegimeAwarePreprocessor must be fitted before transform().")

        # 1. Infer regime IDs using frozen clusterer
        df_tagged = self.clusterer_.transform(df, col_name="regime_id")
        regime_ids = df_tagged["regime_id"].to_numpy()

        # 2. Normalize sensors per regime
        norm_sensors = np.zeros((len(df), len(SENSOR_COLS)), dtype=np.float32)
        sensor_data = df[SENSOR_COLS].to_numpy(dtype=np.float32)

        for r_id in range(self.n_regimes):
            mask = (regime_ids == r_id)
            if not np.any(mask):
                continue

            for j, s in enumerate(SENSOR_COLS):
                mean_val = self.regime_stats_[r_id][s]["mean"]
                std_val = self.regime_stats_[r_id][s]["std"]
                is_zero_var = s in self.zero_variance_channels_[r_id]

                if is_zero_var:
                    # Invariant channel within this regime: normalize to 0.0
                    norm_sensors[mask, j] = 0.0
                else:
                    norm_sensors[mask, j] = (sensor_data[mask, j] - mean_val) / std_val

        sensor_df = pd.DataFrame(norm_sensors, columns=SENSOR_COLS, index=df.index)

        # 3. Assemble full feature set
        cols_to_concat = [df[[ID_COL, CYCLE_COL]]]

        if self.include_settings and self._settings_scaler is not None:
            scaled_settings = self._settings_scaler.transform(df[SETTING_COLS].to_numpy())
            settings_df = pd.DataFrame(scaled_settings, columns=SETTING_COLS, index=df.index)
            cols_to_concat.append(settings_df)

        cols_to_concat.append(sensor_df)

        if self.include_regime_onehot:
            one_hot = np.zeros((len(df), self.n_regimes), dtype=np.float32)
            one_hot[np.arange(len(df)), regime_ids] = 1.0
            one_hot_df = pd.DataFrame(
                one_hot,
                columns=[f"regime_{i}" for i in range(self.n_regimes)],
                index=df.index,
            )
            cols_to_concat.append(one_hot_df)

        return pd.concat(cols_to_concat, axis=1)

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor and transform train dataset."""
        return self.fit(train_df).transform(train_df)

    def get_feature_names(self) -> List[str]:
        """Return ordered list of transformed feature names."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor is not fitted.")
        return list(self.feature_cols_)

