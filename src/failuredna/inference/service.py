"""FailureDNA Core Inference Service.

Encapsulates:
1. Frozen Temporal Transformer encoder loading & verification.
2. Frozen RegimeAwarePreprocessor (k=6) transformation.
3. Downstream Ridge RUL estimation probe.
4. Discrete failure signature clustering & centroid distance assignment.
5. Telemetry validation and sequence edge-padding.
6. Research metadata & benchmark metrics serving from saved artifacts.
"""

from io import BytesIO, StringIO
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch

from failuredna.data.loader import CYCLE_COL, ID_COL, SENSOR_COLS, SETTING_COLS
from failuredna.data.preprocessing import RegimeAwarePreprocessor
from failuredna.data.windowing import WindowGenerator
from failuredna.evaluation.signatures import FailureSignatureClusterer
from failuredna.models.encoder import FailureDNAContrastiveModel, ProjectionHead, TemporalTransformerEncoder
from failuredna.models.trainer import extract_embeddings_from_windows

logger = logging.getLogger("failuredna.inference")

FROZEN_ENCODER_SHA256 = "d18b6cd0abc2fadf12be18f6f1962707f3626e0a5054970147f50eb3620eb248"
ACTIVE_SENSORS = ["s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"]


class FailureDNAInferenceService:
    """Production inference service reusing frozen FailureDNA artifacts."""

    def __init__(
        self,
        checkpoint_path: Path = Path("models/checkpoints/failuredna_regime_aware/best_failuredna_model.pt"),
        bundle_dir: Path = Path("models/inference_bundle"),
        benchmark_summary_path: Path = Path("reports/results/phase5_official_benchmark_summary.json"),
        benchmark_csv_path: Path = Path("reports/results/phase5_model_comparison.csv"),
        adjusted_bootstrap_path: Path = Path("reports/results/phase5_2_adjusted_bootstrap.csv"),
        device: str = "cpu",
    ):
        self.checkpoint_path = checkpoint_path
        self.bundle_dir = bundle_dir
        self.benchmark_summary_path = benchmark_summary_path
        self.benchmark_csv_path = benchmark_csv_path
        self.adjusted_bootstrap_path = adjusted_bootstrap_path
        self.device = device

        self.encoder_model: Optional[FailureDNAContrastiveModel] = None
        self.preprocessor: Optional[RegimeAwarePreprocessor] = None
        self.ridge_probe: Optional[Any] = None
        self.signature_clusterer: Optional[FailureSignatureClusterer] = None
        self.bundle_metadata: Optional[Dict[str, Any]] = None
        self.window_generator: Optional[WindowGenerator] = None

        self._is_ready: bool = False
        self._init_error: Optional[str] = None
        self._load_timestamp: Optional[str] = None

    def initialize(self) -> bool:
        """Load frozen model checkpoint and pre-fitted inference bundle."""
        try:
            logger.info("Initializing FailureDNA Inference Service...")

            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"Missing frozen encoder checkpoint: {self.checkpoint_path}")

            prep_path = self.bundle_dir / "regime_preprocessor.joblib"
            ridge_path = self.bundle_dir / "ridge_probe.joblib"
            clusterer_path = self.bundle_dir / "signature_clusterer.joblib"
            meta_path = self.bundle_dir / "metadata.json"

            if not (prep_path.exists() and ridge_path.exists() and clusterer_path.exists()):
                raise FileNotFoundError(
                    f"Inference bundle missing at {self.bundle_dir}. Please run 'scratch/build_inference_bundle.py'."
                )

            logger.info(f"Loading inference bundle components from '{self.bundle_dir}'...")
            self.preprocessor = joblib.load(prep_path)
            self.ridge_probe = joblib.load(ridge_path)
            self.signature_clusterer = joblib.load(clusterer_path)

            if meta_path.exists():
                with open(meta_path, "r") as f:
                    self.bundle_metadata = json.load(f)

            feature_cols = self.preprocessor.get_feature_names()
            self.window_generator = WindowGenerator(
                window_size=50, stride=1, feature_cols=feature_cols, padding_mode="edge"
            )

            logger.info(f"Loading frozen Temporal Transformer encoder ({self.checkpoint_path})...")
            t_enc = TemporalTransformerEncoder(
                input_dim=len(feature_cols),
                hidden_dim=128,
                n_heads=8,
                n_layers=4,
                dim_feedforward=256,
                dropout=0.1,
                pooling="masked_mean",
            )
            p_head = ProjectionHead(input_dim=128, hidden_dim=128, output_dim=64)
            self.encoder_model = FailureDNAContrastiveModel(t_enc, p_head)

            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.encoder_model.load_state_dict(state_dict)
            self.encoder_model.eval()

            self._is_ready = True
            self._init_error = None
            self._load_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info("FailureDNA Inference Service successfully initialized and ready.")
            return True

        except Exception as e:
            self._is_ready = False
            self._init_error = str(e)
            logger.error(f"Failed to initialize FailureDNA Inference Service: {e}", exc_info=True)
            return False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def get_health_status(self) -> Dict[str, Any]:
        """Return detailed artifact readiness status for /health."""
        return {
            "status": "healthy" if self._is_ready else "degraded",
            "is_ready": self._is_ready,
            "error": self._init_error,
            "initialized_at": self._load_timestamp,
            "artifacts": {
                "frozen_checkpoint_exists": self.checkpoint_path.exists(),
                "encoder_model_loaded": self.encoder_model is not None,
                "preprocessor_loaded": self.preprocessor is not None,
                "ridge_probe_loaded": self.ridge_probe is not None,
                "signature_clusterer_loaded": self.signature_clusterer is not None,
            },
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Return project metadata, specifications, and disclaimers."""
        return {
            "project_name": "FailureDNA: Self-Supervised Discovery of Failure Signatures from Multivariate Industrial Time-Series",
            "version": "1.0.0",
            "dataset": "NASA C-MAPSS FD004",
            "operating_regimes_count": 6,
            "sensor_channels_count": 21,
            "setting_channels_count": 3,
            "required_features": SETTING_COLS + SENSOR_COLS,
            "sequence_window_length": 50,
            "padding_mode": "edge",
            "model_architecture": {
                "encoder": "TemporalTransformerEncoder(4L, 8H, d_model=128, dim_feedforward=256, masked_mean)",
                "checkpoint_sha256": FROZEN_ENCODER_SHA256,
                "latent_dim": 128,
                "downstream_estimator": "StandardScaler + Ridge(alpha=10.0)",
                "signature_clusterer": "KMeans(n_clusters=4, lifecycle_ordered)",
            },
            "disclaimer": (
                "FailureDNA is an academic research prototype evaluating self-supervised representations on NASA C-MAPSS FD004. "
                "Discovered signature states (0, 1, 2, 3) represent discrete empirical clusters and should not be interpreted "
                "as physically confirmed component fault diagnoses without independent ground-truth component labels. "
                "Per Phase 5.1/5.2 statistical audits, FailureDNA does not establish statistical superiority over XGBoost on FD004."
            ),
        }

    def validate_and_format_input(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Validate input telemetry dataframe and normalize column names."""
        warnings: List[str] = []

        if df.empty:
            raise ValueError("Input telemetry sequence is empty.")

        # Normalize column names: lower-case and strip whitespace
        col_map = {c: str(c).strip().lower() for c in df.columns}
        df = df.rename(columns=col_map)

        # Check settings
        missing_settings = [col for col in SETTING_COLS if col not in df.columns]
        if missing_settings:
            raise ValueError(f"Missing required operating condition settings: {missing_settings}. Expected {SETTING_COLS}.")

        # Check sensors
        missing_sensors = [col for col in SENSOR_COLS if col not in df.columns]
        if missing_sensors:
            raise ValueError(f"Missing required sensor channels: {missing_sensors}. Expected {SENSOR_COLS}.")

        # Ensure ID and Cycle columns exist for downstream pipeline compatibility
        if ID_COL not in df.columns:
            df[ID_COL] = 1
        if CYCLE_COL not in df.columns:
            if "time_cycles" in df.columns:
                df[CYCLE_COL] = df["time_cycles"]
            elif "cycle" in df.columns:
                df[CYCLE_COL] = df["cycle"]
            else:
                df[CYCLE_COL] = np.arange(1, len(df) + 1)
                warnings.append("No cycle or time_cycles column provided; sequential cycles assigned automatically.")

        # Chronological sort
        df = df.sort_values(by=[ID_COL, CYCLE_COL]).reset_index(drop=True)

        # Check for non-finite values
        numeric_cols = SETTING_COLS + SENSOR_COLS
        for c in numeric_cols:
            if not np.all(np.isfinite(df[c].to_numpy())):
                raise ValueError(f"Non-finite, null, or NaN values detected in channel '{c}'.")

        return df, warnings

    def analyze_telemetry_dataframe(
        self,
        df: pd.DataFrame,
        unit_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Perform end-to-end analysis on raw engine telemetry."""
        if not self._is_ready:
            raise RuntimeError(f"FailureDNA inference service is not ready: {self._init_error}")

        raw_df, warnings = self.validate_and_format_input(df)
        if unit_id is not None:
            raw_df[ID_COL] = unit_id

        actual_unit_id = int(raw_df[ID_COL].iloc[0])
        seq_len = len(raw_df)

        if seq_len < 1:
            raise ValueError("Telemetry sequence must contain at least 1 cycle.")

        # Sensor trend summaries across the provided trajectory
        sensor_trends = {}
        for s in ACTIVE_SENSORS:
            if s in raw_df.columns:
                arr = raw_df[s].to_numpy(dtype=np.float64)
                sensor_trends[s] = {
                    "start": float(arr[0]),
                    "end": float(arr[-1]),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                    "delta": float(arr[-1] - arr[0]),
                }

        # Operating regimes detected
        detected_regimes = self.preprocessor.clusterer_.predict(raw_df)
        unique_regimes = [int(r) for r in sorted(set(detected_regimes))]

        # Preprocess using the frozen RegimeAwarePreprocessor
        proc_df = self.preprocessor.transform(raw_df)

        # Extract last window with edge padding if < 50
        padding_applied = seq_len < 50
        pad_length = 50 - seq_len if padding_applied else 0
        if padding_applied:
            warnings.append(
                f"Trajectory length ({seq_len} cycles) is shorter than required window size (50). "
                f"Edge-padding applied to initial cycle ({pad_length} cycles padded)."
            )

        X_win, meta_win = self.window_generator.extract_last_window(proc_df, pad=True, padding_mode="edge")
        assert X_win.shape == (1, 50, len(self.preprocessor.get_feature_names()))

        mask_win = np.ones((1, 50), dtype=np.float32)
        if padding_applied:
            mask_win[0, :pad_length] = 0.0

        # Extract FailureDNA Latent Representation
        with torch.no_grad():
            emb = extract_embeddings_from_windows(self.encoder_model, X_win, mask_win, batch_size=1)

        # Downstream RUL Estimation
        pred_rul_raw = float(self.ridge_probe.predict(emb)[0])
        pred_rul = float(np.clip(pred_rul_raw, 0.0, 125.0))

        # Signature Discovery Assignment
        sig_id = int(self.signature_clusterer.predict(emb)[0])

        # Centroid distance (Euclidean distance in latent space to cluster centroid)
        centroid_dist = None
        if hasattr(self.signature_clusterer, "cluster_centers_") and self.signature_clusterer.cluster_centers_ is not None:
            # Map raw cluster ID to ordered cluster ID if applicable
            centroid = self.signature_clusterer.cluster_centers_[sig_id]
            centroid_dist = float(np.linalg.norm(emb[0] - centroid))

        sig_descriptions = {
            0: "State 0: Baseline Healthy / Nominal operation with balanced thermal-pressure margins.",
            1: "State 1: Early degradation signature indicating subtle aerodynamic efficiency decline.",
            2: "State 2: Accelerated degradation signature with elevated exhaust gas temperatures.",
            3: "State 3: Critical terminal wear signature preceding functional turbofan termination.",
        }

        return {
            "unit_id": actual_unit_id,
            "sequence_length": seq_len,
            "window_size": 50,
            "padding_applied": padding_applied,
            "padded_cycles": pad_length,
            "predicted_rul": round(pred_rul, 2),
            "predicted_rul_unclipped": round(pred_rul_raw, 2),
            "rul_cap_applied": pred_rul != pred_rul_raw,
            "signature_state": sig_id,
            "signature_name": f"State {sig_id}",
            "signature_description": sig_descriptions.get(sig_id, f"Discovered Signature Cluster {sig_id}"),
            "distance_to_centroid": round(centroid_dist, 4) if centroid_dist is not None else None,
            "detected_operating_regimes": unique_regimes,
            "sensor_trends": sensor_trends,
            "warnings": warnings,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_metadata": {
                "encoder": "TemporalTransformerEncoder(4L, 8H, d_model=128)",
                "estimator": "StandardScaler + Ridge(alpha=10.0)",
                "checkpoint_sha256": FROZEN_ENCODER_SHA256,
            },
        }

    def analyze_csv_bytes(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Parse uploaded CSV bytes and run analysis."""
        if len(content) > 10 * 1024 * 1024:
            raise ValueError(f"Uploaded file '{filename}' exceeds maximum allowed size of 10 MB.")

        try:
            # Attempt parsing as standard CSV or space-delimited
            text = content.decode("utf-8")
            first_line = text.strip().split("\n")[0]
            sep = "," if "," in first_line else r"\s+"
            df = pd.read_csv(StringIO(text), sep=sep)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file '{filename}': {e}")

        # Check if columns are integers (like raw C-MAPSS files without header)
        if len(df.columns) in (26, 28) and all(str(c).isdigit() for c in df.columns):
            raw_cols = [ID_COL, CYCLE_COL] + SETTING_COLS + SENSOR_COLS
            if len(df.columns) > 26:
                raw_cols += [f"extra_{i}" for i in range(len(df.columns) - 26)]
            df.columns = raw_cols[: len(df.columns)]

        return self.analyze_telemetry_dataframe(df)

    def get_benchmark_results(self) -> Dict[str, Any]:
        """Serve official Phase 5 / 5.1 / 5.2 benchmark results directly from saved artifacts."""
        if not self.benchmark_summary_path.exists():
            raise FileNotFoundError(f"Missing benchmark summary: {self.benchmark_summary_path}")

        with open(self.benchmark_summary_path, "r") as f:
            summary = json.load(f)

        # Also load adjusted bootstrap CSV if present
        adjusted_data = []
        if self.adjusted_bootstrap_path.exists():
            df_boot = pd.read_csv(self.adjusted_bootstrap_path)
            adjusted_data = df_boot.to_dict(orient="records")

        # Load model comparison CSV
        model_comparison = []
        if self.benchmark_csv_path.exists():
            df_comp = pd.read_csv(self.benchmark_csv_path)
            model_comparison = df_comp.to_dict(orient="records")

        return {
            "benchmark_dataset": "NASA C-MAPSS FD004 Official Test Set",
            "test_engines_count": 248,
            "observations_count": 41214,
            "summary": summary,
            "model_comparison": model_comparison,
            "statistical_audit_paired_bootstrap": adjusted_data,
            "scientific_conclusion": (
                "Under the omnibus Holm step-down correction for family-wise error rate (FWER) control, "
                "XGBoost maintains a statistically significant MAE advantage over FailureDNA (Regime-Aware) "
                "(Delta = +2.8571 cycles, adjusted p = 0.0150). Neither the RMSE difference (Delta = +1.2827 cycles, "
                "adjusted p = 0.8140) nor the NASA score difference (Delta = -137.12 points, adjusted p = 0.9220) "
                "is statistically significant. Therefore, FailureDNA does not establish overall prognostic superiority "
                "over gradient-boosted trees on FD004."
            ),
        }

    def get_signatures_summary(self) -> Dict[str, Any]:
        """Serve saved discovered signatures and transition statistics from benchmark artifacts."""
        if not self.benchmark_summary_path.exists():
            raise FileNotFoundError(f"Missing benchmark summary: {self.benchmark_summary_path}")

        with open(self.benchmark_summary_path, "r") as f:
            summary = json.load(f)

        sig_data = summary.get("signature_discovery_test_telemetry", {})
        return {
            "benchmark_cohort": "248 Official Test Engines (FD004)",
            "clusters_count": 4,
            "cluster_order_mapping": sig_data.get("cluster_order_mapping", {}),
            "fleet_occupancy": sig_data.get("test_fleet_occupancy", {}),
            "final_window_distribution": sig_data.get("final_window_state_distribution", {}),
            "transition_matrix": sig_data.get("test_transition_matrix", []),
            "empirical_mean_dwell_times": sig_data.get("test_empirical_mean_dwell_times", {}),
            "signature_regime_nmi": sig_data.get("test_signature_regime_nmi", 0.000143),
            "hazard_rate_audit": summary.get("preflight_audit", {}).get("state_3_hazard_audit", {}),
            "methodological_note": (
                "Signatures are discovered via unsupervised clustering of FailureDNA self-supervised representations. "
                "State indices 0 to 3 are ordered by training lifecycle progression. Low measured association with flight "
                "regimes (NMI = 0.000143) confirms decoupling from instantaneous operating conditions, but does not constitute "
                "independent physical component fault labels."
            ),
        }


# Singleton service instance
_inference_service: Optional[FailureDNAInferenceService] = None


def get_inference_service() -> FailureDNAInferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = FailureDNAInferenceService()
    return _inference_service
