"""Pydantic v2 Request & Response Schemas for FailureDNA REST API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ArtifactsStatus(BaseModel):
    frozen_checkpoint_exists: bool
    encoder_model_loaded: bool
    preprocessor_loaded: bool
    ridge_probe_loaded: bool
    signature_clusterer_loaded: bool


class HealthResponse(BaseModel):
    status: str = Field(..., description="'healthy' if all artifacts are ready, 'degraded' otherwise")
    version: str = Field(..., description="API Version")
    is_ready: bool = Field(..., description="Inference readiness flag")
    initialized_at: Optional[str] = Field(None, description="Timestamp of service initialization")
    error: Optional[str] = Field(None, description="Initialization error message if degraded")
    artifacts: ArtifactsStatus


class TelemetryCycle(BaseModel):
    cycle: Optional[int] = Field(None, description="Operating cycle / time step")
    setting_1: float = Field(..., description="Operating Setting 1 (Altitude)")
    setting_2: float = Field(..., description="Operating Setting 2 (Mach Number)")
    setting_3: float = Field(..., description="Operating Setting 3 (TRA / Throttle)")
    s_1: float
    s_2: float
    s_3: float
    s_4: float
    s_5: float
    s_6: float
    s_7: float
    s_8: float
    s_9: float
    s_10: float
    s_11: float
    s_12: float
    s_13: float
    s_14: float
    s_15: float
    s_16: float
    s_17: float
    s_18: float
    s_19: float
    s_20: float
    s_21: float

    @field_validator("*", mode="before")
    @classmethod
    def check_finite(cls, v: Any) -> Any:
        if isinstance(v, (int, float)):
            import math
            if math.isnan(v) or math.isinf(v):
                raise ValueError("Non-finite numerical values (NaN/Inf) are prohibited.")
        return v


class TelemetrySequenceRequest(BaseModel):
    unit_id: Optional[int] = Field(1, description="Engine unit identifier")
    sequence: List[TelemetryCycle] = Field(
        ..., min_length=1, description="Chronological sequence of engine telemetry observations (>=1 cycles)"
    )


class SensorTrend(BaseModel):
    start: float
    end: float
    min: float
    max: float
    mean: float
    delta: float


class ModelMetadata(BaseModel):
    encoder: str
    estimator: str
    checkpoint_sha256: str


class AnalysisResponse(BaseModel):
    unit_id: int
    sequence_length: int
    window_size: int = 50
    padding_applied: bool
    padded_cycles: int
    predicted_rul: float = Field(..., description="Estimated Remaining Useful Life in cycles (capped [0, 125])")
    predicted_rul_unclipped: float = Field(..., description="Raw continuous linear probe prediction")
    rul_cap_applied: bool
    signature_state: int = Field(..., ge=0, le=3, description="Discovered discrete signature state (0 to 3)")
    signature_name: str
    signature_description: str
    distance_to_centroid: Optional[float] = Field(
        None, description="Euclidean distance in latent space from window embedding to assigned cluster center"
    )
    detected_operating_regimes: List[int]
    sensor_trends: Dict[str, SensorTrend]
    warnings: List[str]
    timestamp: str
    model_metadata: ModelMetadata


class MetadataResponse(BaseModel):
    project_name: str
    version: str
    dataset: str
    operating_regimes_count: int
    sensor_channels_count: int
    setting_channels_count: int
    required_features: List[str]
    sequence_window_length: int
    padding_mode: str
    model_architecture: Dict[str, Any]
    disclaimer: str


class BenchmarkResponse(BaseModel):
    benchmark_dataset: str
    test_engines_count: int
    observations_count: int
    summary: Dict[str, Any]
    model_comparison: List[Dict[str, Any]]
    statistical_audit_paired_bootstrap: List[Dict[str, Any]]
    scientific_conclusion: str


class SignaturesResponse(BaseModel):
    benchmark_cohort: str
    clusters_count: int
    cluster_order_mapping: Dict[str, Any]
    fleet_occupancy: Dict[str, Any]
    final_window_distribution: Dict[str, Any]
    transition_matrix: List[List[float]]
    empirical_mean_dwell_times: Dict[str, Any]
    signature_regime_nmi: float
    hazard_rate_audit: Dict[str, Any]
    methodological_note: str
