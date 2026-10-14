"""API Configuration and Environment Settings."""

import os
from pathlib import Path
from typing import List


class APISettings:
    """Application settings with environment variable overrides."""

    PROJECT_NAME: str = "FailureDNA Production API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"

    # Artifact paths
    CHECKPOINT_PATH: Path = Path(
        os.getenv("CHECKPOINT_PATH", "models/checkpoints/failuredna_regime_aware/best_failuredna_model.pt")
    )
    BUNDLE_DIR: Path = Path(os.getenv("BUNDLE_DIR", "models/inference_bundle"))
    BENCHMARK_SUMMARY_PATH: Path = Path(
        os.getenv("BENCHMARK_SUMMARY_PATH", "reports/results/phase5_official_benchmark_summary.json")
    )
    BENCHMARK_CSV_PATH: Path = Path(
        os.getenv("BENCHMARK_CSV_PATH", "reports/results/phase5_model_comparison.csv")
    )
    ADJUSTED_BOOTSTRAP_PATH: Path = Path(
        os.getenv("ADJUSTED_BOOTSTRAP_PATH", "reports/results/phase5_2_adjusted_bootstrap.csv")
    )

    # CORS Origins
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    # Additional origins from environment
    ENV_CORS = os.getenv("CORS_ORIGINS")
    if ENV_CORS:
        CORS_ORIGINS.extend([origin.strip() for origin in ENV_CORS.split(",") if origin.strip()])

    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB


settings = APISettings()
