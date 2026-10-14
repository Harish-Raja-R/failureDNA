"""Unit and Integration Tests for FailureDNA FastAPI Service."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from failuredna.api.main import app
from failuredna.data.loader import CYCLE_COL, ID_COL, SENSOR_COLS, SETTING_COLS, load_raw_cmapss_file
from failuredna.inference.service import get_inference_service


@pytest.fixture(scope="module")
def client():
    """Create test client with lifespan initialization."""
    with TestClient(app) as test_client:
        yield test_client


def test_api_health_endpoint(client):
    """Verify /health endpoint returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["is_ready"] is True
    assert data["artifacts"]["frozen_checkpoint_exists"] is True
    assert data["artifacts"]["encoder_model_loaded"] is True
    assert data["artifacts"]["preprocessor_loaded"] is True
    assert data["artifacts"]["ridge_probe_loaded"] is True
    assert data["artifacts"]["signature_clusterer_loaded"] is True


def test_api_metadata_endpoint(client):
    """Verify /api/v1/metadata returns specifications and disclaimers."""
    response = client.get("/api/v1/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "FailureDNA" in data["project_name"]
    assert data["dataset"] == "NASA C-MAPSS FD004"
    assert data["sensor_channels_count"] == 21
    assert data["setting_channels_count"] == 3
    assert data["sequence_window_length"] == 50
    assert "disclaimer" in data
    assert "XGBoost" in data["disclaimer"]


def test_api_research_benchmark_endpoint(client):
    """Verify /api/v1/research/benchmark serves official saved metrics."""
    response = client.get("/api/v1/research/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert data["benchmark_dataset"] == "NASA C-MAPSS FD004 Official Test Set"
    assert data["test_engines_count"] == 248

    model_names = [m["model_name"] for m in data["model_comparison"]]
    assert any("FailureDNA (Regime-Aware)" in name for name in model_names)
    assert any("XGBoost" in name for name in model_names)

    # Verify explicit scientific conclusion
    assert "FWER" in data["scientific_conclusion"] or "Holm" in data["scientific_conclusion"]
    assert "statistically significant" in data["scientific_conclusion"]


def test_api_research_signatures_endpoint(client):
    """Verify /api/v1/research/signatures returns transition dynamics and NMI."""
    response = client.get("/api/v1/research/signatures")
    assert response.status_code == 200
    data = response.json()
    assert data["clusters_count"] == 4
    assert len(data["transition_matrix"]) == 4
    assert len(data["transition_matrix"][0]) == 4
    assert abs(data["signature_regime_nmi"] - 0.000143) < 0.0001
    assert "State_0" in data["fleet_occupancy"]


def test_api_analyze_valid_sequence(client):
    """Verify /api/v1/analyze processes a valid sequence and reproduces engine 1 results."""
    raw_test = load_raw_cmapss_file("data/raw/test_FD004.txt")
    eng1 = raw_test[raw_test[ID_COL] == 1].copy()
    seq_records = eng1.to_dict(orient="records")

    payload = {"unit_id": 1, "sequence": seq_records}
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["unit_id"] == 1
    assert data["sequence_length"] == len(eng1)
    assert data["padding_applied"] is False
    assert abs(data["predicted_rul"] - 47.28) < 0.1
    assert data["signature_state"] == 3
    assert data["signature_name"] == "State 3"
    assert data["distance_to_centroid"] is not None
    assert len(data["sensor_trends"]) > 0


def test_api_analyze_short_sequence_edge_padding(client):
    """Verify short sequence (<50 cycles) receives edge-padding and warning."""
    raw_test = load_raw_cmapss_file("data/raw/test_FD004.txt")
    eng1 = raw_test[raw_test[ID_COL] == 1].iloc[:20].copy()
    seq_records = eng1.to_dict(orient="records")

    payload = {"unit_id": 1001, "sequence": seq_records}
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["unit_id"] == 1001
    assert data["sequence_length"] == 20
    assert data["padding_applied"] is True
    assert data["padded_cycles"] == 30
    assert any("shorter than required window size" in w for w in data["warnings"])
    assert 0.0 <= data["predicted_rul"] <= 125.0


def test_api_analyze_rejects_empty_sequence(client):
    """Verify empty sequence returns 422 Unprocessable Entity."""
    payload = {"unit_id": 1, "sequence": []}
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422


def test_api_analyze_rejects_non_finite_values(client):
    """Verify non-numeric or malformed values are rejected with 422."""
    raw_test = load_raw_cmapss_file("data/raw/test_FD004.txt")
    eng1 = raw_test[raw_test[ID_COL] == 1].iloc[:50].copy()
    seq_records = eng1.to_dict(orient="records")
    # Non-numeric string value for sensor
    seq_records[5]["s_2"] = "not_a_number"

    payload = {"unit_id": 1, "sequence": seq_records}
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422

    # Also verify CSV with NaN values is rejected with 422
    eng_nan = raw_test[raw_test[ID_COL] == 1].iloc[:50].copy()
    eng_nan.loc[eng_nan.index[5], "s_2"] = np.nan
    csv_bytes = eng_nan.to_csv(index=False).encode("utf-8")
    files = {"file": ("test_nan.csv", csv_bytes, "text/csv")}
    response = client.post("/api/v1/analyze/csv", files=files)
    assert response.status_code == 422


def test_api_analyze_csv_upload_success(client):
    """Verify /api/v1/analyze/csv successfully parses and analyzes uploaded CSV."""
    raw_test = load_raw_cmapss_file("data/raw/test_FD004.txt")
    eng2 = raw_test[raw_test[ID_COL] == 2].copy()
    csv_bytes = eng2.to_csv(index=False).encode("utf-8")

    files = {"file": ("test_engine_2.csv", csv_bytes, "text/csv")}
    response = client.post("/api/v1/analyze/csv", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["unit_id"] == 2
    # Saved prediction for engine 2: 69.1303
    assert abs(data["predicted_rul"] - 69.13) < 0.1
    # Saved signature for engine 2: 1
    assert data["signature_state"] == 1


def test_api_analyze_csv_rejects_invalid_extension(client):
    """Verify non-csv/txt uploads are rejected with 400 Bad Request."""
    files = {"file": ("malicious.exe", b"MZ...", "application/octet-stream")}
    response = client.post("/api/v1/analyze/csv", files=files)
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_api_analyze_csv_rejects_missing_columns(client):
    """Verify CSV missing required sensor channels returns 422."""
    df_bad = pd.DataFrame({"cycle": [1, 2], "setting_1": [0.0, 0.0]})
    files = {"file": ("bad_schema.csv", df_bad.to_csv(index=False).encode("utf-8"), "text/csv")}
    response = client.post("/api/v1/analyze/csv", files=files)
    assert response.status_code == 422


def test_api_serves_static_frontend(client):
    """Verify root endpoint / serves the built React frontend when dist exists."""
    response = client.get("/")
    assert response.status_code == 200
    assert "FailureDNA" in response.text

