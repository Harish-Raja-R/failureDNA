"""API tests using deterministic telemetry fixtures.

These tests exercise the API contract without requiring the private/raw NASA
C-MAPSS FD004 dataset. Real-data benchmark validation belongs in a separate
opt-in integration test suite.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from failuredna.api.main import app
from failuredna.data.loader import CYCLE_COL, ID_COL, SENSOR_COLS, SETTING_COLS


def make_telemetry(unit_id: int = 1, cycles: int = 60) -> pd.DataFrame:
    """Create deterministic, schema-valid telemetry for API integration tests."""
    data = {
        ID_COL: [unit_id] * cycles,
        CYCLE_COL: np.arange(1, cycles + 1),
        "setting_1": np.zeros(cycles, dtype=float),
        "setting_2": np.zeros(cycles, dtype=float),
        "setting_3": np.full(cycles, 100.0, dtype=float),
    }
    for index, sensor in enumerate(SENSOR_COLS, start=1):
        data[sensor] = np.linspace(float(index), float(index) + 0.5, cycles)
    return pd.DataFrame(data)


def sequence_payload(df: pd.DataFrame, unit_id: int = 1) -> dict:
    """Convert a telemetry frame to the JSON request schema."""
    columns = SETTING_COLS + SENSOR_COLS
    records = []
    for _, row in df.iterrows():
        record = {"cycle": int(row[CYCLE_COL])}
        record.update({column: float(row[column]) for column in columns})
        records.append(record)
    return {"unit_id": unit_id, "sequence": records}


@pytest.fixture(scope="module")
def client():
    """Create a test client and initialize the configured inference service."""
    with TestClient(app) as test_client:
        yield test_client


def test_api_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["is_ready"] is True
    assert data["artifacts"]["encoder_model_loaded"] is True
    assert data["artifacts"]["preprocessor_loaded"] is True
    assert data["artifacts"]["ridge_probe_loaded"] is True
    assert data["artifacts"]["signature_clusterer_loaded"] is True


def test_api_metadata_endpoint(client):
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
    response = client.get("/api/v1/research/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert data["benchmark_dataset"] == "NASA C-MAPSS FD004 Official Test Set"
    assert data["test_engines_count"] == 248
    model_names = [model["model_name"] for model in data["model_comparison"]]
    assert any("FailureDNA (Regime-Aware)" in name for name in model_names)
    assert any("XGBoost" in name for name in model_names)
    assert "FWER" in data["scientific_conclusion"] or "Holm" in data["scientific_conclusion"]
    assert "statistically significant" in data["scientific_conclusion"]


def test_api_research_signatures_endpoint(client):
    response = client.get("/api/v1/research/signatures")
    assert response.status_code == 200
    data = response.json()
    assert data["clusters_count"] == 4
    assert len(data["transition_matrix"]) == 4
    assert all(len(row) == 4 for row in data["transition_matrix"])
    assert "signature_regime_nmi" in data
    assert "State_0" in data["fleet_occupancy"]


def test_api_analyze_valid_sequence(client):
    payload = sequence_payload(make_telemetry(unit_id=7), unit_id=7)
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["unit_id"] == 7
    assert data["sequence_length"] == 60
    assert data["padding_applied"] is False
    assert 0.0 <= data["predicted_rul"] <= 125.0
    assert 0 <= data["signature_state"] <= 3
    assert data["distance_to_centroid"] is not None
    assert len(data["sensor_trends"]) > 0


def test_api_analyze_short_sequence_edge_padding(client):
    payload = sequence_payload(make_telemetry(unit_id=1001, cycles=20), unit_id=1001)
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["unit_id"] == 1001
    assert data["sequence_length"] == 20
    assert data["padding_applied"] is True
    assert data["padded_cycles"] == 30
    assert any("shorter than required window size" in warning for warning in data["warnings"])
    assert 0.0 <= data["predicted_rul"] <= 125.0


def test_api_analyze_rejects_empty_sequence(client):
    response = client.post("/api/v1/analyze", json={"unit_id": 1, "sequence": []})
    assert response.status_code == 422


def test_api_analyze_rejects_non_numeric_values(client):
    payload = sequence_payload(make_telemetry(cycles=50))
    payload["sequence"][5]["s_2"] = "not_a_number"
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422


def test_api_analyze_csv_upload_success(client):
    telemetry = make_telemetry(unit_id=22)
    csv_bytes = telemetry.to_csv(index=False).encode("utf-8")
    response = client.post(
        "/api/v1/analyze/csv",
        files={"file": ("synthetic_engine.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["unit_id"] == 22
    assert data["sequence_length"] == len(telemetry)
    assert 0.0 <= data["predicted_rul"] <= 125.0


def test_api_analyze_csv_rejects_non_finite_values(client):
    telemetry = make_telemetry(cycles=50)
    telemetry.loc[5, "s_2"] = np.nan
    response = client.post(
        "/api/v1/analyze/csv",
        files={"file": ("non_finite.csv", telemetry.to_csv(index=False).encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 422


def test_api_analyze_csv_accepts_uppercase_extension(client):
    telemetry = make_telemetry(unit_id=23)
    response = client.post(
        "/api/v1/analyze/csv",
        files={"file": ("synthetic_engine.CSV", telemetry.to_csv(index=False).encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 200
    assert response.json()["unit_id"] == 23


def test_api_analyze_csv_rejects_multiple_units(client):
    first = make_telemetry(unit_id=1, cycles=10)
    second = make_telemetry(unit_id=2, cycles=10)
    combined = pd.concat([first, second], ignore_index=True)
    response = client.post(
        "/api/v1/analyze/csv",
        files={"file": ("multiple_units.csv", combined.to_csv(index=False).encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 422
    assert "multiple unit IDs" in response.json()["detail"]


def test_api_analyze_csv_rejects_invalid_extension(client):
    response = client.post(
        "/api/v1/analyze/csv",
        files={"file": ("malicious.exe", b"MZ...", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_api_analyze_csv_rejects_missing_columns(client):
    frame = pd.DataFrame({"cycle": [1, 2], "setting_1": [0.0, 0.0]})
    response = client.post(
        "/api/v1/analyze/csv",
        files={"file": ("bad_schema.csv", frame.to_csv(index=False).encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 422


def test_api_serves_static_frontend(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "FailureDNA" in response.text
