# FailureDNA — REST API Specification (Phase 6)

The **FailureDNA REST API** provides programmatic access to self-supervised failure signature discovery, remaining useful life (RUL) estimation, and frozen NASA C-MAPSS FD004 benchmark results.

Interactive documentation and OpenAPI specifications are available at:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc UI:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI Schema:** [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

## 1. General Conventions

- **Base URL:** `/api/v1` (with system endpoints at `/health`)
- **Content Types:**
  - `application/json` for standard structured telemetry requests
  - `multipart/form-data` for file uploads (`POST /api/v1/analyze/csv`)
- **Authentication:** Public research prototype (no API key required)
- **CORS:** Configurable via `CORS_ORIGINS` environment variable

---

## 2. Endpoints Overview

| Method | Path | Description | Response Codes |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Healthcheck and artifact readiness | `200`, `503` |
| `GET` | `/api/v1/metadata` | Model metadata, schema, and capabilities | `200` |
| `POST` | `/api/v1/analyze` | Run inference on a JSON telemetry sequence | `200`, `400`, `422`, `503` |
| `POST` | `/api/v1/analyze/csv` | Run inference on an uploaded CSV telemetry file | `200`, `400`, `422`, `503` |
| `GET` | `/api/v1/research/benchmark` | Retrieve official NASA C-MAPSS FD004 benchmark records | `200` |
| `GET` | `/api/v1/research/signatures` | Retrieve discovered signature profiles & transition matrices | `200` |

---

## 3. Detailed Endpoint Reference

### 3.1 `GET /health`

Checks the availability of the web server and the operational readiness of all loaded inference artifacts (PyTorch checkpoint, regime-aware preprocessor, signature clusterer, and prognostic probe).

#### Example Request
```bash
curl -X GET "http://localhost:8000/health"
```

#### Example Response (`200 OK`)
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "artifacts_loaded": true,
  "device": "cpu",
  "model_type": "TemporalTransformerEncoder",
  "sensor_count": 21,
  "window_length": 50,
  "inference_bundle": {
    "regime_preprocessor": "loaded",
    "signature_clusterer": "loaded (k=4)",
    "prognostic_probe": "loaded (Ridge)"
  }
}
```

---

### 3.2 `GET /api/v1/metadata`

Provides system metadata, expected input feature dimensions, operating conditions, model architectures, and research disclaimers.

#### Example Request
```bash
curl -X GET "http://localhost:8000/api/v1/metadata"
```

#### Example Response (`200 OK`)
```json
{
  "project_name": "FailureDNA",
  "version": "1.0.0",
  "description": "Self-Supervised Discovery of Failure Signatures from Multivariate Industrial Time-Series",
  "dataset": "NASA C-MAPSS FD004",
  "sensor_count": 21,
  "operational_setting_count": 3,
  "window_length": 50,
  "expected_columns": [
    "unit_id", "time_in_cycles", "setting_1", "setting_2", "setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21"
  ],
  "capabilities": {
    "temporal_representation_learning": true,
    "failure_signature_discovery": true,
    "rul_estimation": true,
    "anomaly_detection": true
  },
  "disclaimer": "FailureDNA is a research prototype evaluating self-supervised representation learning on NASA C-MAPSS FD004. Discovered signatures represent statistical clusters in latent space and must not be interpreted as validated physical root-cause diagnoses without external engineering validation."
}
```

---

### 3.3 `POST /api/v1/analyze`

Analyzes an engine telemetry sequence provided as a JSON payload. Telemetry sequences with fewer than 50 cycles are automatically padded via edge-replication.

#### Request Schema
```json
{
  "unit_id": 1,
  "cycles": [1, 2, 3, 4, 5],
  "operational_settings": [
    [42.0003, 0.8400, 100.0],
    [25.0075, 0.6200, 60.0],
    ...
  ],
  "sensors": [
    [445.0, 549.2, 1342.1, 1120.3, 3.91, 5.72, 138.2, 2211.8, 8312.4, 1.02, 41.8, 388.2, 2388.0, 8123.5, 8.32, 0.03, 390.0, 2388.0, 100.0, 38.8, 23.3],
    ...
  ]
}
```

#### Example Request
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "unit_id": 1,
    "cycles": [1, 2],
    "operational_settings": [[42.0, 0.84, 100.0], [42.0, 0.84, 100.0]],
    "sensors": [
      [445.0, 549.2, 1342.1, 1120.3, 3.91, 5.72, 138.2, 2211.8, 8312.4, 1.02, 41.8, 388.2, 2388.0, 8123.5, 8.32, 0.03, 390.0, 2388.0, 100.0, 38.8, 23.3],
      [445.1, 549.3, 1342.3, 1120.5, 3.91, 5.72, 138.3, 2211.9, 8312.5, 1.02, 41.9, 388.3, 2388.0, 8123.6, 8.32, 0.03, 390.1, 2388.0, 100.0, 38.9, 23.4]
    ]
  }'
```

#### Example Response (`200 OK`)
```json
{
  "unit_id": 1,
  "sequence_length": 2,
  "validation_status": "valid",
  "predicted_rul": 128.45,
  "discovered_signature_id": 3,
  "signature_confidence": 0.824,
  "sensor_trends": {
    "sensor_2": { "mean": 549.25, "min": 549.2, "max": 549.3, "std": 0.071 },
    "sensor_4": { "mean": 1120.4, "min": 1120.3, "max": 1120.5, "std": 0.141 }
  },
  "warnings": [
    "Input sequence length (2 cycles) is shorter than model window (50 cycles); edge-padding applied."
  ],
  "model_info": {
    "encoder_checkpoint": "models/checkpoints/failuredna_regime_aware.pt",
    "sha256": "d18b6cd0abc2fadf12be18f6f1962707f3626e0a5054970147f50eb3620eb248",
    "regime_normalization": "RegimeAwarePreprocessor (K-Means k=6)",
    "prognostic_probe": "RidgeRegression (alpha=100.0, cap=130)"
  }
}
```

---

### 3.4 `POST /api/v1/analyze/csv`

Accepts a CSV file representing chronological engine telemetry matching the 26-column NASA C-MAPSS FD004 schema.

#### Requirements
- File size must not exceed `MAX_UPLOAD_SIZE_MB` (default 50 MB).
- Headers may be standard names (`unit_id`, `time_in_cycles`, `setting_1`... `sensor_21`) or 1-indexed integers.
- Data must contain strictly finite numerical values.
- Must contain at least 1 cycle.

#### Example Request
```bash
curl -X POST "http://localhost:8000/api/v1/analyze/csv" \
  -F "file=@data/test_engine_sample.csv"
```

---

### 3.5 `GET /api/v1/research/benchmark`

Retrieves the official NASA C-MAPSS FD004 benchmark results and paired bootstrap statistical tests with omnibus Holm-Bonferroni corrections.

#### Example Request
```bash
curl -X GET "http://localhost:8000/api/v1/research/benchmark"
```

#### Example Response (`200 OK`)
```json
{
  "benchmark_dataset": "NASA C-MAPSS FD004",
  "test_engines_count": 248,
  "evaluation_protocol": "Last-window RUL prediction with 130-cycle early degradation capping",
  "models": [
    {
      "model_name": "XGBoost (Handcrafted)",
      "mae": 19.82,
      "rmse": 26.54,
      "r2": 0.582,
      "nasa_score": 3842.1
    },
    {
      "model_name": "FailureDNA (Regime-Aware + Ridge)",
      "mae": 22.14,
      "rmse": 28.91,
      "r2": 0.504,
      "nasa_score": 4210.5
    },
    {
      "model_name": "Temporal CNN Baseline",
      "mae": 25.43,
      "rmse": 32.18,
      "r2": 0.387,
      "nasa_score": 6820.3
    },
    {
      "model_name": "LSTM Baseline",
      "mae": 26.11,
      "rmse": 33.05,
      "r2": 0.354,
      "nasa_score": 7450.9
    }
  ],
  "statistical_comparison": {
    "primary_comparison": "FailureDNA vs XGBoost",
    "mae_difference": "+2.32 cycles (XGBoost advantage)",
    "raw_p_value_mae": 0.0030,
    "adjusted_p_value_mae_holm": 0.0150,
    "statistically_significant": true,
    "rmse_difference": "+2.37 cycles",
    "adjusted_p_value_rmse_holm": 0.0820,
    "nasa_score_difference": "+368.4",
    "adjusted_p_value_nasa_holm": 0.1420
  },
  "research_conclusion": "FailureDNA has NOT established overall prognostic superiority over XGBoost on FD004. XGBoost maintains a statistically significant MAE advantage under omnibus Holm correction. RMSE and NASA-score differences are not statistically significant."
}
```

---

### 3.6 `GET /api/v1/research/signatures`

Retrieves the 4 discovered failure signatures, their empirical dwell times across validation trajectories, and the state-transition matrix.

#### Example Request
```bash
curl -X GET "http://localhost:8000/api/v1/research/signatures"
```

#### Example Response (`200 OK`)
```json
{
  "cluster_count": 4,
  "clustering_algorithm": "K-Means (k=4) in L2-Normalized Latent Space (R^128)",
  "validation_engine_count": 37,
  "signature_definitions": [
    {
      "signature_id": 0,
      "relative_frequency": 0.284,
      "mean_dwell_cycles": 18.4,
      "median_dwell_cycles": 12.0,
      "interpretation": "Early-life stable operational state across varied flight envelopes"
    },
    {
      "signature_id": 1,
      "relative_frequency": 0.312,
      "mean_dwell_cycles": 24.2,
      "median_dwell_cycles": 16.0,
      "interpretation": "Mid-life progressive thermal and compressor wear"
    },
    {
      "signature_id": 2,
      "relative_frequency": 0.221,
      "mean_dwell_cycles": 14.8,
      "median_dwell_cycles": 8.0,
      "interpretation": "Accelerated degradation cluster"
    },
    {
      "signature_id": 3,
      "relative_frequency": 0.183,
      "mean_dwell_cycles": 20.98,
      "median_dwell_cycles": 14.0,
      "interpretation": "Terminal degradation signature observed prior to complete failure"
    }
  ],
  "absorbing_transition_matrix": [
    [0.945, 0.045, 0.008, 0.002, 0.000],
    [0.012, 0.958, 0.026, 0.004, 0.000],
    [0.004, 0.011, 0.932, 0.053, 0.000],
    [0.000, 0.002, 0.028, 0.952, 0.018],
    [0.000, 0.000, 0.000, 0.000, 1.000]
  ]
}
```

---

## 4. Error Responses

FailureDNA uses standard HTTP status codes accompanied by detailed JSON error bodies:

| Status Code | Reason | Cause |
| :--- | :--- | :--- |
| `400 Bad Request` | Invalid Input | Malformed CSV structure, mismatched sensor dimensions, non-monotonic cycles. |
| `422 Unprocessable Content` | Schema Validation | Pydantic validation error (e.g. non-finite values, NaN/Inf, missing required fields). |
| `413 Payload Too Large` | Size Exceeded | File upload exceeds configured `MAX_UPLOAD_SIZE_MB`. |
| `503 Service Unavailable` | Model Not Ready | Inference checkpoint or bundle file failed to load during startup. |

#### Error Schema
```json
{
  "detail": "Input telemetry contains non-finite values (NaN or Inf) at index 4.",
  "error_code": "INVALID_NUMERICAL_DATA",
  "status_code": 400
}
```

---

## 5. Client Code Examples

### Python (`requests`)
```python
import requests

url = "http://localhost:8000/api/v1/analyze/csv"
with open("data/sample_engine.csv", "rb") as f:
    response = requests.post(url, files={"file": f})

print(response.status_code)
print(response.json())
```

### TypeScript (`fetch`)
```typescript
async function analyzeTelemetry(file: File) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://localhost:8000/api/v1/analyze/csv', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Analysis failed');
  }

  return await response.json();
}
```
