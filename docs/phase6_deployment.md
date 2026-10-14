# FailureDNA — Phase 6: Deployment & Operations Guide

This guide details the deployment architecture, configuration options, local execution pathways, containerization strategies, and public cloud deployment options for the **FailureDNA** industrial prognostic platform.

---

## 1. Architecture & Runtime Model

FailureDNA utilizes a modular decoupled architecture that supports two operational deployment models:

```
[ Web Browser / Client ]
           │
           │ HTTP / HTTPS (JSON & Multipart/form-data)
           ▼
[ Reverse Proxy / Port 8000 (Production Unified Container) ]
           │
     ┌─────┴─────────────────────────┐
     ▼                               ▼
[ React + TS Static Assets ]   [ FastAPI REST Engine ]
  (frontend/dist bundle)       (Uvicorn ASGI Server)
                                     │
                                     ▼
                               [ Lifespan State ]
                               ├── Preprocessor (Regime-wise Normalization)
                               ├── Temporal Transformer (Frozen Checkpoint)
                               ├── K-Means Signature Clusterer (k=4)
                               └── Ridge Prognostic Probe (alpha=100.0)
```

1. **Unified Production Container (Recommended):**
   A multi-stage Docker build builds the React TypeScript Vite frontend into static HTML/JS/CSS assets and copies them into a lightweight Python 3.11-slim container. FastAPI serves the static bundle at root `/` and provides the typed REST endpoints under `/api/v1/` and `/health`.
2. **Decoupled Development Mode:**
   FastAPI runs via `uvicorn` on port 8000 with CORS enabled for `http://localhost:5173`. The Vite development server runs concurrently on port 5173 with hot-module reloading and proxy support.

---

## 2. Environment Variables & Configuration

The application is configured using environment variables with safe production fallbacks:

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `PORT` | int | `8000` | Port for the Uvicorn ASGI server |
| `HOST` | string | `0.0.0.0` | Host IP binding |
| `FAILUREDNA_ENV` | string | `production` | Environment mode (`development` or `production`) |
| `CORS_ORIGINS` | string | `http://localhost:5173,http://localhost:3000,http://localhost:8000` | Comma-separated list of allowed CORS origins |
| `FAILUREDNA_CHECKPOINT_PATH` | string | `models/checkpoints/failuredna_regime_aware.pt` | Path to frozen PyTorch model weights |
| `FAILUREDNA_BUNDLE_DIR` | string | `models/inference_bundle` | Path to preprocessor, probe, and clusterer joblib artifacts |
| `MAX_UPLOAD_SIZE_MB` | int | `50` | Maximum allowable CSV upload payload size |
| `INFERENCE_TIMEOUT_SEC` | int | `30` | Timeout threshold for inference requests |

---

## 3. Local Execution Pathways

### Pathway A: Standalone Python (FastAPI + Built Frontend)

Run the full integrated platform natively on your local machine:

```bash
# 1. Activate your virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Build the frontend static bundle
cd frontend
npm install
npm run build
cd ..

# 3. Launch the FastAPI application
python -m uvicorn failuredna.api.main:app --host 0.0.0.0 --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser to view the React dashboard and interactive API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

### Pathway B: Concurrent Development Mode

For active frontend or backend development:

**Terminal 1 (Backend):**
```bash
python -m uvicorn failuredna.api.main:app --reload --port 8000
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```
Access the Vite dev server at [http://localhost:5173](http://localhost:5173).

---

## 4. Containerized Deployment (Docker & Compose)

### Single Unified Production Container

Build and run the self-contained production image:

```bash
# 1. Build the unified Docker image
docker build -t failuredna:latest .

# 2. Run the container
docker run -d \
  --name failuredna \
  -p 8000:8000 \
  -e FAILUREDNA_ENV=production \
  failuredna:latest

# 3. Verify container health
curl http://localhost:8000/health
```

### Multi-Container Deployment via Docker Compose

To run separate backend and frontend services with explicit network isolation:

```bash
# Build and start services in the background
docker compose up -d --build

# View logs
docker compose logs -f

# Check container status
docker compose ps

# Clean shutdown
docker compose down
```

Compose exposes:
- Frontend Dashboard: [http://localhost:3000](http://localhost:3000)
- Backend REST API: [http://localhost:8000](http://localhost:8000)
- Interactive API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 5. Public Cloud Deployment Guidelines

FailureDNA is fully containerized and designed for zero-infrastructure deployment across leading cloud container platforms. Below are verified configuration recipes for deployment.

### Target 1: Hugging Face Spaces (Docker SDK) — Recommended Free Tier

Hugging Face Spaces provides container hosting for machine learning research applications with CPU and GPU hardware tiers.

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Select **Docker** as the Space SDK.
3. Configure `README.md` YAML metadata in the root of the Space repository:
   ```yaml
   ---
   title: FailureDNA
   emoji: 🧬
   colorFrom: indigo
   colorTo: purple
   sdk: docker
   app_port: 8000
   pinned: false
   ---
   ```
4. Push the FailureDNA repository to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/FailureDNA
   git push space main
   ```
5. Hugging Face will automatically execute the multi-stage `Dockerfile`, build the React UI, initialize the frozen inference bundle, and expose the live web application on port 8000.

### Target 2: Render.com / Railway.app (Web Service)

1. Connect your GitHub repository to Render or Railway.
2. Select **Docker** as the environment runtime.
3. Set Environment Variables:
   - `PORT`: `8000`
   - `FAILUREDNA_ENV`: `production`
   - `CORS_ORIGINS`: `https://<your-render-subdomain>.onrender.com`
4. Health Check Path: `/health`
5. Deploy. The platform will build the multi-stage container and provision free automated TLS (HTTPS).

### Target 3: AWS App Runner / Google Cloud Run

For enterprise or scalable serverless container execution:

1. Build and push the Docker image to your container registry (e.g., AWS ECR or Google Artifact Registry):
   ```bash
   # Authenticate and tag
   docker tag failuredna:latest <account_id>.dkr.ecr.<region>.amazonaws.com/failuredna:latest
   docker push <account_id>.dkr.ecr.<region>.amazonaws.com/failuredna:latest
   ```
2. Create an App Runner or Cloud Run service pointing to the container image.
3. Allocate resource specifications:
   - **vCPU:** 2.0 vCPU
   - **Memory:** 4.0 GB RAM
   - **Port:** 8000
   - **Healthcheck Path:** `/health`
4. Deploy with automatic scaling from 0 to $N$ instances based on concurrent HTTP requests.

---

## 6. Health Checks, Verification & Monitoring

### Health Endpoint Specification

The `/health` endpoint distinguishes general web process availability from model artifact readiness:

```bash
curl -s http://localhost:8000/health | jq
```

Expected Response (`200 OK`):
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

If any critical artifact is missing or corrupted, the endpoint returns `503 Service Unavailable`:
```json
{
  "status": "degraded",
  "version": "1.0.0",
  "artifacts_loaded": false,
  "device": "cpu",
  "model_type": "None",
  "sensor_count": 21,
  "window_length": 50,
  "inference_bundle": {
    "error": "Checkpoint file not found: models/checkpoints/failuredna_regime_aware.pt"
  }
}
```

### Automated Live Verification Script

You can verify that a newly deployed instance is operating properly by running:

```bash
python scratch/verify_api_live.py --base-url http://localhost:8000
```
This script checks:
- Connection and SSL certificate validity
- Artifact loading and device allocation
- Metadata consistency against the NASA C-MAPSS FD004 specification
- End-to-end inference execution on a synthetic telemetry trajectory
- Retrieval of frozen benchmark records and signature transition matrices

---

## 7. Security & Hardening Checklist

1. **Non-Root Execution:** The production `Dockerfile` provisions an unprivileged `appuser` (UID 1000) with isolated home permissions. No process executes as `root`.
2. **Minimal Attack Surface:** The runtime container is built on `python:3.11-slim` with build dependencies (`gcc`, build tools) removed in intermediate stages.
3. **Bounded Uploads:** File upload endpoints enforce strict file type headers (`.csv`, `text/csv`) and payload size limits (`MAX_UPLOAD_SIZE_MB = 50`) to defend against memory exhaustion and DoS attacks.
4. **Input Sanitization:** Telemetry matrices are validated for shape, finiteness, non-negativity of operational cycles, and numerical bounds before tensor conversion.
5. **CORS Isolation:** Cross-Origin Resource Sharing is restricted strictly to permitted hostnames via `CORS_ORIGINS`.
6. **Information Disclosure Prevention:** Public API errors sanitize tracebacks and filesystem paths, returning RFC 7807 structured JSON failure payloads.
