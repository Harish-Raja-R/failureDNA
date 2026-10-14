# FailureDNA: Cloud Deployment Guide (Render.com)

This guide documents the public cloud deployment process for the **FailureDNA** platform on [Render.com](https://render.com), satisfying Requirement 12 (*Public cloud deployment*) of the 19MAM56 CAT-II specification.

---

## 1. Architecture Overview

FailureDNA is containerized using a multi-stage production `Dockerfile`:
- **Stage 1 (Frontend Builder)**: Node.js 22 Alpine compiles the React 18 + TypeScript + Tailwind CSS application into optimized static assets (`frontend/dist/`).
- **Stage 2 (Production Serving)**: Python 3.11 slim installs PyTorch, scikit-learn, XGBoost, and FastAPI, copies frozen model checkpoints, mounts the static frontend at `/`, and runs a production Uvicorn server.
- **Dynamic Port Binding**: Supports `$PORT` environment variable injection (Render assigns dynamic internal ports, e.g. 10000).
- **Automated Healthcheck**: Continuously monitors `http://localhost:${PORT}/health`.

---

## 2. One-Click Blueprint Deployment (Recommended)

Because [render.yaml](file:///c:/Users/Harish/Downloads/FailureDNA/render.yaml) is committed to the root of the repository, you can deploy using Render's Infrastructure-as-Code (IaC) Blueprint:

1. **Sign in to Render**:
   - Go to [dashboard.render.com](https://dashboard.render.com) and log in with GitHub.

2. **Create New Blueprint**:
   - Click the **New +** button in the top navigation bar.
   - Select **Blueprint**.

3. **Connect Repository**:
   - Select your repository: `Harish-Raja-R/failureDNA`.
   - Render automatically parses `render.yaml` and detects:
     - **Service Name**: `failuredna`
     - **Environment**: `Docker`
     - **Plan**: `Free`
     - **Healthcheck Path**: `/health`

4. **Click "Apply"**:
   - Render will build the Docker container and deploy the application.
   - Once complete, you will receive a public HTTPS URL (e.g., `https://failuredna.onrender.com`).

---

## 3. Alternative: Manual Web Service Deployment

If you prefer to configure the Web Service manually via the UI:

1. Go to [dashboard.render.com](https://dashboard.render.com) -> **New +** -> **Web Service**.
2. Select `Harish-Raja-R/failureDNA`.
3. Configure the following fields:
   - **Name**: `failuredna`
   - **Region**: `Oregon (US West)` or `Frankfurt (EU Central)`
   - **Branch**: `main`
   - **Language**: `Docker`
   - **Dockerfile Path**: `./Dockerfile`
   - **Instance Type**: `Free`
4. Expand **Advanced Settings**:
   - **Health Check Path**: `/health`
   - **Auto-Deploy**: `Yes`
5. Click **Create Web Service**.

---

## 4. Verifying Public Cloud Deployment

Once Render finishes deploying (approximately 3–5 minutes):

1. **Web Interface**:
   - Open your assigned URL: `https://<your-service-name>.onrender.com/`
   - Verify that all 5 pages load with the light editorial theme.
2. **Health Endpoint**:
   - Visit `https://<your-service-name>.onrender.com/health`
   - Expected JSON response:
     ```json
     {
       "status": "healthy",
       "version": "1.0.0",
       "is_ready": true,
       "error": null,
       "artifacts": {
         "frozen_checkpoint_exists": true,
         "encoder_model_loaded": true,
         "preprocessor_loaded": true,
         "ridge_probe_loaded": true,
         "signature_clusterer_loaded": true
       }
     }
     ```
3. **Interactive Inference**:
   - Navigate to **Engine Analysis** on your live URL.
   - Click **Unit #098 (Critical)** to execute live inference against the cloud-hosted frozen PyTorch model.
4. **Interactive Swagger Docs**:
   - Visit `https://<your-service-name>.onrender.com/docs` to test endpoints via OpenAPI.
