# FailureDNA: Hugging Face Spaces Deployment Guide

This guide details how to deploy the containerized **FailureDNA** platform to **Hugging Face Spaces** using the Docker SDK.

---

## 1. Why Hugging Face Spaces?

Hugging Face Spaces provides:
- **Free ML Compute**: Free CPU tier with 16 GB RAM and 2 vCPUs.
- **Docker SDK**: Native support for building and running arbitrary Dockerfiles.
- **Public Showcase**: Automatic embeddable demo with a clean public URL (`https://huggingface.co/spaces/<username>/failuredna`).
- **Community Discovery**: Ideal for AI/ML lab evaluations, peer demonstrations, and open-source dissemination.

---

## 2. Configuration Overview

FailureDNA is pre-configured for Hugging Face Spaces:
- **YAML Frontmatter**: The root `README.md` declares `sdk: docker` and `app_port: 7860`.
- **Port Compatibility**: [Dockerfile](file:///c:/Users/Harish/Downloads/FailureDNA/Dockerfile) exposes `7860` and dynamically binds Uvicorn to `${PORT:-8000}`, which Hugging Face sets to `7860`.
- **Security**: The container runs under unprivileged non-root user `appuser` (UID 1000) with home directory `/home/appuser`, satisfying Hugging Face's container sandbox constraints.

---

## 3. Deployment Steps

### Method A: Direct GitHub Repository Sync (Recommended)

1. **Log in to Hugging Face**:
   - Go to [huggingface.co](https://huggingface.co) and sign in.
2. **Create New Space**:
   - Click your profile icon $\rightarrow$ **New Space**.
   - **Space Name**: `failuredna`
   - **License**: `mit`
   - **Space SDK**: Select **Docker** $\rightarrow$ **Blank**.
   - **Visibility**: **Public**.
   - Click **Create Space**.
3. **Link GitHub Repository or Push Code**:
   - In your local terminal or GitHub Actions:
   ```bash
   # Add Hugging Face Space as a remote
   git remote add hf https://huggingface.co/spaces/<YOUR_HF_USERNAME>/failuredna

   # Push main branch to Hugging Face
   git push hf main --force
   ```
   *(When prompted for a password, enter your Hugging Face Access Token with write permissions from `https://huggingface.co/settings/tokens`)*.

---

### Method B: Automated CI/CD Sync via GitHub Actions

The repository includes a GitHub Action workflow ([.github/workflows/deploy-hf.yml](file:///c:/Users/Harish/Downloads/FailureDNA/.github/workflows/deploy-hf.yml)).

To activate automated synchronization on every commit:
1. In your GitHub repository (`Harish-Raja-R/failureDNA`):
   - Go to **Settings** $\rightarrow$ **Secrets and variables** $\rightarrow$ **Actions**.
2. Add two repository secrets:
   - `HF_TOKEN`: Your Hugging Face User Access Token (with Write role).
   - `HF_SPACE`: Your Space slug, e.g. `<username>/failuredna`.
3. Every future push to `main` will automatically build and update your Hugging Face Space.

---

## 4. Verification on Hugging Face Spaces

Once the build finishes in your Space (approximately 3–5 minutes):
1. **Application UI**: The Space renders the full editorial FailureDNA web platform.
2. **Interactive Inference**: The **Engine Analysis** page runs predictions inside the Hugging Face Docker container.
3. **Direct API Access**: The Space exposes public API endpoints accessible at `https://<username>-failuredna.hf.space/health` and `https://<username>-failuredna.hf.space/docs`.
