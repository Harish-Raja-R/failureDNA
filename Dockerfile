# ==============================================================================
# FailureDNA Production Multi-Stage Dockerfile
# Stage 1: Build React 18 + TypeScript + Tailwind CSS Frontend
# Stage 2: Production Python 3.11-slim FastAPI + PyTorch Backend & Serving
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Frontend Build
# ------------------------------------------------------------------------------
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend

# Install dependencies first for Docker layer caching
COPY frontend/package*.json ./
RUN npm install

# Copy frontend source and build static distribution
COPY frontend/ ./
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 2: Production Python Serving Environment
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and frozen models
COPY src/ /app/src/
COPY models/ /app/models/
COPY reports/ /app/reports/
COPY configs/ /app/configs/

# Copy built frontend from Stage 1 into frontend/dist for FastAPI static mounting
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Create a non-root application user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Healthcheck targeting FastAPI /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

EXPOSE 8000 7860

# Start FastAPI production server via Uvicorn with dynamic cloud PORT support
CMD ["sh", "-c", "python -m uvicorn failuredna.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
