"""FastAPI Application Entrypoint for FailureDNA."""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from failuredna.api.config import settings
from failuredna.api.routes import health, inference, research
from failuredna.inference.service import get_inference_service

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("failuredna.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager: initialize model artifacts once at startup."""
    logger.info("Initializing FailureDNA Application Lifespan...")
    service = get_inference_service()
    success = service.initialize()
    if success:
        logger.info("FailureDNA inference pipeline ready for traffic.")
    else:
        logger.warning("FailureDNA service initialized in degraded state. Check artifact paths.")
    yield
    logger.info("Shutting down FailureDNA Application Lifespan...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "FailureDNA REST API: Self-Supervised Discovery of Failure Signatures from "
        "Multivariate Industrial Time-Series on NASA C-MAPSS FD004."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health.router)
app.include_router(inference.router)
app.include_router(research.router)


# Global Exception Handler to avoid exposing stack traces
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception during {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please consult server logs for details."},
    )


# Static frontend mounting (if frontend build exists)
frontend_dist = Path("frontend/dist")
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    logger.info(f"Mounting built static frontend from '{frontend_dist}'...")
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend(full_path: str):
        # Don't intercept API or docs routes
        if full_path.startswith("api") or full_path.startswith("health") or full_path in ("docs", "redoc", "openapi.json"):
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        file_path = frontend_dist / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("failuredna.api.main:app", host="0.0.0.0", port=8000, reload=False)
