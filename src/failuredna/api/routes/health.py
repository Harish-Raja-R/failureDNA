"""Health and Readiness Endpoint."""

from fastapi import APIRouter, Response, status

from failuredna.api.schemas import HealthResponse
from failuredna.inference.service import get_inference_service

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse, summary="Application and Artifact Health")
def get_health(response: Response) -> HealthResponse:
    """Return application availability and artifact initialization status.

    Returns HTTP 200 if all frozen models and inference bundles are initialized.
    Returns HTTP 503 if any required artifact is missing or degraded.
    """
    service = get_inference_service()
    status_dict = service.get_health_status()
    version = service.get_metadata().get("version", "1.0.0")

    if not status_dict["is_ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status=status_dict["status"],
        version=version,
        is_ready=status_dict["is_ready"],
        initialized_at=status_dict["initialized_at"],
        error=status_dict["error"],
        artifacts=status_dict["artifacts"],
    )
