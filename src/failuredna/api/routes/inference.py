"""Telemetry Analysis and Metadata API Routes."""

import logging
from typing import Any, Dict
from fastapi import APIRouter, File, HTTPException, UploadFile, status
import pandas as pd

from failuredna.api.schemas import (
    AnalysisResponse,
    MetadataResponse,
    TelemetrySequenceRequest,
)
from failuredna.inference.service import get_inference_service

logger = logging.getLogger("failuredna.api.inference")
router = APIRouter(prefix="/api/v1", tags=["Inference & Analysis"])


@router.get("/metadata", response_model=MetadataResponse, summary="Project Metadata & Specifications")
def get_metadata() -> MetadataResponse:
    """Return model specifications, dataset provenance, required input channels, and research disclaimers."""
    service = get_inference_service()
    meta = service.get_metadata()
    return MetadataResponse(**meta)


@router.post("/analyze", response_model=AnalysisResponse, summary="Analyze Telemetry Sequence")
def analyze_sequence(payload: TelemetrySequenceRequest) -> AnalysisResponse:
    """Analyze a sequence of multivariate turbofan telemetry observations.

    Validates sequence length, detects operating flight conditions, extracts FailureDNA
    latent representations, predicts remaining useful life (RUL), and assigns the discovered
    degradation signature cluster.
    """
    service = get_inference_service()
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized or model artifacts are unavailable.",
        )

    try:
        # Convert Pydantic sequence to pandas DataFrame
        records = [c.model_dump() for c in payload.sequence]
        df = pd.DataFrame.from_records(records)
        result = service.analyze_telemetry_dataframe(df, unit_id=payload.unit_id)
        return AnalysisResponse(**result)
    except ValueError as e:
        logger.warning(f"Telemetry validation error: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {e}")


@router.post("/analyze/csv", response_model=AnalysisResponse, summary="Upload & Analyze Engine Telemetry CSV")
async def analyze_csv(file: UploadFile = File(..., description="Chronological engine telemetry CSV or text file")) -> AnalysisResponse:
    """Upload a CSV file containing an engine's chronological telemetry.

    Validates format, column names, chronological order, non-finite values, and file size (<=10 MB).
    Applies the frozen preprocessing pipeline and returns RUL estimation, signature clustering, and sensor trend summaries.
    """
    service = get_inference_service()
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized or model artifacts are unavailable.",
        )

    # Validate file extension
    filename = file.filename or "uploaded_telemetry.csv"
    if not (filename.endswith(".csv") or filename.endswith(".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type for '{filename}'. Only .csv or .txt telemetry files are accepted.",
        )

    try:
        content = await file.read()
        result = service.analyze_csv_bytes(content, filename=filename)
        return AnalysisResponse(**result)
    except ValueError as e:
        logger.warning(f"CSV validation failed for '{filename}': {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Failed processing CSV '{filename}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Processing error: {e}")
