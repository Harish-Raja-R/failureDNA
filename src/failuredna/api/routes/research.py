"""Research Results API Routes."""

import logging
from fastapi import APIRouter, HTTPException, status

from failuredna.api.schemas import BenchmarkResponse, SignaturesResponse
from failuredna.inference.service import get_inference_service

logger = logging.getLogger("failuredna.api.research")
router = APIRouter(prefix="/api/v1/research", tags=["Research Results"])


@router.get("/benchmark", response_model=BenchmarkResponse, summary="Official Benchmark Comparison & Statistical Audit")
def get_benchmark() -> BenchmarkResponse:
    """Serve the verified Phase 5 / 5.1 / 5.2 official benchmark metrics on NASA C-MAPSS FD004.

    Includes all 11 model configurations, bootstrap 95% confidence intervals, and multiple-comparisons
    Holm adjustments. Accurately reports that XGBoost achieves a statistically significant MAE advantage,
    while RMSE and NASA score differences are not statistically significant.
    """
    service = get_inference_service()
    try:
        data = service.get_benchmark_results()
        return BenchmarkResponse(**data)
    except FileNotFoundError as e:
        logger.error(f"Benchmark results missing: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load benchmark results: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error retrieving benchmark: {e}")


@router.get("/signatures", response_model=SignaturesResponse, summary="Discovered Failure Signatures & Transition Dynamics")
def get_signatures() -> SignaturesResponse:
    """Serve discovered degradation signatures, fleet occupancy, and Markov state transition dynamics on FD004.

    Includes empirical dwell times, absorbing transition matrices, hazard rate audit, and signature-regime NMI.
    """
    service = get_inference_service()
    try:
        data = service.get_signatures_summary()
        return SignaturesResponse(**data)
    except FileNotFoundError as e:
        logger.error(f"Signature summary missing: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load signatures summary: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error retrieving signatures: {e}")
