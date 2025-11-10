import logging
import time
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse

from engine import get_sentiment_service, SentimentService
from models import (
    ClassifyRequest,
    ClassifyBatchRequest,
    ClassifyResponse,
    ClassifyBatchResponse,
    HealthResponse,
)
from exceptions import SentimentServiceError, get_http_status_code

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify sentiment of a single text",
)
async def classify_single_text(
    request: ClassifyRequest,
    service: SentimentService = Depends(get_sentiment_service)
):
    """Endpoint to classify the sentiment of a single piece of text."""
    try:
        start_time = time.time()
        result = await service.classify(request.text)
        processing_time = time.time() - start_time
        
        return ClassifyResponse(
            label=result.label,
            score=result.score,
            processing_time=processing_time
        )

    except SentimentServiceError as e:
        logger.error(f"Sentiment service error: {e.message}")
        raise HTTPException(status_code=get_http_status_code(e), detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error during classification: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.post(
    "/classify_batch",
    response_model=ClassifyBatchResponse,
    summary="Classify sentiment of a batch of texts",
)
async def classify_batch_text(
    request: ClassifyBatchRequest,
    service: SentimentService = Depends(get_sentiment_service)
):
    """Endpoint to classify the sentiment of a batch of texts."""
    try:
        start_time = time.time()
        results = await service.classify_batch(request.texts)
        processing_time = time.time() - start_time
        
        return ClassifyBatchResponse(results=results, processing_time=processing_time)

    except SentimentServiceError as e:
        logger.error(f"Sentiment service error: {e.message}")
        raise HTTPException(status_code=get_http_status_code(e), detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error during batch classification: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
)
async def health_check(service: SentimentService = Depends(get_sentiment_service)):
    """Endpoint for health checks."""
    health_data = await service.health_check()
    status_code = 200 if health_data["status"] == "healthy" else 503
    return JSONResponse(content=health_data, status_code=status_code)
