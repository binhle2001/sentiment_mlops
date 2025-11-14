import logging

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse

from exceptions import (
    EmbeddingServiceError,
    get_http_status_code,
)
from models import (
    EmbeddingRequest,
    BatchEmbeddingRequest,
    EmbeddingResponse,
    BatchEmbeddingResponse,
    ErrorResponse,
    DetailedErrorResponse,
    HealthResponse,
)
from engine import get_embedding_service, EmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/encode",
    response_model=EmbeddingResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": DetailedErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    },
    summary="Encode single text to embedding",
    description="Encode a single text string into a 1024-dimensional embedding vector using BGE-M3 model."
)
async def encode_text(
    request: EmbeddingRequest,
    service: EmbeddingService = Depends(get_embedding_service)
) -> EmbeddingResponse:
    """Encode a single text to embedding."""
    try:
        logger.info(f"Received encoding request for text length: {len(request.text)}")
        
        # Hardcode pooling_strategy to 'mean'
        result = await service.encode_single(request.text, "mean")
        
        logger.info(f"Successfully encoded text, dimension: {result['dimension']}")
        return EmbeddingResponse(**result)
    
    except EmbeddingServiceError as e:
        logger.error(f"Embedding service error: {e.message}")
        raise HTTPException(status_code=get_http_status_code(e), detail=e.message)
    
    except Exception as e:
        logger.error(f"Unexpected error in encode_text: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/encode_batch",
    response_model=BatchEmbeddingResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": DetailedErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    },
    summary="Encode multiple texts to embeddings",
    description="Encode multiple text strings into 1024-dimensional embedding vectors using BGE-M3 model."
)
async def encode_batch(
    request: BatchEmbeddingRequest,
    service: EmbeddingService = Depends(get_embedding_service)
) -> BatchEmbeddingResponse:
    """Encode multiple texts to embeddings."""
    try:
        logger.info(f"Received batch encoding request for {len(request.texts)} texts")
        
        # Hardcode pooling_strategy to 'mean'
        result = await service.encode_batch(request.texts, "mean")
        
        logger.info(f"Successfully encoded {result['valid_texts']}/{result['total_texts']} texts")
        return BatchEmbeddingResponse(**result)
        
    except EmbeddingServiceError as e:
        logger.error(f"Embedding service error: {e.message}")
        raise HTTPException(status_code=get_http_status_code(e), detail=e.message)
    
    except Exception as e:
        logger.error(f"Unexpected error in encode_batch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
)
async def health_check(service: EmbeddingService = Depends(get_embedding_service)):
    """Endpoint for health checks."""
    health_data = await service.health_check()
    status_code = 200 if health_data["status"] == "healthy" else 503
    return JSONResponse(content=health_data, status_code=status_code)
