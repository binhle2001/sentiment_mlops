import logging
from fastapi import APIRouter, HTTPException, status
from trainer import run_training
from schemas import TrainingResponse
from config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.post("/train", response_model=TrainingResponse, status_code=200)
def train_model() -> TrainingResponse:
    """
    Endpoint để kích hoạt quá trình huấn luyện mô hình embedding.
    Chạy đồng bộ và trả về kết quả sau khi hoàn thành.
    """
    try:
        logger.info("Starting embedding model training...")
        run_training()
        
        logger.info("Training completed successfully")
        return TrainingResponse(
            success=True,
            message="Training completed successfully",
            model_path=settings.output_dir
        )
    except Exception as e:
        logger.exception("Training failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "embedding-training"}