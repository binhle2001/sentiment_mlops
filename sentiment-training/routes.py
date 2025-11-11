import logging
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from schemas import HealthResponse, TrainingTriggerRequest, TrainingTriggerResponse

# Bảo đảm có thể import training_sentiment từ thư mục gốc dự án
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from training_sentiment import TrainingConfig, train_sentiment_pipeline  # noqa: E402

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check cho Sentiment Training Service",
)
async def health_check() -> HealthResponse:
    """Xác nhận service hoạt động bình thường."""
    return HealthResponse(status="healthy", message="Service is ready.")


@router.post(
    "/train",
    response_model=TrainingTriggerResponse,
    summary="Kích hoạt pipeline huấn luyện sentiment",
)
async def trigger_training(body: TrainingTriggerRequest) -> TrainingTriggerResponse:
    """Chạy pipeline huấn luyện sentiment trong threadpool."""
    try:
        config = TrainingConfig.from_env()
        result: Dict[str, Any] = await run_in_threadpool(
            train_sentiment_pipeline,
            config,
            body.force,
        )
        return TrainingTriggerResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to trigger sentiment training: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to trigger sentiment training",
        ) from exc


