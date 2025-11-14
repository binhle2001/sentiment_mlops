import logging
import os
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from routes import router
from schemas import HealthResponse
from training_sentiment import TrainingConfig, train_sentiment_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


def run_training():
    """Hàm chạy huấn luyện trong một luồng riêng biệt."""
    logger.info("Bắt đầu quá trình huấn luyện sentiment tự động...")
    try:
        config = TrainingConfig.from_env()
        train_sentiment_pipeline(config=config, force=True)
        logger.info("Hoàn tất quá trình huấn luyện sentiment tự động.")
    except Exception as e:
        logger.error(f"Lỗi trong quá trình huấn luyện sentiment tự động: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Xử lý logic khi khởi động và tắt ứng dụng."""
    logger.info("Khởi động Sentiment Training Service...")
    model_path = os.path.join("artifacts", "sentiment_mlp", "sentiment_mlp.joblib")
    if not os.path.exists(model_path):
        logger.info(f"Không tìm thấy mô hình tại {model_path}. Bắt đầu huấn luyện...")
        training_thread = threading.Thread(target=run_training)
        training_thread.start()
    else:
        logger.info(f"Đã tìm thấy mô hình tại {model_path}. Bỏ qua huấn luyện.")
    yield
    logger.info("Tắt Sentiment Training Service...")


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="Service chuyên trách huấn luyện mô hình sentiment",
        version=settings.app_version,
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    app.include_router(router, prefix=settings.api_prefix, tags=["sentiment-training"])

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content=HealthResponse(
                status="error",
                message=str(exc) if settings.debug else "Unexpected error",
            ).model_dump(),
        )

    return app


app = create_application()


def main() -> None:
    logging.getLogger().setLevel(getattr(logging, settings.log_level.upper()))
    logger.info(
        "Starting %s on %s:%s", settings.app_name, settings.host, settings.port
    )
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()



