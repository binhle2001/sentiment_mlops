import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from routes import router
from schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Log lifecycle events (có thể mở rộng để pre-load tài nguyên nếu cần)."""
    logger.info("%s v%s starting...", settings.app_name, settings.app_version)
    try:
        yield
    finally:
        logger.info("%s shutting down...", settings.app_name)


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



