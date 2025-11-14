import logging
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from exceptions import SentimentServiceError, get_http_status_code
from models import ErrorResponse
from routes import router
from engine import get_sentiment_service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}...")
    try:
        await get_sentiment_service()
        logger.info(f"{settings.app_name} started successfully.")
    except Exception as e:
        logger.error(f"Failed to start {settings.app_name}: {e}", exc_info=True)
        raise
    
    yield
    
    logger.info(f"Shutting down {settings.app_name}...")
    try:
        service = await get_sentiment_service()
        await service.cleanup()
        logger.info(f"{settings.app_name} shutdown complete.")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)

def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        description="A high-performance service for sentiment classification.",
        version=settings.app_version,
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
        lifespan=lifespan
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    app.include_router(router, prefix=settings.api_prefix, tags=["sentiment"])
    
    # --- Exception Handlers ---
    @app.exception_handler(SentimentServiceError)
    async def sentiment_service_error_handler(request: Request, exc: SentimentServiceError):
        status_code = get_http_status_code(exc)
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(error=exc.error_code, message=exc.message).dict()
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="INTERNAL_SERVER_ERROR",
                message="An unexpected error occurred.",
                detail=str(exc) if settings.debug else None
            ).dict()
        )
        
    return app

app = create_application()

def main():
    """Main entry point for running the service."""
    logging.getLogger().setLevel(getattr(logging, settings.log_level.upper()))
    logger.info(f"Starting {settings.app_name} on {settings.host}:{settings.port}")
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
