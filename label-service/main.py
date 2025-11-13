"""Main application entry point for Label Management Service."""
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from database import init_db, init_pool, close_pool
from routes import router
from training_manager import init_training_manager, shutdown_training_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}...")
    
    try:
        # Initialize database connection pool
        init_pool()
        # Initialize database tables
        init_db()
        await init_training_manager(settings)
        logger.info(f"{settings.app_name} started successfully")
        logger.info(f"API documentation available at {settings.docs_url}")
        
    except Exception as e:
        logger.error(f"Failed to start {settings.app_name}: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {settings.app_name}...")
    try:
        await shutdown_training_manager()
        close_pool()
        logger.info(f"{settings.app_name} shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


def create_application() -> FastAPI:
    """Create and configure FastAPI application."""
    
    # Create FastAPI application
    app = FastAPI(
        title=settings.app_name,
        description="Service for managing hierarchical labels for customer feedback classification",
        version=settings.app_version,
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # Include API routes
    app.include_router(
        router,
        prefix=settings.api_prefix,
        tags=["labels"]
    )
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    return app


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup global exception handlers for the application."""
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "detail": str(exc) if settings.debug else None
            }
        )


def main() -> None:
    """Main entry point for the application."""
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, settings.log_level.upper()))
    
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Host: {settings.host}, Port: {settings.port}")
    logger.info(f"Workers: {settings.workers}, Log Level: {settings.log_level}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Database: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")
    
    # Create application
    app = create_application()
    
    # Run the application
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level,
        reload=settings.debug,
        access_log=settings.debug
    )


if __name__ == "__main__":
    main()



