from fastapi import FastAPI
from config import get_settings
from database import init_pool, close_pool
from routes import router as api_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
)

@app.on_event("startup")
async def startup_event():
    """Initialize database connection pool on startup."""
    init_pool()

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection pool on shutdown."""
    close_pool()

app.include_router(api_router, prefix="/api")