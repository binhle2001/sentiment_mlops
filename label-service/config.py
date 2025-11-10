from typing import Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application settings
    app_name: str = Field(default="Label Management Service", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8001, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    log_level: str = Field(default="info", description="Logging level")
    
    # Database configuration
    postgres_user: str = Field(default="labeluser", description="PostgreSQL user")
    postgres_password: str = Field(default="labelpass123", description="PostgreSQL password")
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_db: str = Field(default="label_db", description="PostgreSQL database name")
    
    # CORS settings
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8001", "http://103.167.84.162:2345", "http://frontend:80"],
        description="CORS allowed origins"
    )
    cors_allow_credentials: bool = Field(default=True, description="CORS allow credentials")
    cors_allow_methods: list[str] = Field(default=["*"], description="CORS allowed methods")
    cors_allow_headers: list[str] = Field(default=["*"], description="CORS allowed headers")
    
    # API settings
    api_prefix: str = Field(default="/api/v1", description="API prefix")
    docs_url: str = Field(default="/docs", description="API documentation URL")
    redoc_url: str = Field(default="/redoc", description="ReDoc documentation URL")
    
    @property
    def database_url(self) -> str:
        """Construct PostgreSQL async database URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def sync_database_url(self) -> str:
        """Construct PostgreSQL sync database URL for Alembic."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["debug", "info", "warning", "error", "critical"]
        if v.lower() not in valid_levels:
            raise ValueError(f"log_level must be one of: {valid_levels}")
        return v.lower()
    
    @validator('port')
    def validate_port(cls, v):
        """Validate port number."""
        if not 1 <= v <= 65535:
            raise ValueError("port must be between 1 and 65535")
        return v
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment variables."""
    global _settings
    _settings = Settings()
    return _settings



