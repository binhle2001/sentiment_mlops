from typing import Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    
    # Application settings
    app_name: str = Field(default="Embedding Service", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    log_level: str = Field(default="info", description="Logging level")
    
    # Model configuration
    model_path: str = Field(
        default="./bge-m3-finetuned-transformer/vn_embedding_bgem3", 
        description="Path to the embedding model"
    )
    device: str = Field(default="cpu", description="Device to run model on")
    max_length: int = Field(default=128, description="Maximum sequence length")
    embedding_dimension: int = Field(default=1024, description="Embedding dimension")
    
    # Performance configuration
    batch_size: int = Field(default=16, description="Default batch size")
    max_batch_size: int = Field(default=32, description="Maximum batch size")
    
    # ONNX Runtime configuration
    onnx_providers: list[str] = Field(default=["CPUExecutionProvider"], description="ONNX providers")
    inter_op_threads: int = Field(default=2, description="ONNX inter-op threads")
    intra_op_threads: int = Field(default=4, description="ONNX intra-op threads")
    
    # CORS settings
    cors_origins: list[str] = Field(default=["*"], description="CORS allowed origins")
    cors_allow_credentials: bool = Field(default=True, description="CORS allow credentials")
    cors_allow_methods: list[str] = Field(default=["*"], description="CORS allowed methods")
    cors_allow_headers: list[str] = Field(default=["*"], description="CORS allowed headers")
    
    # API settings
    api_prefix: str = Field(default="/api/v1", description="API prefix")
    docs_url: str = Field(default="/docs", description="API documentation URL")
    redoc_url: str = Field(default="/redoc", description="ReDoc documentation URL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_prefix = "EMBEDDING_"
        
        # Field aliases for environment variables
        fields = {
            "app_name": {"env": "APP_NAME"},
            "app_version": {"env": "APP_VERSION"},
            "debug": {"env": "DEBUG"},
            "host": {"env": "HOST"},
            "port": {"env": "PORT"},
            "workers": {"env": "WORKERS"},
            "log_level": {"env": "LOG_LEVEL"},
            "model_path": {"env": "MODEL_PATH"},
            "device": {"env": "DEVICE"},
            "max_length": {"env": "MAX_LENGTH"},
            "embedding_dimension": {"env": "DIMENSION"},
            "batch_size": {"env": "BATCH_SIZE"},
            "max_batch_size": {"env": "MAX_BATCH_SIZE"},
            "onnx_providers": {"env": "ONNX_PROVIDERS"},
            "inter_op_threads": {"env": "ONNX_INTER_OP_THREADS"},
            "intra_op_threads": {"env": "ONNX_INTRA_OP_THREADS"},
        }
    
    @validator('device')
    def validate_device(cls, v):
        if v not in ["cpu", "cuda"]:
            raise ValueError("device must be 'cpu' or 'cuda'")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ["debug", "info", "warning", "error", "critical"]
        if v.lower() not in valid_levels:
            raise ValueError(f"log_level must be one of: {valid_levels}")
        return v.lower()
    
    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("port must be between 1 and 65535")
        return v
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError("batch_size must be positive")
        return v
    
    @validator('max_batch_size')
    def validate_max_batch_size(cls, v, values):
        if 'batch_size' in values and v < values['batch_size']:
            raise ValueError("max_batch_size must be >= batch_size")
        if v <= 0:
            raise ValueError("max_batch_size must be positive")
        return v
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.debug


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
