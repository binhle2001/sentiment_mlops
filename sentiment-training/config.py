from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Cấu hình cho Sentiment Training Service."""

    # Thông tin ứng dụng
    app_name: str = Field(default="Sentiment Training Service", description="Tên service")
    app_version: str = Field(default="1.0.0", description="Phiên bản service")
    debug: bool = Field(default=False, description="Bật chế độ debug")

    # Thông tin server
    host: str = Field(default="0.0.0.0", description="Địa chỉ host")
    port: int = Field(default=8010, description="Cổng service")
    workers: int = Field(default=1, description="Số lượng worker uvicorn")
    log_level: str = Field(default="info", description="Mức độ log")

    # CORS
    cors_origins: List[str] = Field(default=["*"], description="Danh sách origin được phép")
    cors_allow_credentials: bool = Field(default=True, description="Có cho phép credentials hay không")
    cors_allow_methods: List[str] = Field(default=["*"], description="Danh sách phương thức HTTP cho phép")
    cors_allow_headers: List[str] = Field(default=["*"], description="Danh sách header cho phép")

    # API prefix
    api_prefix: str = Field(default="/api/v1", description="Prefix cho tất cả route")
    docs_url: str = Field(default="/docs", description="Đường dẫn tới tài liệu Swagger")
    redoc_url: str = Field(default="/redoc", description="Đường dẫn tới ReDoc")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_prefix = "SENTIMENT_TRAINING_"

    @validator("log_level")
    def validate_log_level(cls, value: str) -> str:
        valid_levels = {"debug", "info", "warning", "error", "critical"}
        lv = value.lower()
        if lv not in valid_levels:
            raise ValueError(f"log_level phải thuộc {valid_levels}")
        return lv

    @validator("port")
    def validate_port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("port phải nằm trong khoảng 1-65535")
        return value


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Trả về singleton Settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


