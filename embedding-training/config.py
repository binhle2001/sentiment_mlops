from typing import Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Embedding Training Service"
    app_version: str = "0.1.0"

    server_host: str = "0.0.0.0"
    server_port: int = 8001

    postgres_user: str = Field(..., alias="POSTGRES_USER")
    postgres_password: str = Field(..., alias="POSTGRES_PASSWORD")
    postgres_host: str = Field(..., alias="POSTGRES_HOST")
    postgres_port: int = Field(..., alias="POSTGRES_PORT")
    postgres_db: str = Field(..., alias="POSTGRES_DB")

    model_name: str = "BAAI/bge-m3"
    output_dir: str = "/app/models/bge-m3-finetuned-transformer"
    use_multiple_neg: bool = True
    max_length: int = 128
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    epochs: int = 3
    lr: float = 2e-5
    fp16: bool = True
    logging_steps: int = 50
    save_steps: int = 500
    scale: float = 20.0
    margin: float = 0.2
    seed: int = 42

    @property
    def database_url(self) -> str:
        """Construct PostgreSQL database URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    class Config:
        case_sensitive = False


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings