"""Pydantic schemas for embedding training API."""
from typing import Optional
from pydantic import BaseModel, Field


class TrainingResponse(BaseModel):
    """Response schema for training endpoint."""
    success: bool = Field(..., description="Whether training completed successfully")
    message: str = Field(..., description="Status message")
    model_path: Optional[str] = Field(None, description="Path to saved model")
    error: Optional[str] = Field(None, description="Error message if training failed")

