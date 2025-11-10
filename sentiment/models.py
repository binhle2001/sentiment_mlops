from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

# --- Enums ---

class SentimentLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    EXTREMELY_NEGATIVE = "EXTREMELY_NEGATIVE"

# --- Request Models ---

class ClassifyRequest(BaseModel):
    """Request model for classifying a single text."""
    text: str = Field(..., description="The text to be classified.", min_length=1)

class ClassifyBatchRequest(BaseModel):
    """Request model for classifying a batch of texts."""
    texts: List[str] = Field(..., description="A list of texts to be classified.", min_items=1)

# --- Response Models ---

class SentimentResult(BaseModel):
    """Result for a single sentiment classification."""
    label: SentimentLabel = Field(..., description="The predicted sentiment label.")
    score: float = Field(..., description="The confidence score of the prediction.")

class ClassifyResponse(SentimentResult):
    """Response for a single text classification."""
    status: str = Field("success", description="Status of the operation.")
    processing_time: float = Field(..., description="Time taken for the operation in seconds.")

class ClassifyBatchResponse(BaseModel):
    """Response for a batch classification."""
    results: List[SentimentResult] = Field(..., description="A list of sentiment results for each text.")
    status: str = Field("success", description="Status of the operation.")
    processing_time: float = Field(..., description="Total time taken for the batch operation in seconds.")

# --- System & Error Models ---

class ErrorResponse(BaseModel):
    """Standardized error response model."""
    error: str = Field(..., description="Error type or code.")
    message: str = Field(..., description="A human-readable error message.")
    detail: Optional[str] = Field(None, description="Additional error details.")
    status: str = Field("error", description="Indicates an error response.")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status: 'healthy' or 'unhealthy'.")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Information about the loaded model.")
    reason: Optional[str] = Field(None, description="Reason for an 'unhealthy' status.")
    timestamp: str = Field(..., description="Timestamp of the health check.")
