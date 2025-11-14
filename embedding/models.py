from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class EmbeddingRequest(BaseModel):
    """Request model for single text embedding."""
    text: str = Field(..., description="Text to encode into an embedding.", min_length=1)

    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or contain only whitespace.")
        return v


class BatchEmbeddingRequest(BaseModel):
    """Request model for batch text embedding."""
    texts: List[str] = Field(..., description="A list of texts to encode.", min_items=1)

    @validator('texts')
    def texts_must_not_be_empty(cls, v):
        if not all(isinstance(t, str) and t.strip() for t in v):
            raise ValueError("All texts in the list must be non-empty strings.")
        return v


# --- Response Models ---

class EmbeddingResponse(BaseModel):
    """Response model for a single text embedding."""
    embedding: List[float] = Field(..., description="The embedding vector.")
    dimension: int = Field(..., description="Dimension of the embedding vector.")
    text_length: int = Field(..., description="Length of the original text.")
    pooling_strategy: str = Field(..., description="Pooling strategy used.")
    status: str = Field("success", description="Status of the embedding operation.")
    encoding_time: float = Field(..., description="Time taken for encoding in seconds.")


class BatchEmbeddingResponse(BaseModel):
    """Response model for a batch text embedding."""
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors.")
    dimension: int = Field(..., description="Dimension of each embedding vector.")
    total_texts: int = Field(..., description="Total number of texts in the request.")
    valid_texts: int = Field(..., description="Number of valid texts that were processed.")
    pooling_strategy: str = Field(..., description="Pooling strategy used.")
    status: str = Field("success", description="Status of the batch embedding operation.")
    encoding_time: float = Field(..., description="Time taken for encoding in seconds.")


class ErrorResponse(BaseModel):
    """Standardized error response model."""
    error: str = Field(..., description="Error type or code (e.g., 'INVALID_INPUT').")
    message: str = Field(..., description="A human-readable error message.")
    detail: Optional[str] = Field(None, description="Additional error details, visible in debug mode.")
    status: str = Field("error", description="Indicates an error response.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the error.")


class DetailedErrorResponse(ErrorResponse):
    """Error response that includes validation details."""
    validation_errors: Optional[List[Dict[str, Any]]] = Field(None, description="List of validation errors.")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status: 'healthy' or 'unhealthy'.")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Information about the loaded model.")
    reason: Optional[str] = Field(None, description="Reason for an 'unhealthy' status.")
    timestamp: str = Field(..., description="Timestamp of the health check.")
