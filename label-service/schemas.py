"""Pydantic schemas for request/response validation."""
from datetime import datetime
from typing import Optional, List
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field, validator


class LabelBase(BaseModel):
    """Base label schema."""
    name: str = Field(..., min_length=1, max_length=255, description="Label name")
    level: int = Field(..., ge=1, le=3, description="Label level (1, 2, or 3)")
    parent_id: Optional[UUID] = Field(None, description="Parent label ID")
    description: Optional[str] = Field(None, description="Label description")
    
    @validator('level')
    def validate_level(cls, v):
        """Validate level is 1, 2, or 3."""
        if v not in [1, 2, 3]:
            raise ValueError("Level must be 1, 2, or 3")
        return v
    
    @validator('parent_id', always=True)
    def validate_parent_for_level(cls, v, values):
        """Validate parent_id based on level."""
        level = values.get('level')
        if level == 1 and v is not None:
            raise ValueError("Level 1 labels cannot have a parent")
        if level in [2, 3] and v is None:
            raise ValueError(f"Level {level} labels must have a parent")
        return v


class LabelCreate(LabelBase):
    """Schema for creating a label."""
    pass


class LabelUpdate(BaseModel):
    """Schema for updating a label."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None)
    
    # Note: We don't allow updating level or parent_id to maintain hierarchy integrity


class LabelResponse(LabelBase):
    """Schema for label response."""
    id: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class LabelTreeResponse(LabelResponse):
    """Schema for label tree response with children."""
    children: List["LabelTreeResponse"] = Field(default_factory=list)
    
    class Config:
        from_attributes = True


class LabelListResponse(BaseModel):
    """Schema for list of labels response."""
    labels: List[LabelResponse]
    total: int
    level: Optional[int] = None
    parent_id: Optional[UUID] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    database: str
    version: str


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    message: str
    detail: Optional[str] = None


class BulkLabelCreate(BaseModel):
    """Schema for creating multiple labels at once."""
    labels: List[LabelCreate] = Field(..., min_items=1, max_items=100, description="List of labels to create")


class BulkLabelResult(BaseModel):
    """Result for a single label in bulk operation."""
    success: bool
    label: Optional[LabelResponse] = None
    error: Optional[str] = None
    index: int = Field(..., description="Index in the input array")


class BulkLabelResponse(BaseModel):
    """Response for bulk label creation."""
    total: int = Field(..., description="Total labels submitted")
    successful: int = Field(..., description="Number of successfully created labels")
    failed: int = Field(..., description="Number of failed labels")
    results: List[BulkLabelResult] = Field(..., description="Detailed results for each label")


# Update forward references for recursive model
LabelTreeResponse.model_rebuild()


# --- Feedback Sentiment Schemas ---

class FeedbackSource(str, Enum):
    """Enum for feedback sources."""
    WEB = "web"
    APP = "app"
    MAP = "map"
    SURVEY_FORM = "form khảo sát"
    CALL_CENTER = "tổng đài"


class SentimentLabel(str, Enum):
    """Enum for sentiment labels."""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    EXTREMELY_NEGATIVE = "EXTREMELY_NEGATIVE"


class FeedbackSentimentCreate(BaseModel):
    """Schema for creating a feedback sentiment."""
    feedback_text: str = Field(..., min_length=1, description="Feedback text from customer")
    feedback_source: FeedbackSource = Field(..., description="Source of the feedback")


class FeedbackSentimentUpdate(BaseModel):
    """Schema for updating sentiment label and/or intent for a feedback."""
    sentiment_label: Optional[SentimentLabel] = Field(
        default=None,
        description="Updated sentiment label",
    )
    level1_id: Optional[UUID] = Field(
        default=None,
        description="Selected level 1 label ID (or null to clear intent)",
    )
    level2_id: Optional[UUID] = Field(
        default=None,
        description="Selected level 2 label ID (must be child of level1)",
    )
    level3_id: Optional[UUID] = Field(
        default=None,
        description="Selected level 3 label ID (must be child of level2)",
    )


class FeedbackSentimentResponse(BaseModel):
    """Schema for feedback sentiment response."""
    id: UUID
    feedback_text: str
    sentiment_label: SentimentLabel
    confidence_score: float
    feedback_source: FeedbackSource
    created_at: datetime
    level1_id: Optional[UUID] = None
    level2_id: Optional[UUID] = None
    level3_id: Optional[UUID] = None
    level1_name: Optional[str] = None
    level2_name: Optional[str] = None
    level3_name: Optional[str] = None
    
    class Config:
        from_attributes = True


class FeedbackSentimentListResponse(BaseModel):
    """Schema for list of feedback sentiments response."""
    feedbacks: List[FeedbackSentimentResponse]
    total: int
    sentiment_label: Optional[SentimentLabel] = None
    feedback_source: Optional[FeedbackSource] = None


# --- Intent Analysis Schemas ---

class IntentTriplet(BaseModel):
    """Schema for an intent triplet (level 1, 2, 3 labels)."""
    level1: LabelResponse
    level2: LabelResponse
    level3: LabelResponse
    avg_cosine_similarity: float = Field(..., description="Average cosine similarity score")
    
    class Config:
        from_attributes = True


class FeedbackIntentResponse(BaseModel):
    """Schema for feedback intent analysis response."""
    feedback_id: UUID
    intents: List[IntentTriplet] = Field(..., description="Top 10 intent triplets")
    total_intents: int = Field(..., description="Total number of intents analyzed")
    
    class Config:
        from_attributes = True



