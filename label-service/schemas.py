"""Pydantic schemas for request/response validation."""
from datetime import datetime
from typing import Optional, List
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ValidationInfo


class LabelBase(BaseModel):
    """Base label schema."""
    name: str = Field(..., min_length=1, max_length=255, description="Label name")
    level: int = Field(..., ge=1, le=3, description="Label level (1, 2, or 3)")
    parent_id: Optional[int] = Field(None, ge=1, description="Parent label ID")
    description: Optional[str] = Field(None, description="Label description")
    
    @field_validator('level')
    @classmethod
    def validate_level(cls, v: int) -> int:
        """Validate level is 1, 2, or 3."""
        if v not in [1, 2, 3]:
            raise ValueError("Level must be 1, 2, or 3")
        return v
    
    @field_validator('parent_id')
    @classmethod
    def validate_parent_for_level(cls, v: Optional[int], info: ValidationInfo) -> Optional[int]:
        """Validate parent_id based on level."""
        level = info.data.get('level') if info.data else None
        if level == 1 and v is not None:
            raise ValueError("Level 1 labels cannot have a parent")
        if level in [2, 3] and v is None:
            raise ValueError(f"Level {level} labels must have a parent")
        return v


class LabelCreate(LabelBase):
    """Schema for creating a label."""
    id: Optional[int] = Field(
        default=None,
        ge=1,
        description="Label ID (optional - if omitted, will be assigned automatically)",
    )


class LabelUpdate(BaseModel):
    """Schema for updating a label."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None)
    
    # Note: We don't allow updating level or parent_id to maintain hierarchy integrity


class LabelResponse(LabelBase):
    """Schema for label response."""
    id: int
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
    parent_id: Optional[int] = None


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
    is_model_confirmed: Optional[bool] = Field(
        default=None,
        description="Đánh dấu feedback đã được người dùng xác nhận mô hình đúng",
    )
    level1_id: Optional[int] = Field(
        default=None,
        description="Selected level 1 label ID (or null to clear intent)",
    )
    level2_id: Optional[int] = Field(
        default=None,
        description="Selected level 2 label ID (must be child of level1)",
    )
    level3_id: Optional[int] = Field(
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
    level1_id: Optional[int] = None
    level2_id: Optional[int] = None
    level3_id: Optional[int] = None
    is_model_confirmed: bool = Field(
        default=False,
        description="Feedback đã được người dùng xác nhận mô hình đúng hay chưa",
    )
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



# --- Label Sync Schemas ---

class LabelSyncItem(LabelBase):
    """Payload of a label record to sync from external service."""
    id: int = Field(..., ge=1, description="Unique label ID from source system")
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp của bản ghi nguồn (nếu có) để phục vụ logging"
    )


class LabelSyncRequest(BaseModel):
    """Request schema cho API đồng bộ label."""
    labels: List[LabelSyncItem] = Field(
        ...,
        min_items=1,
        description="Danh sách label cần đồng bộ",
    )


class LabelSyncResultStatus(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


class FeedbackReprocessStatus(str, Enum):
    UPDATED = "updated"
    SKIPPED = "skipped"
    FAILED = "failed"


class FeedbackReprocessResult(BaseModel):
    """Kết quả tái dự đoán cho từng feedback bị ảnh hưởng."""
    feedback_id: UUID
    status: FeedbackReprocessStatus
    message: Optional[str] = None


class LabelSyncResult(BaseModel):
    """Kết quả cho từng label sau khi đồng bộ."""
    id: int
    status: LabelSyncResultStatus
    changes: Optional[List[str]] = Field(
        default=None,
        description="Danh sách field đã thay đổi (nếu có)"
    )
    message: Optional[str] = None


class LabelSyncResponse(BaseModel):
    """Phản hồi API đồng bộ label."""
    created: int
    updated: int
    unchanged: int
    results: List[LabelSyncResult]
    total: int = Field(..., description="Tổng số bản ghi đã xử lý")
    changed_label_ids: List[int] = Field(
        default_factory=list,
        description="Danh sách label đã được cập nhật (ảnh hưởng tới feedback)",
    )
    impacted_feedbacks: int = Field(
        default=0,
        description="Số feedback bị reset lại do thay đổi label",
    )
    reprocessed_feedbacks: List[FeedbackReprocessResult] = Field(
        default_factory=list,
        description="Chi tiết kết quả tái dự đoán cho feedback bị ảnh hưởng",
    )


