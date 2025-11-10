"""Pydantic schemas for request/response validation."""
from datetime import datetime
from typing import Optional, List
from uuid import UUID

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


# Update forward references for recursive model
LabelTreeResponse.model_rebuild()


