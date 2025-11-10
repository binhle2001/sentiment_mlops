"""API routes for label management."""
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID
import httpx

from fastapi import APIRouter, Depends, HTTPException, Query, status

from database import get_db
from crud import LabelCRUD, FeedbackSentimentCRUD
from schemas import (
    LabelCreate,
    LabelUpdate,
    LabelResponse,
    LabelTreeResponse,
    LabelListResponse,
    HealthResponse,
    BulkLabelCreate,
    BulkLabelResponse,
    FeedbackSentimentCreate,
    FeedbackSentimentResponse,
    FeedbackSentimentListResponse,
    FeedbackSource,
    SentimentLabel,
)
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Health check endpoint")
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        database="ok",
        version=settings.app_version
    )


@router.post(
    "/labels",
    response_model=LabelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new label"
)
def create_label(label_data: LabelCreate):
    """Create a new label in the hierarchy."""
    try:
        with get_db() as conn:
            # Check if label with same name and parent already exists
            exists = LabelCRUD.exists_by_name_and_parent(
                conn, label_data.name, label_data.parent_id
            )
            if exists:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Label '{label_data.name}' already exists under this parent"
                )
            
            label = LabelCRUD.create(conn, label_data)
            return label
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating label: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create label"
        )


@router.post(
    "/labels/bulk",
    response_model=BulkLabelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create multiple labels at once"
)
def create_labels_bulk(bulk_data: BulkLabelCreate):
    """Create multiple labels in a single request.
    
    This endpoint allows you to create multiple labels at once.
    Each label is validated and created independently.
    If some labels fail, others will still be created successfully.
    """
    try:
        with get_db() as conn:
            results = LabelCRUD.bulk_create(conn, bulk_data.labels)
            
            # Count successes and failures
            successful = sum(1 for r in results if r['success'])
            failed = sum(1 for r in results if not r['success'])
            
            return BulkLabelResponse(
                total=len(bulk_data.labels),
                successful=successful,
                failed=failed,
                results=results
            )
    except Exception as e:
        logger.error(f"Error in bulk label creation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process bulk label creation"
        )


@router.get(
    "/labels",
    response_model=LabelListResponse,
    summary="Get all labels with optional filters"
)
def get_labels(
    level: Optional[int] = Query(None, ge=1, le=3, description="Filter by level"),
    parent_id: Optional[UUID] = Query(None, description="Filter by parent ID"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
):
    """Get all labels with optional filters."""
    try:
        with get_db() as conn:
            labels = LabelCRUD.get_all(conn, level=level, parent_id=parent_id, skip=skip, limit=limit)
            total = LabelCRUD.count(conn, level=level, parent_id=parent_id)
            
            return LabelListResponse(
                labels=labels,
                total=total,
                level=level,
                parent_id=parent_id
            )
    except Exception as e:
        logger.error(f"Error getting labels: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve labels"
        )


@router.get(
    "/labels/tree",
    response_model=list[LabelTreeResponse],
    summary="Get full label hierarchy as tree"
)
def get_label_tree():
    """Get the full label hierarchy as a tree structure."""
    try:
        with get_db() as conn:
            tree = LabelCRUD.get_tree(conn)
            return tree
    except Exception as e:
        logger.error(f"Error getting label tree: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve label tree"
        )


@router.get(
    "/labels/{label_id}",
    response_model=LabelResponse,
    summary="Get label by ID"
)
def get_label(label_id: UUID):
    """Get a specific label by ID."""
    try:
        with get_db() as conn:
            label = LabelCRUD.get_by_id(conn, label_id)
            if not label:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Label with id {label_id} not found"
                )
            return label
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting label {label_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve label"
        )


@router.get(
    "/labels/{label_id}/children",
    response_model=list[LabelResponse],
    summary="Get children of a label"
)
def get_label_children(label_id: UUID):
    """Get all children of a specific label."""
    try:
        with get_db() as conn:
            # First check if parent exists
            parent = LabelCRUD.get_by_id(conn, label_id)
            if not parent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Label with id {label_id} not found"
                )
            
            children = LabelCRUD.get_children(conn, label_id)
            return children
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting children for label {label_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve label children"
        )


@router.put(
    "/labels/{label_id}",
    response_model=LabelResponse,
    summary="Update a label"
)
def update_label(
    label_id: UUID,
    label_data: LabelUpdate,
):
    """Update a label's name or description."""
    try:
        with get_db() as conn:
            label = LabelCRUD.update(conn, label_id, label_data)
            if not label:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Label with id {label_id} not found"
                )
            return label
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating label {label_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update label"
        )


@router.delete(
    "/labels/{label_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a label"
)
def delete_label(label_id: UUID):
    """Delete a label and all its children (cascade delete)."""
    try:
        with get_db() as conn:
            deleted = LabelCRUD.delete(conn, label_id)
            if not deleted:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Label with id {label_id} not found"
                )
            return None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting label {label_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete label"
        )


# --- Feedback Sentiment Routes ---

async def call_sentiment_service(text: str) -> dict:
    """Call the sentiment analysis service to classify text."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.sentiment_service_url}/classify",
                json={"text": text}
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Error calling sentiment service: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment analysis service is unavailable"
        )
    except Exception as e:
        logger.error(f"Unexpected error calling sentiment service: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze sentiment"
        )


@router.post(
    "/feedbacks",
    response_model=FeedbackSentimentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit feedback and analyze sentiment"
)
async def create_feedback_sentiment(feedback_data: FeedbackSentimentCreate):
    """Submit customer feedback and get sentiment analysis."""
    try:
        # Call sentiment service to classify the feedback
        sentiment_result = await call_sentiment_service(feedback_data.feedback_text)
        
        # Extract sentiment label and score
        sentiment_label = sentiment_result.get("label")
        confidence_score = sentiment_result.get("score")
        
        if not sentiment_label or confidence_score is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid response from sentiment service"
            )
        
        # Save to database
        with get_db() as conn:
            feedback = FeedbackSentimentCRUD.create(
                conn,
                feedback_data,
                sentiment_label,
                confidence_score
            )
            return feedback
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating feedback sentiment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create feedback"
        )


@router.get(
    "/feedbacks",
    response_model=FeedbackSentimentListResponse,
    summary="Get all feedback sentiments with filters"
)
def get_feedback_sentiments(
    sentiment_label: Optional[SentimentLabel] = Query(None, description="Filter by sentiment label"),
    feedback_source: Optional[FeedbackSource] = Query(None, description="Filter by feedback source"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
):
    """Get all feedback sentiments with optional filters."""
    try:
        with get_db() as conn:
            # Convert enums to values for database query
            label_value = sentiment_label.value if sentiment_label else None
            source_value = feedback_source.value if feedback_source else None
            
            feedbacks = FeedbackSentimentCRUD.get_all(
                conn,
                sentiment_label=label_value,
                feedback_source=source_value,
                skip=skip,
                limit=limit
            )
            total = FeedbackSentimentCRUD.count(
                conn,
                sentiment_label=label_value,
                feedback_source=source_value
            )
            
            return FeedbackSentimentListResponse(
                feedbacks=feedbacks,
                total=total,
                sentiment_label=sentiment_label,
                feedback_source=feedback_source
            )
    except Exception as e:
        logger.error(f"Error getting feedback sentiments: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve feedbacks"
        )


@router.get(
    "/feedbacks/{feedback_id}",
    response_model=FeedbackSentimentResponse,
    summary="Get feedback sentiment by ID"
)
def get_feedback_sentiment(feedback_id: UUID):
    """Get a specific feedback sentiment by ID."""
    try:
        with get_db() as conn:
            feedback = FeedbackSentimentCRUD.get_by_id(conn, feedback_id)
            if not feedback:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Feedback with id {feedback_id} not found"
                )
            return feedback
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feedback {feedback_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve feedback"
        )
