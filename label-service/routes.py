"""API routes for label management."""
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from crud import LabelCRUD
from schemas import (
    LabelCreate,
    LabelUpdate,
    LabelResponse,
    LabelTreeResponse,
    LabelListResponse,
    HealthResponse,
)
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Health check endpoint")
async def health_check():
    """Health check endpoint."""
    # Simplified health check without DB query to avoid async issues with docker healthcheck
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
async def create_label(
    label_data: LabelCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new label in the hierarchy."""
    try:
        # Check if label with same name and parent already exists
        exists = await LabelCRUD.exists_by_name_and_parent(
            db, label_data.name, label_data.parent_id
        )
        if exists:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Label '{label_data.name}' already exists under this parent"
            )
        
        label = await LabelCRUD.create(db, label_data)
        await db.commit()
        return label
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating label: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create label"
        )


@router.get(
    "/labels",
    response_model=LabelListResponse,
    summary="Get all labels with optional filters"
)
async def get_labels(
    level: Optional[int] = Query(None, ge=1, le=3, description="Filter by level"),
    parent_id: Optional[UUID] = Query(None, description="Filter by parent ID"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    db: AsyncSession = Depends(get_db)
):
    """Get all labels with optional filters."""
    try:
        labels = await LabelCRUD.get_all(db, level=level, parent_id=parent_id, skip=skip, limit=limit)
        total = await LabelCRUD.count(db, level=level, parent_id=parent_id)
        
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
async def get_label_tree(db: AsyncSession = Depends(get_db)):
    """Get the full label hierarchy as a tree structure."""
    try:
        tree = await LabelCRUD.get_tree(db)
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
async def get_label(
    label_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific label by ID."""
    try:
        label = await LabelCRUD.get_by_id(db, label_id)
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
async def get_label_children(
    label_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get all children of a specific label."""
    try:
        # First check if parent exists
        parent = await LabelCRUD.get_by_id(db, label_id)
        if not parent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Label with id {label_id} not found"
            )
        
        children = await LabelCRUD.get_children(db, label_id)
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
async def update_label(
    label_id: UUID,
    label_data: LabelUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a label's name or description."""
    try:
        label = await LabelCRUD.update(db, label_id, label_data)
        if not label:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Label with id {label_id} not found"
            )
        
        await db.commit()
        return label
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating label {label_id}: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update label"
        )


@router.delete(
    "/labels/{label_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a label"
)
async def delete_label(
    label_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Delete a label and all its children (cascade delete)."""
    try:
        deleted = await LabelCRUD.delete(db, label_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Label with id {label_id} not found"
            )
        
        await db.commit()
        return None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting label {label_id}: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete label"
        )



