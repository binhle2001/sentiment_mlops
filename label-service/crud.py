"""CRUD operations for labels."""
import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models import Label
from schemas import LabelCreate, LabelUpdate

logger = logging.getLogger(__name__)


class LabelCRUD:
    """CRUD operations for Label model."""
    
    @staticmethod
    async def create(db: AsyncSession, label_data: LabelCreate) -> Label:
        """Create a new label."""
        # Validate parent exists if parent_id is provided
        if label_data.parent_id:
            parent = await LabelCRUD.get_by_id(db, label_data.parent_id)
            if not parent:
                raise ValueError(f"Parent label with id {label_data.parent_id} not found")
            
            # Validate hierarchy: level must be parent.level + 1
            if label_data.level != parent.level + 1:
                raise ValueError(
                    f"Invalid level {label_data.level}. "
                    f"Parent is level {parent.level}, so child must be level {parent.level + 1}"
                )
        
        # Create label
        label = Label(
            name=label_data.name,
            level=label_data.level,
            parent_id=label_data.parent_id,
            description=label_data.description
        )
        
        db.add(label)
        await db.flush()
        await db.refresh(label)
        
        logger.info(f"Created label: {label.name} (id={label.id}, level={label.level})")
        return label
    
    @staticmethod
    async def get_by_id(db: AsyncSession, label_id: UUID) -> Optional[Label]:
        """Get label by ID."""
        result = await db.execute(
            select(Label)
            .options(selectinload(Label.children))
            .where(Label.id == label_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_all(
        db: AsyncSession,
        level: Optional[int] = None,
        parent_id: Optional[UUID] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Label]:
        """Get all labels with optional filters."""
        query = select(Label).options(selectinload(Label.children))
        
        # Apply filters
        filters = []
        if level is not None:
            filters.append(Label.level == level)
        if parent_id is not None:
            filters.append(Label.parent_id == parent_id)
        
        if filters:
            query = query.where(and_(*filters))
        
        # Apply pagination
        query = query.offset(skip).limit(limit).order_by(Label.level, Label.name)
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    @staticmethod
    async def count(
        db: AsyncSession,
        level: Optional[int] = None,
        parent_id: Optional[UUID] = None
    ) -> int:
        """Count labels with optional filters."""
        query = select(func.count(Label.id))
        
        filters = []
        if level is not None:
            filters.append(Label.level == level)
        if parent_id is not None:
            filters.append(Label.parent_id == parent_id)
        
        if filters:
            query = query.where(and_(*filters))
        
        result = await db.execute(query)
        return result.scalar_one()
    
    @staticmethod
    async def get_tree(db: AsyncSession) -> List[Label]:
        """Get all labels as a hierarchical tree (only root level 1 labels)."""
        # Get all level 1 labels with their children loaded recursively
        result = await db.execute(
            select(Label)
            .options(selectinload(Label.children))
            .where(Label.level == 1)
            .order_by(Label.name)
        )
        return list(result.scalars().all())
    
    @staticmethod
    async def get_children(db: AsyncSession, parent_id: UUID) -> List[Label]:
        """Get all children of a parent label."""
        result = await db.execute(
            select(Label)
            .where(Label.parent_id == parent_id)
            .order_by(Label.name)
        )
        return list(result.scalars().all())
    
    @staticmethod
    async def update(
        db: AsyncSession,
        label_id: UUID,
        label_data: LabelUpdate
    ) -> Optional[Label]:
        """Update a label."""
        label = await LabelCRUD.get_by_id(db, label_id)
        if not label:
            return None
        
        # Update fields
        update_data = label_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(label, field, value)
        
        await db.flush()
        await db.refresh(label)
        
        logger.info(f"Updated label: {label.name} (id={label.id})")
        return label
    
    @staticmethod
    async def delete(db: AsyncSession, label_id: UUID) -> bool:
        """Delete a label (cascade deletes children)."""
        label = await LabelCRUD.get_by_id(db, label_id)
        if not label:
            return False
        
        await db.delete(label)
        await db.flush()
        
        logger.info(f"Deleted label: {label.name} (id={label_id})")
        return True
    
    @staticmethod
    async def exists_by_name_and_parent(
        db: AsyncSession,
        name: str,
        parent_id: Optional[UUID]
    ) -> bool:
        """Check if a label with the same name and parent exists."""
        query = select(Label).where(Label.name == name)
        
        if parent_id is None:
            query = query.where(Label.parent_id.is_(None))
        else:
            query = query.where(Label.parent_id == parent_id)
        
        result = await db.execute(query)
        return result.scalar_one_or_none() is not None



