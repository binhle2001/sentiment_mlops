"""SQLAlchemy models for label management."""
import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Column, String, Integer, Text, DateTime, ForeignKey, 
    CheckConstraint, UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, Mapped, mapped_column

from database import Base


class Label(Base):
    """Label model with hierarchical structure."""
    
    __tablename__ = "labels"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    
    # Label attributes
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    level: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Hierarchical relationship
    parent_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("labels.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
        index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
    
    # Relationships
    parent: Mapped[Optional["Label"]] = relationship(
        "Label",
        remote_side=[id],
        back_populates="children",
        lazy="selectin"
    )
    children: Mapped[List["Label"]] = relationship(
        "Label",
        back_populates="parent",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    # Table constraints
    __table_args__ = (
        # Unique name per parent
        UniqueConstraint('name', 'parent_id', name='unique_name_per_parent'),
        
        # Level must be 1, 2, or 3
        CheckConstraint('level IN (1, 2, 3)', name='check_valid_level'),
        
        # Level 1 must have no parent
        CheckConstraint(
            '(level = 1 AND parent_id IS NULL) OR (level > 1)',
            name='check_level_1_no_parent'
        ),
        
        # Level 2 and 3 must have parent
        CheckConstraint(
            '(level = 1) OR (level > 1 AND parent_id IS NOT NULL)',
            name='check_level_2_3_has_parent'
        ),
        
        # Indexes
        Index('idx_labels_parent_level', 'parent_id', 'level'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Label(id={self.id}, name={self.name}, level={self.level})>"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "level": self.level,
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def to_tree_dict(self) -> dict:
        """Convert to tree dictionary with children."""
        data = self.to_dict()
        data["children"] = [child.to_tree_dict() for child in self.children]
        return data


