"""CRUD operations with raw SQL queries."""
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime

from schemas import LabelCreate, LabelUpdate

logger = logging.getLogger(__name__)


def row_to_dict(row) -> Optional[Dict[str, Any]]:
    """Convert database row to dictionary."""
    if row is None:
        return None
    return dict(row)


def rows_to_list(rows) -> List[Dict[str, Any]]:
    """Convert database rows to list of dictionaries."""
    return [dict(row) for row in rows] if rows else []


class LabelCRUD:
    """CRUD operations for Label with raw SQL."""
    
    @staticmethod
    def create(conn, label_data: LabelCreate) -> Dict[str, Any]:
        """Create a new label."""
        # Validate parent exists if parent_id is provided
        if label_data.parent_id:
            parent = LabelCRUD.get_by_id(conn, label_data.parent_id)
            if not parent:
                raise ValueError(f"Parent label with id {label_data.parent_id} not found")
            
            # Validate hierarchy: level must be parent.level + 1
            if label_data.level != parent['level'] + 1:
                raise ValueError(
                    f"Invalid level {label_data.level}. "
                    f"Parent is level {parent['level']}, so child must be level {parent['level'] + 1}"
                )
        
        query = """
            INSERT INTO labels (name, level, parent_id, description)
            VALUES (%s, %s, %s, %s)
            RETURNING id, name, level, parent_id, description, created_at, updated_at
        """
        
        from database import execute_query
        # Convert UUID to string if not None
        parent_id_str = str(label_data.parent_id) if label_data.parent_id else None
        
        result = execute_query(
            conn, 
            query, 
            (label_data.name, label_data.level, parent_id_str, label_data.description),
            fetch="one"
        )
        
        label = row_to_dict(result)
        logger.info(f"Created label: {label['name']} (id={label['id']}, level={label['level']})")
        return label
    
    @staticmethod
    def get_by_id(conn, label_id: UUID) -> Optional[Dict[str, Any]]:
        """Get label by ID."""
        query = """
            SELECT id, name, level, parent_id, description, created_at, updated_at
            FROM labels
            WHERE id = %s
        """
        
        from database import execute_query
        result = execute_query(conn, query, (str(label_id),), fetch="one")
        return row_to_dict(result)
    
    @staticmethod
    def get_all(
        conn,
        level: Optional[int] = None,
        parent_id: Optional[UUID] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all labels with optional filters."""
        conditions = []
        params = []
        
        if level is not None:
            conditions.append("level = %s")
            params.append(level)
        
        if parent_id is not None:
            conditions.append("parent_id = %s")
            # Convert UUID to string
            params.append(str(parent_id) if parent_id else None)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
            SELECT id, name, level, parent_id, description, created_at, updated_at
            FROM labels
            {where_clause}
            ORDER BY level, name
            OFFSET %s LIMIT %s
        """
        params.extend([skip, limit])
        
        from database import execute_query
        results = execute_query(conn, query, tuple(params), fetch="all")
        return rows_to_list(results)
    
    @staticmethod
    def count(
        conn,
        level: Optional[int] = None,
        parent_id: Optional[UUID] = None
    ) -> int:
        """Count labels with optional filters."""
        conditions = []
        params = []
        
        if level is not None:
            conditions.append("level = %s")
            params.append(level)
        
        if parent_id is not None:
            conditions.append("parent_id = %s")
            # Convert UUID to string
            params.append(str(parent_id) if parent_id else None)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
            SELECT COUNT(*) as count
            FROM labels
            {where_clause}
        """
        
        from database import execute_query
        result = execute_query(conn, query, tuple(params) if params else None, fetch="one")
        return result['count'] if result else 0
    
    @staticmethod
    def get_tree(conn) -> List[Dict[str, Any]]:
        """Get all labels as a hierarchical tree."""
        # Get all labels
        query = """
            SELECT id, name, level, parent_id, description, created_at, updated_at
            FROM labels
            ORDER BY level, name
        """
        
        from database import execute_query
        results = execute_query(conn, query, fetch="all")
        labels = rows_to_list(results)
        
        # Build tree structure
        label_dict = {str(label['id']): {**label, 'children': []} for label in labels}
        tree = []
        
        for label in labels:
            label_with_children = label_dict[str(label['id'])]
            if label['parent_id'] is None:
                tree.append(label_with_children)
            else:
                parent_id = str(label['parent_id'])
                if parent_id in label_dict:
                    label_dict[parent_id]['children'].append(label_with_children)
        
        return tree
    
    @staticmethod
    def get_children(conn, parent_id: UUID) -> List[Dict[str, Any]]:
        """Get all children of a parent label."""
        query = """
            SELECT id, name, level, parent_id, description, created_at, updated_at
            FROM labels
            WHERE parent_id = %s
            ORDER BY name
        """
        
        from database import execute_query
        results = execute_query(conn, query, (str(parent_id),), fetch="all")
        return rows_to_list(results)
    
    @staticmethod
    def update(
        conn,
        label_id: UUID,
        label_data: LabelUpdate
    ) -> Optional[Dict[str, Any]]:
        """Update a label."""
        label = LabelCRUD.get_by_id(conn, label_id)
        if not label:
            return None
        
        # Build update query dynamically
        update_fields = []
        params = []
        
        update_data = label_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            update_fields.append(f"{field} = %s")
            params.append(value)
        
        if not update_fields:
            return label
        
        params.append(str(label_id))
        
        query = f"""
            UPDATE labels
            SET {', '.join(update_fields)}
            WHERE id = %s
            RETURNING id, name, level, parent_id, description, created_at, updated_at
        """
        
        from database import execute_query
        result = execute_query(conn, query, tuple(params), fetch="one")
        updated_label = row_to_dict(result)
        
        logger.info(f"Updated label: {updated_label['name']} (id={label_id})")
        return updated_label
    
    @staticmethod
    def delete(conn, label_id: UUID) -> bool:
        """Delete a label (cascade deletes children)."""
        label = LabelCRUD.get_by_id(conn, label_id)
        if not label:
            return False
        
        query = "DELETE FROM labels WHERE id = %s"
        
        from database import execute_query
        execute_query(conn, query, (str(label_id),), fetch="none")
        
        logger.info(f"Deleted label: {label['name']} (id={label_id})")
        return True
    
    @staticmethod
    def exists_by_name_and_parent(
        conn,
        name: str,
        parent_id: Optional[UUID]
    ) -> bool:
        """Check if a label with the same name and parent exists."""
        if parent_id is None:
            query = """
                SELECT id FROM labels
                WHERE name = %s AND parent_id IS NULL
            """
            params = (name,)
        else:
            query = """
                SELECT id FROM labels
                WHERE name = %s AND parent_id = %s
            """
            # Convert UUID to string
            params = (name, str(parent_id) if parent_id else None)
        
        from database import execute_query
        result = execute_query(conn, query, params, fetch="one")
        return result is not None
    
    @staticmethod
    def bulk_create(conn, labels_data: List[LabelCreate]) -> List[Dict[str, Any]]:
        """Create multiple labels at once.
        
        Returns list of results with success status and created label or error message.
        """
        results = []
        
        for idx, label_data in enumerate(labels_data):
            try:
                # Check if label with same name and parent already exists
                exists = LabelCRUD.exists_by_name_and_parent(
                    conn, label_data.name, label_data.parent_id
                )
                if exists:
                    results.append({
                        'success': False,
                        'error': f"Label '{label_data.name}' already exists under this parent",
                        'index': idx,
                        'label': None
                    })
                    continue
                
                # Create label
                label = LabelCRUD.create(conn, label_data)
                results.append({
                    'success': True,
                    'error': None,
                    'index': idx,
                    'label': label
                })
                
            except ValueError as e:
                results.append({
                    'success': False,
                    'error': str(e),
                    'index': idx,
                    'label': None
                })
            except Exception as e:
                logger.error(f"Error creating label at index {idx}: {e}", exc_info=True)
                results.append({
                    'success': False,
                    'error': f"Failed to create label: {str(e)}",
                    'index': idx,
                    'label': None
                })
        
        return results
