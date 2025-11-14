"""CRUD operations with raw SQL queries."""
import logging
from typing import List, Optional, Dict, Any, Set
from uuid import UUID
from datetime import datetime

from schemas import (
    LabelCreate,
    LabelUpdate,
    FeedbackSentimentCreate,
    FeedbackSentimentUpdate,
    LabelSyncItem,
    LabelSyncResultStatus,
)

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
    def _generate_next_id(conn) -> int:
        """Generate next available integer ID for labels."""
        from database import execute_query

        result = execute_query(
            conn,
            "SELECT COALESCE(MAX(id), 0) + 1 AS next_id FROM labels",
            fetch="one",
        )
        return int(result["next_id"]) if result and result.get("next_id") is not None else 1

    @staticmethod
    def _validate_parent_for_sync(conn, label_id: int, level: int, parent_id: Optional[int]) -> None:
        """Ensure parent relationship is valid when syncing labels."""
        if level == 1:
            if parent_id is not None:
                raise ValueError("Level 1 labels cannot have a parent")
            return

        if parent_id is None:
            raise ValueError(f"Level {level} labels must have a parent")

        if parent_id == label_id:
            raise ValueError("Label cannot reference itself as parent")

        parent = LabelCRUD.get_by_id(conn, parent_id)
        if not parent:
            raise ValueError(f"Parent label with id {parent_id} not found")

        expected_level = level - 1
        if parent["level"] != expected_level:
            raise ValueError(
                f"Invalid hierarchy: label level {level} must have parent at level {expected_level}"
            )

    @staticmethod
    def _update_from_sync(conn, label_data: "LabelSyncItem") -> Dict[str, Any]:
        """Update label record with full payload coming from sync."""
        from database import execute_query

        query = """
            UPDATE labels
            SET name = %s,
                level = %s,
                parent_id = %s,
                description = %s
            WHERE id = %s
            RETURNING id, name, level, parent_id, description, created_at, updated_at
        """
        result = execute_query(
            conn,
            query,
            (
                label_data.name,
                label_data.level,
                label_data.parent_id,
                label_data.description,
                label_data.id,
            ),
            fetch="one",
        )
        if result is None:
            raise ValueError(f"Failed to update label with id {label_data.id}")
        return row_to_dict(result)

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
        new_id = label_data.id or LabelCRUD._generate_next_id(conn)

        # Ensure provided ID is unique
        if label_data.id is not None:
            existing = LabelCRUD.get_by_id(conn, label_data.id)
            if existing:
                raise ValueError(f"Label id {label_data.id} already exists")

        query = """
            INSERT INTO labels (id, name, level, parent_id, description)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, name, level, parent_id, description, created_at, updated_at
        """
        
        from database import execute_query
        result = execute_query(
            conn,
            query,
            (new_id, label_data.name, label_data.level, label_data.parent_id, label_data.description),
            fetch="one",
        )
        
        label = row_to_dict(result)
        logger.info(f"Created label: {label['name']} (id={label['id']}, level={label['level']})")
        return label
    
    @staticmethod
    def get_by_id(conn, label_id: int) -> Optional[Dict[str, Any]]:
        """Get label by ID."""
        query = """
            SELECT id, name, level, parent_id, description, created_at, updated_at
            FROM labels
            WHERE id = %s
        """
        
        from database import execute_query
        result = execute_query(conn, query, (label_id,), fetch="one")
        return row_to_dict(result)
    
    @staticmethod
    def get_all(
        conn,
        level: Optional[int] = None,
        parent_id: Optional[int] = None,
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
            params.append(parent_id)
        
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
        parent_id: Optional[int] = None
    ) -> int:
        """Count labels with optional filters."""
        conditions = []
        params = []
        
        if level is not None:
            conditions.append("level = %s")
            params.append(level)
        
        if parent_id is not None:
            conditions.append("parent_id = %s")
            params.append(parent_id)
        
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
        label_dict = {label['id']: {**label, 'children': []} for label in labels}
        tree = []
        
        for label in labels:
            label_with_children = label_dict[label['id']]
            if label['parent_id'] is None:
                tree.append(label_with_children)
            else:
                parent_id = label['parent_id']
                if parent_id in label_dict:
                    label_dict[parent_id]['children'].append(label_with_children)
        
        return tree
    
    @staticmethod
    def get_children(conn, parent_id: int) -> List[Dict[str, Any]]:
        """Get all children of a parent label."""
        query = """
            SELECT id, name, level, parent_id, description, created_at, updated_at
            FROM labels
            WHERE parent_id = %s
            ORDER BY name
        """
        
        from database import execute_query
        results = execute_query(conn, query, (parent_id,), fetch="all")
        return rows_to_list(results)
    
    @staticmethod
    def update(
        conn,
        label_id: int,
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
        
        params.append(label_id)
        
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
    def delete(conn, label_id: int) -> bool:
        """Delete a label (cascade deletes children)."""
        label = LabelCRUD.get_by_id(conn, label_id)
        if not label:
            return False
        
        query = "DELETE FROM labels WHERE id = %s"
        
        from database import execute_query
        execute_query(conn, query, (label_id,), fetch="none")
        
        logger.info(f"Deleted label: {label['name']} (id={label_id})")
        return True
    
    @staticmethod
    def exists_by_name_and_parent(
        conn,
        name: str,
        parent_id: Optional[int]
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
            params = (name, parent_id)
        
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
    
    @staticmethod
    def sync_labels(conn, labels_data: List["LabelSyncItem"]) -> Dict[str, Any]:
        """Synchronize labels coming from external systems."""
        if not labels_data:
            return {
                "created": 0,
                "updated": 0,
                "unchanged": 0,
                "results": [],
                "changed_label_ids": [],
            }

        created = 0
        updated = 0
        unchanged = 0
        changed_label_ids: Set[int] = set()
        result_map: Dict[int, Dict[str, Any]] = {}

        # Sort by level to ensure parents are processed before children
        sorted_labels = sorted(labels_data, key=lambda item: (item.level, item.id))

        for item in sorted_labels:
            # Ensure hierarchy is valid before applying changes
            LabelCRUD._validate_parent_for_sync(conn, item.id, item.level, item.parent_id)

            existing = LabelCRUD.get_by_id(conn, item.id)
            if not existing:
                payload = LabelCreate(**item.model_dump(exclude={"updated_at"}))
                label = LabelCRUD.create(conn, payload)
                created += 1

                result_map[item.id] = {
                    "id": label["id"],
                    "status": LabelSyncResultStatus.CREATED,
                    "changes": ["name", "level", "parent_id", "description"],
                    "message": "Created new label from sync",
                }
                continue

            # Determine changed fields
            changes: List[str] = []
            for field in ("name", "level", "parent_id", "description"):
                if existing.get(field) != getattr(item, field):
                    changes.append(field)

            if not changes:
                unchanged += 1
                result_map[item.id] = {
                    "id": existing["id"],
                    "status": LabelSyncResultStatus.UNCHANGED,
                    "changes": None,
                    "message": "No changes detected",
                }
                continue

            updated_label = LabelCRUD._update_from_sync(conn, item)
            updated += 1
            changed_label_ids.add(item.id)

            result_map[item.id] = {
                "id": updated_label["id"],
                "status": LabelSyncResultStatus.UPDATED,
                "changes": changes,
                "message": "Label updated from sync",
            }
            logger.info(
                "Synced label %s: updated fields=%s",
                updated_label["id"],
                ", ".join(changes),
            )

        ordered_results = [result_map[item.id] for item in labels_data]

        return {
            "created": created,
            "updated": updated,
            "unchanged": unchanged,
            "results": ordered_results,
            "changed_label_ids": list(changed_label_ids),
        }
    
    @staticmethod
    def update_embedding(conn, label_id: int, embedding_vector: list) -> bool:
        """Update embedding vector for a label."""
        query = """
            UPDATE labels
            SET embedding = %s
            WHERE id = %s
        """
        
        from database import execute_query
        execute_query(conn, query, (embedding_vector, label_id), fetch="none")
        
        logger.info(f"Updated embedding for label id={label_id}")
        return True
    
    @staticmethod
    def get_all_with_embeddings(conn) -> Dict[int, List[Dict[str, Any]]]:
        """Get all labels with embeddings, grouped by level."""
        query = """
            SELECT id, name, level, parent_id, description, embedding, created_at, updated_at
            FROM labels
            WHERE embedding IS NOT NULL
            ORDER BY level, name
        """
        
        from database import execute_query
        results = execute_query(conn, query, fetch="all")
        labels = rows_to_list(results)
        
        # Group by level
        grouped = {1: [], 2: [], 3: []}
        for label in labels:
            if label['level'] in grouped:
                grouped[label['level']].append(label)
        
        return grouped


class FeedbackSentimentCRUD:
    """CRUD operations for Feedback Sentiment with raw SQL."""
    
    @staticmethod
    def create(
        conn,
        feedback_data: FeedbackSentimentCreate,
        sentiment_label: str,
        confidence_score: float,
        level1_id: Optional[int] = None,
        level2_id: Optional[int] = None,
        level3_id: Optional[int] = None,
        is_model_confirmed: bool = False,
    ) -> Dict[str, Any]:
        """Create a new feedback sentiment record with optional intent labels."""
        query = """
            INSERT INTO feedback_sentiments (
                feedback_text, sentiment_label, confidence_score, feedback_source,
                level1_id, level2_id, level3_id, is_model_confirmed
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, feedback_text, sentiment_label, confidence_score, feedback_source,
                      is_model_confirmed, level1_id, level2_id, level3_id, created_at
        """
        
        from database import execute_query
        
        result = execute_query(
            conn,
            query,
            (
                feedback_data.feedback_text, 
                sentiment_label, 
                confidence_score, 
                feedback_data.feedback_source.value,
                level1_id,
                level2_id,
                level3_id,
                is_model_confirmed,
            ),
            fetch="one"
        )
        
        feedback = row_to_dict(result)
        logger.info(
            "Created feedback sentiment: id=%s, label=%s, intent=(%s, %s, %s)",
            feedback["id"],
            sentiment_label,
            level1_id,
            level2_id,
            level3_id,
        )

        # Enrich with label names for API response
        try:
            full_feedback = FeedbackSentimentCRUD.get_by_id(conn, feedback["id"])
            if full_feedback:
                return full_feedback
        except Exception as exc:  # pragma: no cover - best effort enrichment
            logger.warning("Failed to enrich feedback %s with label names: %s", feedback["id"], exc)
        
        return feedback

    @staticmethod
    def update(
        conn,
        feedback_id: UUID,
        update_data: FeedbackSentimentUpdate,
    ) -> Optional[Dict[str, Any]]:
        """Update sentiment label and/or intent hierarchy for a feedback."""
        existing = FeedbackSentimentCRUD.get_by_id(conn, feedback_id)
        if not existing:
            return None

        update_payload = update_data.model_dump(exclude_unset=True)

        # Normalize sentiment label value (enum -> str)
        sentiment_label = update_payload.get("sentiment_label")
        if sentiment_label is not None and hasattr(sentiment_label, "value"):
            sentiment_label = sentiment_label.value

        # Determine desired state for each level (fallback to existing if not provided)
        new_level1_id = update_payload.get("level1_id", existing.get("level1_id"))
        new_level2_id = update_payload.get("level2_id", existing.get("level2_id"))
        new_level3_id = update_payload.get("level3_id", existing.get("level3_id"))

        # If level1 is explicitly cleared, cascade reset level2 & level3
        if "level1_id" in update_payload and update_payload["level1_id"] is None:
            new_level2_id = None
            new_level3_id = None

        # If level2 cleared (explicitly or due to level1 change), ensure level3 cleared too
        if "level2_id" in update_payload and update_payload["level2_id"] is None:
            new_level3_id = None

        # Validate hierarchy when IDs are present
        def _load_label(label_id: Optional[int]) -> Optional[Dict[str, Any]]:
            if label_id is None:
                return None
            label = LabelCRUD.get_by_id(conn, label_id)
            if not label:
                raise ValueError(f"Label with id {label_id} not found")
            return label

        level1_label = _load_label(new_level1_id)
        level2_label = _load_label(new_level2_id)
        level3_label = _load_label(new_level3_id)

        if level1_label and level1_label["level"] != 1:
            raise ValueError("Level 1 intent phải là label cấp 1")

        if level2_label:
            if level2_label["level"] != 2:
                raise ValueError("Level 2 intent phải là label cấp 2")
            if not new_level1_id:
                raise ValueError("Phải chọn Level 1 trước khi chọn Level 2")
            if level2_label["parent_id"] != new_level1_id:
                raise ValueError("Level 2 được chọn không thuộc Level 1 đã chọn")

        if level3_label:
            if level3_label["level"] != 3:
                raise ValueError("Level 3 intent phải là label cấp 3")
            if not new_level2_id:
                raise ValueError("Phải chọn Level 2 trước khi chọn Level 3")
            if level3_label["parent_id"] != new_level2_id:
                raise ValueError("Level 3 được chọn không thuộc Level 2 đã chọn")

        # Ensure hierarchy consistency when parent missing
        if not new_level1_id:
            new_level2_id = None
            new_level3_id = None
        elif not new_level2_id:
            new_level3_id = None

        update_fields = []
        params = []

        sentiment_changed = False
        if sentiment_label is not None and sentiment_label != existing.get("sentiment_label"):
            update_fields.append("sentiment_label = %s")
            params.append(sentiment_label)
            sentiment_changed = True

        if "is_model_confirmed" in update_payload:
            is_confirmed = update_payload["is_model_confirmed"]
            if is_confirmed is not None and is_confirmed != existing.get("is_model_confirmed"):
                update_fields.append("is_model_confirmed = %s")
                params.append(is_confirmed)

        level1_changed = new_level1_id != existing.get("level1_id")
        level2_changed = new_level2_id != existing.get("level2_id")
        level3_changed = new_level3_id != existing.get("level3_id")
        level_changed = level1_changed or level2_changed or level3_changed

        if new_level1_id != existing.get("level1_id"):
            update_fields.append("level1_id = %s")
            params.append(new_level1_id)

        if new_level2_id != existing.get("level2_id"):
            update_fields.append("level2_id = %s")
            params.append(new_level2_id)

        if new_level3_id != existing.get("level3_id"):
            update_fields.append("level3_id = %s")
            params.append(new_level3_id)

        # Nếu có thay đổi sentiment hoặc intent và không chỉ định is_model_confirmed
        # thì kiểm tra xem có phải đang reset về False không
        # (chỉ reset nếu đang từ True -> False, không reset nếu đang từ False -> True)
        should_reset_confirmation = (
            "is_model_confirmed" not in update_payload
            and (sentiment_changed or level_changed)
            and existing.get("is_model_confirmed")
        )

        # Nếu có chỉ định is_model_confirmed trong payload, dùng giá trị đó
        if "is_model_confirmed" in update_payload:
            is_confirmed = update_payload["is_model_confirmed"]
            if is_confirmed != existing.get("is_model_confirmed"):
                update_fields.append("is_model_confirmed = %s")
                params.append(is_confirmed)
        elif should_reset_confirmation:
            # Chỉ reset về False nếu đang từ True và có thay đổi
            # (logic cũ - giữ lại để tương thích với các trường hợp đặc biệt)
            update_fields.append("is_model_confirmed = %s")
            params.append(False)

        if not update_fields:
            logger.info("No changes detected for feedback %s; skipping update", feedback_id)
            return existing

        params.append(str(feedback_id))
        query = f"""
            UPDATE feedback_sentiments
            SET {', '.join(update_fields)}
            WHERE id = %s
        """

        from database import execute_query

        execute_query(conn, query, tuple(params), fetch="none")

        updated_feedback = FeedbackSentimentCRUD.get_by_id(conn, feedback_id)
        logger.info(
            "Updated feedback %s sentiment/intent -> level1=%s, level2=%s, level3=%s",
            feedback_id,
            new_level1_id,
            new_level2_id,
            new_level3_id,
        )
        return updated_feedback
    
    @staticmethod
    def get_by_id(conn, feedback_id: UUID) -> Optional[Dict[str, Any]]:
        """Get feedback sentiment by ID with labels."""
        query = """
            SELECT 
                fs.id, fs.feedback_text, fs.sentiment_label, fs.confidence_score, 
                fs.feedback_source, fs.is_model_confirmed, fs.created_at,
                fs.level1_id, fs.level2_id, fs.level3_id,
                l1.name as level1_name,
                l2.name as level2_name,
                l3.name as level3_name
            FROM feedback_sentiments fs
            LEFT JOIN labels l1 ON fs.level1_id = l1.id
            LEFT JOIN labels l2 ON fs.level2_id = l2.id
            LEFT JOIN labels l3 ON fs.level3_id = l3.id
            WHERE fs.id = %s
        """
        
        from database import execute_query
        result = execute_query(conn, query, (str(feedback_id),), fetch="one")
        return row_to_dict(result)
    
    @staticmethod
    def get_all(
        conn,
        sentiment_label: Optional[str] = None,
        feedback_source: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all feedback sentiments with optional filters."""
        conditions = []
        params = []
        
        if sentiment_label is not None:
            conditions.append("sentiment_label = %s")
            params.append(sentiment_label)
        
        if feedback_source is not None:
            conditions.append("feedback_source = %s")
            params.append(feedback_source)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
            SELECT 
                fs.id,
                fs.feedback_text,
                fs.sentiment_label,
                fs.confidence_score,
                fs.feedback_source,
                fs.is_model_confirmed,
                fs.created_at,
                fs.level1_id,
                fs.level2_id,
                fs.level3_id,
                l1.name AS level1_name,
                l2.name AS level2_name,
                l3.name AS level3_name
            FROM feedback_sentiments fs
            LEFT JOIN labels l1 ON fs.level1_id = l1.id
            LEFT JOIN labels l2 ON fs.level2_id = l2.id
            LEFT JOIN labels l3 ON fs.level3_id = l3.id
            {where_clause}
            ORDER BY created_at DESC
            OFFSET %s LIMIT %s
        """
        params.extend([skip, limit])
        
        from database import execute_query
        results = execute_query(conn, query, tuple(params), fetch="all")
        return rows_to_list(results)
    
    @staticmethod
    def count(
        conn,
        sentiment_label: Optional[str] = None,
        feedback_source: Optional[str] = None
    ) -> int:
        """Count feedback sentiments with optional filters."""
        conditions = []
        params = []
        
        if sentiment_label is not None:
            conditions.append("sentiment_label = %s")
            params.append(sentiment_label)
        
        if feedback_source is not None:
            conditions.append("feedback_source = %s")
            params.append(feedback_source)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
            SELECT COUNT(*) as count
            FROM feedback_sentiments
            {where_clause}
        """
        
        from database import execute_query
        result = execute_query(conn, query, tuple(params) if params else None, fetch="one")
        return result['count'] if result else 0

    @staticmethod
    def reset_feedback_for_labels(conn, label_ids: List[int]) -> List[UUID]:
        """Clear label assignments and confirmation flags for affected feedbacks."""
        if not label_ids:
            return []

        unique_label_ids = sorted({int(label_id) for label_id in label_ids})
        from database import execute_query

        query = """
            UPDATE feedback_sentiments
            SET
                level1_id = CASE WHEN level1_id = ANY(%s) THEN NULL ELSE level1_id END,
                level2_id = CASE WHEN level2_id = ANY(%s) THEN NULL ELSE level2_id END,
                level3_id = CASE WHEN level3_id = ANY(%s) THEN NULL ELSE level3_id END,
                is_model_confirmed = FALSE
            WHERE level1_id = ANY(%s)
               OR level2_id = ANY(%s)
               OR level3_id = ANY(%s)
            RETURNING id
        """

        results = execute_query(
            conn,
            query,
            (
                unique_label_ids,
                unique_label_ids,
                unique_label_ids,
                unique_label_ids,
                unique_label_ids,
                unique_label_ids,
            ),
            fetch="all",
        )

        feedback_ids = [row["id"] for row in results] if results else []
        if feedback_ids:
            logger.info(
                "Reset %s feedback(s) due to label changes: labels=%s",
                len(feedback_ids),
                unique_label_ids,
            )
        return feedback_ids


class FeedbackIntentCRUD:
    """CRUD operations for Feedback Intent Analysis with raw SQL."""
    
    @staticmethod
    def cosine_similarity(vec1: list, vec2: list) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same length")
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    @staticmethod
    def get_top_intents(
        conn,
        feedback_embedding: list,
        limit: int = 10,
        top_level1: int = 5,
        top_level2_total: int = 15,
        top_level3_total: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Calculate top intent triplets using hierarchical top-down approach.
        
        NEW Algorithm (for Gemini):
        1. Find top 5 level1 labels with highest cosine similarity
        2. From 5 L1, get ALL L2 children → select top 15 L2 (cross all L1)
        3. From 15 L2, get ALL L3 children → select top 50 L3 (cross all L2)
        4. Rerank triplets by avg similarity → return top 10 for Gemini
        
        Args:
            feedback_embedding: Embedding vector of feedback text
            limit: Maximum number of triplets to return for Gemini (default: 10)
            top_level1: Number of top level1 to consider (default: 5)
            top_level2_total: Total number of top level2 across all L1 (default: 15)
            top_level3_total: Total number of top level3 across all L2 (default: 50)
        
        Returns:
            List of top 10 triplets with level1, level2, level3 and avg_cosine_similarity
        """
        # Get all labels with embeddings grouped by level
        labels_by_level = LabelCRUD.get_all_with_embeddings(conn)
        
        level1_labels = labels_by_level.get(1, [])
        level2_labels = labels_by_level.get(2, [])
        level3_labels = labels_by_level.get(3, [])
        
        if not level1_labels or not level2_labels or not level3_labels:
            logger.warning("Not enough labels with embeddings for all levels")
            return []
        
        # Build parent-child relationships
        level2_by_parent: Dict[int, List[Dict[str, Any]]] = {}
        for l2 in level2_labels:
            parent_id = l2['parent_id']
            if parent_id is None:
                continue
            level2_by_parent.setdefault(parent_id, []).append(l2)
        
        level3_by_parent: Dict[int, List[Dict[str, Any]]] = {}
        for l3 in level3_labels:
            parent_id = l3['parent_id']
            if parent_id is None:
                continue
            level3_by_parent.setdefault(parent_id, []).append(l3)
        
        # Step 1: Calculate similarity for all level1 and get top N
        level1_with_sim = []
        for l1 in level1_labels:
            if not l1['embedding']:
                continue
            sim1 = FeedbackIntentCRUD.cosine_similarity(feedback_embedding, l1['embedding'])
            level1_with_sim.append({
                'label': l1,
                'similarity': sim1
            })
        
        # Sort and get top level1
        level1_with_sim.sort(key=lambda x: x['similarity'], reverse=True)
        top_level1_items = level1_with_sim[:top_level1]
        
        logger.debug(f"Selected top {len(top_level1_items)} level1 labels")
        
        # Step 2: Get ALL level2 children from top 5 level1, then select top N total
        all_l2_candidates = []
        for l1_item in top_level1_items:
            l1 = l1_item['label']
            l1_id = l1['id']
            sim1 = l1_item['similarity']
            
            # Get ALL level2 children of this level1
            l2_children = level2_by_parent.get(l1_id, [])
            if not l2_children:
                continue
            
            # Calculate similarity for each level2 child
            for l2 in l2_children:
                if not l2['embedding']:
                    continue
                sim2 = FeedbackIntentCRUD.cosine_similarity(feedback_embedding, l2['embedding'])
                all_l2_candidates.append({
                    'level1': l1,
                    'level1_sim': sim1,
                    'level2': l2,
                    'level2_sim': sim2
                })
        
        # Sort ALL level2 by similarity and take top N (cross all level1)
        all_l2_candidates.sort(key=lambda x: x['level2_sim'], reverse=True)
        level2_with_sim = all_l2_candidates[:top_level2_total]
        
        logger.debug(f"Selected top {len(level2_with_sim)} level2 labels (from {len(all_l2_candidates)} candidates)")
        
        # Step 3: Get ALL level3 children from top 15 level2, then select top N total
        all_triplet_candidates = []
        for l2_item in level2_with_sim:
            l1 = l2_item['level1']
            l2 = l2_item['level2']
            sim1 = l2_item['level1_sim']
            sim2 = l2_item['level2_sim']
            l2_id = l2['id']
            
            # Get ALL level3 children of this level2
            l3_children = level3_by_parent.get(l2_id, [])
            if not l3_children:
                continue
            
            # Calculate similarity for each level3 child
            for l3 in l3_children:
                if not l3['embedding']:
                    continue
                sim3 = FeedbackIntentCRUD.cosine_similarity(feedback_embedding, l3['embedding'])
                
                # Calculate average similarity across all 3 levels
                avg_similarity = (sim1 + sim2 + sim3) / 3.0
                
                all_triplet_candidates.append({
                    'level1': l1,
                    'level2': l2,
                    'level3': l3,
                    'avg_cosine_similarity': avg_similarity,
                    'level1_sim': sim1,
                    'level2_sim': sim2,
                    'level3_sim': sim3
                })
        
        # Sort ALL triplets by level3 similarity and take top N (cross all level2)
        all_triplet_candidates.sort(key=lambda x: x['level3_sim'], reverse=True)
        triplets_top50 = all_triplet_candidates[:top_level3_total]
        
        logger.debug(f"Selected top {len(triplets_top50)} triplets (from {len(all_triplet_candidates)} candidates)")
        
        # Step 4: Rerank by average similarity and return top 10 for Gemini
        triplets_top50.sort(key=lambda x: x['avg_cosine_similarity'], reverse=True)
        final_triplets = triplets_top50[:limit]
        
        logger.info(f"Returning top {len(final_triplets)} triplets for Gemini selection")
        
        return final_triplets
    
    @staticmethod
    def delete_by_feedback_ids(conn, feedback_ids: List[UUID]) -> int:
        """Delete cached intents associated with provided feedback IDs."""
        if not feedback_ids:
            return 0

        unique_ids = list({str(feedback_id) for feedback_id in feedback_ids})

        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM feedback_intents
                WHERE feedback_id = ANY(%s)
                """,
                (unique_ids,),
            )
            deleted = cur.rowcount

        if deleted:
            logger.info(
                "Deleted %s cached intent(s) for feedbacks: %s",
                deleted,
                unique_ids,
            )
        return deleted
