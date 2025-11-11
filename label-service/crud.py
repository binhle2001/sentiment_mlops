"""CRUD operations with raw SQL queries."""
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime

from schemas import LabelCreate, LabelUpdate, FeedbackSentimentCreate

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
    
    @staticmethod
    def update_embedding(conn, label_id: UUID, embedding_vector: list) -> bool:
        """Update embedding vector for a label."""
        query = """
            UPDATE labels
            SET embedding = %s
            WHERE id = %s
        """
        
        from database import execute_query
        execute_query(conn, query, (embedding_vector, str(label_id)), fetch="none")
        
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
        level1_id: Optional[UUID] = None,
        level2_id: Optional[UUID] = None,
        level3_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """Create a new feedback sentiment record with optional intent labels."""
        query = """
            INSERT INTO feedback_sentiments (
                feedback_text, sentiment_label, confidence_score, feedback_source,
                level1_id, level2_id, level3_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id, feedback_text, sentiment_label, confidence_score, feedback_source, 
                      level1_id, level2_id, level3_id, created_at
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
                str(level1_id) if level1_id else None,
                str(level2_id) if level2_id else None,
                str(level3_id) if level3_id else None
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
    def get_by_id(conn, feedback_id: UUID) -> Optional[Dict[str, Any]]:
        """Get feedback sentiment by ID with labels."""
        query = """
            SELECT 
                fs.id, fs.feedback_text, fs.sentiment_label, fs.confidence_score, 
                fs.feedback_source, fs.created_at,
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
        level2_by_parent = {}
        for l2 in level2_labels:
            parent_id = str(l2['parent_id']) if l2['parent_id'] else None
            if parent_id:
                if parent_id not in level2_by_parent:
                    level2_by_parent[parent_id] = []
                level2_by_parent[parent_id].append(l2)
        
        level3_by_parent = {}
        for l3 in level3_labels:
            parent_id = str(l3['parent_id']) if l3['parent_id'] else None
            if parent_id:
                if parent_id not in level3_by_parent:
                    level3_by_parent[parent_id] = []
                level3_by_parent[parent_id].append(l3)
        
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
            l1_id = str(l1['id'])
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
            l2_id = str(l2['id'])
            
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
    
