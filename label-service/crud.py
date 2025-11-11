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
        confidence_score: float
    ) -> Dict[str, Any]:
        """Create a new feedback sentiment record."""
        query = """
            INSERT INTO feedback_sentiments (feedback_text, sentiment_label, confidence_score, feedback_source)
            VALUES (%s, %s, %s, %s)
            RETURNING id, feedback_text, sentiment_label, confidence_score, feedback_source, created_at
        """
        
        from database import execute_query
        
        result = execute_query(
            conn,
            query,
            (feedback_data.feedback_text, sentiment_label, confidence_score, feedback_data.feedback_source.value),
            fetch="one"
        )
        
        feedback = row_to_dict(result)
        logger.info(f"Created feedback sentiment: id={feedback['id']}, label={sentiment_label}")
        return feedback
    
    @staticmethod
    def get_by_id(conn, feedback_id: UUID) -> Optional[Dict[str, Any]]:
        """Get feedback sentiment by ID."""
        query = """
            SELECT id, feedback_text, sentiment_label, confidence_score, feedback_source, created_at
            FROM feedback_sentiments
            WHERE id = %s
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
            SELECT id, feedback_text, sentiment_label, confidence_score, feedback_source, created_at
            FROM feedback_sentiments
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
        limit: int = 50,
        top_level1: int = 5,
        top_level2_per_level1: int = 4,
        top_level3_per_level2: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Calculate top intent triplets using hierarchical top-down approach.
        
        Algorithm:
        1. Find top 5 level1 labels with highest cosine similarity
        2. For each top level1, find top 4 level2 children → ~20 level2
        3. For each top level2, find top 2-3 level3 children → ~50 triplets
        
        Args:
            feedback_embedding: Embedding vector of feedback text
            limit: Maximum number of triplets to return (default: 50)
            top_level1: Number of top level1 to consider (default: 5)
            top_level2_per_level1: Number of top level2 per level1 (default: 4)
            top_level3_per_level2: Number of top level3 per level2 (default: 3)
        
        Returns:
            List of triplets with level1, level2, level3 and avg_cosine_similarity
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
        
        # Step 2: For each top level1, get top level2 children
        level2_with_sim = []
        for l1_item in top_level1_items:
            l1 = l1_item['label']
            l1_id = str(l1['id'])
            sim1 = l1_item['similarity']
            
            # Get level2 children
            l2_children = level2_by_parent.get(l1_id, [])
            if not l2_children:
                continue
            
            # Calculate similarity for each level2 child
            l2_candidates = []
            for l2 in l2_children:
                if not l2['embedding']:
                    continue
                sim2 = FeedbackIntentCRUD.cosine_similarity(feedback_embedding, l2['embedding'])
                l2_candidates.append({
                    'level1': l1,
                    'level1_sim': sim1,
                    'level2': l2,
                    'level2_sim': sim2
                })
            
            # Sort by level2 similarity and take top N
            l2_candidates.sort(key=lambda x: x['level2_sim'], reverse=True)
            level2_with_sim.extend(l2_candidates[:top_level2_per_level1])
        
        logger.debug(f"Selected {len(level2_with_sim)} level2 labels")
        
        # Step 3: For each selected level2, get top level3 children
        triplets = []
        for l2_item in level2_with_sim:
            l1 = l2_item['level1']
            l2 = l2_item['level2']
            sim1 = l2_item['level1_sim']
            sim2 = l2_item['level2_sim']
            l2_id = str(l2['id'])
            
            # Get level3 children
            l3_children = level3_by_parent.get(l2_id, [])
            if not l3_children:
                continue
            
            # Calculate similarity for each level3 child
            l3_candidates = []
            for l3 in l3_children:
                if not l3['embedding']:
                    continue
                sim3 = FeedbackIntentCRUD.cosine_similarity(feedback_embedding, l3['embedding'])
                
                # Calculate average similarity across all 3 levels
                avg_similarity = (sim1 + sim2 + sim3) / 3.0
                
                l3_candidates.append({
                    'level1': l1,
                    'level2': l2,
                    'level3': l3,
                    'avg_cosine_similarity': avg_similarity,
                    'level1_sim': sim1,
                    'level2_sim': sim2,
                    'level3_sim': sim3
                })
            
            # Sort by level3 similarity and take top N
            l3_candidates.sort(key=lambda x: x['level3_sim'], reverse=True)
            triplets.extend(l3_candidates[:top_level3_per_level2])
        
        logger.debug(f"Generated {len(triplets)} triplets")
        
        # Sort all triplets by average similarity and return top N
        triplets.sort(key=lambda x: x['avg_cosine_similarity'], reverse=True)
        
        return triplets[:limit]
    
    @staticmethod
    def save_intents(
        conn,
        feedback_id: UUID,
        intents: List[Dict[str, Any]]
    ) -> int:
        """Save intent analysis results to database."""
        # First, delete existing intents for this feedback
        delete_query = "DELETE FROM feedback_intents WHERE feedback_id = %s"
        
        from database import execute_query
        execute_query(conn, delete_query, (str(feedback_id),), fetch="none")
        
        # Insert new intents
        insert_query = """
            INSERT INTO feedback_intents (feedback_id, level1_id, level2_id, level3_id, avg_cosine_similarity)
            VALUES (%s, %s, %s, %s, %s)
        """
        
        count = 0
        for intent in intents:
            execute_query(
                conn,
                insert_query,
                (
                    str(feedback_id),
                    str(intent['level1']['id']),
                    str(intent['level2']['id']),
                    str(intent['level3']['id']),
                    intent['avg_cosine_similarity']
                ),
                fetch="none"
            )
            count += 1
        
        logger.info(f"Saved {count} intents for feedback id={feedback_id}")
        return count
    
    @staticmethod
    def get_cached_intents(
        conn,
        feedback_id: UUID,
        limit: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached intent analysis results from database."""
        query = """
            SELECT 
                fi.id,
                fi.feedback_id,
                fi.avg_cosine_similarity,
                fi.created_at,
                l1.id as l1_id, l1.name as l1_name, l1.level as l1_level, 
                l1.parent_id as l1_parent_id, l1.description as l1_description,
                l1.created_at as l1_created_at, l1.updated_at as l1_updated_at,
                l2.id as l2_id, l2.name as l2_name, l2.level as l2_level,
                l2.parent_id as l2_parent_id, l2.description as l2_description,
                l2.created_at as l2_created_at, l2.updated_at as l2_updated_at,
                l3.id as l3_id, l3.name as l3_name, l3.level as l3_level,
                l3.parent_id as l3_parent_id, l3.description as l3_description,
                l3.created_at as l3_created_at, l3.updated_at as l3_updated_at
            FROM feedback_intents fi
            JOIN labels l1 ON fi.level1_id = l1.id
            JOIN labels l2 ON fi.level2_id = l2.id
            JOIN labels l3 ON fi.level3_id = l3.id
            WHERE fi.feedback_id = %s
            ORDER BY fi.avg_cosine_similarity DESC
            LIMIT %s
        """
        
        from database import execute_query
        results = execute_query(conn, query, (str(feedback_id), limit), fetch="all")
        
        if not results:
            return None
        
        intents = []
        for row in results:
            row_dict = dict(row)
            intents.append({
                'level1': {
                    'id': row_dict['l1_id'],
                    'name': row_dict['l1_name'],
                    'level': row_dict['l1_level'],
                    'parent_id': row_dict['l1_parent_id'],
                    'description': row_dict['l1_description'],
                    'created_at': row_dict['l1_created_at'],
                    'updated_at': row_dict['l1_updated_at']
                },
                'level2': {
                    'id': row_dict['l2_id'],
                    'name': row_dict['l2_name'],
                    'level': row_dict['l2_level'],
                    'parent_id': row_dict['l2_parent_id'],
                    'description': row_dict['l2_description'],
                    'created_at': row_dict['l2_created_at'],
                    'updated_at': row_dict['l2_updated_at']
                },
                'level3': {
                    'id': row_dict['l3_id'],
                    'name': row_dict['l3_name'],
                    'level': row_dict['l3_level'],
                    'parent_id': row_dict['l3_parent_id'],
                    'description': row_dict['l3_description'],
                    'created_at': row_dict['l3_created_at'],
                    'updated_at': row_dict['l3_updated_at']
                },
                'avg_cosine_similarity': row_dict['avg_cosine_similarity']
            })
        
        return intents
