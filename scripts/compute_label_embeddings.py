#!/usr/bin/env python3
"""
Script to compute embeddings for all labels in the database.
This script connects to the label database and calls the embedding service
to generate embeddings for all labels.

Usage:
    python scripts/compute_label_embeddings.py
"""
import os
import sys
import logging
import requests
from typing import List, Dict, Any
from tqdm import tqdm
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration from environment variables
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'label_db')
DB_USER = os.getenv('POSTGRES_USER', 'labeluser')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'labelpass123')

EMBEDDING_SERVICE_URL = os.getenv('EMBEDDING_SERVICE_URL', 'http://localhost:8000/api/v1')


def get_db_connection():
    """Create database connection."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)


def get_all_labels(conn) -> List[Dict[str, Any]]:
    """Fetch all labels from database."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, level, parent_id, description
                FROM labels
                ORDER BY level, name
            """)
            labels = cur.fetchall()
            return [dict(label) for label in labels]
    except Exception as e:
        logger.error(f"Failed to fetch labels: {e}")
        return []


def get_embedding(text: str) -> List[float]:
    """Call embedding service to get embedding for text."""
    try:
        response = requests.post(
            f"{EMBEDDING_SERVICE_URL}/encode",
            json={"text": text},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result.get('embedding', [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get embedding: {e}")
        return []


def update_label_embedding(conn, label_id: str, embedding: List[float]) -> bool:
    """Update label embedding in database."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE labels
                SET embedding = %s
                WHERE id = %s
            """, (embedding, label_id))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to update embedding for label {label_id}: {e}")
        conn.rollback()
        return False


def compute_label_embeddings():
    """Main function to compute embeddings for all labels."""
    logger.info("Starting label embedding computation...")
    
    # Connect to database
    conn = get_db_connection()
    logger.info(f"Connected to database: {DB_NAME}")
    
    # Get all labels
    labels = get_all_labels(conn)
    logger.info(f"Found {len(labels)} labels to process")
    
    if not labels:
        logger.warning("No labels found in database")
        conn.close()
        return
    
    # Process each label
    success_count = 0
    failure_count = 0
    
    for label in tqdm(labels, desc="Computing embeddings"):
        label_id = label['id']
        label_name = label['name']
        
        # Create text for embedding (name + description if available)
        text = label_name
        if label.get('description'):
            text = f"{label_name}. {label['description']}"
        
        # Get embedding
        embedding = get_embedding(text)
        
        if not embedding:
            logger.warning(f"Failed to get embedding for label: {label_name} (id={label_id})")
            failure_count += 1
            continue
        
        # Update database
        if update_label_embedding(conn, label_id, embedding):
            success_count += 1
            logger.debug(f"Updated embedding for label: {label_name} (id={label_id})")
        else:
            failure_count += 1
    
    # Close connection
    conn.close()
    
    # Summary
    logger.info("=" * 60)
    logger.info("Embedding computation completed!")
    logger.info(f"Total labels: {len(labels)}")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Failed: {failure_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        compute_label_embeddings()
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


