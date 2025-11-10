#!/usr/bin/env python3
"""
Script to compute intent analysis for all feedbacks in the database.
This script reads all feedbacks and calls the label-backend API to compute
and cache intent analysis results.

Usage:
    python scripts/compute_feedback_intents.py
    python scripts/compute_feedback_intents.py --recompute  # Recompute all, including cached
"""
import os
import sys
import argparse
import logging
import requests
from typing import List, Dict, Any
from tqdm import tqdm
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import time

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

LABEL_BACKEND_URL = os.getenv('LABEL_BACKEND_URL', 'http://localhost:8001/api/v1')


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


def get_feedbacks_without_intents(conn) -> List[Dict[str, Any]]:
    """Fetch feedbacks that don't have intent analysis yet."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT fs.id, fs.feedback_text, fs.created_at
                FROM feedback_sentiments fs
                LEFT JOIN feedback_intents fi ON fs.id = fi.feedback_id
                WHERE fi.id IS NULL
                ORDER BY fs.created_at DESC
            """)
            feedbacks = cur.fetchall()
            return [dict(feedback) for feedback in feedbacks]
    except Exception as e:
        logger.error(f"Failed to fetch feedbacks: {e}")
        return []


def get_all_feedbacks(conn) -> List[Dict[str, Any]]:
    """Fetch all feedbacks from database."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, feedback_text, created_at
                FROM feedback_sentiments
                ORDER BY created_at DESC
            """)
            feedbacks = cur.fetchall()
            return [dict(feedback) for feedback in feedbacks]
    except Exception as e:
        logger.error(f"Failed to fetch feedbacks: {e}")
        return []


def compute_intents_for_feedback(feedback_id: str) -> bool:
    """Call label-backend API to compute intents for a feedback."""
    try:
        response = requests.post(
            f"{LABEL_BACKEND_URL}/feedbacks/{feedback_id}/intents",
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"Feedback {feedback_id} not found")
        else:
            logger.error(f"HTTP error computing intents for feedback {feedback_id}: {e}")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to compute intents for feedback {feedback_id}: {e}")
        return False


def compute_feedback_intents(recompute: bool = False):
    """Main function to compute intents for feedbacks."""
    logger.info("Starting feedback intent computation...")
    
    # Connect to database
    conn = get_db_connection()
    logger.info(f"Connected to database: {DB_NAME}")
    
    # Get feedbacks to process
    if recompute:
        logger.info("Recomputing intents for all feedbacks...")
        feedbacks = get_all_feedbacks(conn)
    else:
        logger.info("Computing intents for feedbacks without cached results...")
        feedbacks = get_feedbacks_without_intents(conn)
    
    logger.info(f"Found {len(feedbacks)} feedbacks to process")
    
    if not feedbacks:
        logger.warning("No feedbacks found to process")
        conn.close()
        return
    
    # Process each feedback
    success_count = 0
    failure_count = 0
    
    for feedback in tqdm(feedbacks, desc="Computing intents"):
        feedback_id = feedback['id']
        
        # Compute intents
        if compute_intents_for_feedback(str(feedback_id)):
            success_count += 1
            logger.debug(f"Computed intents for feedback: {feedback_id}")
        else:
            failure_count += 1
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    # Close connection
    conn.close()
    
    # Summary
    logger.info("=" * 60)
    logger.info("Intent computation completed!")
    logger.info(f"Total feedbacks: {len(feedbacks)}")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Failed: {failure_count}")
    logger.info("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute intent analysis for feedbacks in the database"
    )
    parser.add_argument(
        '--recompute',
        action='store_true',
        help='Recompute intents for all feedbacks, including those with cached results'
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        compute_feedback_intents(recompute=args.recompute)
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

