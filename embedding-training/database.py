"""Database connection management with psycopg2."""
import logging
from contextlib import contextmanager
from typing import Generator
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

def init_pool():
    """Initialize connection pool."""
    global connection_pool
    try:
        connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=settings.database_url
        )
        logger.info("Database connection pool created successfully")
    except Exception as e:
        logger.error(f"Error creating connection pool: {e}", exc_info=True)
        raise

def close_pool():
    """Close connection pool."""
    global connection_pool
    if connection_pool:
        connection_pool.closeall()
        logger.info("Database connection pool closed")


@contextmanager
def get_db() -> Generator:
    """Get database connection from pool."""
    if connection_pool is None:
        init_pool()
    
    conn = None
    try:
        conn = connection_pool.getconn()
        conn.autocommit = False
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}", exc_info=True)
        raise
    finally:
        if conn:
            connection_pool.putconn(conn)

def execute_query(conn, query: str, params: tuple = None, fetch: str = "all"):
    """Execute a query and return results."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, params)
        
        if fetch == "one":
            return cur.fetchone()
        elif fetch == "all":
            return cur.fetchall()
        elif fetch == "none":
            return None
        else:
            return cur.fetchall()