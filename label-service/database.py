"""Database connection management with psycopg2."""
import logging
from contextlib import contextmanager
from typing import Generator
from uuid import UUID
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, register_uuid

from config import get_settings

# Register UUID adapter for psycopg2
register_uuid()

logger = logging.getLogger(__name__)
settings = get_settings()

# Parse database URL
# Format: postgresql+asyncpg://user:pass@host:port/dbname
# Convert to psycopg2 format: dbname=... user=... password=... host=... port=...
def parse_db_url(url: str) -> str:
    """Parse database URL to psycopg2 connection string."""
    # Remove postgresql+asyncpg:// or postgresql://
    url = url.replace("postgresql+asyncpg://", "").replace("postgresql://", "")
    
    # Parse user:password@host:port/dbname
    if "@" in url:
        user_pass, host_db = url.split("@", 1)
        if ":" in user_pass:
            user, password = user_pass.split(":", 1)
        else:
            user = user_pass
            password = ""
    else:
        return url
    
    if "/" in host_db:
        host_port, dbname = host_db.split("/", 1)
    else:
        host_port = host_db
        dbname = "postgres"
    
    if ":" in host_port:
        host, port = host_port.split(":", 1)
    else:
        host = host_port
        port = "5432"
    
    conn_str = f"dbname={dbname} user={user} password={password} host={host} port={port}"
    return conn_str


# Create connection pool
connection_pool = None

def init_pool():
    """Initialize connection pool."""
    global connection_pool
    try:
        conn_str = parse_db_url(settings.database_url)
        connection_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=conn_str
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


def init_db():
    """Check database connection and verify tables exist."""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                # Just check if labels table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'labels'
                    );
                """)
                exists = cur.fetchone()[0]
                if exists:
                    logger.info("Database connection OK - labels table exists")
                else:
                    logger.warning("Labels table does not exist - should be created by docker-entrypoint-initdb.d")
    except Exception as e:
        logger.error(f"Error checking database: {e}", exc_info=True)
        raise
