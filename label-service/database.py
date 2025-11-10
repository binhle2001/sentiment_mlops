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
    """Initialize database tables."""
    create_table_sql = """
    -- Enable UUID extension
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    
    CREATE TABLE IF NOT EXISTS labels (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        name VARCHAR(255) NOT NULL,
        level INTEGER NOT NULL CHECK (level IN (1, 2, 3)),
        description TEXT,
        parent_id UUID REFERENCES labels(id) ON DELETE CASCADE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        
        -- Constraints
        CONSTRAINT unique_name_per_parent UNIQUE (name, parent_id),
        CONSTRAINT check_valid_level CHECK (level IN (1, 2, 3)),
        CONSTRAINT check_level_1_no_parent CHECK (
            (level = 1 AND parent_id IS NULL) OR (level > 1)
        ),
        CONSTRAINT check_level_2_3_has_parent CHECK (
            (level = 1) OR (level > 1 AND parent_id IS NOT NULL)
        )
    );
    
    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_labels_id ON labels(id);
    CREATE INDEX IF NOT EXISTS idx_labels_name ON labels(name);
    CREATE INDEX IF NOT EXISTS idx_labels_level ON labels(level);
    CREATE INDEX IF NOT EXISTS idx_labels_parent_id ON labels(parent_id);
    CREATE INDEX IF NOT EXISTS idx_labels_parent_level ON labels(parent_id, level);
    CREATE INDEX IF NOT EXISTS idx_labels_created_at ON labels(created_at);
    
    -- Create trigger for updated_at
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = CURRENT_TIMESTAMP;
        RETURN NEW;
    END;
    $$ language 'plpgsql';
    
    DROP TRIGGER IF EXISTS update_labels_updated_at ON labels;
    CREATE TRIGGER update_labels_updated_at
        BEFORE UPDATE ON labels
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """
    
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
            conn.commit()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        raise
