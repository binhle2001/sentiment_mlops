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
    """Check database connection and ensure UUID extension and default are enabled."""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                # Enable UUID extension if not exists
                cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
                
                # Check if labels table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'labels'
                    );
                """)
                exists = cur.fetchone()[0]
                
                if exists:
                    # Check and add default for created_at
                    cur.execute("""
                        SELECT column_default 
                        FROM information_schema.columns 
                        WHERE table_name = 'labels' AND column_name = 'created_at';
                    """)
                    created_default = cur.fetchone()[0]
                    
                    if created_default is None or 'CURRENT_TIMESTAMP' not in str(created_default).upper():
                        logger.info("Adding CURRENT_TIMESTAMP default to created_at column...")
                        cur.execute("""
                            ALTER TABLE labels 
                            ALTER COLUMN created_at SET DEFAULT CURRENT_TIMESTAMP;
                        """)
                        logger.info("created_at default added successfully")
                    
                    # Check and add default for updated_at
                    cur.execute("""
                        SELECT column_default 
                        FROM information_schema.columns 
                        WHERE table_name = 'labels' AND column_name = 'updated_at';
                    """)
                    updated_default = cur.fetchone()[0]
                    
                    if updated_default is None or 'CURRENT_TIMESTAMP' not in str(updated_default).upper():
                        logger.info("Adding CURRENT_TIMESTAMP default to updated_at column...")
                        cur.execute("""
                            ALTER TABLE labels 
                            ALTER COLUMN updated_at SET DEFAULT CURRENT_TIMESTAMP;
                        """)
                        logger.info("updated_at default added successfully")
                    
                    # Ensure update trigger exists
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM pg_trigger 
                            WHERE tgname = 'update_labels_updated_at'
                        );
                    """)
                    trigger_exists = cur.fetchone()[0]
                    
                    if not trigger_exists:
                        logger.info("Creating updated_at trigger...")
                        cur.execute("""
                            CREATE OR REPLACE FUNCTION update_updated_at_column()
                            RETURNS TRIGGER AS $$
                            BEGIN
                                NEW.updated_at = CURRENT_TIMESTAMP;
                                RETURN NEW;
                            END;
                            $$ language 'plpgsql';
                            
                            CREATE TRIGGER update_labels_updated_at 
                                BEFORE UPDATE ON labels 
                                FOR EACH ROW 
                                EXECUTE FUNCTION update_updated_at_column();
                        """)
                        logger.info("Trigger created successfully")
                    
                    logger.info("Database connection OK - labels table configured correctly")
                else:
                    logger.warning("Labels table does not exist - should be created by docker-entrypoint-initdb.d")
                
                # Check if feedback_sentiments table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'feedback_sentiments'
                    );
                """)
                feedback_table_exists = cur.fetchone()[0]
                
                if not feedback_table_exists:
                    logger.info("Creating feedback_sentiments table...")
                    cur.execute("""
                        CREATE TABLE feedback_sentiments (
                            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                            feedback_text TEXT NOT NULL,
                            sentiment_label VARCHAR(50) NOT NULL,
                            confidence_score FLOAT NOT NULL,
                            feedback_source VARCHAR(50) NOT NULL,
                            is_model_confirmed BOOLEAN NOT NULL DEFAULT FALSE,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                        
                        CREATE INDEX idx_feedback_sentiments_source ON feedback_sentiments(feedback_source);
                        CREATE INDEX idx_feedback_sentiments_label ON feedback_sentiments(sentiment_label);
                        CREATE INDEX idx_feedback_sentiments_created_at ON feedback_sentiments(created_at DESC);
                        
                        COMMENT ON TABLE feedback_sentiments IS 'Customer feedback with sentiment analysis results';
                        COMMENT ON COLUMN feedback_sentiments.sentiment_label IS 'Sentiment classification: POSITIVE, NEGATIVE, NEUTRAL, EXTREMELY_NEGATIVE';
                        COMMENT ON COLUMN feedback_sentiments.feedback_source IS 'Source of feedback: web, app, map, form khảo sát, tổng đài';
                    """)
                    logger.info("feedback_sentiments table created successfully")
                else:
                    logger.info("feedback_sentiments table already exists")

                # Ensure confirmation column exists
                cur.execute("""
                    ALTER TABLE feedback_sentiments
                        ADD COLUMN IF NOT EXISTS is_model_confirmed BOOLEAN NOT NULL DEFAULT FALSE;
                """)
                cur.execute("""
                    COMMENT ON COLUMN feedback_sentiments.is_model_confirmed IS
                        'Đánh dấu feedback đã được người dùng xác nhận mô hình dự đoán đúng';
                """)
            
            conn.commit()
    except Exception as e:
        logger.error(f"Error checking database: {e}", exc_info=True)
        raise
