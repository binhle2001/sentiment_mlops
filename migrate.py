#!/usr/bin/env python3
"""
Database migration script for Intent Analysis feature.
Adds embedding column to labels and creates feedback_intents table.

Usage:
    python migrate.py
"""
import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'label_db')
DB_USER = os.getenv('POSTGRES_USER', 'labeluser')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'labelpass123')


def get_connection():
    """Create database connection."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        return conn
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        print(f"   Host: {DB_HOST}:{DB_PORT}")
        print(f"   Database: {DB_NAME}")
        print(f"   User: {DB_USER}")
        sys.exit(1)


def run_migration(conn):
    """Run database migration."""
    cursor = conn.cursor()
    
    print("=" * 70)
    print("  DATABASE MIGRATION - Intent Analysis Feature")
    print("=" * 70)
    print()
    
    try:
        # Step 1: Add embedding column to labels table
        print("📝 Step 1: Adding embedding column to labels table...")
        cursor.execute("""
            ALTER TABLE labels 
            ADD COLUMN IF NOT EXISTS embedding REAL[]
        """)
        print("   ✅ Column 'embedding' added to labels table")
        
        # Step 2: Create index on embedding column
        print("\n📝 Step 2: Creating index on embedding column...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_labels_embedding 
            ON labels USING GIN(embedding)
        """)
        print("   ✅ Index 'idx_labels_embedding' created")
        
        # Step 3: Create feedback_intents table
        print("\n📝 Step 3: Creating feedback_intents table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_intents (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                feedback_id UUID NOT NULL,
                level1_id UUID NOT NULL,
                level2_id UUID NOT NULL,
                level3_id UUID NOT NULL,
                avg_cosine_similarity REAL NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                
                -- Foreign key constraints
                CONSTRAINT fk_feedback_intents_feedback 
                    FOREIGN KEY (feedback_id) 
                    REFERENCES feedback_sentiments(id) 
                    ON DELETE CASCADE,
                
                CONSTRAINT fk_feedback_intents_level1 
                    FOREIGN KEY (level1_id) 
                    REFERENCES labels(id) 
                    ON DELETE CASCADE,
                
                CONSTRAINT fk_feedback_intents_level2 
                    FOREIGN KEY (level2_id) 
                    REFERENCES labels(id) 
                    ON DELETE CASCADE,
                
                CONSTRAINT fk_feedback_intents_level3 
                    FOREIGN KEY (level3_id) 
                    REFERENCES labels(id) 
                    ON DELETE CASCADE,
                
                -- Ensure each feedback-intent triplet is unique
                CONSTRAINT unique_feedback_intent_triplet 
                    UNIQUE (feedback_id, level1_id, level2_id, level3_id)
            )
        """)
        print("   ✅ Table 'feedback_intents' created")
        
        # Step 4: Create indexes on feedback_intents
        print("\n📝 Step 4: Creating indexes on feedback_intents...")
        
        indexes = [
            ("idx_feedback_intents_feedback_id", "feedback_id"),
            ("idx_feedback_intents_level1_id", "level1_id"),
            ("idx_feedback_intents_level2_id", "level2_id"),
            ("idx_feedback_intents_level3_id", "level3_id"),
            ("idx_feedback_intents_similarity", "avg_cosine_similarity DESC"),
            ("idx_feedback_intents_created_at", "created_at DESC"),
        ]
        
        for index_name, column in indexes:
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {index_name} 
                ON feedback_intents({column})
            """)
            print(f"   ✅ Index '{index_name}' created")
        
        # Step 5: Add comments
        print("\n📝 Step 5: Adding documentation comments...")
        cursor.execute("""
            COMMENT ON COLUMN labels.embedding IS 
            'BGE-M3 embedding vector (1024 dimensions) for similarity computation'
        """)
        cursor.execute("""
            COMMENT ON TABLE feedback_intents IS 
            'Stores top intent triplets for each feedback with cosine similarity scores'
        """)
        cursor.execute("""
            COMMENT ON COLUMN feedback_intents.avg_cosine_similarity IS 
            'Average cosine similarity across level1, level2, and level3 embeddings'
        """)
        print("   ✅ Comments added")
        
        print()
        print("=" * 70)
        print("  ✅ MIGRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print()
        
        # Verify migration
        print("📊 Verification:")
        
        # Check labels table has embedding column
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'labels' AND column_name = 'embedding'
        """)
        if cursor.fetchone():
            print("   ✅ labels.embedding column exists")
        else:
            print("   ⚠️  labels.embedding column NOT found!")
        
        # Check feedback_intents table exists
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name = 'feedback_intents'
        """)
        if cursor.fetchone():
            print("   ✅ feedback_intents table exists")
        else:
            print("   ⚠️  feedback_intents table NOT found!")
        
        # Count labels with embeddings
        cursor.execute("SELECT COUNT(*) FROM labels WHERE embedding IS NOT NULL")
        count = cursor.fetchone()[0]
        print(f"   📊 Labels with embeddings: {count}")
        
        # Count cached intents
        cursor.execute("SELECT COUNT(DISTINCT feedback_id) FROM feedback_intents")
        count = cursor.fetchone()[0]
        print(f"   📊 Feedbacks with cached intents: {count}")
        
        print()
        print("🎯 Next steps:")
        print("   1. Run: python seed_data.py --labels-only")
        print("   2. Run: python seed_data.py --intents-only")
        print()
        
    except psycopg2.Error as e:
        print(f"\n❌ Migration failed: {e}")
        print(f"   Error code: {e.pgcode}")
        print(f"   Error message: {e.pgerror}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cursor.close()


def check_prerequisites(conn):
    """Check if prerequisite tables exist."""
    cursor = conn.cursor()
    
    print("🔍 Checking prerequisites...")
    
    # Check if labels table exists
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_name = 'labels'
    """)
    if not cursor.fetchone():
        print("   ❌ Table 'labels' does not exist!")
        print("   Please run the initial migration first (01-init.sql)")
        cursor.close()
        return False
    print("   ✅ Table 'labels' exists")
    
    # Check if feedback_sentiments table exists
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_name = 'feedback_sentiments'
    """)
    if not cursor.fetchone():
        print("   ❌ Table 'feedback_sentiments' does not exist!")
        print("   Please create the feedback_sentiments table first")
        cursor.close()
        return False
    print("   ✅ Table 'feedback_sentiments' exists")
    
    cursor.close()
    return True


def main():
    """Main function."""
    print()
    print("🚀 Starting database migration...")
    print(f"   Target database: {DB_NAME}@{DB_HOST}:{DB_PORT}")
    print()
    
    # Connect to database
    conn = get_connection()
    print("✅ Connected to database successfully")
    print()
    
    # Check prerequisites
    if not check_prerequisites(conn):
        conn.close()
        sys.exit(1)
    
    print()
    
    # Run migration
    run_migration(conn)
    
    # Close connection
    conn.close()
    print("✅ Database connection closed")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

