#!/usr/bin/env python3
"""
Database migration V2 for Intent Analysis - Gemini Integration.
Adds level1_id, level2_id, level3_id to feedback_sentiments.
Drops feedback_intents table (no longer needed).

Usage:
    python migrate_v2.py
"""
import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_HOST = 'localhost'
DB_PORT = 5499
DB_NAME = 'label_db'
DB_USER =  'postgres'
DB_PASSWORD = 'qwertyxxx'


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
        sys.exit(1)


def run_migration(conn):
    """Run database migration V2."""
    cursor = conn.cursor()
    
    print("=" * 70)
    print("  DATABASE MIGRATION V2 - Gemini Integration")
    print("=" * 70)
    print()
    
    try:
        # Step 1: Add label columns to feedback_sentiments
        print("📝 Step 1: Adding label columns to feedback_sentiments...")
        
        cursor.execute("""
            ALTER TABLE feedback_sentiments 
            ADD COLUMN IF NOT EXISTS level1_id UUID
        """)
        print("   ✅ Column 'level1_id' added")
        
        cursor.execute("""
            ALTER TABLE feedback_sentiments 
            ADD COLUMN IF NOT EXISTS level2_id UUID
        """)
        print("   ✅ Column 'level2_id' added")
        
        cursor.execute("""
            ALTER TABLE feedback_sentiments 
            ADD COLUMN IF NOT EXISTS level3_id UUID
        """)
        print("   ✅ Column 'level3_id' added")
        
        # Step 2: Add foreign key constraints
        print("\n📝 Step 2: Adding foreign key constraints...")
        
        # Check if constraints already exist
        cursor.execute("""
            SELECT constraint_name 
            FROM information_schema.table_constraints 
            WHERE table_name = 'feedback_sentiments' 
            AND constraint_name = 'fk_feedback_level1'
        """)
        
        if not cursor.fetchone():
            cursor.execute("""
                ALTER TABLE feedback_sentiments
                ADD CONSTRAINT fk_feedback_level1 
                FOREIGN KEY (level1_id) REFERENCES labels(id) ON DELETE SET NULL
            """)
            print("   ✅ Constraint 'fk_feedback_level1' added")
        else:
            print("   ⏭️  Constraint 'fk_feedback_level1' already exists")
        
        cursor.execute("""
            SELECT constraint_name 
            FROM information_schema.table_constraints 
            WHERE table_name = 'feedback_sentiments' 
            AND constraint_name = 'fk_feedback_level2'
        """)
        
        if not cursor.fetchone():
            cursor.execute("""
                ALTER TABLE feedback_sentiments
                ADD CONSTRAINT fk_feedback_level2 
                FOREIGN KEY (level2_id) REFERENCES labels(id) ON DELETE SET NULL
            """)
            print("   ✅ Constraint 'fk_feedback_level2' added")
        else:
            print("   ⏭️  Constraint 'fk_feedback_level2' already exists")
        
        cursor.execute("""
            SELECT constraint_name 
            FROM information_schema.table_constraints 
            WHERE table_name = 'feedback_sentiments' 
            AND constraint_name = 'fk_feedback_level3'
        """)
        
        if not cursor.fetchone():
            cursor.execute("""
                ALTER TABLE feedback_sentiments
                ADD CONSTRAINT fk_feedback_level3 
                FOREIGN KEY (level3_id) REFERENCES labels(id) ON DELETE SET NULL
            """)
            print("   ✅ Constraint 'fk_feedback_level3' added")
        else:
            print("   ⏭️  Constraint 'fk_feedback_level3' already exists")
        
        # Step 3: Create indexes
        print("\n📝 Step 3: Creating indexes...")
        
        indexes = [
            "idx_feedback_sentiments_level1_id",
            "idx_feedback_sentiments_level2_id", 
            "idx_feedback_sentiments_level3_id"
        ]
        
        for idx in indexes:
            col = idx.replace("idx_feedback_sentiments_", "")
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {idx} 
                ON feedback_sentiments({col})
            """)
            print(f"   ✅ Index '{idx}' created")
        
        # Step 4: Drop feedback_intents table (no longer needed)
        print("\n📝 Step 4: Dropping feedback_intents table (deprecated)...")
        cursor.execute("""
            DROP TABLE IF EXISTS feedback_intents CASCADE
        """)
        print("   ✅ Table 'feedback_intents' dropped")
        
        # Step 5: Add comments
        print("\n📝 Step 5: Adding documentation comments...")
        cursor.execute("""
            COMMENT ON COLUMN feedback_sentiments.level1_id IS 
            'Level 1 label assigned by Gemini based on intent analysis'
        """)
        cursor.execute("""
            COMMENT ON COLUMN feedback_sentiments.level2_id IS 
            'Level 2 label assigned by Gemini based on intent analysis'
        """)
        cursor.execute("""
            COMMENT ON COLUMN feedback_sentiments.level3_id IS 
            'Level 3 label assigned by Gemini based on intent analysis'
        """)
        print("   ✅ Comments added")
        
        print()
        print("=" * 70)
        print("  ✅ MIGRATION V2 COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print()
        
        # Verify migration
        print("📊 Verification:")
        
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'feedback_sentiments' 
            AND column_name IN ('level1_id', 'level2_id', 'level3_id')
        """)
        cols = cursor.fetchall()
        if len(cols) == 3:
            print(f"   ✅ All 3 label columns exist in feedback_sentiments")
        else:
            print(f"   ⚠️  Only {len(cols)}/3 columns found!")
        
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name = 'feedback_intents'
        """)
        if not cursor.fetchone():
            print("   ✅ feedback_intents table removed")
        else:
            print("   ⚠️  feedback_intents table still exists!")
        
        cursor.execute("""
            SELECT COUNT(*) FROM feedback_sentiments 
            WHERE level1_id IS NOT NULL
        """)
        count = cursor.fetchone()[0]
        print(f"   📊 Feedbacks with labels: {count}")
        
        print()
        print("🎯 Next steps:")
        print("   1. Set GEMINI_API_KEY in .env")
        print("   2. Rebuild services: docker-compose up -d --build")
        print("   3. Test with new feedback submission")
        print()
        
    except psycopg2.Error as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cursor.close()


def main():
    """Main function."""
    print()
    print("🚀 Starting database migration V2...")
    print(f"   Target database: {DB_NAME}@{DB_HOST}:{DB_PORT}")
    print()
    
    conn = get_connection()
    print("✅ Connected to database successfully")
    print()
    
    run_migration(conn)
    
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

