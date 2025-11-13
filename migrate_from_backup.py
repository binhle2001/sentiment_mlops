#!/usr/bin/env python3
"""
Rebuild the PostgreSQL database schema and data from an Excel backup.

Usage:
    python migrate_from_backup.py --backup ./scripts/backups/backup_YYYYMMDD_HHMMSS.xlsx

Steps performed:
    1. Connect to the configured PostgreSQL database.
    2. Drop existing tables related to the label system (if any).
    3. Recreate schema (labels, feedback_sentiments, feedback_intents, triggers, indexes).
    4. Import data from the provided Excel backup file.

Requirements:
    pip install pandas openpyxl psycopg2-binary python-dotenv
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

# Load environment variables if .env exists
load_dotenv()

# Database configuration (align with other scripts)
DB_HOST = "localhost"
DB_PORT = 5499
DB_NAME = "label_db"
DB_USER = "postgres"
DB_PASSWORD = "qwertyxxx"


def get_connection() -> psycopg2.extensions.connection:
    """Create a psycopg2 connection."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        conn.autocommit = False
        return conn
    except Exception as exc:  # pragma: no cover - fatal
        print(f"❌ Không thể kết nối database: {exc}", file=sys.stderr)
        print(f"   Host: {DB_HOST}:{DB_PORT}, DB: {DB_NAME}, User: {DB_USER}", file=sys.stderr)
        raise


def find_latest_backup(default_dir: Path) -> Path:
    """Find the most recent backup_xxx.xlsx in the directory."""
    if not default_dir.exists():
        raise FileNotFoundError(f"Thư mục backup không tồn tại: {default_dir}")

    backups = sorted(default_dir.glob("backup_*.xlsx"))
    if not backups:
        raise FileNotFoundError(f"Không tìm thấy file backup trong {default_dir}")
    return backups[-1]


def recreate_schema(conn: psycopg2.extensions.connection) -> None:
    """Drop existing tables and recreate schema."""
    with conn.cursor() as cur:
        print("🗄️  Đang khởi tạo schema...")

        # Enable uuid extension
        cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')

        # Drop tables in dependency order
        cur.execute("DROP TABLE IF EXISTS feedback_intents CASCADE;")
        cur.execute("DROP TABLE IF EXISTS feedback_sentiments CASCADE;")
        cur.execute("DROP TABLE IF EXISTS labels CASCADE;")

        # Recreate labels table
        cur.execute(
            """
            CREATE TABLE labels (
                id INTEGER PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                level INTEGER NOT NULL CHECK (level IN (1, 2, 3)),
                parent_id INTEGER REFERENCES labels(id) ON DELETE CASCADE,
                description TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                embedding REAL[]
            );
            """
        )

        # Indexes for labels
        cur.execute("CREATE INDEX idx_labels_parent_id ON labels(parent_id);")
        cur.execute("CREATE INDEX idx_labels_level ON labels(level);")
        cur.execute("CREATE INDEX idx_labels_name ON labels(name);")
        cur.execute("CREATE INDEX idx_labels_created_at ON labels(created_at DESC);")
        cur.execute("CREATE INDEX idx_labels_embedding ON labels USING GIN(embedding);")

        # Update trigger for updated_at
        cur.execute(
            """
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
            """
        )

        cur.execute(
            """
            CREATE TRIGGER update_labels_updated_at
                BEFORE UPDATE ON labels
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
            """
        )

        # Recreate feedback_sentiments
        cur.execute(
            """
            CREATE TABLE feedback_sentiments (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                feedback_text TEXT NOT NULL,
                sentiment_label VARCHAR(50) NOT NULL,
                confidence_score REAL NOT NULL,
                feedback_source VARCHAR(50) NOT NULL,
                is_model_confirmed BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                level1_id INTEGER REFERENCES labels(id) ON DELETE SET NULL,
                level2_id INTEGER REFERENCES labels(id) ON DELETE SET NULL,
                level3_id INTEGER REFERENCES labels(id) ON DELETE SET NULL
            );
            """
        )

        cur.execute("CREATE INDEX idx_feedback_sentiments_source ON feedback_sentiments(feedback_source);")
        cur.execute("CREATE INDEX idx_feedback_sentiments_label ON feedback_sentiments(sentiment_label);")
        cur.execute("CREATE INDEX idx_feedback_sentiments_created_at ON feedback_sentiments(created_at DESC);")
        cur.execute("CREATE INDEX idx_feedback_sentiments_level1 ON feedback_sentiments(level1_id);")
        cur.execute("CREATE INDEX idx_feedback_sentiments_level2 ON feedback_sentiments(level2_id);")
        cur.execute("CREATE INDEX idx_feedback_sentiments_level3 ON feedback_sentiments(level3_id);")

        # Recreate feedback_intents
        cur.execute(
            """
            CREATE TABLE feedback_intents (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                feedback_id UUID NOT NULL REFERENCES feedback_sentiments(id) ON DELETE CASCADE,
                level1_id INTEGER NOT NULL REFERENCES labels(id) ON DELETE CASCADE,
                level2_id INTEGER NOT NULL REFERENCES labels(id) ON DELETE CASCADE,
                level3_id INTEGER NOT NULL REFERENCES labels(id) ON DELETE CASCADE,
                avg_cosine_similarity REAL NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (feedback_id, level1_id, level2_id, level3_id)
            );
            """
        )

        cur.execute("CREATE INDEX idx_feedback_intents_feedback_id ON feedback_intents(feedback_id);")
        cur.execute("CREATE INDEX idx_feedback_intents_level1_id ON feedback_intents(level1_id);")
        cur.execute("CREATE INDEX idx_feedback_intents_level2_id ON feedback_intents(level2_id);")
        cur.execute("CREATE INDEX idx_feedback_intents_level3_id ON feedback_intents(level3_id);")
        cur.execute("CREATE INDEX idx_feedback_intents_similarity ON feedback_intents(avg_cosine_similarity DESC);")
        cur.execute("CREATE INDEX idx_feedback_intents_created_at ON feedback_intents(created_at DESC);")

        print("✅ Schema đã được tạo lại thành công.")


def read_sheet(excel_path: Path, sheet_name: str) -> Optional[pd.DataFrame]:
    """Read a sheet as DataFrame if it exists."""
    try:
        return pd.read_excel(excel_path, sheet_name=sheet_name)
    except ValueError:
        # Sheet not found
        return None


def import_labels(conn: psycopg2.extensions.connection, df: pd.DataFrame) -> int:
    """Import labels into database."""
    if df is None or df.empty:
        raise ValueError("Sheet 'labels' không có dữ liệu, không thể khôi phục.")

    required_columns = {"id", "name", "level", "parent_id", "description"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Sheet 'labels' thiếu các cột bắt buộc: {', '.join(sorted(missing))}")

    records = []
    for row in df.itertuples(index=False):
        parent_id = getattr(row, "parent_id")
        parent_id = None if pd.isna(parent_id) else int(parent_id)
        description = getattr(row, "description")
        description = None if pd.isna(description) else description
        records.append(
            (
                int(getattr(row, "id")),
                str(getattr(row, "name")),
                int(getattr(row, "level")),
                parent_id,
                description,
            )
        )

    sql = """
        INSERT INTO labels (id, name, level, parent_id, description)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE
            SET name = EXCLUDED.name,
                level = EXCLUDED.level,
                parent_id = EXCLUDED.parent_id,
                description = EXCLUDED.description
    """

    with conn.cursor() as cur:
        execute_batch(cur, sql, records, page_size=200)

    return len(records)


def import_feedback_sentiments(conn: psycopg2.extensions.connection, df: Optional[pd.DataFrame]) -> int:
    """Import feedback_sentiments if sheet exists."""
    if df is None or df.empty:
        print("ℹ️  Sheet 'feedback_sentiments' không có dữ liệu, skip.")
        return 0

    required_columns = {
        "id",
        "feedback_text",
        "sentiment_label",
        "confidence_score",
        "feedback_source",
        "level1_id",
        "level2_id",
        "level3_id",
    }
    available_columns = set(df.columns)
    missing = required_columns - available_columns
    if missing:
        raise ValueError(f"Sheet 'feedback_sentiments' thiếu cột: {', '.join(sorted(missing))}")

    # Ensure confirmation column exists
    if "is_model_confirmed" not in available_columns:
        df = df.copy()
        df["is_model_confirmed"] = False

    records = []
    for row in df.itertuples(index=False):
        def normalize(value, cast_type=None):
            if pd.isna(value):
                return None
            if cast_type:
                return cast_type(value)
            return value

        is_confirmed = getattr(row, "is_model_confirmed")
        if pd.isna(is_confirmed):
            is_confirmed = False
        else:
            is_confirmed = bool(is_confirmed)

        records.append(
            (
                str(getattr(row, "id")),
                str(getattr(row, "feedback_text")),
                str(getattr(row, "sentiment_label")),
                float(getattr(row, "confidence_score")),
                str(getattr(row, "feedback_source")),
                normalize(getattr(row, "level1_id"), int),
                normalize(getattr(row, "level2_id"), int),
                normalize(getattr(row, "level3_id"), int),
                is_confirmed,
            )
        )

    sql = """
        INSERT INTO feedback_sentiments (
            id,
            feedback_text,
            sentiment_label,
            confidence_score,
            feedback_source,
            level1_id,
            level2_id,
            level3_id,
            is_model_confirmed
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE
            SET feedback_text = EXCLUDED.feedback_text,
                sentiment_label = EXCLUDED.sentiment_label,
                confidence_score = EXCLUDED.confidence_score,
                feedback_source = EXCLUDED.feedback_source,
                level1_id = EXCLUDED.level1_id,
                level2_id = EXCLUDED.level2_id,
                level3_id = EXCLUDED.level3_id,
                is_model_confirmed = EXCLUDED.is_model_confirmed
    """

    with conn.cursor() as cur:
        execute_batch(cur, sql, records, page_size=200)

    return len(records)


def import_feedback_intents(conn: psycopg2.extensions.connection, df: Optional[pd.DataFrame]) -> int:
    """Import feedback_intents if sheet exists."""
    if df is None or df.empty:
        print("ℹ️  Sheet 'feedback_intents' không có dữ liệu, skip.")
        return 0

    required_columns = {
        "id",
        "feedback_id",
        "level1_id",
        "level2_id",
        "level3_id",
        "avg_cosine_similarity",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Sheet 'feedback_intents' thiếu cột: {', '.join(sorted(missing))}")

    records = []
    for row in df.itertuples(index=False):
        records.append(
            (
                str(getattr(row, "id")),
                str(getattr(row, "feedback_id")),
                int(getattr(row, "level1_id")),
                int(getattr(row, "level2_id")),
                int(getattr(row, "level3_id")),
                float(getattr(row, "avg_cosine_similarity")),
            )
        )

    sql = """
        INSERT INTO feedback_intents (
            id,
            feedback_id,
            level1_id,
            level2_id,
            level3_id,
            avg_cosine_similarity
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE
            SET feedback_id = EXCLUDED.feedback_id,
                level1_id = EXCLUDED.level1_id,
                level2_id = EXCLUDED.level2_id,
                level3_id = EXCLUDED.level3_id,
                avg_cosine_similarity = EXCLUDED.avg_cosine_similarity
    """

    with conn.cursor() as cur:
        execute_batch(cur, sql, records, page_size=200)

    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild database from Excel backup.")
    parser.add_argument(
        "--backup",
        type=str,
        default=None,
        help="Đường dẫn tới file backup Excel (mặc định: lấy file mới nhất trong ./scripts/backups)",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="scripts/backups",
        help="Thư mục chứa các file backup (dùng khi không chỉ định --backup)",
    )
    args = parser.parse_args()

    if args.backup:
        backup_path = Path(args.backup)
    else:
        backup_path = find_latest_backup(Path(args.backup_dir))

    if not backup_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file backup: {backup_path}")

    print("=" * 70)
    print("  🚀 DATABASE REBUILD FROM BACKUP")
    print("=" * 70)
    print(f"📁 Backup file: {backup_path.resolve()}")
    print(f"🕒 Start time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    conn = get_connection()

    try:
        recreate_schema(conn)

        print("📥 Đang đọc dữ liệu từ backup...")
        labels_df = read_sheet(backup_path, "labels")
        feedbacks_df = read_sheet(backup_path, "feedback_sentiments")
        intents_df = read_sheet(backup_path, "feedback_intents")

        print("🔄 Đang import labels...")
        label_count = import_labels(conn, labels_df)
        print(f"   ✅ Đã import {label_count} labels.")

        print("🔄 Đang import feedback_sentiments...")
        feedback_count = import_feedback_sentiments(conn, feedbacks_df)
        print(f"   ✅ Đã import {feedback_count} feedback sentiments.")

        print("🔄 Đang import feedback_intents...")
        intent_count = import_feedback_intents(conn, intents_df)
        print(f"   ✅ Đã import {intent_count} feedback intents.")

        conn.commit()
        print("\n🎉 Hoàn tất khôi phục database!")
    except Exception as exc:
        conn.rollback()
        print(f"\n❌ Lỗi trong quá trình khôi phục: {exc}", file=sys.stderr)
        raise
    finally:
        conn.close()
        print("🔚 Đã đóng kết nối database.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⛔ Đã hủy bởi người dùng.")
        sys.exit(1)
    except Exception:
        sys.exit(1)

