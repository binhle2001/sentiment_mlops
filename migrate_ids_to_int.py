#!/usr/bin/env python3
"""
Migration script to convert label identifiers from UUID to INTEGER across the schema.

Steps performed:
1. Add temporary INTEGER columns to hold the new identifiers.
2. Generate a deterministic mapping from existing UUIDs to sequential integers (starting at 1).
3. Update `labels`, `feedback_sentiments`, and (if present) `feedback_intents` with the new identifiers.
4. Swap the UUID columns with the new INTEGER columns and rebuild constraints/indexes.

Usage:
    python migrate_ids_to_int.py

Environment variables (optional, defaults shown in parentheses):
    POSTGRES_HOST (localhost)
    POSTGRES_PORT (5432)
    POSTGRES_DB   (label_db)
    POSTGRES_USER (postgres)
    POSTGRES_PASSWORD (password)
"""
from __future__ import annotations

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB", "label_db")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")


def get_connection() -> psycopg2.extensions.connection:
    """Create a PostgreSQL connection."""
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
    except Exception as exc:  # pragma: no cover - fatal error
        logger.error("Failed to connect to database: %s", exc)
        logger.error("Host=%s Port=%s DB=%s User=%s", DB_HOST, DB_PORT, DB_NAME, DB_USER)
        raise


def table_exists(conn: psycopg2.extensions.connection, table_name: str) -> bool:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_name = %s
            ) AS exists
            """,
            (table_name,),
        )
        row = cur.fetchone()
        return bool(row and row["exists"])


def get_column_data_type(
    conn: psycopg2.extensions.connection, table_name: str, column_name: str
) -> Optional[str]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT data_type
            FROM information_schema.columns
            WHERE table_name = %s AND column_name = %s
            """,
            (table_name, column_name),
        )
        row = cur.fetchone()
        return row["data_type"] if row else None


def ensure_new_columns(
    conn: psycopg2.extensions.connection, *, has_feedback_intents: bool
) -> None:
    """Add helper INTEGER columns used during migration."""
    logger.info("Adding temporary INTEGER columns (if missing)...")
    statements = [
        "ALTER TABLE labels ADD COLUMN IF NOT EXISTS id_int INTEGER",
        "ALTER TABLE labels ADD COLUMN IF NOT EXISTS parent_id_int INTEGER",
        "ALTER TABLE feedback_sentiments ADD COLUMN IF NOT EXISTS level1_id_int INTEGER",
        "ALTER TABLE feedback_sentiments ADD COLUMN IF NOT EXISTS level2_id_int INTEGER",
        "ALTER TABLE feedback_sentiments ADD COLUMN IF NOT EXISTS level3_id_int INTEGER",
    ]

    if has_feedback_intents:
        statements.extend(
            [
                "ALTER TABLE feedback_intents ADD COLUMN IF NOT EXISTS level1_id_int INTEGER",
                "ALTER TABLE feedback_intents ADD COLUMN IF NOT EXISTS level2_id_int INTEGER",
                "ALTER TABLE feedback_intents ADD COLUMN IF NOT EXISTS level3_id_int INTEGER",
            ]
        )

    with conn.cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)


def build_label_mapping(
    conn: psycopg2.extensions.connection,
) -> Tuple[Dict[str, int], List[Dict[str, Optional[str]]]]:
    """Return mapping from UUID -> new int and raw rows for further processing."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                id::text AS old_id,
                name,
                level,
                parent_id::text AS parent_uuid,
                created_at
            FROM labels
            ORDER BY level ASC, parent_uuid NULLS FIRST, name ASC, old_id ASC
            """
        )
        rows = cur.fetchall()

    if not rows:
        raise RuntimeError("No labels found. Cannot build mapping.")

    mapping: Dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        mapping[row["old_id"]] = index

    return mapping, rows


def update_labels_with_mapping(
    conn: psycopg2.extensions.connection,
    mapping: Dict[str, int],
    raw_rows: List[Dict[str, Optional[str]]],
) -> None:
    """Populate id_int and parent_id_int using generated mapping."""
    logger.info("Updating labels with new INTEGER ids...")

    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE labels SET id_int = %s WHERE id = %s",
            [(new_id, old_id) for old_id, new_id in mapping.items()],
        )

        parent_updates: List[Tuple[Optional[int], str]] = []
        for row in raw_rows:
            parent_uuid = row["parent_uuid"]
            parent_int = mapping.get(parent_uuid) if parent_uuid else None
            parent_updates.append((parent_int, row["old_id"]))

        cur.executemany(
            "UPDATE labels SET parent_id_int = %s WHERE id = %s",
            parent_updates,
        )


def update_feedback_sentiments(
    conn: psycopg2.extensions.connection,
    mapping: Dict[str, int],
) -> None:
    """Roll new label mapping into feedback_sentiments."""
    logger.info("Updating feedback_sentiments with INTEGER label ids...")
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id::text AS feedback_id,
                   level1_id::text AS level1_uuid,
                   level2_id::text AS level2_uuid,
                   level3_id::text AS level3_uuid
            FROM feedback_sentiments
            """
        )
        rows = cur.fetchall()

    if not rows:
        return

    updates: List[Tuple[Optional[int], Optional[int], Optional[int], str]] = []
    missing: set[str] = set()

    for row in rows:
        level1 = mapping.get(row["level1_uuid"])
        level2 = mapping.get(row["level2_uuid"])
        level3 = mapping.get(row["level3_uuid"])

        for key in ("level1_uuid", "level2_uuid", "level3_uuid"):
            uuid_value = row[key]
            if uuid_value and uuid_value not in mapping:
                missing.add(uuid_value)

        updates.append((level1, level2, level3, row["feedback_id"]))

    if missing:
        logger.warning(
            "Found %d label ids referenced in feedback_sentiments without mapping; they will be set to NULL.",
            len(missing),
        )

    with conn.cursor() as cur:
        cur.executemany(
            """
            UPDATE feedback_sentiments
            SET level1_id_int = %s,
                level2_id_int = %s,
                level3_id_int = %s
            WHERE id = %s::uuid
            """,
            updates,
        )


def update_feedback_intents(
    conn: psycopg2.extensions.connection,
    mapping: Dict[str, int],
) -> None:
    """Update feedback_intents (if present) to use INTEGER label ids."""
    if not table_exists(conn, "feedback_intents"):
        return

    logger.info("Updating feedback_intents with INTEGER label ids...")
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id::text AS intent_id,
                   level1_id::text AS level1_uuid,
                   level2_id::text AS level2_uuid,
                   level3_id::text AS level3_uuid
            FROM feedback_intents
            """
        )
        rows = cur.fetchall()

    if not rows:
        return

    updates: List[Tuple[int, int, int, str]] = []
    missing: set[str] = set()

    for row in rows:
        level1_uuid = row["level1_uuid"]
        level2_uuid = row["level2_uuid"]
        level3_uuid = row["level3_uuid"]

        try:
            level1 = mapping[level1_uuid]
            level2 = mapping[level2_uuid]
            level3 = mapping[level3_uuid]
        except KeyError as exc:
            missing.add(str(exc))
            continue

        updates.append((level1, level2, level3, row["intent_id"]))

    if missing:
        logger.warning(
            "Skipping %d feedback_intents rows due to missing labels. Consider cleaning manually.",
            len(missing),
        )

    if not updates:
        return

    with conn.cursor() as cur:
        cur.executemany(
            """
            UPDATE feedback_intents
            SET level1_id_int = %s,
                level2_id_int = %s,
                level3_id_int = %s
            WHERE id = %s::uuid
            """,
            updates,
        )


def drop_constraints_and_indexes(
    conn: psycopg2.extensions.connection, *, has_feedback_intents: bool
) -> None:
    """Drop constraints and indexes tied to UUID columns before the swap."""
    logger.info("Dropping constraints and indexes referencing UUID columns...")
    drop_statements = [
        "ALTER TABLE feedback_sentiments DROP CONSTRAINT IF EXISTS fk_feedback_level1",
        "ALTER TABLE feedback_sentiments DROP CONSTRAINT IF EXISTS fk_feedback_level2",
        "ALTER TABLE feedback_sentiments DROP CONSTRAINT IF EXISTS fk_feedback_level3",
        "ALTER TABLE labels DROP CONSTRAINT IF EXISTS fk_parent",
        "ALTER TABLE labels DROP CONSTRAINT IF EXISTS unique_name_per_parent",
        "ALTER TABLE labels DROP CONSTRAINT IF EXISTS check_level_1_no_parent",
        "ALTER TABLE labels DROP CONSTRAINT IF EXISTS check_level_2_3_has_parent",
        "ALTER TABLE labels DROP CONSTRAINT IF EXISTS labels_pkey",
        "DROP INDEX IF EXISTS idx_labels_parent_id",
        "DROP INDEX IF EXISTS idx_labels_level",
        "DROP INDEX IF EXISTS idx_labels_name",
        "DROP INDEX IF EXISTS idx_labels_created_at",
        "DROP INDEX IF EXISTS idx_feedback_sentiments_level1_id",
        "DROP INDEX IF EXISTS idx_feedback_sentiments_level2_id",
        "DROP INDEX IF EXISTS idx_feedback_sentiments_level3_id",
    ]

    if has_feedback_intents:
        drop_statements.extend(
            [
                "ALTER TABLE feedback_intents DROP CONSTRAINT IF EXISTS fk_feedback_intents_level1",
                "ALTER TABLE feedback_intents DROP CONSTRAINT IF EXISTS fk_feedback_intents_level2",
                "ALTER TABLE feedback_intents DROP CONSTRAINT IF EXISTS fk_feedback_intents_level3",
                "ALTER TABLE feedback_intents DROP CONSTRAINT IF EXISTS unique_feedback_intent_triplet",
                "DROP INDEX IF EXISTS idx_feedback_intents_feedback_id",
                "DROP INDEX IF EXISTS idx_feedback_intents_level1_id",
                "DROP INDEX IF EXISTS idx_feedback_intents_level2_id",
                "DROP INDEX IF EXISTS idx_feedback_intents_level3_id",
                "DROP INDEX IF EXISTS idx_feedback_intents_similarity",
                "DROP INDEX IF EXISTS idx_feedback_intents_created_at",
            ]
        )

    with conn.cursor() as cur:
        for stmt in drop_statements:
            cur.execute(stmt)


def swap_columns(
    conn: psycopg2.extensions.connection, *, has_feedback_intents: bool
) -> None:
    """Rename integer columns into place and drop legacy UUID columns."""
    logger.info("Swapping UUID columns with INTEGER columns...")
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE labels RENAME COLUMN id TO id_uuid")
        cur.execute("ALTER TABLE labels RENAME COLUMN id_int TO id")
        cur.execute("ALTER TABLE labels RENAME COLUMN parent_id TO parent_id_uuid")
        cur.execute("ALTER TABLE labels RENAME COLUMN parent_id_int TO parent_id")
        cur.execute("ALTER TABLE labels DROP COLUMN IF EXISTS id_uuid")
        cur.execute("ALTER TABLE labels DROP COLUMN IF EXISTS parent_id_uuid")

        cur.execute("ALTER TABLE feedback_sentiments RENAME COLUMN level1_id TO level1_id_uuid")
        cur.execute("ALTER TABLE feedback_sentiments RENAME COLUMN level1_id_int TO level1_id")
        cur.execute("ALTER TABLE feedback_sentiments RENAME COLUMN level2_id TO level2_id_uuid")
        cur.execute("ALTER TABLE feedback_sentiments RENAME COLUMN level2_id_int TO level2_id")
        cur.execute("ALTER TABLE feedback_sentiments RENAME COLUMN level3_id TO level3_id_uuid")
        cur.execute("ALTER TABLE feedback_sentiments RENAME COLUMN level3_id_int TO level3_id")
        cur.execute("ALTER TABLE feedback_sentiments DROP COLUMN IF EXISTS level1_id_uuid")
        cur.execute("ALTER TABLE feedback_sentiments DROP COLUMN IF EXISTS level2_id_uuid")
        cur.execute("ALTER TABLE feedback_sentiments DROP COLUMN IF EXISTS level3_id_uuid")

        if has_feedback_intents:
            cur.execute("ALTER TABLE feedback_intents RENAME COLUMN level1_id TO level1_id_uuid")
            cur.execute("ALTER TABLE feedback_intents RENAME COLUMN level1_id_int TO level1_id")
            cur.execute("ALTER TABLE feedback_intents RENAME COLUMN level2_id TO level2_id_uuid")
            cur.execute("ALTER TABLE feedback_intents RENAME COLUMN level2_id_int TO level2_id")
            cur.execute("ALTER TABLE feedback_intents RENAME COLUMN level3_id TO level3_id_uuid")
            cur.execute("ALTER TABLE feedback_intents RENAME COLUMN level3_id_int TO level3_id")
            cur.execute("ALTER TABLE feedback_intents DROP COLUMN IF EXISTS level1_id_uuid")
            cur.execute("ALTER TABLE feedback_intents DROP COLUMN IF EXISTS level2_id_uuid")
            cur.execute("ALTER TABLE feedback_intents DROP COLUMN IF EXISTS level3_id_uuid")


def recreate_constraints_and_indexes(
    conn: psycopg2.extensions.connection, *, has_feedback_intents: bool
) -> None:
    """Recreate constraints and indexes for the new INTEGER schema."""
    logger.info("Re-creating constraints and indexes on INTEGER columns...")
    statements = [
        "ALTER TABLE labels ADD CONSTRAINT labels_pkey PRIMARY KEY (id)",
        "ALTER TABLE labels ADD CONSTRAINT fk_parent FOREIGN KEY (parent_id) REFERENCES labels(id) ON DELETE CASCADE",
        "ALTER TABLE labels ADD CONSTRAINT unique_name_per_parent UNIQUE (name, parent_id)",
        "ALTER TABLE labels ADD CONSTRAINT check_level_1_no_parent CHECK ((level = 1 AND parent_id IS NULL) OR (level > 1))",
        "ALTER TABLE labels ADD CONSTRAINT check_level_2_3_has_parent CHECK ((level = 1) OR (level > 1 AND parent_id IS NOT NULL))",
        "CREATE INDEX idx_labels_parent_id ON labels(parent_id)",
        "CREATE INDEX idx_labels_level ON labels(level)",
        "CREATE INDEX idx_labels_name ON labels(name)",
        "CREATE INDEX idx_labels_created_at ON labels(created_at DESC)",
        "ALTER TABLE feedback_sentiments ADD CONSTRAINT fk_feedback_level1 FOREIGN KEY (level1_id) REFERENCES labels(id) ON DELETE SET NULL",
        "ALTER TABLE feedback_sentiments ADD CONSTRAINT fk_feedback_level2 FOREIGN KEY (level2_id) REFERENCES labels(id) ON DELETE SET NULL",
        "ALTER TABLE feedback_sentiments ADD CONSTRAINT fk_feedback_level3 FOREIGN KEY (level3_id) REFERENCES labels(id) ON DELETE SET NULL",
        "CREATE INDEX idx_feedback_sentiments_level1_id ON feedback_sentiments(level1_id)",
        "CREATE INDEX idx_feedback_sentiments_level2_id ON feedback_sentiments(level2_id)",
        "CREATE INDEX idx_feedback_sentiments_level3_id ON feedback_sentiments(level3_id)",
    ]

    if has_feedback_intents:
        statements.extend(
            [
                "ALTER TABLE feedback_intents ADD CONSTRAINT fk_feedback_intents_level1 FOREIGN KEY (level1_id) REFERENCES labels(id) ON DELETE CASCADE",
                "ALTER TABLE feedback_intents ADD CONSTRAINT fk_feedback_intents_level2 FOREIGN KEY (level2_id) REFERENCES labels(id) ON DELETE CASCADE",
                "ALTER TABLE feedback_intents ADD CONSTRAINT fk_feedback_intents_level3 FOREIGN KEY (level3_id) REFERENCES labels(id) ON DELETE CASCADE",
                "ALTER TABLE feedback_intents ADD CONSTRAINT unique_feedback_intent_triplet UNIQUE (feedback_id, level1_id, level2_id, level3_id)",
                "CREATE INDEX idx_feedback_intents_feedback_id ON feedback_intents(feedback_id)",
                "CREATE INDEX idx_feedback_intents_level1_id ON feedback_intents(level1_id)",
                "CREATE INDEX idx_feedback_intents_level2_id ON feedback_intents(level2_id)",
                "CREATE INDEX idx_feedback_intents_level3_id ON feedback_intents(level3_id)",
                "CREATE INDEX idx_feedback_intents_similarity ON feedback_intents(avg_cosine_similarity DESC)",
                "CREATE INDEX idx_feedback_intents_created_at ON feedback_intents(created_at DESC)",
            ]
        )

    with conn.cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)


def run_migration(conn: psycopg2.extensions.connection) -> None:
    logger.info("Checking current schema state...")
    current_type = get_column_data_type(conn, "labels", "id")
    if current_type and current_type.lower() in {"integer", "bigint", "smallint"}:
        logger.info("labels.id is already INTEGER. Nothing to do.")
        return

    if current_type is None:
        raise RuntimeError("labels.id column not found. Aborting migration.")

    has_feedback_intents = table_exists(conn, "feedback_intents")

    logger.info("Detected labels.id type: %s. Starting migration...", current_type)
    ensure_new_columns(conn, has_feedback_intents=has_feedback_intents)

    mapping, raw_rows = build_label_mapping(conn)
    update_labels_with_mapping(conn, mapping, raw_rows)
    update_feedback_sentiments(conn, mapping)
    update_feedback_intents(conn, mapping)

    drop_constraints_and_indexes(conn, has_feedback_intents=has_feedback_intents)
    swap_columns(conn, has_feedback_intents=has_feedback_intents)
    recreate_constraints_and_indexes(conn, has_feedback_intents=has_feedback_intents)

    logger.info("Migration completed successfully. Total labels migrated: %d", len(mapping))


def main() -> None:
    logger.info(
        "Starting UUID -> INTEGER migration for labels (DB=%s, host=%s:%s)",
        DB_NAME,
        DB_HOST,
        DB_PORT,
    )
    conn = get_connection()
    try:
        run_migration(conn)
        conn.commit()
        logger.info("Changes committed.")
    except Exception as exc:  # pragma: no cover - migration failure
        conn.rollback()
        logger.exception("Migration failed. Rolled back all changes.")
        raise SystemExit(1) from exc
    finally:
        conn.close()
        logger.info("Connection closed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover - interactive cancel
        logger.warning("Migration cancelled by user.")
        sys.exit(1)
