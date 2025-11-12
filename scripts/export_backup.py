#!/usr/bin/env python3
"""
Export toàn bộ dữ liệu hệ thống nhãn/feedback ra Excel.

Sinh file Excel với các sheet:
  - labels: bao gồm id tuần tự (bắt đầu 1), parent_id tuần tự, kèm id gốc.
  - feedback_sentiments: có mapping sang id tuần tự cho level1/2/3.
  - feedback_intents (nếu tồn tại): tương tự feedback_sentiments.

Usage:
    python scripts/export_backup.py [--output ./backups]

Yêu cầu:
    pip install -r scripts/requirements.txt
    (đã bao gồm pandas, openpyxl, psycopg2-binary, python-dotenv)
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB", "label_db")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")


def get_connection() -> psycopg2.extensions.connection:
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        return conn
    except Exception as exc:  # pragma: no cover - fatal
        print(f"❌ Không thể kết nối database: {exc}", file=sys.stderr)
        raise


def table_exists(conn: psycopg2.extensions.connection, table: str) -> bool:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            ) AS exists
            """,
            (table,),
        )
        row = cur.fetchone()
        return bool(row and row["exists"])


def load_labels(conn: psycopg2.extensions.connection) -> pd.DataFrame:
    labels = pd.read_sql_query(
        """
        SELECT
            id,
            name,
            level,
            parent_id,
            description,
            created_at,
            updated_at
        FROM labels
        ORDER BY level ASC, COALESCE(parent_id, 0) ASC, name ASC, id ASC
        """,
        conn,
    )

    if labels.empty:
        return labels

    mapping: Dict[int, int] = {
        original_id: idx + 1 for idx, original_id in enumerate(labels["id"])
    }

    labels["id_export"] = labels["id"].map(mapping)
    labels["parent_id_export"] = labels["parent_id"].map(mapping)

    # Convert to pandas nullable Int for consistent Excel output (allows NaN)
    labels["parent_id_export"] = labels["parent_id_export"].astype("Int64")
    labels["parent_id"] = labels["parent_id"].astype("Int64")

    ordered_columns = [
        "id_export",
        "name",
        "level",
        "parent_id_export",
        "description",
        "id",
        "parent_id",
        "created_at",
        "updated_at",
    ]

    labels = labels.rename(
        columns={
            "id_export": "id",
            "parent_id_export": "parent_id",
            "id": "original_id",
            "parent_id": "original_parent_id",
        }
    )[ordered_columns]

    return labels


def load_feedback_sentiments(
    conn: psycopg2.extensions.connection, mapping: Dict[int, int]
) -> pd.DataFrame:
    feedbacks = pd.read_sql_query(
        """
        SELECT
            fs.id,
            fs.feedback_text,
            fs.sentiment_label,
            fs.confidence_score,
            fs.feedback_source,
            fs.created_at,
            fs.level1_id,
            fs.level2_id,
            fs.level3_id,
            l1.name AS level1_name,
            l2.name AS level2_name,
            l3.name AS level3_name
        FROM feedback_sentiments fs
        LEFT JOIN labels l1 ON fs.level1_id = l1.id
        LEFT JOIN labels l2 ON fs.level2_id = l2.id
        LEFT JOIN labels l3 ON fs.level3_id = l3.id
        ORDER BY fs.created_at DESC
        """,
        conn,
    )

    if feedbacks.empty:
        return feedbacks

    for col in ("level1_id", "level2_id", "level3_id"):
        feedbacks[f"{col}_export"] = feedbacks[col].map(mapping).astype("Int64")
        feedbacks[col] = feedbacks[col].astype("Int64")

    ordered_columns = [
        "id",
        "feedback_text",
        "sentiment_label",
        "confidence_score",
        "feedback_source",
        "created_at",
        "level1_id_export",
        "level1_name",
        "level2_id_export",
        "level2_name",
        "level3_id_export",
        "level3_name",
        "level1_id",
        "level2_id",
        "level3_id",
    ]

    feedbacks = feedbacks.rename(
        columns={
            "level1_id_export": "level1_id",
            "level2_id_export": "level2_id",
            "level3_id_export": "level3_id",
            "level1_id": "original_level1_id",
            "level2_id": "original_level2_id",
            "level3_id": "original_level3_id",
        }
    )[ordered_columns]

    return feedbacks


def load_feedback_intents(
    conn: psycopg2.extensions.connection, mapping: Dict[int, int]
) -> Optional[pd.DataFrame]:
    if not table_exists(conn, "feedback_intents"):
        return None

    intents = pd.read_sql_query(
        """
        SELECT
            fi.id,
            fi.feedback_id,
            fi.level1_id,
            fi.level2_id,
            fi.level3_id,
            fi.avg_cosine_similarity,
            fi.created_at,
            l1.name AS level1_name,
            l2.name AS level2_name,
            l3.name AS level3_name
        FROM feedback_intents fi
        LEFT JOIN labels l1 ON fi.level1_id = l1.id
        LEFT JOIN labels l2 ON fi.level2_id = l2.id
        LEFT JOIN labels l3 ON fi.level3_id = l3.id
        ORDER BY fi.created_at DESC NULLS LAST, fi.feedback_id
        """,
        conn,
    )

    if intents.empty:
        return intents

    for col in ("level1_id", "level2_id", "level3_id"):
        intents[f"{col}_export"] = intents[col].map(mapping).astype("Int64")
        intents[col] = intents[col].astype("Int64")

    ordered_columns = [
        "id",
        "feedback_id",
        "avg_cosine_similarity",
        "created_at",
        "level1_id_export",
        "level1_name",
        "level2_id_export",
        "level2_name",
        "level3_id_export",
        "level3_name",
        "level1_id",
        "level2_id",
        "level3_id",
    ]

    intents = intents.rename(
        columns={
            "level1_id_export": "level1_id",
            "level2_id_export": "level2_id",
            "level3_id_export": "level3_id",
            "level1_id": "original_level1_id",
            "level2_id": "original_level2_id",
            "level3_id": "original_level3_id",
        }
    )[ordered_columns]

    return intents


def export_to_excel(
    labels: pd.DataFrame,
    feedbacks: pd.DataFrame,
    intents: Optional[pd.DataFrame],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"backup_{timestamp}.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        labels.to_excel(writer, sheet_name="labels", index=False)

        if not feedbacks.empty:
            feedbacks.to_excel(writer, sheet_name="feedback_sentiments", index=False)

        if intents is not None and not intents.empty:
            intents.to_excel(writer, sheet_name="feedback_intents", index=False)

    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dữ liệu ra Excel backup")
    parser.add_argument(
        "--output",
        type=str,
        default="backups",
        help="Thư mục lưu file Excel (default: ./backups)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    print("🔗 Đang kết nối database...")
    with get_connection() as conn:
        print("✅ Kết nối thành công.")

        print("📥 Đang tải dữ liệu labels...")
        labels_df = load_labels(conn)
        if labels_df.empty:
            print("⚠️ Không có dữ liệu labels, dừng quy trình.")
            return

        label_mapping = {
            int(row.original_id): int(row.id) for row in labels_df.itertuples()
        }

        print("📥 Đang tải dữ liệu feedback_sentiments...")
        feedback_df = load_feedback_sentiments(conn, label_mapping)

        print("📥 Kiểm tra và tải feedback_intents (nếu có)...")
        intents_df = load_feedback_intents(conn, label_mapping)

    print(f"💾 Đang ghi dữ liệu vào {output_dir.resolve()} ...")
    output_path = export_to_excel(labels_df, feedback_df, intents_df, output_dir)

    print("✅ Hoàn tất!")
    print(f"📁 File backup: {output_path}")
    print("📌 Sheet 'labels' dùng id tuần tự (bắt đầu 1); cột original_* giữ lại id gốc.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⛔ Export bị hủy bởi người dùng.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌ Export thất bại: {exc}", file=sys.stderr)
        sys.exit(1)


