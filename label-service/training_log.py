import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

from psycopg2.extensions import connection

from database import get_db

logger = logging.getLogger(__name__)

METADATA_TABLE = "training_log"


def ensure_training_log_table(conn: connection) -> None:
    """Đảm bảo bảng training_log tồn tại trong database."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {METADATA_TABLE} (
                id SERIAL PRIMARY KEY,
                service_name VARCHAR(255) NOT NULL, -- 'sentiment' or 'embedding'
                started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                finished_at TIMESTAMPTZ,
                status VARCHAR(50) NOT NULL, -- 'running', 'success', 'failed'
                triggered_by VARCHAR(255), -- 'startup', 'record_threshold', 'new_label'
                notes TEXT
            );
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{METADATA_TABLE}_service_status_finished
            ON {METADATA_TABLE} (service_name, status, finished_at DESC);
            """
        )
    logger.info(f"Đã đảm bảo bảng '{METADATA_TABLE}' tồn tại.")


def get_last_successful_run(
    service_name: str,
) -> Optional[datetime]:
    """Lấy thời điểm của lần huấn luyện thành công cuối cùng cho một service."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT finished_at
                FROM {METADATA_TABLE}
                WHERE service_name = %s AND status = 'success'
                ORDER BY finished_at DESC
                LIMIT 1;
                """,
                (service_name,),
            )
            result = cur.fetchone()
            return result[0] if result else None


def is_training_in_progress(service_name: str) -> bool:
    """Kiểm tra xem service có đang trong quá trình training không."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM {METADATA_TABLE}
                WHERE service_name = %s AND status = 'running'
                LIMIT 1;
                """,
                (service_name,),
            )
            result = cur.fetchone()
            return result[0] > 0 if result else False


@contextmanager
def training_run_logging(
    service_name: str, triggered_by: str
) -> Iterator[Optional[int]]:
    """Context manager để ghi log một lần chạy huấn luyện."""
    run_id = None
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {METADATA_TABLE} (service_name, status, triggered_by)
                    VALUES (%s, 'running', %s)
                    RETURNING id;
                    """,
                    (service_name, triggered_by),
                )
                run_id = cur.fetchone()[0]
                conn.commit()
                logger.info(
                    f"Bắt đầu lần huấn luyện cho '{service_name}', trigger bởi '{triggered_by}', run_id={run_id}."
                )
        yield run_id
        # Update status to success after the block finishes without errors
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {METADATA_TABLE}
                    SET status = 'success', finished_at = %s
                    WHERE id = %s;
                    """,
                    (datetime.now(timezone.utc), run_id),
                )
                conn.commit()
                logger.info(f"Hoàn thành thành công lần huấn luyện run_id={run_id}.")

    except Exception as e:
        logger.error(f"Lỗi trong lần huấn luyện run_id={run_id}: {e}", exc_info=True)
        if run_id:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        UPDATE {METADATA_TABLE}
                        SET status = 'failed', finished_at = %s, notes = %s
                        WHERE id = %s;
                        """,
                        (datetime.now(timezone.utc), str(e), run_id),
                    )
                    conn.commit()
        # Re-raise the exception to be handled by the caller
        raise
    finally:
        pass