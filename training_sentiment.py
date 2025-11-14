import logging
import os
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
import torch
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


load_dotenv()


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@dataclass
class TrainingConfig:
    """Cấu hình huấn luyện sentiment."""

    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
    feedback_table: str = "feedback_sentiments"
    metadata_table: str = "sentiment_training_runs"
    change_threshold: float = 0.2
    min_changed_records: int = 1
    batch_size: int = 32
    max_length: int = 128
    noise_scale: float = 0.05
    test_size: float = 0.2
    random_state: int = 42
    sample_limit: Optional[int] = None
    model_name: str = "uitnlp/visobert"
    model_hidden_layers: Tuple[int, ...] = (256, 64)
    max_iter: int = 500
    model_output_dir: str = "artifacts/sentiment_mlp"
    save_model: bool = True

    @classmethod
    def from_env(cls) -> "TrainingConfig":
        """Tạo config từ biến môi trường."""

        def _get_int(name: str, default: int) -> int:
            value = os.getenv(name)
            return int(value) if value is not None else default

        def _get_float(name: str, default: float) -> float:
            value = os.getenv(name)
            return float(value) if value is not None else default

        hidden_layers_env = os.getenv("SENTIMENT_HIDDEN_LAYERS")
        if hidden_layers_env:
            hidden_layers = tuple(
                int(x.strip()) for x in hidden_layers_env.split(",") if x.strip()
            )
        else:
            hidden_layers = (256, 64)

        sample_limit_env = os.getenv("SENTIMENT_SAMPLE_LIMIT")

        return cls(
            postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
            postgres_port=_get_int("POSTGRES_PORT", 5432),
            postgres_db=os.getenv("POSTGRES_DB", "label_db"),
            postgres_user=os.getenv("POSTGRES_USER", "postgres"),
            postgres_password=os.getenv("POSTGRES_PASSWORD", "password"),
            feedback_table=os.getenv("SENTIMENT_FEEDBACK_TABLE", "feedback_sentiments"),
            metadata_table=os.getenv("SENTIMENT_METADATA_TABLE", "sentiment_training_runs"),
            change_threshold=_get_float("SENTIMENT_CHANGE_THRESHOLD", 0.2),
            min_changed_records=_get_int("SENTIMENT_MIN_CHANGED", 1),
            batch_size=_get_int("SENTIMENT_BATCH_SIZE", 32),
            max_length=_get_int("SENTIMENT_MAX_LENGTH", 128),
            noise_scale=_get_float("SENTIMENT_NOISE_SCALE", 0.05),
            test_size=_get_float("SENTIMENT_TEST_SIZE", 0.2),
            random_state=_get_int("SENTIMENT_RANDOM_STATE", 42),
            sample_limit=int(sample_limit_env) if sample_limit_env else None,
            model_name=os.getenv("SENTIMENT_MODEL_NAME", "uitnlp/visobert"),
            model_hidden_layers=hidden_layers,
            max_iter=_get_int("SENTIMENT_MAX_ITER", 500),
            model_output_dir=os.getenv("SENTIMENT_MODEL_DIR", "artifacts/sentiment_mlp"),
            save_model=os.getenv("SENTIMENT_SAVE_MODEL", "1").lower() not in {"0", "false"},
        )

    def to_public_dict(self) -> Dict[str, Any]:
        """Chuyển config sang dict (ẩn thông tin nhạy cảm)."""
        data = asdict(self)
        data.pop("postgres_password", None)
        return data


def create_connection(config: TrainingConfig) -> psycopg2.extensions.connection:
    """Tạo kết nối tới Postgres."""
    logger.debug("Kết nối Postgres %s:%s/%s", config.postgres_host, config.postgres_port, config.postgres_db)
    conn = psycopg2.connect(
        host=config.postgres_host,
        port=config.postgres_port,
        dbname=config.postgres_db,
        user=config.postgres_user,
        password=config.postgres_password,
    )
    conn.autocommit = True
    return conn


def ensure_metadata_table(conn: psycopg2.extensions.connection, config: TrainingConfig) -> None:
    """Đảm bảo bảng lưu lịch sử huấn luyện tồn tại."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {config.metadata_table} (
                id SERIAL PRIMARY KEY,
                started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                finished_at TIMESTAMPTZ,
                status VARCHAR(32) NOT NULL DEFAULT 'running',
                total_records INTEGER,
                changed_records INTEGER,
                model_path TEXT,
                notes TEXT
            );
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{config.metadata_table}_status_finished
            ON {config.metadata_table} (status, finished_at DESC);
            """
        )


def get_table_columns(conn: psycopg2.extensions.connection, table_name: str) -> List[str]:
    """Lấy danh sách cột của bảng."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
            """,
            (table_name,),
        )
        return [row[0] for row in cur.fetchall()]


def get_last_successful_training(
    conn: psycopg2.extensions.connection, config: TrainingConfig
) -> Optional[datetime]:
    """Lấy thời điểm huấn luyện thành công gần nhất."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT finished_at
            FROM {config.metadata_table}
            WHERE status = %s AND finished_at IS NOT NULL
            ORDER BY finished_at DESC
            LIMIT 1
            """,
            ("success",),
        )
        result = cur.fetchone()
        return result[0] if result else None


def count_total_records(
    conn: psycopg2.extensions.connection, config: TrainingConfig
) -> int:
    """Đếm tổng số bản ghi trong bảng feedback."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {config.feedback_table}")
        return cur.fetchone()[0]


def count_changed_records(
    conn: psycopg2.extensions.connection,
    config: TrainingConfig,
    last_trained_at: Optional[datetime],
    has_updated_at: bool,
) -> int:
    """Đếm số bản ghi được tạo/sửa kể từ lần train cuối."""
    if last_trained_at is None:
        return count_total_records(conn, config)

    timestamp = last_trained_at.astimezone(timezone.utc)
    column_expr = "COALESCE(updated_at, created_at)" if has_updated_at else "created_at"
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM {config.feedback_table}
            WHERE {column_expr} > %s
            """,
            (timestamp,),
        )
        return cur.fetchone()[0]


def should_trigger_training(
    conn: psycopg2.extensions.connection,
    config: TrainingConfig,
    force: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Kiểm tra có cần chạy huấn luyện không."""
    ensure_metadata_table(conn, config)
    last_trained_at = get_last_successful_training(conn, config)
    columns = get_table_columns(conn, config.feedback_table)
    has_updated_at = "updated_at" in [col.lower() for col in columns]

    total_records = count_total_records(conn, config)
    changed_records = count_changed_records(conn, config, last_trained_at, has_updated_at)
    change_ratio = (changed_records / total_records) if total_records else 0.0

    reason = "force" if force else ""
    should_run = force or (
        total_records > 0
        and changed_records >= config.min_changed_records
        and change_ratio >= config.change_threshold
    )

    details = {
        "total_records": total_records,
        "changed_records": changed_records,
        "change_ratio": round(change_ratio, 4),
        "threshold": config.change_threshold,
        "min_changed_records": config.min_changed_records,
        "last_trained_at": last_trained_at.isoformat() if last_trained_at else None,
        "has_updated_at": has_updated_at,
        "reason": reason or ("threshold_met" if should_run else "threshold_not_met"),
    }

    return should_run, details


def fetch_feedback_dataframe(
    conn: psycopg2.extensions.connection,
    config: TrainingConfig,
) -> pd.DataFrame:
    """Tải dữ liệu feedback từ Postgres."""
    columns = get_table_columns(conn, config.feedback_table)
    select_columns = ["id", "feedback_text", "sentiment_label", "confidence_score", "created_at"]
    optional_columns = ["feedback_source", "updated_at"]
    for col in optional_columns:
        if col in columns:
            select_columns.append(col)

    query = f"""
        SELECT {', '.join(select_columns)}
        FROM {config.feedback_table}
        WHERE sentiment_label IS NOT NULL
          AND TRIM(sentiment_label) <> ''
          AND feedback_text IS NOT NULL
          AND TRIM(feedback_text) <> ''
    """

    params: Tuple[Any, ...] = ()
    if config.sample_limit:
        query += " LIMIT %s"
        params = (config.sample_limit,)

    logger.info("Đang tải dữ liệu feedback...")
    return pd.read_sql_query(query, conn, params=params)


def preprocess_feedback(df: pd.DataFrame) -> pd.DataFrame:
    """Tiền xử lý dữ liệu văn bản + nhãn."""
    if df.empty:
        return df

    df = df.copy()
    df["feedback_text"] = df["feedback_text"].astype(str).str.strip()
    df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip()

    df = df[(df["feedback_text"] != "") & (df["sentiment_label"] != "")]
    df = df.drop_duplicates(subset=["feedback_text", "sentiment_label"], keep="last")
    df = df.reset_index(drop=True)
    logger.info("Sau tiền xử lý còn %s bản ghi", len(df))
    return df


def encode_batches(
    texts: Iterable[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Encoder văn bản theo batch với ViSoBERT."""
    embeddings: List[np.ndarray] = []
    text_list = list(texts)

    for start in tqdm(range(0, len(text_list), batch_size), desc="Encoding feedback", unit="batch"):
        batch_texts = text_list[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(emb)

    matrix = np.vstack(embeddings)
    return normalize(matrix)


def augment_with_gaussian(
    X: np.ndarray,
    y: np.ndarray,
    noise_scale: float,
    target_count: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tăng cường dữ liệu cho lớp hiếm bằng nhiễu Gaussian."""
    unique, counts = np.unique(y, return_counts=True)
    max_count = max(counts) if counts.size else 0
    min_count = target_count or max_count

    X_aug, y_aug = [X], [y]
    rng = np.random.default_rng(seed=42)

    for label, count in zip(unique, counts):
        if count < min_count:
            need = min_count - count
            subset = X[y == label]
            idx = rng.choice(len(subset), size=need, replace=True)
            noise = rng.normal(0, noise_scale, subset[idx].shape)
            X_new = subset[idx] + noise
            X_aug.append(X_new)
            y_aug.append(np.array([label] * need))

    return np.vstack(X_aug), np.concatenate(y_aug)


def start_training_run(
    conn: psycopg2.extensions.connection,
    config: TrainingConfig,
    stats: Dict[str, Any],
) -> Tuple[int, datetime]:
    """Ghi nhận bắt đầu huấn luyện."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"""
            INSERT INTO {config.metadata_table} (status, total_records, changed_records, notes)
            VALUES (%s, %s, %s, %s)
            RETURNING id, started_at
            """,
            (
                "running",
                stats.get("total_records"),
                stats.get("changed_records"),
                f"ratio={stats.get('change_ratio')}",
            ),
        )
        result = cur.fetchone()
        return result["id"], result["started_at"]


def finish_training_run(
    conn: psycopg2.extensions.connection,
    config: TrainingConfig,
    run_id: int,
    status: str,
    notes: Optional[str] = None,
    model_path: Optional[str] = None,
) -> None:
    """Cập nhật trạng thái huấn luyện."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {config.metadata_table}
            SET finished_at = %s,
                status = %s,
                notes = %s,
                model_path = COALESCE(%s, model_path)
            WHERE id = %s
            """,
            (datetime.now(timezone.utc), status, notes, model_path, run_id),
        )


def train_sentiment_pipeline(
    config: TrainingConfig,
    force: bool = False,
) -> Dict[str, Any]:
    """Pipeline chính: kiểm tra trigger, tải dữ liệu, huấn luyện và lưu kết quả."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Sử dụng thiết bị: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModel.from_pretrained(config.model_name)
    model.to(device)
    model.eval()

    with closing(create_connection(config)) as conn:
        should_run, stats = should_trigger_training(conn, config, force=force)
        if not should_run:
            logger.info(
                "Bỏ qua huấn luyện sentiment (changed=%s, ratio=%.4f < threshold %.4f)",
                stats["changed_records"],
                stats["change_ratio"],
                stats["threshold"],
            )
            return {"triggered": False, "details": stats}

        run_id, started_at = start_training_run(conn, config, stats)
        logger.info("Bắt đầu huấn luyện sentiment run_id=%s", run_id)

        try:
            raw_df = fetch_feedback_dataframe(conn, config)
            df = preprocess_feedback(raw_df)

            if df.empty:
                raise ValueError("Không có dữ liệu hợp lệ để huấn luyện sentiment.")

            texts = df["feedback_text"].tolist()
            labels = df["sentiment_label"].tolist()

            X_emb = encode_batches(
                texts,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=config.batch_size,
                max_length=config.max_length,
            )
            y = np.array(labels)

            if np.unique(y).size < 2:
                raise ValueError("Cần tối thiểu 2 nhãn sentiment khác nhau để huấn luyện.")

            X_bal, y_bal = augment_with_gaussian(X_emb, y, noise_scale=config.noise_scale)

            stratify_labels = y_bal if np.unique(y_bal).size > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X_bal,
                y_bal,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=stratify_labels,
            )

            clf = MLPClassifier(
                hidden_layer_sizes=config.model_hidden_layers,
                max_iter=config.max_iter,
                random_state=config.random_state,
            )
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            report_text = classification_report(y_test, y_pred)
            report_dict = classification_report(y_test, y_pred, output_dict=True)

            model_path = None
            if config.save_model:
                if joblib is None:
                    logger.warning("joblib chưa được cài đặt, bỏ qua bước lưu mô hình.")
                else:
                    os.makedirs(config.model_output_dir, exist_ok=True)
                    model_path = os.path.join(config.model_output_dir, "sentiment_mlp.joblib")
                    joblib.dump(
                        {
                            "classifier": clf,
                            "config": config.to_public_dict(),
                            "report": report_dict,
                            "trained_at": datetime.now(timezone.utc),
                        },
                        model_path,
                    )
                    logger.info("Đã lưu mô hình MLP tại %s", model_path)

            finish_training_run(
                conn,
                config,
                run_id=run_id,
                status="success",
                notes=report_text,
                model_path=model_path,
            )

            stats.update(
                {
                    "triggered": True,
                    "run_id": run_id,
                    "started_at": started_at.isoformat(),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "samples": len(df),
                    "augmented_samples": len(y_bal),
                    "model_path": model_path,
                    "report": report_dict,
                }
            )

            logger.info("Hoàn tất huấn luyện sentiment run_id=%s", run_id)
            return {"triggered": True, "details": stats}

        except Exception as exc:
            logger.exception("Huấn luyện sentiment thất bại: %s", exc)
            finish_training_run(
                conn,
                config,
                run_id=run_id,
                status="failed",
                notes=str(exc),
            )
            raise


def main() -> None:
    """Entry point cho script CLI."""
    config = TrainingConfig.from_env()
    result = train_sentiment_pipeline(config=config, force=False)
    logger.info("Kết quả huấn luyện: %s", result)


if __name__ == "__main__":
    main()