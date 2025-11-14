import asyncio
import logging
from datetime import datetime

import httpx

from config import get_settings
from database import get_db
from training_log import ensure_training_log_table, get_last_successful_run

logger = logging.getLogger(__name__)
settings = get_settings()


class TrainingManager:
    """Quản lý và kích hoạt các quy trình huấn luyện theo kịch bản MLOps."""

    def __init__(self, check_interval_seconds: int = 300, record_threshold: int = 200):
        self.check_interval = check_interval_seconds
        self.record_threshold = record_threshold
        self._task: asyncio.Task = None
        self._running = False

    async def start(self):
        """Khởi động vòng lặp kiểm tra định kỳ."""
        if self._running:
            return
        logger.info(
            f"Khởi động TrainingManager: kiểm tra mỗi {self.check_interval} giây, ngưỡng {self.record_threshold} bản ghi."
        )
        # Đảm bảo bảng log tồn tại khi khởi động
        with get_db() as conn:
            ensure_training_log_table(conn)
            conn.commit()

        self._running = True
        self._task = asyncio.create_task(self._run_check_loop())

    async def stop(self):
        """Dừng vòng lặp kiểm tra."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            logger.info("Đã dừng TrainingManager.")

    async def _run_check_loop(self):
        """Vòng lặp chính, kiểm tra định kỳ."""
        while self._running:
            try:
                await self.check_and_trigger_all(triggered_by="record_threshold")
            except Exception as e:
                logger.error(f"Lỗi trong vòng lặp kiểm tra của TrainingManager: {e}", exc_info=True)
            await asyncio.sleep(self.check_interval)

    async def check_and_trigger_all(self, triggered_by: str):
        """Kiểm tra và kích hoạt huấn luyện cho tất cả các service nếu cần."""
        logger.info(f"Đang kiểm tra điều kiện huấn luyện (trigger: {triggered_by})...")
        # Hiện tại chỉ có một nguồn dữ liệu chung, nên trigger cả hai cùng lúc
        await self.check_and_trigger_service("sentiment", triggered_by)
        await self.check_and_trigger_service("embedding", triggered_by)

    def _count_new_confirmed_records(self, last_run_time: datetime | None) -> int:
        """Đếm số bản ghi được xác nhận mới kể từ lần chạy cuối."""
        with get_db() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT COUNT(*)
                    FROM feedback_sentiment
                    WHERE is_model_confirmed = TRUE
                """
                params = []
                if last_run_time:
                    query += " AND updated_at > %s"
                    params.append(last_run_time)

                cur.execute(query, tuple(params))
                count = cur.fetchone()[0]
                return count

    async def check_and_trigger_service(self, service_name: str, triggered_by: str):
        """Kiểm tra và kích hoạt cho một service cụ thể."""
        # Vì cả 2 model dùng chung data, ta chỉ cần check 1 lần và dùng chung last_run
        # để tránh gọi DB nhiều lần.
        last_run = get_last_successful_run(service_name)
        new_records = self._count_new_confirmed_records(last_run)

        logger.info(
            f"Kiểm tra '{service_name}': {new_records} bản ghi mới được xác nhận kể từ {last_run or 'lần đầu'}."
        )

        should_trigger = False
        if triggered_by == "new_label":
            total_records = self._count_new_confirmed_records(None)
            if total_records > self.record_threshold:
                should_trigger = True
                logger.info(f"Trigger '{service_name}' do có nhãn mới và tổng số bản ghi ({total_records}) > {self.record_threshold}")
        elif new_records > self.record_threshold:
            should_trigger = True
            logger.info(f"Trigger '{service_name}' do số bản ghi mới ({new_records}) > {self.record_threshold}")

        if should_trigger:
            await self._trigger_training_api(service_name, triggered_by)

    async def _trigger_training_api(self, service_name: str, triggered_by: str):
        """Gọi API để bắt đầu huấn luyện."""
        if service_name == "sentiment":
            url = settings.sentiment_training_url
        elif service_name == "embedding":
            # Giả sử có setting cho embedding training url
            url = settings.embedding_training_url
        else:
            logger.error(f"Tên service không hợp lệ: {service_name}")
            return

        if not url:
            logger.warning(f"Không có URL cấu hình cho '{service_name}', không thể trigger.")
            return

        # Các service huấn luyện đều có endpoint /train
        full_url = f"{url.rstrip('/')}/train"
        logger.info(f"Gửi yêu cầu POST tới {full_url} để bắt đầu huấn luyện (trigger: {triggered_by})...")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(full_url, json={"triggered_by": triggered_by})
                response.raise_for_status()
            logger.info(
                f"Đã kích hoạt thành công huấn luyện cho '{service_name}'. Status: {response.status_code}, Response: {response.text}"
            )
        except httpx.RequestError as e:
            logger.error(f"Lỗi khi gọi API huấn luyện cho '{service_name}': {e}")
        except Exception as e:
            logger.error(f"Lỗi không xác định khi trigger '{service_name}': {e}", exc_info=True)


# Singleton pattern cho TrainingManager
_training_manager_instance: TrainingManager | None = None
_manager_lock = asyncio.Lock()


async def get_training_manager() -> TrainingManager:
    """Lấy hoặc tạo instance của TrainingManager."""
    global _training_manager_instance
    async with _manager_lock:
        if _training_manager_instance is None:
            _training_manager_instance = TrainingManager(
                check_interval_seconds=settings.training_check_interval,
                record_threshold=settings.training_record_threshold,
            )
    return _training_manager_instance


async def startup_manager():
    """Khởi tạo và bắt đầu TrainingManager."""
    manager = await get_training_manager()
    await manager.start()


async def shutdown_manager():
    """Dừng TrainingManager."""
    if _training_manager_instance:
        await _training_manager_instance.stop()

