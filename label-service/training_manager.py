import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from config import Settings
from database import execute_query, get_db

logger = logging.getLogger(__name__)

ModelType = str


@dataclass
class ModelState:
    confirmed: int = 0
    relabel: int = 0
    last_event_at: Optional[float] = None
    samples: List[Dict[str, Any]] = field(default_factory=list)
    training: bool = False

    def reset(self) -> None:
        self.confirmed = 0
        self.relabel = 0
        self.last_event_at = None
        self.samples.clear()
        self.training = False


class TrainingManager:
    """Quản lý trigger huấn luyện cho sentiment và intent."""

    SAMPLE_BUFFER_SIZE = 500

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = asyncio.Lock()
        self._model_states: Dict[ModelType, ModelState] = {
            "intent": ModelState(),
            "sentiment": ModelState(),
        }
        self._current_trigger: Optional[ModelType] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._runner_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if not self.settings.enable_training_trigger:
            logger.info("Training trigger is disabled via configuration.")
            return

        if self._runner_task is not None:
            return

        self._loop = asyncio.get_running_loop()
        self._running = True
        self._runner_task = asyncio.create_task(self._run_loop(), name="training-trigger-loop")
        logger.info("Training manager started.")

    async def shutdown(self) -> None:
        self._running = False
        if self._runner_task:
            self._runner_task.cancel()
            try:
                await self._runner_task
            except asyncio.CancelledError:
                pass
            self._runner_task = None
        logger.info("Training manager stopped.")

    async def get_status(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                "current_trigger": self._current_trigger,
                "models": {
                    model: {
                        "confirmed": state.confirmed,
                        "relabel": state.relabel,
                        "last_event_at": state.last_event_at,
                        "training": state.training,
                        "samples_buffer": len(state.samples),
                    }
                    for model, state in self._model_states.items()
                },
            }

    def record_confirm(self, model: ModelType, feedback: Optional[Dict[str, Any]] = None) -> None:
        if not self.settings.enable_training_trigger:
            return
        if self._loop is None:
            logger.debug("Attempt to record confirm before training manager started.")
            return
        asyncio.run_coroutine_threadsafe(
            self._record_event("confirm", model, feedback),
            self._loop,
        )

    def record_relabel(self, model: ModelType, feedback: Optional[Dict[str, Any]] = None) -> None:
        if not self.settings.enable_training_trigger:
            return
        if self._loop is None:
            logger.debug("Attempt to record relabel before training manager started.")
            return
        asyncio.run_coroutine_threadsafe(
            self._record_event("relabel", model, feedback),
            self._loop,
        )

    async def record_confirm_async(self, model: ModelType, feedback: Optional[Dict[str, Any]] = None) -> None:
        if not self.settings.enable_training_trigger:
            return
        await self._record_event("confirm", model, feedback)

    async def record_relabel_async(self, model: ModelType, feedback: Optional[Dict[str, Any]] = None) -> None:
        if not self.settings.enable_training_trigger:
            return
        await self._record_event("relabel", model, feedback)

    async def _record_event(self, event_type: str, model: ModelType, feedback: Optional[Dict[str, Any]]) -> None:
        if model not in self._model_states:
            logger.warning("Unknown model type '%s' passed to training manager.", model)
            return

        sample: Optional[Dict[str, Any]] = None
        if model == "intent" and feedback:
            sample = await asyncio.get_running_loop().run_in_executor(
                None,
                self._build_intent_sample,
                feedback,
            )

        async with self._lock:
            state = self._model_states[model]
            if event_type == "confirm":
                state.confirmed += 1
            elif event_type == "relabel":
                state.relabel += 1

            now = time.monotonic()
            state.last_event_at = now

            if sample:
                state.samples.append(sample)
                if len(state.samples) > self.SAMPLE_BUFFER_SIZE:
                    state.samples = state.samples[-self.SAMPLE_BUFFER_SIZE :]

            if self._current_trigger is None:
                self._current_trigger = model
            logger.debug(
                "Recorded %s for %s: confirmed=%s, relabel=%s",
                event_type,
                model,
                state.confirmed,
                state.relabel,
            )

    async def _run_loop(self) -> None:
        interval = max(1, self.settings.training_check_interval_seconds)
        try:
            while self._running:
                await asyncio.sleep(interval)
                await self._tick()
        except asyncio.CancelledError:
            logger.debug("Training loop cancelled.")
        except Exception:  # pragma: no cover - defensive
            logger.exception("Unexpected error inside training manager loop.")

    async def _tick(self) -> None:
        if not self.settings.enable_training_trigger:
            return

        model: Optional[ModelType] = None
        state_snapshot: Optional[ModelState] = None

        async with self._lock:
            model = self._current_trigger or self._choose_next_model_locked()
            if model is None:
                return

            state = self._model_states[model]
            if state.training:
                return

            if state.last_event_at is None:
                return

            idle_seconds = time.monotonic() - state.last_event_at
            if idle_seconds < self.settings.training_idle_seconds:
                return

            thresholds = self._get_thresholds(model)
            if state.confirmed <= thresholds["confirmed"] or state.relabel <= thresholds["relabel"]:
                return

            state.training = True
            state_snapshot = ModelState(
                confirmed=state.confirmed,
                relabel=state.relabel,
                last_event_at=state.last_event_at,
                samples=list(state.samples),
                training=True,
            )
            self._current_trigger = model
            logger.info(
                "Triggering %s training (confirmed=%s, relabel=%s, idle=%.1fs).",
                model,
                state.confirmed,
                state.relabel,
                idle_seconds,
            )

        if model is None or state_snapshot is None:
            return

        success = await self._call_training_service(model)

        async with self._lock:
            state = self._model_states[model]
            if success:
                logger.info("Training service for %s acknowledged trigger, resetting counters.", model)
                state.reset()
                self._current_trigger = self._choose_next_model_locked(exclude=model)
            else:
                logger.warning("Training service for %s failed, keeping counters for retry.", model)
                state.training = False
                state.last_event_at = time.monotonic()

    async def _call_training_service(self, model: ModelType) -> bool:
        url = (
            self.settings.intent_training_service_url
            if model == "intent"
            else self.settings.sentiment_training_service_url
        )

        timeout = httpx.Timeout(30.0, connect=10.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url)
            if response.status_code not in (200, 202):
                logger.error(
                    "Training service for %s returned %s: %s",
                    model,
                    response.status_code,
                    response.text,
                )
                return False
            return True
        except httpx.HTTPError as exc:
            logger.error("Error calling training service for %s: %s", model, exc)
            return False

    def _choose_next_model_locked(self, exclude: Optional[ModelType] = None) -> Optional[ModelType]:
        candidates = []
        for model, state in self._model_states.items():
            if model == exclude:
                continue
            if state.last_event_at is None:
                continue
            if state.confirmed == 0 and state.relabel == 0:
                continue
            candidates.append((model, state.last_event_at))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[1])
        return candidates[0][0]

    def _get_thresholds(self, model: ModelType) -> Dict[str, int]:
        if model == "intent":
            return {
                "confirmed": self.settings.intent_confirm_threshold,
                "relabel": self.settings.intent_relabel_threshold,
            }
        return {
            "confirmed": self.settings.sentiment_confirm_threshold,
            "relabel": self.settings.sentiment_relabel_threshold,
        }

    def _build_intent_sample(self, feedback: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        positive_label_id = feedback.get("level3_id")
        if positive_label_id is None:
            return None

        negative_ids = self._pick_negative_labels(positive_label_id, limit=3)

        return {
            "feedback_id": str(feedback.get("id")),
            "feedback_text": feedback.get("feedback_text"),
            "positive_label_id": positive_label_id,
            "positive_label_name": feedback.get("level3_name"),
            "negative_label_ids": negative_ids,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def _pick_negative_labels(self, positive_label_id: int, limit: int = 3) -> List[int]:
        with get_db() as conn:
            rows = execute_query(
                conn,
                """
                SELECT id
                FROM labels
                WHERE level = 3 AND id <> %s
                ORDER BY RANDOM()
                LIMIT %s
                """,
                (positive_label_id, limit),
                fetch="all",
            )
        return [row["id"] for row in rows] if rows else []


_manager_instance: Optional[TrainingManager] = None
_manager_lock = asyncio.Lock()


async def init_training_manager(settings: Settings) -> TrainingManager:
    global _manager_instance
    async with _manager_lock:
        if _manager_instance is None:
            manager = TrainingManager(settings)
            await manager.start()
            _manager_instance = manager
    return _manager_instance


async def shutdown_training_manager() -> None:
    global _manager_instance
    async with _manager_lock:
        if _manager_instance is not None:
            await _manager_instance.shutdown()
            _manager_instance = None


def get_training_manager() -> TrainingManager:
    if _manager_instance is None:
        raise RuntimeError("Training manager has not been initialized yet.")
    return _manager_instance

