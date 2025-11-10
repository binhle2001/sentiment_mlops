import logging
import asyncio
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from config import Settings, get_settings
from exceptions import (
    ClassificationError,
    ServiceUnavailableError,
    ModelNotLoadedError,
    InvalidInputError,
)
from models import SentimentResult, SentimentLabel

logger = logging.getLogger(__name__)


class SentimentService:
    """
    Unified sentiment classification service using ONNX.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: Optional[ort.InferenceSession] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._model_loaded = False
        self.lock = asyncio.Lock()
        # Mapping from model output index to sentiment label
        self.sentiment_map = {
            0: SentimentLabel.EXTREMELY_NEGATIVE,
            1: SentimentLabel.NEGATIVE,
            2: SentimentLabel.NEUTRAL,
            3: SentimentLabel.POSITIVE,
        }

    async def initialize(self) -> None:
        """Load the ONNX sentiment model and tokenizer."""
        async with self.lock:
            if self._model_loaded:
                logger.info("Sentiment model already loaded.")
                return

            try:
                model_path = self.settings.model_path
                onnx_model_file = os.path.join(model_path, "visobert_sentiment_full.onnx")
                logger.info(f"Loading sentiment model from: {onnx_model_file}")

                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model path not found in {model_path}")
                print(os.listdir(model_path))
                if not os.path.exists(onnx_model_file):
                    raise FileNotFoundError(f"ONNX file not found in {onnx_model_file}")

                self.tokenizer, self.model = await asyncio.to_thread(
                    self._load_model_components, model_path, onnx_model_file
                )

                self._model_loaded = True
                logger.info("ONNX sentiment model and tokenizer loaded successfully.")
                await self._validate_model()

            except Exception as e:
                logger.error(f"Failed to load ONNX sentiment model: {e}", exc_info=True)
                raise ServiceUnavailableError(f"Failed to initialize sentiment model: {e}")

    def _load_model_components(self, model_path: str, onnx_model_file: str) -> Tuple[AutoTokenizer, ort.InferenceSession]:
        """Synchronous helper to load tokenizer and ONNX session."""
        tokenizer_path = os.path.join(model_path, "visobert_tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = self.settings.inter_op_threads
        sess_options.intra_op_num_threads = self.settings.intra_op_threads
        
        model = ort.InferenceSession(
            onnx_model_file,
            providers=self.settings.onnx_providers,
            sess_options=sess_options
        )
        return tokenizer, model

    async def _validate_model(self):
        """Run a test classification to validate the loaded model."""
        try:
            logger.info("Validating sentiment model with a test run...")
            result = await self.classify("đây là một câu văn bản thử nghiệm")
            assert isinstance(result, SentimentResult)
            logger.info(f"Model validation successful. Test result: {result.label.value}")
        except Exception as e:
            self._model_loaded = False
            logger.error(f"Sentiment model validation failed: {e}", exc_info=True)
            raise ServiceUnavailableError(f"Model validation failed after loading: {e}")

    async def classify(self, text: str) -> SentimentResult:
        """Classify the sentiment of a single text."""
        if not self._model_loaded:
            raise ModelNotLoadedError()
        if not text or not text.strip():
            raise InvalidInputError("Text cannot be empty.")

        try:
            results = await asyncio.to_thread(self._run_inference_batch, [text])
            return results[0]
        except Exception as e:
            logger.error(f"Classification failed for text: '{text[:50]}...': {e}", exc_info=True)
            raise ClassificationError(f"Failed to classify text: {e}")

    async def classify_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Classify the sentiment of a batch of texts."""
        if not self._model_loaded:
            raise ModelNotLoadedError()
        if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
            raise InvalidInputError("All texts in the list must be non-empty strings.")

        try:
            results = await asyncio.to_thread(self._run_inference_batch, texts)
            return results
        except Exception as e:
            logger.error(f"Batch classification failed: {e}", exc_info=True)
            raise ClassificationError(f"Failed to classify batch: {e}")

    def _run_inference_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Synchronous batch inference and post-processing."""
        # Preprocess
        inputs = self._preprocess_texts(texts)
        
        # Run inference
        outputs = self.model.run(None, inputs)
        logits = outputs[0]
        
        # Postprocess
        results = self._postprocess_outputs(logits)
        return results

    def _preprocess_texts(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Preprocess texts for model input."""
        encodings = self.tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self.settings.max_length,
        )
        
        inputs = {name: encodings[name] for name in self.tokenizer.model_input_names}
        return inputs

    def _postprocess_outputs(self, logits: np.ndarray) -> List[SentimentResult]:
        """Postprocess model outputs to sentiment results."""
        probabilities = self._softmax(logits)
        results = []
        for probs in probabilities:
            predicted_class_idx = np.argmax(probs)
            label = self.sentiment_map[predicted_class_idx]
            score = float(probs[predicted_class_idx])
            results.append(SentimentResult(label=label, score=score))
        return results

    def _softmax(self, x: np.ndarray, axis=-1) -> np.ndarray:
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check."""
        if not self._model_loaded:
            return {"status": "unhealthy", "reason": "Model not loaded.", "timestamp": datetime.utcnow().isoformat()}
        
        try:
            await self.classify("health check")
            return {
                "status": "healthy",
                "model_info": {"model_loaded": True, "model_type": "ONNX Sentiment"},
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Sentiment health check failed: {e}", exc_info=True)
            return {"status": "unhealthy", "reason": str(e), "timestamp": datetime.utcnow().isoformat()}

    async def cleanup(self) -> None:
        """Clean up resources."""
        async with self.lock:
            self.model = None
            self.tokenizer = None
            self._model_loaded = False
            logger.info("Sentiment service cleaned up.")

# --- Singleton Management ---
_service_instance: Optional[SentimentService] = None
_service_lock = asyncio.Lock()

async def get_sentiment_service() -> SentimentService:
    """Dependency provider for the SentimentService singleton."""
    global _service_instance
    async with _service_lock:
        if _service_instance is None:
            settings = get_settings()
            _service_instance = SentimentService(settings)
            await _service_instance.initialize()
    return _service_instance
