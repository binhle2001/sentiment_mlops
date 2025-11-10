"""
Optimized and simplified embedding service using ONNX Runtime.
"""
import logging
import asyncio
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from config import Settings, get_settings
from exceptions import (
    InvalidInputError,
    EncodingError,
    ServiceUnavailableError,
    ModelNotLoadedError,
)

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Unified embedding service using ONNX for inference.
    This class handles model loading, encoding, pooling, and health checks.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: Optional[ort.InferenceSession] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._model_loaded = False
        self.lock = asyncio.Lock()

    async def initialize(self) -> None:
        """
        Load the ONNX embedding model and tokenizer.
        """
        async with self.lock:
            if self._model_loaded:
                logger.info("Model already loaded.")
                return

            try:
                model_path = self.settings.model_path
                onnx_model_file = os.path.join(model_path, "model.onnx")
                logger.info(f"Loading model from: {model_path}")

                if not os.path.exists(model_path) or not os.path.exists(onnx_model_file):
                    raise FileNotFoundError(f"Model path or ONNX file not found in {model_path}")

                # Load tokenizer and model in a thread pool to avoid blocking
                self.tokenizer, self.model = await asyncio.to_thread(
                    self._load_model_components, model_path, onnx_model_file
                )

                self._model_loaded = True
                logger.info("ONNX model and tokenizer loaded successfully.")

                # Validate model with a test run
                await self._validate_model()

            except Exception as e:
                logger.error(f"Failed to load ONNX model: {e}", exc_info=True)
                raise ServiceUnavailableError(f"Failed to initialize embedding model: {e}")

    def _load_model_components(self, model_path: str, onnx_model_file: str):
        """Synchronous helper to load tokenizer and ONNX session."""
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        
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
        """Run a test encoding to validate the loaded model."""
        try:
            logger.info("Validating model with a test encoding...")
            test_embedding = await self.encode_single("test", "mean")
            dim = test_embedding.get("dimension")
            logger.info(dim)
            if dim != self.settings.embedding_dimension:
                raise ValueError(f"Model output dimension {dim} does not match configured dimension {self.settings.embedding_dimension}")
            logger.info(f"Model validation successful. Embedding dimension: {dim}")
        except Exception as e:
            self._model_loaded = False # Invalidate model on failed test
            logger.error(f"Model validation failed: {e}", exc_info=True)
            raise ServiceUnavailableError(f"Model validation failed after loading: {e}")


    async def encode_single(self, text: str, pooling_strategy: str) -> Dict[str, Any]:
        """Encode a single text to an embedding."""
        if not self._model_loaded:
            raise ModelNotLoadedError("Model is not available for encoding.")

        start_time = time.time()
        
        try:
            if not text or not text.strip():
                raise InvalidInputError("Text cannot be empty.")
            
            embedding = await asyncio.to_thread(self._encode_text, text, pooling_strategy)
            
            encoding_time = time.time() - start_time
            
            return {
                "embedding": embedding.tolist(),
                "dimension": len(embedding),
                "text_length": len(text),
                "pooling_strategy": pooling_strategy,
                "status": "success",
                "encoding_time": encoding_time,
            }
        except Exception as e:
            logger.error(f"Encoding failed for text: '{text[:50]}...': {e}", exc_info=True)
            raise EncodingError(f"Failed to encode text: {e}")

    async def encode_batch(self, texts: List[str], pooling_strategy: str) -> Dict[str, Any]:
        """Encode a batch of texts to embeddings."""
        if not self._model_loaded:
            raise ModelNotLoadedError("Model is not available for encoding.")

        start_time = time.time()

        try:
            if not texts:
                raise InvalidInputError("Texts list cannot be empty.")

            embeddings = await asyncio.to_thread(self._encode_batch, texts, pooling_strategy)
            
            encoding_time = time.time() - start_time

            return {
                "embeddings": [e.tolist() for e in embeddings],
                "dimension": len(embeddings[0]) if embeddings else 0,
                "total_texts": len(texts),
                "valid_texts": len(embeddings),
                "pooling_strategy": pooling_strategy,
                "status": "success",
                "encoding_time": encoding_time,
            }
        except Exception as e:
            logger.error(f"Batch encoding failed: {e}", exc_info=True)
            raise EncodingError(f"Failed to encode batch: {e}")

    def _encode_text(self, text: str, pooling_strategy: str) -> np.ndarray:
        """Synchronous text encoding using ONNX."""
        enc = self.tokenizer(
            [text],
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self.settings.max_length,
            return_attention_mask=True
        )
        ort_inputs = self._ensure_inputs(enc)
        model_outputs = self.model.run(None, ort_inputs)
        token_embeddings = model_outputs[0]
        attention_mask = enc['attention_mask']
        sentence_embedding = self._apply_pooling(
            token_embeddings[0], attention_mask[0], pooling_strategy
        )
        return sentence_embedding.astype(np.float32)

    def _encode_batch(self, texts: List[str], pooling_strategy: str) -> List[np.ndarray]:
        """Synchronous batch encoding using ONNX."""
        enc = self.tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self.settings.max_length,
            return_attention_mask=True
        )
        ort_inputs = self._ensure_inputs(enc)
        model_outputs = self.model.run(None, ort_inputs)
        token_embeddings = model_outputs[0]
        attention_masks = enc['attention_mask']
        
        sentence_embeddings = [
            self._apply_pooling(token_embeddings[i], attention_masks[i], pooling_strategy).astype(np.float32)
            for i in range(len(texts))
        ]
        return sentence_embeddings

    def _ensure_inputs(self, enc_np: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Ensure all required ONNX inputs are present."""
        inputs = {}
        required_inputs = {inp.name for inp in self.model.get_inputs()}
        
        for name in required_inputs:
            if name in enc_np:
                inputs[name] = enc_np[name]
            elif name == "token_type_ids":
                inputs[name] = np.zeros_like(enc_np["input_ids"], dtype=np.int64)
            else:
                raise ValueError(f"Missing required ONNX input: {name}")
        return inputs

    def _apply_pooling(self, token_embeddings: np.ndarray, attention_mask: np.ndarray, pooling_strategy: str) -> np.ndarray:
        """Apply pooling to convert token embeddings to a sentence embedding."""
        if pooling_strategy == 'cls':
            return token_embeddings[0]
        
        elif pooling_strategy == 'mean':
            attention_mask_expanded = np.expand_dims(attention_mask, axis=-1)
            masked_embeddings = token_embeddings * attention_mask_expanded
            sum_embeddings = np.sum(masked_embeddings, axis=0)
            sum_mask = np.sum(attention_mask)
            return sum_embeddings / sum_mask if sum_mask > 0 else sum_embeddings
        
        elif pooling_strategy == 'max':
            attention_mask_expanded = np.expand_dims(attention_mask, axis=-1)
            masked_embeddings = np.where(attention_mask_expanded, token_embeddings, -1e9)
            return np.max(masked_embeddings, axis=0)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check."""
        if not self._model_loaded:
            return {
                "status": "unhealthy", 
                "reason": "Model not loaded.", 
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Simplified health check: only verify model is loaded.
        # The validation step during initialize() is sufficient to check model integrity.
        return {
            "status": "healthy",
            "model_info": {
                "model_loaded": True, 
                "dimension": self.settings.embedding_dimension,
                "model_type": "ONNX"
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def cleanup(self) -> None:
        """Clean up resources."""
        async with self.lock:
            self.model = None
            self.tokenizer = None
            self._model_loaded = False
            logger.info("Embedding service cleaned up.")


# --- Singleton Management ---
_service_instance: Optional[EmbeddingService] = None
_service_lock = asyncio.Lock()


async def get_embedding_service() -> EmbeddingService:
    """Dependency provider for the EmbeddingService singleton."""
    global _service_instance
    async with _service_lock:
        if _service_instance is None:
            settings = get_settings()
            _service_instance = EmbeddingService(settings)
            await _service_instance.initialize()
    return _service_instance
