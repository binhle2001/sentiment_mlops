from typing import Any, Dict, Optional

class SentimentServiceError(Exception):
    """Base exception for sentiment service errors."""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)

class ModelNotLoadedError(SentimentServiceError):
    """Raised when model is not loaded."""
    def __init__(self, message: str = "Sentiment model not loaded", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MODEL_NOT_LOADED", details)

class InvalidInputError(SentimentServiceError):
    """Raised when input validation fails."""
    def __init__(self, message: str = "Invalid input", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "INVALID_INPUT", details)

class ClassificationError(SentimentServiceError):
    """Raised when the classification process fails."""
    def __init__(self, message: str = "Classification failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CLASSIFICATION_ERROR", details)

class ServiceUnavailableError(SentimentServiceError):
    """Raised when the service is unavailable."""
    def __init__(self, message: str = "Service unavailable", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "SERVICE_UNAVAILABLE", details)

# --- HTTP Status Mapping ---
EXCEPTION_STATUS_MAP = {
    ModelNotLoadedError: 503,
    InvalidInputError: 422,
    ClassificationError: 500,
    ServiceUnavailableError: 503,
    SentimentServiceError: 500,
}

def get_http_status_code(exception: SentimentServiceError) -> int:
    """Get appropriate HTTP status code for an exception."""
    for exc_type, status_code in EXCEPTION_STATUS_MAP.items():
        if isinstance(exception, exc_type):
            return status_code
    return 500
