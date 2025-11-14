"""
Unified exception classes for Embedding Service.
"""
from typing import Any, Dict, Optional


class EmbeddingServiceError(Exception):
    """Base exception for embedding service errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


class ModelNotLoadedError(EmbeddingServiceError):
    """Raised when model is not loaded but required for operations."""
    
    def __init__(self, message: str = "Model not loaded", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MODEL_NOT_LOADED", details)


class InvalidInputError(EmbeddingServiceError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str = "Invalid input", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "INVALID_INPUT", details)


class ConfigurationError(EmbeddingServiceError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str = "Configuration error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CONFIGURATION_ERROR", details)


class ModelLoadError(EmbeddingServiceError):
    """Raised when model loading fails."""
    
    def __init__(self, message: str = "Failed to load model", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MODEL_LOAD_ERROR", details)


class EncodingError(EmbeddingServiceError):
    """Raised when text encoding fails."""
    
    def __init__(self, message: str = "Encoding failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "ENCODING_ERROR", details)


class ServiceUnavailableError(EmbeddingServiceError):
    """Raised when service is unavailable."""
    
    def __init__(self, message: str = "Service unavailable", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "SERVICE_UNAVAILABLE", details)


class ValidationError(EmbeddingServiceError):
    """Raised when validation fails."""
    
    def __init__(self, message: str = "Validation error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", details)


class RateLimitError(EmbeddingServiceError):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "RATE_LIMIT_EXCEEDED", details)


class AuthenticationError(EmbeddingServiceError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "AUTHENTICATION_ERROR", details)


class AuthorizationError(EmbeddingServiceError):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Authorization failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "AUTHORIZATION_ERROR", details)


# Exception mapping for HTTP status codes
EXCEPTION_STATUS_MAP = {
    EmbeddingServiceError: 500,
    ModelNotLoadedError: 503,
    InvalidInputError: 400,
    ConfigurationError: 500,
    ModelLoadError: 503,
    EncodingError: 500,
    ServiceUnavailableError: 503,
    ValidationError: 422,
    RateLimitError: 429,
    AuthenticationError: 401,
    AuthorizationError: 403,
}


def get_http_status_code(exception: EmbeddingServiceError) -> int:
    """Get appropriate HTTP status code for an exception."""
    for exc_type, status_code in EXCEPTION_STATUS_MAP.items():
        if isinstance(exception, exc_type):
            return status_code
    return 500


def is_client_error(exception: EmbeddingServiceError) -> bool:
    """Check if exception is a client error (4xx)."""
    status_code = get_http_status_code(exception)
    return 400 <= status_code < 500


def is_server_error(exception: EmbeddingServiceError) -> bool:
    """Check if exception is a server error (5xx)."""
    status_code = get_http_status_code(exception)
    return 500 <= status_code < 600
