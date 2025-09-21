"""
Custom exception classes and error handlers for the Researcher Agent application.
Provides unified error handling with proper HTTP status codes and logging.
"""

from typing import Any, Dict, Optional, Union
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

logger = get_logger(__name__)


class ResearcherAgentException(Exception):
    """Base exception class for Researcher Agent application."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        self.message = message
        self.details = details or {}
        self.error_code = error_code
        super().__init__(self.message)


class ValidationError(ResearcherAgentException):
    """Raised when validation fails."""
    pass


class FileProcessingError(ResearcherAgentException):
    """Raised when file processing fails."""
    pass


class DocumentNotFoundError(ResearcherAgentException):
    """Raised when a document is not found."""
    pass


class EmbeddingError(ResearcherAgentException):
    """Raised when embedding generation fails."""
    pass


class DocumentError(ResearcherAgentException):
    """Raised when document operations fail."""
    pass


class ProcessingError(ResearcherAgentException):
    """Raised when document processing fails."""
    pass


class VectorStoreError(ResearcherAgentException):
    """Raised when vector store operations fail."""
    pass


class SearchError(ResearcherAgentException):
    """Raised when search operations fail."""
    pass


class LLMError(ResearcherAgentException):
    """Raised when LLM operations fail."""
    pass


class ConfigurationError(ResearcherAgentException):
    """Raised when configuration is invalid."""
    pass


class ServiceUnavailableError(ResearcherAgentException):
    """Raised when a service is unavailable."""
    pass


# Exception to HTTP status code mapping
EXCEPTION_STATUS_CODES = {
    ValidationError: status.HTTP_400_BAD_REQUEST,
    FileProcessingError: status.HTTP_422_UNPROCESSABLE_ENTITY,
    DocumentNotFoundError: status.HTTP_404_NOT_FOUND,
    EmbeddingError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    VectorStoreError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    SearchError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    LLMError: status.HTTP_502_BAD_GATEWAY,
    ConfigurationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ServiceUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
}


def create_error_response(
    status_code: int,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    error_code: Optional[str] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        status_code: HTTP status code
        message: Error message
        details: Additional error details
        error_code: Application-specific error code
        request_id: Request ID for tracking
        
    Returns:
        Standardized error response dictionary
    """
    response = {
        "error": True,
        "status_code": status_code,
        "message": message,
        "timestamp": None  # Will be set by middleware
    }
    
    if details:
        response["details"] = details
    
    if error_code:
        response["error_code"] = error_code
    
    if request_id:
        response["request_id"] = request_id
    
    return response


async def researcher_agent_exception_handler(
    request: Request,
    exc: ResearcherAgentException
) -> JSONResponse:
    """
    Handle custom ResearcherAgentException.
    
    Args:
        request: FastAPI request object
        exc: The exception instance
        
    Returns:
        JSON response with error details
    """
    status_code = EXCEPTION_STATUS_CODES.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    # Get request ID if available
    request_id = getattr(request.state, "request_id", None)
    
    # Log the error
    logger.error(
        "Application exception occurred",
        exception=type(exc).__name__,
        message=exc.message,
        details=exc.details,
        error_code=exc.error_code,
        status_code=status_code,
        request_id=request_id,
        path=request.url.path,
        method=request.method
    )
    
    response_data = create_error_response(
        status_code=status_code,
        message=exc.message,
        details=exc.details,
        error_code=exc.error_code,
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status_code,
        content=response_data
    )


async def http_exception_handler(
    request: Request,
    exc: Union[HTTPException, StarletteHTTPException]
) -> JSONResponse:
    """
    Handle HTTP exceptions.
    
    Args:
        request: FastAPI request object
        exc: The HTTP exception instance
        
    Returns:
        JSON response with error details
    """
    request_id = getattr(request.state, "request_id", None)
    
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        request_id=request_id,
        path=request.url.path,
        method=request.method
    )
    
    response_data = create_error_response(
        status_code=exc.status_code,
        message=str(exc.detail),
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handle request validation errors.
    
    Args:
        request: FastAPI request object
        exc: The validation exception instance
        
    Returns:
        JSON response with validation error details
    """
    request_id = getattr(request.state, "request_id", None)
    
    # Format validation errors
    validation_errors = []
    for error in exc.errors():
        validation_errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    logger.warning(
        "Validation error occurred",
        errors=validation_errors,
        request_id=request_id,
        path=request.url.path,
        method=request.method
    )
    
    response_data = create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        message="Validation error",
        details={"validation_errors": validation_errors},
        error_code="VALIDATION_ERROR",
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response_data
    )


async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """
    Handle unexpected exceptions.
    
    Args:
        request: FastAPI request object
        exc: The exception instance
        
    Returns:
        JSON response with generic error message
    """
    request_id = getattr(request.state, "request_id", None)
    
    logger.error(
        "Unexpected exception occurred",
        exception=type(exc).__name__,
        message=str(exc),
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        exc_info=True
    )
    
    response_data = create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        message="An unexpected error occurred",
        error_code="INTERNAL_ERROR",
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_data
    )


def setup_exception_handlers(app):
    """Setup exception handlers for the FastAPI app."""
    app.add_exception_handler(ResearcherAgentException, researcher_agent_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
