"""
L-kn Gateway Utilities
Shared utilities for logging, request tracking, and error handling.
"""

import uuid
import time
import orjson
import structlog
from typing import Dict, Any, Optional
from fastapi import Request
from fastapi.responses import JSONResponse

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(serializer=orjson.dumps)
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


def generate_request_id() -> str:
    """Generate unique request ID."""
    return str(uuid.uuid4())


class RequestContextMiddleware:
    """Middleware to inject request_id into all logs."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request_id = generate_request_id()
            scope["request_id"] = request_id
            structlog.contextvars.clear_contextvars()
            structlog.contextvars.bind_contextvars(request_id=request_id)
        
        await self.app(scope, receive, send)


def error_response(
    status_code: int,
    error_type: str,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """Build standardized error response."""
    content = {
        "error": {
            "type": error_type,
            "message": message,
        }
    }
    if details:
        content["error"]["details"] = details
    
    return JSONResponse(
        status_code=status_code,
        content=content
    )


class LatencyTracker:
    """Track request latency."""
    
    def __init__(self):
        self.start_time = None
    
    def start(self):
        """Start timing."""
        self.start_time = time.perf_counter()
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time is None:
            return 0.0
        return (time.perf_counter() - self.start_time) * 1000


def classify_engine_error(exception: Exception) -> tuple[str, str]:
    """
    Classify engine errors into RSI categories for structured logging.
    
    Args:
        exception: The exception to classify
    
    Returns:
        Tuple of (category, reason) where:
        - category: "real" (infrastructure) or "simbólico" (protocol/schema)
        - reason: specific error type for logging
    """
    import httpx
    
    error_type = type(exception).__name__
    
    # Real (infrastructure failures)
    if isinstance(exception, (httpx.TimeoutException, TimeoutError)):
        return ("real", "engine_timeout")
    elif isinstance(exception, (httpx.ConnectError, ConnectionError)):
        return ("real", "engine_unreachable")
    elif isinstance(exception, RuntimeError) and "Circuit breaker" in str(exception):
        return ("real", "circuit_breaker_open")
    
    # Simbólico (protocol/schema issues)
    elif isinstance(exception, httpx.HTTPStatusError):
        status_code = exception.response.status_code
        return ("simbólico", f"http_error_{status_code}")
    elif isinstance(exception, (KeyError, ValueError, TypeError)):
        return ("simbólico", "schema_mismatch")
    
    # Unknown
    else:
        return ("real", f"unknown_error_{error_type}")


def log_request(
    request_id: str,
    mode: str,
    entropy_norm: Optional[float],
    latency_ms: float,
    status: str = "success",
    engine_status: Optional[str] = None,
    failure_reason: Optional[str] = None
):
    """
    Log structured request information.
    
    Args:
        request_id: Unique request identifier
        mode: Homeostatic mode (FLUIDO/ANALITICO/passthrough)
        entropy_norm: Normalized entropy if available
        latency_ms: Request latency in milliseconds
        status: Request status (success/failed/streaming)
        engine_status: Engine state (ok/timeout/circuit_open/http_error)
        failure_reason: Specific failure reason if status != success
    """
    log_data = {
        "request_id": request_id,
        "mode": mode,
        "entropy_norm": entropy_norm,
        "latency_ms": round(latency_ms, 2),
        "status": status
    }
    
    if engine_status:
        log_data["engine_status"] = engine_status
    if failure_reason:
        log_data["failure_reason"] = failure_reason
    
    if status == "success":
        logger.info("request_completed", **log_data)
    else:
        logger.error("request_failed", **log_data)
