"""
L-kn Gateway Utilities
Shared utilities for logging, request tracking, and error handling.
"""

import uuid
import time
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
        self.start_time = time.time()
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) * 1000


def log_request(
    request_id: str,
    mode: str,
    entropy_norm: Optional[float],
    latency_ms: float,
    status: str = "success"
):
    """Log structured request information."""
    logger.info(
        "request_completed",
        request_id=request_id,
        mode=mode,
        entropy_norm=entropy_norm,
        latency_ms=round(latency_ms, 2),
        status=status
    )
