"""
L-kn Engine Client
Async HTTP client for SGLang engine with circuit breaker and retry logic.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from enum import Enum
import httpx
import structlog
from pydantic import BaseModel

logger = structlog.get_logger()


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker:
    """Simple circuit breaker implementation."""
    
    def __init__(self, threshold: int = 5, timeout: int = 30):
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def record_success(self):
        """Record successful request."""
        self.failures = 0
        self.state = CircuitState.CLOSED
    
    def record_failure(self):
        """Record failed request."""
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.threshold:
            self.state = CircuitState.OPEN
            logger.warning("circuit_breaker_opened", failures=self.failures)
    
    def can_attempt(self) -> bool:
        """Check if request can be attempted."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if timeout expired
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("circuit_breaker_half_open")
                return True
            return False
        
        # HALF_OPEN: allow one attempt
        return True


class EngineClient:
    """Async HTTP client for SGLang engine."""
    
    def __init__(
        self,
        base_url: str,
        timeout: int = 120,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 30
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = httpx.Timeout(connect=30.0, read=float(timeout), write=30.0, pool=5.0)
        self.max_retries = max_retries
        self.circuit_breaker = CircuitBreaker(circuit_breaker_threshold, circuit_breaker_timeout)
        
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
    
    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> httpx.Response:
        """Make HTTP request with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.request(method, f"{self.base_url}{endpoint}", **kwargs)
                response.raise_for_status()
                return response
            
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    backoff = 2 ** attempt
                    logger.warning(
                        "engine_request_retry",
                        attempt=attempt + 1,
                        backoff_s=backoff,
                        error=str(e)
                    )
                    await asyncio.sleep(backoff)
                continue
            
            except httpx.HTTPStatusError as e:
                # Don't retry 4xx errors
                if 400 <= e.response.status_code < 500:
                    raise
                last_exception = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
        
        raise last_exception
    
    async def health_check(self) -> bool:
        """Check if engine is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.error("health_check_failed", error=str(e))
            return False
    
    async def generate(
        self,
        text: str,
        sampling_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call native /generate endpoint for probe."""
        if not self.circuit_breaker.can_attempt():
            raise RuntimeError("Circuit breaker is OPEN")
        
        payload = {
            "text": text,
            "sampling_params": sampling_params or {}
        }
        
        try:
            response = await self._request_with_retry("POST", "/generate", json=payload)
            self.circuit_breaker.record_success()
            return response.json()
        
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error("generate_request_failed", error=str(e))
            raise
    
    async def chat_completions(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ):
        """Call OpenAI-compatible /v1/chat/completions endpoint."""
        if not self.circuit_breaker.can_attempt():
            raise RuntimeError("Circuit breaker is OPEN")
        
        payload = {
            "model": "default",
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        try:
            if stream:
                # Return async generator for streaming
                async def stream_generator():
                    try:
                        async with self.client.stream(
                            "POST",
                            f"{self.base_url}/v1/chat/completions",
                            json=payload
                        ) as response:
                            response.raise_for_status()
                            async for chunk in response.aiter_text():
                                yield chunk
                        
                        # Only record success if stream completes without error
                        self.circuit_breaker.record_success()
                        
                    except Exception as e:
                        # Record failure on any streaming exception (timeout, disconnect, etc)
                        self.circuit_breaker.record_failure()
                        logger.error("stream_failed_mid_flight", error=str(e))
                        raise

                return stream_generator()
            
            else:
                response = await self._request_with_retry("POST", "/v1/chat/completions", json=payload)
                self.circuit_breaker.record_success()
                return response.json()
        
        except Exception as e:
            # Catch errors during setup or non-streaming request
            self.circuit_breaker.record_failure()
            logger.error("chat_completions_failed", error=str(e))
            raise
