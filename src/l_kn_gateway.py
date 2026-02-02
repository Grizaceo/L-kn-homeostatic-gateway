"""
L-kn Gateway - FastAPI Application
OpenAI-compatible gateway with homeostatic decision-making.
"""

import os
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import orjson
import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from engine_client import EngineClient
from homeostatic import HomeostaticSystem, HomeostaticConfig
from utils import RequestContextMiddleware, error_response, LatencyTracker, log_request

logger = structlog.get_logger()


class Settings(BaseSettings):
    """Application settings from environment."""
    engine_host: str = "127.0.0.1"
    engine_port: int = 30000
    engine_timeout: int = 120
    
    gateway_host: str = "127.0.0.1"
    gateway_port: int = 8000
    gateway_log_level: str = "INFO"
    
    lkn_entropy_threshold: float = 0.6
    lkn_probe_top_k: int = 10
    lkn_analytic_system_prompt: str = "Think carefully, verify your assumptions, and reason step-by-step before answering."
    lkn_mode: str = "homeostatic"  # "homeostatic" or "passthrough"
    
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 30
    max_retries: int = 3
    
    class Config:
        env_file = "config/.env"
        env_file_encoding = "utf-8"


# Pydantic models for OpenAI API compatibility
class Message(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: List[Message]
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=512, ge=1)
    stream: bool = False


# Global state
settings = Settings()
engine_client: Optional[EngineClient] = None
homeostatic_system: Optional[HomeostaticSystem] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global engine_client, homeostatic_system
    
    # Startup
    logger.info("gateway_startup", version="0.1.0")
    
    engine_base_url = f"http://{settings.engine_host}:{settings.engine_port}"
    engine_client = EngineClient(
        base_url=engine_base_url,
        timeout=settings.engine_timeout,
        max_retries=settings.max_retries,
        circuit_breaker_threshold=settings.circuit_breaker_threshold,
        circuit_breaker_timeout=settings.circuit_breaker_timeout
    )
    
    # Initialize homeostatic system
    homeostatic_config = HomeostaticConfig(
        entropy_threshold=settings.lkn_entropy_threshold,
        probe_top_k=settings.lkn_probe_top_k,
        analytic_system_prompt=settings.lkn_analytic_system_prompt
    )
    homeostatic_system = HomeostaticSystem(homeostatic_config, engine_client)
    
    logger.info("gateway_ready", mode=settings.lkn_mode)
    
    yield
    
    # Shutdown
    logger.info("gateway_shutdown")
    if engine_client:
        await engine_client.close()


# Initialize FastAPI app
app = FastAPI(
    title="L-kn Gateway",
    description="Homeostatic inference gateway",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware (restricted to localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request context middleware
app.add_middleware(RequestContextMiddleware)


@app.get("/health")
async def health():
    """Health check endpoint."""
    engine_healthy = await engine_client.health_check() if engine_client else False
    
    return {
        "status": "healthy" if engine_healthy else "degraded",
        "engine": "connected" if engine_healthy else "disconnected",
        "mode": settings.lkn_mode
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """
    OpenAI-compatible chat completions endpoint with homeostatic intervention.
    """
    tracker = LatencyTracker()
    tracker.start()
    
    request_id = raw_request.scope.get("request_id", "unknown")
    
    try:
        # Convert Pydantic messages to dicts
        messages = [msg.model_dump() for msg in request.messages]
        
        # Validate non-empty messages
        if not messages:
            return error_response(
                400,
                "invalid_request",
                "Messages list cannot be empty",
                {}
            )
        
        # Homeostatic decision
        decision = None
        if settings.lkn_mode == "homeostatic":
            decision = await homeostatic_system.decide_mode(messages)
            
            # Apply intervention if needed
            if decision.intervention_applied:
                messages = homeostatic_system.apply_intervention(messages, decision)
        
        # Prepare kwargs for engine
        engine_kwargs = {}
        if request.temperature is not None:
            engine_kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            engine_kwargs["top_p"] = request.top_p
        if request.max_tokens is not None:
            engine_kwargs["max_tokens"] = request.max_tokens
        
        # Call engine
        if request.stream:
            # Check circuit breaker before streaming
            if not engine_client.circuit_breaker.can_attempt():
                return error_response(
                    503,
                    "service_unavailable",
                    "Engine is temporarily unavailable (circuit breaker open)",
                    {"retry_after": settings.circuit_breaker_timeout}
                )
            
            # Streaming response
            async def generate_stream():
                stream_successful = False
                try:
                    stream_gen = await engine_client.chat_completions(
                        messages=messages,
                        stream=True,
                        **engine_kwargs
                    )
                    
                    async for chunk in stream_gen:
                        yield chunk
                        stream_successful = True
                    
                    # Mark as successful after complete stream
                    if stream_successful and engine_client:
                        engine_client.circuit_breaker.record_success()
                
                except Exception as e:
                    logger.error("stream_error", error=str(e), error_type=type(e).__name__)
                    error_data = {
                        "error": {
                            "type": "stream_error",
                            "message": str(e)
                        }
                    }
                    yield f"data: {orjson.dumps(error_data).decode()}\n\n"
                    yield "data: [DONE]\n\n"
            
            # Log before streaming starts
            mode = decision.mode if decision else "passthrough"
            entropy_norm = decision.entropy_norm if decision else None
            log_request(request_id, mode, entropy_norm, tracker.elapsed_ms(), "streaming")
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream"
            )
        
        else:
            # Non-streaming response
            response = await engine_client.chat_completions(
                messages=messages,
                stream=False,
                **engine_kwargs
            )
            
            # Log completion
            mode = decision.mode if decision else "passthrough"
            entropy_norm = decision.entropy_norm if decision else None
            log_request(request_id, mode, entropy_norm, tracker.elapsed_ms())
            
            return JSONResponse(content=response)
    
    except RuntimeError as e:
        if "Circuit breaker" in str(e):
            return error_response(
                503,
                "service_unavailable",
                "Engine is temporarily unavailable (circuit breaker open)",
                {"retry_after": settings.circuit_breaker_timeout}
            )
        raise
    
    except Exception as e:
        logger.error("request_failed", error=str(e), error_type=type(e).__name__)
        return error_response(
            500,
            "internal_error",
            "Request processing failed",
            {"error": str(e)}
        )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "L-kn Gateway",
        "version": "0.1.0",
        "mode": settings.lkn_mode,
        "endpoints": {
            "health": "/health",
            "chat_completions": "/v1/chat/completions"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "l_kn_gateway:app",
        host=settings.gateway_host,
        port=settings.gateway_port,
        log_level=settings.gateway_log_level.lower(),
        reload=False
    )
