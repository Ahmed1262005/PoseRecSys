"""
FastAPI middleware for request tracing and logging.

This module provides middleware that:
- Generates unique request IDs for tracing
- Logs request/response information
- Binds context for correlation
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from core.logging import bind_context, clear_context, get_logger


logger = get_logger(__name__)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds request tracing and logging.
    
    Features:
    - Generates unique request_id for each request
    - Logs request start/end with timing
    - Binds context for all logs during request processing
    - Adds X-Request-ID header to response
    
    Usage:
        from core.middleware import RequestTracingMiddleware
        
        app = FastAPI()
        app.add_middleware(RequestTracingMiddleware)
    """
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        
        # Bind context for all logs during this request
        bind_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        
        # Extract user_id if present in query params
        user_id = request.query_params.get("user_id")
        if user_id:
            bind_context(user_id=user_id)
        
        start_time = time.perf_counter()
        
        # Log request start
        logger.info(
            "Request started",
            query_params=dict(request.query_params) if request.query_params else None,
        )
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log request completion
            logger.info(
                "Request completed",
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            logger.error(
                "Request failed",
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=round(duration_ms, 2),
            )
            raise
            
        finally:
            # Clear context to prevent leaking to next request
            clear_context()


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Simple middleware that adds X-Response-Time header.
    
    Lighter weight than RequestTracingMiddleware when you don't need full logging.
    """
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        start_time = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        return response
