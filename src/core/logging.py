"""
Structured logging configuration using structlog.

This module provides a consistent logging setup across the application.
It supports both development (colored console) and production (JSON) output.

Usage:
    from core.logging import configure_logging, get_logger
    
    # Configure once at startup
    configure_logging(json_logs=False)  # Development
    configure_logging(json_logs=True)   # Production
    
    # Get a logger
    logger = get_logger(__name__)
    
    # Log with context
    logger.info("Processing request", user_id="123", action="feed")
    logger.error("Failed to fetch", error=str(e), product_id="abc")
"""

import logging
import sys
from typing import Any, Optional

import structlog
from structlog.types import Processor


def configure_logging(
    json_logs: bool = False,
    log_level: str = "INFO",
    include_timestamp: bool = True,
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        json_logs: If True, output JSON format (for production).
                   If False, output colored console format (for development).
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        include_timestamp: Whether to include timestamp in logs
    """
    # Common processors for all configurations
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if include_timestamp:
        shared_processors.insert(0, structlog.processors.TimeStamper(fmt="iso"))
    
    if json_logs:
        # Production: JSON output
        shared_processors.append(structlog.processors.format_exc_info)
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        # Development: Colored console output
        shared_processors.append(structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        ))
    
    # Configure structlog
    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__). If None, returns root logger.
        
    Returns:
        A structlog BoundLogger instance
        
    Usage:
        logger = get_logger(__name__)
        logger.info("Message", key="value")
    """
    return structlog.get_logger(name)


# Convenience function for adding context to all subsequent logs
def bind_context(**kwargs: Any) -> None:
    """
    Bind context variables to all subsequent logs in the current context.
    
    Useful for adding request-scoped context like user_id, request_id.
    
    Args:
        **kwargs: Key-value pairs to bind
        
    Usage:
        bind_context(user_id="123", request_id="abc")
        logger.info("Processing")  # Will include user_id and request_id
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """
    Clear all bound context variables.
    
    Call this at the end of request processing to avoid context leaking.
    """
    structlog.contextvars.clear_contextvars()


def unbind_context(*keys: str) -> None:
    """
    Unbind specific context variables.
    
    Args:
        *keys: Keys to unbind
    """
    structlog.contextvars.unbind_contextvars(*keys)


# Pre-configured loggers for common use cases
class LoggerMixin:
    """
    Mixin class that provides a logger property.
    
    Usage:
        class MyService(LoggerMixin):
            def process(self):
                self.logger.info("Processing")
    """
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        return get_logger(self.__class__.__name__)
