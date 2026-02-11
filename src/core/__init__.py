"""
Core module for cross-cutting concerns.

This module provides:
- Structured logging configuration
- Request tracing middleware
- Authentication utilities
- Common utilities
"""

from core.logging import configure_logging, get_logger
from core.auth import require_auth, get_current_user, SupabaseUser
from core.utils import convert_numpy, normalize_string_set, safe_get

__all__ = [
    "configure_logging",
    "get_logger",
    "require_auth",
    "get_current_user",
    "SupabaseUser",
    "convert_numpy",
    "normalize_string_set",
    "safe_get",
]
