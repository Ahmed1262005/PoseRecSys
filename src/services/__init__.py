"""
Services module for business logic.

Provides session management for style learning sessions.
"""

from services.session_manager import (
    SessionManager,
    get_women_session_manager,
    get_unified_session_manager,
    get_men_session_manager,
)

__all__ = [
    "SessionManager",
    "get_women_session_manager",
    "get_unified_session_manager",
    "get_men_session_manager",
]
