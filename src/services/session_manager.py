"""
Session manager for in-memory session state.

This service manages session state for style learning sessions,
including preferences, current items, and test info.

In production, this should be backed by Redis for horizontal scaling.
"""

from typing import Any, Dict, Generic, List, Optional, TypeVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading

from core.logging import LoggerMixin


T = TypeVar("T")


@dataclass
class SessionData(Generic[T]):
    """Container for session data with metadata."""
    
    data: T
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: int = 86400  # 24 hours default
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        expiry = self.created_at + timedelta(seconds=self.ttl_seconds)
        return datetime.utcnow() > expiry
    
    def touch(self) -> None:
        """Update the last accessed time."""
        self.updated_at = datetime.utcnow()


class SessionManager(LoggerMixin):
    """
    Thread-safe in-memory session manager.
    
    Manages multiple types of session data:
    - preferences: User preference objects
    - current_items: Currently displayed items
    - test_info: Current test state
    
    Usage:
        manager = SessionManager()
        
        # Store session data
        manager.set_preferences("user_123", prefs_object)
        manager.set_current_items("user_123", ["item1", "item2"])
        
        # Retrieve session data
        prefs = manager.get_preferences("user_123")
        items = manager.get_current_items("user_123")
        
        # Delete session
        manager.delete_session("user_123")
    """
    
    def __init__(self, ttl_seconds: int = 86400):
        """
        Initialize session manager.
        
        Args:
            ttl_seconds: Session TTL in seconds (default 24 hours)
        """
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        
        # Session stores by type
        self._preferences: Dict[str, SessionData] = {}
        self._current_items: Dict[str, SessionData[List[str]]] = {}
        self._test_info: Dict[str, SessionData[Dict[str, Any]]] = {}
    
    # =========================================================================
    # Preferences
    # =========================================================================
    
    def get_preferences(self, session_key: str) -> Optional[Any]:
        """
        Get preferences for a session.
        
        Args:
            session_key: Session identifier
            
        Returns:
            Preferences object or None if not found/expired
        """
        with self._lock:
            session = self._preferences.get(session_key)
            if session is None:
                return None
            if session.is_expired():
                del self._preferences[session_key]
                return None
            session.touch()
            return session.data
    
    def set_preferences(self, session_key: str, preferences: Any) -> None:
        """
        Set preferences for a session.
        
        Args:
            session_key: Session identifier
            preferences: Preferences object to store
        """
        with self._lock:
            self._preferences[session_key] = SessionData(
                data=preferences,
                ttl_seconds=self._ttl_seconds
            )
    
    def has_preferences(self, session_key: str) -> bool:
        """Check if session has active preferences."""
        return self.get_preferences(session_key) is not None
    
    # =========================================================================
    # Current Items
    # =========================================================================
    
    def get_current_items(self, session_key: str) -> Optional[List[str]]:
        """
        Get current items for a session.
        
        Args:
            session_key: Session identifier
            
        Returns:
            List of item IDs or None
        """
        with self._lock:
            session = self._current_items.get(session_key)
            if session is None:
                return None
            if session.is_expired():
                del self._current_items[session_key]
                return None
            session.touch()
            return session.data
    
    def set_current_items(self, session_key: str, items: List[str]) -> None:
        """
        Set current items for a session.
        
        Args:
            session_key: Session identifier
            items: List of item IDs
        """
        with self._lock:
            self._current_items[session_key] = SessionData(
                data=items,
                ttl_seconds=self._ttl_seconds
            )
    
    # =========================================================================
    # Test Info
    # =========================================================================
    
    def get_test_info(self, session_key: str) -> Optional[Dict[str, Any]]:
        """
        Get test info for a session.
        
        Args:
            session_key: Session identifier
            
        Returns:
            Test info dict or None
        """
        with self._lock:
            session = self._test_info.get(session_key)
            if session is None:
                return None
            if session.is_expired():
                del self._test_info[session_key]
                return None
            session.touch()
            return session.data
    
    def set_test_info(self, session_key: str, info: Dict[str, Any]) -> None:
        """
        Set test info for a session.
        
        Args:
            session_key: Session identifier
            info: Test info dict
        """
        with self._lock:
            self._test_info[session_key] = SessionData(
                data=info,
                ttl_seconds=self._ttl_seconds
            )
    
    # =========================================================================
    # Session Management
    # =========================================================================
    
    def delete_session(self, session_key: str) -> None:
        """
        Delete all data for a session.
        
        Args:
            session_key: Session identifier
        """
        with self._lock:
            self._preferences.pop(session_key, None)
            self._current_items.pop(session_key, None)
            self._test_info.pop(session_key, None)
    
    def clear_expired(self) -> int:
        """
        Clear all expired sessions.
        
        Returns:
            Number of sessions cleared
        """
        cleared = 0
        with self._lock:
            # Clear preferences
            expired_keys = [
                k for k, v in self._preferences.items() 
                if v.is_expired()
            ]
            for key in expired_keys:
                del self._preferences[key]
                cleared += 1
            
            # Clear current items
            expired_keys = [
                k for k, v in self._current_items.items() 
                if v.is_expired()
            ]
            for key in expired_keys:
                del self._current_items[key]
                cleared += 1
            
            # Clear test info
            expired_keys = [
                k for k, v in self._test_info.items() 
                if v.is_expired()
            ]
            for key in expired_keys:
                del self._test_info[key]
                cleared += 1
        
        if cleared:
            self.logger.info("Cleared expired sessions", count=cleared)
        return cleared
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get session statistics.
        
        Returns:
            Dict with counts by session type
        """
        with self._lock:
            return {
                "preferences": len(self._preferences),
                "current_items": len(self._current_items),
                "test_info": len(self._test_info),
            }


# Global session managers by domain
# These are singletons to maintain state across requests

_women_sessions: Optional[SessionManager] = None
_unified_sessions: Optional[SessionManager] = None
_men_sessions: Optional[SessionManager] = None


def get_women_session_manager() -> SessionManager:
    """Get the women's fashion session manager singleton."""
    global _women_sessions
    if _women_sessions is None:
        _women_sessions = SessionManager()
    return _women_sessions


def get_unified_session_manager() -> SessionManager:
    """Get the unified (gender-aware) session manager singleton."""
    global _unified_sessions
    if _unified_sessions is None:
        _unified_sessions = SessionManager()
    return _unified_sessions


def get_men_session_manager() -> SessionManager:
    """Get the men's fashion session manager singleton."""
    global _men_sessions
    if _men_sessions is None:
        _men_sessions = SessionManager()
    return _men_sessions
