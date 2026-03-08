"""
Search session cache for infinite-scroll pagination.

Caches merged+reranked search results server-side so that page 2+
can be served instantly from cache (~1ms) instead of re-running the
full hybrid pipeline (~15s).

Architecture:
- Thread-safe in-memory dict (production: swap for Redis).
- 10-minute TTL per session (search sessions are short-lived).
- Background cleanup every 2 minutes.
- Each cached session stores the full ranked result list, facets,
  follow-ups, and metadata needed to reconstruct paginated responses.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Cursor helpers
# ---------------------------------------------------------------------------

def encode_cursor(page: int, offset: int) -> str:
    """Encode pagination state as an opaque base64 cursor."""
    payload = json.dumps({"p": page, "o": offset}, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode()


def decode_cursor(cursor: str) -> dict:
    """Decode a cursor back to pagination state.

    Returns {"p": <page>, "o": <offset>} or raises ValueError.
    """
    try:
        raw = base64.urlsafe_b64decode(cursor.encode()).decode()
        data = json.loads(raw)
        if "p" not in data or "o" not in data:
            raise ValueError("missing fields")
        return data
    except Exception as exc:
        raise ValueError(f"Invalid search cursor: {exc}") from exc


# ---------------------------------------------------------------------------
# Cached session entry
# ---------------------------------------------------------------------------

@dataclass
class SearchSessionEntry:
    """A single cached search session."""

    session_id: str
    query: str
    intent: str
    sort_by: str
    all_results: List[dict]          # Full merged+reranked result dicts
    facets: Optional[Dict[str, Any]] = None
    follow_ups: Optional[List[Any]] = None
    applied_filters: Optional[Dict[str, Any]] = None
    answered_dimensions: Optional[List[str]] = None
    timing: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    page_size: int = 50

    @property
    def total_results(self) -> int:
        return len(self.all_results)

    def get_page(self, page: int) -> tuple:
        """Return (page_results, has_more, next_cursor).

        ``page`` is 1-indexed.
        """
        start = (page - 1) * self.page_size
        end = start + self.page_size
        page_results = self.all_results[start:end]
        has_more = end < len(self.all_results)
        next_cursor = encode_cursor(page + 1, end) if has_more else None
        return page_results, has_more, next_cursor

    def is_expired(self, ttl_seconds: int) -> bool:
        return (time.time() - self.created_at) > ttl_seconds


# ---------------------------------------------------------------------------
# Session cache
# ---------------------------------------------------------------------------

_DEFAULT_TTL = 600        # 10 minutes
_CLEANUP_INTERVAL = 120   # 2 minutes


class SearchSessionCache:
    """Thread-safe in-memory cache for search pagination sessions."""

    _instance: Optional["SearchSessionCache"] = None
    _lock_cls = threading.Lock()

    def __init__(self, ttl_seconds: int = _DEFAULT_TTL):
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._store: Dict[str, SearchSessionEntry] = {}
        self._last_cleanup = time.time()

    # -- Singleton -----------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "SearchSessionCache":
        if cls._instance is None:
            with cls._lock_cls:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # -- Public API ----------------------------------------------------------

    def generate_session_id(self) -> str:
        """Generate a unique search session ID."""
        return f"ss_{uuid.uuid4().hex[:12]}"

    def store(self, entry: SearchSessionEntry) -> None:
        """Cache a search session entry."""
        with self._lock:
            self._store[entry.session_id] = entry
            self._maybe_cleanup()
        logger.info(
            "Cached search session",
            session_id=entry.session_id,
            total_results=entry.total_results,
            page_size=entry.page_size,
            pages_available=max(1, -(-entry.total_results // entry.page_size)),
        )

    def get(self, session_id: str) -> Optional[SearchSessionEntry]:
        """Retrieve a cached session (None if missing or expired)."""
        with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                return None
            if entry.is_expired(self._ttl):
                del self._store[session_id]
                logger.info("Search session expired", session_id=session_id)
                return None
            return entry

    def delete(self, session_id: str) -> bool:
        """Explicitly remove a session. Returns True if it existed."""
        with self._lock:
            return self._store.pop(session_id, None) is not None

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    # -- Internals -----------------------------------------------------------

    def _maybe_cleanup(self) -> None:
        """Evict expired sessions periodically (called under lock)."""
        now = time.time()
        if now - self._last_cleanup < _CLEANUP_INTERVAL:
            return
        self._last_cleanup = now
        expired = [
            sid for sid, entry in self._store.items()
            if entry.is_expired(self._ttl)
        ]
        for sid in expired:
            del self._store[sid]
        if expired:
            logger.info(
                "Cleaned up expired search sessions",
                evicted=len(expired),
                remaining=len(self._store),
            )
