"""
Search session cache for "extend search" pagination.

Instead of caching stale results, caches the *plan state* from page 1:
- LLM planner outputs (intent, filters, algolia_query, semantic queries)
- Pre-computed FashionCLIP embeddings (numpy arrays)
- Algolia filter strings, RRF weights, reranker config
- Seen product IDs (grows each page)

Page 2+ reuses the cached plan to extend the search:
- Algolia: native page=N pagination (same query/filters)
- Semantic: reuse cached embeddings + exclude seen IDs → fresh pgvector results
- RRF merge + rerank on fresh candidates

This gives ~2-3s page 2+ instead of 12-15s (re-running full pipeline).

Architecture:
- Thread-safe in-memory dict (production: swap for Redis).
- 10-minute TTL per session.
- Background cleanup every 2 minutes.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Cursor helpers
# ---------------------------------------------------------------------------

def encode_cursor(page: int) -> str:
    """Encode pagination state as an opaque base64 cursor."""
    payload = json.dumps({"p": page}, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode()


def decode_cursor(cursor: str) -> dict:
    """Decode a cursor back to pagination state.

    Returns {"p": <page>} or raises ValueError.
    """
    try:
        raw = base64.urlsafe_b64decode(cursor.encode()).decode()
        data = json.loads(raw)
        if "p" not in data:
            raise ValueError("missing 'p' field")
        return data
    except Exception as exc:
        raise ValueError(f"Invalid search cursor: {exc}") from exc


# ---------------------------------------------------------------------------
# Cached session entry — stores plan state, not results
# ---------------------------------------------------------------------------

@dataclass
class SearchSessionEntry:
    """Cached plan state from page 1 for extend-search pagination."""

    session_id: str
    query: str
    intent: str                         # "exact" / "specific" / "vague"
    sort_by: str

    # Algolia state (for native pagination)
    algolia_query: str = ""
    algolia_filters: str = ""
    algolia_optional_filters: Optional[List[str]] = None

    # Semantic state (for extend search with exclude_ids)
    semantic_queries: Optional[List[str]] = None
    semantic_embeddings: Optional[List[np.ndarray]] = None  # pre-computed
    semantic_request_updates: Optional[Dict[str, Any]] = None  # relaxed filter overrides

    # RRF weights
    algolia_weight: float = 0.5
    semantic_weight: float = 0.5

    # Reranker config carried from page 1
    rerank_kwargs: Optional[Dict[str, Any]] = None

    # Pagination tracking
    seen_product_ids: Set[str] = field(default_factory=set)
    algolia_page: int = 0               # last Algolia page fetched (0-indexed, page 1 uses 0)
    page_size: int = 50
    fetch_size: int = 150               # per-source fetch size

    # Response metadata from page 1 (returned on all pages)
    facets: Optional[Dict[str, Any]] = None
    follow_ups: Optional[List[Any]] = None
    applied_filters: Optional[Dict[str, Any]] = None
    answered_dimensions: Optional[List[str]] = None

    # Algolia catalog count (nbHits) — surfaced as total_results on all pages
    algolia_total_hits: int = 0

    # Post-filter criteria for endless semantic search (page 2+).
    # Structural filters from the planner that semantic results must satisfy.
    post_filter_criteria: Optional[Dict[str, Any]] = None

    # Flags
    skip_algolia: bool = False          # empty query + no brand filter
    use_attribute_search: bool = False  # attribute-filtered semantic path

    # Attribute search state (if applicable)
    attribute_filters: Optional[Any] = None  # AttributeFilters object

    created_at: float = field(default_factory=time.time)

    def add_seen_ids(self, product_ids: List[str]) -> None:
        """Add product IDs to the seen set."""
        self.seen_product_ids.update(product_ids)

    def next_algolia_page(self) -> int:
        """Return the next Algolia page (0-indexed) and increment."""
        self.algolia_page += 1
        return self.algolia_page

    def is_expired(self, ttl_seconds: int) -> bool:
        return (time.time() - self.created_at) > ttl_seconds


# ---------------------------------------------------------------------------
# Session cache
# ---------------------------------------------------------------------------

_DEFAULT_TTL = 1800       # 30 minutes
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
        self.last_get_status: str = ""  # "hit", "miss_not_found", "miss_expired"

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
            "Cached search session (plan state)",
            session_id=entry.session_id,
            intent=entry.intent,
            seen_ids=len(entry.seen_product_ids),
            has_embeddings=entry.semantic_embeddings is not None,
            semantic_queries=len(entry.semantic_queries or []),
        )

    def get(self, session_id: str) -> Optional[SearchSessionEntry]:
        """Retrieve a cached session (None if missing or expired).

        Returns a (entry, status) style result via the entry itself.
        The ``last_get_status`` attribute is set for diagnostics.
        """
        with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                self.last_get_status = "miss_not_found"
                logger.info(
                    "Search session not found in cache",
                    session_id=session_id,
                    cache_size=len(self._store),
                )
                return None
            if entry.is_expired(self._ttl):
                del self._store[session_id]
                self.last_get_status = "miss_expired"
                logger.info(
                    "Search session expired",
                    session_id=session_id,
                    age_s=round(time.time() - entry.created_at),
                    ttl_s=self._ttl,
                )
                return None
            self.last_get_status = "hit"
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
