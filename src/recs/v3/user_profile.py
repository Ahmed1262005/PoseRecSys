"""
V3 User Profile Loader — wraps CandidateSelectionModule with LRU cache.

Caches UserState objects in-process to avoid repeated DB loads within
the same server lifetime.
"""

import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Cache configuration
CACHE_TTL = 300       # 5 minutes
CACHE_MAX_SIZE = 500  # Max cached users


@dataclass
class _CacheEntry:
    """Cached UserState with expiry timestamp."""
    user_state: Any
    expires_at: float


class UserProfileLoader:
    """
    Loads user profiles via CandidateSelectionModule.load_user_state()
    with an in-process LRU cache (TTL 5 min, max 500 entries).
    """

    def __init__(self, supabase_client: Any = None) -> None:
        self._supabase = supabase_client
        self._csm = None
        self._cache: Dict[str, _CacheEntry] = {}
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def _get_csm(self):
        """Lazy-init CandidateSelectionModule."""
        if self._csm is None:
            from recs.candidate_selection import CandidateSelectionModule
            self._csm = CandidateSelectionModule(self._supabase)
        return self._csm

    def load(self, user_id: str) -> Any:
        """Load user state, using cache if available."""
        # Check cache
        cached = self._cache_get(user_id)
        if cached is not None:
            return cached

        # Cache miss — load from DB
        try:
            csm = self._get_csm()
            user_state = csm.load_user_state(user_id)
        except Exception as e:
            logger.error("Failed to load user profile for %s: %s", user_id, e)
            return None

        self._cache_put(user_id, user_state)
        return user_state

    def _cache_get(self, user_id: str) -> Optional[Any]:
        """Get from cache if present and not expired."""
        with self._lock:
            entry = self._cache.get(user_id)
            if entry is None:
                self._misses += 1
                return None
            if time.time() > entry.expires_at:
                # Expired
                del self._cache[user_id]
                self._misses += 1
                return None
            self._hits += 1
            return entry.user_state

    def _cache_put(self, user_id: str, user_state: Any) -> None:
        """Store in cache, evicting oldest if at capacity."""
        with self._lock:
            # Evict if full
            if len(self._cache) >= CACHE_MAX_SIZE and user_id not in self._cache:
                # Remove the oldest entry (first inserted)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[user_id] = _CacheEntry(
                user_state=user_state,
                expires_at=time.time() + CACHE_TTL,
            )

    def invalidate(self, user_id: str) -> None:
        """Remove a specific user from cache."""
        with self._lock:
            self._cache.pop(user_id, None)

    def clear_cache(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def cache_stats(self) -> Dict[str, int]:
        """Return cache hit/miss statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "max_size": CACHE_MAX_SIZE,
                "ttl": CACHE_TTL,
            }
