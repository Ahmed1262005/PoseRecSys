"""
V3 Session Store — Redis-backed (production) and in-memory (testing).

Stores session profiles, shown ID sets, and candidate pools.
Redis keys are mode-specific: v3:session:{session_id}:pool:{mode}

See docs/V3_FEED_ARCHITECTURE_PLAN.md §8 for specification.
"""

import json
import logging
import os
import time
from threading import Lock
from typing import Any, Dict, List, Optional, Set

from recs.v3.models import CandidatePool, SessionProfile

logger = logging.getLogger(__name__)

# TTLs
SESSION_TTL = 86400   # 24 hours
POOL_TTL = 7200       # 2 hours

CATALOG_VERSION_KEY = "v3:catalog:version"


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _state_key(session_id: str) -> str:
    return f"v3:session:{session_id}:state"


def _shown_key(session_id: str) -> str:
    return f"v3:session:{session_id}:shown"


def _pool_key(session_id: str, mode: str) -> str:
    return f"v3:session:{session_id}:pool:{mode}"


def _version_key(session_id: str, mode: str) -> str:
    return f"v3:session:{session_id}:version:{mode}"


# =========================================================================
# In-Memory Store (testing / development)
# =========================================================================

class InMemorySessionStore:
    """Thread-safe in-memory session store for development and testing."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: Dict[str, dict] = {}
        self._shown: Dict[str, Set[str]] = {}
        self._pools: Dict[str, dict] = {}
        self._catalog_version: str = ""

    def get_or_create_session(
        self, session_id: str, user_id: str
    ) -> SessionProfile:
        with self._lock:
            key = _state_key(session_id)
            if key in self._sessions:
                return SessionProfile.from_dict(self._sessions[key])
            session = SessionProfile(session_id=session_id, user_id=user_id)
            self._sessions[key] = session.to_dict()
            return session

    def save_session(self, session_id: str, session: SessionProfile) -> None:
        with self._lock:
            self._sessions[_state_key(session_id)] = session.to_dict()

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(_state_key(session_id), None)
            self._shown.pop(_shown_key(session_id), None)
            # Remove all pools for this session
            prefix = f"v3:session:{session_id}:pool:"
            keys_to_remove = [k for k in self._pools if k.startswith(prefix)]
            for k in keys_to_remove:
                self._pools.pop(k, None)

    def load_shown_set(self, session_id: str) -> Set[str]:
        with self._lock:
            key = _shown_key(session_id)
            return set(self._shown.get(key, set()))

    def add_shown(self, session_id: str, item_ids: Set[str]) -> None:
        if not item_ids:
            return
        with self._lock:
            key = _shown_key(session_id)
            if key not in self._shown:
                self._shown[key] = set()
            self._shown[key].update(item_ids)

    def get_shown_count(self, session_id: str) -> int:
        with self._lock:
            return len(self._shown.get(_shown_key(session_id), set()))

    def get_pool(
        self, session_id: str, mode: str
    ) -> Optional[CandidatePool]:
        with self._lock:
            key = _pool_key(session_id, mode)
            data = self._pools.get(key)
            if data is None:
                return None
            return CandidatePool.from_dict(data)

    def save_pool(self, session_id: str, pool: CandidatePool) -> None:
        with self._lock:
            key = _pool_key(session_id, pool.mode)
            self._pools[key] = pool.to_dict()

    def delete_pool(self, session_id: str, mode: str) -> None:
        with self._lock:
            self._pools.pop(_pool_key(session_id, mode), None)

    def get_catalog_version(self) -> str:
        return self._catalog_version

    def set_catalog_version(self, version: str) -> None:
        self._catalog_version = version

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "sessions": len(self._sessions),
                "pools": len(self._pools),
                "backend": "memory",
            }


# =========================================================================
# Redis-backed Store (production)
# =========================================================================

class V3SessionStore:
    """
    Redis-backed session store for V3 feed.

    Session state / scoring profiles: 24h TTL
    Candidate pools: 2h TTL
    Shown IDs: Redis SET with 24h TTL
    """

    def __init__(self, redis_url: str) -> None:
        import redis
        self._redis = redis.from_url(redis_url, decode_responses=True)
        logger.info("V3SessionStore connected to Redis")

    def get_or_create_session(
        self, session_id: str, user_id: str
    ) -> SessionProfile:
        key = _state_key(session_id)
        raw = self._redis.get(key)
        if raw:
            return SessionProfile.from_dict(json.loads(raw))
        session = SessionProfile(session_id=session_id, user_id=user_id)
        self.save_session(session_id, session)
        return session

    def save_session(self, session_id: str, session: SessionProfile) -> None:
        key = _state_key(session_id)
        self._redis.set(key, json.dumps(session.to_dict()), ex=SESSION_TTL)

    def delete_session(self, session_id: str) -> None:
        self._redis.delete(_state_key(session_id))
        self._redis.delete(_shown_key(session_id))
        # Scan and delete pool keys
        pattern = f"v3:session:{session_id}:pool:*"
        cursor = 0
        while True:
            cursor, keys = self._redis.scan(cursor, match=pattern, count=100)
            if keys:
                self._redis.delete(*keys)
            if cursor == 0:
                break

    def load_shown_set(self, session_id: str) -> Set[str]:
        key = _shown_key(session_id)
        return self._redis.smembers(key) or set()

    def add_shown(self, session_id: str, item_ids: Set[str]) -> None:
        if not item_ids:
            return
        key = _shown_key(session_id)
        self._redis.sadd(key, *item_ids)
        self._redis.expire(key, SESSION_TTL)

    def get_shown_count(self, session_id: str) -> int:
        return self._redis.scard(_shown_key(session_id)) or 0

    def get_pool(
        self, session_id: str, mode: str
    ) -> Optional[CandidatePool]:
        key = _pool_key(session_id, mode)
        raw = self._redis.get(key)
        if raw is None:
            return None
        return CandidatePool.from_dict(json.loads(raw))

    def save_pool(self, session_id: str, pool: CandidatePool) -> None:
        key = _pool_key(session_id, pool.mode)
        self._redis.set(key, pool.to_json(), ex=POOL_TTL)

    def delete_pool(self, session_id: str, mode: str) -> None:
        self._redis.delete(_pool_key(session_id, mode))

    def get_catalog_version(self) -> str:
        return self._redis.get(CATALOG_VERSION_KEY) or ""

    def set_catalog_version(self, version: str) -> None:
        self._redis.set(CATALOG_VERSION_KEY, version)

    def get_stats(self) -> dict:
        session_count = 0
        pool_count = 0
        cursor = 0
        while True:
            cursor, keys = self._redis.scan(
                cursor, match="v3:session:*:state", count=100
            )
            session_count += len(keys)
            if cursor == 0:
                break
        cursor = 0
        while True:
            cursor, keys = self._redis.scan(
                cursor, match="v3:session:*:pool:*", count=100
            )
            pool_count += len(keys)
            if cursor == 0:
                break
        return {
            "sessions": session_count,
            "pools": pool_count,
            "backend": "redis",
        }


# =========================================================================
# Factory
# =========================================================================

def get_session_store(backend: str = "auto"):
    """
    Factory: create session store.

    backend: "auto" (try Redis, fall back to memory), "redis", "memory"
    """
    if backend == "memory":
        return InMemorySessionStore()

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    if backend == "redis":
        return V3SessionStore(redis_url)

    # auto: try Redis, fall back to memory
    try:
        store = V3SessionStore(redis_url)
        store._redis.ping()
        return store
    except Exception as e:
        logger.warning("Redis unavailable (%s), falling back to in-memory store", e)
        return InMemorySessionStore()
