"""
Session State Service for Endless Scroll V2 (Keyset Pagination)

Manages per-session state for recommendation pagination:
- seen_item_ids: Items already displayed (authoritative dedup via Redis SET)
- keyset_cursor: (score, item_id) for O(1) keyset pagination
- feed_version: Snapshot version for stable ordering within session
- user_signals: Recent interactions (views, clicks, skips)

Supports two backends:
1. InMemory: For development/testing (default)
2. Redis: For production (when redis package is available)

V2 Correctness Guarantees:
A. No duplicates within a session (authoritative seen tracking)
B. Stable ordering within session (feed versioning)
C. Graceful degradation (Redis unavailable)
D. Session isolation
"""

import os
import uuid
import json
import time
import base64
import hashlib
from typing import Set, List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock


# =============================================================================
# Session State Data Model
# =============================================================================

@dataclass
class KeysetCursor:
    """
    Keyset cursor for O(1) pagination.

    Instead of tracking seen_ids array (O(n) exclusion), we use a cursor
    based on the last item's (score, id) for constant-time pagination.
    """
    score: float
    item_id: str
    page: int = 0  # Page number for reference

    def to_dict(self) -> Dict[str, Any]:
        return {"score": self.score, "item_id": self.item_id, "page": self.page}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeysetCursor":
        return cls(
            score=float(data.get("score", 0.0)),
            item_id=str(data.get("item_id", "")),
            page=int(data.get("page", 0))
        )

    def encode(self) -> str:
        """Encode cursor as opaque base64 string for API response."""
        json_str = json.dumps(self.to_dict())
        return base64.b64encode(json_str.encode('utf-8')).decode('utf-8')

    @classmethod
    def decode(cls, encoded: str) -> Optional["KeysetCursor"]:
        """Decode cursor from API request. Returns None if invalid."""
        try:
            json_str = base64.b64decode(encoded.encode('utf-8')).decode('utf-8')
            data = json.loads(json_str)
            return cls.from_dict(data)
        except Exception:
            return None


@dataclass
class FeedVersion:
    """
    Feed version for stable ordering within a session.

    Stores the scoring parameters used at session start so that
    subsequent pages use the same ranking criteria.
    """
    version_id: str
    taste_vector_hash: Optional[str] = None  # Hash of user's taste vector
    filters_hash: Optional[str] = None  # Hash of applied filters
    scoring_weights: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "taste_vector_hash": self.taste_vector_hash,
            "filters_hash": self.filters_hash,
            "scoring_weights": self.scoring_weights,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedVersion":
        return cls(
            version_id=data.get("version_id", ""),
            taste_vector_hash=data.get("taste_vector_hash"),
            filters_hash=data.get("filters_hash"),
            scoring_weights=data.get("scoring_weights", {}),
            created_at=data.get("created_at", time.time())
        )

    @staticmethod
    def generate_version_id() -> str:
        """Generate a unique version ID."""
        return f"v_{int(time.time())}_{uuid.uuid4().hex[:8]}"


@dataclass
class SessionState:
    """
    State for a user session (V2 with keyset pagination).

    Contains:
    - session_id: Unique session identifier
    - seen_item_ids: Set of items already shown (authoritative dedup)
    - cursor: Keyset cursor for O(1) pagination
    - feed_version: Version info for stable ordering
    - current_offset: Legacy offset (for backwards compatibility)
    - user_signals: User interaction signals
    """
    session_id: str
    seen_item_ids: Set[str] = field(default_factory=set)
    cursor: Optional[KeysetCursor] = None
    feed_version: Optional[FeedVersion] = None
    current_offset: int = 0  # Legacy - kept for backwards compatibility
    user_signals: Dict[str, List[str]] = field(default_factory=lambda: {"views": [], "clicks": [], "skips": []})
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "seen_item_ids": list(self.seen_item_ids),
            "cursor": self.cursor.to_dict() if self.cursor else None,
            "feed_version": self.feed_version.to_dict() if self.feed_version else None,
            "current_offset": self.current_offset,
            "user_signals": self.user_signals,
            "created_at": self.created_at,
            "last_access": self.last_access
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Deserialize from dictionary."""
        cursor_data = data.get("cursor")
        feed_version_data = data.get("feed_version")

        return cls(
            session_id=data["session_id"],
            seen_item_ids=set(data.get("seen_item_ids", [])),
            cursor=KeysetCursor.from_dict(cursor_data) if cursor_data else None,
            feed_version=FeedVersion.from_dict(feed_version_data) if feed_version_data else None,
            current_offset=data.get("current_offset", 0),
            user_signals=data.get("user_signals", {"views": [], "clicks": [], "skips": []}),
            created_at=data.get("created_at", time.time()),
            last_access=data.get("last_access", time.time())
        )


# =============================================================================
# In-Memory Backend (Default)
# =============================================================================

class InMemorySessionBackend:
    """
    In-memory session storage for development/testing.

    Note: Sessions are lost on server restart.
    Use Redis backend for production.
    """

    def __init__(self, ttl_seconds: int = 86400):
        """
        Initialize in-memory backend.

        Args:
            ttl_seconds: Time-to-live for sessions (default 24 hours)
        """
        self._sessions: Dict[str, SessionState] = {}
        self._lock = Lock()
        self._ttl = ttl_seconds
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # Cleanup every hour

    def _maybe_cleanup(self):
        """Remove expired sessions if cleanup interval has passed."""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = now

    def _cleanup_expired(self):
        """Remove sessions older than TTL."""
        cutoff = time.time() - self._ttl
        expired = [sid for sid, state in self._sessions.items()
                   if state.last_access < cutoff]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            print(f"[SessionState] Cleaned up {len(expired)} expired sessions")

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID."""
        self._maybe_cleanup()
        with self._lock:
            state = self._sessions.get(session_id)
            if state:
                state.last_access = time.time()
            return state

    def create_session(self, session_id: str) -> SessionState:
        """Create a new session."""
        with self._lock:
            state = SessionState(session_id=session_id)
            self._sessions[session_id] = state
            return state

    def get_or_create_session(self, session_id: str) -> SessionState:
        """Get existing session or create new one."""
        state = self.get_session(session_id)
        if state is None:
            state = self.create_session(session_id)
        return state

    def update_session(self, state: SessionState):
        """Update session state."""
        with self._lock:
            state.last_access = time.time()
            self._sessions[state.session_id] = state

    def delete_session(self, session_id: str):
        """Delete a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics."""
        return {
            "backend": "in_memory",
            "active_sessions": len(self._sessions),
            "ttl_seconds": self._ttl
        }


# =============================================================================
# Redis Backend (Optional - for production)
# =============================================================================

class RedisSessionBackend:
    """
    Redis-based session storage for production.

    Requires: pip install redis
    """

    def __init__(self, redis_url: str = None, ttl_seconds: int = 86400):
        """
        Initialize Redis backend.

        Args:
            redis_url: Redis connection URL (default from env REDIS_URL)
            ttl_seconds: Time-to-live for sessions (default 24 hours)
        """
        try:
            import redis
        except ImportError:
            raise ImportError("Redis backend requires 'redis' package. Install with: pip install redis")

        self._ttl = ttl_seconds
        redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._redis = redis.from_url(redis_url, decode_responses=True)

        # Test connection
        self._redis.ping()
        print(f"[SessionState] Connected to Redis: {redis_url.split('@')[-1]}")

    def _key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"session:{session_id}"

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID."""
        data = self._redis.get(self._key(session_id))
        if data:
            state = SessionState.from_dict(json.loads(data))
            # Update last access
            state.last_access = time.time()
            self._redis.setex(self._key(session_id), self._ttl, json.dumps(state.to_dict()))
            return state
        return None

    def create_session(self, session_id: str) -> SessionState:
        """Create a new session."""
        state = SessionState(session_id=session_id)
        self._redis.setex(self._key(session_id), self._ttl, json.dumps(state.to_dict()))
        return state

    def get_or_create_session(self, session_id: str) -> SessionState:
        """Get existing session or create new one."""
        state = self.get_session(session_id)
        if state is None:
            state = self.create_session(session_id)
        return state

    def update_session(self, state: SessionState):
        """Update session state."""
        state.last_access = time.time()
        self._redis.setex(self._key(state.session_id), self._ttl, json.dumps(state.to_dict()))

    def delete_session(self, session_id: str):
        """Delete a session."""
        self._redis.delete(self._key(session_id))

    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics."""
        # Count session keys
        cursor = 0
        count = 0
        while True:
            cursor, keys = self._redis.scan(cursor, match="session:*", count=1000)
            count += len(keys)
            if cursor == 0:
                break

        return {
            "backend": "redis",
            "active_sessions": count,
            "ttl_seconds": self._ttl
        }


# =============================================================================
# Session State Service (Main Interface)
# =============================================================================

class SessionStateService:
    """
    High-level session state service.

    Provides methods for:
    - Managing seen items (for exclusion)
    - Tracking pagination offset
    - Recording user signals

    Automatically selects backend:
    - Redis if available and REDIS_URL is set
    - In-memory otherwise
    """

    def __init__(self, backend: str = "auto", ttl_seconds: int = 86400):
        """
        Initialize session state service.

        Args:
            backend: "auto", "redis", or "memory"
            ttl_seconds: Session TTL (default 24 hours)
        """
        if backend == "auto":
            # Try Redis first, fall back to memory
            if os.getenv("REDIS_URL"):
                try:
                    self._backend = RedisSessionBackend(ttl_seconds=ttl_seconds)
                    print("[SessionState] Using Redis backend")
                except Exception as e:
                    print(f"[SessionState] Redis unavailable ({e}), using in-memory backend")
                    self._backend = InMemorySessionBackend(ttl_seconds=ttl_seconds)
            else:
                print("[SessionState] No REDIS_URL set, using in-memory backend")
                self._backend = InMemorySessionBackend(ttl_seconds=ttl_seconds)
        elif backend == "redis":
            self._backend = RedisSessionBackend(ttl_seconds=ttl_seconds)
        else:
            self._backend = InMemorySessionBackend(ttl_seconds=ttl_seconds)

    # =========================================================
    # Session ID Generation
    # =========================================================

    @staticmethod
    def generate_session_id() -> str:
        """Generate a new unique session ID."""
        return f"sess_{uuid.uuid4().hex[:12]}"

    # =========================================================
    # Seen Items Management
    # =========================================================

    def get_seen_items(self, session_id: str) -> Set[str]:
        """
        Get set of item IDs already shown in this session.

        Args:
            session_id: Session identifier

        Returns:
            Set of item IDs to exclude from future pages
        """
        state = self._backend.get_session(session_id)
        if state:
            return state.seen_item_ids.copy()
        return set()

    def add_seen_items(self, session_id: str, item_ids: List[str]):
        """
        Add items to the seen set for this session.

        Call this after returning items to the user.

        Args:
            session_id: Session identifier
            item_ids: List of item IDs that were just shown
        """
        state = self._backend.get_or_create_session(session_id)
        state.seen_item_ids.update(item_ids)
        self._backend.update_session(state)

    def get_seen_count(self, session_id: str) -> int:
        """Get count of items seen in this session."""
        state = self._backend.get_session(session_id)
        if state:
            return len(state.seen_item_ids)
        return 0

    # =========================================================
    # Offset Management
    # =========================================================

    def get_offset(self, session_id: str) -> int:
        """Get current SQL offset for this session."""
        state = self._backend.get_session(session_id)
        if state:
            return state.current_offset
        return 0

    def increment_offset(self, session_id: str, amount: int):
        """Increment the SQL offset for this session."""
        state = self._backend.get_or_create_session(session_id)
        state.current_offset += amount
        self._backend.update_session(state)

    def set_offset(self, session_id: str, offset: int):
        """Set the SQL offset for this session."""
        state = self._backend.get_or_create_session(session_id)
        state.current_offset = offset
        self._backend.update_session(state)

    # =========================================================
    # User Signals
    # =========================================================

    def record_signal(self, session_id: str, signal_type: str, item_id: str):
        """
        Record a user interaction signal.

        Args:
            session_id: Session identifier
            signal_type: "view", "click", or "skip"
            item_id: Item that was interacted with
        """
        if signal_type not in ("views", "clicks", "skips"):
            # Normalize signal type
            signal_type = {
                "view": "views",
                "click": "clicks",
                "skip": "skips"
            }.get(signal_type, signal_type)

        state = self._backend.get_or_create_session(session_id)
        if signal_type in state.user_signals:
            # Keep only last 100 signals per type
            if len(state.user_signals[signal_type]) >= 100:
                state.user_signals[signal_type] = state.user_signals[signal_type][-99:]
            state.user_signals[signal_type].append(item_id)
            self._backend.update_session(state)

    def get_signals(self, session_id: str) -> Dict[str, List[str]]:
        """Get all user signals for this session."""
        state = self._backend.get_session(session_id)
        if state:
            return state.user_signals.copy()
        return {"views": [], "clicks": [], "skips": []}

    # =========================================================
    # Keyset Cursor Management (V2)
    # =========================================================

    def get_cursor(self, session_id: str) -> Optional[KeysetCursor]:
        """
        Get the keyset cursor for this session.

        Args:
            session_id: Session identifier

        Returns:
            KeysetCursor with (score, item_id) for pagination, or None
        """
        state = self._backend.get_session(session_id)
        if state and state.cursor:
            return state.cursor
        return None

    def set_cursor(self, session_id: str, score: float, item_id: str, page: int = 0):
        """
        Set the keyset cursor for this session.

        Call this after returning items to update the pagination position.

        Args:
            session_id: Session identifier
            score: Score of the last item returned
            item_id: ID of the last item returned
            page: Current page number
        """
        state = self._backend.get_or_create_session(session_id)
        state.cursor = KeysetCursor(score=score, item_id=item_id, page=page)
        self._backend.update_session(state)

    def get_cursor_encoded(self, session_id: str) -> Optional[str]:
        """Get cursor as base64 encoded string for API response."""
        cursor = self.get_cursor(session_id)
        if cursor:
            return cursor.encode()
        return None

    def decode_cursor(self, encoded: str) -> Optional[KeysetCursor]:
        """Decode cursor from API request."""
        return KeysetCursor.decode(encoded)

    # =========================================================
    # Feed Version Management (V2)
    # =========================================================

    def get_feed_version(self, session_id: str) -> Optional[FeedVersion]:
        """
        Get the feed version for this session.

        Args:
            session_id: Session identifier

        Returns:
            FeedVersion with scoring parameters, or None
        """
        state = self._backend.get_session(session_id)
        if state and state.feed_version:
            return state.feed_version
        return None

    def set_feed_version(
        self,
        session_id: str,
        taste_vector: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
        scoring_weights: Optional[Dict[str, float]] = None
    ) -> FeedVersion:
        """
        Set the feed version for this session.

        Call this on the first page request to lock in scoring parameters.

        Args:
            session_id: Session identifier
            taste_vector: User's taste vector (will be hashed)
            filters: Applied filters (will be hashed)
            scoring_weights: Scoring weights used

        Returns:
            The created FeedVersion
        """
        state = self._backend.get_or_create_session(session_id)

        # Hash taste vector if provided
        taste_hash = None
        if taste_vector:
            taste_str = ",".join(f"{v:.6f}" for v in taste_vector[:10])  # First 10 values
            taste_hash = hashlib.md5(taste_str.encode()).hexdigest()[:16]

        # Hash filters if provided
        filters_hash = None
        if filters:
            filters_str = json.dumps(filters, sort_keys=True)
            filters_hash = hashlib.md5(filters_str.encode()).hexdigest()[:16]

        feed_version = FeedVersion(
            version_id=FeedVersion.generate_version_id(),
            taste_vector_hash=taste_hash,
            filters_hash=filters_hash,
            scoring_weights=scoring_weights or {}
        )

        state.feed_version = feed_version
        self._backend.update_session(state)

        return feed_version

    def has_feed_version(self, session_id: str) -> bool:
        """Check if session has a feed version (i.e., not first page)."""
        state = self._backend.get_session(session_id)
        return state is not None and state.feed_version is not None

    # =========================================================
    # Session Management
    # =========================================================

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get full session info for debugging."""
        state = self._backend.get_session(session_id)
        if state:
            info = {
                "session_id": state.session_id,
                "seen_count": len(state.seen_item_ids),
                "current_offset": state.current_offset,
                "signals": {
                    "views": len(state.user_signals.get("views", [])),
                    "clicks": len(state.user_signals.get("clicks", [])),
                    "skips": len(state.user_signals.get("skips", []))
                },
                "created_at": datetime.fromtimestamp(state.created_at).isoformat(),
                "last_access": datetime.fromtimestamp(state.last_access).isoformat(),
                "age_seconds": int(time.time() - state.created_at)
            }
            # V2: Add cursor info
            if state.cursor:
                info["cursor"] = {
                    "score": state.cursor.score,
                    "item_id": state.cursor.item_id,
                    "page": state.cursor.page
                }
            # V2: Add feed version info
            if state.feed_version:
                info["feed_version"] = {
                    "version_id": state.feed_version.version_id,
                    "created_at": datetime.fromtimestamp(state.feed_version.created_at).isoformat()
                }
            return info
        return None

    def clear_session(self, session_id: str):
        """Clear a session (for testing or reset)."""
        self._backend.delete_session(session_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return self._backend.get_stats()


# =============================================================================
# Testing
# =============================================================================

def test_session_state():
    """Test the session state service."""
    print("=" * 70)
    print("Testing Session State Service")
    print("=" * 70)

    # Initialize service (will use in-memory backend)
    service = SessionStateService(backend="memory")
    stats = service.get_stats()
    print(f"\n1. Initialized with backend: {stats['backend']}")

    # Test 2: Generate session ID
    session_id = SessionStateService.generate_session_id()
    print(f"\n2. Generated session ID: {session_id}")

    # Test 3: Add seen items
    print("\n3. Testing seen items...")
    items_page_1 = [f"item_{i}" for i in range(50)]
    service.add_seen_items(session_id, items_page_1)
    seen = service.get_seen_items(session_id)
    print(f"   Added 50 items, seen count: {len(seen)}")

    # Test 4: Add more items (simulating page 2)
    items_page_2 = [f"item_{i}" for i in range(50, 100)]
    service.add_seen_items(session_id, items_page_2)
    seen = service.get_seen_items(session_id)
    print(f"\n4. Added 50 more items, seen count: {len(seen)}")

    # Test 5: Verify exclusion works
    print("\n5. Testing exclusion...")
    all_items = [f"item_{i}" for i in range(150)]
    unseen = [item for item in all_items if item not in seen]
    print(f"   Total items: {len(all_items)}")
    print(f"   Seen items: {len(seen)}")
    print(f"   Unseen items: {len(unseen)}")

    # Test 6: Offset management
    print("\n6. Testing offset management...")
    print(f"   Initial offset: {service.get_offset(session_id)}")
    service.increment_offset(session_id, 50)
    print(f"   After +50: {service.get_offset(session_id)}")
    service.increment_offset(session_id, 50)
    print(f"   After +50: {service.get_offset(session_id)}")

    # Test 7: User signals
    print("\n7. Testing user signals...")
    service.record_signal(session_id, "view", "item_5")
    service.record_signal(session_id, "click", "item_10")
    service.record_signal(session_id, "skip", "item_15")
    signals = service.get_signals(session_id)
    print(f"   Views: {signals['views']}")
    print(f"   Clicks: {signals['clicks']}")
    print(f"   Skips: {signals['skips']}")

    # Test 8: Session info
    print("\n8. Session info:")
    info = service.get_session_info(session_id)
    for key, value in info.items():
        print(f"   {key}: {value}")

    # Test 9: Service stats
    print("\n9. Service stats:")
    stats = service.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 70)
    print("Session State test complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_session_state()
