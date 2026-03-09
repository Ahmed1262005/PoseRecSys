"""V3 Feed data models.

Lightweight dataclasses for the V3 recommendation pipeline.
Designed for compact Redis serialisation and minimal memory footprint.
"""

import base64
import hashlib
import json
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

@dataclass
class FeedRequest:
    user_id: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: Optional[str] = None
    page_size: int = 24
    mode: str = "explore"
    hard_filters: Optional[Any] = None
    soft_preferences: Optional[Dict[str, Any]] = None
    cursor: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    debug: bool = False


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

@dataclass
class CandidateStub:
    """Lightweight item reference from a retrieval source."""

    item_id: str
    source: str
    retrieval_score: float = 0.0
    brand: Optional[str] = None
    broad_category: Optional[str] = None
    cluster_id: Optional[str] = None
    article_type: Optional[str] = None
    price: Optional[float] = None
    image_dedup_key: Optional[str] = None
    embedding_score: float = 0.0
    retrieval_key: Optional[str] = None


@dataclass
class ScoringMeta:
    """Minimal per-item metadata stored in pool.

    ~8 fields, 500 items = 50-80 KB in Redis.
    """

    source: str
    retrieval_score: float
    brand: str
    cluster_id: str
    broad_category: str
    article_type: str
    price: float
    image_dedup_key: Optional[str] = None

    def to_dict(self) -> Dict:
        """Compact keys for Redis storage."""
        return {
            "s": self.source,
            "rs": self.retrieval_score,
            "b": self.brand,
            "c": self.cluster_id,
            "bc": self.broad_category,
            "at": self.article_type,
            "p": self.price,
            "dk": self.image_dedup_key,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ScoringMeta":
        """Rebuild from compact keys."""
        return cls(
            source=d["s"],
            retrieval_score=d["rs"],
            brand=d["b"],
            cluster_id=d["c"],
            broad_category=d["bc"],
            article_type=d["at"],
            price=d["p"],
            image_dedup_key=d.get("dk"),
        )


# ---------------------------------------------------------------------------
# Candidate Pool
# ---------------------------------------------------------------------------

@dataclass
class CandidatePool:
    """Cached pool of ranked candidates for session+mode.

    Persisted in Redis at ``v3:session:{session_id}:pool:{mode}``
    """

    session_id: str
    mode: str
    ordered_ids: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, ScoringMeta] = field(default_factory=dict)
    served_count: int = 0
    created_at: float = field(default_factory=time.time)
    retrieval_signature: str = ""
    ranking_signature: str = ""
    catalog_version: str = ""
    source_mix: Dict[str, int] = field(default_factory=dict)
    last_rerank_action_seq: int = 0

    # -- properties ---------------------------------------------------------

    @property
    def remaining(self) -> int:
        return max(0, len(self.ordered_ids) - self.served_count)

    @property
    def has_more(self) -> bool:
        return self.remaining > 0

    # -- page helpers -------------------------------------------------------

    def next_page_ids(self, page_size: int) -> List[str]:
        """Return next *page_size* IDs without advancing served_count."""
        return self.ordered_ids[self.served_count : self.served_count + page_size]

    def current_page(self) -> int:
        """1-indexed page number."""
        if self.served_count > 0:
            return (self.served_count // 24) + 1
        return 1

    # -- cursor -------------------------------------------------------------

    def get_cursor(self) -> Optional[str]:
        """Base64-encoded JSON cursor. *None* when no items remain."""
        if self.remaining <= 0:
            return None
        payload = json.dumps({"sc": self.served_count, "m": self.mode})
        return base64.b64encode(payload.encode()).decode()

    @staticmethod
    def decode_cursor(cursor: str) -> Optional[dict]:
        """Decode a base64 JSON cursor. Returns *None* if invalid."""
        try:
            payload = base64.b64decode(cursor.encode()).decode()
            return json.loads(payload)
        except Exception:
            return None

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        """Compact keys for Redis."""
        return {
            "sid": self.session_id,
            "m": self.mode,
            "ids": self.ordered_ids,
            "sc": self.scores,
            "svd": self.served_count,
            "cat": self.created_at,
            "rsig": self.retrieval_signature,
            "rksig": self.ranking_signature,
            "cv": self.catalog_version,
            "smix": self.source_mix,
            "lras": self.last_rerank_action_seq,
            "meta": {k: v.to_dict() for k, v in self.meta.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CandidatePool":
        """Rebuild from compact keys."""
        meta = {k: ScoringMeta.from_dict(v) for k, v in d.get("meta", {}).items()}
        return cls(
            session_id=d["sid"],
            mode=d["m"],
            ordered_ids=d.get("ids", []),
            scores=d.get("sc", {}),
            meta=meta,
            served_count=d.get("svd", 0),
            created_at=d.get("cat", 0.0),
            retrieval_signature=d.get("rsig", ""),
            ranking_signature=d.get("rksig", ""),
            catalog_version=d.get("cv", ""),
            source_mix=d.get("smix", {}),
            last_rerank_action_seq=d.get("lras", 0),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "CandidatePool":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Session Profile
# ---------------------------------------------------------------------------

@dataclass
class SessionProfile:
    session_id: str
    user_id: str
    shown_ids: Set[str] = field(default_factory=set)
    clicked_ids: Set[str] = field(default_factory=set)
    saved_ids: Set[str] = field(default_factory=set)
    skipped_ids: Set[str] = field(default_factory=set)
    hidden_ids: Set[str] = field(default_factory=set)
    explicit_negative_brands: Set[str] = field(default_factory=set)
    brand_exposure: Counter = field(default_factory=Counter)
    cluster_exposure: Counter = field(default_factory=Counter)
    category_exposure: Counter = field(default_factory=Counter)
    exploration_budget: float = 0.15
    intent_strength: float = 0.0
    action_seq: int = 0
    last_action_at: float = 0.0
    recent_actions: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    MAX_RECENT_ACTIONS: int = 50

    def record_action(
        self,
        action: str,
        item_id: str,
        brand: Optional[str] = None,
        cluster_id: Optional[str] = None,
        article_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.action_seq += 1
        self.last_action_at = time.time()

        # -- set membership -------------------------------------------------
        if action in ("click", "save", "cart", "purchase"):
            self.clicked_ids.add(item_id)
            if action == "save":
                self.saved_ids.add(item_id)
            if action == "purchase":
                self.saved_ids.add(item_id)

        if action == "skip":
            self.skipped_ids.add(item_id)

        if action == "hide":
            self.hidden_ids.add(item_id)
            if brand:
                hide_count = sum(
                    1
                    for a in self.recent_actions
                    if a.get("action") == "hide" and a.get("brand") == brand
                )
                if hide_count >= 2:
                    self.explicit_negative_brands.add(brand)

        # -- exposure counters ----------------------------------------------
        if brand:
            self.brand_exposure[brand] += 1
        if cluster_id:
            self.cluster_exposure[cluster_id] += 1
        if article_type:
            self.category_exposure[article_type] += 1

        # -- recent actions log ---------------------------------------------
        entry: Dict[str, Any] = {
            "action": action,
            "item_id": item_id,
            "brand": brand,
            **(metadata or {}),
        }
        self.recent_actions.append(entry)
        if len(self.recent_actions) > self.MAX_RECENT_ACTIONS:
            self.recent_actions = self.recent_actions[-self.MAX_RECENT_ACTIONS :]

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "sid": self.session_id,
            "uid": self.user_id,
            "clicked": list(self.clicked_ids),
            "saved": list(self.saved_ids),
            "skipped": list(self.skipped_ids),
            "hidden": list(self.hidden_ids),
            "neg_brands": list(self.explicit_negative_brands),
            "brand_exp": dict(self.brand_exposure),
            "cluster_exp": dict(self.cluster_exposure),
            "cat_exp": dict(self.category_exposure),
            "explore_budget": self.exploration_budget,
            "intent_str": self.intent_strength,
            "action_seq": self.action_seq,
            "last_action": self.last_action_at,
            "recent": self.recent_actions,
            "created": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionProfile":
        return cls(
            session_id=d["sid"],
            user_id=d["uid"],
            clicked_ids=set(d.get("clicked", [])),
            saved_ids=set(d.get("saved", [])),
            skipped_ids=set(d.get("skipped", [])),
            hidden_ids=set(d.get("hidden", [])),
            explicit_negative_brands=set(d.get("neg_brands", [])),
            brand_exposure=Counter(d.get("brand_exp", {})),
            cluster_exposure=Counter(d.get("cluster_exp", {})),
            category_exposure=Counter(d.get("cat_exp", {})),
            exploration_budget=d.get("explore_budget", 0.15),
            intent_strength=d.get("intent_str", 0.0),
            action_seq=d.get("action_seq", 0),
            last_action_at=d.get("last_action", 0.0),
            recent_actions=d.get("recent", []),
            created_at=d.get("created", 0.0),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "SessionProfile":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Pool Decision
# ---------------------------------------------------------------------------

@dataclass
class PoolDecision:
    action: str
    reason: str
    remaining: int = 0


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------

def _stable_json(obj) -> str:
    """Deterministic JSON string for hashing."""
    return json.dumps(obj, sort_keys=True, default=str)


def compute_retrieval_signature(
    mode: str,
    hard_filters_hashable,
    key_family: str,
    source_config_version: str = "v1",
) -> str:
    """Content-addressable retrieval signature (first 16 hex chars of MD5)."""
    payload = _stable_json([mode, hard_filters_hashable, key_family, source_config_version])
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def compute_ranking_signature(weights, session_action_seq: int) -> str:
    """Ranking signature that changes every 3 actions."""
    payload = _stable_json([weights, session_action_seq // 3])
    return hashlib.md5(payload.encode()).hexdigest()[:16]
