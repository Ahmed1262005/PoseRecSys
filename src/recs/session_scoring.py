"""
Session-Aware Scoring Engine for Fashion Recommendations.

This module implements the core scoring logic that makes the feed feel
"like a human stylist":

1. SessionScores: Live per-session preference state
   - brand_scores, type_scores, attr_scores, cluster_scores
   - Each dimension stores Dict[str, PreferenceState] with multi-resolution EMA
   - Updated on every interaction, search, and filter action

2. score_item(): Additive scoring formula
   score(item) = cluster_affinity + brand_affinity + type_affinity
                + attr_match + search_intent + trending - seen_penalty

3. Action processing: Different actions map to [-1, 1] signal values
   - skip = -0.5, hover = 0.1, click = 0.5, cart = 0.8, purchase = 1.0

4. Multi-Resolution EMA (Exponential Moving Average):
   - Each preference has a FAST track (α=0.3, last ~3 signals)
     and a SLOW track (α=0.05, last ~20 signals)
   - Self-normalizing: if inputs are in [-1, 1], output stays in [-1, 1]
   - Time-based idle decay on the fast track
   - No caps needed (unlike the old scalar accumulation)

Architecture:
- Designed for Redis storage (serializable to JSON)
- EMA replaces rolling windows (less manual tuning, mathematically bounded)
- All scores become ML features later (just log them)
"""

import math
import time
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple

from recs.brand_clusters import (
    get_cluster_for_item,
    compute_cluster_scores_from_brands,
    DEFAULT_CLUSTER,
    CLUSTER_TRAITS,
)


# =============================================================================
# PreferenceState: Multi-Resolution EMA
# =============================================================================

@dataclass
class PreferenceState:
    """
    Multi-resolution EMA state for a single preference dimension.

    Two tracks run in parallel:
    - **fast** (α=0.3): Reacts to the last ~3 signals. Captures impulse.
    - **slow** (α=0.05): Reacts to the last ~20 signals. Captures taste.

    EMA update rule:
        μ_new = α × signal + (1 - α) × μ_old

    Self-normalizing: if inputs are in [-1, 1], output stays in [-1, 1].
    No capping needed.
    """
    fast: float = 0.0
    slow: float = 0.0
    count: int = 0
    last_update: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fast": self.fast,
            "slow": self.slow,
            "count": self.count,
            "last_update": self.last_update,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferenceState":
        return cls(
            fast=data.get("fast", 0.0),
            slow=data.get("slow", 0.0),
            count=data.get("count", 0),
            last_update=data.get("last_update", 0.0),
        )

    def score(self, fast_weight: float = 0.7, slow_weight: float = 0.3,
              confidence_threshold: int = 3) -> float:
        """
        Compute blended score from fast + slow tracks.

        Returns:
            float in [-1, 1], scaled by confidence.
        """
        raw = fast_weight * self.fast + slow_weight * self.slow
        # Confidence ramp: full weight after `confidence_threshold` signals
        confidence = min(1.0, self.count / max(confidence_threshold, 1))
        return raw * confidence


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class ScoringConfig:
    """Tunable parameters for session scoring."""

    # --- Action signal values ---
    # Values > 1.0 are intentional for high-intent actions: the EMA formula
    # (alpha * signal + (1-alpha) * old) naturally dampens them, but they
    # spike the fast track hard enough to trigger instant retrieval pivots.
    ACTION_VALUES: Dict[str, float] = field(default_factory=lambda: {
        "purchase": 1.0,
        "add_to_cart": 0.8,
        "add_to_wishlist": 2.0,   # High-intent: spike fast track to ~0.6
        "click": 0.5,
        "search": 2.0,            # High-intent: spike fast track to ~0.6
        "hover": 0.1,
        "skip": -0.5,
        "impression_no_action": -0.1,
    })

    # --- EMA parameters ---
    FAST_ALPHA: float = 0.3          # Immediate reaction (~3 signal memory)
    SLOW_ALPHA: float = 0.05         # Long-term taste (~20 signal memory)
    FAST_WEIGHT: float = 0.7         # Blend: 70% impulse
    SLOW_WEIGHT: float = 0.3         # Blend: 30% profile
    HALF_LIFE_SECONDS: float = 1800  # 30-min idle decay on fast track
    CONFIDENCE_THRESHOLD: int = 1    # Full weight from first signal (EMA already dampens)

    # --- Action log bound (for debugging / future ML) ---
    LONG_TERM_WINDOW: int = 500    # Max action log entries to keep

    # --- Scoring weights (how much each signal contributes) ---
    W_CLUSTER: float = 0.25
    W_BRAND: float = 0.20
    W_TYPE: float = 0.20
    W_ATTR: float = 0.15
    W_SEARCH_INTENT: float = 0.10
    W_TRENDING: float = 0.05
    W_NOVELTY: float = 0.05

    # --- Session amplification ---
    SESSION_MULTIPLIER: float = 3.0  # Amplifies session score vs base score spread

    # --- Mismatch penalty ---
    # When user has ANY brand/type signal but item doesn't match, push it down.
    # Threshold is low (0.05) so penalty fires after a single interaction.
    MISMATCH_PENALTY_THRESHOLD: float = 0.05  # Min blended score to trigger mismatch
    MISMATCH_PENALTY_FACTOR: float = 0.8      # Fraction of max signal applied as penalty

    # --- Penalties ---
    SEEN_PENALTY: float = 0.8      # How much to penalize already-seen items
    SKIP_PENALTY: float = 0.3      # Additional penalty for skipped items

    # --- Search intent decay ---
    SEARCH_INTENT_SESSIONS: int = 3  # How many search queries to remember


DEFAULT_CONFIG = ScoringConfig()


# =============================================================================
# Action Log Entry
# =============================================================================

@dataclass
class ActionEntry:
    """A single user action with context."""
    action: str            # click, hover, skip, purchase, add_to_cart, etc.
    product_id: str
    brand: str
    item_type: str         # broad_category or article_type
    cluster_id: str        # brand's cluster
    attributes: Dict[str, str]  # {fit: "slim", color_family: "Neutrals", ...}
    timestamp: float
    source: str = "feed"   # feed, search, similar, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "product_id": self.product_id,
            "brand": self.brand,
            "item_type": self.item_type,
            "cluster_id": self.cluster_id,
            "attributes": self.attributes,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionEntry":
        return cls(
            action=data["action"],
            product_id=data["product_id"],
            brand=data.get("brand", ""),
            item_type=data.get("item_type", ""),
            cluster_id=data.get("cluster_id", ""),
            attributes=data.get("attributes", {}),
            timestamp=data.get("timestamp", time.time()),
            source=data.get("source", "feed"),
        )


# =============================================================================
# Preference Map Serialization Helpers
# =============================================================================

def _serialize_pref_map(prefs: Dict[str, "PreferenceState"]) -> Dict[str, Any]:
    """Serialize Dict[str, PreferenceState] to JSON-safe dict."""
    return {k: v.to_dict() for k, v in prefs.items()}


def _deserialize_pref_map(raw: Dict[str, Any]) -> Dict[str, "PreferenceState"]:
    """Deserialize preference map with backward-compat migration.

    Handles two formats:
    - New format: {"key": {"fast": 0.3, "slow": 0.1, "count": 5, ...}}
    - Old format: {"key": 1.5}  (plain float -> seed into slow track)
    """
    result: Dict[str, PreferenceState] = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "fast" in v:
            # New format
            result[k] = PreferenceState.from_dict(v)
        elif isinstance(v, (int, float)):
            # Old format: migrate scalar into slow track
            # Clamp the old value to [-1, 1] since old system could accumulate
            clamped = max(-1.0, min(1.0, float(v)))
            result[k] = PreferenceState(slow=clamped, count=1)
        else:
            # Unknown format, skip
            pass
    return result


# =============================================================================
# Session Scores (the live state)
# =============================================================================

@dataclass
class SessionScores:
    """
    Live session preference state, updated on every interaction.

    This is the core data structure that makes the feed adaptive.
    Stored in Redis, serialized to JSON.

    Each score map stores Dict[str, PreferenceState] with multi-resolution
    EMA tracks (fast + slow).  The old Dict[str, float] format is
    auto-migrated on deserialization.
    """
    # Preference maps: key -> PreferenceState (multi-resolution EMA)
    cluster_scores: Dict[str, PreferenceState] = field(default_factory=dict)
    brand_scores: Dict[str, PreferenceState] = field(default_factory=dict)
    type_scores: Dict[str, PreferenceState] = field(default_factory=dict)
    attr_scores: Dict[str, PreferenceState] = field(default_factory=dict)

    # Search intent: what the user is actively looking for
    search_intents: Dict[str, PreferenceState] = field(default_factory=dict)

    # Action log (for debugging + future ML features)
    action_log: List[Dict[str, Any]] = field(default_factory=list)

    # Skipped product IDs (for skip penalty)
    skipped_ids: Set[str] = field(default_factory=set)

    # Metadata
    action_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    # --- Convenience accessors for external consumers ---

    def get_score(self, dimension: str, key: str,
                  config: "ScoringConfig" = None) -> float:
        """Get the blended EMA score for a key in a dimension.

        Args:
            dimension: One of "cluster", "brand", "type", "attr", "intent"
            key: The preference key (e.g., "reformation", "fit:slim")
            config: Optional ScoringConfig for EMA weights

        Returns:
            float in [-1, 1], or 0.0 if key not found.
        """
        cfg = config or DEFAULT_CONFIG
        pref_map = {
            "cluster": self.cluster_scores,
            "brand": self.brand_scores,
            "type": self.type_scores,
            "attr": self.attr_scores,
            "intent": self.search_intents,
        }.get(dimension, {})
        pref = pref_map.get(key)
        if pref is None:
            return 0.0
        return pref.score(
            fast_weight=cfg.FAST_WEIGHT,
            slow_weight=cfg.SLOW_WEIGHT,
            confidence_threshold=cfg.CONFIDENCE_THRESHOLD,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict (for Redis)."""
        return {
            "cluster_scores": _serialize_pref_map(self.cluster_scores),
            "brand_scores": _serialize_pref_map(self.brand_scores),
            "type_scores": _serialize_pref_map(self.type_scores),
            "attr_scores": _serialize_pref_map(self.attr_scores),
            "search_intents": _serialize_pref_map(self.search_intents),
            "action_log": self.action_log,
            "skipped_ids": list(self.skipped_ids),
            "action_count": self.action_count,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionScores":
        """Deserialize from dict (auto-migrates old float format)."""
        return cls(
            cluster_scores=_deserialize_pref_map(data.get("cluster_scores", {})),
            brand_scores=_deserialize_pref_map(data.get("brand_scores", {})),
            type_scores=_deserialize_pref_map(data.get("type_scores", {})),
            attr_scores=_deserialize_pref_map(data.get("attr_scores", {})),
            search_intents=_deserialize_pref_map(data.get("search_intents", {})),
            action_log=data.get("action_log", []),
            skipped_ids=set(data.get("skipped_ids", [])),
            action_count=data.get("action_count", 0),
            created_at=data.get("created_at", time.time()),
            last_updated=data.get("last_updated", time.time()),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "SessionScores":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Session Scoring Engine
# =============================================================================

class SessionScoringEngine:
    """
    Manages session scores and computes item scores.

    Uses Multi-Resolution EMA (Exponential Moving Average) for all
    preference tracking.  Each dimension (brand, type, attr, cluster)
    stores a Dict[str, PreferenceState] with fast + slow tracks.

    Usage:
        engine = SessionScoringEngine()

        # Initialize from onboarding (cold start)
        scores = engine.initialize_from_onboarding(preferred_brands=["Reformation", "Zara"])

        # Process user actions
        engine.process_action(scores, action="click", product_id="p1", brand="Zara", ...)

        # Process search signals
        engine.process_search_signal(scores, query="blue midi dress", filters={"fit_types": ["slim"]})

        # Score candidates
        for item in candidates:
            item.final_score = engine.score_item(scores, item)
    """

    def __init__(self, config: ScoringConfig = None):
        self.config = config or DEFAULT_CONFIG

    # =========================================================
    # Core EMA Operations
    # =========================================================

    def _update_preference(
        self,
        pref_map: Dict[str, PreferenceState],
        key: str,
        signal: float,
        now: float = None,
    ) -> None:
        """
        Update a single preference using multi-resolution EMA.

        EMA rule: μ_new = α × signal + (1 - α) × μ_old

        Also applies idle time-decay to the fast track:
            fast *= exp(-dt / HALF_LIFE)

        Args:
            pref_map: The preference dict to update (mutated in-place)
            key: Preference key (e.g., "reformation", "fit:slim")
            signal: Signal value in [-1, 1]
            now: Current timestamp (default: time.time())
        """
        now = now or time.time()
        cfg = self.config

        pref = pref_map.get(key)
        if pref is None:
            pref = PreferenceState()
            pref_map[key] = pref

        # Apply idle time-decay to fast track
        if pref.last_update > 0 and cfg.HALF_LIFE_SECONDS > 0:
            dt = now - pref.last_update
            if dt > 0:
                decay = math.exp(-dt / cfg.HALF_LIFE_SECONDS)
                pref.fast *= decay

        # EMA update
        pref.fast = cfg.FAST_ALPHA * signal + (1 - cfg.FAST_ALPHA) * pref.fast
        pref.slow = cfg.SLOW_ALPHA * signal + (1 - cfg.SLOW_ALPHA) * pref.slow
        pref.count += 1
        pref.last_update = now

    def _get_preference_score(
        self, pref_map: Dict[str, PreferenceState], key: str
    ) -> float:
        """
        Get blended EMA score for a preference key.

        Returns:
            float in [-1, 1], or 0.0 if key not found.
        """
        pref = pref_map.get(key)
        if pref is None:
            return 0.0
        cfg = self.config
        return pref.score(cfg.FAST_WEIGHT, cfg.SLOW_WEIGHT, cfg.CONFIDENCE_THRESHOLD)

    # =========================================================
    # Initialization
    # =========================================================

    def initialize_from_onboarding(
        self,
        preferred_brands: List[str] = None,
        onboarding_profile: Any = None,
    ) -> SessionScores:
        """
        Create initial session scores from onboarding data (cold-start solver).

        Seeds the **slow** EMA track only (these are prior preferences,
        not impulse signals).

        Args:
            preferred_brands: Brand names from onboarding
            onboarding_profile: Full OnboardingProfile (optional, for attr priors)

        Returns:
            SessionScores with initial cluster/brand priors in slow track
        """
        scores = SessionScores()

        # Initialize cluster scores from preferred brands
        if preferred_brands:
            cluster_raw = compute_cluster_scores_from_brands(preferred_brands)
            for cluster_id, val in cluster_raw.items():
                scores.cluster_scores[cluster_id] = PreferenceState(
                    slow=val, count=1,
                )
            # Also seed brand scores
            for brand in preferred_brands:
                scores.brand_scores[brand.lower()] = PreferenceState(
                    slow=0.5, count=1,
                )

        # Initialize attribute scores from onboarding profile
        if onboarding_profile:
            self._seed_from_profile(scores, onboarding_profile)

        return scores

    def _seed_from_profile(self, scores: SessionScores, profile: Any) -> None:
        """Seed attribute slow priors from onboarding profile."""
        # Positive attribute priors (seed slow track at 0.3)
        attr_sources = [
            ("preferred_fits", "fit"),
            ("preferred_sleeves", "sleeve"),
            ("preferred_lengths", "length"),
            ("preferred_necklines", "neckline"),
            ("patterns_liked", "pattern"),
            ("occasions", "occasion"),
            ("style_persona", "style"),
        ]
        for attr_name, prefix in attr_sources:
            for val in getattr(profile, attr_name, []) or []:
                if val:
                    key = f"{prefix}:{val.lower()}"
                    scores.attr_scores[key] = PreferenceState(slow=0.3, count=1)

        # Categories -> type scores
        for cat in getattr(profile, "categories", []) or []:
            if cat:
                scores.type_scores[cat.lower()] = PreferenceState(slow=0.3, count=1)

        # Negative signals from profile
        for brand in getattr(profile, "brands_to_avoid", []) or []:
            if brand:
                scores.brand_scores[brand.lower()] = PreferenceState(slow=-1.0, count=1)
        for color in getattr(profile, "colors_to_avoid", []) or []:
            if color:
                scores.attr_scores[f"color:{color.lower()}"] = PreferenceState(
                    slow=-0.5, count=1,
                )

    # =========================================================
    # Action Processing
    # =========================================================

    def process_action(
        self,
        scores: SessionScores,
        action: str,
        product_id: str,
        brand: str = "",
        item_type: str = "",
        attributes: Dict[str, str] = None,
        source: str = "feed",
    ) -> None:
        """
        Update session scores based on a user action.

        Args:
            scores: Current session scores (mutated in place)
            action: Action type (click, hover, skip, purchase, add_to_cart, etc.)
            product_id: Product ID
            brand: Product brand
            item_type: Product type (broad_category or article_type)
            attributes: Product attributes {fit, color_family, pattern, ...}
            source: Action source (feed, search, similar, etc.)
        """
        attributes = attributes or {}
        signal = self.config.ACTION_VALUES.get(action, 0.0)
        cluster_id = get_cluster_for_item(brand) or DEFAULT_CLUSTER
        now = time.time()

        # Record the action in the log (for debugging + future ML)
        entry = ActionEntry(
            action=action,
            product_id=product_id,
            brand=brand.lower(),
            item_type=item_type.lower() if item_type else "",
            cluster_id=cluster_id,
            attributes={k: v.lower() if isinstance(v, str) else v for k, v in attributes.items()},
            timestamp=now,
            source=source,
        )

        # Append to action log (keep bounded)
        scores.action_log.append(entry.to_dict())
        if len(scores.action_log) > self.config.LONG_TERM_WINDOW:
            scores.action_log = scores.action_log[-self.config.LONG_TERM_WINDOW:]

        # Track skips
        if action == "skip":
            scores.skipped_ids.add(product_id)

        # Update EMA score maps
        self._update_scores(scores, entry, signal, now)

        scores.action_count += 1
        scores.last_updated = now

    def _update_scores(
        self,
        scores: SessionScores,
        entry: ActionEntry,
        signal: float,
        now: float,
    ) -> None:
        """Apply EMA update to all relevant score dimensions."""
        # Cluster
        if entry.cluster_id:
            self._update_preference(
                scores.cluster_scores, entry.cluster_id, signal, now
            )

        # Brand
        if entry.brand:
            self._update_preference(
                scores.brand_scores, entry.brand, signal, now
            )

        # Type
        if entry.item_type:
            self._update_preference(
                scores.type_scores, entry.item_type, signal, now
            )

        # Attributes
        for attr_key, attr_val in entry.attributes.items():
            if attr_val and attr_val not in ("", "none", "null", "n/a"):
                key = f"{attr_key}:{attr_val}"
                self._update_preference(scores.attr_scores, key, signal, now)

    # =========================================================
    # Search Signal Processing
    # =========================================================

    def process_search_signal(
        self,
        scores: SessionScores,
        query: str = "",
        filters: Dict[str, Any] = None,
    ) -> None:
        """
        Update session scores from search query + filter usage.

        Accepts **structured** signals extracted by the caller (search route,
        Gradio demo, etc.).  Free-text query parsing does NOT belong here —
        callers should use :func:`extract_search_signals` to convert raw
        query text into structured filters before calling this method.

        Structured filter keys processed:
            brands, categories, fit_types, colors, patterns, occasions,
            styles, necklines, sleeve_types, lengths, materials

        Args:
            scores: Current session scores (mutated in-place)
            query: Original query text (stored as raw intent if non-empty)
            filters: Structured search filters
        """
        filters = filters or {}
        search_signal = self.config.ACTION_VALUES.get("search", 0.4)
        now = time.time()

        # -----------------------------------------------------------
        # 1. Process structured filter signals
        # -----------------------------------------------------------
        filter_attr_map = {
            "fit_types": "fit",
            "colors": "color",
            "patterns": "pattern",
            "occasions": "occasion",
            "styles": "style",
            "necklines": "neckline",
            "sleeve_types": "sleeve",
            "lengths": "length",
            "materials": "material",
            "categories": "type",
        }

        for filter_key, attr_prefix in filter_attr_map.items():
            values = filters.get(filter_key, [])
            if isinstance(values, str):
                values = [values]
            for val in values:
                if val:
                    key = f"{attr_prefix}:{val.lower()}"
                    self._update_preference(
                        scores.search_intents, key, search_signal, now
                    )

        # Process brand signals (from filters or extracted from query)
        for brand in filters.get("brands", []):
            if brand:
                b_lower = brand.lower()
                self._update_preference(
                    scores.brand_scores, b_lower, search_signal, now
                )
                # Update cluster scores for matched brands
                cluster_id = get_cluster_for_item(b_lower)
                if cluster_id:
                    self._update_preference(
                        scores.cluster_scores, cluster_id, search_signal, now
                    )

        # Process type signals (from filters or extracted from query)
        for item_type in filters.get("types", []):
            if item_type:
                t_lower = item_type.lower()
                self._update_preference(
                    scores.type_scores, t_lower, search_signal, now
                )
                self._update_preference(
                    scores.search_intents, f"type:{t_lower}", search_signal, now
                )

        # Store raw query as a weak intent
        if query and query.strip():
            q_lower = query.lower().strip()
            self._update_preference(
                scores.search_intents, f"query:{q_lower}",
                search_signal * 0.3, now,
            )

        # -----------------------------------------------------------
        # 2. Housekeeping — prune search intents to top 50
        # -----------------------------------------------------------
        if len(scores.search_intents) > 50:
            sorted_intents = sorted(
                scores.search_intents.items(),
                key=lambda x: -x[1].score(
                    self.config.FAST_WEIGHT,
                    self.config.SLOW_WEIGHT,
                    self.config.CONFIDENCE_THRESHOLD,
                ),
            )
            scores.search_intents = dict(sorted_intents[:50])

        scores.action_count += 1
        scores.last_updated = now

    # =========================================================
    # Blended Score Computation (EMA-based)
    # =========================================================

    def compute_blended_scores(
        self, scores: SessionScores
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute blended scores from EMA preference states.

        Reads each PreferenceState and produces a blended float score
        via: score = fast_weight * fast + slow_weight * slow, scaled
        by confidence.

        Returns a dict with scores for each dimension:
        {
            "cluster": {cluster_id: float_score},
            "brand": {brand: float_score},
            "type": {type: float_score},
            "attr": {attr_key: float_score},
        }

        This output format is the same as before — downstream consumers
        (score_item, pipeline, Gradio) don't need to change.
        """
        dims = {
            "cluster": scores.cluster_scores,
            "brand": scores.brand_scores,
            "type": scores.type_scores,
            "attr": scores.attr_scores,
        }
        cfg = self.config

        blended: Dict[str, Dict[str, float]] = {}
        for dim_name, pref_map in dims.items():
            blended[dim_name] = {
                key: pref.score(cfg.FAST_WEIGHT, cfg.SLOW_WEIGHT, cfg.CONFIDENCE_THRESHOLD)
                for key, pref in pref_map.items()
            }

        # NOTE: search_intents are NOT folded into attr here.
        # They are read directly by score_item() component #5.
        # Folding them in caused double-counting: broad intents
        # (pattern:solid, style:casual) matched most items and
        # created a flat floor that killed differentiation.

        return blended

    # =========================================================
    # Item Scoring
    # =========================================================

    def score_item(
        self,
        scores: SessionScores,
        item: Any,
        blended: Dict[str, Dict[str, float]] = None,
    ) -> float:
        """
        Compute a session-aware score for a candidate item.

        Args:
            scores: Current session scores
            item: A Candidate object (from recs.models)
            blended: Pre-computed blended scores (optional, for batch efficiency)

        Returns:
            Float score (higher = better match for this session).
            Self-bounded by EMA — no clamping needed.
        """
        if blended is None:
            blended = self.compute_blended_scores(scores)

        cfg = self.config
        total = 0.0

        # 1. Cluster affinity
        brand = getattr(item, "brand", "") or ""
        cluster_id = get_cluster_for_item(brand) or DEFAULT_CLUSTER
        cluster_score = blended.get("cluster", {}).get(cluster_id, 0.0)
        total += cfg.W_CLUSTER * cluster_score

        # 2. Brand affinity
        brand_key = brand.lower()
        brand_score = blended.get("brand", {}).get(brand_key, 0.0)
        total += cfg.W_BRAND * brand_score

        # 3. Type affinity — check both article_type and broad_category,
        #    take the stronger signal.
        type_map = blended.get("type", {})
        article_type = (getattr(item, "article_type", "") or "").lower()
        broad_cat = (getattr(item, "broad_category", "") or "").lower()
        type_score = max(
            type_map.get(article_type, 0.0),
            type_map.get(broad_cat, 0.0),
        )
        total += cfg.W_TYPE * type_score

        # 4. Attribute match (average of matching attributes)
        attr_total = 0.0
        attr_count = 0
        attr_map = blended.get("attr", {})

        item_attrs = self._extract_item_attributes(item)
        for key in item_attrs:
            if key in attr_map:
                attr_total += attr_map[key]
                attr_count += 1

        if attr_count > 0:
            total += cfg.W_ATTR * (attr_total / max(attr_count, 1))

        # 5. Search intent match (direct EMA read)
        intent_match = 0.0
        intent_count = 0
        for key in item_attrs:
            intent_score = self._get_preference_score(scores.search_intents, key)
            if intent_score != 0.0:
                intent_match += intent_score
                intent_count += 1
        if intent_count > 0:
            total += cfg.W_SEARCH_INTENT * (intent_match / max(intent_count, 1))

        # 6. Trending / newness boost
        is_new = getattr(item, "is_new", False)
        is_sale = getattr(item, "is_on_sale", False)
        if is_new:
            total += cfg.W_TRENDING * 0.5
        if is_sale:
            total += cfg.W_TRENDING * 0.3

        # 7. Novelty (items from unseen clusters get a small boost)
        if cluster_id not in blended.get("cluster", {}):
            total += cfg.W_NOVELTY * 0.5

        # 8. Penalties
        product_id = getattr(item, "item_id", "") or getattr(item, "product_id", "")
        if product_id in scores.skipped_ids:
            total -= cfg.SKIP_PENALTY

        # 9. Mismatch penalty — push down items that don't match strong session signals.
        #    Without this, non-matching items keep score=0 from session (neutral),
        #    so they can outrank matching items if their base score is high.
        brand_map = blended.get("brand", {})
        if brand_map and brand_score == 0.0:
            max_brand = max(brand_map.values(), default=0)
            if max_brand > cfg.MISMATCH_PENALTY_THRESHOLD:
                total -= cfg.W_BRAND * max_brand * cfg.MISMATCH_PENALTY_FACTOR

        type_map_all = blended.get("type", {})
        if type_map_all and type_score == 0.0:
            max_type = max(type_map_all.values(), default=0)
            if max_type > cfg.MISMATCH_PENALTY_THRESHOLD:
                total -= cfg.W_TYPE * max_type * cfg.MISMATCH_PENALTY_FACTOR

        # No clamping — EMA is self-normalizing.
        # Score is bounded by sum of weights (≈1.0) * [-1, 1] = [-1, 1]
        return total

    def score_candidates(
        self,
        scores: SessionScores,
        candidates: List[Any],
    ) -> List[Any]:
        """
        Score a batch of candidates using session scores.

        Session scoring is **additive**: it adjusts the existing score
        (embedding similarity + profile + context) rather than replacing it.
        This preserves the base ranking while allowing session signals to
        meaningfully reorder items.

        Modifies each candidate's final_score in place.
        Returns candidates sorted by final_score descending.
        """
        blended = self.compute_blended_scores(scores)

        for item in candidates:
            session_score = self.score_item(scores, item, blended)
            # Additive: session boost adjusts the existing ranking
            # Multiplier ensures session signals can overtake base score spread
            existing = getattr(item, "final_score", 0.0) or 0.0
            item.final_score = existing + session_score * self.config.SESSION_MULTIPLIER

        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates

    @staticmethod
    def _extract_item_attributes(item: Any) -> List[str]:
        """Extract attribute keys from a Candidate for matching."""
        keys = []
        attr_fields = [
            ("fit", "fit"),
            ("color_family", "color"),
            ("pattern", "pattern"),
            ("formality", "formality"),
            ("neckline", "neckline"),
            ("sleeve", "sleeve"),
            ("length", "length"),
        ]
        for field_name, prefix in attr_fields:
            val = getattr(item, field_name, None)
            if val and val.lower() not in ("", "none", "null", "n/a"):
                keys.append(f"{prefix}:{val.lower()}")

        # List attributes
        for tag in getattr(item, "style_tags", []) or []:
            if tag:
                keys.append(f"style:{tag.lower()}")
        for occ in getattr(item, "occasions", []) or []:
            if occ:
                keys.append(f"occasion:{occ.lower()}")

        # Broad category as type
        bc = getattr(item, "broad_category", "") or ""
        if bc:
            keys.append(f"type:{bc.lower()}")

        return keys


# =============================================================================
# Session-Aware Retrieval: Extract intent filters for candidate recall
# =============================================================================

def extract_intent_filters(
    scores: SessionScores,
    config: ScoringConfig = None,
    min_score: float = 0.15,
    max_brands: int = 5,
    max_types: int = 5,
) -> Dict[str, Any]:
    """
    Extract the top session intent signals for candidate retrieval.

    Reads the EMA preference states and returns structured filters
    suitable for querying the product database.  Uses **raw** blended
    scores (no confidence ramp) so that even a single high-intent
    action (search, wishlist) can trigger retrieval immediately.

    This is the bridge between session scoring (ranking) and candidate
    retrieval (recall).  The pipeline uses these filters to inject
    session-relevant items into the candidate pool.

    Args:
        scores: Current session scores.
        config: Optional ScoringConfig for EMA weights.
        min_score: Minimum *raw* EMA score to include a signal (no
            confidence ramp — we want aggressive retrieval).
        max_brands: Maximum number of brands to return.
        max_types: Maximum number of types to return.

    Returns:
        Dict with keys:
            brands: List[str] — top brand signals
            types: List[str] — top type signals
            has_intent: bool — whether any meaningful signals exist
            signal_count: int — total number of qualifying signals

    Example:
        >>> filters = extract_intent_filters(scores)
        >>> filters
        {'brands': ['alo yoga', 'lululemon'], 'types': ['leggings'],
         'has_intent': True, 'signal_count': 3}
    """
    cfg = config or DEFAULT_CONFIG

    def _top_keys(pref_map: Dict[str, PreferenceState], max_n: int) -> List[str]:
        """Return top-N keys sorted by raw blended score, filtered by min_score.

        Uses raw score (fast_weight * fast + slow_weight * slow) WITHOUT
        the confidence ramp.  For retrieval we care about "is there a
        signal?" not "how confident are we?" — a single search at 2.0
        signal strength should immediately trigger candidate fetching.
        """
        scored = []
        for key, pref in pref_map.items():
            # Raw score: no confidence multiplier
            s = cfg.FAST_WEIGHT * pref.fast + cfg.SLOW_WEIGHT * pref.slow
            if s >= min_score:
                scored.append((key, s))
        scored.sort(key=lambda x: -x[1])
        return [k for k, _ in scored[:max_n]]

    brands = _top_keys(scores.brand_scores, max_brands)
    types = _top_keys(scores.type_scores, max_types)

    # Exclude onboarding-only priors (count=1, no fast signal) — these
    # are static profile preferences, not live session intent.
    brands = [b for b in brands if scores.brand_scores[b].count > 1
              or scores.brand_scores[b].fast != 0.0]
    types = [t for t in types if scores.type_scores[t].count > 1
             or scores.type_scores[t].fast != 0.0]

    return {
        "brands": brands,
        "types": types,
        "has_intent": len(brands) > 0 or len(types) > 0,
        "signal_count": len(brands) + len(types),
    }


# =============================================================================
# Extract structured signals from Algolia search results
# =============================================================================

def extract_search_signals(
    results: List[Dict[str, Any]],
    top_n: int = 10,
) -> Dict[str, Any]:
    """Extract structured session-scoring signals from search results.

    Instead of parsing query text with hardcoded keyword maps, this looks
    at what Algolia/hybrid search **actually returned**.  The top-N results
    are the best evidence of what the user is browsing.

    Args:
        results: Search result dicts (or ProductResult-like objects).
                 Expected keys: brand, article_type, style_tags, occasions,
                 color_family, pattern, fit_type, categories.
        top_n:   How many top results to consider (default 10).

    Returns:
        Dict suitable for ``filters`` kwarg of
        :meth:`SessionScoringEngine.process_search_signal`::

            {
                "brands":     ["alo yoga", "lululemon"],
                "types":      ["leggings"],
                "styles":     ["athleisure", "sporty"],
                "occasions":  ["workout"],
                "colors":     ["black"],
                "patterns":   ["solid"],
                "fit_types":  ["fitted"],
            }
    """
    if not results:
        return {}

    slice_ = results[:top_n]
    from collections import Counter

    brand_counts: Counter = Counter()
    type_counts: Counter = Counter()
    style_counts: Counter = Counter()
    occasion_counts: Counter = Counter()
    color_counts: Counter = Counter()
    pattern_counts: Counter = Counter()
    fit_counts: Counter = Counter()

    for r in slice_:
        # Support both dicts and pydantic models / objects
        _get = r.get if isinstance(r, dict) else lambda k, d=None: getattr(r, k, d)

        brand = _get("brand", None)
        if brand:
            brand_counts[brand.lower().strip()] += 1

        atype = _get("article_type", None)
        if atype:
            type_counts[atype.lower().strip()] += 1

        for tag in (_get("style_tags", None) or []):
            if tag:
                style_counts[tag.lower().strip()] += 1

        for occ in (_get("occasions", None) or []):
            if occ:
                occasion_counts[occ.lower().strip()] += 1

        color = _get("color_family", None)
        if color:
            color_counts[color.lower().strip()] += 1

        pat = _get("pattern", None)
        if pat and pat.lower() not in ("", "n/a", "none", "null"):
            pattern_counts[pat.lower().strip()] += 1

        fit = _get("fit_type", None)
        if fit and fit.lower() not in ("", "n/a", "none", "null"):
            fit_counts[fit.lower().strip()] += 1

    # Only include signals that appear in >= 20% of top results (signal, not noise)
    threshold = max(1, len(slice_) * 0.2)
    signals: Dict[str, list] = {}

    def _above_threshold(counter: Counter) -> List[str]:
        return [k for k, v in counter.most_common() if v >= threshold]

    brands = _above_threshold(brand_counts)
    if brands:
        signals["brands"] = brands
    types = _above_threshold(type_counts)
    if types:
        signals["types"] = types
    styles = _above_threshold(style_counts)
    if styles:
        signals["styles"] = styles
    occasions = _above_threshold(occasion_counts)
    if occasions:
        signals["occasions"] = occasions
    colors = _above_threshold(color_counts)
    if colors:
        signals["colors"] = colors
    patterns = _above_threshold(pattern_counts)
    if patterns:
        signals["patterns"] = patterns
    fits = _above_threshold(fit_counts)
    if fits:
        signals["fit_types"] = fits

    return signals


# =============================================================================
# Singleton
# =============================================================================

_engine_instance: Optional[SessionScoringEngine] = None
_engine_lock = None

def get_session_scoring_engine(config: ScoringConfig = None) -> SessionScoringEngine:
    """Get or create the singleton scoring engine."""
    global _engine_instance, _engine_lock
    if _engine_lock is None:
        import threading
        _engine_lock = threading.Lock()

    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = SessionScoringEngine(config)
    return _engine_instance
