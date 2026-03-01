"""
Soft-Penalty Greedy Reranker for Fashion Feed.

Builds the feed one item at a time using SOFT score penalties for diversity
instead of hard caps that reject items. This ensures the feed always fills
regardless of candidate pool size.

Algorithm:
1. Sort candidates by session score (descending)
2. For each position in the feed:
   a. Scan all remaining candidates
   b. Compute adjusted_score = raw_score * brand_decay * cluster_decay
                               * combo_penalty * category_overshoot_penalty
   c. Pick the candidate with the highest adjusted_score
   d. Only HARD-REJECT if brand exceeds max_brand_share of feed
   e. Update counters
3. Continue until the feed is full

Why soft penalties instead of hard caps:
- Hard caps (max 4 per brand) kill feeds when the candidate pool is small
  (e.g., after attribute filtering for "evening" → 83 candidates, 5 brands)
- Soft penalties naturally adapt: with 80 brands, repeats get pushed down
  by fresh-brand items; with 5 brands, repeats still fill the feed
- One config works for both filtered and unfiltered feeds
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from recs.brand_clusters import get_cluster_for_item, DEFAULT_CLUSTER


# =============================================================================
# Category group mapping — maps raw broad_category / article_type values
# to a canonical group used for proportional allocation.
# =============================================================================

CATEGORY_GROUP_MAP: Dict[str, str] = {
    # Tops
    "tops": "tops", "knitwear": "tops", "shirts": "tops",
    "blouses": "tops", "sweaters": "tops", "t-shirts": "tops",
    "tank tops": "tops", "crop tops": "tops", "bodysuits": "tops",
    # Bottoms
    "bottoms": "bottoms", "jeans": "bottoms", "pants": "bottoms",
    "trousers": "bottoms", "shorts": "bottoms", "skirts": "bottoms",
    "leggings": "bottoms",
    # Dresses
    "dresses": "dresses", "dress": "dresses", "jumpsuits": "dresses",
    "rompers": "dresses",
    # Outerwear
    "outerwear": "outerwear", "jackets": "outerwear", "coats": "outerwear",
}

# Default proportional targets (must sum to ~1.0).
# Each value is the fraction of the target feed size reserved for that group.
DEFAULT_CATEGORY_PROPORTIONS: Dict[str, float] = {
    "tops": 0.35,
    "bottoms": 0.25,
    "dresses": 0.18,
    "outerwear": 0.10,
    "_other": 0.12,   # anything not in the map above
}


def _resolve_category_group(candidate: Any) -> str:
    """Map a candidate's broad_category to a canonical group name."""
    raw = (
        getattr(candidate, "broad_category", "")
        or getattr(candidate, "article_type", "")
        or ""
    ).lower().strip()
    return CATEGORY_GROUP_MAP.get(raw, "_other")


def compute_category_caps(
    target_size: int,
    proportions: Dict[str, float],
    min_per_group: int = 2,
) -> Dict[str, int]:
    """Convert proportional targets into absolute per-group caps.

    Each group gets at least *min_per_group* slots (so small categories
    still appear) but never more than its proportional share rounded up.
    """
    caps: Dict[str, int] = {}
    for group, frac in proportions.items():
        caps[group] = max(min_per_group, math.ceil(target_size * frac))
    return caps


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class RerankerConfig:
    """Tunable parameters for the soft-penalty greedy reranker.

    Diversity is achieved through score decay factors, not hard caps.
    The only hard reject is max_brand_share (safety net for degenerate pools).
    """

    # --- Soft penalty decay factors ---
    # Each additional item from the same brand/cluster multiplies the
    # adjusted score by the decay factor. e.g., brand_decay=0.85 means:
    #   1st item from brand: 1.0x, 2nd: 0.85x, 3rd: 0.72x, 4th: 0.61x
    brand_decay: float = 0.85
    cluster_decay: float = 0.92

    # Flat penalty for items with identical (cluster, category, color, fit)
    combo_penalty: float = 0.70

    # Penalty when a category group exceeds its proportional target
    category_overshoot_penalty: float = 0.80

    # --- Hard safety net ---
    # No single brand can exceed this fraction of the feed.
    # At target_size=50, max_brand_share=0.40 means max 20 items per brand.
    max_brand_share: float = 0.40

    # --- Strict diversity (top of feed only) ---
    # In the first N positions, require unique brands (hard reject).
    # Set to 3 — achievable even with small pools.
    strict_diversity_positions: int = 3

    # --- Category proportional targets (for overshoot penalty) ---
    category_proportions: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CATEGORY_PROPORTIONS)
    )

    # --- Exploration ---
    exploration_rate: float = 0.08  # 8% of positions are exploration
    exploration_min_position: int = 5  # Don't explore in first 5 positions
    exploration_cluster_penalty: float = 0.3  # How different exploration must be


DEFAULT_RERANKER_CONFIG = RerankerConfig()


# =============================================================================
# Reranker
# =============================================================================

class GreedyConstrainedReranker:
    """
    Build the feed one item at a time with soft score penalties.

    For each position, scans all remaining candidates and picks the one
    with the highest adjusted score. Diversity emerges naturally because
    repeated brands/clusters get progressively penalized.

    The only hard rejects are:
    - max_brand_share: prevents a single brand dominating the entire feed
    - strict_diversity_positions: first 3 items must be different brands
    """

    def __init__(self, config: RerankerConfig = None):
        self.config = config or DEFAULT_RERANKER_CONFIG

    def rerank(
        self,
        candidates: List[Any],
        target_size: int = 50,
        seen_ids: Set[str] = None,
        skipped_ids: Set[str] = None,
        exploration_pool: List[Any] = None,
    ) -> List[Any]:
        """
        Build a diverse feed from scored candidates using soft penalties.

        Args:
            candidates: Pre-scored candidates (must have final_score, brand, etc.)
            target_size: How many items to return
            seen_ids: Product IDs already shown (hard exclude)
            skipped_ids: Product IDs the user skipped (not used by reranker)
            exploration_pool: Optional separate pool for exploration injection

        Returns:
            Reranked list of candidates (length <= target_size)
        """
        cfg = self.config
        seen_ids = seen_ids or set()

        # Step 1: Remove already-seen items, mark available
        available = [
            c for c in candidates
            if (getattr(c, "item_id", None) or getattr(c, "product_id", ""))
            not in seen_ids
        ]

        # Step 2: Sort by raw score (descending) for tie-breaking
        available.sort(key=lambda c: getattr(c, "final_score", 0.0), reverse=True)

        # Step 3: Compute proportional category targets (for overshoot detection)
        category_caps = compute_category_caps(
            target_size, cfg.category_proportions
        )

        # Hard brand cap (the only hard diversity limit)
        max_brand_items = max(3, int(target_size * cfg.max_brand_share))

        # Step 4: Greedy selection with soft penalties
        result: List[Any] = []
        brand_counts: Dict[str, int] = defaultdict(int)
        cluster_counts: Dict[str, int] = defaultdict(int)
        group_counts: Dict[str, int] = defaultdict(int)
        used_combos: Set[str] = set()
        used_ids: Set[str] = set()

        for position in range(target_size):
            # Check if we should inject exploration
            if (
                position >= cfg.exploration_min_position
                and random.random() < cfg.exploration_rate
                and exploration_pool
            ):
                exp_item = self._pick_exploration_item(
                    exploration_pool, used_ids, brand_counts, cluster_counts
                )
                if exp_item:
                    result.append(exp_item)
                    self._update_counters(
                        exp_item, brand_counts, cluster_counts,
                        used_combos, used_ids, group_counts
                    )
                    continue

            # Scan all remaining candidates, pick the best adjusted score
            best_candidate = None
            best_adjusted = -float("inf")

            for candidate in available:
                pid = (
                    getattr(candidate, "item_id", None)
                    or getattr(candidate, "product_id", "")
                )
                if pid in used_ids:
                    continue

                brand = (getattr(candidate, "brand", "") or "").lower()
                cluster_id = (
                    get_cluster_for_item(brand) if brand else DEFAULT_CLUSTER
                )

                # === HARD REJECTS (only two) ===

                # 1. Brand share cap — safety net
                if brand and brand_counts[brand] >= max_brand_items:
                    continue

                # 2. Strict diversity in first N positions
                if position < cfg.strict_diversity_positions:
                    if brand and brand_counts[brand] > 0:
                        continue

                # === SOFT PENALTIES ===

                raw_score = getattr(candidate, "final_score", 0.0)
                adjusted = raw_score

                # Brand decay: 0.85^n for nth item from same brand
                if brand:
                    adjusted *= cfg.brand_decay ** brand_counts[brand]

                # Cluster decay: 0.92^n for nth item from same cluster
                if cluster_id:
                    adjusted *= cfg.cluster_decay ** cluster_counts[cluster_id]

                # Combo penalty: 0.70 if duplicate attribute combo
                combo_key = self._get_combo_key(candidate, cluster_id)
                if combo_key and combo_key in used_combos:
                    adjusted *= cfg.combo_penalty

                # Category overshoot penalty: 0.80 if group exceeds target
                group = _resolve_category_group(candidate)
                cap = category_caps.get(group, 999)
                if group_counts[group] >= cap:
                    adjusted *= cfg.category_overshoot_penalty

                if adjusted > best_adjusted:
                    best_adjusted = adjusted
                    best_candidate = candidate

            if best_candidate is None:
                # Truly out of candidates
                break

            result.append(best_candidate)
            self._update_counters(
                best_candidate, brand_counts, cluster_counts,
                used_combos, used_ids, group_counts
            )

        return result

    def _update_counters(
        self,
        candidate: Any,
        brand_counts: Dict[str, int],
        cluster_counts: Dict[str, int],
        used_combos: Set[str],
        used_ids: Set[str],
        group_counts: Dict[str, int],
    ) -> None:
        """Update counters after placing an item."""
        brand = (getattr(candidate, "brand", "") or "").lower()
        cluster_id = get_cluster_for_item(brand) if brand else DEFAULT_CLUSTER

        pid = (
            getattr(candidate, "item_id", None)
            or getattr(candidate, "product_id", "")
        )
        used_ids.add(pid)

        if brand:
            brand_counts[brand] += 1
        if cluster_id:
            cluster_counts[cluster_id] += 1

        group = _resolve_category_group(candidate)
        group_counts[group] += 1

        combo = self._get_combo_key(candidate, cluster_id)
        if combo:
            used_combos.add(combo)

    @staticmethod
    def _get_combo_key(candidate: Any, cluster_id: str = "") -> str:
        """
        Build attribute combo key for soft penalty.

        Items with the same (cluster, type, color_family, fit) are "too similar."
        """
        parts = [
            cluster_id or "",
            (getattr(candidate, "broad_category", "") or "").lower(),
            (getattr(candidate, "color_family", "") or "").lower(),
            (getattr(candidate, "fit", "") or "").lower(),
        ]
        key = "|".join(parts)
        # Don't penalize if most fields are empty
        if key.count("|") >= 3 and key.replace("|", "") == "":
            return ""
        return key

    def _pick_exploration_item(
        self,
        pool: List[Any],
        used_ids: Set[str],
        brand_counts: Dict[str, int],
        cluster_counts: Dict[str, int],
    ) -> Optional[Any]:
        """
        Pick an exploration item from a separate pool.

        Prefers items from under-represented clusters and brands.
        """
        random.shuffle(pool)
        for item in pool:
            pid = getattr(item, "item_id", None) or getattr(item, "product_id", "")
            if pid in used_ids:
                continue

            brand = (getattr(item, "brand", "") or "").lower()
            cluster_id = get_cluster_for_item(brand) if brand else DEFAULT_CLUSTER

            # Prefer items from clusters not yet well-represented
            if cluster_counts.get(cluster_id, 0) < 2:
                return item
            # Also accept if brand is not yet shown
            if brand_counts.get(brand, 0) == 0:
                return item

        # Fallback: any unused item
        for item in pool:
            pid = getattr(item, "item_id", None) or getattr(item, "product_id", "")
            if pid not in used_ids:
                return item

        return None

    def get_diversity_stats(self, result: List[Any]) -> Dict[str, Any]:
        """Get diversity statistics for a reranked feed (for monitoring)."""
        brand_counts: Dict[str, int] = defaultdict(int)
        cluster_counts: Dict[str, int] = defaultdict(int)
        type_counts: Dict[str, int] = defaultdict(int)
        group_counts: Dict[str, int] = defaultdict(int)

        for item in result:
            brand = (getattr(item, "brand", "") or "").lower()
            cluster_id = get_cluster_for_item(brand) if brand else DEFAULT_CLUSTER
            item_type = (
                getattr(item, "broad_category", "")
                or getattr(item, "article_type", "")
                or ""
            ).lower()

            if brand:
                brand_counts[brand] += 1
            if cluster_id:
                cluster_counts[cluster_id] += 1
            if item_type:
                type_counts[item_type] += 1
            group_counts[_resolve_category_group(item)] += 1

        return {
            "total_items": len(result),
            "unique_brands": len(brand_counts),
            "unique_clusters": len(cluster_counts),
            "unique_types": len(type_counts),
            "top_brands": sorted(brand_counts.items(), key=lambda x: -x[1])[:5],
            "top_clusters": sorted(cluster_counts.items(), key=lambda x: -x[1])[:5],
            "top_types": sorted(type_counts.items(), key=lambda x: -x[1])[:5],
            "category_groups": dict(sorted(group_counts.items(), key=lambda x: -x[1])),
            "brand_entropy": self._entropy(brand_counts),
            "cluster_entropy": self._entropy(cluster_counts),
        }

    @staticmethod
    def _entropy(counts: Dict[str, int]) -> float:
        """Shannon entropy of a distribution (higher = more diverse)."""
        total = sum(counts.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return round(entropy, 3)


# =============================================================================
# Light Diversity for Sorted Feeds
# =============================================================================

def apply_sort_diversity(
    candidates: List[Any],
    max_consecutive: int = 2,
    max_brand_share: float = 0.30,
    seen_ids: Optional[Set[str]] = None,
) -> List[Any]:
    """
    Apply lightweight brand diversity to a *deterministically sorted* list.

    Unlike the full greedy reranker (which reorders by adjusted score), this
    function preserves the caller's sort order as much as possible.  It only
    intervenes when:

    1. **Consecutive-brand limit** — more than ``max_consecutive`` items in a
       row share the same brand.  The excess item is deferred (pushed down a
       few positions) rather than removed.
    2. **Brand-share cap** — a single brand exceeds ``max_brand_share`` of the
       total page.  Excess items are deferred to the end.

    Deferred items retain their relative sort order among themselves.

    Args:
        candidates: Pre-sorted candidate list (e.g. by price ASC).
        max_consecutive: Maximum items in a row from the same brand (default 2).
        max_brand_share: Maximum fraction of the result from one brand (default 0.30).
        seen_ids: Optional set of product IDs to exclude entirely.

    Returns:
        Diversified list (same length or shorter if seen_ids removed items).
    """
    seen_ids = seen_ids or set()

    # Strip already-seen items first
    pool = [
        c for c in candidates
        if (getattr(c, "item_id", None) or getattr(c, "product_id", ""))
        not in seen_ids
    ]

    if not pool:
        return []

    max_per_brand = max(3, int(len(pool) * max_brand_share))

    result: List[Any] = []
    deferred: List[Any] = []
    brand_counts: Dict[str, int] = defaultdict(int)

    for item in pool:
        brand = (getattr(item, "brand", "") or "").lower()
        pid = getattr(item, "item_id", None) or getattr(item, "product_id", "")

        # Check brand share cap
        if brand and brand_counts[brand] >= max_per_brand:
            deferred.append(item)
            continue

        # Check consecutive brand limit
        if brand and len(result) >= max_consecutive:
            recent_brands = [
                (getattr(r, "brand", "") or "").lower()
                for r in result[-max_consecutive:]
            ]
            if all(b == brand for b in recent_brands):
                deferred.append(item)
                continue

        result.append(item)
        if brand:
            brand_counts[brand] += 1

    # Append deferred items at the end (preserving their relative sort order)
    result.extend(deferred)
    return result


# =============================================================================
# Singleton
# =============================================================================

_reranker_instance: Optional[GreedyConstrainedReranker] = None

def get_feed_reranker(config: RerankerConfig = None) -> GreedyConstrainedReranker:
    """Get or create the singleton reranker."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = GreedyConstrainedReranker(config)
    return _reranker_instance
