"""
Greedy Constrained List-wise Reranker for Fashion Feed.

This is the "secret sauce" that makes the feed feel like a human stylist.

Instead of taking the top-N by score, we BUILD the feed one item at a time,
enforcing constraints at each step:

1. Sort candidates by session score (descending)
2. For each position in the feed:
   a. Pick the highest-scoring candidate that passes ALL constraints
   b. Update constraint counters
   c. At certain positions, inject exploration items
3. Continue until the feed is full

Constraints:
- max_per_brand: No more than N items from the same brand
- max_per_cluster: No more than N items from the same persona cluster
- max_per_type: No more than N items from the same item type
- no_repeat_combo: Avoid showing items with identical attribute combos
- exploration_rate: Inject X% items from outside the user's comfort zone

This is the standard approach used by Zalando, Pinterest, Etsy, and Amazon
for fashion/marketplace feeds. It scales and converts to ML later because
every item position + constraints become features for learning-to-rank.
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
    """Tunable parameters for the greedy constrained reranker."""

    # --- Diversity caps ---
    max_per_brand: int = 4
    max_per_cluster: int = 8
    max_per_type: int = 6          # fallback for raw types not in a group

    # --- Category proportional caps ---
    # If set, overrides max_per_type for grouped categories.
    # Keys: "tops", "bottoms", "dresses", "outerwear", "_other"
    # Values: fraction of target_size (e.g. 0.35 = 35%)
    category_proportions: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CATEGORY_PROPORTIONS)
    )

    # --- Attribute combo dedup ---
    # Items with the same (brand_cluster, broad_category, color_family, fit)
    # are considered "too similar" -- only keep the first one.
    combo_dedup: bool = True

    # --- Exploration ---
    exploration_rate: float = 0.08  # 8% of positions are exploration
    exploration_min_position: int = 5  # Don't explore in first 5 positions
    exploration_cluster_penalty: float = 0.3  # How different exploration must be

    # --- Position-aware diversity ---
    # In the first N items, enforce stricter diversity (no brand repeats)
    strict_diversity_positions: int = 10

    # --- Seen penalty ---
    seen_penalty_weight: float = 0.8


DEFAULT_RERANKER_CONFIG = RerankerConfig()


# =============================================================================
# Reranker
# =============================================================================

class GreedyConstrainedReranker:
    """
    Build the feed one item at a time with constraints.

    Every serious commerce feed uses this pattern.
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
        Build a constrained, diverse feed from scored candidates.

        Args:
            candidates: Pre-scored candidates (must have final_score, brand, etc.)
            target_size: How many items to return
            seen_ids: Product IDs already shown (hard exclude)
            skipped_ids: Product IDs the user skipped (soft penalty)
            exploration_pool: Optional separate pool for exploration injection

        Returns:
            Reranked list of candidates (length <= target_size)
        """
        cfg = self.config
        seen_ids = seen_ids or set()
        skipped_ids = skipped_ids or set()

        # Step 1: Remove already-seen items
        available = [
            c for c in candidates
            if (getattr(c, "item_id", None) or getattr(c, "product_id", ""))
            not in seen_ids
        ]

        # Step 2: Sort by score (descending)
        available.sort(key=lambda c: getattr(c, "final_score", 0.0), reverse=True)

        # Step 3: Compute proportional category caps
        category_caps = compute_category_caps(
            target_size, cfg.category_proportions
        )

        # Step 4: Greedy selection with constraints
        result: List[Any] = []
        brand_counts: Dict[str, int] = defaultdict(int)
        cluster_counts: Dict[str, int] = defaultdict(int)
        type_counts: Dict[str, int] = defaultdict(int)
        group_counts: Dict[str, int] = defaultdict(int)
        used_combos: Set[str] = set()
        used_ids: Set[str] = set()

        # Track which candidates we've consumed
        candidate_idx = 0

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
                        type_counts, used_combos, used_ids, group_counts
                    )
                    continue

            # Find the next valid candidate
            picked = False
            scan_start = candidate_idx
            scan_count = 0

            while candidate_idx < len(available):
                candidate = available[candidate_idx]
                candidate_idx += 1
                scan_count += 1

                pid = getattr(candidate, "item_id", None) or getattr(candidate, "product_id", "")
                if pid in used_ids:
                    continue

                # Check constraints
                if self._passes_constraints(
                    candidate, position, brand_counts, cluster_counts,
                    type_counts, used_combos, group_counts, category_caps
                ):
                    result.append(candidate)
                    self._update_counters(
                        candidate, brand_counts, cluster_counts,
                        type_counts, used_combos, used_ids, group_counts
                    )
                    picked = True
                    break

                # If we've scanned too many without finding a match, relax constraints
                if scan_count > 50:
                    break

            if not picked:
                # Fallback: take the next available item regardless of constraints
                # (better to show something than nothing)
                for fallback_idx in range(candidate_idx, len(available)):
                    candidate = available[fallback_idx]
                    pid = getattr(candidate, "item_id", None) or getattr(candidate, "product_id", "")
                    if pid not in used_ids:
                        result.append(candidate)
                        self._update_counters(
                            candidate, brand_counts, cluster_counts,
                            type_counts, used_combos, used_ids, group_counts
                        )
                        candidate_idx = fallback_idx + 1
                        break
                else:
                    # Truly out of candidates
                    break

        return result

    def _passes_constraints(
        self,
        candidate: Any,
        position: int,
        brand_counts: Dict[str, int],
        cluster_counts: Dict[str, int],
        type_counts: Dict[str, int],
        used_combos: Set[str],
        group_counts: Dict[str, int] = None,
        category_caps: Dict[str, int] = None,
    ) -> bool:
        """Check if a candidate passes all diversity constraints."""
        cfg = self.config

        brand = (getattr(candidate, "brand", "") or "").lower()
        cluster_id = get_cluster_for_item(brand) if brand else DEFAULT_CLUSTER
        item_type = (
            getattr(candidate, "broad_category", "")
            or getattr(candidate, "article_type", "")
            or ""
        ).lower()

        # Brand cap
        if brand and brand_counts[brand] >= cfg.max_per_brand:
            return False

        # Strict diversity in first N positions (no brand repeats)
        if position < cfg.strict_diversity_positions:
            if brand and brand_counts[brand] > 0:
                return False

        # Cluster cap
        if cluster_id and cluster_counts[cluster_id] >= cfg.max_per_cluster:
            return False

        # Category group proportional cap (preferred over flat max_per_type)
        if category_caps and group_counts is not None:
            group = _resolve_category_group(candidate)
            if group == "_other":
                # Unknown categories: use the more generous of proportional
                # cap or flat max_per_type so we don't over-restrict misc items
                cap = max(category_caps.get(group, cfg.max_per_type), cfg.max_per_type)
            else:
                cap = category_caps.get(group, cfg.max_per_type)
            if group_counts[group] >= cap:
                return False
        elif item_type and type_counts[item_type] >= cfg.max_per_type:
            # Fallback: flat type cap when no proportional caps provided
            return False

        # Attribute combo dedup
        if cfg.combo_dedup:
            combo = self._get_combo_key(candidate, cluster_id)
            if combo and combo in used_combos:
                return False

        return True

    def _update_counters(
        self,
        candidate: Any,
        brand_counts: Dict[str, int],
        cluster_counts: Dict[str, int],
        type_counts: Dict[str, int],
        used_combos: Set[str],
        used_ids: Set[str],
        group_counts: Dict[str, int] = None,
    ) -> None:
        """Update constraint counters after placing an item."""
        brand = (getattr(candidate, "brand", "") or "").lower()
        cluster_id = get_cluster_for_item(brand) if brand else DEFAULT_CLUSTER
        item_type = (
            getattr(candidate, "broad_category", "")
            or getattr(candidate, "article_type", "")
            or ""
        ).lower()

        pid = getattr(candidate, "item_id", None) or getattr(candidate, "product_id", "")
        used_ids.add(pid)

        if brand:
            brand_counts[brand] += 1
        if cluster_id:
            cluster_counts[cluster_id] += 1
        if item_type:
            type_counts[item_type] += 1

        # Track category group counts for proportional allocation
        if group_counts is not None:
            group = _resolve_category_group(candidate)
            group_counts[group] += 1

        combo = self._get_combo_key(candidate, cluster_id)
        if combo:
            used_combos.add(combo)

    @staticmethod
    def _get_combo_key(candidate: Any, cluster_id: str = "") -> str:
        """
        Build attribute combo key for dedup.

        Items with the same (cluster, type, color_family, fit) are "too similar."
        """
        parts = [
            cluster_id or "",
            (getattr(candidate, "broad_category", "") or "").lower(),
            (getattr(candidate, "color_family", "") or "").lower(),
            (getattr(candidate, "fit", "") or "").lower(),
        ]
        key = "|".join(parts)
        # Don't dedup if most fields are empty
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
        import math
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
# Singleton
# =============================================================================

_reranker_instance: Optional[GreedyConstrainedReranker] = None

def get_feed_reranker(config: RerankerConfig = None) -> GreedyConstrainedReranker:
    """Get or create the singleton reranker."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = GreedyConstrainedReranker(config)
    return _reranker_instance
