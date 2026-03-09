"""
V3 Greedy Constrained Diversity Reranker.

Enforces brand diversity, category proportions, exploration injection,
and near-duplicate suppression via combo keys.
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from recs.brand_clusters import get_cluster_for_item, DEFAULT_CLUSTER

logging = __import__("logging")
logger = logging.getLogger(__name__)


# =========================================================================
# Category group mapping
# =========================================================================

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
    "dresses": "dresses",
    # Outerwear
    "outerwear": "outerwear",
}

DEFAULT_CATEGORY_PROPORTIONS: Dict[str, float] = {
    "tops": 0.35,
    "bottoms": 0.25,
    "dresses": 0.18,
    "outerwear": 0.10,
    "_other": 0.12,
}


def _resolve_category_group(candidate: Any) -> str:
    """Map a candidate's broad_category to a canonical group name."""
    bc = (getattr(candidate, "broad_category", None) or "").lower().strip()
    at = (getattr(candidate, "article_type", None) or "").lower().strip()
    # Try broad_category first, then article_type
    return CATEGORY_GROUP_MAP.get(bc) or CATEGORY_GROUP_MAP.get(at) or "_other"


def _compute_category_caps(
    target_size: int,
    proportions: Dict[str, float],
    min_per_group: int = 1,
) -> Dict[str, int]:
    """Convert proportional targets into absolute per-group caps."""
    caps = {}
    for group, frac in proportions.items():
        caps[group] = max(min_per_group, math.ceil(target_size * frac))
    return caps


# =========================================================================
# Config
# =========================================================================

@dataclass
class V3RerankerConfig:
    """Tunable parameters for the V3 greedy constrained reranker.

    Same constants as V2 feed_reranker.py RerankerConfig.
    """
    brand_decay: float = 0.85
    cluster_decay: float = 0.92
    combo_penalty: float = 0.7
    category_overshoot_penalty: float = 0.8
    max_brand_share: float = 0.4
    strict_diversity_positions: int = 3
    category_proportions: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CATEGORY_PROPORTIONS)
    )
    exploration_rate: float = 0.08
    exploration_min_position: int = 5


DEFAULT_V3_RERANKER_CONFIG = V3RerankerConfig()


# =========================================================================
# Reranker
# =========================================================================

class V3Reranker:
    """
    Greedy constrained diversity reranker.

    Picks items one-at-a-time, applying:
      - Strict brand diversity in first 3 positions
      - Brand share cap (max 40%)
      - Category proportion targets
      - Exploration injection
      - Near-duplicate suppression (combo keys)
    """

    def __init__(self, config: V3RerankerConfig = None) -> None:
        self.config = config or DEFAULT_V3_RERANKER_CONFIG

    def rerank(
        self,
        candidates: List[Any],
        target_size: int = 24,
        seen_ids: Optional[Set[str]] = None,
        exploration_pool: Optional[List[Any]] = None,
    ) -> List[Any]:
        """Greedy constrained reranking.

        Optimizations over naive O(target_size * pool_size):
        - Pre-compute brand/cluster/group/combo for all candidates once
        - Pre-compute unique_brands once for strict diversity relaxation
        - Remove chosen items via swap-to-end instead of list.pop(middle)
        """
        seen = seen_ids or set()
        exp_pool = list(exploration_pool or [])

        # Filter out already-seen items
        pool = [c for c in candidates if _get_item_id(c) not in seen]

        # Sort by final_score descending
        pool.sort(key=lambda c: getattr(c, "final_score", 0.0), reverse=True)

        if not pool:
            return []

        # Pre-compute per-candidate metadata once (avoids repeated getattr + lower() + lookups)
        _pre_ids: List[str] = []
        _pre_brands: List[str] = []
        _pre_clusters: List[str] = []
        _pre_groups: List[str] = []
        _pre_combos: List[str] = []
        _pre_scores: List[float] = []

        for c in pool:
            _pre_ids.append(_get_item_id(c))
            brand = (getattr(c, "brand", None) or "").lower()
            _pre_brands.append(brand)
            cluster = get_cluster_for_item(brand)
            _pre_clusters.append(cluster)
            _pre_groups.append(_resolve_category_group(c))
            _pre_combos.append(_get_combo_key(c, cluster))
            _pre_scores.append(getattr(c, "final_score", 0.0))

        # Pre-compute unique brand count for strict diversity relaxation
        unique_brand_count = len(set(b for b in _pre_brands if b))

        result: List[Any] = []
        brand_counts: Dict[str, int] = defaultdict(int)
        cluster_counts: Dict[str, int] = defaultdict(int)
        used_combos: Set[str] = set()
        used_ids: Set[str] = set()
        group_counts: Dict[str, int] = defaultdict(int)

        max_brand_count = max(1, int(target_size * self.config.max_brand_share))
        category_caps = _compute_category_caps(
            target_size, self.config.category_proportions,
        )

        # Active pool tracking: indices into pool that are still available
        active: List[int] = list(range(len(pool)))

        while len(result) < target_size and active:
            # Exploration injection
            if (
                exp_pool
                and len(result) >= self.config.exploration_min_position
                and random.random() < self.config.exploration_rate
            ):
                exp_item = self._pick_exploration_item(
                    exp_pool, used_ids, brand_counts, cluster_counts,
                )
                if exp_item:
                    result.append(exp_item)
                    self._update_counters(
                        exp_item, brand_counts, cluster_counts,
                        used_combos, used_ids, group_counts,
                    )
                    continue

            best_active_pos = -1
            best_score = float("-inf")

            pos = len(result)
            is_strict_pos = pos < self.config.strict_diversity_positions

            for ai, idx in enumerate(active):
                cid = _pre_ids[idx]
                if cid in used_ids:
                    continue

                brand = _pre_brands[idx]
                score = _pre_scores[idx]

                # Strict diversity: first N positions must have unique brands
                if is_strict_pos and brand and brand_counts.get(brand, 0) > 0:
                    if unique_brand_count > 2:
                        continue

                cluster = _pre_clusters[idx]
                group = _pre_groups[idx]
                combo_key = _pre_combos[idx]

                # Brand cap
                if brand and brand_counts.get(brand, 0) >= max_brand_count:
                    score *= 0.8

                # Brand decay
                bc = brand_counts.get(brand, 0)
                if brand and bc > 0:
                    score *= self.config.brand_decay ** bc

                # Cluster decay
                if cluster != DEFAULT_CLUSTER:
                    cc = cluster_counts.get(cluster, 0)
                    if cc > 0:
                        score *= self.config.cluster_decay ** cc

                # Combo penalty
                if combo_key and combo_key in used_combos:
                    score *= self.config.combo_penalty

                # Category overshoot
                cap = category_caps.get(group, 999)
                if group_counts.get(group, 0) >= cap:
                    score *= self.config.category_overshoot_penalty

                if score > best_score:
                    best_score = score
                    best_active_pos = ai

            if best_active_pos < 0:
                break

            chosen_idx = active[best_active_pos]
            chosen = pool[chosen_idx]

            # Remove from active list: swap with last, then pop
            active[best_active_pos] = active[-1]
            active.pop()

            result.append(chosen)
            # Update counters using pre-computed values
            brand = _pre_brands[chosen_idx]
            cluster = _pre_clusters[chosen_idx]
            group = _pre_groups[chosen_idx]
            combo_key = _pre_combos[chosen_idx]
            cid = _pre_ids[chosen_idx]

            used_ids.add(cid)
            if brand:
                brand_counts[brand] = brand_counts.get(brand, 0) + 1
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
            group_counts[group] = group_counts.get(group, 0) + 1
            if combo_key:
                used_combos.add(combo_key)

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
        """Update all counters after placing an item."""
        cid = _get_item_id(candidate)
        used_ids.add(cid)

        brand = (getattr(candidate, "brand", None) or "").lower()
        if brand:
            brand_counts[brand] = brand_counts.get(brand, 0) + 1

        cluster = get_cluster_for_item(brand) or "unknown"
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

        group = _resolve_category_group(candidate)
        group_counts[group] = group_counts.get(group, 0) + 1

        combo_key = _get_combo_key(candidate, cluster)
        used_combos.add(combo_key)

    def _pick_exploration_item(
        self,
        pool: List[Any],
        used_ids: Set[str],
        brand_counts: Dict[str, int],
        cluster_counts: Dict[str, int],
    ) -> Optional[Any]:
        """Pick an exploration item from a separate pool."""
        for i, c in enumerate(pool):
            cid = _get_item_id(c)
            if cid in used_ids:
                continue
            brand = (getattr(c, "brand", None) or "").lower()
            # Prefer items from under-represented brands
            if brand and brand_counts.get(brand, 0) < 2:
                return pool.pop(i)
        # Fallback: pick first unused
        for i, c in enumerate(pool):
            if _get_item_id(c) not in used_ids:
                return pool.pop(i)
        return None

    def get_diversity_stats(self, result: List[Any]) -> Dict[str, Any]:
        """Get diversity statistics for monitoring."""
        brand_dist: Dict[str, int] = defaultdict(int)
        group_dist: Dict[str, int] = defaultdict(int)

        for c in result:
            brand = (getattr(c, "brand", None) or "").lower()
            if brand:
                brand_dist[brand] += 1
            group = _resolve_category_group(c)
            group_dist[group] += 1

        n = len(result) or 1
        return {
            "total_items": len(result),
            "unique_brands": len(brand_dist),
            "brand_distribution": dict(
                sorted(brand_dist.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
            "category_distribution": dict(group_dist),
            "brand_entropy": _entropy(brand_dist),
            "category_entropy": _entropy(group_dist),
            "max_brand_share": max(brand_dist.values(), default=0) / n,
        }


# =========================================================================
# Helpers
# =========================================================================

def _get_item_id(candidate: Any) -> str:
    """Extract item ID from candidate."""
    return getattr(candidate, "item_id", None) or getattr(candidate, "product_id", "")


def _get_combo_key(candidate: Any, cluster_id: str) -> str:
    """Build attribute combo key for the soft combo penalty."""
    parts = [
        (getattr(candidate, "broad_category", None) or "").lower(),
        (getattr(candidate, "color_family", None) or "").lower(),
        (getattr(candidate, "fit", None) or "").lower(),
    ]
    key = "|".join(parts)
    # Only penalize if combo has 3+ distinct non-empty parts
    if key.count("|") >= 2 and key.replace("|", ""):
        return key
    return ""


def _entropy(counts: Dict[str, int]) -> float:
    """Shannon entropy of a distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return round(entropy, 3)
