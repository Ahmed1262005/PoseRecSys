"""
V3 Feature Hydrator — batch-fetches full Candidate features from
the product_serving materialized view via v3_hydrate_candidates RPC.

Two usage patterns:
  - Pool rebuild hydration: 500 items via RPC
  - Page hydration: 24 items, instant from in-memory cache (0ms)

After rebuild, all hydrated candidates are cached in-process.
Subsequent page serves read from cache — zero DB calls on reuse.
"""

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from recs.models import Candidate

logger = logging.getLogger(__name__)

# Default LRU cache size: ~3000 items covers 6 concurrent sessions × 500 pool items
_DEFAULT_CACHE_SIZE = 3000


def _row_to_candidate(row: Dict[str, Any]) -> Candidate:
    """
    Convert a product_serving MV row to a Candidate model.

    Maps MV column names to Candidate field names.
    """
    gallery = row.get("gallery_images") or []
    if isinstance(gallery, str):
        gallery = [gallery] if gallery else []

    # pa_construction can be jsonb (dict) — extract closure_type or set None
    construction_raw = row.get("pa_construction")
    if isinstance(construction_raw, dict):
        construction_raw = construction_raw.get("closure_type") or None
    elif not isinstance(construction_raw, str):
        construction_raw = None

    # pa_coverage_details can be jsonb — leave as list or extract
    coverage_details = row.get("pa_coverage_details")
    if isinstance(coverage_details, dict):
        coverage_details = []
    elif not isinstance(coverage_details, list):
        coverage_details = coverage_details or []

    return Candidate(
        item_id=str(row.get("id", "")),
        name=row.get("name", ""),
        brand=row.get("brand", ""),
        category=row.get("category", ""),
        broad_category=row.get("broad_category", ""),
        article_type=row.get("article_type", ""),
        colors=row.get("colors") or [],
        materials=row.get("materials") or [],
        price=float(row.get("price", 0)) if row.get("price") is not None else 0.0,
        fit=row.get("fit"),
        length=row.get("length"),
        sleeve=row.get("sleeve"),
        neckline=row.get("neckline"),
        rise=row.get("rise"),
        style_tags=row.get("style_tags") or [],
        image_url=row.get("primary_image_url", ""),
        hero_image_url=row.get("hero_image_url"),
        gallery_images=gallery,
        # From product_attributes (pa_ prefixed)
        occasions=row.get("pa_occasions") or [],
        pattern=row.get("pa_pattern"),
        formality=row.get("pa_formality"),
        color_family=row.get("pa_color_family"),
        seasons=row.get("pa_seasons") or [],
        silhouette=row.get("pa_silhouette"),
        construction=construction_raw,
        coverage_level=row.get("pa_coverage_level"),
        skin_exposure=row.get("pa_skin_exposure"),
        coverage_details=coverage_details,
        model_body_type=row.get("pa_model_body_type"),
        model_size_estimate=row.get("pa_model_size_estimate"),
        # Computed fields
        original_price=row.get("original_price"),
        is_on_sale=bool(row.get("is_on_sale", False)),
        is_new=bool(row.get("is_new", False)),
        created_at=row.get("created_at"),
        gender=row.get("gender") or [],
        computed_occasion_scores=row.get("computed_occasion_scores") or {},
        computed_style_scores=row.get("computed_style_scores") or {},
        computed_pattern_scores=row.get("computed_pattern_scores") or {},
        image_dedup_key=row.get("image_dedup_key", ""),
        in_stock=True,
    )


class FeatureHydrator:
    """
    Batch fetches full product features from the product_serving
    materialized view.

    Maintains an in-process LRU cache so that:
    - Pool rebuild hydration: fetches from DB, populates cache
    - Page hydration on reuse: instant cache hits (0ms, no DB call)

    Cache is bounded by max_cache_size (default 3000 items).
    """

    def __init__(
        self,
        supabase_client: Any,
        max_cache_size: int = _DEFAULT_CACHE_SIZE,
    ) -> None:
        self._supabase = supabase_client
        self._max_cache_size = max_cache_size
        # OrderedDict for LRU eviction: most recently accessed at end
        self._cache: OrderedDict[str, Candidate] = OrderedDict()
        self._stats = {"hits": 0, "misses": 0, "rpc_calls": 0}

    # -- public API ---------------------------------------------------------

    def hydrate(self, item_ids: List[str]) -> List[Candidate]:
        """Fetch full Candidate objects for a list of item IDs.

        Uses cache for items already hydrated. Only fetches missing
        items from the DB.
        """
        if not item_ids:
            return []

        cached: Dict[str, Candidate] = {}
        uncached_ids: List[str] = []

        for iid in item_ids:
            if iid in self._cache:
                self._cache.move_to_end(iid)
                cached[iid] = self._cache[iid]
                self._stats["hits"] += 1
            else:
                uncached_ids.append(iid)
                self._stats["misses"] += 1

        # Fetch only uncached items from DB
        fetched: Dict[str, Candidate] = {}
        if uncached_ids:
            fetched = self._fetch_from_db(uncached_ids)
            # Populate cache
            for iid, candidate in fetched.items():
                self._put_cache(iid, candidate)

        # Combine cached + fetched, preserving input order is done by caller
        all_candidates = []
        merged = {**cached, **fetched}
        for iid in item_ids:
            if iid in merged:
                all_candidates.append(merged[iid])

        return all_candidates

    def hydrate_ordered(self, item_ids: List[str]) -> List[Candidate]:
        """
        Hydrate and return in the same order as item_ids.

        Items missing from the MV are skipped (no gap, just shorter list).
        This is the primary method for page serving where order matters.

        On reuse pages, this is near-instant (all items cached from rebuild).
        """
        if not item_ids:
            return []

        # Fast path: all items in cache
        if all(iid in self._cache for iid in item_ids):
            self._stats["hits"] += len(item_ids)
            result = []
            for iid in item_ids:
                self._cache.move_to_end(iid)
                result.append(self._cache[iid])
            return result

        # Slow path: mixed cache + DB
        return self.hydrate(item_ids)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics for debugging."""
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self._max_cache_size,
            **self._stats,
        }

    def invalidate(self, item_ids: Optional[List[str]] = None) -> None:
        """Remove items from cache. If item_ids is None, clear all."""
        if item_ids is None:
            self._cache.clear()
        else:
            for iid in item_ids:
                self._cache.pop(iid, None)

    # -- internal -----------------------------------------------------------

    def _fetch_from_db(self, item_ids: List[str]) -> Dict[str, Candidate]:
        """Fetch candidates from Supabase RPC. Returns dict keyed by item_id."""
        self._stats["rpc_calls"] += 1
        try:
            result = self._supabase.rpc(
                "v3_hydrate_candidates",
                {"p_ids": item_ids},
            ).execute()
        except Exception as e:
            logger.error("Hydration failed for %d items: %s", len(item_ids), e)
            return {}

        rows = result.data or []
        candidates: Dict[str, Candidate] = {}

        for row in rows:
            rid = str(row.get("id", "?"))
            try:
                c = _row_to_candidate(row)
                candidates[c.item_id] = c
            except Exception as e:
                logger.warning("Failed to hydrate row %s: %s", rid, e)

        missing = len(item_ids) - len(candidates)
        if missing > 0:
            logger.info(
                "Hydrated %d/%d items (%d missing from MV)",
                len(candidates), len(item_ids), missing,
            )

        return candidates

    def _put_cache(self, item_id: str, candidate: Candidate) -> None:
        """Add item to LRU cache, evicting oldest if at capacity."""
        if item_id in self._cache:
            self._cache.move_to_end(item_id)
            self._cache[item_id] = candidate
        else:
            if len(self._cache) >= self._max_cache_size:
                self._cache.popitem(last=False)  # evict oldest
            self._cache[item_id] = candidate
