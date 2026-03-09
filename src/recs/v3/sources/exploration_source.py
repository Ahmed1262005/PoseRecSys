"""
V3 Exploration Source — 3-tier discovery source.

Tiers: Safe (50%), Discovery (35%), Serendipity (15%)
Uses brand clusters for tier assignment and alternate key families
for serendipity.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Set

from recs.brand_clusters import get_cluster_adjacent_brands, CLUSTER_TO_BRANDS
from recs.v3.models import CandidateStub, FeedRequest, SessionProfile
from recs.v3.sources.preference_source import _assign_key_family

logger = logging.getLogger(__name__)

# Tier shares (sum to 1.0)
SAFE_SHARE = 0.50
DISCOVERY_SHARE = 0.35
SERENDIPITY_SHARE = 0.15


def _alternate_key_family(user_id: str) -> str:
    """Return a different explore_key family than the user's primary."""
    primary = _assign_key_family(user_id)
    families = ["a", "b", "c"]
    # Use a different hash to pick an alternate
    h = hashlib.md5((user_id + "_alt").encode()).hexdigest()
    idx = int(h[:8], 16) % 3
    alt = families[idx]
    if alt == primary:
        alt = families[(idx + 1) % 3]
    return alt


class ExplorationSource:
    """
    3-tier exploration source: Safe / Discovery / Serendipity.

    Safe: adjacent clusters, same explore_key family
    Discovery: clusters where session exposure < 2
    Serendipity: different key family, no cluster filter
    """

    def __init__(self, supabase_client: Any) -> None:
        self._supabase = supabase_client

    def retrieve(
        self,
        user_state: Any,
        session: SessionProfile,
        request: FeedRequest,
        exclude_ids: Set[str],
        limit: int = 75,
    ) -> List[CandidateStub]:
        if limit <= 0:
            return []

        safe_limit = max(1, int(limit * SAFE_SHARE))
        discovery_limit = max(1, int(limit * DISCOVERY_SHARE))
        serendipity_limit = limit - safe_limit - discovery_limit

        stubs: List[CandidateStub] = []

        # Tier 1: Safe — adjacent cluster brands
        safe_stubs = self._retrieve_safe(
            user_state, session, request, exclude_ids, safe_limit
        )
        stubs.extend(safe_stubs)

        # Tier 2: Discovery — low-exposure clusters
        discovery_stubs = self._retrieve_discovery(
            session, request, exclude_ids, discovery_limit
        )
        stubs.extend(discovery_stubs)

        # Tier 3: Serendipity — different key family
        serendipity_stubs = self._retrieve_serendipity(
            request, exclude_ids, serendipity_limit
        )
        stubs.extend(serendipity_stubs)

        return stubs[:limit]

    def _retrieve_safe(
        self,
        user_state: Any,
        session: SessionProfile,
        request: FeedRequest,
        exclude_ids: Set[str],
        limit: int,
    ) -> List[CandidateStub]:
        """Adjacent clusters, same explore_key family."""
        key_family = _assign_key_family(request.user_id)
        params = self._build_base_params(request, key_family)
        params["p_limit"] = limit

        # Get adjacent brands from user's top clusters
        adjacent_brands = []
        for cluster_id, count in session.cluster_exposure.most_common(3):
            adj = get_cluster_adjacent_brands([cluster_id])
            adjacent_brands.extend(list(adj)[:5])

        if adjacent_brands:
            params["p_include_brands"] = list(set(adjacent_brands))[:10]

        try:
            result = self._supabase.rpc(
                "v3_get_candidates_by_explore_key", params
            ).execute()
            return self._rows_to_stubs(result.data or [], exclude_ids, "exploration")
        except Exception as e:
            logger.error("ExplorationSource safe tier failed: %s", e)
            return []

    def _retrieve_discovery(
        self,
        session: SessionProfile,
        request: FeedRequest,
        exclude_ids: Set[str],
        limit: int,
    ) -> List[CandidateStub]:
        """Clusters where session exposure < 2."""
        key_family = _assign_key_family(request.user_id)
        params = self._build_base_params(request, key_family)
        params["p_limit"] = limit

        # Find low-exposure clusters
        low_exposure_brands = []
        for cluster_id, brands in CLUSTER_TO_BRANDS.items():
            if session.cluster_exposure.get(cluster_id, 0) < 2:
                low_exposure_brands.extend(list(brands)[:3])

        if low_exposure_brands:
            params["p_include_brands"] = list(set(low_exposure_brands))[:10]

        try:
            result = self._supabase.rpc(
                "v3_get_candidates_by_explore_key", params
            ).execute()
            return self._rows_to_stubs(result.data or [], exclude_ids, "exploration")
        except Exception as e:
            logger.error("ExplorationSource discovery tier failed: %s", e)
            return []

    def _retrieve_serendipity(
        self,
        request: FeedRequest,
        exclude_ids: Set[str],
        limit: int,
    ) -> List[CandidateStub]:
        """Different key family, no cluster filter."""
        alt_family = _alternate_key_family(request.user_id)
        params = self._build_base_params(request, alt_family)
        params["p_limit"] = limit

        try:
            result = self._supabase.rpc(
                "v3_get_candidates_by_explore_key", params
            ).execute()
            return self._rows_to_stubs(result.data or [], exclude_ids, "exploration")
        except Exception as e:
            logger.error("ExplorationSource serendipity tier failed: %s", e)
            return []

    def _build_base_params(
        self, request: FeedRequest, key_family: str
    ) -> Dict[str, Any]:
        """Build base RPC params."""
        hf = request.hard_filters
        params: Dict[str, Any] = {"p_key_family": key_family}

        if hf:
            gender = getattr(hf, "gender", None)
            if gender:
                params["p_gender"] = gender

            categories = getattr(hf, "categories", None)
            if categories:
                params["p_categories"] = categories

            min_price = getattr(hf, "min_price", None)
            if min_price is not None:
                params["p_min_price"] = min_price

            max_price = getattr(hf, "max_price", None)
            if max_price is not None:
                params["p_max_price"] = max_price

            exclude_brands = getattr(hf, "exclude_brands", None)
            if exclude_brands:
                params["p_exclude_brands"] = exclude_brands

        if request.mode == "sale":
            params["p_on_sale_only"] = True

        return params

    @staticmethod
    def _rows_to_stubs(
        rows: List[Dict], exclude_ids: Set[str], source: str
    ) -> List[CandidateStub]:
        stubs = []
        for row in rows:
            item_id = str(row["id"])
            if item_id in exclude_ids:
                continue
            stubs.append(CandidateStub(
                item_id=item_id,
                source=source,
                retrieval_score=float(row.get("explore_score", 0) or 0),
                brand=row.get("brand"),
                broad_category=row.get("broad_category"),
                article_type=row.get("article_type"),
                price=float(row.get("price", 0)) if row.get("price") else None,
                image_dedup_key=row.get("image_dedup_key"),
            ))
        return stubs
