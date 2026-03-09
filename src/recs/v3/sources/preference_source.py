"""
V3 Preference Source — stable baseline retrieval based on user's
long-term profile.

Uses precomputed explore_key for deterministic pseudo-random ordering.
If preferred brands exist: 60% preferred, 40% general.
When include_brands is active: skips the preferred/general split.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Set

from recs.v3.models import CandidateStub, FeedRequest, SessionProfile

logger = logging.getLogger(__name__)


def _assign_key_family(user_id: str) -> str:
    """Deterministic key family from user_id via MD5 hash."""
    h = hashlib.md5(user_id.encode()).hexdigest()
    idx = int(h[:8], 16) % 3
    return ["a", "b", "c"][idx]


def _row_to_stub(row: Dict[str, Any], source: str, key_family: str) -> CandidateStub:
    """Convert Supabase RPC row to CandidateStub."""
    return CandidateStub(
        item_id=str(row["id"]),
        source=source,
        retrieval_score=float(row.get("explore_score", 0) or 0),
        brand=row.get("brand"),
        broad_category=row.get("broad_category"),
        article_type=row.get("article_type"),
        price=float(row.get("price", 0)) if row.get("price") else None,
        image_dedup_key=row.get("image_dedup_key"),
        retrieval_key=key_family,
    )


class PreferenceSource:
    """
    Stable baseline retrieval based on user's long-term profile.

    Uses precomputed explore_key for deterministic pseudo-random ordering.
    If preferred brands exist and include_brands is NOT active:
      60% preferred brands, 40% general.
    Otherwise: single general query.
    """

    def __init__(self, supabase_client: Any) -> None:
        self._supabase = supabase_client

    def retrieve(
        self,
        user_state: Any,
        session: SessionProfile,
        request: FeedRequest,
        exclude_ids: Set[str],
        limit: int = 225,
    ) -> List[CandidateStub]:
        if limit <= 0:
            return []

        key_family = _assign_key_family(request.user_id)
        hf = request.hard_filters

        # Check if include_brands is active
        include_brands = getattr(hf, "include_brands", None) if hf else None

        # Extract preferred brands from user profile (if any)
        preferred_brands = None
        if user_state and not include_brands:
            onboarding = getattr(user_state, "onboarding_profile", None)
            if onboarding:
                preferred_brands = getattr(onboarding, "preferred_brands", None)

        base_params = self._build_params(request, key_family)

        if preferred_brands and not include_brands:
            # Split: 60% preferred, 40% general
            pref_limit = int(limit * 0.6)
            gen_limit = limit - pref_limit

            pref_params = {**base_params, "p_preferred_brands": preferred_brands, "p_limit": pref_limit}
            gen_params = {**base_params, "p_limit": gen_limit}

            stubs = []
            try:
                pref_result = self._supabase.rpc(
                    "v3_get_candidates_by_explore_key", pref_params
                ).execute()
                for row in (pref_result.data or []):
                    if str(row["id"]) not in exclude_ids:
                        stubs.append(_row_to_stub(row, "preference", key_family))
            except Exception as e:
                logger.error("PreferenceSource preferred query failed: %s", e)

            try:
                gen_result = self._supabase.rpc(
                    "v3_get_candidates_by_explore_key", gen_params
                ).execute()
                for row in (gen_result.data or []):
                    if str(row["id"]) not in exclude_ids:
                        stubs.append(_row_to_stub(row, "preference", key_family))
            except Exception as e:
                logger.error("PreferenceSource general query failed: %s", e)

            return stubs[:limit]
        else:
            # Single general query (or include_brands forces single query)
            params = {**base_params, "p_limit": limit}
            if include_brands:
                params["p_include_brands"] = include_brands

            try:
                result = self._supabase.rpc(
                    "v3_get_candidates_by_explore_key", params
                ).execute()
                stubs = []
                for row in (result.data or []):
                    if str(row["id"]) not in exclude_ids:
                        stubs.append(_row_to_stub(row, "preference", key_family))
                return stubs[:limit]
            except Exception as e:
                logger.error("PreferenceSource query failed: %s", e)
                return []

    def _build_params(
        self, request: FeedRequest, key_family: str
    ) -> Dict[str, Any]:
        """Build base RPC params from request."""
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

            include_brands = getattr(hf, "include_brands", None)
            if include_brands:
                params["p_include_brands"] = include_brands

        if request.mode == "sale":
            params["p_on_sale_only"] = True
        elif request.mode == "new_arrivals":
            params["p_new_arrivals"] = True

        return params
