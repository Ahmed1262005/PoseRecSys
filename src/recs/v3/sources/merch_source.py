"""
V3 Merch Source — merchandising source for new arrivals, sale items,
and promoted products.

Uses v3_get_candidates_by_freshness RPC.
Sale mode: only on-sale items.
New arrivals: last 14 days.
Explore mode: last 30 days.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from recs.v3.models import CandidateStub, FeedRequest, SessionProfile

logger = logging.getLogger(__name__)


def _row_to_merch_stub(row: Dict[str, Any]) -> CandidateStub:
    """Convert freshness RPC row to CandidateStub."""
    return CandidateStub(
        item_id=str(row["id"]),
        source="merch",
        retrieval_score=0.5,  # Fixed score for merch items
        brand=row.get("brand"),
        broad_category=row.get("broad_category"),
        article_type=row.get("article_type"),
        price=float(row.get("price", 0)) if row.get("price") else None,
        image_dedup_key=row.get("image_dedup_key"),
    )


class MerchSource:
    """
    Merchandising source for new arrivals, sale items, promoted products.

    Uses v3_get_candidates_by_freshness RPC.
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

        params = self._build_params(request)
        params["p_limit"] = limit

        try:
            result = self._supabase.rpc(
                "v3_get_candidates_by_freshness", params
            ).execute()

            stubs = []
            for row in (result.data or []):
                item_id = str(row["id"])
                if item_id not in exclude_ids:
                    stubs.append(_row_to_merch_stub(row))

            return stubs[:limit]
        except Exception as e:
            logger.error("MerchSource query failed: %s", e)
            return []

    def _build_params(self, request: FeedRequest) -> Dict[str, Any]:
        """Build RPC params from request."""
        hf = request.hard_filters
        params: Dict[str, Any] = {}

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

        # Mode-specific
        if request.mode == "sale":
            params["p_on_sale_only"] = True
            params["p_days"] = 90  # Wider window for sale
        elif request.mode == "new_arrivals":
            params["p_days"] = 14
        else:
            params["p_days"] = 30

        return params
