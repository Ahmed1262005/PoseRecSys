"""
V3 Session Source — retrieves candidates based on recent positive
session signals (clicks, saves, purchases).

Extracts top-K positive actions, weights by recency, queries
v3_get_candidates_by_explore_key with brand/type filters.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from recs.v3.models import CandidateStub, FeedRequest, SessionProfile

logger = logging.getLogger(__name__)

# Signal extraction constants
MAX_RECENT_POSITIVES = 10
RECENCY_DECAY = 0.85
MAX_PER_BRAND = 3
MAX_PER_CATEGORY = 4


def _extract_session_signals(session: SessionProfile) -> List[Dict[str, Any]]:
    """
    Extract top recent positive actions with recency weighting.

    Returns list of signal dicts with brand, article_type, weight.
    Caps per-brand and per-category counts.
    """
    positive_actions = {"click", "save", "cart", "purchase"}
    signals = []

    brand_counts: Dict[str, int] = {}
    cat_counts: Dict[str, int] = {}

    for i, action in enumerate(reversed(session.recent_actions)):
        if len(signals) >= MAX_RECENT_POSITIVES:
            break

        act_type = action.get("action", "")
        if act_type not in positive_actions:
            continue

        brand = action.get("brand") or ""
        article_type = action.get("article_type") or ""

        # Cap per-brand
        if brand:
            if brand_counts.get(brand, 0) >= MAX_PER_BRAND:
                continue
            brand_counts[brand] = brand_counts.get(brand, 0) + 1

        # Cap per-category
        if article_type:
            if cat_counts.get(article_type, 0) >= MAX_PER_CATEGORY:
                continue
            cat_counts[article_type] = cat_counts.get(article_type, 0) + 1

        # Recency decay
        weight = RECENCY_DECAY ** i

        signals.append({
            "brand": brand,
            "article_type": article_type,
            "weight": weight,
            "item_id": action.get("item_id", ""),
        })

    # Collapse near-duplicates (same brand+type)
    seen_combos: Set[str] = set()
    deduped = []
    for sig in signals:
        combo = f"{sig['brand']}|{sig['article_type']}"
        if combo in seen_combos:
            continue
        seen_combos.add(combo)
        deduped.append(sig)

    return deduped


class SessionSource:
    """
    Retrieves candidates based on recent positive session signals.

    If no positive actions exist, returns empty (fallback handled by mixer).
    """

    def __init__(self, supabase_client: Any) -> None:
        self._supabase = supabase_client

    def retrieve(
        self,
        user_state: Any,
        session: SessionProfile,
        request: FeedRequest,
        exclude_ids: Set[str],
        limit: int = 125,
    ) -> List[CandidateStub]:
        if limit <= 0:
            return []

        signals = _extract_session_signals(session)
        if not signals:
            return []

        stubs: List[CandidateStub] = []
        per_signal_limit = max(10, limit // max(len(signals), 1))

        for sig in signals:
            params = self._build_params(request)
            params["p_limit"] = per_signal_limit

            # Filter by signal's brand/type if available
            if sig["brand"]:
                params["p_include_brands"] = [sig["brand"]]

            try:
                result = self._supabase.rpc(
                    "v3_get_candidates_by_explore_key", params
                ).execute()

                weight = sig.get("weight", 1.0)
                for row in (result.data or []):
                    item_id = str(row["id"])
                    if item_id not in exclude_ids and item_id != sig.get("item_id"):
                        stubs.append(CandidateStub(
                            item_id=item_id,
                            source="session",
                            retrieval_score=float(row.get("explore_score", 0) or 0) * weight,
                            brand=row.get("brand"),
                            broad_category=row.get("broad_category"),
                            article_type=row.get("article_type"),
                            price=float(row.get("price", 0)) if row.get("price") else None,
                            image_dedup_key=row.get("image_dedup_key"),
                        ))
            except Exception as e:
                logger.error("SessionSource query failed: %s", e)

        return stubs[:limit]

    def _build_params(self, request: FeedRequest) -> Dict[str, Any]:
        """Build base RPC params from request."""
        hf = request.hard_filters
        params: Dict[str, Any] = {"p_key_family": "a"}

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
        elif request.mode == "new_arrivals":
            params["p_new_arrivals"] = True

        return params
