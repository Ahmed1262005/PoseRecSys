"""
V3 Feature Hydrator — batch-fetches full Candidate features from
the product_serving materialized view via v3_hydrate_candidates RPC.

Two usage patterns:
  - Page hydration: 24 items, < 20ms
  - Pool rebuild hydration: 500 items, < 100ms
"""

import logging
from typing import Any, Dict, List, Optional

from recs.models import Candidate

logger = logging.getLogger(__name__)


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

    Two usage patterns:
    - Page hydration: 24 items, < 20ms
    - Pool rebuild hydration: 500 items, < 100ms
    """

    def __init__(self, supabase_client: Any) -> None:
        self._supabase = supabase_client

    def hydrate(self, item_ids: List[str]) -> List[Candidate]:
        """Fetch full Candidate objects for a list of item IDs."""
        if not item_ids:
            return []

        try:
            result = self._supabase.rpc(
                "v3_hydrate_candidates",
                {"p_ids": item_ids},
            ).execute()
        except Exception as e:
            logger.error("Hydration failed for %d items: %s", len(item_ids), e)
            return []

        rows = result.data or []
        candidates = []
        row_map: Dict[str, Dict] = {}

        for row in rows:
            rid = str(row.get("id", "?"))
            row_map[rid] = row
            try:
                candidates.append(_row_to_candidate(row))
            except Exception as e:
                logger.warning("Failed to hydrate row %s: %s", rid, e)

        missing = len(item_ids) - len(candidates)
        if missing > 0:
            logger.info(
                "Hydrated %d/%d items (%d missing from MV)",
                len(candidates), len(item_ids), missing,
            )

        return candidates

    def hydrate_ordered(self, item_ids: List[str]) -> List[Candidate]:
        """
        Hydrate and return in the same order as item_ids.

        Items missing from the MV are skipped (no gap, just shorter list).
        This is the primary method for page serving where order matters.
        """
        candidates = self.hydrate(item_ids)
        by_id = {c.item_id: c for c in candidates}
        return [by_id[iid] for iid in item_ids if iid in by_id]
