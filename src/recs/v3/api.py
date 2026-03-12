"""
V3 Feed API — thin FastAPI endpoint layer.

Routes:
    GET  /api/recs/v3/feed                      — personalized feed
    POST /api/recs/v3/feed/action               — record user action
    GET  /api/recs/v3/feed/health               — V3 health check
    GET  /api/recs/v3/feed/session/{session_id}  — session info
    DELETE /api/recs/v3/feed/session/{session_id} — clear session

All endpoints except health require Supabase JWT auth.
Param parsing matches V2 for feature parity. The endpoint bodies
are thin: parse → FeedRequest → orchestrator → JSON.

See docs/V3_FEED_ARCHITECTURE_PLAN.md §Slice 5.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from core.auth import SupabaseUser, require_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recs/v3", tags=["Recommendations V3 (Pool-based)"])

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_orchestrator = None


def get_orchestrator():
    """
    Lazy-init the FeedOrchestrator with all 10 dependencies.

    Same pattern as V2's ``get_pipeline()`` — module-level singleton
    created on first call. Returns the cached instance thereafter.
    """
    global _orchestrator
    if _orchestrator is not None:
        return _orchestrator

    try:
        from config.database import get_supabase_client
        from recs.v3.eligibility import EligibilityFilter
        from recs.v3.events import EventLogger
        from recs.v3.feature_hydrator import FeatureHydrator
        from recs.v3.mixer import CandidateMixer
        from recs.v3.orchestrator import FeedOrchestrator
        from recs.v3.pool_manager import PoolManager
        from recs.v3.ranker import FeedRanker
        from recs.v3.reranker import V3Reranker
        from recs.v3.session_store import get_session_store
        from recs.v3.sources.exploration_source import ExplorationSource
        from recs.v3.sources.merch_source import MerchSource
        from recs.v3.sources.preference_source import PreferenceSource
        from recs.v3.sources.session_source import SessionSource
        from recs.v3.user_profile import UserProfileLoader

        supabase = get_supabase_client()

        _orchestrator = FeedOrchestrator(
            session_store=get_session_store("auto"),
            user_profile=UserProfileLoader(supabase),
            pool_manager=PoolManager(),
            sources={
                "preference": PreferenceSource(supabase),
                "session": SessionSource(supabase),
                "exploration": ExplorationSource(supabase),
                "merch": MerchSource(supabase),
            },
            mixer=CandidateMixer(),
            hydrator=FeatureHydrator(supabase),
            eligibility=EligibilityFilter(),
            ranker=FeedRanker(),
            reranker=V3Reranker(),
            events=EventLogger(supabase),
        )

        logger.info("V3 FeedOrchestrator initialized")
        return _orchestrator

    except Exception as e:
        logger.error("Failed to initialize V3 FeedOrchestrator: %s", e)
        raise


def _get_orchestrator_or_503():
    """Get orchestrator, raising 503 if not available."""
    try:
        return get_orchestrator()
    except Exception as e:
        raise HTTPException(status_code=503, detail="V3 feed service unavailable: " + str(e))


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class V3ActionRequest(BaseModel):
    """Request body for POST /feed/action."""
    session_id: str = Field(..., description="Session ID from feed response")
    product_id: str = Field(..., description="Product UUID that was interacted with")
    action: str = Field(..., description="Action type: click, save, cart, purchase, skip, hide, hover")
    source: str = Field(default="feed", description="Source: feed, search, similar")
    position: Optional[int] = Field(default=None, description="Position in feed when interacted")
    brand: Optional[str] = Field(default=None, description="Product brand (for session scoring)")
    item_type: Optional[str] = Field(default=None, description="Product article type")
    cluster_id: Optional[str] = Field(default=None, description="Brand cluster ID")
    attributes: Optional[Dict[str, Any]] = Field(default=None, description="Product attributes: {fit, color_family, pattern, ...}")


class V3ActionResponse(BaseModel):
    """Response from POST /feed/action."""
    status: str
    action_seq: Optional[int] = None


class V3SearchSignalRequest(BaseModel):
    """Request body for POST /feed/search-signal."""
    session_id: str = Field(..., description="V3 session ID (from feed response)")
    query: str = Field(..., description="Search query text")
    categories: Optional[List[str]] = Field(default=None, description="Matched categories (e.g. ['dresses'])")
    brands: Optional[List[str]] = Field(default=None, description="Matched brands")
    article_types: Optional[List[str]] = Field(default=None, description="Matched article types (e.g. ['midi dress', 'maxi dress'])")


VALID_ACTIONS = {
    "click", "save", "wishlist", "cart", "add_to_cart",
    "purchase", "skip", "hide", "dislike", "hover",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_csv(val: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated string to list, or None."""
    if not val:
        return None
    return [v.strip() for v in val.split(",") if v.strip()]


def _build_hard_filters(
    gender: Optional[str],
    categories: Optional[str],
    article_types: Optional[str],
    exclude_colors: Optional[str],
    include_colors: Optional[str],
    exclude_brands: Optional[str],
    include_brands: Optional[str],
    min_price: Optional[float],
    max_price: Optional[float],
    exclude_styles: Optional[str],
    include_occasions: Optional[str],
    include_patterns: Optional[str],
    exclude_patterns: Optional[str],
) -> Any:
    """Build HardFilters from query params."""
    # Any param that is None after _parse_csv stays None (not filtered)
    from recs.models import HardFilters

    return HardFilters(
        gender=gender,
        categories=_parse_csv(categories),
        article_types=_parse_csv(article_types),
        exclude_colors=_parse_csv(exclude_colors),
        include_colors=_parse_csv(include_colors),
        exclude_brands=_parse_csv(exclude_brands),
        include_brands=_parse_csv(include_brands),
        min_price=min_price,
        max_price=max_price,
        exclude_styles=_parse_csv(exclude_styles),
        include_occasions=_parse_csv(include_occasions),
        include_patterns=_parse_csv(include_patterns),
        exclude_patterns=_parse_csv(exclude_patterns),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/feed",
    summary="Get V3 personalized feed (pool-based)",
    description="""
    Get personalized product recommendations using the V3 pool-based pipeline.

    **Architecture:** Pool-based serving with multi-source retrieval.
    First page builds a pool (~400-700ms), subsequent pages serve from pool (~30-50ms).

    **Pipeline:**
    1. 4 parallel retrieval sources (preference, session, exploration, merch)
    2. Mix + dedup (500 item pool)
    3. Unified scoring (9 signal components)
    4. Greedy constrained diversity reranking
    5. Serve from cached pool with serve-time validation

    **Pagination:**
    - First request: Don't send `cursor` or `session_id`
    - Subsequent: Send back `cursor` and `session_id` from previous response

    **Modes:**
    - `explore` (default): Full personalized feed
    - `sale`: Sale items only
    - `new_arrivals`: New arrivals only
    """,
)
async def get_v3_feed(
    # Auth
    user: SupabaseUser = Depends(require_auth),
    # Pagination / session
    session_id: Optional[str] = Query(None, description="Session ID (returned in response, send back for pagination)"),
    mode: str = Query("explore", description="Feed mode: explore (default), sale, new_arrivals"),
    page_size: int = Query(24, ge=1, le=100, description="Items per page (max 100)"),
    cursor: Optional[str] = Query(None, description="Cursor from previous response (for pagination)"),
    debug: bool = Query(False, description="Include debug metadata in response"),
    # Hard filters
    gender: Optional[str] = Query("female", description="Gender filter"),
    categories: Optional[str] = Query(None, description="Comma-separated broad categories (tops, bottoms, dresses, outerwear)"),
    article_types: Optional[str] = Query(None, description="Comma-separated article types (jeans, t-shirts, sweaters)"),
    exclude_styles: Optional[str] = Query(None, description="Comma-separated coverage styles to avoid (deep-necklines, sheer, cutouts, backless, strapless)"),
    include_occasions: Optional[str] = Query(None, description="Comma-separated occasions (casual, office, evening, beach, active)"),
    exclude_occasions: Optional[str] = Query(None, description="Comma-separated occasions to exclude"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price"),
    max_price: Optional[float] = Query(None, description="Maximum price"),
    exclude_brands: Optional[str] = Query(None, description="Comma-separated brands to exclude"),
    include_brands: Optional[str] = Query(None, description="Comma-separated brands to include (hard filter)"),
    exclude_colors: Optional[str] = Query(None, description="Comma-separated colors to exclude"),
    include_colors: Optional[str] = Query(None, description="Comma-separated colors to include"),
    include_patterns: Optional[str] = Query(None, description="Comma-separated patterns to include"),
    exclude_patterns: Optional[str] = Query(None, description="Comma-separated patterns to exclude"),
    # Extended filters (PA fields)
    include_formality: Optional[str] = Query(None, description="Comma-separated formality levels"),
    exclude_formality: Optional[str] = Query(None, description="Comma-separated formality levels to exclude"),
    include_seasons: Optional[str] = Query(None, description="Comma-separated seasons"),
    exclude_seasons: Optional[str] = Query(None, description="Comma-separated seasons to exclude"),
    include_style_tags: Optional[str] = Query(None, description="Comma-separated style tags"),
    exclude_style_tags: Optional[str] = Query(None, description="Comma-separated style tags to exclude"),
    include_color_family: Optional[str] = Query(None, description="Comma-separated color families"),
    exclude_color_family: Optional[str] = Query(None, description="Comma-separated color families to exclude"),
    include_silhouette: Optional[str] = Query(None, description="Comma-separated silhouettes"),
    exclude_silhouette: Optional[str] = Query(None, description="Comma-separated silhouettes to exclude"),
    include_fit: Optional[str] = Query(None, description="Comma-separated fits (slim, regular, relaxed, oversized)"),
    exclude_fit: Optional[str] = Query(None, description="Comma-separated fits to exclude"),
    include_length: Optional[str] = Query(None, description="Comma-separated lengths (cropped, standard, long)"),
    exclude_length: Optional[str] = Query(None, description="Comma-separated lengths to exclude"),
    include_sleeves: Optional[str] = Query(None, description="Comma-separated sleeve types"),
    exclude_sleeves: Optional[str] = Query(None, description="Comma-separated sleeve types to exclude"),
    include_neckline: Optional[str] = Query(None, description="Comma-separated necklines"),
    exclude_neckline: Optional[str] = Query(None, description="Comma-separated necklines to exclude"),
    include_rise: Optional[str] = Query(None, description="Comma-separated rises (high, mid, low)"),
    exclude_rise: Optional[str] = Query(None, description="Comma-separated rises to exclude"),
    include_coverage: Optional[str] = Query(None, description="Comma-separated coverage levels"),
    exclude_coverage: Optional[str] = Query(None, description="Comma-separated coverage levels to exclude"),
    include_materials: Optional[str] = Query(None, description="Comma-separated materials"),
    exclude_materials: Optional[str] = Query(None, description="Comma-separated materials to exclude"),
    on_sale_only: Optional[bool] = Query(None, description="Only show items on sale"),
):
    """Get personalized feed using V3 pool-based pipeline."""
    try:
        from recs.v3.models import FeedRequest

        orch = _get_orchestrator_or_503()

        # Effective mode
        effective_mode = mode or "explore"
        if on_sale_only and effective_mode == "explore":
            effective_mode = "sale"

        # Build hard filters from core params
        hard_filters = _build_hard_filters(
            gender=gender,
            categories=categories,
            article_types=article_types,
            exclude_colors=exclude_colors,
            include_colors=include_colors,
            exclude_brands=exclude_brands,
            include_brands=include_brands,
            min_price=min_price,
            max_price=max_price,
            exclude_styles=exclude_styles,
            include_occasions=include_occasions,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )

        # Build soft preferences from extended PA filters
        soft_prefs: Dict[str, Any] = {}
        for key, val in [
            ("include_formality", include_formality),
            ("exclude_formality", exclude_formality),
            ("include_seasons", include_seasons),
            ("exclude_seasons", exclude_seasons),
            ("include_style_tags", include_style_tags),
            ("exclude_style_tags", exclude_style_tags),
            ("include_color_family", include_color_family),
            ("exclude_color_family", exclude_color_family),
            ("include_silhouette", include_silhouette),
            ("exclude_silhouette", exclude_silhouette),
            ("include_fit", include_fit),
            ("exclude_fit", exclude_fit),
            ("include_length", include_length),
            ("exclude_length", exclude_length),
            ("include_sleeves", include_sleeves),
            ("exclude_sleeves", exclude_sleeves),
            ("include_neckline", include_neckline),
            ("exclude_neckline", exclude_neckline),
            ("include_rise", include_rise),
            ("exclude_rise", exclude_rise),
            ("include_coverage", include_coverage),
            ("exclude_coverage", exclude_coverage),
            ("include_materials", include_materials),
            ("exclude_materials", exclude_materials),
            ("exclude_occasions", exclude_occasions),
        ]:
            parsed = _parse_csv(val)
            if parsed:
                soft_prefs[key] = parsed

        request = FeedRequest(
            user_id=user.id,
            session_id=session_id,
            page_size=page_size,
            mode=effective_mode,
            hard_filters=hard_filters,
            soft_preferences=soft_prefs if soft_prefs else None,
            cursor=cursor,
            context=user.user_metadata,
            debug=debug,
        )

        response = orch.get_feed(request)
        return response.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error("V3 feed error: user=%s error=%s", user.id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Feed error: " + str(e))


@router.post(
    "/feed/action",
    response_model=V3ActionResponse,
    summary="Record user interaction",
    description="""
    Record an explicit user interaction with a product.

    **Response is instant** (~1ms) — session scoring updates in-memory,
    Supabase persistence happens in background (non-blocking).

    **Valid Actions:**
    - `click`: User tapped to view product details
    - `save` / `wishlist`: User saved/liked the item
    - `cart` / `add_to_cart`: User added to cart
    - `purchase`: User completed purchase
    - `skip`: User scrolled past without interaction
    - `hide` / `dislike`: User explicitly dismissed
    - `hover`: User swiped through gallery
    """,
)
async def record_v3_action(
    request: V3ActionRequest,
    user: SupabaseUser = Depends(require_auth),
):
    """Record a user interaction (instant response, background persistence)."""
    if request.action not in VALID_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{request.action}'. Must be one of: " + ", ".join(sorted(VALID_ACTIONS)),
        )

    try:
        orch = _get_orchestrator_or_503()

        metadata = {
            "brand": request.brand,
            "article_type": request.item_type,
            "cluster_id": request.cluster_id,
            "source": request.source or "feed",
            "position": request.position,
            "attributes": request.attributes,
        }

        orch.record_action(
            session_id=request.session_id,
            user_id=user.id,
            action=request.action,
            product_id=request.product_id,
            metadata=metadata,
        )

        return V3ActionResponse(status="success")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("V3 action error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Feed error: " + str(e))


@router.post(
    "/feed/search-signal",
    summary="Forward search query to feed session",
    description="""
    Called by the search module after each hybrid search so that the
    feed ranker can boost items matching recent search intent.

    The search route extracts categories/brands/article_types from the
    query and forwards them here.  The next feed request will apply an
    additive score boost (up to +0.18) for matching candidates.
    """,
)
async def record_v3_search_signal(
    request: V3SearchSignalRequest,
    user: SupabaseUser = Depends(require_auth),
):
    """Record a search query as a feed intent signal."""
    try:
        orch = _get_orchestrator_or_503()
        orch.record_search_signal(
            session_id=request.session_id,
            user_id=user.id,
            query=request.query,
            categories=request.categories,
            brands=request.brands,
            article_types=request.article_types,
        )
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("V3 search-signal error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Search signal error: " + str(e))


@router.get(
    "/feed/health",
    summary="V3 feed health check",
    description="Check if the V3 feed pipeline is healthy and ready.",
)
async def v3_health():
    """V3-specific health check."""
    try:
        orch = get_orchestrator()
        store_stats = orch.session_store.get_stats()
        return {
            "status": "healthy",
            "service": "v3-feed",
            "session_store": store_stats,
        }
    except Exception as e:
        return {
            "status": "degraded",
            "service": "v3-feed",
            "error": str(e),
        }


@router.get(
    "/feed/session/{session_id}",
    summary="Get V3 session info",
    description="Get information about a V3 session (shown count, signals, exposure).",
)
async def get_v3_session_info(
    session_id: str,
    user: SupabaseUser = Depends(require_auth),
):
    """Get V3 session info for debugging."""
    try:
        orch = _get_orchestrator_or_503()
        session = orch.session_store.get_or_create_session(session_id, user.id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to load session: " + str(e))

    shown_count = orch.session_store.get_shown_count(session_id)

    # Pool info per mode
    pool_info = {}
    for mode in ("explore", "sale", "new_arrivals"):
        pool = orch.session_store.get_pool(session_id, mode)
        if pool:
            pool_info[mode] = {
                "pool_size": len(pool.ordered_ids),
                "served_count": pool.served_count,
                "remaining": pool.remaining,
                "source_mix": pool.source_mix,
                "last_rerank_action_seq": pool.last_rerank_action_seq,
            }

    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "action_seq": session.action_seq,
        "shown_count": shown_count,
        "clicked_count": len(session.clicked_ids),
        "saved_count": len(session.saved_ids),
        "skipped_count": len(session.skipped_ids),
        "hidden_count": len(session.hidden_ids),
        "negative_brands": list(session.explicit_negative_brands),
        "brand_exposure": dict(session.brand_exposure),
        "category_exposure": dict(session.category_exposure),
        "cluster_exposure": dict(session.cluster_exposure),
        "exploration_budget": session.exploration_budget,
        "intent_strength": session.intent_strength,
        "recent_actions_count": len(session.recent_actions),
        "pools": pool_info,
    }


@router.delete(
    "/feed/session/{session_id}",
    summary="Clear V3 session",
    description="Clear a V3 session to start fresh. Deletes session state, shown set, and all pools.",
)
async def clear_v3_session(
    session_id: str,
    user: SupabaseUser = Depends(require_auth),
):
    """Clear a V3 session."""
    try:
        orch = _get_orchestrator_or_503()
        orch.session_store.delete_session(session_id)
        return {
            "status": "success",
            "session_id": session_id,
            "message": "Session cleared. Next feed request will build a fresh pool.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to clear session: " + str(e))
