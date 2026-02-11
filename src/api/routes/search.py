"""
Search API Routes.

Provides hybrid search (Algolia + FashionCLIP), autocomplete, and analytics.

All search endpoints require JWT authentication.
Analytics click/conversion endpoints also require auth.

NOTE: Routes use `def` (not `async def`) because the underlying services
(Algolia SDK, Supabase client, PyTorch inference) are all synchronous.
FastAPI automatically runs sync route handlers in a thread pool, avoiding
event-loop blocking that would occur with `async def` + sync calls.
"""

import time
import threading
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query

from core.auth import require_auth, SupabaseUser
from core.logging import get_logger
from search.models import (
    HybridSearchRequest,
    HybridSearchResponse,
    AutocompleteResponse,
    SearchClickRequest,
    SearchConversionRequest,
)
from search.hybrid_search import get_hybrid_search_service
from search.autocomplete import get_autocomplete_service
from search.analytics import get_search_analytics
from scoring import ContextResolver, UserContext

logger = get_logger(__name__)

# TTL cache for user profiles (avoid DB hit on every search request)
_PROFILE_CACHE_TTL = 300  # 5 minutes
_profile_cache: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}
_profile_cache_lock = threading.Lock()

# Shared ContextResolver for building UserContext (thread-safe, singleton)
_context_resolver: Optional[ContextResolver] = None
_context_resolver_lock = threading.Lock()

router = APIRouter(prefix="/api/search", tags=["Search"])


# =============================================================================
# Hybrid Search
# =============================================================================

@router.post(
    "/hybrid",
    response_model=HybridSearchResponse,
    summary="Hybrid search (Algolia + FashionCLIP)",
)
def hybrid_search(
    request: HybridSearchRequest,
    user: SupabaseUser = Depends(require_auth),
) -> HybridSearchResponse:
    """
    Search products using hybrid Algolia + FashionCLIP pipeline.

    - **Typo tolerant** (Algolia handles "sweter" -> "sweater")
    - **Synonym aware** (pants = trousers = slacks)
    - **Semantic understanding** for vague queries ("quiet luxury blazer")
    - **All Gemini Vision attributes** available as filters
    - **Session-aware reranking** with profile boosts

    Query intent is auto-classified:
    - exact: Brand/product name -> Algolia dominates
    - specific: Category + attribute -> balanced merge
    - vague: Style/occasion/vibe -> FashionCLIP dominates
    """
    service = get_hybrid_search_service()

    # Load user profile for reranking (optional)
    user_profile = _load_user_profile(user.id)

    # Build UserContext for context-aware scoring (age + weather)
    user_context = _build_user_context(user, user_profile)

    # Load live session scores if session_id provided (for session-aware reranking)
    session_scores = _load_session_scores(request.session_id)

    result = service.search(
        request=request,
        user_id=user.id,
        user_profile=user_profile,
        user_context=user_context,
        session_scores=session_scores,
    )

    # Wire search signals into recommendation session scoring (non-blocking)
    try:
        _forward_search_signal(
            user_id=user.id,
            session_id=request.session_id,
            query=request.query,
            request=request,
        )
    except Exception:
        pass  # Non-fatal — don't let search signal wiring break search

    return result


# =============================================================================
# Autocomplete
# =============================================================================

@router.get(
    "/autocomplete",
    response_model=AutocompleteResponse,
    summary="Search autocomplete (products first, then brands)",
)
def autocomplete(
    q: str = Query(..., min_length=1, max_length=200, description="Search query"),
    limit: int = Query(10, ge=1, le=20, description="Max product suggestions"),
    user: SupabaseUser = Depends(require_auth),
) -> AutocompleteResponse:
    """
    Fast autocomplete powered by Algolia.

    Returns product name suggestions first, then brand suggestions.
    """
    service = get_autocomplete_service()
    return service.autocomplete(query=q, limit=limit)


# =============================================================================
# Analytics Events
# =============================================================================

@router.post(
    "/click",
    summary="Record a search result click",
    status_code=201,
)
def record_click(
    request: SearchClickRequest,
    user: SupabaseUser = Depends(require_auth),
) -> Dict[str, str]:
    """Record when a user clicks a search result."""
    analytics = get_search_analytics()
    analytics.log_click(
        query=request.query,
        product_id=request.product_id,
        position=request.position,
        user_id=user.id,
    )
    return {"status": "ok"}


@router.post(
    "/conversion",
    summary="Record a search conversion",
    status_code=201,
)
def record_conversion(
    request: SearchConversionRequest,
    user: SupabaseUser = Depends(require_auth),
) -> Dict[str, str]:
    """Record when a user converts (add to cart / purchase) from search."""
    analytics = get_search_analytics()
    analytics.log_conversion(
        query=request.query,
        product_id=request.product_id,
        user_id=user.id,
    )
    return {"status": "ok"}


# =============================================================================
# Health / Info
# =============================================================================

@router.get(
    "/health",
    summary="Search service health check",
)
def search_health() -> Dict[str, Any]:
    """Check Algolia and semantic engine connectivity."""
    from search.algolia_client import get_algolia_client

    status: Dict[str, Any] = {"service": "search", "algolia": "unknown", "semantic": "unknown"}

    # Check Algolia
    try:
        client = get_algolia_client()
        resp = client.search(query="", hits_per_page=1)
        status["algolia"] = "healthy"
        status["index_records"] = resp.get("nbHits", 0)
    except Exception as e:
        status["algolia"] = "unhealthy"
        status["algolia_error"] = str(e)

    # Check semantic engine (FashionCLIP / pgvector)
    try:
        from women_search_engine import get_women_search_engine
        engine = get_women_search_engine()
        # Check if model can be loaded (lazy init)
        if engine.supabase is not None:
            status["semantic"] = "healthy"
        else:
            status["semantic"] = "degraded"
            status["semantic_error"] = "Supabase client not initialized"
    except Exception as e:
        status["semantic"] = "unhealthy"
        status["semantic_error"] = str(e)

    # Overall status
    if status["algolia"] == "healthy" and status["semantic"] == "healthy":
        status["status"] = "healthy"
    elif status["algolia"] == "unhealthy" and status["semantic"] == "unhealthy":
        status["status"] = "unhealthy"
    else:
        status["status"] = "degraded"

    return status


# =============================================================================
# Helpers
# =============================================================================

def _load_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Load user's onboarding profile for reranking (best-effort, TTL cached).

    Caches profiles for _PROFILE_CACHE_TTL seconds to avoid hitting the
    database on every search request for the same user.
    """
    now = time.time()

    # Check cache first
    cached = _profile_cache.get(user_id)
    if cached:
        cached_at, profile = cached
        if now - cached_at < _PROFILE_CACHE_TTL:
            return profile

    # Cache miss or expired - load from DB
    try:
        from women_search_engine import get_women_search_engine
        engine = get_women_search_engine()
        profile = engine.load_user_profile(user_id=user_id)
    except Exception as e:
        logger.warning("Failed to load user profile for reranking", user_id=user_id, error=str(e))
        profile = None

    # Store in cache (thread-safe)
    with _profile_cache_lock:
        _profile_cache[user_id] = (now, profile)

    return profile


def _get_context_resolver() -> ContextResolver:
    """Get or create the shared ContextResolver (thread-safe singleton)."""
    global _context_resolver
    if _context_resolver is None:
        with _context_resolver_lock:
            if _context_resolver is None:
                from config.settings import get_settings
                settings = get_settings()
                _context_resolver = ContextResolver(
                    weather_api_key=settings.openweather_api_key,
                )
    return _context_resolver


def _build_user_context(
    user: SupabaseUser,
    user_profile: Optional[Dict[str, Any]],
) -> Optional[UserContext]:
    """Build UserContext from JWT metadata + cached onboarding profile.

    Returns None on any failure (context scoring will be skipped).
    """
    try:
        resolver = _get_context_resolver()

        # Extract birthdate from the cached user_profile
        birthdate = None
        profile_dict = None
        if user_profile:
            # user_profile structure is loaded by women_search_engine.load_user_profile
            # which returns the full onboarding profile dict
            birthdate = user_profile.get("birthdate")
            # Build profile dict with coverage flags for the resolver
            profile_dict = {
                "no_sleeveless": user_profile.get("no_sleeveless", False),
                "no_tanks": user_profile.get("no_tanks", False),
                "no_crop": user_profile.get("no_crop", False),
                "no_athletic": user_profile.get("no_athletic", False),
                "no_revealing": user_profile.get("no_revealing", False),
                "styles_to_avoid": user_profile.get("styles_to_avoid") or user_profile.get("hard_filters", {}).get("exclude_styles") or [],
                "modesty": user_profile.get("modesty"),
            }

        ctx = resolver.resolve(
            user_id=user.id,
            jwt_user_metadata=user.user_metadata,
            birthdate=birthdate,
            onboarding_profile=profile_dict,
        )
        # Only return if we have something useful
        if ctx.age_group or ctx.weather:
            return ctx
        return None
    except Exception as e:
        logger.debug("Failed to build UserContext for search", user_id=user.id, error=str(e))
        return None


def _load_session_scores(session_id: Optional[str]):
    """Load live session scores from the recommendation pipeline.

    Returns None if no session_id, no pipeline, or no scores exist.
    Used to pass session-learned preferences to the search reranker.
    """
    if not session_id:
        return None
    try:
        from recs.api_endpoints import get_pipeline
        pipeline = get_pipeline()
        return pipeline.get_session_scores(session_id)
    except Exception:
        return None  # Session scores are optional for search


def _forward_search_signal(
    user_id: str,
    session_id: Optional[str],
    query: str,
    request: HybridSearchRequest,
) -> None:
    """Forward search query/filters to the recommendation pipeline's session scoring.

    This wires search intent signals into the feed pipeline so the
    recommendation feed can adapt to what the user is searching for.
    """
    if not session_id:
        return  # No session to update

    try:
        from recs.api_endpoints import get_pipeline
        pipeline = get_pipeline()

        filters: Dict[str, Any] = {}
        if request.categories:
            filters["categories"] = request.categories
        if request.brands:
            filters["brands"] = request.brands
        if request.colors:
            filters["colors"] = request.colors
        if request.occasions:
            filters["occasions"] = request.occasions
        if request.patterns:
            filters["patterns"] = request.patterns
        if request.style_tags:
            filters["style_tags"] = request.style_tags
        if request.min_price is not None:
            filters["min_price"] = request.min_price
        if request.max_price is not None:
            filters["max_price"] = request.max_price

        pipeline.update_session_scores_from_search(
            session_id=session_id,
            query=query,
            filters=filters,
        )
    except Exception as e:
        logger.debug("Failed to forward search signal", error=str(e))
