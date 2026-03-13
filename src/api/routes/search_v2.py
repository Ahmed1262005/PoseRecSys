"""
Search V2 API Routes.

Upgraded search pipeline with:
- Groq Llama 4 Scout planner (faster than OpenAI for non-reasoning queries)
- Heuristic planner bypass for simple queries (brand-only, bare category, category+color)
- Lane-structured semantic queries (core, style_variant, silhouette_variant, edge)

V1 (/api/search/hybrid) remains untouched. V2 lives at /api/search/v2/hybrid.

NOTE: Routes use `def` (not `async def`) because the underlying services
are synchronous. FastAPI runs sync handlers in a thread pool.
"""

import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from core.auth import require_auth, SupabaseUser
from core.logging import get_logger
from core.sanitization import strip_tags
from search.models import HybridSearchRequest, HybridSearchResponse
from search.hybrid_search import get_hybrid_search_service
from search.analytics import get_search_analytics

# Reuse v1 helpers — profile loading, context building, signal forwarding
from api.routes.search import (
    _load_user_profile,
    _build_user_context,
    _build_planner_context,
    _load_session_scores,
    _forward_search_signal,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/search/v2", tags=["Search V2"])


# =============================================================================
# V2 Hybrid Search — with heuristic bypass + Groq planner
# =============================================================================

@router.post(
    "/hybrid",
    response_model=HybridSearchResponse,
    summary="V2 Hybrid search with heuristic bypass + Groq planner",
)
def hybrid_search_v2(
    request: HybridSearchRequest,
    background_tasks: BackgroundTasks,
    user: SupabaseUser = Depends(require_auth),
) -> HybridSearchResponse:
    """
    V2 search pipeline. Same semantics as /api/search/hybrid but with:

    1. **Heuristic bypass** — simple queries (pure brand, bare category,
       category+color) skip the LLM entirely. ~0ms planner latency.
    2. **Groq planner** — when LLM is needed, uses Groq Llama 4 Scout
       (~200-600ms) instead of gpt-4.1-mini (~1.5-4s).
    3. **Lane-structured semantic queries** — diversity by design, not by
       accident. Each semantic query has an explicit role (core, style
       variant, silhouette variant, edge).

    Falls back to the full LLM planner if heuristic bypass doesn't match.
    """
    service = get_hybrid_search_service()

    # Load user profile for reranking (optional)
    user_profile = _load_user_profile(user.id)
    user_context = _build_user_context(user, user_profile)
    planner_context = (
        getattr(request, "planner_context", None)
        or _build_planner_context(user_profile, user_context)
    )
    session_scores = _load_session_scores(request.session_id)

    # Extract follow-up refinement fields
    selected_filters = getattr(request, "selected_filters", None)
    selection_labels = getattr(request, "selection_labels", None)

    # ------------------------------------------------------------------
    # V2 addition: try heuristic bypass BEFORE calling the LLM planner.
    # If it returns a SearchPlan, pass it as pre_plan to skip the LLM.
    # ------------------------------------------------------------------
    pre_plan = None
    planner_source = "llm"
    t0 = time.perf_counter()

    # Only attempt heuristic on fresh page-1 searches (no cursor, no
    # search_session_id, no follow-up refinement).
    is_fresh_search = (
        not getattr(request, "search_session_id", None)
        and not getattr(request, "cursor", None)
        and not selected_filters
    )

    if is_fresh_search:
        from search.query_planner import get_query_planner
        planner = get_query_planner()
        heuristic_plan = planner.try_heuristic_plan(request.query)
        if heuristic_plan is not None:
            pre_plan = heuristic_plan
            planner_source = "heuristic"
            logger.info(
                "V2 heuristic bypass hit",
                query=request.query,
                intent=heuristic_plan.intent,
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            )

    result = service.search(
        request=request,
        user_id=user.id,
        user_profile=user_profile,
        user_context=user_context,
        session_scores=session_scores,
        planner_context=planner_context,
        selected_filters=selected_filters,
        selection_labels=selection_labels,
        pre_plan=pre_plan,
    )

    # Tag the response with v2 metadata via timing dict
    result.timing["search_version"] = "v2"
    result.timing["planner_source"] = planner_source

    # Build v2_meta for full observability
    from config.settings import get_settings
    v2_settings = get_settings()
    faiss_info = {"backend": "pgvector", "vectors_loaded": 0}
    if v2_settings.use_local_faiss:
        try:
            from search.local_vector_store import get_local_vector_store
            store = get_local_vector_store()
            if store.ready:
                faiss_info = {"backend": "faiss", "vectors_loaded": store.count}
        except Exception:
            pass

    from search.query_planner import get_query_planner
    v2_planner = get_query_planner()

    # Detect plan cache hit: planner_ms < 10ms on a non-heuristic LLM call
    plan_cached = (
        planner_source == "llm"
        and result.timing.get("planner_ms", 999) < 10
    )

    result.v2_meta = {
        "search_version": "v2",
        "planner_source": planner_source,
        "planner_provider": v2_planner._provider,
        "planner_model": v2_planner._model,
        "plan_cached": plan_cached,
        "semantic_backend": faiss_info["backend"],
        "faiss_vectors_loaded": faiss_info["vectors_loaded"],
        "cache_status": result.timing.get("cache_status", "unknown"),
        "heuristic_plan": {
            "intent": pre_plan.intent,
            "algolia_query": strip_tags(pre_plan.algolia_query),
            "semantic_queries": [strip_tags(q) for q in (pre_plan.semantic_queries or [])],
            "brand": strip_tags(pre_plan.brand),
            "attributes": pre_plan.attributes,
            "confidence": pre_plan.confidence,
        } if pre_plan else None,
    }

    # Wire search signals into recommendation session scoring (non-blocking)
    try:
        _forward_search_signal(
            user_id=user.id,
            session_id=request.session_id,
            query=request.query,
            request=request,
        )
    except Exception:
        pass

    # Log impressions in background
    product_ids = [r.product_id for r in result.results]
    if product_ids:
        analytics = get_search_analytics()
        background_tasks.add_task(
            analytics.log_impressions,
            query=request.query,
            product_ids=product_ids,
            page=request.page,
            user_id=user.id,
        )

    return result


# =============================================================================
# V2 Health
# =============================================================================

@router.get("/health", summary="V2 search health check")
def search_v2_health():
    """Health check for V2 search pipeline."""
    from search.query_planner import get_query_planner
    planner = get_query_planner()
    return {
        "status": "ok",
        "version": "v2",
        "planner_provider": planner.provider,
        "planner_model": planner._model,
        "planner_enabled": planner.enabled,
        "heuristic_bypass": planner._heuristic_bypass,
    }
