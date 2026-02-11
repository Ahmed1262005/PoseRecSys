"""
Women's Fashion Style Learning Routes.

Endpoints for the Tinder-style 4-choice preference learning system
for women's fashion.

All endpoints require JWT authentication.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.auth import require_auth, SupabaseUser
from core.utils import convert_numpy
from engines import (
    get_women_engine,
    get_image_url,
    get_search_engine,
    PredictivePreferences,
)
from services.session_manager import get_women_session_manager


router = APIRouter(prefix="/api/women", tags=["Women's Fashion"])


# =============================================================================
# Session State (using SessionManager)
# =============================================================================

_session_manager = None


def get_session_manager():
    """Get the women's session manager singleton."""
    global _session_manager
    if _session_manager is None:
        _session_manager = get_women_session_manager()
    return _session_manager


# =============================================================================
# Request/Response Models
# =============================================================================

class SessionStartRequest(BaseModel):
    """Request to start a style learning session."""
    colors_to_avoid: Optional[List[str]] = Field(
        default=None,
        description="Colors to exclude from recommendations"
    )
    materials_to_avoid: Optional[List[str]] = Field(
        default=None,
        description="Materials to exclude from recommendations"
    )
    selected_categories: Optional[List[str]] = Field(
        default=None,
        description="Categories to focus on (e.g., ['tops_woven', 'dresses'])"
    )


class ChoiceRequest(BaseModel):
    """Request to record a choice."""
    winner_id: str = Field(..., description="ID of the chosen item")


class SearchRequest(BaseModel):
    """Request for text search."""
    query: str = Field(..., description="Search query text")
    categories: Optional[List[str]] = Field(default=None)
    colors_to_avoid: Optional[List[str]] = Field(default=None)
    materials_to_avoid: Optional[List[str]] = Field(default=None)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


# =============================================================================
# Helper Functions
# =============================================================================

def format_item_response(item_id: str, item_info: Dict[str, Any]) -> Dict[str, Any]:
    """Format an item for API response."""
    return {
        "id": item_id,
        "image_url": get_image_url(item_id, "female"),
        "category": item_info.get("category", ""),
        "brand": item_info.get("brand", ""),
        "color": item_info.get("color", ""),
        "cluster": item_info.get("cluster", 0),
    }


def format_test_info(test_info: Dict[str, Any]) -> Dict[str, Any]:
    """Format test info for API response."""
    return {
        "category": test_info.get("category", ""),
        "category_index": test_info.get("category_index", 0),
        "total_categories": test_info.get("total_categories", 0),
        "round_in_category": test_info.get("round_in_category", 0),
        "clusters_shown": test_info.get("clusters_shown", []),
        "prediction": test_info.get("prediction"),
        "categories_completed": test_info.get("categories_completed", []),
    }


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/session/start", summary="Start a style learning session")
async def start_session(
    request: SessionStartRequest,
    user: SupabaseUser = Depends(require_auth)
) -> Dict[str, Any]:
    """
    Start a new style learning session.
    
    Returns 4 items to choose from along with test metadata.
    """
    user_id = user.id
    sessions = get_session_manager()
    
    try:
        engine = get_women_engine()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Women's fashion engine not available: {str(e)}"
        )
    
    # Create preferences
    prefs = PredictivePreferences(
        user_id=user_id,
        gender="female",
        colors_to_avoid=set(c.lower().strip() for c in (request.colors_to_avoid or []) if c),
        materials_to_avoid=set(m.lower().strip() for m in (request.materials_to_avoid or []) if m),
    )
    
    # Initialize with selected categories
    prefs = engine.initialize_session(prefs, request.selected_categories)
    
    # Get first 4 items
    four_items, test_info = engine.get_four_items(prefs)
    
    if len(four_items) < 4:
        raise HTTPException(
            status_code=503,
            detail="Not enough items available for style learning"
        )
    
    # Store session state
    sessions.set_preferences(user_id, prefs)
    sessions.set_current_items(user_id, four_items)
    sessions.set_test_info(user_id, test_info)
    
    # Format response
    items = [
        format_item_response(item_id, engine.get_item_info(item_id))
        for item_id in four_items
    ]
    
    return convert_numpy({
        "status": "success",
        "items": items,
        "test_info": format_test_info(test_info),
        "stats": engine.get_preference_summary(prefs),
    })


@router.post("/session/choose", summary="Record a choice")
async def record_choice(
    request: ChoiceRequest,
    user: SupabaseUser = Depends(require_auth)
) -> Dict[str, Any]:
    """
    Record the user's choice and get next items.
    
    Call this when user selects one of the 4 items.
    """
    user_id = user.id
    sessions = get_session_manager()
    
    # Get session state
    prefs = sessions.get_preferences(user_id)
    current_items = sessions.get_current_items(user_id)
    test_info = sessions.get_test_info(user_id)
    
    if prefs is None:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call /session/start first."
        )
    
    if current_items is None or request.winner_id not in current_items:
        raise HTTPException(
            status_code=400,
            detail="Invalid item selection"
        )
    
    engine = get_women_engine()
    
    # Record the choice
    prefs = engine.record_choice(prefs, request.winner_id, current_items, test_info)
    
    # Check if session complete
    summary = engine.get_preference_summary(prefs)
    
    if summary.get("session_complete"):
        # Clean up session
        sessions.delete_session(user_id)
        
        return convert_numpy({
            "status": "complete",
            "message": "Style learning complete!",
            "stats": summary,
        })
    
    # Get next 4 items
    four_items, test_info = engine.get_four_items(prefs)
    
    if len(four_items) < 4:
        sessions.delete_session(user_id)
        return convert_numpy({
            "status": "complete",
            "message": "Style learning complete!",
            "stats": summary,
        })
    
    # Update session state
    sessions.set_preferences(user_id, prefs)
    sessions.set_current_items(user_id, four_items)
    sessions.set_test_info(user_id, test_info)
    
    # Format response
    items = [
        format_item_response(item_id, engine.get_item_info(item_id))
        for item_id in four_items
    ]
    
    return convert_numpy({
        "status": "success",
        "items": items,
        "test_info": format_test_info(test_info),
        "stats": summary,
    })


@router.post("/session/skip", summary="Skip current items")
async def skip_items(
    user: SupabaseUser = Depends(require_auth)
) -> Dict[str, Any]:
    """
    Skip all 4 current items and get new ones.
    
    Call this when user doesn't like any of the 4 items.
    """
    user_id = user.id
    sessions = get_session_manager()
    
    # Get session state
    prefs = sessions.get_preferences(user_id)
    current_items = sessions.get_current_items(user_id)
    test_info = sessions.get_test_info(user_id)
    
    if prefs is None:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call /session/start first."
        )
    
    engine = get_women_engine()
    
    # Record the skip
    prefs = engine.record_skip(prefs, current_items, test_info)
    
    # Get next 4 items
    four_items, test_info = engine.get_four_items(prefs)
    summary = engine.get_preference_summary(prefs)
    
    if len(four_items) < 4 or summary.get("session_complete"):
        sessions.delete_session(user_id)
        return convert_numpy({
            "status": "complete",
            "message": "Style learning complete!",
            "stats": summary,
        })
    
    # Update session state
    sessions.set_preferences(user_id, prefs)
    sessions.set_current_items(user_id, four_items)
    sessions.set_test_info(user_id, test_info)
    
    # Format response
    items = [
        format_item_response(item_id, engine.get_item_info(item_id))
        for item_id in four_items
    ]
    
    return convert_numpy({
        "status": "success",
        "items": items,
        "test_info": format_test_info(test_info),
        "stats": summary,
    })


@router.get("/session/summary", summary="Get session summary")
async def get_summary(
    user: SupabaseUser = Depends(require_auth)
) -> Dict[str, Any]:
    """
    Get the current session summary and learned preferences.
    """
    user_id = user.id
    sessions = get_session_manager()
    
    prefs = sessions.get_preferences(user_id)
    
    if prefs is None:
        raise HTTPException(
            status_code=404,
            detail="No active session found"
        )
    
    engine = get_women_engine()
    summary = engine.get_preference_summary(prefs)
    
    return convert_numpy({
        "status": "success",
        "user_id": user_id,
        "summary": summary,
    })


@router.get("/feed", summary="Get personalized feed")
async def get_feed(
    user: SupabaseUser = Depends(require_auth),
    page: int = 1,
    page_size: int = 20,
) -> Dict[str, Any]:
    """
    Get personalized fashion recommendations based on learned preferences.
    
    Uses the recommendation service to fetch items from Supabase pgvector
    based on the user's taste vector from style learning.
    """
    from recs.recommendation_service import RecommendationService
    
    user_id = user.id
    
    try:
        service = RecommendationService()
        results = service.get_feed(
            user_id=user_id,
            gender="female",
            page=page,
            limit=page_size,
        )
        
        return convert_numpy({
            "status": "success",
            "user_id": user_id,
            "items": results.get("items", []),
            "strategy": results.get("strategy", "unknown"),
            "pagination": {
                "page": page,
                "page_size": page_size,
                "has_more": len(results.get("items", [])) >= page_size,
            }
        })
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get feed: {str(e)}"
        )


# =============================================================================
# Complete the Fit
# =============================================================================

class CompleteFitRequest(BaseModel):
    """Request for complete-the-fit recommendations."""
    product_id: str = Field(..., description="UUID of the source product")
    items_per_category: int = Field(
        default=4, ge=1, le=20,
        description="Items per category in carousel mode"
    )
    category: Optional[str] = Field(
        default=None,
        description="Target category for feed mode (e.g. 'tops', 'outerwear'). "
                    "Omit for carousel mode (all complementary categories)."
    )
    offset: int = Field(default=0, ge=0, description="Skip first N items (feed mode)")
    limit: Optional[int] = Field(
        default=None, ge=1, le=100,
        description="Max items to return (feed mode, overrides items_per_category)"
    )


@router.post(
    "/complete-fit",
    summary="Complete the Look - Complementary items",
    description="""
    Returns complementary items to complete an outfit with the given product.

    **Carousel mode** (default): Omit `category`. Returns top N items from each
    complementary category (tops→bottoms+outerwear, bottoms→tops+outerwear, etc.).

    **Feed mode**: Set `category` to a specific category. Returns paginated items
    for that category only (use `offset`/`limit` for infinite scroll).
    """
)
async def complete_fit(
    request: CompleteFitRequest,
    user: SupabaseUser = Depends(require_auth),
) -> Dict[str, Any]:
    """Find complementary items using FashionCLIP semantic search."""
    try:
        search_engine = get_search_engine()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Search engine not available: {str(e)}"
        )

    result = search_engine.complete_the_fit(
        product_id=request.product_id,
        items_per_category=request.items_per_category,
        target_category=request.category,
        offset=request.offset,
        limit=request.limit,
    )

    if result.get("error") and result.get("source_product") is None:
        if "not found" in result["error"].lower():
            raise HTTPException(status_code=404, detail=result["error"])
        raise HTTPException(status_code=500, detail=result["error"])

    return convert_numpy(result)


# =============================================================================
# Search
# =============================================================================

@router.post("/search", summary="Search women's fashion")
async def search(
    request: SearchRequest,
    user: SupabaseUser = Depends(require_auth)
) -> Dict[str, Any]:
    """
    Text search for women's fashion using FashionCLIP + pgvector.
    
    Supports natural language queries like:
    - "flowy blue dress"
    - "black blazer"
    - "something elegant for a wedding"
    """
    try:
        search_engine = get_search_engine()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Search engine not available: {str(e)}"
        )
    
    # Execute search
    results = search_engine.search(
        query=request.query,
        categories=request.categories,
        colors_to_avoid=request.colors_to_avoid,
        materials_to_avoid=request.materials_to_avoid,
        page=request.page,
        page_size=request.page_size,
    )
    
    return convert_numpy({
        "status": "success",
        "query": request.query,
        "results": results.get("results", []),
        "total": results.get("total", 0),
        "pagination": {
            "page": request.page,
            "page_size": request.page_size,
            "has_more": results.get("has_more", False),
        }
    })
