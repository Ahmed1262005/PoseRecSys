"""
Unified Gender-Aware Style Learning Routes.

Endpoints that work for both men's and women's fashion,
with gender specified in the request.

All endpoints require JWT authentication.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.auth import require_auth, SupabaseUser
from core.utils import convert_numpy
from engines import (
    get_engine,
    get_image_url,
    normalize_gender,
    PredictivePreferences,
)
from services.session_manager import get_unified_session_manager


router = APIRouter(prefix="/api/unified", tags=["Unified"])


# =============================================================================
# Session State (using SessionManager)
# =============================================================================

_session_manager = None


def get_session_manager():
    """Get the unified session manager singleton."""
    global _session_manager
    if _session_manager is None:
        _session_manager = get_unified_session_manager()
    return _session_manager


def _get_session_key(user_id: str, gender: str) -> str:
    """Generate session key combining user_id and gender."""
    return f"{user_id}:{normalize_gender(gender)}"


# =============================================================================
# Request/Response Models
# =============================================================================

class UnifiedStartRequest(BaseModel):
    """Request to start a unified style learning session."""
    gender: str = Field(..., description="Gender: 'female'/'women' or 'male'/'men'")
    colors_to_avoid: Optional[List[str]] = Field(default=None)
    materials_to_avoid: Optional[List[str]] = Field(default=None)
    selected_categories: Optional[List[str]] = Field(default=None)


class UnifiedChoiceRequest(BaseModel):
    """Request to record a choice in unified session."""
    gender: str = Field(..., description="Gender for session lookup")
    winner_id: str = Field(..., description="ID of the chosen item")


class UnifiedSkipRequest(BaseModel):
    """Request to skip items in unified session."""
    gender: str = Field(..., description="Gender for session lookup")


# =============================================================================
# Helper Functions
# =============================================================================

def format_item_response(item_id: str, item_info: Dict[str, Any], gender: str) -> Dict[str, Any]:
    """Format an item for API response."""
    return {
        "id": item_id,
        "image_url": get_image_url(item_id, gender),
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

@router.post("/four/start", summary="Start a unified style learning session")
async def start_session(
    request: UnifiedStartRequest,
    user: SupabaseUser = Depends(require_auth)
) -> Dict[str, Any]:
    """
    Start a new gender-aware four-choice session.
    
    The gender determines which product catalog and engine configuration to use.
    """
    user_id = user.id
    gender = request.gender
    gender_key = normalize_gender(gender)
    session_key = _get_session_key(user_id, gender)
    sessions = get_session_manager()
    
    # Get gender-specific engine
    engine = get_engine(gender)
    
    # Create preferences
    prefs = PredictivePreferences(
        user_id=user_id,
        gender=gender_key,
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
            detail=f"Not enough {gender_key} items available for style learning"
        )
    
    # Store session state
    sessions.set_preferences(session_key, prefs)
    sessions.set_current_items(session_key, four_items)
    sessions.set_test_info(session_key, test_info)
    
    # Format response
    items = [
        format_item_response(item_id, engine.get_item_info(item_id), gender_key)
        for item_id in four_items
    ]
    
    return convert_numpy({
        "status": "success",
        "gender": gender_key,
        "items": items,
        "test_info": format_test_info(test_info),
        "stats": engine.get_preference_summary(prefs),
    })


@router.post("/four/choose", summary="Record a choice")
async def record_choice(
    request: UnifiedChoiceRequest,
    user: SupabaseUser = Depends(require_auth)
) -> Dict[str, Any]:
    """
    Record the user's choice and get next items.
    """
    user_id = user.id
    gender = request.gender
    gender_key = normalize_gender(gender)
    session_key = _get_session_key(user_id, gender)
    sessions = get_session_manager()
    
    # Get session state
    prefs = sessions.get_preferences(session_key)
    current_items = sessions.get_current_items(session_key)
    test_info = sessions.get_test_info(session_key)
    
    if prefs is None:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call /four/start first."
        )
    
    if current_items is None or request.winner_id not in current_items:
        raise HTTPException(
            status_code=400,
            detail="Invalid item selection"
        )
    
    engine = get_engine(gender)
    
    # Record the choice
    prefs = engine.record_choice(prefs, request.winner_id, current_items, test_info)
    
    # Check if session complete
    summary = engine.get_preference_summary(prefs)
    
    if summary.get("session_complete"):
        sessions.delete_session(session_key)
        return convert_numpy({
            "status": "complete",
            "gender": gender_key,
            "message": "Style learning complete!",
            "stats": summary,
        })
    
    # Get next 4 items
    four_items, test_info = engine.get_four_items(prefs)
    
    if len(four_items) < 4:
        sessions.delete_session(session_key)
        return convert_numpy({
            "status": "complete",
            "gender": gender_key,
            "message": "Style learning complete!",
            "stats": summary,
        })
    
    # Update session state
    sessions.set_preferences(session_key, prefs)
    sessions.set_current_items(session_key, four_items)
    sessions.set_test_info(session_key, test_info)
    
    # Format response
    items = [
        format_item_response(item_id, engine.get_item_info(item_id), gender_key)
        for item_id in four_items
    ]
    
    return convert_numpy({
        "status": "success",
        "gender": gender_key,
        "items": items,
        "test_info": format_test_info(test_info),
        "stats": summary,
    })


@router.post("/four/skip", summary="Skip current items")
async def skip_items(
    request: UnifiedSkipRequest,
    user: SupabaseUser = Depends(require_auth)
) -> Dict[str, Any]:
    """
    Skip all 4 current items and get new ones.
    """
    user_id = user.id
    gender = request.gender
    gender_key = normalize_gender(gender)
    session_key = _get_session_key(user_id, gender)
    sessions = get_session_manager()
    
    # Get session state
    prefs = sessions.get_preferences(session_key)
    current_items = sessions.get_current_items(session_key)
    test_info = sessions.get_test_info(session_key)
    
    if prefs is None:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call /four/start first."
        )
    
    engine = get_engine(gender)
    
    # Record the skip
    prefs = engine.record_skip(prefs, current_items, test_info)
    
    # Get next 4 items
    four_items, test_info = engine.get_four_items(prefs)
    summary = engine.get_preference_summary(prefs)
    
    if len(four_items) < 4 or summary.get("session_complete"):
        sessions.delete_session(session_key)
        return convert_numpy({
            "status": "complete",
            "gender": gender_key,
            "message": "Style learning complete!",
            "stats": summary,
        })
    
    # Update session state
    sessions.set_preferences(session_key, prefs)
    sessions.set_current_items(session_key, four_items)
    sessions.set_test_info(session_key, test_info)
    
    # Format response
    items = [
        format_item_response(item_id, engine.get_item_info(item_id), gender_key)
        for item_id in four_items
    ]
    
    return convert_numpy({
        "status": "success",
        "gender": gender_key,
        "items": items,
        "test_info": format_test_info(test_info),
        "stats": summary,
    })


@router.get("/four/summary/{gender}", summary="Get session summary")
async def get_summary(
    gender: str,
    user: SupabaseUser = Depends(require_auth)
) -> Dict[str, Any]:
    """
    Get the current session summary and learned preferences.
    """
    user_id = user.id
    gender_key = normalize_gender(gender)
    session_key = _get_session_key(user_id, gender)
    sessions = get_session_manager()
    
    prefs = sessions.get_preferences(session_key)
    
    if prefs is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active {gender_key} session found"
        )
    
    engine = get_engine(gender)
    summary = engine.get_preference_summary(prefs)
    
    return convert_numpy({
        "status": "success",
        "user_id": user_id,
        "gender": gender_key,
        "summary": summary,
    })
