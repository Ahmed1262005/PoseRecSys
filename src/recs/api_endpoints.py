#!/usr/bin/env python3
"""
Recommendation API Endpoints - Supabase Integration

These endpoints connect the Tinder-style preference test to Supabase
for persistent storage and pgvector-based recommendations.

Endpoints (Legacy):
- POST /api/recs/save-preferences      - Save Tinder test results to Supabase
- GET  /api/recs/feed/{user_id}        - Get personalized feed from Supabase
- GET  /api/recs/similar/{product_id}  - Get similar products
- GET  /api/recs/trending              - Get trending products
- GET  /api/recs/categories            - Get product categories

New Pipeline Endpoints:
- POST /api/recs/v2/onboarding         - Save 9-module onboarding profile
- GET  /api/recs/v2/feed               - Get feed using full pipeline (SASRec + filters)
- GET  /api/recs/v2/info               - Get pipeline configuration info
"""

import os
import sys
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.recs.recommendation_service import RecommendationService
from src.recs.models import (
    OnboardingProfile,
    TopsPrefs,
    BottomsPrefs,
    DressesPrefs,
    OuterwearPrefs,
    SkirtsPrefs,
    OnePiecePrefs,
    PriceRange,
    FeedRequest,
    FeedResponse as PipelineFeedResponse,
    FeedItem,
    SaveOnboardingRequest,
    SaveOnboardingResponse,
    StyleDiscoveryModule,
    StyleDiscoverySelection,
    StyleDiscoverySummary,
)
from src.recs.pipeline import RecommendationPipeline

# Create router
router = APIRouter(prefix="/api/recs", tags=["Recommendations"])

# Initialize service (lazy loading)
_service: Optional[RecommendationService] = None
_pipeline: Optional[RecommendationPipeline] = None


def get_service() -> RecommendationService:
    """Get or create recommendation service."""
    global _service
    if _service is None:
        _service = RecommendationService()
    return _service


def get_pipeline() -> RecommendationPipeline:
    """Get or create recommendation pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RecommendationPipeline(load_sasrec=True)
    return _pipeline


# =============================================================================
# Request/Response Models
# =============================================================================

class SavePreferencesRequest(BaseModel):
    """Request to save Tinder test preferences."""
    user_id: Optional[str] = Field(None, description="UUID user ID (for logged-in users)")
    anon_id: Optional[str] = Field(None, description="Anonymous user ID")
    gender: str = Field("female", description="Gender: 'female' or 'male'")
    rounds_completed: int = Field(0, description="Number of Tinder rounds completed")
    categories_tested: List[str] = Field(default_factory=list, description="Categories tested")
    attribute_preferences: Dict[str, Any] = Field(default_factory=dict, description="Learned attribute preferences")
    prediction_accuracy: Optional[float] = Field(None, description="Prediction accuracy score")
    taste_vector: Optional[List[float]] = Field(None, description="512-dim taste vector from CLIP")


class SavePreferencesResponse(BaseModel):
    """Response from saving preferences."""
    status: str
    preference_id: Optional[str] = None
    user_id: str
    seed_source: str


class ProductItem(BaseModel):
    """A product item in the feed."""
    product_id: str
    name: str
    brand: Optional[str] = None
    category: str
    gender: List[str]
    price: float
    image_url: str
    similarity: Optional[float] = None
    trending_score: Optional[float] = None
    reason: Optional[str] = None


class FeedResponse(BaseModel):
    """Response from feed endpoint."""
    user_id: Optional[str]
    strategy: str
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class CategoryCount(BaseModel):
    """Category with product count."""
    category: str
    product_count: int


# =============================================================================
# User Interaction Tracking Models
# =============================================================================

# Valid actions for tracking (explicit engagement only)
VALID_INTERACTION_ACTIONS = {'click', 'hover', 'add_to_wishlist', 'add_to_cart', 'purchase'}


class RecordActionRequest(BaseModel):
    """Request to record a user interaction."""
    anon_id: Optional[str] = Field(None, description="Anonymous user ID")
    user_id: Optional[str] = Field(None, description="UUID user ID")
    session_id: str = Field(..., description="Session ID from feed response")
    product_id: str = Field(..., description="Product UUID that was interacted with")
    action: str = Field(..., description="Action type: click, hover, add_to_wishlist, add_to_cart, purchase")
    source: Optional[str] = Field("feed", description="Source: feed, search, similar, style-this")
    position: Optional[int] = Field(None, description="Position in feed when interacted")


class RecordActionResponse(BaseModel):
    """Response from recording an action."""
    status: str
    interaction_id: Optional[str] = None


class SyncSessionRequest(BaseModel):
    """Request to sync session seen_ids for training data."""
    anon_id: Optional[str] = Field(None, description="Anonymous user ID")
    user_id: Optional[str] = Field(None, description="UUID user ID")
    session_id: str = Field(..., description="Session ID from feed response")
    seen_ids: List[str] = Field(..., description="List of product UUIDs shown in this session")


class SyncSessionResponse(BaseModel):
    """Response from syncing session data."""
    status: str
    synced_count: int


# =============================================================================
# API Endpoints
# =============================================================================

@router.post(
    "/save-preferences",
    response_model=SavePreferencesResponse,
    summary="Save Tinder test preferences to Supabase",
    description="""
    Save user preferences from the Tinder-style test to Supabase.

    This should be called after the user completes the style learning session.
    The taste_vector (512-dim) enables personalized recommendations via pgvector.

    **Required**: Either `user_id` or `anon_id` must be provided.
    """
)
async def save_preferences(request: SavePreferencesRequest):
    """Save Tinder test results to Supabase."""

    if not request.user_id and not request.anon_id:
        raise HTTPException(
            status_code=400,
            detail="Either user_id or anon_id must be provided"
        )

    service = get_service()

    result = service.save_tinder_preferences(
        user_id=request.user_id,
        anon_id=request.anon_id,
        gender=request.gender,
        rounds_completed=request.rounds_completed,
        categories_tested=request.categories_tested,
        attribute_preferences=request.attribute_preferences,
        prediction_accuracy=request.prediction_accuracy,
        taste_vector=request.taste_vector
    )

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))

    return SavePreferencesResponse(**result)


@router.get(
    "/feed/{user_id}",
    response_model=FeedResponse,
    summary="Get personalized feed from Supabase",
    description="""
    Get personalized product recommendations based on user's taste vector.

    **Strategy Selection:**
    - `seed_vector`: User has completed Tinder test - uses pgvector similarity
    - `trending`: New user - returns trending products

    **Parameters:**
    - `user_id`: User ID (UUID or anon_id)
    - `gender`: Filter by gender (default: female)
    - `categories`: Comma-separated category filter
    - `limit`: Number of results (default: 50)
    - `offset`: Number of results to skip for pagination (default: 0)
    """
)
async def get_feed(
    user_id: str,
    gender: str = Query("female", description="Gender filter"),
    categories: Optional[str] = Query(None, description="Comma-separated categories"),
    limit: int = Query(50, ge=1, le=200, description="Number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip (pagination)")
):
    """Get personalized feed from Supabase."""

    service = get_service()

    # Parse categories
    cat_list = None
    if categories:
        cat_list = [c.strip() for c in categories.split(",")]

    # Check if it's a UUID or anon_id
    is_uuid = len(user_id) == 36 and user_id.count("-") == 4

    result = service.get_recommendations(
        user_id=user_id if is_uuid else None,
        anon_id=user_id if not is_uuid else None,
        gender=gender,
        categories=cat_list,
        limit=limit,
        offset=offset
    )

    return FeedResponse(**result)


@router.get(
    "/similar/{product_id}",
    summary="Get similar products",
    description="""
    Find products visually similar to the given product using pgvector.

    Uses FashionCLIP embeddings for similarity matching.
    """
)
async def get_similar(
    product_id: str,
    gender: str = Query("female", description="Gender filter"),
    category: Optional[str] = Query(None, description="Category filter"),
    limit: int = Query(20, ge=1, le=100, description="Number of results")
):
    """Get similar products."""

    service = get_service()

    results = service.get_similar_products(
        product_id=product_id,
        gender=gender,
        category=category,
        limit=limit
    )

    if not results:
        # Check if product exists
        product = service.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail=f"Product not found: {product_id}")
        # Product exists but no embedding
        return {
            "product_id": product_id,
            "similar": [],
            "message": "Product has no embedding for similarity search"
        }

    return {
        "product_id": product_id,
        "similar": results
    }


@router.get(
    "/trending",
    summary="Get trending products",
    description="""
    Get trending/popular products.

    Used as fallback when no personalization is available.
    Supports pagination with offset parameter.
    """
)
async def get_trending(
    gender: str = Query("female", description="Gender filter"),
    category: Optional[str] = Query(None, description="Category filter"),
    limit: int = Query(50, ge=1, le=200, description="Number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip (pagination)")
):
    """Get trending products."""

    service = get_service()

    results = service.get_trending_products(
        gender=gender,
        category=category,
        limit=limit,
        offset=offset
    )

    return {
        "gender": gender,
        "category": category,
        "results": results,
        "count": len(results),
        "offset": offset,
        "has_more": len(results) == limit
    }


@router.get(
    "/categories",
    response_model=List[CategoryCount],
    summary="Get product categories",
    description="Get list of product categories with counts."
)
async def get_categories(
    gender: str = Query("female", description="Gender filter")
):
    """Get product categories."""

    service = get_service()

    results = service.get_product_categories(gender)

    # Filter out NULL categories
    return [CategoryCount(**r) for r in results if r.get("category")]


@router.get(
    "/product/{product_id}",
    summary="Get product details",
    description="Get details for a single product."
)
async def get_product(product_id: str):
    """Get product details."""

    service = get_service()

    result = service.get_product(product_id)

    if not result:
        raise HTTPException(status_code=404, detail=f"Product not found: {product_id}")

    return result


@router.get(
    "/health",
    summary="Health check",
    description="Check if the recommendation service is healthy."
)
async def health_check():
    """Health check."""

    try:
        service = get_service()
        categories = service.get_product_categories("female")

        return {
            "status": "healthy",
            "supabase_connected": True,
            "categories_count": len(categories),
            "total_products": sum(c["product_count"] for c in categories)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# =============================================================================
# V2 Pipeline Endpoints (NEW)
# =============================================================================

# Create v2 router
v2_router = APIRouter(prefix="/api/recs/v2", tags=["Recommendations V2 (Pipeline)"])


# -----------------------------------------------------------------------------
# Frontend-matching Request Models (exactly as sent by frontend)
# -----------------------------------------------------------------------------

class FrontendCoreSetup(BaseModel):
    """Module 1: Core Setup - exactly as sent by frontend."""
    selectedCategories: List[str] = Field(default_factory=list)
    sizes: List[str] = Field(default_factory=list)
    birthdate: Optional[str] = None
    colorsToAvoid: List[str] = Field(default_factory=list)
    materialsToAvoid: List[str] = Field(default_factory=list)
    enabled: bool = True


class FrontendTopsModule(BaseModel):
    """Module 2: Tops - exactly as sent by frontend."""
    topTypes: List[str] = Field(default_factory=list)
    fits: List[str] = Field(default_factory=list)
    sleeves: List[str] = Field(default_factory=list)
    priceComfort: Optional[float] = None
    enabled: bool = True


class FrontendBottomsModule(BaseModel):
    """Module 3: Bottoms - exactly as sent by frontend."""
    bottomTypes: List[str] = Field(default_factory=list)
    fits: List[str] = Field(default_factory=list)
    rises: List[str] = Field(default_factory=list)
    lengths: List[str] = Field(default_factory=list)
    numericWaist: Optional[int] = None
    numericHip: Optional[int] = None
    priceComfort: Optional[float] = None
    enabled: bool = True


class FrontendSkirtsModule(BaseModel):
    """Module 4: Skirts - exactly as sent by frontend."""
    skirtTypes: List[str] = Field(default_factory=list)
    lengths: List[str] = Field(default_factory=list)
    fits: List[str] = Field(default_factory=list)
    numericWaist: Optional[int] = None
    priceComfort: Optional[float] = None
    enabled: bool = True


class FrontendDressesModule(BaseModel):
    """Module 5: Dresses - exactly as sent by frontend."""
    dressTypes: List[str] = Field(default_factory=list)
    fits: List[str] = Field(default_factory=list)
    lengths: List[str] = Field(default_factory=list)
    sleeves: List[str] = Field(default_factory=list)
    priceComfort: Optional[float] = None
    enabled: bool = True


class FrontendOnePieceModule(BaseModel):
    """Module 6: One-piece - exactly as sent by frontend."""
    onePieceTypes: List[str] = Field(default_factory=list)
    fits: List[str] = Field(default_factory=list)
    lengths: List[str] = Field(default_factory=list)
    numericWaist: Optional[int] = None
    priceComfort: Optional[float] = None
    enabled: bool = True


class FrontendOuterwearModule(BaseModel):
    """Module 7: Outerwear - exactly as sent by frontend."""
    outerwearTypes: List[str] = Field(default_factory=list)
    fits: List[str] = Field(default_factory=list)
    sleeves: List[str] = Field(default_factory=list)
    priceComfort: Optional[float] = None
    enabled: bool = True


class FrontendStyleModule(BaseModel):
    """Module 8: Style - exactly as sent by frontend."""
    styleDirections: List[str] = Field(default_factory=list)
    modestyPreference: Optional[str] = None
    enabled: bool = True


class FrontendBrandsModule(BaseModel):
    """Module 9: Brands - exactly as sent by frontend."""
    preferredBrands: List[str] = Field(default_factory=list)
    brandsToAvoid: List[str] = Field(default_factory=list)
    brandOpenness: Optional[str] = None
    enabled: bool = True


class FrontendStyleDiscoverySelection(BaseModel):
    """A single selection in style discovery."""
    round: Optional[int] = None
    category: Optional[str] = None
    winnerId: Optional[str] = None
    loserId: Optional[str] = None
    timestamp: Optional[str] = None


class FrontendStyleDiscoverySummary(BaseModel):
    """Summary of style discovery results - flexible to accept all frontend fields."""
    attribute_preferences: Dict[str, Any] = Field(default_factory=dict)
    taste_stability: Optional[float] = None
    taste_vector: Optional[List[float]] = None
    # Additional fields from frontend (all optional)
    session_complete: Optional[bool] = None
    total_swipes: Optional[int] = None
    likes: Optional[int] = None
    dislikes: Optional[int] = None
    coverage: Optional[str] = None
    clusters_rejected: Optional[int] = None
    cluster_health: Optional[Dict[str, Any]] = None
    brand_preferences: Optional[Dict[str, Any]] = None
    style_profile: Optional[Dict[str, Any]] = None
    colors_avoided: Optional[List[str]] = None
    gender: Optional[str] = None
    rounds_completed: Optional[int] = None
    categories_completed: Optional[List[str]] = None
    total_categories: Optional[int] = None
    current_category: Optional[str] = None
    prediction_accuracy: Optional[float] = None
    total_predictions: Optional[int] = None
    correct_predictions: Optional[int] = None
    category_stats: Optional[Dict[str, Any]] = None
    information_bits: Optional[int] = None
    pairwise_comparisons: Optional[int] = None

    class Config:
        extra = "allow"  # Allow additional fields not explicitly defined


class FrontendStyleDiscoveryModule(BaseModel):
    """Module 10: Style Discovery (Tinder test) - exactly as sent by frontend."""
    userId: Optional[str] = None
    # selections can be either list of strings (item IDs) or list of selection objects
    selections: List[Any] = Field(default_factory=list)
    roundsCompleted: int = 0
    sessionComplete: bool = False
    summary: Optional[FrontendStyleDiscoverySummary] = None
    enabled: bool = True


class FullOnboardingRequest(BaseModel):
    """
    Complete 10-module onboarding request - exactly as sent by frontend.

    This model accepts the exact JSON structure from the frontend.
    """
    # User identification
    user_id: Optional[str] = Field(None, alias="userId")
    anon_id: Optional[str] = Field(None, alias="anonId")
    gender: str = "female"

    # All 10 modules (using frontend field names with hyphens as Python identifiers)
    core_setup: Optional[FrontendCoreSetup] = Field(None, alias="core-setup")
    tops: Optional[FrontendTopsModule] = None
    bottoms: Optional[FrontendBottomsModule] = None
    skirts: Optional[FrontendSkirtsModule] = None
    dresses: Optional[FrontendDressesModule] = None
    one_piece: Optional[FrontendOnePieceModule] = Field(None, alias="one-piece")
    outerwear: Optional[FrontendOuterwearModule] = None
    style: Optional[FrontendStyleModule] = None
    brands: Optional[FrontendBrandsModule] = None
    style_discovery: Optional[FrontendStyleDiscoveryModule] = Field(None, alias="style-discovery")

    # Metadata
    completedAt: Optional[str] = None

    class Config:
        populate_by_name = True  # Allow both alias and field name


class OnboardingResponse(BaseModel):
    """Response from saving onboarding profile."""
    status: str
    user_id: str
    modules_saved: int
    categories_selected: List[str]
    has_taste_vector: bool = False


def transform_frontend_to_profile(request: FullOnboardingRequest, user_key: str) -> OnboardingProfile:
    """Transform frontend request structure to internal OnboardingProfile."""

    # Start with basic profile
    profile = OnboardingProfile(user_id=user_key)

    # Module 1: Core Setup
    if request.core_setup and request.core_setup.enabled:
        profile.categories = request.core_setup.selectedCategories
        profile.sizes = request.core_setup.sizes
        profile.birthdate = request.core_setup.birthdate
        profile.colors_to_avoid = request.core_setup.colorsToAvoid
        profile.materials_to_avoid = request.core_setup.materialsToAvoid

    # Module 2: Tops
    if request.tops and request.tops.enabled:
        profile.tops = TopsPrefs(
            types=request.tops.topTypes,
            fits=request.tops.fits,
            sleeves=request.tops.sleeves,
            price_comfort=request.tops.priceComfort,
            enabled=True
        )

    # Module 3: Bottoms
    if request.bottoms and request.bottoms.enabled:
        profile.bottoms = BottomsPrefs(
            types=request.bottoms.bottomTypes,
            fits=request.bottoms.fits,
            rises=request.bottoms.rises,
            lengths=request.bottoms.lengths,
            numeric_waist=request.bottoms.numericWaist,
            numeric_hip=request.bottoms.numericHip,
            price_comfort=request.bottoms.priceComfort,
            enabled=True
        )

    # Module 4: Skirts
    if request.skirts and request.skirts.enabled:
        profile.skirts = SkirtsPrefs(
            types=request.skirts.skirtTypes,
            lengths=request.skirts.lengths,
            fits=request.skirts.fits,
            numeric_waist=request.skirts.numericWaist,
            price_comfort=request.skirts.priceComfort,
            enabled=True
        )

    # Module 5: Dresses
    if request.dresses and request.dresses.enabled:
        profile.dresses = DressesPrefs(
            types=request.dresses.dressTypes,
            fits=request.dresses.fits,
            lengths=request.dresses.lengths,
            sleeves=request.dresses.sleeves,
            price_comfort=request.dresses.priceComfort,
            enabled=True
        )

    # Module 6: One-piece
    if request.one_piece and request.one_piece.enabled:
        profile.one_piece = OnePiecePrefs(
            types=request.one_piece.onePieceTypes,
            fits=request.one_piece.fits,
            lengths=request.one_piece.lengths,
            numeric_waist=request.one_piece.numericWaist,
            price_comfort=request.one_piece.priceComfort,
            enabled=True
        )

    # Module 7: Outerwear
    if request.outerwear and request.outerwear.enabled:
        profile.outerwear = OuterwearPrefs(
            types=request.outerwear.outerwearTypes,
            fits=request.outerwear.fits,
            sleeves=request.outerwear.sleeves,
            price_comfort=request.outerwear.priceComfort,
            enabled=True
        )

    # Module 8: Style
    if request.style and request.style.enabled:
        profile.style_directions = request.style.styleDirections
        profile.modesty = request.style.modestyPreference

    # Module 9: Brands
    if request.brands and request.brands.enabled:
        profile.preferred_brands = request.brands.preferredBrands
        profile.brands_to_avoid = request.brands.brandsToAvoid
        profile.brand_openness = request.brands.brandOpenness

    # Module 10: Style Discovery (Tinder test)
    if request.style_discovery and request.style_discovery.enabled:
        selections = []
        for sel in request.style_discovery.selections:
            # Handle both string selections (item IDs) and object selections
            if isinstance(sel, str):
                # Simple string selection - just store the item ID
                selections.append(StyleDiscoverySelection(
                    round=0,
                    category=None,
                    winner_id=sel,
                    loser_id="",
                    timestamp=None
                ))
            elif isinstance(sel, dict):
                # Dict from JSON
                selections.append(StyleDiscoverySelection(
                    round=sel.get('round', 0),
                    category=sel.get('category'),
                    winner_id=sel.get('winnerId', ''),
                    loser_id=sel.get('loserId', ''),
                    timestamp=sel.get('timestamp')
                ))
            else:
                # Pydantic model object
                selections.append(StyleDiscoverySelection(
                    round=getattr(sel, 'round', 0) or 0,
                    category=getattr(sel, 'category', None),
                    winner_id=getattr(sel, 'winnerId', '') or '',
                    loser_id=getattr(sel, 'loserId', '') or '',
                    timestamp=getattr(sel, 'timestamp', None)
                ))

        summary = None
        if request.style_discovery.summary:
            summary = StyleDiscoverySummary(
                attribute_preferences=request.style_discovery.summary.attribute_preferences,
                taste_stability=request.style_discovery.summary.taste_stability,
                taste_vector=request.style_discovery.summary.taste_vector
            )

        profile.style_discovery = StyleDiscoveryModule(
            user_id=request.style_discovery.userId,
            selections=selections,
            rounds_completed=request.style_discovery.roundsCompleted,
            session_complete=request.style_discovery.sessionComplete,
            summary=summary,
            enabled=True
        )

    # Metadata
    profile.completed_at = request.completedAt

    return profile


@v2_router.post(
    "/onboarding",
    response_model=OnboardingResponse,
    summary="Save 10-module onboarding profile",
    description="""
    Save user's complete onboarding profile from all 10 modules.

    **Modules:**
    1. Core Setup (selectedCategories, sizes, birthdate, colorsToAvoid, materialsToAvoid) - HARD FILTERS
    2. Tops preferences (topTypes, fits, sleeves, priceComfort)
    3. Bottoms preferences (bottomTypes, fits, rises, lengths, numericWaist, numericHip, priceComfort)
    4. Skirts preferences (skirtTypes, lengths, fits, numericWaist, priceComfort)
    5. Dresses preferences (dressTypes, fits, lengths, sleeves, priceComfort)
    6. One-piece preferences (onePieceTypes, fits, lengths, numericWaist, priceComfort)
    7. Outerwear preferences (outerwearTypes, fits, sleeves, priceComfort)
    8. Style (styleDirections, modestyPreference)
    9. Brands (preferredBrands, brandsToAvoid, brandOpenness)
    10. Style Discovery - Tinder test results (selections, summary with taste_vector)

    The request body should match the exact structure sent by the frontend.
    """
)
async def save_onboarding(request: FullOnboardingRequest):
    """Save 10-module onboarding profile."""

    user_key = request.user_id or request.anon_id
    if not user_key:
        raise HTTPException(
            status_code=400,
            detail="Either user_id/userId or anon_id/anonId must be provided"
        )

    # Transform frontend structure to internal profile
    profile = transform_frontend_to_profile(request, user_key)

    # Count enabled modules
    modules_saved = 0
    if request.core_setup and request.core_setup.enabled:
        modules_saved += 1
    if request.tops and request.tops.enabled:
        modules_saved += 1
    if request.bottoms and request.bottoms.enabled:
        modules_saved += 1
    if request.skirts and request.skirts.enabled:
        modules_saved += 1
    if request.dresses and request.dresses.enabled:
        modules_saved += 1
    if request.one_piece and request.one_piece.enabled:
        modules_saved += 1
    if request.outerwear and request.outerwear.enabled:
        modules_saved += 1
    if request.style and request.style.enabled:
        modules_saved += 1
    if request.brands and request.brands.enabled:
        modules_saved += 1
    if request.style_discovery and request.style_discovery.enabled:
        modules_saved += 1

    # Check if we have a taste vector (ensure boolean, not None)
    has_taste_vector = bool(
        request.style_discovery and
        request.style_discovery.summary and
        request.style_discovery.summary.taste_vector is not None
    )

    # Save via pipeline
    pipeline = get_pipeline()
    result = pipeline.save_onboarding(profile, gender=request.gender)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))

    return OnboardingResponse(
        status=result.get("status", "success"),
        user_id=user_key,
        modules_saved=modules_saved,
        categories_selected=profile.categories,
        has_taste_vector=has_taste_vector
    )


# =============================================================================
# Category Mapping: Onboarding -> Tinder Test Categories
# =============================================================================

# Mapping from broad onboarding categories to specific Tinder test categories
ONBOARDING_TO_TINDER_CATEGORIES = {
    "tops": ["tops_knitwear", "tops_woven", "tops_sleeveless", "tops_special"],
    "bottoms": ["bottoms_trousers", "bottoms_skorts"],
    "dresses": ["dresses"],
    "skirts": ["bottoms_skorts"],  # Skirts are in bottoms_skorts
    "outerwear": ["outerwear"],
    "one-piece": ["dresses"],  # Jumpsuits often grouped with dresses
    "sportswear": ["sportswear"],
}

def map_onboarding_to_tinder_categories(onboarding_categories: List[str]) -> List[str]:
    """Map broad onboarding categories to specific Tinder test categories."""
    tinder_categories = set()
    for cat in onboarding_categories:
        cat_lower = cat.lower().replace("-", "_").replace(" ", "_")
        if cat_lower in ONBOARDING_TO_TINDER_CATEGORIES:
            tinder_categories.update(ONBOARDING_TO_TINDER_CATEGORIES[cat_lower])
    return list(tinder_categories) if tinder_categories else None  # None = all categories


class PartialOnboardingRequest(BaseModel):
    """Request to save partial onboarding (core-setup only)."""
    user_id: Optional[str] = Field(None, alias="userId")
    anon_id: Optional[str] = Field(None, alias="anonId")
    gender: str = "female"
    core_setup: FrontendCoreSetup = Field(..., alias="core-setup")

    class Config:
        populate_by_name = True


class PartialOnboardingResponse(BaseModel):
    """Response from partial onboarding with mapped Tinder categories."""
    status: str
    user_id: str
    categories_selected: List[str]
    tinder_categories: List[str]  # Mapped categories for Tinder test
    colors_to_avoid: List[str]
    materials_to_avoid: List[str]


@v2_router.post(
    "/onboarding/core-setup",
    response_model=PartialOnboardingResponse,
    summary="Save core-setup and get Tinder test categories",
    description="""
    Save just the core-setup module and get the mapped categories for the Tinder test.

    Call this BEFORE starting the style-discovery (Tinder) test so the test
    only shows items from the user's selected categories.

    **Flow:**
    1. User completes core-setup in onboarding
    2. Frontend calls this endpoint with core-setup data
    3. Backend returns `tinder_categories` - use these when calling `/api/women/session/start`
    4. User completes Tinder test with filtered categories
    5. Frontend calls `/api/recs/v2/onboarding` with complete 10-module data
    """
)
async def save_core_setup(request: PartialOnboardingRequest):
    """Save core-setup and return mapped Tinder categories."""

    user_key = request.user_id or request.anon_id
    if not user_key:
        raise HTTPException(
            status_code=400,
            detail="Either user_id/userId or anon_id/anonId must be provided"
        )

    core = request.core_setup

    # Map onboarding categories to Tinder categories
    tinder_cats = map_onboarding_to_tinder_categories(core.selectedCategories)

    # Save partial profile (will be overwritten when full onboarding is saved)
    profile = OnboardingProfile(
        user_id=user_key,
        categories=core.selectedCategories,
        sizes=core.sizes,
        birthdate=core.birthdate,
        colors_to_avoid=core.colorsToAvoid,
        materials_to_avoid=core.materialsToAvoid,
    )

    pipeline = get_pipeline()
    result = pipeline.save_onboarding(profile, gender=request.gender)

    return PartialOnboardingResponse(
        status="success",
        user_id=user_key,
        categories_selected=core.selectedCategories,
        tinder_categories=tinder_cats if tinder_cats else [
            "tops_knitwear", "tops_woven", "tops_sleeveless", "tops_special",
            "dresses", "bottoms_trousers", "bottoms_skorts", "outerwear"
        ],
        colors_to_avoid=core.colorsToAvoid,
        materials_to_avoid=core.materialsToAvoid
    )


@v2_router.get(
    "/categories/mapping",
    summary="Get category mapping for Tinder test",
    description="Get the mapping from onboarding categories to Tinder test categories."
)
async def get_category_mapping():
    """Get category mapping."""
    return {
        "onboarding_to_tinder": ONBOARDING_TO_TINDER_CATEGORIES,
        "tinder_categories": [
            {"id": "tops_knitwear", "label": "Sweaters & Knits", "broad": "tops"},
            {"id": "tops_woven", "label": "Blouses & Shirts", "broad": "tops"},
            {"id": "tops_sleeveless", "label": "Tank Tops & Camis", "broad": "tops"},
            {"id": "tops_special", "label": "Bodysuits", "broad": "tops"},
            {"id": "dresses", "label": "Dresses", "broad": "dresses"},
            {"id": "bottoms_trousers", "label": "Pants & Trousers", "broad": "bottoms"},
            {"id": "bottoms_skorts", "label": "Skirts & Shorts", "broad": "bottoms,skirts"},
            {"id": "outerwear", "label": "Outerwear", "broad": "outerwear"},
            {"id": "sportswear", "label": "Sportswear", "broad": "sportswear"},
        ],
        "onboarding_categories": [
            {"id": "tops", "label": "Tops", "tinder_maps_to": ["tops_knitwear", "tops_woven", "tops_sleeveless", "tops_special"]},
            {"id": "bottoms", "label": "Bottoms", "tinder_maps_to": ["bottoms_trousers", "bottoms_skorts"]},
            {"id": "dresses", "label": "Dresses", "tinder_maps_to": ["dresses"]},
            {"id": "skirts", "label": "Skirts", "tinder_maps_to": ["bottoms_skorts"]},
            {"id": "outerwear", "label": "Outerwear", "tinder_maps_to": ["outerwear"]},
            {"id": "one-piece", "label": "One-Piece", "tinder_maps_to": ["dresses"]},
        ]
    }


@v2_router.get(
    "/feed",
    summary="Get personalized feed using full pipeline",
    description="""
    Get personalized product recommendations using the full recommendation pipeline.

    **Now uses O(1) keyset pagination internally** - supports infinite scroll through
    entire catalog without performance degradation.

    **Pipeline Stages:**
    1. Hard filtering (colors, materials, brands to avoid)
    2. Candidate retrieval (taste_vector similarity OR trending for cold users)
    3. Soft preference scoring (fit, style, length matching)
    4. SASRec ranking (for warm users with 5+ interactions)
    5. Diversity constraints (max 8 per category)
    6. Exploration injection (10% for discovery)

    **User States:**
    - `cold_start`: No Tinder test → trending only
    - `tinder_complete`: Has taste_vector → personalized embedding similarity
    - `warm_user`: 5+ interactions → SASRec + embedding + preference

    **Pagination:**
    - First request: Don't send `cursor` or `session_id`
    - Subsequent requests: Send back the `cursor` and `session_id` from previous response
    - No duplicates within session guaranteed
    """
)
async def get_pipeline_feed(
    user_id: Optional[str] = Query(None, description="UUID user ID"),
    anon_id: Optional[str] = Query(None, description="Anonymous user ID"),
    session_id: Optional[str] = Query(None, description="Session ID (returned in response, send back for pagination)"),
    gender: str = Query("female", description="Gender filter"),
    categories: Optional[str] = Query(None, description="Comma-separated broad category filter (tops, bottoms, dresses, outerwear)"),
    article_types: Optional[str] = Query(None, description="Comma-separated article type filter (e.g., 'jeans,t-shirts,tank tops')"),
    cursor: Optional[str] = Query(None, description="Cursor from previous response (for pagination)"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page")
):
    """Get personalized feed using the full recommendation pipeline with keyset pagination."""

    if not user_id and not anon_id:
        raise HTTPException(
            status_code=400,
            detail="Either user_id or anon_id must be provided"
        )

    pipeline = get_pipeline()

    # Parse categories
    cat_list = None
    if categories:
        cat_list = [c.strip() for c in categories.split(",")]

    # Parse article_types
    article_type_list = None
    if article_types:
        article_type_list = [a.strip() for a in article_types.split(",")]

    # Use keyset pagination internally for O(1) performance
    response = pipeline.get_feed_keyset(
        user_id=user_id,
        anon_id=anon_id,
        session_id=session_id,
        gender=gender,
        categories=cat_list,
        article_types=article_type_list,
        cursor=cursor,
        page_size=page_size
    )

    return response


@v2_router.get(
    "/feed/endless",
    summary="Get endless scroll feed with session state",
    description="""
    Get personalized recommendations with TRUE endless scroll support.

    **Key Differences from /feed:**
    1. Uses session state to track seen items (no duplicates within session)
    2. Generates FRESH candidates per page (not slicing fixed pool)
    3. Returns `session_id` for client to persist (in sessionStorage)
    4. Each page request retrieves NEW candidates from database

    **How to Use:**
    1. First request: Call without session_id (auto-generated and returned)
    2. Subsequent requests: Include returned session_id to continue session
    3. New tab/refresh: Omit session_id to start fresh session

    **Session Behavior:**
    - Sessions last 24 hours
    - Each page shows items you haven't seen in this session
    - has_more=true until database is exhausted
    - Closing browser tab preserves session (if session_id persisted)

    **Example Flow:**
    1. GET /feed/endless?anon_id=abc123&page_size=50
       -> Returns 50 items + session_id=sess_xyz123
    2. GET /feed/endless?anon_id=abc123&session_id=sess_xyz123&page=1
       -> Returns NEXT 50 items (no duplicates)
    3. Repeat until has_more=false
    """
)
async def get_endless_feed(
    user_id: Optional[str] = Query(None, description="UUID user ID"),
    anon_id: Optional[str] = Query(None, description="Anonymous user ID"),
    session_id: Optional[str] = Query(None, description="Session ID (auto-generated if not provided)"),
    gender: str = Query("female", description="Gender filter"),
    categories: Optional[str] = Query(None, description="Comma-separated category filter"),
    page: int = Query(0, ge=0, description="Page number (0-indexed)"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page")
):
    """Get endless scroll feed with session state tracking."""

    if not user_id and not anon_id:
        raise HTTPException(
            status_code=400,
            detail="Either user_id or anon_id must be provided"
        )

    pipeline = get_pipeline()

    # Parse categories
    cat_list = None
    if categories:
        cat_list = [c.strip() for c in categories.split(",")]

    response = pipeline.get_feed_endless(
        user_id=user_id,
        anon_id=anon_id,
        session_id=session_id,
        gender=gender,
        categories=cat_list,
        page=page,
        page_size=page_size
    )

    return response


@v2_router.get(
    "/feed/keyset",
    summary="Get feed with keyset cursor pagination (V2)",
    description="""
    Get personalized recommendations with O(1) keyset cursor pagination.

    **V2 Improvements over /feed/endless:**
    1. Uses keyset cursor (score, id) instead of exclusion arrays
    2. O(1) pagination regardless of page depth (page 100 same speed as page 1)
    3. Feed versioning for stable ordering within session
    4. Authoritative dedup via Redis SET (no false positives)

    **How to Use:**
    1. First request: Call without cursor (returns first page + cursor)
    2. Subsequent requests: Pass returned cursor to get next page
    3. Session_id is auto-generated and returned

    **API Contract:**
    - cursor: Opaque base64 string from previous response (NULL for first page)
    - Returns cursor for next page (use this in next request)
    - has_more: true until catalog exhausted

    **Example Flow:**
    1. GET /feed/keyset?anon_id=abc123&page_size=50
       -> Returns 50 items + session_id + cursor="eyJz..."
    2. GET /feed/keyset?anon_id=abc123&session_id=sess_xxx&cursor=eyJz...
       -> Returns NEXT 50 items + new cursor
    3. Repeat until has_more=false

    **Why Keyset Pagination:**
    - Exclusion arrays (V1) are O(n) per row - slow with 1000+ seen items
    - Keyset cursor is O(1) - uses WHERE (score, id) < (cursor_score, cursor_id)
    - Index makes it constant time regardless of pagination depth
    """
)
async def get_keyset_feed(
    user_id: Optional[str] = Query(None, description="UUID user ID"),
    anon_id: Optional[str] = Query(None, description="Anonymous user ID"),
    session_id: Optional[str] = Query(None, description="Session ID (auto-generated if not provided)"),
    gender: str = Query("female", description="Gender filter"),
    categories: Optional[str] = Query(None, description="Comma-separated category filter"),
    cursor: Optional[str] = Query(None, description="Opaque cursor from previous response (NULL for first page)"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page")
):
    """Get keyset cursor paginated feed."""

    if not user_id and not anon_id:
        raise HTTPException(
            status_code=400,
            detail="Either user_id or anon_id must be provided"
        )

    pipeline = get_pipeline()

    # Parse categories
    cat_list = None
    if categories:
        cat_list = [c.strip() for c in categories.split(",")]

    response = pipeline.get_feed_keyset(
        user_id=user_id,
        anon_id=anon_id,
        session_id=session_id,
        gender=gender,
        categories=cat_list,
        cursor=cursor,
        page_size=page_size
    )

    return response


@v2_router.get(
    "/feed/session/{session_id}",
    summary="Get session info for debugging",
    description="Get information about a specific session (seen count, signals, etc.)"
)
async def get_session_info(session_id: str):
    """Get session info for debugging."""

    pipeline = get_pipeline()
    info = pipeline.get_session_info(session_id)

    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )

    return info


@v2_router.delete(
    "/feed/session/{session_id}",
    summary="Clear session (reset seen items)",
    description="Clear a session to start fresh. Use for testing or user-requested reset."
)
async def clear_session(session_id: str):
    """Clear a session."""

    pipeline = get_pipeline()
    pipeline.clear_session(session_id)

    return {
        "status": "success",
        "session_id": session_id,
        "message": "Session cleared. Next feed request will show fresh items."
    }


@v2_router.get(
    "/info",
    summary="Get pipeline configuration info",
    description="Get information about the recommendation pipeline configuration."
)
async def get_pipeline_info():
    """Get pipeline configuration info."""

    try:
        pipeline = get_pipeline()
        return pipeline.get_pipeline_info()
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@v2_router.get(
    "/health",
    summary="Pipeline health check",
    description="Check if the recommendation pipeline is healthy and ready."
)
async def pipeline_health():
    """Pipeline health check."""

    try:
        pipeline = get_pipeline()
        info = pipeline.get_pipeline_info()

        return {
            "status": "healthy",
            "sasrec_loaded": info["sasrec_ranker"]["model_loaded"],
            "sasrec_vocab_size": info["sasrec_ranker"]["vocab_size"],
            "pipeline_ready": True
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "pipeline_ready": False
        }


# =============================================================================
# User Interaction Tracking Endpoints
# =============================================================================

@v2_router.post(
    "/feed/action",
    response_model=RecordActionResponse,
    summary="Record user interaction",
    description="""
    Record an explicit user interaction with a product.

    **Valid Actions:**
    - `click`: User tapped to view product details
    - `hover`: User swiped through photo gallery
    - `add_to_wishlist`: User saved/liked the item (strong positive signal)
    - `add_to_cart`: User added to cart (conversion intent)
    - `purchase`: User completed purchase (conversion)

    **Note:** View and skip actions are NOT tracked here - they are implicit
    from the session seen_ids (tracked via /session/sync endpoint).

    Call this endpoint immediately when a user takes an action.
    """
)
async def record_action(request: RecordActionRequest):
    """Record a user interaction."""

    # Validate action type
    if request.action not in VALID_INTERACTION_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{request.action}'. Must be one of: {VALID_INTERACTION_ACTIONS}"
        )

    # Require user identifier
    if not request.anon_id and not request.user_id:
        raise HTTPException(
            status_code=400,
            detail="Either anon_id or user_id must be provided"
        )

    service = get_service()

    try:
        result = service.record_user_interaction(
            anon_id=request.anon_id,
            user_id=request.user_id,
            session_id=request.session_id,
            product_id=request.product_id,
            action=request.action,
            source=request.source,
            position=request.position
        )

        return RecordActionResponse(
            status="success",
            interaction_id=result.get("id")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v2_router.post(
    "/session/sync",
    response_model=SyncSessionResponse,
    summary="Sync session seen_ids for ML training",
    description="""
    Sync the list of product IDs shown to the user in this session.

    **Purpose:** This data is used for ML training - products shown but not
    interacted with become implicit negative samples.

    **When to Call:**
    - Every N pages (e.g., every 5 pages of scrolling)
    - On app close/background (window.onbeforeunload)
    - On session end

    **Not for:** Real-time tracking. Use /feed/action for explicit interactions.
    """
)
async def sync_session(request: SyncSessionRequest):
    """Sync session seen_ids for training data."""

    # Require user identifier
    if not request.anon_id and not request.user_id:
        raise HTTPException(
            status_code=400,
            detail="Either anon_id or user_id must be provided"
        )

    if not request.seen_ids:
        return SyncSessionResponse(status="success", synced_count=0)

    service = get_service()

    try:
        result = service.sync_session_seen_ids(
            anon_id=request.anon_id,
            user_id=request.user_id,
            session_id=request.session_id,
            seen_ids=request.seen_ids
        )

        return SyncSessionResponse(
            status="success",
            synced_count=len(request.seen_ids)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Helper to integrate with existing swipe_server
# =============================================================================

def integrate_with_app(app):
    """
    Integrate these endpoints with an existing FastAPI app.

    Usage in swipe_server.py:
        from src.recs.api_endpoints import integrate_with_app
        integrate_with_app(app)
    """
    app.include_router(router)
    app.include_router(v2_router)
    print("Recommendation API endpoints integrated: /api/recs/*")
    print("Pipeline V2 endpoints integrated: /api/recs/v2/*")


# =============================================================================
# Standalone testing
# =============================================================================

if __name__ == "__main__":
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI(title="Recommendation API Test")
    app.include_router(router)

    print("Starting test server on http://localhost:8081")
    print("API docs: http://localhost:8081/docs")
    uvicorn.run(app, host="0.0.0.0", port=8081)
