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
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from core.auth import require_auth, SupabaseUser
from core.utils import sanitize_product_images
from recs.recommendation_service import RecommendationService
from recs.models import (
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
from recs.pipeline import RecommendationPipeline

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
VALID_INTERACTION_ACTIONS = {'click', 'hover', 'add_to_wishlist', 'add_to_cart', 'purchase', 'skip'}


class RecordActionRequest(BaseModel):
    """Request to record a user interaction."""
    anon_id: Optional[str] = Field(None, description="Anonymous user ID")
    user_id: Optional[str] = Field(None, description="UUID user ID")
    session_id: str = Field(..., description="Session ID from feed response")
    product_id: str = Field(..., description="Product UUID that was interacted with")
    action: str = Field(..., description="Action type: click, hover, add_to_wishlist, add_to_cart, purchase")
    source: Optional[str] = Field("feed", description="Source: feed, search, similar, style-this")
    position: Optional[int] = Field(None, description="Position in feed when interacted")
    # Session scoring fields (optional, sent by frontend for real-time scoring)
    brand: Optional[str] = Field(None, description="Product brand (for session scoring)")
    item_type: Optional[str] = Field(None, description="Product type/category (for session scoring)")
    attributes: Optional[Dict[str, str]] = Field(None, description="Product attributes for session scoring: {fit, color_family, pattern, ...}")


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
# Background Task Helpers
# =============================================================================


def _bg_persist_interaction(
    user_id: str,
    session_id: str,
    product_id: str,
    action: str,
    source: str,
    position: Optional[int],
):
    """Background task: persist interaction to Supabase (non-blocking)."""
    try:
        service = get_service()
        service.record_user_interaction(
            anon_id=None,
            user_id=user_id,
            session_id=session_id,
            product_id=product_id,
            action=action,
            source=source,
            position=position,
        )
    except Exception as e:
        print(f"[BG] Failed to persist interaction: {e}")


def _bg_persist_seen_ids(
    user_id: str,
    session_id: str,
    seen_ids: List[str],
):
    """Background task: persist seen_ids to Supabase for ML training (non-blocking)."""
    if not seen_ids:
        return
    try:
        service = get_service()
        service.sync_session_seen_ids(
            anon_id=None,
            user_id=user_id,
            session_id=session_id,
            seen_ids=seen_ids,
        )
    except Exception as e:
        print(f"[BG] Failed to persist seen_ids: {e}")


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

    **Pagination:**
    - Use `offset` for infinite scroll (0, 20, 40, ...)
    - `has_more` indicates if more results are available
    - Each item has a `rank` field (1-indexed, continuous across pages)

    **Carousel Usage:**
    - Just use `limit=10` without offset for top 10 similar items

    **Feed/Infinite Scroll Usage:**
    - Page 1: `?limit=20&offset=0`
    - Page 2: `?limit=20&offset=20`
    - Continue until `has_more=false`
    """
)
async def get_similar(
    product_id: str,
    gender: str = Query("female", description="Gender filter"),
    category: Optional[str] = Query(None, description="Category filter"),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip (for pagination)"),
    user: SupabaseUser = Depends(require_auth),
):
    """Get similar products with pagination support."""

    service = get_service()

    results = service.get_similar_products(
        product_id=product_id,
        gender=gender,
        category=category,
        limit=limit,
        offset=offset
    )

    if not results and offset == 0:
        # Check if product exists (only on first page)
        product = service.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail=f"Product not found: {product_id}")
        # Product exists but no embedding
        return {
            "product_id": product_id,
            "results": [],
            "pagination": {
                "offset": offset,
                "limit": limit,
                "returned": 0,
                "has_more": False
            },
            "message": "Product has no embedding for similarity search"
        }

    # Add rank to each result (1-indexed, continuous across pages)
    ranked_results = []
    for i, item in enumerate(results):
        ranked_results.append(sanitize_product_images({
            **item,
            "rank": offset + i + 1
        }))

    # Check if there are more results
    has_more = len(results) == limit

    return {
        "product_id": product_id,
        "results": ranked_results,
        "pagination": {
            "offset": offset,
            "limit": limit,
            "returned": len(ranked_results),
            "has_more": has_more
        }
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


class FrontendLifestyleModule(BaseModel):
    """Module 11: Lifestyle preferences - exactly as sent by frontend."""
    occasions: List[str] = Field(default_factory=list)  # ["casual", "office", "evening", "beach"]
    stylesToAvoid: List[str] = Field(default_factory=list)  # ["deep-necklines", "sheer", "cutouts", "backless", "strapless"]
    patterns: Dict[str, str] = Field(default_factory=dict)  # {"floral": "like", "animal-print": "avoid"}
    stylePersona: List[str] = Field(default_factory=list)  # ["classic", "minimal", "trendy"]
    enabled: bool = True


# =============================================================================
# V3 Frontend Models (NEW SPEC)
# =============================================================================

class FrontendCoreSetupV3(BaseModel):
    """V3 Core Setup - split sizes by category."""
    categories: List[str] = Field(default_factory=list)  # Renamed from selectedCategories
    birthdate: Optional[str] = None
    topSize: List[str] = Field(default_factory=list)  # NEW: split from sizes
    bottomSize: List[str] = Field(default_factory=list)  # NEW: split from sizes
    outerwearSize: List[str] = Field(default_factory=list)  # NEW: split from sizes
    colorsToAvoid: List[str] = Field(default_factory=list)
    materialsToAvoid: List[str] = Field(default_factory=list)


class FrontendFitCategoryMapping(BaseModel):
    """Maps a fit ID to the categories it applies to."""
    fitId: str
    categories: List[str] = Field(default_factory=list)


class FrontendSleeveCategoryMapping(BaseModel):
    """Maps a sleeve ID to the categories it applies to."""
    sleeveId: str
    categories: List[str] = Field(default_factory=list)


class FrontendLengthCategoryMapping(BaseModel):
    """Maps a length ID to the categories it applies to."""
    lengthId: str
    categories: List[str] = Field(default_factory=list)


class FrontendAttributePreferences(BaseModel):
    """V3 Flat attribute preferences with category mappings."""
    # Fit preferences
    fit: List[str] = Field(default_factory=list)  # ["regular", "relaxed"]
    fitCategories: List[Dict[str, Any]] = Field(default_factory=list)  # [{fitId, categories}]

    # Sleeve preferences
    sleeve: List[str] = Field(default_factory=list)  # ["short", "long"]
    sleeveCategories: List[Dict[str, Any]] = Field(default_factory=list)

    # Length preferences (for tops/bottoms)
    length: List[str] = Field(default_factory=list)  # ["cropped", "standard", "long"]
    lengthCategories: List[Dict[str, Any]] = Field(default_factory=list)

    # Length preferences for skirts/dresses
    lengthSkirtsDresses: List[str] = Field(default_factory=list)  # ["mini", "midi", "maxi"]
    lengthSkirtsDressesCategories: List[Dict[str, Any]] = Field(default_factory=list)

    # Rise preferences
    rise: List[str] = Field(default_factory=list)  # ["high", "mid", "low"]


class FrontendTypePreferencesCategory(BaseModel):
    """Types for a single category."""
    types: List[str] = Field(default_factory=list)


class FrontendTypePreferences(BaseModel):
    """V3 Simplified type preferences - skirts merged into bottoms, one-piece into dresses."""
    tops: Optional[FrontendTypePreferencesCategory] = None
    bottoms: Optional[FrontendTypePreferencesCategory] = None  # Now includes skirt types
    dresses: Optional[FrontendTypePreferencesCategory] = None  # Now includes jumpsuit/romper
    outerwear: Optional[FrontendTypePreferencesCategory] = None


class FrontendLifestyleV3(BaseModel):
    """V3 Lifestyle preferences with separate pattern arrays."""
    occasions: List[str] = Field(default_factory=list)
    stylesToAvoid: List[str] = Field(default_factory=list)
    patternsLiked: List[str] = Field(default_factory=list)  # NEW: was dict
    patternsAvoided: List[str] = Field(default_factory=list)  # NEW: was dict
    stylePersona: List[str] = Field(default_factory=list)  # NOW STORED


class FrontendBrandsV3(BaseModel):
    """V3 Brands - renamed fields."""
    preferred: List[str] = Field(default_factory=list)  # Renamed from preferredBrands
    toAvoid: List[str] = Field(default_factory=list)  # Renamed from brandsToAvoid
    openness: Optional[str] = None  # Renamed from brandOpenness


class FrontendStyleDiscoveryV3(BaseModel):
    """V3 Simplified style discovery."""
    completed: bool = False  # Renamed from sessionComplete
    swipedItems: List[str] = Field(default_factory=list)  # Simplified from selections


class FrontendStyleDiscoverySummaryV3(BaseModel):
    """V3 Style discovery summary - primarily for taste_vector."""
    taste_vector: Optional[List[float]] = None  # 512-dim FashionCLIP embedding


class FullOnboardingRequestV3(BaseModel):
    """
    V3 Complete onboarding request - NEW frontend spec.

    Key changes from V2:
    - coreSetup has split sizes (topSize, bottomSize, outerwearSize)
    - attributePreferences is flat with category mappings
    - typePreferences simplified (skirts -> bottoms, one-piece -> dresses)
    - lifestyle has separate patternsLiked/patternsAvoided arrays
    - brands has renamed fields
    - styleDiscovery simplified to completed + swipedItems
    """
    # User identification
    userId: str
    gender: str = "female"

    # All modules with new structure
    coreSetup: FrontendCoreSetupV3
    attributePreferences: Optional[FrontendAttributePreferences] = None
    typePreferences: Optional[FrontendTypePreferences] = None
    lifestyle: Optional[FrontendLifestyleV3] = None
    brands: Optional[FrontendBrandsV3] = None
    styleDiscovery: Optional[FrontendStyleDiscoveryV3] = None

    # Summary containing taste_vector (sent separately)
    summary: Optional[FrontendStyleDiscoverySummaryV3] = None

    # Metadata
    completedAt: Optional[str] = None


def transform_frontend_to_profile_v3(request: FullOnboardingRequestV3) -> OnboardingProfile:
    """
    Transform V3 frontend request structure to internal OnboardingProfile.

    This handles the NEW frontend spec:
    - Split sizes -> top_sizes, bottom_sizes, outerwear_sizes
    - Flat attributePreferences -> preferred_fits, fit_category_mapping, etc.
    - Simplified typePreferences -> top_types, bottom_types, dress_types, outerwear_types
    - Separate pattern arrays -> patterns_liked, patterns_avoided
    - stylePersona is now stored
    """
    profile = OnboardingProfile(user_id=request.userId)

    # Core Setup
    profile.categories = request.coreSetup.categories
    profile.birthdate = request.coreSetup.birthdate
    profile.top_sizes = request.coreSetup.topSize
    profile.bottom_sizes = request.coreSetup.bottomSize
    profile.outerwear_sizes = request.coreSetup.outerwearSize
    profile.colors_to_avoid = request.coreSetup.colorsToAvoid
    profile.materials_to_avoid = request.coreSetup.materialsToAvoid

    # Attribute Preferences (flat with category mappings)
    if request.attributePreferences:
        ap = request.attributePreferences
        profile.preferred_fits = ap.fit
        profile.fit_category_mapping = ap.fitCategories
        profile.preferred_sleeves = ap.sleeve
        profile.sleeve_category_mapping = ap.sleeveCategories
        profile.preferred_lengths = ap.length
        profile.length_category_mapping = ap.lengthCategories
        profile.preferred_lengths_dresses = ap.lengthSkirtsDresses
        profile.length_dresses_category_mapping = ap.lengthSkirtsDressesCategories
        profile.preferred_rises = ap.rise

    # Type Preferences (simplified)
    if request.typePreferences:
        tp = request.typePreferences
        if tp.tops:
            profile.top_types = tp.tops.types
        if tp.bottoms:
            profile.bottom_types = tp.bottoms.types  # Now includes skirt types
        if tp.dresses:
            profile.dress_types = tp.dresses.types  # Now includes jumpsuit/romper
        if tp.outerwear:
            profile.outerwear_types = tp.outerwear.types

    # Lifestyle
    if request.lifestyle:
        lf = request.lifestyle
        profile.occasions = lf.occasions
        profile.styles_to_avoid = lf.stylesToAvoid
        profile.patterns_liked = lf.patternsLiked
        profile.patterns_avoided = lf.patternsAvoided
        profile.style_persona = lf.stylePersona

    # Brands
    if request.brands:
        br = request.brands
        profile.preferred_brands = br.preferred
        profile.brands_to_avoid = br.toAvoid
        profile.brand_openness = br.openness

    # Style Discovery (simplified)
    if request.styleDiscovery:
        sd = request.styleDiscovery
        profile.style_discovery_complete = sd.completed
        profile.swiped_items = sd.swipedItems

    # Taste Vector from summary
    if request.summary and request.summary.taste_vector:
        profile.taste_vector = request.summary.taste_vector

    # Metadata
    profile.completed_at = request.completedAt

    return profile


class FullOnboardingRequest(BaseModel):
    """
    Complete 11-module onboarding request - exactly as sent by frontend.

    This model accepts the exact JSON structure from the frontend.
    """
    # User identification
    user_id: Optional[str] = Field(None, alias="userId")
    anon_id: Optional[str] = Field(None, alias="anonId")
    gender: str = "female"

    # All 11 modules (using frontend field names with hyphens as Python identifiers)
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
    lifestyle: Optional[FrontendLifestyleModule] = None  # NEW: Module 11

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

    # Module 11: Lifestyle preferences (NEW)
    if request.lifestyle and request.lifestyle.enabled:
        profile.occasions = request.lifestyle.occasions
        profile.styles_to_avoid = request.lifestyle.stylesToAvoid

        # Parse patterns into preferred and avoided lists
        if request.lifestyle.patterns:
            profile.patterns_to_avoid = [
                pattern for pattern, preference in request.lifestyle.patterns.items()
                if preference == 'avoid'
            ]
            profile.patterns_preferred = [
                pattern for pattern, preference in request.lifestyle.patterns.items()
                if preference == 'like'
            ]

    # Metadata
    profile.completed_at = request.completedAt

    return profile


@v2_router.post(
    "/onboarding",
    response_model=OnboardingResponse,
    summary="Save complete 10-module onboarding profile",
    description="""
    Save user's complete onboarding profile from the 10-module flow.

    **Modules (all optional except core-setup):**
    1. Core Setup (sizes, categories, exclusions) - REQUIRED
    2. Lifestyle (occupation, routines, budget)
    3. Tops preferences (topTypes, fits, sleeves, necklines, priceComfort)
    4. Bottoms preferences (bottomTypes, fits, rises, lengths, priceComfort)
    5. Dresses preferences (dressTypes, fits, lengths, sleeves, priceComfort)
    6. One-piece preferences (onePieceTypes, fits, lengths, numericWaist, priceComfort)
    7. Outerwear preferences (outerwearTypes, fits, sleeves, priceComfort)
    8. Style (styleDirections, modestyPreference)
    9. Brands (preferredBrands, brandsToAvoid, brandOpenness)
    10. Style Discovery - Tinder test results (selections, summary with taste_vector)

    The request body should match the exact structure sent by the frontend.
    """
)
async def save_onboarding(
    request: FullOnboardingRequest,
    user: SupabaseUser = Depends(require_auth)
):
    """Save 10-module onboarding profile."""

    user_key = user.id

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
    if request.lifestyle and request.lifestyle.enabled:
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
async def save_core_setup(
    request: PartialOnboardingRequest,
    user: SupabaseUser = Depends(require_auth)
):
    """Save core-setup and return mapped Tinder categories."""

    user_key = user.id

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


# =============================================================================
# V3 Onboarding Endpoint (NEW SPEC)
# =============================================================================

class OnboardingResponseV3(BaseModel):
    """Response from saving V3 onboarding profile."""
    status: str
    user_id: str
    categories_selected: List[str]
    has_taste_vector: bool = False
    has_attribute_preferences: bool = False
    has_type_preferences: bool = False


@v2_router.post(
    "/onboarding/v3",
    response_model=OnboardingResponseV3,
    summary="Save V3 onboarding profile (NEW SPEC)",
    description="""
    Save user's complete onboarding profile using the NEW frontend spec.

    **V3 Changes:**
    - `coreSetup.categories` (renamed from selectedCategories)
    - `coreSetup.topSize`, `coreSetup.bottomSize`, `coreSetup.outerwearSize` (split from sizes)
    - `attributePreferences` - flat structure with category mappings
    - `typePreferences` - simplified (skirts merged into bottoms, one-piece into dresses)
    - `lifestyle.patternsLiked` / `lifestyle.patternsAvoided` (was dict)
    - `lifestyle.stylePersona` - NOW STORED
    - `brands.preferred` / `brands.toAvoid` / `brands.openness` (renamed)
    - `styleDiscovery.completed` / `styleDiscovery.swipedItems` (simplified)
    - `summary.taste_vector` - sent separately
    """
)
async def save_onboarding_v3(
    request: FullOnboardingRequestV3,
    user: SupabaseUser = Depends(require_auth)
):
    """Save V3 onboarding profile."""

    # User ID comes from JWT token
    user_id = user.id

    # Transform V3 frontend structure to internal profile
    profile = transform_frontend_to_profile_v3(request)

    # Check what we have
    has_taste_vector = profile.taste_vector is not None and len(profile.taste_vector) == 512
    has_attribute_prefs = bool(profile.preferred_fits or profile.preferred_sleeves or profile.preferred_lengths)
    has_type_prefs = bool(profile.top_types or profile.bottom_types or profile.dress_types or profile.outerwear_types)

    # Save via pipeline
    pipeline = get_pipeline()
    result = pipeline.save_onboarding_v3(profile, gender=request.gender)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))

    return OnboardingResponseV3(
        status=result.get("status", "success"),
        user_id=user_id,
        categories_selected=profile.categories,
        has_taste_vector=has_taste_vector,
        has_attribute_preferences=has_attribute_prefs,
        has_type_preferences=has_type_prefs
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
    2. Lifestyle filtering (styles_to_avoid, occasions) using FashionCLIP scores
    3. Candidate retrieval (taste_vector similarity OR trending for cold users)
    4. Soft preference scoring (fit, style, length matching)
    5. SASRec ranking (for warm users with 5+ interactions)
    6. Diversity constraints (max 8 per category)
    7. Exploration injection (10% for discovery)

    **User States:**
    - `cold_start`: No Tinder test → trending only
    - `tinder_complete`: Has taste_vector → personalized embedding similarity
    - `warm_user`: 5+ interactions → SASRec + embedding + preference

    **Pagination:**
    - First request: Don't send `cursor` or `session_id`
    - Subsequent requests: Send back the `cursor` and `session_id` from previous response
    - No duplicates within session guaranteed

    **Category Filters:**
    - `categories`: Broad categories (tops, bottoms, dresses, outerwear)
    - `article_types`: Specific types (jeans, t-shirts, sweaters, knitwear, sweatpants, etc.)

    **Color Filters:**
    - `exclude_colors`: Colors to avoid (hard filter)
    - `include_colors`: Colors to include - item must have at least one (hard filter)

    **Lifestyle Filters:**
    - `exclude_styles`: Coverage styles to avoid (deep-necklines, sheer, cutouts, backless, strapless)
    - `include_occasions`: Occasions to match (casual, office, evening, beach, active)

    **Pattern Filters:**
    - `include_patterns`: Preferred patterns (solid, stripes, floral, geometric, animal-print, plaid)
    - `exclude_patterns`: Patterns to avoid

    **Attribute Filters (soft scoring):**
    - `fit`: Fit preferences (slim, regular, relaxed, oversized)
    - `length`: Length preferences (cropped, standard, long)
    - `sleeves`: Sleeve preferences (short, long, sleeveless, 3/4)
    - `neckline`: Neckline preferences (crew, v-neck, scoop, turtleneck, mock)
    - `rise`: Rise preferences for bottoms (high, mid, low)

    **Price & Brand Filters:**
    - `min_price`, `max_price`: Price range filter
    - `include_brands`: Brands to include (hard filter - ONLY show these brands)
    - `exclude_brands`: Brands to avoid (hard filter)
    """
)
async def get_pipeline_feed(
    background_tasks: BackgroundTasks,
    user: SupabaseUser = Depends(require_auth),
    session_id: Optional[str] = Query(None, description="Session ID (returned in response, send back for pagination)"),
    gender: str = Query("female", description="Gender filter"),
    categories: Optional[str] = Query(None, description="Comma-separated broad category filter (tops, bottoms, dresses, outerwear)"),
    article_types: Optional[str] = Query(None, description="Comma-separated article type filter (e.g., 'jeans,t-shirts,tank tops')"),
    exclude_styles: Optional[str] = Query(None, description="Comma-separated styles to avoid (deep-necklines, sheer, cutouts, backless, strapless)"),
    include_occasions: Optional[str] = Query(None, description="Comma-separated occasions to include (casual, office, evening, beach, active)"),
    # NEW FILTERS
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price filter"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price filter"),
    exclude_brands: Optional[str] = Query(None, description="Comma-separated brands to exclude"),
    include_brands: Optional[str] = Query(None, description="Comma-separated brands to include (hard filter - ONLY these brands)"),
    preferred_brands: Optional[str] = Query(None, description="[DEPRECATED] Use include_brands instead"),
    exclude_colors: Optional[str] = Query(None, description="Comma-separated colors to exclude"),
    include_colors: Optional[str] = Query(None, description="Comma-separated colors to include (hard filter - item must have at least one)"),
    # Attribute filters (hard include/exclude)
    fit: Optional[str] = Query(None, description="[DEPRECATED] Use include_fit. Comma-separated fits: slim, regular, relaxed, oversized"),
    length: Optional[str] = Query(None, description="[DEPRECATED] Use include_length. Comma-separated lengths: cropped, standard, long"),
    sleeves: Optional[str] = Query(None, description="[DEPRECATED] Use include_sleeves. Comma-separated sleeves: short, long, sleeveless, 3/4"),
    neckline: Optional[str] = Query(None, description="[DEPRECATED] Use include_neckline. Comma-separated necklines: crew, v-neck, scoop, turtleneck, mock"),
    rise: Optional[str] = Query(None, description="[DEPRECATED] Use include_rise. Comma-separated rises: high, mid, low"),
    include_patterns: Optional[str] = Query(None, description="Comma-separated patterns to include (solid, stripes, floral, geometric, animal-print, plaid)"),
    exclude_patterns: Optional[str] = Query(None, description="Comma-separated patterns to exclude"),
    # NEW: Comprehensive attribute hard filters (include/exclude)
    include_formality: Optional[str] = Query(None, description="Comma-separated formality levels to include (Casual, Smart Casual, Semi-Formal, Formal)"),
    exclude_formality: Optional[str] = Query(None, description="Comma-separated formality levels to exclude"),
    include_seasons: Optional[str] = Query(None, description="Comma-separated seasons to include (Spring, Summer, Fall, Winter)"),
    exclude_seasons: Optional[str] = Query(None, description="Comma-separated seasons to exclude"),
    include_style_tags: Optional[str] = Query(None, description="Comma-separated styles to include (Classic, Trendy, Bold, Minimal, Street, Boho, Romantic)"),
    exclude_style_tags: Optional[str] = Query(None, description="Comma-separated styles to exclude"),
    include_color_family: Optional[str] = Query(None, description="Comma-separated color families to include (Neutrals, Blues, Browns, Greens, etc.)"),
    exclude_color_family: Optional[str] = Query(None, description="Comma-separated color families to exclude"),
    include_silhouette: Optional[str] = Query(None, description="Comma-separated silhouettes to include (Fitted, A-Line, Straight, Wide Leg, Skinny, etc.)"),
    exclude_silhouette: Optional[str] = Query(None, description="Comma-separated silhouettes to exclude"),
    include_fit: Optional[str] = Query(None, description="Comma-separated fits to include (slim, regular, relaxed, oversized)"),
    exclude_fit: Optional[str] = Query(None, description="Comma-separated fits to exclude"),
    include_length: Optional[str] = Query(None, description="Comma-separated lengths to include (cropped, standard, long, mini, midi, maxi)"),
    exclude_length: Optional[str] = Query(None, description="Comma-separated lengths to exclude"),
    include_sleeves: Optional[str] = Query(None, description="Comma-separated sleeves to include (short, long, sleeveless, 3/4)"),
    exclude_sleeves: Optional[str] = Query(None, description="Comma-separated sleeves to exclude"),
    include_neckline: Optional[str] = Query(None, description="Comma-separated necklines to include (crew, v-neck, scoop, turtleneck, mock)"),
    exclude_neckline: Optional[str] = Query(None, description="Comma-separated necklines to exclude"),
    include_rise: Optional[str] = Query(None, description="Comma-separated rises to include (high, mid, low)"),
    exclude_rise: Optional[str] = Query(None, description="Comma-separated rises to exclude"),
    include_coverage: Optional[str] = Query(None, description="Comma-separated coverage levels to include (Full, Moderate, Partial, Minimal)"),
    exclude_coverage: Optional[str] = Query(None, description="Comma-separated coverage levels to exclude"),
    include_materials: Optional[str] = Query(None, description="Comma-separated materials to include (cotton, linen, silk, etc.)"),
    exclude_materials_filter: Optional[str] = Query(None, alias="exclude_materials", description="Comma-separated materials to exclude"),
    exclude_occasions: Optional[str] = Query(None, description="Comma-separated occasions to exclude"),
    on_sale_only: bool = Query(False, description="Only show items on sale"),
    cursor: Optional[str] = Query(None, description="Cursor from previous response (for pagination)"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page")
):
    """Get personalized feed using the full recommendation pipeline with keyset pagination."""
    
    # User ID from JWT auth
    user_id = user.id

    pipeline = get_pipeline()

    # Parse categories
    cat_list = None
    if categories:
        cat_list = [c.strip() for c in categories.split(",")]

    # Parse article_types
    article_type_list = None
    if article_types:
        article_type_list = [a.strip() for a in article_types.split(",")]

    # Parse lifestyle filters
    exclude_styles_list = None
    if exclude_styles:
        exclude_styles_list = [s.strip() for s in exclude_styles.split(",")]

    include_occasions_list = None
    if include_occasions:
        include_occasions_list = [o.strip() for o in include_occasions.split(",")]

    # Parse new filters
    exclude_brands_list = None
    if exclude_brands:
        exclude_brands_list = [b.strip() for b in exclude_brands.split(",")]

    # include_brands is the hard filter, preferred_brands is deprecated alias
    include_brands_list = None
    if include_brands:
        include_brands_list = [b.strip() for b in include_brands.split(",")]
    elif preferred_brands:
        # Backward compat: use preferred_brands as include_brands
        include_brands_list = [b.strip() for b in preferred_brands.split(",")]

    exclude_colors_list = None
    if exclude_colors:
        exclude_colors_list = [c.strip() for c in exclude_colors.split(",")]

    include_colors_list = None
    if include_colors:
        include_colors_list = [c.strip() for c in include_colors.split(",")]

    # Parse attribute filters
    fit_list = None
    if fit:
        fit_list = [f.strip() for f in fit.split(",")]

    length_list = None
    if length:
        length_list = [l.strip() for l in length.split(",")]

    sleeves_list = None
    if sleeves:
        sleeves_list = [s.strip() for s in sleeves.split(",")]

    neckline_list = None
    if neckline:
        neckline_list = [n.strip() for n in neckline.split(",")]

    rise_list = None
    if rise:
        rise_list = [r.strip() for r in rise.split(",")]

    include_patterns_list = None
    if include_patterns:
        include_patterns_list = [p.strip() for p in include_patterns.split(",")]

    exclude_patterns_list = None
    if exclude_patterns:
        exclude_patterns_list = [p.strip() for p in exclude_patterns.split(",")]

    # Parse new attribute filters (comma-separated -> list)
    def _parse_csv(val: Optional[str]) -> Optional[list]:
        return [v.strip() for v in val.split(",")] if val else None

    include_formality_list = _parse_csv(include_formality)
    exclude_formality_list = _parse_csv(exclude_formality)
    include_seasons_list = _parse_csv(include_seasons)
    exclude_seasons_list = _parse_csv(exclude_seasons)
    include_style_tags_list = _parse_csv(include_style_tags)
    exclude_style_tags_list = _parse_csv(exclude_style_tags)
    include_color_family_list = _parse_csv(include_color_family)
    exclude_color_family_list = _parse_csv(exclude_color_family)
    include_silhouette_list = _parse_csv(include_silhouette)
    exclude_silhouette_list = _parse_csv(exclude_silhouette)
    include_fit_list = _parse_csv(include_fit)
    exclude_fit_list = _parse_csv(exclude_fit)
    include_length_list = _parse_csv(include_length)
    exclude_length_list = _parse_csv(exclude_length)
    include_sleeves_list = _parse_csv(include_sleeves)
    exclude_sleeves_list = _parse_csv(exclude_sleeves)
    include_neckline_list = _parse_csv(include_neckline)
    exclude_neckline_list = _parse_csv(exclude_neckline)
    include_rise_list = _parse_csv(include_rise)
    exclude_rise_list = _parse_csv(exclude_rise)
    include_coverage_list = _parse_csv(include_coverage)
    exclude_coverage_list = _parse_csv(exclude_coverage)
    include_materials_list = _parse_csv(include_materials)
    exclude_materials_filter_list = _parse_csv(exclude_materials_filter)
    exclude_occasions_list = _parse_csv(exclude_occasions)

    # Use keyset pagination internally for O(1) performance
    response = pipeline.get_feed_keyset(
        user_id=user_id,
        anon_id=None,  # No anonymous users - auth required
        session_id=session_id,
        gender=gender,
        categories=cat_list,
        article_types=article_type_list,
        exclude_styles=exclude_styles_list,
        include_occasions=include_occasions_list,
        min_price=min_price,
        max_price=max_price,
        exclude_brands=exclude_brands_list,
        preferred_brands=include_brands_list,  # include_brands is the hard filter
        exclude_colors=exclude_colors_list,
        include_colors=include_colors_list,
        include_patterns=include_patterns_list,
        exclude_patterns=exclude_patterns_list,
        # Legacy attribute filters (backward compat)
        fit=fit_list,
        length=length_list,
        sleeves=sleeves_list,
        neckline=neckline_list,
        rise=rise_list,
        cursor=cursor,
        page_size=page_size,
        on_sale_only=on_sale_only,
        # Context scoring inputs
        user_metadata=user.user_metadata,
        # NEW: Comprehensive attribute hard filters
        include_formality=include_formality_list,
        exclude_formality=exclude_formality_list,
        include_seasons=include_seasons_list,
        exclude_seasons=exclude_seasons_list,
        include_style_tags=include_style_tags_list,
        exclude_style_tags=exclude_style_tags_list,
        include_color_family=include_color_family_list,
        exclude_color_family=exclude_color_family_list,
        include_silhouette=include_silhouette_list,
        exclude_silhouette=exclude_silhouette_list,
        include_fit=include_fit_list,
        exclude_fit=exclude_fit_list,
        include_length=include_length_list,
        exclude_length=exclude_length_list,
        include_sleeves=include_sleeves_list,
        exclude_sleeves=exclude_sleeves_list,
        include_neckline=include_neckline_list,
        exclude_neckline=exclude_neckline_list,
        include_rise=include_rise_list,
        exclude_rise=exclude_rise_list,
        include_coverage=include_coverage_list,
        exclude_coverage=exclude_coverage_list,
        include_materials=include_materials_list,
        exclude_materials=exclude_materials_filter_list,
        exclude_occasions=exclude_occasions_list,
    )

    # Auto-persist seen_ids to Supabase in background (replaces manual /session/sync)
    shown_ids = [item["product_id"] for item in response.get("results", [])]
    if shown_ids:
        background_tasks.add_task(
            _bg_persist_seen_ids,
            user_id=user.id,
            session_id=response.get("session_id", session_id or ""),
            seen_ids=shown_ids,
        )

    return response


@v2_router.get(
    "/sale",
    summary="Picks on Sale - Personalized sale items",
    description="""
    Get personalized sale items (items where original_price > price).

    **Features:**
    - Calls the same pipeline as /feed with on_sale_only=true
    - All existing filters work (categories, occasions, colors, brands, patterns)
    - Sale items ordered by highest discount percentage first
    - Returns original_price, discount_percent, is_on_sale fields

    **Response includes:**
    - `original_price`: Original price before discount
    - `discount_percent`: Discount as integer percentage (e.g., 25 for 25% off)
    - `is_on_sale`: Always true for this endpoint
    """
)
async def get_sale_items(
    background_tasks: BackgroundTasks,
    user: SupabaseUser = Depends(require_auth),
    session_id: Optional[str] = Query(None, description="Session ID (returned in response, send back for pagination)"),
    gender: str = Query("female", description="Gender filter"),
    categories: Optional[str] = Query(None, description="Comma-separated broad category filter"),
    article_types: Optional[str] = Query(None, description="Comma-separated article type filter"),
    exclude_styles: Optional[str] = Query(None, description="Comma-separated styles to avoid"),
    include_occasions: Optional[str] = Query(None, description="Comma-separated occasions to include"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price filter"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price filter"),
    exclude_brands: Optional[str] = Query(None, description="Comma-separated brands to exclude"),
    include_brands: Optional[str] = Query(None, description="Comma-separated brands to include"),
    exclude_colors: Optional[str] = Query(None, description="Comma-separated colors to exclude"),
    include_colors: Optional[str] = Query(None, description="Comma-separated colors to include"),
    include_patterns: Optional[str] = Query(None, description="Comma-separated patterns to prefer"),
    exclude_patterns: Optional[str] = Query(None, description="Comma-separated patterns to avoid"),
    cursor: Optional[str] = Query(None, description="Cursor from previous response (for pagination)"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page")
):
    """Get personalized sale items."""
    
    user_id = user.id

    pipeline = get_pipeline()

    # Parse comma-separated filters
    cat_list = [c.strip() for c in categories.split(",")] if categories else None
    article_type_list = [a.strip() for a in article_types.split(",")] if article_types else None
    exclude_styles_list = [s.strip() for s in exclude_styles.split(",")] if exclude_styles else None
    include_occasions_list = [o.strip() for o in include_occasions.split(",")] if include_occasions else None
    exclude_brands_list = [b.strip() for b in exclude_brands.split(",")] if exclude_brands else None
    include_brands_list = [b.strip() for b in include_brands.split(",")] if include_brands else None
    exclude_colors_list = [c.strip() for c in exclude_colors.split(",")] if exclude_colors else None
    include_colors_list = [c.strip() for c in include_colors.split(",")] if include_colors else None
    include_patterns_list = [p.strip() for p in include_patterns.split(",")] if include_patterns else None
    exclude_patterns_list = [p.strip() for p in exclude_patterns.split(",")] if exclude_patterns else None

    response = pipeline.get_feed_keyset(
        user_id=user_id,
        anon_id=None,
        session_id=session_id,
        gender=gender,
        categories=cat_list,
        article_types=article_type_list,
        exclude_styles=exclude_styles_list,
        include_occasions=include_occasions_list,
        min_price=min_price,
        max_price=max_price,
        exclude_brands=exclude_brands_list,
        preferred_brands=include_brands_list,
        exclude_colors=exclude_colors_list,
        include_colors=include_colors_list,
        include_patterns=include_patterns_list,
        exclude_patterns=exclude_patterns_list,
        cursor=cursor,
        page_size=page_size,
        on_sale_only=True,
        user_metadata=user.user_metadata,
    )

    # Auto-persist seen_ids in background
    shown_ids = [item["product_id"] for item in response.get("results", [])]
    if shown_ids:
        background_tasks.add_task(
            _bg_persist_seen_ids,
            user_id=user.id,
            session_id=response.get("session_id", session_id or ""),
            seen_ids=shown_ids,
        )

    return response


@v2_router.get(
    "/new-arrivals",
    summary="Just In - Personalized new arrivals",
    description="""
    Get personalized new arrivals (items added in the last 7 days).

    **Features:**
    - Calls the same pipeline as /feed with new_arrivals_only=true
    - All existing filters work (categories, occasions, colors, brands, patterns)
    - Items ordered by most recent first
    - Returns is_new field (always true for this endpoint)

    **Response includes:**
    - `is_new`: Always true for this endpoint
    - Items from the last 7 days
    """
)
async def get_new_arrivals(
    background_tasks: BackgroundTasks,
    user: SupabaseUser = Depends(require_auth),
    session_id: Optional[str] = Query(None, description="Session ID (returned in response, send back for pagination)"),
    gender: str = Query("female", description="Gender filter"),
    categories: Optional[str] = Query(None, description="Comma-separated broad category filter"),
    article_types: Optional[str] = Query(None, description="Comma-separated article type filter"),
    exclude_styles: Optional[str] = Query(None, description="Comma-separated styles to avoid"),
    include_occasions: Optional[str] = Query(None, description="Comma-separated occasions to include"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price filter"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price filter"),
    exclude_brands: Optional[str] = Query(None, description="Comma-separated brands to exclude"),
    include_brands: Optional[str] = Query(None, description="Comma-separated brands to include"),
    exclude_colors: Optional[str] = Query(None, description="Comma-separated colors to exclude"),
    include_colors: Optional[str] = Query(None, description="Comma-separated colors to include"),
    include_patterns: Optional[str] = Query(None, description="Comma-separated patterns to prefer"),
    exclude_patterns: Optional[str] = Query(None, description="Comma-separated patterns to avoid"),
    cursor: Optional[str] = Query(None, description="Cursor from previous response (for pagination)"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page")
):
    """Get personalized new arrivals."""
    
    user_id = user.id

    pipeline = get_pipeline()

    # Parse comma-separated filters
    cat_list = [c.strip() for c in categories.split(",")] if categories else None
    article_type_list = [a.strip() for a in article_types.split(",")] if article_types else None
    exclude_styles_list = [s.strip() for s in exclude_styles.split(",")] if exclude_styles else None
    include_occasions_list = [o.strip() for o in include_occasions.split(",")] if include_occasions else None
    exclude_brands_list = [b.strip() for b in exclude_brands.split(",")] if exclude_brands else None
    include_brands_list = [b.strip() for b in include_brands.split(",")] if include_brands else None
    exclude_colors_list = [c.strip() for c in exclude_colors.split(",")] if exclude_colors else None
    include_colors_list = [c.strip() for c in include_colors.split(",")] if include_colors else None
    include_patterns_list = [p.strip() for p in include_patterns.split(",")] if include_patterns else None
    exclude_patterns_list = [p.strip() for p in exclude_patterns.split(",")] if exclude_patterns else None

    response = pipeline.get_feed_keyset(
        user_id=user_id,
        anon_id=None,
        session_id=session_id,
        gender=gender,
        categories=cat_list,
        article_types=article_type_list,
        exclude_styles=exclude_styles_list,
        include_occasions=include_occasions_list,
        min_price=min_price,
        max_price=max_price,
        exclude_brands=exclude_brands_list,
        preferred_brands=include_brands_list,
        exclude_colors=exclude_colors_list,
        include_colors=include_colors_list,
        include_patterns=include_patterns_list,
        exclude_patterns=exclude_patterns_list,
        cursor=cursor,
        page_size=page_size,
        new_arrivals_only=True,
        user_metadata=user.user_metadata,
    )

    # Auto-persist seen_ids in background
    shown_ids = [item["product_id"] for item in response.get("results", [])]
    if shown_ids:
        background_tasks.add_task(
            _bg_persist_seen_ids,
            user_id=user.id,
            session_id=response.get("session_id", session_id or ""),
            seen_ids=shown_ids,
        )

    return response


@v2_router.get(
    "/feed/endless",
    summary="[DEPRECATED] Use GET /feed instead",
    description="""
    **DEPRECATED** -- Use `GET /api/recs/v2/feed` instead, which provides the same
    functionality with keyset cursor pagination, 23+ filters, session scoring,
    and context-aware scoring (age + weather).

    This endpoint uses offset-based dedup which is O(n) per page.
    The `/feed` endpoint uses keyset cursors which are O(1).

    Kept for backward compatibility. Will be removed in a future version.
    """,
    deprecated=True,
)
async def get_endless_feed(
    user: SupabaseUser = Depends(require_auth),
    session_id: Optional[str] = Query(None, description="Session ID (auto-generated if not provided)"),
    gender: str = Query("female", description="Gender filter"),
    categories: Optional[str] = Query(None, description="Comma-separated category filter"),
    page: int = Query(0, ge=0, description="Page number (0-indexed)"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page")
):
    """Get endless scroll feed with session state tracking."""
    
    user_id = user.id

    pipeline = get_pipeline()

    # Parse categories
    cat_list = None
    if categories:
        cat_list = [c.strip() for c in categories.split(",")]

    response = pipeline.get_feed_endless(
        user_id=user_id,
        anon_id=None,  # No anonymous users - auth required
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
    user: SupabaseUser = Depends(require_auth),
    session_id: Optional[str] = Query(None, description="Session ID (auto-generated if not provided)"),
    gender: str = Query("female", description="Gender filter"),
    categories: Optional[str] = Query(None, description="Comma-separated broad category filter"),
    article_types: Optional[str] = Query(None, description="Comma-separated article type filter"),
    exclude_styles: Optional[str] = Query(None, description="Comma-separated styles to avoid"),
    include_occasions: Optional[str] = Query(None, description="Comma-separated occasions to include"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price filter"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price filter"),
    exclude_brands: Optional[str] = Query(None, description="Comma-separated brands to exclude"),
    include_brands: Optional[str] = Query(None, description="Comma-separated brands to include (hard filter)"),
    exclude_colors: Optional[str] = Query(None, description="Comma-separated colors to exclude"),
    include_colors: Optional[str] = Query(None, description="Comma-separated colors to include"),
    include_patterns: Optional[str] = Query(None, description="Comma-separated patterns to include"),
    exclude_patterns: Optional[str] = Query(None, description="Comma-separated patterns to exclude"),
    # Attribute filters (include/exclude)
    include_formality: Optional[str] = Query(None, description="Comma-separated formality levels to include"),
    exclude_formality: Optional[str] = Query(None, description="Comma-separated formality levels to exclude"),
    include_seasons: Optional[str] = Query(None, description="Comma-separated seasons to include"),
    exclude_seasons: Optional[str] = Query(None, description="Comma-separated seasons to exclude"),
    include_style_tags: Optional[str] = Query(None, description="Comma-separated styles to include"),
    exclude_style_tags: Optional[str] = Query(None, description="Comma-separated styles to exclude"),
    include_color_family: Optional[str] = Query(None, description="Comma-separated color families to include"),
    exclude_color_family: Optional[str] = Query(None, description="Comma-separated color families to exclude"),
    include_silhouette: Optional[str] = Query(None, description="Comma-separated silhouettes to include"),
    exclude_silhouette: Optional[str] = Query(None, description="Comma-separated silhouettes to exclude"),
    include_fit: Optional[str] = Query(None, description="Comma-separated fits to include"),
    exclude_fit: Optional[str] = Query(None, description="Comma-separated fits to exclude"),
    include_length: Optional[str] = Query(None, description="Comma-separated lengths to include"),
    exclude_length: Optional[str] = Query(None, description="Comma-separated lengths to exclude"),
    include_sleeves: Optional[str] = Query(None, description="Comma-separated sleeves to include"),
    exclude_sleeves: Optional[str] = Query(None, description="Comma-separated sleeves to exclude"),
    include_neckline: Optional[str] = Query(None, description="Comma-separated necklines to include"),
    exclude_neckline: Optional[str] = Query(None, description="Comma-separated necklines to exclude"),
    include_rise: Optional[str] = Query(None, description="Comma-separated rises to include"),
    exclude_rise: Optional[str] = Query(None, description="Comma-separated rises to exclude"),
    include_coverage: Optional[str] = Query(None, description="Comma-separated coverage levels to include"),
    exclude_coverage: Optional[str] = Query(None, description="Comma-separated coverage levels to exclude"),
    include_materials: Optional[str] = Query(None, description="Comma-separated materials to include"),
    exclude_materials_filter: Optional[str] = Query(None, alias="exclude_materials", description="Comma-separated materials to exclude"),
    exclude_occasions: Optional[str] = Query(None, description="Comma-separated occasions to exclude"),
    on_sale_only: bool = Query(False, description="Only show items on sale"),
    cursor: Optional[str] = Query(None, description="Opaque cursor from previous response (NULL for first page)"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page")
):
    """Get keyset cursor paginated feed with full filter support."""
    
    user_id = user.id

    pipeline = get_pipeline()

    # Parse all comma-separated filters
    def _parse_csv(val: Optional[str]) -> Optional[list]:
        return [v.strip() for v in val.split(",")] if val else None

    response = pipeline.get_feed_keyset(
        user_id=user_id,
        anon_id=None,
        session_id=session_id,
        gender=gender,
        categories=_parse_csv(categories),
        article_types=_parse_csv(article_types),
        exclude_styles=_parse_csv(exclude_styles),
        include_occasions=_parse_csv(include_occasions),
        min_price=min_price,
        max_price=max_price,
        exclude_brands=_parse_csv(exclude_brands),
        preferred_brands=_parse_csv(include_brands),
        exclude_colors=_parse_csv(exclude_colors),
        include_colors=_parse_csv(include_colors),
        include_patterns=_parse_csv(include_patterns),
        exclude_patterns=_parse_csv(exclude_patterns),
        cursor=cursor,
        page_size=page_size,
        on_sale_only=on_sale_only,
        user_metadata=user.user_metadata,
        # Comprehensive attribute hard filters
        include_formality=_parse_csv(include_formality),
        exclude_formality=_parse_csv(exclude_formality),
        include_seasons=_parse_csv(include_seasons),
        exclude_seasons=_parse_csv(exclude_seasons),
        include_style_tags=_parse_csv(include_style_tags),
        exclude_style_tags=_parse_csv(exclude_style_tags),
        include_color_family=_parse_csv(include_color_family),
        exclude_color_family=_parse_csv(exclude_color_family),
        include_silhouette=_parse_csv(include_silhouette),
        exclude_silhouette=_parse_csv(exclude_silhouette),
        include_fit=_parse_csv(include_fit),
        exclude_fit=_parse_csv(exclude_fit),
        include_length=_parse_csv(include_length),
        exclude_length=_parse_csv(exclude_length),
        include_sleeves=_parse_csv(include_sleeves),
        exclude_sleeves=_parse_csv(exclude_sleeves),
        include_neckline=_parse_csv(include_neckline),
        exclude_neckline=_parse_csv(exclude_neckline),
        include_rise=_parse_csv(include_rise),
        exclude_rise=_parse_csv(exclude_rise),
        include_coverage=_parse_csv(include_coverage),
        exclude_coverage=_parse_csv(exclude_coverage),
        include_materials=_parse_csv(include_materials),
        exclude_materials=_parse_csv(exclude_materials_filter),
        exclude_occasions=_parse_csv(exclude_occasions),
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

    **Response is instant** (~1ms) — session scoring updates in-memory,
    Supabase persistence happens in background (non-blocking).

    **Valid Actions:**
    - `click`: User tapped to view product details (signal: 0.5)
    - `hover`: User swiped through photo gallery (signal: 0.1)
    - `add_to_wishlist`: User saved/liked the item (signal: 2.0, strong positive)
    - `add_to_cart`: User added to cart (signal: 0.8)
    - `purchase`: User completed purchase (signal: 1.0)
    - `skip`: User explicitly dismissed (signal: -0.5)

    **Session Scoring:**
    Send `brand`, `item_type`, and `attributes` to enable multi-dimensional
    preference learning. The next feed request will reflect these preferences.

    Call this endpoint immediately when a user takes an action.
    """
)
async def record_action(
    request: RecordActionRequest,
    background_tasks: BackgroundTasks,
    user: SupabaseUser = Depends(require_auth),
):
    """Record a user interaction (instant response, background persistence)."""

    # Validate action type
    if request.action not in VALID_INTERACTION_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{request.action}'. Must be one of: {VALID_INTERACTION_ACTIONS}"
        )

    # Step 1: Update session scores FIRST (instant, in-memory/Redis ~1ms)
    try:
        pipeline = get_pipeline()
        pipeline.update_session_scores_from_action(
            session_id=request.session_id,
            action=request.action,
            product_id=request.product_id,
            brand=request.brand or "",
            item_type=request.item_type or "",
            attributes=request.attributes or {},
            source=request.source or "feed",
        )
    except Exception:
        pass  # Session scoring failure is non-fatal

    # Step 2: Enqueue Supabase write as background task (non-blocking)
    background_tasks.add_task(
        _bg_persist_interaction,
        user_id=user.id,
        session_id=request.session_id,
        product_id=request.product_id,
        action=request.action,
        source=request.source or "feed",
        position=request.position,
    )

    # Step 3: Return immediately (~1ms total)
    return RecordActionResponse(status="success", interaction_id=None)


@v2_router.post(
    "/session/sync",
    response_model=SyncSessionResponse,
    summary="[DEPRECATED] Sync session seen_ids",
    description="""
    **DEPRECATED** -- Seen_ids are now auto-persisted by the server on each
    feed request via background tasks. This endpoint is no longer needed.

    Kept for backward compatibility. Will be removed in a future version.
    """,
    deprecated=True,
)
async def sync_session(
    request: SyncSessionRequest,
    user: SupabaseUser = Depends(require_auth)
):
    """Sync session seen_ids for training data."""

    if not request.seen_ids:
        return SyncSessionResponse(status="success", synced_count=0)

    service = get_service()

    try:
        result = service.sync_session_seen_ids(
            anon_id=None,
            user_id=user.id,
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
# Helper to integrate with FastAPI app
# =============================================================================

def integrate_with_app(app):
    """
    Integrate these endpoints with a FastAPI app.

    Usage:
        from recs.api_endpoints import integrate_with_app
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
