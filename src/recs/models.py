"""
Pydantic models for the recommendation pipeline.

Models cover:
- User onboarding preferences (10 modules)
- User state for recommendation context
- Candidate items with multi-score ranking
- API request/response schemas
"""

from enum import Enum
from typing import List, Optional, Set, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# =============================================================================
# Enums
# =============================================================================

class UserStateType(str, Enum):
    """User state for determining recommendation strategy."""
    COLD_START = "cold_start"           # No Tinder test, no interaction history
    TINDER_COMPLETE = "tinder_complete"  # Has taste_vector from Tinder, no history
    WARM_USER = "warm_user"              # Has 5+ interactions (SASRec eligible)


class BrandOpenness(str, Enum):
    """How open user is to discovering new brands."""
    STICK_TO_FAVORITES = "stick_to_favorites"
    MIX = "mix"
    MIX_FAVORITES_NEW = "mix-favorites-new"
    DISCOVER_NEW = "discover_new"


class Modesty(str, Enum):
    """User's modesty preference."""
    MODEST = "modest"
    BALANCED = "balanced"
    REVEALING = "revealing"


class StyleDirection(str, Enum):
    """Style direction preferences."""
    MINIMAL = "minimal"
    CLASSIC = "classic"
    TRENDY = "trendy"
    STATEMENT = "statement"


# =============================================================================
# Category Preferences (Modules 2-7: Soft Preferences)
# =============================================================================

class PriceRange(BaseModel):
    """Price range filter."""
    min_price: Optional[float] = None
    max_price: Optional[float] = None


class TopsPrefs(BaseModel):
    """Module 2: Tops preferences (soft scoring)."""
    types: List[str] = Field(default_factory=list)  # tee, blouse, sweater, etc.
    fits: List[str] = Field(default_factory=list)   # slim, regular, relaxed, oversized
    lengths: List[str] = Field(default_factory=list)  # cropped, regular, longline
    sleeves: List[str] = Field(default_factory=list)  # short-sleeve, long-sleeve, sleeveless
    necklines: List[str] = Field(default_factory=list)  # crew, v-neck, scoop, turtleneck
    price_comfort: Optional[float] = None  # Max price user is comfortable with
    enabled: bool = True


class BottomsPrefs(BaseModel):
    """Module 3: Bottoms preferences (soft scoring)."""
    types: List[str] = Field(default_factory=list)  # jeans, pants, shorts, leggings
    fits: List[str] = Field(default_factory=list)   # skinny, straight, wide-leg, relaxed
    rises: List[str] = Field(default_factory=list)  # low-rise, mid-rise, high-rise
    lengths: List[str] = Field(default_factory=list)  # cropped, ankle, full-length
    numeric_waist: Optional[int] = None  # Waist measurement
    numeric_hip: Optional[int] = None  # Hip measurement
    price_comfort: Optional[float] = None
    enabled: bool = True


class SkirtsPrefs(BaseModel):
    """Module 4: Skirts preferences (soft scoring)."""
    types: List[str] = Field(default_factory=list)  # a-line, midi, mini, maxi, pencil
    lengths: List[str] = Field(default_factory=list)  # mini, midi, maxi
    fits: List[str] = Field(default_factory=list)
    numeric_waist: Optional[int] = None
    price_comfort: Optional[float] = None
    enabled: bool = True


class DressesPrefs(BaseModel):
    """Module 5: Dresses preferences (soft scoring)."""
    types: List[str] = Field(default_factory=list)  # wrap, a-line, shift, bodycon
    fits: List[str] = Field(default_factory=list)  # fitted, regular, relaxed
    lengths: List[str] = Field(default_factory=list)  # mini, midi, maxi
    sleeves: List[str] = Field(default_factory=list)  # sleeveless, short, long
    price_comfort: Optional[float] = None
    enabled: bool = True


class OnePiecePrefs(BaseModel):
    """Module 6: One-piece preferences (jumpsuits, rompers, etc.)."""
    types: List[str] = Field(default_factory=list)  # jumpsuit, romper, overalls
    fits: List[str] = Field(default_factory=list)
    lengths: List[str] = Field(default_factory=list)  # short, regular, long
    numeric_waist: Optional[int] = None
    price_comfort: Optional[float] = None
    enabled: bool = True


class OuterwearPrefs(BaseModel):
    """Module 7: Outerwear preferences (soft scoring)."""
    types: List[str] = Field(default_factory=list)  # coat, puffer, blazer, jacket
    fits: List[str] = Field(default_factory=list)  # regular, oversized, fitted
    sleeves: List[str] = Field(default_factory=list)  # long-sleeve, 3/4-sleeve
    price_comfort: Optional[float] = None
    enabled: bool = True


# =============================================================================
# Style Discovery (Module 10 - Tinder-style swipe test)
# =============================================================================

class StyleDiscoverySelection(BaseModel):
    """A single selection from the style discovery (Tinder) test."""
    round: int
    category: Optional[str] = None
    winner_id: str
    loser_id: str
    timestamp: Optional[str] = None


class StyleDiscoverySummary(BaseModel):
    """Summary of learned preferences from style discovery test."""
    attribute_preferences: Dict[str, Any] = Field(default_factory=dict)
    taste_stability: Optional[float] = None
    taste_vector: Optional[List[float]] = None  # 512-dim FashionCLIP embedding


class StyleDiscoveryModule(BaseModel):
    """Module 10: Style Discovery (Tinder-style swipe test)."""
    user_id: Optional[str] = None
    selections: List[StyleDiscoverySelection] = Field(default_factory=list)
    rounds_completed: int = 0
    session_complete: bool = False
    summary: Optional[StyleDiscoverySummary] = None
    enabled: bool = True


# =============================================================================
# Onboarding Profile (All 10 Modules Combined)
# =============================================================================

class OnboardingProfile(BaseModel):
    """
    Complete user onboarding profile from 10 modules.

    Module 1: Core Setup (HARD filters - SQL WHERE)
    Modules 2-7: Per-category preferences (SOFT scoring)
    Module 8: Style preferences (SOFT scoring)
    Module 9: Brand preferences (MIXED - both hard and soft)
    Module 10: Style Discovery - Tinder test results (taste_vector)
    """
    user_id: str

    # -------------------------------------------------------------------------
    # Module 1: Core Setup (HARD FILTERS)
    # -------------------------------------------------------------------------
    categories: List[str] = Field(
        default_factory=list,
        description="Selected broad categories: tops, bottoms, one_piece, outerwear, dresses, skirts"
    )
    sizes: List[str] = Field(
        default_factory=list,
        description="User's sizes: XS, S, M, L, XL, XXL"
    )
    birthdate: Optional[str] = Field(
        default=None,
        description="User's birthdate (YYYY-MM-DD format)"
    )
    colors_to_avoid: List[str] = Field(
        default_factory=list,
        description="Colors to exclude from recommendations"
    )
    materials_to_avoid: List[str] = Field(
        default_factory=list,
        description="Materials to exclude (e.g., polyester, wool)"
    )

    # -------------------------------------------------------------------------
    # Modules 2-7: Per-Category Soft Preferences
    # -------------------------------------------------------------------------
    tops: Optional[TopsPrefs] = None
    bottoms: Optional[BottomsPrefs] = None
    skirts: Optional[SkirtsPrefs] = None
    dresses: Optional[DressesPrefs] = None
    one_piece: Optional[OnePiecePrefs] = None
    outerwear: Optional[OuterwearPrefs] = None

    # -------------------------------------------------------------------------
    # Module 8: Style Preferences (SOFT)
    # -------------------------------------------------------------------------
    style_directions: List[str] = Field(
        default_factory=list,
        description="Style directions: minimal, classic, trendy, statement"
    )
    modesty: Optional[str] = Field(
        default=None,
        description="Modesty preference: modest, balanced, revealing"
    )

    # -------------------------------------------------------------------------
    # Module 9: Brand Preferences (MIXED)
    # -------------------------------------------------------------------------
    preferred_brands: List[str] = Field(
        default_factory=list,
        description="Brands to boost in ranking (soft)"
    )
    brands_to_avoid: List[str] = Field(
        default_factory=list,
        description="Brands to exclude (hard filter)"
    )
    brand_openness: Optional[str] = Field(
        default=None,
        description="stick_to_favorites, mix, mix-favorites-new, discover_new"
    )

    # -------------------------------------------------------------------------
    # Module 10: Style Discovery (Tinder test results)
    # -------------------------------------------------------------------------
    style_discovery: Optional[StyleDiscoveryModule] = None

    # -------------------------------------------------------------------------
    # Global Price Range (optional override)
    # -------------------------------------------------------------------------
    global_min_price: Optional[float] = None
    global_max_price: Optional[float] = None

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    completed_at: Optional[str] = None


# =============================================================================
# User State (Recommendation Context)
# =============================================================================

class UserState(BaseModel):
    """
    User state for recommendation context.

    Combines:
    - Onboarding profile (9 modules)
    - Tinder test result (taste_vector)
    - Interaction history (for SASRec)
    - Session context (seen/disliked items)
    """
    user_id: str
    state_type: UserStateType = UserStateType.COLD_START

    # From Tinder swipe test
    taste_vector: Optional[List[float]] = Field(
        default=None,
        description="512-dim FashionCLIP embedding from Tinder test"
    )

    # From onboarding (9 modules)
    onboarding_profile: Optional[OnboardingProfile] = None

    # From interaction history (for SASRec - last 50 items)
    interaction_sequence: List[str] = Field(
        default_factory=list,
        description="Last 50 interacted item IDs (chronological)"
    )

    # Session context
    disliked_ids: Set[str] = Field(
        default_factory=set,
        description="Permanently disliked item IDs"
    )
    session_seen_ids: Set[str] = Field(
        default_factory=set,
        description="Items seen in current session"
    )

    class Config:
        # Allow Set type serialization
        json_encoders = {
            set: list
        }


# =============================================================================
# Candidate (Scored Item)
# =============================================================================

class Candidate(BaseModel):
    """
    Scored candidate item from the recommendation pipeline.

    Scores:
    - embedding_score: Cosine similarity with taste_vector (0-1)
    - preference_score: Soft preference match from onboarding (0-1)
    - sasrec_score: Sequential model score (normalized 0-1)
    - final_score: Weighted combination of above scores
    """
    item_id: str

    # Individual scores (normalized 0-1)
    embedding_score: float = 0.0
    preference_score: float = 0.0
    sasrec_score: float = 0.0

    # Final combined score
    final_score: float = 0.0

    # SASRec vocabulary status
    is_oov: bool = Field(
        default=False,
        description="True if item not in SASRec vocabulary (uses median score)"
    )

    # Item metadata (from Supabase) - all Optional to handle NULL from DB
    category: Optional[str] = ""
    broad_category: Optional[str] = ""
    article_type: Optional[str] = ""  # Specific article type (e.g., jeans, t-shirts)
    brand: Optional[str] = ""
    price: float = 0.0
    colors: List[str] = Field(default_factory=list)
    materials: List[str] = Field(default_factory=list)
    fit: Optional[str] = None
    length: Optional[str] = None
    sleeve: Optional[str] = None
    neckline: Optional[str] = None
    style_tags: List[str] = Field(default_factory=list)
    image_url: Optional[str] = ""
    gallery_images: List[str] = Field(default_factory=list)
    name: Optional[str] = ""

    # Source tracking
    source: str = Field(
        default="taste_vector",
        description="Candidate source: taste_vector, trending, exploration"
    )


# =============================================================================
# Hard Filters (for SQL query building)
# =============================================================================

class HardFilters(BaseModel):
    """
    Hard filters derived from OnboardingProfile for SQL WHERE clauses.
    Passed to Supabase pgvector RPC function.
    """
    gender: Optional[str] = None
    categories: Optional[List[str]] = None
    article_types: Optional[List[str]] = None  # Specific article types (e.g., jeans, t-shirts)
    exclude_colors: Optional[List[str]] = None
    exclude_materials: Optional[List[str]] = None
    exclude_brands: Optional[List[str]] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    exclude_product_ids: Optional[List[str]] = None

    @classmethod
    def from_user_state(cls, user_state: UserState, gender: str = "female") -> "HardFilters":
        """Build hard filters from user state."""
        profile = user_state.onboarding_profile

        # Combine disliked + session seen for exclusion
        exclude_ids = list(user_state.disliked_ids | user_state.session_seen_ids)

        if profile is None:
            return cls(
                gender=gender,
                exclude_product_ids=exclude_ids if exclude_ids else None
            )

        return cls(
            gender=gender,
            categories=profile.categories if profile.categories else None,
            exclude_colors=profile.colors_to_avoid if profile.colors_to_avoid else None,
            exclude_materials=profile.materials_to_avoid if profile.materials_to_avoid else None,
            exclude_brands=profile.brands_to_avoid if profile.brands_to_avoid else None,
            min_price=profile.global_min_price,
            max_price=profile.global_max_price,
            exclude_product_ids=exclude_ids if exclude_ids else None
        )


# =============================================================================
# API Request/Response Models
# =============================================================================

class SaveOnboardingRequest(BaseModel):
    """Request to save user onboarding profile."""
    user_id: Optional[str] = None
    anon_id: Optional[str] = None
    gender: str = "female"
    onboarding_profile: OnboardingProfile


class SaveOnboardingResponse(BaseModel):
    """Response after saving onboarding profile."""
    status: str = "success"
    user_id: str
    modules_saved: int
    categories_selected: List[str]


class FeedRequest(BaseModel):
    """Request for personalized feed."""
    user_id: Optional[str] = None
    anon_id: Optional[str] = None
    session_id: str
    gender: str = "female"
    categories: Optional[List[str]] = None
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class FeedItem(BaseModel):
    """Single item in feed response."""
    product_id: str
    rank: int
    score: float
    reason: str  # personalized, style_matched, trending, explore
    category: str
    brand: str
    name: str
    price: float
    image_url: str
    gallery_images: List[str] = Field(default_factory=list)
    colors: List[str] = Field(default_factory=list)


class FeedResponse(BaseModel):
    """Response for personalized feed."""
    user_id: str
    strategy: str  # sasrec, seed_vector, trending
    results: List[FeedItem]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    pagination: Dict[str, Any] = Field(default_factory=dict)


class UserStateResponse(BaseModel):
    """Debug endpoint: user state summary."""
    user_id: str
    state_type: str
    sequence_length: int
    last_5_interactions: List[str]
    has_taste_vector: bool
    has_onboarding: bool
    session_seen_count: int
    disliked_count: int
    recommendation_strategy: str
