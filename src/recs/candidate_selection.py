"""
Candidate Selection Module

Multi-source candidate retrieval with hard filtering and soft preference scoring.

Pipeline:
1. Build hard filters from OnboardingProfile (SQL WHERE)
2. Multi-source retrieval:
   - taste_vector pgvector search (300 items)
   - trending products (100 items)
   - random exploration (50 items)
3. Merge & dedupe candidates
4. Apply soft preference scoring
5. Return scored Candidate objects for SASRec ranking
"""

import os
import sys
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass

from dotenv import load_dotenv
from supabase import create_client, Client

from recs.models import (
    UserState,
    UserStateType,
    OnboardingProfile,
    Candidate,
    HardFilters,
    TopsPrefs,
    BottomsPrefs,
    SkirtsPrefs,
    DressesPrefs,
    OnePiecePrefs,
    OuterwearPrefs,
)
from recs.feasibility_filter import canonicalize_article_type, canonicalize_name
from core.utils import filter_gallery_images

load_dotenv()


# =============================================================================
# Type Mappings for Article Types
# =============================================================================

# Mapping from onboarding type selections to database article_type values
# Used to determine include_article_types filter based on user's module selections
TYPE_MAPPINGS = {
    # Tops mappings
    'tee': ['t-shirts', 'tees'],
    't-shirt': ['t-shirts', 'tees'],
    'blouse': ['blouses'],
    'sweater': ['sweaters', 'knitwear'],
    'cardigan': ['cardigans'],
    'cardigan-top': ['cardigans'],
    'tank': ['tank tops', 'camisoles'],
    'tank-top': ['tank tops', 'camisoles'],
    'camisole': ['camisoles', 'tank tops'],
    'crop-top': ['crop tops'],
    'bodysuit': ['bodysuits'],
    'hoodie': ['hoodies', 'sweatshirts'],
    'sweatshirt': ['sweatshirts', 'hoodies'],

    # Knitwear mappings (virtual category - sweaters/cardigans grouped together)
    'knitwear': ['sweaters', 'cardigans', 'knitwear', 'turtlenecks', 'pullovers'],
    'turtleneck': ['turtlenecks', 'turtle necks', 'sweaters'],
    'pullover': ['pullovers', 'sweaters'],

    # Bottoms mappings
    'jeans': ['jeans', 'denim'],
    'pants': ['pants', 'trousers'],
    'trousers': ['pants', 'trousers'],
    'shorts': ['shorts'],
    'leggings': ['leggings'],
    'sweatpants': ['sweatpants', 'joggers', 'track pants'],
    'joggers': ['joggers', 'sweatpants', 'track pants'],

    # Skirt mappings (V6: single "skirts" option maps to all skirt types)
    'skirts': ['skirts', 'a-line skirts', 'pencil skirts', 'mini skirts', 'midi skirts', 'maxi skirts', 'pleated skirts', 'wrap skirts'],
    # Legacy individual skirt mappings (backward compat)
    'a-line-skirt': ['skirts', 'a-line skirts'],
    'pencil-skirt': ['skirts', 'pencil skirts'],
    'mini-skirt': ['skirts', 'mini skirts'],
    'midi-skirt': ['skirts', 'midi skirts'],
    'maxi-skirt': ['skirts', 'maxi skirts'],
    'pleated-skirt': ['skirts', 'pleated skirts'],
    'wrap-skirt': ['skirts', 'wrap skirts'],

    # Dress mappings (V6: single "dresses" option maps to all dress types)
    'dresses': ['dresses', 'mini dresses', 'midi dresses', 'maxi dresses', 'wrap dresses', 'a-line dresses', 'bodycon dresses', 'shift dresses', 'slip dresses', 'shirt dresses'],
    # Legacy individual dress mappings (backward compat)
    'wrap-dress': ['dresses', 'wrap dresses'],
    'a-line-dress': ['dresses', 'a-line dresses'],
    'bodycon': ['dresses', 'bodycon dresses'],
    'shift': ['dresses', 'shift dresses'],
    'slip-dress': ['dresses', 'slip dresses'],
    'shirt-dress': ['dresses', 'shirt dresses'],
    'maxi-dress': ['maxi dresses', 'dresses'],
    'midi-dress': ['midi dresses', 'dresses'],
    'mini-dress': ['mini dresses', 'dresses'],

    # One-piece mappings
    'jumpsuit': ['jumpsuits'],
    'romper': ['rompers'],
    'overalls': ['overalls'],

    # Outerwear mappings
    'coats': ['coats'],
    'coat': ['coats'],
    'jackets': ['jackets'],
    'jacket': ['jackets'],
    'blazers': ['blazers'],
    'blazer': ['blazers'],
    'puffer': ['puffer jackets', 'puffers'],
    'trench': ['trench coats'],
}


def get_include_article_types(profile: OnboardingProfile) -> Optional[List[str]]:
    """
    Determine which article types to INCLUDE based on user's type preferences.

    Supports both:
    - V3 format: flat type preferences (top_types, bottom_types, dress_types, outerwear_types)
    - Legacy format: per-module preferences (profile.tops.types, profile.bottoms.types, etc.)

    Returns None if no specific types selected (include all).

    FALLBACK: If no type prefs are set but categories are selected, we use
    the categories to determine basic exclusions.
    """
    include_types = set()

    # ==========================================================================
    # V3 Format: Flat type preferences
    # ==========================================================================
    if profile.top_types:
        for type_key in profile.top_types:
            type_key_lower = type_key.lower().replace(' ', '-')
            if type_key_lower in TYPE_MAPPINGS:
                include_types.update(TYPE_MAPPINGS[type_key_lower])
            else:
                include_types.add(type_key)

    if profile.bottom_types:
        # V3: bottom_types now includes skirt types
        for type_key in profile.bottom_types:
            type_key_lower = type_key.lower().replace(' ', '-')
            if type_key_lower in TYPE_MAPPINGS:
                include_types.update(TYPE_MAPPINGS[type_key_lower])
            else:
                include_types.add(type_key)

    if profile.dress_types:
        # V3: dress_types now includes jumpsuit/romper
        for type_key in profile.dress_types:
            type_key_lower = type_key.lower().replace(' ', '-')
            if type_key_lower in TYPE_MAPPINGS:
                include_types.update(TYPE_MAPPINGS[type_key_lower])
            else:
                include_types.add(type_key)

    if profile.outerwear_types:
        for type_key in profile.outerwear_types:
            type_key_lower = type_key.lower().replace(' ', '-')
            if type_key_lower in TYPE_MAPPINGS:
                include_types.update(TYPE_MAPPINGS[type_key_lower])
            else:
                include_types.add(type_key)

    # ==========================================================================
    # Legacy Format: Per-module preferences (backward compat)
    # ==========================================================================
    if not include_types:
        # Tops
        if profile.tops and profile.tops.enabled and profile.tops.types:
            for type_key in profile.tops.types:
                type_key_lower = type_key.lower().replace(' ', '-')
                if type_key_lower in TYPE_MAPPINGS:
                    include_types.update(TYPE_MAPPINGS[type_key_lower])
                else:
                    include_types.add(type_key)

        # Bottoms
        if profile.bottoms and profile.bottoms.enabled and profile.bottoms.types:
            for type_key in profile.bottoms.types:
                type_key_lower = type_key.lower().replace(' ', '-')
                if type_key_lower in TYPE_MAPPINGS:
                    include_types.update(TYPE_MAPPINGS[type_key_lower])
                else:
                    include_types.add(type_key)

        # Skirts (only if enabled)
        if profile.skirts and profile.skirts.enabled and profile.skirts.types:
            for type_key in profile.skirts.types:
                type_key_lower = type_key.lower().replace(' ', '-')
                if type_key_lower in TYPE_MAPPINGS:
                    include_types.update(TYPE_MAPPINGS[type_key_lower])
                else:
                    include_types.add(type_key)
        elif profile.skirts and profile.skirts.enabled:
            include_types.add('skirts')

        # Dresses (only if enabled)
        if profile.dresses and profile.dresses.enabled and profile.dresses.types:
            for type_key in profile.dresses.types:
                type_key_lower = type_key.lower().replace(' ', '-')
                if type_key_lower in TYPE_MAPPINGS:
                    include_types.update(TYPE_MAPPINGS[type_key_lower])
                else:
                    include_types.add(type_key)
        elif profile.dresses and profile.dresses.enabled:
            include_types.add('dresses')

        # One-piece (only if enabled)
        if profile.one_piece and profile.one_piece.enabled and profile.one_piece.types:
            for type_key in profile.one_piece.types:
                type_key_lower = type_key.lower().replace(' ', '-')
                if type_key_lower in TYPE_MAPPINGS:
                    include_types.update(TYPE_MAPPINGS[type_key_lower])
                else:
                    include_types.add(type_key)
        elif profile.one_piece and profile.one_piece.enabled:
            include_types.update(['jumpsuits', 'rompers', 'overalls'])

        # Outerwear (only if enabled)
        if profile.outerwear and profile.outerwear.enabled and profile.outerwear.types:
            for type_key in profile.outerwear.types:
                type_key_lower = type_key.lower().replace(' ', '-')
                if type_key_lower in TYPE_MAPPINGS:
                    include_types.update(TYPE_MAPPINGS[type_key_lower])
                else:
                    include_types.add(type_key)
        elif profile.outerwear and profile.outerwear.enabled:
            include_types.update(['coats', 'jackets', 'blazers'])

    # ==========================================================================
    # FALLBACK: Use categories if no type preferences set
    # ==========================================================================
    if not include_types and profile.categories:
        CATEGORY_TO_ARTICLE_TYPES = {
            'tops': ['t-shirts', 'tees', 'blouses', 'sweaters', 'cardigans', 'tank tops',
                     'camisoles', 'crop tops', 'bodysuits', 'hoodies', 'sweatshirts', 'shirts'],
            'bottoms': ['jeans', 'denim', 'pants', 'trousers', 'shorts', 'leggings',
                        'skirts', 'a-line skirts', 'pencil skirts', 'mini skirts', 'midi skirts', 'maxi skirts'],
            'skirts': ['skirts', 'a-line skirts', 'pencil skirts', 'mini skirts', 'midi skirts', 'maxi skirts'],
            'dresses': ['dresses', 'mini dresses', 'midi dresses', 'maxi dresses', 'wrap dresses',
                        'bodycon dresses', 'shift dresses', 'a-line dresses', 'jumpsuits', 'rompers', 'overalls'],
            'one-piece': ['jumpsuits', 'rompers', 'overalls'],
            'outerwear': ['coats', 'jackets', 'blazers', 'puffer jackets', 'trench coats'],
        }

        selected_categories = set(c.lower() for c in profile.categories)

        for cat in selected_categories:
            if cat in CATEGORY_TO_ARTICLE_TYPES:
                include_types.update(CATEGORY_TO_ARTICLE_TYPES[cat])

        if include_types:
            print(f"[CandidateSelection] FALLBACK: Using categories {profile.categories} to derive article types: {len(include_types)} types")

    return list(include_types) if include_types else None


# =============================================================================
# Occasion Mapping (Frontend → Database)
# =============================================================================

# The frontend uses different occasion names than the product_attributes table.
# This mapping expands user-facing occasion names to database values.
OCCASION_MAP = {
    'office': ['Office', 'Work'],
    'casual': ['Everyday', 'Casual', 'Weekend', 'Lounging'],
    'evening': ['Date Night', 'Party', 'Evening Event'],
    'smart-casual': ['Brunch', 'Vacation'],
    'beach': ['Vacation', 'Beach'],
    'active': ['Workout'],
    # Direct mappings (DB values passed through)
    'everyday': ['Everyday'],
    'date night': ['Date Night'],
    'party': ['Party'],
    'brunch': ['Brunch'],
    'vacation': ['Vacation'],
    'workout': ['Workout'],
}


def expand_occasions(user_occasions: List[str]) -> List[str]:
    """
    Expand frontend occasion names to DB values.

    Args:
        user_occasions: List of occasion names from frontend/profile

    Returns:
        List of expanded occasion names matching product_attributes.occasions
    """
    if not user_occasions:
        return []

    expanded = set()
    for occ in user_occasions:
        occ_lower = occ.lower().strip()
        if occ_lower in OCCASION_MAP:
            expanded.update(OCCASION_MAP[occ_lower])
        else:
            # Pass through unknown occasion names as-is (title case)
            expanded.add(occ.title())
    return list(expanded)


# =============================================================================
# Configuration
# =============================================================================

# NOTE: For simple attribute boost/demote scoring (fit, sleeve, length) used in
# search results, see filter_utils.SoftScoringWeights. The weights below are for
# direct database attribute matching (occasions array, pattern field, etc.).

@dataclass
class CandidateSelectionConfig:
    """Configuration for candidate selection."""
    # Candidate counts by source
    PRIMARY_CANDIDATES: int = 300      # From taste_vector pgvector search
    CONTEXTUAL_CANDIDATES: int = 100   # From trending within categories
    EXPLORATION_CANDIDATES: int = 50   # Random diverse for discovery

    # Soft preference scoring weights (OR logic - any match adds to score)
    # These weights are for feed ranking with computed CLIP scores.
    # For simpler search result boosting, see filter_utils.SoftScoringWeights.
    SOFT_WEIGHTS: Dict[str, float] = None

    # Brand boost multipliers based on brand_openness setting
    BRAND_OPENNESS_MULTIPLIERS: Dict[str, float] = None

    def __post_init__(self):
        if self.SOFT_WEIGHTS is None:
            self.SOFT_WEIGHTS = {
                # =============================================================
                # Brand preferences (INCREASED for stronger brand loyalty)
                # =============================================================
                'brand_preferred': 0.80,  # Strong boost for preferred brands
                'brand_new': 0.15,        # Small boost for new brands (discovery)

                # =============================================================
                # Occasion scoring (direct array match)
                # =============================================================
                'occasion_match': 0.50,        # Strong - lifestyle fit is important
                'multi_occasion_bonus': 0.20,  # Reward versatile items

                # =============================================================
                # Pattern scoring (direct string match)
                # =============================================================
                'pattern_match': 0.30,         # Boost for preferred patterns (e.g., Solid, Stripes)
                'pattern_avoid_penalty': -0.20,  # Penalty for disliked patterns (e.g., Floral)

                # =============================================================
                # Attribute matches (direct field comparison)
                # =============================================================
                'type_match': 0.25,       # Boost for matching type preferences
                'fit': 0.15,
                'length': 0.10,
                'neckline': 0.10,
                'sleeve': 0.10,
                'rise': 0.10,
            }

        # Brand diversity cap - prevents single brand from dominating feed
        self.BRAND_DIVERSITY_CAP: float = 0.25  # Max 25% of results from any single brand

        if self.BRAND_OPENNESS_MULTIPLIERS is None:
            self.BRAND_OPENNESS_MULTIPLIERS = {
                'stick_to_favorites': 2.0,    # Strong brand loyalty
                'stick-to-favorites': 2.0,
                'mix': 1.2,                   # Boosted brand preference
                'mix-favorites-new': 1.0,     # Normal brand boost with discovery
                'discover_new': 0.7,          # Still respect some brand pref
                'discover-new': 0.7,
            }


# =============================================================================
# Candidate Selection Module
# =============================================================================

class CandidateSelectionModule:
    """
    Multi-source candidate retrieval with hard filtering.

    Retrieves candidates from:
    1. taste_vector similarity (pgvector ANN search)
    2. trending/popular products
    3. random exploration items

    Applies:
    - Hard filters at SQL level (colors_to_avoid, materials_to_avoid, etc.)
    - Soft preference scoring in Python (fit, style, length, etc.)
    """

    def __init__(self, config: Optional[CandidateSelectionConfig] = None):
        """Initialize with Supabase client."""
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")

        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

        self.supabase: Client = create_client(url, key)
        self.config = config or CandidateSelectionConfig()

        # Shared profile scorer for attribute-driven preference matching
        from scoring.profile_scorer import ProfileScorer
        self._profile_scorer = ProfileScorer()

    # =========================================================
    # Main Candidate Retrieval
    # =========================================================

    def get_candidates(
        self,
        user_state: UserState,
        gender: str = "female",
        exclude_ids: Optional[Set[str]] = None,
        use_endless: bool = False
    ) -> List[Candidate]:
        """
        Get candidates from all sources with hard filtering.

        Args:
            user_state: User's current state (onboarding, taste_vector, history)
            gender: Target gender for products
            exclude_ids: Set of item IDs to exclude (already seen in session)
            use_endless: Use endless scroll SQL functions (with exclusion-based pagination)

        Returns:
            List of Candidate objects with embedding_score and preference_score set
        """
        # Step 1: Build hard filters from onboarding profile
        hard_filters = HardFilters.from_user_state(user_state, gender)

        # Add session exclusions to hard filters
        if exclude_ids:
            existing_exclusions = set(hard_filters.exclude_product_ids or [])
            existing_exclusions.update(exclude_ids)
            hard_filters.exclude_product_ids = list(existing_exclusions)

        # Step 2: Multi-source retrieval
        candidates: Dict[str, Candidate] = {}

        # Source A: taste_vector similarity (if available)
        if user_state.taste_vector:
            taste_candidates = self._retrieve_by_taste_vector(
                taste_vector=user_state.taste_vector,
                hard_filters=hard_filters,
                limit=self.config.PRIMARY_CANDIDATES,
                use_endless=use_endless
            )
            for c in taste_candidates:
                candidates[c.item_id] = c

        # Source B: Trending within selected categories
        trending_candidates = self._retrieve_trending(
            hard_filters=hard_filters,
            limit=self.config.CONTEXTUAL_CANDIDATES,
            use_endless=use_endless
        )
        for c in trending_candidates:
            if c.item_id not in candidates:
                candidates[c.item_id] = c

        # Source C: Random exploration
        exploration_candidates = self._retrieve_exploration(
            hard_filters=hard_filters,
            limit=self.config.EXPLORATION_CANDIDATES,
            exclude_ids=set(candidates.keys()),
            use_endless=use_endless
        )
        for c in exploration_candidates:
            if c.item_id not in candidates:
                candidates[c.item_id] = c

        # Step 3: Apply soft preference scoring (ProfileScorer + legacy fallback)
        candidate_list = list(candidates.values())
        scored_candidates = self._apply_profile_scoring(
            candidate_list,
            user_state.onboarding_profile
        )

        return scored_candidates

    # =========================================================
    # Keyset Pagination (V2)
    # =========================================================

    def get_candidates_keyset(
        self,
        user_state: UserState,
        gender: str = "female",
        cursor_score: Optional[float] = None,
        cursor_id: Optional[str] = None,
        page_size: int = 50,
        exclude_ids: Optional[Set[str]] = None,
        article_types: Optional[List[str]] = None,
        # NEW: Sale/New arrivals filters
        on_sale_only: bool = False,
        new_arrivals_only: bool = False,
        new_arrivals_days: int = 7
    ) -> List[Candidate]:
        """
        Get candidates using keyset cursor for O(1) pagination.

        V2 improvement: Instead of tracking seen_ids (O(n) exclusion),
        we use a cursor based on (score, id) for constant-time pagination.

        Args:
            user_state: User's current state (onboarding, taste_vector, history)
            gender: Target gender for products
            cursor_score: Score from last item of previous page (None for first page)
            cursor_id: ID from last item of previous page (None for first page)
            page_size: Number of items to return
            exclude_ids: Set of product IDs to exclude (e.g., seen history)
            article_types: Optional list of specific article types to filter by (e.g., ['jeans', 't-shirts'])
            on_sale_only: If True, only return items on sale (original_price > price)
            new_arrivals_only: If True, only return items added in last N days
            new_arrivals_days: Number of days to consider for new arrivals (default 7)

        Returns:
            List of Candidate objects, sorted by score descending
        """
        # Step 1: Build hard filters from onboarding profile
        hard_filters = HardFilters.from_user_state(user_state, gender)

        # Step 1b: Compute include_article_types
        # Priority: API request > profile-computed
        if article_types:
            # API request specifies exact types to include
            hard_filters.include_article_types = article_types
            print(f"[CandidateSelection] Using API-specified article_types: {article_types}")
        elif user_state.onboarding_profile and not hard_filters.include_article_types:
            # Compute from profile if not already set
            profile = user_state.onboarding_profile

            # Debug: Log what modules are enabled and their types
            print(f"[CandidateSelection] Profile modules:")
            if profile.tops:
                print(f"  - tops: enabled={profile.tops.enabled}, types={profile.tops.types}")
            if profile.bottoms:
                print(f"  - bottoms: enabled={profile.bottoms.enabled}, types={profile.bottoms.types}")
            if profile.skirts:
                print(f"  - skirts: enabled={profile.skirts.enabled}, types={profile.skirts.types}")
            if profile.dresses:
                print(f"  - dresses: enabled={profile.dresses.enabled}, types={profile.dresses.types}")
            if profile.one_piece:
                print(f"  - one_piece: enabled={profile.one_piece.enabled}, types={profile.one_piece.types}")
            if profile.outerwear:
                print(f"  - outerwear: enabled={profile.outerwear.enabled}, types={profile.outerwear.types}")

            profile_types = get_include_article_types(user_state.onboarding_profile)
            print(f"[CandidateSelection] Computed include_article_types: {profile_types}")

            if profile_types:
                hard_filters.include_article_types = profile_types

        # Step 2: Choose retrieval strategy based on user state
        # ALWAYS use keyset functions for proper cursor-based pagination
        # Exclusion is handled in Python after fetching (to preserve unlimited scroll)

        # EXPERIMENT: Always use exploration, disable taste_vector similarity
        # if user_state.taste_vector:
        #     # Warm user: use taste vector similarity with keyset cursor
        #     candidates = self._retrieve_by_taste_vector_keyset(
        #         taste_vector=user_state.taste_vector,
        #         hard_filters=hard_filters,
        #         cursor_score=cursor_score,
        #         cursor_id=cursor_id,
        #         limit=page_size
        #     )
        # else:

        # Always use exploration (deterministic random) with keyset cursor
        # Use user_id as seed for consistent ordering across pages
        random_seed = user_state.user_id or user_state.anon_id or "default"

        # Get preferred brands from profile for brand-priority retrieval
        preferred_brands = None
        if user_state.onboarding_profile and user_state.onboarding_profile.preferred_brands:
            preferred_brands = user_state.onboarding_profile.preferred_brands

        candidates = self._retrieve_exploration_keyset(
            hard_filters=hard_filters,
            random_seed=random_seed,
            cursor_score=cursor_score,
            cursor_id=cursor_id,
            limit=page_size,
            preferred_brands=preferred_brands,
            on_sale_only=on_sale_only,
            new_arrivals_only=new_arrivals_only,
            new_arrivals_days=new_arrivals_days
        )

        # Step 2b: Filter out excluded items in Python (preserves keyset cursor pagination)
        if exclude_ids:
            candidates = [c for c in candidates if c.item_id not in exclude_ids]

        # Step 2c: Filter by include_article_types if specified (Python-level fallback)
        # Note: SQL functions now handle this, but keep as fallback until migration applied
        if hard_filters.include_article_types and hard_filters.article_types:
            # Old code path for backward compatibility
            candidates = self._enrich_with_article_types(candidates)
            article_types_lower = [at.lower() for at in hard_filters.article_types]
            candidates = [c for c in candidates if (c.article_type or '').lower() in article_types_lower]

        # Step 3: Apply soft preference scoring (ProfileScorer + legacy fallback)
        scored_candidates = self._apply_profile_scoring(
            candidates,
            user_state.onboarding_profile
        )

        return scored_candidates

    def _retrieve_by_taste_vector_keyset(
        self,
        taste_vector: List[float],
        hard_filters: HardFilters,
        cursor_score: Optional[float],
        cursor_id: Optional[str],
        limit: int
    ) -> List[Candidate]:
        """Retrieve candidates by pgvector similarity with keyset cursor."""

        # Convert vector to pgvector string format
        vector_str = f"[{','.join(map(str, taste_vector))}]"

        try:
            params = {
                'query_embedding': vector_str,
                'filter_gender': hard_filters.gender,
                'filter_categories': hard_filters.categories,
                'exclude_colors': hard_filters.exclude_colors,
                'exclude_materials': hard_filters.exclude_materials,
                'exclude_brands': hard_filters.exclude_brands,
                'min_price': hard_filters.min_price,
                'max_price': hard_filters.max_price,
                'cursor_score': cursor_score,
                'cursor_id': cursor_id,
                'p_limit': limit,
                # Lifestyle filters - occasions now filtered in Python via product_attributes
                'exclude_styles': hard_filters.exclude_styles,
                'include_occasions': None,  # Filtered in Python via product_attributes.occasions
                'include_article_types': None,  # Python handles article_type filtering via name matching
                'style_threshold': 0.25,  # Default for backward compat with SQL functions
                'occasion_threshold': 0.18,
                # Pattern filters - patterns now filtered in Python via product_attributes.pattern
                'include_patterns': None,  # Filtered in Python via product_attributes.pattern
                'exclude_patterns': None,
                'pattern_threshold': 0.30,
            }
            result = self.supabase.rpc('match_products_keyset', params).execute()
            candidates = [self._row_to_candidate(row, source="taste_vector") for row in (result.data or [])]
            return self._enrich_with_attributes(candidates)
        except Exception as e:
            error_str = str(e)
            if 'PGRST202' in error_str or 'could not find' in error_str.lower():
                print(f"[CandidateSelection] Keyset function not found, falling back to endless function")
                # Fall back to endless function
                candidates = self._retrieve_by_taste_vector(taste_vector, hard_filters, limit, use_endless=True)
                return self._enrich_with_attributes(candidates)
            else:
                print(f"[CandidateSelection] Error in taste_vector keyset retrieval: {e}")
                return []

    def _retrieve_trending_keyset(
        self,
        hard_filters: HardFilters,
        cursor_score: Optional[float],
        cursor_id: Optional[str],
        limit: int
    ) -> List[Candidate]:
        """Retrieve trending products with keyset cursor."""

        try:
            params = {
                'filter_gender': hard_filters.gender,
                'filter_categories': hard_filters.categories,
                'exclude_colors': hard_filters.exclude_colors,
                'exclude_materials': hard_filters.exclude_materials,
                'exclude_brands': hard_filters.exclude_brands,
                'min_price': hard_filters.min_price,
                'max_price': hard_filters.max_price,
                'cursor_score': cursor_score,
                'cursor_id': cursor_id,
                'p_limit': limit,
                # Lifestyle filters - occasions now filtered in Python via product_attributes
                'exclude_styles': hard_filters.exclude_styles,
                'include_occasions': None,  # Filtered in Python via product_attributes.occasions
                'include_article_types': None,  # Python handles article_type filtering via name matching
                'style_threshold': 0.25,  # Default for backward compat with SQL functions
                'occasion_threshold': 0.18,
                # Pattern filters - patterns now filtered in Python via product_attributes.pattern
                'include_patterns': None,  # Filtered in Python via product_attributes.pattern
                'exclude_patterns': None,
                'pattern_threshold': 0.30,
            }
            result = self.supabase.rpc('get_trending_keyset', params).execute()
            candidates = [self._row_to_candidate(row, source="trending") for row in (result.data or [])]
            return self._enrich_with_attributes(candidates)
        except Exception as e:
            error_str = str(e)
            if 'PGRST202' in error_str or 'could not find' in error_str.lower():
                print(f"[CandidateSelection] Trending keyset function not found, falling back to endless function")
                candidates = self._retrieve_trending(hard_filters, limit, use_endless=True)
                return self._enrich_with_attributes(candidates)
            else:
                print(f"[CandidateSelection] Error in trending keyset retrieval: {e}")
                return []

    def _retrieve_exploration_keyset(
        self,
        hard_filters: HardFilters,
        random_seed: str,
        cursor_score: Optional[float],
        cursor_id: Optional[str],
        limit: int,
        preferred_brands: Optional[List[str]] = None,
        # NEW: Sale/New arrivals filters
        on_sale_only: bool = False,
        new_arrivals_only: bool = False,
        new_arrivals_days: int = 7
    ) -> List[Candidate]:
        """Retrieve exploration items with deterministic ordering.

        If preferred_brands is provided, uses brand-priority function to
        boost preferred brands to the top of results.

        Args:
            hard_filters: Hard filters from user state
            random_seed: Seed for deterministic random ordering
            cursor_score: Score from last item for keyset pagination
            cursor_id: ID from last item for keyset pagination
            limit: Number of items to fetch
            preferred_brands: Brands to boost in results
            on_sale_only: If True, only return items on sale
            new_arrivals_only: If True, only return items added in last N days
            new_arrivals_days: Number of days to consider for new arrivals
        """

        try:
            params = {
                'filter_gender': hard_filters.gender,
                'filter_categories': hard_filters.categories,
                'exclude_colors': hard_filters.exclude_colors,
                'exclude_materials': hard_filters.exclude_materials,
                'exclude_brands': hard_filters.exclude_brands,
                'min_price': hard_filters.min_price,
                'max_price': hard_filters.max_price,
                'random_seed': random_seed,
                'cursor_score': cursor_score,
                'cursor_id': cursor_id,
                'p_limit': limit,
                # Lifestyle filters - occasions now filtered in Python via product_attributes
                'exclude_styles': hard_filters.exclude_styles,
                'include_occasions': None,  # Filtered in Python via product_attributes.occasions
                'include_article_types': None,  # Python handles article_type filtering via name matching
                'style_threshold': 0.25,  # Default for backward compat with SQL functions
                'occasion_threshold': 0.18,
                # Pattern filters - patterns now filtered in Python via product_attributes.pattern
                'include_patterns': None,  # Filtered in Python via product_attributes.pattern
                'exclude_patterns': None,
                'pattern_threshold': 0.30,
                # Sale/New arrivals filters
                'on_sale_only': on_sale_only,
                'new_arrivals_only': new_arrivals_only,
                'new_arrivals_days': new_arrivals_days,
            }

            # Debug log article type filtering
            if hard_filters.include_article_types:
                print(f"[CandidateSelection] SQL filter include_article_types: {hard_filters.include_article_types}")

            # Log sale/new filters
            if on_sale_only:
                print(f"[CandidateSelection] Sale filter enabled")
            if new_arrivals_only:
                print(f"[CandidateSelection] New arrivals filter enabled ({new_arrivals_days} days)")

            # Use brand-boosted function if preferred brands are provided
            # This gives +0.5 SQL-level boost to preferred brands, ensuring they appear in results
            if preferred_brands:
                print(f"[CandidateSelection] Preferred brands (SQL + Python boost): {preferred_brands}")
                params['preferred_brands'] = preferred_brands
                result = self.supabase.rpc('get_exploration_keyset_with_brands', params).execute()
            else:
                result = self.supabase.rpc('get_exploration_keyset', params).execute()

            candidates = [self._row_to_candidate(row, source="exploration") for row in (result.data or [])]

            # Enrich with product_attributes for direct attribute filtering
            return self._enrich_with_attributes(candidates)
        except Exception as e:
            print(f"[CandidateSelection] Error in exploration keyset retrieval: {e}")
            return []

    # =========================================================
    # Source A: Taste Vector Similarity
    # =========================================================

    def _retrieve_by_taste_vector(
        self,
        taste_vector: List[float],
        hard_filters: HardFilters,
        limit: int,
        use_endless: bool = False,
        offset: int = 0
    ) -> List[Candidate]:
        """Retrieve candidates by pgvector similarity search."""

        # Convert vector to pgvector string format
        vector_str = f"[{','.join(map(str, taste_vector))}]"

        # Try endless function first if requested
        if use_endless:
            try:
                params = {
                    'query_embedding': vector_str,
                    'filter_gender': hard_filters.gender,
                    'filter_categories': hard_filters.categories,
                    'exclude_colors': hard_filters.exclude_colors,
                    'exclude_materials': hard_filters.exclude_materials,
                    'exclude_brands': hard_filters.exclude_brands,
                    'min_price': hard_filters.min_price,
                    'max_price': hard_filters.max_price,
                    'exclude_product_ids': hard_filters.exclude_product_ids,
                    'p_offset': offset,
                    'p_limit': limit,
                }
                result = self.supabase.rpc('match_products_endless', params).execute()
                return [self._row_to_candidate(row, source="taste_vector") for row in (result.data or [])]
            except Exception as e:
                error_str = str(e)
                if 'PGRST202' in error_str or 'could not find' in error_str.lower():
                    print(f"[CandidateSelection] Endless function not found, falling back to standard function")
                else:
                    print(f"[CandidateSelection] Error in taste_vector endless retrieval: {e}")
                    return []

        # Standard function (or fallback)
        try:
            params = {
                'query_embedding': vector_str,
                'filter_gender': hard_filters.gender,
                'filter_categories': hard_filters.categories,
                'exclude_colors': hard_filters.exclude_colors,
                'exclude_materials': hard_filters.exclude_materials,
                'exclude_brands': hard_filters.exclude_brands,
                'min_price': hard_filters.min_price,
                'max_price': hard_filters.max_price,
                'exclude_product_ids': hard_filters.exclude_product_ids,
                'match_count': limit,
            }
            result = self.supabase.rpc('match_products_with_hard_filters', params).execute()
            return [self._row_to_candidate(row, source="taste_vector") for row in (result.data or [])]
        except Exception as e:
            print(f"[CandidateSelection] Error in taste_vector retrieval: {e}")
            return []

    # =========================================================
    # Source B: Trending Products
    # =========================================================

    def _retrieve_trending(
        self,
        hard_filters: HardFilters,
        limit: int,
        use_endless: bool = False,
        offset: int = 0
    ) -> List[Candidate]:
        """Retrieve trending products with hard filters."""

        # Try endless function first if requested
        if use_endless:
            try:
                params = {
                    'filter_gender': hard_filters.gender,
                    'filter_categories': hard_filters.categories,
                    'exclude_colors': hard_filters.exclude_colors,
                    'exclude_materials': hard_filters.exclude_materials,
                    'exclude_brands': hard_filters.exclude_brands,
                    'min_price': hard_filters.min_price,
                    'max_price': hard_filters.max_price,
                    'exclude_product_ids': hard_filters.exclude_product_ids,
                    'p_offset': offset,
                    'p_limit': limit,
                }
                result = self.supabase.rpc('get_trending_endless', params).execute()
                return [self._row_to_candidate(row, source="trending") for row in (result.data or [])]
            except Exception as e:
                error_str = str(e)
                if 'PGRST202' in error_str or 'could not find' in error_str.lower():
                    print(f"[CandidateSelection] Endless trending function not found, falling back to standard function")
                else:
                    print(f"[CandidateSelection] Error in trending endless retrieval: {e}")
                    return []

        # Standard function (or fallback)
        try:
            params = {
                'filter_gender': hard_filters.gender,
                'filter_categories': hard_filters.categories,
                'exclude_colors': hard_filters.exclude_colors,
                'exclude_materials': hard_filters.exclude_materials,
                'exclude_brands': hard_filters.exclude_brands,
                'min_price': hard_filters.min_price,
                'max_price': hard_filters.max_price,
                'exclude_product_ids': hard_filters.exclude_product_ids,
                'result_limit': limit,
            }
            result = self.supabase.rpc('get_trending_with_hard_filters', params).execute()
            return [self._row_to_candidate(row, source="trending") for row in (result.data or [])]
        except Exception as e:
            print(f"[CandidateSelection] Error in trending retrieval: {e}")
            return []

    # =========================================================
    # Source C: Random Exploration
    # =========================================================

    def _retrieve_exploration(
        self,
        hard_filters: HardFilters,
        limit: int,
        exclude_ids: Set[str],
        use_endless: bool = False
    ) -> List[Candidate]:
        """Retrieve random exploration items for discovery."""

        # Add already-retrieved items to exclusion list
        all_exclusions = list(exclude_ids)
        if hard_filters.exclude_product_ids:
            all_exclusions.extend(hard_filters.exclude_product_ids)

        # Try endless function first if requested
        if use_endless:
            try:
                params = {
                    'filter_gender': hard_filters.gender,
                    'filter_categories': hard_filters.categories,
                    'exclude_colors': hard_filters.exclude_colors,
                    'exclude_materials': hard_filters.exclude_materials,
                    'exclude_brands': hard_filters.exclude_brands,
                    'exclude_product_ids': all_exclusions if all_exclusions else None,
                    'p_limit': limit,
                }
                result = self.supabase.rpc('get_exploration_endless', params).execute()
                return [self._row_to_candidate(row, source="exploration") for row in (result.data or [])]
            except Exception as e:
                error_str = str(e)
                if 'PGRST202' in error_str or 'could not find' in error_str.lower():
                    print(f"[CandidateSelection] Endless exploration function not found, falling back to standard function")
                else:
                    print(f"[CandidateSelection] Error in exploration endless retrieval: {e}")
                    return []

        # Standard function (or fallback)
        try:
            params = {
                'filter_gender': hard_filters.gender,
                'filter_categories': hard_filters.categories,
                'exclude_colors': hard_filters.exclude_colors,
                'exclude_materials': hard_filters.exclude_materials,
                'exclude_brands': hard_filters.exclude_brands,
                'exclude_product_ids': all_exclusions if all_exclusions else None,
                'result_limit': limit,
            }
            result = self.supabase.rpc('get_random_exploration_items', params).execute()
            return [self._row_to_candidate(row, source="exploration") for row in (result.data or [])]
        except Exception as e:
            print(f"[CandidateSelection] Error in exploration retrieval: {e}")
            return []

    # =========================================================
    # Count Available Products (for has_more calculation)
    # =========================================================

    def count_available_products(
        self,
        hard_filters: HardFilters,
        exclude_ids: Optional[Set[str]] = None
    ) -> int:
        """Count total products available after filtering (for has_more)."""

        try:
            all_exclusions = list(exclude_ids) if exclude_ids else []
            if hard_filters.exclude_product_ids:
                all_exclusions.extend(hard_filters.exclude_product_ids)

            result = self.supabase.rpc('count_available_products', {
                'filter_gender': hard_filters.gender,
                'filter_categories': hard_filters.categories,
                'exclude_colors': hard_filters.exclude_colors,
                'exclude_materials': hard_filters.exclude_materials,
                'exclude_brands': hard_filters.exclude_brands,
                'exclude_product_ids': all_exclusions if all_exclusions else None,
            }).execute()

            return result.data if result.data else 0

        except Exception as e:
            print(f"[CandidateSelection] Error counting products: {e}")
            return 0

    # =========================================================
    # Article Type Enrichment (Workaround until SQL functions updated)
    # =========================================================

    def _enrich_with_article_types(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Fetch article_type for candidates that don't have it.

        This is a workaround for SQL functions that don't return article_type.
        Once SQL migration 016_add_article_type.sql is applied, this becomes a no-op.
        """
        # Check if we need to fetch article_types
        needs_enrichment = [c for c in candidates if not c.article_type]
        if not needs_enrichment:
            return candidates

        # Get product IDs that need enrichment
        product_ids = [c.item_id for c in needs_enrichment]

        # Batch fetch article_types from products table
        try:
            result = self.supabase.table('products') \
                .select('id, article_type') \
                .in_('id', product_ids) \
                .execute()

            # Build lookup map
            article_type_map = {str(row['id']): row.get('article_type', '') for row in (result.data or [])}

            # Enrich candidates
            for c in candidates:
                if not c.article_type and c.item_id in article_type_map:
                    c.article_type = article_type_map[c.item_id]

            return candidates
        except Exception as e:
            print(f"[CandidateSelection] Error enriching article_types: {e}")
            return candidates

    # =========================================================
    # Product Attributes Enrichment
    # =========================================================

    def _enrich_with_attributes(self, candidates: List[Candidate], require_attributes: bool = True) -> List[Candidate]:
        """
        Fetch product_attributes for candidates and enrich them.

        This replaces the CLIP-based computed scores with direct database fields:
        - occasions: text[] (e.g., ['Office', 'Everyday', 'Date Night'])
        - style_tags: text[] (e.g., ['Casual', 'Trendy', 'Chic'])
        - pattern: text (e.g., 'Solid', 'Floral', 'Striped')
        - formality: text (e.g., 'Casual', 'Smart Casual', 'Semi-Formal', 'Formal')
        - fit_type: text (e.g., 'Regular', 'Fitted', 'Slim', 'Relaxed', 'Loose')
        - color_family: text (e.g., 'Neutrals', 'Blues', 'Browns')
        - seasons: text[] (e.g., ['Spring', 'Summer', 'Fall', 'Winter'])
        - construction: jsonb (contains neckline, sleeve_type, length)

        Args:
            candidates: List of candidates to enrich
            require_attributes: If True, filter out candidates without product_attributes data
        """
        if not candidates:
            return candidates

        # Get all product IDs
        product_ids = [c.item_id for c in candidates]

        try:
            # Query product_attributes table directly
            result = self.supabase.table('product_attributes') \
                .select('sku_id, occasions, style_tags, pattern, formality, fit_type, color_family, seasons, construction, coverage_level, skin_exposure, coverage_details, model_body_type, model_size_estimate') \
                .in_('sku_id', product_ids) \
                .execute()

            # Build lookup dict
            attrs_by_id = {str(r['sku_id']): r for r in (result.data or [])}

            # Track enrichment stats
            enriched_count = 0
            enriched_candidates = []

            # Enrich candidates
            for c in candidates:
                attrs = attrs_by_id.get(c.item_id)
                if attrs:
                    enriched_count += 1
                    c.occasions = attrs.get('occasions') or []
                    c.style_tags = attrs.get('style_tags') or []
                    c.pattern = attrs.get('pattern')
                    c.formality = attrs.get('formality')
                    c.color_family = attrs.get('color_family')
                    c.seasons = attrs.get('seasons') or []

                    # Override fit from product_attributes if available
                    if attrs.get('fit_type'):
                        c.fit = attrs.get('fit_type')

                    # Extract from construction jsonb
                    construction = attrs.get('construction') or {}
                    if construction.get('neckline') and not c.neckline:
                        c.neckline = construction.get('neckline')
                    if construction.get('sleeve_type') and not c.sleeve:
                        c.sleeve = construction.get('sleeve_type')
                    if construction.get('length') and not c.length:
                        c.length = construction.get('length')

                    # Coverage & body type (from Gemini Vision)
                    c.coverage_level = attrs.get('coverage_level')
                    c.skin_exposure = attrs.get('skin_exposure')
                    c.coverage_details = attrs.get('coverage_details') or []
                    c.model_body_type = attrs.get('model_body_type')
                    c.model_size_estimate = attrs.get('model_size_estimate')

                    enriched_candidates.append(c)
                elif not require_attributes:
                    # Keep candidates without attributes if not required
                    enriched_candidates.append(c)

            filtered_count = len(candidates) - len(enriched_candidates)
            print(f"[CandidateSelection] Enriched {enriched_count}/{len(candidates)} candidates with product_attributes")
            if require_attributes and filtered_count > 0:
                print(f"[CandidateSelection] Filtered out {filtered_count} candidates without product_attributes")

            return enriched_candidates

        except Exception as e:
            print(f"[CandidateSelection] Error enriching with product_attributes: {e}")
            # On error, return original candidates if attributes not required, else empty
            return [] if require_attributes else candidates

    # =========================================================
    # Row to Candidate Conversion
    # =========================================================

    def _row_to_candidate(self, row: Dict[str, Any], source: str) -> Candidate:
        """Convert a database row to a Candidate object."""
        # Compute canonical type for FeasibilityFilter
        article_type_raw = row.get('article_type', '') or ''
        name_raw = row.get('name', '') or ''
        canonical_type = canonicalize_article_type(article_type_raw)
        if canonical_type == "unknown":
            canonical_type = canonicalize_name(name_raw)

        # Sale/New arrival fields
        original_price = row.get('original_price')
        is_on_sale = row.get('is_on_sale', False)
        discount_percent = row.get('discount_percent')
        is_new = row.get('is_new', False)

        return Candidate(
            item_id=str(row.get('product_id', '')),
            embedding_score=float(row.get('similarity', 0.0)),
            preference_score=0.0,  # Will be set by soft scoring
            sasrec_score=0.0,      # Will be set by SASRecRanker
            final_score=0.0,
            is_oov=False,
            category=row.get('category', ''),
            broad_category=row.get('broad_category', ''),
            article_type=article_type_raw,
            canonical_type=canonical_type,
            brand=row.get('brand', ''),
            price=float(row.get('price', 0) or 0),
            colors=row.get('colors', []) or [],
            materials=row.get('materials', []) or [],
            fit=row.get('fit'),
            length=row.get('length'),
            sleeve=row.get('sleeve'),
            neckline=row.get('neckline'),
            rise=row.get('rise'),
            style_tags=row.get('style_tags', []) or [],
            image_url=row.get('primary_image_url') or '',
            gallery_images=filter_gallery_images(row.get('gallery_images', []) or []),
            name=name_raw,
            source=source,
            # Sale/New arrival fields
            original_price=float(original_price) if original_price else None,
            is_on_sale=bool(is_on_sale),
            discount_percent=int(discount_percent) if discount_percent else None,
            is_new=bool(is_new),
            # product_attributes fields - will be populated by _enrich_with_attributes()
            occasions=[],
            pattern=None,
            formality=None,
            color_family=None,
            seasons=[],
        )

    # =========================================================
    # Profile Scoring (shared ProfileScorer)
    # =========================================================

    def _apply_profile_scoring(
        self,
        candidates: List[Candidate],
        profile: Optional[OnboardingProfile]
    ) -> List[Candidate]:
        """
        Apply attribute-driven profile scoring using the shared ProfileScorer.

        Uses direct Gemini attribute matching against the onboarding profile:
        brand (+0.25 preferred, +0.10 cluster-adjacent), style tags, formality,
        fit/sleeve/length/neckline/rise (category-aware), type preferences,
        pattern, occasion, color avoidance, price range, and coverage hard-kills.

        Falls back to the legacy _apply_soft_scoring for backward compatibility
        if ProfileScorer fails.
        """
        if not profile:
            return candidates

        for c in candidates:
            c.preference_score = self._profile_scorer.score_item(
                c.to_scoring_dict(), profile
            )

        return candidates

    # =========================================================
    # Soft Preference Scoring (Legacy - kept for reference)
    # =========================================================

    def _apply_soft_scoring(
        self,
        candidates: List[Candidate],
        profile: Optional[OnboardingProfile]
    ) -> List[Candidate]:
        """
        Apply soft preference scoring to candidates.

        Supports both:
        - V3 format: flat attribute preferences (preferred_fits, preferred_sleeves, etc.)
        - Legacy format: per-category preferences (profile.tops.fits, etc.)

        Uses OR logic: any match adds to the preference_score.
        Weights from config.SOFT_WEIGHTS.

        Key scoring factors:
        - Type match: Does item type match user's selected types? (+0.25)
        - Brand match: Is this a preferred brand? (+0.60, modified by brand_openness)
        - Occasion match: Does item match user's preferred occasions? (+0.30)
        - Fit match: Does fit match preferences? (+0.15)
        - Style match: Does item style match user's style directions? (+0.20)
        - Attribute matches: Length, sleeve, neckline, rise (+0.10 each)
        """
        if not profile:
            return candidates

        weights = self.config.SOFT_WEIGHTS

        # Get brand openness multiplier (default to 1.0 if not set)
        brand_multiplier = 1.0
        if profile.brand_openness:
            brand_multiplier = self.config.BRAND_OPENNESS_MULTIPLIERS.get(
                profile.brand_openness.lower(), 1.0
            )

        # Pre-compute preferred brands (case-insensitive)
        preferred_brands_lower = set(b.lower() for b in profile.preferred_brands) if profile.preferred_brands else set()

        # Check if we have V3 flat preferences
        has_v3_prefs = bool(
            profile.preferred_fits or profile.preferred_sleeves or
            profile.preferred_lengths or profile.preferred_rises
        )

        for c in candidates:
            score = 0.0
            item_category = (c.broad_category or c.category or '').lower()

            # =================================================================
            # V3 Format: Use flat attribute preferences with category mappings
            # =================================================================
            if has_v3_prefs:
                # Type match using V3 type lists
                type_match = self._check_type_match_v3(profile, c.article_type or c.category, item_category)
                if type_match:
                    score += weights['type_match']

                # Fit match with category mapping
                if profile.preferred_fits and c.fit:
                    fit_applies = self._attribute_applies_to_category(
                        c.fit, item_category, profile.preferred_fits, profile.fit_category_mapping, 'fitId'
                    )
                    if fit_applies:
                        score += weights['fit']

                # Sleeve match with category mapping
                if profile.preferred_sleeves and c.sleeve:
                    sleeve_applies = self._attribute_applies_to_category(
                        c.sleeve, item_category, profile.preferred_sleeves, profile.sleeve_category_mapping, 'sleeveId'
                    )
                    if sleeve_applies:
                        score += weights['sleeve']

                # Length match with category mapping
                if c.length:
                    # Check standard lengths (for tops/bottoms)
                    if profile.preferred_lengths:
                        length_applies = self._attribute_applies_to_category(
                            c.length, item_category, profile.preferred_lengths, profile.length_category_mapping, 'lengthId'
                        )
                        if length_applies:
                            score += weights['length']

                    # Check dress/skirt lengths (mini, midi, maxi)
                    if profile.preferred_lengths_dresses and item_category in ['dresses', 'skirts']:
                        length_applies = self._attribute_applies_to_category(
                            c.length, item_category, profile.preferred_lengths_dresses,
                            profile.length_dresses_category_mapping, 'lengthId'
                        )
                        if length_applies:
                            score += weights['length']

                # Rise match (bottoms only, no category mapping needed)
                if profile.preferred_rises and hasattr(c, 'rise') and c.rise:
                    if c.rise.lower() in [r.lower() for r in profile.preferred_rises]:
                        score += weights['rise']

            # =================================================================
            # Legacy Format: Use per-category preferences
            # =================================================================
            else:
                cat_prefs = self._get_category_prefs(profile, c.broad_category or c.category)

                if cat_prefs:
                    # Type match
                    type_match = self._check_type_match(cat_prefs, c.article_type or c.category)
                    if type_match:
                        score += weights['type_match']

                    # Fit match
                    if hasattr(cat_prefs, 'fits') and cat_prefs.fits and c.fit:
                        if c.fit.lower() in [f.lower() for f in cat_prefs.fits]:
                            score += weights['fit']

                    # Length match
                    if hasattr(cat_prefs, 'lengths') and cat_prefs.lengths and c.length:
                        if c.length.lower() in [l.lower() for l in cat_prefs.lengths]:
                            score += weights['length']

                    # Neckline match
                    if hasattr(cat_prefs, 'necklines') and cat_prefs.necklines and c.neckline:
                        if c.neckline.lower() in [n.lower() for n in cat_prefs.necklines]:
                            score += weights['neckline']

                    # Sleeve match
                    if hasattr(cat_prefs, 'sleeves') and cat_prefs.sleeves and c.sleeve:
                        if c.sleeve.lower() in [s.lower() for s in cat_prefs.sleeves]:
                            score += weights['sleeve']

                    # Rise match (for bottoms)
                    if hasattr(cat_prefs, 'rises') and cat_prefs.rises and hasattr(c, 'rise') and c.rise:
                        if c.rise.lower() in [r.lower() for r in cat_prefs.rises]:
                            score += weights['rise']

            # =================================================================
            # Occasion match - simple array intersection using product_attributes
            # =================================================================
            if profile.occasions and c.occasions:
                # Expand user occasions to match DB values
                user_occasions_expanded = expand_occasions(profile.occasions)
                user_occasions_set = set(o.lower() for o in user_occasions_expanded)
                item_occasions_set = set(o.lower() for o in c.occasions)

                # Check for intersection
                matched_occasions = user_occasions_set & item_occasions_set
                if matched_occasions:
                    # Apply occasion boost
                    score += weights['occasion_match']

                    # Multi-occasion bonus: extra boost for items that match multiple user occasions
                    if len(matched_occasions) >= 2:
                        bonus_factor = min(1.0, len(matched_occasions) / len(user_occasions_expanded))
                        score += weights['multi_occasion_bonus'] * bonus_factor

            # =================================================================
            # Pattern preference scoring - simple string match
            # =================================================================
            # Get user's pattern preferences (V3 or legacy)
            patterns_liked = profile.patterns_liked or profile.patterns_preferred or []
            patterns_avoided = profile.patterns_avoided or profile.patterns_to_avoid or []

            if c.pattern:
                item_pattern_lower = c.pattern.lower()

                # Boost for liked patterns - simple string comparison
                if patterns_liked:
                    patterns_liked_lower = [p.lower() for p in patterns_liked]
                    if item_pattern_lower in patterns_liked_lower:
                        score += weights['pattern_match']

                # Penalty for avoided patterns
                if patterns_avoided:
                    patterns_avoided_lower = [p.lower() for p in patterns_avoided]
                    if item_pattern_lower in patterns_avoided_lower:
                        score += weights['pattern_avoid_penalty']

            # =================================================================
            # Style tag matching using product_attributes.style_tags
            # =================================================================
            style_prefs = profile.style_persona if profile.style_persona else profile.style_directions
            if style_prefs and c.style_tags:
                style_prefs_lower = set(s.lower() for s in style_prefs)
                item_styles_lower = set(t.lower() for t in c.style_tags)

                # Check for intersection - any style match gets full boost
                if style_prefs_lower & item_styles_lower:
                    score += weights.get('style', 0.30)

            # =================================================================
            # Brand preference match
            # =================================================================
            if c.brand:
                brand_lower = c.brand.lower()
                if preferred_brands_lower and brand_lower in preferred_brands_lower:
                    brand_boost = weights['brand_preferred'] * brand_multiplier
                    score += brand_boost
                elif profile.brand_openness in ['discover_new', 'discover-new', 'mix-favorites-new']:
                    score += weights['brand_new']

            c.preference_score = score

        return candidates

    def _attribute_applies_to_category(
        self,
        item_value: str,
        item_category: str,
        preferences: List[str],
        category_mapping: List[Dict[str, Any]],
        mapping_key: str
    ) -> bool:
        """
        Check if an attribute preference applies to the given item/category.

        Uses category mapping to determine if the preference applies.
        If no mapping exists for the preference, assume it applies to all categories.

        Args:
            item_value: The item's attribute value (e.g., "regular" for fit)
            item_category: The item's category (e.g., "tops", "dresses")
            preferences: List of preferred values (e.g., ["regular", "relaxed"])
            category_mapping: List of {mappingKey, categories} dicts
            mapping_key: The key in the mapping (e.g., "fitId", "sleeveId")
        """
        item_value_lower = item_value.lower()
        item_category_lower = item_category.lower()

        # Check if item value is in preferences
        if item_value_lower not in [p.lower() for p in preferences]:
            return False

        # If no mapping, assume applies to all categories
        if not category_mapping:
            return True

        # Find the mapping for this preference value
        for mapping in category_mapping:
            if mapping.get(mapping_key, '').lower() == item_value_lower:
                categories = mapping.get('categories', [])
                # Check if item category is in the mapped categories
                if any(item_category_lower in cat.lower() or cat.lower() in item_category_lower
                       for cat in categories):
                    return True
                return False

        # No mapping found for this value - assume applies to all
        return True

    def _check_type_match_v3(self, profile: OnboardingProfile, article_type: str, category: str) -> bool:
        """
        Check if item's article_type matches user's V3 type preferences.

        V3 has flat type lists: top_types, bottom_types, dress_types, outerwear_types
        """
        if not article_type:
            return False

        article_type_lower = article_type.lower()

        # Determine which type list to check based on category
        types_list = None
        if 'top' in category or 'knit' in category or 'woven' in category:
            types_list = profile.top_types
        elif 'bottom' in category or 'trouser' in category or 'pant' in category or 'skirt' in category:
            types_list = profile.bottom_types
        elif 'dress' in category or 'jumpsuit' in category or 'romper' in category:
            types_list = profile.dress_types
        elif 'outer' in category or 'jacket' in category or 'coat' in category:
            types_list = profile.outerwear_types

        if not types_list:
            return True  # No type preferences = accept all

        # Check if article_type matches any of the user's selected types
        for pref_type in types_list:
            pref_type_lower = pref_type.lower().replace(' ', '-')

            # Direct match
            if pref_type_lower == article_type_lower or pref_type_lower.replace('-', ' ') == article_type_lower:
                return True

            # Check mappings
            if pref_type_lower in TYPE_MAPPINGS:
                if article_type_lower in TYPE_MAPPINGS[pref_type_lower]:
                    return True

            # Fuzzy match
            if pref_type_lower.replace('-', '') in article_type_lower.replace('-', '').replace(' ', ''):
                return True

        return False

    def _check_type_match(self, cat_prefs, article_type: str) -> bool:
        """
        Check if item's article_type matches user's selected types for the category.

        Maps onboarding type names to product article_types.
        """
        if not article_type:
            return False

        article_type_lower = article_type.lower()

        # Get the types list from category prefs
        types_list = None
        if hasattr(cat_prefs, 'types') and cat_prefs.types:
            types_list = cat_prefs.types

        if not types_list:
            # No type preferences = accept all
            return True

        # Create mapping from onboarding types to article_types
        type_mappings = {
            # Tops mappings
            'blouse': ['blouses', 'blouse'],
            'tee': ['t-shirts', 't-shirt', 'tees', 'tee'],
            't-shirt': ['t-shirts', 't-shirt', 'tees'],
            'sweater': ['sweaters', 'sweater', 'knitwear'],
            'cardigan': ['cardigans', 'cardigan'],
            'cardigan-top': ['cardigans', 'cardigan'],
            'tank': ['tank tops', 'tank top', 'camisoles'],
            'tank-top': ['tank tops', 'tank top', 'camisoles'],
            'crop-top': ['crop tops', 'crop top'],
            'bodysuit': ['bodysuits', 'bodysuit'],
            'hoodie': ['hoodies', 'hoodie', 'sweatshirts'],
            'sweatshirt': ['sweatshirts', 'sweatshirt', 'hoodies'],
            # Bottoms mappings
            'jeans': ['jeans', 'denim'],
            'pants': ['pants', 'trousers'],
            'trousers': ['pants', 'trousers'],
            'shorts': ['shorts'],
            'leggings': ['leggings'],
            # Skirt mappings
            'a-line-skirt': ['skirts', 'a-line skirt'],
            'pencil-skirt': ['skirts', 'pencil skirt'],
            'mini-skirt': ['skirts', 'mini skirt'],
            'midi-skirt': ['skirts', 'midi skirt'],
            'maxi-skirt': ['skirts', 'maxi skirt'],
            # Dress mappings
            'wrap-dress': ['dresses', 'wrap dress'],
            'a-line-dress': ['dresses', 'a-line dress'],
            'bodycon': ['dresses', 'bodycon dress'],
            'shift': ['dresses', 'shift dress'],
            'maxi': ['maxi dresses', 'maxi dress'],
            'midi': ['midi dresses', 'midi dress'],
            'mini': ['mini dresses', 'mini dress'],
            # One-piece mappings
            'jumpsuit': ['jumpsuits', 'jumpsuit'],
            'romper': ['rompers', 'romper'],
            'overalls': ['overalls'],
            # Outerwear mappings
            'coats': ['coats', 'coat'],
            'jackets': ['jackets', 'jacket'],
            'blazers': ['blazers', 'blazer'],
            'puffer': ['puffer jackets', 'puffer'],
            'trench': ['trench coats', 'trench'],
        }

        # Check if article_type matches any of the user's selected types
        for pref_type in types_list:
            pref_type_lower = pref_type.lower().replace(' ', '-')

            # Direct match
            if pref_type_lower == article_type_lower or pref_type_lower.replace('-', ' ') == article_type_lower:
                return True

            # Check mappings
            if pref_type_lower in type_mappings:
                if article_type_lower in type_mappings[pref_type_lower]:
                    return True

            # Fuzzy match - check if article_type contains the preference type
            if pref_type_lower.replace('-', '') in article_type_lower.replace('-', '').replace(' ', ''):
                return True

        return False

    def _get_category_prefs(self, profile: OnboardingProfile, category: str):
        """Get category-specific preferences from profile."""
        category_lower = category.lower() if category else ''

        # Map broad categories to profile fields
        if 'top' in category_lower or 'knit' in category_lower or 'woven' in category_lower:
            return profile.tops
        elif 'bottom' in category_lower or 'trouser' in category_lower or 'pant' in category_lower:
            return profile.bottoms
        elif 'skirt' in category_lower or 'skort' in category_lower:
            return profile.skirts
        elif 'dress' in category_lower:
            return profile.dresses
        elif 'one' in category_lower or 'jumpsuit' in category_lower or 'romper' in category_lower:
            return profile.one_piece
        elif 'outer' in category_lower or 'jacket' in category_lower or 'coat' in category_lower:
            return profile.outerwear

        return None

    # =========================================================
    # User State Loading
    # =========================================================

    def load_user_state(
        self,
        user_id: Optional[str] = None,
        anon_id: Optional[str] = None
    ) -> UserState:
        """
        Load complete user state from Supabase.

        Combines:
        - Tinder test results (taste_vector)
        - Full onboarding profile (10 modules including per-category preferences)
        - (Interaction history would be loaded separately for SASRec)

        This enables:
        - Cold start mitigation using onboarding preferences
        - Type-based filtering (topTypes, bottomTypes, etc.)
        - Brand preference boosting
        - Attribute-based soft scoring (fit, sleeve, rise, length)
        """
        state = UserState(
            user_id=user_id or anon_id or "",
            state_type=UserStateType.COLD_START
        )

        if not user_id and not anon_id:
            return state

        try:
            # Load user recommendation state from Supabase
            result = self.supabase.rpc('get_user_recommendation_state', {
                'p_user_id': user_id,
                'p_anon_id': anon_id
            }).execute()

            if result.data and len(result.data) > 0:
                row = result.data[0]

                # Set taste vector
                taste_vec = row.get('taste_vector')
                if taste_vec:
                    if isinstance(taste_vec, str):
                        taste_vec = [float(x) for x in taste_vec.strip('[]').split(',')]
                    if len(taste_vec) == 512:
                        state.taste_vector = taste_vec

                # Build full onboarding profile with per-category preferences
                if row.get('has_onboarding'):
                    state.onboarding_profile = self._build_full_onboarding_profile(
                        user_id or anon_id or "",
                        row
                    )

                # Determine state type
                # IMPORTANT: Even without taste_vector, if user has onboarding,
                # treat as TINDER_COMPLETE to use preferences for cold start mitigation
                state_str = row.get('state_type', 'cold_start')
                if state_str == 'warm_user':
                    state.state_type = UserStateType.WARM_USER
                elif state_str == 'tinder_complete' or state.taste_vector:
                    state.state_type = UserStateType.TINDER_COMPLETE
                elif row.get('has_onboarding'):
                    # User has onboarding but no taste_vector - use onboarding preferences
                    # This is critical for cold start mitigation
                    state.state_type = UserStateType.TINDER_COMPLETE
                else:
                    state.state_type = UserStateType.COLD_START

        except Exception as e:
            print(f"[CandidateSelection] Error loading user state: {e}")

        return state

    def _build_full_onboarding_profile(self, user_key: str, row: Dict[str, Any]) -> OnboardingProfile:
        """
        Build full OnboardingProfile from DB row.

        Supports both:
        - V3 format: flat attribute/type preferences
        - Legacy format: per-category module preferences

        This extracts all preferences saved during onboarding.
        """
        profile = OnboardingProfile(
            user_id=user_key,
            # Core setup
            categories=row.get('onboarding_categories', []) or [],
            birthdate=row.get('birthdate'),
            colors_to_avoid=row.get('colors_to_avoid', []) or [],
            materials_to_avoid=row.get('materials_to_avoid', []) or [],
            # Legacy single sizes (for backward compat)
            sizes=row.get('sizes', []) or [],
            # Brands
            preferred_brands=row.get('preferred_brands', []) or [],
            brands_to_avoid=row.get('brands_to_avoid', []) or [],
            brand_openness=row.get('brand_openness'),
            # Legacy style
            style_directions=row.get('style_directions', []) or [],
            modesty=row.get('modesty'),
            # Global price
            global_min_price=row.get('global_min_price'),
            global_max_price=row.get('global_max_price'),
        )

        # =================================================================
        # V3 Fields: Split sizes
        # =================================================================
        profile.top_sizes = row.get('top_sizes', []) or []
        profile.bottom_sizes = row.get('bottom_sizes', []) or []
        profile.outerwear_sizes = row.get('outerwear_sizes', []) or []

        # =================================================================
        # V3 Fields: Flat attribute preferences
        # =================================================================
        profile.preferred_fits = row.get('preferred_fits', []) or []
        profile.fit_category_mapping = row.get('fit_category_mapping', []) or []
        profile.preferred_sleeves = row.get('preferred_sleeves', []) or []
        profile.sleeve_category_mapping = row.get('sleeve_category_mapping', []) or []
        profile.preferred_lengths = row.get('preferred_lengths', []) or []
        profile.length_category_mapping = row.get('length_category_mapping', []) or []
        profile.preferred_lengths_dresses = row.get('preferred_lengths_dresses', []) or []
        profile.length_dresses_category_mapping = row.get('length_dresses_category_mapping', []) or []
        profile.preferred_rises = row.get('preferred_rises', []) or []

        # =================================================================
        # V3 Fields: Simplified type preferences
        # =================================================================
        profile.top_types = row.get('top_types', []) or []
        profile.bottom_types = row.get('bottom_types', []) or []
        profile.dress_types = row.get('dress_types', []) or []
        profile.outerwear_types = row.get('outerwear_types', []) or []

        # =================================================================
        # V3 Fields: Lifestyle & Style persona
        # =================================================================
        profile.occasions = row.get('occasions', []) or []
        profile.styles_to_avoid = row.get('styles_to_avoid', []) or []
        profile.patterns_liked = row.get('patterns_liked', []) or []
        profile.patterns_avoided = row.get('patterns_avoided', []) or []
        profile.style_persona = row.get('style_persona', []) or []

        # Legacy pattern fields (for backward compat)
        profile.patterns_to_avoid = row.get('patterns_to_avoid', []) or []
        profile.patterns_preferred = row.get('patterns_preferred', []) or []

        # =================================================================
        # V3 Fields: Simplified style discovery
        # =================================================================
        profile.style_discovery_complete = row.get('style_discovery_complete', False) or False
        profile.swiped_items = row.get('swiped_items', []) or []

        # =================================================================
        # Legacy: Per-category preferences from JSONB columns
        # =================================================================
        # Tops preferences
        tops_prefs = row.get('tops_prefs')
        if tops_prefs and isinstance(tops_prefs, dict):
            profile.tops = TopsPrefs(
                types=tops_prefs.get('types', []) or [],
                fits=tops_prefs.get('fits', []) or [],
                lengths=tops_prefs.get('lengths', []) or [],
                sleeves=tops_prefs.get('sleeves', []) or [],
                necklines=tops_prefs.get('necklines', []) or [],
                price_comfort=tops_prefs.get('price_comfort'),
                enabled=tops_prefs.get('enabled', True)
            )

        # Bottoms preferences
        bottoms_prefs = row.get('bottoms_prefs')
        if bottoms_prefs and isinstance(bottoms_prefs, dict):
            profile.bottoms = BottomsPrefs(
                types=bottoms_prefs.get('types', []) or [],
                fits=bottoms_prefs.get('fits', []) or [],
                rises=bottoms_prefs.get('rises', []) or [],
                lengths=bottoms_prefs.get('lengths', []) or [],
                numeric_waist=bottoms_prefs.get('numeric_waist'),
                numeric_hip=bottoms_prefs.get('numeric_hip'),
                price_comfort=bottoms_prefs.get('price_comfort'),
                enabled=bottoms_prefs.get('enabled', True)
            )

        # Dresses preferences
        dresses_prefs = row.get('dresses_prefs')
        if dresses_prefs and isinstance(dresses_prefs, dict):
            profile.dresses = DressesPrefs(
                types=dresses_prefs.get('types', []) or [],
                fits=dresses_prefs.get('fits', []) or [],
                lengths=dresses_prefs.get('lengths', []) or [],
                sleeves=dresses_prefs.get('sleeves', []) or [],
                price_comfort=dresses_prefs.get('price_comfort'),
                enabled=dresses_prefs.get('enabled', True)
            )

        # Outerwear preferences
        outerwear_prefs = row.get('outerwear_prefs')
        if outerwear_prefs and isinstance(outerwear_prefs, dict):
            profile.outerwear = OuterwearPrefs(
                types=outerwear_prefs.get('types', []) or [],
                fits=outerwear_prefs.get('fits', []) or [],
                sleeves=outerwear_prefs.get('sleeves', []) or [],
                price_comfort=outerwear_prefs.get('price_comfort'),
                enabled=outerwear_prefs.get('enabled', True)
            )

        # Skirts preferences
        skirts_prefs = row.get('skirts_prefs')
        if skirts_prefs and isinstance(skirts_prefs, dict):
            profile.skirts = SkirtsPrefs(
                types=skirts_prefs.get('types', []) or [],
                lengths=skirts_prefs.get('lengths', []) or [],
                fits=skirts_prefs.get('fits', []) or [],
                numeric_waist=skirts_prefs.get('numeric_waist'),
                price_comfort=skirts_prefs.get('price_comfort'),
                enabled=skirts_prefs.get('enabled', True)
            )

        # One-piece preferences
        one_piece_prefs = row.get('one_piece_prefs')
        if one_piece_prefs and isinstance(one_piece_prefs, dict):
            profile.one_piece = OnePiecePrefs(
                types=one_piece_prefs.get('types', []) or [],
                fits=one_piece_prefs.get('fits', []) or [],
                lengths=one_piece_prefs.get('lengths', []) or [],
                numeric_waist=one_piece_prefs.get('numeric_waist'),
                price_comfort=one_piece_prefs.get('price_comfort'),
                enabled=one_piece_prefs.get('enabled', True)
            )

        return profile

    # =========================================================
    # Onboarding Profile Save
    # =========================================================

    def save_onboarding_profile(self, profile: OnboardingProfile, gender: str = "female") -> Dict[str, Any]:
        """Save user's onboarding profile to Supabase."""

        try:
            # Determine user_id vs anon_id
            user_id = None
            anon_id = None

            # Check if user_id looks like a UUID
            if profile.user_id and len(profile.user_id) == 36 and '-' in profile.user_id:
                user_id = profile.user_id
            else:
                anon_id = profile.user_id

            # Build style discovery data
            style_discovery_data = None
            taste_vector = None
            if profile.style_discovery:
                style_discovery_data = profile.style_discovery.model_dump()
                # Extract taste vector if present
                if profile.style_discovery.summary and profile.style_discovery.summary.taste_vector:
                    taste_vector = profile.style_discovery.summary.taste_vector

            result = self.supabase.rpc('save_onboarding_profile_v2', {
                'p_user_id': user_id,
                'p_anon_id': anon_id,
                'p_gender': gender,
                'p_categories': profile.categories,
                'p_sizes': profile.sizes,
                'p_birthdate': profile.birthdate,
                'p_colors_to_avoid': profile.colors_to_avoid,
                'p_materials_to_avoid': profile.materials_to_avoid,
                'p_tops_prefs': profile.tops.model_dump() if profile.tops else {},
                'p_bottoms_prefs': profile.bottoms.model_dump() if profile.bottoms else {},
                'p_skirts_prefs': profile.skirts.model_dump() if profile.skirts else {},
                'p_dresses_prefs': profile.dresses.model_dump() if profile.dresses else {},
                'p_one_piece_prefs': profile.one_piece.model_dump() if profile.one_piece else {},
                'p_outerwear_prefs': profile.outerwear.model_dump() if profile.outerwear else {},
                'p_style_directions': profile.style_directions,
                'p_modesty': profile.modesty,
                'p_preferred_brands': profile.preferred_brands,
                'p_brands_to_avoid': profile.brands_to_avoid,
                'p_brand_openness': profile.brand_openness,
                'p_global_min_price': profile.global_min_price,
                'p_global_max_price': profile.global_max_price,
                'p_style_discovery': style_discovery_data,
                'p_taste_vector': taste_vector,
                'p_completed_at': profile.completed_at,
                # Lifestyle fields (NEW)
                'p_styles_to_avoid': profile.styles_to_avoid if profile.styles_to_avoid else [],
                'p_occasions': profile.occasions if profile.occasions else [],
                'p_patterns_to_avoid': profile.patterns_to_avoid if profile.patterns_to_avoid else [],
                'p_patterns_preferred': profile.patterns_preferred if profile.patterns_preferred else []
            }).execute()

            return {
                "status": "success",
                "profile_id": str(result.data) if result.data else None,
                "user_id": profile.user_id,
                "modules_saved": self._count_modules_saved(profile),
                "categories_selected": profile.categories,
                "has_taste_vector": taste_vector is not None
            }

        except Exception as e:
            print(f"[CandidateSelection] Error saving onboarding profile: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _count_modules_saved(self, profile: OnboardingProfile) -> int:
        """Count how many modules have data."""
        count = 0

        # Module 1: Core (always counts if categories selected)
        if profile.categories:
            count += 1

        # Modules 2-7: Per-category
        for cat_prefs in [profile.tops, profile.bottoms, profile.skirts,
                          profile.dresses, profile.one_piece, profile.outerwear]:
            if cat_prefs:
                count += 1

        # Module 8: Style
        if profile.style_directions or profile.modesty:
            count += 1

        # Module 9: Brands
        if profile.preferred_brands or profile.brands_to_avoid:
            count += 1

        return count

    def save_onboarding_profile_v3(self, profile: OnboardingProfile, gender: str = "female") -> Dict[str, Any]:
        """
        Save user's V3 onboarding profile to Supabase.

        V3 format uses:
        - Split sizes (top_sizes, bottom_sizes, outerwear_sizes)
        - Flat attribute preferences with category mappings
        - Simplified type preferences (top_types, bottom_types, dress_types, outerwear_types)
        - style_persona instead of style_directions
        - Simplified style discovery (style_discovery_complete, swiped_items)
        """
        try:
            # Determine user_id vs anon_id
            user_id = None
            anon_id = None

            if profile.user_id and len(profile.user_id) == 36 and '-' in profile.user_id:
                user_id = profile.user_id
            else:
                anon_id = profile.user_id

            result = self.supabase.rpc('save_onboarding_profile_v3', {
                'p_user_id': user_id,
                'p_anon_id': anon_id,
                'p_gender': gender,
                # Core setup
                'p_categories': profile.categories,
                'p_birthdate': profile.birthdate,
                'p_top_sizes': profile.top_sizes,
                'p_bottom_sizes': profile.bottom_sizes,
                'p_outerwear_sizes': profile.outerwear_sizes,
                'p_colors_to_avoid': profile.colors_to_avoid,
                'p_materials_to_avoid': profile.materials_to_avoid,
                # Attribute preferences
                'p_preferred_fits': profile.preferred_fits,
                'p_fit_category_mapping': profile.fit_category_mapping,
                'p_preferred_sleeves': profile.preferred_sleeves,
                'p_sleeve_category_mapping': profile.sleeve_category_mapping,
                'p_preferred_lengths': profile.preferred_lengths,
                'p_length_category_mapping': profile.length_category_mapping,
                'p_preferred_lengths_dresses': profile.preferred_lengths_dresses,
                'p_length_dresses_category_mapping': profile.length_dresses_category_mapping,
                'p_preferred_rises': profile.preferred_rises,
                # Type preferences
                'p_top_types': profile.top_types,
                'p_bottom_types': profile.bottom_types,
                'p_dress_types': profile.dress_types,
                'p_outerwear_types': profile.outerwear_types,
                # Lifestyle
                'p_occasions': profile.occasions,
                'p_styles_to_avoid': profile.styles_to_avoid,
                'p_patterns_liked': profile.patterns_liked,
                'p_patterns_avoided': profile.patterns_avoided,
                'p_style_persona': profile.style_persona,
                # Brands
                'p_preferred_brands': profile.preferred_brands,
                'p_brands_to_avoid': profile.brands_to_avoid,
                'p_brand_openness': profile.brand_openness,
                # Style discovery
                'p_style_discovery_complete': profile.style_discovery_complete,
                'p_swiped_items': profile.swiped_items,
                'p_taste_vector': profile.taste_vector,
                # Metadata
                'p_completed_at': profile.completed_at,
            }).execute()

            return {
                "status": "success",
                "profile_id": str(result.data) if result.data else None,
                "user_id": profile.user_id,
                "has_taste_vector": profile.taste_vector is not None and len(profile.taste_vector) == 512
            }

        except Exception as e:
            print(f"[CandidateSelection] Error saving V3 onboarding profile: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    # =========================================================
    # User Seen History (for permanent deduplication)
    # =========================================================

    def get_user_seen_history(
        self,
        anon_id: Optional[str],
        user_id: Optional[str],
        limit: int = 5000
    ) -> Set[str]:
        """
        Get all product IDs this user has been shown across all sessions.

        Enables permanent deduplication - items shown once will never
        be shown again to this user, even across page refreshes.

        Args:
            anon_id: Anonymous user identifier
            user_id: UUID user identifier
            limit: Max items to load (default 5000)

        Returns:
            Set of product UUIDs the user has seen
        """
        seen_ids: Set[str] = set()

        try:
            # Build query based on available identifier
            if user_id:
                result = self.supabase.table("session_seen_ids").select(
                    "seen_ids"
                ).eq("user_id", user_id).execute()
            elif anon_id:
                result = self.supabase.table("session_seen_ids").select(
                    "seen_ids"
                ).eq("anon_id", anon_id).execute()
            else:
                return seen_ids

            # Flatten all seen_ids arrays into a single set
            if result.data:
                for row in result.data:
                    if row.get("seen_ids"):
                        seen_ids.update(str(sid) for sid in row["seen_ids"])

            # Limit to prevent memory issues
            if len(seen_ids) > limit:
                seen_ids = set(list(seen_ids)[:limit])

            print(f"[CandidateSelection] Loaded {len(seen_ids)} seen items for user from DB")
            return seen_ids

        except Exception as e:
            print(f"[CandidateSelection] Error loading user seen history: {e}")
            return seen_ids


# =============================================================================
# Testing
# =============================================================================

def test_candidate_selection():
    """Test the candidate selection module."""
    print("=" * 70)
    print("Testing Candidate Selection Module")
    print("=" * 70)

    module = CandidateSelectionModule()

    # Test 1: Load user state (cold start)
    print("\n1. Loading user state (cold start)...")
    state = module.load_user_state(anon_id="test_cold_user")
    print(f"   State type: {state.state_type.value}")
    print(f"   Has taste vector: {state.taste_vector is not None}")
    print(f"   Has onboarding: {state.onboarding_profile is not None}")

    # Test 2: Get candidates (cold start - should only get trending + exploration)
    print("\n2. Getting candidates (cold start)...")
    candidates = module.get_candidates(state, gender="female")
    print(f"   Total candidates: {len(candidates)}")

    by_source = {}
    for c in candidates:
        by_source[c.source] = by_source.get(c.source, 0) + 1
    print(f"   By source: {by_source}")

    if candidates:
        print(f"   Sample candidate: {candidates[0].name[:40]}...")
        print(f"     - Category: {candidates[0].category}")
        print(f"     - Brand: {candidates[0].brand}")
        print(f"     - Price: ${candidates[0].price:.2f}")

    # Test 3: Save onboarding profile
    print("\n3. Saving test onboarding profile...")
    test_profile = OnboardingProfile(
        user_id="test_onboarding_user",
        categories=["tops", "dresses"],
        colors_to_avoid=["orange", "yellow"],
        materials_to_avoid=["polyester"],
        style_directions=["minimal", "classic"],
        preferred_brands=["Zara", "H&M"]
    )
    save_result = module.save_onboarding_profile(test_profile)
    print(f"   Save status: {save_result.get('status')}")
    print(f"   Modules saved: {save_result.get('modules_saved', 0)}")

    # Test 4: Load user state with onboarding
    print("\n4. Loading user state with onboarding...")
    state_with_onboard = module.load_user_state(anon_id="test_onboarding_user")
    print(f"   State type: {state_with_onboard.state_type.value}")
    print(f"   Has onboarding: {state_with_onboard.onboarding_profile is not None}")
    if state_with_onboard.onboarding_profile:
        print(f"   Categories: {state_with_onboard.onboarding_profile.categories}")
        print(f"   Colors to avoid: {state_with_onboard.onboarding_profile.colors_to_avoid}")

    # Test 5: Get candidates with filters
    print("\n5. Getting candidates with onboarding filters...")
    filtered_candidates = module.get_candidates(state_with_onboard, gender="female")
    print(f"   Total candidates: {len(filtered_candidates)}")

    # Check if filtering worked (shouldn't see orange/yellow/polyester items)
    filter_violations = 0
    for c in filtered_candidates:
        if any(color in ['orange', 'yellow'] for color in c.colors):
            filter_violations += 1
        if 'polyester' in [m.lower() for m in c.materials]:
            filter_violations += 1
    print(f"   Filter violations: {filter_violations}")

    print("\n" + "=" * 70)
    print("Candidate Selection test complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_candidate_selection()
