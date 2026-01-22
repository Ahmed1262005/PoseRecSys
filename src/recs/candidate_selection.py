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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
    DressesPrefs,
    OuterwearPrefs,
)

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CandidateSelectionConfig:
    """Configuration for candidate selection."""
    # Candidate counts by source
    PRIMARY_CANDIDATES: int = 300      # From taste_vector pgvector search
    CONTEXTUAL_CANDIDATES: int = 100   # From trending within categories
    EXPLORATION_CANDIDATES: int = 50   # Random diverse for discovery

    # Soft preference scoring weights (OR logic - any match adds to score)
    SOFT_WEIGHTS: Dict[str, float] = None

    def __post_init__(self):
        if self.SOFT_WEIGHTS is None:
            self.SOFT_WEIGHTS = {
                'fit': 0.20,
                'style': 0.25,
                'length': 0.15,
                'neckline': 0.15,
                'sleeve': 0.10,
                'brand': 0.15,
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

        # Step 3: Apply soft preference scoring
        candidate_list = list(candidates.values())
        scored_candidates = self._apply_soft_scoring(
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
        article_types: Optional[List[str]] = None
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

        Returns:
            List of Candidate objects, sorted by score descending
        """
        # Step 1: Build hard filters from onboarding profile
        hard_filters = HardFilters.from_user_state(user_state, gender)

        # Step 1b: Add article_types filter from request (not stored in profile)
        if article_types:
            hard_filters.article_types = article_types

        # Step 2: Choose retrieval strategy based on user state
        # ALWAYS use keyset functions for proper cursor-based pagination
        # Exclusion is handled in Python after fetching (to preserve unlimited scroll)
        if user_state.taste_vector:
            # Warm user: use taste vector similarity with keyset cursor
            candidates = self._retrieve_by_taste_vector_keyset(
                taste_vector=user_state.taste_vector,
                hard_filters=hard_filters,
                cursor_score=cursor_score,
                cursor_id=cursor_id,
                limit=page_size
            )
        else:
            # Cold user: use trending with keyset cursor
            candidates = self._retrieve_trending_keyset(
                hard_filters=hard_filters,
                cursor_score=cursor_score,
                cursor_id=cursor_id,
                limit=page_size
            )

        # Step 2b: Filter out excluded items in Python (preserves keyset cursor pagination)
        if exclude_ids:
            candidates = [c for c in candidates if c.item_id not in exclude_ids]

        # Step 2c: Filter by article_types if specified (Python-level filter)
        if hard_filters.article_types:
            # Enrich candidates with article_type if not already present
            # (Workaround until SQL migration 016_add_article_type.sql is applied)
            candidates = self._enrich_with_article_types(candidates)
            article_types_lower = [at.lower() for at in hard_filters.article_types]
            candidates = [c for c in candidates if (c.article_type or '').lower() in article_types_lower]

        # Step 3: Apply soft preference scoring
        scored_candidates = self._apply_soft_scoring(
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
            }
            result = self.supabase.rpc('match_products_keyset', params).execute()
            return [self._row_to_candidate(row, source="taste_vector") for row in (result.data or [])]
        except Exception as e:
            error_str = str(e)
            if 'PGRST202' in error_str or 'could not find' in error_str.lower():
                print(f"[CandidateSelection] Keyset function not found, falling back to endless function")
                # Fall back to endless function
                return self._retrieve_by_taste_vector(taste_vector, hard_filters, limit, use_endless=True)
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
            }
            result = self.supabase.rpc('get_trending_keyset', params).execute()
            return [self._row_to_candidate(row, source="trending") for row in (result.data or [])]
        except Exception as e:
            error_str = str(e)
            if 'PGRST202' in error_str or 'could not find' in error_str.lower():
                print(f"[CandidateSelection] Trending keyset function not found, falling back to endless function")
                return self._retrieve_trending(hard_filters, limit, use_endless=True)
            else:
                print(f"[CandidateSelection] Error in trending keyset retrieval: {e}")
                return []

    def _retrieve_exploration_keyset(
        self,
        hard_filters: HardFilters,
        random_seed: str,
        cursor_score: Optional[float],
        cursor_id: Optional[str],
        limit: int
    ) -> List[Candidate]:
        """Retrieve exploration items with deterministic ordering."""

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
            }
            result = self.supabase.rpc('get_exploration_keyset', params).execute()
            return [self._row_to_candidate(row, source="exploration") for row in (result.data or [])]
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
    # Row to Candidate Conversion
    # =========================================================

    def _row_to_candidate(self, row: Dict[str, Any], source: str) -> Candidate:
        """Convert a database row to a Candidate object."""
        return Candidate(
            item_id=str(row.get('product_id', '')),
            embedding_score=float(row.get('similarity', 0.0)),
            preference_score=0.0,  # Will be set by soft scoring
            sasrec_score=0.0,      # Will be set by SASRecRanker
            final_score=0.0,
            is_oov=False,
            category=row.get('category', ''),
            broad_category=row.get('broad_category', ''),
            article_type=row.get('article_type', ''),
            brand=row.get('brand', ''),
            price=float(row.get('price', 0) or 0),
            colors=row.get('colors', []) or [],
            materials=row.get('materials', []) or [],
            fit=row.get('fit'),
            length=row.get('length'),
            sleeve=row.get('sleeve'),
            neckline=row.get('neckline'),
            style_tags=row.get('style_tags', []) or [],
            image_url=row.get('primary_image_url') or '',
            gallery_images=row.get('gallery_images', []) or [],
            name=row.get('name', ''),
            source=source
        )

    # =========================================================
    # Soft Preference Scoring
    # =========================================================

    def _apply_soft_scoring(
        self,
        candidates: List[Candidate],
        profile: Optional[OnboardingProfile]
    ) -> List[Candidate]:
        """
        Apply soft preference scoring to candidates.

        Uses OR logic: any match adds to the preference_score.
        Weights from config.SOFT_WEIGHTS.
        """
        if not profile:
            return candidates

        weights = self.config.SOFT_WEIGHTS

        for c in candidates:
            score = 0.0

            # Get category-specific preferences
            cat_prefs = self._get_category_prefs(profile, c.broad_category or c.category)

            if cat_prefs:
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

            # Style direction match (applies to all categories)
            if profile.style_directions and c.style_tags:
                if any(s.lower() in [t.lower() for t in c.style_tags] for s in profile.style_directions):
                    score += weights['style']

            # Brand preference match (soft boost)
            if profile.preferred_brands and c.brand:
                if c.brand in profile.preferred_brands:
                    score += weights['brand']

            c.preference_score = score

        return candidates

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
        - Onboarding profile (9 modules)
        - (Interaction history would be loaded separately for SASRec)
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

                # Build onboarding profile
                if row.get('has_onboarding'):
                    state.onboarding_profile = OnboardingProfile(
                        user_id=user_id or anon_id or "",
                        categories=row.get('onboarding_categories', []),
                        colors_to_avoid=row.get('colors_to_avoid', []),
                        materials_to_avoid=row.get('materials_to_avoid', []),
                        preferred_brands=row.get('preferred_brands', []),
                        brands_to_avoid=row.get('brands_to_avoid', []),
                        style_directions=row.get('style_directions', [])
                    )

                # Determine state type
                state_str = row.get('state_type', 'cold_start')
                if state_str == 'warm_user':
                    state.state_type = UserStateType.WARM_USER
                elif state_str == 'tinder_complete' or state.taste_vector:
                    state.state_type = UserStateType.TINDER_COMPLETE
                else:
                    state.state_type = UserStateType.COLD_START

        except Exception as e:
            print(f"[CandidateSelection] Error loading user state: {e}")

        return state

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
                'p_completed_at': profile.completed_at
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
