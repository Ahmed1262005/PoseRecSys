"""
Recommendation Pipeline

Orchestrates the full recommendation flow:
1. Load user state (onboarding + Tinder + history)
2. Candidate selection with hard filtering
3. SASRec ranking with OOV handling
4. Diversity constraints (max 8 per category)
5. Exploration injection (10%)
6. Return paginated feed

User States:
- COLD_START: No Tinder, no history -> trending only
- TINDER_COMPLETE: Has taste_vector, no history -> embedding + preference
- WARM_USER: 5+ interactions -> sasrec + embedding + preference
"""

import os
import sys
import random
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from recs.models import (
    UserState,
    UserStateType,
    Candidate,
    OnboardingProfile,
    FeedRequest,
    FeedResponse,
    FeedItem,
    FeedSortBy,
    HardFilters,
)
from recs.candidate_selection import CandidateSelectionModule, CandidateSelectionConfig, expand_occasions
from recs.sasrec_ranker import SASRecRanker, SASRecRankerConfig
from recs.session_state import SessionStateService
from recs.feasibility_filter import filter_by_feasibility, FeasibilityFilter
from recs.filter_utils import (
    deduplicate_candidates,
    apply_diversity_candidates,
    get_diversity_limit,
    DiversityConfig,
)
from recs.session_scoring import (
    SessionScoringEngine,
    SessionScores,
    get_session_scoring_engine,
)
from recs.feed_reranker import (
    GreedyConstrainedReranker,
    get_feed_reranker,
    apply_sort_diversity,
)
from scoring import ContextScorer, ContextResolver, UserContext


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the recommendation pipeline."""

    # Diversity constraints
    MAX_PER_CATEGORY: int = 8         # Max items from any single category
    EXPLORATION_RATE: float = 0.10    # Fraction of results for exploration items

    # Feed defaults
    DEFAULT_LIMIT: int = 50
    MAX_LIMIT: int = 200

    # Candidate selection config
    candidate_config: CandidateSelectionConfig = field(default_factory=CandidateSelectionConfig)

    # SASRec ranker config
    ranker_config: SASRecRankerConfig = field(default_factory=SASRecRankerConfig)


# =============================================================================
# Recommendation Pipeline
# =============================================================================

class RecommendationPipeline:
    """
    Full recommendation pipeline.

    Flow:
    1. Load user state from Supabase
    2. Get candidates (taste_vector + trending + exploration)
    3. Rank with SASRec (warm users) or embedding/preference (cold users)
    4. Apply diversity constraints
    5. Inject exploration items
    6. Return paginated feed
    """

    def __init__(self, config: Optional[PipelineConfig] = None, load_sasrec: bool = True):
        """
        Initialize the recommendation pipeline.

        Args:
            config: Pipeline configuration
            load_sasrec: Whether to load the SASRec model
        """
        self.config = config or PipelineConfig()

        print("[RecommendationPipeline] Initializing...")

        # Initialize candidate selection module
        self.candidate_module = CandidateSelectionModule(self.config.candidate_config)
        print("[RecommendationPipeline] CandidateSelectionModule initialized")

        # Initialize SASRec ranker
        self.ranker = SASRecRanker(self.config.ranker_config, load_model=load_sasrec)
        print("[RecommendationPipeline] SASRecRanker initialized")

        # Initialize session state service for endless scroll
        self.session_service = SessionStateService(backend="auto")
        print("[RecommendationPipeline] SessionStateService initialized")

        # Initialize session scoring engine + feed reranker
        self.scoring_engine = get_session_scoring_engine()
        self.feed_reranker = get_feed_reranker()
        print("[RecommendationPipeline] SessionScoringEngine + FeedReranker initialized")

        # Initialize scoring backend (Redis if REDIS_URL set, else in-memory)
        # + L1 process-local cache for fast repeated reads
        self._scoring_backend = self._init_scoring_backend()
        self._scores_l1_cache: Dict[str, SessionScores] = {}
        self._SCORES_L1_MAX = 200
        print("[RecommendationPipeline] ScoringBackend initialized")

        # Initialize context-aware scoring (age affinity + weather/season)
        from config.settings import get_settings
        settings = get_settings()
        self.context_scorer = ContextScorer()
        self.context_resolver = ContextResolver(
            weather_api_key=settings.openweather_api_key,
        )
        print("[RecommendationPipeline] ContextScorer + ContextResolver initialized")

        print("[RecommendationPipeline] Ready")

    # =========================================================
    # Main Feed Generation
    # =========================================================

    def get_feed(
        self,
        user_id: Optional[str] = None,
        anon_id: Optional[str] = None,
        session_id: Optional[str] = None,
        gender: str = "female",
        categories: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> FeedResponse:
        """
        Generate personalized feed for a user.

        Args:
            user_id: Logged-in user UUID
            anon_id: Anonymous user ID
            session_id: Current session ID
            gender: Target gender
            categories: Optional category filter
            limit: Number of results (max 200)
            offset: Pagination offset

        Returns:
            FeedResponse with ranked items and metadata
        """
        # Validate limit
        limit = min(limit, self.config.MAX_LIMIT)

        # Step 1: Load user state
        user_state = self._load_user_state(user_id, anon_id, session_id)

        # Optional: Override categories from request
        if categories and user_state.onboarding_profile:
            user_state.onboarding_profile.categories = categories

        # Step 2: Get candidates
        candidates = self.candidate_module.get_candidates(user_state, gender)

        # Step 3: Rank candidates
        ranked_candidates = self.ranker.rank_candidates(user_state, candidates)

        # Step 4: Apply diversity constraints
        diverse_candidates = self._apply_diversity(ranked_candidates, user_state)

        # Step 4b: Deduplicate by image hash to remove same products across different brands
        diverse_candidates = self._deduplicate_by_image(diverse_candidates)

        # Step 5: Inject exploration items
        final_candidates = self._inject_exploration(diverse_candidates)

        # Step 6: Apply pagination
        total_available = len(final_candidates)
        paginated = final_candidates[offset:offset + limit]

        # Determine strategy used
        # EXPERIMENT: Always exploration (taste_vector disabled)
        # if user_state.state_type == UserStateType.WARM_USER:
        #     strategy = "sasrec"
        # elif user_state.state_type == UserStateType.TINDER_COMPLETE:
        #     strategy = "seed_vector"
        # else:
        #     strategy = "exploration"
        strategy = "session_aware" if session_scores.action_count > 0 else "exploration"

        # Convert to response
        results = []
        for rank, candidate in enumerate(paginated, start=offset + 1):
            results.append(FeedItem(
                product_id=candidate.item_id,
                rank=rank,
                score=candidate.final_score,
                reason=self._get_reason(candidate, user_state),
                category=candidate.category,
                brand=candidate.brand,
                name=candidate.name,
                price=candidate.price,
                image_url=candidate.image_url,
                colors=candidate.colors
            ))

        # Build metadata
        by_source = defaultdict(int)
        for c in final_candidates:
            by_source[c.source] += 1

        metadata = {
            "candidates_retrieved": len(candidates),
            "candidates_after_scoring": len(ranked_candidates),
            "sasrec_available": self.ranker.model is not None,
            "seed_vector_available": user_state.taste_vector is not None,
            "has_onboarding": user_state.onboarding_profile is not None,
            "user_state_type": user_state.state_type.value,
            "exploration_count": by_source.get("exploration", 0),
            "by_source": dict(by_source),
            "offset": offset,
            "has_more": offset + limit < total_available
        }

        return FeedResponse(
            user_id=user_id or anon_id or "anonymous",
            strategy=strategy,
            results=results,
            metadata=metadata,
            pagination={
                "limit": limit,
                "offset": offset,
                "total": total_available
            }
        )

    # =========================================================
    # Endless Scroll Feed Generation
    # =========================================================

    def get_feed_endless(
        self,
        user_id: Optional[str] = None,
        anon_id: Optional[str] = None,
        session_id: Optional[str] = None,
        gender: str = "female",
        categories: Optional[List[str]] = None,
        page: int = 0,
        page_size: int = 50,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate endless scroll feed with session state tracking.

        Unlike get_feed(), this method:
        1. Uses session state to track seen items
        2. Generates FRESH candidates per page (excludes seen items)
        3. Never shows duplicates within a session
        4. Returns session_id for client to persist

        Args:
            user_id: Logged-in user UUID
            anon_id: Anonymous user ID
            session_id: Session ID (auto-generated if not provided)
            gender: Target gender
            categories: Optional category filter
            page: Page number (0-indexed)
            page_size: Items per page (default 50)

        Returns:
            Dict with results, pagination, and session_id
        """
        # Generate session_id if not provided (new session)
        if not session_id:
            session_id = SessionStateService.generate_session_id()
            print(f"[Pipeline] New session created: {session_id}")

        # Validate page_size
        page_size = min(page_size, self.config.MAX_LIMIT)

        # Step 1: Load user state
        user_state = self._load_user_state(user_id, anon_id, session_id)

        # Override categories from request (even for cold-start users)
        if categories:
            if user_state.onboarding_profile:
                user_state.onboarding_profile.categories = categories
            else:
                # Create minimal onboarding profile with just categories
                from recs.models import OnboardingProfile
                user_state.onboarding_profile = OnboardingProfile(
                    user_id=user_state.user_id,
                    categories=categories
                )

        # Step 2: Load session state (seen items)
        seen_ids = self.session_service.get_seen_items(session_id)
        seen_count = len(seen_ids)

        # Step 3: Get FRESH candidates (excluding seen items)
        candidates = self.candidate_module.get_candidates(
            user_state,
            gender,
            exclude_ids=seen_ids,
            use_endless=True
        )

        # Step 4: Rank candidates
        ranked_candidates = self.ranker.rank_candidates(user_state, candidates)

        # Step 5: Apply diversity constraints
        diverse_candidates = self._apply_diversity(ranked_candidates, user_state)

        # Step 5b: Deduplicate by image hash to remove same products across different brands
        diverse_candidates = self._deduplicate_by_image(diverse_candidates)

        # Step 6: Inject exploration items
        final_candidates = self._inject_exploration(diverse_candidates)

        # Step 7: Take top N for this page
        page_results = final_candidates[:page_size]

        # Step 8: Update session state with newly shown items
        shown_ids = [c.item_id for c in page_results]
        if shown_ids:
            self.session_service.add_seen_items(session_id, shown_ids)

        # Step 9: Calculate has_more (approximate - based on whether we got full page)
        has_more = len(page_results) >= page_size and len(candidates) > page_size

        # Determine strategy used
        # EXPERIMENT: Always exploration (taste_vector disabled)
        # if user_state.state_type == UserStateType.WARM_USER:
        #     strategy = "sasrec"
        # elif user_state.state_type == UserStateType.TINDER_COMPLETE:
        #     strategy = "seed_vector"
        # else:
        #     strategy = "exploration"
        strategy = "exploration"

        # Convert to response format
        results = []
        for rank, candidate in enumerate(page_results, start=seen_count + 1):
            results.append({
                "product_id": candidate.item_id,
                "rank": rank,
                "score": candidate.final_score,
                "reason": self._get_reason(candidate, user_state),
                "category": candidate.category,
                "broad_category": candidate.broad_category,
                "brand": candidate.brand,
                "name": candidate.name,
                "price": candidate.price,
                "image_url": candidate.image_url,
                "gallery_images": candidate.gallery_images,
                "colors": candidate.colors,
                "source": candidate.source
            })

        # Build metadata
        by_source = defaultdict(int)
        for c in page_results:
            by_source[c.source] += 1

        resp = {
            "user_id": user_id or anon_id or "anonymous",
            "session_id": session_id,
            "strategy": strategy,
            "results": results,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "items_returned": len(results),
                "session_seen_count": seen_count + len(results),
                "has_more": has_more
            },
            "metadata": {
                "candidates_retrieved": len(candidates),
                "candidates_after_scoring": len(ranked_candidates),
                "sasrec_available": self.ranker.model is not None,
                "seed_vector_available": user_state.taste_vector is not None,
                "has_onboarding": user_state.onboarding_profile is not None,
                "user_state_type": user_state.state_type.value,
                "by_source": dict(by_source),
                "endless_scroll": True
            }
        }

        if not debug:
            resp.pop("metadata", None)

        return resp

    # =========================================================
    # Keyset Cursor Feed Generation (V2)
    # =========================================================

    def get_feed_keyset(
        self,
        user_id: Optional[str] = None,
        anon_id: Optional[str] = None,
        session_id: Optional[str] = None,
        gender: str = "female",
        categories: Optional[List[str]] = None,
        article_types: Optional[List[str]] = None,
        exclude_styles: Optional[List[str]] = None,
        include_occasions: Optional[List[str]] = None,
        # Price/Brand/Color filters
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        exclude_brands: Optional[List[str]] = None,
        preferred_brands: Optional[List[str]] = None,
        exclude_colors: Optional[List[str]] = None,
        include_colors: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        # Legacy attribute filters (kept for backward compat, now also used as hard filters)
        fit: Optional[List[str]] = None,
        length: Optional[List[str]] = None,
        sleeves: Optional[List[str]] = None,
        neckline: Optional[List[str]] = None,
        rise: Optional[List[str]] = None,
        cursor: Optional[str] = None,
        page_size: int = 50,
        # Sale/New arrivals filters
        on_sale_only: bool = False,
        new_arrivals_only: bool = False,
        new_arrivals_days: int = 7,
        # Context scoring inputs
        user_metadata: Optional[Dict[str, Any]] = None,
        # ============================================================
        # NEW: Comprehensive attribute hard filters (include/exclude)
        # ============================================================
        include_formality: Optional[List[str]] = None,
        exclude_formality: Optional[List[str]] = None,
        include_seasons: Optional[List[str]] = None,
        exclude_seasons: Optional[List[str]] = None,
        include_style_tags: Optional[List[str]] = None,
        exclude_style_tags: Optional[List[str]] = None,
        include_color_family: Optional[List[str]] = None,
        exclude_color_family: Optional[List[str]] = None,
        include_silhouette: Optional[List[str]] = None,
        exclude_silhouette: Optional[List[str]] = None,
        include_fit: Optional[List[str]] = None,
        exclude_fit: Optional[List[str]] = None,
        include_length: Optional[List[str]] = None,
        exclude_length: Optional[List[str]] = None,
        include_sleeves: Optional[List[str]] = None,
        exclude_sleeves: Optional[List[str]] = None,
        include_neckline: Optional[List[str]] = None,
        exclude_neckline: Optional[List[str]] = None,
        include_rise: Optional[List[str]] = None,
        exclude_rise: Optional[List[str]] = None,
        include_coverage: Optional[List[str]] = None,
        exclude_coverage: Optional[List[str]] = None,
        include_materials: Optional[List[str]] = None,
        exclude_materials: Optional[List[str]] = None,
        exclude_occasions: Optional[List[str]] = None,
        debug: bool = False,
        # ============================================================
        # Sort mode (relevance | price_asc | price_desc)
        # ============================================================
        sort_by: FeedSortBy = FeedSortBy.RELEVANCE,
    ) -> Dict[str, Any]:
        """
        Generate feed using keyset cursor for O(1) pagination.

        V2 Improvements over get_feed_endless():
        1. Uses keyset cursor (score, id) instead of exclusion arrays
        2. O(1) pagination regardless of page depth
        3. Feed versioning for stable ordering within session
        4. Still tracks seen items for authoritative dedup
        5. Lifestyle filtering (styles_to_avoid, occasions)

        When ``sort_by`` is not RELEVANCE, the full scoring/reranking pipeline
        is skipped and candidates are returned in deterministic price order
        with light brand diversity applied.

        Args:
            user_id: Logged-in user UUID
            anon_id: Anonymous user ID
            session_id: Session ID (auto-generated if not provided)
            gender: Target gender
            categories: Optional broad category filter (tops, bottoms, dresses, outerwear)
            article_types: Optional specific article type filter (jeans, t-shirts, mini dresses, etc.)
            exclude_styles: Optional list of coverage styles to exclude (deep-necklines, sheer, cutouts, backless, strapless)
            include_occasions: Optional list of occasions to include (casual, office, evening, beach)
            cursor: Base64 encoded keyset cursor (None for first page)
            page_size: Items per page (default 50)
            sort_by: Sort mode — RELEVANCE (default), PRICE_ASC, or PRICE_DESC

        Returns:
            Dict with results, cursor, pagination, and session_id
        """
        from recs.session_state import KeysetCursor
        import hashlib

        # ---------------------------------------------------------------
        # Sort-mode dispatch: non-relevance sorts use a dedicated fast path
        # that skips scoring/reranking to preserve deterministic order.
        # ---------------------------------------------------------------
        if sort_by != FeedSortBy.RELEVANCE:
            return self._get_feed_sorted(
                sort_by=sort_by,
                user_id=user_id,
                anon_id=anon_id,
                session_id=session_id,
                gender=gender,
                categories=categories,
                article_types=article_types,
                exclude_styles=exclude_styles,
                include_occasions=include_occasions,
                min_price=min_price,
                max_price=max_price,
                exclude_brands=exclude_brands,
                preferred_brands=preferred_brands,
                exclude_colors=exclude_colors,
                include_colors=include_colors,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                fit=fit,
                length=length,
                sleeves=sleeves,
                neckline=neckline,
                rise=rise,
                cursor=cursor,
                page_size=page_size,
                on_sale_only=on_sale_only,
                new_arrivals_only=new_arrivals_only,
                new_arrivals_days=new_arrivals_days,
                include_formality=include_formality,
                exclude_formality=exclude_formality,
                include_seasons=include_seasons,
                exclude_seasons=exclude_seasons,
                include_style_tags=include_style_tags,
                exclude_style_tags=exclude_style_tags,
                include_color_family=include_color_family,
                exclude_color_family=exclude_color_family,
                include_silhouette=include_silhouette,
                exclude_silhouette=exclude_silhouette,
                include_fit=include_fit,
                exclude_fit=exclude_fit,
                include_length=include_length,
                exclude_length=exclude_length,
                include_sleeves=include_sleeves,
                exclude_sleeves=exclude_sleeves,
                include_neckline=include_neckline,
                exclude_neckline=exclude_neckline,
                include_rise=include_rise,
                exclude_rise=exclude_rise,
                include_coverage=include_coverage,
                exclude_coverage=exclude_coverage,
                include_materials=include_materials,
                exclude_materials=exclude_materials,
                exclude_occasions=exclude_occasions,
                debug=debug,
            )

        # Session ID logic:
        # - If session_id provided: use it (explicit session tracking)
        # - If cursor provided but no session_id: derive from anon_id/user_id (auto-link)
        # - If neither: generate new session
        is_new_session = session_id is None
        if is_new_session:
            if cursor and (anon_id or user_id):
                # Auto-link session to user for pagination continuity
                seed = user_id or anon_id
                session_id = f"sess_{hashlib.md5(seed.encode()).hexdigest()[:12]}"
            else:
                # New feed request - generate fresh session
                session_id = SessionStateService.generate_session_id()
                print(f"[Pipeline] New keyset session created: {session_id}")

        # Validate page_size
        page_size = min(page_size, self.config.MAX_LIMIT)

        # Step 1: Decode cursor (if provided)
        cursor_obj = None
        cursor_score = None
        cursor_id = None
        page = 0

        if cursor:
            cursor_obj = self.session_service.decode_cursor(cursor)
            if cursor_obj:
                cursor_score = cursor_obj.score
                cursor_id = cursor_obj.item_id
                page = cursor_obj.page + 1

        # Step 2: Load user state
        user_state = self._load_user_state(user_id, anon_id, session_id)

        # Override categories from request
        if categories:
            if user_state.onboarding_profile:
                user_state.onboarding_profile.categories = categories
            else:
                from recs.models import OnboardingProfile
                user_state.onboarding_profile = OnboardingProfile(
                    user_id=user_state.user_id,
                    categories=categories
                )

        # Merge legacy attribute params into new include_ params
        # (backward compat: fit=["slim"] becomes include_fit=["slim"] if include_fit not set)
        if fit and not include_fit:
            include_fit = fit
        if length and not include_length:
            include_length = length
        if sleeves and not include_sleeves:
            include_sleeves = sleeves
        if neckline and not include_neckline:
            include_neckline = neckline
        if rise and not include_rise:
            include_rise = rise

        # Override filters from request
        has_any_filter = any([
            exclude_styles, include_occasions, min_price, max_price,
            exclude_brands, preferred_brands, exclude_colors, include_colors,
            include_patterns, exclude_patterns,
            include_fit, include_length, include_sleeves, include_neckline, include_rise,
        ])
        if has_any_filter:
            if not user_state.onboarding_profile:
                from recs.models import OnboardingProfile
                user_state.onboarding_profile = OnboardingProfile(
                    user_id=user_state.user_id
                )
            profile = user_state.onboarding_profile
            # Lifestyle filters
            if exclude_styles:
                profile.styles_to_avoid = exclude_styles
            if include_occasions:
                profile.occasions = include_occasions
            # Price filters
            if min_price is not None:
                profile.global_min_price = min_price
            if max_price is not None:
                profile.global_max_price = max_price
            # Brand filters
            if exclude_brands:
                profile.brands_to_avoid = exclude_brands
            if preferred_brands:
                profile.preferred_brands = preferred_brands
            # Color filters
            if exclude_colors:
                profile.colors_to_avoid = exclude_colors
            if include_colors:
                profile.colors_preferred = include_colors
            # Pattern filters
            if include_patterns:
                profile.patterns_liked = include_patterns
            if exclude_patterns:
                profile.patterns_avoided = exclude_patterns
            # Attribute filters (soft scoring via profile)
            if include_fit:
                profile.preferred_fits = include_fit
            if include_length:
                profile.preferred_lengths = include_length
            if include_sleeves:
                profile.preferred_sleeves = include_sleeves
            if include_neckline:
                profile.preferred_necklines = include_neckline
            if include_rise:
                profile.preferred_rises = include_rise

        # Step 3: Set feed version on first page (for stable ordering)
        if is_new_session or not self.session_service.has_feed_version(session_id):
            self.session_service.set_feed_version(
                session_id,
                taste_vector=user_state.taste_vector,
                filters={
                    "gender": gender,
                    "categories": categories
                },
                scoring_weights=self.ranker.config.WARM_WEIGHTS if user_state.taste_vector else self.ranker.config.COLD_WEIGHTS
            )

        # Step 4: Load DB seen history first (for SQL-level exclusion)
        db_seen_ids = self.candidate_module.get_user_seen_history(anon_id, user_id)
        db_history_count = len(db_seen_ids)

        # Calculate how many candidates to fetch
        # With SQL-level attribute filtering (JOIN product_attributes), no need for
        # massive over-fetch. Only over-fetch for Python-level filters (colors, brands, article_types, rise).
        has_python_filters = bool(
            include_colors or exclude_colors or preferred_brands or exclude_brands
            or article_types or include_rise or exclude_rise
        )

        filter_multiplier = 3 if has_python_filters else 1
        base_fetch = max(500, page_size * 10)  # At least 500, or 10x requested page
        fetch_size = base_fetch * filter_multiplier
        fetch_size = min(fetch_size, 5000)  # Safety cap

        # Step 5: Get candidates using keyset cursor
        # Attribute filters are applied directly in SQL via LEFT JOIN to product_attributes.
        # No pre-filter step needed — SQL handles all attribute filtering in one query.
        #
        # Expand frontend filter values to DB values.
        # The frontend sends user-friendly slugs (e.g., "casual", "active", "beach")
        # but the DB stores Gemini-generated values (e.g., "Everyday", "Workout", "Vacation").
        # We expand each frontend value to ALL matching DB values, and also handle
        # cross-dimension routing (e.g., "smart-casual" -> formality filter, not occasions).

        # --- Occasion expansion ---
        # Frontend occasion slugs -> DB product_attributes.occasions values
        _OCCASION_EXPANSION = {
            'casual':       ['Everyday', 'Weekend', 'Brunch', 'Casual Outings'],
            'active':       ['Workout'],
            'activewear':   ['Workout'],
            'beach':        ['Vacation'],
            'vacation':     ['Vacation'],
            'evening':      ['Date Night', 'Party', 'Night Out', 'Evening', 'Evening Event'],
            'date-night':   ['Date Night'],
            'date night':   ['Date Night'],
            'party':        ['Party', 'Night Out'],
            'office':       ['Office', 'Work'],
            'work':         ['Office', 'Work'],
            'formal':       ['Formal Event', 'Wedding Guest'],
            'wedding':      ['Wedding Guest', 'Wedding'],
            'lounge':       ['Lounging'],
            'lounging':     ['Lounging'],
            'brunch':       ['Brunch'],
            'weekend':      ['Weekend'],
            'workout':      ['Workout'],
            'night-out':    ['Night Out', 'Party'],
        }

        # Cross-dimension routing: some "occasion" values should trigger
        # formality or style_tags filters instead of (or in addition to) occasions.
        _OCCASION_CROSS_DIMENSION = {
            'smart-casual':  {'formality': ['Smart Casual', 'Business Casual']},
            'smart casual':  {'formality': ['Smart Casual', 'Business Casual']},
            'formal':        {'formality': ['Formal', 'Semi-Formal']},
        }

        def _expand(vals, expansion_map):
            """Expand frontend values using synonym map. Unknown values are title-cased as-is."""
            if not vals:
                return None
            expanded = []
            for v in vals:
                key = v.lower().strip()
                if key in expansion_map:
                    expanded.extend(expansion_map[key])
                else:
                    # Unknown value: title-case it and pass through
                    expanded.append(v.title())
            return list(dict.fromkeys(expanded)) if expanded else None  # dedupe, preserve order

        def _tc(vals):
            """Title-case normalize a list of filter values, or None."""
            if not vals:
                return vals
            return [v.title() for v in vals]

        # Handle cross-dimension routing FIRST (before occasion expansion)
        # e.g., "smart-casual" should add formality filter, not occasion filter.
        # Values that are purely cross-dimensional are removed from the occasion list.
        occasion_values_for_expansion = list(include_occasions) if include_occasions else []
        if include_occasions:
            for v in list(include_occasions):
                key = v.lower().strip()
                if key in _OCCASION_CROSS_DIMENSION:
                    cross = _OCCASION_CROSS_DIMENSION[key]
                    if 'formality' in cross:
                        if include_formality is None:
                            include_formality = []
                        include_formality = list(set(include_formality) | set(cross['formality']))
                        print(f"[Pipeline] Cross-dimension: '{v}' -> include_formality={cross['formality']}")
                    # Remove from occasion list if it's ONLY a cross-dimension value
                    # (not also a valid occasion expansion key)
                    if key not in _OCCASION_EXPANSION:
                        occasion_values_for_expansion = [x for x in occasion_values_for_expansion if x.lower().strip() != key]

        # Now expand remaining occasion values
        expanded_occasions = _expand(occasion_values_for_expansion, _OCCASION_EXPANSION) if occasion_values_for_expansion else None
        expanded_exclude_occasions = _expand(exclude_occasions, _OCCASION_EXPANSION) if exclude_occasions else None

        if expanded_occasions:
            print(f"[Pipeline] Occasion expansion: {include_occasions} -> {expanded_occasions}")
        if include_formality:
            print(f"[Pipeline] Formality filter (after cross-dim): {include_formality}")

        # Determine if Python-level brand bucketing will be active.
        # When it is, skip the SQL brand boost so we get a diverse candidate pool
        # for the 60/30/10 tier allocation to work on.
        _onboarding_brands = (
            user_state.onboarding_profile.preferred_brands
            if user_state.onboarding_profile else []
        ) or []
        will_bucket_brands = bool(_onboarding_brands) and not preferred_brands

        candidates = self.candidate_module.get_candidates_keyset(
            user_state,
            gender,
            cursor_score=cursor_score,
            cursor_id=cursor_id,
            page_size=fetch_size,
            exclude_ids=db_seen_ids if db_seen_ids else None,
            article_types=article_types,
            on_sale_only=on_sale_only,
            new_arrivals_only=new_arrivals_only,
            new_arrivals_days=new_arrivals_days,
            include_materials=include_materials,
            # Attribute filters via SQL JOIN (maps API params to SQL column names)
            # Occasions use synonym expansion; other fields use title-case normalization
            attr_include_formality=_tc(include_formality),
            attr_exclude_formality=_tc(exclude_formality),
            attr_include_seasons=_tc(include_seasons),
            attr_exclude_seasons=_tc(exclude_seasons),
            attr_include_style_tags=_tc(include_style_tags),
            attr_exclude_style_tags=_tc(exclude_style_tags),
            attr_include_color_family=_tc(include_color_family),
            attr_exclude_color_family=_tc(exclude_color_family),
            attr_include_silhouette=_tc(include_silhouette),
            attr_exclude_silhouette=_tc(exclude_silhouette),
            attr_include_fit_type=_tc(include_fit),          # API param 'fit' -> SQL column 'fit_type'
            attr_exclude_fit_type=_tc(exclude_fit),
            attr_include_coverage=_tc(include_coverage),      # SQL column 'coverage_level'
            attr_exclude_coverage=_tc(exclude_coverage),
            attr_include_pattern=_tc(include_patterns),       # API param 'patterns' -> SQL 'pattern'
            attr_exclude_pattern=_tc(exclude_patterns),
            attr_include_neckline=_tc(include_neckline),      # SQL reads from construction JSONB
            attr_exclude_neckline=_tc(exclude_neckline),
            attr_include_sleeve_type=_tc(include_sleeves),    # API 'sleeves' -> SQL 'sleeve_type'
            attr_exclude_sleeve_type=_tc(exclude_sleeves),
            attr_include_length=_tc(include_length),          # SQL reads from construction JSONB
            attr_exclude_length=_tc(exclude_length),
            attr_include_occasions=expanded_occasions,        # Already expanded with DB values
            attr_exclude_occasions=expanded_exclude_occasions,
            hybrid_brand_fetch=will_bucket_brands,
        )

        # Step 5a: Session-Aware Retrieval (Contextual Recall)
        # If the user has live session signals (from searches/clicks),
        # inject additional candidates matching those signals.
        # This bridges session scoring (ranking) with candidate retrieval (recall).
        session_intent_count = 0
        session_scores_for_recall = self._get_or_create_session_scores(
            session_id, user_state
        ) if session_id else None

        if session_scores_for_recall and session_scores_for_recall.action_count > 0:
            try:
                from recs.session_scoring import extract_intent_filters
                intent_filters = extract_intent_filters(session_scores_for_recall)

                if intent_filters["has_intent"]:
                    existing_ids = {c.item_id for c in candidates}
                    intent_candidates = self._retrieve_session_intent_candidates(
                        user_state=user_state,
                        gender=gender,
                        intent_brands=intent_filters["brands"],
                        intent_types=intent_filters["types"],
                        exclude_ids=existing_ids | db_seen_ids,
                        limit=100,
                    )
                    # Merge: add intent candidates not already in pool
                    for c in intent_candidates:
                        if c.item_id not in existing_ids:
                            c.source = "session_intent"
                            candidates.append(c)
                            existing_ids.add(c.item_id)
                            session_intent_count += 1

                    if session_intent_count > 0:
                        print(f"[Pipeline] Session-intent recall: +{session_intent_count} candidates "
                              f"(brands={intent_filters['brands']}, types={intent_filters['types']})")
            except Exception as e:
                print(f"[Pipeline] Session-intent recall skipped (non-fatal): {e}")

        # Track candidates retrieved for metadata
        pre_filter_count_for_meta = len(candidates)

        # Step 5b: Apply Python-level hard filters (colors, brands, article_types)
        # These are more reliable than SQL-level filtering
        pre_python_filter_count = len(candidates)

        # Apply color filter
        if include_colors or exclude_colors:
            candidates = self._apply_color_filter(candidates, include_colors, exclude_colors)

        # Apply brand filter (preferred_brands = hard include filter)
        if preferred_brands or exclude_brands:
            candidates = self._apply_brand_filter(candidates, preferred_brands, exclude_brands)

        # Apply article_type filter (name-based matching)
        if article_types:
            candidates = self._apply_article_type_filter(candidates, article_types)

        post_filter_count = len(candidates)
        if pre_python_filter_count != post_filter_count:
            print(f"[Pipeline] Python filters: {pre_python_filter_count} -> {post_filter_count} candidates")

        # Step 5b2: Rise filter (Python-only — no product_attributes column)
        # All other attribute filters are now handled in SQL via JOIN.
        if include_rise or exclude_rise:
            pre_rise = len(candidates)
            filtered = []
            inc_set = {v.lower() for v in include_rise} if include_rise else None
            exc_set = {v.lower() for v in exclude_rise} if exclude_rise else None
            for c in candidates:
                val = (c.rise or '').lower() if c.rise else None
                if inc_set and (not val or val not in inc_set):
                    continue
                if exc_set and val and val in exc_set:
                    continue
                filtered.append(c)
            candidates = filtered
            post_filter_count = len(candidates)
            if pre_rise != post_filter_count:
                print(f"[Pipeline] Rise filter: {pre_rise} -> {post_filter_count} candidates")

        # Step 5c: Apply FeasibilityFilter (HARD constraint-based filtering)
        # This runs BEFORE soft scoring and uses canonicalized article types
        feasibility_stats = {}
        user_exclusions = []
        if user_state.onboarding_profile:
            user_exclusions = user_state.onboarding_profile.get_user_exclusions()

        if include_occasions or user_exclusions:
            pre_feasibility = len(candidates)
            candidates, feasibility_stats = filter_by_feasibility(
                candidates,
                occasions=include_occasions,
                user_exclusions=user_exclusions,
                verbose=False
            )
            post_feasibility = len(candidates)
            if pre_feasibility != post_feasibility:
                print(f"[Pipeline] Feasibility filter: {pre_feasibility} -> {post_feasibility} candidates")
                # Log top blocked reasons for debugging
                if feasibility_stats.get("blocked_reasons"):
                    top_reasons = sorted(feasibility_stats["blocked_reasons"].items(), key=lambda x: -x[1])[:3]
                    for reason, count in top_reasons:
                        print(f"  - {reason}: {count}")

        # Step 5d: Filter by occasions using product_attributes.occasions array
        # Simple array membership check - much simpler than old CLIP-based gate
        occasion_filter_stats = {}
        if include_occasions:
            pre_occasion_count = len(candidates)
            candidates = self._filter_by_occasions(candidates, include_occasions)
            post_occasion_count = len(candidates)
            occasion_filter_stats = {
                'pre_count': pre_occasion_count,
                'post_count': post_occasion_count,
                'filtered': pre_occasion_count - post_occasion_count,
            }
            if pre_occasion_count != post_occasion_count:
                print(f"[Pipeline] Occasion filter ({include_occasions}): {pre_occasion_count} -> {post_occasion_count} candidates")

        # Step 5e: Brand tier bucketing (3-tier: preferred / cluster-adjacent / discovery)
        # Only apply when the user has onboarding preferred brands and NO runtime
        # brand filter is active (runtime brand filter already hard-includes brands
        # via _apply_brand_filter, making bucketing redundant).
        # Uses will_bucket_brands computed before SQL retrieval (which also skipped
        # the SQL brand boost to ensure a diverse candidate pool).
        if will_bucket_brands:
            bucket_target = max(200, page_size * 5)
            seed_str = user_id or anon_id or "default"
            candidates = self._apply_brand_tier_bucketing(
                candidates=candidates,
                preferred_brands=_onboarding_brands,
                target_size=bucket_target,
                seed=seed_str,
            )

        # Step 6: Rank candidates (SASRec + embedding scoring)
        ranked_candidates = self.ranker.rank_candidates(user_state, candidates)

        # Step 6b: Session-aware scoring
        # Load or initialize session scores for this user
        session_scores = self._get_or_create_session_scores(
            session_id, user_state
        )
        # Apply session scoring on top of existing scores
        ranked_candidates = self.scoring_engine.score_candidates(
            session_scores, ranked_candidates
        )

        # Step 6c: Context-aware scoring (age affinity + weather/season)
        context_scoring_meta = {}
        try:
            birthdate = None
            profile_dict = None
            if user_state.onboarding_profile:
                birthdate = user_state.onboarding_profile.birthdate
                # Build profile dict for coverage prefs extraction
                profile_dict = {}
                profile = user_state.onboarding_profile
                for flag in ("no_sleeveless", "no_tanks", "no_crop", "no_athletic", "no_revealing"):
                    profile_dict[flag] = getattr(profile, flag, False)
                profile_dict["styles_to_avoid"] = getattr(profile, "styles_to_avoid", []) or []
                profile_dict["modesty"] = getattr(profile, "modesty", None)

            user_context = self.context_resolver.resolve(
                user_id=user_id or anon_id or "anonymous",
                jwt_user_metadata=user_metadata,
                birthdate=birthdate,
                onboarding_profile=profile_dict,
            )

            if user_context.age_group or user_context.weather:
                for c in ranked_candidates:
                    item_dict = c.to_scoring_dict()
                    adj = self.context_scorer.score_item(item_dict, user_context)
                    c.final_score += adj

                context_scoring_meta = {
                    "age_group": user_context.age_group.value if user_context.age_group else None,
                    "has_weather": user_context.weather is not None,
                    "season": user_context.weather.season.value if user_context.weather else None,
                    "city": user_context.city,
                }
                print(f"[Pipeline] Context scoring applied: age={context_scoring_meta.get('age_group')}, season={context_scoring_meta.get('season')}")
        except Exception as e:
            print(f"[Pipeline] Context scoring skipped (non-fatal): {e}")
            context_scoring_meta = {"error": str(e)}

        # Step 7: Get seen items for dedup + reranking
        session_seen_ids = self.session_service.get_seen_items(session_id)
        all_seen_ids = session_seen_ids | db_seen_ids

        # Step 7b: Deduplicate by image hash
        ranked_candidates = self._deduplicate_by_image(ranked_candidates)

        # Step 8: Greedy constrained list-wise reranking
        # Builds the feed one item at a time with diversity constraints:
        # - max per brand, max per cluster, max per type
        # - no repeated attribute combos
        # - exploration injection
        filtered_candidates = self.feed_reranker.rerank(
            ranked_candidates,
            target_size=page_size + 10,  # Fetch slightly more for safety
            seen_ids=all_seen_ids,
            skipped_ids=session_scores.skipped_ids,
        )

        # Step 9: Take top N for this page
        page_results = filtered_candidates[:page_size]

        # Step 10: Update session state
        if page_results:
            # Add to seen items
            shown_ids = [c.item_id for c in page_results]
            self.session_service.add_seen_items(session_id, shown_ids)

            # Update cursor with last item
            # IMPORTANT: Use embedding_score (SQL-side score) for cursor, not final_score
            # The keyset cursor must match what SQL uses for ordering
            last_item = page_results[-1]
            cursor_score = last_item.embedding_score if last_item.embedding_score > 0 else 0.5
            self.session_service.set_cursor(
                session_id,
                score=cursor_score,
                item_id=last_item.item_id,
                page=page
            )

        # Step 11: Get next cursor
        next_cursor = None
        if page_results:
            last_item = page_results[-1]
            # Use embedding_score for cursor (matches SQL ordering)
            cursor_score = last_item.embedding_score if last_item.embedding_score > 0 else 0.5
            next_cursor_obj = KeysetCursor(
                score=cursor_score,
                item_id=last_item.item_id,
                page=page
            )
            next_cursor = next_cursor_obj.encode()

        # Calculate has_more
        # True if we have more filtered candidates OR if we fetched a full batch (indicating more in DB)
        has_more = len(filtered_candidates) > page_size or len(candidates) >= fetch_size

        # Determine strategy used
        # EXPERIMENT: Always exploration (taste_vector disabled)
        # if user_state.state_type == UserStateType.WARM_USER:
        #     strategy = "sasrec"
        # elif user_state.state_type == UserStateType.TINDER_COMPLETE:
        #     strategy = "seed_vector"
        # else:
        #     strategy = "exploration"
        strategy = "exploration"

        # Get total seen count (session only for this response, DB history is permanent)
        total_seen = len(session_seen_ids) + len(page_results)

        # Convert to response format
        results = []
        for rank, candidate in enumerate(page_results, start=total_seen - len(page_results) + 1):
            results.append({
                "product_id": candidate.item_id,
                "rank": rank,
                "score": candidate.final_score,
                "reason": self._get_reason(candidate, user_state),
                "category": candidate.category,
                "broad_category": candidate.broad_category,
                "brand": candidate.brand,
                "name": candidate.name,
                "price": candidate.price,
                "image_url": candidate.image_url,
                "gallery_images": candidate.gallery_images,
                "colors": candidate.colors,
                "source": candidate.source,
                # Sale/New arrival fields
                "original_price": candidate.original_price,
                "is_on_sale": candidate.is_on_sale,
                "discount_percent": candidate.discount_percent,
                "is_new": candidate.is_new,
            })

        # Build metadata
        by_source = defaultdict(int)
        for c in page_results:
            by_source[c.source] += 1

        resp = {
            "user_id": user_id or anon_id or "anonymous",
            "session_id": session_id,
            "cursor": next_cursor,
            "sort_by": FeedSortBy.RELEVANCE.value,
            "strategy": strategy,
            "results": results,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "items_returned": len(results),
                "session_seen_count": total_seen,
                "has_more": has_more
            },
            "metadata": {
                "candidates_retrieved": pre_filter_count_for_meta,
                "candidates_after_python_filters": post_filter_count,
                "sql_attribute_filters_active": has_python_filters or bool(expanded_occasions or include_formality or exclude_formality or include_seasons or exclude_seasons or include_style_tags or exclude_style_tags),
                "candidates_after_feasibility_filter": feasibility_stats.get("passed") if feasibility_stats else post_filter_count,
                "candidates_after_occasion_filter": len(candidates) if include_occasions else None,
                "candidates_after_scoring": len(ranked_candidates),
                "candidates_after_dedup": len(filtered_candidates),
                "sasrec_available": self.ranker.model is not None,
                "seed_vector_available": user_state.taste_vector is not None,
                "has_onboarding": user_state.onboarding_profile is not None,
                "user_state_type": user_state.state_type.value,
                "by_source": dict(by_source),
                "keyset_pagination": True,
                "session_scoring": {
                    "action_count": session_scores.action_count,
                    "active_clusters": list(session_scores.cluster_scores.keys()),
                    "active_brands": len(session_scores.brand_scores),
                    "search_intents": len(session_scores.search_intents),
                    "session_intent_candidates": session_intent_count,
                },
                "context_scoring": context_scoring_meta if context_scoring_meta else None,
                "db_seen_history_count": db_history_count,
                "fetch_size_used": fetch_size,
                "python_filters_applied": {
                    "include_colors": include_colors,
                    "exclude_colors": exclude_colors,
                    "include_brands": preferred_brands,
                    "exclude_brands": exclude_brands,
                    "article_types": article_types,
                    "include_rise": include_rise,
                    "exclude_rise": exclude_rise,
                } if has_python_filters else None,
                "feasibility_filter": {
                    "occasions": include_occasions,
                    "user_exclusions": user_exclusions,
                    "blocked": feasibility_stats.get("blocked", 0),
                    "blocked_reasons": feasibility_stats.get("blocked_reasons", {}),
                    "warnings": feasibility_stats.get("warnings", [])
                } if feasibility_stats else None,
                "occasion_filter_applied": {
                    "occasions": include_occasions,
                    "filtered_count": occasion_filter_stats.get("filtered", 0),
                } if include_occasions else None,
                "feed_version": self.session_service.get_feed_version(session_id).version_id if self.session_service.get_feed_version(session_id) else None
            }
        }

        # Strip internal metadata in production (only include when debug=True)
        if not debug:
            resp.pop("metadata", None)

        return resp

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session info for debugging."""
        return self.session_service.get_session_info(session_id)

    def clear_session(self, session_id: str):
        """Clear a session (reset seen items)."""
        self.session_service.clear_session(session_id)

    # =========================================================
    # Sorted Feed (price_asc / price_desc fast path)
    # =========================================================

    def _get_feed_sorted(
        self,
        sort_by: FeedSortBy,
        user_id: Optional[str] = None,
        anon_id: Optional[str] = None,
        session_id: Optional[str] = None,
        gender: str = "female",
        categories: Optional[List[str]] = None,
        article_types: Optional[List[str]] = None,
        exclude_styles: Optional[List[str]] = None,
        include_occasions: Optional[List[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        exclude_brands: Optional[List[str]] = None,
        preferred_brands: Optional[List[str]] = None,
        exclude_colors: Optional[List[str]] = None,
        include_colors: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        fit: Optional[List[str]] = None,
        length: Optional[List[str]] = None,
        sleeves: Optional[List[str]] = None,
        neckline: Optional[List[str]] = None,
        rise: Optional[List[str]] = None,
        cursor: Optional[str] = None,
        page_size: int = 50,
        on_sale_only: bool = False,
        new_arrivals_only: bool = False,
        new_arrivals_days: int = 7,
        include_formality: Optional[List[str]] = None,
        exclude_formality: Optional[List[str]] = None,
        include_seasons: Optional[List[str]] = None,
        exclude_seasons: Optional[List[str]] = None,
        include_style_tags: Optional[List[str]] = None,
        exclude_style_tags: Optional[List[str]] = None,
        include_color_family: Optional[List[str]] = None,
        exclude_color_family: Optional[List[str]] = None,
        include_silhouette: Optional[List[str]] = None,
        exclude_silhouette: Optional[List[str]] = None,
        include_fit: Optional[List[str]] = None,
        exclude_fit: Optional[List[str]] = None,
        include_length: Optional[List[str]] = None,
        exclude_length: Optional[List[str]] = None,
        include_sleeves: Optional[List[str]] = None,
        exclude_sleeves: Optional[List[str]] = None,
        include_neckline: Optional[List[str]] = None,
        exclude_neckline: Optional[List[str]] = None,
        include_rise: Optional[List[str]] = None,
        exclude_rise: Optional[List[str]] = None,
        include_coverage: Optional[List[str]] = None,
        exclude_coverage: Optional[List[str]] = None,
        include_materials: Optional[List[str]] = None,
        exclude_materials: Optional[List[str]] = None,
        exclude_occasions: Optional[List[str]] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Fast-path feed for deterministic sort modes (price_asc, price_desc).

        Skips: SASRec scoring, session scoring, context scoring, brand tier
        bucketing, greedy constrained reranker, image dedup, session-intent
        recall, feasibility filter, occasion filter.

        Keeps: user state loading (for hard filters from onboarding), request
        filter overrides, cursor decode with sort_mode validation, DB seen
        history exclusion, Python-level hard filters (colors, brands,
        article_types, rise), light brand diversity via apply_sort_diversity.
        """
        from recs.session_state import KeysetCursor
        import hashlib

        sort_mode = sort_by.value  # "price_asc" or "price_desc"
        print(f"[Pipeline] Sorted feed path: sort_by={sort_mode}")

        # ----- Session ID logic (same as relevance path) -----
        is_new_session = session_id is None
        if is_new_session:
            if cursor and (anon_id or user_id):
                seed = user_id or anon_id
                session_id = f"sess_{hashlib.md5(seed.encode()).hexdigest()[:12]}"
            else:
                session_id = SessionStateService.generate_session_id()
                print(f"[Pipeline] New sorted-feed session: {session_id}")

        page_size = min(page_size, self.config.MAX_LIMIT)

        # ----- Decode cursor & validate sort_mode -----
        cursor_value: Optional[float] = None
        cursor_id: Optional[str] = None
        page = 0

        if cursor:
            cursor_obj = self.session_service.decode_cursor(cursor)
            if cursor_obj:
                if cursor_obj.sort_mode == sort_mode:
                    cursor_value = cursor_obj.score  # holds price
                    cursor_id = cursor_obj.item_id
                    page = cursor_obj.page + 1
                else:
                    # Sort mode changed — reset to page 1
                    print(f"[Pipeline] Cursor sort_mode mismatch: "
                          f"cursor={cursor_obj.sort_mode}, request={sort_mode}. "
                          f"Resetting to page 1.")

        # ----- Load user state (for hard filters from onboarding) -----
        user_state = self._load_user_state(user_id, anon_id, session_id)

        # Override categories from request
        if categories:
            if user_state.onboarding_profile:
                user_state.onboarding_profile.categories = categories
            else:
                from recs.models import OnboardingProfile
                user_state.onboarding_profile = OnboardingProfile(
                    user_id=user_state.user_id,
                    categories=categories,
                )

        # Merge legacy attribute params into new include_ params
        if fit and not include_fit:
            include_fit = fit
        if length and not include_length:
            include_length = length
        if sleeves and not include_sleeves:
            include_sleeves = sleeves
        if neckline and not include_neckline:
            include_neckline = neckline
        if rise and not include_rise:
            include_rise = rise

        # Apply request filter overrides to profile (same as relevance path)
        has_any_filter = any([
            exclude_styles, include_occasions, min_price, max_price,
            exclude_brands, preferred_brands, exclude_colors, include_colors,
            include_patterns, exclude_patterns,
            include_fit, include_length, include_sleeves, include_neckline, include_rise,
        ])
        if has_any_filter:
            if not user_state.onboarding_profile:
                from recs.models import OnboardingProfile
                user_state.onboarding_profile = OnboardingProfile(
                    user_id=user_state.user_id
                )
            profile = user_state.onboarding_profile
            if exclude_styles:
                profile.styles_to_avoid = exclude_styles
            if include_occasions:
                profile.occasions = include_occasions
            if min_price is not None:
                profile.global_min_price = min_price
            if max_price is not None:
                profile.global_max_price = max_price
            if exclude_brands:
                profile.brands_to_avoid = exclude_brands
            if preferred_brands:
                profile.preferred_brands = preferred_brands
            if exclude_colors:
                profile.colors_to_avoid = exclude_colors
            if include_colors:
                profile.colors_preferred = include_colors
            if include_patterns:
                profile.patterns_liked = include_patterns
            if exclude_patterns:
                profile.patterns_avoided = exclude_patterns
            if include_fit:
                profile.preferred_fits = include_fit
            if include_length:
                profile.preferred_lengths = include_length
            if include_sleeves:
                profile.preferred_sleeves = include_sleeves
            if include_neckline:
                profile.preferred_necklines = include_neckline
            if include_rise:
                profile.preferred_rises = include_rise

        # ----- Title-case helper + occasion expansion (reuse from relevance) -----
        def _tc(vals):
            if not vals:
                return vals
            return [v.title() for v in vals]

        _OCCASION_EXPANSION = {
            'casual': ['Everyday', 'Weekend', 'Brunch', 'Casual Outings'],
            'active': ['Workout'], 'activewear': ['Workout'],
            'beach': ['Vacation'], 'vacation': ['Vacation'],
            'evening': ['Date Night', 'Party', 'Night Out', 'Evening', 'Evening Event'],
            'date-night': ['Date Night'], 'date night': ['Date Night'],
            'party': ['Party', 'Night Out'],
            'office': ['Office', 'Work'], 'work': ['Office', 'Work'],
            'formal': ['Formal Event', 'Wedding Guest'],
            'wedding': ['Wedding Guest', 'Wedding'],
            'lounge': ['Lounging'], 'lounging': ['Lounging'],
            'brunch': ['Brunch'], 'weekend': ['Weekend'],
            'workout': ['Workout'], 'night-out': ['Night Out', 'Party'],
        }

        def _expand(vals, expansion_map):
            if not vals:
                return None
            expanded = []
            for v in vals:
                key = v.lower().strip()
                if key in expansion_map:
                    expanded.extend(expansion_map[key])
                else:
                    expanded.append(v.title())
            return list(dict.fromkeys(expanded)) if expanded else None

        expanded_occasions = _expand(include_occasions, _OCCASION_EXPANSION) if include_occasions else None
        expanded_exclude_occasions = _expand(exclude_occasions, _OCCASION_EXPANSION) if exclude_occasions else None

        # ----- DB seen history (for SQL-level exclusion) -----
        db_seen_ids = self.candidate_module.get_user_seen_history(anon_id, user_id)

        # Fetch more than page_size to account for Python-level filtering
        has_python_filters = bool(
            include_colors or exclude_colors or preferred_brands or exclude_brands
            or article_types or include_rise or exclude_rise
        )
        buffer_mult = 3 if has_python_filters else 1
        fetch_size = min(max(200, page_size * 5) * buffer_mult, 3000)

        # ----- Retrieve sorted candidates from SQL -----
        candidates = self.candidate_module.get_candidates_sorted_keyset(
            user_state,
            gender,
            sort_mode=sort_mode,
            cursor_value=cursor_value,
            cursor_id=cursor_id,
            page_size=fetch_size,
            exclude_ids=db_seen_ids if db_seen_ids else None,
            article_types=article_types,
            on_sale_only=on_sale_only,
            new_arrivals_only=new_arrivals_only,
            new_arrivals_days=new_arrivals_days,
            include_materials=include_materials,
            attr_include_formality=_tc(include_formality),
            attr_exclude_formality=_tc(exclude_formality),
            attr_include_seasons=_tc(include_seasons),
            attr_exclude_seasons=_tc(exclude_seasons),
            attr_include_style_tags=_tc(include_style_tags),
            attr_exclude_style_tags=_tc(exclude_style_tags),
            attr_include_color_family=_tc(include_color_family),
            attr_exclude_color_family=_tc(exclude_color_family),
            attr_include_silhouette=_tc(include_silhouette),
            attr_exclude_silhouette=_tc(exclude_silhouette),
            attr_include_fit_type=_tc(include_fit),
            attr_exclude_fit_type=_tc(exclude_fit),
            attr_include_coverage=_tc(include_coverage),
            attr_exclude_coverage=_tc(exclude_coverage),
            attr_include_pattern=_tc(include_patterns),
            attr_exclude_pattern=_tc(exclude_patterns),
            attr_include_neckline=_tc(include_neckline),
            attr_exclude_neckline=_tc(exclude_neckline),
            attr_include_sleeve_type=_tc(include_sleeves),
            attr_exclude_sleeve_type=_tc(exclude_sleeves),
            attr_include_length=_tc(include_length),
            attr_exclude_length=_tc(exclude_length),
            attr_include_occasions=expanded_occasions,
            attr_exclude_occasions=expanded_exclude_occasions,
        )

        pre_filter_count = len(candidates)

        # ----- Python-level hard filters (same as relevance path) -----
        if include_colors or exclude_colors:
            candidates = self._apply_color_filter(candidates, include_colors, exclude_colors)
        if preferred_brands or exclude_brands:
            candidates = self._apply_brand_filter(candidates, preferred_brands, exclude_brands)
        if article_types:
            candidates = self._apply_article_type_filter(candidates, article_types)
        if include_rise or exclude_rise:
            inc_set = {v.lower() for v in include_rise} if include_rise else None
            exc_set = {v.lower() for v in exclude_rise} if exclude_rise else None
            filtered = []
            for c in candidates:
                val = (c.rise or '').lower() if c.rise else None
                if inc_set and (not val or val not in inc_set):
                    continue
                if exc_set and val and val in exc_set:
                    continue
                filtered.append(c)
            candidates = filtered

        post_filter_count = len(candidates)
        if pre_filter_count != post_filter_count:
            print(f"[Pipeline] Sorted feed Python filters: {pre_filter_count} -> {post_filter_count}")

        # ----- Light brand diversity (preserves sort order) -----
        session_seen_ids = self.session_service.get_seen_items(session_id)
        all_seen_ids = session_seen_ids | db_seen_ids

        diversified = apply_sort_diversity(
            candidates,
            max_consecutive=2,
            max_brand_share=0.30,
            seen_ids=all_seen_ids,
        )

        # ----- Take page -----
        page_results = diversified[:page_size]

        # ----- Update session state -----
        if page_results:
            shown_ids = [c.item_id for c in page_results]
            self.session_service.add_seen_items(session_id, shown_ids)

            last_item = page_results[-1]
            # Cursor value = price (stored in embedding_score by SQL, or fall back to .price)
            last_price = last_item.embedding_score if last_item.embedding_score > 0 else last_item.price
            self.session_service.set_cursor(
                session_id,
                score=last_price,
                item_id=last_item.item_id,
                page=page,
                sort_mode=sort_mode,
            )

        # ----- Build next cursor -----
        next_cursor = None
        if page_results:
            last_item = page_results[-1]
            last_price = last_item.embedding_score if last_item.embedding_score > 0 else last_item.price
            next_cursor_obj = KeysetCursor(
                score=last_price,
                item_id=last_item.item_id,
                page=page,
                sort_mode=sort_mode,
            )
            next_cursor = next_cursor_obj.encode()

        has_more = len(diversified) > page_size or len(candidates) >= fetch_size

        total_seen = len(session_seen_ids) + len(page_results)

        # ----- Convert to response format -----
        results = []
        for rank, candidate in enumerate(page_results, start=total_seen - len(page_results) + 1):
            results.append({
                "product_id": candidate.item_id,
                "rank": rank,
                "score": candidate.price,  # Use price as the score for sorted feeds
                "reason": "sorted",
                "category": candidate.category,
                "broad_category": candidate.broad_category,
                "brand": candidate.brand,
                "name": candidate.name,
                "price": candidate.price,
                "image_url": candidate.image_url,
                "gallery_images": candidate.gallery_images,
                "colors": candidate.colors,
                "source": candidate.source,
                "original_price": candidate.original_price,
                "is_on_sale": candidate.is_on_sale,
                "discount_percent": candidate.discount_percent,
                "is_new": candidate.is_new,
            })

        resp = {
            "user_id": user_id or anon_id or "anonymous",
            "session_id": session_id,
            "cursor": next_cursor,
            "sort_by": sort_mode,
            "strategy": "sorted",
            "results": results,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "items_returned": len(results),
                "session_seen_count": total_seen,
                "has_more": has_more,
            },
            "metadata": {
                "sort_mode": sort_mode,
                "candidates_retrieved": pre_filter_count,
                "candidates_after_python_filters": post_filter_count,
                "candidates_after_diversity": len(diversified),
                "db_seen_history_count": len(db_seen_ids),
                "fetch_size_used": fetch_size,
            },
        }

        if not debug:
            resp.pop("metadata", None)

        return resp

    # =========================================================
    # Session Scoring (Redis-backed with L1 cache)
    # =========================================================

    def _init_scoring_backend(self):
        """Initialize scoring backend (Redis if REDIS_URL set, else in-memory)."""
        from recs.session_state import get_scoring_backend
        return get_scoring_backend(backend="auto")

    def _get_or_create_session_scores(
        self, session_id: str, user_state: UserState
    ) -> SessionScores:
        """
        Get or create session scores for a user.

        Lookup order: L1 cache -> scoring backend (Redis/memory) -> cold start.
        On first call (cold start): initializes from onboarding brands.
        On subsequent calls: returns the existing session scores.
        """
        # L1 cache check
        if session_id in self._scores_l1_cache:
            return self._scores_l1_cache[session_id]

        # Backend check (Redis or in-memory)
        scores = self._scoring_backend.get_scores(session_id)
        if scores is not None:
            self._scores_l1_cache[session_id] = scores
            return scores

        # Cold start - initialize from onboarding
        preferred_brands = []
        profile = user_state.onboarding_profile
        if profile:
            preferred_brands = profile.preferred_brands or []

        scores = self.scoring_engine.initialize_from_onboarding(
            preferred_brands=preferred_brands,
            onboarding_profile=profile,
        )

        # Persist to backend and L1 cache
        self._scoring_backend.save_scores(session_id, scores)
        self._scores_l1_cache[session_id] = scores

        # Evict L1 if too large
        if len(self._scores_l1_cache) > self._SCORES_L1_MAX:
            sorted_keys = sorted(
                self._scores_l1_cache.keys(),
                key=lambda k: self._scores_l1_cache[k].last_updated,
            )
            for k in sorted_keys[: len(sorted_keys) // 2]:
                del self._scores_l1_cache[k]

        return scores

    def get_session_scores(self, session_id: str) -> Optional[SessionScores]:
        """Get session scores for external use (e.g., search reranker).

        Lookup order: L1 cache -> scoring backend (Redis/memory).
        """
        scores = self._scores_l1_cache.get(session_id)
        if scores is None:
            scores = self._scoring_backend.get_scores(session_id)
            if scores is not None:
                self._scores_l1_cache[session_id] = scores
        return scores

    def update_session_scores_from_action(
        self,
        session_id: str,
        action: str,
        product_id: str,
        brand: str = "",
        item_type: str = "",
        attributes: Dict[str, str] = None,
        source: str = "feed",
    ) -> None:
        """
        Update session scores from a user action.
        Called by the /api/recs/v2/feed/action endpoint.
        Persists back to scoring backend after mutation.
        """
        # Load from L1 or backend
        scores = self._scores_l1_cache.get(session_id)
        if scores is None:
            scores = self._scoring_backend.get_scores(session_id)
        if scores is None:
            scores = SessionScores()

        self.scoring_engine.process_action(
            scores, action=action, product_id=product_id,
            brand=brand, item_type=item_type,
            attributes=attributes or {}, source=source,
        )

        # Persist back
        self._scoring_backend.save_scores(session_id, scores)
        self._scores_l1_cache[session_id] = scores

    def update_session_scores_from_search(
        self,
        session_id: str,
        query: str = "",
        filters: Dict[str, Any] = None,
    ) -> None:
        """
        Update session scores from a search/filter action.
        Called by the search endpoint to feed signals into the recommendation feed.
        Persists back to scoring backend after mutation.
        """
        # Load from L1 or backend
        scores = self._scores_l1_cache.get(session_id)
        if scores is None:
            scores = self._scoring_backend.get_scores(session_id)
        if scores is None:
            scores = SessionScores()

        self.scoring_engine.process_search_signal(
            scores, query=query, filters=filters or {},
        )

        # Persist back
        self._scoring_backend.save_scores(session_id, scores)
        self._scores_l1_cache[session_id] = scores

    # =========================================================
    # User State Loading
    # =========================================================

    def _load_user_state(
        self,
        user_id: Optional[str],
        anon_id: Optional[str],
        session_id: Optional[str]
    ) -> UserState:
        """
        Load complete user state.

        Combines:
        - Onboarding profile (from Supabase)
        - Tinder taste_vector (from Supabase)
        - Interaction history (for SASRec)
        - Session context (seen items)
        """
        # Load from candidate module (handles Supabase queries)
        state = self.candidate_module.load_user_state(user_id, anon_id)

        # TODO: Load interaction history for SASRec from events table
        # For now, this would need to be passed in or loaded separately
        # state.interaction_sequence = await load_interaction_sequence(user_id or anon_id)

        # Determine final state type based on sequence length
        if state.taste_vector and len(state.interaction_sequence) >= self.config.ranker_config.MIN_SEQUENCE_FOR_SASREC:
            state.state_type = UserStateType.WARM_USER
        elif state.taste_vector:
            state.state_type = UserStateType.TINDER_COMPLETE
        else:
            state.state_type = UserStateType.COLD_START

        return state

    # =========================================================
    # Deduplication
    # =========================================================

    def _deduplicate_by_image(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Deduplicate candidates by image hash to remove same products across different brands.

        Many fashion retailers (e.g., Boohoo/Nasty Gal) sell identical items under different
        brand names. Uses shared filter_utils for consistent deduplication logic.
        """
        return deduplicate_candidates(candidates)

    # =========================================================
    # Session-Aware Retrieval (Contextual Recall)
    # =========================================================

    def _retrieve_session_intent_candidates(
        self,
        user_state: "UserState",
        gender: str,
        intent_brands: List[str],
        intent_types: List[str],
        exclude_ids: set,
        limit: int = 100,
    ) -> List[Candidate]:
        """
        Retrieve additional candidates matching live session intent signals.

        Uses direct PostgREST table queries (not an RPC function) to fetch
        products whose brand or article_type matches the user's in-session
        signals.  Results are enriched with product_attributes and converted
        to Candidate objects with source="session_intent".

        Args:
            user_state: Current user state (for hard filter extraction).
            gender: Gender filter string (e.g. "women").
            intent_brands: Brands the user has shown session affinity for.
            intent_types: Article types the user has shown session affinity for.
            exclude_ids: Product IDs already in the candidate pool or seen.
            limit: Maximum candidates to retrieve.

        Returns:
            List of Candidate objects (may be empty on error or no matches).
        """
        if not intent_brands and not intent_types:
            return []

        supabase = self.candidate_module.supabase

        # Columns matching what the RPC functions return, plus extras for enrichment
        SELECT_COLS = (
            "id, name, brand, category, broad_category, article_type, "
            "price, original_price, in_stock, fit, length, sleeve, "
            "neckline, rise, colors, materials, style_tags, "
            "primary_image_url, gallery_images, gender"
        )

        try:
            # Build OR filter for brands and types
            or_clauses = []
            for b in intent_brands:
                # Use ilike for case-insensitive brand matching
                escaped = b.replace("%", "").replace("_", "")
                or_clauses.append(f"brand.ilike.%{escaped}%")
            for t in intent_types:
                escaped = t.replace("%", "").replace("_", "")
                or_clauses.append(f"article_type.ilike.%{escaped}%")

            if not or_clauses:
                return []

            or_filter = ",".join(or_clauses)

            # Query products table with filters
            query = supabase.table("products") \
                .select(SELECT_COLS) \
                .eq("in_stock", True) \
                .not_.is_("primary_image_url", "null") \
                .or_(or_filter)

            # Apply gender filter if available
            if gender:
                query = query.contains("gender", [gender])

            # Exclude already-seen/pooled products
            if exclude_ids:
                # PostgREST has a max URL length, so limit exclusions to 500
                exclude_list = list(exclude_ids)[:500]
                # Use not_.in_ to exclude IDs we already have
                # Note: supabase-py uses .not_.in_() for NOT IN
                # But for large sets, we filter in Python to avoid URL limits
                pass  # Will filter in Python below

            query = query.limit(limit * 2)  # Over-fetch to account for Python filtering
            result = query.execute()

            if not result.data:
                return []

            # Python-level filtering: exclude IDs already in pool
            rows = [
                r for r in result.data
                if str(r.get("id", "")) not in exclude_ids
            ][:limit]

            if not rows:
                return []

            # Convert to Candidate objects using the same pattern as CandidateSelectionModule
            candidates = []
            for row in rows:
                # Map 'id' -> 'product_id' to match the format _row_to_candidate expects
                row["product_id"] = row.get("id", "")
                row["similarity"] = 0.0  # No embedding score for direct queries
                c = self.candidate_module._row_to_candidate(row, source="session_intent")
                candidates.append(c)

            # Enrich with product_attributes (occasions, pattern, formality, etc.)
            candidates = self.candidate_module._enrich_with_attributes(
                candidates, require_attributes=False
            )

            print(f"[Pipeline] Session-intent retrieval: queried {len(result.data)} rows, "
                  f"returning {len(candidates)} candidates "
                  f"(brands={intent_brands}, types={intent_types})")

            return candidates

        except Exception as e:
            print(f"[Pipeline] Error in session-intent retrieval: {e}")
            return []

    # =========================================================
    # Python-Level Hard Filters (Colors, Article Types)
    # =========================================================

    def _apply_color_filter(
        self,
        candidates: List[Candidate],
        include_colors: Optional[List[str]] = None,
        exclude_colors: Optional[List[str]] = None
    ) -> List[Candidate]:
        """
        Apply hard color filtering in Python.

        Args:
            candidates: List of candidates to filter
            include_colors: Colors that MUST be present (item must have at least one)
            exclude_colors: Colors that must NOT be present

        Returns:
            Filtered list of candidates
        """
        if not include_colors and not exclude_colors:
            return candidates

        # Normalize color names for comparison
        def normalize_color(color: str) -> str:
            return color.lower().strip()

        include_set = {normalize_color(c) for c in include_colors} if include_colors else None
        exclude_set = {normalize_color(c) for c in exclude_colors} if exclude_colors else None

        filtered = []
        for candidate in candidates:
            item_colors = [normalize_color(c) for c in (candidate.colors or [])]

            # Check exclude_colors - skip if ANY excluded color is present
            if exclude_set:
                has_excluded = any(
                    any(exc in item_color for exc in exclude_set)
                    for item_color in item_colors
                )
                if has_excluded:
                    continue

            # Check include_colors - skip if NONE of the included colors are present
            if include_set:
                has_included = any(
                    any(inc in item_color for inc in include_set)
                    for item_color in item_colors
                )
                if not has_included:
                    continue

            filtered.append(candidate)

        return filtered

    def _apply_brand_filter(
        self,
        candidates: List[Candidate],
        include_brands: Optional[List[str]] = None,
        exclude_brands: Optional[List[str]] = None
    ) -> List[Candidate]:
        """
        Apply hard brand filtering in Python.

        Args:
            candidates: List of candidates to filter
            include_brands: Brands that MUST match (item brand must be one of these)
            exclude_brands: Brands that must NOT match

        Returns:
            Filtered list of candidates
        """
        if not include_brands and not exclude_brands:
            return candidates

        # Normalize brand names for comparison
        def normalize_brand(brand: str) -> str:
            return brand.lower().strip()

        include_set = {normalize_brand(b) for b in include_brands} if include_brands else None
        exclude_set = {normalize_brand(b) for b in exclude_brands} if exclude_brands else None

        filtered = []
        for candidate in candidates:
            item_brand = normalize_brand(candidate.brand or '')

            # Check exclude_brands - skip if brand matches any excluded
            if exclude_set and item_brand:
                is_excluded = any(exc in item_brand or item_brand in exc for exc in exclude_set)
                if is_excluded:
                    continue

            # Check include_brands - skip if brand doesn't match any included
            if include_set:
                if not item_brand:
                    continue  # No brand info, skip
                is_included = any(inc in item_brand or item_brand in inc for inc in include_set)
                if not is_included:
                    continue

            filtered.append(candidate)

        return filtered

    def _apply_brand_tier_bucketing(
        self,
        candidates: List[Candidate],
        preferred_brands: List[str],
        target_size: int = 200,
        seed: str = "default",
        tier1_pct: float = 0.60,
        tier2_pct: float = 0.30,
    ) -> List[Candidate]:
        """
        Apply 3-tier brand bucketing to shape the candidate pool.

        Matches the Gradio demo's approach: instead of a flat SQL boost,
        explicitly allocate candidate slots by brand affinity tier.

        Tier 1 (60%): Preferred brands — user explicitly chose these
        Tier 2 (30%): Cluster-adjacent brands — same style cluster, not preferred
        Tier 3 (10%): Random — diversity / discovery from outside any cluster

        If any tier is short, remaining budget is backfilled from other tiers.
        Deterministic shuffle using seed for consistent ordering across pages.

        Args:
            candidates: Full candidate pool after hard filtering.
            preferred_brands: User's preferred brands from onboarding.
            target_size: Number of candidates to keep (default 200).
            seed: Deterministic seed for shuffle (typically user_id).
            tier1_pct: Fraction for preferred brands (default 0.60).
            tier2_pct: Fraction for cluster-adjacent brands (default 0.30).

        Returns:
            Bucketed candidate list with balanced brand distribution.
        """
        from recs.brand_clusters import get_cluster_adjacent_brands, get_cluster_for_item

        if not preferred_brands:
            return candidates

        pref_lower = {b.lower().strip() for b in preferred_brands}
        adjacent_lower = get_cluster_adjacent_brands(preferred_brands)

        # Bucket candidates into 3 tiers
        tier1: List[Candidate] = []  # preferred brands
        tier2: List[Candidate] = []  # cluster-adjacent brands
        tier3: List[Candidate] = []  # everything else (discovery)

        for c in candidates:
            brand_lower = (c.brand or "").lower().strip()
            if not brand_lower:
                tier3.append(c)
                continue

            if brand_lower in pref_lower:
                tier1.append(c)
            elif brand_lower in adjacent_lower:
                tier2.append(c)
            else:
                tier3.append(c)

        # Deterministic shuffle for consistent ordering
        rng = random.Random(hash(seed))
        rng.shuffle(tier1)
        rng.shuffle(tier2)
        rng.shuffle(tier3)

        # If the pool is already smaller than target, keep everything
        total = len(candidates)
        if total <= target_size:
            # Still shuffle to mix tiers, but keep all items
            combined = tier1 + tier2 + tier3
            rng.shuffle(combined)
            print(f"[Pipeline] Brand bucketing: pool ({total}) <= target ({target_size}), "
                  f"keeping all (T1={len(tier1)}, T2={len(tier2)}, T3={len(tier3)})")
            return combined

        # Allocate budgets: 60% / 30% / 10%
        budget1 = min(len(tier1), int(target_size * tier1_pct))
        budget2 = min(len(tier2), int(target_size * tier2_pct))
        budget3 = target_size - budget1 - budget2

        selected = tier1[:budget1] + tier2[:budget2] + tier3[:max(0, budget3)]

        # Backfill if short (e.g., not enough cluster-adjacent brands)
        if len(selected) < target_size:
            remaining = tier1[budget1:] + tier2[budget2:] + tier3[max(0, budget3):]
            rng.shuffle(remaining)
            selected.extend(remaining[:target_size - len(selected)])

        # Final shuffle to interleave tiers
        rng.shuffle(selected)

        actual_t1 = sum(1 for c in selected if (c.brand or "").lower().strip() in pref_lower)
        actual_t2 = sum(1 for c in selected if (c.brand or "").lower().strip() in adjacent_lower)
        actual_t3 = len(selected) - actual_t1 - actual_t2
        print(f"[Pipeline] Brand bucketing: {total} -> {len(selected)} candidates "
              f"(T1={actual_t1}/{budget1}, T2={actual_t2}/{budget2}, T3={actual_t3}/{max(0, budget3)})")

        return selected

    def _apply_article_type_filter(
        self,
        candidates: List[Candidate],
        article_types: Optional[List[str]] = None
    ) -> List[Candidate]:
        """
        Apply article type filtering using name/category matching.

        Uses TYPE_MAPPINGS to expand user-friendly type names to database values,
        then matches against item name and category.

        Args:
            candidates: List of candidates to filter
            article_types: List of article types to include (e.g., ['knitwear', 'jeans', 'sweatpants'])

        Returns:
            Filtered list of candidates
        """
        if not article_types:
            return candidates

        from recs.candidate_selection import TYPE_MAPPINGS

        # Expand article_types using TYPE_MAPPINGS
        expanded_types = set()
        for article_type in article_types:
            type_key = article_type.lower().replace(' ', '-')
            if type_key in TYPE_MAPPINGS:
                expanded_types.update(TYPE_MAPPINGS[type_key])
            else:
                expanded_types.add(article_type.lower())

        # Also add the original types for direct matching
        for t in article_types:
            expanded_types.add(t.lower())

        # Keywords to look for in item names
        # Map expanded types to search keywords
        search_keywords = set()
        for t in expanded_types:
            # Add the type itself
            search_keywords.add(t.lower())
            # Add variations
            search_keywords.add(t.lower().replace('-', ' '))
            search_keywords.add(t.lower().replace(' ', ''))

        # Special keyword mappings for common types
        keyword_mappings = {
            'knitwear': ['knit', 'sweater', 'cardigan', 'pullover', 'turtleneck', 'jumper'],
            'sweaters': ['sweater', 'knit', 'pullover', 'jumper'],
            'cardigans': ['cardigan', 'cardi'],
            'turtlenecks': ['turtleneck', 'turtle neck', 'mock neck'],
            'sweatpants': ['sweatpant', 'jogger', 'track pant', 'jogging'],
            'joggers': ['jogger', 'sweatpant', 'track pant'],
            't-shirts': ['t-shirt', 'tee', 'tshirt'],
            'tees': ['tee', 't-shirt', 'tshirt'],
            'tank tops': ['tank', 'cami', 'camisole', 'sleeveless top'],
            'jeans': ['jean', 'denim pant'],
            'leggings': ['legging', 'tight'],
            'hoodies': ['hoodie', 'hoody', 'hood'],
            'blouses': ['blouse'],
            'crop tops': ['crop top', 'cropped top'],
            'bodysuits': ['bodysuit', 'body suit'],
        }

        for t in expanded_types:
            if t in keyword_mappings:
                search_keywords.update(keyword_mappings[t])

        filtered = []
        for candidate in candidates:
            name_lower = (candidate.name or '').lower()
            category_lower = (candidate.category or '').lower()
            article_type_lower = (candidate.article_type or '').lower()

            # Check if any keyword matches
            matches = False

            # Direct article_type match (if populated)
            if article_type_lower:
                for keyword in search_keywords:
                    if keyword in article_type_lower:
                        matches = True
                        break

            # Name-based matching
            if not matches:
                for keyword in search_keywords:
                    if keyword in name_lower:
                        matches = True
                        break

            # Category-based matching (looser)
            if not matches:
                for keyword in search_keywords:
                    if keyword in category_lower:
                        matches = True
                        break

            if matches:
                filtered.append(candidate)

        return filtered

    # NOTE: _apply_attribute_filters() has been removed.
    # All attribute filtering now happens in SQL via LEFT JOIN to product_attributes.
    # Only rise filter remains in Python (no product_attributes column for rise).

    # =========================================================
    # Occasion Filtering (using product_attributes.occasions)
    # =========================================================

    def _filter_by_occasions(
        self,
        candidates: List[Candidate],
        required_occasions: Optional[List[str]]
    ) -> List[Candidate]:
        """
        Filter candidates by occasion using direct array membership.

        Much simpler than the old CLIP-score-based occasion gate.
        Uses product_attributes.occasions array for direct matching.

        Args:
            candidates: List of candidates to filter
            required_occasions: List of occasions to filter by (e.g., ['office'])

        Returns:
            Filtered list of candidates that have at least one matching occasion
        """
        if not required_occasions:
            return candidates

        # Expand user occasions to match DB values
        expanded_occasions = expand_occasions(required_occasions)
        required_set = set(o.lower() for o in expanded_occasions)

        def has_occasion(c: Candidate) -> bool:
            if not c.occasions:
                return False  # No attributes = excluded when filtering by occasion
            item_occasions = set(o.lower() for o in c.occasions)
            return bool(required_set & item_occasions)

        return [c for c in candidates if has_occasion(c)]

    # =========================================================
    # Diversity Constraints
    # =========================================================

    def _apply_diversity(self, candidates: List[Candidate], user_state: UserState) -> List[Candidate]:
        """
        Apply category diversity constraints.

        Ensures no more than MAX_PER_CATEGORY items from any single category.
        Uses shared filter_utils for consistent diversity logic.
        """
        # Get user's selected categories
        user_categories = []
        if user_state.onboarding_profile and user_state.onboarding_profile.categories:
            user_categories = user_state.onboarding_profile.categories

        # Calculate dynamic limit based on user state
        # Uses shared config with pipeline-specific overrides
        diversity_config = DiversityConfig(
            default_limit=self.config.MAX_PER_CATEGORY,
            single_category_limit=self.config.DEFAULT_LIMIT,
        )

        max_per_cat = get_diversity_limit(
            user_categories=user_categories,
            has_taste_vector=user_state.taste_vector is not None,
            config=diversity_config
        )

        return apply_diversity_candidates(candidates, max_per_cat)

    # =========================================================
    # Exploration Injection
    # =========================================================

    def _inject_exploration(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Inject exploration items at random positions.

        Exploration items (source="exploration") are placed at random positions
        to allow discovery of new items beyond the user's usual preferences.
        """
        # Separate exploration from non-exploration
        exploration = [c for c in candidates if c.source == "exploration"]
        non_exploration = [c for c in candidates if c.source != "exploration"]

        if not exploration:
            return candidates

        # Calculate how many exploration items to inject
        total_target = len(non_exploration)
        exploration_count = int(total_target * self.config.EXPLORATION_RATE)
        exploration_to_inject = exploration[:exploration_count]

        if not exploration_to_inject:
            return non_exploration

        # Inject at random positions
        result = non_exploration.copy()
        for exp_item in exploration_to_inject:
            # Random position (not at very top)
            min_pos = max(5, int(len(result) * 0.1))  # At least position 5 or 10%
            max_pos = len(result)
            pos = random.randint(min_pos, max_pos)
            result.insert(pos, exp_item)

        return result

    # =========================================================
    # Reason Determination
    # =========================================================

    def _get_reason(self, candidate: Candidate, user_state: UserState) -> str:
        """Determine the reason string for a candidate."""
        if candidate.source == "exploration":
            return "explore"

        if user_state.state_type == UserStateType.WARM_USER and candidate.sasrec_score > 0:
            return "personalized"

        if candidate.embedding_score > 0.7:
            return "style_matched"

        if candidate.preference_score > 0.5:
            return "preference_matched"

        return "exploration"

    # =========================================================
    # Onboarding Profile Management
    # =========================================================

    def save_onboarding(self, profile: OnboardingProfile, gender: str = "female") -> Dict[str, Any]:
        """Save user's onboarding profile (legacy V2 format)."""
        return self.candidate_module.save_onboarding_profile(profile, gender=gender)

    def save_onboarding_v3(self, profile: OnboardingProfile, gender: str = "female") -> Dict[str, Any]:
        """
        Save user's V3 onboarding profile.

        V3 format uses:
        - Split sizes (top_sizes, bottom_sizes, outerwear_sizes)
        - Flat attribute preferences with category mappings
        - Simplified type preferences
        - style_persona
        - Simplified style discovery
        """
        return self.candidate_module.save_onboarding_profile_v3(profile, gender=gender)

    # =========================================================
    # Utility Methods
    # =========================================================

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration."""
        ranker_info = self.ranker.get_model_info()

        return {
            "pipeline_config": {
                "max_per_category": self.config.MAX_PER_CATEGORY,
                "exploration_rate": self.config.EXPLORATION_RATE,
                "default_limit": self.config.DEFAULT_LIMIT,
                "max_limit": self.config.MAX_LIMIT
            },
            "candidate_selection": {
                "primary_candidates": self.config.candidate_config.PRIMARY_CANDIDATES,
                "contextual_candidates": self.config.candidate_config.CONTEXTUAL_CANDIDATES,
                "exploration_candidates": self.config.candidate_config.EXPLORATION_CANDIDATES,
                "soft_weights": self.config.candidate_config.SOFT_WEIGHTS
            },
            "sasrec_ranker": ranker_info
        }


# =============================================================================
# Testing
# =============================================================================

def test_pipeline():
    """Test the recommendation pipeline."""
    print("=" * 70)
    print("Testing Recommendation Pipeline")
    print("=" * 70)

    # Test 1: Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = RecommendationPipeline(load_sasrec=True)
    info = pipeline.get_pipeline_info()
    print(f"   Max per category: {info['pipeline_config']['max_per_category']}")
    print(f"   Exploration rate: {info['pipeline_config']['exploration_rate']}")
    print(f"   SASRec loaded: {info['sasrec_ranker']['model_loaded']}")

    # Test 2: Get feed for cold start user
    print("\n2. Getting feed for cold start user...")
    cold_response = pipeline.get_feed(
        anon_id="test_cold_pipeline_user",
        gender="female",
        limit=10
    )
    print(f"   Strategy: {cold_response.strategy}")
    print(f"   Results: {len(cold_response.results)}")
    print(f"   User state: {cold_response.metadata.get('user_state_type')}")

    if cold_response.results:
        print(f"   Top result: {cold_response.results[0].name[:40]}...")
        print(f"     - Score: {cold_response.results[0].score:.3f}")
        print(f"     - Reason: {cold_response.results[0].reason}")

    # Test 3: Save onboarding profile
    print("\n3. Saving onboarding profile...")
    test_profile = OnboardingProfile(
        user_id="test_pipeline_user",
        categories=["tops", "dresses", "bottoms"],
        colors_to_avoid=["orange", "neon"],
        materials_to_avoid=["polyester"],
        style_directions=["minimal", "classic"],
        preferred_brands=["Zara", "Mango"]
    )
    save_result = pipeline.save_onboarding(test_profile)
    print(f"   Save status: {save_result.get('status')}")

    # Test 4: Get feed for user with onboarding
    print("\n4. Getting feed with onboarding profile...")
    onboard_response = pipeline.get_feed(
        anon_id="test_pipeline_user",
        gender="female",
        limit=10
    )
    print(f"   Strategy: {onboard_response.strategy}")
    print(f"   Results: {len(onboard_response.results)}")
    print(f"   Has onboarding: {onboard_response.metadata.get('has_onboarding')}")
    print(f"   By source: {onboard_response.metadata.get('by_source')}")

    # Test 5: Category distribution check
    print("\n5. Checking category distribution...")
    large_response = pipeline.get_feed(
        anon_id="test_pipeline_user",
        gender="female",
        limit=50
    )

    category_counts = defaultdict(int)
    for item in large_response.results:
        category_counts[item.category] += 1

    print(f"   Categories in results:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"   - {cat}: {count}")

    max_in_category = max(category_counts.values()) if category_counts else 0
    print(f"   Max in any category: {max_in_category} (limit: {pipeline.config.MAX_PER_CATEGORY})")

    # Test 6: Pagination
    print("\n6. Testing pagination...")
    page1 = pipeline.get_feed(anon_id="test_pipeline_user", gender="female", limit=10, offset=0)
    page2 = pipeline.get_feed(anon_id="test_pipeline_user", gender="female", limit=10, offset=10)

    print(f"   Page 1: {len(page1.results)} items, has_more={page1.metadata.get('has_more')}")
    print(f"   Page 2: {len(page2.results)} items, offset={page2.metadata.get('offset')}")

    if page1.results and page2.results:
        overlap = set(r.product_id for r in page1.results) & set(r.product_id for r in page2.results)
        print(f"   Overlap between pages: {len(overlap)} (should be 0)")

    print("\n" + "=" * 70)
    print("Pipeline test complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_pipeline()
