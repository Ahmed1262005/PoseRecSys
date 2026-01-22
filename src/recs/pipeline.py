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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from recs.models import (
    UserState,
    UserStateType,
    Candidate,
    OnboardingProfile,
    FeedRequest,
    FeedResponse,
    FeedItem,
    HardFilters,
)
from recs.candidate_selection import CandidateSelectionModule, CandidateSelectionConfig
from recs.sasrec_ranker import SASRecRanker, SASRecRankerConfig
from recs.session_state import SessionStateService


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
        if user_state.state_type == UserStateType.WARM_USER:
            strategy = "sasrec"
        elif user_state.state_type == UserStateType.TINDER_COMPLETE:
            strategy = "seed_vector"
        else:
            strategy = "trending"

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
            "candidates_after_diversity": len(diverse_candidates),
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
        page_size: int = 50
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
                from src.recs.models import OnboardingProfile
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
        if user_state.state_type == UserStateType.WARM_USER:
            strategy = "sasrec"
        elif user_state.state_type == UserStateType.TINDER_COMPLETE:
            strategy = "seed_vector"
        else:
            strategy = "trending"

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

        return {
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
                "candidates_after_diversity": len(diverse_candidates),
                "sasrec_available": self.ranker.model is not None,
                "seed_vector_available": user_state.taste_vector is not None,
                "has_onboarding": user_state.onboarding_profile is not None,
                "user_state_type": user_state.state_type.value,
                "by_source": dict(by_source),
                "endless_scroll": True
            }
        }

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
        cursor: Optional[str] = None,
        page_size: int = 50
    ) -> Dict[str, Any]:
        """
        Generate feed using keyset cursor for O(1) pagination.

        V2 Improvements over get_feed_endless():
        1. Uses keyset cursor (score, id) instead of exclusion arrays
        2. O(1) pagination regardless of page depth
        3. Feed versioning for stable ordering within session
        4. Still tracks seen items for authoritative dedup

        Args:
            user_id: Logged-in user UUID
            anon_id: Anonymous user ID
            session_id: Session ID (auto-generated if not provided)
            gender: Target gender
            categories: Optional broad category filter (tops, bottoms, dresses, outerwear)
            article_types: Optional specific article type filter (jeans, t-shirts, mini dresses, etc.)
            cursor: Base64 encoded keyset cursor (None for first page)
            page_size: Items per page (default 50)

        Returns:
            Dict with results, cursor, pagination, and session_id
        """
        from recs.session_state import KeysetCursor

        # Generate session_id if not provided (new session)
        is_new_session = session_id is None
        if is_new_session:
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
                from src.recs.models import OnboardingProfile
                user_state.onboarding_profile = OnboardingProfile(
                    user_id=user_state.user_id,
                    categories=categories
                )

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

        # Step 4: Load DB seen history first (to determine fetch size)
        db_seen_ids = self.candidate_module.get_user_seen_history(anon_id, user_id)
        db_history_count = len(db_seen_ids)

        # Calculate how many candidates to fetch
        # Need to fetch extra to account for exclusions done in Python
        # Fetch enough to skip past seen items while maintaining keyset cursor pagination
        fetch_size = page_size + db_history_count + 100
        fetch_size = min(fetch_size, 2000)  # Increased cap for large histories

        # Step 5: Get candidates using keyset cursor (or endless if we have exclude_ids)
        candidates = self.candidate_module.get_candidates_keyset(
            user_state,
            gender,
            cursor_score=cursor_score,
            cursor_id=cursor_id,
            page_size=fetch_size,
            exclude_ids=db_seen_ids if db_seen_ids else None,
            article_types=article_types
        )

        # Step 5: Rank candidates (lightweight re-scoring)
        ranked_candidates = self.ranker.rank_candidates(user_state, candidates)

        # Step 6: Apply diversity constraints
        diverse_candidates = self._apply_diversity(ranked_candidates, user_state)

        # Step 7: Get seen items and filter duplicates (session + DB history)
        # Load from session (current page views)
        session_seen_ids = self.session_service.get_seen_items(session_id)

        # db_seen_ids already loaded in Step 4
        # Combine both sources for permanent deduplication
        all_seen_ids = session_seen_ids | db_seen_ids
        filtered_candidates = [c for c in diverse_candidates if c.item_id not in all_seen_ids]

        # Step 7b: Deduplicate by image hash to remove same products across different brands
        filtered_candidates = self._deduplicate_by_image(filtered_candidates)

        # Step 8: Take top N for this page
        page_results = filtered_candidates[:page_size]

        # Step 9: Update session state
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

        # Step 10: Get next cursor
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
        if user_state.state_type == UserStateType.WARM_USER:
            strategy = "sasrec"
        elif user_state.state_type == UserStateType.TINDER_COMPLETE:
            strategy = "seed_vector"
        else:
            strategy = "trending"

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
                "source": candidate.source
            })

        # Build metadata
        by_source = defaultdict(int)
        for c in page_results:
            by_source[c.source] += 1

        return {
            "user_id": user_id or anon_id or "anonymous",
            "session_id": session_id,
            "cursor": next_cursor,
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
                "candidates_retrieved": len(candidates),
                "candidates_after_diversity": len(diverse_candidates),
                "candidates_after_dedup": len(filtered_candidates),
                "sasrec_available": self.ranker.model is not None,
                "seed_vector_available": user_state.taste_vector is not None,
                "has_onboarding": user_state.onboarding_profile is not None,
                "user_state_type": user_state.state_type.value,
                "by_source": dict(by_source),
                "keyset_pagination": True,
                "db_seen_history_count": db_history_count,
                "fetch_size_used": fetch_size,
                "feed_version": self.session_service.get_feed_version(session_id).version_id if self.session_service.get_feed_version(session_id) else None
            }
        }

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session info for debugging."""
        return self.session_service.get_session_info(session_id)

    def clear_session(self, session_id: str):
        """Clear a session (reset seen items)."""
        self.session_service.clear_session(session_id)

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
        brand names. This catches duplicates by extracting the image hash from URLs like:
        .../original_0_85a218f8.jpg -> hash is 85a218f8
        """
        import re

        def get_image_hash(url):
            if not url:
                return None
            match = re.search(r'original_\d+_([a-f0-9]+)\.', url)
            return match.group(1) if match else None

        seen_image_hashes = set()
        seen_name_brand = set()
        deduped = []

        for c in candidates:
            # Primary dedup: image hash (catches cross-brand duplicates)
            img_hash = get_image_hash(c.image_url)
            if img_hash and img_hash in seen_image_hashes:
                continue  # Skip - same image already shown

            # Secondary dedup: (name, brand) for products without matching image hash
            name_brand_key = (c.name, c.brand) if c.name and c.brand else None
            if name_brand_key and name_brand_key in seen_name_brand:
                continue  # Skip - same name+brand already shown

            deduped.append(c)
            if img_hash:
                seen_image_hashes.add(img_hash)
            if name_brand_key:
                seen_name_brand.add(name_brand_key)

        return deduped

    # =========================================================
    # Diversity Constraints
    # =========================================================

    def _apply_diversity(self, candidates: List[Candidate], user_state: UserState) -> List[Candidate]:
        """
        Apply category diversity constraints.

        Ensures no more than MAX_PER_CATEGORY items from any single category.
        When user has few categories selected, allows more items per category.
        """
        # Calculate dynamic limit based on user's selected categories
        user_categories = []
        if user_state.onboarding_profile and user_state.onboarding_profile.categories:
            user_categories = user_state.onboarding_profile.categories

        # If user has 1-2 categories, allow more items per category
        # Default limit of 8 is for users with many categories
        if len(user_categories) <= 1:
            max_per_cat = self.config.DEFAULT_LIMIT  # Allow up to 50 items for single category
        elif len(user_categories) == 2:
            max_per_cat = 25  # 25 per category for 2 categories
        elif len(user_categories) == 3:
            max_per_cat = 16  # 16 per category for 3 categories
        else:
            max_per_cat = self.config.MAX_PER_CATEGORY  # Default 8 for 4+ categories

        result = []
        category_counts = defaultdict(int)

        for candidate in candidates:
            cat = candidate.broad_category or candidate.category or "unknown"

            if category_counts[cat] < max_per_cat:
                result.append(candidate)
                category_counts[cat] += 1

        return result

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

        return "trending"

    # =========================================================
    # Onboarding Profile Management
    # =========================================================

    def save_onboarding(self, profile: OnboardingProfile, gender: str = "female") -> Dict[str, Any]:
        """Save user's onboarding profile."""
        return self.candidate_module.save_onboarding_profile(profile, gender=gender)

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
