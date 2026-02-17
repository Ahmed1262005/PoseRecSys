"""
Women's Fashion Text Search using FashionCLIP + Supabase pgvector

Provides text-to-image search for women's fashion items using:
- FashionCLIP for text encoding (512-dim vectors)
- Supabase pgvector for similarity search in the products database
"""
import os
import sys
import threading
import numpy as np
from typing import List, Dict, Optional, Set, Any, Tuple
from collections import defaultdict

from dotenv import load_dotenv
from supabase import create_client, Client

from core.logging import get_logger
from core.utils import filter_gallery_images

load_dotenv()

logger = get_logger(__name__)

# Import occasion expansion for simple array filtering
from recs.candidate_selection import expand_occasions
from recs.models import Candidate
from recs.filter_utils import (
    extract_image_hash,
    deduplicate_dicts,
    apply_diversity_dicts,
    apply_soft_scoring,
    SoftScoringWeights,
    DEFAULT_SOFT_WEIGHTS,
)
from recs.candidate_factory import candidate_from_dict


class WomenSearchEngine:
    """
    Text search engine for women's fashion using FashionCLIP + Supabase pgvector.

    Encodes text queries with FashionCLIP and searches the products database
    using pgvector similarity search.
    """

    def __init__(self):
        """Initialize with Supabase client."""
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")

        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

        self.supabase: Client = create_client(url, key)

        # Lazy-load CLIP model (only when search is called)
        self._model = None
        self._processor = None
        self._model_lock = threading.Lock()

        # Lazy-load session service
        self._session_service = None

    # =========================================================================
    # Session State Management
    # =========================================================================

    @property
    def session_service(self):
        """Lazy-load session service."""
        if self._session_service is None:
            from recs.session_state import SessionStateService
            self._session_service = SessionStateService(backend="auto")
        return self._session_service

    def _get_session_seen_ids(self, session_id: Optional[str]) -> Set[str]:
        """Get product IDs already shown in this session."""
        if not session_id:
            return set()
        return self.session_service.get_seen_items(session_id)

    def _update_session_seen(self, session_id: str, product_ids: List[str]):
        """Add shown product IDs to session state."""
        if session_id and product_ids:
            self.session_service.add_seen_items(session_id, product_ids)

    @staticmethod
    def generate_session_id() -> str:
        """Generate a new session ID."""
        from recs.session_state import SessionStateService
        return SessionStateService.generate_session_id()

    # =========================================================================
    # User Profile Loading
    # =========================================================================

    def load_user_profile(
        self,
        user_id: Optional[str] = None,
        anon_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Public interface. Delegates to _load_user_profile."""
        return self._load_user_profile(user_id=user_id, anon_id=anon_id)

    def _load_user_profile(
        self,
        user_id: Optional[str] = None,
        anon_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load user's complete onboarding profile from Supabase.

        Returns a flat dict whose keys match ``OnboardingProfile`` field names
        so it can be passed directly to ``ProfileScorer.score_item()``.

        For backward compatibility the legacy nested ``hard_filters``,
        ``soft_prefs``, ``type_prefs`` and ``taste_vector`` keys are also
        included at the top level.
        """
        _EMPTY: Dict[str, Any] = {
            "hard_filters": {}, "soft_prefs": {}, "type_prefs": {},
            "taste_vector": None,
        }
        if not user_id and not anon_id:
            return _EMPTY

        try:
            # Build query based on user_id or anon_id
            query = self.supabase.table('user_onboarding_profiles').select(
                # Core filters
                'categories',
                'colors_to_avoid',
                'materials_to_avoid',
                'brands_to_avoid',
                # Lifestyle filters
                'styles_to_avoid',
                'occasions',
                'patterns_liked',
                'patterns_avoided',
                'modesty',
                # Soft preferences
                'preferred_fits',
                'preferred_sleeves',
                'preferred_lengths',
                'preferred_lengths_dresses',
                'preferred_rises',
                'preferred_brands',
                'style_persona',
                # Brand behaviour
                'brand_openness',
                # Type preferences
                'top_types',
                'bottom_types',
                'dress_types',
                'outerwear_types',
                # V3 category-aware mappings
                'fit_category_mapping',
                'sleeve_category_mapping',
                'length_category_mapping',
                # Price range
                'global_min_price',
                'global_max_price',
                # Identity
                'birthdate',
                # Style discovery
                'taste_vector',
            )

            if user_id:
                query = query.eq('user_id', user_id)
            else:
                query = query.eq('anon_id', anon_id)

            result = query.limit(1).execute()

            if not result.data:
                return _EMPTY

            profile = result.data[0]

            # ----------------------------------------------------------
            # Flat keys — match OnboardingProfile / ProfileScorer fields
            # ----------------------------------------------------------
            flat: Dict[str, Any] = {
                # Brand
                "preferred_brands": profile.get('preferred_brands') or [],
                "brand_openness": profile.get('brand_openness'),
                # Style
                "style_persona": profile.get('style_persona') or [],
                # Attribute preferences
                "preferred_fits": profile.get('preferred_fits') or [],
                "fit_category_mapping": profile.get('fit_category_mapping') or [],
                "preferred_sleeves": profile.get('preferred_sleeves') or [],
                "sleeve_category_mapping": profile.get('sleeve_category_mapping') or [],
                "preferred_lengths": profile.get('preferred_lengths') or [],
                "length_category_mapping": profile.get('length_category_mapping') or [],
                "preferred_lengths_dresses": profile.get('preferred_lengths_dresses') or [],
                "preferred_necklines": [],  # no dedicated DB column yet
                "preferred_rises": profile.get('preferred_rises') or [],
                # Type preferences
                "top_types": profile.get('top_types') or [],
                "bottom_types": profile.get('bottom_types') or [],
                "dress_types": profile.get('dress_types') or [],
                "outerwear_types": profile.get('outerwear_types') or [],
                # Patterns
                "patterns_liked": profile.get('patterns_liked') or [],
                "patterns_avoided": profile.get('patterns_avoided') or [],
                # Occasions
                "occasions": profile.get('occasions') or [],
                # Avoidances
                "colors_to_avoid": profile.get('colors_to_avoid') or [],
                "styles_to_avoid": profile.get('styles_to_avoid') or [],
                # Coverage exclusion flags — derived from styles_to_avoid
                # (no dedicated DB columns; ProfileScorer._ProfileFields
                # also handles styles_to_avoid -> exclusion mapping)
                "no_crop": False,
                "no_revealing": False,
                "no_sleeveless": False,
                "no_deep_necklines": False,
                "no_tanks": False,
                # Price
                "global_min_price": profile.get('global_min_price'),
                "global_max_price": profile.get('global_max_price'),
                # Identity
                "birthdate": profile.get('birthdate'),
            }

            # ----------------------------------------------------------
            # Legacy nested keys — backward compatibility for callers
            # that still read hard_filters / soft_prefs / type_prefs
            # ----------------------------------------------------------
            flat["hard_filters"] = {
                "categories": profile.get('categories'),
                "exclude_colors": profile.get('colors_to_avoid'),
                "exclude_materials": profile.get('materials_to_avoid'),
                "exclude_brands": profile.get('brands_to_avoid'),
                "exclude_styles": profile.get('styles_to_avoid'),
                "include_occasions": profile.get('occasions'),
                "include_patterns": profile.get('patterns_liked'),
                "exclude_patterns": profile.get('patterns_avoided'),
            }
            flat["soft_prefs"] = {
                "preferred_fits": flat["preferred_fits"],
                "preferred_sleeves": flat["preferred_sleeves"],
                "preferred_lengths": flat["preferred_lengths"],
                "preferred_lengths_dresses": flat["preferred_lengths_dresses"],
                "preferred_rises": flat["preferred_rises"],
                "preferred_brands": flat["preferred_brands"],
                "style_persona": flat["style_persona"],
                "preferred_necklines": flat["preferred_necklines"],
            }
            flat["type_prefs"] = {
                "top_types": flat["top_types"],
                "bottom_types": flat["bottom_types"],
                "dress_types": flat["dress_types"],
                "outerwear_types": flat["outerwear_types"],
            }
            flat["taste_vector"] = profile.get('taste_vector')

            return flat
        except Exception as e:
            logger.warning("Error loading user profile", error=str(e))
            return _EMPTY

    # =========================================================================
    # Soft Preference Scoring
    # =========================================================================

    def _apply_soft_scoring(
        self,
        results: List[Dict],
        soft_prefs: Dict[str, Any],
        type_prefs: Dict[str, Any],
        user_hard: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """
        Apply soft preference scoring to boost/demote items based on user preferences.
        Uses shared filter_utils for consistent scoring logic.
        """
        return apply_soft_scoring(results, soft_prefs, type_prefs, user_hard, DEFAULT_SOFT_WEIGHTS)

    # =========================================================================
    # Diversity Constraints
    # =========================================================================

    def _apply_diversity(
        self,
        results: List[Dict],
        max_per_category: int = 15
    ) -> List[Dict]:
        """
        Apply category diversity constraints.
        Uses shared filter_utils for consistent diversity logic.
        """
        return apply_diversity_dicts(results, max_per_category)

    # =========================================================================
    # Occasion Gate Integration
    # =========================================================================

    def _apply_occasion_gate(
        self,
        results: List[Dict],
        include_occasions: Optional[List[str]] = None,
        verbose: bool = False
    ) -> Tuple[List[Dict], Dict[str, int]]:
        """
        Apply occasion filter to search results using product_attributes.occasions.

        Uses simple array membership check instead of CLIP-based scoring.

        Args:
            results: List of search result dicts
            include_occasions: Occasions to filter by (e.g., ['office'])
            verbose: Print debug info

        Returns:
            (filtered_results, blocked_stats) - stats include filtered count
        """
        if not include_occasions:
            return results, {}

        # Expand user occasions to match DB values
        expanded_occasions = expand_occasions(include_occasions)
        required_set = set(o.lower() for o in expanded_occasions)

        filtered = []
        blocked_count = 0

        for item in results:
            # Get occasions from the item (product_attributes.occasions)
            item_occasions = item.get('occasions') or []

            # Check for intersection
            if item_occasions:
                item_occasions_set = set(o.lower() for o in item_occasions)
                if required_set & item_occasions_set:
                    filtered.append(item)
                else:
                    blocked_count += 1
            else:
                # No occasion data - exclude when filtering by occasion
                blocked_count += 1

        if verbose and blocked_count > 0:
            logger.debug("Occasion filter applied", before=len(results), after=len(filtered), blocked=blocked_count)

        return filtered, {"blocked": blocked_count}

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _empty_response(
        self,
        query: str,
        page: int,
        page_size: int,
        filters_applied: Dict,
        session_id: Optional[str] = None,
        use_hybrid_search: bool = True
    ) -> Dict:
        """Return empty response structure."""
        return {
            "query": query,
            "results": [],
            "count": 0,
            "filters_applied": filters_applied,
            "user_prefs_applied": False,
            "hybrid_search": use_hybrid_search,
            "session_id": session_id,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "has_more": False
            }
        }

    def _load_model(self):
        """Lazy load FashionCLIP model (thread-safe, double-check locking)."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    import torch
                    from transformers import CLIPProcessor, CLIPModel

                    logger.info("Loading FashionCLIP model...")
                    model = CLIPModel.from_pretrained('patrickjohncyh/fashion-clip')
                    processor = CLIPProcessor.from_pretrained('patrickjohncyh/fashion-clip')

                    # Move to GPU if available
                    if torch.cuda.is_available():
                        model = model.cuda()

                    model.eval()
                    # Assign last to avoid partial initialization visible to other threads
                    self._processor = processor
                    self._model = model
                    logger.info("FashionCLIP model loaded")

    def encode_text(self, query: str) -> np.ndarray:
        """Encode text query to embedding vector."""
        import torch
        self._load_model()

        with torch.no_grad():
            inputs = self._processor(text=[query], return_tensors='pt', padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            emb = self._model.get_text_features(**inputs)
            # transformers >=5.x: get_text_features returns BaseModelOutputWithPooling
            # instead of a raw tensor. The correct embedding is in pooler_output
            # (which is the EOS token projected through text_projection).
            # NOTE: last_hidden_state[:, 0, :] (CLS token) is NOT the right embedding
            # -- it produces identical vectors for all queries.
            if isinstance(emb, torch.Tensor):
                # transformers <5.x: direct tensor (already projected)
                pass
            elif hasattr(emb, 'pooler_output') and emb.pooler_output is not None:
                # transformers >=5.x: use the pooled + projected output
                emb = emb.pooler_output
            elif hasattr(emb, 'text_embeds') and emb.text_embeds is not None:
                emb = emb.text_embeds
            else:
                raise ValueError(
                    f"Unexpected return type from get_text_features: {type(emb)}. "
                    f"Available attrs: {[a for a in dir(emb) if not a.startswith('_')]}"
                )
            # Normalize
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.cpu().numpy().flatten()

    def search_all(
        self,
        query: str,
        max_results: int = 10000,
        categories: Optional[List[str]] = None
    ) -> Dict:
        """
        Search and return ALL matching results (auto-paginate through Supabase 1000 limit).

        Args:
            query: Text description
            max_results: Maximum results to fetch (default 10000)
            categories: Optional category filter

        Returns:
            Dict with all results
        """
        all_results = []
        page = 1
        page_size = 1000  # Supabase max

        # Encode once
        text_embedding = self.encode_text(query)
        text_embedding = text_embedding.astype('float32').tolist()
        vector_str = f"[{','.join(map(str, text_embedding))}]"

        while len(all_results) < max_results:
            offset = (page - 1) * page_size

            try:
                result = self.supabase.rpc('text_search_products', {
                    'query_embedding': vector_str,
                    'match_count': page_size,
                    'match_offset': offset,
                    'filter_category': categories[0] if categories else None
                }).execute()

                if not result.data or len(result.data) == 0:
                    break

                for row in result.data:
                    all_results.append({
                        "product_id": row.get('product_id'),
                        "similarity": float(row.get('similarity', 0)),
                        "name": row.get('name', ''),
                        "brand": row.get('brand', ''),
                        "category": row.get('category', ''),
                        "broad_category": row.get('broad_category', ''),
                        "price": float(row.get('price', 0) or 0),
                        "image_url": row.get('primary_image_url', ''),
                        "gallery_images": filter_gallery_images(row.get('gallery_images', []) or []),
                        "colors": row.get('colors', []) or [],
                        "materials": row.get('materials', []) or [],
                    })

                if len(result.data) < page_size:
                    break

                page += 1

            except Exception as e:
                logger.warning("Error fetching page", page=page, error=str(e))
                break

        return {
            "query": query,
            "results": all_results[:max_results],
            "count": len(all_results[:max_results]),
            "total_fetched": len(all_results)
        }

    def _get_image_hash(self, url: str) -> Optional[str]:
        """Extract image hash from URL. Uses shared filter_utils for consistent logic."""
        return extract_image_hash(url)

    def _deduplicate_results(self, results: List[Dict], limit: Optional[int] = None) -> List[Dict]:
        """
        Remove duplicate products based on image hash and (name, brand).
        Uses shared filter_utils for consistent deduplication logic.
        """
        return deduplicate_dicts(results, limit=limit)

    def search(
        self,
        query: str,
        page: int = 1,
        page_size: int = 50,
        categories: Optional[List[str]] = None
    ) -> Dict:
        """
        Search for items matching text query with pagination.

        Args:
            query: Text description (e.g., "flowy blue dress")
            page: Page number (1-indexed)
            page_size: Number of results per page (max 200)
            categories: Optional list of categories to filter by

        Returns:
            Dict with results and pagination info
        """
        # Validate pagination (no upper limit)
        page = max(1, page)
        page_size = max(1, page_size)

        # Fetch extra results to account for duplicates being filtered out
        # For page 1, we fetch 3x. For later pages, we need to re-fetch and skip
        fetch_multiplier = 3
        fetch_count = page_size * fetch_multiplier

        # Encode text query with FashionCLIP (fast GPU inference)
        text_embedding = self.encode_text(query)
        text_embedding = text_embedding.astype('float32').tolist()

        # Convert to pgvector string format
        vector_str = f"[{','.join(map(str, text_embedding))}]"

        try:
            # For pagination with deduplication, we need to fetch from start and deduplicate
            # Then slice to the requested page
            total_needed = page * page_size
            fetch_count = total_needed * fetch_multiplier

            result = self.supabase.rpc('text_search_products', {
                'query_embedding': vector_str,
                'match_count': fetch_count,
                'match_offset': 0,
                'filter_category': categories[0] if categories else None
            }).execute()

            if not result.data:
                return {
                    "query": query,
                    "results": [],
                    "count": 0,
                    "pagination": {
                        "page": page,
                        "page_size": page_size,
                        "has_more": False
                    }
                }

            # Format results
            all_results = []
            for row in result.data:
                all_results.append({
                    "product_id": row.get('product_id'),
                    "similarity": float(row.get('similarity', 0)),
                    "name": row.get('name', ''),
                    "brand": row.get('brand', ''),
                    "category": row.get('category', ''),
                    "broad_category": row.get('broad_category', ''),
                    "price": float(row.get('price', 0) or 0),
                    "image_url": row.get('primary_image_url', ''),
                    "gallery_images": filter_gallery_images(row.get('gallery_images', []) or []),
                    "colors": row.get('colors', []) or [],
                    "materials": row.get('materials', []) or [],
                })

            # Deduplicate results
            unique_results = self._deduplicate_results(all_results)

            # Paginate the deduplicated results
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            results = unique_results[start_idx:end_idx]

            return {
                "query": query,
                "results": results,
                "count": len(results),
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "has_more": len(results) == page_size
                }
            }

        except Exception as e:
            logger.error("Error in text search", error=str(e))
            return {
                "query": query,
                "results": [],
                "count": 0,
                "error": str(e),
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "has_more": False
                }
            }

    def search_with_filters(
        self,
        query: str,
        page: int = 1,
        page_size: int = 50,
        # User identification (NEW)
        user_id: Optional[str] = None,
        anon_id: Optional[str] = None,
        session_id: Optional[str] = None,
        # Existing filters (override user profile if provided)
        categories: Optional[List[str]] = None,
        exclude_colors: Optional[List[str]] = None,
        exclude_materials: Optional[List[str]] = None,
        exclude_brands: Optional[List[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        exclude_product_ids: Optional[List[str]] = None,
        # Extended filters
        article_types: Optional[List[str]] = None,
        include_colors: Optional[List[str]] = None,
        include_materials: Optional[List[str]] = None,
        include_brands: Optional[List[str]] = None,
        fits: Optional[List[str]] = None,
        occasions: Optional[List[str]] = None,
        exclude_styles: Optional[List[str]] = None,
        patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        # NEW options
        apply_user_prefs: bool = True,
        apply_diversity: bool = False,
        max_per_category: int = 15,
        use_hybrid_search: bool = True,
        semantic_weight: float = 0.90,  # UPDATED: was 0.7, reduced keyword influence
        keyword_weight: float = 0.10,   # UPDATED: was 0.3, prevents override of semantic
    ) -> Dict:
        """
        Search with comprehensive filters, user preferences, and hybrid search.

        Pipeline:
        1. Load user profile (if user_id/anon_id provided)
        2. Merge request filters with profile filters (request takes precedence)
        3. Get session seen IDs
        4. Execute hybrid search (semantic + keyword)
        5. Filter out session-seen items
        6. Deduplicate by image hash
        7. Apply soft preference scoring
        8. Apply diversity constraints (optional)
        9. Paginate results
        10. Update session state

        Args:
            query: Text description
            page: Page number (1-indexed)
            page_size: Results per page
            user_id: User UUID for profile loading
            anon_id: Anonymous ID for profile loading
            session_id: Session ID for deduplication across pages
            categories: Broad category filter (tops, bottoms, dresses, outerwear)
            exclude_colors: Colors to exclude
            exclude_materials: Materials to exclude
            exclude_brands: Brands to exclude
            min_price: Minimum price
            max_price: Maximum price
            exclude_product_ids: Product IDs to exclude
            article_types: Specific article types (jeans, t-shirts, midi dresses)
            include_colors: Colors to include (positive filter)
            include_materials: Materials to include (positive filter)
            include_brands: Brands to include (positive filter)
            fits: Fit types to include (slim, regular, relaxed, oversized)
            occasions: Occasions to filter by (casual, office, evening, beach)
            exclude_styles: Styles to exclude (sheer, cutouts, backless, etc.)
            patterns: Patterns to include (solid, stripes, floral, plaid)
            exclude_patterns: Patterns to exclude
            apply_user_prefs: Whether to load and apply user profile
            apply_diversity: Whether to limit items per category
            max_per_category: Max items per category when diversity is enabled
            use_hybrid_search: Enable keyword matching for brand/name
            semantic_weight: Weight for CLIP similarity (0-1)
            keyword_weight: Weight for keyword match boost (0-1)

        Returns:
            Dict with filtered results and pagination
        """
        page = max(1, page)
        page_size = max(1, page_size)

        # =========================================
        # STEP 1: Load user profile
        # =========================================
        user_profile = {"hard_filters": {}, "soft_prefs": {}, "type_prefs": {}, "taste_vector": None}
        if apply_user_prefs and (user_id or anon_id):
            user_profile = self._load_user_profile(user_id, anon_id)

        # User profile is used ONLY for soft scoring (boost/demote), NOT hard filters
        user_hard = user_profile.get('hard_filters', {})
        soft_prefs = user_profile.get('soft_prefs', {})
        type_prefs = user_profile.get('type_prefs', {})

        # =========================================
        # STEP 2: Build filters (request params only - no user profile hard filters)
        # =========================================
        # User profile preferences are applied as soft boosts, not hard filters
        final_filters = {
            'filter_categories': categories,
            'exclude_colors': exclude_colors,
            'exclude_materials': exclude_materials,
            'exclude_brands': exclude_brands,
            'exclude_styles': exclude_styles,
            'include_occasions': occasions,
            'include_patterns': patterns,
            'exclude_patterns': exclude_patterns,
            'min_price': min_price,
            'max_price': max_price,
            'filter_article_types': article_types,
            'include_colors': include_colors,
            'include_materials': include_materials,
            'include_brands': include_brands,
            'include_fits': fits,
            'exclude_product_ids': exclude_product_ids,
        }

        # =========================================
        # STEP 3: Get session seen IDs
        # =========================================
        seen_ids = set()
        if session_id:
            seen_ids = self._get_session_seen_ids(session_id)

        # =========================================
        # STEP 4: Execute search
        # =========================================
        fetch_multiplier = 3
        fetch_count = (page * page_size + len(seen_ids)) * fetch_multiplier

        text_embedding = self.encode_text(query)
        text_embedding = text_embedding.astype('float32').tolist()
        vector_str = f"[{','.join(map(str, text_embedding))}]"

        try:
            # Choose search function based on hybrid setting
            if use_hybrid_search:
                rpc_name = 'text_search_products_hybrid'
                rpc_params = {
                    'query_embedding': vector_str,
                    'query_text': query,
                    'match_count': fetch_count,
                    'match_offset': 0,
                    'semantic_weight': semantic_weight,
                    'keyword_weight': keyword_weight,
                }
            else:
                rpc_name = 'text_search_products_filtered'
                rpc_params = {
                    'query_embedding': vector_str,
                    'match_count': fetch_count,
                    'match_offset': 0,
                }

            # Add filters to params (only non-None and non-empty values)
            for key, value in final_filters.items():
                if value is not None and value != [] and value != '':
                    rpc_params[key] = value

            result = self.supabase.rpc(rpc_name, rpc_params).execute()

            if not result.data:
                return self._empty_response(query, page, page_size, final_filters, session_id, use_hybrid_search)

            # =========================================
            # STEP 5: Format and filter results
            # =========================================
            all_results = []
            for row in result.data:
                # Preserve raw CLIP score as base_similarity for transparency
                raw_similarity = float(row.get('similarity') or 0)
                combined_score = float(row.get('combined_score') or raw_similarity)

                all_results.append({
                    "product_id": row.get('product_id'),
                    "base_similarity": raw_similarity,  # Raw CLIP score before any boosts
                    "similarity": combined_score,       # Score used for ranking (may include SQL-level boosts)
                    "keyword_match": row.get('keyword_match', False),
                    "brand_match": row.get('brand_match', False),
                    "exact_match": row.get('exact_match', False),  # NEW: Exact name match flag
                    "name": row.get('name', ''),
                    "brand": row.get('brand', ''),
                    "category": row.get('category', ''),
                    "broad_category": row.get('broad_category', ''),
                    "article_type": row.get('article_type', ''),
                    "price": float(row.get('price', 0) or 0),
                    "image_url": row.get('primary_image_url', ''),
                    "gallery_images": filter_gallery_images(row.get('gallery_images', []) or []),
                    "colors": row.get('colors', []) or [],
                    "materials": row.get('materials', []) or [],
                    "fit": row.get('fit'),
                    "length": row.get('length'),
                    "sleeve": row.get('sleeve'),
                    # product_attributes fields for occasion filtering
                    "occasions": row.get('occasions', []) or [],
                    "style_tags": row.get('style_tags', []) or [],
                    "pattern": row.get('pattern'),
                })

            # Filter out session-seen items
            if seen_ids:
                all_results = [r for r in all_results if r['product_id'] not in seen_ids]

            # =========================================
            # STEP 6: Deduplicate
            # =========================================
            unique_results = self._deduplicate_results(all_results)

            # =========================================
            # STEP 7: Apply soft preference scoring (boosts + demotes)
            # =========================================
            if soft_prefs or type_prefs or user_hard:
                unique_results = self._apply_soft_scoring(unique_results, soft_prefs, type_prefs, user_hard)

            # =========================================
            # STEP 8: Apply diversity constraints
            # =========================================
            if apply_diversity:
                unique_results = self._apply_diversity(unique_results, max_per_category)

            # =========================================
            # STEP 8.5: Apply occasion filter (simple array membership)
            # =========================================
            occasion_filter_stats = {}
            if occasions:
                pre_filter_count = len(unique_results)
                unique_results, occasion_filter_stats = self._apply_occasion_gate(
                    unique_results, occasions, verbose=True
                )
                if pre_filter_count != len(unique_results):
                    logger.debug("Occasion filter in search_with_filters", occasions=occasions, before=pre_filter_count, after=len(unique_results))

            # =========================================
            # STEP 9: Paginate
            # =========================================
            start_idx = (page - 1) * page_size
            results = unique_results[start_idx:start_idx + page_size]

            # =========================================
            # STEP 10: Update session state
            # =========================================
            if session_id and results:
                self._update_session_seen(session_id, [r['product_id'] for r in results])

            # Check if any user prefs were actually applied (boosts or demotes)
            has_user_prefs = bool(
                soft_prefs.get('preferred_fits') or
                soft_prefs.get('preferred_sleeves') or
                soft_prefs.get('preferred_brands') or
                any(type_prefs.values()) or
                user_hard.get('exclude_colors') or
                user_hard.get('exclude_materials') or
                user_hard.get('exclude_brands')
            )

            return {
                "query": query,
                "results": results,
                "count": len(results),
                "filters_applied": final_filters,
                "user_prefs_applied": has_user_prefs,
                "hybrid_search": use_hybrid_search,
                "session_id": session_id,
                "occasion_filter_applied": {
                    "occasions": occasions,
                    "filtered_count": occasion_filter_stats.get("blocked", 0)
                } if occasions else None,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "has_more": len(results) == page_size
                }
            }

        except Exception as e:
            logger.error("Error in filtered search", error=str(e))
            import traceback
            traceback.print_exc()
            return {
                "query": query,
                "results": [],
                "count": 0,
                "error": str(e),
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "has_more": False
                }
            }

    def search_multimodal(
        self,
        query: str,
        limit: int = 100,
        embedding_version: int = 1,
        categories: Optional[List[str]] = None,
        exclude_brands: Optional[List[str]] = None,
        include_brands: Optional[List[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        exclude_product_ids: Optional[List[str]] = None,
    ) -> Dict:
        """
        Search using multimodal embeddings (combined image + text vectors).

        Calls the search_multimodal RPC which searches against pre-computed
        combined embeddings that encode BOTH visual appearance and product
        text attributes (name, fabric, texture, color, etc.).

        This enables matching on descriptive terms like "ribbed", "quilted",
        or "pleated" that exist in product text but not in images.

        Args:
            query: Search query text (encoded with FashionCLIP text encoder).
            limit: Max results to return.
            embedding_version: 1 = attributes only, 2 = attributes + description.
            categories: Filter by broad_category.
            exclude_brands: Brands to exclude.
            include_brands: Brands to include.
            min_price: Minimum price filter.
            max_price: Maximum price filter.
            exclude_product_ids: Product IDs to exclude.

        Returns:
            Dict with "results" list and metadata.
        """
        text_embedding = self.encode_text(query)
        text_embedding = text_embedding.astype('float32').tolist()
        vector_str = f"[{','.join(map(str, text_embedding))}]"

        rpc_params: Dict[str, Any] = {
            'query_embedding': vector_str,
            'match_count': limit,
            'match_offset': 0,
            'embedding_version': embedding_version,
        }

        # Add optional filters (only non-None values)
        if categories:
            rpc_params['filter_categories'] = categories
        if exclude_brands:
            rpc_params['exclude_brands'] = exclude_brands
        if include_brands:
            rpc_params['include_brands'] = include_brands
        if min_price is not None:
            rpc_params['min_price'] = min_price
        if max_price is not None:
            rpc_params['max_price'] = max_price
        if exclude_product_ids:
            rpc_params['exclude_product_ids'] = exclude_product_ids

        try:
            result = self.supabase.rpc('search_multimodal', rpc_params).execute()

            if not result.data:
                return {"query": query, "results": [], "count": 0}

            results = []
            for row in result.data:
                results.append({
                    "product_id": row.get('product_id'),
                    "similarity": float(row.get('similarity') or 0),
                    "name": row.get('name', ''),
                    "brand": row.get('brand', ''),
                    "category": row.get('category', ''),
                    "broad_category": row.get('broad_category', ''),
                    "article_type": row.get('article_type', ''),
                    "price": float(row.get('price', 0) or 0),
                    "original_price": float(row.get('original_price', 0) or 0) or None,
                    "is_on_sale": bool(
                        row.get('original_price')
                        and float(row.get('original_price') or 0) > float(row.get('price') or 0)
                    ),
                    "image_url": row.get('primary_image_url', ''),
                    "gallery_images": filter_gallery_images(row.get('gallery_images', []) or []),
                    "colors": row.get('colors', []) or [],
                    "materials": row.get('materials', []) or [],
                    "fit": row.get('fit'),
                    "length": row.get('length'),
                    "sleeve": row.get('sleeve'),
                })

            return {
                "query": query,
                "results": results,
                "count": len(results),
                "embedding_version": embedding_version,
            }

        except Exception as e:
            logger.error(
                "Multimodal search failed, falling back to image-only",
                error=str(e),
                query=query,
            )
            # Return empty -- caller can fall back to image-only search
            return {"query": query, "results": [], "count": 0, "error": str(e)}

    def get_similar(
        self,
        product_id: str,
        page: int = 1,
        page_size: int = 50,
        same_category: bool = False
    ) -> Dict:
        """
        Find products visually similar to a given product.

        Args:
            product_id: UUID of the source product
            page: Page number (1-indexed)
            page_size: Results per page
            same_category: If True, only return items from same category

        Returns:
            Dict with similar products and pagination
        """
        page = max(1, page)
        page_size = max(1, page_size)

        # Fetch extra to account for duplicates
        fetch_multiplier = 3
        total_needed = page * page_size
        fetch_count = total_needed * fetch_multiplier

        try:
            # Get source product's category if filtering
            filter_category = None
            if same_category:
                prod = self.supabase.table('products').select('category').eq('id', product_id).limit(1).execute()
                if prod.data:
                    filter_category = prod.data[0].get('category')

            result = self.supabase.rpc('get_similar_products_v2', {
                'source_product_id': product_id,
                'match_count': fetch_count,
                'match_offset': 0,
                'filter_category': filter_category
            }).execute()

            if not result.data:
                return {
                    "source_product_id": product_id,
                    "results": [],
                    "count": 0,
                    "pagination": {
                        "page": page,
                        "page_size": page_size,
                        "has_more": False
                    }
                }

            all_results = []
            for row in result.data:
                all_results.append({
                    "product_id": row.get('product_id'),
                    "similarity": float(row.get('similarity', 0)),
                    "name": row.get('name', ''),
                    "brand": row.get('brand', ''),
                    "category": row.get('category', ''),
                    "broad_category": row.get('broad_category', ''),
                    "price": float(row.get('price', 0) or 0),
                    "image_url": row.get('primary_image_url', ''),
                    "gallery_images": filter_gallery_images(row.get('gallery_images', []) or []),
                    "colors": row.get('colors', []) or [],
                    "materials": row.get('materials', []) or [],
                })

            # Deduplicate results
            unique_results = self._deduplicate_results(all_results)

            # Paginate the deduplicated results
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            results = unique_results[start_idx:end_idx]

            return {
                "source_product_id": product_id,
                "results": results,
                "count": len(results),
                "same_category": same_category,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "has_more": len(results) == page_size
                }
            }

        except Exception as e:
            logger.error("Error getting similar products", error=str(e))
            return {
                "source_product_id": product_id,
                "results": [],
                "count": 0,
                "error": str(e),
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "has_more": False
                }
            }

    # =========================================================================
    # Complete The Fit - CLIP-based Complementary Items Logic
    # =========================================================================

    # Category mapping: what categories complement each other
    # Based on actual database categories: tops, bottoms, dresses, outerwear
    COMPLEMENTARY_CATEGORIES = {
        'tops': ['bottoms', 'outerwear'],
        'bottoms': ['tops', 'outerwear'],
        'dresses': ['outerwear'],
        'outerwear': ['tops', 'bottoms', 'dresses'],
    }

    # Human-readable category names for prompt generation
    CATEGORY_NAMES = {
        'tops': 'top, blouse, or sweater',
        'bottoms': 'pants, skirt, or shorts',
        'dresses': 'dress',
        'outerwear': 'jacket or coat',
    }

    def _build_complement_query(self, source_product: Dict, target_category: str) -> str:
        """
        Build a CLIP text query describing what would complement the source item.

        Uses source product attributes to generate semantic search queries like:
        "elegant black jacket to pair with red evening dress"
        """
        source_name = source_product.get('name', '')
        source_color = source_product.get('base_color', '')
        source_category = source_product.get('category', '')
        occasions = source_product.get('occasions') or []
        usage = source_product.get('usage', '')

        target_name = self.CATEGORY_NAMES.get(target_category, target_category)

        # Build descriptive query parts
        parts = []

        # Add occasion/style context
        if occasions:
            occasion = occasions[0] if isinstance(occasions, list) else occasions
            parts.append(occasion)
        elif usage:
            parts.append(usage)

        # Add target category
        parts.append(target_name)

        # Add pairing context with source
        parts.append("to wear with")

        # Describe source item
        if source_color:
            parts.append(source_color)

        # Add source category context
        source_cat_name = self.CATEGORY_NAMES.get(source_category, source_category)
        parts.append(source_cat_name)

        query = " ".join(parts)
        return query

    def complete_the_fit(
        self,
        product_id: str,
        items_per_category: int = 4,
        target_category: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None
    ) -> Dict:
        """
        Find complementary items using CLIP semantic search.

        Strategy:
        1. Get source product details
        2. For each complementary category, generate a text query describing
           the ideal complement (e.g., "elegant jacket to wear with red dress")
        3. Use CLIP text search to find semantically matching items
        4. Return top items per category

        Args:
            product_id: UUID of the source product
            items_per_category: Number of items to return per category (default 4) - for carousel mode
            target_category: If specified, only return items from this category (for feed/pagination mode)
            offset: Number of items to skip (for pagination in feed mode)
            limit: Number of items to return (for pagination in feed mode, overrides items_per_category)

        Returns:
            Dict with source product, recommendations by category, and complete outfit
        """
        try:
            # 1. Get source product details directly from table
            source_result = self.supabase.table('products').select(
                'id, name, brand, category, broad_category, price, '
                'primary_image_url, gallery_images, colors, base_color, '
                'materials, occasions, usage'
            ).eq('id', product_id).limit(1).execute()

            if not source_result.data:
                return {
                    "error": f"Product {product_id} not found",
                    "source_product": None,
                    "recommendations": {}
                }

            source_product = source_result.data[0]
            # Map 'id' to 'product_id' for consistency
            source_product['product_id'] = source_product.pop('id', product_id)
            source_category = source_product.get('category', '')

            # 2. Determine complementary categories
            all_target_categories = self.COMPLEMENTARY_CATEGORIES.get(
                source_category,
                ['tops_knitwear', 'tops_woven', 'bottoms_trousers', 'outerwear']
            )

            # If specific category requested, filter to just that one
            if target_category:
                if target_category in all_target_categories:
                    target_categories = [target_category]
                else:
                    # Allow any category for feed mode
                    target_categories = [target_category]
            else:
                target_categories = all_target_categories

            if not target_categories:
                return {
                    "source_product": self._format_product(source_product),
                    "recommendations": {},
                    "message": f"No complementary categories defined for {source_category}"
                }

            # Determine fetch size based on mode
            if target_category and limit:
                # Feed/pagination mode - fetch more for the specific category
                fetch_size = offset + limit + 20  # Extra for dedup
            else:
                # Carousel mode - use items_per_category
                fetch_size = items_per_category * 3
                limit = items_per_category

            # 3. For each target category, do CLIP text search
            recommendations = {}
            all_recommended_items = []
            queries_used = {}

            for target_cat in target_categories:
                # Build semantic query for this category
                query = self._build_complement_query(source_product, target_cat)
                queries_used[target_cat] = query

                # Search using CLIP text encoding
                search_result = self.search(
                    query=query,
                    page=1,
                    page_size=fetch_size,
                    categories=[target_cat]
                )

                # Filter out source product
                all_items = []
                for item in search_result.get('results', []):
                    if item.get('product_id') != product_id:
                        all_items.append(item)

                # Apply pagination if in feed mode (specific category)
                if target_category:
                    paginated_items = all_items[offset:offset + limit]
                    has_more = len(all_items) > offset + limit

                    # Add rank to each item (1-indexed, continuous across pages)
                    ranked_items = []
                    for i, item in enumerate(paginated_items):
                        ranked_items.append({
                            **item,
                            "rank": offset + i + 1
                        })

                    recommendations[target_cat] = {
                        "items": ranked_items,
                        "pagination": {
                            "offset": offset,
                            "limit": limit,
                            "returned": len(ranked_items),
                            "has_more": has_more
                        }
                    }
                    all_recommended_items.extend(ranked_items)
                else:
                    # Carousel mode - just top N items with rank
                    items = all_items[:limit]
                    ranked_items = []
                    for i, item in enumerate(items):
                        ranked_items.append({
                            **item,
                            "rank": i + 1
                        })

                    recommendations[target_cat] = {
                        "items": ranked_items,
                        "pagination": {
                            "offset": 0,
                            "limit": limit,
                            "returned": len(ranked_items),
                            "has_more": len(all_items) > limit
                        }
                    }
                    all_recommended_items.extend(ranked_items)

            # 4. Build complete outfit response
            total_price = float(source_product.get('price', 0) or 0)
            for item in all_recommended_items:
                total_price += float(item.get('price', 0) or 0)

            return {
                "source_product": self._format_product(source_product),
                "recommendations": recommendations,
                "queries_used": queries_used,
                "complete_outfit": {
                    "items": [product_id] + [
                        item.get('product_id') for item in all_recommended_items
                    ],
                    "total_price": round(total_price, 2),
                    "item_count": 1 + len(all_recommended_items)
                }
            }

        except Exception as e:
            logger.error("Error in complete_the_fit", error=str(e))
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "source_product": None,
                "recommendations": {}
            }

    def _format_product(self, product: Dict) -> Dict:
        """Format product for API response."""
        return {
            "product_id": product.get('product_id'),
            "name": product.get('name'),
            "brand": product.get('brand'),
            "category": product.get('category'),
            "price": float(product.get('price', 0) or 0),
            "base_color": product.get('base_color'),
            "colors": product.get('colors'),
            "occasions": product.get('occasions'),
            "usage": product.get('usage'),
            "image_url": product.get('primary_image_url'),
        }

    def get_stats(self) -> Dict:
        """Get engine statistics."""
        try:
            # Count all products with embeddings
            result = self.supabase.table("image_embeddings").select(
                "sku_id", count="exact"
            ).execute()

            return {
                "total_products_with_embeddings": result.count if result.count else 0,
                "clip_model_loaded": self._model is not None,
                "database": "supabase_pgvector",
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# Singleton instance for API use (thread-safe)
_engine: Optional[WomenSearchEngine] = None
_engine_lock = threading.Lock()


def get_women_search_engine() -> WomenSearchEngine:
    """Get or create WomenSearchEngine singleton (thread-safe)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = WomenSearchEngine()
    return _engine


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    engine = get_women_search_engine()

    stats = engine.get_stats()
    logger.info("Engine stats", **stats)

    test_queries = [
        "red dress",
        "flowy blue dress",
        "black blazer",
        "striped sweater",
        "casual white top",
    ]

    for query in test_queries:
        logger.info(f"Query: '{query}'")
        results = engine.search(query, k=5, gender="female")
        if results:
            for i, r in enumerate(results, 1):
                logger.info(
                    f"  {i}. {r['name'][:50]} (sim: {r['similarity']:.3f}) "
                    f"Brand: {r['brand']}, ${r['price']:.2f}"
                )
        else:
            logger.info("  No results found")
