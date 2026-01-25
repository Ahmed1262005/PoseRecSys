"""
Women's Fashion Text Search using FashionCLIP + Supabase pgvector

Provides text-to-image search for women's fashion items using:
- FashionCLIP for text encoding (512-dim vectors)
- Supabase pgvector for similarity search in the products database
"""
import os
import sys
import numpy as np
from typing import List, Dict, Optional, Set, Any, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Import occasion gate for multi-signal filtering
from recs.occasion_gate import check_occasion_gate, filter_candidates_by_occasion
from recs.models import Candidate


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

    def _load_user_profile(
        self,
        user_id: Optional[str] = None,
        anon_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load user's complete onboarding profile from Supabase.

        Returns:
            Dict with all profile fields organized into:
            - hard_filters: Applied in SQL (colors_to_avoid, etc.)
            - soft_prefs: Used for scoring (preferred_fits, etc.)
            - type_prefs: Article type preferences (top_types, etc.)
            - taste_vector: 512-dim embedding from style discovery
        """
        if not user_id and not anon_id:
            return {"hard_filters": {}, "soft_prefs": {}, "type_prefs": {}, "taste_vector": None}

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
                # Soft preferences
                'preferred_fits',
                'preferred_sleeves',
                'preferred_lengths',
                'preferred_lengths_dresses',
                'preferred_rises',
                'preferred_brands',
                'style_persona',
                # Type preferences
                'top_types',
                'bottom_types',
                'dress_types',
                'outerwear_types',
                # Style discovery
                'taste_vector'
            )

            if user_id:
                query = query.eq('user_id', user_id)
            else:
                query = query.eq('anon_id', anon_id)

            result = query.limit(1).execute()

            if not result.data:
                return {"hard_filters": {}, "soft_prefs": {}, "type_prefs": {}, "taste_vector": None}

            profile = result.data[0]

            return {
                "hard_filters": {
                    "categories": profile.get('categories'),
                    "exclude_colors": profile.get('colors_to_avoid'),
                    "exclude_materials": profile.get('materials_to_avoid'),
                    "exclude_brands": profile.get('brands_to_avoid'),
                    "exclude_styles": profile.get('styles_to_avoid'),
                    "include_occasions": profile.get('occasions'),
                    "include_patterns": profile.get('patterns_liked'),
                    "exclude_patterns": profile.get('patterns_avoided'),
                },
                "soft_prefs": {
                    "preferred_fits": profile.get('preferred_fits') or [],
                    "preferred_sleeves": profile.get('preferred_sleeves') or [],
                    "preferred_lengths": profile.get('preferred_lengths') or [],
                    "preferred_lengths_dresses": profile.get('preferred_lengths_dresses') or [],
                    "preferred_rises": profile.get('preferred_rises') or [],
                    "preferred_brands": profile.get('preferred_brands') or [],
                    "style_persona": profile.get('style_persona') or [],
                },
                "type_prefs": {
                    "top_types": profile.get('top_types') or [],
                    "bottom_types": profile.get('bottom_types') or [],
                    "dress_types": profile.get('dress_types') or [],
                    "outerwear_types": profile.get('outerwear_types') or [],
                },
                "taste_vector": profile.get('taste_vector'),
            }
        except Exception as e:
            print(f"[WomenSearchEngine] Error loading user profile: {e}")
            return {"hard_filters": {}, "soft_prefs": {}, "type_prefs": {}, "taste_vector": None}

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

        IMPORTANT: Boosts are CAPPED to prevent overriding semantic relevance.
        - MAX_TOTAL_BOOST = 0.15 (positive boosts never exceed this)
        - SEMANTIC_FLOOR = 0.50 (items below this get no boosts)

        BOOST weights (positive preferences) - REDUCED from previous values:
        - Fit match: +0.03 (was 0.05)
        - Sleeve match: +0.03
        - Length match: +0.03
        - Rise match: +0.02
        - Preferred brand: +0.05 (was 0.08)
        - Type match: +0.03 (was 0.04)

        DEMOTE weights (things to avoid) - unchanged:
        - Color to avoid: -0.15
        - Style to avoid: -0.12
        - Material to avoid: -0.10
        - Brand to avoid: -0.15
        """
        if not soft_prefs and not type_prefs and not user_hard:
            return results

        user_hard = user_hard or {}

        # === CAPS to prevent boosts from overriding semantic relevance ===
        MAX_TOTAL_BOOST = 0.15  # Positive boosts never exceed this
        SEMANTIC_FLOOR = 0.25  # Items below this semantic score get no boosts
        # Note: Text-to-image CLIP scores typically range 0.28-0.40, not 0.70-0.95

        # === BOOST preferences (positive) ===
        preferred_fits = {f.lower() for f in soft_prefs.get('preferred_fits', [])}
        preferred_sleeves = {s.lower() for s in soft_prefs.get('preferred_sleeves', [])}
        preferred_lengths = {l.lower() for l in soft_prefs.get('preferred_lengths', [])}
        preferred_lengths_dresses = {l.lower() for l in soft_prefs.get('preferred_lengths_dresses', [])}
        preferred_rises = {r.lower() for r in soft_prefs.get('preferred_rises', [])}
        preferred_brands = {b.lower() for b in soft_prefs.get('preferred_brands', [])}

        # Type preferences by category
        all_type_prefs = set()
        for types in type_prefs.values():
            all_type_prefs.update(t.lower() for t in (types or []))

        # === DEMOTE preferences (things to avoid) ===
        colors_to_avoid = {c.lower() for c in user_hard.get('exclude_colors', []) or []}
        styles_to_avoid = {s.lower() for s in user_hard.get('exclude_styles', []) or []}
        materials_to_avoid = {m.lower() for m in user_hard.get('exclude_materials', []) or []}
        brands_to_avoid = {b.lower() for b in user_hard.get('exclude_brands', []) or []}

        # Scoring weights - BOOSTS (REDUCED to prevent override)
        BOOST_FIT = 0.03      # was 0.05
        BOOST_SLEEVE = 0.03
        BOOST_LENGTH = 0.03
        BOOST_RISE = 0.02
        BOOST_BRAND = 0.05    # was 0.08
        BOOST_TYPE = 0.03     # was 0.04

        # Scoring weights - DEMOTES (negative) - unchanged
        DEMOTE_COLOR = -0.15
        DEMOTE_STYLE = -0.12
        DEMOTE_MATERIAL = -0.10
        DEMOTE_BRAND = -0.15

        for item in results:
            positive_boost = 0.0
            negative_boost = 0.0
            match_reasons = []
            demote_reasons = []

            # Get base semantic score (before any boosts)
            base_semantic = item.get('base_similarity', item.get('similarity', 0))

            item_brand = (item.get('brand') or '').lower()
            item_colors = [c.lower() for c in (item.get('colors') or [])]
            item_materials = [m.lower() for m in (item.get('materials') or [])]

            # === BOOSTS (only apply if semantic score meets floor) ===
            if base_semantic >= SEMANTIC_FLOOR:
                # Fit match
                item_fit = (item.get('fit') or '').lower()
                if item_fit and item_fit in preferred_fits:
                    positive_boost += BOOST_FIT
                    match_reasons.append('fit')

                # Sleeve match
                item_sleeve = (item.get('sleeve') or '').lower()
                if item_sleeve and item_sleeve in preferred_sleeves:
                    positive_boost += BOOST_SLEEVE
                    match_reasons.append('sleeve')

                # Length match (use dress lengths for dresses, regular for others)
                item_length = (item.get('length') or '').lower()
                item_category = (item.get('broad_category') or '').lower()
                if item_length:
                    if item_category in ['dresses', 'skirts']:
                        if item_length in preferred_lengths_dresses:
                            positive_boost += BOOST_LENGTH
                            match_reasons.append('length')
                    else:
                        if item_length in preferred_lengths:
                            positive_boost += BOOST_LENGTH
                            match_reasons.append('length')

                # Rise match (for bottoms)
                item_rise = (item.get('rise') or '').lower()
                if item_rise and item_rise in preferred_rises:
                    positive_boost += BOOST_RISE
                    match_reasons.append('rise')

                # Brand match (boost preferred brands)
                if item_brand and item_brand in preferred_brands:
                    positive_boost += BOOST_BRAND
                    match_reasons.append('brand')

                # Type match
                item_type = (item.get('article_type') or '').lower()
                if item_type and item_type in all_type_prefs:
                    positive_boost += BOOST_TYPE
                    match_reasons.append('type')

            # === DEMOTES (always apply, no floor) ===

            # Demote colors to avoid (partial match - "red" matches "Wine Red", "Cherry Red", etc.)
            if colors_to_avoid and item_colors:
                for item_color in item_colors:
                    if any(avoided in item_color for avoided in colors_to_avoid):
                        negative_boost += DEMOTE_COLOR
                        demote_reasons.append('color_avoid')
                        break  # Only demote once per item

            # Demote styles to avoid (would need style_tags field)
            # Note: style scores are complex, simplified check here
            # if styles_to_avoid:
            #     # Check if item has avoided styles - requires style_tags field
            #     pass

            # Demote materials to avoid (partial match - "wool" matches "82% Wool", etc.)
            if materials_to_avoid and item_materials:
                for item_material in item_materials:
                    if any(avoided in item_material for avoided in materials_to_avoid):
                        negative_boost += DEMOTE_MATERIAL
                        demote_reasons.append('material_avoid')
                        break  # Only demote once per item

            # Demote brands to avoid
            if brands_to_avoid and item_brand:
                if item_brand in brands_to_avoid:
                    negative_boost += DEMOTE_BRAND
                    demote_reasons.append('brand_avoid')

            # Cap positive boosts to prevent overriding semantic relevance
            capped_positive_boost = min(MAX_TOTAL_BOOST, positive_boost)
            total_boost = capped_positive_boost + negative_boost  # negative_boost is already negative

            # Apply boost to similarity score (clamp to 0-1 range)
            original_similarity = item.get('similarity', 0)
            item['similarity'] = max(0.0, min(1.0, original_similarity + total_boost))
            item['preference_boost'] = round(total_boost, 4)
            item['preference_matches'] = match_reasons
            item['preference_demotes'] = demote_reasons

        # Re-sort by boosted similarity
        results.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        return results

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

        Ensures varied results by limiting items from any single category.
        Uses broad_category for grouping (tops, bottoms, dresses, outerwear).
        """
        category_counts = defaultdict(int)
        diverse_results = []

        for item in results:
            cat = item.get('broad_category') or item.get('category') or 'unknown'

            if category_counts[cat] < max_per_category:
                diverse_results.append(item)
                category_counts[cat] += 1

        return diverse_results

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
        Apply multi-signal occasion gate to search results.

        This provides the same hard gating logic as the feed pipeline:
        - Positive occasion score threshold
        - Negative occasion score threshold (contrastive)
        - Cross-occasion blocking (high active blocks office)
        - Style score blocking (high sleeveless blocks office)
        - Sleeve/length hard blocks
        - Brand/keyword blocks

        Args:
            results: List of search result dicts
            include_occasions: Occasions to filter by (e.g., ['office'])
            verbose: Print debug info

        Returns:
            (filtered_results, blocked_reasons) - blocked_reasons is count by reason
        """
        if not include_occasions:
            return results, {}

        # Check if any results have computed scores
        # If SQL function doesn't return them, we can only use hard blockers (brand, keywords)
        has_scores = any(item.get('computed_occasion_scores') for item in results[:10])

        if not has_scores and verbose:
            print("[WomenSearchEngine] Note: computed_occasion_scores not available from SQL")
            print("  Applying brand/keyword blockers only. Run SQL migration 036 to enable full gate.")

        filtered = []
        blocked_reasons = defaultdict(int)

        for item in results:
            # Convert dict to Candidate-like object for occasion gate
            # If scores are missing, pass empty dicts - gate will use hard blockers only
            occasion_scores = item.get('computed_occasion_scores')
            style_scores = item.get('computed_style_scores')

            # Handle None and convert JSONB to dict if needed
            if occasion_scores is None:
                occasion_scores = {}
            elif isinstance(occasion_scores, str):
                import json
                try:
                    occasion_scores = json.loads(occasion_scores)
                except:
                    occasion_scores = {}

            if style_scores is None:
                style_scores = {}
            elif isinstance(style_scores, str):
                import json
                try:
                    style_scores = json.loads(style_scores)
                except:
                    style_scores = {}

            candidate = Candidate(
                item_id=item.get('product_id', ''),
                name=item.get('name'),
                brand=item.get('brand'),
                article_type=item.get('article_type'),
                sleeve=item.get('sleeve'),
                length=item.get('length'),
                fit=item.get('fit'),
                computed_occasion_scores=occasion_scores,
                computed_style_scores=style_scores,
            )

            passes, reason = check_occasion_gate(candidate, include_occasions, verbose=False)
            if passes:
                filtered.append(item)
            else:
                blocked_reasons[reason] += 1

        if verbose and blocked_reasons:
            print(f"[WomenSearchEngine] Occasion gate: {len(results)} -> {len(filtered)} results")
            for reason, count in sorted(blocked_reasons.items(), key=lambda x: -x[1])[:3]:
                print(f"  - {reason}: {count}")

        return filtered, dict(blocked_reasons)

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
        """Lazy load FashionCLIP model directly for faster inference."""
        if self._model is None:
            import torch
            from transformers import CLIPProcessor, CLIPModel

            self._model = CLIPModel.from_pretrained('patrickjohncyh/fashion-clip')
            self._processor = CLIPProcessor.from_pretrained('patrickjohncyh/fashion-clip')

            # Move to GPU if available
            if torch.cuda.is_available():
                self._model = self._model.cuda()

            self._model.eval()

    def encode_text(self, query: str) -> np.ndarray:
        """Encode text query to embedding vector."""
        import torch
        self._load_model()

        with torch.no_grad():
            inputs = self._processor(text=[query], return_tensors='pt', padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            emb = self._model.get_text_features(**inputs)
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
                        "gallery_images": row.get('gallery_images', []) or [],
                        "colors": row.get('colors', []) or [],
                        "materials": row.get('materials', []) or [],
                    })

                if len(result.data) < page_size:
                    break

                page += 1

            except Exception as e:
                print(f"[WomenSearchEngine] Error fetching page {page}: {e}")
                break

        return {
            "query": query,
            "results": all_results[:max_results],
            "count": len(all_results[:max_results]),
            "total_fetched": len(all_results)
        }

    def _get_image_hash(self, url: str) -> Optional[str]:
        """Extract image hash from URL like .../original_0_85a218f8.jpg -> 85a218f8"""
        import re
        if not url:
            return None
        match = re.search(r'original_\d+_([a-f0-9]+)\.', url)
        return match.group(1) if match else None

    def _deduplicate_results(self, results: List[Dict], limit: Optional[int] = None) -> List[Dict]:
        """
        Remove duplicate products based on image hash and (name, brand).

        Handles two types of duplicates:
        1. Same image across different brands (e.g., Boohoo/Nasty Gal selling identical items)
        2. Same name+brand with different IDs (same product scraped multiple times)

        Keeps the first (highest similarity) occurrence.
        """
        seen_image_hashes = set()
        seen_name_brand = set()
        unique_results = []

        for item in results:
            # Primary dedup: image hash (catches cross-brand duplicates like Boohoo/Nasty Gal)
            img_url = item.get('image_url') or item.get('primary_image_url')
            img_hash = self._get_image_hash(img_url)
            if img_hash and img_hash in seen_image_hashes:
                continue  # Same image already shown

            # Secondary dedup: (name, brand) for products without matching image hash
            name = (item.get('name') or '').lower().strip()
            brand = (item.get('brand') or '').lower().strip()
            name_brand_key = (name, brand) if name and brand else None
            if name_brand_key and name_brand_key in seen_name_brand:
                continue  # Same name+brand already shown

            unique_results.append(item)
            if img_hash:
                seen_image_hashes.add(img_hash)
            if name_brand_key:
                seen_name_brand.add(name_brand_key)

            # Stop if we have enough results
            if limit and len(unique_results) >= limit:
                break

        return unique_results

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
                    "gallery_images": row.get('gallery_images', []) or [],
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
            print(f"[WomenSearchEngine] Error in text search: {e}")
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
                    "gallery_images": row.get('gallery_images', []) or [],
                    "colors": row.get('colors', []) or [],
                    "materials": row.get('materials', []) or [],
                    "fit": row.get('fit'),
                    "length": row.get('length'),
                    "sleeve": row.get('sleeve'),
                    # Computed scores for occasion gate (multi-signal filtering)
                    "computed_occasion_scores": row.get('computed_occasion_scores', {}),
                    "computed_style_scores": row.get('computed_style_scores', {}),
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
            # STEP 8.5: Apply occasion gate (multi-signal filtering)
            # =========================================
            occasion_gate_reasons = {}
            if occasions:
                pre_gate_count = len(unique_results)
                unique_results, occasion_gate_reasons = self._apply_occasion_gate(
                    unique_results, occasions, verbose=True
                )
                if pre_gate_count != len(unique_results):
                    print(f"[WomenSearchEngine] Occasion gate ({occasions}): {pre_gate_count} -> {len(unique_results)}")

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
                "occasion_gate_applied": {
                    "occasions": occasions,
                    "blocked_reasons": occasion_gate_reasons
                } if occasions else None,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "has_more": len(results) == page_size
                }
            }

        except Exception as e:
            print(f"[WomenSearchEngine] Error in filtered search: {e}")
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
                    "gallery_images": row.get('gallery_images', []) or [],
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
            print(f"[WomenSearchEngine] Error getting similar products: {e}")
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
        items_per_category: int = 4
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
            items_per_category: Number of items to return per category (default 4)

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
            target_categories = self.COMPLEMENTARY_CATEGORIES.get(
                source_category,
                ['tops_knitwear', 'tops_woven', 'bottoms_trousers', 'outerwear']
            )

            if not target_categories:
                return {
                    "source_product": self._format_product(source_product),
                    "recommendations": {},
                    "message": f"No complementary categories defined for {source_category}"
                }

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
                    page_size=items_per_category * 3,  # Get more for filtering
                    categories=[target_cat]
                )

                # Filter out source product and take top items
                items = []
                for item in search_result.get('results', []):
                    if item.get('product_id') != product_id:
                        items.append(item)
                    if len(items) >= items_per_category:
                        break

                recommendations[target_cat] = items
                all_recommended_items.extend(items)

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
            print(f"[WomenSearchEngine] Error in complete_the_fit: {e}")
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


# Singleton instance for API use
_engine: Optional[WomenSearchEngine] = None


def get_women_search_engine() -> WomenSearchEngine:
    """Get or create WomenSearchEngine singleton."""
    global _engine
    if _engine is None:
        _engine = WomenSearchEngine()
    return _engine


if __name__ == "__main__":
    # Test the search engine
    print("Loading WomenSearchEngine (Supabase)...")
    engine = get_women_search_engine()

    stats = engine.get_stats()
    print(f"\nEngine Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Test queries
    test_queries = [
        "red dress",
        "flowy blue dress",
        "black blazer",
        "striped sweater",
        "casual white top",
    ]

    print("\n" + "="*60)
    print("Test Queries:")
    print("="*60)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = engine.search(query, k=5, gender="female")
        if results:
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r['name'][:50]}... (sim: {r['similarity']:.3f})")
                print(f"     Brand: {r['brand']}, Category: {r['category']}, Price: ${r['price']:.2f}")
        else:
            print("  No results found")
