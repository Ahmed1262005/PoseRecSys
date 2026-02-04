#!/usr/bin/env python3
"""
Recommendation Service - Supabase Integration

Provides personalized recommendations using:
1. Vector similarity search (pgvector via image_embeddings.sku_id -> products.id)
2. Trending/popular products (via products table)
3. User preferences (from Tinder-style test)

Schema:
- products: 61K products with metadata
- image_embeddings: 60K embeddings (sku_id links to products.id)
- user_seed_preferences: Tinder test results
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))


class RecommendationStrategy(Enum):
    """Strategy used for generating recommendations."""
    TRENDING = "trending"          # No personalization, just popular items
    SEED_VECTOR = "seed_vector"    # Cold start with Tinder preferences
    SIMILAR = "similar"            # Similar to a specific product
    SASREC = "sasrec"              # Warm user with sequence history (future)


@dataclass
class UserState:
    """User state for recommendation context."""
    user_id: Optional[str] = None
    anon_id: Optional[str] = None
    seed_vector: Optional[List[float]] = None
    seed_source: str = "none"  # 'tinder', 'behavior', 'none'
    tinder_preferences: Optional[Dict] = None
    sequence_length: int = 0
    last_interactions: List[str] = field(default_factory=list)
    disliked_skus: set = field(default_factory=set)
    session_seen_skus: set = field(default_factory=set)


class RecommendationService:
    """
    Main recommendation service.

    Integrates:
    - Supabase pgvector for similarity search
    - Products table for trending/popular items
    - User preferences from Tinder test
    """

    # Configuration
    MIN_SEQUENCE_FOR_SASREC = 5
    K_CANDIDATES = 200
    EXPLORATION_RATE = 0.1
    MAX_PER_CATEGORY = 8

    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")

        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

        self.supabase: Client = create_client(url, key)

    # =========================================================
    # Main Recommendation Methods
    # =========================================================

    def get_recommendations(
        self,
        user_id: Optional[str] = None,
        anon_id: Optional[str] = None,
        gender: str = "female",
        categories: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0,
        exclude_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get personalized recommendations.

        Flow:
        1. Load user state (preferences, history)
        2. Determine strategy (seed_vector vs trending)
        3. Retrieve candidates
        4. Return ranked results

        Args:
            offset: Number of results to skip (for pagination)
        """

        # Step 1: Load user state
        user_state = self._load_user_state(user_id, anon_id)

        # Step 2: Determine strategy
        if user_state.seed_vector is not None:
            strategy = RecommendationStrategy.SEED_VECTOR
        else:
            strategy = RecommendationStrategy.TRENDING

        # Step 3: Retrieve candidates (fetch limit + offset, then slice)
        fetch_limit = limit + offset

        if strategy == RecommendationStrategy.SEED_VECTOR:
            all_results = self._retrieve_by_vector(
                vector=user_state.seed_vector,
                gender=gender,
                category=categories[0] if categories else None,
                limit=fetch_limit
            )
            reason = "style_matched"
        else:
            all_results = self.get_trending_products(
                gender=gender,
                category=categories[0] if categories else None,
                limit=fetch_limit
            )
            reason = "trending"

        # Apply offset pagination
        results = all_results[offset:offset + limit]

        # Add reason to results
        for r in results:
            r['reason'] = reason

        return {
            "user_id": user_id or anon_id,
            "strategy": strategy.value,
            "results": results,
            "metadata": {
                "candidates_retrieved": len(results),
                "seed_vector_available": user_state.seed_vector is not None,
                "seed_source": user_state.seed_source,
                "offset": offset,
                "has_more": len(all_results) > offset + limit
            }
        }

    def _deduplicate_products(self, products: List[Dict]) -> List[Dict]:
        """
        Deduplicate products by image hash to remove same products across different brands.

        Many fashion retailers (e.g., Boohoo/Nasty Gal) sell identical items under different
        brand names. This catches duplicates by extracting the image hash from URLs.
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

        for p in products:
            # Primary dedup: image hash (catches cross-brand duplicates)
            img_url = p.get('image_url') or p.get('primary_image_url')
            img_hash = get_image_hash(img_url)
            if img_hash and img_hash in seen_image_hashes:
                continue

            # Secondary dedup: (name, brand)
            name = p.get('name')
            brand = p.get('brand')
            name_brand_key = (name, brand) if name and brand else None
            if name_brand_key and name_brand_key in seen_name_brand:
                continue

            deduped.append(p)
            if img_hash:
                seen_image_hashes.add(img_hash)
            if name_brand_key:
                seen_name_brand.add(name_brand_key)

        return deduped

    def get_similar_products(
        self,
        product_id: str,
        gender: str = "female",
        category: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict]:
        """Find products similar to a given product with pagination support."""

        try:
            # Fetch extra to account for deduplication and offset
            fetch_limit = limit + offset + 50
            result = self.supabase.rpc('get_similar_products', {
                'source_product_id': product_id,
                'match_count': fetch_limit,
                'filter_gender': gender,
                'filter_category': category
            }).execute()

            products = result.data or []
            deduped = self._deduplicate_products(products)

            # Apply offset pagination
            return deduped[offset:offset + limit]

        except Exception as e:
            print(f"Error getting similar products: {e}")
            return []

    def get_trending_products(
        self,
        gender: str = "female",
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """Get trending products with pagination support."""

        try:
            # Fetch extra to account for deduplication
            fetch_limit = limit + offset + 50

            result = self.supabase.rpc('get_trending_products', {
                'filter_gender': gender,
                'filter_category': category,
                'result_limit': fetch_limit
            }).execute()

            all_results = result.data or []
            deduped = self._deduplicate_products(all_results)

            # Apply offset pagination
            return deduped[offset:offset + limit]

        except Exception as e:
            print(f"Error getting trending products: {e}")
            return []

    def get_product_categories(self, gender: str = "female") -> List[Dict]:
        """Get available product categories with counts."""

        try:
            result = self.supabase.rpc('get_product_categories', {
                'filter_gender': gender
            }).execute()

            return result.data or []

        except Exception as e:
            print(f"Error getting categories: {e}")
            return []

    # =========================================================
    # Vector Search Methods
    # =========================================================

    def _retrieve_by_vector(
        self,
        vector: List[float],
        gender: str,
        category: Optional[str],
        limit: int
    ) -> List[Dict]:
        """Retrieve products by vector similarity."""

        try:
            # Convert to pgvector format
            vector_str = f"[{','.join(map(str, vector))}]"

            result = self.supabase.rpc('match_products_by_embedding', {
                'query_embedding': vector_str,
                'match_count': limit,
                'filter_gender': gender,
                'filter_category': category
            }).execute()

            return result.data or []

        except Exception as e:
            print(f"Error in vector retrieval: {e}")
            # Fallback to trending
            return self.get_trending_products(gender, category, limit)

    def get_product_embedding(self, product_id: str) -> Optional[str]:
        """Get the embedding vector for a product."""

        try:
            result = self.supabase.rpc('get_product_embedding', {
                'p_product_id': product_id
            }).execute()

            return result.data

        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    # =========================================================
    # User State Management
    # =========================================================

    def _load_user_state(
        self,
        user_id: Optional[str],
        anon_id: Optional[str]
    ) -> UserState:
        """Load user state from Supabase."""

        state = UserState(user_id=user_id, anon_id=anon_id)

        if not user_id and not anon_id:
            return state

        try:
            query = self.supabase.table("user_seed_preferences").select(
                "attribute_preferences, taste_vector, completed_at"
            )

            if user_id:
                query = query.eq("user_id", user_id)
            else:
                query = query.eq("anon_id", anon_id)

            result = query.limit(1).execute()

            if result.data and result.data[0].get("completed_at"):
                prefs = result.data[0]
                state.tinder_preferences = prefs.get("attribute_preferences")

                # Load taste vector if available
                taste_vec = prefs.get("taste_vector")
                if taste_vec:
                    # Handle pgvector string format "[0.123, 0.456, ...]"
                    if isinstance(taste_vec, str):
                        taste_vec = [float(x) for x in taste_vec.strip('[]').split(',')]

                    if len(taste_vec) == 512:
                        state.seed_vector = taste_vec
                        state.seed_source = "tinder"

        except Exception as e:
            print(f"Error loading user state: {e}")

        return state

    # =========================================================
    # Tinder Preference Ingestion
    # =========================================================

    def save_tinder_preferences(
        self,
        user_id: Optional[str] = None,
        anon_id: Optional[str] = None,
        gender: str = "female",
        rounds_completed: int = 0,
        categories_tested: List[str] = None,
        attribute_preferences: Dict = None,
        prediction_accuracy: Optional[float] = None,
        taste_vector: Optional[List[float]] = None
    ) -> Dict:
        """
        Save Tinder test results to user_seed_preferences.

        Called after user completes Tinder-style preference test.
        """

        if not user_id and not anon_id:
            raise ValueError("Either user_id or anon_id must be provided")

        try:
            result = self.supabase.rpc('save_tinder_preferences', {
                'p_user_id': user_id,
                'p_anon_id': anon_id,
                'p_gender': gender,
                'p_rounds_completed': rounds_completed,
                'p_categories_tested': categories_tested or [],
                'p_attribute_preferences': attribute_preferences or {},
                'p_cluster_preferences': {},
                'p_prediction_accuracy': prediction_accuracy,
                'p_taste_vector': taste_vector
            }).execute()

            return {
                "status": "success",
                "preference_id": str(result.data) if result.data else None,
                "user_id": user_id or anon_id,
                "seed_source": "tinder" if taste_vector else "attributes"
            }

        except Exception as e:
            print(f"Error saving preferences: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    # =========================================================
    # Product Queries
    # =========================================================

    def get_product(self, product_id: str) -> Optional[Dict]:
        """Get a single product by ID."""

        try:
            result = self.supabase.table("products").select(
                "id, name, brand, category, gender, price, "
                "primary_image_url, hero_image_url, in_stock"
            ).eq("id", product_id).limit(1).execute()

            return result.data[0] if result.data else None

        except Exception as e:
            print(f"Error getting product: {e}")
            return None

    def search_products(
        self,
        gender: str = "female",
        categories: Optional[List[str]] = None,
        brands: Optional[List[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Search products with filters."""

        try:
            query = self.supabase.table("products").select(
                "id, name, brand, category, gender, price, "
                "primary_image_url, hero_image_url, trending_score"
            ).contains("gender", [gender]).eq("in_stock", True)

            if categories:
                query = query.in_("category", categories)

            if brands:
                query = query.in_("brand", brands)

            if min_price is not None:
                query = query.gte("price", min_price)

            if max_price is not None:
                query = query.lte("price", max_price)

            query = query.order("trending_score", desc=True, nullsfirst=False)
            query = query.limit(limit)

            result = query.execute()
            return result.data or []

        except Exception as e:
            print(f"Error searching products: {e}")
            return []

    # =========================================================
    # User Interaction Tracking
    # =========================================================

    def record_user_interaction(
        self,
        anon_id: Optional[str],
        user_id: Optional[str],
        session_id: str,
        product_id: str,
        action: str,
        source: Optional[str] = "feed",
        position: Optional[int] = None
    ) -> Dict:
        """
        Record a user interaction to the database.

        Args:
            anon_id: Anonymous user identifier
            user_id: UUID user identifier
            session_id: Session ID from feed response
            product_id: Product UUID that was interacted with
            action: One of: click, hover, add_to_wishlist, add_to_cart, purchase
            source: Where the interaction happened (feed, search, similar, style-this)
            position: Position in the feed when interacted

        Returns:
            Dict with interaction_id if successful
        """
        data = {
            "session_id": session_id,
            "product_id": product_id,
            "action": action,
            "source": source,
        }

        if anon_id:
            data["anon_id"] = anon_id
        if user_id:
            data["user_id"] = user_id
        if position is not None:
            data["position"] = position

        try:
            result = self.supabase.table("user_interactions").insert(data).execute()
            if result.data:
                return {"id": str(result.data[0]["id"])}
            return {}
        except Exception as e:
            print(f"Error recording interaction: {e}")
            raise

    def sync_session_seen_ids(
        self,
        anon_id: Optional[str],
        user_id: Optional[str],
        session_id: str,
        seen_ids: List[str]
    ) -> Dict:
        """
        Persist seen_ids for ML training data.

        This is called periodically from frontend (every N pages or on app close)
        to batch-persist which products were shown to the user.

        Used for negative sampling: items shown but not interacted with
        are implicit negatives.

        Args:
            anon_id: Anonymous user identifier
            user_id: UUID user identifier
            session_id: Session ID from feed response
            seen_ids: List of product UUIDs that were shown

        Returns:
            Dict with sync record id if successful
        """
        data = {
            "session_id": session_id,
            "seen_ids": seen_ids,  # PostgreSQL will store as uuid[]
        }

        if anon_id:
            data["anon_id"] = anon_id
        if user_id:
            data["user_id"] = user_id

        try:
            result = self.supabase.table("session_seen_ids").insert(data).execute()
            if result.data:
                return {"id": str(result.data[0]["id"])}
            return {}
        except Exception as e:
            print(f"Error syncing session seen_ids: {e}")
            raise

    def get_user_seen_history(
        self,
        anon_id: Optional[str],
        user_id: Optional[str],
        limit: int = 5000
    ) -> Set[str]:
        """
        Get all product IDs this user has been shown across all sessions.

        This enables permanent deduplication - items shown once will never
        be shown again to this user, even across page refreshes.

        Args:
            anon_id: Anonymous user identifier
            user_id: UUID user identifier
            limit: Max items to load (default 5000)

        Returns:
            Set of product UUIDs the user has seen
        """
        seen_ids = set()

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
                        seen_ids.update(row["seen_ids"])

            # Limit to prevent memory issues
            if len(seen_ids) > limit:
                # Keep most recent (though we don't have order here, just truncate)
                seen_ids = set(list(seen_ids)[:limit])

            print(f"[Service] Loaded {len(seen_ids)} seen items for user from DB")
            return seen_ids

        except Exception as e:
            print(f"Error loading user seen history: {e}")
            return seen_ids


# =========================================================
# Testing
# =========================================================

def test_service():
    """Test the recommendation service."""

    print("=" * 70)
    print("Testing Recommendation Service")
    print("=" * 70)

    service = RecommendationService()

    # Test 1: Get categories
    print("\n1. Getting product categories...")
    categories = service.get_product_categories("female")
    print(f"   Found {len(categories)} categories:")
    for cat in categories[:5]:
        print(f"   - {cat['category']}: {cat['product_count']} products")

    # Test 2: Get trending products
    print("\n2. Getting trending products...")
    trending = service.get_trending_products("female", None, 5)
    print(f"   Found {len(trending)} trending products:")
    for p in trending[:3]:
        print(f"   - {p['name'][:45]}...")

    # Test 3: Get similar products
    print("\n3. Getting similar products...")
    if trending:
        product_id = trending[0]['product_id']
        similar = service.get_similar_products(product_id, "female", None, 5)
        print(f"   Found {len(similar)} similar to '{trending[0]['name'][:30]}...':")
        for p in similar[:3]:
            print(f"   - {p['name'][:45]}... (sim: {p['similarity']:.3f})")

    # Test 4: Get recommendations (cold start)
    print("\n4. Getting recommendations (cold start - no user)...")
    recs = service.get_recommendations(gender="female", limit=5)
    print(f"   Strategy: {recs['strategy']}")
    print(f"   Results: {len(recs['results'])} products")

    # Test 5: Save and load preferences
    print("\n5. Testing preference save/load...")
    test_prefs = {
        "pattern": {"preferred": [["solid", 0.8]], "avoided": []},
        "style": {"preferred": [["casual", 0.7]], "avoided": []}
    }

    save_result = service.save_tinder_preferences(
        anon_id="test_service_user",
        gender="female",
        rounds_completed=5,
        categories_tested=["tops", "dresses"],
        attribute_preferences=test_prefs,
        prediction_accuracy=0.75
    )
    print(f"   Save result: {save_result['status']}")

    # Load and verify
    user_state = service._load_user_state(None, "test_service_user")
    print(f"   Loaded preferences: {user_state.tinder_preferences is not None}")

    print("\n" + "=" * 70)
    print("Service test complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_service()
