"""
Women's Fashion Text Search using FashionCLIP + Supabase pgvector

Provides text-to-image search for women's fashion items using:
- FashionCLIP for text encoding (512-dim vectors)
- Supabase pgvector for similarity search in the products database
"""
import os
import sys
import numpy as np
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))


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
        categories: Optional[List[str]] = None,
        exclude_colors: Optional[List[str]] = None,
        exclude_materials: Optional[List[str]] = None,
        exclude_brands: Optional[List[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        exclude_product_ids: Optional[List[str]] = None,
    ) -> Dict:
        """
        Search with additional filters and pagination.

        Args:
            query: Text description
            page: Page number (1-indexed)
            page_size: Results per page (max 200)
            categories: Category filter
            exclude_colors: Colors to exclude
            exclude_materials: Materials to exclude
            exclude_brands: Brands to exclude
            min_price: Minimum price
            max_price: Maximum price
            exclude_product_ids: Product IDs to exclude

        Returns:
            Dict with filtered results and pagination
        """
        # Validate pagination (no upper limit)
        page = max(1, page)
        page_size = max(1, page_size)

        # Fetch extra results to account for duplicates being filtered out
        fetch_multiplier = 3
        total_needed = page * page_size
        fetch_count = total_needed * fetch_multiplier

        # Encode text query (fast GPU inference)
        text_embedding = self.encode_text(query)
        text_embedding = text_embedding.astype('float32').tolist()

        vector_str = f"[{','.join(map(str, text_embedding))}]"

        try:
            # Fetch from offset 0 with extra results, then deduplicate and paginate
            result = self.supabase.rpc('text_search_products_filtered', {
                'query_embedding': vector_str,
                'match_count': fetch_count,
                'match_offset': 0,
                'filter_categories': categories,
                'exclude_colors': exclude_colors,
                'exclude_materials': exclude_materials,
                'exclude_brands': exclude_brands,
                'min_price': min_price,
                'max_price': max_price,
                'exclude_product_ids': exclude_product_ids,
            }).execute()

            if not result.data:
                return {
                    "query": query,
                    "results": [],
                    "count": 0,
                    "filters_applied": {
                        "categories": categories,
                        "exclude_colors": exclude_colors,
                        "exclude_materials": exclude_materials,
                        "min_price": min_price,
                        "max_price": max_price,
                    },
                    "pagination": {
                        "page": page,
                        "page_size": page_size,
                        "has_more": False
                    }
                }

            # Format all results
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
                "filters_applied": {
                    "categories": categories,
                    "exclude_colors": exclude_colors,
                    "exclude_materials": exclude_materials,
                    "min_price": min_price,
                    "max_price": max_price,
                },
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "has_more": len(results) == page_size
                }
            }

        except Exception as e:
            print(f"[WomenSearchEngine] Error in filtered search: {e}")
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
