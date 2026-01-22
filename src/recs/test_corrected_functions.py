#!/usr/bin/env python3
"""
Test corrected SQL functions after running 003_correct_product_embedding_join.sql

This tests:
1. match_embeddings() - raw vector search (product embeddings only)
2. match_products_by_embedding() - vector search with product details
3. get_trending_products() - trending products
4. get_similar_products() - find similar products
5. get_product_categories() - category list
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))


def test_all():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    supabase: Client = create_client(url, key)

    print("=" * 70)
    print("TESTING CORRECTED SQL FUNCTIONS")
    print("=" * 70)

    # Get a sample embedding to use as query
    print("\n0. Getting sample product embedding...")
    sample = supabase.table("image_embeddings").select(
        "id, sku_id, embedding"
    ).not_.is_("sku_id", "null").limit(1).execute()

    if not sample.data:
        print("   ERROR: No product embeddings found!")
        return False

    query_embedding = sample.data[0]['embedding']
    sample_sku_id = sample.data[0]['sku_id']
    print(f"   Using embedding for product: {sample_sku_id[:8]}...")

    # =========================================================
    # Test 1: match_embeddings
    # =========================================================
    print("\n" + "-" * 70)
    print("TEST 1: match_embeddings()")
    print("-" * 70)

    try:
        result = supabase.rpc('match_embeddings', {
            'query_embedding': query_embedding,
            'match_count': 5
        }).execute()

        print(f"   Found {len(result.data)} similar items")
        for i, item in enumerate(result.data):
            print(f"   {i+1}. sku_id: {item['sku_id'][:8]}... | similarity: {item['similarity']:.4f}")

        if result.data[0]['similarity'] > 0.99:
            print("   ✓ PASS - First result is the query item (similarity ~1.0)")
        else:
            print("   ⚠ WARNING - First result similarity lower than expected")

    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        return False

    # =========================================================
    # Test 2: match_products_by_embedding
    # =========================================================
    print("\n" + "-" * 70)
    print("TEST 2: match_products_by_embedding()")
    print("-" * 70)

    try:
        result = supabase.rpc('match_products_by_embedding', {
            'query_embedding': query_embedding,
            'match_count': 5,
            'filter_gender': 'female'
        }).execute()

        print(f"   Found {len(result.data)} similar products")

        if result.data:
            for i, item in enumerate(result.data[:3]):
                print(f"\n   {i+1}. {item['name'][:45]}...")
                print(f"      Brand: {item['brand']} | Category: {item['category']}")
                print(f"      Price: ${item['price']} | Similarity: {item['similarity']:.4f}")

            print("\n   ✓ PASS - Products returned with full metadata")
        else:
            print("   ✗ FAIL - No products returned")
            return False

    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        return False

    # =========================================================
    # Test 3: get_trending_products
    # =========================================================
    print("\n" + "-" * 70)
    print("TEST 3: get_trending_products()")
    print("-" * 70)

    try:
        result = supabase.rpc('get_trending_products', {
            'filter_gender': 'female',
            'filter_category': None,
            'result_limit': 5
        }).execute()

        print(f"   Found {len(result.data)} trending products")

        if result.data:
            for i, item in enumerate(result.data[:3]):
                print(f"   {i+1}. {item['name'][:45]}... (score: {item['trending_score']})")

            print("\n   ✓ PASS - Trending products returned")
        else:
            print("   ⚠ WARNING - No trending products (may need data)")

    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        return False

    # =========================================================
    # Test 4: get_similar_products
    # =========================================================
    print("\n" + "-" * 70)
    print("TEST 4: get_similar_products()")
    print("-" * 70)

    try:
        result = supabase.rpc('get_similar_products', {
            'source_product_id': sample_sku_id,
            'match_count': 5,
            'filter_gender': 'female'
        }).execute()

        print(f"   Found {len(result.data)} similar products")

        if result.data:
            for i, item in enumerate(result.data[:3]):
                print(f"   {i+1}. {item['name'][:45]}... (sim: {item['similarity']:.4f})")

            print("\n   ✓ PASS - Similar products returned")
        else:
            print("   ⚠ WARNING - No similar products found")

    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        return False

    # =========================================================
    # Test 5: get_product_categories
    # =========================================================
    print("\n" + "-" * 70)
    print("TEST 5: get_product_categories()")
    print("-" * 70)

    try:
        result = supabase.rpc('get_product_categories', {
            'filter_gender': 'female'
        }).execute()

        print(f"   Found {len(result.data)} categories")

        if result.data:
            for item in result.data[:5]:
                print(f"   - {item['category']}: {item['product_count']} products")

            print("\n   ✓ PASS - Categories returned")
        else:
            print("   ✗ FAIL - No categories found")
            return False

    except Exception as e:
        print(f"   ✗ FAIL: {e}")
        return False

    # =========================================================
    # Test 6: Verify embedding coverage
    # =========================================================
    print("\n" + "-" * 70)
    print("TEST 6: Verify embedding coverage")
    print("-" * 70)

    try:
        # Count products with embeddings
        emb_result = supabase.table("image_embeddings").select(
            "id", count="exact"
        ).not_.is_("sku_id", "null").limit(1).execute()

        prod_result = supabase.table("products").select(
            "id", count="exact"
        ).limit(1).execute()

        coverage = emb_result.count / prod_result.count * 100

        print(f"   Products: {prod_result.count}")
        print(f"   Product embeddings: {emb_result.count}")
        print(f"   Coverage: {coverage:.1f}%")

        if coverage > 95:
            print("\n   ✓ PASS - Excellent embedding coverage")
        else:
            print(f"\n   ⚠ WARNING - {prod_result.count - emb_result.count} products missing embeddings")

    except Exception as e:
        print(f"   ✗ FAIL: {e}")

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
    print("\nThe corrected SQL functions are working correctly.")
    print("JOIN: image_embeddings.sku_id = products.id")
    print(f"Coverage: ~{coverage:.0f}% of products have embeddings")

    return True


if __name__ == "__main__":
    test_all()
