#!/usr/bin/env python3
"""Test pgvector similarity search functions."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))


def test_match_embeddings():
    """Test the match_embeddings RPC function."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    supabase: Client = create_client(url, key)

    print("=" * 60)
    print("Testing pgvector Similarity Search")
    print("=" * 60)

    # Get a sample embedding to use as query
    print("\n1. Fetching a sample embedding as query vector...")
    result = supabase.table("image_embeddings").select("id, embedding, review_sku_id").limit(1).execute()

    if not result.data:
        print("  ERROR: No embeddings found")
        return False

    sample = result.data[0]
    query_embedding = sample['embedding']
    print(f"  Query embedding ID: {sample['id']}")

    # Test match_embeddings function
    print("\n2. Testing match_embeddings() function...")
    try:
        result = supabase.rpc('match_embeddings', {
            'query_embedding': query_embedding,
            'match_count': 5
        }).execute()

        print(f"  Found {len(result.data)} similar items:")
        for i, item in enumerate(result.data):
            print(f"    {i+1}. ID: {item['id'][:8]}... | SKU: {item['review_sku_id'][:8] if item['review_sku_id'] else 'N/A'}... | Similarity: {item['similarity']:.4f}")

    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    # Test match_products_by_embedding function
    print("\n3. Testing match_products_by_embedding() function...")
    try:
        result = supabase.rpc('match_products_by_embedding', {
            'query_embedding': query_embedding,
            'match_count': 5,
            'filter_gender': 'female'
        }).execute()

        print(f"  Found {len(result.data)} similar products:")
        for i, item in enumerate(result.data):
            print(f"    {i+1}. {item['name'][:40]}...")
            print(f"       Brand: {item['brand']} | Category: {item['category']} | Price: ${item['price']}")
            print(f"       Similarity: {item['similarity']:.4f}")

    except Exception as e:
        print(f"  ERROR: {e}")

    # Test get_trending_products function
    print("\n4. Testing get_trending_products() function...")
    try:
        result = supabase.rpc('get_trending_products', {
            'filter_gender': 'female',
            'result_limit': 5
        }).execute()

        print(f"  Found {len(result.data)} trending products:")
        for i, item in enumerate(result.data):
            print(f"    {i+1}. {item['name'][:40]}...")
            print(f"       Brand: {item['brand']} | Trending: {item['trending_score']}")

    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("Vector similarity search test complete!")
    print("=" * 60)

    return True


def test_save_preferences():
    """Test saving user preferences."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    supabase: Client = create_client(url, key)

    print("\n" + "=" * 60)
    print("Testing save_tinder_preferences() function")
    print("=" * 60)

    # Test with anonymous user
    test_prefs = {
        "pattern": {
            "preferred": [["solid", 0.78], ["floral", 0.65]],
            "avoided": [["animal_print", 0.22]]
        },
        "style": {
            "preferred": [["casual", 0.72], ["minimalist", 0.68]],
            "avoided": [["evening", 0.25]]
        }
    }

    try:
        result = supabase.rpc('save_tinder_preferences', {
            'p_anon_id': 'test_user_123',
            'p_gender': 'female',
            'p_rounds_completed': 12,
            'p_categories_tested': ['tops', 'dresses'],
            'p_attribute_preferences': test_prefs,
            'p_prediction_accuracy': 0.667
        }).execute()

        print(f"  Saved preferences with ID: {result.data}")

        # Verify it was saved
        verify = supabase.table("user_seed_preferences").select("*").eq("anon_id", "test_user_123").execute()
        if verify.data:
            print(f"  Verified: Found saved preferences for test_user_123")
            print(f"  Gender: {verify.data[0]['gender']}")
            print(f"  Rounds: {verify.data[0]['rounds_completed']}")

    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    return True


if __name__ == "__main__":
    test_match_embeddings()
    test_save_preferences()
