#!/usr/bin/env python3
"""Test pgvector similarity search on Supabase."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))


def test_vector_search():
    """Test pgvector similarity search."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    supabase: Client = create_client(url, key)

    print("=" * 60)
    print("Testing pgvector Similarity Search")
    print("=" * 60)

    # First, get a sample embedding to use as query
    print("\n1. Fetching a sample embedding...")
    try:
        result = supabase.table("image_embeddings").select("id, embedding, review_sku_id").limit(1).execute()
        if not result.data:
            print("  ERROR: No embeddings found")
            return False

        sample = result.data[0]
        sample_embedding = sample['embedding']
        sample_id = sample['id']

        print(f"  Sample ID: {sample_id}")
        print(f"  Embedding type: {type(sample_embedding)}")

        # Parse the embedding string to check its format
        if isinstance(sample_embedding, str):
            # It's stored as a pgvector string like "[0.123, 0.456, ...]"
            # Remove brackets and split
            emb_str = sample_embedding.strip('[]')
            values = [float(x) for x in emb_str.split(',')]
            print(f"  Embedding dimension: {len(values)}")
            print(f"  First 5 values: {values[:5]}")

    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    # Test vector similarity search using RPC
    print("\n2. Testing similarity search via RPC...")
    print("  Note: This requires a match_embeddings function in your database")

    try:
        # Try to call a similarity search function
        # This may need to be created in Supabase
        result = supabase.rpc('match_embeddings', {
            'query_embedding': sample_embedding,
            'match_count': 5
        }).execute()

        print(f"  Found {len(result.data)} similar items")
        for i, item in enumerate(result.data[:3]):
            print(f"    {i+1}. ID: {item.get('id', 'N/A')}, Similarity: {item.get('similarity', 'N/A')}")

    except Exception as e:
        error_str = str(e)
        if "function" in error_str.lower() and "does not exist" in error_str.lower():
            print("  The match_embeddings RPC function doesn't exist yet.")
            print("  You need to create it in Supabase SQL Editor.")
            print("\n  Here's the SQL to create it:")
            print("""
  CREATE OR REPLACE FUNCTION match_embeddings(
    query_embedding vector(512),
    match_count int DEFAULT 10
  )
  RETURNS TABLE (
    id uuid,
    review_sku_id uuid,
    similarity float
  )
  LANGUAGE plpgsql
  AS $$
  BEGIN
    RETURN QUERY
    SELECT
      ie.id,
      ie.review_sku_id,
      1 - (ie.embedding <=> query_embedding) as similarity
    FROM image_embeddings ie
    ORDER BY ie.embedding <=> query_embedding
    LIMIT match_count;
  END;
  $$;
            """)
        else:
            print(f"  ERROR: {e}")

    # Test raw SQL query via RPC
    print("\n3. Testing direct vector comparison...")
    print("  Checking if we can use vector operators...")

    try:
        # Get 5 random embeddings and check if they have vectors
        result = supabase.table("image_embeddings") \
            .select("id, embedding") \
            .limit(5) \
            .execute()

        print(f"  Retrieved {len(result.data)} embeddings")

        # Check if embeddings are proper pgvector type
        for i, item in enumerate(result.data):
            emb = item['embedding']
            if isinstance(emb, str) and emb.startswith('['):
                print(f"    {i+1}. pgvector format (string) - OK")
            elif isinstance(emb, list):
                print(f"    {i+1}. Python list format - may need conversion")
            else:
                print(f"    {i+1}. Unknown format: {type(emb)}")

    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

    return True


def check_pgvector_extension():
    """Check if pgvector extension is enabled."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    supabase: Client = create_client(url, key)

    print("\nChecking pgvector extension status...")

    try:
        # Use a raw query to check extensions
        result = supabase.rpc('get_extensions', {}).execute()
        print(f"  Extensions: {result.data}")
    except Exception as e:
        print("  Note: get_extensions function not available")
        print("  Run this in SQL Editor to verify pgvector:")
        print("    SELECT * FROM pg_extension WHERE extname = 'vector';")


if __name__ == "__main__":
    test_vector_search()
    check_pgvector_extension()
