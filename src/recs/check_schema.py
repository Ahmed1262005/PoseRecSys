#!/usr/bin/env python3
"""Check existing Supabase table schemas."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))


def check_schema():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    supabase: Client = create_client(url, key)

    print("=" * 60)
    print("Checking existing table schemas")
    print("=" * 60)

    # Check image_embeddings
    print("\n--- image_embeddings (sample row) ---")
    try:
        result = supabase.table("image_embeddings").select("*").limit(1).execute()
        if result.data:
            row = result.data[0]
            for key, value in row.items():
                if key == 'embedding':
                    # Show first few values of embedding
                    if isinstance(value, list):
                        print(f"  {key}: list[{len(value)}] = [{value[0]:.4f}, {value[1]:.4f}, ...]")
                    elif isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: str (len={len(value)}) - likely vector string")
                    else:
                        print(f"  {key}: {type(value).__name__}")
                else:
                    print(f"  {key}: {value}")
    except Exception as e:
        print(f"  Error: {e}")

    # Check products
    print("\n--- products (sample row) ---")
    try:
        result = supabase.table("products").select("*").limit(1).execute()
        if result.data:
            row = result.data[0]
            for key, value in row.items():
                if isinstance(value, str) and len(str(value)) > 100:
                    print(f"  {key}: str (len={len(value)}) = {str(value)[:50]}...")
                elif isinstance(value, list):
                    print(f"  {key}: list[{len(value)}]")
                elif isinstance(value, dict):
                    print(f"  {key}: dict with keys {list(value.keys())[:5]}...")
                else:
                    print(f"  {key}: {value}")
    except Exception as e:
        print(f"  Error: {e}")

    # Count by gender
    print("\n--- Data counts ---")
    try:
        result = supabase.table("image_embeddings").select("gender", count="exact").execute()
        print(f"  Total embeddings: {result.count}")
    except Exception as e:
        print(f"  Error counting embeddings: {e}")

    try:
        result = supabase.table("products").select("gender", count="exact").execute()
        print(f"  Total products: {result.count}")
    except Exception as e:
        print(f"  Error counting products: {e}")

    # Check if there are female embeddings
    print("\n--- Checking for female items ---")
    try:
        result = supabase.table("image_embeddings").select("*", count="exact").eq("gender", "female").limit(1).execute()
        print(f"  Female embeddings: {result.count}")
        if result.data:
            print(f"  Sample female SKU: {result.data[0].get('sku_id', 'N/A')}")
    except Exception as e:
        print(f"  Error: {e}")

    try:
        result = supabase.table("products").select("*", count="exact").eq("gender", "female").limit(1).execute()
        print(f"  Female products: {result.count}")
        if result.data:
            print(f"  Sample female product: {result.data[0].get('title', 'N/A')[:50]}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    check_schema()
