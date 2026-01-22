#!/usr/bin/env python3
"""Investigate the relationship between image_embeddings and products tables."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))


def investigate():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    supabase: Client = create_client(url, key)

    print("=" * 60)
    print("Investigating Schema Relationships")
    print("=" * 60)

    # Check image_embeddings sample
    print("\n1. Sample from image_embeddings:")
    result = supabase.table("image_embeddings").select("id, review_sku_id, review_image_id").limit(3).execute()
    for row in result.data:
        print(f"  ID: {row['id']}")
        print(f"  review_sku_id: {row['review_sku_id']}")
        print(f"  review_image_id: {row['review_image_id']}")
        print()

    # Check if review_sku_id matches products.id
    print("\n2. Checking if review_sku_id matches products.id...")
    sample_sku_id = result.data[0]['review_sku_id'] if result.data else None

    if sample_sku_id:
        product_result = supabase.table("products").select("id, name, gender").eq("id", sample_sku_id).execute()
        if product_result.data:
            print(f"  MATCH FOUND!")
            print(f"  Product: {product_result.data[0]['name'][:50]}...")
            print(f"  Gender: {product_result.data[0]['gender']}")
        else:
            print(f"  NO MATCH for review_sku_id: {sample_sku_id}")

    # Check products sample
    print("\n3. Sample from products:")
    result = supabase.table("products").select("id, name, gender, category").limit(3).execute()
    for row in result.data:
        print(f"  ID: {row['id']}")
        print(f"  Name: {row['name'][:40]}...")
        print(f"  Gender: {row['gender']}")
        print(f"  Category: {row['category']}")
        print()

    # Count products with female gender
    print("\n4. Counting female products...")
    # Since gender is an array, we need to use a different approach
    result = supabase.table("products").select("id", count="exact").limit(1).execute()
    print(f"  Total products: {result.count}")

    # Sample female products
    print("\n5. Sample products with 'female' in gender array:")
    result = supabase.table("products").select("id, name, gender, category, in_stock").contains("gender", ["female"]).limit(5).execute()
    if result.data:
        for row in result.data:
            print(f"  - {row['name'][:40]}... | in_stock: {row['in_stock']}")
    else:
        print("  No products found with gender containing 'female'")

    # Check how many embeddings have matching products
    print("\n6. Checking embedding-product join...")
    # Get a few embedding IDs and check if they have products
    emb_result = supabase.table("image_embeddings").select("review_sku_id").limit(10).execute()
    matches = 0
    for emb in emb_result.data:
        if emb['review_sku_id']:
            prod = supabase.table("products").select("id").eq("id", emb['review_sku_id']).limit(1).execute()
            if prod.data:
                matches += 1
    print(f"  Out of 10 embeddings, {matches} have matching products")


if __name__ == "__main__":
    investigate()
