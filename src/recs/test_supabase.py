#!/usr/bin/env python3
"""Test Supabase connection and pgvector."""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))


def test_connection():
    """Test basic Supabase connection."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        print("ERROR: SUPABASE_URL or SUPABASE_SERVICE_KEY not set in .env")
        return False

    print(f"Connecting to: {url}")

    try:
        supabase: Client = create_client(url, key)
        print("Supabase client created successfully!")

        # Test by listing tables (this works even without tables)
        # We'll try a simple query to the pgvector-enabled table
        print("\nTesting connection with a simple query...")

        # Try to get any data (will return empty if tables don't exist yet)
        try:
            result = supabase.table("image_embeddings").select("*").limit(1).execute()
            print(f"image_embeddings table exists! Rows: {len(result.data)}")
        except Exception as e:
            if "relation" in str(e).lower() and "does not exist" in str(e).lower():
                print("image_embeddings table doesn't exist yet - you need to create it via SQL Editor")
            else:
                print(f"Query error: {e}")

        return True

    except Exception as e:
        print(f"ERROR: Failed to connect to Supabase: {e}")
        return False


def test_pgvector():
    """Test if pgvector extension is enabled."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        return False

    try:
        supabase: Client = create_client(url, key)

        # Try to use pgvector functions via RPC
        # This will fail if pgvector isn't enabled
        print("\nTesting pgvector extension...")

        # Create a test function to check pgvector
        # We'll try to create a simple vector and check if it works
        try:
            # Try a simple vector operation
            result = supabase.rpc('check_extension', {'ext_name': 'vector'}).execute()
            print(f"pgvector check result: {result.data}")
        except Exception as e:
            if "function" in str(e).lower():
                print("Note: check_extension function doesn't exist, but that's OK")
                print("You should verify pgvector is enabled by running this in SQL Editor:")
                print("  SELECT * FROM pg_extension WHERE extname = 'vector';")
            else:
                print(f"Error checking pgvector: {e}")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def check_tables():
    """Check which tables exist."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        return

    supabase: Client = create_client(url, key)

    print("\nChecking for required tables...")

    tables = [
        "image_embeddings",
        "products",
        "user_seed_preferences",
        "user_vectors",
        "sku_popularity"
    ]

    for table in tables:
        try:
            result = supabase.table(table).select("count", count="exact").limit(0).execute()
            print(f"  {table}: EXISTS (count={result.count})")
        except Exception as e:
            if "does not exist" in str(e).lower():
                print(f"  {table}: NOT FOUND - needs to be created")
            else:
                print(f"  {table}: ERROR - {e}")


def main():
    print("=" * 60)
    print("Supabase Connection Test")
    print("=" * 60)

    # Test basic connection
    if not test_connection():
        print("\nConnection test FAILED")
        return False

    # Test pgvector
    test_pgvector()

    # Check tables
    check_tables()

    print("\n" + "=" * 60)
    print("Connection test completed!")
    print("=" * 60)

    print("\nNext steps:")
    print("1. If tables don't exist, run the SQL migration in Supabase SQL Editor")
    print("2. Run migrate_embeddings.py to populate image_embeddings")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
