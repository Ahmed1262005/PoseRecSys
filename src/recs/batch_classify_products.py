#!/usr/bin/env python3
"""
Batch Classification Script (Parallel Processing)

Classifies all products in Supabase using FashionCLIP embeddings.
Stores computed scores and attributes:
- computed_style_scores (JSONB) - coverage styles
- computed_occasion_scores (JSONB) - occasions
- computed_pattern_scores (JSONB) - patterns (NEW)
- fit, length, sleeve (text) - computed attributes (NEW)

Usage:
    python batch_classify_products.py [--batch-size 100] [--dry-run] [--limit 1000]
    python batch_classify_products.py --all --include-attributes  # Full classification
    python batch_classify_products.py --all --include-attributes --workers 8 --batch-size 1000  # Parallel

Requirements:
    - Run SQL migration 017_lifestyle_filtering.sql first
    - Run SQL migration 028_add_computed_attributes.sql for pattern/attribute support
    - SUPABASE_URL and SUPABASE_SERVICE_KEY env vars set
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from supabase import create_client, Client

from recs.style_classifier import StyleClassifier, StyleClassifierConfig

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))

# Thread-local storage for Supabase clients
_thread_local = threading.local()


def get_supabase_client() -> Client:
    """Create Supabase client."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

    return create_client(url, key)


def get_thread_local_supabase() -> Client:
    """Get a thread-local Supabase client for parallel writes."""
    if not hasattr(_thread_local, 'supabase'):
        _thread_local.supabase = get_supabase_client()
    return _thread_local.supabase


def count_products_to_classify(
    supabase: Client,
    reclassify_all: bool = False
) -> int:
    """Count products that need classification."""
    if reclassify_all:
        # Count all products with embeddings
        result = supabase.table('image_embeddings').select('sku_id', count='exact').execute()
        return result.count or 0
    else:
        # Count products with embeddings but no style scores
        # This requires a join, so we'll use RPC or raw query
        result = supabase.rpc('count_products_needing_classification', {}).execute()
        return result.data if result.data else 0


def fetch_products_batch(
    supabase: Client,
    offset: int,
    limit: int,
    reclassify_all: bool = False
) -> List[Dict[str, Any]]:
    """
    Fetch a batch of products with their embeddings.

    Returns products that either:
    - Don't have computed_style_scores yet (default)
    - All products with embeddings (if reclassify_all=True)
    """
    # Fetch from image_embeddings joined with products
    # We need the embedding and product_id

    if reclassify_all:
        result = supabase.table('image_embeddings').select(
            'sku_id, embedding'
        ).range(offset, offset + limit - 1).execute()
    else:
        # Fetch products where style scores are empty
        # Using a raw query via RPC
        result = supabase.rpc('get_products_needing_classification', {
            'p_offset': offset,
            'p_limit': limit
        }).execute()

    return result.data or []


def update_product_scores(
    supabase: Client,
    product_id: str,
    style_scores: Dict[str, float],
    occasion_scores: Dict[str, float],
    pattern_scores: Optional[Dict[str, float]] = None,
    fit: Optional[str] = None,
    length: Optional[str] = None,
    sleeve: Optional[str] = None,
    rise: Optional[str] = None,
) -> bool:
    """Update a single product's computed scores and attributes."""
    try:
        update_data = {
            'computed_style_scores': style_scores,
            'computed_occasion_scores': occasion_scores
        }

        if pattern_scores is not None:
            update_data['computed_pattern_scores'] = pattern_scores
        if fit is not None:
            update_data['fit'] = fit
        if length is not None:
            update_data['length'] = length
        if sleeve is not None:
            update_data['sleeve'] = sleeve
        if rise is not None:
            update_data['rise'] = rise

        supabase.table('products').update(update_data).eq('id', product_id).execute()
        return True
    except Exception as e:
        print(f"[ERROR] Failed to update product {product_id}: {e}")
        return False


def _update_single_product(update: Dict[str, Any], include_patterns: bool, include_attributes: bool) -> bool:
    """Update a single product (for parallel execution)."""
    try:
        supabase = get_thread_local_supabase()
        update_data = {
            'computed_style_scores': update['style_scores'],
            'computed_occasion_scores': update['occasion_scores']
        }

        if include_patterns and 'pattern_scores' in update:
            update_data['computed_pattern_scores'] = update['pattern_scores']

        if include_attributes:
            if update.get('fit'):
                update_data['fit'] = update['fit']
            if update.get('length'):
                update_data['length'] = update['length']
            if update.get('sleeve'):
                update_data['sleeve'] = update['sleeve']
            if update.get('rise'):
                update_data['rise'] = update['rise']

        supabase.table('products').update(update_data).eq('id', update['product_id']).execute()
        return True
    except Exception as e:
        return False


def batch_update_product_scores(
    supabase: Client,
    updates: List[Dict[str, Any]],
    include_patterns: bool = True,
    include_attributes: bool = True,
    num_workers: int = 16,
) -> int:
    """
    Batch update multiple products' scores and attributes using parallel writes.

    Returns number of successful updates.
    """
    if num_workers <= 1:
        # Sequential fallback
        success_count = 0
        for update in updates:
            if _update_single_product(update, include_patterns, include_attributes):
                success_count += 1
        return success_count

    # Parallel updates
    success_count = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_update_single_product, update, include_patterns, include_attributes)
            for update in updates
        ]
        for future in as_completed(futures):
            if future.result():
                success_count += 1

    return success_count


def run_classification(
    batch_size: int = 100,
    dry_run: bool = False,
    limit: Optional[int] = None,
    reclassify_all: bool = False,
    verbose: bool = False,
    include_patterns: bool = True,
    include_attributes: bool = False,
    include_negative_scores: bool = True,
    num_workers: int = 16,
):
    """
    Run batch classification on all products with parallel DB writes.

    Args:
        batch_size: Number of products to process per batch
        dry_run: If True, don't write to database
        limit: Maximum number of products to process (None = all)
        reclassify_all: If True, reclassify even products with existing scores
        verbose: Print detailed progress
        include_patterns: Classify patterns (solid, stripes, floral, etc.)
        include_attributes: Classify fit, length, sleeve attributes
        include_negative_scores: Classify negative occasion scores for hard gating (default True)
        num_workers: Number of parallel workers for DB writes (default 16)
    """
    print("=" * 70)
    print("Batch Product Classification (Parallel)")
    print("=" * 70)

    # Initialize
    supabase = get_supabase_client()
    classifier = StyleClassifier()

    # Force model load
    print("\nLoading classifier model...")
    classifier._ensure_model_loaded()

    # Get total count
    print("\nCounting products to classify...")

    # Simple count using image_embeddings table
    if reclassify_all:
        count_result = supabase.table('image_embeddings').select('sku_id', count='exact').execute()
        total_products = count_result.count or 0
    else:
        # Count products where computed_style_scores is empty or null
        # We'll just process all and let the update handle it
        count_result = supabase.table('image_embeddings').select('sku_id', count='exact').execute()
        total_products = count_result.count or 0

    if limit:
        total_products = min(total_products, limit)

    print(f"Products to process: {total_products}")
    print(f"Batch size: {batch_size}")
    print(f"Parallel workers: {num_workers}")
    print(f"Dry run: {dry_run}")
    print(f"Reclassify all: {reclassify_all}")
    print(f"Include patterns: {include_patterns}")
    print(f"Include attributes: {include_attributes}")
    print(f"Include negative scores: {include_negative_scores}")

    if dry_run:
        print("\n*** DRY RUN MODE - No database writes ***\n")

    # Process in batches
    processed = 0
    updated = 0
    errors = 0
    start_time = time.time()

    offset = 0
    while processed < total_products:
        # Fetch batch of embeddings
        batch_result = supabase.table('image_embeddings').select(
            'sku_id, embedding'
        ).range(offset, offset + batch_size - 1).execute()

        batch = batch_result.data or []
        if not batch:
            break

        # Extract embeddings
        embeddings = []
        product_ids = []

        for row in batch:
            if row.get('embedding') and row.get('sku_id'):
                embeddings.append(row['embedding'])
                product_ids.append(str(row['sku_id']))

        # Fetch product metadata (name, brand, article_type, broad_category) for occasion adjustments
        # Also used for attribute classification
        metadata_map = {}
        categories = []
        if product_ids:
            # Fetch product metadata in smaller batches to avoid query limits
            meta_batch_size = 200  # Supabase limit for IN clause
            for i in range(0, len(product_ids), meta_batch_size):
                batch_ids = product_ids[i:i + meta_batch_size]
                meta_result = supabase.table('products').select(
                    'id, name, brand, article_type, broad_category, category'
                ).in_('id', batch_ids).execute()
                for r in (meta_result.data or []):
                    metadata_map[str(r['id'])] = {
                        'name': r.get('name', ''),
                        'brand': r.get('brand', ''),
                        'article_type': r.get('article_type', ''),
                        'broad_category': r.get('broad_category', ''),
                        'category': r.get('category', ''),
                    }

            # Build ordered lists matching product_ids order
            categories = [metadata_map.get(pid, {}).get('category', '') for pid in product_ids]
        else:
            categories = [''] * len(product_ids)

        # Build product_metadata list for occasion adjustments
        product_metadata = [metadata_map.get(pid, {}) for pid in product_ids]

        if not embeddings:
            offset += batch_size
            continue

        # Classify batch - styles, occasions, patterns (and negative scores if requested)
        # Pass product_metadata for intelligent occasion scoring adjustments
        try:
            if include_negative_scores:
                # Use full classification which includes negative occasion scores
                classifications = classifier.classify_products_batch_full(
                    embeddings,
                    include_patterns=include_patterns,
                    include_negative_scores=True,
                    product_metadata=product_metadata,
                )
            else:
                # Standard classification without negative scores
                classifications = classifier.classify_products_batch(
                    embeddings,
                    include_patterns=include_patterns,
                    product_metadata=product_metadata,
                )
        except Exception as e:
            print(f"[ERROR] Classification failed for batch at offset {offset}: {e}")
            errors += len(embeddings)
            offset += batch_size
            continue

        # Classify attributes if requested
        attributes = None
        if include_attributes:
            try:
                attributes = classifier.classify_attributes_batch(embeddings, categories)
            except Exception as e:
                print(f"[ERROR] Attribute classification failed for batch at offset {offset}: {e}")
                # Continue without attributes

        # Prepare updates
        updates = []
        for i, classification in enumerate(classifications):
            update = {
                'product_id': product_ids[i],
                'style_scores': classification['styles'],
                'occasion_scores': classification['occasions']
            }

            if include_patterns and 'patterns' in classification:
                update['pattern_scores'] = classification['patterns']

            if attributes and i < len(attributes):
                update['fit'] = attributes[i].get('fit')
                update['length'] = attributes[i].get('length')
                update['sleeve'] = attributes[i].get('sleeve')
                update['rise'] = attributes[i].get('rise')

            updates.append(update)

        # Write to database (unless dry run) - parallel writes
        if not dry_run:
            batch_updated = batch_update_product_scores(
                supabase, updates,
                include_patterns=include_patterns,
                include_attributes=include_attributes,
                num_workers=num_workers
            )
            updated += batch_updated
            errors += len(updates) - batch_updated
        else:
            updated += len(updates)

        processed += len(embeddings)
        offset += batch_size

        # Progress update
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total_products - processed) / rate if rate > 0 else 0

        print(f"Progress: {processed}/{total_products} ({processed/total_products*100:.1f}%) | "
              f"Updated: {updated} | Errors: {errors} | "
              f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}m")

        if verbose and updates:
            # Show sample classification
            sample = updates[0]
            print(f"  Sample (product {sample['product_id'][:8]}...):")
            print(f"    Styles: {sample['style_scores']}")
            print(f"    Occasions: {sample['occasion_scores']}")
            if include_patterns:
                print(f"    Patterns: {sample.get('pattern_scores', {})}")
            if include_attributes:
                print(f"    Fit: {sample.get('fit')}, Length: {sample.get('length')}, Sleeve: {sample.get('sleeve')}, Rise: {sample.get('rise')}")

        # Respect rate limits
        if not dry_run:
            time.sleep(0.1)  # Small delay between batches

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("Classification Complete")
    print("=" * 70)
    print(f"Total processed: {processed}")
    print(f"Successfully updated: {updated}")
    print(f"Errors: {errors}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Average rate: {processed/elapsed:.1f} products/second")


def create_rpc_functions(supabase: Client):
    """
    Create helper RPC functions for classification.

    Run this once to set up the helper functions.
    """
    sql = """
    -- Count products needing classification
    CREATE OR REPLACE FUNCTION count_products_needing_classification()
    RETURNS int
    LANGUAGE plpgsql AS $$
    BEGIN
        RETURN (
            SELECT COUNT(*)
            FROM image_embeddings ie
            JOIN products p ON p.id = ie.sku_id
            WHERE (p.computed_style_scores IS NULL OR p.computed_style_scores = '{}')
              AND ie.embedding IS NOT NULL
        );
    END;
    $$;

    -- Get products needing classification
    CREATE OR REPLACE FUNCTION get_products_needing_classification(
        p_offset int DEFAULT 0,
        p_limit int DEFAULT 100
    )
    RETURNS TABLE(sku_id uuid, embedding vector(512))
    LANGUAGE plpgsql AS $$
    BEGIN
        RETURN QUERY
        SELECT ie.sku_id, ie.embedding
        FROM image_embeddings ie
        JOIN products p ON p.id = ie.sku_id
        WHERE (p.computed_style_scores IS NULL OR p.computed_style_scores = '{}')
          AND ie.embedding IS NOT NULL
        ORDER BY ie.sku_id
        OFFSET p_offset
        LIMIT p_limit;
    END;
    $$;

    -- Get embeddings with category (for attribute classification)
    CREATE OR REPLACE FUNCTION get_embeddings_with_category(
        p_offset int DEFAULT 0,
        p_limit int DEFAULT 100
    )
    RETURNS TABLE(sku_id uuid, embedding vector(512), broad_category text)
    LANGUAGE plpgsql AS $$
    BEGIN
        RETURN QUERY
        SELECT ie.sku_id, ie.embedding, p.broad_category
        FROM image_embeddings ie
        JOIN products p ON p.id = ie.sku_id
        WHERE ie.embedding IS NOT NULL
        ORDER BY ie.sku_id
        OFFSET p_offset
        LIMIT p_limit;
    END;
    $$;

    GRANT EXECUTE ON FUNCTION count_products_needing_classification TO anon, authenticated;
    GRANT EXECUTE ON FUNCTION get_products_needing_classification TO anon, authenticated;
    GRANT EXECUTE ON FUNCTION get_embeddings_with_category TO anon, authenticated;
    """

    print("Creating helper RPC functions...")
    # Note: This would need to be run via Supabase dashboard or psql
    print("SQL to run:")
    print(sql)


def main():
    parser = argparse.ArgumentParser(description='Batch classify products for lifestyle filtering (parallel)')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Number of products per batch (default: 1000)')
    parser.add_argument('--workers', type=int, default=32,
                        help='Number of parallel workers for DB writes (default: 32)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run without writing to database')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of products to process')
    parser.add_argument('--reclassify-all', '--all', action='store_true',
                        help='Reclassify all products, even those with existing scores')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed progress')
    parser.add_argument('--create-rpc', action='store_true',
                        help='Print SQL for helper RPC functions')
    parser.add_argument('--include-patterns', action='store_true', default=True,
                        help='Classify patterns (solid, stripes, floral, etc.) - enabled by default')
    parser.add_argument('--no-patterns', action='store_true',
                        help='Skip pattern classification')
    parser.add_argument('--include-attributes', action='store_true',
                        help='Classify fit, length, sleeve, rise attributes (requires category data)')
    parser.add_argument('--include-negative-scores', action='store_true', default=True,
                        help='Classify negative occasion scores for hard gating (enabled by default)')
    parser.add_argument('--no-negative-scores', action='store_true',
                        help='Skip negative occasion score classification')

    args = parser.parse_args()

    if args.create_rpc:
        supabase = get_supabase_client()
        create_rpc_functions(supabase)
        return

    include_patterns = args.include_patterns and not args.no_patterns
    include_negative_scores = args.include_negative_scores and not args.no_negative_scores

    run_classification(
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        limit=args.limit,
        reclassify_all=args.reclassify_all,
        verbose=args.verbose,
        include_patterns=include_patterns,
        include_attributes=args.include_attributes,
        include_negative_scores=include_negative_scores,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
