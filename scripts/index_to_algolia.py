#!/usr/bin/env python3
"""
Bulk index products from Supabase to Algolia.

Fetches products + product_attributes (Gemini Vision) and pushes
them to the Algolia 'products' index.

Usage:
    # From project root, with venv active:
    PYTHONPATH=src python scripts/index_to_algolia.py

    # Dry run - just count products:
    PYTHONPATH=src python scripts/index_to_algolia.py --dry-run

    # Index specific brand:
    PYTHONPATH=src python scripts/index_to_algolia.py --brand "Boohoo"

    # Clear and re-index:
    PYTHONPATH=src python scripts/index_to_algolia.py --clear-first

    # Only configure settings + synonyms (no indexing):
    PYTHONPATH=src python scripts/index_to_algolia.py --configure-only

    # Only index products missing from Algolia:
    PYTHONPATH=src python scripts/index_to_algolia.py --missing-only
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from supabase import create_client

from search.algolia_client import AlgoliaClient
from search.algolia_config import product_to_algolia_record


def get_supabase():
    """Create Supabase client."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    return create_client(url, key)


def fetch_products_batch(supabase, offset: int, batch_size: int, brand: str = None):
    """Fetch a batch of products with their attributes."""
    query = supabase.table("products").select(
        "id, name, brand, category, broad_category, article_type, "
        "price, original_price, in_stock, fit, length, sleeve, "
        "neckline, rise, base_color, colors, materials, style_tags, "
        "primary_image_url, gallery_images, trending_score, created_at, "
        "gender, "
        "product_attributes!left("
        "  category_l1, category_l2, category_l3, "
        "  construction, primary_color, color_family, secondary_colors, "
        "  pattern, pattern_scale, apparent_fabric, texture, sheen, "
        "  style_tags, occasions, seasons, formality, trend_tags, "
        "  fit_type, stretch, rise, leg_shape, silhouette"
        ")"
    ).eq("in_stock", True)

    if brand:
        query = query.eq("brand", brand)

    result = query.range(offset, offset + batch_size - 1).execute()
    return result.data or []


def fetch_total_count(supabase, brand: str = None) -> int:
    """Get total product count."""
    query = supabase.table("products").select("id", count="exact").eq("in_stock", True)
    if brand:
        query = query.eq("brand", brand)
    result = query.execute()
    return result.count or 0


def extract_brands(supabase) -> list:
    """Extract unique brands from the products table."""
    all_brands = set()
    offset = 0
    batch_size = 1000

    while True:
        result = (
            supabase.table("products")
            .select("brand")
            .eq("in_stock", True)
            .neq("brand", None)
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        if not result.data:
            break
        for row in result.data:
            b = row.get("brand")
            if b:
                all_brands.add(b)
        if len(result.data) < batch_size:
            break
        offset += batch_size

    return sorted(all_brands)


def fetch_all_product_ids(supabase, brand: str = None) -> set:
    """Fetch all in-stock product IDs from Supabase using cursor-based pagination."""
    all_ids = set()
    # Supabase caps at 1000 rows per request; use order+gt cursor to go beyond 65K
    batch_size = 1000
    last_id = None

    while True:
        query = supabase.table("products").select("id").eq("in_stock", True).order("id")
        if brand:
            query = query.eq("brand", brand)
        if last_id:
            query = query.gt("id", last_id)
        result = query.limit(batch_size).execute()
        if not result.data:
            break
        for row in result.data:
            all_ids.add(str(row["id"]))
        last_id = result.data[-1]["id"]
        if len(result.data) < batch_size:
            break
        if len(all_ids) % 10000 == 0:
            print(f"    ... fetched {len(all_ids)} IDs so far")

    return all_ids


def fetch_algolia_object_ids(algolia) -> set:
    """Fetch all objectIDs currently in the Algolia index via browse API."""
    from algoliasearch.search.models.browse_params_object import BrowseParamsObject

    algolia_ids = set()
    print("  Browsing Algolia index for existing objectIDs...")

    def aggregator(response):
        hits = response.hits or []
        for hit in hits:
            oid = hit.get("objectID") if isinstance(hit, dict) else getattr(hit, "object_id", None)
            if oid:
                algolia_ids.add(str(oid))
        if len(algolia_ids) % 20000 < 1001:
            print(f"    ... found {len(algolia_ids)} Algolia records so far")

    try:
        algolia._client.browse_objects(
            index_name=algolia.index_name,
            aggregator=aggregator,
            browse_params=BrowseParamsObject(
                attributes_to_retrieve=["objectID"],
                hits_per_page=1000,
            ),
        )
    except Exception as e:
        print(f"  [WARN] browse_objects failed: {e}")
        print(f"  Got {len(algolia_ids)} IDs before failure")

    return algolia_ids


def fetch_products_by_ids(supabase, product_ids: list):
    """Fetch products with attributes by specific IDs (batch of up to 100)."""
    query = supabase.table("products").select(
        "id, name, brand, category, broad_category, article_type, "
        "price, original_price, in_stock, fit, length, sleeve, "
        "neckline, rise, base_color, colors, materials, style_tags, "
        "primary_image_url, gallery_images, trending_score, created_at, "
        "gender, "
        "product_attributes!left("
        "  category_l1, category_l2, category_l3, "
        "  construction, primary_color, color_family, secondary_colors, "
        "  pattern, pattern_scale, apparent_fabric, texture, sheen, "
        "  style_tags, occasions, seasons, formality, trend_tags, "
        "  fit_type, stretch, rise, leg_shape, silhouette"
        ")"
    ).in_("id", product_ids)

    result = query.execute()
    return result.data or []


def save_with_retry(algolia, records, batch_size=500, max_retries=3, retry_delay=5):
    """Save records to Algolia with retry logic on timeout/error."""
    for attempt in range(1, max_retries + 1):
        try:
            algolia.save_objects(records, batch_size=batch_size)
            return True, len(records)
        except Exception as e:
            err_str = str(e)
            if attempt < max_retries:
                # On timeout or transient error, try smaller batches
                if "timeout" in err_str.lower() or "timed out" in err_str.lower():
                    smaller_batch = max(50, batch_size // 2)
                    print(f"    [RETRY {attempt}/{max_retries}] Timeout, retrying with batch_size={smaller_batch} in {retry_delay}s...")
                    time.sleep(retry_delay)
                    batch_size = smaller_batch
                    continue
                else:
                    print(f"    [RETRY {attempt}/{max_retries}] Error: {e}, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
            else:
                print(f"    [FAILED] All {max_retries} attempts failed: {e}")
                return False, 0
    return False, 0


def main():
    parser = argparse.ArgumentParser(description="Index products to Algolia")
    parser.add_argument("--dry-run", action="store_true", help="Just count, don't index")
    parser.add_argument("--brand", type=str, help="Filter by brand")
    parser.add_argument("--clear-first", action="store_true", help="Clear index before indexing")
    parser.add_argument("--configure-only", action="store_true", help="Only set settings + synonyms")
    parser.add_argument("--batch-size", type=int, default=500, help="Records per batch (default 500, lower = fewer timeouts)")
    parser.add_argument("--extract-brands", action="store_true", help="Print all brand names and exit")
    parser.add_argument("--missing-only", action="store_true", help="Only index products not already in Algolia")
    parser.add_argument("--retries", type=int, default=3, help="Max retries per batch on failure")
    args = parser.parse_args()

    supabase = get_supabase()
    algolia = AlgoliaClient()

    # --extract-brands: just print brand names
    if args.extract_brands:
        print("Extracting brand names from products table...")
        brands = extract_brands(supabase)
        print(f"\nFound {len(brands)} brands:\n")
        for b in brands:
            print(f"  {b}")
        return

    # Configure settings + synonyms
    print("Configuring Algolia index settings...")
    resp = algolia.configure_index()
    print(f"  Settings applied: {resp}")

    print("Configuring synonyms...")
    resp = algolia.configure_synonyms()
    print(f"  Synonyms applied: {resp}")

    if args.configure_only:
        print("\nDone (configure-only mode).")
        return

    # Count products
    total = fetch_total_count(supabase, args.brand)
    print(f"\nTotal in-stock products: {total}")
    if args.brand:
        print(f"  Brand filter: {args.brand}")

    # --missing-only: find products not yet in Algolia
    missing_ids = None
    if args.missing_only:
        print("\n--- Missing-Only Mode ---")
        print("Fetching all product IDs from Supabase...")
        supabase_ids = fetch_all_product_ids(supabase, args.brand)
        print(f"  Supabase in-stock: {len(supabase_ids)}")

        print("Fetching existing objectIDs from Algolia...")
        algolia_ids = fetch_algolia_object_ids(algolia)
        print(f"  Algolia indexed: {len(algolia_ids)}")

        missing_ids = supabase_ids - algolia_ids
        print(f"  Missing from Algolia: {len(missing_ids)}")

        if not missing_ids:
            print("\nAll products are already indexed! Nothing to do.")
            return

        if args.dry_run:
            print(f"\n[DRY RUN] Would index {len(missing_ids)} missing products.")
            # Show a sample
            sample = sorted(list(missing_ids))[:10]
            print(f"  Sample IDs: {sample}")
            return

        total = len(missing_ids)
        print(f"\nWill index {total} missing products (direct ID fetch)...")

        # --- Direct ID-based indexing for missing products ---
        indexed = 0
        skipped = 0
        failed = 0
        # Supabase .in_() works best with chunks of ~100 IDs
        id_batch_size = 100
        missing_list = sorted(list(missing_ids))
        t_start = time.time()

        for i in range(0, len(missing_list), id_batch_size):
            chunk_ids = missing_list[i : i + id_batch_size]

            # Fetch this chunk directly by IDs
            try:
                products = fetch_products_by_ids(supabase, chunk_ids)
            except Exception as e:
                print(f"  [ERROR] Fetch chunk at {i}: {e}")
                failed += len(chunk_ids)
                continue

            # Convert to Algolia records
            records = []
            for product in products:
                attrs_list = product.pop("product_attributes", None)
                attrs = None
                if attrs_list and isinstance(attrs_list, list) and len(attrs_list) > 0:
                    attrs = attrs_list[0]
                elif attrs_list and isinstance(attrs_list, dict):
                    attrs = attrs_list

                try:
                    record = product_to_algolia_record(product, attrs)
                    records.append(record)
                except Exception as e:
                    skipped += 1
                    if skipped <= 10:
                        print(f"  [SKIP] {product.get('id')}: {e}")

            # Push to Algolia with retry
            if records:
                success, count = save_with_retry(
                    algolia, records,
                    batch_size=min(args.batch_size, 500),
                    max_retries=args.retries,
                )
                if success:
                    indexed += count
                else:
                    failed += len(records)

            # Progress every 10 chunks
            if (i // id_batch_size) % 10 == 0 or i + id_batch_size >= len(missing_list):
                elapsed = time.time() - t_start
                rate = indexed / elapsed if elapsed > 0 else 0
                pct = min(100.0, (i + len(chunk_ids)) / total * 100)
                print(
                    f"  [{pct:5.1f}%] Indexed: {indexed} | "
                    f"Skipped: {skipped} | "
                    f"Failed: {failed} | "
                    f"Rate: {rate:.0f}/s | "
                    f"Elapsed: {elapsed:.0f}s"
                )

        # Summary
        elapsed = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"MISSING-ONLY INDEXING COMPLETE")
        print(f"{'='*60}")
        print(f"  Target:         {total}")
        print(f"  Total indexed:  {indexed}")
        print(f"  Skipped:        {skipped}")
        print(f"  Failed:         {failed}")
        print(f"  Time:           {elapsed:.1f}s")
        print(f"  Rate:           {indexed / elapsed:.0f} products/s" if elapsed > 0 else "")
        print(f"{'='*60}")
        return

    elif args.dry_run:
        print("\n[DRY RUN] Would index the above products. Run without --dry-run to execute.")
        return

    # Clear if requested
    if args.clear_first:
        print("\nClearing existing index objects...")
        algolia.clear_objects()
        print("  Index cleared.")

    # Index in batches (full re-index mode)
    fetch_batch_size = args.batch_size
    print(f"\nIndexing {total} products in batches of {fetch_batch_size}...")
    offset = 0
    indexed = 0
    skipped = 0
    failed = 0
    t_start = time.time()

    while offset < total:
        # Fetch batch from Supabase
        products = fetch_products_batch(supabase, offset, fetch_batch_size, args.brand)
        if not products:
            break

        # Convert to Algolia records
        records = []
        for product in products:
            # product_attributes comes as a list (left join) - take first if exists
            attrs_list = product.pop("product_attributes", None)
            attrs = None
            if attrs_list and isinstance(attrs_list, list) and len(attrs_list) > 0:
                attrs = attrs_list[0]
            elif attrs_list and isinstance(attrs_list, dict):
                attrs = attrs_list

            try:
                record = product_to_algolia_record(product, attrs)
                records.append(record)
            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    print(f"  [SKIP] {product.get('id')}: {e}")

        # Push to Algolia with retry
        if records:
            success, count = save_with_retry(
                algolia, records,
                batch_size=min(args.batch_size, 500),
                max_retries=args.retries,
            )
            if success:
                indexed += count
            else:
                failed += len(records)
                print(f"  [FAILED] Batch at offset {offset}: {len(records)} records lost")

        # Progress
        elapsed = time.time() - t_start
        rate = indexed / elapsed if elapsed > 0 else 0
        pct = (offset + len(products)) / total * 100 if total > 0 else 100
        print(
            f"  [{pct:5.1f}%] Indexed: {indexed} | "
            f"Skipped: {skipped} | "
            f"Failed: {failed} | "
            f"Rate: {rate:.0f}/s | "
            f"Elapsed: {elapsed:.0f}s"
        )

        offset += len(products)

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"INDEXING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total indexed:  {indexed}")
    print(f"  Skipped:        {skipped}")
    print(f"  Failed:         {failed}")
    print(f"  Time:           {elapsed:.1f}s")
    print(f"  Rate:           {indexed / elapsed:.0f} products/s" if elapsed > 0 else "")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
