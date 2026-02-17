"""
Generate multimodal embeddings for all products.

Combines FashionCLIP image embeddings (visual) with FashionCLIP text embeddings
(product name + Gemini-extracted attributes) into a single 512d vector per product.

    combined = alpha * image_embedding + (1 - alpha) * text_embedding
    combined = combined / ||combined||

This enables semantic search that matches descriptive terms like "ribbed",
"quilted", or "turtleneck" that exist in product text but not in images.

Two versions are generated for A/B testing:
    v1: Structured attributes only (name, brand, category, color, fabric, etc.)
    v2: Structured attributes + source_description excerpt

Usage:
    # Generate v1 (attributes only) with 4 DB workers
    PYTHONPATH=src python scripts/generate_multimodal_embeddings.py --version 1 --workers 4

    # Generate v2 (attributes + description)
    PYTHONPATH=src python scripts/generate_multimodal_embeddings.py --version 2

    # Generate both versions
    PYTHONPATH=src python scripts/generate_multimodal_embeddings.py --version both

    # Resume from where you left off (skips already-processed products)
    PYTHONPATH=src python scripts/generate_multimodal_embeddings.py --version 1 --resume

    # Process a specific batch size with more workers
    PYTHONPATH=src python scripts/generate_multimodal_embeddings.py --version 1 --batch-size 200 --workers 6

    # Dry run (don't write to DB, just show text templates)
    PYTHONPATH=src python scripts/generate_multimodal_embeddings.py --version 1 --dry-run --limit 5
"""

import argparse
import json
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Attribute Text Templates
# ============================================================================

_SKIP_VALUES = {None, "", "N/A", "n/a", "None", "null", "Unknown", "Other"}


def _safe(val: Any) -> Optional[str]:
    """Return string value if not a skip value, else None."""
    if val in _SKIP_VALUES:
        return None
    if isinstance(val, str):
        return val.strip() if val.strip() not in _SKIP_VALUES else None
    return None


def build_attribute_text_v1(
    product: Dict[str, Any],
    attributes: Dict[str, Any],
) -> str:
    """
    Build attribute text string (v1: structured attributes only).

    Example output:
        "Ribbed Turtleneck by Another Tomorrow. Tops, Sweater. Black, Neutrals.
         Solid pattern. Knit fabric. Quilted texture. Slim fit. Turtleneck neckline.
         Long sleeves. Minimalist, Classic. Everyday, Work. Fall, Winter."

    Designed to fit within CLIP's 77-token limit (~60 tokens typical).
    """
    parts = []

    name = _safe(product.get("name"))
    if name:
        parts.append(name)

    brand = _safe(product.get("brand"))
    if brand:
        parts.append(f"by {brand}")

    cat_parts = []
    cat_l1 = _safe(attributes.get("category_l1"))
    cat_l2 = _safe(attributes.get("category_l2"))
    cat_l3 = _safe(attributes.get("category_l3"))
    if cat_l1:
        cat_parts.append(cat_l1)
    if cat_l2:
        cat_parts.append(cat_l2)
    if cat_l3 and cat_l3 != cat_l2:
        cat_parts.append(cat_l3)
    if cat_parts:
        parts.append(", ".join(cat_parts))

    color_parts = []
    primary_color = _safe(attributes.get("primary_color"))
    color_family = _safe(attributes.get("color_family"))
    if primary_color:
        color_parts.append(primary_color)
    if color_family and color_family != primary_color:
        color_parts.append(color_family)
    if color_parts:
        parts.append(", ".join(color_parts))

    pattern = _safe(attributes.get("pattern"))
    if pattern and pattern.lower() != "solid":
        parts.append(f"{pattern} pattern")

    fabric = _safe(attributes.get("apparent_fabric"))
    if fabric:
        parts.append(f"{fabric} fabric")

    texture = _safe(attributes.get("texture"))
    if texture and texture.lower() not in ("smooth", "standard"):
        parts.append(f"{texture} texture")

    construction = attributes.get("construction") or {}
    if isinstance(construction, str):
        try:
            construction = json.loads(construction)
        except (json.JSONDecodeError, TypeError):
            construction = {}

    neckline = _safe(construction.get("neckline"))
    if neckline:
        parts.append(f"{neckline} neckline")

    sleeve_type = _safe(construction.get("sleeve_type"))
    if sleeve_type:
        parts.append(f"{sleeve_type} sleeves")

    length = _safe(construction.get("length"))
    if length and length.lower() != "regular":
        parts.append(f"{length} length")

    fit_type = _safe(attributes.get("fit_type"))
    if fit_type:
        parts.append(f"{fit_type} fit")

    silhouette = _safe(attributes.get("silhouette"))
    if silhouette and silhouette != fit_type:
        parts.append(f"{silhouette} silhouette")

    style_tags = attributes.get("style_tags") or []
    if isinstance(style_tags, list) and style_tags:
        valid_tags = [t for t in style_tags[:3] if _safe(t)]
        if valid_tags:
            parts.append(", ".join(valid_tags))

    occasions = attributes.get("occasions") or []
    if isinstance(occasions, list) and occasions:
        valid_occasions = [o for o in occasions[:2] if _safe(o)]
        if valid_occasions:
            parts.append(", ".join(valid_occasions))

    seasons = attributes.get("seasons") or []
    if isinstance(seasons, list) and seasons:
        valid_seasons = [s for s in seasons[:2] if _safe(s)]
        if valid_seasons:
            parts.append(", ".join(valid_seasons))

    formality = _safe(attributes.get("formality"))
    if formality and formality.lower() != "casual":
        parts.append(f"{formality}")

    return ". ".join(parts)


def build_attribute_text_v2(
    product: Dict[str, Any],
    attributes: Dict[str, Any],
) -> str:
    """
    Build attribute text string (v2: structured attributes + description excerpt).

    Appends first ~100 chars of source_description to v1 text.
    May approach the 77-token limit for products with long descriptions.
    """
    base = build_attribute_text_v1(product, attributes)

    description = _safe(attributes.get("source_description"))
    if description:
        excerpt = description[:100]
        last_space = excerpt.rfind(" ")
        if last_space > 50:
            excerpt = excerpt[:last_space]
        base = f"{base}. {excerpt}"

    return base


# ============================================================================
# DB I/O Functions
# ============================================================================

def fetch_products_with_attributes(
    supabase_client,
    batch_size: int = 500,
    offset: int = 0,
) -> List[Dict]:
    """Fetch products with their Gemini attributes."""
    resp = (
        supabase_client.table("product_attributes")
        .select(
            "sku_id, apparent_fabric, texture, pattern, primary_color, color_family, "
            "category_l1, category_l2, category_l3, construction, fit_type, formality, "
            "silhouette, style_tags, occasions, seasons, sheen, stretch, rise, "
            "source_description"
        )
        .range(offset, offset + batch_size - 1)
        .execute()
    )
    return resp.data or []


def fetch_product_info(
    supabase_client,
    product_ids: List[str],
) -> Dict[str, Dict]:
    """Fetch product name, brand, article_type for a list of product IDs."""
    if not product_ids:
        return {}
    resp = (
        supabase_client.table("products")
        .select("id, name, brand, article_type, broad_category, in_stock")
        .in_("id", product_ids)
        .execute()
    )
    return {str(row["id"]): row for row in (resp.data or [])}


def fetch_primary_image_embeddings(
    supabase_client,
    product_ids: List[str],
) -> Dict[str, np.ndarray]:
    """Fetch the primary image embedding for each product."""
    if not product_ids:
        return {}
    result = {}
    for i in range(0, len(product_ids), 100):
        chunk = product_ids[i:i + 100]
        resp = (
            supabase_client.table("image_embeddings")
            .select("sku_id, embedding")
            .in_("sku_id", chunk)
            .not_.is_("sku_id", "null")
            .order("id")
            .execute()
        )
        for row in (resp.data or []):
            sid = str(row["sku_id"])
            if sid not in result:
                emb_str = row["embedding"]
                if isinstance(emb_str, str):
                    emb_str = emb_str.strip("[]")
                    emb = np.array([float(x) for x in emb_str.split(",")], dtype=np.float32)
                elif isinstance(emb_str, list):
                    emb = np.array(emb_str, dtype=np.float32)
                else:
                    continue
                result[sid] = emb
    return result


def fetch_existing_product_ids(supabase_client, version: int) -> set:
    """Fetch product_ids that already have multimodal embeddings."""
    existing = set()
    offset = 0
    batch_size = 1000
    while True:
        resp = (
            supabase_client.table("product_multimodal_embeddings")
            .select("product_id")
            .eq("version", version)
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            break
        for row in rows:
            existing.add(str(row["product_id"]))
        offset += batch_size
    return existing


def upsert_records(supabase_client, records: List[Dict]) -> Tuple[int, int]:
    """Upsert a batch of records. Returns (success_count, error_count)."""
    if not records:
        return 0, 0
    try:
        supabase_client.table("product_multimodal_embeddings").upsert(
            records, on_conflict="product_id,version",
        ).execute()
        return len(records), 0
    except Exception as e:
        print(f"  ERROR upserting batch of {len(records)}: {e}")
        return 0, len(records)


# ============================================================================
# Embedding Functions
# ============================================================================

def combine_embeddings(
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """Combine image and text embeddings with weighted average + L2 norm."""
    combined = alpha * image_emb + (1.0 - alpha) * text_emb
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    return combined


def embedding_to_pgvector_str(emb: np.ndarray) -> str:
    """Convert numpy embedding to pgvector string format."""
    return "[" + ",".join(f"{x:.8f}" for x in emb) + "]"


def encode_text_batch(encode_fn, texts: List[str]) -> List[np.ndarray]:
    """Encode a list of texts. FashionCLIP processes one at a time."""
    return [encode_fn(t) for t in texts]


# ============================================================================
# Parallel Batch Prefetcher
# ============================================================================

class BatchPrefetcher:
    """
    Pre-fetches DB data (attributes, product info, image embeddings) in
    background threads while the main thread runs FashionCLIP encoding.

    Architecture:
        Thread pool fetches next batch's data from Supabase while
        main thread encodes current batch with FashionCLIP.
    """

    def __init__(self, supabase_client, num_workers: int = 4):
        self.sb = supabase_client
        self.pool = ThreadPoolExecutor(max_workers=num_workers)

    def prefetch_batch_data(
        self,
        attrs_batch: List[Dict],
        existing_ids: set,
    ) -> Dict:
        """
        Fetch all data needed for a batch in parallel.

        Runs 2 parallel DB queries:
        1. Product info (name, brand, in_stock)
        2. Image embeddings

        Returns dict with product_info, image_embeddings, attr_by_id, to_process_ids.
        """
        product_ids = [str(row["sku_id"]) for row in attrs_batch]
        to_process_ids = [pid for pid in product_ids if pid not in existing_ids]

        if not to_process_ids:
            return {
                "to_process_ids": [],
                "product_info": {},
                "image_embeddings": {},
                "attr_by_id": {},
                "skipped": len(product_ids),
            }

        # Run both DB fetches in parallel
        future_info = self.pool.submit(fetch_product_info, self.sb, to_process_ids)
        future_embs = self.pool.submit(fetch_primary_image_embeddings, self.sb, to_process_ids)

        product_info = future_info.result()
        image_embeddings = future_embs.result()
        attr_by_id = {str(row["sku_id"]): row for row in attrs_batch}

        return {
            "to_process_ids": to_process_ids,
            "product_info": product_info,
            "image_embeddings": image_embeddings,
            "attr_by_id": attr_by_id,
            "skipped": len(product_ids) - len(to_process_ids),
        }

    def shutdown(self):
        self.pool.shutdown(wait=False)


# ============================================================================
# Core Processing
# ============================================================================

def process_batch_parallel(
    prefetcher: BatchPrefetcher,
    encode_text_fn,
    attrs_batch: List[Dict],
    version: int,
    alpha: float,
    existing_ids: set,
    dry_run: bool = False,
    upsert_workers: int = 2,
) -> Tuple[int, int, int]:
    """
    Process a batch with parallel DB I/O.

    1. Prefetch product info + image embeddings in parallel threads
    2. Build text strings (CPU, fast)
    3. Encode all texts with FashionCLIP (CPU/GPU, bottleneck)
    4. Combine embeddings (CPU, fast)
    5. Upsert in background thread

    Returns (processed, skipped, errors).
    """
    # Step 1: Parallel DB fetch
    data = prefetcher.prefetch_batch_data(attrs_batch, existing_ids)

    to_process_ids = data["to_process_ids"]
    product_info = data["product_info"]
    image_embeddings = data["image_embeddings"]
    attr_by_id = data["attr_by_id"]
    skipped = data["skipped"]
    processed = 0
    errors = 0

    if not to_process_ids:
        return 0, skipped, 0

    # Step 2: Build text strings + filter valid products
    items_to_encode = []  # (pid, product, attrs, img_emb, text)

    for pid in to_process_ids:
        product = product_info.get(pid)
        attrs = attr_by_id.get(pid)

        if not product or not attrs:
            errors += 1
            continue

        if not product.get("in_stock", True):
            skipped += 1
            continue

        img_emb = image_embeddings.get(pid)
        if img_emb is None:
            skipped += 1
            continue

        text = build_attribute_text_v1(product, attrs) if version == 1 else build_attribute_text_v2(product, attrs)

        if dry_run:
            print(f"\n--- {product.get('name', '?')} ({product.get('brand', '?')}) ---")
            print(f"  Text (v{version}): {text}")
            print(f"  Image embedding: {img_emb.shape}, norm={np.linalg.norm(img_emb):.4f}")
            processed += 1
            continue

        items_to_encode.append((pid, product, attrs, img_emb, text))

    if dry_run or not items_to_encode:
        return processed, skipped, errors

    # Step 3: Batch encode all texts with FashionCLIP
    texts = [item[4] for item in items_to_encode]
    try:
        text_embeddings = encode_text_batch(encode_text_fn, texts)
    except Exception as e:
        print(f"  ERROR batch encoding: {e}")
        return 0, skipped, len(items_to_encode)

    # Step 4: Combine embeddings and build upsert records
    records = []
    for i, (pid, product, attrs, img_emb, text) in enumerate(items_to_encode):
        text_emb = text_embeddings[i]
        multimodal_emb = combine_embeddings(img_emb, text_emb, alpha=alpha)

        records.append({
            "product_id": pid,
            "version": version,
            "image_embedding": embedding_to_pgvector_str(img_emb),
            "text_embedding": embedding_to_pgvector_str(text_emb),
            "multimodal_embedding": embedding_to_pgvector_str(multimodal_emb),
            "text_used": text[:2000],
            "alpha": alpha,
        })
        processed += 1

    # Step 5: Upsert in chunks (parallel if multiple chunks)
    chunk_size = 50
    chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]

    with ThreadPoolExecutor(max_workers=upsert_workers) as upsert_pool:
        futures = [
            upsert_pool.submit(upsert_records, prefetcher.sb, chunk)
            for chunk in chunks
        ]
        for future in as_completed(futures):
            ok, err = future.result()
            if err:
                errors += err
                processed -= err

    return processed, skipped, errors


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate multimodal embeddings (image + text) for all products."
    )
    parser.add_argument(
        "--version", type=str, default="1",
        choices=["1", "2", "both"],
        help="Embedding version: 1=attributes only, 2=attributes+description, both=generate both",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.6,
        help="Image embedding weight (text weight = 1 - alpha). Default: 0.6",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Products per batch. Default: 100",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of DB I/O worker threads. Default: 4",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip products that already have embeddings for this version",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Don't write to DB, just show text templates for first N products",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max products to process (useful for testing). Default: all",
    )

    args = parser.parse_args()
    versions = [1, 2] if args.version == "both" else [int(args.version)]

    print("=" * 60)
    print("Multimodal Embedding Generator")
    print("=" * 60)
    print(f"  Versions:    {versions}")
    print(f"  Alpha:       {args.alpha} (image) / {1 - args.alpha:.1f} (text)")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Workers:     {args.workers} (DB I/O threads)")
    print(f"  Resume:      {args.resume}")
    print(f"  Dry run:     {args.dry_run}")
    print(f"  Limit:       {args.limit or 'all'}")
    print()

    # Initialize Supabase client
    from config.database import get_supabase_client
    sb = get_supabase_client()
    print("[1/3] Supabase client initialized")

    # Initialize FashionCLIP model (skip for dry run to save time)
    encode_text_fn = None
    if not args.dry_run:
        from women_search_engine import get_women_search_engine
        engine = get_women_search_engine()
        engine._load_model()  # Force model load now
        encode_text_fn = engine.encode_text
        print("[2/3] FashionCLIP model loaded")
    else:
        print("[2/3] FashionCLIP model skipped (dry run)")

    # Count total product_attributes
    count_resp = sb.table("product_attributes").select("sku_id", count="exact").limit(1).execute()
    total_attrs = count_resp.count or 0
    print(f"[3/3] Found {total_attrs:,} product attribute records")
    print()

    # Create prefetcher with worker threads
    prefetcher = BatchPrefetcher(sb, num_workers=args.workers)

    for version in versions:
        print(f"{'='*60}")
        print(f"Generating v{version} embeddings...")
        print(f"{'='*60}")

        # Fetch existing IDs if resuming
        existing_ids = set()
        if args.resume:
            existing_ids = fetch_existing_product_ids(sb, version)
            print(f"  Found {len(existing_ids):,} existing v{version} embeddings (will skip)")

        offset = 0
        total_processed = 0
        total_skipped = 0
        total_errors = 0
        t_start = time.time()

        # Pipeline: prefetch next batch while processing current batch
        # Fetch first batch
        next_batch = fetch_products_with_attributes(sb, args.batch_size, offset)

        while next_batch:
            current_batch = next_batch
            offset += args.batch_size

            # Check limit
            if args.limit and total_processed >= args.limit:
                print(f"\n  Reached limit of {args.limit} products")
                break

            # Start prefetching next batch in background
            next_future = prefetcher.pool.submit(
                fetch_products_with_attributes, sb, args.batch_size, offset
            )

            # Process current batch
            batch_t = time.time()
            processed, skipped, errors = process_batch_parallel(
                prefetcher=prefetcher,
                encode_text_fn=encode_text_fn,
                attrs_batch=current_batch,
                version=version,
                alpha=args.alpha,
                existing_ids=existing_ids,
                dry_run=args.dry_run,
                upsert_workers=min(2, args.workers),
            )

            total_processed += processed
            total_skipped += skipped
            total_errors += errors
            batch_ms = int((time.time() - batch_t) * 1000)

            # Progress
            progress = offset
            elapsed = time.time() - t_start
            rate = total_processed / elapsed if elapsed > 0 else 0
            eta = (total_attrs - progress) / rate / 60 if rate > 0 else 0

            batch_num = (offset - args.batch_size) // args.batch_size + 1
            print(
                f"  Batch {batch_num}: "
                f"processed={processed}, skipped={skipped}, errors={errors} "
                f"({batch_ms}ms) | "
                f"Total: {total_processed:,}/{total_attrs:,} "
                f"({rate:.1f}/s, ETA {eta:.1f}min)"
            )

            # Get prefetched next batch
            next_batch = next_future.result()

            # Check limit again
            if args.limit and total_processed >= args.limit:
                break

        elapsed = time.time() - t_start
        print(f"\n  v{version} complete:")
        print(f"    Processed: {total_processed:,}")
        print(f"    Skipped:   {total_skipped:,}")
        print(f"    Errors:    {total_errors:,}")
        print(f"    Time:      {elapsed:.1f}s ({elapsed/60:.1f}min)")
        if total_processed > 0:
            print(f"    Rate:      {total_processed/elapsed:.1f} products/sec")
        print()

    prefetcher.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
