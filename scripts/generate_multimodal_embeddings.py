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
    # Generate v1 (attributes only)
    PYTHONPATH=src python scripts/generate_multimodal_embeddings.py --version 1

    # Generate v2 (attributes + description)
    PYTHONPATH=src python scripts/generate_multimodal_embeddings.py --version 2

    # Generate both versions
    PYTHONPATH=src python scripts/generate_multimodal_embeddings.py --version both

    # Resume from where you left off (skips already-processed products)
    PYTHONPATH=src python scripts/generate_multimodal_embeddings.py --version 1 --resume

    # Faster batching with shorter delay (if Supabase plan allows)
    PYTHONPATH=src python scripts/generate_multimodal_embeddings.py --version 1 --batch-delay 0.2

    # Dry run (don't write to DB, just show text templates)
    PYTHONPATH=src python scripts/generate_multimodal_embeddings.py --version 1 --dry-run --limit 5
"""

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Retry Helper
# ============================================================================

def retry_with_backoff(fn, *args, max_retries=3, base_delay=1.0, **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"    Retry {attempt+1}/{max_retries} after {delay:.1f}s: {e}")
            time.sleep(delay)


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
    """Upsert a batch of records with retry. Returns (success_count, error_count)."""
    if not records:
        return 0, 0
    total_ok = 0
    total_err = 0
    # Small chunks to avoid overwhelming Supabase
    chunk_size = 25
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        try:
            retry_with_backoff(
                lambda c=chunk: supabase_client.table("product_multimodal_embeddings").upsert(
                    c, on_conflict="product_id,version",
                ).execute(),
                max_retries=4,
                base_delay=2.0,
            )
            total_ok += len(chunk)
        except Exception as e:
            print(f"  ERROR upserting chunk of {len(chunk)} (gave up after retries): {e}")
            total_err += len(chunk)
        # Brief pause between chunks to avoid connection exhaustion
        if i + chunk_size < len(records):
            time.sleep(0.3)
    return total_ok, total_err


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


# ============================================================================
# Batch Data Fetcher (sequential with retry)
# ============================================================================

def fetch_batch_data(
    supabase_client,
    attrs_batch: List[Dict],
    existing_ids: set,
) -> Dict:
    """
    Fetch all data needed for a batch. Sequential DB calls with retry
    to avoid overwhelming Supabase's HTTP/2 connection pool.

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

    # Sequential DB fetches with retry (no parallel — avoids HTTP/2 exhaustion)
    product_info = retry_with_backoff(
        fetch_product_info, supabase_client, to_process_ids,
        max_retries=4, base_delay=2.0,
    )
    time.sleep(0.2)  # Brief pause between requests

    image_embeddings = retry_with_backoff(
        fetch_primary_image_embeddings, supabase_client, to_process_ids,
        max_retries=4, base_delay=2.0,
    )

    attr_by_id = {str(row["sku_id"]): row for row in attrs_batch}

    return {
        "to_process_ids": to_process_ids,
        "product_info": product_info,
        "image_embeddings": image_embeddings,
        "attr_by_id": attr_by_id,
        "skipped": len(product_ids) - len(to_process_ids),
    }


# ============================================================================
# Core Processing
# ============================================================================

def process_batch(
    supabase_client,
    encode_text_fn,
    attrs_batch: List[Dict],
    version: int,
    alpha: float,
    existing_ids: set,
    dry_run: bool = False,
) -> Tuple[int, int, int]:
    """
    Process a batch with sequential DB I/O and retry logic.

    1. Fetch product info + image embeddings (sequential, with retry)
    2. Build text strings (CPU, fast)
    3. Encode texts one at a time with FashionCLIP (skips on error)
    4. Combine embeddings (CPU, fast)
    5. Upsert in small sequential chunks with retry

    Returns (processed, skipped, errors).
    """
    # Step 1: Sequential DB fetch with retry
    try:
        data = fetch_batch_data(supabase_client, attrs_batch, existing_ids)
    except Exception as e:
        print(f"  ERROR fetching batch data (gave up after retries): {e}")
        return 0, 0, len(attrs_batch)

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

    # Step 3: Encode texts individually (skip failures instead of failing entire batch)
    records = []
    for pid, product, attrs, img_emb, text in items_to_encode:
        try:
            text_emb = encode_text_fn(text)
        except Exception as e:
            # Skip this product but continue with the rest
            errors += 1
            continue

        # Step 4: Combine embeddings
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

    # Step 5: Upsert with retry (sequential small chunks inside upsert_records)
    if records:
        ok, err = upsert_records(supabase_client, records)
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
        "--batch-delay", type=float, default=0.5,
        help="Seconds to wait between batches (avoids Supabase rate limits). Default: 0.5",
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
    print(f"  Batch delay: {args.batch_delay}s")
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
        batch_num = 0
        t_start = time.time()

        while True:
            # Check limit
            if args.limit and total_processed >= args.limit:
                print(f"\n  Reached limit of {args.limit} products")
                break

            # Fetch next batch of attributes
            try:
                attrs_batch = retry_with_backoff(
                    fetch_products_with_attributes, sb, args.batch_size, offset,
                    max_retries=4, base_delay=2.0,
                )
            except Exception as e:
                print(f"  ERROR fetching attrs batch at offset {offset}: {e}")
                break

            if not attrs_batch:
                break  # No more data

            batch_num += 1
            offset += args.batch_size

            # Process current batch (all sequential with retry)
            batch_t = time.time()
            processed, skipped, errors = process_batch(
                supabase_client=sb,
                encode_text_fn=encode_text_fn,
                attrs_batch=attrs_batch,
                version=version,
                alpha=args.alpha,
                existing_ids=existing_ids,
                dry_run=args.dry_run,
            )

            total_processed += processed
            total_skipped += skipped
            total_errors += errors
            batch_ms = int((time.time() - batch_t) * 1000)

            # Progress
            elapsed = time.time() - t_start
            rate = total_processed / elapsed if elapsed > 0 else 0
            eta = (total_attrs - offset) / rate / 60 if rate > 0 else 0

            print(
                f"  Batch {batch_num}: "
                f"processed={processed}, skipped={skipped}, errors={errors} "
                f"({batch_ms}ms) | "
                f"Total: {total_processed:,}/{total_attrs:,} "
                f"({rate:.1f}/s, ETA {eta:.1f}min)"
            )

            # Delay between batches to avoid overwhelming Supabase
            if args.batch_delay > 0:
                time.sleep(args.batch_delay)

        elapsed = time.time() - t_start
        print(f"\n  v{version} complete:")
        print(f"    Processed: {total_processed:,}")
        print(f"    Skipped:   {total_skipped:,}")
        print(f"    Errors:    {total_errors:,}")
        print(f"    Time:      {elapsed:.1f}s ({elapsed/60:.1f}min)")
        if total_processed > 0:
            print(f"    Rate:      {total_processed/elapsed:.1f} products/sec")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
