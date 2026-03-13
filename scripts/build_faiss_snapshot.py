#!/usr/bin/env python3
"""
Build FAISS snapshot from Supabase.

Downloads all multimodal embeddings + product metadata + Gemini attributes
from Supabase, builds a FAISS IndexFlatIP, and saves to data/faiss/.

The build is atomic: writes to a staging directory first, validates the
output, then swaps staging → live in a single rename.  The running server
detects the new version.json timestamp and hot-reloads automatically.

Usage:
    PYTHONPATH=src python scripts/build_faiss_snapshot.py

Output:
    data/faiss/index.faiss       — FAISS index (~250MB)
    data/faiss/product_ids.npy   — row→product_id mapping
    data/faiss/metadata.pkl      — product metadata dict
    data/faiss/version.json      — build metadata

Scheduling (cron, every 6 hours):
    0 */6 * * * cd /mnt/d/ecommerce/recommendationSystem && \\
        PYTHONPATH=src .venv/bin/python scripts/build_faiss_snapshot.py \\
        >> /var/log/faiss_rebuild.log 2>&1
"""

import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

LIVE_DIR = "data/faiss"
STAGING_DIR = "data/faiss_staging"
OLD_DIR = "data/faiss_old"

BATCH_SIZE = 500          # smaller batches for reliability over slow links
MAX_RETRIES = 3
RETRY_DELAY = 5           # seconds
MIN_VECTORS = 50_000      # safety: abort if fewer vectors than this


def _create_client():
    """Create a Supabase client with a generous timeout for bulk fetching."""
    from supabase import create_client, ClientOptions
    from config.settings import get_settings

    settings = get_settings()
    opts = ClientOptions(postgrest_client_timeout=120)
    return create_client(settings.supabase_url, settings.supabase_service_key, opts)


def _fetch_batch(table_query, offset, batch_size, label=""):
    """Fetch a batch with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = table_query.range(offset, offset + batch_size - 1).execute()
            return resp.data
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  Retry {attempt + 1}/{MAX_RETRIES} for {label} offset={offset}: {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise


def fetch_embeddings(client):
    """Fetch all multimodal embeddings."""
    print("Step 1/4: Fetching multimodal embeddings...")
    all_embeddings = []
    all_product_ids = []
    offset = 0

    while True:
        query = (
            client.table("product_multimodal_embeddings")
            .select("product_id, multimodal_embedding")
            .eq("version", 1)
        )
        rows = _fetch_batch(query, offset, BATCH_SIZE, "embeddings")
        if not rows:
            break

        for row in rows:
            pid = str(row["product_id"])
            emb_raw = row["multimodal_embedding"]
            if isinstance(emb_raw, str):
                emb = np.array(json.loads(emb_raw), dtype=np.float32)
            elif isinstance(emb_raw, list):
                emb = np.array(emb_raw, dtype=np.float32)
            else:
                continue
            all_embeddings.append(emb)
            all_product_ids.append(pid)

        offset += BATCH_SIZE
        if len(rows) < BATCH_SIZE:
            break
        if offset % 5000 == 0:
            print(f"  {offset:,} embeddings fetched...")

    print(f"  Total: {len(all_embeddings):,} embeddings")
    return all_embeddings, all_product_ids


def fetch_product_metadata(client):
    """Fetch product metadata (name, brand, price, etc.)."""
    print("Step 2/4: Fetching product metadata...")
    metadata = {}
    offset = 0

    while True:
        query = (
            client.table("products")
            .select(
                "id, name, brand, price, original_price, in_stock, "
                "primary_image_url, gallery_images, category, "
                "broad_category, article_type, colors, materials"
            )
            .eq("in_stock", True)
        )
        rows = _fetch_batch(query, offset, BATCH_SIZE, "products")
        if not rows:
            break

        for row in rows:
            pid = str(row["id"])
            metadata[pid] = {
                "name": row.get("name", ""),
                "brand": row.get("brand", ""),
                "price": float(row.get("price") or 0),
                "original_price": row.get("original_price"),
                "in_stock": row.get("in_stock", True),
                "image_url": row.get("primary_image_url"),
                "gallery_images": row.get("gallery_images") or [],
                "category": row.get("category"),
                "broad_category": row.get("broad_category"),
                "article_type": row.get("article_type"),
                "colors": row.get("colors") or [],
                "materials": row.get("materials") or [],
            }

        offset += BATCH_SIZE
        if len(rows) < BATCH_SIZE:
            break
        if offset % 10000 == 0:
            print(f"  {offset:,} products fetched...")

    print(f"  Total: {len(metadata):,} products")
    return metadata


def fetch_gemini_attributes(client, metadata):
    """Fetch Gemini-extracted attributes and merge into metadata."""
    print("Step 3/4: Fetching Gemini attributes (product_attributes)...")

    _ATTR_FIELDS = (
        "sku_id, category_l1, category_l2, category_l3, primary_color, "
        "color_family, formality, silhouette, rise, "
        "apparent_fabric, seasons, fit_type, pattern, pattern_scale, "
        "style_tags, occasions, is_fashion_item, "
        "arm_coverage, shoulder_coverage, neckline_depth, "
        "midriff_exposure, back_openness, sheerness_visual, "
        "body_cling_visual, structure_level, drape_level, "
        "cropped_degree, waist_definition_visual, leg_volume_visual, "
        "has_pockets_visible, slit_presence, slit_height, "
        "detail_tags, lining_status_likely, "
        "vibe_tags, coverage_level, styling_role, "
        "construction"
    )
    _CONSTRUCTION_KEYS = ("neckline", "sleeve_type", "length")

    attr_count = 0
    offset = 0

    while True:
        query = client.table("product_attributes").select(_ATTR_FIELDS)
        rows = _fetch_batch(query, offset, BATCH_SIZE, "attributes")
        if not rows:
            break

        for row in rows:
            pid = str(row.get("sku_id", ""))
            if pid not in metadata:
                continue
            meta = metadata[pid]
            for key, val in row.items():
                if key == "sku_id":
                    continue
                if key == "construction" and isinstance(val, dict):
                    for ck in _CONSTRUCTION_KEYS:
                        cv = val.get(ck)
                        if cv is not None:
                            meta[ck] = cv
                    continue
                if val is not None:
                    store_key = "has_pockets" if key == "has_pockets_visible" else key
                    meta[store_key] = val
            attr_count += 1

        offset += BATCH_SIZE
        if len(rows) < BATCH_SIZE:
            break
        if offset % 10000 == 0:
            print(f"  {offset:,} attribute records fetched...")

    print(f"  Merged attributes for {attr_count:,} products")
    return metadata


def build_to_staging(all_embeddings, all_product_ids, metadata):
    """Build FAISS index and write to the staging directory.

    Returns the version dict on success, or raises on failure.
    """
    import faiss

    print("Step 4/4: Building FAISS index...")

    embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_matrix /= norms

    dim = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_matrix)

    # Write to staging dir (not live)
    p = Path(STAGING_DIR)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)

    faiss.write_index(index, str(p / "index.faiss"))
    np.save(str(p / "product_ids.npy"), np.array(all_product_ids, dtype=object))
    with open(p / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    version = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "count": len(all_product_ids),
        "embedding_version": 1,
        "dimension": dim,
        "has_gemini_attributes": True,
    }
    with open(p / "version.json", "w") as f:
        json.dump(version, f, indent=2)

    print(f"  Staging: {len(all_product_ids):,} vectors, {dim}-dim → {STAGING_DIR}/")
    return version


def validate_staging(version):
    """Sanity-check the staging snapshot before promoting to live.

    Returns True if valid, False otherwise.
    """
    p = Path(STAGING_DIR)
    required = ["index.faiss", "product_ids.npy", "metadata.pkl", "version.json"]

    for fname in required:
        fpath = p / fname
        if not fpath.exists():
            print(f"  VALIDATION FAILED: missing {fpath}")
            return False
        if fpath.stat().st_size == 0:
            print(f"  VALIDATION FAILED: empty file {fpath}")
            return False

    count = version.get("count", 0)
    dim = version.get("dimension", 0)

    if count < MIN_VECTORS:
        print(f"  VALIDATION FAILED: only {count:,} vectors (minimum: {MIN_VECTORS:,})")
        return False

    if dim != 512:
        print(f"  VALIDATION FAILED: dimension={dim} (expected 512)")
        return False

    print(f"  Validation passed: {count:,} vectors, {dim}-dim")
    return True


def promote_staging():
    """Atomically swap staging → live.

    1. Rename live → old  (if live exists)
    2. Rename staging → live
    3. Delete old
    """
    live = Path(LIVE_DIR)
    staging = Path(STAGING_DIR)
    old = Path(OLD_DIR)

    # Clean up any leftover old dir from a previous run
    if old.exists():
        shutil.rmtree(old)

    # Move current live aside
    if live.exists():
        live.rename(old)

    # Promote staging to live
    staging.rename(live)

    # Clean up old
    if old.exists():
        shutil.rmtree(old)

    print(f"  Promoted: {STAGING_DIR}/ → {LIVE_DIR}/")


def main():
    print("=" * 60)
    print("FAISS Snapshot Build")
    print(f"  Time:       {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"  Supabase:   {os.environ.get('SUPABASE_URL', 'NOT SET')[:50]}...")
    print(f"  Batch size: {BATCH_SIZE}, retries: {MAX_RETRIES}")
    print(f"  Staging:    {STAGING_DIR}/")
    print(f"  Live:       {LIVE_DIR}/")
    print("=" * 60)
    print()

    t0 = time.perf_counter()
    client = _create_client()

    # ---- Fetch data from Supabase ----
    all_embeddings, all_product_ids = fetch_embeddings(client)
    if not all_embeddings:
        print("ERROR: No embeddings found. Aborting.")
        sys.exit(1)

    metadata = fetch_product_metadata(client)
    metadata = fetch_gemini_attributes(client, metadata)

    # ---- Build to staging ----
    version = build_to_staging(all_embeddings, all_product_ids, metadata)

    # ---- Validate ----
    if not validate_staging(version):
        print("\nABORTED: Validation failed. Live index untouched.")
        # Leave staging dir for debugging
        sys.exit(1)

    # ---- Atomic swap: staging → live ----
    promote_staging()

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Vectors: {version['count']:,}")
    print(f"  Built at: {version['built_at']}")

    # Quick verification
    sample_pid = all_product_ids[0]
    meta = metadata.get(sample_pid, {})
    has_attrs = "category_l1" in meta
    print(f"  Gemini attributes baked in: {has_attrs}")
    if has_attrs:
        print(f"  Sample: category_l1={meta.get('category_l1')}, "
              f"neckline={meta.get('neckline')}, "
              f"sleeve_type={meta.get('sleeve_type')}, "
              f"formality={meta.get('formality')}")

    sys.exit(0)


if __name__ == "__main__":
    main()
