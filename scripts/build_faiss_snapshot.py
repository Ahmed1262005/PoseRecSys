#!/usr/bin/env python3
"""
Build FAISS snapshot from Supabase.

Downloads all multimodal embeddings + product metadata + Gemini attributes
from Supabase, builds a FAISS IndexFlatIP, and saves to data/faiss/.

Usage:
    PYTHONPATH=src python scripts/build_faiss_snapshot.py

Output:
    data/faiss/index.faiss       — FAISS index (~250MB)
    data/faiss/product_ids.npy   — row→product_id mapping
    data/faiss/metadata.pkl      — product metadata dict
    data/faiss/version.json      — build metadata
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SNAPSHOT_DIR = "data/faiss"
BATCH_SIZE = 500          # smaller batches for reliability over slow links
MAX_RETRIES = 3
RETRY_DELAY = 5           # seconds


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


def build_and_save(all_embeddings, all_product_ids, metadata):
    """Build FAISS index and save snapshot."""
    import faiss

    print("Step 4/4: Building FAISS index...")

    embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_matrix /= norms

    dim = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_matrix)

    p = Path(SNAPSHOT_DIR)
    p.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(p / "index.faiss"))
    np.save(str(p / "product_ids.npy"), np.array(all_product_ids, dtype=object))
    with open(p / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(p / "version.json", "w") as f:
        json.dump({
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "count": len(all_product_ids),
            "embedding_version": 1,
            "dimension": dim,
            "has_gemini_attributes": True,
        }, f, indent=2)

    print(f"  Index: {len(all_product_ids):,} vectors, {dim}-dim")


def main():
    print("Building FAISS snapshot from Supabase...")
    print(f"  Supabase URL: {os.environ.get('SUPABASE_URL', 'NOT SET')[:50]}...")
    print(f"  Batch size: {BATCH_SIZE}, retries: {MAX_RETRIES}")
    print()

    t0 = time.perf_counter()
    client = _create_client()

    all_embeddings, all_product_ids = fetch_embeddings(client)
    if not all_embeddings:
        print("No embeddings found. Aborting.")
        sys.exit(1)

    metadata = fetch_product_metadata(client)
    metadata = fetch_gemini_attributes(client, metadata)

    build_and_save(all_embeddings, all_product_ids, metadata)

    elapsed = time.perf_counter() - t0
    print(f"\nDone! Snapshot saved to {SNAPSHOT_DIR}/ in {elapsed:.1f}s")

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


if __name__ == "__main__":
    main()
