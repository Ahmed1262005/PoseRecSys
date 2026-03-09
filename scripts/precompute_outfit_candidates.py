#!/usr/bin/env python
"""Precompute outfit candidate pools for all products using faiss.

Pipeline:
  1. Export embeddings from Supabase image_embeddings (batched, parallel)
  2. Export category mapping from product_attributes
  3. Deduplicate to 1 embedding per product
  4. Build faiss IndexFlatIP per target category
  5. For each source product, query complementary category indexes (top K)
  6. Upload results to outfit_candidates table (batched, parallel)

Usage:
    PYTHONPATH=src .venv/bin/python scripts/precompute_outfit_candidates.py

Environment:
    WORKERS=8              Number of parallel upload workers (default 8)
    CANDIDATES_PER_POOL=60 Candidates per source×category pool (default 60)
    BATCH_SIZE=5000        Upload batch size (default 5000)
    EXPORT_BATCH=1000      Embedding export batch size (default 1000)
    SKIP_EXPORT=1          Skip export if local .npy files exist (default 0)
    DRY_RUN=1              Compute but don't upload (default 0)
"""

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("precompute")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WORKERS = int(os.environ.get("WORKERS", 8))
CANDIDATES_PER_POOL = int(os.environ.get("CANDIDATES_PER_POOL", 60))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 5000))
EXPORT_BATCH = int(os.environ.get("EXPORT_BATCH", 1000))
SKIP_EXPORT = os.environ.get("SKIP_EXPORT", "0") == "1"
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

CACHE_DIR = ROOT / "data" / "precompute_cache"
EMB_FILE = CACHE_DIR / "embeddings.npy"
IDS_FILE = CACHE_DIR / "product_ids.npy"
CATS_FILE = CACHE_DIR / "categories.npy"

# Source category → target categories
COMPLEMENTARY = {
    "tops":      ["bottoms", "outerwear"],
    "bottoms":   ["tops", "outerwear"],
    "dresses":   ["outerwear"],
    "outerwear": ["tops", "bottoms"],
}

# Gemini L1 → broad category (mirrors gemini_l1_to_broad SQL function)
_L1_TO_BROAD = {
    "tops": "tops", "top": "tops",
    "bottoms": "bottoms", "bottom": "bottoms",
    "dresses": "dresses", "dress": "dresses",
    "outerwear": "outerwear", "jackets & coats": "outerwear",
    "coats & jackets": "outerwear",
    "activewear": "tops", "swimwear": "tops",
    "jumpsuits & rompers": "dresses", "jumpsuit": "dresses",
}


def _broad(l1: Optional[str]) -> Optional[str]:
    if not l1:
        return None
    return _L1_TO_BROAD.get(l1.lower().strip(), l1.lower().strip())


# ---------------------------------------------------------------------------
# Phase 1: Export embeddings
# ---------------------------------------------------------------------------

def _get_supabase():
    from config.database import get_supabase_client
    return get_supabase_client()


def export_embeddings():
    """Pull all embeddings from Supabase, deduplicate per product, save to .npy."""
    if SKIP_EXPORT and EMB_FILE.exists() and IDS_FILE.exists() and CATS_FILE.exists():
        log.info("SKIP_EXPORT=1 and cache files exist, loading from disk")
        embeddings = np.load(str(EMB_FILE))
        product_ids = np.load(str(IDS_FILE), allow_pickle=True)
        categories = np.load(str(CATS_FILE), allow_pickle=True)
        log.info("Loaded %d embeddings, %d IDs, %d categories",
                 len(embeddings), len(product_ids), len(categories))
        return embeddings, product_ids, categories

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sb = _get_supabase()

    # Step 1: Get total count
    r = sb.table("image_embeddings").select("id", count="exact").limit(1).execute()
    total = r.count
    log.info("Total image_embeddings rows: %d", total)

    # Step 2: Export in batches using parallel workers
    all_ids: List[str] = []
    all_embeddings: List[np.ndarray] = []
    seen_products: Set[str] = set()

    def _fetch_batch(offset: int) -> List[dict]:
        """Fetch one batch of embeddings."""
        _sb = _get_supabase()
        r = _sb.table("image_embeddings").select(
            "sku_id, embedding"
        ).range(offset, offset + EXPORT_BATCH - 1).execute()
        return r.data or []

    offsets = list(range(0, total, EXPORT_BATCH))
    log.info("Exporting %d batches of %d with %d workers...",
             len(offsets), EXPORT_BATCH, WORKERS)

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_fetch_batch, off): off for off in offsets}
        for future in as_completed(futures):
            rows = future.result()
            for row in rows:
                pid = str(row["sku_id"])
                if pid in seen_products:
                    continue  # deduplicate: keep first embedding per product
                seen_products.add(pid)
                emb_str = row["embedding"]
                # Parse "[0.01,0.02,...]" string to float array
                emb = np.fromstring(emb_str.strip("[]"), sep=",", dtype=np.float32)
                if len(emb) == 512:
                    all_ids.append(pid)
                    all_embeddings.append(emb)
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(offsets) - done) / rate if rate > 0 else 0
                log.info("  exported %d/%d batches (%.0f/s) | %d unique products | ETA %.0fs",
                         done, len(offsets), rate, len(all_ids), eta)

    elapsed = time.time() - t0
    log.info("Export done: %d unique products with embeddings in %.0fs", len(all_ids), elapsed)

    # Step 3: Get category mapping
    log.info("Fetching category mapping from product_attributes...")
    cat_map: Dict[str, str] = {}

    def _fetch_cats(offset: int) -> List[dict]:
        _sb = _get_supabase()
        r = _sb.table("product_attributes").select(
            "sku_id, category_l1"
        ).range(offset, offset + 5000 - 1).execute()
        return r.data or []

    # Get total
    r = sb.table("product_attributes").select("sku_id", count="exact").limit(1).execute()
    cat_total = r.count
    cat_offsets = list(range(0, cat_total, 5000))

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_fetch_cats, off): off for off in cat_offsets}
        for future in as_completed(futures):
            rows = future.result()
            for row in rows:
                cat_map[str(row["sku_id"])] = _broad(row.get("category_l1")) or ""

    log.info("Category mapping: %d products", len(cat_map))

    # Step 4: Build aligned arrays
    embeddings = np.array(all_embeddings, dtype=np.float32)
    product_ids = np.array(all_ids, dtype=object)
    categories = np.array([cat_map.get(pid, "") for pid in all_ids], dtype=object)

    # L2-normalize for inner product = cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    # Save cache
    np.save(str(EMB_FILE), embeddings)
    np.save(str(IDS_FILE), product_ids)
    np.save(str(CATS_FILE), categories)
    log.info("Saved to %s (%dMB)", CACHE_DIR, embeddings.nbytes // 1024 // 1024)

    return embeddings, product_ids, categories


# ---------------------------------------------------------------------------
# Phase 2: Build faiss indexes per category
# ---------------------------------------------------------------------------

def build_category_indexes(
    embeddings: np.ndarray,
    product_ids: np.ndarray,
    categories: np.ndarray,
) -> Dict[str, Tuple[faiss.IndexFlatIP, np.ndarray]]:
    """Build one faiss IndexFlatIP per target category.

    Returns dict: category -> (faiss_index, product_ids_in_index).
    """
    indexes: Dict[str, Tuple[faiss.IndexFlatIP, np.ndarray]] = {}

    for cat in ["tops", "bottoms", "outerwear", "dresses"]:
        mask = categories == cat
        cat_embeddings = embeddings[mask]
        cat_ids = product_ids[mask]

        if len(cat_embeddings) == 0:
            log.warning("No embeddings for category '%s', skipping", cat)
            continue

        # Flat inner product index (cosine on normalized vectors)
        index = faiss.IndexFlatIP(512)
        index.add(cat_embeddings)
        indexes[cat] = (index, cat_ids)
        log.info("  %s: %d vectors indexed", cat, len(cat_ids))

    return indexes


# ---------------------------------------------------------------------------
# Phase 3: Compute neighbor pools
# ---------------------------------------------------------------------------

def compute_all_pools(
    embeddings: np.ndarray,
    product_ids: np.ndarray,
    categories: np.ndarray,
    indexes: Dict[str, Tuple[faiss.IndexFlatIP, np.ndarray]],
) -> List[Tuple[str, str, str, float, int]]:
    """For each source product, find top-K neighbors in each complementary category.

    Returns list of (source_id, target_category, candidate_id, similarity, rank).
    """
    results: List[Tuple[str, str, str, float, int]] = []
    K = CANDIDATES_PER_POOL

    # Group source products by their category
    cat_to_source_mask: Dict[str, np.ndarray] = {}
    for src_cat in COMPLEMENTARY:
        mask = categories == src_cat
        cat_to_source_mask[src_cat] = mask

    total_pools = 0
    t0 = time.time()

    for src_cat, target_cats in COMPLEMENTARY.items():
        src_mask = cat_to_source_mask.get(src_cat)
        if src_mask is None or not src_mask.any():
            continue

        src_embs = embeddings[src_mask]
        src_ids = product_ids[src_mask]
        n_sources = len(src_ids)

        for tgt_cat in target_cats:
            if tgt_cat not in indexes:
                continue

            tgt_index, tgt_ids = indexes[tgt_cat]
            n_targets = len(tgt_ids)

            log.info("Computing %s → %s: %d sources × %d targets (top %d)...",
                     src_cat, tgt_cat, n_sources, n_targets, K)

            # faiss batch search — this is the fast part
            actual_k = min(K + 5, n_targets)  # extra buffer for self-dedup
            t1 = time.time()
            similarities, indices = tgt_index.search(src_embs, actual_k)
            search_time = time.time() - t1

            log.info("  faiss search: %.2fs (%.0f queries/s)",
                     search_time, n_sources / search_time if search_time > 0 else 0)

            # Convert to result tuples
            for i in range(n_sources):
                src_id = src_ids[i]
                rank = 0
                for j in range(actual_k):
                    if rank >= K:
                        break
                    idx = indices[i, j]
                    if idx < 0:
                        continue
                    cand_id = tgt_ids[idx]
                    if cand_id == src_id:
                        continue  # skip self
                    sim = float(similarities[i, j])
                    rank += 1
                    results.append((src_id, tgt_cat, cand_id, sim, rank))

                total_pools += 1

            elapsed = time.time() - t0
            log.info("  %d pools computed so far, %d total rows, %.0fs elapsed",
                     total_pools, len(results), elapsed)

    log.info("All pools computed: %d pools, %d rows in %.0fs",
             total_pools, len(results), time.time() - t0)
    return results


# ---------------------------------------------------------------------------
# Phase 4: Upload to Supabase
# ---------------------------------------------------------------------------

def upload_results(results: List[Tuple[str, str, str, float, int]]):
    """Upload results to outfit_candidates table in parallel batches."""
    if DRY_RUN:
        log.info("DRY_RUN=1, skipping upload. Would insert %d rows.", len(results))
        return

    sb = _get_supabase()

    # Truncate existing data
    log.info("Clearing existing outfit_candidates...")
    try:
        sb.table("outfit_candidates").delete().neq("source_id", "00000000-0000-0000-0000-000000000000").execute()
        log.info("  cleared.")
    except Exception as e:
        log.warning("  clear failed (table may not exist yet): %s", e)

    # Prepare batches
    total = len(results)
    batches = []
    for i in range(0, total, BATCH_SIZE):
        batch = []
        for src_id, tgt_cat, cand_id, sim, rank in results[i:i + BATCH_SIZE]:
            batch.append({
                "source_id": src_id,
                "target_category": tgt_cat,
                "candidate_id": cand_id,
                "cosine_similarity": round(sim, 6),
                "rank": rank,
            })
        batches.append(batch)

    log.info("Uploading %d rows in %d batches of %d with %d workers...",
             total, len(batches), BATCH_SIZE, WORKERS)

    uploaded = 0
    errors = 0
    t0 = time.time()

    def _upload_batch(batch: List[dict]) -> int:
        _sb = _get_supabase()
        try:
            _sb.table("outfit_candidates").upsert(batch).execute()
            return len(batch)
        except Exception as e:
            log.error("Batch upload failed: %s", str(e)[:200])
            return -1

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_upload_batch, b): i for i, b in enumerate(batches)}
        for future in as_completed(futures):
            n = future.result()
            if n > 0:
                uploaded += n
            else:
                errors += 1
            done = uploaded + errors * BATCH_SIZE
            if (uploaded // BATCH_SIZE) % 20 == 0 and uploaded > 0:
                elapsed = time.time() - t0
                rate = uploaded / elapsed
                eta = (total - uploaded) / rate if rate > 0 else 0
                log.info("  uploaded %d/%d rows (%.0f/s) | %d errors | ETA %.0fs",
                         uploaded, total, rate, errors, eta)

    elapsed = time.time() - t0
    log.info("Upload done: %d rows in %.0fs (%d errors)", uploaded, elapsed, errors)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("Precompute Outfit Candidates")
    log.info("  WORKERS=%d  K=%d  BATCH=%d  DRY_RUN=%s",
             WORKERS, CANDIDATES_PER_POOL, BATCH_SIZE, DRY_RUN)
    log.info("=" * 60)

    t_total = time.time()

    # Phase 1: Export
    log.info("\n--- Phase 1: Export embeddings ---")
    embeddings, product_ids, categories = export_embeddings()
    log.info("  %d products, %d dimensions, categories: %s",
             len(product_ids), embeddings.shape[1],
             {c: int((categories == c).sum()) for c in ["tops", "bottoms", "outerwear", "dresses"]})

    # Phase 2: Build indexes
    log.info("\n--- Phase 2: Build faiss indexes ---")
    indexes = build_category_indexes(embeddings, product_ids, categories)

    # Phase 3: Compute pools
    log.info("\n--- Phase 3: Compute neighbor pools ---")
    results = compute_all_pools(embeddings, product_ids, categories, indexes)

    # Phase 4: Upload
    log.info("\n--- Phase 4: Upload to Supabase ---")
    upload_results(results)

    elapsed = time.time() - t_total
    log.info("\n" + "=" * 60)
    log.info("DONE in %.0fs (%.1f min)", elapsed, elapsed / 60)
    log.info("  %d products, %d pools, %d candidate rows",
             len(product_ids), len(set((r[0], r[1]) for r in results)), len(results))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
