#!/usr/bin/env python
"""Precompute outfit candidate pools with full TATTOO scoring pipeline.

Pipeline:
  Phase 1a: Export embeddings from Supabase image_embeddings (batched, parallel)
  Phase 1b: Export full product catalog (products + product_attributes)
  Phase 2:  Build faiss IndexFlatIP per target category (L2 exclusions applied)
  Phase 3:  For each source × target category:
            - faiss retrieve top-K_RETRIEVE neighbors (over-fetch buffer)
            - Build AestheticProfiles for source + all candidates
            - Full filtering: category gate, dedup, set removal, hard avoids
            - Full TATTOO scoring: compat + cosine + style_adj + avoid_adj + styling_adj
            - Density caps (cardigan cap in outerwear)
            - Keep top CANDIDATES_PER_POOL by TATTOO score
  Phase 4:  Upload to outfit_candidates table (batched, parallel)

This mirrors the serve-time scoring in outfit_engine._score_category() but
runs offline so serve-time is a simple indexed lookup (<100ms).
User-specific adjustments (profile_adj) are applied at serve-time only.

Usage (server):
    PYTHONPATH=src python scripts/precompute_outfit_candidates.py

Usage (local):
    PYTHONPATH=src .venv/bin/python scripts/precompute_outfit_candidates.py

Environment:
    WORKERS=8              Parallel upload/export workers (default 8)
    CANDIDATES_PER_POOL=60 Final candidates per source×category pool (default 60)
    K_RETRIEVE=150         Over-retrieve from faiss before filtering (default 150)
    BATCH_SIZE=5000        Upload batch size (default 5000)
    EXPORT_BATCH=1000      Embedding export batch size (default 1000)
    CATALOG_BATCH=1000     Product/attrs export batch size (must be ≤1000, Supabase row limit)
    SKIP_EMBEDDINGS=1      Skip embedding export if cache exists (default 0)
    SKIP_CATALOG=1         Skip catalog export if cache exists (default 0)
    DRY_RUN=1              Compute but don't upload (default 0)
"""

import dataclasses
import json
import logging
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# Imports from the scoring pipeline
# ---------------------------------------------------------------------------
from services.outfit_engine import (
    AestheticProfile,
    compute_compatibility_score,
    compute_novelty_score,
    get_complementary_targets,
    _filter_by_gemini_category,
    _deduplicate,
    _remove_sets_and_non_outfit,
    _gemini_broad,
)
from services.outfit_avoids import (
    compute_avoid_penalties,
    filter_hard_avoids,
)
from services.styling_scorer import compute_styling_adjustment

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
K_RETRIEVE = int(os.environ.get("K_RETRIEVE", 150))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 5000))
EXPORT_BATCH = int(os.environ.get("EXPORT_BATCH", 1000))
CATALOG_BATCH = int(os.environ.get("CATALOG_BATCH", 1000))
SKIP_EMBEDDINGS = os.environ.get("SKIP_EMBEDDINGS", "0") == "1"
SKIP_CATALOG = os.environ.get("SKIP_CATALOG", "0") == "1"
# Legacy: SKIP_EXPORT=1 sets both SKIP_EMBEDDINGS and SKIP_CATALOG
if os.environ.get("SKIP_EXPORT", "0") == "1":
    SKIP_EMBEDDINGS = True
    SKIP_CATALOG = True
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

CACHE_DIR = ROOT / "data" / "precompute_cache"
EMB_FILE = CACHE_DIR / "embeddings.npy"
IDS_FILE = CACHE_DIR / "product_ids.npy"
CATS_FILE = CACHE_DIR / "categories.npy"
L2_FILE = CACHE_DIR / "category_l2.npy"
PRODUCTS_FILE = CACHE_DIR / "products.pkl"
ATTRS_FILE = CACHE_DIR / "attrs.pkl"

# Source category → target categories (mirrors COMPLEMENTARY_CATEGORIES in outfit_engine)
COMPLEMENTARY = {
    "tops":      ["bottoms", "outerwear"],
    "bottoms":   ["tops", "outerwear"],
    "dresses":   ["outerwear"],
    "outerwear": ["tops", "bottoms"],
}

# Gemini L1 → broad category (mirrors _GEMINI_L1_TO_BROAD in outfit_engine)
_L1_TO_BROAD = {
    "tops": "tops", "top": "tops",
    "bottoms": "bottoms", "bottom": "bottoms",
    "dresses": "dresses", "dress": "dresses",
    "outerwear": "outerwear", "jackets & coats": "outerwear",
    "coats & jackets": "outerwear",
    "activewear": "tops", "swimwear": "tops",
    "jumpsuits & rompers": "dresses", "jumpsuit": "dresses",
}

# L2 types excluded from the outerwear faiss index AND as outerwear sources.
# These are knit/soft layers that dominate via high cosine similarity but
# function as tops/mid-layers, not structured outerwear.
# Mirrors _OUTERWEAR_WAISTCOAT_L2 + cardigans in outfit_engine.py.
_OUTERWEAR_EXCLUDE_L2 = {"cardigan", "vest", "hoodie"}

# --- TATTOO scoring constants (must stay in sync with outfit_engine._score_category) ---
W_COMPAT = 0.70
W_COSINE = 0.30

# Outerwear styling heuristics (v3.2)
_STRUCTURED_OUTERWEAR = frozenset({
    "blazer", "jacket", "coat", "trench", "trench coat",
    "leather jacket", "denim jacket", "bomber", "parka",
})
_STRUCTURED_BONUS = 0.03
_KNIT_CARDIGAN_PENALTY = -0.06
_IMPLICIT_KNIT_L2 = frozenset({
    "t-shirt", "tank top", "camisole", "henley", "long sleeve t-shirt",
    "crop top", "bodysuit", "jersey top", "knit top", "polo",
    "turtleneck", "mock neck top",
})

# Cardigan density cap in outerwear results
_MAX_CARDIGANS_OUTERWEAR = 1

# Columns fetched from products table (mirrors _PRODUCT_SELECT in outfit_engine)
_PRODUCT_COLUMNS = (
    "id, name, brand, category, broad_category, price, "
    "primary_image_url, base_color, colors, materials, style_tags, fit"
)

# Columns fetched from product_attributes (mirrors _ATTRS_SELECT in outfit_engine)
_ATTRS_COLUMNS = (
    "sku_id, category_l1, category_l2, "
    "occasions, style_tags, pattern, formality, fit_type, "
    "color_family, primary_color, secondary_colors, seasons, silhouette, "
    "construction, apparent_fabric, texture, coverage_level, "
    "sheen, rise, leg_shape, stretch, "
    "styling_metadata, styling_role, appearance_top_tags, vibe_tags, extractor_version"
)


def _broad(l1: Optional[str]) -> Optional[str]:
    if not l1:
        return None
    return _L1_TO_BROAD.get(l1.lower().strip(), l1.lower().strip())


# ===========================================================================
# Supabase helpers
# ===========================================================================

def _get_supabase():
    """Return the singleton Supabase client (for single-threaded use only)."""
    from config.database import get_supabase_client
    return get_supabase_client()


def _new_supabase():
    """Create a brand-new Supabase client (thread-safe, no shared connection pool).

    Each call returns an independent client with its own HTTP transport.
    Use this inside thread workers to avoid the HTTP/2 hpack race condition
    that occurs when multiple threads share a singleton client.
    """
    from supabase import create_client
    from config.settings import get_settings
    s = get_settings()
    return create_client(s.supabase_url, s.supabase_service_key)


# ===========================================================================
# Phase 1a: Export embeddings
# ===========================================================================

def export_embeddings():
    """Pull all embeddings from Supabase, deduplicate per product, save to .npy.

    Returns (embeddings, product_ids, categories, category_l2) numpy arrays.
    """
    if SKIP_EMBEDDINGS and EMB_FILE.exists() and IDS_FILE.exists() and CATS_FILE.exists():
        log.info("SKIP_EMBEDDINGS=1 and cache files exist, loading from disk")
        embeddings = np.load(str(EMB_FILE))
        product_ids = np.load(str(IDS_FILE), allow_pickle=True)
        categories = np.load(str(CATS_FILE), allow_pickle=True)
        cat_l2 = np.load(str(L2_FILE), allow_pickle=True) if L2_FILE.exists() else np.array([""] * len(product_ids), dtype=object)
        log.info("Loaded %d embeddings, %d IDs, %d categories, %d L2",
                 len(embeddings), len(product_ids), len(categories), len(cat_l2))
        return embeddings, product_ids, categories, cat_l2

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
        _sb = _new_supabase()
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
                    continue
                seen_products.add(pid)
                emb_str = row["embedding"]
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

    # Step 3: Get category mapping (L1 + L2)
    log.info("Fetching category mapping from product_attributes...")
    cat_map: Dict[str, str] = {}
    l2_map: Dict[str, str] = {}

    def _fetch_cats(offset: int) -> List[dict]:
        _sb = _new_supabase()
        r = _sb.table("product_attributes").select(
            "sku_id, category_l1, category_l2"
        ).range(offset, offset + CATALOG_BATCH - 1).execute()
        return r.data or []

    r = sb.table("product_attributes").select("sku_id", count="exact").limit(1).execute()
    cat_total = r.count
    cat_offsets = list(range(0, cat_total, CATALOG_BATCH))

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_fetch_cats, off): off for off in cat_offsets}
        for future in as_completed(futures):
            rows = future.result()
            for row in rows:
                pid = str(row["sku_id"])
                cat_map[pid] = _broad(row.get("category_l1")) or ""
                l2_map[pid] = (row.get("category_l2") or "").lower().strip()

    log.info("Category mapping from product_attributes: %d products (L1+L2)", len(cat_map))

    # Step 3b: Fallback categories from products.category for products
    # that have embeddings but no product_attributes row.
    missing_cats = [pid for pid in all_ids if pid not in cat_map or not cat_map[pid]]
    if missing_cats:
        log.info("Fetching fallback categories from products.category for %d products...", len(missing_cats))

        # Broad-category mapping for products.category values
        _PRODUCTS_CAT_TO_BROAD = {
            "tops": "tops", "bottoms": "bottoms", "dresses": "dresses",
            "outerwear": "outerwear", "activewear": "tops", "swimwear": "tops",
        }

        # Batch fetch products.category (must stay ≤1000 per request)
        fallback_count = 0
        for i in range(0, len(missing_cats), 500):
            batch_ids = missing_cats[i:i + 500]
            try:
                _sb = _new_supabase()
                r = _sb.table("products").select(
                    "id, category"
                ).in_("id", batch_ids).execute()
                for row in (r.data or []):
                    pid = str(row["id"])
                    raw_cat = (row.get("category") or "").lower().strip()
                    broad = _PRODUCTS_CAT_TO_BROAD.get(raw_cat)
                    if broad and (pid not in cat_map or not cat_map[pid]):
                        cat_map[pid] = broad
                        fallback_count += 1
            except Exception as e:
                log.warning("Fallback category batch failed at offset %d: %s", i, e)

        log.info("  Fallback categories applied: %d products now have categories", fallback_count)

    log.info("Total category mapping: %d products", len([v for v in cat_map.values() if v]))

    # Step 4: Build aligned arrays
    embeddings = np.array(all_embeddings, dtype=np.float32)
    product_ids = np.array(all_ids, dtype=object)
    categories = np.array([cat_map.get(pid, "") for pid in all_ids], dtype=object)
    category_l2 = np.array([l2_map.get(pid, "") for pid in all_ids], dtype=object)

    # L2-normalize for inner product = cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    # Save cache
    np.save(str(EMB_FILE), embeddings)
    np.save(str(IDS_FILE), product_ids)
    np.save(str(CATS_FILE), categories)
    np.save(str(L2_FILE), category_l2)
    log.info("Saved to %s (%dMB)", CACHE_DIR, embeddings.nbytes // 1024 // 1024)

    return embeddings, product_ids, categories, category_l2


# ===========================================================================
# Phase 1b: Export full product catalog
# ===========================================================================

def export_product_catalog() -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Export products + product_attributes for profile building.

    Returns:
        product_data: Dict[product_id → products row dict]
        attrs_data:   Dict[product_id → product_attributes row dict]
    """
    if SKIP_CATALOG and PRODUCTS_FILE.exists() and ATTRS_FILE.exists():
        log.info("SKIP_CATALOG=1 and catalog cache exists, loading from disk")
        with open(PRODUCTS_FILE, "rb") as f:
            product_data = pickle.load(f)
        with open(ATTRS_FILE, "rb") as f:
            attrs_data = pickle.load(f)
        log.info("Loaded %d products, %d attrs from cache", len(product_data), len(attrs_data))
        return product_data, attrs_data

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sb = _get_supabase()

    # --- Export products table ---
    log.info("Exporting products table...")
    product_data: Dict[str, dict] = {}

    r = sb.table("products").select("id", count="exact").limit(1).execute()
    prod_total = r.count
    prod_offsets = list(range(0, prod_total, CATALOG_BATCH))
    log.info("  %d products in %d batches", prod_total, len(prod_offsets))

    def _fetch_products(offset: int) -> List[dict]:
        _sb = _new_supabase()
        r = _sb.table("products").select(
            _PRODUCT_COLUMNS
        ).range(offset, offset + CATALOG_BATCH - 1).execute()
        return r.data or []

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_fetch_products, off): off for off in prod_offsets}
        for future in as_completed(futures):
            rows = future.result()
            for row in rows:
                pid = str(row.get("id", ""))
                if pid:
                    product_data[pid] = row
            done += 1
            if done % 10 == 0:
                log.info("  products: %d/%d batches, %d rows", done, len(prod_offsets), len(product_data))

    log.info("  Products exported: %d rows in %.0fs", len(product_data), time.time() - t0)

    # --- Export product_attributes table ---
    log.info("Exporting product_attributes table...")
    attrs_data: Dict[str, dict] = {}

    r = sb.table("product_attributes").select("sku_id", count="exact").limit(1).execute()
    attrs_total = r.count
    attrs_offsets = list(range(0, attrs_total, CATALOG_BATCH))
    log.info("  %d attrs in %d batches", attrs_total, len(attrs_offsets))

    def _fetch_attrs(offset: int) -> List[dict]:
        _sb = _new_supabase()
        r = _sb.table("product_attributes").select(
            _ATTRS_COLUMNS
        ).range(offset, offset + CATALOG_BATCH - 1).execute()
        return r.data or []

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_fetch_attrs, off): off for off in attrs_offsets}
        for future in as_completed(futures):
            rows = future.result()
            for row in rows:
                pid = str(row.get("sku_id", ""))
                if pid:
                    attrs_data[pid] = row
            done += 1
            if done % 10 == 0:
                log.info("  attrs: %d/%d batches, %d rows", done, len(attrs_offsets), len(attrs_data))

    log.info("  Attrs exported: %d rows in %.0fs", len(attrs_data), time.time() - t0)

    # Save cache
    with open(PRODUCTS_FILE, "wb") as f:
        pickle.dump(product_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ATTRS_FILE, "wb") as f:
        pickle.dump(attrs_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("  Catalog cached to %s", CACHE_DIR)

    return product_data, attrs_data


# ===========================================================================
# Phase 2: Build faiss indexes per category
# ===========================================================================

def build_category_indexes(
    embeddings: np.ndarray,
    product_ids: np.ndarray,
    categories: np.ndarray,
    category_l2: np.ndarray,
) -> Dict[str, Tuple[faiss.IndexFlatIP, np.ndarray]]:
    """Build one faiss IndexFlatIP per target category.

    For outerwear, excludes soft-layer L2 types (cardigan, vest, hoodie)
    that dominate via high cosine similarity but function as mid-layers.

    Returns dict: category → (faiss_index, product_ids_in_index).
    """
    indexes: Dict[str, Tuple[faiss.IndexFlatIP, np.ndarray]] = {}

    for cat in ["tops", "bottoms", "outerwear", "dresses"]:
        mask = categories == cat

        # Outerwear: exclude soft-layer L2 types from the index
        if cat == "outerwear":
            l2_vals = category_l2[mask]
            keep = np.array([v not in _OUTERWEAR_EXCLUDE_L2 for v in l2_vals])
            base_indices = np.where(mask)[0]
            filtered_indices = base_indices[keep]
            cat_embeddings = embeddings[filtered_indices]
            cat_ids = product_ids[filtered_indices]
            excluded = int((~keep).sum())
            log.info("  %s: excluded %d soft-layer products (cardigan/vest/hoodie)",
                     cat, excluded)
        else:
            cat_embeddings = embeddings[mask]
            cat_ids = product_ids[mask]

        if len(cat_embeddings) == 0:
            log.warning("No embeddings for category '%s', skipping", cat)
            continue

        index = faiss.IndexFlatIP(512)
        index.add(cat_embeddings)
        indexes[cat] = (index, cat_ids)
        log.info("  %s: %d vectors indexed", cat, len(cat_ids))

    return indexes


# ===========================================================================
# Phase 3a: Build AestheticProfile cache
# ===========================================================================

def build_profiles_cache(
    product_data: Dict[str, dict],
    attrs_data: Dict[str, dict],
    product_ids: np.ndarray,
) -> Dict[str, AestheticProfile]:
    """Pre-build AestheticProfile for every product that has product data.

    Products with product_attributes get full profiles (all Gemini fields).
    Products without attrs get "lite" profiles (basic fields only — category,
    brand, price, color). TATTOO scoring degrades gracefully: cosine + basic
    compat dimensions, no styling scorer, no hard avoids.

    Returns Dict[product_id → AestheticProfile].
    """
    cache: Dict[str, AestheticProfile] = {}
    missing_product = 0
    full_profiles = 0
    lite_profiles = 0
    build_errors = 0
    total = len(product_ids)

    for pid in product_ids:
        pid_str = str(pid)
        prod = product_data.get(pid_str)
        if not prod:
            missing_product += 1
            continue
        attrs = attrs_data.get(pid_str, {})
        try:
            profile = AestheticProfile.from_product_and_attrs(prod, attrs)
            cache[pid_str] = profile
            if attrs:
                full_profiles += 1
            else:
                lite_profiles += 1
        except Exception as e:
            build_errors += 1
            if build_errors <= 5:
                log.warning("Failed to build profile for %s: %s", pid_str[:12], e)

    log.info("Profile cache: %d/%d profiles built (%.0f%%)",
             len(cache), total, 100 * len(cache) / total if total else 0)
    log.info("  %d full (with attrs), %d lite (no attrs), %d missing from products, %d errors",
             full_profiles, lite_profiles, missing_product, build_errors)
    log.info("  product_data has %d entries, attrs_data has %d entries",
             len(product_data), len(attrs_data))

    if len(cache) < total * 0.5:
        log.warning("  LOW COVERAGE: only %.0f%% of embedding products have profiles. "
                     "Check CATALOG_BATCH and run without SKIP_CATALOG.",
                     100 * len(cache) / total if total else 0)

    return cache


# ===========================================================================
# Phase 3b: Score all pools with full TATTOO pipeline
# ===========================================================================

def _score_one_pool(
    source: AestheticProfile,
    candidates: List[AestheticProfile],
    target_broad: str,
) -> List[Dict[str, Any]]:
    """Score candidates for one source × target_category using full TATTOO pipeline.

    Mirrors outfit_engine._score_category() scoring loop:
      tattoo = W_COMPAT*compat + W_COSINE*cosine + style_adj + avoid_adj + styling_adj

    Returns list of scored dicts sorted by TATTOO score descending.
    """
    # --- Outerwear styling heuristics (v3.2) ---
    source_l2 = (source.gemini_category_l2 or "").lower().strip()
    source_is_knit = (
        source.material_family == "knit"
        or source_l2 in _IMPLICIT_KNIT_L2
    )

    scored = []
    for cand in candidates:
        cand.broad_category = _gemini_broad(cand.gemini_category_l1) or target_broad

        compat, dims = compute_compatibility_score(source, cand)
        cosine = cand.similarity
        tattoo = W_COMPAT * compat + W_COSINE * cosine

        # Outerwear style adjustment
        style_adj = 0.0
        if target_broad == "outerwear":
            cand_l2 = (cand.gemini_category_l2 or "").lower().strip()
            if source_is_knit and cand_l2 == "cardigan":
                style_adj += _KNIT_CARDIGAN_PENALTY
            if cand_l2 in _STRUCTURED_OUTERWEAR:
                style_adj += _STRUCTURED_BONUS
        tattoo += style_adj

        # Outfit avoids (no user_styles at precompute time)
        avoid_adj, _triggered = compute_avoid_penalties(source, cand, set())
        tattoo += avoid_adj

        scored.append({
            "profile": cand,
            "compat": compat,
            "cosine": cosine,
            "tattoo": tattoo,
            "style_adjustment": round(style_adj, 4),
            "avoid_adjustment": round(avoid_adj, 4),
            "styling_adjustment": 0.0,
        })

    # --- Styling metadata scorer (v3.4) ---
    # Deterministic scoring from pre-computed styling_metadata.
    # Graceful: if source lacks v1.0.0.2 metadata, all adjustments stay 0.
    if scored and source.styling_metadata:
        for entry in scored:
            cp = entry["profile"]
            try:
                result = compute_styling_adjustment(
                    source=source,
                    candidate=cp,
                    target_category=target_broad,
                    source_styling_metadata=source.styling_metadata,
                    candidate_appearance_tags=cp.appearance_top_tags,
                    candidate_vibe_tags=cp.vibe_tags,
                    source_appearance_tags=source.appearance_top_tags,
                    source_vibe_tags=source.vibe_tags,
                )
                styling_adj = result.styling_adj
                entry["styling_adjustment"] = round(styling_adj, 4)
                entry["tattoo"] += styling_adj
            except Exception:
                pass  # graceful degradation

    # Sort by TATTOO descending
    scored.sort(key=lambda e: e["tattoo"], reverse=True)

    # --- Cardigan density cap (v3.2) ---
    if target_broad == "outerwear":
        kept, overflow, card_count = [], [], 0
        for entry in scored:
            l2 = (entry["profile"].gemini_category_l2 or "").lower().strip()
            if l2 == "cardigan":
                card_count += 1
                if card_count <= _MAX_CARDIGANS_OUTERWEAR:
                    kept.append(entry)
                else:
                    overflow.append(entry)
            else:
                kept.append(entry)
        scored = kept + overflow

    return scored


def score_all_pools(
    embeddings: np.ndarray,
    product_ids: np.ndarray,
    categories: np.ndarray,
    category_l2: np.ndarray,
    indexes: Dict[str, Tuple[faiss.IndexFlatIP, np.ndarray]],
    profiles_cache: Dict[str, AestheticProfile],
) -> List[Tuple[str, str, str, float, float, int]]:
    """Retrieve, filter, score all source×target pools.

    For each source product:
      1. faiss batch retrieve top-K_RETRIEVE from each complementary category
      2. Build candidate profiles (from cache) with faiss cosine as similarity
      3. Full filtering: _filter_by_gemini_category, _deduplicate,
         _remove_sets_and_non_outfit, filter_hard_avoids
      4. Full TATTOO scoring: compat + cosine + style_adj + avoid_adj + styling_adj
      5. Density caps (cardigan cap for outerwear)
      6. Keep top CANDIDATES_PER_POOL

    Returns list of (source_id, target_category, candidate_id,
                     cosine_similarity, tattoo_score, rank).
    """
    results: List[Tuple[str, str, str, float, float, int]] = []
    K = K_RETRIEVE
    FINAL_K = CANDIDATES_PER_POOL

    # Build source masks per category (exclude soft-layer outerwear sources)
    cat_to_source_mask: Dict[str, np.ndarray] = {}
    for src_cat in COMPLEMENTARY:
        mask = categories == src_cat
        if src_cat == "outerwear":
            l2_vals = category_l2[mask]
            keep = np.array([v not in _OUTERWEAR_EXCLUDE_L2 for v in l2_vals])
            base_indices = np.where(mask)[0]
            filtered_indices = base_indices[keep]
            excluded = int((~keep).sum())
            if excluded:
                log.info("  Excluded %d soft-layer outerwear sources (cardigan/vest/hoodie)", excluded)
            new_mask = np.zeros(len(categories), dtype=bool)
            new_mask[filtered_indices] = True
            mask = new_mask
        cat_to_source_mask[src_cat] = mask

    total_pools = 0
    total_scored = 0
    skipped_no_profile = 0
    t0 = time.time()

    # Stats accumulators
    filter_stats = {"pre_filter": 0, "post_filter": 0, "post_hard_avoid": 0}
    score_stats = {"styling_active": 0, "styling_total": 0}

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

            log.info("Scoring %s → %s: %d sources × %d targets (retrieve %d, keep %d)...",
                     src_cat, tgt_cat, n_sources, n_targets, K, FINAL_K)

            # --- Batch faiss search (fast: all sources at once) ---
            actual_k = min(K + 10, n_targets)
            t1 = time.time()
            similarities, indices = tgt_index.search(src_embs, actual_k)
            search_time = time.time() - t1
            log.info("  faiss search: %.2fs (%.0f queries/s)",
                     search_time, n_sources / search_time if search_time > 0 else 0)

            # --- Per-source scoring ---
            t_score = time.time()
            pool_count = 0
            src_skipped = 0
            cand_cache_hits = 0
            cand_cache_misses = 0
            for i in range(n_sources):
                src_pid = str(src_ids[i])
                src_profile = profiles_cache.get(src_pid)
                if not src_profile:
                    skipped_no_profile += 1
                    src_skipped += 1
                    continue

                # Build candidate list with faiss similarities
                candidates: List[AestheticProfile] = []
                for j in range(actual_k):
                    idx = indices[i, j]
                    if idx < 0:
                        continue
                    cand_pid = str(tgt_ids[idx])
                    if cand_pid == src_pid:
                        continue
                    cand_profile = profiles_cache.get(cand_pid)
                    if not cand_profile:
                        cand_cache_misses += 1
                        continue
                    cand_cache_hits += 1
                    # Create copy with faiss similarity set
                    cand_copy = dataclasses.replace(cand_profile, similarity=float(similarities[i, j]))
                    candidates.append(cand_copy)

                if not candidates:
                    continue

                filter_stats["pre_filter"] += len(candidates)

                # --- Full filtering pipeline ---
                candidates = _filter_by_gemini_category(src_profile, candidates, tgt_cat)
                candidates = _deduplicate(candidates)
                candidates = _remove_sets_and_non_outfit(src_profile, candidates)

                filter_stats["post_filter"] += len(candidates)

                # Hard avoids (no user_styles at precompute time)
                candidates = filter_hard_avoids(src_profile, candidates, set())

                filter_stats["post_hard_avoid"] += len(candidates)

                if not candidates:
                    continue

                # --- Full TATTOO scoring ---
                scored = _score_one_pool(src_profile, candidates, tgt_cat)

                if src_profile.styling_metadata:
                    score_stats["styling_active"] += 1
                score_stats["styling_total"] += 1

                # Take top FINAL_K
                top = scored[:FINAL_K]

                # Collect results
                for rank, entry in enumerate(top, 1):
                    cp = entry["profile"]
                    results.append((
                        src_pid,
                        tgt_cat,
                        cp.product_id,
                        round(entry["cosine"], 6),
                        round(entry["tattoo"], 6),
                        rank,
                    ))

                pool_count += 1
                total_scored += len(top)

            total_pools += pool_count
            score_time = time.time() - t_score
            elapsed = time.time() - t0

            total_lookups = cand_cache_hits + cand_cache_misses
            hit_rate = 100 * cand_cache_hits / total_lookups if total_lookups else 0
            log.info("  scored %d pools in %.1fs (%.0f pools/s) | "
                     "src_skipped=%d | cand_hits=%d misses=%d (%.0f%% hit rate)",
                     pool_count, score_time,
                     pool_count / score_time if score_time > 0 else 0,
                     src_skipped, cand_cache_hits, cand_cache_misses, hit_rate)
            log.info("  running totals: %d pools, %d rows, %.0fs elapsed",
                     total_pools, len(results), elapsed)

    # Summary stats
    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("Scoring complete: %d pools, %d candidate rows in %.0fs (%.1f min)",
             total_pools, len(results), elapsed, elapsed / 60)
    log.info("  Skipped (no profile): %d sources", skipped_no_profile)

    pre = filter_stats["pre_filter"]
    post = filter_stats["post_filter"]
    hard = filter_stats["post_hard_avoid"]
    if pre > 0:
        log.info("  Filtering: %d pre → %d post-filter (%.0f%% removed) → %d post-hard-avoid",
                 pre, post, 100 * (1 - post / pre), hard)

    sa = score_stats["styling_active"]
    st = score_stats["styling_total"]
    if st > 0:
        log.info("  Styling metadata: %d/%d sources had v1.0.0.2 metadata (%.0f%%)",
                 sa, st, 100 * sa / st)

    return results


# ===========================================================================
# Phase 4: Upload to Supabase
# ===========================================================================

def upload_results(results: List[Tuple[str, str, str, float, float, int]]):
    """Upload results to outfit_candidates table in parallel batches.

    Each result tuple: (source_id, target_category, candidate_id,
                        cosine_similarity, tattoo_score, rank)
    """
    if DRY_RUN:
        log.info("DRY_RUN=1, skipping upload. Would insert %d rows.", len(results))
        # Print sample
        for r in results[:5]:
            log.info("  sample: src=%s tgt=%s cand=%s cos=%.4f tat=%.4f rank=%d",
                     r[0][:12], r[1], r[2][:12], r[3], r[4], r[5])
        return

    sb = _get_supabase()

    # Truncate existing data (batched to avoid Supabase statement timeout)
    log.info("Clearing existing outfit_candidates...")
    for tgt in ["tops", "bottoms", "outerwear", "dresses"]:
        try:
            sb.table("outfit_candidates").delete().eq(
                "target_category", tgt
            ).execute()
            log.info("  cleared target_category=%s", tgt)
        except Exception as e:
            log.warning("  clear target_category=%s failed: %s", tgt, str(e)[:120])
    # Catch any rows with unexpected target_category values
    try:
        sb.table("outfit_candidates").delete().neq(
            "source_id", "00000000-0000-0000-0000-000000000000"
        ).execute()
        log.info("  cleared remaining rows")
    except Exception as e:
        log.warning("  final clear failed (may already be empty): %s", str(e)[:120])

    # Prepare batches
    total = len(results)
    batches = []
    for i in range(0, total, BATCH_SIZE):
        batch = []
        for src_id, tgt_cat, cand_id, cosine, tattoo, rank in results[i:i + BATCH_SIZE]:
            batch.append({
                "source_id": src_id,
                "target_category": tgt_cat,
                "candidate_id": cand_id,
                "cosine_similarity": cosine,
                "rank": rank,
            })
        batches.append(batch)

    log.info("Uploading %d rows in %d batches of %d with %d workers...",
             total, len(batches), BATCH_SIZE, WORKERS)

    uploaded = 0
    errors = 0
    t0 = time.time()

    def _upload_batch(batch: List[dict]) -> int:
        _sb = _new_supabase()
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
            if (uploaded // BATCH_SIZE) % 20 == 0 and uploaded > 0:
                elapsed = time.time() - t0
                rate = uploaded / elapsed
                eta = (total - uploaded) / rate if rate > 0 else 0
                log.info("  uploaded %d/%d rows (%.0f/s) | %d errors | ETA %.0fs",
                         uploaded, total, rate, errors, eta)

    elapsed = time.time() - t0
    log.info("Upload done: %d rows in %.0fs (%d errors)", uploaded, elapsed, errors)


# ===========================================================================
# Main
# ===========================================================================

def main():
    log.info("=" * 60)
    log.info("Precompute Outfit Candidates (Full TATTOO Pipeline)")
    log.info("  WORKERS=%d  K_RETRIEVE=%d  FINAL_K=%d  BATCH=%d  DRY_RUN=%s",
             WORKERS, K_RETRIEVE, CANDIDATES_PER_POOL, BATCH_SIZE, DRY_RUN)
    log.info("  SKIP_EMBEDDINGS=%s  SKIP_CATALOG=%s  CATALOG_BATCH=%d",
             SKIP_EMBEDDINGS, SKIP_CATALOG, CATALOG_BATCH)
    log.info("=" * 60)

    t_total = time.time()

    # Phase 1a: Export embeddings
    log.info("\n--- Phase 1a: Export embeddings ---")
    embeddings, product_ids, categories, category_l2 = export_embeddings()
    log.info("  %d products, %d dimensions, categories: %s",
             len(product_ids), embeddings.shape[1],
             {c: int((categories == c).sum()) for c in ["tops", "bottoms", "outerwear", "dresses"]})

    # Phase 1b: Export product catalog
    log.info("\n--- Phase 1b: Export product catalog ---")
    product_data, attrs_data = export_product_catalog()

    # Phase 2: Build faiss indexes
    log.info("\n--- Phase 2: Build faiss indexes ---")
    indexes = build_category_indexes(embeddings, product_ids, categories, category_l2)

    # Phase 3a: Build profile cache
    log.info("\n--- Phase 3a: Build AestheticProfile cache ---")
    profiles_cache = build_profiles_cache(product_data, attrs_data, product_ids)

    # Free catalog data — profiles_cache holds everything we need
    del product_data, attrs_data

    # Phase 3b: Score all pools
    log.info("\n--- Phase 3b: Score all pools (full TATTOO pipeline) ---")
    results = score_all_pools(
        embeddings, product_ids, categories, category_l2,
        indexes, profiles_cache,
    )

    # Phase 4: Upload
    log.info("\n--- Phase 4: Upload to Supabase ---")
    upload_results(results)

    elapsed = time.time() - t_total
    log.info("\n" + "=" * 60)
    log.info("DONE in %.0fs (%.1f min)", elapsed, elapsed / 60)
    log.info("  %d profiles, %d pools, %d candidate rows",
             len(profiles_cache),
             len(set((r[0], r[1]) for r in results)),
             len(results))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
