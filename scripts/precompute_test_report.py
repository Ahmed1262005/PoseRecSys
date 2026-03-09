#!/usr/bin/env python
"""Precompute outfit candidate pools for 100 random products and generate
an HTML visual report for quality review.

Pipeline:
  1. Export all embeddings from Supabase (uses cache from precompute script)
  2. Build faiss indexes per category
  3. Pick 100 random v1.0.0.2 source products (no dresses)
  4. Compute top-8 candidates per complementary category via faiss
  5. Fetch product details (name, brand, price, image) for display
  6. Generate dark-themed HTML report

Usage:
    PYTHONPATH=src .venv/bin/python scripts/precompute_test_report.py

Environment:
    N_SOURCES=100   Number of source products to test (default 100)
    TOP_K=8         Candidates shown per category (default 8)
    SKIP_EXPORT=1   Skip embedding export if cache exists (default 1)
"""

import logging
import os
import random
import sys
import time
from datetime import datetime
from html import escape
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
log = logging.getLogger("precompute_test")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_SOURCES = int(os.environ.get("N_SOURCES", 100))
TOP_K = int(os.environ.get("TOP_K", 8))
OUTPUT_FILE = ROOT / "scripts" / "precompute_test_report.html"

# Force SKIP_EXPORT=1 by default (use cache if available)
os.environ.setdefault("SKIP_EXPORT", "1")

# Import precompute pipeline functions
from precompute_outfit_candidates import (
    export_embeddings,
    build_category_indexes,
    COMPLEMENTARY,
    _broad,
    CACHE_DIR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_supabase():
    from config.database import get_supabase_client
    return get_supabase_client()


def pick_source_products(
    product_ids: np.ndarray,
    categories: np.ndarray,
    n: int,
) -> List[Tuple[str, str]]:
    """Pick n random source products (no dresses), preferring v1.0.0.2.

    Returns list of (product_id, broad_category).
    """
    sb = _get_supabase()

    # Try to get v1.0.0.2 products first
    log.info("Fetching v1.0.0.2 product IDs...")
    r = sb.table("product_attributes").select(
        "sku_id, category_l1"
    ).eq("extractor_version", "v1.0.0.2").limit(5000).execute()
    v2_products = {}
    for row in (r.data or []):
        broad = _broad(row.get("category_l1"))
        if broad and broad != "dresses":
            v2_products[str(row["sku_id"])] = broad
    log.info("  %d v1.0.0.2 non-dress products found", len(v2_products))

    # Filter to products that have embeddings (exist in our arrays)
    id_set = set(product_ids.tolist())
    eligible = [(pid, cat) for pid, cat in v2_products.items() if pid in id_set]
    log.info("  %d have embeddings", len(eligible))

    random.shuffle(eligible)
    selected = eligible[:n]

    # If we need more, fill from general pool
    if len(selected) < n:
        needed = n - len(selected)
        selected_ids = {s[0] for s in selected}
        general = []
        for i, pid in enumerate(product_ids):
            cat = str(categories[i])
            if cat != "dresses" and cat and pid not in selected_ids:
                general.append((str(pid), cat))
        random.shuffle(general)
        selected.extend(general[:needed])

    log.info("Selected %d source products: %s",
             len(selected),
             {cat: sum(1 for _, c in selected if c == cat) for cat in ["tops", "bottoms", "outerwear"]})
    return selected


def compute_pools_for_sources(
    sources: List[Tuple[str, str]],
    embeddings: np.ndarray,
    product_ids: np.ndarray,
    categories: np.ndarray,
    indexes: Dict[str, Tuple[faiss.IndexFlatIP, np.ndarray]],
    top_k: int = 8,
) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    """Compute candidate pools for selected source products.

    Returns: {source_id: {target_cat: [(candidate_id, similarity), ...]}}
    """
    # Build product_id -> index mapping for fast embedding lookup
    id_to_idx = {str(pid): i for i, pid in enumerate(product_ids)}

    results: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    k_search = top_k + 5  # buffer for self-dedup

    for src_id, src_cat in sources:
        idx = id_to_idx.get(src_id)
        if idx is None:
            continue

        src_emb = embeddings[idx:idx+1]  # shape (1, 512)
        target_cats = COMPLEMENTARY.get(src_cat, [])
        results[src_id] = {}

        for tgt_cat in target_cats:
            if tgt_cat not in indexes:
                continue
            tgt_index, tgt_ids = indexes[tgt_cat]
            actual_k = min(k_search, tgt_index.ntotal)

            sims, idxs = tgt_index.search(src_emb, actual_k)

            candidates = []
            for j in range(actual_k):
                if len(candidates) >= top_k:
                    break
                ii = idxs[0, j]
                if ii < 0:
                    continue
                cand_id = str(tgt_ids[ii])
                if cand_id == src_id:
                    continue
                candidates.append((cand_id, float(sims[0, j])))
            results[src_id][tgt_cat] = candidates

    return results


def fetch_product_details(
    product_ids_needed: Set[str],
) -> Dict[str, dict]:
    """Batch-fetch product details from Supabase."""
    sb = _get_supabase()
    details: Dict[str, dict] = {}

    ids_list = list(product_ids_needed)
    log.info("Fetching product details for %d products...", len(ids_list))

    for i in range(0, len(ids_list), 500):
        batch = ids_list[i:i+500]
        try:
            r = sb.table("products").select(
                "id, name, brand, price, primary_image_url, category, broad_category"
            ).in_("id", batch).execute()
            for row in (r.data or []):
                details[str(row["id"])] = row
        except Exception as e:
            log.warning("Product fetch failed for batch %d: %s", i, e)

    # Also fetch gemini attributes for category info
    for i in range(0, len(ids_list), 500):
        batch = ids_list[i:i+500]
        try:
            r = sb.table("product_attributes").select(
                "sku_id, category_l1, category_l2, formality, fit_type, "
                "color_family, pattern, apparent_fabric, style_tags, "
                "styling_metadata, extractor_version"
            ).in_("sku_id", batch).execute()
            for row in (r.data or []):
                pid = str(row["sku_id"])
                if pid in details:
                    details[pid]["attrs"] = row
                else:
                    details[pid] = {"attrs": row}
        except Exception as e:
            log.warning("Attrs fetch failed for batch %d: %s", i, e)

    log.info("  fetched %d product details", len(details))
    return details


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def build_html(
    sources: List[Tuple[str, str]],
    pools: Dict[str, Dict[str, List[Tuple[str, float]]]],
    details: Dict[str, dict],
    total_time: float,
    n_embeddings: int,
    cat_counts: Dict[str, int],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_ok = sum(1 for s, _ in sources if s in pools and pools[s])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Precomputed Outfit Pools Report - {now}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 20px; }}
  h1 {{ color: #58a6ff; margin-bottom: 8px; }}
  .meta {{ color: #8b949e; margin-bottom: 20px; font-size: 14px; line-height: 1.6; }}
  .meta b {{ color: #c9d1d9; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                   gap: 12px; margin-bottom: 24px; }}
  .stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                padding: 14px; text-align: center; }}
  .stat-val {{ font-size: 28px; font-weight: 700; color: #58a6ff; }}
  .stat-label {{ font-size: 12px; color: #8b949e; margin-top: 4px; }}
  .outfit {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px;
             margin-bottom: 24px; overflow: hidden; }}
  .outfit-header {{ background: #1c2333; padding: 12px 18px; display: flex;
                    align-items: center; gap: 14px; border-bottom: 1px solid #30363d;
                    flex-wrap: wrap; }}
  .outfit-num {{ background: #238636; color: white; font-weight: 700; font-size: 14px;
                 border-radius: 6px; padding: 3px 10px; }}
  .outfit-meta {{ font-size: 12px; color: #8b949e; }}
  .outfit-meta b {{ color: #58a6ff; }}
  .tag {{ display: inline-block; font-size: 10px; padding: 2px 6px; border-radius: 4px;
          margin-left: 4px; }}
  .tag-v2 {{ background: #1f6feb33; color: #58a6ff; border: 1px solid #1f6feb; }}
  .tag-styling {{ background: #23863633; color: #7ee787; border: 1px solid #238636; }}
  .tag-cat {{ background: #30363d; color: #c9d1d9; }}
  .outfit-body {{ display: flex; gap: 0; }}
  .source-col {{ width: 200px; min-width: 200px; background: #1a1f2e;
                 border-right: 2px solid #238636; padding: 12px; text-align: center; }}
  .source-col img {{ width: 170px; height: 240px; object-fit: cover; border-radius: 8px;
                     border: 2px solid #238636; }}
  .source-label {{ color: #238636; font-weight: 700; font-size: 11px;
                   text-transform: uppercase; margin-bottom: 6px; }}
  .source-name {{ font-size: 11px; color: #c9d1d9; margin-top: 6px;
                  max-height: 30px; overflow: hidden; }}
  .source-brand {{ font-size: 10px; color: #8b949e; }}
  .source-price {{ font-size: 12px; color: #58a6ff; font-weight: 600; }}
  .source-detail {{ font-size: 9px; color: #6e7681; margin-top: 4px; line-height: 1.4; }}
  .cats-col {{ flex: 1; padding: 12px; overflow-x: auto; }}
  .cat-section {{ margin-bottom: 12px; }}
  .cat-label {{ font-size: 11px; font-weight: 700; color: #d2a8ff;
                text-transform: uppercase; margin-bottom: 6px;
                border-bottom: 1px solid #30363d; padding-bottom: 4px; }}
  .items-row {{ display: flex; gap: 8px; flex-wrap: nowrap; overflow-x: auto; }}
  .item-card {{ width: 120px; min-width: 120px; text-align: center; }}
  .item-card img {{ width: 110px; height: 155px; object-fit: cover; border-radius: 6px;
                    border: 1px solid #30363d; }}
  .item-rank {{ display: inline-block; background: #30363d; color: #c9d1d9;
                border-radius: 4px; padding: 1px 5px; font-size: 10px; font-weight: 700;
                margin-bottom: 3px; }}
  .item-name {{ font-size: 9px; color: #c9d1d9; max-height: 24px; overflow: hidden;
                line-height: 1.2; margin-top: 3px; }}
  .item-brand {{ font-size: 9px; color: #8b949e; }}
  .item-score {{ font-size: 10px; color: #7ee787; font-family: monospace; margin-top: 2px; }}
  .item-detail {{ font-size: 8px; color: #6e7681; font-family: monospace; }}
  .sim-bar {{ height: 3px; border-radius: 2px; margin-top: 3px; }}
  .no-pool {{ color: #f85149; font-size: 12px; padding: 20px; }}
</style>
</head>
<body>
<h1>Precomputed Outfit Pools Report</h1>
<p class="meta">
  Generated {now} | <b>{n_ok}</b> source products with pools | Total: <b>{total_time:.0f}s</b>
</p>
<div class="summary-grid">
  <div class="stat-card">
    <div class="stat-val">{n_embeddings:,}</div>
    <div class="stat-label">Total Embeddings</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{cat_counts.get('tops', 0):,}</div>
    <div class="stat-label">Tops in Index</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{cat_counts.get('bottoms', 0):,}</div>
    <div class="stat-label">Bottoms in Index</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{cat_counts.get('outerwear', 0):,}</div>
    <div class="stat-label">Outerwear in Index</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{cat_counts.get('dresses', 0):,}</div>
    <div class="stat-label">Dresses in Index</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{N_SOURCES}</div>
    <div class="stat-label">Sources Tested</div>
  </div>
</div>
"""

    for i, (src_id, src_cat) in enumerate(sources):
        src_detail = details.get(src_id, {})
        src_attrs = src_detail.get("attrs", {})
        src_name = src_detail.get("name", "Unknown")
        src_brand = src_detail.get("brand", "?")
        src_price = src_detail.get("price", 0) or 0
        src_img = src_detail.get("primary_image_url", "")
        src_l1 = src_attrs.get("category_l1", "?")
        src_l2 = src_attrs.get("category_l2", "")
        src_formality = src_attrs.get("formality", "")
        src_fabric = src_attrs.get("apparent_fabric", "")
        src_fit = src_attrs.get("fit_type", "")
        src_pattern = src_attrs.get("pattern", "")
        src_ver = src_attrs.get("extractor_version", "")
        has_styling = bool(src_attrs.get("styling_metadata"))

        pool = pools.get(src_id, {})
        total_cands = sum(len(c) for c in pool.values())

        ver_tag = f'<span class="tag tag-v2">v1.0.0.2</span>' if src_ver == "v1.0.0.2" else ""
        styling_tag = f'<span class="tag tag-styling">styling_metadata</span>' if has_styling else ""

        html += f"""
<div class="outfit">
  <div class="outfit-header">
    <span class="outfit-num">#{i+1}</span>
    <div class="outfit-meta">
      <b>{escape(src_cat)}</b> | {escape(src_l1)} / {escape(src_l2 or '?')}
      | {total_cands} candidates across {len(pool)} categories
      {ver_tag} {styling_tag}
    </div>
  </div>
  <div class="outfit-body">
    <div class="source-col">
      <div class="source-label">Source</div>
      <img src="{escape(src_img)}" alt="source" loading="lazy"
           onerror="this.style.display='none'">
      <div class="source-name">{escape(src_name[:50])}</div>
      <div class="source-brand">{escape(src_brand)}</div>
      <div class="source-price">${src_price:.0f}</div>
      <div class="source-detail">
        {escape(src_formality)} | {escape(src_fit)}<br>
        {escape(src_fabric)} | {escape(src_pattern)}
      </div>
    </div>
    <div class="cats-col">
"""

        if not pool:
            html += '      <div class="no-pool">No complementary categories found</div>\n'
        else:
            for tgt_cat, candidates in pool.items():
                html += f'      <div class="cat-section">\n'
                html += f'        <div class="cat-label">{escape(tgt_cat)} ({len(candidates)} candidates)</div>\n'
                html += f'        <div class="items-row">\n'

                for rank, (cand_id, sim) in enumerate(candidates, 1):
                    d = details.get(cand_id, {})
                    ca = d.get("attrs", {})
                    c_name = d.get("name", "?")
                    c_brand = d.get("brand", "?")
                    c_price = d.get("price", 0) or 0
                    c_img = d.get("primary_image_url", "")
                    c_l2 = ca.get("category_l2", "")
                    c_formality = ca.get("formality", "")
                    c_fit = ca.get("fit_type", "")
                    c_fabric = ca.get("apparent_fabric", "")
                    # Color the sim bar: green > 0.85, yellow 0.7-0.85, red < 0.7
                    bar_pct = max(0, min(100, int(sim * 100)))
                    if sim >= 0.85:
                        bar_color = "#238636"
                    elif sim >= 0.70:
                        bar_color = "#d29922"
                    else:
                        bar_color = "#f85149"

                    html += f"""          <div class="item-card">
            <span class="item-rank">#{rank}</span>
            <img src="{escape(c_img)}" alt="" loading="lazy"
                 onerror="this.style.display='none'">
            <div class="item-name">{escape(c_name[:40])}</div>
            <div class="item-brand">{escape(c_brand)} | ${c_price:.0f}</div>
            <div class="item-score">cos={sim:.3f}</div>
            <div class="item-detail">{escape(c_l2 or '?')} | {escape(c_formality or '?')}</div>
            <div class="item-detail">{escape(c_fit or '?')} | {escape(c_fabric or '?')}</div>
            <div class="sim-bar" style="width:{bar_pct}%;background:{bar_color}"></div>
          </div>
"""
                html += '        </div>\n      </div>\n'

        html += '    </div>\n  </div>\n</div>\n'

    html += """
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.time()

    log.info("=" * 60)
    log.info("Precompute Test Report")
    log.info("  N_SOURCES=%d  TOP_K=%d", N_SOURCES, TOP_K)
    log.info("=" * 60)

    # Phase 1: Export embeddings (cached)
    log.info("\n--- Phase 1: Export embeddings ---")
    embeddings, product_ids, categories = export_embeddings()
    n_embeddings = len(product_ids)
    log.info("  %d products, %d dimensions", n_embeddings, embeddings.shape[1])

    # Phase 2: Build faiss indexes
    log.info("\n--- Phase 2: Build faiss indexes ---")
    indexes = build_category_indexes(embeddings, product_ids, categories)
    cat_counts = {cat: idx[0].ntotal for cat, idx in indexes.items()}

    # Phase 3: Pick source products
    log.info("\n--- Phase 3: Pick %d source products ---", N_SOURCES)
    sources = pick_source_products(product_ids, categories, N_SOURCES)

    # Phase 4: Compute pools
    log.info("\n--- Phase 4: Compute candidate pools ---")
    t0 = time.time()
    pools = compute_pools_for_sources(
        sources, embeddings, product_ids, categories, indexes, TOP_K,
    )
    pool_time = time.time() - t0
    total_candidates = sum(
        sum(len(cands) for cands in p.values())
        for p in pools.values()
    )
    log.info("  %d pools computed, %d total candidates in %.2fs",
             len(pools), total_candidates, pool_time)

    # Phase 5: Fetch product details
    log.info("\n--- Phase 5: Fetch product details ---")
    all_ids_needed: Set[str] = set()
    for src_id, _ in sources:
        all_ids_needed.add(src_id)
    for src_pools in pools.values():
        for cands in src_pools.values():
            for cand_id, _ in cands:
                all_ids_needed.add(cand_id)
    details = fetch_product_details(all_ids_needed)

    # Phase 6: Generate HTML report
    log.info("\n--- Phase 6: Generate HTML report ---")
    total_time = time.time() - t_total
    html = build_html(sources, pools, details, total_time, n_embeddings, cat_counts)
    OUTPUT_FILE.write_text(html, encoding="utf-8")

    log.info("\n" + "=" * 60)
    log.info("DONE in %.0fs", total_time)
    log.info("  %d sources, %d candidate pools, %d total candidates",
             len(sources), len(pools), total_candidates)
    log.info("  Faiss pool computation: %.2fs (%.1fms per source)",
             pool_time, pool_time / len(sources) * 1000 if sources else 0)
    log.info("  Report: %s", OUTPUT_FILE)
    log.info("=" * 60)
    print(f"\nReport: {OUTPUT_FILE}")
    print(f"Open in browser: file://{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
