#!/usr/bin/env python
"""Side-by-side diversity comparison report: BEFORE vs AFTER.

Runs build_outfit() on the SAME products with two different diversity configs:
  - BEFORE: Pure MMR (lambda=0.08, floor=0.08, 5-dim signature, no L2 awareness)
  - AFTER:  v3.5 L2-stratified round-robin + stronger MMR (lambda=0.25, floor=0.18,
            6-dim signature with category_l2 weighted 2x)

Generates an HTML report showing both results side-by-side with L2 distribution stats.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/diversity_comparison_report.py
    N_OUTFITS=5 PYTHONPATH=src .venv/bin/python scripts/diversity_comparison_report.py
"""

import copy
import logging
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("diversity_report")
logger.setLevel(logging.INFO)

from config.database import get_supabase_client
from services.outfit_engine import (
    OutfitEngine,
    _diverse_select,
    _item_signature,
    _sig_overlap,
    AestheticProfile,
)

N_OUTFITS = int(os.environ.get("N_OUTFITS", 10))
OUTPUT_FILE = ROOT / "scripts" / "diversity_comparison_report.html"

# ---------------------------------------------------------------------------
# OLD diversity functions (monkey-patched for BEFORE runs)
# These restore the pre-v3.5 behavior: 5-dim signature, weak pure-MMR
# ---------------------------------------------------------------------------

def _normalize_color_safe(val):
    """Safely normalize color."""
    try:
        from services.outfit_engine import _normalize_color
        return _normalize_color(val)
    except Exception:
        return (val or "").lower().strip() or None


def _item_signature_old(entry: Dict) -> Dict[str, Optional[str]]:
    """OLD 5-dim signature (no category_l2)."""
    p: AestheticProfile = entry["profile"]
    return {
        "color": _normalize_color_safe(p.color_family) or _normalize_color_safe(p.primary_color),
        "fabric": (p.material_family or "").lower() or None,
        "pattern": (p.pattern or "").lower() or None,
        "silhouette": (p.silhouette or "").lower() or None,
        "brand": (p.brand or "").lower() or None,
    }


def _sig_overlap_old(a: Dict[str, Optional[str]], b: Dict[str, Optional[str]]) -> float:
    """OLD equal-weight overlap (no category_l2)."""
    keys = ["color", "fabric", "pattern", "silhouette", "brand"]
    matches = 0
    compared = 0
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if va and vb:
            compared += 1
            if va == vb:
                matches += 1
    if compared == 0:
        return 0.0
    return matches / compared


def _diverse_select_old(
    scored: List[Dict],
    diversity_lambda: float = 0.08,
    quality_floor: float = 0.08,
) -> List[Dict]:
    """OLD pure-MMR with weak lambda=0.08, floor=0.08, no L2 round-robin."""
    if len(scored) <= 1:
        return scored

    selected: List[Dict] = []
    selected_sigs: List[Dict[str, Optional[str]]] = []
    remaining = list(scored)

    best_score = remaining[0]["tattoo"]
    selected.append(remaining.pop(0))
    selected_sigs.append(_item_signature_old(selected[0]))

    while remaining:
        best_idx = 0
        best_adjusted = -999.0

        for i, cand in enumerate(remaining):
            if cand["tattoo"] < best_score - quality_floor:
                continue
            cand_sig = _item_signature_old(cand)
            max_overlap = max(
                _sig_overlap_old(cand_sig, sel_sig)
                for sel_sig in selected_sigs
            )
            adjusted = (1.0 - diversity_lambda) * cand["tattoo"] - diversity_lambda * max_overlap
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_idx = i

        if best_adjusted <= -999.0 and remaining:
            best_idx = 0

        picked = remaining.pop(best_idx)
        selected.append(picked)
        selected_sigs.append(_item_signature_old(picked))

    return selected


# ---------------------------------------------------------------------------
# Fetch products with precomputed pools
# ---------------------------------------------------------------------------

def fetch_precomputed_products(sb, n: int = 10) -> List[str]:
    """Get product IDs that have precomputed outfit pools (no dresses)."""
    try:
        r = sb.table("outfit_candidates").select(
            "source_id"
        ).eq("rank", 1).limit(n * 20).execute()
        if r.data:
            ids = list({row["source_id"] for row in r.data})
            random.shuffle(ids)
            return ids[:n]
    except Exception as e:
        logger.warning("outfit_candidates query failed: %s", e)
    return []


# ---------------------------------------------------------------------------
# Run build_outfit with monkey-patched diversity
# ---------------------------------------------------------------------------

def run_outfit(engine: OutfitEngine, pid: str, use_old: bool = False) -> Dict[str, Any]:
    """Run build_outfit.  use_old=True patches in the pre-v3.5 diversity code."""
    import services.outfit_engine as mod

    # Save current (v3.5) functions
    orig_diverse = mod._diverse_select
    orig_sig = mod._item_signature
    orig_overlap = mod._sig_overlap

    if use_old:
        mod._diverse_select = _diverse_select_old
        mod._item_signature = _item_signature_old
        mod._sig_overlap = _sig_overlap_old

    try:
        t0 = time.time()
        result = engine.build_outfit(product_id=pid, items_per_category=4, user_id=None)
        elapsed = time.time() - t0
        result["_elapsed"] = elapsed
        return result
    finally:
        # Restore v3.5 functions
        mod._diverse_select = orig_diverse
        mod._item_signature = orig_sig
        mod._sig_overlap = orig_overlap


# ---------------------------------------------------------------------------
# Extract diversity metrics from result
# ---------------------------------------------------------------------------

def extract_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract diversity metrics from a build_outfit result."""
    metrics = {}
    recs = result.get("recommendations", {})

    for cat, cat_data in recs.items():
        items = cat_data.get("items", [])
        if not items:
            continue

        l2_list = []
        colors = []
        brands = []
        fabrics = []
        tattoo_scores = []

        for item in items:
            l2_list.append(item.get("category_l2", item.get("gemini_category_l2", "unknown")) or "unknown")
            colors.append(item.get("color_family", "?") or "?")
            brands.append(item.get("brand", "?") or "?")
            fabrics.append(item.get("material_family", "?") or "?")
            tattoo_scores.append(item.get("tattoo_score", 0) or 0)

        metrics[cat] = {
            "count": len(items),
            "l2_distribution": dict(Counter(l2_list)),
            "unique_l2": len(set(l2_list)),
            "color_distribution": dict(Counter(colors)),
            "unique_colors": len(set(colors)),
            "brand_distribution": dict(Counter(brands)),
            "unique_brands": len(set(brands)),
            "fabric_distribution": dict(Counter(fabrics)),
            "avg_tattoo": sum(tattoo_scores) / len(tattoo_scores) if tattoo_scores else 0,
            "min_tattoo": min(tattoo_scores) if tattoo_scores else 0,
            "max_tattoo": max(tattoo_scores) if tattoo_scores else 0,
        }

    return metrics


# ---------------------------------------------------------------------------
# HTML Report Builder
# ---------------------------------------------------------------------------

def build_html(comparisons: List[Dict], total_time: float) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Aggregate stats
    total_before_l2 = 0
    total_after_l2 = 0
    n_cats = 0
    tattoo_deltas = []

    for comp in comparisons:
        for cat in comp.get("before_metrics", {}):
            bm = comp["before_metrics"][cat]
            am = comp.get("after_metrics", {}).get(cat, {})
            total_before_l2 += bm.get("unique_l2", 0)
            total_after_l2 += am.get("unique_l2", 0)
            n_cats += 1
            if am.get("avg_tattoo"):
                tattoo_deltas.append(am["avg_tattoo"] - bm["avg_tattoo"])

    avg_before_l2 = total_before_l2 / n_cats if n_cats else 0
    avg_after_l2 = total_after_l2 / n_cats if n_cats else 0
    avg_tattoo_delta = sum(tattoo_deltas) / len(tattoo_deltas) if tattoo_deltas else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Diversity Comparison Report — {now}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 20px; max-width: 1800px; margin: 0 auto; }}
  h1 {{ color: #58a6ff; margin-bottom: 8px; }}
  h2 {{ color: #d2a8ff; margin: 16px 0 8px 0; font-size: 18px; }}
  .meta {{ color: #8b949e; margin-bottom: 16px; font-size: 14px; }}

  /* Summary stats */
  .summary {{ display: flex; gap: 20px; margin-bottom: 24px; flex-wrap: wrap; }}
  .stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                padding: 16px; min-width: 200px; text-align: center; }}
  .stat-value {{ font-size: 28px; font-weight: 700; }}
  .stat-label {{ font-size: 12px; color: #8b949e; margin-top: 4px; }}
  .stat-good {{ color: #7ee787; }}
  .stat-neutral {{ color: #d29922; }}
  .stat-bad {{ color: #f85149; }}

  /* Comparison outfit */
  .comparison {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px;
                 margin-bottom: 28px; overflow: hidden; }}
  .comp-header {{ background: #1c2333; padding: 14px 18px; display: flex;
                   align-items: center; gap: 16px; border-bottom: 1px solid #30363d; }}
  .comp-num {{ background: #238636; color: white; font-weight: 700; font-size: 16px;
               border-radius: 8px; padding: 4px 12px; }}
  .comp-meta {{ font-size: 13px; color: #8b949e; }}
  .comp-meta b {{ color: #58a6ff; }}

  .comp-body {{ display: flex; gap: 0; }}
  .source-col {{ width: 180px; min-width: 180px; background: #1a1f2e;
                 border-right: 2px solid #238636; padding: 12px; text-align: center; }}
  .source-col img {{ width: 150px; height: 210px; object-fit: cover; border-radius: 8px;
                     border: 2px solid #238636; }}
  .source-label {{ color: #238636; font-weight: 700; font-size: 11px;
                   text-transform: uppercase; margin-bottom: 4px; }}
  .source-name {{ font-size: 11px; color: #c9d1d9; margin-top: 4px;
                  max-height: 28px; overflow: hidden; }}
  .source-brand {{ font-size: 10px; color: #8b949e; }}
  .source-price {{ font-size: 12px; color: #58a6ff; font-weight: 600; }}

  /* Side-by-side columns */
  .sides {{ flex: 1; display: flex; gap: 0; }}
  .side {{ flex: 1; padding: 10px; }}
  .side-before {{ border-right: 1px dashed #30363d; }}
  .side-label {{ font-size: 11px; font-weight: 700; text-transform: uppercase;
                 padding: 4px 10px; border-radius: 4px; margin-bottom: 8px;
                 display: inline-block; }}
  .side-label-before {{ background: #30363d; color: #c9d1d9; }}
  .side-label-after {{ background: #1f3d2a; color: #7ee787; }}

  .cat-section {{ margin-bottom: 12px; }}
  .cat-header {{ font-size: 11px; font-weight: 700; color: #d2a8ff;
                 text-transform: uppercase; margin-bottom: 4px;
                 border-bottom: 1px solid #30363d; padding-bottom: 3px;
                 display: flex; justify-content: space-between; }}
  .l2-dist {{ font-size: 10px; color: #8b949e; font-weight: 400; }}

  .items-row {{ display: flex; gap: 8px; flex-wrap: nowrap; overflow-x: auto; }}
  .item-card {{ width: 120px; min-width: 120px; text-align: center; }}
  .item-card img {{ width: 110px; height: 155px; object-fit: cover; border-radius: 6px;
                    border: 1px solid #30363d; }}
  .item-rank {{ display: inline-block; background: #30363d; color: #c9d1d9;
               border-radius: 3px; padding: 1px 5px; font-size: 10px; font-weight: 700;
               margin-bottom: 3px; }}
  .item-name {{ font-size: 9px; color: #c9d1d9; max-height: 24px; overflow: hidden;
               line-height: 1.2; margin-top: 3px; }}
  .item-brand {{ font-size: 9px; color: #8b949e; }}
  .item-l2 {{ font-size: 9px; color: #d2a8ff; font-weight: 600; }}
  .item-score {{ font-size: 9px; color: #7ee787; font-family: monospace; }}
  .item-detail {{ font-size: 9px; color: #8b949e; font-family: monospace; }}
</style>
</head>
<body>
<h1>Diversity Comparison Report</h1>
<p class="meta">Generated {now} | {len(comparisons)} outfits | Total: {total_time:.0f}s</p>
<p class="meta">
  <b>BEFORE:</b> Pure MMR — lambda=0.08, floor=0.08, 5-dim signature (color/fabric/pattern/silhouette/brand), no L2 awareness
  <br><b>AFTER (v3.5):</b> L2-stratified round-robin (Phase 1: best item per unique L2 subcategory) + stronger MMR (Phase 2: lambda=0.25, floor=0.18, 6-dim signature with category_l2 weighted 2x)
</p>

<div class="summary">
  <div class="stat-card">
    <div class="stat-value stat-neutral">{avg_before_l2:.1f}</div>
    <div class="stat-label">Avg unique L2 types (BEFORE)</div>
  </div>
  <div class="stat-card">
    <div class="stat-value stat-good">{avg_after_l2:.1f}</div>
    <div class="stat-label">Avg unique L2 types (AFTER)</div>
  </div>
  <div class="stat-card">
    <div class="stat-value {"stat-good" if avg_after_l2 > avg_before_l2 else "stat-bad"}">{avg_after_l2 - avg_before_l2:+.1f}</div>
    <div class="stat-label">L2 diversity improvement</div>
  </div>
  <div class="stat-card">
    <div class="stat-value {"stat-good" if avg_tattoo_delta >= -0.03 else "stat-bad"}">{avg_tattoo_delta:+.3f}</div>
    <div class="stat-label">Avg TATTOO score delta</div>
  </div>
</div>
"""

    for idx, comp in enumerate(comparisons):
        src = comp.get("source", {})
        before = comp.get("before", {})
        after = comp.get("after", {})
        before_m = comp.get("before_metrics", {})
        after_m = comp.get("after_metrics", {})
        before_time = comp.get("before_time", 0)
        after_time = comp.get("after_time", 0)

        html += f"""
<div class="comparison">
  <div class="comp-header">
    <span class="comp-num">#{idx+1}</span>
    <div class="comp-meta">
      <b>{escape(src.get('name','?')[:50])}</b> — {escape(src.get('brand','?'))}
      — ${src.get('price', 0):.0f}
      | before: {before_time:.1f}s, after: {after_time:.1f}s
    </div>
  </div>
  <div class="comp-body">
    <div class="source-col">
      <div class="source-label">Source</div>
      <img src="{escape(src.get('image_url',''))}" alt="source" loading="lazy"
           onerror="this.style.display='none'">
      <div class="source-name">{escape(src.get('name','?')[:40])}</div>
      <div class="source-brand">{escape(src.get('brand','?'))}</div>
      <div class="source-price">${src.get('price', 0):.0f}</div>
    </div>
    <div class="sides">
      <div class="side side-before">
        <div class="side-label side-label-before">BEFORE (current)</div>
"""

        # Render BEFORE side
        before_recs = before.get("recommendations", {})
        for cat, cat_data in before_recs.items():
            items = cat_data.get("items", [])
            m = before_m.get(cat, {})
            l2_dist = m.get("l2_distribution", {})
            l2_str = ", ".join(f"{v}×{k}" for k, v in sorted(l2_dist.items(), key=lambda x: -x[1]))
            unique_l2 = m.get("unique_l2", 0)

            html += f'        <div class="cat-section"><div class="cat-header"><span>{escape(cat)} ({len(items)})</span><span class="l2-dist">L2: {unique_l2} types — {escape(l2_str)}</span></div><div class="items-row">\n'
            for item in items:
                html += _render_item(item)
            html += "        </div></div>\n"

        html += """      </div>
      <div class="side side-after">
        <div class="side-label side-label-after">AFTER (proposed)</div>
"""

        # Render AFTER side
        after_recs = after.get("recommendations", {})
        for cat, cat_data in after_recs.items():
            items = cat_data.get("items", [])
            m = after_m.get(cat, {})
            l2_dist = m.get("l2_distribution", {})
            l2_str = ", ".join(f"{v}×{k}" for k, v in sorted(l2_dist.items(), key=lambda x: -x[1]))
            unique_l2 = m.get("unique_l2", 0)

            html += f'        <div class="cat-section"><div class="cat-header"><span>{escape(cat)} ({len(items)})</span><span class="l2-dist">L2: {unique_l2} types — {escape(l2_str)}</span></div><div class="items-row">\n'
            for item in items:
                html += _render_item(item)
            html += "        </div></div>\n"

        html += """      </div>
    </div>
  </div>
</div>
"""

    html += "</body></html>"
    return html


def _render_item(item: Dict) -> str:
    """Render a single item card."""
    img_url = item.get("image_url", "")
    l2 = item.get("category_l2", "") or item.get("gemini_category_l2", "") or "?"
    tattoo = item.get("tattoo_score", 0)
    color = item.get("color_family", "") or "?"
    fabric = item.get("material_family", "") or "?"

    return f"""          <div class="item-card">
            <span class="item-rank">#{item.get('rank','?')}</span>
            <img src="{escape(img_url)}" alt="" loading="lazy" onerror="this.style.display='none'">
            <div class="item-l2">{escape(str(l2))}</div>
            <div class="item-name">{escape((item.get('name','') or '?')[:35])}</div>
            <div class="item-brand">{escape(item.get('brand','?') or '?')}</div>
            <div class="item-score">T={tattoo:.3f}</div>
            <div class="item-detail">{escape(str(color))} · {escape(str(fabric))}</div>
          </div>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sb = get_supabase_client()
    engine = OutfitEngine(sb)

    logger.info("Fetching products with precomputed pools...")
    pids = fetch_precomputed_products(sb, n=N_OUTFITS + 5)
    if not pids:
        logger.error("No products with precomputed pools found!")
        sys.exit(1)
    logger.info("Got %d candidate products", len(pids))

    comparisons = []
    t_total = time.time()

    for i, pid in enumerate(pids):
        if len(comparisons) >= N_OUTFITS:
            break

        logger.info("[%d/%d] Product %s...", i + 1, len(pids), pid[:12])

        # Run BEFORE (old diversity: weak MMR, 5-dim, no L2 round-robin)
        logger.info("  BEFORE run...")
        before = run_outfit(engine, pid, use_old=True)
        before_time = before.get("_elapsed", 0)

        if before.get("error") and not before.get("recommendations"):
            logger.warning("  Skipping %s: %s", pid[:12], before.get("error"))
            continue

        # Run AFTER (v3.5: L2 round-robin + stronger MMR)
        logger.info("  AFTER run...")
        after = run_outfit(engine, pid, use_old=False)
        after_time = after.get("_elapsed", 0)

        before_metrics = extract_metrics(before)
        after_metrics = extract_metrics(after)

        # Log quick summary
        for cat in before_metrics:
            bl2 = before_metrics[cat].get("unique_l2", 0)
            al2 = after_metrics.get(cat, {}).get("unique_l2", 0)
            bt = before_metrics[cat].get("avg_tattoo", 0)
            at = after_metrics.get(cat, {}).get("avg_tattoo", 0)
            logger.info("  %s: L2 types %d→%d | avg TATTOO %.3f→%.3f",
                        cat, bl2, al2, bt, at)

        comparisons.append({
            "source": before.get("source_product", {}),
            "before": before,
            "after": after,
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "before_time": before_time,
            "after_time": after_time,
        })

        # Write incremental report
        total_time = time.time() - t_total
        html = build_html(comparisons, total_time)
        OUTPUT_FILE.write_text(html, encoding="utf-8")

    total_time = time.time() - t_total
    html = build_html(comparisons, total_time)
    OUTPUT_FILE.write_text(html, encoding="utf-8")

    logger.info("\nDone! %d comparisons in %.0fs", len(comparisons), total_time)
    logger.info("Report: %s", OUTPUT_FILE)
    print(f"\nReport: file://{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
