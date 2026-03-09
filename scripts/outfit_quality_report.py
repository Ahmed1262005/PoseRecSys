#!/usr/bin/env python
"""Generate an HTML report of 40 random outfit builds for quality review.

For each random source product (tops only, no dresses):
  1. Calls OutfitEngine.build_outfit() with limit=4
  2. Shows source image + all recommended items per category
  3. Displays TATTOO scores, avoid adjustments, engine version

Usage:
    PYTHONPATH=src .venv/bin/python scripts/outfit_quality_report.py
"""

import logging
import os
import random
import sys
import time
from datetime import datetime
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("outfit_report")
logger.setLevel(logging.INFO)

from config.database import get_supabase_client
from services.outfit_engine import OutfitEngine

N_OUTFITS = int(os.environ.get("N_OUTFITS", 40))
OUTPUT_FILE = ROOT / "scripts" / "outfit_quality_report.html"


def fetch_random_product_ids(sb, n: int = 50) -> list:
    """Get random product IDs that have precomputed outfit pools (no dresses).

    Prefers products with precomputed pools so the fast Strategy P path
    fires. Falls back to random in-stock tops if the outfit_candidates
    table is empty or doesn't exist.
    """
    try:
        # Pick source_ids that have precomputed pools (exclude dresses)
        r = sb.rpc("get_precomputed_source_ids", {"p_limit": n * 5}).execute()
        if r.data:
            ids = [row["source_id"] for row in r.data]
            random.shuffle(ids)
            logger.info("Using %d products with precomputed pools", min(len(ids), n))
            return ids[:n]
    except Exception:
        pass

    # Fallback: query outfit_candidates directly for distinct source_ids
    try:
        r = sb.table("outfit_candidates").select(
            "source_id"
        ).neq("target_category", "dresses").limit(n * 10).execute()
        if r.data:
            ids = list({row["source_id"] for row in r.data})
            random.shuffle(ids)
            logger.info("Using %d products from outfit_candidates table", min(len(ids), n))
            return ids[:n]
    except Exception:
        pass

    # Final fallback: random in-stock tops
    logger.info("No precomputed pools found, falling back to random tops")
    r = sb.table("products").select("id").eq("in_stock", True).eq(
        "category", "tops"
    ).limit(500).execute()
    ids = [p["id"] for p in (r.data or [])]
    random.shuffle(ids)
    return ids[:n]


def build_html(results: list, total_time: float) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    successful = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Outfit Quality Report — {now}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 20px; }}
  h1 {{ color: #58a6ff; margin-bottom: 8px; }}
  .meta {{ color: #8b949e; margin-bottom: 24px; font-size: 14px; }}
  .outfit {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px;
             margin-bottom: 28px; overflow: hidden; }}
  .outfit-header {{ background: #1c2333; padding: 14px 18px; display: flex;
                    align-items: center; gap: 16px; border-bottom: 1px solid #30363d; }}
  .outfit-num {{ background: #238636; color: white; font-weight: 700; font-size: 16px;
                 border-radius: 8px; padding: 4px 12px; min-width: 36px; text-align: center; }}
  .outfit-meta {{ font-size: 13px; color: #8b949e; }}
  .outfit-meta b {{ color: #58a6ff; }}
  .outfit-body {{ display: flex; gap: 0; }}
  .source-col {{ width: 220px; min-width: 220px; background: #1a1f2e;
                 border-right: 2px solid #238636; padding: 12px; text-align: center; }}
  .source-col img {{ width: 190px; height: 260px; object-fit: cover; border-radius: 8px;
                     border: 2px solid #238636; }}
  .source-label {{ color: #238636; font-weight: 700; font-size: 12px;
                   text-transform: uppercase; margin-bottom: 6px; }}
  .source-name {{ font-size: 12px; color: #c9d1d9; margin-top: 6px;
                  max-height: 32px; overflow: hidden; }}
  .source-brand {{ font-size: 11px; color: #8b949e; }}
  .source-price {{ font-size: 13px; color: #58a6ff; font-weight: 600; }}
  .cats-col {{ flex: 1; padding: 12px; overflow-x: auto; }}
  .cat-section {{ margin-bottom: 14px; }}
  .cat-label {{ font-size: 12px; font-weight: 700; color: #d2a8ff;
                text-transform: uppercase; margin-bottom: 8px;
                border-bottom: 1px solid #30363d; padding-bottom: 4px; }}
  .items-row {{ display: flex; gap: 10px; flex-wrap: nowrap; }}
  .item-card {{ width: 140px; min-width: 140px; text-align: center; }}
  .item-card img {{ width: 130px; height: 180px; object-fit: cover; border-radius: 6px;
                    border: 1px solid #30363d; }}
  .item-rank {{ display: inline-block; background: #30363d; color: #c9d1d9;
                border-radius: 4px; padding: 1px 6px; font-size: 11px; font-weight: 700;
                margin-bottom: 4px; }}
  .item-name {{ font-size: 10px; color: #c9d1d9; max-height: 28px; overflow: hidden;
                line-height: 1.3; margin-top: 4px; }}
  .item-brand {{ font-size: 10px; color: #8b949e; }}
  .item-score {{ font-size: 10px; color: #7ee787; font-family: monospace; }}
   .item-avoid {{ font-size: 10px; color: #f85149; font-family: monospace; }}
  .judge-notes {{ background: #1a1f2e; border-top: 1px solid #30363d; padding: 10px 18px;
                   font-size: 12px; color: #8b949e; line-height: 1.5; }}
  .judge-notes b {{ color: #d2a8ff; }}
  .judge-notes .note {{ color: #c9d1d9; font-style: italic; margin: 4px 0; }}
  .fail {{ color: #f85149; }}
  .summary {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
               padding: 16px; margin-bottom: 20px; }}
</style>
</head>
<body>
<h1>Outfit Quality Report</h1>
<p class="meta">Generated {now} | {len(successful)} outfits built | {len(failed)} failed | Total: {total_time:.0f}s</p>
"""

    for i, r in enumerate(successful):
        src = r["source"]
        recs = r["recommendations"]
        info = r["scoring_info"]
        engine_ver = info.get("engine", "?")
        outfit_ranked = info.get("outfit_ranking", False)
        judge_on = info.get("stylist_judge", False)
        elapsed = r.get("elapsed", 0)

        styling_on = info.get("styling_scorer", False)
        retrieval = info.get("retrieval_strategies", {})
        badges = f"<b>{engine_ver}</b>"
        if styling_on:
            badges += " | styling scorer ON"
        if judge_on:
            badges += " | stylist judge ON"
        if outfit_ranked:
            badges += " | outfit ranked"
        # Show retrieval strategy per category
        for cat, strat in retrieval.items():
            color = "#238636" if strat == "precomputed" else "#d29922"
            badges += f' | <span style="color:{color}">{cat}={strat}</span>'

        html += f"""
<div class="outfit">
  <div class="outfit-header">
    <span class="outfit-num">#{i+1}</span>
    <div class="outfit-meta">{badges} | {elapsed:.1f}s</div>
  </div>
  <div class="outfit-body">
    <div class="source-col">
      <div class="source-label">Source</div>
      <img src="{escape(src.get('image_url',''))}" alt="source" loading="lazy"
           onerror="this.style.display='none'">
      <div class="source-name">{escape(src.get('name','?'))}</div>
      <div class="source-brand">{escape(src.get('brand','?'))}</div>
      <div class="source-price">${src.get('price', 0):.0f}</div>
    </div>
    <div class="cats-col">
"""

        for cat_name, cat_data in recs.items():
            items = cat_data.get("items", [])
            if not items:
                continue
            html += f'      <div class="cat-section"><div class="cat-label">{escape(cat_name)} ({len(items)})</div><div class="items-row">\n'
            for item in items:
                tattoo = item.get("tattoo_score", "?")
                avoid = item.get("avoid_adjustment", 0)
                avoid_str = f"{avoid}" if avoid and avoid != 0 else ""
                styling = item.get("styling_adjustment", 0)
                styling_str = f"{styling}" if styling and styling != 0 else ""
                img_url = item.get("image_url", "")
                html += f"""        <div class="item-card">
          <span class="item-rank">#{item.get('rank','?')}</span>
          <img src="{escape(img_url)}" alt="" loading="lazy" onerror="this.style.display='none'">
          <div class="item-name">{escape(item.get('name','?')[:45])}</div>
          <div class="item-brand">{escape(item.get('brand','?'))}</div>
          <div class="item-score">T={tattoo}</div>
          {'<div class="item-avoid">avoid=' + avoid_str + '</div>' if avoid_str else ''}
          {'<div class="item-score">styling=' + styling_str + '</div>' if styling_str else ''}
        </div>
"""
            html += "      </div></div>\n"

        html += "    </div>\n  </div>\n"

        # Judge notes section
        judge_notes = info.get("judge_notes", {})
        cat_notes = judge_notes.get("category_notes", [])
        outfit_note = judge_notes.get("outfit_note", "")
        if cat_notes or outfit_note:
            html += '  <div class="judge-notes">\n'
            for cn in cat_notes:
                if isinstance(cn, dict):
                    cat = cn.get("category", "?")
                    note = cn.get("note", "")
                elif isinstance(cn, str):
                    cat, note = "?", cn
                else:
                    continue
                if note:
                    html += f'    <b>Stylist ({escape(cat)}):</b> <span class="note">{escape(note)}</span><br>\n'
            if outfit_note:
                html += f'    <b>Outfit ranking:</b> <span class="note">{escape(outfit_note)}</span>\n'
            html += '  </div>\n'

        html += "</div>\n"

    if failed:
        html += '<div class="summary"><h3 class="fail">Failed builds</h3><ul>'
        for r in failed:
            html += f'<li class="fail">{escape(r.get("pid","?"))}: {escape(str(r.get("error","")))}</li>'
        html += "</ul></div>"

    html += "</body></html>"
    return html


def main():
    sb = get_supabase_client()
    engine = OutfitEngine(sb)

    logger.info("Fetching random product IDs...")
    pids = fetch_random_product_ids(sb, n=N_OUTFITS + 2)  # small buffer for failures
    logger.info("Got %d candidate product IDs", len(pids))

    results = []
    t_total = time.time()

    for i, pid in enumerate(pids):
        if len([r for r in results if r.get("ok")]) >= N_OUTFITS:
            break
        logger.info("[%d/%d] Building outfit for %s...", i + 1, len(pids), pid[:12])
        t0 = time.time()
        try:
            result = engine.build_outfit(product_id=pid, limit=4)
            elapsed = time.time() - t0
            # Check if it actually returned recommendations
            recs = result.get("recommendations", {})
            total_items = sum(len(c.get("items", [])) for c in recs.values())
            if total_items == 0:
                results.append({"pid": pid, "ok": False, "error": "No recommendations returned"})
            else:
                results.append({
                    "ok": True,
                    "pid": pid,
                    "source": result.get("source_product", {}),
                    "recommendations": recs,
                    "scoring_info": result.get("scoring_info", {}),
                    "elapsed": elapsed,
                })
                logger.info("  -> %d categories, %d items, %.1fs", len(recs), total_items, elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            logger.warning("  -> FAILED: %s (%.1fs)", e, elapsed)
            results.append({"pid": pid, "ok": False, "error": str(e)})

        # Write report after every outfit so you can preview in browser
        total_time = time.time() - t_total
        html = build_html(results, total_time)
        OUTPUT_FILE.write_text(html, encoding="utf-8")

    total_time = time.time() - t_total
    logger.info("Done. %d successful, %d failed, %.0fs total",
                len([r for r in results if r.get("ok")]),
                len([r for r in results if not r.get("ok")]),
                total_time)

    # Final write
    html = build_html(results, total_time)
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", OUTPUT_FILE)
    print(f"\nReport: {OUTPUT_FILE}")
    print(f"Open in browser: file://{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
