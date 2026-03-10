#!/usr/bin/env python
"""Outerwear Appropriateness Gate Report (v3.5).

Side-by-side comparison showing how the three-group outerwear gate changes
outerwear recommendations for products across different formality levels.

BEFORE: All structured outerwear gets +0.03 uniformly; volume/heavy ungated.
AFTER:  Three-group gating by source formality & occasions:

  Group 1 — Formal (blazer, suit jacket):
    Form. 1 + no occasion -> HARD FILTERED
    Form. 2 + no occasion -> -0.10 penalty
    Form. 2 + occasion    -> neutral
    Form. 3+              -> +0.03 bonus

  Group 2 — Volume (poncho, cape, kimono, bolero, shrug):
    Form. 1               -> HARD FILTERED (no occasion override)
    Form. 2               -> -0.10 penalty (no occasion override)
    Form. 3+              -> neutral (0.00)

  Group 3 — Heavy (parka, coat):
    Form. 1 + no occasion -> -0.10 penalty
    Form. 1 + occasion    -> +0.03 bonus
    Form. 2+              -> +0.03 bonus

Usage:
    PYTHONPATH=src .venv/bin/python scripts/blazer_gate_report.py
    N_OUTFITS=10 PYTHONPATH=src .venv/bin/python scripts/blazer_gate_report.py
"""

import logging
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("blazer_report")
logger.setLevel(logging.INFO)

from config.database import get_supabase_client
from services.outfit_engine import (
    OutfitEngine,
    AestheticProfile,
    _FORMAL_OUTER_L2,
    _VOLUME_OUTER_L2,
    _HEAVY_OUTER_L2,
    _OUTERWEAR_GATE_OCCASIONS,
    _OUTERWEAR_MIN_FORMALITY,
    _OUTERWEAR_CASUAL_PENALTY,
)

N_OUTFITS = int(os.environ.get("N_OUTFITS", 8))
OUTPUT_FILE = ROOT / "scripts" / "outerwear_gate_report.html"

_FORMALITY_NAMES = {1: "Casual", 2: "Smart Casual", 3: "Business Casual", 4: "Semi-Formal", 5: "Formal"}

# All gated L2 types for tagging items in the report
_ALL_GATED_L2 = _FORMAL_OUTER_L2 | _VOLUME_OUTER_L2 | _HEAVY_OUTER_L2


def _classify_outerwear_group(l2: str) -> Optional[str]:
    """Return the gate group for an outerwear L2 type, or None."""
    l2 = l2.lower().strip()
    if l2 in _FORMAL_OUTER_L2:
        return "formal"
    if l2 in _VOLUME_OUTER_L2:
        return "volume"
    if l2 in _HEAVY_OUTER_L2:
        return "heavy"
    return None


# ---------------------------------------------------------------------------
# Monkey-patch helpers for BEFORE (no outerwear gates)
# ---------------------------------------------------------------------------

def _disable_outerwear_gates():
    """Disable all three outerwear gates by replacing module-level constants."""
    import services.outfit_engine as mod
    saved = {
        "_FORMAL_OUTER_L2": mod._FORMAL_OUTER_L2,
        "_VOLUME_OUTER_L2": mod._VOLUME_OUTER_L2,
        "_HEAVY_OUTER_L2": mod._HEAVY_OUTER_L2,
        "_OUTERWEAR_GATE_OCCASIONS": mod._OUTERWEAR_GATE_OCCASIONS,
        "_OUTERWEAR_MIN_FORMALITY": mod._OUTERWEAR_MIN_FORMALITY,
        "_OUTERWEAR_CASUAL_PENALTY": mod._OUTERWEAR_CASUAL_PENALTY,
        # Backward-compat aliases
        "_BLAZER_GATE_L2": mod._BLAZER_GATE_L2,
        "_BLAZER_GATE_OCCASIONS": mod._BLAZER_GATE_OCCASIONS,
        "_BLAZER_MIN_FORMALITY": mod._BLAZER_MIN_FORMALITY,
        "_BLAZER_CASUAL_PENALTY": mod._BLAZER_CASUAL_PENALTY,
    }
    # Disable hard filters: empty sets means nothing matches
    mod._FORMAL_OUTER_L2 = frozenset()
    mod._VOLUME_OUTER_L2 = frozenset()
    mod._HEAVY_OUTER_L2 = frozenset()
    mod._BLAZER_GATE_L2 = frozenset()
    # Disable scoring: min formality 0 so everything gets +0.03
    mod._OUTERWEAR_MIN_FORMALITY = 0
    mod._BLAZER_MIN_FORMALITY = 0
    mod._OUTERWEAR_CASUAL_PENALTY = 0.0
    mod._BLAZER_CASUAL_PENALTY = 0.0
    return saved


def _restore_outerwear_gates(saved: dict):
    """Restore original outerwear gate constants."""
    import services.outfit_engine as mod
    for key, val in saved.items():
        setattr(mod, key, val)


# ---------------------------------------------------------------------------
# Fetch products by formality spread
# ---------------------------------------------------------------------------

def fetch_products_by_formality(sb, n: int = 8) -> List[dict]:
    """Get products that have precomputed pools, spread across formality levels."""
    try:
        # Get source IDs from outfit_candidates
        r = sb.table("outfit_candidates").select(
            "source_id"
        ).eq("target_category", "outerwear").eq("rank", 1).limit(500).execute()

        if not r.data:
            return []

        source_ids = list({row["source_id"] for row in r.data})
        random.shuffle(source_ids)

        # Fetch product details + attributes for these sources
        batch = source_ids[:min(100, len(source_ids))]
        products = []
        for chunk_start in range(0, len(batch), 20):
            chunk = batch[chunk_start:chunk_start + 20]
            r2 = sb.table("product_attributes").select(
                "sku_id, formality"
            ).in_("sku_id", chunk).execute()
            if r2.data:
                for row in r2.data:
                    products.append(row)

        # Group by formality level
        formality_map = {
            "casual": 1, "smart casual": 2, "business casual": 3,
            "semi-formal": 4, "formal": 5,
        }
        by_level: Dict[int, List[str]] = {1: [], 2: [], 3: [], 4: [], 5: []}
        for p in products:
            f = (p.get("formality") or "").lower().strip()
            level = formality_map.get(f, 2)
            by_level[level].append(p["sku_id"])

        # Pick evenly across levels
        result = []
        per_level = max(1, n // 3)  # Emphasize casual/smart-casual but include formal
        # Prioritize levels 1 and 2 (where blazer gate has most impact)
        for level in [1, 2, 3, 4, 5]:
            ids = by_level.get(level, [])
            random.shuffle(ids)
            take = per_level if level <= 2 else max(1, per_level // 2)
            for pid in ids[:take]:
                if len(result) < n:
                    result.append({"sku_id": pid, "formality_level": level})

        random.shuffle(result)
        return result[:n]

    except Exception as e:
        logger.warning("Failed to fetch products: %s", e)
        return []


# ---------------------------------------------------------------------------
# Run build_outfit
# ---------------------------------------------------------------------------

def run_outfit(engine: OutfitEngine, pid: str, disable_gates: bool = False) -> Dict[str, Any]:
    """Run build_outfit with or without outerwear gates."""
    saved = None
    if disable_gates:
        saved = _disable_outerwear_gates()
    try:
        t0 = time.time()
        result = engine.build_outfit(product_id=pid, items_per_category=4, user_id=None)
        result["_elapsed"] = time.time() - t0
        return result
    finally:
        if saved:
            _restore_outerwear_gates(saved)


# ---------------------------------------------------------------------------
# Extract outerwear details
# ---------------------------------------------------------------------------

def extract_outerwear(result: Dict[str, Any]) -> List[Dict]:
    """Extract outerwear items with gate group metadata."""
    recs = result.get("recommendations", {})
    ow = recs.get("outerwear", {})
    items = ow.get("items", [])
    enriched = []
    for item in items:
        l2 = (item.get("category_l2") or item.get("gemini_category_l2") or "unknown").lower().strip()
        group = _classify_outerwear_group(l2)
        enriched.append({
            **item,
            "l2_lower": l2,
            "gate_group": group,  # "formal", "volume", "heavy", or None
            "is_gated": group is not None,
            "is_blazer": l2 in _FORMAL_OUTER_L2,  # backward-compat
        })
    return enriched


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def build_html(comparisons: List[Dict], total_time: float) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Aggregate stats per group
    gated_before = Counter()  # group -> count before
    gated_after = Counter()   # group -> count after
    by_treatment = Counter()  # "formal-filtered", "volume-filtered", etc.
    total_items_before = 0

    for comp in comparisons:
        before_items = comp.get("before_outerwear", [])
        after_items = comp.get("after_outerwear", [])
        total_items_before += len(before_items)
        fl = comp.get("formality_level", 2)
        src_occs = comp.get("source_occasions", [])
        src_occs_lower = {o.lower().strip() for o in src_occs}
        has_formal_occ = bool(src_occs_lower & _OUTERWEAR_GATE_OCCASIONS)

        for item in before_items:
            g = item.get("gate_group")
            if g:
                gated_before[g] += 1
        for item in after_items:
            g = item.get("gate_group")
            if g:
                gated_after[g] += 1

        # Track treatment by group
        for item in before_items:
            g = item.get("gate_group")
            if not g:
                continue
            if g == "formal":
                if fl <= 1 and not has_formal_occ:
                    by_treatment["formal-filtered"] += 1
                elif fl == 2 and not has_formal_occ:
                    by_treatment["formal-penalized"] += 1
                else:
                    by_treatment["formal-kept"] += 1
            elif g == "volume":
                if fl <= 1:
                    by_treatment["volume-filtered"] += 1
                elif fl <= 2:
                    by_treatment["volume-penalized"] += 1
                else:
                    by_treatment["volume-kept"] += 1
            elif g == "heavy":
                if fl <= 1 and not has_formal_occ:
                    by_treatment["heavy-penalized"] += 1
                else:
                    by_treatment["heavy-kept"] += 1

    total_gated_before = sum(gated_before.values())
    total_gated_after = sum(gated_after.values())
    total_removed = total_gated_before - total_gated_after

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Outerwear Appropriateness Gate Report -- {now}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 24px; max-width: 1600px; margin: 0 auto; }}
  h1 {{ color: #58a6ff; margin-bottom: 6px; font-size: 24px; }}
  h2 {{ color: #d2a8ff; margin: 20px 0 10px; font-size: 16px; }}
  .meta {{ color: #8b949e; font-size: 13px; margin-bottom: 6px; line-height: 1.6; }}
  .rule-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
               padding: 14px 18px; margin-bottom: 20px; font-size: 13px; line-height: 1.7; }}
  .rule-box b {{ color: #58a6ff; }}
  .rule-box .red {{ color: #f85149; }}
  .rule-box .yellow {{ color: #d29922; }}
  .rule-box .green {{ color: #7ee787; }}
  .rule-box .purple {{ color: #d2a8ff; }}

  .summary {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
  .stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                padding: 14px 18px; min-width: 150px; text-align: center; }}
  .stat-value {{ font-size: 28px; font-weight: 700; }}
  .stat-label {{ font-size: 10px; color: #8b949e; margin-top: 4px; }}
  .good {{ color: #7ee787; }}
  .warn {{ color: #d29922; }}
  .bad {{ color: #f85149; }}
  .neutral {{ color: #8b949e; }}
  .purple {{ color: #d2a8ff; }}

  .comparison {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px;
                 margin-bottom: 24px; overflow: hidden; }}
  .comp-header {{ padding: 12px 18px; border-bottom: 1px solid #30363d;
                   display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
  .comp-num {{ background: #238636; color: white; font-weight: 700; font-size: 14px;
               border-radius: 6px; padding: 3px 10px; }}
  .comp-title {{ color: #58a6ff; font-weight: 600; font-size: 14px; }}
  .comp-detail {{ font-size: 12px; color: #8b949e; }}
  .comp-detail span {{ margin-right: 14px; }}

  .formality-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
                      font-size: 11px; font-weight: 700; }}
  .f1 {{ background: #1f1f22; color: #8b949e; border: 1px solid #30363d; }}
  .f2 {{ background: #1c2333; color: #58a6ff; border: 1px solid #1f6feb; }}
  .f3 {{ background: #1f3d2a; color: #7ee787; border: 1px solid #238636; }}
  .f4 {{ background: #2d1f3d; color: #d2a8ff; border: 1px solid #8957e5; }}
  .f5 {{ background: #3d1f1f; color: #f78166; border: 1px solid #da3633; }}

  .treatment {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
                font-size: 11px; font-weight: 700; text-transform: uppercase; margin-left: 8px; }}
  .t-filtered {{ background: #3d1f1f; color: #f85149; }}
  .t-penalized {{ background: #3d2f1f; color: #d29922; }}
  .t-neutral {{ background: #1c2333; color: #58a6ff; }}
  .t-boosted {{ background: #1f3d2a; color: #7ee787; }}

  .comp-body {{ display: flex; gap: 0; }}
  .source-col {{ width: 170px; min-width: 170px; background: #1a1f2e;
                 border-right: 2px solid #238636; padding: 12px; text-align: center; }}
  .source-col img {{ width: 140px; height: 200px; object-fit: cover; border-radius: 8px;
                     border: 2px solid #238636; }}
  .source-label {{ color: #238636; font-weight: 700; font-size: 10px;
                   text-transform: uppercase; margin-bottom: 4px; }}
  .source-name {{ font-size: 10px; color: #c9d1d9; margin-top: 4px;
                  max-height: 26px; overflow: hidden; }}
  .source-brand {{ font-size: 10px; color: #8b949e; }}

  .sides {{ flex: 1; display: flex; gap: 0; }}
  .side {{ flex: 1; padding: 12px; }}
  .side-before {{ border-right: 1px dashed #30363d; }}
  .side-label {{ font-size: 11px; font-weight: 700; text-transform: uppercase;
                 padding: 3px 10px; border-radius: 4px; margin-bottom: 10px;
                 display: inline-block; }}
  .label-before {{ background: #30363d; color: #c9d1d9; }}
  .label-after {{ background: #1f3d2a; color: #7ee787; }}

  .items-row {{ display: flex; gap: 10px; flex-wrap: nowrap; overflow-x: auto; }}
  .item-card {{ width: 130px; min-width: 130px; text-align: center; position: relative; }}
  .item-card img {{ width: 120px; height: 170px; object-fit: cover; border-radius: 6px;
                    border: 2px solid #30363d; }}
  .item-card.gated-formal img {{ border-color: #f85149; }}
  .item-card.gated-volume img {{ border-color: #d2a8ff; }}
  .item-card.gated-heavy img {{ border-color: #d29922; }}
  .item-card.gated-kept img {{ border-color: #7ee787; }}
  .item-l2 {{ font-size: 10px; font-weight: 700; margin-top: 3px; }}
  .item-l2.is-gated {{ color: #f85149; }}
  .item-l2.not-gated {{ color: #d2a8ff; }}
  .item-name {{ font-size: 9px; color: #c9d1d9; max-height: 22px; overflow: hidden;
               line-height: 1.2; margin-top: 2px; }}
  .item-brand {{ font-size: 9px; color: #8b949e; }}
  .item-score {{ font-size: 10px; font-family: monospace; margin-top: 2px; }}
  .item-adj {{ font-size: 9px; font-family: monospace; color: #8b949e; }}
  .group-tag {{ position: absolute; top: 4px; right: 8px;
                color: white; font-size: 8px; font-weight: 700; padding: 1px 5px;
                border-radius: 3px; text-transform: uppercase; }}
  .tag-formal {{ background: #f85149; }}
  .tag-volume {{ background: #8957e5; }}
  .tag-heavy {{ background: #d29922; }}
  .no-items {{ color: #8b949e; font-style: italic; font-size: 12px; padding: 20px; }}

  .occasions-list {{ font-size: 11px; color: #8b949e; }}
  .occ-match {{ color: #7ee787; font-weight: 600; }}
</style>
</head>
<body>
<h1>Outerwear Appropriateness Gate Report</h1>
<p class="meta">Generated {now} | {len(comparisons)} products | {total_time:.0f}s total</p>

<div class="rule-box">
  <b>BEFORE:</b> All structured outerwear gets uniform <b>+0.03</b>; volume/heavy ungated.<br>
  <b>AFTER (v3.5):</b> Three-group outerwear gates:<br><br>
  <b>Group 1 -- Formal</b> (blazer, suit jacket):<br>
  &nbsp;&nbsp;<span class="red">Form. 1 + no occasion -> HARD FILTERED</span><br>
  &nbsp;&nbsp;<span class="yellow">Form. 2 + no occasion -> PENALIZED -0.10</span><br>
  &nbsp;&nbsp;Form. 2 + formal occasion -> NEUTRAL (0.00)<br>
  &nbsp;&nbsp;<span class="green">Form. 3+ -> BOOSTED +0.03</span><br><br>
  <b>Group 2 -- Volume</b> (poncho, cape, kimono, bolero, shrug):<br>
  &nbsp;&nbsp;<span class="red">Form. 1 -> HARD FILTERED (no occasion override)</span><br>
  &nbsp;&nbsp;<span class="yellow">Form. 2 -> PENALIZED -0.10 (no occasion override)</span><br>
  &nbsp;&nbsp;<span class="purple">Form. 3+ -> NEUTRAL (0.00, no bonus)</span><br><br>
  <b>Group 3 -- Heavy</b> (parka, coat):<br>
  &nbsp;&nbsp;<span class="yellow">Form. 1 + no occasion -> PENALIZED -0.10</span><br>
  &nbsp;&nbsp;<span class="green">Form. 1 + occasion / Form. 2+ -> BOOSTED +0.03</span>
</div>

<div class="summary">
  <div class="stat-card">
    <div class="stat-value neutral">{total_items_before}</div>
    <div class="stat-label">Total outerwear BEFORE</div>
  </div>
  <div class="stat-card">
    <div class="stat-value {"good" if total_removed > 0 else "neutral"}">{total_removed}</div>
    <div class="stat-label">Gated items removed</div>
  </div>
  <div class="stat-card">
    <div class="stat-value bad">{by_treatment.get("formal-filtered", 0)}</div>
    <div class="stat-label">Formal: hard-filtered</div>
  </div>
  <div class="stat-card">
    <div class="stat-value warn">{by_treatment.get("formal-penalized", 0)}</div>
    <div class="stat-label">Formal: penalized</div>
  </div>
  <div class="stat-card">
    <div class="stat-value bad">{by_treatment.get("volume-filtered", 0)}</div>
    <div class="stat-label">Volume: hard-filtered</div>
  </div>
  <div class="stat-card">
    <div class="stat-value warn">{by_treatment.get("volume-penalized", 0)}</div>
    <div class="stat-label">Volume: penalized</div>
  </div>
  <div class="stat-card">
    <div class="stat-value warn">{by_treatment.get("heavy-penalized", 0)}</div>
    <div class="stat-label">Heavy: penalized</div>
  </div>
  <div class="stat-card">
    <div class="stat-value good">{by_treatment.get("formal-kept", 0) + by_treatment.get("volume-kept", 0) + by_treatment.get("heavy-kept", 0)}</div>
    <div class="stat-label">Kept / boosted</div>
  </div>
</div>
"""

    # Sort comparisons by formality level (casual first — most impact)
    comparisons_sorted = sorted(comparisons, key=lambda c: c.get("formality_level", 2))

    for idx, comp in enumerate(comparisons_sorted):
        src = comp.get("source", {})
        fl = comp.get("formality_level", 2)
        fl_name = _FORMALITY_NAMES.get(fl, "Unknown")
        src_occs = comp.get("source_occasions", [])
        src_occs_lower = {o.lower().strip() for o in src_occs}
        has_formal_occ = bool(src_occs_lower & _OUTERWEAR_GATE_OCCASIONS)

        before_items = comp.get("before_outerwear", [])
        after_items = comp.get("after_outerwear", [])
        before_time = comp.get("before_time", 0)
        after_time = comp.get("after_time", 0)

        # Determine overall treatment badge for this formality level
        if fl <= 1 and not has_formal_occ:
            treatment = "formal+volume filtered"
            t_class = "t-filtered"
        elif fl == 2 and not has_formal_occ:
            treatment = "formal+volume penalized"
            t_class = "t-penalized"
        elif fl == 2 and has_formal_occ:
            treatment = "occasion override"
            t_class = "t-neutral"
        else:
            treatment = "all boosted/neutral"
            t_class = "t-boosted"

        # Format occasions with matches highlighted
        occ_html_parts = []
        for occ in src_occs:
            if occ.lower().strip() in _OUTERWEAR_GATE_OCCASIONS:
                occ_html_parts.append(f'<span class="occ-match">{escape(occ)}</span>')
            else:
                occ_html_parts.append(escape(occ))
        occ_html = ", ".join(occ_html_parts) if occ_html_parts else "<em>none</em>"

        gated_b = sum(1 for i in before_items if i.get("is_gated"))
        gated_a = sum(1 for i in after_items if i.get("is_gated"))

        html += f"""
<div class="comparison">
  <div class="comp-header">
    <span class="comp-num">#{idx+1}</span>
    <span class="comp-title">{escape((src.get('name') or '?')[:55])}</span>
    <span class="formality-badge f{fl}">{fl_name} ({fl})</span>
    <span class="treatment {t_class}">{treatment}</span>
    <div class="comp-detail">
      <span>{escape(src.get('brand') or '?')} -- ${src.get('price', 0):.0f}</span>
      <span>L2: {escape(src.get('gemini_category_l2') or src.get('category_l2') or '?')}</span>
      <span>Gated: {gated_b} -> {gated_a}</span>
      <span>{before_time:.1f}s / {after_time:.1f}s</span>
    </div>
    <div class="occasions-list">Occasions: {occ_html}</div>
  </div>
  <div class="comp-body">
    <div class="source-col">
      <div class="source-label">Source</div>
      <img src="{escape(src.get('image_url') or '')}" alt="source" loading="lazy"
           onerror="this.style.display='none'">
      <div class="source-name">{escape((src.get('name') or '?')[:40])}</div>
      <div class="source-brand">{escape(src.get('brand') or '?')}</div>
    </div>
    <div class="sides">
      <div class="side side-before">
        <div class="side-label label-before">BEFORE (no gates)</div>
        <div class="items-row">
"""
        if before_items:
            for item in before_items:
                html += _render_item(item, gate_active=False)
        else:
            html += '          <div class="no-items">No outerwear results</div>\n'

        html += f"""        </div>
      </div>
      <div class="side side-after">
        <div class="side-label label-after">AFTER (outerwear gates)</div>
        <div class="items-row">
"""
        if after_items:
            for item in after_items:
                html += _render_item(item, gate_active=True, fl=fl)
        else:
            html += '          <div class="no-items">No outerwear results</div>\n'

        html += """        </div>
      </div>
    </div>
  </div>
</div>
"""

    html += "</body></html>"
    return html


def _render_item(item: Dict, gate_active: bool = False, fl: int = 2) -> str:
    """Render a single outerwear item card."""
    img = item.get("image_url", "")
    l2 = item.get("l2_lower") or item.get("category_l2") or "?"
    group = item.get("gate_group")
    tattoo = item.get("tattoo_score", 0) or 0
    style_adj = item.get("style_adjustment", 0) or 0
    name = (item.get("name") or "?")[:35]
    brand = item.get("brand") or "?"

    l2_class = "is-gated" if group else "not-gated"
    card_class = "item-card"
    if group:
        if gate_active and fl >= _OUTERWEAR_MIN_FORMALITY:
            card_class += " gated-kept"
        else:
            card_class += f" gated-{group}"

    # Group tag (shown for all gated items)
    tag_html = ""
    if group:
        tag_label = {"formal": "FORMAL", "volume": "VOLUME", "heavy": "HEAVY"}
        tag_html = f'<span class="group-tag tag-{group}">{tag_label.get(group, group.upper())}</span>'

    score_color = "#7ee787" if tattoo >= 0.85 else "#d29922" if tattoo >= 0.75 else "#f85149"
    adj_str = f"{style_adj:+.2f}" if style_adj != 0 else "0.00"

    return f"""          <div class="{card_class}">
            {tag_html}
            <img src="{escape(img)}" alt="" loading="lazy" onerror="this.style.display='none'">
            <div class="item-l2 {l2_class}">{escape(str(l2))}</div>
            <div class="item-name">{escape(name)}</div>
            <div class="item-brand">{escape(brand)}</div>
            <div class="item-score" style="color:{score_color}">T={tattoo:.3f}</div>
            <div class="item-adj">style: {adj_str}</div>
          </div>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sb = get_supabase_client()
    engine = OutfitEngine(sb)

    logger.info("Fetching products across formality levels...")
    products = fetch_products_by_formality(sb, n=N_OUTFITS + 4)
    if not products:
        logger.error("No products found!")
        sys.exit(1)
    logger.info("Got %d candidate products", len(products))

    comparisons = []
    t_total = time.time()

    for i, prod in enumerate(products):
        if len(comparisons) >= N_OUTFITS:
            break

        pid = prod["sku_id"]
        logger.info("[%d/%d] Product %s (formality %d)...",
                    i + 1, len(products), pid[:12], prod["formality_level"])

        # Run BEFORE (outerwear gates disabled)
        logger.info("  BEFORE (no gates)...")
        before = run_outfit(engine, pid, disable_gates=True)
        before_time = before.get("_elapsed", 0)

        if before.get("error") and not before.get("recommendations"):
            logger.warning("  Skipping %s: %s", pid[:12], before.get("error"))
            continue

        # Run AFTER (outerwear gates active)
        logger.info("  AFTER (gates active)...")
        after = run_outfit(engine, pid, disable_gates=False)
        after_time = after.get("_elapsed", 0)

        before_ow = extract_outerwear(before)
        after_ow = extract_outerwear(after)

        src = before.get("source_product", {})
        src_occs = []
        # Try to get occasions from source profile
        src_profile = before.get("_source_profile")
        if src_profile and hasattr(src_profile, "occasions"):
            src_occs = src_profile.occasions or []
        elif src.get("occasions"):
            src_occs = src["occasions"]

        fl = prod["formality_level"]

        gated_b = sum(1 for x in before_ow if x.get("is_gated"))
        gated_a = sum(1 for x in after_ow if x.get("is_gated"))
        logger.info("  Gated items: %d -> %d | outerwear: %d -> %d",
                    gated_b, gated_a, len(before_ow), len(after_ow))

        comparisons.append({
            "source": src,
            "source_occasions": src_occs,
            "formality_level": fl,
            "before_outerwear": before_ow,
            "after_outerwear": after_ow,
            "before_time": before_time,
            "after_time": after_time,
        })

        # Incremental write
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
