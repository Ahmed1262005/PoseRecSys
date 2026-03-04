#!/usr/bin/env python3
"""
Gradio Outfit Builder Demo (TATTOO v3.1 + Vision Judge)
========================================================

Visual demo that lets you pick a product and builds a complete outfit
using the production OutfitEngine — TATTOO scoring, avoids system,
Vision Judge (gpt-4o-mini) with per-item veto + outfit ranking,
MMR diversity, and profile-aware boosts.

Usage:
    PYTHONPATH=src python scripts/experiments/outfit_builder_gradio.py
    -> Open http://localhost:7861
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src and experiments dir to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import gradio as gr
from supabase import create_client, Client

# Import real production engine
from services.outfit_engine import get_outfit_engine, COMPLEMENTARY_CATEGORIES
from config.database import get_supabase_client


# =============================================================================
# DATABASE
# =============================================================================

def get_supabase() -> Client:
    return get_supabase_client()


SUPABASE = get_supabase()
ENGINE = None


def _get_engine():
    """Lazy-init the outfit engine."""
    global ENGINE
    if ENGINE is None:
        ENGINE = get_outfit_engine()
    return ENGINE


# =============================================================================
# PRODUCT SEARCH
# =============================================================================

def search_products(query: str, category: str, brand: str, limit: int = 20) -> List[Dict]:
    """
    Search products by name, category, or brand.
    Uses Gemini category_l1 to verify category, not the unreliable DB field.
    """
    # Join with product_attributes to get Gemini's true category
    q = SUPABASE.table("products").select(
        "id, name, brand, category, price, primary_image_url, base_color, "
        "product_attributes!inner(category_l1)"
    ).eq("in_stock", True)

    if category and category != "All":
        # Filter by Gemini's category_l1 (the source of truth)
        q = q.eq("product_attributes.category_l1", category.capitalize())
    if brand and brand != "All":
        q = q.eq("brand", brand)
    if query:
        q = q.ilike("name", f"%{query}%")

    result = q.limit(limit).execute()

    # Flatten: replace DB category with Gemini's
    products = []
    for row in (result.data or []):
        attrs = row.pop("product_attributes", None) or {}
        gemini_l1 = attrs.get("category_l1", row.get("category", ""))
        row["category"] = gemini_l1.lower() if gemini_l1 else row.get("category", "")
        products.append(row)
    return products


def get_random_products(category: str = "All", limit: int = 12) -> List[Dict]:
    """
    Get random products, filtered by Gemini category_l1 (not unreliable DB field).
    """
    q = SUPABASE.table("products").select(
        "id, name, brand, category, price, primary_image_url, base_color, "
        "product_attributes!inner(category_l1)"
    ).eq("in_stock", True)

    if category and category != "All":
        q = q.eq("product_attributes.category_l1", category.capitalize())

    result = q.limit(limit).execute()

    products = []
    for row in (result.data or []):
        attrs = row.pop("product_attributes", None) or {}
        gemini_l1 = attrs.get("category_l1", row.get("category", ""))
        row["category"] = gemini_l1.lower() if gemini_l1 else row.get("category", "")
        products.append(row)
    return products


# =============================================================================
# OUTFIT BUILDING — uses real production OutfitEngine
# =============================================================================

def build_outfit(
    source_product_id: str,
    items_per_category: int = 6,
) -> Dict:
    """
    Build a complete outfit using the production OutfitEngine.
    Runs the full pipeline: TATTOO scoring, avoids, LLM judge, MMR diversity.
    """
    t0 = time.time()
    engine = _get_engine()
    result = engine.build_outfit(
        product_id=source_product_id,
        items_per_category=items_per_category,
    )
    elapsed = time.time() - t0
    result["_elapsed"] = elapsed
    return result


# =============================================================================
# HTML RENDERING
# =============================================================================

def render_product_card(item: Dict, is_source: bool = False) -> str:
    """Render a single product as an HTML card from engine response dict."""
    name = item.get("name", "Unknown")
    brand = item.get("brand", "")
    price = float(item.get("price", 0) or 0)
    image_url = item.get("image_url", "")
    category = item.get("category", "")
    rank = item.get("rank", 0)
    tattoo = item.get("tattoo_score", 0)
    dim_scores = item.get("dimension_scores", {})
    avoid_adj = item.get("avoid_adjustment", 0)
    vision_pass = item.get("vision_pass", True)

    card_cls = "outfit-card source-card" if is_source else "outfit-card"
    badge = ""
    if is_source:
        badge = '<span style="position:absolute;top:8px;left:8px;background:#6366f1;color:white;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:600;z-index:1;">SOURCE</span>'
    elif rank > 0:
        badge = f'<span style="position:absolute;top:8px;left:8px;background:#10b981;color:white;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;z-index:1;">#{rank}</span>'

    # Score line
    score_html = ""
    if not is_source and tattoo > 0:
        parts = [f'TATTOO: {tattoo:.3f}']
        if avoid_adj and avoid_adj != 0:
            parts.append(f'<span class="avoid">Avoid: {avoid_adj:+.3f}</span>')
        if not vision_pass:
            parts.append('<span style="color:#ef4444;">Vision: FAIL</span>')
        score_html = f'<div class="card-score">{" | ".join(parts)}</div>'

    # Dimension bars
    dim_html = ""
    if dim_scores and not is_source:
        bars = ""
        dim_labels = {"formality": "FRM", "occasion": "OCC", "color": "CLR", "style": "STY", "season": "SEA", "material": "MAT", "balance": "BAL", "pattern": "PAT", "price": "PRC"}
        for dim, label in dim_labels.items():
            val = dim_scores.get(dim, 0)
            pct = int(val * 100)
            bar_color = "#10b981" if val >= 0.7 else "#f59e0b" if val >= 0.4 else "#ef4444"
            bars += f'''
                <div style="display:flex;align-items:center;gap:4px;margin:1px 0;">
                    <span class="dim-label">{label}</span>
                    <div class="dim-track">
                        <div style="width:{pct}%;background:{bar_color};height:6px;border-radius:3px;"></div>
                    </div>
                    <span class="dim-value">{val:.2f}</span>
                </div>
            '''
        dim_html = f'<div style="margin-top:6px;">{bars}</div>'

    img_html = (
        f'<img src="{image_url}" class="card-img" loading="lazy" onerror="this.style.display=\'none\'" />'
        if image_url else
        '<div class="card-img-empty">No image</div>'
    )

    return f'''
    <div class="{card_cls}">
        {badge}
        {img_html}
        <div class="card-body">
            <div class="card-name">{name}</div>
            <div class="card-brand">{brand}</div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px;">
                <span class="card-price">${price:.0f}</span>
                <span class="card-category">{category}</span>
            </div>
            {score_html}
            {dim_html}
        </div>
    </div>
    '''


def render_outfit_html(result: Dict) -> str:
    """Render the full outfit as HTML from the real OutfitEngine response."""
    if "error" in result:
        return f'<div style="text-align:center;padding:40px;color:#ef4444;font-size:18px;">{result["error"]}</div>'

    source = result.get("source_product", {})
    recs = result.get("recommendations", {})
    status = result.get("status", "ok")
    scoring = result.get("scoring_info", {})
    outfit_info = result.get("complete_outfit", {})
    elapsed = result.get("_elapsed", 0)

    engine_label = scoring.get("engine", "?")
    has_judge = scoring.get("vision_judge", False) or scoring.get("llm_judge", False)
    has_outfit_ranking = scoring.get("outfit_ranking", False)
    fusion_label = scoring.get("fusion", "?")

    # Aesthetic profile from source
    ap = source.get("aesthetic_profile", {})

    # Build outfit sections + compute total
    sections_html = ""
    total_price = float(source.get("price", 0) or 0)
    n_pieces = 1

    for cat, cat_data in recs.items():
        items = cat_data.get("items", [])
        if not items:
            sections_html += f'''
            <div class="cat-section" style="margin-top:24px;">
                <h3>{cat} <span class="cat-count">(no matches)</span></h3>
            </div>
            '''
            continue

        top_pick = items[0]
        total_price += float(top_pick.get("price", 0) or 0)
        n_pieces += 1

        # Source card inline at the start of first category section
        cards_html = "".join(render_product_card(item) for item in items)

        sections_html += f'''
        <div class="cat-section" style="margin-top:28px;">
            <h3>{cat} <span class="cat-count">({len(items)} options)</span></h3>
            <div style="display:grid;grid-template-columns:repeat(auto-fill, minmax(220px, 1fr));gap:14px;">
                {cards_html}
            </div>
        </div>
        '''

    # Engine info badges
    judge_badge = (
        '<span style="background:#10b981;color:white;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;">Vision Judge ON</span>'
        if has_judge else
        '<span style="background:#6b7280;color:white;padding:2px 8px;border-radius:10px;font-size:11px;">Vision Judge OFF</span>'
    )
    if has_outfit_ranking:
        judge_badge += ' <span style="background:#8b5cf6;color:white;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;">Outfit Ranked</span>'

    # Status banner
    status_banner = ""
    if status == "set":
        status_banner = (
            '<div class="status-banner set">'
            '<strong>This is a set/co-ord product</strong> &mdash; it already covers multiple categories. '
            'Only outerwear suggestions are shown.</div>'
        )
    elif status == "activewear":
        status_banner = (
            '<div class="status-banner active">'
            '<strong>Activewear source</strong> &mdash; showing only activewear-compatible suggestions.</div>'
        )

    return f'''
    <div style="font-family:system-ui,-apple-system,sans-serif;max-width:1200px;margin:0 auto;">

        <!-- Engine info + source -->
        <div class="outfit-info">
            <h2>Your Outfit</h2>
            <p class="info-meta">
                Total: <strong>${total_price:.0f}</strong> &nbsp;|&nbsp; {n_pieces} pieces &nbsp;|&nbsp; {elapsed:.1f}s
            </p>
            <p class="info-engine">
                Engine: <strong>{engine_label}</strong> &nbsp;|&nbsp; {fusion_label} &nbsp;|&nbsp; {judge_badge}
            </p>
        </div>

        <!-- Source product -->
        <div style="margin-bottom:20px;">
            <div style="display:grid;grid-template-columns:minmax(220px, 280px);gap:14px;">
                {render_product_card(source, is_source=True)}
            </div>
        </div>

        <!-- Aesthetic profile -->
        <div class="aesthetic-profile">
            <h3>Source Aesthetic Profile</h3>
            <div style="display:grid;grid-template-columns:repeat(auto-fill, minmax(160px, 1fr));gap:6px;font-size:12px;">
                <span class="ap-pill" style="background:#e0e7ff;color:#4338ca;">Formality: {ap.get('formality', '?')}</span>
                <span class="ap-pill" style="background:#fce7f3;color:#be185d;">Color: {ap.get('color_family', '?')}</span>
                <span class="ap-pill" style="background:#d1fae5;color:#065f46;">Pattern: {ap.get('pattern', '?')}</span>
                <span class="ap-pill" style="background:#fef3c7;color:#92400e;">Fit: {ap.get('fit_type', '?')}</span>
                <span class="ap-pill" style="background:#ede9fe;color:#5b21b6;">Fabric: {ap.get('apparent_fabric', '?')}</span>
                <span class="ap-pill" style="background:#fdf4ff;color:#86198f;">Texture: {ap.get('texture', '?')}</span>
                <span class="ap-pill" style="background:#e0f2fe;color:#0369a1;">Silhouette: {ap.get('silhouette', '?')}</span>
                <span class="ap-pill" style="background:#f0f9ff;color:#0c4a6e;">Coverage: {ap.get('coverage_level', '?')}</span>
                <span class="ap-pill" style="background:#fffbeb;color:#78350f;">Length: {ap.get('length', '?')}</span>
            </div>
        </div>

        {status_banner}

        <!-- Category Sections -->
        {sections_html}
    </div>
    '''


def render_product_grid(products: List[Dict]) -> str:
    """Render a grid of products for selection."""
    if not products:
        return '<div style="text-align:center;padding:40px;color:#999;">No products found. Try a different search.</div>'

    cards = ""
    for p in products:
        pid = p.get("id", "")
        name = p.get("name", "Unknown")
        brand = p.get("brand", "")
        price = float(p.get("price", 0) or 0)
        img = p.get("primary_image_url", "")
        cat = p.get("category", "")

        img_html = (
            f'<img src="{img}" class="card-img" loading="lazy" onerror="this.style.display=\'none\'" />'
            if img else
            '<div class="card-img-empty">No image</div>'
        )

        cards += f'''
        <div class="grid-card" onclick="
            document.getElementById('selected-product-id').querySelector('textarea').value = '{pid}';
            document.getElementById('selected-product-id').querySelector('textarea').dispatchEvent(new Event('input', {{bubbles: true}}));
        ">
            {img_html}
            <div class="card-body">
                <div class="card-name">{name}</div>
                <div class="card-brand">{brand}</div>
                <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px;">
                    <span class="card-price">${price:.0f}</span>
                    <span class="card-category">{cat}</span>
                </div>
            </div>
        </div>
        '''

    return f'''
    <div style="display:grid;grid-template-columns:repeat(auto-fill, minmax(200px, 1fr));gap:12px;">
        {cards}
    </div>
    '''


# =============================================================================
# GRADIO EVENT HANDLERS
# =============================================================================

def handle_search(query: str, category: str, brand: str):
    """Search for products and return HTML grid."""
    products = search_products(query, category, brand, limit=16)
    return render_product_grid(products)


def handle_browse(category: str):
    """Browse random products in a category."""
    products = get_random_products(category, limit=16)
    return render_product_grid(products)


def handle_build_outfit(product_id: str, items_per_cat: int):
    """Build an outfit from the selected product."""
    if not product_id or not product_id.strip():
        return '<div style="text-align:center;padding:60px;color:#999;font-size:16px;">Enter a product ID or click a product above to select it.</div>'

    product_id = product_id.strip()
    result = build_outfit(product_id, items_per_category=int(items_per_cat))
    return render_outfit_html(result)


def handle_random_outfit(category: str, items_per_cat: int):
    """Pick a random product and build an outfit."""
    products = get_random_products(category if category != "All" else "tops", limit=5)

    # Find one with attributes
    for p in products:
        pid = p["id"]
        attrs = SUPABASE.table("product_attributes").select("sku_id").eq("sku_id", pid).limit(1).execute()
        if attrs.data:
            result = build_outfit(pid, items_per_category=int(items_per_cat))
            return render_outfit_html(result), pid

    # Fallback: just use first
    if products:
        result = build_outfit(products[0]["id"], items_per_category=int(items_per_cat))
        return render_outfit_html(result), products[0]["id"]

    return '<div style="text-align:center;padding:60px;color:#ef4444;">No products found.</div>', ""


# =============================================================================
# GRADIO UI
# =============================================================================

# =============================================================================
# SHOWCASE: AUTO-FIND PRODUCTS FOR EVERY CASE TYPE
# =============================================================================

def _find_product_for_case(case_type: str) -> Optional[str]:
    """Find a product ID for a specific showcase case type."""
    try:
        if case_type in ("Tops", "Bottoms", "Dresses", "Outerwear"):
            # Normal category: find a product with good Gemini attributes
            result = SUPABASE.table("products").select(
                "id, name, product_attributes!inner(category_l1, apparent_fabric, texture, silhouette)"
            ).eq("in_stock", True).eq(
                "product_attributes.category_l1", case_type
            ).not_.is_("product_attributes.apparent_fabric", "null").not_.is_(
                "product_attributes.texture", "null"
            ).limit(10).execute()
            if result.data:
                return result.data[0]["id"]
            # Fallback: without material requirements
            result = SUPABASE.table("products").select(
                "id, product_attributes!inner(category_l1)"
            ).eq("in_stock", True).eq(
                "product_attributes.category_l1", case_type
            ).limit(5).execute()
            if result.data:
                return result.data[0]["id"]

        elif case_type == "Set":
            # Find a co-ord or set product
            for term in ["%co-ord%", "%coord set%", "%two piece%", "%matching set%"]:
                result = SUPABASE.table("products").select(
                    "id, name"
                ).eq("in_stock", True).ilike("name", term).limit(3).execute()
                if result.data:
                    return result.data[0]["id"]

        elif case_type == "Activewear":
            result = SUPABASE.table("products").select(
                "id, product_attributes!inner(category_l1)"
            ).eq("in_stock", True).eq(
                "product_attributes.category_l1", "Activewear"
            ).limit(5).execute()
            if result.data:
                return result.data[0]["id"]

        elif case_type == "Intimates":
            result = SUPABASE.table("products").select(
                "id, product_attributes!inner(category_l1)"
            ).eq("in_stock", True).eq(
                "product_attributes.category_l1", "Intimates"
            ).limit(3).execute()
            if result.data:
                return result.data[0]["id"]

        elif case_type == "Swimwear":
            result = SUPABASE.table("products").select(
                "id, product_attributes!inner(category_l1)"
            ).eq("in_stock", True).eq(
                "product_attributes.category_l1", "Swimwear"
            ).limit(3).execute()
            if result.data:
                return result.data[0]["id"]

    except Exception as e:
        print(f"[Showcase] Error finding {case_type}: {e}")
    return None


def _render_showcase_section(
    case_label: str,
    case_description: str,
    expected_status: str,
    result: Dict,
    case_num: int,
) -> str:
    """Render a single showcase case as HTML."""
    # Status badge color
    status_colors = {
        "ok": ("#10b981", "#ecfdf5", "#065f46"),
        "set": ("#f59e0b", "#fffbeb", "#92400e"),
        "activewear": ("#3b82f6", "#dbeafe", "#1e40af"),
        "blocked": ("#ef4444", "#fef2f2", "#991b1b"),
    }

    if "error" in result:
        actual_status = "blocked"
    else:
        actual_status = result.get("status", "ok")

    bg, badge_bg, badge_fg = status_colors.get(actual_status, ("#6b7280", "#f9fafb", "#374151"))

    # Case header
    html = f'''
    <div style="border:2px solid {bg};border-radius:16px;margin-bottom:24px;overflow:hidden;">
        <div style="background:{badge_bg};padding:14px 20px;border-bottom:2px solid {bg};">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <span style="background:{bg};color:white;padding:3px 12px;border-radius:20px;font-size:12px;font-weight:700;margin-right:10px;">
                        CASE {case_num}
                    </span>
                    <span style="font-weight:700;font-size:16px;color:#1f2937;">{case_label}</span>
                </div>
                <span style="background:{badge_bg};border:1px solid {bg};color:{badge_fg};padding:3px 12px;border-radius:20px;font-size:12px;font-weight:600;">
                    Status: {actual_status.upper()}
                </span>
            </div>
            <div style="font-size:13px;color:#6b7280;margin-top:6px;">{case_description}</div>
        </div>
    '''

    if "error" in result:
        html += f'''
        <div style="padding:30px;text-align:center;">
            <div style="font-size:16px;color:#ef4444;font-weight:600;">{result["error"]}</div>
            <div style="font-size:13px;color:#9ca3af;margin-top:8px;">This is the expected behavior for blocked products.</div>
        </div>
        '''
    else:
        # Render the outfit
        html += f'<div style="padding:16px;">{render_outfit_html(result)}</div>'

    html += '</div>'
    return html


def handle_showcase():
    """Run through all 8 case types and build outfits for each."""
    import time as _time

    cases = [
        ("Tops", "Normal: Top as source", "Find complementary bottoms + outerwear", "ok"),
        ("Bottoms", "Normal: Bottom as source", "Find complementary tops + outerwear", "ok"),
        ("Dresses", "Normal: Dress as source", "Find complementary outerwear", "ok"),
        ("Outerwear", "Normal: Outerwear as source", "Find complementary tops + bottoms + dresses", "ok"),
        ("Set", "Edge: Co-ord / Set product", "Already covers multiple slots -- outerwear only", "set"),
        ("Activewear", "Edge: Activewear source", "Candidates filtered to workout/athletic items only", "activewear"),
        ("Intimates", "Blocked: Intimates", "Non-outfit category -- pipeline should refuse", "blocked"),
        ("Swimwear", "Blocked: Swimwear", "Non-outfit category -- pipeline should refuse", "blocked"),
    ]

    t0 = _time.time()
    sections = []
    found = 0
    total = len(cases)
    status_msg_parts = []

    for i, (case_type, label, desc, expected) in enumerate(cases, 1):
        pid = _find_product_for_case(case_type)
        if not pid:
            sections.append(f'''
            <div style="border:2px dashed #d1d5db;border-radius:16px;margin-bottom:24px;padding:30px;text-align:center;">
                <span style="background:#6b7280;color:white;padding:3px 12px;border-radius:20px;font-size:12px;font-weight:700;margin-right:8px;">
                    CASE {i}
                </span>
                <span style="font-weight:600;color:#6b7280;">{label}</span>
                <div style="color:#9ca3af;margin-top:8px;">No product found for this case type in the database.</div>
            </div>
            ''')
            status_msg_parts.append(f"Case {i} ({case_type}): SKIPPED")
            continue

        result = build_outfit(pid, items_per_category=4)
        sections.append(_render_showcase_section(label, desc, expected, result, i))

        actual = "blocked" if "error" in result else result.get("status", "ok")
        match = "OK" if actual == expected else f"MISMATCH (got {actual})"
        status_msg_parts.append(f"Case {i} ({case_type}): {match}")
        found += 1

    elapsed = _time.time() - t0

    # Build final HTML
    header = f'''
    <div style="font-family:system-ui,-apple-system,sans-serif;max-width:1200px;margin:0 auto;">
        <div class="showcase-header">
            <h2>TATTOO v3.1 + Vision Judge Showcase</h2>
            <p>
                {found}/{total} cases executed in {elapsed:.1f}s.
                Each outfit is scored across <strong>8 dimensions</strong>
                + vision judge (per-item veto + outfit ranking) + avoids system.
            </p>
        </div>
        {"".join(sections)}
    </div>
    '''

    status_summary = f"Done: {found}/{total} cases in {elapsed:.1f}s | " + " | ".join(status_msg_parts)
    return header, status_summary


CUSTOM_CSS = """
.main-container { max-width: 1200px; margin: 0 auto; }
.outfit-output { min-height: 400px; }

/* ── Product cards ───────────────────────────────────────────── */
.outfit-card {
    background: #ffffff;
    color: #1f2937;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    position: relative;
}
.outfit-card.source-card { border: 3px solid #6366f1; }
.outfit-card .card-img {
    width: 100%; height: 220px;
    object-fit: contain;
    background: #f9fafb;
    border-bottom: 1px solid #f3f4f6;
    display: block;
}
.outfit-card .card-img-empty {
    width: 100%; height: 220px;
    background: #f3f4f6;
    display: flex; align-items: center; justify-content: center;
    color: #aaa; font-size: 13px;
    border-bottom: 1px solid #f3f4f6;
}
.outfit-card .card-body       { padding: 10px 12px; }
.outfit-card .card-name       { font-weight: 600; font-size: 13px; line-height: 1.3; height: 34px; overflow: hidden; color: #1f2937; }
.outfit-card .card-brand      { font-size: 11px; color: #6b7280; margin-top: 2px; }
.outfit-card .card-price      { font-weight: 700; font-size: 14px; color: #1f2937; }
.outfit-card .card-category   { font-size: 10px; color: #9ca3af; text-transform: uppercase; }
.outfit-card .card-score      { font-size: 11px; color: #6366f1; font-weight: 600; margin-top: 4px; }
.outfit-card .card-score .avoid { color: #ef4444; }
.outfit-card .dim-label       { font-size: 9px; color: #6b7280; width: 24px; }
.outfit-card .dim-track       { flex: 1; background: #f3f4f6; border-radius: 3px; height: 6px; }
.outfit-card .dim-value       { font-size: 9px; color: #9ca3af; width: 22px; text-align: right; }

/* ── Product grid (selection panel) ──────────────────────────── */
.grid-card {
    background: #ffffff;
    color: #1f2937;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    overflow: hidden;
    cursor: pointer;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s, border-color 0.2s;
}
.grid-card:hover { border-color: #6366f1; box-shadow: 0 4px 12px rgba(99,102,241,0.15); }
.grid-card .card-img {
    width: 100%; height: 200px;
    object-fit: contain;
    background: #f9fafb;
    border-bottom: 1px solid #f3f4f6;
    display: block;
}
.grid-card .card-img-empty {
    width: 100%; height: 200px;
    background: #f3f4f6;
    display: flex; align-items: center; justify-content: center;
    color: #aaa; font-size: 13px;
    border-bottom: 1px solid #f3f4f6;
}
.grid-card .card-body     { padding: 10px 12px; }
.grid-card .card-name     { font-weight: 600; font-size: 12px; line-height: 1.3; height: 32px; overflow: hidden; color: #1f2937; }
.grid-card .card-brand    { font-size: 11px; color: #6b7280; }
.grid-card .card-price    { font-weight: 700; font-size: 14px; color: #1f2937; }
.grid-card .card-category { font-size: 10px; color: #9ca3af; text-transform: uppercase; }

/* ── Outfit info banner ──────────────────────────────────────── */
.outfit-info {
    background: linear-gradient(135deg, #f0f0ff, #fdf2f8);
    border-radius: 16px; padding: 20px 24px; margin-bottom: 24px;
}
.outfit-info h2 { margin: 0 0 4px; font-size: 20px; color: #1f2937; }
.outfit-info .info-meta { margin: 0 0 4px; font-size: 13px; color: #6b7280; }
.outfit-info .info-engine { margin: 0; font-size: 12px; color: #6b7280; }

/* ── Aesthetic profile ───────────────────────────────────────── */
.aesthetic-profile {
    background: #f8fafc;
    border-radius: 12px; padding: 16px; margin-bottom: 16px;
}
.aesthetic-profile h3 { margin: 0 0 8px; font-size: 15px; color: #374151; }
.aesthetic-profile .ap-pill {
    padding: 3px 10px; border-radius: 20px; font-size: 12px; display: inline-block;
}

/* ── Section headers ─────────────────────────────────────────── */
.cat-section h3 { margin: 0 0 12px; font-size: 16px; color: #374151; text-transform: capitalize; }
.cat-section h3 .cat-count { color: #9ca3af; font-size: 13px; font-weight: 400; }

/* ── Status banners ──────────────────────────────────────────── */
.status-banner { border-radius: 10px; padding: 12px 16px; margin-bottom: 16px; font-size: 14px; }
.status-banner.set     { background: #fef3c7; border: 1px solid #f59e0b; color: #92400e; }
.status-banner.active  { background: #dbeafe; border: 1px solid #3b82f6; color: #1e40af; }

/* ── Showcase ────────────────────────────────────────────────── */
.showcase-header {
    background: linear-gradient(135deg, #ede9fe, #dbeafe, #fdf2f8);
    border-radius: 16px; padding: 20px 24px; margin-bottom: 24px;
}
.showcase-header h2 { margin: 0 0 8px; color: #1f2937; }
.showcase-header p  { margin: 0; color: #6b7280; font-size: 14px; }

/* ══════════════════════════════════════════════════════════════
   DARK MODE — Gradio adds .dark to a parent container
   ══════════════════════════════════════════════════════════════ */

/* Cards */
.dark .outfit-card {
    background: #1e293b; color: #e2e8f0;
    border-color: #334155;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}
.dark .outfit-card.source-card { border-color: #818cf8; }
.dark .outfit-card .card-img       { background: #0f172a; border-color: #1e293b; }
.dark .outfit-card .card-img-empty { background: #0f172a; border-color: #1e293b; color: #475569; }
.dark .outfit-card .card-name      { color: #f1f5f9; }
.dark .outfit-card .card-brand     { color: #94a3b8; }
.dark .outfit-card .card-price     { color: #f1f5f9; }
.dark .outfit-card .card-category  { color: #64748b; }
.dark .outfit-card .card-score     { color: #a5b4fc; }
.dark .outfit-card .dim-label      { color: #94a3b8; }
.dark .outfit-card .dim-track      { background: #334155; }
.dark .outfit-card .dim-value      { color: #64748b; }

/* Grid cards */
.dark .grid-card {
    background: #1e293b; color: #e2e8f0;
    border-color: #334155;
}
.dark .grid-card:hover { border-color: #818cf8; box-shadow: 0 4px 12px rgba(129,140,248,0.2); }
.dark .grid-card .card-img       { background: #0f172a; border-color: #1e293b; }
.dark .grid-card .card-img-empty { background: #0f172a; border-color: #1e293b; color: #475569; }
.dark .grid-card .card-name      { color: #f1f5f9; }
.dark .grid-card .card-brand     { color: #94a3b8; }
.dark .grid-card .card-price     { color: #f1f5f9; }
.dark .grid-card .card-category  { color: #64748b; }

/* Info & profile */
.dark .outfit-info { background: linear-gradient(135deg, #1e1b4b, #172554, #3b0764); }
.dark .outfit-info h2 { color: #e2e8f0; }
.dark .outfit-info .info-meta, .dark .outfit-info .info-engine { color: #94a3b8; }

.dark .aesthetic-profile { background: #1e293b; }
.dark .aesthetic-profile h3 { color: #e2e8f0; }

/* Section headers */
.dark .cat-section h3 { color: #e2e8f0; }
.dark .cat-section h3 .cat-count { color: #64748b; }

/* Status banners */
.dark .status-banner.set    { background: #422006; border-color: #a16207; color: #fde68a; }
.dark .status-banner.active { background: #172554; border-color: #2563eb; color: #93c5fd; }

/* Showcase */
.dark .showcase-header { background: linear-gradient(135deg, #1e1b4b, #172554, #3b0764); }
.dark .showcase-header h2 { color: #e2e8f0; }
.dark .showcase-header p  { color: #94a3b8; }
"""

def build_app():
    with gr.Blocks(title="Outfit Builder - TATTOO v3.1 + Vision Judge") as app:
        gr.Markdown("""
        # Outfit Builder (TATTOO v3.1 + Vision Judge)
        **Production OutfitEngine** with 8-dimension TATTOO scoring, outfit avoids system,
        Vision Judge (gpt-4o-mini) with per-item veto + outfit-level ranking, MMR diversity,
        and profile-aware boosts.
        """)

        with gr.Tabs():
            # ── Tab 1: Build Outfit ──────────────────────────────────
            with gr.Tab("Build Outfit"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Find a Product")
                        search_query = gr.Textbox(label="Search by name", placeholder="e.g. floral midi dress")
                        with gr.Row():
                            search_cat = gr.Dropdown(
                                choices=["All", "tops", "bottoms", "outerwear"],
                                value="All", label="Category",
                            )
                            search_brand = gr.Dropdown(
                                choices=["All", "Boohoo", "Nasty Gal", "Missguided", "Forever 21",
                                         "Princess Polly", "Reformation", "Free People", "Alo Yoga",
                                         "Joe's Jeans", "Universal Standard", "J.Crew", "Old Navy",
                                         "Rag & Bone", "The Frankie Shop", "Club Monaco", "DL1961",
                                         "Scotch & Soda", "Abercrombie & Fitch", "Gap"],
                                value="All", label="Brand",
                            )
                        search_btn = gr.Button("Search", variant="primary")
                        browse_btn = gr.Button("Browse Random", variant="secondary")
                        product_grid = gr.HTML(value="<div style='text-align:center;padding:40px;color:#999;'>Search or browse to find products</div>")

                    with gr.Column(scale=2):
                        gr.Markdown("### Outfit")
                        with gr.Row():
                            product_id_input = gr.Textbox(
                                label="Product ID", placeholder="Paste product UUID or click a product",
                                elem_id="selected-product-id",
                            )
                            build_btn = gr.Button("Build Outfit", variant="primary", scale=0)

                        items_slider = gr.Slider(minimum=3, maximum=10, value=6, step=1, label="Items per category")

                        outfit_output = gr.HTML(
                            value="<div style='text-align:center;padding:80px;color:#999;font-size:16px;'>Select a product to build an outfit</div>",
                            elem_classes=["outfit-output"],
                        )

                # Events
                search_btn.click(handle_search, [search_query, search_cat, search_brand], [product_grid])
                search_query.submit(handle_search, [search_query, search_cat, search_brand], [product_grid])
                browse_btn.click(handle_browse, [search_cat], [product_grid])
                build_btn.click(handle_build_outfit, [product_id_input, items_slider], [outfit_output])

            # ── Tab 2: Random Outfits ────────────────────────────────
            with gr.Tab("Random Outfits"):
                gr.Markdown("### Generate random outfits across categories")
                with gr.Row():
                    rand_cat = gr.Dropdown(
                        choices=["tops", "bottoms", "outerwear"],
                        value="tops", label="Source category",
                    )
                    rand_items = gr.Slider(minimum=3, maximum=8, value=4, step=1, label="Items per category")
                    rand_btn = gr.Button("Generate Random Outfit", variant="primary")

                rand_pid = gr.Textbox(visible=False)
                rand_output = gr.HTML(
                    value="<div style='text-align:center;padding:80px;color:#999;'>Click Generate to build a random outfit</div>",
                )

                rand_btn.click(handle_random_outfit, [rand_cat, rand_items], [rand_output, rand_pid])

            # ── Tab 3: Showcase All Cases ────────────────────────────
            with gr.Tab("Showcase All Cases"):
                gr.Markdown("""
                ### TATTOO v3.1 + Vision Judge Showcase
                Automatically finds products for **every case type** (normal x4 categories,
                set/co-ord, activewear, blocked) and builds outfits using the full pipeline:
                8-dimension TATTOO scoring, avoids penalties, vision judge (per-item veto + outfit ranking), and MMR diversity.
                """)

                with gr.Row():
                    showcase_btn = gr.Button(
                        "Run Full Showcase (8 cases)", variant="primary", size="lg",
                    )
                    showcase_progress = gr.Textbox(
                        label="Status", value="Ready", interactive=False,
                    )
                showcase_output = gr.HTML(
                    value="<div style='text-align:center;padding:80px;color:#999;font-size:16px;'>Click the button to run through all clothing types and edge cases</div>",
                )

                showcase_btn.click(
                    handle_showcase,
                    inputs=[],
                    outputs=[showcase_output, showcase_progress],
                )

    return app


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Starting Outfit Builder Gradio demo...")
    print("Open http://localhost:7863 in your browser")
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False,
        css=CUSTOM_CSS,
    )
