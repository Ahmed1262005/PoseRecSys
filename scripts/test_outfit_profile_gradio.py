#!/usr/bin/env python3
"""
Gradio test UI for Profile-Aware Outfit Recommendations (v2.2).

Side-by-side comparison: baseline (no profile) vs personalized (with profile).
Exercises the cluster complement prompts + ProfileScorer integration.

Tabs:
  1. Complete the Fit — compare baseline vs personalized outfit
  2. Similar Items — compare baseline vs personalized similar items
  3. Profile Explorer — see cluster resolution and prompt injection

Usage:
    1. Start the API server (or just run this script directly — it calls the engine):
       PYTHONPATH=src python scripts/test_outfit_profile_gradio.py
    2. Open http://localhost:7863 in your browser
"""

import html
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gradio as gr
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from supabase import create_client

from services.outfit_engine import OutfitEngine, _get_cluster_prompts
from recs.brand_clusters import (
    BRAND_CLUSTER_MAP,
    CLUSTER_COMPLEMENT_PROMPTS,
    CLUSTER_TRAITS,
    PERSONA_TO_CLUSTERS,
)

# =============================================================================
# Supabase client
# =============================================================================

def _get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY required in .env")
    return create_client(url, key)


SUPABASE = _get_supabase()
ENGINE = OutfitEngine(SUPABASE)

# =============================================================================
# Constants
# =============================================================================

PERSONA_CHOICES = [
    "classic", "minimal", "elegant", "trendy", "casual",
    "streetwear", "sporty", "bohemian", "romantic", "edgy",
    "preppy", "glamorous", "athleisure", "vintage",
]

# Popular brands from the catalog (sorted)
DB_BRANDS = sorted([
    "Zara", "Mango", "H&M", "Reformation", "Ba&sh", "COS",
    "Anine Bing", "AllSaints", "Theory", "Vince", "Reiss",
    "Boohoo", "Missguided", "PrettyLittleThing", "Princess Polly",
    "Free People", "Urban Outfitters", "Levi's", "Gap",
    "Abercrombie & Fitch", "American Eagle Outfitters",
    "Nike", "Adidas", "Lululemon", "Alo Yoga",
    "J.Crew", "Ann Taylor", "Everlane", "Uniqlo",
    "Sandro", "Maje", "& Other Stories", "Aritzia",
    "Good American", "Agolde", "Citizens of Humanity",
    "The North Face", "Patagonia",
    "Alice + Olivia", "Zimmermann", "Staud",
    "Skims", "Diesel", "True Religion",
    "Forever 21", "Shein", "Fashion Nova", "Nasty Gal",
    "Oak + Fort", "Rails", "Club Monaco",
])

# Some good test product IDs (mix of categories)
SAMPLE_PRODUCTS = [
    ("Black Blazer (Outerwear)", "use your own product ID"),
    ("Floral Midi Dress", "use your own product ID"),
    ("Premium Denim Jeans", "use your own product ID"),
    ("Silk Camisole Top", "use your own product ID"),
]


# =============================================================================
# HTML rendering
# =============================================================================

CUSTOM_CSS = """
/* ── Result grid ────────────────────────────────────── */
.result-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
    padding: 8px;
}

/* ── Cards (results) ────────────────────────────────── */
.card {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    overflow: hidden;
    background: white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    position: relative;
    transition: box-shadow 0.2s, transform 0.15s;
}
.card:hover { box-shadow: 0 6px 16px rgba(0,0,0,0.12); transform: translateY(-2px); }
.card-personalized { border: 2px solid #8b5cf6; }
.card-new {
    border: 2px solid #10b981;
    box-shadow: 0 0 0 2px rgba(16,185,129,0.15);
}
.card img.card-img {
    width: 100%; height: 320px; object-fit: cover; object-position: top;
    background: #f8f9fa;
}
.card .card-img-placeholder {
    width: 100%; height: 320px;
    background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
    display: flex; align-items: center; justify-content: center;
    color: #9ca3af; font-size: 13px;
}
.card .card-body { padding: 10px 12px; }
.card .card-name {
    font-weight: 600; font-size: 13px; line-height: 1.3;
    height: 34px; overflow: hidden;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}
.card .card-brand {
    font-size: 11px; color: #6b7280; margin-top: 3px;
    font-weight: 500; text-transform: uppercase; letter-spacing: 0.3px;
}
.card .card-price-row {
    display: flex; justify-content: space-between; align-items: center; margin-top: 6px;
}
.card .card-price { font-weight: 700; font-size: 15px; color: #111; }
.card .card-cat { font-size: 10px; color: #9ca3af; text-transform: uppercase; }
.card .card-scores {
    font-size: 10px; color: #6366f1; font-weight: 600; margin-top: 6px;
    padding-top: 6px; border-top: 1px solid #f3f4f6;
}

/* ── Source hero card ───────────────────────────────── */
.hero-source {
    display: flex; gap: 24px; align-items: stretch;
    background: white; border: 2px solid #6366f1;
    border-radius: 16px; overflow: hidden;
    box-shadow: 0 4px 20px rgba(99,102,241,0.12);
    margin-bottom: 20px; max-width: 680px;
}
.hero-source .hero-img-wrap {
    flex: 0 0 280px; min-height: 400px; position: relative;
    overflow: hidden; background: #f8f9fa;
}
.hero-source .hero-img-wrap img {
    width: 100%; height: 100%; object-fit: cover; object-position: top;
}
.hero-source .hero-details {
    flex: 1; padding: 24px 24px 24px 0;
    display: flex; flex-direction: column; justify-content: center;
}
.hero-source .hero-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 11px; font-weight: 700; color: white;
    background: #6366f1; margin-bottom: 12px; width: fit-content;
    letter-spacing: 0.5px;
}
.hero-source .hero-name {
    font-size: 20px; font-weight: 700; color: #111;
    line-height: 1.3; margin-bottom: 6px;
}
.hero-source .hero-brand {
    font-size: 13px; color: #6b7280; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;
}
.hero-source .hero-price {
    font-size: 26px; font-weight: 800; color: #111; margin-bottom: 12px;
}
.hero-source .hero-meta {
    font-size: 12px; color: #9ca3af; margin-top: 4px;
}
.hero-source .hero-meta span {
    display: inline-block; background: #f3f4f6; padding: 3px 10px;
    border-radius: 6px; margin: 2px 4px 2px 0; font-weight: 500;
}
.hero-source .hero-pid {
    font-size: 10px; color: #c4b5fd; font-family: monospace;
    margin-top: 12px;
}

/* ── Badges ─────────────────────────────────────────── */
.badge {
    position: absolute; top: 8px; left: 8px;
    padding: 3px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 700; color: white;
    letter-spacing: 0.3px;
}
.badge-rank { background: #374151; }
.badge-profile { background: #8b5cf6; }
.badge-new {
    background: #10b981;
    animation: pulse-badge 1.5s ease-in-out infinite;
}
@keyframes pulse-badge {
    0%, 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.3); }
    50% { box-shadow: 0 0 0 4px rgba(16,185,129,0); }
}

/* ── Profile adjustment chips ───────────────────────── */
.profile-adj-pos { color: #059669; font-weight: 700; }
.profile-adj-neg { color: #dc2626; font-weight: 700; }
.adj-chip {
    display: inline-block; padding: 1px 7px; border-radius: 4px;
    font-size: 10px; font-weight: 700;
}
.adj-chip-pos { background: #ecfdf5; color: #059669; }
.adj-chip-neg { background: #fef2f2; color: #dc2626; }
.adj-chip-zero { background: #f3f4f6; color: #9ca3af; }

/* ── Dimension mini-bars ────────────────────────────── */
.dim-bar {
    display: flex; align-items: center; gap: 4px; margin: 1px 0;
}
.dim-bar .dim-label { font-size: 9px; color: #6b7280; width: 28px; }
.dim-bar .dim-track { flex: 1; background: #f3f4f6; border-radius: 3px; height: 5px; }
.dim-bar .dim-fill { height: 5px; border-radius: 3px; }
.dim-bar .dim-val { font-size: 9px; color: #9ca3af; width: 26px; text-align: right; }

/* ── Section and header ─────────────────────────────── */
.section-title {
    font-size: 15px; font-weight: 700; margin: 16px 0 8px 0;
    padding-bottom: 4px; border-bottom: 2px solid #e5e7eb;
    color: #374151;
}
.comparison-header {
    text-align: center; font-size: 13px; font-weight: 700;
    padding: 8px 12px; border-radius: 8px; margin-bottom: 10px;
}
.header-baseline { background: #f3f4f6; color: #374151; }
.header-personalized { background: #ede9fe; color: #6d28d9; }

/* ── Cluster / prompt badges ────────────────────────── */
.cluster-badge {
    display: inline-block; padding: 2px 8px; border-radius: 8px;
    font-size: 11px; font-weight: 600; margin: 2px;
    background: #ede9fe; color: #6d28d9;
}
.prompt-box {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 8px 12px; margin: 4px 0; font-size: 12px; font-family: monospace;
}
.prompt-cluster { border-left: 3px solid #8b5cf6; }
.prompt-source { border-left: 3px solid #6366f1; }

/* ── Info / warning boxes ───────────────────────────── */
.info-box {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px;
    padding: 12px; margin: 8px 0; font-size: 13px;
}
.warn-box {
    background: #fef3c7; border: 1px solid #fde68a; border-radius: 8px;
    padding: 12px; margin: 8px 0; font-size: 13px;
}
.diff-summary {
    background: #ede9fe; border: 1px solid #c4b5fd; border-radius: 8px;
    padding: 10px 14px; margin: 8px 0; font-size: 12px; color: #4c1d95;
}
"""

DIM_LABELS = {
    "occasion_formality": "OCC",
    "style": "STY",
    "fabric": "FAB",
    "silhouette": "SIL",
    "color": "CLR",
    "seasonality": "SEA",
    "pattern": "PAT",
    "price": "PRC",
}


def _dim_bars(dims: Dict[str, float]) -> str:
    bars = ""
    for dim_key, label in DIM_LABELS.items():
        val = dims.get(dim_key, 0)
        pct = int(val * 100)
        color = "#10b981" if val >= 0.7 else "#f59e0b" if val >= 0.4 else "#ef4444"
        bars += f'''
        <div class="dim-bar">
            <span class="dim-label">{label}</span>
            <div class="dim-track">
                <div class="dim-fill" style="width:{pct}%;background:{color};"></div>
            </div>
            <span class="dim-val">{val:.2f}</span>
        </div>'''
    return bars


def _profile_adj_html(adj: float) -> str:
    if adj == 0:
        return ""
    cls = "profile-adj-pos" if adj > 0 else "profile-adj-neg"
    sign = "+" if adj > 0 else ""
    return f' <span class="{cls}">[{sign}{adj:.3f}]</span>'


def render_card(
    item: Dict[str, Any],
    show_dims: bool = True,
    show_profile_adj: bool = False,
    is_new: bool = False,
) -> str:
    name = html.escape(str(item.get("name", "Unknown")))
    brand = html.escape(str(item.get("brand", "")))
    price = float(item.get("price", 0))
    img_url = item.get("image_url") or item.get("primary_image_url") or ""
    category = item.get("category", "")
    rank = item.get("rank", 0)
    tattoo = item.get("tattoo_score", 0)
    cosine = item.get("cosine_similarity", item.get("similarity", 0))
    compat = item.get("compatibility_score", 0)
    profile_adj = item.get("profile_adjustment", 0)
    dims = item.get("dimension_scores", {})
    pid = item.get("product_id", "")

    # CSS classes
    card_cls = "card"
    if is_new:
        card_cls += " card-new"
    elif profile_adj and abs(profile_adj) > 0.01:
        card_cls += " card-personalized"

    # Badge
    if is_new:
        badge = f'<span class="badge badge-new">NEW #{rank}</span>'
    elif rank > 0:
        badge = f'<span class="badge badge-rank">#{rank}</span>'
    else:
        badge = ""

    # Image with graceful fallback
    if img_url:
        img_html = (
            f'<img class="card-img" src="{img_url}" loading="lazy" '
            f'onerror="this.outerHTML=\'<div class=card-img-placeholder>Image unavailable</div>\'" />'
        )
    else:
        img_html = '<div class="card-img-placeholder">No image</div>'

    # Scores line
    scores_parts = []
    if tattoo:
        scores_parts.append(f"<b>{tattoo:.3f}</b>")
    if cosine:
        scores_parts.append(f"cos {cosine:.3f}")
    if compat:
        scores_parts.append(f"compat {compat:.3f}")
    scores_html = f'<div class="card-scores">{" &middot; ".join(scores_parts)}</div>' if scores_parts else ""

    # Profile adjustment chip
    adj_html = ""
    if show_profile_adj and profile_adj:
        if profile_adj > 0.005:
            chip_cls = "adj-chip adj-chip-pos"
        elif profile_adj < -0.005:
            chip_cls = "adj-chip adj-chip-neg"
        else:
            chip_cls = "adj-chip adj-chip-zero"
        adj_html = f'<span class="{chip_cls}">profile {profile_adj:+.3f}</span>'

    # Dimension bars
    dim_html = ""
    if show_dims and dims:
        dim_html = f'<div style="margin-top:6px;">{_dim_bars(dims)}</div>'

    return f'''
    <div class="{card_cls}">
        {badge}
        {img_html}
        <div class="card-body">
            <div class="card-name">{name}</div>
            <div class="card-brand">{brand}</div>
            <div class="card-price-row">
                <span class="card-price">${price:.0f}</span>
                <span class="card-cat">{category}</span>
            </div>
            {scores_html}
            {adj_html}
            {dim_html}
        </div>
    </div>'''


def render_source_card(result: Dict) -> str:
    """Render source product as a large hero card with image + details side-by-side."""
    src = result.get("source_product", {})
    if not src:
        return ""

    name = html.escape(str(src.get("name", "Unknown")))
    brand = html.escape(str(src.get("brand", "")))
    price = float(src.get("price", 0))
    img_url = src.get("image_url") or ""
    category = src.get("category", "")
    pid = src.get("product_id", "")
    aesthetic = src.get("aesthetic_profile", {})

    # Build meta tags from aesthetic profile
    meta_tags = []
    for key in ("style_tags", "pattern", "formality", "color_family", "fit_type"):
        val = aesthetic.get(key)
        if val:
            if isinstance(val, list):
                meta_tags.extend(str(v) for v in val[:3])
            elif str(val).lower() not in ("none", "n/a", ""):
                meta_tags.append(str(val))
    meta_html = "".join(
        f'<span>{html.escape(t)}</span>' for t in meta_tags[:6]
    )

    if img_url:
        img_html = (
            f'<img src="{img_url}" alt="{name}" '
            f'onerror="this.outerHTML=\'<div style=padding:40px;color:#aaa;text-align:center>Image unavailable</div>\'" />'
        )
    else:
        img_html = '<div style="padding:40px;color:#aaa;text-align:center;">No image</div>'

    return f'''
    <div class="hero-source">
        <div class="hero-img-wrap">
            {img_html}
        </div>
        <div class="hero-details">
            <div class="hero-badge">SOURCE PRODUCT</div>
            <div class="hero-name">{name}</div>
            <div class="hero-brand">{brand}</div>
            <div class="hero-price">${price:.0f}</div>
            <div class="hero-meta">{meta_html}</div>
            <div class="hero-meta"><span>{category}</span></div>
            <div class="hero-pid">{pid}</div>
        </div>
    </div>'''


def render_outfit_results(
    result: Dict,
    show_profile_adj: bool = False,
    baseline_ids: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Render complete-fit results as HTML.

    If baseline_ids is provided (mapping category -> list of product_ids),
    items NOT in the baseline are marked as NEW.
    """
    if not result or result.get("error"):
        err = result.get("error", "No results") if result else "No results"
        return f'<div class="warn-box">{html.escape(str(err))}</div>'

    scoring_info = result.get("scoring_info", {})
    personalized = scoring_info.get("personalized", False)
    clusters = scoring_info.get("clusters", [])
    engine_ver = scoring_info.get("engine", "?")

    # Header
    header_cls = "header-personalized" if personalized else "header-baseline"
    label = "PERSONALIZED" if personalized else "BASELINE"
    cluster_text = f" | Clusters: {', '.join(clusters)}" if clusters else ""
    header = f'<div class="comparison-header {header_cls}">{label} ({engine_ver}){cluster_text}</div>'

    # Diff summary (only for personalized when we have baseline to compare)
    diff_html = ""
    total_new = 0

    sections = ""
    recs = result.get("recommendations", {})
    for cat_name, cat_data in recs.items():
        items = cat_data.get("items", [])
        if not items:
            continue
        sections += f'<div class="section-title">{cat_name.title()} ({len(items)} items)</div>'
        sections += '<div class="result-grid">'
        b_ids = set(baseline_ids.get(cat_name, [])) if baseline_ids else set()
        for item in items:
            is_new = bool(b_ids and item.get("product_id") not in b_ids)
            if is_new:
                total_new += 1
            sections += render_card(item, show_profile_adj=show_profile_adj, is_new=is_new)
        sections += '</div>'

    if baseline_ids and total_new > 0:
        diff_html = (
            f'<div class="diff-summary">'
            f'Profile changed the results: <b>{total_new} new item{"s" if total_new != 1 else ""}</b> '
            f'surfaced by personalization (marked with green NEW badge)'
            f'</div>'
        )

    return header + diff_html + sections


def render_similar_results(
    result: Dict,
    show_profile_adj: bool = False,
    baseline_ids: Optional[List[str]] = None,
) -> str:
    """Render similar items results as HTML."""
    if not result:
        return '<div class="warn-box">No results</div>'

    personalized = result.get("personalized", False)
    items = result.get("results", [])
    b_ids = set(baseline_ids) if baseline_ids else set()

    header_cls = "header-personalized" if personalized else "header-baseline"
    label = "PERSONALIZED" if personalized else "BASELINE"
    header = f'<div class="comparison-header {header_cls}">{label} ({len(items)} items)</div>'

    total_new = 0
    grid = '<div class="result-grid">'
    for item in items:
        is_new = bool(b_ids and item.get("product_id") not in b_ids)
        if is_new:
            total_new += 1
        grid += render_card(item, show_profile_adj=show_profile_adj, is_new=is_new)
    grid += '</div>'

    diff_html = ""
    if b_ids and total_new > 0:
        diff_html = (
            f'<div class="diff-summary">'
            f'Profile changed the results: <b>{total_new} new item{"s" if total_new != 1 else ""}</b> '
            f'surfaced by personalization (marked with green NEW badge)'
            f'</div>'
        )

    return header + diff_html + grid


def render_profile_summary(
    brands: List[str], personas: List[str], clusters: List[str],
) -> str:
    """Render profile summary with cluster info."""
    parts = []

    if brands:
        brand_badges = " ".join(
            f'<span style="display:inline-block;padding:2px 8px;border-radius:8px;font-size:11px;background:#dcfce7;color:#166534;margin:2px;">{html.escape(b)}</span>'
            for b in brands
        )
        parts.append(f"<b>Brands:</b> {brand_badges}")

    if personas:
        persona_badges = " ".join(
            f'<span style="display:inline-block;padding:2px 8px;border-radius:8px;font-size:11px;background:#fef3c7;color:#92400e;margin:2px;">{html.escape(p)}</span>'
            for p in personas
        )
        parts.append(f"<b>Style:</b> {persona_badges}")

    if clusters:
        cluster_html = ""
        for cid in clusters:
            traits = CLUSTER_TRAITS.get(cid)
            name = traits.name if traits else cid
            cluster_html += f'<span class="cluster-badge">{cid}: {html.escape(name)}</span> '
        parts.append(f"<b>Clusters:</b> {cluster_html}")

    return '<div class="info-box">' + "<br>".join(parts) + '</div>' if parts else ""


# =============================================================================
# Fake profile builder (simulates what _load_user_profile returns)
# =============================================================================

def build_fake_profile(
    brands: List[str],
    personas: List[str],
    min_price: float = 0,
    max_price: float = 500,
) -> dict:
    """Build a profile dict matching the format _load_user_profile returns."""
    return {
        "preferred_brands": brands or [],
        "brand_openness": "open",
        "style_persona": personas or [],
        "preferred_fits": [],
        "fit_category_mapping": [],
        "preferred_sleeves": [],
        "sleeve_category_mapping": [],
        "preferred_lengths": [],
        "length_category_mapping": [],
        "preferred_lengths_dresses": [],
        "preferred_necklines": [],
        "preferred_rises": [],
        "top_types": [],
        "bottom_types": [],
        "dress_types": [],
        "outerwear_types": [],
        "patterns_liked": [],
        "patterns_avoided": [],
        "occasions": [],
        "colors_to_avoid": [],
        "styles_to_avoid": [],
        "no_crop": False,
        "no_revealing": False,
        "no_sleeveless": False,
        "no_deep_necklines": False,
        "no_tanks": False,
        "global_min_price": min_price if min_price > 0 else None,
        "global_max_price": max_price if max_price < 500 else None,
        "birthdate": None,
        "taste_vector": None,
    }


# =============================================================================
# Core logic: run comparison
# =============================================================================

def run_outfit_comparison(
    product_id: str,
    brands: List[str],
    personas: List[str],
    min_price: float,
    max_price: float,
    items_per_cat: int,
) -> Tuple[str, str, str]:
    """Run build_outfit with and without profile, return HTML for both + timing."""
    product_id = product_id.strip()
    if not product_id:
        return '<div class="warn-box">Enter a product ID</div>', "", ""

    profile = build_fake_profile(brands, personas, min_price, max_price)
    clusters = ENGINE._resolve_user_clusters(profile)
    profile_summary = render_profile_summary(brands, personas, clusters)

    # Baseline (no profile) — inject None user_id so _load_user_profile returns None
    t0 = time.time()
    baseline = ENGINE.build_outfit(
        product_id=product_id,
        items_per_category=items_per_cat,
        user_id=None,
    )
    t_baseline = time.time() - t0

    # Personalized — inject profile directly into engine cache so it's used
    fake_user_id = "__gradio_test_user__"
    import time as _t
    with ENGINE._profile_cache_lock:
        ENGINE._profile_cache[fake_user_id] = (profile, _t.monotonic())

    t1 = time.time()
    personalized = ENGINE.build_outfit(
        product_id=product_id,
        items_per_category=items_per_cat,
        user_id=fake_user_id,
    )
    t_personalized = time.time() - t1

    # Source card (shared — shown in info panel, not duplicated in each column)
    source_html = render_source_card(baseline or personalized)

    # Extract baseline product IDs per category for diff highlighting
    baseline_ids: Dict[str, List[str]] = {}
    for cat_name, cat_data in (baseline or {}).get("recommendations", {}).items():
        baseline_ids[cat_name] = [
            it.get("product_id", "") for it in cat_data.get("items", [])
        ]

    # Timing info
    timing = (
        f'<div class="info-box">'
        f'Baseline: {t_baseline:.1f}s | Personalized: {t_personalized:.1f}s'
        f'</div>'
    )

    baseline_html = render_outfit_results(baseline, show_profile_adj=False)
    personal_html = render_outfit_results(
        personalized, show_profile_adj=True, baseline_ids=baseline_ids,
    )

    return baseline_html, personal_html, source_html + profile_summary + timing


def run_similar_comparison(
    product_id: str,
    brands: List[str],
    personas: List[str],
    min_price: float,
    max_price: float,
    limit: int,
) -> Tuple[str, str, str]:
    """Run get_similar_scored with and without profile."""
    product_id = product_id.strip()
    if not product_id:
        return '<div class="warn-box">Enter a product ID</div>', "", ""

    profile = build_fake_profile(brands, personas, min_price, max_price)
    clusters = ENGINE._resolve_user_clusters(profile)
    profile_summary = render_profile_summary(brands, personas, clusters)

    # Baseline
    t0 = time.time()
    baseline = ENGINE.get_similar_scored(
        product_id=product_id,
        limit=limit,
        user_id=None,
    )
    t_baseline = time.time() - t0

    # Personalized
    fake_user_id = "__gradio_test_user__"
    import time as _t
    with ENGINE._profile_cache_lock:
        ENGINE._profile_cache[fake_user_id] = (profile, _t.monotonic())

    t1 = time.time()
    personalized = ENGINE.get_similar_scored(
        product_id=product_id,
        limit=limit,
        user_id=fake_user_id,
    )
    t_personalized = time.time() - t1

    # Extract baseline IDs for diff highlighting
    baseline_pids = [
        it.get("product_id", "") for it in (baseline or {}).get("results", [])
    ]

    timing = (
        f'<div class="info-box">'
        f'Baseline: {t_baseline:.1f}s | Personalized: {t_personalized:.1f}s'
        f'</div>'
    )

    # Source card from the baseline result
    source_html = ""
    if baseline and baseline.get("product_id"):
        # Fetch source info for hero card (similar items doesn't include source_product)
        try:
            src = ENGINE._fetch_product_with_attrs(baseline["product_id"])
            if src:
                source_html = render_source_card({
                    "source_product": {
                        "name": src.name, "brand": src.brand,
                        "price": src.price, "image_url": src.image_url,
                        "category": src.gemini_category_l1 or src.category,
                        "product_id": src.product_id,
                        "aesthetic_profile": src.to_api_dict(),
                    }
                })
        except Exception:
            pass

    baseline_html = render_similar_results(baseline, show_profile_adj=False)
    personal_html = render_similar_results(
        personalized, show_profile_adj=True, baseline_ids=baseline_pids,
    )

    return baseline_html, personal_html, source_html + profile_summary + timing


def explore_clusters(
    brands: List[str],
    personas: List[str],
    target_category: str,
) -> str:
    """Show cluster resolution and the prompts that would be injected."""
    profile = build_fake_profile(brands, personas)
    clusters = ENGINE._resolve_user_clusters(profile)

    parts = []
    parts.append(render_profile_summary(brands, personas, clusters))

    if not clusters:
        parts.append('<div class="warn-box">No clusters resolved. Select brands or style personas above.</div>')
        return "\n".join(parts)

    # Show cluster details
    for cid in clusters:
        traits = CLUSTER_TRAITS.get(cid)
        if traits:
            parts.append(
                f'<div style="margin:8px 0;padding:10px;border:1px solid #e2e8f0;border-radius:8px;">'
                f'<b>Cluster {cid}: {html.escape(traits.name)}</b><br>'
                f'<span style="font-size:12px;color:#666;">{html.escape(traits.description)}</span><br>'
                f'<span style="font-size:11px;">Style: {", ".join(traits.style_tags)} | '
                f'Palette: {traits.palette} | Formality: {traits.formality} | '
                f'Price: {traits.price_tier} (${traits.typical_price_range[0]}-${traits.typical_price_range[1]})</span>'
                f'</div>'
            )

    # Show prompts per category
    target = target_category.lower() if target_category else "tops"
    parts.append(f'<div class="section-title">Prompts for target: {target}</div>')

    # Source-derived prompts (example — would need a real source product)
    parts.append('<div style="font-size:12px;color:#666;margin-bottom:4px;">Source-derived prompts (3):</div>')
    parts.append('<div class="prompt-box prompt-source"><i>Generated from source product attributes at runtime</i></div>')

    # Cluster prompts
    cluster_prompts = _get_cluster_prompts(clusters, target)
    parts.append(f'<div style="font-size:12px;color:#6d28d9;margin:8px 0 4px;">Cluster prompts ({len(cluster_prompts)}):</div>')
    for i, prompt in enumerate(cluster_prompts):
        cid = clusters[i] if i < len(clusters) else "?"
        parts.append(f'<div class="prompt-box prompt-cluster"><b>[{cid}]</b> {html.escape(prompt)}</div>')

    # Show all prompts for this cluster x all categories
    parts.append(f'<div class="section-title">All cluster prompts (all categories)</div>')
    for cid in clusters:
        cat_prompts = CLUSTER_COMPLEMENT_PROMPTS.get(cid, {})
        for cat, prompts_list in cat_prompts.items():
            for j, p in enumerate(prompts_list):
                highlight = " style='background:#ede9fe;'" if cat == target else ""
                parts.append(f'<div class="prompt-box prompt-cluster"{highlight}><b>[{cid}/{cat}/{j}]</b> {html.escape(p)}</div>')

    return "\n".join(parts)


def get_random_product(category_filter: str = "") -> str:
    """Fetch a random in-stock product that has Gemini attributes."""
    try:
        q = SUPABASE.table("product_attributes").select(
            "sku_id, category_l1"
        )
        if category_filter:
            q = q.eq("category_l1", category_filter)
        # Grab a batch and pick randomly
        result = q.limit(200).execute()
        if not result.data:
            return ""
        import random
        row = random.choice(result.data)
        return str(row["sku_id"])
    except Exception as e:
        print(f"Random product error: {e}")
        return ""


def search_products(query: str) -> str:
    """Quick product search for finding product IDs to test with."""
    if not query or len(query) < 2:
        return '<div class="warn-box">Enter at least 2 characters to search</div>'

    try:
        result = SUPABASE.table("products").select(
            "id, name, brand, category, price, primary_image_url"
        ).ilike("name", f"%{query}%").eq(
            "in_stock", True
        ).limit(12).execute()

        if not result.data:
            return '<div class="warn-box">No products found</div>'

        html_out = '<div class="result-grid">'
        for p in result.data:
            pid = p.get("id", "")
            html_out += f'''
            <div class="card" style="cursor:pointer;" onclick="navigator.clipboard.writeText('{pid}')">
                <span class="badge" style="background:#374151;font-size:9px;">CLICK TO COPY ID</span>
                {f'<img class="card-img" src="{p.get("primary_image_url", "")}" loading="lazy" onerror="this.style.display=&apos;none&apos;" />' if p.get("primary_image_url") else ""}
                <div class="card-body">
                    <div class="card-name">{html.escape(p.get("name", ""))}</div>
                    <div class="card-brand">{html.escape(p.get("brand", ""))}</div>
                    <div class="card-price-row">
                        <span class="card-price">${float(p.get("price", 0)):.0f}</span>
                        <span class="card-cat">{p.get("category", "")}</span>
                    </div>
                    <div style="font-size:9px;color:#666;font-family:monospace;margin-top:4px;">{pid}</div>
                </div>
            </div>'''
        html_out += '</div>'
        return html_out

    except Exception as e:
        return f'<div class="warn-box">Search error: {html.escape(str(e))}</div>'


# =============================================================================
# Gradio UI
# =============================================================================

def _random_and_compare_outfit(
    category: str, brands: List[str], personas: List[str],
    min_price: float, max_price: float, items_per_cat: int,
) -> Tuple[str, str, str, str]:
    """Pick a random product, then run outfit comparison."""
    pid = get_random_product(category)
    if not pid:
        msg = '<div class="warn-box">Could not find a random product. Try a different category.</div>'
        return pid, msg, "", ""
    b, p, info = run_outfit_comparison(pid, brands, personas, min_price, max_price, items_per_cat)
    return pid, b, p, info


def _random_and_compare_similar(
    category: str, brands: List[str], personas: List[str],
    min_price: float, max_price: float, limit: int,
) -> Tuple[str, str, str, str]:
    """Pick a random product, then run similar comparison."""
    pid = get_random_product(category)
    if not pid:
        msg = '<div class="warn-box">Could not find a random product. Try a different category.</div>'
        return pid, msg, "", ""
    b, p, info = run_similar_comparison(pid, brands, personas, min_price, max_price, limit)
    return pid, b, p, info


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Outfit Engine v2.2 - Profile Testing") as demo:
        gr.Markdown("# Outfit Engine v2.2 - Profile-Aware Testing")
        gr.Markdown(
            "Side-by-side comparison: **Baseline** (no profile) vs **Personalized** "
            "(with cluster prompts + ProfileScorer). "
            "Select brands/personas, pick a category, hit **Random Product & Compare**."
        )

        # ---- Shared profile controls ----
        with gr.Accordion("User Profile", open=True):
            with gr.Row():
                brand_input = gr.Dropdown(
                    choices=DB_BRANDS,
                    value=["Alo Yoga", "Ann Taylor", "Aritzia", "Everlane"],
                    multiselect=True,
                    label="Preferred Brands",
                    info="Select brands to determine style clusters",
                )
                persona_input = gr.Dropdown(
                    choices=PERSONA_CHOICES,
                    value=["classic", "minimal"],
                    multiselect=True,
                    label="Style Personas (fallback if no brands)",
                    info="Used only if no brands are selected",
                )
            with gr.Row():
                min_price = gr.Slider(0, 500, value=0, step=10, label="Min Price ($)")
                max_price = gr.Slider(0, 500, value=500, step=10, label="Max Price ($)")

        # ---- Product search helper ----
        with gr.Accordion("Find Product IDs", open=False):
            with gr.Row():
                search_query = gr.Textbox(label="Search by name", placeholder="e.g. blazer, midi dress, jeans...")
                search_btn = gr.Button("Search", size="sm")
            search_results = gr.HTML()
            search_btn.click(search_products, inputs=[search_query], outputs=[search_results])

        # ---- Tabs ----
        with gr.Tabs():

            # === Tab 1: Complete the Fit ===
            with gr.TabItem("Complete the Fit"):
                with gr.Row():
                    outfit_cat = gr.Dropdown(
                        choices=["", "Tops", "Bottoms", "Dresses", "Outerwear"],
                        value="",
                        label="Random from category (blank = any)",
                    )
                    outfit_pid = gr.Textbox(
                        label="Product ID (auto-filled or paste)",
                        placeholder="Click Random or paste a UUID",
                    )
                    outfit_n = gr.Slider(2, 8, value=4, step=1, label="Items/cat")
                with gr.Row():
                    outfit_random_btn = gr.Button("Random Product & Compare", variant="primary", scale=2)
                    outfit_btn = gr.Button("Compare (use current ID)", variant="secondary", scale=1)

                outfit_info = gr.HTML()

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Baseline (no profile)")
                        outfit_baseline = gr.HTML()
                    with gr.Column():
                        gr.Markdown("### Personalized (with profile)")
                        outfit_personal = gr.HTML()

                outfit_random_btn.click(
                    _random_and_compare_outfit,
                    inputs=[outfit_cat, brand_input, persona_input, min_price, max_price, outfit_n],
                    outputs=[outfit_pid, outfit_baseline, outfit_personal, outfit_info],
                )
                outfit_btn.click(
                    run_outfit_comparison,
                    inputs=[outfit_pid, brand_input, persona_input, min_price, max_price, outfit_n],
                    outputs=[outfit_baseline, outfit_personal, outfit_info],
                )

            # === Tab 2: Similar Items ===
            with gr.TabItem("Similar Items"):
                with gr.Row():
                    similar_cat = gr.Dropdown(
                        choices=["", "Tops", "Bottoms", "Dresses", "Outerwear"],
                        value="",
                        label="Random from category (blank = any)",
                    )
                    similar_pid = gr.Textbox(
                        label="Product ID (auto-filled or paste)",
                        placeholder="Click Random or paste a UUID",
                    )
                    similar_n = gr.Slider(5, 30, value=12, step=1, label="Results")
                with gr.Row():
                    similar_random_btn = gr.Button("Random Product & Compare", variant="primary", scale=2)
                    similar_btn = gr.Button("Compare (use current ID)", variant="secondary", scale=1)

                similar_info = gr.HTML()

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Baseline (cosine only)")
                        similar_baseline = gr.HTML()
                    with gr.Column():
                        gr.Markdown("### Personalized (cosine + profile)")
                        similar_personal = gr.HTML()

                similar_random_btn.click(
                    _random_and_compare_similar,
                    inputs=[similar_cat, brand_input, persona_input, min_price, max_price, similar_n],
                    outputs=[similar_pid, similar_baseline, similar_personal, similar_info],
                )
                similar_btn.click(
                    run_similar_comparison,
                    inputs=[similar_pid, brand_input, persona_input, min_price, max_price, similar_n],
                    outputs=[similar_baseline, similar_personal, similar_info],
                )

            # === Tab 3: Profile Explorer ===
            with gr.TabItem("Profile Explorer"):
                gr.Markdown(
                    "Explore how brands/personas resolve to clusters, "
                    "and see the premade FashionCLIP prompts that get injected."
                )
                with gr.Row():
                    explore_cat = gr.Dropdown(
                        choices=["tops", "bottoms", "dresses", "outerwear"],
                        value="tops",
                        label="Target category",
                    )
                    explore_btn = gr.Button("Explore", variant="primary")

                explore_html = gr.HTML()
                explore_btn.click(
                    explore_clusters,
                    inputs=[brand_input, persona_input, explore_cat],
                    outputs=[explore_html],
                )

    return demo


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False,
        css=CUSTOM_CSS,
    )
