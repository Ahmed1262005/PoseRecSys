#!/usr/bin/env python3
"""
Fashion Discovery Demo
======================

Each product is a "mega card" showing the product detail, gallery,
similar items, and complete-the-fit items — all pre-loaded.

Recommendations loaded in parallel via ThreadPoolExecutor.

Run:
    PYTHONPATH=src python scripts/experiments/fashion_discovery_demo.py
    -> http://localhost:7863
"""

import html as _html
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.database import get_supabase_client
from services.outfit_engine import get_outfit_engine

logger = logging.getLogger(__name__)

# ===========================================================================
# CONSTANTS
# ===========================================================================

CATEGORIES = ["Tops", "Bottoms", "Dresses", "Outerwear"]
COLOR_FAMILIES = [
    "Blacks", "Whites", "Greys", "Browns", "Beiges", "Creams",
    "Reds", "Pinks", "Oranges", "Yellows", "Greens",
    "Blues", "Purples", "Metallics", "Multi",
]
PATTERNS = [
    "Solid", "Striped", "Floral", "Plaid/Check", "Animal Print",
    "Polka Dot", "Geometric", "Abstract", "Tie-Dye", "Lace",
]

PRODUCT_SELECT = (
    "id, name, brand, price, original_price, primary_image_url, "
    "gallery_images, source_url, base_color, category"
)
ATTRS_SELECT = (
    "sku_id, category_l1, category_l2, color_family, primary_color, "
    "pattern, formality, occasions, style_tags, seasons, fit_type, "
    "silhouette, apparent_fabric, texture, coverage_level, construction"
)

SIM_LIMIT = 4
FIT_LIMIT = 4
MAX_WORKERS = 6

DIM_LABELS = {
    "formality": "FRM", "occasion": "OCC", "color": "CLR",
    "style": "STY", "season": "SEA", "price": "PRC",
    "pattern": "PAT", "material": "MAT", "balance": "BAL",
}


# ===========================================================================
# CSS
# ===========================================================================

CUSTOM_CSS = """
/* ---- Mega Card ---- */
.mega {
    border: 1px solid #ddd; border-radius: 14px; background: #fff;
    margin-bottom: 20px; overflow: hidden;
}
.mega-top {
    display: flex; gap: 18px; padding: 16px;
}
.mega-imgs { flex-shrink: 0; }
.mega-main-img {
    width: 240px; height: 320px; object-fit: cover; border-radius: 10px;
    display: block; background: #f0f0f0;
}
.mega-gallery { display: flex; gap: 4px; margin-top: 6px; flex-wrap: wrap; }
.mega-thumb {
    width: 44px; height: 56px; object-fit: cover; border-radius: 5px;
    cursor: pointer; opacity: .55; transition: all .12s; border: 2px solid transparent;
}
.mega-thumb:hover, .mega-thumb.on { opacity: 1; border-color: #333; }
.mega-info { flex: 1; min-width: 0; }
.mega-cat {
    font-size: 10px; color: #999; text-transform: uppercase; letter-spacing: .4px;
}
.mega-name { font-size: 18px; font-weight: 700; margin: 3px 0; line-height: 1.25; }
.mega-brand { font-size: 13px; color: #666; }
.mega-price { font-size: 17px; font-weight: 700; margin: 5px 0 8px; }
.mega-price .sale { color: #d32f2f; }
.mega-price .orig { text-decoration: line-through; color: #bbb; font-weight: 400; font-size: 12px; margin-right: 4px; }
.mega-price .disc { background: #d32f2f; color: #fff; font-size: 10px; padding: 2px 6px; border-radius: 4px; margin-left: 5px; font-weight: 600; }
.mega-attrs { display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0; }
.mega-attr {
    padding: 2px 8px; border-radius: 12px; font-size: 10px;
    background: #f2f2f8; color: #555;
}
.mega-link {
    display: inline-block; margin-top: 6px; padding: 6px 12px;
    background: #1a1a2e; color: #fff !important; border-radius: 6px;
    text-decoration: none; font-size: 11px; font-weight: 500;
}
.mega-link:hover { background: #282850; }
.mega-id { font-size: 9px; color: #ccc; font-family: monospace; margin-top: 6px; }

/* ---- Rec Sections inside mega card ---- */
.rec-sec {
    padding: 10px 16px 14px; border-top: 1px solid #eee;
}
.rec-sec-title {
    font-size: 13px; font-weight: 700; color: #444; margin-bottom: 6px;
}
.rec-sec-meta { font-size: 10px; color: #aaa; margin-left: 8px; font-weight: 400; }
.rec-row {
    display: flex; gap: 10px; overflow-x: auto; padding-bottom: 4px;
}
.rec-cat-label {
    font-size: 11px; font-weight: 600; color: #666; margin: 8px 0 4px;
    padding-bottom: 2px; border-bottom: 1px solid #eee;
}

/* ---- Mini Rec Card ---- */
.mini {
    flex: 0 0 160px; border: 1px solid #e8e8e8; border-radius: 9px;
    overflow: hidden; background: #fff;
}
.mini-img {
    width: 160px; height: 200px; object-fit: cover; display: block; background: #f0f0f0;
}
.mini-body { padding: 6px 8px 8px; }
.mini-name {
    font-size: 10px; font-weight: 500; line-height: 1.3; height: 2.6em; overflow: hidden;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}
.mini-sub { font-size: 9px; color: #888; margin-top: 2px; }
.mini-score { display: flex; align-items: center; gap: 4px; margin-top: 4px; }
.mini-bar-bg { flex: 1; height: 4px; background: #eee; border-radius: 2px; overflow: hidden; }
.mini-bar-fg { height: 100%; border-radius: 2px; }
.mini-num { font-size: 10px; font-weight: 700; min-width: 26px; }
.mini-dims { display: flex; flex-wrap: wrap; gap: 2px; margin-top: 3px; }
.mini-dim { font-size: 7px; font-weight: 700; padding: 1px 3px; border-radius: 4px; }
.mini a { text-decoration: none; color: inherit; }
.mini a:hover .mini-name { text-decoration: underline; }

/* ---- Status ---- */
.empty { text-align: center; padding: 30px; color: #aaa; font-size: 13px; }
.timing { font-size: 11px; color: #999; margin-bottom: 12px; }

/* ---- Dark mode ---- */
.dark .mega { background: #1e1e2e; border-color: #444; }
.dark .mega-name { color: #e0e0e0; }
.dark .mega-brand { color: #aaa; }
.dark .mega-attr { background: #2a2d4a; color: #bbc; }
.dark .rec-sec { border-color: #333; }
.dark .rec-sec-title { color: #bbc; }
.dark .rec-cat-label { color: #999; border-color: #333; }
.dark .mini { background: #1e1e2e; border-color: #444; }
.dark .mini-name { color: #ddd; }
.dark .mini-sub { color: #999; }
.dark .mini-bar-bg { background: #333; }
.dark .mega-link { background: #2a2d5a; }
"""


# ===========================================================================
# DATA HELPERS
# ===========================================================================

def _load_top_brands(limit=60):
    try:
        sb = get_supabase_client()
        r = sb.table("products").select("brand").eq("in_stock", True).limit(8000).execute()
        counts = Counter(p["brand"] for p in r.data if p.get("brand"))
        return sorted(counts.keys(), key=lambda b: -counts[b])[:limit]
    except Exception as e:
        logger.warning("Failed to load brands: %s", e)
        return ["Boohoo", "Missguided", "Forever 21", "Princess Polly", "Nasty Gal"]


def search_products(query, categories, brands, color_families, patterns,
                    min_price, max_price, on_sale, count):
    sb = get_supabase_client()
    attr_ids = None
    if categories or color_families or patterns:
        aq = sb.table("product_attributes").select("sku_id")
        if categories:
            aq = aq.in_("category_l1", categories)
        if color_families:
            aq = aq.in_("color_family", color_families)
        if patterns:
            aq = aq.in_("pattern", patterns)
        ar = aq.limit(2000).execute()
        attr_ids = [row["sku_id"] for row in (ar.data or [])]
        if not attr_ids:
            return []
    q = sb.table("products").select(PRODUCT_SELECT).eq("in_stock", True)
    if query and query.strip():
        safe = query.strip().replace("%", "").replace("_", "")
        q = q.or_(f"name.ilike.%{safe}%,brand.ilike.%{safe}%")
    if brands:
        q = q.in_("brand", brands)
    if min_price and float(min_price) > 0:
        q = q.gte("price", float(min_price))
    if max_price and float(max_price) > 0:
        q = q.lte("price", float(max_price))
    if attr_ids is not None:
        q = q.in_("id", attr_ids[:500])
    products = q.limit(int(count)).execute().data or []
    if on_sale:
        products = [p for p in products if float(p.get("original_price") or 0) > float(p.get("price") or 0)]
    return products


def random_products(count=8):
    sb = get_supabase_client()
    offset = random.randint(0, 80000)
    r = sb.table("products").select(PRODUCT_SELECT).eq(
        "in_stock", True
    ).range(offset, offset + int(count) - 1).execute()
    products = r.data or []
    random.shuffle(products)
    return products


def fetch_attrs(product_id):
    sb = get_supabase_client()
    ar = sb.table("product_attributes").select(ATTRS_SELECT).eq(
        "sku_id", product_id
    ).limit(1).execute()
    return ar.data[0] if ar.data else {}


# ===========================================================================
# PARALLEL RECOMMENDATION LOADING
# ===========================================================================

def _load_recs_for_product(product):
    """Load similar + fit for one product. Called in thread pool."""
    pid = product["id"]
    result = {"product": product, "attrs": {}, "similar": None, "fit": None,
              "sim_ms": 0, "fit_ms": 0}
    try:
        result["attrs"] = fetch_attrs(pid)
    except Exception:
        pass
    try:
        engine = get_outfit_engine()
        t0 = time.time()
        result["similar"] = engine.get_similar_scored(product_id=pid, limit=SIM_LIMIT)
        result["sim_ms"] = (time.time() - t0) * 1000
    except Exception as e:
        logger.warning("Similar failed for %s: %s", pid, e)
    try:
        engine = get_outfit_engine()
        t0 = time.time()
        result["fit"] = engine.build_outfit(product_id=pid, items_per_category=FIT_LIMIT)
        result["fit_ms"] = (time.time() - t0) * 1000
    except Exception as e:
        logger.warning("Fit failed for %s: %s", pid, e)
    return result


def load_all_recs(products):
    """Load recommendations for all products in parallel."""
    results = [None] * len(products)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_load_recs_for_product, p): i for i, p in enumerate(products)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error("Thread failed: %s", e)
                results[idx] = {"product": products[idx], "attrs": {},
                                "similar": None, "fit": None, "sim_ms": 0, "fit_ms": 0}
    return results


# ===========================================================================
# HTML RENDERERS
# ===========================================================================

def _esc(s):
    return _html.escape(str(s)) if s else ""


def _score_color(v):
    return "#4caf50" if v >= 0.7 else "#ff9800" if v >= 0.5 else "#ef5350"


def _price_html(price, orig):
    price = float(price or 0)
    orig = float(orig or 0)
    if orig > price > 0:
        disc = int((1 - price / orig) * 100)
        return (f'<span class="orig">${orig:.0f}</span> '
                f'<span class="sale">${price:.0f}</span>'
                f'<span class="disc">-{disc}%</span>')
    return f"${price:.0f}" if price else ""


def _render_mini_card(item):
    """Render a small recommendation card."""
    name = _esc(item.get("name"))
    brand = _esc(item.get("brand"))
    price = float(item.get("price") or 0)
    img = item.get("primary_image_url") or item.get("image_url") or ""
    source = item.get("source_url") or ""
    tattoo = float(item.get("tattoo_score") or 0)
    pct = int(tattoo * 100)
    col = _score_color(tattoo)

    img_h = (f'<img class="mini-img" src="{_esc(img)}" loading="lazy" '
             f'onerror="this.style.display=\'none\'">'
             if img else '<div class="mini-img" style="display:flex;align-items:center;'
             'justify-content:center;color:#ccc;font-size:11px">No img</div>')

    # Dimension pills
    dims = item.get("dimension_scores") or {}
    dim_h = ""
    if dims:
        pills = []
        for key, label in DIM_LABELS.items():
            v = dims.get(key, 0)
            c = _score_color(v)
            pills.append(f'<span class="mini-dim" style="background:{c}15;color:{c}">{label}{v:.0%}</span>')
        dim_h = f'<div class="mini-dims">{"".join(pills)}</div>'

    inner = f"""{img_h}
        <div class="mini-body">
            <div class="mini-name">{name}</div>
            <div class="mini-sub">{brand} &middot; ${price:.0f}</div>
            <div class="mini-score">
                <div class="mini-bar-bg"><div class="mini-bar-fg" style="width:{pct}%;background:{col}"></div></div>
                <span class="mini-num" style="color:{col}">{tattoo:.2f}</span>
            </div>
            {dim_h}
        </div>"""

    if source:
        return f'<div class="mini"><a href="{_esc(source)}" target="_blank" rel="noopener">{inner}</a></div>'
    return f'<div class="mini">{inner}</div>'


def _render_mega_card(data):
    """Render one mega card: product detail + similar + fit."""
    product = data["product"]
    attrs = data.get("attrs") or {}
    similar = data.get("similar")
    fit = data.get("fit")
    sim_ms = data.get("sim_ms", 0)
    fit_ms = data.get("fit_ms", 0)

    pid = product["id"]
    name = _esc(product.get("name"))
    brand = _esc(product.get("brand"))
    img = product.get("primary_image_url") or ""
    gallery = product.get("gallery_images") or []
    source_url = product.get("source_url") or ""
    price_h = _price_html(product.get("price"), product.get("original_price"))

    # Category
    cat1 = attrs.get("category_l1") or product.get("category") or ""
    cat2 = attrs.get("category_l2") or ""
    cat_s = cat1 + (f" / {cat2}" if cat2 else "")

    # Main image
    uid = pid[:8]  # unique prefix for this card's image swap
    main_img = (f'<img id="mi-{uid}" class="mega-main-img" src="{_esc(img)}" alt="{name}">'
                if img else '<div class="mega-main-img" style="display:flex;align-items:center;'
                'justify-content:center;color:#bbb">No image</div>')

    # Gallery
    seen = set()
    all_imgs = []
    for u in [img] + list(gallery):
        if u and u not in seen:
            seen.add(u)
            all_imgs.append(u)
    gal_h = ""
    if len(all_imgs) > 1:
        thumbs = []
        for i, u in enumerate(all_imgs[:8]):
            on = " on" if i == 0 else ""
            click = (f"document.getElementById('mi-{uid}').src='{_esc(u)}';"
                     f"this.parentElement.querySelectorAll('.mega-thumb').forEach(t=>t.classList.remove('on'));"
                     f"this.classList.add('on')")
            thumbs.append(f'<img class="mega-thumb{on}" src="{_esc(u)}" onclick="{click}" loading="lazy">')
        gal_h = f'<div class="mega-gallery">{"".join(thumbs)}</div>'

    # Attribute badges
    badges = []
    for key, label in [
        ("formality", "Formality"), ("pattern", "Pattern"), ("color_family", "Color"),
        ("fit_type", "Fit"), ("silhouette", "Silhouette"),
        ("apparent_fabric", "Fabric"), ("texture", "Texture"),
    ]:
        v = attrs.get(key)
        if v and str(v) not in ("N/A", "Unknown", "None", ""):
            badges.append(f'<span class="mega-attr">{label}: {_esc(v)}</span>')
    construction = attrs.get("construction") or {}
    if isinstance(construction, str):
        try:
            construction = json.loads(construction)
        except (json.JSONDecodeError, TypeError):
            construction = {}
    for key, label in [("neckline", "Neckline"), ("sleeve_type", "Sleeve"), ("length", "Length")]:
        v = construction.get(key)
        if v and str(v) not in ("N/A", "Unknown", "None", ""):
            badges.append(f'<span class="mega-attr">{label}: {_esc(v)}</span>')
    for key, label in [("occasions", "Occasions"), ("style_tags", "Styles"), ("seasons", "Seasons")]:
        vals = attrs.get(key) or []
        if isinstance(vals, str):
            try:
                vals = json.loads(vals)
            except (json.JSONDecodeError, TypeError):
                vals = [vals] if vals else []
        if vals:
            badges.append(f'<span class="mega-attr">{label}: {_esc(", ".join(str(v) for v in vals[:3]))}</span>')
    badges_h = f'<div class="mega-attrs">{"".join(badges)}</div>' if badges else ""

    source_h = (f'<a href="{_esc(source_url)}" target="_blank" rel="noopener" class="mega-link">'
                f'View on retailer &#8599;</a>') if source_url else ""

    # --- Top section: product detail ---
    top_html = f"""<div class="mega-top">
        <div class="mega-imgs">{main_img}{gal_h}</div>
        <div class="mega-info">
            <div class="mega-cat">{_esc(cat_s)}</div>
            <div class="mega-name">{name}</div>
            <div class="mega-brand">{brand}</div>
            <div class="mega-price">{price_h}</div>
            {badges_h}
            {source_h}
            <div class="mega-id">{pid}</div>
        </div>
    </div>"""

    # --- Similar section ---
    sim_html = ""
    if similar and similar.get("results"):
        items = similar["results"]
        cards = "".join(_render_mini_card(i) for i in items)
        sim_html = f"""<div class="rec-sec">
            <div class="rec-sec-title">Similar Items ({len(items)})
                <span class="rec-sec-meta">{sim_ms:.0f}ms</span>
            </div>
            <div class="rec-row">{cards}</div>
        </div>"""
    else:
        sim_html = '<div class="rec-sec"><div class="rec-sec-title">Similar Items</div><div style="color:#bbb;font-size:11px">No similar items found</div></div>'

    # --- Fit section ---
    fit_html = ""
    if fit and fit.get("recommendations"):
        recs = fit["recommendations"]
        status = fit.get("status", "ok")
        outfit = fit.get("complete_outfit") or {}
        total = outfit.get("total_price", 0)
        count = outfit.get("item_count", 0)

        cat_sections = ""
        for cat, cdata in recs.items():
            items = cdata.get("items", [])
            if not items:
                continue
            cards = "".join(_render_mini_card(i) for i in items)
            label = cat.replace("_", " ").title()
            cat_sections += f"""<div class="rec-cat-label">{_esc(label)} ({len(items)})</div>
                <div class="rec-row">{cards}</div>"""

        if cat_sections:
            fit_html = f"""<div class="rec-sec">
                <div class="rec-sec-title">Complete the Fit
                    <span class="rec-sec-meta">{count} pieces &middot; ${total:.2f} &middot; {status} &middot; {fit_ms:.0f}ms</span>
                </div>
                {cat_sections}
            </div>"""
        else:
            fit_html = '<div class="rec-sec"><div class="rec-sec-title">Complete the Fit</div><div style="color:#bbb;font-size:11px">No complementary items</div></div>'
    elif fit and fit.get("error"):
        fit_html = f'<div class="rec-sec"><div class="rec-sec-title">Complete the Fit</div><div style="color:#bbb;font-size:11px">{_esc(fit["error"])}</div></div>'
    else:
        fit_html = '<div class="rec-sec"><div class="rec-sec-title">Complete the Fit</div><div style="color:#bbb;font-size:11px">Not available</div></div>'

    return f'<div class="mega">{top_html}{sim_html}{fit_html}</div>'


def render_all(results, total_ms):
    """Render the full page of mega cards."""
    if not results:
        return '<div class="empty">No products found.</div>'
    html = f'<div class="timing">Loaded {len(results)} products with recommendations in {total_ms:.1f}s</div>'
    for data in results:
        html += _render_mega_card(data)
    return html


# ===========================================================================
# EVENT HANDLERS
# ===========================================================================

def handle_search(query, categories, brands, colors, patterns,
                  min_price, max_price, on_sale, count):
    count = int(count or 8)
    t0 = time.time()
    products = search_products(
        query, categories or [], brands or [], colors or [], patterns or [],
        min_price, max_price, on_sale, count,
    )
    if not products:
        return "**0 products** found", '<div class="empty">No products found. Try different filters.</div>'

    results = load_all_recs(products)
    total_s = time.time() - t0
    info = f"**{len(products)} products** with recommendations loaded in **{total_s:.1f}s**"
    return info, render_all(results, total_s)


def handle_random(count):
    count = int(count or 8)
    t0 = time.time()
    products = random_products(count)
    if not products:
        return "No products found", '<div class="empty">No products found.</div>'
    results = load_all_recs(products)
    total_s = time.time() - t0
    info = f"**{len(products)} random products** with recommendations loaded in **{total_s:.1f}s**"
    return info, render_all(results, total_s)


# ===========================================================================
# GRADIO APP
# ===========================================================================

def build_app(brands):
    with gr.Blocks(title="Fashion Discovery") as app:

        gr.Markdown(
            "# Fashion Discovery\n"
            "Each product card shows its **similar items** and **complete-the-fit** "
            "outfit recommendations powered by TATTOO 9-dimension scoring.\n\n"
            "Rec items link to the retailer site. Gallery thumbnails are clickable."
        )

        # ---- Search ----
        with gr.Row():
            search_input = gr.Textbox(
                placeholder="Search by name or brand...", show_label=False, scale=3,
            )
            cat_dd = gr.Dropdown(choices=CATEGORIES, label="Category", multiselect=True, scale=1)
            brand_dd = gr.Dropdown(choices=brands, label="Brand", multiselect=True, scale=1,
                                   allow_custom_value=True)

        with gr.Row():
            color_dd = gr.Dropdown(choices=COLOR_FAMILIES, label="Color", multiselect=True, scale=1)
            pattern_dd = gr.Dropdown(choices=PATTERNS, label="Pattern", multiselect=True, scale=1)
            min_p = gr.Number(label="Min $", value=0, minimum=0, scale=0)
            max_p = gr.Number(label="Max $", value=0, minimum=0, scale=0)
            sale_cb = gr.Checkbox(label="Sale", value=False, scale=0)

        with gr.Row():
            count_slider = gr.Slider(minimum=2, maximum=12, value=6, step=1,
                                     label="Products to load", scale=1)
            search_btn = gr.Button("Search", variant="primary", size="sm")
            random_btn = gr.Button("Random", variant="secondary", size="sm")

        info_md = gr.Markdown("")
        results_html = gr.HTML()

        # ---- Wiring ----
        s_in = [search_input, cat_dd, brand_dd, color_dd, pattern_dd,
                min_p, max_p, sale_cb, count_slider]
        s_out = [info_md, results_html]

        search_btn.click(handle_search, s_in, s_out)
        search_input.submit(handle_search, s_in, s_out)
        random_btn.click(handle_random, [count_slider], s_out)

        # Preload random products on start
        app.load(handle_random, [count_slider], s_out)

    return app


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Loading brands...")
    top_brands = _load_top_brands()
    print(f"  {len(top_brands)} brands")

    print("Building app...")
    app = build_app(top_brands)

    print("http://localhost:7863")
    app.launch(
        server_name="0.0.0.0", server_port=7863, share=False,
        css=CUSTOM_CSS, theme=gr.themes.Soft(),
    )
