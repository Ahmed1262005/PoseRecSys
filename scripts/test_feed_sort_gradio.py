"""
Feed Sort Test — Gradio UI for testing the sort_by feature end-to-end.

Calls RecommendationPipeline.get_feed_keyset() directly with sort_by parameter,
which hits the Supabase RPC `get_feed_sorted_keyset` (migration 048).

Usage:
    cd /mnt/d/ecommerce/recommendationSystem
    PYTHONPATH=src python scripts/test_feed_sort_gradio.py
    # -> http://localhost:7862
"""

import html as _html
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

# -- path setup ---------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import gradio as gr

from recs.pipeline import RecommendationPipeline
from recs.models import FeedSortBy

# =============================================================================
# GLOBALS
# =============================================================================

pipeline: Optional[RecommendationPipeline] = None
_state: Dict[str, Any] = {}  # track cursors per sort mode


def init_pipeline():
    global pipeline
    if pipeline is None:
        print("[TestFeedSort] Initializing pipeline...")
        pipeline = RecommendationPipeline(load_sasrec=False)
        print("[TestFeedSort] Pipeline ready.")
    return pipeline


# =============================================================================
# CSS
# =============================================================================

CSS = """
.product-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
    padding: 8px;
}
.product-card {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    overflow: hidden;
    background: #fafafa;
    transition: box-shadow 0.2s;
}
.product-card:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.1); }
.product-card img {
    width: 100%;
    height: 240px;
    object-fit: cover;
}
.product-card .card-info {
    padding: 10px;
}
.product-card .card-name {
    font-weight: 600;
    font-size: 13px;
    line-height: 1.3;
    margin-bottom: 4px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.product-card .card-brand {
    color: #666;
    font-size: 12px;
    margin-bottom: 4px;
}
.product-card .card-price {
    font-weight: 700;
    font-size: 15px;
    color: #1a237e;
}
.product-card .card-price .sale {
    color: #e74c3c;
}
.product-card .card-price .original {
    text-decoration: line-through;
    color: #999;
    font-weight: 400;
    font-size: 12px;
    margin-right: 4px;
}
.product-card .card-rank {
    color: #aaa;
    font-size: 10px;
    font-family: monospace;
}
.product-card .card-meta {
    color: #888;
    font-size: 10px;
    margin-top: 4px;
}
.status-bar {
    padding: 10px 14px;
    background: #f0f4ff;
    border-radius: 8px;
    font-size: 13px;
    color: #333;
    margin-bottom: 8px;
}
.compare-section h3 {
    margin: 12px 0 6px 0;
    font-size: 15px;
    color: #1a237e;
}

/* Dark mode */
.dark .product-card { background: #1e1e2e; border-color: #444; }
.dark .product-card .card-name { color: #e0e0e0; }
.dark .product-card .card-brand { color: #aaa; }
.dark .product-card .card-price { color: #8e99f3; }
.dark .product-card .card-rank { color: #777; }
.dark .product-card .card-meta { color: #888; }
.dark .status-bar { background: #1e2340; color: #e0e0e0; }
.dark .compare-section h3 { color: #8e99f3; }
"""


# =============================================================================
# RENDERING
# =============================================================================

def render_card(item: dict, idx: int) -> str:
    name = _html.escape(str(item.get("name", "Unknown")))
    brand = _html.escape(str(item.get("brand", "Unknown")))
    price = float(item.get("price", 0) or 0)
    img = item.get("image_url", "")
    pid = str(item.get("product_id", ""))[:12]
    rank = item.get("rank", idx)
    sale = item.get("is_on_sale", False)
    is_new = item.get("is_new", False)
    source = item.get("source", "")
    category = item.get("category", "")

    img_html = (
        f'<img src="{_html.escape(img)}" alt="{name}" loading="lazy">'
        if img else
        f'<div style="width:100%;height:240px;background:#eee;display:flex;'
        f'align-items:center;justify-content:center;color:#999;font-size:12px">'
        f'{_html.escape(category)}</div>'
    )

    price_html = f"${price:.2f}"
    if sale:
        orig = float(item.get("original_price", 0) or 0)
        if orig > price:
            price_html = (
                f'<span class="original">${orig:.2f}</span>'
                f'<span class="sale">${price:.2f}</span>'
            )
        else:
            price_html = f'<span class="sale">${price:.2f}</span>'

    new_badge = ""
    if is_new:
        new_badge = ' <span style="color:#1565c0;font-size:10px;font-weight:bold">NEW</span>'

    return f'''<div class="product-card">
        {img_html}
        <div class="card-info">
            <div class="card-rank">#{rank} &middot; {_html.escape(pid)}</div>
            <div class="card-name">{name}</div>
            <div class="card-brand">{brand}</div>
            <div class="card-price">{price_html}{new_badge}</div>
            <div class="card-meta">{_html.escape(source)} &middot; {_html.escape(category)}</div>
        </div>
    </div>'''


def render_grid(items: list, title: str = "") -> str:
    if not items:
        return '<div class="status-bar">No results returned.</div>'
    header = f"<h3>{_html.escape(title)}</h3>" if title else ""
    cards = "\n".join(render_card(item, i + 1) for i, item in enumerate(items))
    return f'{header}<div class="product-grid">{cards}</div>'


def render_status(resp: dict, elapsed: float) -> str:
    pagination = resp.get("pagination", {})
    metadata = resp.get("metadata", {})
    sort_by = resp.get("sort_by", "?")
    page = pagination.get("page", 0)
    returned = pagination.get("items_returned", 0)
    has_more = pagination.get("has_more", False)
    session_seen = pagination.get("session_seen_count", 0)

    parts = [
        f"<b>Sort:</b> {_html.escape(sort_by)}",
        f"<b>Page:</b> {page}",
        f"<b>Returned:</b> {returned}",
        f"<b>Has more:</b> {has_more}",
        f"<b>Session seen:</b> {session_seen}",
        f"<b>Time:</b> {elapsed:.2f}s",
    ]
    if metadata:
        parts.append(f"<b>DB candidates:</b> {metadata.get('candidates_retrieved', '?')}")
        parts.append(f"<b>After filters:</b> {metadata.get('candidates_after_python_filters', '?')}")
        parts.append(f"<b>After diversity:</b> {metadata.get('candidates_after_diversity', '?')}")

    return f'<div class="status-bar">{" &nbsp;|&nbsp; ".join(parts)}</div>'


# =============================================================================
# CORE FEED CALL
# =============================================================================

def call_feed(
    sort_by: str,
    gender: str,
    page_size: int,
    categories: str,
    min_price: Optional[float],
    max_price: Optional[float],
    brands: str,
    on_sale_only: bool,
    cursor: Optional[str] = None,
) -> Dict[str, Any]:
    """Call pipeline.get_feed_keyset() with given params. Returns the raw response dict."""
    p = init_pipeline()

    sort_enum = FeedSortBy(sort_by)

    # Parse comma-separated inputs
    cat_list = [c.strip() for c in categories.split(",") if c.strip()] if categories else None
    brand_list = [b.strip() for b in brands.split(",") if b.strip()] if brands else None

    # Use a stable anon_id so session/cursor state persists across pages
    anon_id = f"sort_test_{sort_by}"

    resp = p.get_feed_keyset(
        anon_id=anon_id,
        gender=gender,
        sort_by=sort_enum,
        categories=cat_list,
        preferred_brands=brand_list,
        min_price=min_price if min_price and min_price > 0 else None,
        max_price=max_price if max_price and max_price > 0 else None,
        on_sale_only=on_sale_only,
        page_size=page_size,
        cursor=cursor,
        debug=True,
    )
    return resp


# =============================================================================
# TAB 1: SINGLE FEED
# =============================================================================

def tab1_fetch(sort_by, gender, page_size, categories, min_price, max_price, brands, on_sale):
    """Fetch first page."""
    try:
        _state.pop("tab1_cursor", None)
        t0 = time.time()
        resp = call_feed(sort_by, gender, int(page_size), categories, min_price, max_price, brands, on_sale)
        elapsed = time.time() - t0

        _state["tab1_cursor"] = resp.get("cursor")
        _state["tab1_params"] = dict(
            sort_by=sort_by, gender=gender, page_size=int(page_size),
            categories=categories, min_price=min_price, max_price=max_price,
            brands=brands, on_sale_only=on_sale,
        )

        items = resp.get("results", [])
        status = render_status(resp, elapsed)
        grid = render_grid(items, f"{sort_by} - Page {resp['pagination'].get('page', 0)}")
        raw = json.dumps(resp, indent=2, default=str)

        return status, grid, raw
    except Exception as e:
        tb = traceback.format_exc()
        return f'<div class="status-bar">Error: {_html.escape(str(e))}</div>', "", tb


def tab1_next_page():
    """Fetch next page using stored cursor."""
    cursor = _state.get("tab1_cursor")
    params = _state.get("tab1_params")
    if not cursor:
        return '<div class="status-bar">No cursor — fetch first page first, or no more results.</div>', "", ""
    if not params:
        return '<div class="status-bar">No params stored — fetch first page first.</div>', "", ""

    try:
        t0 = time.time()
        resp = call_feed(cursor=cursor, **params)
        elapsed = time.time() - t0

        _state["tab1_cursor"] = resp.get("cursor")

        items = resp.get("results", [])
        status = render_status(resp, elapsed)
        grid = render_grid(items, f"{params['sort_by']} - Page {resp['pagination'].get('page', 0)}")
        raw = json.dumps(resp, indent=2, default=str)

        return status, grid, raw
    except Exception as e:
        tb = traceback.format_exc()
        return f'<div class="status-bar">Error: {_html.escape(str(e))}</div>', "", tb


def tab1_reset():
    """Clear session state for a fresh start."""
    _state.pop("tab1_cursor", None)
    _state.pop("tab1_params", None)
    return '<div class="status-bar">Session reset. Ready for a fresh fetch.</div>', "", ""


# =============================================================================
# TAB 2: COMPARE SORT MODES
# =============================================================================

def tab2_compare(gender, page_size, categories, min_price, max_price, brands, on_sale):
    """Fetch same query with all 3 sort modes side by side."""
    all_html = []
    all_raw = {}

    for mode in ["relevance", "price_asc", "price_desc"]:
        try:
            t0 = time.time()
            resp = call_feed(mode, gender, int(page_size), categories, min_price, max_price, brands, on_sale)
            elapsed = time.time() - t0

            items = resp.get("results", [])
            status = render_status(resp, elapsed)
            grid = render_grid(items[:12], f"{mode}")  # show top 12 per mode

            all_html.append(f'<div class="compare-section">{status}{grid}</div>')
            all_raw[mode] = {
                "time": f"{elapsed:.2f}s",
                "returned": len(items),
                "first_price": items[0]["price"] if items else None,
                "last_price": items[-1]["price"] if items else None,
                "brands": list(dict.fromkeys(i["brand"] for i in items[:12])),
            }
        except Exception as e:
            all_html.append(
                f'<div class="compare-section"><div class="status-bar">'
                f'{_html.escape(mode)}: Error - {_html.escape(str(e))}</div></div>'
            )
            all_raw[mode] = {"error": str(e)}

    html_out = "\n".join(all_html)
    raw_out = json.dumps(all_raw, indent=2, default=str)
    return html_out, raw_out


# =============================================================================
# TAB 3: PAGINATION STRESS TEST
# =============================================================================

def tab3_paginate(sort_by, gender, page_size, categories, min_price, max_price, brands, on_sale, num_pages):
    """Fetch multiple pages and verify ordering + no duplicates."""
    num_pages = int(num_pages)
    page_size = int(page_size)

    all_ids = []
    all_prices = []
    pages_info = []
    cursor = None

    for pg in range(num_pages):
        try:
            t0 = time.time()
            resp = call_feed(sort_by, gender, page_size, categories, min_price, max_price, brands, on_sale, cursor=cursor)
            elapsed = time.time() - t0

            items = resp.get("results", [])
            cursor = resp.get("cursor")

            page_ids = [i["product_id"] for i in items]
            page_prices = [float(i.get("price", 0)) for i in items]

            all_ids.extend(page_ids)
            all_prices.extend(page_prices)

            pages_info.append({
                "page": pg,
                "items": len(items),
                "price_range": f"${min(page_prices):.2f} - ${max(page_prices):.2f}" if page_prices else "n/a",
                "time": f"{elapsed:.2f}s",
                "has_more": resp.get("pagination", {}).get("has_more", False),
            })

            if not cursor or not resp.get("pagination", {}).get("has_more", False):
                break
        except Exception as e:
            pages_info.append({"page": pg, "error": str(e)})
            break

    # Check for duplicates
    unique_ids = set(all_ids)
    dup_count = len(all_ids) - len(unique_ids)

    # Check sort order
    order_ok = True
    violations = []
    if sort_by == "price_asc":
        for i in range(1, len(all_prices)):
            if all_prices[i] < all_prices[i - 1] - 0.001:
                order_ok = False
                violations.append(f"  idx {i-1}=${all_prices[i-1]:.2f} > idx {i}=${all_prices[i]:.2f}")
                if len(violations) >= 5:
                    break
    elif sort_by == "price_desc":
        for i in range(1, len(all_prices)):
            if all_prices[i] > all_prices[i - 1] + 0.001:
                order_ok = False
                violations.append(f"  idx {i-1}=${all_prices[i-1]:.2f} < idx {i}=${all_prices[i]:.2f}")
                if len(violations) >= 5:
                    break

    # Build report
    report_lines = [
        f"Sort mode: {sort_by}",
        f"Pages fetched: {len(pages_info)}",
        f"Total items: {len(all_ids)}",
        f"Unique items: {len(unique_ids)}",
        f"Duplicates: {dup_count}",
        f"Sort order correct: {'YES' if order_ok else 'NO'}",
    ]
    if violations:
        report_lines.append("Order violations:")
        report_lines.extend(violations)
    if all_prices:
        report_lines.append(f"Overall price range: ${min(all_prices):.2f} - ${max(all_prices):.2f}")

    report_lines.append("")
    report_lines.append("Per-page breakdown:")
    for p in pages_info:
        if "error" in p:
            report_lines.append(f"  Page {p['page']}: ERROR - {p['error']}")
        else:
            report_lines.append(
                f"  Page {p['page']}: {p['items']} items, "
                f"prices {p['price_range']}, {p['time']}, "
                f"has_more={p['has_more']}"
            )

    report = "\n".join(report_lines)

    # Show first page visually
    status_html = (
        f'<div class="status-bar">'
        f'<b>Pages:</b> {len(pages_info)} &nbsp;|&nbsp; '
        f'<b>Total items:</b> {len(all_ids)} &nbsp;|&nbsp; '
        f'<b>Duplicates:</b> {dup_count} &nbsp;|&nbsp; '
        f'<b>Order OK:</b> {"YES" if order_ok else "NO"}'
        f'</div>'
    )

    return status_html, report


# =============================================================================
# BUILD APP
# =============================================================================

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Feed Sort Test") as app:
        gr.Markdown("# Feed Sort Test\nEnd-to-end test of `sort_by` via `RecommendationPipeline.get_feed_keyset()` -> Supabase RPC `get_feed_sorted_keyset`.")

        # ---- Tab 1: Single Feed ----
        with gr.Tab("1. Feed Browser"):
            gr.Markdown("Fetch a feed page with a specific sort mode. Use **Next Page** to paginate.")
            with gr.Row():
                with gr.Column(scale=1):
                    t1_sort = gr.Radio(
                        choices=["relevance", "price_asc", "price_desc"],
                        value="price_asc",
                        label="Sort By",
                    )
                    t1_gender = gr.Radio(choices=["female", "male"], value="female", label="Gender")
                    t1_page_size = gr.Slider(minimum=5, maximum=100, value=20, step=5, label="Page Size")
                    t1_categories = gr.Textbox(label="Categories (comma-separated)", placeholder="e.g. dresses, tops")
                    with gr.Row():
                        t1_min_price = gr.Number(label="Min Price", value=0)
                        t1_max_price = gr.Number(label="Max Price", value=0)
                    t1_brands = gr.Textbox(label="Preferred Brands (comma-separated)", placeholder="e.g. Boohoo, Zara")
                    t1_on_sale = gr.Checkbox(label="On Sale Only", value=False)
                    with gr.Row():
                        t1_fetch_btn = gr.Button("Fetch", variant="primary")
                        t1_next_btn = gr.Button("Next Page")
                        t1_reset_btn = gr.Button("Reset Session")

                with gr.Column(scale=3):
                    t1_status = gr.HTML(label="Status")
                    t1_grid = gr.HTML(label="Results")
                    with gr.Accordion("Raw Response", open=False):
                        t1_raw = gr.Code(language="json", label="JSON")

            t1_inputs = [t1_sort, t1_gender, t1_page_size, t1_categories, t1_min_price, t1_max_price, t1_brands, t1_on_sale]
            t1_outputs = [t1_status, t1_grid, t1_raw]
            t1_fetch_btn.click(tab1_fetch, inputs=t1_inputs, outputs=t1_outputs)
            t1_next_btn.click(tab1_next_page, outputs=t1_outputs)
            t1_reset_btn.click(tab1_reset, outputs=t1_outputs)

        # ---- Tab 2: Compare Sort Modes ----
        with gr.Tab("2. Compare Sort Modes"):
            gr.Markdown("Same filters, all 3 sort modes side by side. Shows top 12 per mode.")
            with gr.Row():
                with gr.Column(scale=1):
                    t2_gender = gr.Radio(choices=["female", "male"], value="female", label="Gender")
                    t2_page_size = gr.Slider(minimum=5, maximum=50, value=20, step=5, label="Page Size")
                    t2_categories = gr.Textbox(label="Categories", placeholder="e.g. dresses")
                    with gr.Row():
                        t2_min_price = gr.Number(label="Min Price", value=0)
                        t2_max_price = gr.Number(label="Max Price", value=0)
                    t2_brands = gr.Textbox(label="Preferred Brands", placeholder="e.g. Boohoo")
                    t2_on_sale = gr.Checkbox(label="On Sale Only", value=False)
                    t2_compare_btn = gr.Button("Compare All 3", variant="primary")

                with gr.Column(scale=3):
                    t2_html = gr.HTML(label="Comparison")
                    with gr.Accordion("Summary", open=True):
                        t2_raw = gr.Code(language="json", label="Summary JSON")

            t2_inputs = [t2_gender, t2_page_size, t2_categories, t2_min_price, t2_max_price, t2_brands, t2_on_sale]
            t2_compare_btn.click(tab2_compare, inputs=t2_inputs, outputs=[t2_html, t2_raw])

        # ---- Tab 3: Pagination Stress Test ----
        with gr.Tab("3. Pagination Test"):
            gr.Markdown("Fetch N pages sequentially and verify: correct sort order across pages, no duplicate items.")
            with gr.Row():
                with gr.Column(scale=1):
                    t3_sort = gr.Radio(
                        choices=["price_asc", "price_desc", "relevance"],
                        value="price_asc",
                        label="Sort By",
                    )
                    t3_gender = gr.Radio(choices=["female", "male"], value="female", label="Gender")
                    t3_page_size = gr.Slider(minimum=10, maximum=50, value=20, step=5, label="Page Size")
                    t3_num_pages = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of Pages")
                    t3_categories = gr.Textbox(label="Categories", placeholder="e.g. dresses")
                    with gr.Row():
                        t3_min_price = gr.Number(label="Min Price", value=0)
                        t3_max_price = gr.Number(label="Max Price", value=0)
                    t3_brands = gr.Textbox(label="Preferred Brands", placeholder="")
                    t3_on_sale = gr.Checkbox(label="On Sale Only", value=False)
                    t3_run_btn = gr.Button("Run Pagination Test", variant="primary")

                with gr.Column(scale=3):
                    t3_status = gr.HTML(label="Status")
                    t3_report = gr.Textbox(label="Report", lines=20, interactive=False)

            t3_inputs = [t3_sort, t3_gender, t3_page_size, t3_categories, t3_min_price, t3_max_price, t3_brands, t3_on_sale, t3_num_pages]
            t3_run_btn.click(tab3_paginate, inputs=t3_inputs, outputs=[t3_status, t3_report])

    return app


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("[TestFeedSort] Building Gradio app...")
    app = build_app()
    print("[TestFeedSort] Launching on http://localhost:7862")
    app.launch(server_name="0.0.0.0", server_port=7862, share=False, css=CSS)
