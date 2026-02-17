"""
Search Feed — clean Gradio UI with infinite scroll.

Calls HybridSearchService directly (no API server needed).
Loads .env automatically so the LLM query planner works.

Usage:
    PYTHONPATH=src python scripts/search_feed_gradio.py

    Open http://localhost:7680 in your browser
"""

import os
import sys
import time

# Setup path and env before any src imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import gradio as gr

from search.hybrid_search import HybridSearchService
from search.models import HybridSearchRequest, SortBy

PAGE_SIZE = 20

# ---------------------------------------------------------------------------
# Initialize search service (one-time, loads FashionCLIP etc.)
# ---------------------------------------------------------------------------

print("Initializing HybridSearchService...")
_service = HybridSearchService()
print("Preloading FashionCLIP model...")
_service.semantic_engine._load_model()
print("Service ready.")

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* Overall layout */
.main-container { max-width: 1200px; margin: 0 auto; }

/* Search bar */
.search-row { position: sticky; top: 0; z-index: 100; background: var(--background-fill-primary); padding: 8px 0 12px; }

/* Product grid */
.product-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    padding: 8px 0;
}
@media (max-width: 1100px) { .product-grid { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 768px) { .product-grid { grid-template-columns: repeat(2, 1fr); } }

/* Product card */
.product-card {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    overflow: hidden;
    background: #fff;
    transition: box-shadow 0.2s, transform 0.15s;
    cursor: default;
}
.product-card:hover {
    box-shadow: 0 4px 20px rgba(0,0,0,0.10);
    transform: translateY(-2px);
}
.product-card img {
    width: 100%;
    aspect-ratio: 3/4;
    object-fit: cover;
    display: block;
    background: #f3f4f6;
}
.product-card .card-info {
    padding: 10px 12px 12px;
}
.product-card .card-brand {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #6b7280;
    margin-bottom: 2px;
}
.product-card .card-name {
    font-size: 13px;
    font-weight: 500;
    color: #111827;
    line-height: 1.3;
    margin-bottom: 6px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.product-card .card-price-row {
    display: flex;
    align-items: center;
    gap: 6px;
}
.product-card .card-price {
    font-size: 14px;
    font-weight: 700;
    color: #111827;
}
.product-card .card-original-price {
    font-size: 12px;
    color: #9ca3af;
    text-decoration: line-through;
}
.product-card .card-sale-badge {
    font-size: 10px;
    font-weight: 700;
    color: #dc2626;
    background: #fef2f2;
    padding: 1px 6px;
    border-radius: 4px;
}
.product-card .card-attrs {
    margin-top: 6px;
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
}
.product-card .card-attr-tag {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 6px;
    background: #f3f4f6;
    color: #4b5563;
}
.product-card .card-source {
    font-size: 10px;
    color: #9ca3af;
    margin-top: 4px;
    font-family: monospace;
}

/* Stats bar */
.stats-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 4px;
    font-size: 13px;
    color: #6b7280;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 8px;
}
.stats-bar .stat-highlight { color: #111827; font-weight: 600; }

/* Load more */
.load-more-row {
    display: flex;
    justify-content: center;
    padding: 24px 0;
}
"""

# ---------------------------------------------------------------------------
# Card builder
# ---------------------------------------------------------------------------

def _build_card(product) -> str:
    """Build a single product card HTML. Accepts ProductResult or dict."""
    # Support both Pydantic model and dict
    def _get(key, default=None):
        if isinstance(product, dict):
            return product.get(key, default)
        return getattr(product, key, default)

    img_url = _get("image_url") or ""
    name = _get("name", "Unknown")
    brand = _get("brand", "")
    price = _get("price", 0)
    original_price = _get("original_price")
    is_on_sale = _get("is_on_sale", False)

    # Image
    if img_url:
        img_html = f'<img src="{img_url}" alt="{name[:40]}" loading="lazy" />'
    else:
        img_html = '<div style="width:100%;aspect-ratio:3/4;background:#f3f4f6;display:flex;align-items:center;justify-content:center;color:#9ca3af;font-size:12px;">No Image</div>'

    # Price
    price_html = f'<span class="card-price">${price:.2f}</span>'
    if is_on_sale and original_price and original_price > price:
        discount = int((1 - price / original_price) * 100)
        price_html += f' <span class="card-original-price">${original_price:.2f}</span>'
        price_html += f' <span class="card-sale-badge">-{discount}%</span>'

    # Key attribute tags
    tags = []
    for key in ["category_l2", "primary_color", "pattern", "apparent_fabric", "fit_type", "neckline"]:
        val = _get(key)
        if val and val not in ("N/A", "null", "None", "Other", "Solid"):
            tags.append(f'<span class="card-attr-tag">{val}</span>')

    tags_html = f'<div class="card-attrs">{"".join(tags[:5])}</div>' if tags else ""

    # Source indicator
    source_parts = []
    if _get("algolia_rank"):
        source_parts.append(f"A#{_get('algolia_rank')}")
    if _get("semantic_rank"):
        source_parts.append(f"S#{_get('semantic_rank')}")
    if _get("rrf_score"):
        source_parts.append(f"RRF:{_get('rrf_score'):.4f}")
    source_html = f'<div class="card-source">{" ".join(source_parts)}</div>' if source_parts else ""

    return f"""<div class="product-card">
  {img_html}
  <div class="card-info">
    <div class="card-brand">{brand}</div>
    <div class="card-name">{name}</div>
    <div class="card-price-row">{price_html}</div>
    {tags_html}
    {source_html}
  </div>
</div>"""


def _build_grid(products: list) -> str:
    """Build the full product grid HTML."""
    if not products:
        return '<div style="text-align:center;padding:40px;color:#9ca3af;">No results found.</div>'
    cards = [_build_card(p) for p in products]
    return f'<div class="product-grid">{"".join(cards)}</div>'


def _build_stats(total_loaded: int, timing: dict, intent: str, page: int) -> str:
    """Build the stats bar HTML."""
    total_ms = timing.get("total_ms", 0)
    planner_ms = timing.get("planner_ms", 0)
    algolia_ms = timing.get("algolia_ms", 0)
    semantic_ms = timing.get("semantic_ms", 0)

    return (
        f'<div class="stats-bar">'
        f'<span><span class="stat-highlight">{total_loaded}</span> products loaded &middot; page {page}</span>'
        f'<span>Intent: <span class="stat-highlight">{intent}</span> &middot; '
        f'Planner: {planner_ms}ms &middot; Algolia: {algolia_ms}ms &middot; Semantic: {semantic_ms}ms &middot; Total: {total_ms}ms</span>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Search logic — calls HybridSearchService directly
# ---------------------------------------------------------------------------

def do_search(query: str, state: dict) -> tuple:
    """New search — reset state, fetch page 1."""
    if not query or not query.strip():
        return (
            "",
            '<div style="text-align:center;padding:40px;color:#9ca3af;">Enter a search query above.</div>',
            gr.update(visible=False),
            {"query": "", "page": 0, "products": [], "has_more": False, "timing": {}, "intent": ""},
        )

    state = {"query": query.strip(), "page": 1, "products": [], "has_more": False, "timing": {}, "intent": ""}
    return _fetch_page(state)


def do_load_more(state: dict) -> tuple:
    """Load next page, append to existing results."""
    if not state or not state.get("query"):
        return "", "", gr.update(visible=False), state

    state["page"] = state.get("page", 1) + 1
    return _fetch_page(state)


def _fetch_page(state: dict) -> tuple:
    """Run a search page via HybridSearchService directly."""
    query = state["query"]
    page = state["page"]

    try:
        request = HybridSearchRequest(
            query=query,
            page=page,
            page_size=PAGE_SIZE,
            sort_by=SortBy.RELEVANCE,
        )

        response = _service.search(request)

        new_products = response.results
        timing = response.timing
        intent = response.intent
        has_more = response.pagination.has_more

        # Accumulate products
        state["products"].extend(new_products)
        state["has_more"] = has_more
        state["timing"] = timing
        state["intent"] = intent

        # Build outputs
        stats_html = _build_stats(len(state["products"]), timing, intent, page)
        grid_html = _build_grid(state["products"])
        show_load_more = gr.update(visible=has_more)

        return stats_html, grid_html, show_load_more, state

    except Exception as e:
        error_html = f'<div style="padding:20px;color:#dc2626;">Error: {e}</div>'
        return "", error_html, gr.update(visible=False), state


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Search Feed",
    css=CUSTOM_CSS,
) as app:

    # State to track accumulated results
    feed_state = gr.State({
        "query": "",
        "page": 0,
        "products": [],
        "has_more": False,
        "timing": {},
        "intent": "",
    })

    # Header
    gr.Markdown("## Search Feed")

    # Search bar
    with gr.Row(elem_classes="search-row"):
        search_input = gr.Textbox(
            placeholder="Search for anything... e.g. fitted ribbed turtleneck in burgundy",
            show_label=False,
            lines=1,
            scale=5,
            container=False,
        )
        search_btn = gr.Button("Search", variant="primary", scale=1, min_width=100)

    # Stats bar
    stats_output = gr.HTML("")

    # Product grid
    grid_output = gr.HTML(
        '<div style="text-align:center;padding:60px;color:#9ca3af;">Enter a search query above.</div>'
    )

    # Load More button
    load_more_btn = gr.Button(
        "Load More",
        variant="secondary",
        visible=False,
        size="lg",
        elem_classes="load-more-row",
    )

    # Wire up events
    search_btn.click(
        do_search,
        inputs=[search_input, feed_state],
        outputs=[stats_output, grid_output, load_more_btn, feed_state],
    )
    search_input.submit(
        do_search,
        inputs=[search_input, feed_state],
        outputs=[stats_output, grid_output, load_more_btn, feed_state],
    )
    load_more_btn.click(
        do_load_more,
        inputs=[feed_state],
        outputs=[stats_output, grid_output, load_more_btn, feed_state],
    )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7680, share=False)
