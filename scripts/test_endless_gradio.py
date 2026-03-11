"""
Gradio demo for Endless Semantic Search pagination.

Shows page-by-page results with product cards, timing stats, and
a "Load More" button that fetches the next page via the endless
semantic pipeline.

Usage:
    1. Start the API server:
       PYTHONPATH=src uvicorn api.app:app --port 8000

    2. Run this script:
       PYTHONPATH=src python scripts/test_endless_gradio.py

    3. Open http://localhost:7680
"""

import os
import sys
import time
import requests
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL = os.getenv("API_URL", "http://localhost:8000")
SEARCH_URL = f"{API_URL}/api/search"


def _make_token(user_id: str = "gradio-endless-user") -> str:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    import jwt as pyjwt
    secret = os.getenv("SUPABASE_JWT_SECRET")
    now = int(time.time())
    return pyjwt.encode({
        "sub": user_id, "aud": "authenticated", "role": "authenticated",
        "email": f"{user_id}@test.com", "aal": "aal1",
        "exp": now + 86400, "iat": now, "is_anonymous": False,
    }, secret, algorithm="HS256")


TOKEN = _make_token()
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_state = {
    "search_session_id": None,
    "cursor": None,
    "current_page": 0,
    "query": "",
    "all_ids": set(),
    "page_data": [],  # list of (page_num, results, timing, pagination)
}


def _reset():
    _state.update({
        "search_session_id": None,
        "cursor": None,
        "current_page": 0,
        "query": "",
        "all_ids": set(),
        "page_data": [],
    })


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
.page-divider {
    border-top: 3px solid #6366f1;
    margin: 24px 0 12px 0;
    padding-top: 8px;
}
.page-header {
    font-size: 15px;
    font-weight: 700;
    color: #4338ca;
    margin-bottom: 8px;
}
.page-stats {
    font-size: 12px;
    color: #666;
    font-family: monospace;
    margin-bottom: 12px;
    background: #f5f3ff;
    padding: 6px 10px;
    border-radius: 6px;
    line-height: 1.6;
}
.product-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}
.product-card {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
    background: #fff;
    transition: box-shadow 0.2s;
}
.product-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.10);
}
.product-card img {
    width: 100%;
    height: 230px;
    object-fit: cover;
}
.product-card .no-img {
    width: 100%;
    height: 230px;
    background: #f3f4f6;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #9ca3af;
    font-size: 13px;
}
.product-info {
    padding: 8px 10px;
}
.product-name {
    font-size: 12px;
    font-weight: 600;
    line-height: 1.3;
    max-height: 32px;
    overflow: hidden;
    margin-bottom: 3px;
}
.product-brand {
    font-size: 11px;
    color: #6b7280;
}
.product-price {
    font-size: 13px;
    font-weight: 700;
    color: #111;
    margin-top: 2px;
}
.product-price .sale {
    color: #ef4444;
    font-size: 11px;
    margin-left: 4px;
}
.product-meta {
    font-size: 10px;
    color: #9ca3af;
    margin-top: 3px;
    font-family: monospace;
}
.summary-bar {
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 12px;
    font-size: 13px;
}
"""

# ---------------------------------------------------------------------------
# Card builder
# ---------------------------------------------------------------------------


def _product_card(i: int, p: dict, page: int) -> str:
    img = p.get("image_url") or ""
    if img:
        img_html = f'<img src="{img}" loading="lazy" alt="" />'
    else:
        img_html = '<div class="no-img">No image</div>'

    price_str = f"${p.get('price', 0):.2f}"
    sale_html = ""
    if p.get("is_on_sale") and p.get("original_price"):
        sale_html = f'<span class="sale">was ${p["original_price"]:.0f}</span>'

    cat = p.get("category_l1") or ""
    sim = p.get("semantic_score")
    sim_str = f" | sim={sim:.3f}" if sim else ""

    return f"""<div class="product-card">
  {img_html}
  <div class="product-info">
    <div class="product-name">{p.get('name','')[:60]}</div>
    <div class="product-brand">{p.get('brand','')}</div>
    <div class="product-price">{price_str} {sale_html}</div>
    <div class="product-meta">P{page} #{i+1} | {cat}{sim_str}</div>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# Search logic
# ---------------------------------------------------------------------------


def do_search(query: str) -> tuple:
    """Run page 1 search."""
    _reset()
    _state["query"] = query

    body = {"query": query, "page_size": 50}
    t0 = time.time()
    try:
        r = requests.post(f"{SEARCH_URL}/hybrid", json=body, headers=HEADERS, timeout=90)
        r.raise_for_status()
    except Exception as e:
        return f"<b>Error:</b> {e}", gr.update(interactive=False)

    wall_ms = int((time.time() - t0) * 1000)
    data = r.json()

    _state["search_session_id"] = data.get("search_session_id")
    _state["cursor"] = data.get("cursor")
    _state["current_page"] = 1

    results = data.get("results", [])
    ids = {p["product_id"] for p in results}
    _state["all_ids"] = ids
    _state["page_data"] = [(1, results, data.get("timing", {}), data.get("pagination", {}))]

    html = _render_all_pages(data.get("intent", "?"))
    has_more = data.get("pagination", {}).get("has_more", False)
    return html, gr.update(interactive=has_more and bool(_state["cursor"]))


def do_load_more() -> tuple:
    """Fetch next page."""
    if not _state["cursor"] or not _state["search_session_id"]:
        return _render_all_pages("?"), gr.update(interactive=False)

    next_page = _state["current_page"] + 1
    body = {
        "query": _state["query"],
        "page": next_page,
        "page_size": 50,
        "search_session_id": _state["search_session_id"],
        "cursor": _state["cursor"],
    }

    t0 = time.time()
    try:
        r = requests.post(f"{SEARCH_URL}/hybrid", json=body, headers=HEADERS, timeout=90)
        r.raise_for_status()
    except Exception as e:
        return f"<b>Error loading page {next_page}:</b> {e}", gr.update(interactive=False)

    data = r.json()

    _state["cursor"] = data.get("cursor")
    _state["current_page"] = next_page

    results = data.get("results", [])
    new_ids = {p["product_id"] for p in results}
    overlap = _state["all_ids"] & new_ids
    _state["all_ids"].update(new_ids)
    _state["page_data"].append((next_page, results, data.get("timing", {}), data.get("pagination", {})))

    intent = _state["page_data"][0][3].get("intent") if _state["page_data"] else "?"
    # Get intent from the first page's response
    if _state["page_data"]:
        # intent was stored in page 1 response
        pass

    html = _render_all_pages()
    has_more = data.get("pagination", {}).get("has_more", False)
    return html, gr.update(interactive=has_more and bool(_state["cursor"]))


def _render_all_pages(intent: str = None) -> str:
    """Render all fetched pages as HTML."""
    parts = []

    # Summary bar
    total_products = len(_state["all_ids"])
    total_pages = len(_state["page_data"])
    if _state["page_data"]:
        p1_pagination = _state["page_data"][0][3]
        total_results = p1_pagination.get("total_results", "?")
        if intent is None:
            # Try to get from first page timing or just use stored
            intent = "?"
            for _, _, timing, pag in _state["page_data"]:
                break
        p1_intent = _state["page_data"][0][3].get("intent") if "intent" in _state["page_data"][0][3] else intent
    else:
        total_results = "?"
        p1_intent = intent

    # Get intent from the response data
    if _state["page_data"]:
        # Check if we stored it
        first_timing = _state["page_data"][0][2]

    parts.append(
        f'<div class="summary-bar">'
        f'<b>{_state["query"]!r}</b> &mdash; '
        f'{total_products} unique products loaded across {total_pages} page(s) '
        f'&mdash; catalog total: {total_results}'
        f'</div>'
    )

    for page_num, results, timing, pagination in _state["page_data"]:
        total_ms = timing.get("total_ms", "?")
        rounds = timing.get("rounds", "-")
        is_endless = timing.get("endless_semantic", False)
        is_extend = timing.get("extend_search", False)
        seen_ids = timing.get("seen_ids_total", "?")

        if page_num == 1:
            mode = "full pipeline"
        elif is_endless:
            mode = f"endless semantic (rounds={rounds})"
        elif is_extend:
            mode = "extend search (Algolia)"
        else:
            mode = "unknown"

        # Check overlap
        if page_num > 1:
            page_ids = {p["product_id"] for p in results}
            prev_ids = set()
            for pn, pr, _, _ in _state["page_data"]:
                if pn < page_num:
                    prev_ids.update(p["product_id"] for p in pr)
            overlap = len(page_ids & prev_ids)
            overlap_str = f" | overlap={overlap}" if overlap else ""
        else:
            overlap_str = ""

        parts.append(f'<div class="page-divider"></div>')
        parts.append(f'<div class="page-header">Page {page_num} &mdash; {len(results)} results</div>')
        parts.append(
            f'<div class="page-stats">'
            f'mode: {mode} | time: {total_ms}ms | seen_ids: {seen_ids}{overlap_str}'
            f'</div>'
        )

        if results:
            cards = [_product_card(i, p, page_num) for i, p in enumerate(results)]
            parts.append(f'<div class="product-grid">{"".join(cards)}</div>')
        else:
            parts.append('<div style="color:#999;padding:12px;">No results</div>')

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Endless Semantic Search Demo",
    css=CSS,
    theme=gr.themes.Soft(),
) as app:
    gr.Markdown("## Endless Semantic Search Demo")
    gr.Markdown(
        "Search, then click **Load More** to paginate. "
        "SPECIFIC/VAGUE queries use the endless semantic pipeline (pgvector). "
        "EXACT queries use Algolia-native pagination."
    )

    with gr.Row():
        query_input = gr.Textbox(
            label="Search query",
            placeholder="e.g. black midi dress, summer vibes, tops...",
            scale=4,
        )
        search_btn = gr.Button("Search", variant="primary", scale=1)

    load_more_btn = gr.Button("Load More", variant="secondary", interactive=False)
    results_html = gr.HTML(value="<div style='color:#999;padding:20px;'>Enter a query and click Search</div>")

    search_btn.click(
        fn=do_search,
        inputs=[query_input],
        outputs=[results_html, load_more_btn],
    )
    query_input.submit(
        fn=do_search,
        inputs=[query_input],
        outputs=[results_html, load_more_btn],
    )
    load_more_btn.click(
        fn=do_load_more,
        inputs=[],
        outputs=[results_html, load_more_btn],
    )

app.launch(server_name="0.0.0.0", server_port=7680, share=False)
