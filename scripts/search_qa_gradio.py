"""
Search QA Gradio — Full onboarding + search + follow-up interactive UI.

Features:
- Onboarding: age slider, brand cluster selection (22 clusters with names),
  price range, modesty preference
- Search: query input, product grid with images, timing, intent, facets
- Follow-ups: clickable buttons that refine the search via /api/search/refine
- Session state: planner_context built from onboarding, carried across searches

Usage:
    1. Start the API server:
       PYTHONPATH=src uvicorn api.app:app --port 8000

    2. Run this script:
       PYTHONPATH=src python scripts/search_qa_gradio.py

    3. Open http://localhost:7862 in your browser
"""

import os
import sys
import time
import json
import uuid
import requests
import gradio as gr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL = os.getenv("API_URL", "http://localhost:8000")
SEARCH_URL = f"{API_URL}/api/search"
PORT = int(os.getenv("GRADIO_PORT", "7862"))

# Make sure src/ is importable for JWT generation + brand clusters
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _make_token(user_id: str = "test-qa-user") -> str:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    import jwt as pyjwt

    secret = os.getenv("SUPABASE_JWT_SECRET")
    now = int(time.time())
    return pyjwt.encode(
        {
            "sub": user_id,
            "aud": "authenticated",
            "role": "authenticated",
            "email": f"{user_id}@test.com",
            "aal": "aal1",
            "exp": now + 86400,
            "iat": now,
            "is_anonymous": False,
        },
        secret,
        algorithm="HS256",
    )


TOKEN = _make_token()
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Brand cluster data (imported from the codebase)
# ---------------------------------------------------------------------------

from recs.brand_clusters import CLUSTER_TRAITS, BRAND_CLUSTER_MAP, BRAND_TYPE_TO_CLUSTERS

# Build human-readable cluster info for the onboarding UI
CLUSTER_INFO = {}
for cid, traits in sorted(CLUSTER_TRAITS.items()):
    brands_in_cluster = sorted(
        {b for b, (c, _) in BRAND_CLUSTER_MAP.items() if c == cid},
        key=str.lower,
    )
    CLUSTER_INFO[cid] = {
        "name": traits.name,
        "description": traits.description,
        "icp_age": traits.icp_age,
        "price_range": traits.typical_price_range,
        "price_tier": traits.price_tier,
        "palette": traits.palette,
        "revealing": traits.revealing,
        "formality": traits.formality,
        "brands": brands_in_cluster,
    }


# Map age group value from slider to the label format the planner expects
def _age_to_group(age: int) -> str:
    if age < 18:
        return "under-18"
    elif age < 25:
        return "18-24"
    elif age < 35:
        return "25-34"
    elif age < 45:
        return "35-44"
    elif age < 55:
        return "45-54"
    elif age < 65:
        return "55-64"
    else:
        return "65+"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ---- Onboarding sidebar ---- */
.cluster-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
}
.cluster-item {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 13px;
    background: #fafafa;
    cursor: pointer;
    transition: all 0.15s;
}
.cluster-item:hover {
    border-color: #667eea;
    background: #f0f2ff;
}
.cluster-item.selected {
    border-color: #667eea;
    background: #eef0ff;
    font-weight: 600;
}
.cluster-name {
    font-weight: 600;
    font-size: 13px;
}
.cluster-meta {
    color: #888;
    font-size: 11px;
    margin-top: 2px;
}
.cluster-brands {
    color: #666;
    font-size: 11px;
    margin-top: 3px;
    font-style: italic;
}

/* ---- Result cards ---- */
.results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 12px;
    padding: 8px 0;
}
.product-card {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
    background: #fff;
    transition: box-shadow 0.2s, transform 0.15s;
    cursor: default;
}
.product-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.10);
    transform: translateY(-2px);
}
.product-img {
    width: 100%;
    height: 240px;
    object-fit: cover;
    display: block;
}
.product-img-placeholder {
    width: 100%;
    height: 240px;
    background: #f3f4f6;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #9ca3af;
    font-size: 13px;
}
.product-info {
    padding: 8px 10px 10px;
}
.product-brand {
    font-size: 11px;
    text-transform: uppercase;
    color: #6b7280;
    letter-spacing: 0.5px;
    margin-bottom: 2px;
}
.product-name {
    font-size: 13px;
    font-weight: 500;
    color: #111;
    line-height: 1.3;
    margin-bottom: 4px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.product-price {
    font-size: 13px;
    font-weight: 600;
    color: #111;
}
.product-price-sale {
    color: #dc2626;
}
.product-price-original {
    text-decoration: line-through;
    color: #9ca3af;
    font-weight: 400;
    margin-left: 4px;
}
.product-meta {
    font-size: 10px;
    color: #9ca3af;
    margin-top: 3px;
}

/* ---- Follow-up questions ---- */
.followup-section {
    margin-top: 12px;
    padding: 14px;
    border: 1px solid #e0e7ff;
    border-radius: 10px;
    background: #f8f9ff;
}
.followup-question {
    font-size: 14px;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 8px;
}
.followup-options {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.followup-btn {
    padding: 6px 14px;
    border: 1px solid #c7d2fe;
    border-radius: 20px;
    background: #fff;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s;
}
.followup-btn:hover {
    background: #eef2ff;
    border-color: #818cf8;
}

/* ---- Timing / meta ---- */
.search-meta {
    display: flex;
    gap: 16px;
    padding: 8px 0;
    font-size: 12px;
    color: #6b7280;
    flex-wrap: wrap;
}
.search-meta-item {
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.search-meta-badge {
    background: #e0e7ff;
    color: #3730a3;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
}

/* ---- Profile summary ---- */
.profile-summary {
    padding: 10px 12px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    font-size: 12px;
    line-height: 1.6;
}
.profile-summary strong {
    color: #166534;
}
"""


# ---------------------------------------------------------------------------
# Helper: Build planner context from onboarding state
# ---------------------------------------------------------------------------

def build_planner_context(
    age: int,
    selected_clusters: list,
    min_price: float,
    max_price: float,
    modesty: str,
) -> dict:
    """Build the planner_context dict that would normally come from the user profile."""
    ctx = {}

    # Age group
    ctx["age_group"] = _age_to_group(age)

    # Brand clusters + descriptions
    if selected_clusters:
        ctx["brand_clusters"] = sorted(selected_clusters)
        descs = []
        for cid in sorted(selected_clusters):
            info = CLUSTER_INFO.get(cid)
            if info:
                descs.append(info["name"])
        if descs:
            ctx["cluster_descriptions"] = descs

    # Price range
    price_range = {}
    if min_price > 0:
        price_range["min"] = int(min_price)
    if max_price < 500:
        price_range["max"] = int(max_price)
    if price_range:
        ctx["price_range"] = price_range

    # Modesty
    if modesty and modesty != "No preference":
        ctx["modesty"] = modesty.lower()

    return ctx


# ---------------------------------------------------------------------------
# Helper: Format product card HTML
# ---------------------------------------------------------------------------

def _product_card_html(item: dict, idx: int) -> str:
    img_url = item.get("image_url") or ""
    brand = item.get("brand") or "Unknown"
    name = item.get("name") or "Product"
    price = item.get("price") or 0
    original_price = item.get("original_price")
    is_sale = item.get("is_on_sale", False)
    cat_l1 = item.get("category_l1") or ""
    cat_l2 = item.get("category_l2") or ""
    semantic_score = item.get("semantic_score") or item.get("rrf_score") or 0

    # Image
    if img_url:
        img_html = f'<img class="product-img" src="{img_url}" alt="{name}" loading="lazy">'
    else:
        img_html = '<div class="product-img-placeholder">No image</div>'

    # Price
    if is_sale and original_price and original_price > price:
        price_html = (
            f'<span class="product-price-sale">${price:.0f}</span>'
            f'<span class="product-price-original">${original_price:.0f}</span>'
        )
    else:
        price_html = f'${price:.0f}'

    # Meta
    meta_parts = []
    if cat_l1:
        meta_parts.append(cat_l1)
    if cat_l2:
        meta_parts.append(cat_l2)
    if semantic_score:
        meta_parts.append(f"score: {semantic_score:.3f}")
    meta_str = " | ".join(meta_parts)

    return f"""
    <div class="product-card">
        {img_html}
        <div class="product-info">
            <div class="product-brand">{brand}</div>
            <div class="product-name">{name}</div>
            <div class="product-price">{price_html}</div>
            <div class="product-meta">{meta_str}</div>
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Helper: Format follow-up questions HTML
# ---------------------------------------------------------------------------

def _followup_html(follow_ups: list) -> str:
    """Render follow-up questions as static HTML (for display only)."""
    if not follow_ups:
        return ""

    sections = []
    for fu in follow_ups:
        question = fu.get("question", "")
        sections.append(f"""
        <div class="followup-section">
            <div class="followup-question">{question}</div>
            <div style="font-size:12px; color:#6b7280; margin-top:2px;">
                Click an option below to refine your search
            </div>
        </div>
        """)

    return "\n".join(sections)


# Max follow-up slots: 4 questions x 5 options = 20 buttons
MAX_FU_QUESTIONS = 4
MAX_FU_OPTIONS = 5


# ---------------------------------------------------------------------------
# Helper: Format search meta HTML
# ---------------------------------------------------------------------------

def _search_meta_html(response: dict) -> str:
    intent = response.get("intent", "unknown")
    timing = response.get("timing", {})
    pagination = response.get("pagination", {})
    sort_by = response.get("sort_by", "relevance")

    parts = [
        f'<span class="search-meta-badge">{intent}</span>',
        f'<span class="search-meta-item">Sort: <b>{sort_by}</b></span>',
    ]

    total = pagination.get("total_results", 0)
    parts.append(f'<span class="search-meta-item">Results: <b>{total}</b></span>')

    # Timing breakdown
    for key in ["total_ms", "planner_ms", "algolia_ms", "semantic_ms", "reranker_ms"]:
        val = timing.get(key)
        if val is not None:
            label = key.replace("_ms", "").replace("_", " ").title()
            parts.append(f'<span class="search-meta-item">{label}: <b>{val}ms</b></span>')

    return f'<div class="search-meta">{"".join(parts)}</div>'


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def do_search(
    query: str,
    planner_context: dict,
    page: int = 1,
    page_size: int = 30,
    session_id: str = None,
) -> dict:
    """Call /api/search/hybrid. Plain search, no extra filters."""
    body = {
        "query": query,
        "page": page,
        "page_size": page_size,
        "planner_context": planner_context,
    }
    if session_id:
        body["session_id"] = session_id

    try:
        resp = requests.post(
            f"{SEARCH_URL}/hybrid",
            json=body,
            headers=HEADERS,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to API at {API_URL}. Is the server running?"}
    except Exception as e:
        return {"error": str(e)}


def do_refine(
    original_query: str,
    selected_filters: dict,
    page: int = 1,
    page_size: int = 30,
    session_id: str = None,
) -> dict:
    """Call /api/search/refine with follow-up filters."""
    body = {
        "original_query": original_query,
        "selected_filters": selected_filters,
        "page": page,
        "page_size": page_size,
    }
    if session_id:
        body["session_id"] = session_id

    try:
        resp = requests.post(
            f"{SEARCH_URL}/refine",
            json=body,
            headers=HEADERS,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to API at {API_URL}. Is the server running?"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def create_app():
    session_id = str(uuid.uuid4())

    with gr.Blocks(
        title="Search QA",
    ) as app:
        gr.Markdown("# Search QA — Onboarding + Search + Follow-ups")

        with gr.Row():
            # ===========================================================
            # LEFT SIDEBAR — Onboarding
            # ===========================================================
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("## Profile Setup")

                age_slider = gr.Slider(
                    minimum=14,
                    maximum=70,
                    value=28,
                    step=1,
                    label="Age",
                    info="Sets your age group for personalized follow-ups",
                )

                # Price range
                with gr.Row():
                    min_price = gr.Number(
                        value=0, label="Min Price ($)", minimum=0, maximum=5000,
                    )
                    max_price = gr.Number(
                        value=500, label="Max Price ($)", minimum=0, maximum=5000,
                    )

                modesty = gr.Radio(
                    choices=["No preference", "Modest", "Balanced", "Revealing ok"],
                    value="No preference",
                    label="Coverage Preference",
                )

                gr.Markdown("### Brand Style Clusters")
                gr.Markdown(
                    "*Select clusters that match your style. "
                    "This personalizes follow-up questions and scoring.*",
                )

                # Build cluster checkboxes with rich labels
                cluster_choices = []
                for cid in sorted(CLUSTER_INFO.keys()):
                    info = CLUSTER_INFO[cid]
                    pr = info["price_range"]
                    price_str = f"${pr[0]}-${pr[1]}" if pr[0] > 0 else ""
                    # Top 5 brands as preview
                    brand_preview = ", ".join(info["brands"][:5])
                    if len(info["brands"]) > 5:
                        brand_preview += f" +{len(info['brands'])-5} more"
                    label = f"[{cid}] {info['name']}"
                    cluster_choices.append(label)

                cluster_checkboxes = gr.CheckboxGroup(
                    choices=cluster_choices,
                    value=[],
                    label="",
                    info="",
                )

                # Cluster detail display
                cluster_detail = gr.HTML(value="<i>Select clusters above to see details</i>")

                # Profile summary
                profile_summary = gr.HTML(value="")

                def update_cluster_detail(selected):
                    if not selected:
                        return "<i>Select clusters above to see details</i>"
                    html_parts = []
                    for sel in selected:
                        cid = sel.split("]")[0].replace("[", "").strip()
                        info = CLUSTER_INFO.get(cid, {})
                        if not info:
                            continue
                        pr = info.get("price_range", (0, 0))
                        brands = ", ".join(info.get("brands", [])[:8])
                        if len(info.get("brands", [])) > 8:
                            brands += f" +{len(info['brands'])-8}"
                        html_parts.append(f"""
                        <div style="padding:6px 0; border-bottom:1px solid #eee;">
                            <b>[{cid}] {info.get('name','')}</b><br>
                            <span style="font-size:12px; color:#555;">{info.get('description','')}</span><br>
                            <span style="font-size:11px; color:#888;">
                                Age: {info.get('icp_age','')} | 
                                ${pr[0]}-${pr[1]} |
                                {info.get('price_tier','')} |
                                {info.get('formality','')}
                            </span><br>
                            <span style="font-size:11px; color:#667; font-style:italic;">
                                {brands}
                            </span>
                        </div>
                        """)
                    return "\n".join(html_parts) if html_parts else ""

                cluster_checkboxes.change(
                    fn=update_cluster_detail,
                    inputs=[cluster_checkboxes],
                    outputs=[cluster_detail],
                )

                def update_profile_summary(age, clusters, mn, mx, mod):
                    selected_ids = [
                        s.split("]")[0].replace("[", "").strip() for s in clusters
                    ]
                    ctx = build_planner_context(age, selected_ids, mn, mx, mod)
                    parts = [f"<b>Age group:</b> {ctx.get('age_group', '?')}"]
                    if ctx.get("brand_clusters"):
                        descs = ctx.get("cluster_descriptions", ctx["brand_clusters"])
                        parts.append(f"<b>Clusters:</b> {', '.join(descs)}")
                    if ctx.get("price_range"):
                        pr = ctx["price_range"]
                        parts.append(
                            f"<b>Price:</b> "
                            f"${pr.get('min', 0)}-${pr.get('max', '...')}"
                        )
                    if ctx.get("modesty"):
                        parts.append(f"<b>Coverage:</b> {ctx['modesty']}")
                    return (
                        '<div class="profile-summary">'
                        + "<br>".join(parts)
                        + "</div>"
                    )

                for comp in [age_slider, cluster_checkboxes, min_price, max_price, modesty]:
                    comp.change(
                        fn=update_profile_summary,
                        inputs=[age_slider, cluster_checkboxes, min_price, max_price, modesty],
                        outputs=[profile_summary],
                    )

            # ===========================================================
            # MAIN PANEL — Search + Results + Follow-ups
            # ===========================================================
            with gr.Column(scale=3):
                gr.Markdown("## Search")

                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="Try: 'quiet luxury', 'red midi dress', 'outfit for a wedding'...",
                        label="",
                        scale=5,
                        lines=1,
                    )
                    search_btn = gr.Button("Search", variant="primary", scale=1)

                # Meta info (timing, intent, etc.)
                meta_html = gr.HTML(value="")

                # Planner context display (collapsible)
                with gr.Accordion("Planner Context (sent to LLM)", open=False):
                    planner_ctx_display = gr.JSON(value={}, label="")

                # ---- Follow-ups FIRST (before results) ----
                gr.Markdown("### Follow-up Questions")
                followup_html = gr.HTML(value="")

                # State: follow-up data, original query, and accumulated selections
                fu_state = gr.State(value=[])           # list of {dimension, question, options}
                fu_original_query = gr.State(value="")
                fu_selections = gr.State(value={})      # {question_idx: [{label, filters}, ...]}

                # Pre-allocate a grid of buttons for follow-up options.
                # Each question gets a label + row of option buttons.
                fu_labels = []
                fu_buttons = []  # flat list of all buttons
                fu_button_map = []  # (question_idx, option_idx) per button

                for qi in range(MAX_FU_QUESTIONS):
                    fu_label = gr.Markdown(value="", visible=False)
                    fu_labels.append(fu_label)
                    with gr.Row():
                        for oi in range(MAX_FU_OPTIONS):
                            btn = gr.Button(
                                value="",
                                visible=False,
                                variant="secondary",
                                size="sm",
                                min_width=80,
                            )
                            fu_buttons.append(btn)
                            fu_button_map.append((qi, oi))

                # Summary of current selections + Apply button
                selection_summary = gr.HTML(value="")
                apply_btn = gr.Button(
                    "Search with selections",
                    variant="primary",
                    visible=False,
                )

                # ---- Results (after follow-ups) ----
                results_html = gr.HTML(value='<div style="color:#999; padding:40px; text-align:center;">Enter a query and click Search</div>')

                # Raw response (debug)
                with gr.Accordion("Raw API Response", open=False):
                    raw_json = gr.JSON(value={}, label="")

                # ===========================================================
                # Helpers: build button grid updates
                # ===========================================================

                def _build_button_updates(follow_ups, selections=None):
                    """Return gr.update() list for all labels + buttons.

                    selections is {qi_str: [{label, filters}, ...]} (multi-select).
                    Selected buttons show with primary variant; others secondary.
                    """
                    if selections is None:
                        selections = {}
                    updates = []

                    # Labels (MAX_FU_QUESTIONS)
                    for qi in range(MAX_FU_QUESTIONS):
                        if qi < len(follow_ups):
                            fu = follow_ups[qi]
                            question = fu.get("question", "")
                            sel_list = selections.get(str(qi), [])
                            if sel_list:
                                names = ", ".join(s["label"] for s in sel_list)
                                checkmark = f' -- *Selected: {names}*'
                            else:
                                checkmark = ""
                            updates.append(gr.update(
                                value=f"**{question}**{checkmark}",
                                visible=True,
                            ))
                        else:
                            updates.append(gr.update(value="", visible=False))

                    # Buttons (MAX_FU_QUESTIONS x MAX_FU_OPTIONS)
                    for qi in range(MAX_FU_QUESTIONS):
                        for oi in range(MAX_FU_OPTIONS):
                            if qi < len(follow_ups):
                                options = follow_ups[qi].get("options", [])
                                if oi < len(options):
                                    label = options[oi].get("label", "")
                                    sel_list = selections.get(str(qi), [])
                                    sel_labels = {s["label"] for s in sel_list}
                                    is_selected = label in sel_labels
                                    display = f">> {label} <<" if is_selected else label
                                    variant = "primary" if is_selected else "secondary"
                                    updates.append(gr.update(
                                        value=display,
                                        visible=True,
                                        variant=variant,
                                    ))
                                else:
                                    updates.append(gr.update(value="", visible=False))
                            else:
                                updates.append(gr.update(value="", visible=False))

                    return updates

                def _hide_all_buttons():
                    """Return updates to hide all labels + buttons."""
                    updates = []
                    for _ in range(MAX_FU_QUESTIONS):
                        updates.append(gr.update(value="", visible=False))
                    for _ in range(MAX_FU_QUESTIONS * MAX_FU_OPTIONS):
                        updates.append(gr.update(value="", visible=False))
                    return updates

                def _selection_summary_html(selections):
                    """Render the current selection summary (multi-select aware)."""
                    if not selections:
                        return ""
                    parts = []
                    for qi_str in sorted(selections.keys()):
                        sel_list = selections[qi_str]
                        if not sel_list:
                            continue
                        for sel in sel_list:
                            parts.append(
                                f'<span style="display:inline-block; background:#e0e7ff; '
                                f'color:#3730a3; padding:3px 10px; border-radius:12px; '
                                f'font-size:12px; margin:2px 4px;">'
                                f'{sel["label"]}</span>'
                            )
                    if not parts:
                        return ""
                    return (
                        '<div style="padding:8px 0;">'
                        '<span style="font-size:13px; font-weight:600; color:#374151;">'
                        'Selected: </span>'
                        + "".join(parts)
                        + '</div>'
                    )

                # All button-grid outputs: labels first, then buttons
                all_fu_outputs = fu_labels + fu_buttons

                # ===========================================================
                # Search callback
                # ===========================================================

                def on_search(query, age, clusters, mn, mx, mod):
                    empty_btns = _hide_all_buttons()
                    if not query.strip():
                        return [
                            "",
                            '<div style="color:#999; padding:20px;">Enter a query first</div>',
                            "",
                            {},
                            {},
                            [],   # fu_state
                            "",   # fu_original_query
                            {},   # fu_selections (reset)
                            "",   # selection_summary
                            gr.update(visible=False),  # apply_btn
                        ] + empty_btns

                    # Build planner context from onboarding
                    selected_ids = [
                        s.split("]")[0].replace("[", "").strip() for s in clusters
                    ]
                    ctx = build_planner_context(age, selected_ids, mn, mx, mod)

                    # Call API
                    response = do_search(
                        query=query.strip(),
                        planner_context=ctx,
                        page=1,
                        page_size=30,
                        session_id=session_id,
                    )

                    if "error" in response:
                        return [
                            f'<div style="color:red; padding:10px;">{response["error"]}</div>',
                            "",
                            "",
                            ctx,
                            response,
                            [],
                            "",
                            {},
                            "",
                            gr.update(visible=False),
                        ] + empty_btns

                    # Meta
                    meta = _search_meta_html(response)

                    # Results grid
                    results = response.get("results", [])
                    if results:
                        cards = [_product_card_html(r, i) for i, r in enumerate(results)]
                        grid = f'<div class="results-grid">{"".join(cards)}</div>'
                    else:
                        grid = '<div style="color:#999; padding:20px;">No results found</div>'

                    # Follow-ups
                    follow_ups = response.get("follow_ups") or []
                    fu_html = _followup_html(follow_ups)

                    # Fresh buttons (no selections yet)
                    btn_updates = _build_button_updates(follow_ups)

                    # Show apply button only if there are follow-ups
                    show_apply = gr.update(visible=bool(follow_ups))

                    return [
                        meta,
                        grid,
                        fu_html,
                        ctx,
                        response,
                        follow_ups,         # fu_state
                        query.strip(),      # fu_original_query
                        {},                 # fu_selections (reset)
                        "",                 # selection_summary (empty)
                        show_apply,         # apply_btn visibility
                    ] + btn_updates

                search_outputs = [
                    meta_html, results_html, followup_html,
                    planner_ctx_display, raw_json,
                    fu_state, fu_original_query,
                    fu_selections, selection_summary, apply_btn,
                ] + all_fu_outputs

                search_btn.click(
                    fn=on_search,
                    inputs=[query_input, age_slider, cluster_checkboxes, min_price, max_price, modesty],
                    outputs=search_outputs,
                )
                query_input.submit(
                    fn=on_search,
                    inputs=[query_input, age_slider, cluster_checkboxes, min_price, max_price, modesty],
                    outputs=search_outputs,
                )

                # ===========================================================
                # Follow-up button click: toggle selection (no search yet)
                # ===========================================================

                def make_toggle_handler(question_idx, option_idx):
                    """Factory: create a click handler that toggles selection (multi-select)."""

                    def on_click(fu_data, selections):
                        if not fu_data or question_idx >= len(fu_data):
                            return [selections, "", gr.update(visible=False)] + _hide_all_buttons()

                        fu = fu_data[question_idx]
                        options = fu.get("options", [])
                        if option_idx >= len(options):
                            return [selections, "", gr.update(visible=False)] + _hide_all_buttons()

                        opt = options[option_idx]
                        label = opt.get("label", "")
                        filters = opt.get("filters", {})
                        qi_key = str(question_idx)

                        # Multi-select toggle: add or remove this option from the list
                        new_selections = {k: list(v) for k, v in (selections or {}).items()}
                        current = new_selections.get(qi_key, [])
                        existing_labels = {s["label"] for s in current}

                        if label in existing_labels:
                            # Deselect: remove this option
                            current = [s for s in current if s["label"] != label]
                        else:
                            # Select: add this option
                            current.append({"label": label, "filters": filters})

                        if current:
                            new_selections[qi_key] = current
                        else:
                            new_selections.pop(qi_key, None)

                        # Rebuild button visuals with new selections
                        btn_updates = _build_button_updates(fu_data, new_selections)
                        summary = _selection_summary_html(new_selections)
                        has_any = any(bool(v) for v in new_selections.values())
                        show_apply = gr.update(visible=has_any)

                        return [new_selections, summary, show_apply] + btn_updates

                    return on_click

                # Outputs for toggle: selections state + summary + apply visibility + all buttons
                toggle_outputs = [fu_selections, selection_summary, apply_btn] + all_fu_outputs

                for btn_idx, btn in enumerate(fu_buttons):
                    qi, oi = fu_button_map[btn_idx]
                    handler = make_toggle_handler(qi, oi)
                    btn.click(
                        fn=handler,
                        inputs=[fu_state, fu_selections],
                        outputs=toggle_outputs,
                    )

                # ===========================================================
                # Apply: full NEW search with merged follow-up filters
                # ===========================================================

                def _merge_multi_selections(selections):
                    """Merge multi-select filters into a single dict.

                    Within the same question: union list values, min of min_prices,
                    max of max_prices. Across questions: later keys overwrite.
                    Returns (merged_filters: dict, all_labels: list[str]).
                    """
                    merged = {}
                    all_labels = []

                    for qi_key in sorted(selections.keys()):
                        sel_list = selections[qi_key]
                        if not sel_list:
                            continue

                        # First: union all filters within this question
                        q_merged = {}
                        for sel in sel_list:
                            all_labels.append(sel["label"])
                            for fk, fv in sel.get("filters", {}).items():
                                if fk == "min_price":
                                    # Widest floor: take the smallest min_price
                                    cur = q_merged.get("min_price")
                                    val = float(fv) if fv is not None else None
                                    if val is not None:
                                        q_merged["min_price"] = min(cur, val) if cur is not None else val
                                elif fk == "max_price":
                                    # Widest ceiling: take the largest max_price
                                    cur = q_merged.get("max_price")
                                    val = float(fv) if fv is not None else None
                                    if val is not None:
                                        q_merged["max_price"] = max(cur, val) if cur is not None else val
                                elif isinstance(fv, list):
                                    # Union list values (deduplicated, order preserved)
                                    existing = q_merged.get(fk, [])
                                    seen = set(existing)
                                    for v in fv:
                                        if v not in seen:
                                            existing.append(v)
                                            seen.add(v)
                                    q_merged[fk] = existing
                                else:
                                    q_merged[fk] = fv

                        # Merge this question's filters into the overall dict
                        # (across questions: later keys overwrite)
                        merged.update(q_merged)

                    return merged, all_labels

                def on_apply(fu_data, orig_query, selections, age, clusters, mn, mx, mod):
                    empty_btns = _hide_all_buttons()
                    if not selections or not orig_query:
                        return [
                            "",
                            '<div style="color:#999;">No selections to apply</div>',
                            "",
                            {},
                            {},
                            [],
                            orig_query,
                            {},
                            "",
                            gr.update(visible=False),
                        ] + empty_btns

                    # Merge multi-select filters
                    merged_filters, labels_used = _merge_multi_selections(selections)

                    # Extract concise keywords from merged filter values
                    keywords = []
                    _CAT_WORD = {
                        "Dresses": "dress", "Tops": "top", "Bottoms": "pants",
                        "Outerwear": "jacket", "Activewear": "activewear",
                    }

                    for fk, fv in merged_filters.items():
                        if fk == "modes":
                            if isinstance(fv, list):
                                for m in fv:
                                    if m in ("modest", "revealing", "formal", "casual"):
                                        keywords.append(m)
                            continue
                        if not fv:
                            continue
                        if fk in ("min_price", "max_price"):
                            continue  # Don't add price as a keyword
                        if fk == "category_l1" and isinstance(fv, list):
                            for cat in fv:
                                keywords.append(_CAT_WORD.get(cat, cat.lower()))
                        elif fk == "formality" and isinstance(fv, list):
                            keywords.append(fv[0].lower())
                        elif fk == "occasions" and isinstance(fv, list):
                            keywords.append(f"for {fv[0].lower()}")
                        elif fk == "colors" and isinstance(fv, list):
                            keywords.extend(c.lower() for c in fv[:2])
                        elif fk == "length" and isinstance(fv, list):
                            keywords.append(fv[0].lower())
                        elif fk == "materials" and isinstance(fv, list):
                            keywords.append(fv[0].lower())
                        elif fk == "patterns" and isinstance(fv, list):
                            keywords.append(fv[0].lower())
                        elif fk == "style_tags" and isinstance(fv, list):
                            keywords.append(fv[0].lower())
                        elif isinstance(fv, list) and fv:
                            keywords.append(fv[0].lower())

                    # Dedupe and build concise query
                    seen = set()
                    unique_kw = []
                    for kw in keywords:
                        if kw not in seen and kw not in orig_query.lower():
                            seen.add(kw)
                            unique_kw.append(kw)

                    if unique_kw:
                        new_query = f"{' '.join(unique_kw)} {orig_query}"
                    else:
                        new_query = orig_query

                    # Build planner context from onboarding (same as initial search)
                    selected_ids = [
                        s.split("]")[0].replace("[", "").strip() for s in clusters
                    ]
                    ctx = build_planner_context(age, selected_ids, mn, mx, mod)

                    # Full fresh search — no extra filters, just the enriched query
                    response = do_search(
                        query=new_query,
                        planner_context=ctx,
                        page=1,
                        page_size=30,
                        session_id=session_id,
                    )

                    if "error" in response:
                        return [
                            f'<div style="color:red;">{response["error"]}</div>',
                            "",
                            "",
                            ctx,
                            response,
                            fu_data,
                            orig_query,
                            selections,
                            _selection_summary_html(selections),
                            gr.update(visible=True),
                        ] + _build_button_updates(fu_data, selections)

                    # Meta
                    meta = _search_meta_html(response)
                    meta += (
                        f'<div style="font-size:12px; color:#059669; margin-top:4px;">'
                        f'Answers: <b>{" + ".join(labels_used)}</b></div>'
                        f'<div style="font-size:12px; color:#374151; margin-top:2px;">'
                        f'New query: <b>{new_query}</b></div>'
                    )

                    # Results grid
                    results = response.get("results", [])
                    if results:
                        cards = [_product_card_html(r, i) for i, r in enumerate(results)]
                        grid = f'<div class="results-grid">{"".join(cards)}</div>'
                    else:
                        grid = '<div style="color:#999; padding:20px;">No results found</div>'

                    # New follow-ups from the fresh search
                    new_follow_ups = response.get("follow_ups") or []
                    fu_html = _followup_html(new_follow_ups)
                    btn_updates = _build_button_updates(new_follow_ups)

                    return [
                        meta, grid, fu_html, ctx, response,
                        new_follow_ups,     # new fu_state
                        orig_query,         # keep original query
                        {},                 # reset selections
                        "",                 # clear summary
                        gr.update(visible=bool(new_follow_ups)),
                    ] + btn_updates

                apply_outputs = [
                    meta_html, results_html, followup_html,
                    planner_ctx_display, raw_json,
                    fu_state, fu_original_query,
                    fu_selections, selection_summary, apply_btn,
                ] + all_fu_outputs

                apply_btn.click(
                    fn=on_apply,
                    inputs=[
                        fu_state, fu_original_query, fu_selections,
                        age_slider, cluster_checkboxes, min_price, max_price, modesty,
                    ],
                    outputs=apply_outputs,
                )

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = create_app()
    print(f"\n  Search QA Gradio starting on http://localhost:{PORT}\n")
    app.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
    )
