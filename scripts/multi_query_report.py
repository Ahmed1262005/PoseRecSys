"""
Generate an HTML report showing multi-query semantic search results
WITH follow-up refinements.

For each query:
1. Run the initial hybrid search (shows semantic queries + follow-ups)
2. For each follow-up question, pick the FIRST option and call /refine
3. Show the refined results side-by-side

Usage: PYTHONPATH=src python scripts/multi_query_report.py
"""

import json
import time
import sys
import os
import html as html_mod

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jwt
import requests
from config.settings import get_settings

API_URL = "http://localhost:8000"
settings = get_settings()

# Generate auth token
TOKEN = jwt.encode(
    {
        "sub": "test-report-user",
        "aud": "authenticated",
        "role": "authenticated",
        "exp": int(time.time()) + 3600,
        "user_metadata": {},
    },
    settings.supabase_jwt_secret,
    algorithm="HS256",
)
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

QUERIES = [
    "outfit for a night out",
    "something cute for this weekend",
    "what to wear to a wedding",
    "something for work but not boring",
    "red midi dress",
    "i need new clothes",
]


def run_query(query: str) -> dict:
    """Run a query against the hybrid search API."""
    print(f'  Search: "{query}" ...', end="", flush=True)
    t0 = time.time()
    try:
        r = requests.post(
            f"{API_URL}/api/search/hybrid",
            json={"query": query, "page_size": 30},
            headers=HEADERS,
            timeout=120,
        )
        elapsed = time.time() - t0
        print(f" {elapsed:.1f}s")
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}", "query": query, "elapsed": elapsed}
        data = r.json()
        data["elapsed"] = elapsed
        return data
    except Exception as e:
        elapsed = time.time() - t0
        print(f" ERROR: {e}")
        return {"error": str(e), "query": query, "elapsed": elapsed}


def run_refine(query: str, selected_filters: dict, label: str) -> dict:
    """Run a refine query with selected follow-up filters."""
    print(f'    Refine [{label}] ...', end="", flush=True)
    t0 = time.time()
    try:
        r = requests.post(
            f"{API_URL}/api/search/refine",
            json={
                "original_query": query,
                "selected_filters": selected_filters,
                "page_size": 20,
            },
            headers=HEADERS,
            timeout=120,
        )
        elapsed = time.time() - t0
        print(f" {elapsed:.1f}s")
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}", "query": query, "elapsed": elapsed}
        data = r.json()
        data["elapsed"] = elapsed
        return data
    except Exception as e:
        elapsed = time.time() - t0
        print(f" ERROR: {e}")
        return {"error": str(e), "query": query, "elapsed": elapsed}


# =========================================================================
# HTML Building
# =========================================================================

CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f0f; color: #e0e0e0; padding: 20px; }
h1 { text-align: center; margin: 20px 0 30px; color: #fff; font-size: 28px; }
h1 small { display: block; font-size: 14px; color: #888; font-weight: normal; margin-top: 6px; }

.query-section { background: #1a1a1a; border-radius: 12px; margin-bottom: 40px; overflow: hidden; border: 1px solid #333; }
.query-header { padding: 18px 24px; background: #222; border-bottom: 1px solid #333; }
.query-text { font-size: 22px; font-weight: 600; color: #fff; }
.query-meta { display: flex; gap: 16px; margin-top: 10px; flex-wrap: wrap; }
.meta-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.intent-vague { background: #7c3aed22; color: #a78bfa; border: 1px solid #7c3aed44; }
.intent-specific { background: #05966922; color: #6ee7b7; border: 1px solid #05966944; }
.intent-exact { background: #2563eb22; color: #93c5fd; border: 1px solid #2563eb44; }
.meta-timing { color: #888; font-size: 13px; line-height: 28px; }

.semantic-queries { padding: 14px 24px; background: #1e1e2e; border-bottom: 1px solid #333; }
.semantic-queries h3 { font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
.sq-item { display: flex; gap: 8px; margin-bottom: 6px; align-items: flex-start; }
.sq-num { background: #7c3aed; color: #fff; width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; flex-shrink: 0; margin-top: 1px; }
.sq-text { font-size: 13px; color: #c4b5fd; font-style: italic; line-height: 1.5; }

.follow-ups { padding: 14px 24px; background: #1a2332; border-bottom: 1px solid #333; }
.follow-ups h3 { font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
.fu-question { margin-bottom: 12px; }
.fu-q-text { font-size: 14px; color: #93c5fd; font-weight: 600; margin-bottom: 6px; }
.fu-dim { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
.fu-options { display: flex; gap: 8px; flex-wrap: wrap; }
.fu-option { padding: 5px 14px; background: #2563eb22; border: 1px solid #2563eb44; border-radius: 20px; font-size: 12px; color: #93c5fd; }
.fu-option.selected { background: #2563eb; color: #fff; border-color: #2563eb; }

.section-label { padding: 12px 24px; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid #333; }
.section-label.original { background: #1a1a2e; color: #888; }
.section-label.refined { background: #1a2e1a; color: #6ee7b7; }
.refine-header { padding: 14px 24px; background: #132218; border-bottom: 1px solid #2d4a2d; }
.refine-title { font-size: 16px; font-weight: 600; color: #6ee7b7; }
.refine-desc { font-size: 13px; color: #888; margin-top: 4px; }
.refine-filters { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
.refine-chip { padding: 3px 10px; background: #6ee7b722; border: 1px solid #6ee7b744; border-radius: 12px; font-size: 11px; color: #6ee7b7; }
.refine-timing { font-size: 12px; color: #4a7a4a; margin-top: 6px; }

.results-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; padding: 14px 24px; }
.product-card { background: #222; border-radius: 8px; overflow: hidden; border: 1px solid #333; transition: border-color 0.2s; }
.product-card:hover { border-color: #7c3aed; }
.product-img { width: 100%; height: 240px; object-fit: cover; background: #333; }
.product-info { padding: 8px 10px; }
.product-name { font-size: 12px; font-weight: 600; color: #fff; line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.product-brand { font-size: 11px; color: #888; margin-top: 2px; }
.product-price { font-size: 13px; font-weight: 700; color: #6ee7b7; margin-top: 3px; }
.product-price .original { text-decoration: line-through; color: #666; font-weight: 400; font-size: 11px; margin-left: 4px; }
.product-tags { display: flex; gap: 3px; flex-wrap: wrap; margin-top: 4px; }
.product-tag { font-size: 9px; padding: 2px 5px; border-radius: 3px; background: #333; color: #aaa; }
.product-rank { font-size: 9px; color: #555; margin-top: 3px; }

.error-box { padding: 24px; color: #f87171; font-size: 16px; }
.no-results { padding: 24px; color: #888; font-size: 14px; text-align: center; }

.stats-bar { display: flex; gap: 20px; padding: 8px 24px; background: #161616; border-top: 1px solid #333; font-size: 11px; color: #666; flex-wrap: wrap; }
.stats-bar span { white-space: nowrap; }
.stats-bar strong { color: #aaa; }

.overlap-bar { padding: 8px 24px; background: #1a1a1a; border-top: 1px solid #333; font-size: 12px; color: #d97706; }

.divider { height: 2px; background: linear-gradient(90deg, #333 0%, #7c3aed 50%, #333 100%); margin: 0; }
"""


def _e(text: str) -> str:
    """HTML escape shorthand."""
    return html_mod.escape(str(text)) if text else ""


def build_product_cards(results: list, max_items: int = 30) -> str:
    """Build HTML product card grid."""
    if not results:
        return '<div class="no-results">No results found.</div>'
    cards = ""
    for i, p in enumerate(results[:max_items]):
        img = p.get("image_url") or ""
        name = p.get("name", "?")
        brand = p.get("brand", "?")
        price = p.get("price", 0)
        orig_price = p.get("original_price")
        cat = p.get("category_l1", "")
        article = p.get("article_type", "")
        source = p.get("source", "")

        price_html = f"${price:.0f}"
        if orig_price and orig_price > price:
            price_html += f' <span class="original">${orig_price:.0f}</span>'

        tags = [t for t in [cat, article, source] if t]
        tags_html = "".join(f'<span class="product-tag">{_e(t)}</span>' for t in tags)

        cards += f"""
<div class="product-card">
  <img class="product-img" src="{_e(img)}" alt="{_e(name)}" loading="lazy" onerror="this.style.display='none'">
  <div class="product-info">
    <div class="product-name">{_e(name)}</div>
    <div class="product-brand">{_e(brand)}</div>
    <div class="product-price">{price_html}</div>
    <div class="product-tags">{tags_html}</div>
    <div class="product-rank">#{i+1}</div>
  </div>
</div>"""
    return f'<div class="results-grid">{cards}</div>'


def build_stats_bar(results: list) -> str:
    """Brand + category stats."""
    brands, cats = {}, {}
    for r in results:
        b = r.get("brand", "?")
        brands[b] = brands.get(b, 0) + 1
        c = r.get("category_l1", "?")
        cats[c] = cats.get(c, 0) + 1
    top_brands = sorted(brands.items(), key=lambda x: -x[1])[:5]
    brands_str = ", ".join(f"{b} ({c})" for b, c in top_brands)
    cats_str = ", ".join(f"{c}: {n}" for c, n in cats.items())
    return f"""<div class="stats-bar">
  <span><strong>{len(brands)}</strong> brands: {brands_str}</span>
  <span>Categories: {cats_str}</span>
</div>"""


def build_query_section(data: dict, refinements: list) -> str:
    """Build HTML for one query + its refinements."""
    if "error" in data:
        return f"""
<div class="query-section">
  <div class="query-header">
    <div class="query-text">&ldquo;{_e(data.get('query', '?'))}&rdquo;</div>
  </div>
  <div class="error-box">Error: {_e(data['error'])}</div>
</div>"""

    query = data.get("query", "?")
    intent = data.get("intent", "?")
    timing = data.get("timing", {})
    results = data.get("results", [])
    follow_ups = data.get("follow_ups") or []

    total_ms = timing.get("total_ms", 0)
    planner_ms = timing.get("planner_ms", 0)
    algolia_ms = timing.get("algolia_ms", 0)
    semantic_ms = timing.get("semantic_ms", 0)
    sq_count = timing.get("semantic_query_count", 1)
    semantic_queries = timing.get("plan_semantic_queries") or []

    intent_cls = f"intent-{intent}"

    # -- Semantic queries --
    sq_html = ""
    if semantic_queries:
        sq_items = "".join(
            f'<div class="sq-item"><div class="sq-num">{i}</div><div class="sq-text">{_e(sq)}</div></div>\n'
            for i, sq in enumerate(semantic_queries, 1)
        )
        sq_html = f"""
<div class="semantic-queries">
  <h3>Semantic Queries ({len(semantic_queries)}) &mdash; each targets a different visual angle</h3>
  {sq_items}
</div>"""

    # -- Follow-ups with selected highlights --
    fu_html = ""
    if follow_ups:
        fu_items = ""
        for fu in follow_ups:
            opts_html = ""
            for j, o in enumerate(fu.get("options", [])):
                cls = "fu-option selected" if j == 0 else "fu-option"
                opts_html += f'<span class="{cls}">{_e(o["label"])}</span>'
            fu_items += f"""
<div class="fu-question">
  <div class="fu-dim">{_e(fu.get('dimension', ''))}</div>
  <div class="fu-q-text">{_e(fu.get('question', ''))}</div>
  <div class="fu-options">{opts_html}</div>
</div>"""
        fu_html = f"""
<div class="follow-ups">
  <h3>Follow-Up Questions ({len(follow_ups)}) &mdash; first option auto-selected for refinement</h3>
  {fu_items}
</div>"""

    # -- Original results --
    original_html = f"""
<div class="section-label original">Original Results ({len(results)})</div>
{build_product_cards(results)}
{build_stats_bar(results)}"""

    # -- Refinement results --
    refine_sections = ""
    for ref in refinements:
        ref_data = ref["data"]
        ref_label = ref["label"]
        ref_filters = ref["filters"]
        ref_question = ref["question"]
        ref_dimension = ref["dimension"]

        if "error" in ref_data:
            refine_sections += f"""
<div class="refine-header">
  <div class="refine-title">Refined: {_e(ref_label)}</div>
  <div class="error-box">Error: {_e(ref_data['error'])}</div>
</div>"""
            continue

        ref_results = ref_data.get("results", [])
        ref_timing = ref_data.get("timing", {})
        ref_elapsed = ref_data.get("elapsed", 0)

        # Compute overlap with original
        orig_ids = {r.get("product_id") for r in results}
        ref_ids = {r.get("product_id") for r in ref_results}
        overlap = orig_ids & ref_ids
        new_items = ref_ids - orig_ids
        overlap_pct = len(overlap) / max(len(ref_ids), 1) * 100

        # Filter chips
        chips = ""
        for k, v in ref_filters.items():
            val = json.dumps(v) if isinstance(v, (list, dict)) else str(v)
            chips += f'<span class="refine-chip">{_e(k)}: {_e(val)}</span>'

        refine_sections += f"""
<div class="divider"></div>
<div class="refine-header">
  <div class="refine-title">Refined: &ldquo;{_e(ref_label)}&rdquo;</div>
  <div class="refine-desc">{_e(ref_dimension)}: {_e(ref_question)}</div>
  <div class="refine-filters">{chips}</div>
  <div class="refine-timing">
    {ref_timing.get('total_ms', 0)}ms total &middot;
    {len(ref_results)} results &middot;
    {len(new_items)} new items &middot;
    {overlap_pct:.0f}% overlap with original
  </div>
</div>
{build_product_cards(ref_results, max_items=20)}
{build_stats_bar(ref_results)}
<div class="overlap-bar">{len(new_items)} new products not in original results &middot; {len(overlap)} shared &middot; {overlap_pct:.0f}% overlap</div>"""

    return f"""
<div class="query-section">
  <div class="query-header">
    <div class="query-text">&ldquo;{_e(query)}&rdquo;</div>
    <div class="query-meta">
      <span class="meta-badge {intent_cls}">{intent}</span>
      <span class="meta-timing">
        Total: <strong>{total_ms}ms</strong> &middot;
        Planner: <strong>{planner_ms}ms</strong> &middot;
        Algolia: <strong>{algolia_ms}ms</strong> &middot;
        Semantic: <strong>{semantic_ms}ms</strong> ({sq_count} queries) &middot;
        {len(results)} results
      </span>
    </div>
  </div>
  {sq_html}
  {fu_html}
  {original_html}
  {refine_sections}
</div>"""


def build_html(sections_html: list) -> str:
    """Build the full HTML document."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multi-Query Semantic Search + Follow-Up Refinement Report</title>
<style>{CSS}</style>
</head>
<body>
<h1>Multi-Query Semantic Search + Follow-Up Refinement
<small>Generated {time.strftime('%Y-%m-%d %H:%M:%S')} &mdash; Shows original results, then each follow-up option refined</small>
</h1>
{"".join(sections_html)}
</body>
</html>"""


# =========================================================================
# Main
# =========================================================================

def main():
    print("Multi-Query Search + Follow-Up Refinement Report")
    print(f"API: {API_URL}")
    print(f"Queries: {len(QUERIES)}")
    print()

    # Verify API
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        print(f"API health: {r.json()}")
    except Exception as e:
        print(f"ERROR: API not reachable: {e}")
        sys.exit(1)

    print()
    all_sections = []

    for query in QUERIES:
        print(f'--- "{query}" ---')
        data = run_query(query)

        refinements = []
        follow_ups = data.get("follow_ups") or []

        if follow_ups and "error" not in data:
            # For each follow-up question, select the FIRST option and run refine
            for fu in follow_ups:
                dimension = fu.get("dimension", "")
                question = fu.get("question", "")
                options = fu.get("options", [])

                if not options:
                    continue
                opt = options[0]  # Pick first option only
                label = opt.get("label", "?")
                filters = opt.get("filters", {})
                if not filters:
                    # Try second option if first has empty filters
                    if len(options) > 1:
                        opt = options[1]
                        label = opt.get("label", "?")
                        filters = opt.get("filters", {})
                if not filters:
                    continue

                ref_data = run_refine(query, dict(filters), label)
                refinements.append({
                    "label": label,
                    "filters": filters,
                    "question": question,
                    "dimension": dimension,
                    "data": ref_data,
                })

        section = build_query_section(data, refinements)
        all_sections.append(section)
        print()

    html = build_html(all_sections)
    out_path = os.path.join(os.path.dirname(__file__), "multi_query_report.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Report written to: {out_path}")
    print(f"Open: file://{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
