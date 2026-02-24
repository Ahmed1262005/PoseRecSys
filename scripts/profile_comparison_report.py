"""
Generate an HTML report comparing how different user profiles get
different personalized follow-up questions for the SAME queries.

For each query, runs it once per profile and shows:
- The planner context injected
- Follow-up questions + option ordering (personalized per profile)
- Product results (top 15)

Usage:
  1. Start the API server:  PYTHONPATH=src uvicorn api.app:app --port 8000
  2. Run this script:       PYTHONPATH=src python scripts/profile_comparison_report.py

Output: scripts/profile_comparison_report.html
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

TOKEN = jwt.encode(
    {
        "sub": "test-profile-report",
        "aud": "authenticated",
        "role": "authenticated",
        "exp": int(time.time()) + 7200,
        "user_metadata": {},
    },
    settings.supabase_jwt_secret,
    algorithm="HS256",
)
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# =========================================================================
# Profiles — synthetic user contexts for the LLM planner
# =========================================================================

PROFILES = [
    {
        "id": "no_profile",
        "label": "No Profile (Baseline)",
        "color": "#888888",
        "description": "Anonymous user — no onboarding data. Default follow-up ordering.",
        "planner_context": None,  # No context → baseline
    },
    {
        "id": "gen_z_trendy",
        "label": "Gen-Z Trendy",
        "color": "#f472b6",
        "description": "20yo, loves Boohoo/PrettyLittleThing, party outfits, budget-conscious.",
        "planner_context": {
            "age_group": "gen_z",
            "style_persona": ["trendy", "going-out"],
            "cluster_descriptions": [
                "Ultra-Fast Fashion / High-Trend",
                "Trendy Feminine / Going-Out Elevated",
            ],
            "price_range": {"min": 10, "max": 50},
            "brand_openness": "open",
            "modesty": "balanced",
        },
    },
    {
        "id": "mid_career_polished",
        "label": "Mid-Career Polished",
        "color": "#6ee7b7",
        "description": "38yo, Reiss/Theory/Veronica Beard, office-to-dinner, premium budget, covered.",
        "planner_context": {
            "age_group": "mid_career",
            "style_persona": ["polished", "tailored"],
            "cluster_descriptions": [
                "Modern Classics / Elevated Everyday",
                "Premium Contemporary Designer",
                "Premium Contemporary Staples / Quiet-Lux",
            ],
            "price_range": {"min": 80, "max": 400},
            "brand_openness": "selective",
            "modesty": "covered",
        },
    },
]

# Shared queries — same query across all profiles to compare follow-ups
SHARED_QUERIES = [
    "outfit for a night out",
    "something for work but not boring",
    "what to wear to a wedding",
]


def run_query(query: str, planner_context=None) -> dict:
    """Run a query with optional planner_context override."""
    t0 = time.time()
    try:
        body = {"query": query, "page_size": 30}
        if planner_context:
            body["planner_context"] = planner_context
        r = requests.post(
            f"{API_URL}/api/search/hybrid",
            json=body,
            headers=HEADERS,
            timeout=120,
        )
        elapsed = time.time() - t0
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}", "query": query, "elapsed": elapsed}
        data = r.json()
        data["elapsed"] = elapsed
        return data
    except Exception as e:
        elapsed = time.time() - t0
        return {"error": str(e), "query": query, "elapsed": elapsed}


# =========================================================================
# HTML
# =========================================================================

CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f0f; color: #e0e0e0; padding: 20px; }
h1 { text-align: center; margin: 20px 0 10px; color: #fff; font-size: 28px; }
h1 small { display: block; font-size: 14px; color: #888; font-weight: normal; margin-top: 6px; }
h2 { font-size: 22px; color: #fff; margin: 40px 0 20px; padding-bottom: 10px; border-bottom: 2px solid #333; }
h2 .query-label { color: #a78bfa; }

.profiles-legend { display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; margin: 16px 0 30px; }
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 13px; }
.legend-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
.legend-desc { color: #888; font-size: 11px; }

.comparison-row { display: grid; gap: 16px; margin-bottom: 40px; }
.comparison-row.cols-2 { grid-template-columns: repeat(2, 1fr); }
.comparison-row.cols-3 { grid-template-columns: repeat(3, 1fr); }
.comparison-row.cols-4 { grid-template-columns: repeat(4, 1fr); }
.comparison-row.cols-5 { grid-template-columns: repeat(5, 1fr); }

.profile-card { background: #1a1a1a; border-radius: 12px; overflow: hidden; border: 2px solid #333; }
.profile-header { padding: 12px 16px; border-bottom: 1px solid #333; }
.profile-name { font-size: 15px; font-weight: 700; }
.profile-desc { font-size: 11px; color: #888; margin-top: 2px; }
.profile-context { font-size: 10px; color: #555; margin-top: 4px; font-family: monospace; line-height: 1.4; white-space: pre-wrap; word-break: break-all; }

.meta-row { display: flex; gap: 8px; padding: 8px 16px; background: #161616; border-bottom: 1px solid #333; flex-wrap: wrap; }
.meta-badge { display: inline-block; padding: 3px 10px; border-radius: 16px; font-size: 11px; font-weight: 600; }
.intent-vague { background: #7c3aed22; color: #a78bfa; border: 1px solid #7c3aed44; }
.intent-specific { background: #05966922; color: #6ee7b7; border: 1px solid #05966944; }
.intent-exact { background: #2563eb22; color: #93c5fd; border: 1px solid #2563eb44; }
.meta-timing { color: #666; font-size: 11px; }

.sq-section { padding: 10px 16px; background: #1e1e2e; border-bottom: 1px solid #333; }
.sq-section h4 { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.sq-text { font-size: 11px; color: #c4b5fd; font-style: italic; margin-bottom: 3px; padding-left: 8px; border-left: 2px solid #7c3aed44; }

.fu-section { padding: 10px 16px; background: #1a2332; border-bottom: 1px solid #333; }
.fu-section h4 { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
.fu-question { margin-bottom: 10px; }
.fu-dim { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
.fu-q-text { font-size: 12px; color: #93c5fd; font-weight: 600; margin: 2px 0 4px; }
.fu-options { display: flex; gap: 4px; flex-wrap: wrap; }
.fu-option { padding: 3px 10px; background: #2563eb22; border: 1px solid #2563eb44; border-radius: 16px; font-size: 10px; color: #93c5fd; }
.fu-option:first-child { background: #2563eb; color: #fff; border-color: #2563eb; font-weight: 600; }

.results-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 6px; padding: 10px 16px; }
.product-card { background: #222; border-radius: 6px; overflow: hidden; border: 1px solid #333; }
.product-img { width: 100%; height: 160px; object-fit: cover; background: #333; }
.product-info { padding: 5px 7px; }
.product-name { font-size: 10px; font-weight: 600; color: #fff; line-height: 1.2; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.product-brand { font-size: 9px; color: #888; margin-top: 1px; }
.product-price { font-size: 11px; font-weight: 700; color: #6ee7b7; margin-top: 2px; }
.product-rank { font-size: 8px; color: #444; }

.stats-bar { padding: 6px 16px; background: #161616; border-top: 1px solid #333; font-size: 10px; color: #555; }
.stats-bar strong { color: #888; }

.error-box { padding: 16px; color: #f87171; font-size: 13px; }
.no-results { padding: 16px; color: #888; font-size: 12px; text-align: center; }

.diff-highlight { background: #fbbf2422; border: 1px solid #fbbf2444; border-radius: 4px; padding: 2px 4px; }
"""


def _e(text) -> str:
    return html_mod.escape(str(text)) if text else ""


def build_product_cards(results: list, max_items: int = 15) -> str:
    if not results:
        return '<div class="no-results">No results</div>'
    cards = ""
    for i, p in enumerate(results[:max_items]):
        img = p.get("image_url") or ""
        name = p.get("name", "?")
        brand = p.get("brand", "?")
        price = p.get("price", 0)
        cards += f"""<div class="product-card">
  <img class="product-img" src="{_e(img)}" loading="lazy" onerror="this.style.display='none'">
  <div class="product-info">
    <div class="product-name">{_e(name)}</div>
    <div class="product-brand">{_e(brand)}</div>
    <div class="product-price">${price:.0f}</div>
    <div class="product-rank">#{i+1}</div>
  </div>
</div>"""
    return f'<div class="results-grid">{cards}</div>'


def build_stats(results: list) -> str:
    brands, cats = {}, {}
    for r in results:
        b = r.get("brand", "?")
        brands[b] = brands.get(b, 0) + 1
        c = r.get("category_l1", "?")
        cats[c] = cats.get(c, 0) + 1
    top_b = sorted(brands.items(), key=lambda x: -x[1])[:4]
    b_str = ", ".join(f"{b}({c})" for b, c in top_b)
    c_str = ", ".join(f"{c}:{n}" for c, n in cats.items())
    return f'<div class="stats-bar"><strong>{len(brands)}</strong> brands: {b_str} | {c_str}</div>'


def build_profile_card(profile: dict, data: dict) -> str:
    """Build one profile's result card for a query."""
    pid = profile["id"]
    color = profile["color"]
    label = profile["label"]
    desc = profile["description"]
    ctx = profile["planner_context"]

    header = f"""<div class="profile-header" style="border-top: 3px solid {color};">
  <div class="profile-name" style="color: {color};">{_e(label)}</div>
  <div class="profile-desc">{_e(desc)}</div>"""

    if ctx:
        ctx_str = json.dumps(ctx, indent=1, ensure_ascii=False)
        # Truncate for display
        if len(ctx_str) > 300:
            ctx_str = ctx_str[:300] + "..."
        header += f'\n  <div class="profile-context">{_e(ctx_str)}</div>'
    else:
        header += '\n  <div class="profile-context">No context (anonymous)</div>'
    header += "\n</div>"

    if "error" in data:
        return f'<div class="profile-card">{header}<div class="error-box">{_e(data["error"])}</div></div>'

    intent = data.get("intent", "?")
    timing = data.get("timing", {})
    results = data.get("results", [])
    follow_ups = data.get("follow_ups") or []
    sq = timing.get("plan_semantic_queries") or []

    # Meta row
    meta = f"""<div class="meta-row">
  <span class="meta-badge intent-{intent}">{intent}</span>
  <span class="meta-timing">{timing.get('total_ms', 0)}ms | {timing.get('planner_ms', 0)}ms planner | {len(results)} results</span>
</div>"""

    # Semantic queries
    sq_html = ""
    if sq:
        sq_items = "\n".join(f'<div class="sq-text">{_e(s)}</div>' for s in sq)
        sq_html = f'<div class="sq-section"><h4>Semantic Queries ({len(sq)})</h4>{sq_items}</div>'

    # Follow-ups
    fu_html = ""
    if follow_ups:
        fu_items = ""
        for fu in follow_ups:
            opts = "".join(f'<span class="fu-option">{_e(o.get("label", "?"))}</span>' for o in fu.get("options", []))
            fu_items += f"""<div class="fu-question">
  <div class="fu-dim">{_e(fu.get('dimension', ''))}</div>
  <div class="fu-q-text">{_e(fu.get('question', ''))}</div>
  <div class="fu-options">{opts}</div>
</div>"""
        fu_html = f'<div class="fu-section"><h4>Follow-Up Questions ({len(follow_ups)})</h4>{fu_items}</div>'
    else:
        fu_html = '<div class="fu-section"><h4>Follow-Up Questions</h4><div class="no-results">None generated</div></div>'

    return f"""<div class="profile-card">
{header}
{meta}
{sq_html}
{fu_html}
{build_product_cards(results)}
{build_stats(results)}
</div>"""


def build_html(query_sections: list) -> str:
    n = len(PROFILES)
    legend_items = "".join(
        f'<div class="legend-item"><div class="legend-dot" style="background:{p["color"]};"></div>'
        f'<span>{_e(p["label"])}</span><span class="legend-desc">{_e(p["description"][:60])}</span></div>'
        for p in PROFILES
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Personalized Follow-Up Comparison Report</title>
<style>{CSS}</style>
</head>
<body>
<h1>Personalized Follow-Up Comparison
<small>Same query, different profiles &rarr; different follow-up questions &amp; option ordering</small>
<small>Generated {time.strftime('%Y-%m-%d %H:%M:%S')} &mdash; {len(SHARED_QUERIES)} queries &times; {n} profiles = {len(SHARED_QUERIES) * n} searches</small>
</h1>
<div class="profiles-legend">{legend_items}</div>
{"".join(query_sections)}
</body>
</html>"""


# =========================================================================
# Main
# =========================================================================

def main():
    n_profiles = len(PROFILES)
    n_queries = len(SHARED_QUERIES)
    total = n_queries * n_profiles
    out_path = os.path.join(os.path.dirname(__file__), "profile_comparison_report.html")

    print("=" * 60)
    print("Personalized Follow-Up Comparison Report")
    print(f"API: {API_URL}")
    print(f"Profiles: {n_profiles}")
    print(f"Queries: {n_queries}")
    print(f"Total searches: {total}")
    print("=" * 60)
    print()

    # Verify API
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        print(f"API health: {r.json()}")
    except Exception as e:
        print(f"ERROR: API not reachable at {API_URL}: {e}")
        sys.exit(1)

    print()
    all_sections = []
    done = 0

    for qi, query in enumerate(SHARED_QUERIES):
        print(f'=== Query {qi+1}/{n_queries}: "{query}" ===')
        cards = []

        for profile in PROFILES:
            pid = profile["id"]
            ctx = profile["planner_context"]
            print(f'  [{pid}] searching...', end="", flush=True)
            data = run_query(query, planner_context=ctx)
            elapsed = data.get("elapsed", 0)
            n_results = len(data.get("results", []))
            n_fu = len(data.get("follow_ups") or [])
            print(f" {elapsed:.1f}s, {n_results} results, {n_fu} follow-ups")
            cards.append(build_profile_card(profile, data))
            done += 1

        cols_cls = f"cols-{n_profiles}" if n_profiles <= 5 else "cols-3"
        section = f"""<h2><span class="query-label">Query {qi+1}:</span> &ldquo;{_e(query)}&rdquo;</h2>
<div class="comparison-row {cols_cls}">
{"".join(cards)}
</div>"""
        all_sections.append(section)

        # Write incrementally so partial results are available
        html = build_html(all_sections)
        with open(out_path, "w") as f:
            f.write(html)
        print(f"  >> Saved ({qi+1}/{n_queries} queries done)")
        print()

    print("=" * 60)
    print(f"Report written to: {out_path}")
    print(f"Open: file://{os.path.abspath(out_path)}")
    print(f"Total searches: {done}")
    print("=" * 60)


if __name__ == "__main__":
    main()
