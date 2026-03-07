"""
Generate an HTML report showing vibe-brand query results.

Tests queries like:
  - "Like Zara but better quality"
  - "Boho maxi dress like Anthropologie"
  - "Like Aritzia vibe basics"
  - "Antropologie style tops" (misspelled)
  - "Ba&sh dress" (exact brand — control)

Runs the full hybrid search pipeline (planner → Algolia + FashionCLIP → RRF → reranker)
and shows the planner output + result cards for each query.

Run:
    PYTHONPATH=src python scripts/vibe_brand_report.py
"""
import os, sys, time, html as html_mod

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dotenv import load_dotenv
load_dotenv()

from search.hybrid_search import get_hybrid_search_service
from search.models import HybridSearchRequest

service = get_hybrid_search_service()

# ── Queries to test ──────────────────────────────────────────────────────
QUERIES = [
    {
        "query": "Like Zara but better quality",
        "description": "Vibe brand + quality upgrade. Should set vibe_brand=Zara, min_price raised, "
                       "boost Cluster A brands (COS, Massimo Dutti, & Other Stories, Reformation).",
        "expect_vibe": "Zara",
    },
    {
        "query": "Boho maxi dress like Anthropologie",
        "description": "Vibe brand + specific item. Should set vibe_brand=Anthropologie, "
                       "boho modes, Maxi+Dresses, boost Cluster M brands (Free People, Spell).",
        "expect_vibe": "Anthropologie",
    },
    {
        "query": "Like Aritzia vibe basics",
        "description": "Vibe brand + category. Should set vibe_brand=Aritzia, "
                       "quiet-lux/minimalist, boost Cluster K brands (Theory, Vince, AllSaints).",
        "expect_vibe": "Aritzia",
    },
    {
        "query": "Antropologie style tops",
        "description": "Misspelled vibe brand. LLM should correct to Anthropologie, "
                       "set vibe_brand=Anthropologie, boho/bohemian modes, Tops only.",
        "expect_vibe": "Anthropologie",
    },
    {
        "query": "Cheaper alternative to Reformation",
        "description": "Vibe brand + budget modifier. Should set vibe_brand=Reformation, "
                       "max_price capped, boost same-aesthetic lower-tier brands.",
        "expect_vibe": "Reformation",
    },
    {
        "query": "Ba&sh dress",
        "description": "CONTROL: Exact brand search. Should set brand=Ba&sh (hard filter), "
                       "vibe_brand=None. Only Ba&sh products returned.",
        "expect_vibe": None,
    },
]

# ── Run searches ─────────────────────────────────────────────────────────
results = []
for q_info in QUERIES:
    query = q_info["query"]
    print(f"\n{'='*60}")
    print(f"Query: \"{query}\"")
    print(f"{'='*60}")

    request = HybridSearchRequest(query=query, page_size=20)
    t0 = time.time()
    try:
        response = service.search(request=request)
        elapsed = time.time() - t0
        total = response.pagination.total_results if response.pagination else len(response.results)
        print(f"  Results: {total}")
        print(f"  Time: {elapsed:.1f}s")

        # Extract planner info from timing
        timing = response.timing or {}
        plan_vibe = timing.get("plan_vibe_brand")
        plan_algolia = timing.get("plan_algolia_query", "")
        plan_semantic = timing.get("plan_semantic_queries", [])
        plan_modes = timing.get("plan_modes", [])
        plan_attrs = timing.get("plan_attributes", {})
        plan_filters = timing.get("plan_applied_filters", {})
        vibe_filters_count = timing.get("vibe_brand_filters", 0)

        print(f"  vibe_brand: {plan_vibe}")
        print(f"  modes: {plan_modes}")
        print(f"  applied_filters: {plan_filters}")

        # Collect brand distribution
        brand_counts = {}
        for r in response.results:
            b = r.brand or "Unknown"
            brand_counts[b] = brand_counts.get(b, 0) + 1

        results.append({
            "query": query,
            "description": q_info["description"],
            "expect_vibe": q_info["expect_vibe"],
            "total": response.pagination.total_results if response.pagination else len(response.results),
            "elapsed": elapsed,
            "products": response.results[:20],
            "plan_vibe": plan_vibe,
            "plan_algolia": plan_algolia,
            "plan_semantic": plan_semantic,
            "plan_modes": plan_modes,
            "plan_attrs": plan_attrs,
            "plan_filters": plan_filters,
            "vibe_filters_count": vibe_filters_count,
            "brand_counts": brand_counts,
            "timing": timing,
        })
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ERROR: {e}")
        results.append({
            "query": query,
            "description": q_info["description"],
            "expect_vibe": q_info["expect_vibe"],
            "total": 0,
            "elapsed": elapsed,
            "products": [],
            "plan_vibe": None,
            "plan_algolia": "",
            "plan_semantic": [],
            "plan_modes": [],
            "plan_attrs": {},
            "plan_filters": {},
            "vibe_filters_count": 0,
            "brand_counts": {},
            "timing": {},
            "error": str(e),
        })

# ── Generate HTML ────────────────────────────────────────────────────────
def product_card(product):
    """Render a single product card."""
    name = html_mod.escape(getattr(product, 'name', '?') or '?')[:55]
    brand = html_mod.escape(getattr(product, 'brand', '?') or '?')
    price = getattr(product, 'price', 0) or 0
    img = html_mod.escape(getattr(product, 'image_url', '') or '')
    score = getattr(product, 'rrf_score', 0) or 0
    alg_rank = getattr(product, 'algolia_rank', None)
    sem_rank = getattr(product, 'semantic_rank', None)
    source_parts = []
    if alg_rank: source_parts.append(f"A#{alg_rank}")
    if sem_rank: source_parts.append(f"S#{sem_rank}")
    source = html_mod.escape(" ".join(source_parts) if source_parts else "—")

    return f"""
    <div style="border:1px solid #ddd; border-radius:10px; padding:10px; width:200px;
                display:inline-block; vertical-align:top; margin:6px; background:#fff;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
      <img src="{img}" style="width:180px; height:240px; object-fit:cover; border-radius:6px;"
           onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22180%22 height=%22240%22><rect fill=%22%23eee%22 width=%22180%22 height=%22240%22/><text x=%2250%%22 y=%2250%%22 font-size=%2214%22 fill=%22%23999%22 text-anchor=%22middle%22>No image</text></svg>'">
      <div style="margin-top:8px;">
        <div style="font-weight:600; font-size:12px; line-height:1.3; min-height:32px;">{name}</div>
        <div style="color:#666; font-size:12px; margin-top:2px;">
          <span style="font-weight:600; color:#333;">{brand}</span> &middot; ${price:.0f}
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:4px;">
          <span style="color:#09c; font-size:10px;">score: {score:.3f}</span>
          <span style="color:#888; font-size:10px;">{source}</span>
        </div>
      </div>
    </div>"""


def brand_bar(brand_counts, max_brands=12):
    """Render a horizontal brand frequency bar."""
    if not brand_counts:
        return "<p style='color:#999;'>No results</p>"
    sorted_brands = sorted(brand_counts.items(), key=lambda x: -x[1])[:max_brands]
    max_count = max(c for _, c in sorted_brands)
    bars = []
    for brand, count in sorted_brands:
        pct = count / max_count * 100
        bars.append(
            f'<div style="display:flex; align-items:center; margin:2px 0;">'
            f'<span style="width:140px; font-size:11px; text-align:right; padding-right:8px; '
            f'color:#555; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
            f'{html_mod.escape(brand)}</span>'
            f'<div style="background:linear-gradient(90deg, #4a90d9, #7eb8f0); height:16px; '
            f'border-radius:3px; width:{pct:.0f}%; min-width:20px; display:flex; align-items:center; '
            f'padding-left:6px;">'
            f'<span style="color:#fff; font-size:10px; font-weight:600;">{count}</span>'
            f'</div></div>'
        )
    return "".join(bars)


def status_badge(actual_vibe, expected_vibe):
    """Green check or red X for vibe_brand detection."""
    if expected_vibe is None:
        # Exact brand query — vibe should be None
        ok = actual_vibe is None
    else:
        ok = actual_vibe and actual_vibe.lower() == expected_vibe.lower()
    color = "#2ea043" if ok else "#d1242f"
    icon = "&#10003;" if ok else "&#10007;"
    label = f"vibe_brand={actual_vibe or 'None'}"
    return f'<span style="background:{color}; color:#fff; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:600;">{icon} {html_mod.escape(label)}</span>'


sections_html = []
for r in results:
    # Plan details box
    plan_html = f"""
    <div style="background:#f8f9fa; border:1px solid #e1e4e8; border-radius:8px; padding:14px; margin:10px 0;">
      <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
        {status_badge(r['plan_vibe'], r['expect_vibe'])}
        <span style="color:#666; font-size:12px;">{r['elapsed']:.1f}s &middot; {r['total']} results
        {f' &middot; {r["vibe_filters_count"]} cluster brands boosted' if r['vibe_filters_count'] else ''}</span>
      </div>
      <div style="font-size:11px; color:#555; line-height:1.6;">
        <b>Algolia query:</b> <code>{html_mod.escape(r['plan_algolia'] or '(empty)')}</code><br>
        <b>Modes:</b> {', '.join(r['plan_modes']) or '(none)'}<br>
        <b>Applied filters:</b> <code>{html_mod.escape(str(r['plan_filters']))}</code><br>
        <b>Semantic queries:</b><br>
        {'<br>'.join(f'&nbsp;&nbsp;{i+1}. <i>{html_mod.escape(sq)}</i>' for i, sq in enumerate(r['plan_semantic'][:4])) or '(none)'}
      </div>
    </div>
    """

    # Brand distribution
    brand_html = f"""
    <div style="margin:10px 0;">
      <b style="font-size:12px; color:#555;">Brand distribution:</b>
      <div style="max-width:400px; margin-top:4px;">
        {brand_bar(r['brand_counts'])}
      </div>
    </div>
    """

    # Product cards
    cards = "".join(product_card(p) for p in r["products"][:16])
    if not cards:
        cards = '<p style="color:#c00; font-size:14px;">No results returned.</p>'
        if r.get("error"):
            cards += f'<p style="color:#c00; font-size:12px;">Error: {html_mod.escape(r["error"])}</p>'

    sections_html.append(f"""
    <div style="margin-bottom:40px;">
      <h2 style="color:#222; border-bottom:2px solid #e1e4e8; padding-bottom:6px; margin-bottom:4px;">
        "{html_mod.escape(r['query'])}"
      </h2>
      <p style="color:#666; font-size:13px; margin-top:4px;">{html_mod.escape(r['description'])}</p>
      {plan_html}
      {brand_html}
      <div style="margin-top:10px;">{cards}</div>
    </div>
    """)


full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Vibe-Brand Search Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1400px; margin: 20px auto; padding: 0 20px; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; margin-bottom: 4px; }}
code {{ background: #e8edf2; padding: 1px 5px; border-radius: 3px; font-size: 11px; }}
</style></head><body>
<h1>Vibe-Brand Query Report</h1>
<p style="color:#666; margin-top:0;">
  Tests the <code>vibe_brand</code> feature: LLM detects brand-as-style references,
  decomposes aesthetic into search attributes, and boosts cluster-adjacent brands via optionalFilters.
</p>
<div style="background:#fff; border:1px solid #ddd; border-radius:8px; padding:14px; margin:16px 0;">
  <b>Summary:</b> {sum(1 for r in results if r['total'] > 0)}/{len(results)} queries returned results.
  Vibe detection: {sum(1 for r in results if (r['expect_vibe'] is None and r['plan_vibe'] is None) or (r['expect_vibe'] and r['plan_vibe'] and r['plan_vibe'].lower() == r['expect_vibe'].lower()))}/{len(results)} correct.
</div>
{''.join(sections_html)}
</body></html>"""

out_path = os.path.join(os.path.dirname(__file__), "vibe_brand_report.html")
with open(out_path, "w") as f:
    f.write(full_html)
print(f"\nReport written to {out_path}")
