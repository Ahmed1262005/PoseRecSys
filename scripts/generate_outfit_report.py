#!/usr/bin/env python3
"""
Generate an HTML report with real product examples from the Complete the Fit engine.
Picks diverse source products, runs build_outfit(), and outputs a visual report
with images, dimension scores, and outfit breakdowns.

Usage:
    PYTHONPATH=src python scripts/generate_outfit_report.py
"""

import os, sys, json, time, html
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from services.outfit_engine import get_outfit_engine

# ── Scenario definitions ──────────────────────────────────────────────────
# Each scenario: (label, description, query to find a matching product)
# We'll query Supabase directly to find products that match each scenario.

SCENARIOS = [
    {
        "label": "Classic Office Look",
        "desc": "Formal blouse → what bottoms + outerwear complement it?",
        "query": {"formality": "Business Casual", "category_l1": "Tops", "category_l2": "Blouse"},
    },
    {
        "label": "Casual Denim",
        "desc": "Blue jeans → what tops + outerwear pair with denim?",
        "query": {"category_l1": "Bottoms", "apparent_fabric": "Denim"},
    },
    {
        "label": "Evening Dress",
        "desc": "Formal/semi-formal dress → what outerwear completes the look?",
        "query": {"formality": "Semi-Formal", "category_l1": "Dresses"},
    },
    {
        "label": "Streetwear Hoodie",
        "desc": "Casual hoodie → can it find relaxed bottoms + streetwear outerwear?",
        "query": {"category_l1": "Tops", "category_l2": "Hoodie"},
    },
    {
        "label": "Leather Jacket",
        "desc": "Edgy outerwear → what tops, bottoms, and dresses work underneath?",
        "query": {"category_l1": "Outerwear", "apparent_fabric": "Faux Leather"},
    },
    {
        "label": "Summer Linen",
        "desc": "Light summer top → does seasonality keep it warm-weather?",
        "query": {"category_l1": "Tops", "apparent_fabric": "Linen"},
    },
    {
        "label": "Statement Sequin Top",
        "desc": "High-strength statement piece → does it find supporting (low-strength) bottoms?",
        "query": {"category_l1": "Tops", "apparent_fabric": "Sequin"},
    },
    {
        "label": "Bohemian Maxi Skirt",
        "desc": "Boho flowy bottom → style matrix should prefer romantic/casual tops.",
        "query": {"category_l1": "Bottoms", "silhouette": "A-Line"},
    },
    {
        "label": "Blazer (Bridge Item)",
        "desc": "Classic blazer bridges formality gaps → how versatile are its recs?",
        "query": {"category_l1": "Outerwear", "category_l2": "Blazer"},
    },
    {
        "label": "Cropped Fitted Top",
        "desc": "Cropped top → waist logic should prefer high-rise bottoms.",
        "query": {"category_l1": "Tops", "silhouette": "Fitted", "category_l2": "Crop Top"},
    },
    {
        "label": "Knit Sweater (Winter)",
        "desc": "Winter knit → seasonality should match with coats, not sandals.",
        "query": {"category_l1": "Tops", "apparent_fabric": "Knit", "category_l2": "Sweater"},
    },
    {
        "label": "Floral Print Dress",
        "desc": "Bold pattern → pattern scorer should pair it with solid outerwear.",
        "query": {"category_l1": "Dresses", "pattern": "Floral"},
    },
]


DIM_LABELS = {
    "occasion_formality": "Occasion",
    "style": "Style",
    "fabric": "Fabric",
    "silhouette": "Silhouette",
    "color": "Color",
    "seasonality": "Season",
    "pattern": "Pattern",
    "price": "Price",
}

DIM_COLORS = {
    "occasion_formality": "#7c6ff7",
    "style": "#f472b6",
    "fabric": "#fbbf24",
    "silhouette": "#34d399",
    "color": "#60a5fa",
    "seasonality": "#22d3ee",
    "pattern": "#fb923c",
    "price": "#a78bfa",
}


def find_source_product(supabase, query: dict) -> dict | None:
    """Find a product matching the scenario query."""
    # Start with product_attributes table
    q = supabase.table("product_attributes").select("sku_id")

    for key, val in query.items():
        if key == "apparent_fabric":
            q = q.ilike(key, f"%{val}%")
        else:
            q = q.eq(key, val)

    q = q.limit(20)
    result = q.execute()

    if not result.data:
        # Fallback: try with ilike for more flexibility
        q2 = supabase.table("product_attributes").select("sku_id")
        for key, val in query.items():
            if key != "construction":
                q2 = q2.ilike(key, f"%{val}%")
        q2 = q2.limit(20)
        result = q2.execute()

    if not result.data:
        return None

    # Pick one that's in stock and has an image
    sku_ids = [r["sku_id"] for r in result.data]
    products = (
        supabase.table("products")
        .select("id, name, brand, price, primary_image_url, in_stock")
        .in_("id", sku_ids)
        .eq("in_stock", True)
        .not_.is_("primary_image_url", "null")
        .limit(5)
        .execute()
    )

    if products.data:
        # Pick one with a reasonable image
        for p in products.data:
            if p.get("primary_image_url") and p.get("name"):
                return p

    return None


def generate_html(scenarios_data: list) -> str:
    """Generate the full HTML report."""

    cards_html = []
    for i, scenario in enumerate(scenarios_data):
        if scenario.get("error"):
            cards_html.append(f"""
            <div class="scenario error-scenario">
              <div class="scenario-header">
                <span class="scenario-num">{i+1}</span>
                <div>
                  <h2>{html.escape(scenario['label'])}</h2>
                  <p class="scenario-desc">{html.escape(scenario['desc'])}</p>
                </div>
              </div>
              <div class="error-msg">Could not find a matching product for this scenario.</div>
            </div>""")
            continue

        source = scenario["source_product"]
        profile = source.get("aesthetic_profile", {})
        status = scenario.get("status", "ok")
        recs = scenario.get("recommendations", {})

        # Source product card
        profile_pills = []
        if profile.get("formality"):
            profile_pills.append(f'<span class="pill form">{html.escape(profile["formality"])}</span>')
        for tag in (profile.get("style_tags") or [])[:3]:
            profile_pills.append(f'<span class="pill style">{html.escape(tag)}</span>')
        if profile.get("apparent_fabric"):
            profile_pills.append(f'<span class="pill fabric">{html.escape(profile["apparent_fabric"])}</span>')
        if profile.get("pattern") and profile["pattern"].lower() not in ("solid", "plain", "none"):
            profile_pills.append(f'<span class="pill pattern">{html.escape(profile["pattern"])}</span>')
        if profile.get("silhouette"):
            profile_pills.append(f'<span class="pill sil">{html.escape(profile["silhouette"])}</span>')
        if profile.get("color_family"):
            profile_pills.append(f'<span class="pill color">{html.escape(profile["color_family"])}</span>')
        for s in (profile.get("seasons") or [])[:2]:
            profile_pills.append(f'<span class="pill season">{html.escape(s)}</span>')

        pills_html = " ".join(profile_pills)

        # Recommendation categories
        cat_sections = []
        for cat_name, cat_data in recs.items():
            items = cat_data.get("items", [])
            if not items:
                continue

            item_cards = []
            for item in items[:6]:  # show up to 6
                dim_scores = item.get("dimension_scores", {})
                bars = []
                for dim_key, dim_label in DIM_LABELS.items():
                    val = dim_scores.get(dim_key, 0)
                    color = DIM_COLORS.get(dim_key, "#888")
                    pct = int(val * 100)
                    bars.append(f"""
                      <div class="dim-row">
                        <span class="dim-label">{dim_label}</span>
                        <div class="dim-track"><div class="dim-fill" style="width:{pct}%;background:{color}"></div></div>
                        <span class="dim-val">{val:.2f}</span>
                      </div>""")

                bars_html = "\n".join(bars)
                tattoo = item.get("tattoo_score", 0)
                compat = item.get("compatibility_score", 0)
                cosine = item.get("cosine_similarity", 0)
                novelty = item.get("novelty_score", 0)

                score_class = "score-great" if tattoo >= 0.70 else "score-good" if tattoo >= 0.55 else "score-mid" if tattoo >= 0.40 else "score-bad"

                novelty_pct = int(novelty * 100)

                item_cards.append(f"""
                <div class="rec-card">
                  <div class="rec-img-wrap">
                    <img src="{html.escape(item.get('image_url', '') or '')}" alt="{html.escape(item.get('name', ''))}" loading="lazy" onerror="this.style.display='none'">
                    <div class="rec-rank">#{item.get('rank', '?')}</div>
                    <div class="rec-score {score_class}">{tattoo:.3f}</div>
                  </div>
                  <div class="rec-info">
                    <div class="rec-name">{html.escape((item.get('name', '') or '')[:55])}</div>
                    <div class="rec-brand">{html.escape(item.get('brand', '') or '')} &middot; ${item.get('price', 0):.0f}</div>
                    <div class="rec-scores-detail">
                      <span title="Attribute compatibility">Compat: {compat:.3f}</span>
                      <span title="Contrast novelty">Novelty: {novelty:.3f}</span>
                      <span title="Visual cosine similarity">Cosine: {cosine:.3f}</span>
                    </div>
                    <div class="dim-row" style="margin-bottom:.4rem">
                      <span class="dim-label" style="color:var(--pink)">Novelty</span>
                      <div class="dim-track"><div class="dim-fill" style="width:{novelty_pct}%;background:var(--pink)"></div></div>
                      <span class="dim-val">{novelty:.2f}</span>
                    </div>
                    <div class="dim-bars">{bars_html}</div>
                  </div>
                </div>""")

            item_cards_html = "\n".join(item_cards)
            cat_sections.append(f"""
            <div class="cat-section">
              <h3 class="cat-title">{html.escape(cat_name.title())}</h3>
              <div class="rec-grid">{item_cards_html}</div>
            </div>""")

        cats_html = "\n".join(cat_sections)

        cards_html.append(f"""
        <div class="scenario">
          <div class="scenario-header">
            <span class="scenario-num">{i+1}</span>
            <div>
              <h2>{html.escape(scenario['label'])} <span class="status-badge status-{status}">{status}</span></h2>
              <p class="scenario-desc">{html.escape(scenario['desc'])}</p>
            </div>
          </div>

          <div class="source-section">
            <div class="source-card">
              <div class="source-img-wrap">
                <img src="{html.escape(source.get('image_url', '') or '')}" alt="{html.escape(source.get('name', ''))}" loading="lazy" onerror="this.style.display='none'">
              </div>
              <div class="source-info">
                <div class="source-label">SOURCE PRODUCT</div>
                <div class="source-name">{html.escape(source.get('name', '') or '')}</div>
                <div class="source-brand">{html.escape(source.get('brand', '') or '')} &middot; ${source.get('price', 0):.0f}</div>
                <div class="source-cat">{html.escape(source.get('category', '') or '')}</div>
                <div class="pills">{pills_html}</div>
              </div>
            </div>
          </div>

          {cats_html}
        </div>""")

    all_cards = "\n".join(cards_html)
    ts = datetime.now().strftime("%B %d, %Y %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Complete the Fit — Live Examples Report</title>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d27; --surface2: #222633;
    --border: #2e3348; --text: #e2e4ec; --text2: #9da2b8;
    --accent: #7c6ff7; --accent2: #a78bfa;
    --green: #34d399; --red: #f87171; --yellow: #fbbf24; --blue: #60a5fa;
    --pink: #f472b6; --cyan: #22d3ee;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
  }}
  .wrapper {{ max-width: 1400px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }}

  header {{
    text-align: center; padding: 3rem 0 2rem;
    border-bottom: 1px solid var(--border); margin-bottom: 2.5rem;
  }}
  header h1 {{
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--pink));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  header p {{ color: var(--text2); font-size: .95rem; margin-top: .4rem; }}
  header .meta {{ color: var(--text2); font-size: .8rem; margin-top: .6rem; }}

  /* ── Scenario ──────────────────────────────── */
  .scenario {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 1.8rem; margin-bottom: 2rem;
  }}
  .error-scenario {{ opacity: 0.5; }}
  .error-msg {{ color: var(--red); font-size: .9rem; padding: 1rem 0; }}

  .scenario-header {{
    display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 1.5rem;
  }}
  .scenario-num {{
    display: flex; align-items: center; justify-content: center;
    width: 2.2rem; height: 2.2rem; flex-shrink: 0;
    background: var(--accent); color: #fff; border-radius: 10px;
    font-weight: 700; font-size: .95rem;
  }}
  .scenario-header h2 {{ font-size: 1.25rem; font-weight: 700; }}
  .scenario-desc {{ color: var(--text2); font-size: .87rem; margin-top: .15rem; }}

  .status-badge {{
    display: inline-block; font-size: .7rem; font-weight: 600;
    padding: .12rem .5rem; border-radius: 999px; text-transform: uppercase;
    vertical-align: middle; margin-left: .4rem;
  }}
  .status-ok {{ background: rgba(52,211,153,.15); color: var(--green); }}
  .status-set {{ background: rgba(251,191,36,.15); color: var(--yellow); }}
  .status-blocked {{ background: rgba(248,113,113,.15); color: var(--red); }}
  .status-activewear {{ background: rgba(96,165,250,.15); color: var(--blue); }}

  /* ── Source product ────────────────────────── */
  .source-section {{ margin-bottom: 1.5rem; }}
  .source-card {{
    display: flex; gap: 1.2rem; background: var(--surface2);
    border: 1px solid var(--border); border-radius: 12px; padding: 1rem;
  }}
  .source-img-wrap {{
    width: 180px; height: 240px; flex-shrink: 0;
    border-radius: 10px; overflow: hidden; background: #15171f;
  }}
  .source-img-wrap img {{
    width: 100%; height: 100%; object-fit: cover;
  }}
  .source-info {{ flex: 1; padding: .3rem 0; }}
  .source-label {{
    font-size: .7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .08em; color: var(--accent2); margin-bottom: .3rem;
  }}
  .source-name {{ font-size: 1.05rem; font-weight: 600; margin-bottom: .2rem; }}
  .source-brand {{ font-size: .88rem; color: var(--text2); margin-bottom: .15rem; }}
  .source-cat {{ font-size: .82rem; color: var(--text2); margin-bottom: .6rem; }}

  .pills {{ display: flex; flex-wrap: wrap; gap: .35rem; }}
  .pill {{
    font-size: .72rem; font-weight: 500; padding: .2rem .55rem;
    border-radius: 999px; border: 1px solid var(--border);
    background: var(--surface);
  }}
  .pill.form {{ border-color: rgba(124,111,247,.4); color: var(--accent2); }}
  .pill.style {{ border-color: rgba(244,114,182,.4); color: var(--pink); }}
  .pill.fabric {{ border-color: rgba(251,191,36,.4); color: var(--yellow); }}
  .pill.pattern {{ border-color: rgba(251,146,60,.4); color: #fb923c; }}
  .pill.sil {{ border-color: rgba(52,211,153,.4); color: var(--green); }}
  .pill.color {{ border-color: rgba(96,165,250,.4); color: var(--blue); }}
  .pill.season {{ border-color: rgba(34,211,238,.4); color: var(--cyan); }}

  /* ── Category section ──────────────────────── */
  .cat-section {{ margin-bottom: 1.2rem; }}
  .cat-title {{
    font-size: 1rem; font-weight: 700; color: var(--accent2);
    margin-bottom: .8rem; padding-bottom: .4rem;
    border-bottom: 1px solid var(--border);
  }}

  /* ── Rec grid ──────────────────────────────── */
  .rec-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: .8rem;
  }}
  .rec-card {{
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden; transition: border-color .2s;
  }}
  .rec-card:hover {{ border-color: var(--accent); }}

  .rec-img-wrap {{
    position: relative; width: 100%; height: 260px;
    background: #15171f; overflow: hidden;
  }}
  .rec-img-wrap img {{
    width: 100%; height: 100%; object-fit: cover;
  }}
  .rec-rank {{
    position: absolute; top: .5rem; left: .5rem;
    background: rgba(0,0,0,.7); color: #fff; font-size: .75rem;
    font-weight: 700; padding: .15rem .45rem; border-radius: 6px;
  }}
  .rec-score {{
    position: absolute; top: .5rem; right: .5rem;
    font-size: .8rem; font-weight: 700;
    padding: .2rem .5rem; border-radius: 6px;
  }}
  .score-great {{ background: rgba(52,211,153,.2); color: var(--green); }}
  .score-good  {{ background: rgba(96,165,250,.2); color: var(--blue); }}
  .score-mid   {{ background: rgba(251,191,36,.2); color: var(--yellow); }}
  .score-bad   {{ background: rgba(248,113,113,.2); color: var(--red); }}

  .rec-info {{ padding: .7rem .8rem .9rem; }}
  .rec-name {{ font-size: .82rem; font-weight: 600; line-height: 1.3; margin-bottom: .2rem;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }}
  .rec-brand {{ font-size: .78rem; color: var(--text2); margin-bottom: .3rem; }}
  .rec-scores-detail {{ font-size: .72rem; color: var(--text2); margin-bottom: .5rem; display:flex; gap:.6rem; }}

  /* ── Dimension bars ────────────────────────── */
  .dim-bars {{ display: flex; flex-direction: column; gap: .2rem; }}
  .dim-row {{ display: flex; align-items: center; gap: .3rem; }}
  .dim-label {{ width: 52px; font-size: .68rem; color: var(--text2); text-align: right; flex-shrink: 0; }}
  .dim-track {{
    flex: 1; height: 8px; background: rgba(255,255,255,.06);
    border-radius: 4px; overflow: hidden;
  }}
  .dim-fill {{ height: 100%; border-radius: 4px; transition: width .3s; }}
  .dim-val {{ width: 28px; font-size: .68rem; color: var(--text2); font-weight: 500; }}

  @media (max-width: 700px) {{
    .source-card {{ flex-direction: column; }}
    .source-img-wrap {{ width: 100%; height: 300px; }}
    .rec-grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>
<div class="wrapper">
  <header>
    <h1>Complete the Fit — Live Examples</h1>
    <p>Real products from the database scored by the TATTOO v2 engine</p>
    <div class="meta">{ts} &middot; {len([s for s in scenarios_data if not s.get('error')])} scenarios &middot; outfit_engine.py tattoo_v2</div>
  </header>
  {all_cards}
  <div style="text-align:center;padding:2rem 0;color:var(--text2);font-size:.78rem;border-top:1px solid var(--border);margin-top:1rem;">
    Complete the Fit &mdash; TATTOO v2 Engine &middot; Generated {ts}
  </div>
</div>
</body>
</html>"""


def main():
    print("Initializing outfit engine...")
    engine = get_outfit_engine()
    supabase = engine.supabase
    print("Engine ready.\n")

    scenarios_data = []

    for i, scenario in enumerate(SCENARIOS):
        label = scenario["label"]
        print(f"[{i+1}/{len(SCENARIOS)}] {label}...")

        # Find a matching source product
        product = find_source_product(supabase, scenario["query"])
        if not product:
            print(f"  -> No matching product found, skipping.")
            scenarios_data.append({
                "label": label,
                "desc": scenario["desc"],
                "error": True,
            })
            continue

        pid = product["id"]
        print(f"  -> Found: {product['name'][:60]} ({product['brand']}, ${product.get('price', 0):.0f})")

        # Run the outfit engine
        t0 = time.time()
        try:
            result = engine.build_outfit(product_id=pid, items_per_category=6)
        except Exception as e:
            print(f"  -> ERROR: {e}")
            scenarios_data.append({
                "label": label,
                "desc": scenario["desc"],
                "error": True,
            })
            continue

        elapsed = time.time() - t0
        print(f"  -> build_outfit() took {elapsed:.1f}s")

        if result.get("error"):
            print(f"  -> Engine error: {result['error']}")
            scenarios_data.append({
                "label": label,
                "desc": scenario["desc"],
                "error": True,
            })
            continue

        # Count total rec items
        total_items = sum(
            len(cat.get("items", []))
            for cat in result.get("recommendations", {}).values()
        )
        print(f"  -> Status: {result.get('status', '?')}, {total_items} items across {len(result.get('recommendations', {}))} categories")

        scenarios_data.append({
            "label": label,
            "desc": scenario["desc"],
            "source_product": result["source_product"],
            "recommendations": result["recommendations"],
            "status": result.get("status", "ok"),
        })

    # Generate HTML
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "docs", "outfit_examples_report.html"
    )
    output_path = os.path.normpath(output_path)

    print(f"\nGenerating HTML report...")
    html_content = generate_html(scenarios_data)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Report saved to: {output_path}")
    print(f"Total scenarios: {len(scenarios_data)}, successful: {len([s for s in scenarios_data if not s.get('error')])}")


if __name__ == "__main__":
    main()
