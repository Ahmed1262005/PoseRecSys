#!/usr/bin/env python3
"""
Generate an HTML report for Similar Items — pure cosine ranking.
Picks diverse source products, runs get_similar_scored(), outputs visual report.

Usage:
    PYTHONPATH=src python scripts/generate_similar_report.py
"""

import os, sys, html, time, random
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from services.outfit_engine import get_outfit_engine

# ── Scenarios ──────────────────────────────────────────────────────────────
SCENARIOS = [
    {"label": "Floral Blouse", "query": {"category_l1": "Tops", "category_l2": "Blouse", "pattern": "Floral"}},
    {"label": "Black Dress", "query": {"category_l1": "Dresses", "color_family": "Black"}},
    {"label": "Denim Jeans", "query": {"category_l1": "Bottoms", "apparent_fabric": "Denim"}},
    {"label": "Leather Jacket", "query": {"category_l1": "Outerwear", "apparent_fabric": "Faux Leather"}},
    {"label": "Knit Sweater", "query": {"category_l1": "Tops", "category_l2": "Sweater"}},
    {"label": "Midi Skirt", "query": {"category_l1": "Bottoms", "category_l2": "Skirt"}},
    {"label": "Sequin Top", "query": {"category_l1": "Tops", "apparent_fabric": "Sequin"}},
    {"label": "Casual T-Shirt", "query": {"category_l1": "Tops", "category_l2": "T-Shirt"}},
    {"label": "Maxi Dress", "query": {"category_l1": "Dresses", "silhouette": "A-Line"}},
    {"label": "Blazer", "query": {"category_l1": "Outerwear", "category_l2": "Blazer"}},
]

LIMIT = 8  # items per scenario


def find_source_product(supabase, query: dict) -> dict | None:
    q = supabase.table("product_attributes").select("sku_id")
    for key, val in query.items():
        if key == "apparent_fabric":
            q = q.ilike(key, f"%{val}%")
        else:
            q = q.eq(key, val)
    q = q.limit(20)
    result = q.execute()

    if not result.data:
        q2 = supabase.table("product_attributes").select("sku_id")
        for key, val in query.items():
            if key != "construction":
                q2 = q2.ilike(key, f"%{val}%")
        q2 = q2.limit(20)
        result = q2.execute()

    if not result.data:
        return None

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
    valid = [p for p in (products.data or []) if p.get("primary_image_url") and p.get("name")]
    return random.choice(valid) if valid else None


def generate_html(sections: list, total_time: float) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    body_parts = []

    for sec in sections:
        if sec.get("error"):
            body_parts.append(f'<div class="section"><h2>{html.escape(sec["label"])}</h2>'
                              f'<p style="color:#888">No product found for this scenario.</p></div>')
            continue

        src = sec["source"]
        items = sec["items"]
        elapsed = sec["elapsed"]

        # Source card
        src_html = (
            f'<div class="source">'
            f'<img src="{html.escape(src["primary_image_url"])}" onerror="this.style.display=\'none\'">'
            f'<div class="source-info">'
            f'<h3>{html.escape(src["name"])}</h3>'
            f'<div class="detail">{html.escape(src.get("brand",""))} &bull; '
            f'${src.get("price",0)} &bull; '
            f'<span class="l2-tag">{html.escape(sec["label"])}</span></div>'
            f'</div></div>'
        )

        # Stats
        stats_html = (
            f'<div class="stats">{len(items)} similar items '
            f'<span class="timing">{elapsed:.1f}s</span></div>'
        )

        # Item cards
        cards = []
        for i, item in enumerate(items):
            cos = item.get("cosine_similarity", item.get("tattoo_score", 0))
            compat = item.get("compatibility_score", 0)
            img = item.get("primary_image_url", "")
            name = item.get("name", "?")
            brand = item.get("brand", "")
            price = item.get("price", 0)

            cards.append(
                f'<div class="card-wrap"><div class="rank">#{i+1}</div><div class="card">'
                f'<img src="{html.escape(img)}" onerror="this.style.background=\'#eee\'">'
                f'<div class="card-info">'
                f'<div class="name">{html.escape(name)}</div>'
                f'<div class="meta">{html.escape(brand)} &bull; ${price}</div>'
                f'<div class="scores">'
                f'<span class="badge badge-cos">cos {cos:.2f}</span>'
                f'<span class="badge badge-compat">compat {compat:.2f}</span>'
                f'</div></div></div></div>'
            )

        grid_html = '<div class="grid">' + "".join(cards) + '</div>'
        body_parts.append(f'<div class="section">{src_html}{stats_html}{grid_html}</div>')

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Similar Items Report — Pure Cosine (match_count=500)</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;margin:20px auto;max-width:1200px;background:#f5f5f5;color:#222}}
h1{{text-align:center;margin-bottom:5px}}
.subtitle{{text-align:center;color:#666;font-size:14px;margin-bottom:30px}}
.section{{background:#fff;border-radius:12px;padding:24px;margin:30px 0;box-shadow:0 2px 8px rgba(0,0,0,0.08)}}
.source{{display:flex;align-items:center;gap:20px;padding:16px;background:linear-gradient(135deg,#f0f4ff,#f8f0ff);border-radius:10px;margin-bottom:16px}}
.source img{{width:120px;height:160px;object-fit:cover;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.15)}}
.source-info h3{{margin:0 0 6px;font-size:18px}}
.source-info .detail{{color:#555;font-size:14px;line-height:1.6}}
.source-info .l2-tag{{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;background:#e8e0f0;color:#5a3d8a}}
.stats{{font-size:13px;color:#555;margin-bottom:12px}}
.timing{{float:right;font-size:12px;color:#888;background:#f0f0f0;padding:3px 8px;border-radius:4px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:14px}}
.card{{background:#fafafa;border-radius:8px;overflow:hidden;border:1px solid #eee;transition:transform 0.15s}}
.card:hover{{transform:translateY(-2px);box-shadow:0 4px 12px rgba(0,0,0,0.1)}}
.card img{{width:100%;height:230px;object-fit:cover}}
.card-info{{padding:10px;font-size:12px}}
.card-info .name{{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:3px}}
.card-info .meta{{color:#666;margin-bottom:6px}}
.scores{{display:flex;gap:4px;flex-wrap:wrap;align-items:center}}
.badge{{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600}}
.badge-cos{{background:#dbeafe;color:#1e40af}}
.badge-compat{{background:#dcfce7;color:#166534}}
.rank{{position:absolute;top:6px;left:6px;background:rgba(0,0,0,0.65);color:#fff;font-size:11px;font-weight:700;padding:2px 7px;border-radius:4px}}
.card-wrap{{position:relative}}
</style></head><body>
<h1>Similar Items Report — Pure Cosine Ranking (match_count=500)</h1>
<p class="subtitle">Ranked by FashionCLIP cosine similarity only. Compat shown for reference, not used in ranking. Generated {ts}. Total: {total_time:.1f}s</p>
{"".join(body_parts)}
</body></html>"""


def main():
    print("Loading engine...")
    engine = get_outfit_engine()
    supabase = engine.supabase

    sections = []
    t_total = time.time()

    for i, scenario in enumerate(SCENARIOS):
        label = scenario["label"]
        print(f"\n[{i+1}/{len(SCENARIOS)}] {label}...")

        src = find_source_product(supabase, scenario["query"])
        if not src:
            print(f"  -> No product found, skipping")
            sections.append({"label": label, "error": True})
            continue

        print(f"  Source: {src['name'][:50]} ({src['brand']}, ${src['price']})")

        t0 = time.time()
        try:
            result = engine.get_similar_scored(product_id=src["id"], limit=LIMIT)
        except Exception as e:
            print(f"  -> Error: {e}")
            sections.append({"label": label, "error": True})
            continue
        elapsed = time.time() - t0

        items = result.get("results", []) if result else []
        print(f"  -> {len(items)} items in {elapsed:.1f}s")

        # Print top 3 cosine scores
        for j, item in enumerate(items[:3]):
            cos = item.get("cosine_similarity", item.get("tattoo_score", 0))
            print(f"     #{j+1}: cos={cos:.3f} — {item.get('name','?')[:40]}")

        sections.append({
            "label": label,
            "source": src,
            "items": items,
            "elapsed": elapsed,
        })

    total_time = time.time() - t_total
    print(f"\nTotal time: {total_time:.1f}s")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "similar_items_report.html")
    report_html = generate_html(sections, total_time)
    with open(out_path, "w") as f:
        f.write(report_html)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
