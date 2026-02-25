#!/usr/bin/env python3
"""
Compare similar items: OLD v1 RPC (HNSW-friendly) vs CURRENT v2 RPC (DISTINCT ON).

Picks diverse products, runs both RPCs side-by-side, generates HTML comparison report.

Usage:
    PYTHONPATH=src python scripts/compare_similar_v1_v2.py
"""

import os, sys, html, time, random
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from config.database import get_supabase_client

LIMIT = 10  # items per system per scenario

# Diverse test scenarios — find products by Gemini L1/L2
SCENARIOS = [
    {"label": "Floral Blouse", "query": {"category_l1": "Tops", "category_l2": "Blouse", "pattern": "Floral"}},
    {"label": "Black Dress", "query": {"category_l1": "Dresses", "color_family": "Black"}},
    {"label": "Denim Jeans", "query": {"category_l1": "Bottoms", "apparent_fabric": "Denim"}},
    {"label": "Leather Jacket", "query": {"category_l1": "Outerwear", "apparent_fabric": "Faux Leather"}},
    {"label": "Knit Sweater", "query": {"category_l1": "Tops", "category_l2": "Sweater"}},
    {"label": "Sequin Top", "query": {"category_l1": "Tops", "apparent_fabric": "Sequin"}},
    {"label": "Casual T-Shirt", "query": {"category_l1": "Tops", "category_l2": "T-Shirt"}},
    {"label": "Blazer", "query": {"category_l1": "Outerwear", "category_l2": "Blazer"}},
]


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
        .select("id, name, brand, price, category, primary_image_url, in_stock")
        .in_("id", sku_ids)
        .eq("in_stock", True)
        .not_.is_("primary_image_url", "null")
        .limit(5)
        .execute()
    )
    valid = [p for p in (products.data or []) if p.get("primary_image_url") and p.get("name")]
    return random.choice(valid) if valid else None


def run_v1(supabase, product_id: str, category: str | None) -> tuple[list, float]:
    """Old get_similar_products — no DISTINCT ON, has in_stock, HNSW-friendly."""
    t0 = time.time()
    try:
        result = supabase.rpc("get_similar_products", {
            "source_product_id": product_id,
            "match_count": LIMIT + 50,  # old pattern: extra for dedup
            "filter_gender": "female",
            "filter_category": category,
        }).execute()
        rows = result.data or []
    except Exception as e:
        print(f"    v1 error: {e}")
        rows = []
    elapsed = time.time() - t0

    # Deduplicate by product_id (old Python did this)
    seen = set()
    deduped = []
    for r in rows:
        pid = str(r.get("product_id", ""))
        if pid and pid not in seen:
            seen.add(pid)
            deduped.append(r)
    return deduped[:LIMIT], elapsed


def run_v2(supabase, product_id: str, category: str | None) -> tuple[list, float]:
    """Current get_similar_products_v2 — DISTINCT ON, no in_stock, subquery."""
    t0 = time.time()
    try:
        result = supabase.rpc("get_similar_products_v2", {
            "source_product_id": product_id,
            "match_count": 500,  # current setting
            "match_offset": 0,
            "filter_category": category,
        }).execute()
        rows = result.data or []
    except Exception as e:
        print(f"    v2 error: {e}")
        rows = []
    elapsed = time.time() - t0

    # Deduplicate by product_id
    seen = set()
    deduped = []
    for r in rows:
        pid = str(r.get("product_id", ""))
        if pid and pid not in seen:
            seen.add(pid)
            deduped.append(r)
    return deduped[:LIMIT], elapsed


def make_card(item: dict, rank: int, system: str) -> str:
    img = html.escape(item.get("primary_image_url", ""))
    name = html.escape(item.get("name", "?")[:45])
    brand = html.escape(item.get("brand", ""))
    price = item.get("price", 0)
    sim = item.get("similarity", 0)
    in_stock = item.get("in_stock", "?")

    stock_badge = ""
    if in_stock is False:
        stock_badge = '<span class="badge badge-oos">OOS</span>'

    return (
        f'<div class="card-wrap"><div class="rank">#{rank}</div><div class="card">'
        f'<img src="{img}" onerror="this.style.background=\'#eee\'">'
        f'<div class="card-info">'
        f'<div class="name">{name}</div>'
        f'<div class="meta">{brand} &bull; ${price}</div>'
        f'<div class="scores">'
        f'<span class="badge badge-cos">cos {sim:.3f}</span>'
        f'{stock_badge}'
        f'</div></div></div></div>'
    )


def generate_html(sections: list, total_time: float) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    body_parts = []

    for sec in sections:
        if sec.get("error"):
            body_parts.append(f'<div class="section"><h2>{html.escape(sec["label"])}</h2>'
                              f'<p style="color:#888">No product found.</p></div>')
            continue

        src = sec["source"]
        v1_items = sec["v1_items"]
        v2_items = sec["v2_items"]
        v1_time = sec["v1_time"]
        v2_time = sec["v2_time"]

        # Overlap analysis
        v1_ids = {str(r.get("product_id", "")) for r in v1_items}
        v2_ids = {str(r.get("product_id", "")) for r in v2_items}
        overlap = v1_ids & v2_ids
        overlap_pct = len(overlap) / max(len(v1_ids | v2_ids), 1) * 100

        # Avg similarity
        v1_avg = sum(r.get("similarity", 0) for r in v1_items) / max(len(v1_items), 1)
        v2_avg = sum(r.get("similarity", 0) for r in v2_items) / max(len(v2_items), 1)

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

        # Stats comparison
        stats_html = (
            f'<div class="comparison-stats">'
            f'<div class="stat-box v1">'
            f'<strong>OLD (v1)</strong>: {len(v1_items)} items, {v1_time:.2f}s, avg cos={v1_avg:.3f}'
            f'</div>'
            f'<div class="stat-box v2">'
            f'<strong>NEW (v2)</strong>: {len(v2_items)} items, {v2_time:.2f}s, avg cos={v2_avg:.3f}'
            f'</div>'
            f'<div class="stat-box overlap">'
            f'Overlap: {len(overlap)}/{max(len(v1_ids | v2_ids), 1)} ({overlap_pct:.0f}%)'
            f'</div>'
            f'</div>'
        )

        # Side-by-side grids
        v1_cards = "".join(make_card(r, i+1, "v1") for i, r in enumerate(v1_items))
        v2_cards = "".join(make_card(r, i+1, "v2") for i, r in enumerate(v2_items))

        grids_html = (
            f'<div class="side-by-side">'
            f'<div class="half"><h3 class="system-label v1-label">OLD (v1) — No DISTINCT ON, in_stock=true, HNSW-friendly</h3>'
            f'<div class="grid">{v1_cards}</div></div>'
            f'<div class="half"><h3 class="system-label v2-label">NEW (v2) — DISTINCT ON, no in_stock, subquery</h3>'
            f'<div class="grid">{v2_cards}</div></div>'
            f'</div>'
        )

        body_parts.append(f'<div class="section">{src_html}{stats_html}{grids_html}</div>')

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Similar Items: V1 vs V2 Comparison</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;margin:20px auto;max-width:1400px;background:#f5f5f5;color:#222}}
h1{{text-align:center;margin-bottom:5px}}
.subtitle{{text-align:center;color:#666;font-size:14px;margin-bottom:30px}}
.section{{background:#fff;border-radius:12px;padding:24px;margin:30px 0;box-shadow:0 2px 8px rgba(0,0,0,0.08)}}
.source{{display:flex;align-items:center;gap:20px;padding:16px;background:linear-gradient(135deg,#f0f4ff,#f8f0ff);border-radius:10px;margin-bottom:16px}}
.source img{{width:120px;height:160px;object-fit:cover;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.15)}}
.source-info h3{{margin:0 0 6px;font-size:18px}}
.source-info .detail{{color:#555;font-size:14px;line-height:1.6}}
.source-info .l2-tag{{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;background:#e8e0f0;color:#5a3d8a}}
.comparison-stats{{display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap}}
.stat-box{{padding:8px 14px;border-radius:8px;font-size:13px;flex:1;min-width:200px}}
.stat-box.v1{{background:#dcfce7;border:1px solid #86efac}}
.stat-box.v2{{background:#dbeafe;border:1px solid #93c5fd}}
.stat-box.overlap{{background:#fef3c7;border:1px solid #fcd34d}}
.side-by-side{{display:flex;gap:20px}}
.half{{flex:1;min-width:0}}
.system-label{{font-size:13px;margin:0 0 10px;padding:6px 12px;border-radius:6px;text-align:center}}
.v1-label{{background:#dcfce7;color:#166534}}
.v2-label{{background:#dbeafe;color:#1e40af}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:10px}}
.card{{background:#fafafa;border-radius:8px;overflow:hidden;border:1px solid #eee;transition:transform 0.15s}}
.card:hover{{transform:translateY(-2px);box-shadow:0 4px 12px rgba(0,0,0,0.1)}}
.card img{{width:100%;height:170px;object-fit:cover}}
.card-info{{padding:8px;font-size:11px}}
.card-info .name{{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:2px}}
.card-info .meta{{color:#666;margin-bottom:4px}}
.scores{{display:flex;gap:3px;flex-wrap:wrap;align-items:center}}
.badge{{display:inline-block;padding:2px 6px;border-radius:4px;font-size:9px;font-weight:600}}
.badge-cos{{background:#dbeafe;color:#1e40af}}
.badge-oos{{background:#fee2e2;color:#991b1b}}
.rank{{position:absolute;top:4px;left:4px;background:rgba(0,0,0,0.65);color:#fff;font-size:10px;font-weight:700;padding:2px 6px;border-radius:4px}}
.card-wrap{{position:relative}}
</style></head><body>
<h1>Similar Items: V1 (Old) vs V2 (Current)</h1>
<p class="subtitle">Side-by-side comparison. V1 = original RPC (HNSW-friendly, in_stock=true). V2 = current RPC (DISTINCT ON, no in_stock). Generated {ts}. Total: {total_time:.1f}s</p>
{"".join(body_parts)}
</body></html>"""


def main():
    print("Connecting to Supabase...")
    supabase = get_supabase_client()

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

        pid = src["id"]
        cat = src.get("category")  # DB category for v1 (that's what it uses)
        print(f"  Source: {src['name'][:50]} ({src['brand']}, ${src['price']}, cat={cat})")

        # Run both
        v1_items, v1_time = run_v1(supabase, pid, cat)
        v2_items, v2_time = run_v2(supabase, pid, cat)

        print(f"  V1: {len(v1_items)} items in {v1_time:.2f}s (top cos: {v1_items[0]['similarity']:.3f})" if v1_items else f"  V1: 0 items in {v1_time:.2f}s")
        print(f"  V2: {len(v2_items)} items in {v2_time:.2f}s (top cos: {v2_items[0]['similarity']:.3f})" if v2_items else f"  V2: 0 items in {v2_time:.2f}s")

        # Overlap
        v1_ids = {str(r.get("product_id", "")) for r in v1_items}
        v2_ids = {str(r.get("product_id", "")) for r in v2_items}
        overlap = v1_ids & v2_ids
        print(f"  Overlap: {len(overlap)}/{len(v1_ids | v2_ids)} items in common")

        sections.append({
            "label": label,
            "source": src,
            "v1_items": v1_items,
            "v2_items": v2_items,
            "v1_time": v1_time,
            "v2_time": v2_time,
        })

    total_time = time.time() - t_total
    print(f"\nTotal time: {total_time:.1f}s")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "similar_v1_v2_comparison.html")
    report_html = generate_html(sections, total_time)
    with open(out_path, "w") as f:
        f.write(report_html)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
