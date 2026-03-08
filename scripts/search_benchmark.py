"""
Search benchmark: run 140+ queries across categories and generate an HTML report.

Tests category distribution, brand diversity, intent detection, and result quality
across coverage, vibe, occasion, concrete attribute, fit, color, and season queries.

Run:
    PYTHONPATH=src python scripts/search_benchmark.py

The report shows per-query: category breakdown, brand distribution, timing,
planner output, and product cards.
"""
import os, sys, time, html as html_mod
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dotenv import load_dotenv
load_dotenv()

from search.hybrid_search import get_hybrid_search_service
from search.models import HybridSearchRequest

service = get_hybrid_search_service()

# ── All benchmark queries with tags ──────────────────────────────────────
QUERIES = [
    # Coverage / body
    ("coverage", "Help me find a top that hides my arms"),
    ("coverage", "Help me find a dress that doesn't show my stomach"),
    # Vibe
    ("vibe", "Help me find something that looks expensive"),
    # Occasion
    ("occasion", "Help me find an outfit for a first date"),
    # Vibe
    ("vibe", "Help me find something cute but not too try-hard"),
    # Coverage
    ("coverage", "Help me find something modest but not frumpy"),
    # Occasion
    ("occasion", "Help me find something sexy but classy"),
    # Utility
    ("utility", "Help me find something I can wear 3 different ways"),
    # Weather
    ("weather", "Help me find something breathable for hot weather"),
    ("weather", "Help me find something warm but not bulky"),
    # Care
    ("care", "Help me find something that won't wrinkle"),
    # Travel
    ("travel", "Help me find something that travels well"),
    # Performance
    ("performance", "Help me find something that won't show sweat"),
    # Fit
    ("fit", "Help me find something that doesn't show underwear lines"),
    # Occasion - events
    ("occasion", "Outfit for a wedding guest"),
    ("occasion", "Dress for a black tie wedding"),
    ("occasion", "Semi formal wedding outfit"),
    # Occasion - work
    ("work", "Business casual outfits for work"),
    ("work", "Interview outfit for a creative job"),
    ("work", "Outfit for a work dinner"),
    ("work", "What to wear to a conference"),
    # Occasion - travel/social
    ("travel", "Vacation outfits for Europe"),
    ("occasion", "Beach dinner outfit"),
    ("occasion", "Brunch outfit"),
    ("weather", "Night out outfit in winter"),
    ("occasion", "Outfit for clubbing but not too revealing"),
    ("occasion", "Date night outfit"),
    ("work", "Casual Friday outfit"),
    ("occasion", "Outfit for family gathering"),
    ("occasion", "Funeral outfit (simple and respectful)"),
    ("occasion", "Eid outfit"),
    ("occasion", "Ramadan iftar dinner outfit"),
    ("occasion", "Graduation dress"),
    ("occasion", "Birthday outfit"),
    ("occasion", "Engagement party outfit"),
    # Vibe - aesthetic
    ("aesthetic", "Quiet luxury outfit"),
    ("aesthetic", "Old money style dress"),
    ("aesthetic", "Clean girl outfit"),
    ("aesthetic", "Model off duty look"),
    ("aesthetic", "French girl outfit"),
    ("aesthetic", "Scandi minimalist outfit"),
    ("aesthetic", "Coastal grandmother outfit"),
    ("aesthetic", "Y2K top"),
    ("aesthetic", "Soft girl dress"),
    ("aesthetic", "Mob wife coat"),
    ("aesthetic", "Balletcore skirt"),
    ("aesthetic", "Office siren outfit"),
    ("aesthetic", "Dark academia outfit"),
    ("aesthetic", "Coquette top with bows"),
    ("aesthetic", "Edgy streetwear jacket"),
    # Vibe - brand vibe
    ("brand_vibe", "Boho maxi dress like Anthropologie"),
    ("brand_vibe", "Like Zara but better quality"),
    ("brand_vibe", "Like Aritzia vibe basics"),
    ("brand_vibe", "Reformation-style dress but cheaper"),
    # Concrete attributes - outerwear
    ("outerwear", "Jacket with zippered pockets"),
    ("outerwear", "Waterproof rain jacket with hood"),
    ("outerwear", "Windbreaker with adjustable waist"),
    ("outerwear", "Puffer jacket that's not too puffy"),
    ("outerwear", "Wool coat with belt"),
    ("outerwear", "Trench coat with storm flap"),
    ("outerwear", "Leather jacket oversized"),
    ("outerwear", "Bomber jacket cropped"),
    ("outerwear", "Denim jacket lined"),
    ("outerwear", "Coat with hidden buttons"),
    ("outerwear", "Jacket with two-way zipper"),
    ("outerwear", "Long coat that covers my butt"),
    # Concrete attributes - tops
    ("tops", "Ribbed knit top with square neckline"),
    ("tops", "Button down that doesn't gape at the chest"),
    ("tops", "Wrap top that stays closed"),
    ("tops", "T-shirt that's thick not see-through"),
    ("tops", "Blouse with covered buttons"),
    ("tops", "Top with longer sleeves"),
    ("tops", "Cropped top but not too cropped"),
    ("tops", "Longline tank"),
    ("tops", "Bodysuit with snap closure"),
    ("tops", "Top that isn't clingy"),
    # Concrete attributes - bottoms
    ("bottoms", "High rise wide leg jeans"),
    ("bottoms", "Mid rise straight jeans no rips"),
    ("bottoms", "Low rise baggy jeans"),
    ("bottoms", "Pants with elastic waistband but look tailored"),
    ("bottoms", "Trousers with pleats and belt loops"),
    ("bottoms", "Skirt with shorts underneath"),
    ("bottoms", "Maxi skirt with slit"),
    ("bottoms", "Pockets that don't flare out"),
    ("bottoms", "Leggings squat proof"),
    ("bottoms", "Shorts with 5 inch inseam"),
    # Concrete attributes - dresses
    ("dresses", "Midi dress with sleeves"),
    ("dresses", "Maxi dress with open back"),
    ("dresses", "Wrap dress but not too low cut"),
    ("dresses", "Dress with corset bodice"),
    ("dresses", "Slip dress satin"),
    ("dresses", "Bodycon dress thick material"),
    ("dresses", "Dress with pockets"),
    ("dresses", "Dress with adjustable straps"),
    ("dresses", "Dress with higher neckline"),
    ("dresses", "Dress that covers shoulders"),
    # Fit / body
    ("fit", "Jeans for short legs"),
    ("fit", "Pants for tall girls 5'10"),
    ("fit", "Petite blazer"),
    ("fit", "Long torso bodysuit"),
    ("fit", "Wide calf boots"),
    ("fit", "Skirt for big hips small waist"),
    ("fit", "Dresses for apple shape"),
    ("fit", "Outfit to hide belly"),
    ("fit", "Jeans that don't gap at waist"),
    ("fit", "Dress that doesn't cling to thighs"),
    ("size", "Plus size wedding guest dress"),
    ("size", "Maternity dress for wedding"),
    ("fit", "Postpartum friendly outfits"),
    ("coverage", "Outfits that hide upper arms"),
    ("coverage", "Outfits that cover my back"),
    # Coverage / modesty
    ("coverage", "Long sleeves but lightweight"),
    ("coverage", "Not see through"),
    ("coverage", "No cleavage"),
    ("coverage", "High neck top"),
    ("coverage", "Full length maxi dress no slit"),
    ("coverage", "Midi skirt that's not tight"),
    ("coverage", "Loose fit pants modest"),
    ("coverage", "Longline blazer for coverage"),
    ("coverage", "No backless"),
    ("coverage", "Not cropped"),
    ("coverage", "Covers shoulders"),
    ("coverage", "Hijab-friendly dress"),
    ("coverage", "Modest wedding guest dress"),
    # Color / print
    ("color", "Chocolate brown dress"),
    ("color", "Butter yellow top"),
    ("color", "Cherry red mini dress"),
    ("color", "Navy blazer"),
    ("color", "Black dress not boring"),
    ("color", "White top that isn't see-through"),
    ("print", "Leopard print skirt"),
    ("print", "Floral dress not grandma"),
    ("print", "Striped knit top"),
    ("print", "Polka dot midi dress"),
    ("basics", "Solid color basics"),
    ("basics", "Neutral capsule wardrobe pieces"),
    ("color", "Monochrome beige outfit"),
    ("color", "Colorful summer set"),
    # Season / weather
    ("season", "Winter work outfits"),
    ("season", "Summer dresses breathable"),
    ("weather", "Hot weather pants"),
    ("work", "Layering tops for cold office"),
    ("weather", "Rainy day outfit"),
    ("travel", "Vacation outfits for humid weather"),
    ("weather", "Coat for 10 degrees"),
    ("weather", "Outfit for windy weather"),
]


# ── Run searches ─────────────────────────────────────────────────────────
results = []
total_queries = len(QUERIES)
for idx, (tag, query) in enumerate(QUERIES, 1):
    print(f"[{idx}/{total_queries}] ({tag}) \"{query}\"")
    request = HybridSearchRequest(query=query, page_size=20)
    t0 = time.time()
    try:
        response = service.search(request=request)
        elapsed = time.time() - t0
        total_results = response.pagination.total_results if response.pagination else len(response.results)
        timing = response.timing or {}

        # Category distribution from results
        cat_counts = Counter()
        for r in response.results:
            cat = (
                getattr(r, "category_l1", None)
                or getattr(r, "broad_category", None)
                or getattr(r, "article_type", None)
                or "Unknown"
            )
            cat_counts[cat] += 1

        # Brand distribution
        brand_counts = Counter()
        for r in response.results:
            brand_counts[r.brand or "Unknown"] += 1

        results.append({
            "tag": tag,
            "query": query,
            "total": total_results,
            "shown": len(response.results),
            "elapsed": elapsed,
            "products": response.results[:20],
            "cat_counts": dict(cat_counts),
            "brand_counts": dict(brand_counts),
            "intent": timing.get("plan_intent", "?"),
            "plan_algolia": timing.get("plan_algolia_query", ""),
            "plan_semantic": timing.get("plan_semantic_queries", []),
            "plan_modes": timing.get("plan_modes", []),
            "plan_filters": timing.get("plan_applied_filters", {}),
            "plan_vibe": timing.get("plan_vibe_brand"),
            "timing": timing,
            "error": None,
        })
        # Compact progress
        cats_str = ", ".join(f"{c}:{n}" for c, n in cat_counts.most_common(4))
        print(f"  {elapsed:.1f}s | {len(response.results)} results | {cats_str}")

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ERROR ({elapsed:.1f}s): {e}")
        results.append({
            "tag": tag, "query": query, "total": 0, "shown": 0,
            "elapsed": elapsed, "products": [], "cat_counts": {},
            "brand_counts": {}, "intent": "?", "plan_algolia": "",
            "plan_semantic": [], "plan_modes": [], "plan_filters": {},
            "plan_vibe": None, "timing": {}, "error": str(e),
        })


# ── Generate HTML ────────────────────────────────────────────────────────
def product_card(product):
    name = html_mod.escape(getattr(product, 'name', '?') or '?')[:55]
    brand = html_mod.escape(getattr(product, 'brand', '?') or '?')
    price = getattr(product, 'price', 0) or 0
    img = html_mod.escape(getattr(product, 'image_url', '') or '')
    score = getattr(product, 'rrf_score', 0) or 0
    cat = html_mod.escape(
        getattr(product, 'category_l1', None)
        or getattr(product, 'broad_category', None)
        or '?'
    )
    alg_rank = getattr(product, 'algolia_rank', None)
    sem_rank = getattr(product, 'semantic_rank', None)
    source_parts = []
    if alg_rank: source_parts.append(f"A#{alg_rank}")
    if sem_rank: source_parts.append(f"S#{sem_rank}")
    source = html_mod.escape(" ".join(source_parts) if source_parts else "—")

    return f"""
    <div style="border:1px solid #ddd; border-radius:10px; padding:10px; width:180px;
                display:inline-block; vertical-align:top; margin:4px; background:#fff;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
      <img src="{img}" style="width:160px; height:213px; object-fit:cover; border-radius:6px;"
           onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22160%22 height=%22213%22><rect fill=%22%23eee%22 width=%22160%22 height=%22213%22/><text x=%2250%%22 y=%2250%%22 font-size=%2212%22 fill=%22%23999%22 text-anchor=%22middle%22>No image</text></svg>'">
      <div style="margin-top:6px;">
        <div style="font-weight:600; font-size:11px; line-height:1.3; min-height:28px;">{name}</div>
        <div style="color:#666; font-size:11px; margin-top:2px;">
          <span style="font-weight:600; color:#333;">{brand}</span> &middot; ${price:.0f}
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:3px;">
          <span style="color:#09c; font-size:9px;">{score:.3f}</span>
          <span style="background:#eef; color:#669; font-size:9px; padding:1px 4px; border-radius:3px;">{cat}</span>
          <span style="color:#888; font-size:9px;">{source}</span>
        </div>
      </div>
    </div>"""


def cat_bar(cat_counts):
    if not cat_counts:
        return "<span style='color:#999;'>—</span>"
    total = sum(cat_counts.values())
    CAT_COLORS = {
        "Tops": "#4a90d9", "Bottoms": "#d94a4a", "Dresses": "#9b59b6",
        "Outerwear": "#e67e22", "Accessories": "#2ecc71",
    }
    parts = []
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        color = CAT_COLORS.get(cat, "#95a5a6")
        parts.append(
            f'<div style="display:inline-block; background:{color}; color:#fff; '
            f'padding:2px 6px; border-radius:3px; font-size:10px; margin:1px 2px;">'
            f'{html_mod.escape(str(cat))} {count} ({pct:.0f}%)</div>'
        )
    return "".join(parts)


def brand_bar_compact(brand_counts, max_brands=8):
    if not brand_counts:
        return ""
    sorted_brands = sorted(brand_counts.items(), key=lambda x: -x[1])[:max_brands]
    return ", ".join(
        f"<b>{html_mod.escape(b)}</b>({c})" for b, c in sorted_brands
    )


# Group by tag
from collections import OrderedDict
tag_groups = OrderedDict()
for r in results:
    tag_groups.setdefault(r["tag"], []).append(r)

# Summary stats
total_time = sum(r["elapsed"] for r in results)
total_with_results = sum(1 for r in results if r["total"] > 0)
total_errors = sum(1 for r in results if r["error"])

# Global category distribution across ALL queries
global_cats = Counter()
for r in results:
    for cat, count in r["cat_counts"].items():
        global_cats[cat] += count

sections_html = []
for tag, group in tag_groups.items():
    # Tag header
    tag_cats = Counter()
    for r in group:
        for cat, count in r["cat_counts"].items():
            tag_cats[cat] += count

    sections_html.append(f"""
    <div style="margin-top:30px;">
      <h2 style="color:#1a1a2e; border-bottom:3px solid #4a90d9; padding-bottom:4px;
                  display:inline-block; text-transform:uppercase; font-size:16px; letter-spacing:1px;">
        {html_mod.escape(tag)} <span style="color:#999; font-weight:normal; font-size:13px;">({len(group)} queries)</span>
      </h2>
      <div style="margin:6px 0;">{cat_bar(dict(tag_cats))}</div>
    </div>
    """)

    for r in group:
        error_html = ""
        if r["error"]:
            error_html = f'<span style="color:#c00; font-size:11px;">ERROR: {html_mod.escape(r["error"])}</span>'

        filters_str = html_mod.escape(str(r["plan_filters"])) if r["plan_filters"] else "(none)"
        modes_str = ", ".join(r["plan_modes"]) or "(none)"
        vibe_str = f' | vibe={r["plan_vibe"]}' if r["plan_vibe"] else ""

        cards = "".join(product_card(p) for p in r["products"][:12])
        if not cards:
            cards = '<p style="color:#999;">No results</p>'

        sections_html.append(f"""
        <div style="margin:16px 0 24px 0; padding:12px; background:#fff; border:1px solid #e1e4e8;
                    border-radius:8px; box-shadow:0 1px 2px rgba(0,0,0,0.04);">
          <div style="font-weight:700; font-size:14px; color:#222;">"{html_mod.escape(r['query'])}"</div>
          <div style="margin:4px 0; font-size:11px; color:#666;">
            {r['elapsed']:.1f}s | {r['shown']} results | intent: {html_mod.escape(str(r['intent']))}{vibe_str}
            {error_html}
          </div>
          <div style="margin:4px 0;">{cat_bar(r['cat_counts'])}</div>
          <div style="font-size:10px; color:#888; margin:2px 0;">
            Brands: {brand_bar_compact(r['brand_counts'])}
          </div>
          <details style="margin-top:6px;">
            <summary style="cursor:pointer; font-size:11px; color:#4a90d9;">Planner details</summary>
            <div style="font-size:10px; color:#555; margin-top:4px; padding:6px; background:#f8f9fa; border-radius:4px;">
              <b>Algolia:</b> <code>{html_mod.escape(r['plan_algolia'] or '(empty)')}</code><br>
              <b>Modes:</b> {modes_str}<br>
              <b>Filters:</b> <code>{filters_str}</code><br>
              <b>Semantic:</b><br>
              {'<br>'.join(f'&nbsp;&nbsp;{i+1}. <i>{html_mod.escape(sq)}</i>' for i, sq in enumerate(r['plan_semantic'][:4])) or '(none)'}
            </div>
          </details>
          <div style="margin-top:8px; overflow-x:auto; white-space:nowrap;">{cards}</div>
        </div>
        """)

full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Search Benchmark Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1600px; margin: 20px auto; padding: 0 20px; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; margin-bottom: 4px; }}
code {{ background: #e8edf2; padding: 1px 5px; border-radius: 3px; font-size: 10px; }}
details > summary {{ outline: none; }}
</style></head><body>
<h1>Search Benchmark Report</h1>
<p style="color:#666; margin-top:0; font-size:13px;">
  {total_queries} queries | {total_with_results} with results | {total_errors} errors | {total_time:.0f}s total
</p>
<div style="background:#fff; border:1px solid #ddd; border-radius:8px; padding:14px; margin:10px 0;">
  <b>Global category distribution:</b><br>
  <div style="margin-top:6px;">{cat_bar(dict(global_cats))}</div>
</div>
{''.join(sections_html)}
</body></html>"""

out_path = os.path.join(os.path.dirname(__file__), "search_benchmark.html")
with open(out_path, "w") as f:
    f.write(full_html)
print(f"\nReport written to {out_path}")
print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"Results: {total_with_results}/{total_queries} queries returned results")
