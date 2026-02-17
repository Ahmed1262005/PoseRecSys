"""
Test ~230 real-world user queries and generate an HTML results page.

Usage:
    PYTHONPATH=src python scripts/test_20_queries_html.py
    # Opens: scripts/search_results.html
"""

import html
import json
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from search.hybrid_search import HybridSearchService
from search.models import HybridSearchRequest, SortBy

_QUERY_DATA = [
    # (query, category)
    # --- Coverage / body concerns ---
    ("Help me find a top that hides my arms", "Coverage"),
    ("Help me find a dress that doesn't show my stomach", "Coverage"),
    ("Help me find something that looks expensive", "Coverage / Vibe"),
    ("Help me find an outfit for a first date", "Coverage / Occasion"),
    ("Help me find something cute but not too try-hard", "Coverage / Vibe"),
    ("Help me find something modest but not frumpy", "Coverage / Modesty"),
    ("Help me find something sexy but classy", "Coverage / Vibe"),
    ("Help me find something I can wear 3 different ways", "Utility"),
    ("Help me find something breathable for hot weather", "Weather"),
    ("Help me find something warm but not bulky", "Weather"),
    ("Help me find something that won't wrinkle", "Care"),
    ("Help me find something that travels well", "Travel"),
    ("Help me find something that won't show sweat", "Performance"),
    ("Help me find something that doesn't show underwear lines", "Fit / Coverage"),
    # --- Occasion ---
    ("Outfit for a wedding guest", "Occasion"),
    ("Dress for a black tie wedding", "Occasion"),
    ("Semi formal wedding outfit", "Occasion"),
    ("Business casual outfits for work", "Occasion / Work"),
    ("Interview outfit for a creative job", "Occasion / Work"),
    ("Outfit for a work dinner", "Occasion / Work"),
    ("What to wear to a conference", "Occasion / Work"),
    ("Vacation outfits for Europe", "Occasion / Travel"),
    ("Beach dinner outfit", "Occasion"),
    ("Brunch outfit", "Occasion"),
    ("Night out outfit in winter", "Occasion / Weather"),
    ("Outfit for clubbing but not too revealing", "Occasion"),
    ("Date night outfit", "Occasion"),
    ("Casual Friday outfit", "Occasion / Work"),
    ("Outfit for family gathering", "Occasion"),
    ("Funeral outfit (simple and respectful)", "Occasion"),
    ("Eid outfit", "Occasion"),
    ("Ramadan iftar dinner outfit", "Occasion"),
    ("Graduation dress", "Occasion"),
    ("Birthday outfit", "Occasion"),
    ("Engagement party outfit", "Occasion"),
    # --- Aesthetic / vibe ---
    ("Quiet luxury outfit", "Aesthetic"),
    ("Old money style dress", "Aesthetic"),
    ("Clean girl outfit", "Aesthetic"),
    ("Model off duty look", "Aesthetic"),
    ("French girl outfit", "Aesthetic"),
    ("Scandi minimalist outfit", "Aesthetic"),
    ("Coastal grandmother outfit", "Aesthetic"),
    ("Y2K top", "Aesthetic"),
    ("Soft girl dress", "Aesthetic"),
    ("Mob wife coat", "Aesthetic"),
    ("Balletcore skirt", "Aesthetic"),
    ("Office siren outfit", "Aesthetic"),
    ("Dark academia outfit", "Aesthetic"),
    ("Coquette top with bows", "Aesthetic"),
    ("Edgy streetwear jacket", "Aesthetic"),
    ("Boho maxi dress like Anthropologie", "Brand Vibe"),
    ("Like Zara but better quality", "Brand Vibe"),
    ("Like Aritzia vibe basics", "Brand Vibe"),
    ("Reformation-style dress but cheaper", "Brand Vibe"),
    # --- Concrete attributes: outerwear ---
    ("Jacket with zippered pockets", "Outerwear"),
    ("Waterproof rain jacket with hood", "Outerwear"),
    ("Windbreaker with adjustable waist", "Outerwear"),
    ("Puffer jacket that's not too puffy", "Outerwear"),
    ("Wool coat with belt", "Outerwear"),
    ("Trench coat with storm flap", "Outerwear"),
    ("Leather jacket oversized", "Outerwear"),
    ("Bomber jacket cropped", "Outerwear"),
    ("Denim jacket lined", "Outerwear"),
    ("Coat with hidden buttons", "Outerwear"),
    ("Jacket with two-way zipper", "Outerwear"),
    ("Long coat that covers my butt", "Outerwear"),
    # --- Concrete attributes: tops ---
    ("Ribbed knit top with square neckline", "Tops"),
    ("Button down that doesn't gape at the chest", "Tops"),
    ("Wrap top that stays closed", "Tops"),
    ("T-shirt that's thick not see-through", "Tops"),
    ("Blouse with covered buttons", "Tops"),
    ("Top with longer sleeves", "Tops"),
    ("Cropped top but not too cropped", "Tops"),
    ("Longline tank", "Tops"),
    ("Bodysuit with snap closure", "Tops"),
    ("Top that isn't clingy", "Tops"),
    # --- Concrete attributes: bottoms ---
    ("High rise wide leg jeans", "Bottoms"),
    ("Mid rise straight jeans no rips", "Bottoms"),
    ("Low rise baggy jeans", "Bottoms"),
    ("Pants with elastic waistband but look tailored", "Bottoms"),
    ("Trousers with pleats and belt loops", "Bottoms"),
    ("Skirt with shorts underneath", "Bottoms"),
    ("Maxi skirt with slit", "Bottoms"),
    ("Pockets that don't flare out", "Bottoms"),
    ("Leggings squat proof", "Bottoms"),
    ("Shorts with 5 inch inseam", "Bottoms"),
    # --- Concrete attributes: dresses ---
    ("Midi dress with sleeves", "Dresses"),
    ("Maxi dress with open back", "Dresses"),
    ("Wrap dress but not too low cut", "Dresses"),
    ("Dress with corset bodice", "Dresses"),
    ("Slip dress satin", "Dresses"),
    ("Bodycon dress thick material", "Dresses"),
    ("Dress with pockets", "Dresses"),
    ("Dress with adjustable straps", "Dresses"),
    ("Dress with higher neckline", "Dresses"),
    ("Dress that covers shoulders", "Dresses"),
    # --- Fit / body type ---
    ("Jeans for short legs", "Fit / Body"),
    ("Pants for tall girls 5'10", "Fit / Body"),
    ("Petite blazer", "Fit / Body"),
    ("Long torso bodysuit", "Fit / Body"),
    ("Wide calf boots", "Fit / Body"),
    ("Skirt for big hips small waist", "Fit / Body"),
    ("Dresses for apple shape", "Fit / Body"),
    ("Outfit to hide belly", "Fit / Body"),
    ("Jeans that don't gap at waist", "Fit / Body"),
    ("Dress that doesn't cling to thighs", "Fit / Body"),
    ("Plus size wedding guest dress", "Fit / Size"),
    ("Maternity dress for wedding", "Fit / Size"),
    ("Postpartum friendly outfits", "Fit / Body"),
    ("Outfits that hide upper arms", "Fit / Coverage"),
    ("Outfits that cover my back", "Fit / Coverage"),
    # --- Coverage / modesty ---
    ("Long sleeves but lightweight", "Modesty"),
    ("Not see through", "Modesty"),
    ("No cleavage", "Modesty"),
    ("High neck top", "Modesty"),
    ("Full length maxi dress no slit", "Modesty"),
    ("Midi skirt that's not tight", "Modesty"),
    ("Loose fit pants modest", "Modesty"),
    ("Longline blazer for coverage", "Modesty"),
    ("No backless", "Modesty"),
    ("Not cropped", "Modesty"),
    ("Covers shoulders", "Modesty"),
    ("Hijab-friendly dress", "Modesty"),
    ("Modest wedding guest dress", "Modesty / Occasion"),
    # --- Color / print ---
    ("Chocolate brown dress", "Color"),
    ("Butter yellow top", "Color"),
    ("Cherry red mini dress", "Color"),
    ("Navy blazer", "Color"),
    ("Black dress not boring", "Color"),
    ("White top that isn't see-through", "Color"),
    ("Leopard print skirt", "Print"),
    ("Floral dress not grandma", "Print"),
    ("Striped knit top", "Print"),
    ("Polka dot midi dress", "Print"),
    ("Solid color basics", "Color"),
    ("Neutral capsule wardrobe pieces", "Color"),
    ("Monochrome beige outfit", "Color"),
    ("Colorful summer set", "Color"),
    # --- Season / weather ---
    ("Winter work outfits", "Season"),
    ("Summer dresses breathable", "Season"),
    ("Hot weather pants", "Weather"),
    ("Layering tops for cold office", "Weather"),
    ("Rainy day outfit", "Weather"),
    ("Vacation outfits for humid weather", "Weather / Travel"),
    ("Coat for 10 degrees", "Weather"),
    ("Outfit for windy weather", "Weather"),
    ("Transitional spring jacket", "Season"),
    ("Fall capsule wardrobe", "Season"),
    ("Ski trip outfits", "Season / Travel"),
    # --- Styling / outfit building ---
    ("Outfit ideas with a black blazer", "Styling"),
    ("What goes with wide leg jeans", "Styling"),
    ("Top to wear with a satin skirt", "Styling"),
    ("Shoes that go with this dress", "Styling"),
    ("Layering piece for this top", "Styling"),
    ("Find matching set", "Styling"),
    ("Find pants to match this blazer", "Styling"),
    ("Find a top that matches these pants", "Styling"),
    ("Find an outfit for this skirt", "Styling"),
    ("Find similar items to this", "Similarity"),
    ("Complete the look", "Styling"),
    # --- Similarity ---
    ("Similar to this jacket but cheaper", "Similarity"),
    ("Similar to this dress but longer", "Similarity"),
    ("Similar but with sleeves", "Similarity"),
    ("Same style but in black", "Similarity"),
    ("Same but more modest", "Similarity"),
    ("Same but less bodycon", "Similarity"),
    ("Dupes for this", "Similarity"),
    ("Something like this from Zara", "Similarity / Brand"),
    ("Something like Aritzia effortless pants", "Similarity / Brand"),
    ("Reformation-style floral midi", "Similarity / Brand"),
    ("Skims-like bodysuit but thicker", "Similarity / Brand"),
    ("Lululemon dupe leggings", "Similarity / Brand"),
    # --- Brand ---
    ("Aritzia-style basics", "Brand"),
    ("Anthropologie boho dress", "Brand"),
    ("COS minimalist dress", "Brand"),
    ("Zara blazer", "Brand"),
    ("H&M satin skirt", "Brand"),
    ("Abercrombie jeans curve love", "Brand"),
    ("Levi's 501 style jeans", "Brand"),
    ("Sézane cardigan style", "Brand"),
    ("Sandro-style tweed jacket", "Brand"),
    ("Like The Row vibes", "Brand"),
    ("Skims dupe", "Brand"),
    ("Reformation dupe", "Brand"),
    # --- Price / deals ---
    ("Under $50 date night dress", "Price"),
    ("Affordable work pants", "Price"),
    ("On sale coats", "Deals"),
    ("Best value basics", "Price"),
    ("Cheap but doesn't look cheap", "Price"),
    ("Designer-looking baggy jeans under $80", "Price"),
    ("Wedding guest dresses under $100", "Price"),
    ("Only show items on sale", "Deals"),
    ("Discounted matching sets", "Deals"),
    ("Clearance party dresses", "Deals"),
    # --- Fabric / feel ---
    ("100% cotton t-shirt", "Fabric"),
    ("Linen pants not see-through", "Fabric"),
    ("Wool coat", "Fabric"),
    ("Cashmere sweater", "Fabric"),
    ("Silk blouse", "Fabric"),
    ("Non-itchy sweater", "Fabric / Comfort"),
    ("Soft loungewear", "Fabric / Comfort"),
    ("Breathable fabrics", "Fabric"),
    ("Sweat-wicking top", "Fabric / Performance"),
    ("Quick dry shorts", "Fabric / Performance"),
    ("Stretchy but structured pants", "Fabric / Fit"),
    ("No pilling knit", "Fabric / Care"),
    ("Machine washable blazer", "Fabric / Care"),
    ("Wrinkle resistant dress", "Fabric / Care"),
    # --- Constraints / features ---
    ("With pockets", "Constraints"),
    ("With belt loops", "Constraints"),
    ("With adjustable straps", "Constraints"),
    ("With built in shorts", "Constraints"),
    ("With lining", "Constraints"),
    ("Double lined", "Constraints"),
    ("No polyester", "Constraints"),
    ("No itchy material", "Constraints"),
    ("No dry clean", "Constraints"),
    ("No rips / no distressing", "Constraints"),
    ("No logos", "Constraints"),
    ("No shoulder pads", "Constraints"),
    ("No slit", "Constraints"),
    ("No low back", "Constraints"),
    # --- Sizing ---
    ("Runs small?", "Sizing"),
    ("True to size?", "Sizing"),
    ("I'm between sizes", "Sizing"),
    ("I'm 5'3 which length should I get", "Sizing"),
    ("Petite friendly", "Sizing"),
    ("Tall friendly", "Sizing"),
    ("Plus size options", "Sizing"),
    ("What size should I buy", "Sizing"),
    ("This brand sizing is weird", "Sizing"),
    ("Something that fits a 34DD", "Sizing"),
    # --- Stress test ---
    ("black mini going out", "Stress Test"),
    ("top for jeans cute", "Stress Test"),
    ("wedding guest dress not ugly", "Stress Test"),
    ("coat warm cute", "Stress Test"),
    ("jeans no gap waist", "Stress Test"),
    ("not itchy sweater pls", "Stress Test"),
    ("need something last minute", "Stress Test"),
    ("something like this but long sleeve", "Stress Test"),
    ("tight but not too tight", "Stress Test"),
    ("formal but chill", "Stress Test"),
    ("office but hot", "Stress Test"),
    ("zip pockets jacket", "Stress Test"),
    # --- Dialogue / natural language ---
    ("Help me find a top that will be good for a work dinner but still cute", "Dialogue"),
    ("I need a jacket that has zippered pockets and looks minimal and modern", "Dialogue"),
    ("I need a dress that's flattering but I don't want to show my arms", "Dialogue"),
    ("I want pants that look tailored but feel like sweatpants", "Dialogue"),
    ("I'm going to a wedding and want something elegant not too revealing and not super expensive", "Dialogue"),
    ("I want a cardigan that looks expensive and isn't itchy", "Dialogue"),
    ("I need a going-out top that isn't cropped and doesn't show cleavage", "Dialogue"),
    ("Find me a travel outfit that's comfortable and still looks put together", "Dialogue"),
    ("I need a blazer that doesn't look boxy on me", "Dialogue"),
    ("I want an outfit for cold weather that still looks cute for a night out", "Dialogue"),
]

QUERIES = [q for q, _ in _QUERY_DATA]
CATEGORIES = [c for _, c in _QUERY_DATA]

TOP_N = 8


def run_tests():
    print("Initializing hybrid search service...")
    service = HybridSearchService()
    print("Service ready.\n")

    all_results = []
    total_start = time.time()

    for idx, query in enumerate(QUERIES):
        i = idx + 1
        print(f"[{i:3d}/{len(QUERIES)}] Searching: \"{query}\"...", end=" ", flush=True)

        request = HybridSearchRequest(
            query=query,
            page=1,
            page_size=20,
            sort_by=SortBy.RELEVANCE,
        )

        t_start = time.time()
        try:
            response = service.search(request)
            elapsed_ms = int((time.time() - t_start) * 1000)

            products = []
            for p in response.results[:TOP_N]:
                products.append({
                    "name": p.name,
                    "brand": p.brand,
                    "price": p.price,
                    "original_price": p.original_price,
                    "is_on_sale": p.is_on_sale,
                    "image_url": p.image_url,
                    "pattern": p.pattern,
                    "category_l2": p.category_l2,
                    "category_l1": p.category_l1,
                    "primary_color": p.primary_color,
                    "formality": p.formality,
                    "algolia_rank": p.algolia_rank,
                    "semantic_rank": p.semantic_rank,
                    "semantic_score": p.semantic_score,
                    "rrf_score": p.rrf_score,
                })

            algolia_count = sum(1 for r in response.results if r.algolia_rank)
            semantic_count = sum(1 for r in response.results if r.semantic_rank)

            all_results.append({
                "query": query,
                "category": CATEGORIES[idx],
                "intent": response.intent,
                "total": response.pagination.total_results or len(response.results),
                "timing": response.timing,
                "elapsed_ms": elapsed_ms,
                "products": products,
                "algolia_count": algolia_count,
                "semantic_count": semantic_count,
                "success": True,
            })
            print(f"{len(products)} results ({elapsed_ms}ms)")

        except Exception as e:
            elapsed_ms = int((time.time() - t_start) * 1000)
            all_results.append({
                "query": query,
                "category": CATEGORIES[idx],
                "intent": "error",
                "total": 0,
                "timing": {},
                "elapsed_ms": elapsed_ms,
                "products": [],
                "algolia_count": 0,
                "semantic_count": 0,
                "success": False,
                "error": str(e),
            })
            print(f"ERROR ({elapsed_ms}ms): {e}")

    total_elapsed = int(time.time() - total_start)
    generate_html(all_results, total_elapsed)


def generate_html(results, total_elapsed):
    succeeded = sum(1 for r in results if r["success"])
    with_results = sum(1 for r in results if r["total"] > 0)
    avg_time = sum(r["elapsed_ms"] for r in results) // len(results)

    # Build query cards
    cards_html = ""
    for i, r in enumerate(results):
        q = html.escape(r["query"])
        intent_class = r["intent"]
        intent_color = {"exact": "#3b82f6", "specific": "#10b981", "vague": "#8b5cf6", "error": "#ef4444"}.get(intent_class, "#6b7280")

        # Timing pills
        timing = r.get("timing", {})
        timing_pills = ""
        for key in ["planner_ms", "algolia_ms", "semantic_ms", "total_ms"]:
            if key in timing:
                label = key.replace("_ms", "").replace("_", " ").title()
                timing_pills += f'<span class="timing-pill">{label}: {timing[key]}ms</span>'

        # Source badge
        src_text = f'{r["algolia_count"]} Algolia / {r["semantic_count"]} Semantic'

        # Product grid
        products_html = ""
        if r["products"]:
            for p in r["products"]:
                name = html.escape(p["name"][:60])
                brand = html.escape(p["brand"])
                img = p.get("image_url") or ""
                price = p["price"]
                orig = p.get("original_price")
                on_sale = p.get("is_on_sale", False)
                pattern = html.escape(p.get("pattern") or "-")
                cat = html.escape(p.get("category_l2") or "-")
                color = html.escape(p.get("primary_color") or "-")

                # Source indicator
                has_alg = p.get("algolia_rank") is not None
                has_sem = p.get("semantic_rank") is not None
                if has_alg and has_sem:
                    source_badge = '<span class="source-badge both">Both</span>'
                elif has_alg:
                    source_badge = '<span class="source-badge algolia">Algolia</span>'
                else:
                    source_badge = '<span class="source-badge semantic">Semantic</span>'

                # Price display
                if on_sale and orig and orig > price:
                    price_html = f'<span class="price-sale">${price:.2f}</span> <span class="price-original">${orig:.2f}</span>'
                else:
                    price_html = f'<span class="price">${price:.2f}</span>'

                # Score
                rrf = p.get("rrf_score")
                score_html = f'<span class="score">RRF: {rrf:.4f}</span>' if rrf else ""

                products_html += f'''
                <div class="product-card">
                    <div class="product-image-container">
                        <img src="{img}" alt="{name}" class="product-image" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 200 260%22><rect fill=%22%23f3f4f6%22 width=%22200%22 height=%22260%22/><text x=%2250%%22 y=%2250%%22 text-anchor=%22middle%22 fill=%22%239ca3af%22 font-size=%2214%22>No Image</text></svg>'"/>
                        {source_badge}
                    </div>
                    <div class="product-info">
                        <p class="product-brand">{brand}</p>
                        <p class="product-name" title="{name}">{name}</p>
                        <div class="product-meta">
                            {price_html}
                        </div>
                        <div class="product-tags">
                            <span class="tag">{cat}</span>
                            <span class="tag">{pattern}</span>
                            <span class="tag">{color}</span>
                        </div>
                        {score_html}
                    </div>
                </div>'''
        else:
            products_html = '<div class="no-results">No results found</div>'

        cards_html += f'''
        <div class="query-section" id="query-{i+1}">
            <div class="query-header">
                <div class="query-number">#{i+1}</div>
                <div class="query-text-container">
                    <h2 class="query-text">"{q}"</h2>
                    <span class="category-label">{html.escape(r["category"])}</span>
                </div>
                <div class="query-meta">
                    <span class="intent-badge" style="background:{intent_color}">{intent_class}</span>
                    <span class="result-count">{r["total"]} results</span>
                    <span class="time-badge">{r["elapsed_ms"]}ms</span>
                </div>
            </div>
            <div class="timing-row">{timing_pills}</div>
            <div class="source-row">Sources: {src_text}</div>
            <div class="products-grid">{products_html}</div>
        </div>'''

    # Summary rows
    summary_rows = ""
    for i, r in enumerate(results):
        q = html.escape(r["query"])
        intent = r["intent"]
        color = {"exact": "#3b82f6", "specific": "#10b981", "vague": "#8b5cf6"}.get(intent, "#6b7280")
        count_class = "zero" if r["total"] == 0 else ("low" if r["total"] < 5 else "")
        summary_rows += f'''
        <tr onclick="document.getElementById('query-{i+1}').scrollIntoView({{behavior:'smooth'}})">
            <td class="row-num">{i+1}</td>
            <td class="row-query">{q}</td>
            <td><span class="intent-badge-sm" style="background:{color}">{intent}</span></td>
            <td class="row-count {count_class}">{r["total"]}</td>
            <td class="row-time">{r["elapsed_ms"]}ms</td>
        </tr>'''

    page_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Search Results - LLM Query Planner Test</title>
<style>
:root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242836;
    --border: #2e3345;
    --text: #e4e7ef;
    --text2: #9ca3b8;
    --accent: #7c6ef6;
    --green: #10b981;
    --blue: #3b82f6;
    --pink: #ec4899;
    --orange: #f59e0b;
    --red: #ef4444;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
}}
.container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}

/* Header */
.header {{
    text-align: center;
    padding: 48px 0 32px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
}}
.header h1 {{
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 8px;
    background: linear-gradient(135deg, var(--accent), var(--pink));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.header p {{ color: var(--text2); font-size: 14px; }}

/* Stats bar */
.stats-bar {{
    display: flex;
    gap: 16px;
    justify-content: center;
    margin: 24px 0;
    flex-wrap: wrap;
}}
.stat-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 24px;
    text-align: center;
    min-width: 140px;
}}
.stat-value {{ font-size: 28px; font-weight: 700; color: var(--accent); }}
.stat-label {{ font-size: 12px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.05em; }}

/* Summary table */
.summary-section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 40px;
}}
.summary-section h3 {{
    font-size: 16px;
    margin-bottom: 16px;
    color: var(--text2);
}}
.summary-table {{
    width: 100%;
    border-collapse: collapse;
}}
.summary-table th {{
    text-align: left;
    padding: 10px 12px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text2);
    border-bottom: 1px solid var(--border);
}}
.summary-table td {{
    padding: 10px 12px;
    font-size: 13px;
    border-bottom: 1px solid var(--border);
}}
.summary-table tr {{ cursor: pointer; transition: background 0.15s; }}
.summary-table tr:hover {{ background: var(--surface2); }}
.row-num {{ color: var(--text2); width: 40px; }}
.row-query {{ font-weight: 500; }}
.row-count {{ font-weight: 600; text-align: right; }}
.row-count.zero {{ color: var(--red); }}
.row-count.low {{ color: var(--orange); }}
.row-time {{ color: var(--text2); text-align: right; font-size: 12px; }}
.intent-badge-sm {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    color: white;
}}

/* Query sections */
.query-section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
}}
.query-header {{
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 12px;
    flex-wrap: wrap;
}}
.query-number {{
    background: var(--accent);
    color: white;
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 14px;
    flex-shrink: 0;
}}
.query-text-container {{ flex: 1; min-width: 200px; }}
.query-text {{ font-size: 18px; font-weight: 600; }}
.category-label {{
    font-size: 11px;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.query-meta {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
.intent-badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    color: white;
}}
.result-count {{
    font-size: 13px;
    font-weight: 600;
    color: var(--text2);
    background: var(--surface2);
    padding: 4px 10px;
    border-radius: 6px;
}}
.time-badge {{
    font-size: 12px;
    color: var(--text2);
    background: var(--surface2);
    padding: 4px 10px;
    border-radius: 6px;
}}
.timing-row {{
    display: flex;
    gap: 6px;
    margin-bottom: 8px;
    flex-wrap: wrap;
}}
.timing-pill {{
    font-size: 11px;
    color: var(--text2);
    background: var(--bg);
    padding: 3px 8px;
    border-radius: 4px;
    border: 1px solid var(--border);
}}
.source-row {{
    font-size: 12px;
    color: var(--text2);
    margin-bottom: 16px;
}}

/* Product grid */
.products-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 12px;
}}
.product-card {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    transition: transform 0.15s, box-shadow 0.15s;
}}
.product-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}}
.product-image-container {{
    position: relative;
    width: 100%;
    aspect-ratio: 3/4;
    background: #1e2130;
    overflow: hidden;
}}
.product-image {{
    width: 100%;
    height: 100%;
    object-fit: cover;
}}
.source-badge {{
    position: absolute;
    top: 6px;
    right: 6px;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}
.source-badge.algolia {{ background: var(--blue); color: white; }}
.source-badge.semantic {{ background: var(--green); color: white; }}
.source-badge.both {{ background: var(--pink); color: white; }}
.product-info {{ padding: 10px; }}
.product-brand {{
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--accent);
    margin-bottom: 2px;
}}
.product-name {{
    font-size: 12px;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 6px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    line-height: 1.4;
}}
.product-meta {{ margin-bottom: 6px; }}
.price {{ font-size: 14px; font-weight: 700; color: var(--text); }}
.price-sale {{ font-size: 14px; font-weight: 700; color: var(--red); }}
.price-original {{ font-size: 11px; color: var(--text2); text-decoration: line-through; }}
.product-tags {{ display: flex; gap: 4px; flex-wrap: wrap; }}
.tag {{
    font-size: 9px;
    padding: 1px 5px;
    border-radius: 3px;
    background: var(--bg);
    color: var(--text2);
    border: 1px solid var(--border);
}}
.score {{
    display: block;
    font-size: 10px;
    color: var(--text2);
    margin-top: 4px;
}}
.no-results {{
    grid-column: 1 / -1;
    text-align: center;
    padding: 40px;
    color: var(--red);
    font-weight: 500;
    background: rgba(239,68,68,0.08);
    border-radius: 8px;
}}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>LLM Query Planner - Search Results</h1>
        <p>{len(results)} real-world user queries tested through the full hybrid pipeline with gpt-4o-mini query planner</p>
        <p style="margin-top:4px;color:var(--text2);font-size:12px;">{datetime.now().strftime("%B %d, %Y at %H:%M")}</p>
    </div>

    <div class="stats-bar">
        <div class="stat-card">
            <div class="stat-value">{succeeded}/{len(results)}</div>
            <div class="stat-label">Queries Succeeded</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{with_results}/{len(results)}</div>
            <div class="stat-label">With Results</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_time}ms</div>
            <div class="stat-label">Avg Latency</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{total_elapsed}s</div>
            <div class="stat-label">Total Time</div>
        </div>
    </div>

    <div class="summary-section">
        <h3>All Queries (click to jump)</h3>
        <table class="summary-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Query</th>
                    <th>Intent</th>
                    <th style="text-align:right">Results</th>
                    <th style="text-align:right">Time</th>
                </tr>
            </thead>
            <tbody>{summary_rows}</tbody>
        </table>
    </div>

    {cards_html}
</div>
</body>
</html>'''

    out_path = "scripts/search_results.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page_html)
    print(f"\nHTML report saved to: {out_path}")


if __name__ == "__main__":
    run_tests()
