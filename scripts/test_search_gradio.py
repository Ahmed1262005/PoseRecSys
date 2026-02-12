"""
Gradio test UI for the Hybrid Search API.

Enhanced search-focused UI with:
- All 23 filter fields with multi-select support
- Gallery image carousel in result cards
- Session deduplication
- Pagination controls
- Click analytics tracking
- Side-by-side query comparison
- Filter summary display
- Comprehensive quick tests

Usage:
    1. Start the API server:
       PYTHONPATH=src uvicorn api.app:app --port 8000

    2. Run this script:
       python scripts/test_search_gradio.py

    3. Open http://localhost:7860 in your browser
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

# Generate JWT token for auth
def _make_token(user_id: str = "test-gradio-user") -> str:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
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

# Session tracking for deduplication
_session_id: str = str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Facet values (for dropdowns) - comprehensive lists from Algolia config
# ---------------------------------------------------------------------------

BROAD_CATEGORIES = ["tops", "bottoms", "dresses", "outerwear"]
CATEGORIES_L1 = ["Tops", "Bottoms", "Dresses", "Outerwear", "Activewear", "Swimwear", "Shoes", "Intimates"]
CATEGORIES_L2 = [
    "Blouse", "T-Shirt", "Tank Top", "Camisole", "Bodysuit", "Sweater", "Cardigan",
    "Hoodie", "Sweatshirt", "Crop Top", "Tunic", "Polo",
    "Jeans", "Trousers", "Shorts", "Skirt", "Leggings", "Wide Leg Pants", "Joggers",
    "Midi Dress", "Mini Dress", "Maxi Dress", "Wrap Dress", "Shirt Dress", "Slip Dress",
    "Blazer", "Jacket", "Coat", "Puffer", "Trench Coat", "Denim Jacket",
    "Jumpsuit", "Romper", "Playsuit",
]
ARTICLE_TYPES = [
    "jeans", "t-shirt", "midi dress", "mini dress", "maxi dress", "blouse",
    "sweater", "cardigan", "hoodie", "tank top", "crop top", "bodysuit",
    "trousers", "shorts", "skirt", "leggings", "joggers",
    "blazer", "jacket", "coat", "puffer jacket", "denim jacket",
    "jumpsuit", "romper", "shirt", "polo", "tunic",
]
STYLE_TAGS = [
    "Minimalist", "Boho", "Bohemian", "Preppy", "Streetwear", "Y2K", "Coquette",
    "Dark Academia", "Clean Girl", "Old Money", "Quiet Luxury", "Coastal",
    "Cottagecore", "Grunge", "Romantic", "Sporty", "Classic", "Trendy",
    "Vintage", "Retro", "Elegant", "Edgy", "Feminine", "Tomboy",
]
FORMALITY = ["Casual", "Smart Casual", "Business Casual", "Semi-Formal", "Formal"]
FIT_TYPES = ["Slim", "Regular", "Fitted", "Relaxed", "Loose", "Oversized"]
PATTERNS = ["Solid", "Floral", "Striped", "Plaid", "Polka Dot", "Animal Print", "Geometric", "Abstract", "Colorblock", "Tie Dye", "Camo"]
COLORS = ["Black", "White", "Blue", "Red", "Pink", "Green", "Navy Blue", "Beige", "Brown", "Gray", "Cream", "Burgundy", "Multi", "Purple", "Orange", "Yellow", "Coral", "Teal", "Ivory", "Olive"]
COLOR_FAMILIES = ["Neutrals", "Blacks", "Blues", "Pinks", "Reds", "Greens", "Browns", "Purples", "Yellows", "Oranges", "Multi"]
NECKLINES = ["V-Neck", "Crew", "Round", "Scoop", "Square", "Turtleneck", "Collared", "Strapless", "Off-Shoulder", "Halter", "Sweetheart", "Cowl", "Hooded", "Mock Neck", "Boat Neck"]
SLEEVES = ["Sleeveless", "Short", "Long", "3/4", "Cap", "Puff", "Bell", "Flutter", "Balloon", "Dolman"]
LENGTHS = ["Mini", "Midi", "Maxi", "Cropped", "Regular", "Knee-Length", "Ankle", "Floor-length"]
SILHOUETTES = ["A-Line", "Fitted", "Straight", "Bodycon", "Shift", "Wrap", "Fit & Flare", "Oversized", "Relaxed", "Column", "Wide Leg", "Tapered", "Flared", "Peplum"]
RISES = ["High", "Mid", "Low", "Ultra High", "Regular"]
FABRICS = ["Cotton", "Polyester", "Silk", "Linen", "Denim", "Wool", "Cashmere", "Satin", "Chiffon", "Velvet", "Leather", "Faux Leather", "Jersey", "Knit", "Lace", "Mesh", "Fleece", "Rayon", "Nylon", "Spandex"]
SEASONS = ["Spring", "Summer", "Fall", "Winter"]
OCCASIONS = ["Everyday", "Office", "Date Night", "Party", "Wedding", "Brunch", "Beach", "Workout", "Casual", "Vacation", "Nightclub", "Formal Event", "Weekend", "Festival", "Travel"]
BRANDS_POPULAR = ["Boohoo", "Missguided", "Forever 21", "Princess Polly", "Nasty Gal", "Reformation", "Free People", "White House Black Market", "Aje", "Alo Yoga", "Skims", "Universal Standard", "Club Monaco", "Old Navy", "Rouje", "American Eagle Outfitters", "The Frankie Shop", "Rails", "J.Crew", "Ann Taylor", "Gap", "Ba&sh", "Scotch & Soda", "L'AGENCE", "Rag & Bone", "Joe's Jeans", "Pull&Bear", "Abercrombie & Fitch", "Re/Done", "Staud"]

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.result-card {
    display: flex;
    gap: 14px;
    padding: 14px;
    margin: 8px 0;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    background: #fafafa;
    transition: box-shadow 0.2s;
}
.result-card:hover {
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
.card-image-container {
    position: relative;
    flex-shrink: 0;
}
.card-image {
    width: 130px;
    height: 175px;
    object-fit: cover;
    border-radius: 8px;
    cursor: pointer;
}
.card-image-placeholder {
    width: 130px;
    height: 175px;
    background: #eee;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #999;
    font-size: 12px;
}
.gallery-thumbs {
    display: flex;
    gap: 3px;
    margin-top: 4px;
    overflow-x: auto;
    max-width: 130px;
}
.gallery-thumb {
    width: 28px;
    height: 36px;
    object-fit: cover;
    border-radius: 4px;
    cursor: pointer;
    opacity: 0.7;
    transition: opacity 0.2s;
    border: 1px solid #ddd;
}
.gallery-thumb:hover {
    opacity: 1;
    border-color: #666;
}
.card-body {
    flex: 1;
    min-width: 0;
}
.card-title {
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 3px;
    line-height: 1.3;
}
.card-brand-price {
    color: #555;
    font-size: 13px;
    margin-bottom: 4px;
}
.card-sale {
    color: #e74c3c;
    font-weight: bold;
}
.card-rank {
    color: #888;
    font-size: 11px;
    margin-top: 4px;
    font-family: monospace;
}
.card-attrs {
    color: #555;
    font-size: 11px;
    margin-top: 4px;
    line-height: 1.5;
}
.card-tags {
    margin-top: 4px;
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
}
.card-tag {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 10px;
    font-size: 10px;
    background: #eef2ff;
    color: #4338ca;
}
.card-tag-occasion {
    background: #fef3c7;
    color: #92400e;
}
.card-tag-season {
    background: #d1fae5;
    color: #065f46;
}
.card-id {
    color: #bbb;
    font-size: 10px;
    margin-top: 3px;
    word-break: break-all;
    font-family: monospace;
}
.filter-summary {
    background: #f0f4ff;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 12px;
    margin-bottom: 8px;
    border: 1px solid #c7d2fe;
}
.filter-chip {
    display: inline-block;
    padding: 2px 8px;
    margin: 2px 3px;
    background: #818cf8;
    color: white;
    border-radius: 12px;
    font-size: 11px;
}
.intent-exact { color: #059669; font-weight: bold; }
.intent-specific { color: #2563eb; font-weight: bold; }
.intent-vague { color: #7c3aed; font-weight: bold; }
.timing-fast { color: #059669; }
.timing-medium { color: #d97706; }
.timing-slow { color: #dc2626; }
.pagination-bar {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 12px;
    margin: 16px 0;
    padding: 10px;
    background: #f9fafb;
    border-radius: 8px;
}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _multi_to_list(val):
    """Convert Gradio multi-select value to list or None."""
    if not val:
        return None
    if isinstance(val, str):
        return [val] if val.strip() else None
    if isinstance(val, list):
        filtered = [v for v in val if v and v.strip()]
        return filtered if filtered else None
    return None


def _build_filter_summary(body: dict) -> str:
    """Build an HTML filter summary from the request body."""
    filter_fields = [
        ("categories", "Category (broad)"), ("category_l1", "Category L1"),
        ("category_l2", "Category L2"), ("article_type", "Article Type"),
        ("brands", "Brand"), ("exclude_brands", "Exclude Brand"),
        ("colors", "Color"), ("color_family", "Color Family"),
        ("patterns", "Pattern"), ("materials", "Material"),
        ("occasions", "Occasion"), ("seasons", "Season"),
        ("formality", "Formality"), ("fit_type", "Fit"),
        ("neckline", "Neckline"), ("sleeve_type", "Sleeve"),
        ("length", "Length"), ("rise", "Rise"),
        ("silhouette", "Silhouette"), ("style_tags", "Style"),
    ]
    chips = []
    for field, label in filter_fields:
        vals = body.get(field)
        if vals:
            for v in (vals if isinstance(vals, list) else [vals]):
                chips.append(f'<span class="filter-chip">{label}: {v}</span>')

    if body.get("min_price") and float(body["min_price"]) > 0:
        chips.append(f'<span class="filter-chip">Min: ${body["min_price"]}</span>')
    if body.get("max_price") and float(body["max_price"]) > 0:
        chips.append(f'<span class="filter-chip">Max: ${body["max_price"]}</span>')
    if body.get("on_sale_only"):
        chips.append('<span class="filter-chip">On Sale</span>')

    if not chips:
        return ""
    return f'<div class="filter-summary">Active filters: {"".join(chips)}</div>'


def _format_timing(ms: int) -> str:
    """Color-code timing values."""
    if ms < 500:
        return f'<span class="timing-fast">{ms}ms</span>'
    elif ms < 2000:
        return f'<span class="timing-medium">{ms}ms</span>'
    else:
        return f'<span class="timing-slow">{ms}ms</span>'


def _build_product_card(i: int, p: dict, query: str = "") -> str:
    """Build an HTML card for a single product result."""
    # Rank info
    rank_parts = []
    if p.get("algolia_rank"):
        rank_parts.append(f"Algolia #{p['algolia_rank']}")
    if p.get("semantic_rank"):
        rank_parts.append(f"Semantic #{p['semantic_rank']}")
    if p.get("rrf_score"):
        rank_parts.append(f"RRF: {p['rrf_score']:.4f}")
    if p.get("semantic_score"):
        rank_parts.append(f"Sim: {p['semantic_score']:.3f}")
    rank_str = " | ".join(rank_parts)

    # Attributes
    attr_items = []
    for attr_key, attr_label in [
        ("category_l1", "Cat"), ("category_l2", "Type"), ("article_type", "Article"),
        ("formality", "Formality"), ("fit_type", "Fit"), ("primary_color", "Color"),
        ("color_family", "Family"), ("pattern", "Pattern"), ("apparent_fabric", "Fabric"),
        ("silhouette", "Silhouette"), ("neckline", "Neckline"), ("sleeve_type", "Sleeve"),
        ("length", "Length"), ("rise", "Rise"),
    ]:
        v = p.get(attr_key)
        if v:
            attr_items.append(f"<b>{attr_label}:</b> {v}")
    attrs_html = " &middot; ".join(attr_items) if attr_items else ""

    # Tags
    tags_html = ""
    tag_parts = []
    for tag in (p.get("style_tags") or [])[:4]:
        tag_parts.append(f'<span class="card-tag">{tag}</span>')
    for occ in (p.get("occasions") or [])[:3]:
        tag_parts.append(f'<span class="card-tag card-tag-occasion">{occ}</span>')
    for sea in (p.get("seasons") or [])[:2]:
        tag_parts.append(f'<span class="card-tag card-tag-season">{sea}</span>')
    if tag_parts:
        tags_html = f'<div class="card-tags">{"".join(tag_parts)}</div>'

    # Price & sale
    price_str = f"${p.get('price', 0):.2f}"
    sale_html = ""
    if p.get("is_on_sale") and p.get("original_price"):
        sale_html = f' <span class="card-sale">SALE</span> <s>${p["original_price"]:.0f}</s>'

    # Main image
    img_url = p.get("image_url") or ""
    if img_url:
        img_html = f'<img class="card-image" src="{img_url}" alt="{p.get("name","")[:40]}" loading="lazy" />'
    else:
        img_html = '<div class="card-image-placeholder">No Image</div>'

    # Gallery thumbnails
    gallery = p.get("gallery_images") or []
    gallery_html = ""
    if gallery and len(gallery) > 1:
        thumbs = []
        for gimg in gallery[:4]:
            thumbs.append(f'<img class="gallery-thumb" src="{gimg}" loading="lazy" />')
        if len(gallery) > 4:
            thumbs.append(f'<span style="font-size:10px;color:#888;align-self:center;">+{len(gallery)-4}</span>')
        gallery_html = f'<div class="gallery-thumbs">{"".join(thumbs)}</div>'

    product_id = p.get("product_id", "")

    return f"""
<div class="result-card" data-product-id="{product_id}">
  <div class="card-image-container">
    {img_html}
    {gallery_html}
  </div>
  <div class="card-body">
    <div class="card-title">#{i+1} {p.get('name','')[:90]}</div>
    <div class="card-brand-price">{p.get('brand','')} &mdash; <b>{price_str}</b>{sale_html}</div>
    <div class="card-rank">{rank_str}</div>
    <div class="card-attrs">{attrs_html}</div>
    {tags_html}
    <div class="card-id">{product_id}</div>
  </div>
</div>"""


def _new_session():
    """Generate a new session ID."""
    global _session_id
    _session_id = str(uuid.uuid4())
    return _session_id


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def do_search(
    query, page,
    # Category filters
    categories, category_l1, category_l2, article_type,
    # Brand filters
    brands, exclude_brands,
    # Color filters
    colors, color_family,
    # Pattern & Material
    patterns, materials,
    # Style & Occasion
    style_tags, formality, occasions, seasons,
    # Construction
    fit_type, neckline, sleeve_type, length, silhouette, rise,
    # Price & Options
    min_price, max_price, on_sale_only, page_size, semantic_boost,
    # Session
    use_session,
):
    """Call the hybrid search API and return formatted results."""
    if not query or not query.strip():
        return "Enter a search query.", "", "", ""

    # Debug: log the raw query received from Gradio
    print(f"[GRADIO] Raw query: {repr(query)}")

    body = {
        "query": query.strip(),
        "page": int(page) if page else 1,
        "page_size": int(page_size),
    }

    # Category filters
    if _multi_to_list(categories):
        body["categories"] = _multi_to_list(categories)
    if _multi_to_list(category_l1):
        body["category_l1"] = _multi_to_list(category_l1)
    if _multi_to_list(category_l2):
        body["category_l2"] = _multi_to_list(category_l2)
    if _multi_to_list(article_type):
        body["article_type"] = _multi_to_list(article_type)

    # Brand filters
    if _multi_to_list(brands):
        body["brands"] = _multi_to_list(brands)
    if _multi_to_list(exclude_brands):
        body["exclude_brands"] = _multi_to_list(exclude_brands)

    # Color filters
    if _multi_to_list(colors):
        body["colors"] = _multi_to_list(colors)
    if _multi_to_list(color_family):
        body["color_family"] = _multi_to_list(color_family)

    # Pattern & Material
    if _multi_to_list(patterns):
        body["patterns"] = _multi_to_list(patterns)
    if _multi_to_list(materials):
        body["materials"] = _multi_to_list(materials)

    # Style & Occasion
    if _multi_to_list(style_tags):
        body["style_tags"] = _multi_to_list(style_tags)
    if _multi_to_list(formality):
        body["formality"] = _multi_to_list(formality)
    if _multi_to_list(occasions):
        body["occasions"] = _multi_to_list(occasions)
    if _multi_to_list(seasons):
        body["seasons"] = _multi_to_list(seasons)

    # Construction
    if _multi_to_list(fit_type):
        body["fit_type"] = _multi_to_list(fit_type)
    if _multi_to_list(neckline):
        body["neckline"] = _multi_to_list(neckline)
    if _multi_to_list(sleeve_type):
        body["sleeve_type"] = _multi_to_list(sleeve_type)
    if _multi_to_list(length):
        body["length"] = _multi_to_list(length)
    if _multi_to_list(silhouette):
        body["silhouette"] = _multi_to_list(silhouette)
    if _multi_to_list(rise):
        body["rise"] = _multi_to_list(rise)

    # Price
    if min_price and float(min_price) > 0:
        body["min_price"] = float(min_price)
    if max_price and float(max_price) > 0:
        body["max_price"] = float(max_price)
    if on_sale_only:
        body["on_sale_only"] = True
    if semantic_boost != 0.4:
        body["semantic_boost"] = float(semantic_boost)

    # Session
    if use_session:
        body["session_id"] = _session_id

    try:
        t = time.time()
        r = requests.post(f"{SEARCH_URL}/hybrid", json=body, headers=HEADERS, timeout=30)
        elapsed = time.time() - t

        if r.status_code != 200:
            return f"**Error {r.status_code}:** {r.text[:500]}", "", "", ""

        data = r.json()
        intent = data["intent"]
        timing = data["timing"]
        results = data["results"]
        pagination = data["pagination"]

        # Intent class for coloring
        intent_cls = f"intent-{intent}"

        # Filter summary
        filter_html = _build_filter_summary(body)

        # Metadata summary
        total_ms = timing.get("total_ms", 0)
        algolia_ms = timing.get("algolia_ms", 0)
        semantic_ms = timing.get("semantic_ms", 0)

        meta_lines = [
            filter_html,
            f'**Intent:** <span class="{intent_cls}">{intent}</span> '
            f'| **Results:** {len(results)} '
            f'| **Page:** {pagination.get("page", 1)} '
            f'| **Has more:** {pagination["has_more"]}',
            f'**Timing:** total={_format_timing(total_ms)}, '
            f'algolia={_format_timing(algolia_ms)}, '
            f'semantic={_format_timing(semantic_ms)}',
            f'**Round-trip:** {elapsed*1000:.0f}ms'
            + (f' | **Session:** `{_session_id[:8]}...`' if use_session else ''),
        ]
        meta = "\n\n".join(meta_lines)

        if not results:
            return meta, "No results found.", "", ""

        # Build result cards
        cards = [_build_product_card(i, p, query) for i, p in enumerate(results)]
        results_html = "\n".join(cards)

        # Raw JSON (first 3 results)
        raw = json.dumps({
            "query": data["query"], "intent": intent, "timing": timing,
            "pagination": pagination, "request_body": body,
            "results_sample": results[:3],
        }, indent=2)

        return meta, results_html, raw, ""

    except requests.ConnectionError:
        return "**Connection error** - is the API server running on " + API_URL + "?", "", "", ""
    except Exception as e:
        return f"**Error:** {e}", "", "", ""


def do_compare(
    query_a, query_b, page_size_cmp, semantic_boost_cmp,
):
    """Run two queries side by side for comparison."""
    if not query_a or not query_a.strip():
        return "Enter Query A.", "", ""
    if not query_b or not query_b.strip():
        return "", "Enter Query B.", ""

    results_parts = []
    raw_parts = {}

    for label, query in [("A", query_a), ("B", query_b)]:
        body = {
            "query": query.strip(),
            "page_size": int(page_size_cmp),
            "semantic_boost": float(semantic_boost_cmp),
        }
        try:
            t = time.time()
            r = requests.post(f"{SEARCH_URL}/hybrid", json=body, headers=HEADERS, timeout=30)
            elapsed = time.time() - t

            if r.status_code != 200:
                results_parts.append(f"**Query {label} Error {r.status_code}:** {r.text[:300]}")
                continue

            data = r.json()
            intent = data["intent"]
            timing = data["timing"]
            results = data["results"]

            header = (
                f'<div style="background:#f0f4ff;padding:8px 12px;border-radius:8px;margin-bottom:8px;">'
                f'<b>Query {label}:</b> "{query}" | '
                f'<span class="intent-{intent}">{intent}</span> | '
                f'{len(results)} results | '
                f'{timing.get("total_ms",0)}ms total | '
                f'{elapsed*1000:.0f}ms round-trip'
                f'</div>'
            )

            cards = [_build_product_card(i, p) for i, p in enumerate(results)]
            results_parts.append(header + "\n".join(cards))
            raw_parts[f"query_{label}"] = {
                "query": query, "intent": intent, "timing": timing,
                "result_count": len(results),
                "top_3": [{"name": p.get("name"), "brand": p.get("brand"),
                          "rrf_score": p.get("rrf_score")} for p in results[:3]],
            }

        except Exception as e:
            results_parts.append(f"**Query {label} Error:** {e}")

    # Find overlap
    overlap_info = ""
    if len(raw_parts) == 2:
        try:
            r_a = requests.post(f"{SEARCH_URL}/hybrid", json={"query": query_a.strip(), "page_size": int(page_size_cmp)}, headers=HEADERS, timeout=30).json()
            r_b = requests.post(f"{SEARCH_URL}/hybrid", json={"query": query_b.strip(), "page_size": int(page_size_cmp)}, headers=HEADERS, timeout=30).json()
            ids_a = {p["product_id"] for p in r_a["results"]}
            ids_b = {p["product_id"] for p in r_b["results"]}
            overlap = ids_a & ids_b
            raw_parts["overlap"] = {
                "count": len(overlap),
                "percentage_of_a": f"{len(overlap)/max(len(ids_a),1)*100:.0f}%",
                "percentage_of_b": f"{len(overlap)/max(len(ids_b),1)*100:.0f}%",
                "product_ids": list(overlap)[:10],
            }
            overlap_info = f"\n\n**Overlap:** {len(overlap)} products in common ({len(overlap)/max(len(ids_a),1)*100:.0f}% of A, {len(overlap)/max(len(ids_b),1)*100:.0f}% of B)"
        except:
            pass

    result_a = results_parts[0] if len(results_parts) > 0 else ""
    result_b = results_parts[1] if len(results_parts) > 1 else ""
    raw = json.dumps(raw_parts, indent=2)

    return result_a, result_b, raw + overlap_info


def do_autocomplete(query):
    """Call the autocomplete API."""
    if not query or len(query) < 1:
        return "Type at least 1 character."
    try:
        t = time.time()
        r = requests.get(
            f"{SEARCH_URL}/autocomplete",
            params={"q": query, "limit": 10},
            headers=HEADERS, timeout=10,
        )
        elapsed = time.time() - t

        if r.status_code != 200:
            return f"**Error {r.status_code}:** {r.text[:300]}"

        data = r.json()
        lines = [f"**Query:** `{data['query']}` | **Time:** {elapsed*1000:.0f}ms\n"]

        if data["products"]:
            lines.append("### Products")
            for p in data["products"]:
                name = p.get("highlighted_name") or p.get("name", "")
                price = p.get("price", "?")
                img = p.get("image_url", "")
                lines.append(f"- **{p.get('brand', '')}** - {name} (${price})")

        if data["brands"]:
            lines.append("\n### Brands")
            for b in data["brands"]:
                highlighted = b.get("highlighted") or b.get("name", "")
                lines.append(f"- **{highlighted}**")

        if not data["products"] and not data["brands"]:
            lines.append("No suggestions found.")

        return "\n".join(lines)
    except requests.ConnectionError:
        return "**Connection error** - is the API server running?"
    except Exception as e:
        return f"**Error:** {e}"


def log_click(query, product_id, position):
    """Log a click event to analytics."""
    if not query or not product_id:
        return "Missing query or product_id"
    try:
        r = requests.post(
            f"{SEARCH_URL}/click",
            json={"query": query, "product_id": product_id, "position": int(position)},
            headers=HEADERS, timeout=5,
        )
        return f"Click logged: {r.status_code}"
    except Exception as e:
        return f"Error: {e}"


def check_health():
    """Check search API health."""
    try:
        r = requests.get(f"{SEARCH_URL}/health", timeout=5)
        data = r.json()
        status = data.get("status", "unknown")
        algolia = data.get("algolia", "unknown")
        records = data.get("index_records", "?")
        emoji = "OK" if status == "healthy" else "WARN"
        rec_str = f"{records:,}" if isinstance(records, int) else str(records)
        return f"**[{emoji}]** Status: {status} | Algolia: {algolia} | Index: {rec_str} records"
    except Exception as e:
        return f"**[DOWN]** {e}"


def run_quick_tests():
    """Run comprehensive preset test queries."""
    tests = [
        # (name, request_body, expected_status, expected_intent, description)
        ("Exact: brand only", {"query": "boohoo", "page_size": 5}, 200, "exact", "Single brand name"),
        ("Exact: brand variant", {"query": "Ba&sh", "page_size": 5}, 200, "exact", "Brand with special chars"),
        ("Specific: color+cat", {"query": "black midi dress", "page_size": 5}, 200, "specific", "Color + length + category"),
        ("Specific: fabric+cat", {"query": "silk blouse", "page_size": 5}, 200, "specific", "Fabric + category"),
        ("Specific: attribute", {"query": "oversized sweater", "page_size": 5}, 200, "specific", "Fit + category"),
        ("Vague: style vibe", {"query": "quiet luxury", "page_size": 5}, 200, "vague", "Style keyword"),
        ("Vague: occasion", {"query": "date night outfit", "page_size": 5}, 200, "vague", "Occasion-based"),
        ("Vague: aesthetic", {"query": "dark academia", "page_size": 5}, 200, "vague", "Aesthetic keyword"),
        ("Typo tolerance", {"query": "sweter", "page_size": 5}, 200, None, "Misspelled sweater"),
        ("Typo: florral", {"query": "florral dress", "page_size": 5}, 200, None, "Misspelled floral"),
        ("Brand filter", {"query": "dress", "brands": ["Boohoo"], "page_size": 5}, 200, None, "Brand inclusion filter"),
        ("Exclude brand", {"query": "dress", "exclude_brands": ["Boohoo"], "page_size": 5}, 200, None, "Brand exclusion filter"),
        ("Price range", {"query": "top", "min_price": 10, "max_price": 30, "page_size": 5}, 200, None, "Price bounds"),
        ("On sale only", {"query": "dress", "on_sale_only": True, "page_size": 5}, 200, None, "Sale filter"),
        ("Category L1", {"query": "black", "category_l1": ["Tops"], "page_size": 5}, 200, None, "L1 category filter"),
        ("Formality", {"query": "dress", "formality": ["Casual"], "page_size": 5}, 200, None, "Formality filter"),
        ("Pattern", {"query": "dress", "patterns": ["Floral"], "page_size": 5}, 200, None, "Pattern filter"),
        ("Multi-filter", {"query": "top", "category_l1": ["Tops"], "colors": ["Black"], "formality": ["Casual"], "page_size": 5}, 200, None, "3 filters combined"),
        ("Neckline filter", {"query": "dress", "neckline": ["V-Neck"], "page_size": 5}, 200, None, "Neckline filter"),
        ("Sleeve filter", {"query": "top", "sleeve_type": ["Long"], "page_size": 5}, 200, None, "Sleeve filter"),
        ("Silhouette", {"query": "dress", "silhouette": ["A-Line"], "page_size": 5}, 200, None, "Silhouette filter"),
        ("Season", {"query": "dress", "seasons": ["Summer"], "page_size": 5}, 200, None, "Season filter"),
        ("Semantic high", {"query": "quiet luxury", "page_size": 5, "semantic_boost": 0.9}, 200, None, "High semantic weight"),
        ("Semantic low", {"query": "quiet luxury", "page_size": 5, "semantic_boost": 0.1}, 200, None, "Low semantic weight"),
        ("Empty query", {"query": "", "page_size": 5}, 422, None, "Should reject empty"),
        ("Invalid price", {"query": "dress", "min_price": 100, "max_price": 50}, 422, None, "min > max should fail"),
        ("Pagination p2", {"query": "dress", "page": 2, "page_size": 5}, 200, None, "Page 2 results"),
        ("Large page", {"query": "top", "page_size": 100}, 200, None, "Max page size"),
    ]

    rows = []
    for name, body, expected_status, expected_intent, desc in tests:
        try:
            t = time.time()
            r = requests.post(f"{SEARCH_URL}/hybrid", json=body, headers=HEADERS, timeout=30)
            elapsed = (time.time() - t) * 1000

            status_ok = r.status_code == expected_status
            status_mark = "PASS" if status_ok else "FAIL"

            if r.status_code == 200:
                d = r.json()
                n = len(d["results"])
                intent = d["intent"]
                intent_ok = expected_intent is None or intent == expected_intent
                intent_mark = "PASS" if intent_ok else "FAIL"
                total_ms = d["timing"].get("total_ms", 0)
                rows.append(
                    f"| {status_mark} | {name} | `{body.get('query', '')[:20]}` | "
                    f"{r.status_code} | {intent} ({intent_mark}) | {n} | "
                    f"{total_ms}ms | {elapsed:.0f}ms | {desc} |"
                )
            elif r.status_code == 422:
                rows.append(
                    f"| {status_mark} | {name} | `{body.get('query', '')[:20]}` | "
                    f"422 | -- | -- | -- | {elapsed:.0f}ms | {desc} |"
                )
            else:
                rows.append(
                    f"| FAIL | {name} | `{body.get('query', '')[:20]}` | "
                    f"{r.status_code} | ERROR | -- | -- | {elapsed:.0f}ms | {r.text[:50]} |"
                )
        except Exception as e:
            rows.append(
                f"| ERR | {name} | `{body.get('query', '')[:20]}` | "
                f"-- | -- | -- | -- | -- | {e} |"
            )

    header = (
        "| Result | Test | Query | Status | Intent | Count | API Time | RTT | Notes |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )

    passed = sum(1 for r in rows if r.startswith("| PASS"))
    total = len(rows)
    summary = f"\n\n**Results: {passed}/{total} passed**"

    return header + "\n".join(rows) + summary


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Search Test UI") as app:
    gr.Markdown("# Hybrid Search Test UI")
    gr.Markdown("Algolia + FashionCLIP hybrid search -- all 23 filters, gallery images, session dedup, comparison mode")

    # Health check bar
    with gr.Row():
        health_btn = gr.Button("Check API Health", size="sm", variant="secondary")
        session_display = gr.Markdown(f"**Session:** `{_session_id[:8]}...`")
        new_session_btn = gr.Button("New Session", size="sm", variant="secondary")
        health_out = gr.Markdown("")

    health_btn.click(check_health, outputs=health_out)
    new_session_btn.click(lambda: f"**Session:** `{_new_session()[:8]}...`", outputs=session_display)

    with gr.Tabs():
        # ==================================================================
        # Tab 1: Hybrid Search (full filters)
        # ==================================================================
        with gr.TabItem("Hybrid Search"):
            with gr.Row():
                # Left column: Query + all filters
                with gr.Column(scale=1):
                    query = gr.Textbox(
                        label="Search Query",
                        placeholder="e.g. black midi dress, quiet luxury, boohoo, silk blouse...",
                        lines=1,
                    )
                    with gr.Row():
                        search_btn = gr.Button("Search", variant="primary", size="lg")
                        page_num = gr.Number(label="Page", value=1, minimum=1, maximum=100, step=1, scale=1)

                    # --- Category Filters ---
                    with gr.Accordion("Category Filters", open=True):
                        with gr.Row():
                            f_categories = gr.Dropdown(
                                BROAD_CATEGORIES, label="Broad Category", multiselect=True,
                                info="tops, bottoms, dresses, outerwear",
                            )
                            f_category_l1 = gr.Dropdown(
                                CATEGORIES_L1, label="Category L1", multiselect=True,
                                info="Gemini vision categories",
                            )
                        with gr.Row():
                            f_category_l2 = gr.Dropdown(
                                CATEGORIES_L2, label="Category L2", multiselect=True,
                                info="Specific type: Blouse, Jeans, Midi Dress...",
                            )
                            f_article_type = gr.Dropdown(
                                ARTICLE_TYPES, label="Article Type", multiselect=True,
                                allow_custom_value=True,
                                info="jeans, t-shirt, midi dress...",
                            )

                    # --- Brand Filters ---
                    with gr.Accordion("Brand Filters", open=False):
                        with gr.Row():
                            f_brands = gr.Dropdown(
                                BRANDS_POPULAR, label="Include Brands", multiselect=True,
                                allow_custom_value=True,
                            )
                            f_exclude_brands = gr.Dropdown(
                                BRANDS_POPULAR, label="Exclude Brands", multiselect=True,
                                allow_custom_value=True,
                            )

                    # --- Color & Pattern ---
                    with gr.Accordion("Color & Pattern", open=False):
                        with gr.Row():
                            f_colors = gr.Dropdown(COLORS, label="Colors", multiselect=True)
                            f_color_family = gr.Dropdown(COLOR_FAMILIES, label="Color Family", multiselect=True)
                        with gr.Row():
                            f_patterns = gr.Dropdown(PATTERNS, label="Patterns", multiselect=True)
                            f_materials = gr.Dropdown(FABRICS, label="Materials", multiselect=True)

                    # --- Style & Occasion ---
                    with gr.Accordion("Style & Occasion", open=False):
                        with gr.Row():
                            f_style_tags = gr.Dropdown(STYLE_TAGS, label="Style Tags", multiselect=True)
                            f_formality = gr.Dropdown(FORMALITY, label="Formality", multiselect=True)
                        with gr.Row():
                            f_occasions = gr.Dropdown(OCCASIONS, label="Occasions", multiselect=True)
                            f_seasons = gr.Dropdown(SEASONS, label="Seasons", multiselect=True)

                    # --- Construction ---
                    with gr.Accordion("Construction Details", open=False):
                        with gr.Row():
                            f_fit_type = gr.Dropdown(FIT_TYPES, label="Fit Type", multiselect=True)
                            f_neckline = gr.Dropdown(NECKLINES, label="Neckline", multiselect=True)
                        with gr.Row():
                            f_sleeve_type = gr.Dropdown(SLEEVES, label="Sleeve Type", multiselect=True)
                            f_length = gr.Dropdown(LENGTHS, label="Length", multiselect=True)
                        with gr.Row():
                            f_silhouette = gr.Dropdown(SILHOUETTES, label="Silhouette", multiselect=True)
                            f_rise = gr.Dropdown(RISES, label="Rise", multiselect=True)

                    # --- Price & Options ---
                    with gr.Accordion("Price & Options", open=False):
                        with gr.Row():
                            f_min_price = gr.Number(label="Min Price", value=0, minimum=0)
                            f_max_price = gr.Number(label="Max Price", value=0, minimum=0)
                        with gr.Row():
                            f_on_sale_only = gr.Checkbox(label="On Sale Only", value=False)
                            f_use_session = gr.Checkbox(label="Session Dedup", value=False, info="Avoid repeating products across searches")
                        with gr.Row():
                            f_page_size = gr.Slider(1, 100, value=20, step=1, label="Results Per Page")
                            f_semantic_boost = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Semantic Boost", info="0=Algolia only, 1=Semantic only")

                # Right column: Results
                with gr.Column(scale=2):
                    meta_out = gr.Markdown(label="Search Metadata")
                    results_out = gr.HTML(label="Results")
                    with gr.Accordion("Raw JSON (first 3 results)", open=False):
                        raw_out = gr.Code(language="json", label="Raw Response")
                    error_out = gr.Markdown(visible=False)

            all_inputs = [
                query, page_num,
                f_categories, f_category_l1, f_category_l2, f_article_type,
                f_brands, f_exclude_brands,
                f_colors, f_color_family,
                f_patterns, f_materials,
                f_style_tags, f_formality, f_occasions, f_seasons,
                f_fit_type, f_neckline, f_sleeve_type, f_length, f_silhouette, f_rise,
                f_min_price, f_max_price, f_on_sale_only, f_page_size, f_semantic_boost,
                f_use_session,
            ]
            all_outputs = [meta_out, results_out, raw_out, error_out]

            search_btn.click(do_search, inputs=all_inputs, outputs=all_outputs)
            query.submit(do_search, inputs=all_inputs, outputs=all_outputs)

        # ==================================================================
        # Tab 2: Side-by-Side Comparison
        # ==================================================================
        with gr.TabItem("Compare Queries"):
            gr.Markdown("### Compare two queries side by side to evaluate relevance")
            with gr.Row():
                cmp_query_a = gr.Textbox(label="Query A", placeholder="e.g. black dress", lines=1)
                cmp_query_b = gr.Textbox(label="Query B", placeholder="e.g. dark evening gown", lines=1)
            with gr.Row():
                cmp_page_size = gr.Slider(5, 50, value=10, step=1, label="Results Per Query")
                cmp_semantic = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Semantic Boost")
            cmp_btn = gr.Button("Compare", variant="primary", size="lg")

            with gr.Row():
                cmp_out_a = gr.HTML(label="Query A Results")
                cmp_out_b = gr.HTML(label="Query B Results")
            with gr.Accordion("Comparison Data", open=False):
                cmp_raw = gr.Code(language="json", label="Raw Comparison")

            cmp_btn.click(
                do_compare,
                inputs=[cmp_query_a, cmp_query_b, cmp_page_size, cmp_semantic],
                outputs=[cmp_out_a, cmp_out_b, cmp_raw],
            )

        # ==================================================================
        # Tab 3: Autocomplete
        # ==================================================================
        with gr.TabItem("Autocomplete"):
            ac_query = gr.Textbox(label="Type to autocomplete", placeholder="boo, dre, silk...", lines=1)
            ac_btn = gr.Button("Get Suggestions", variant="primary")
            ac_out = gr.Markdown("")
            ac_btn.click(do_autocomplete, inputs=ac_query, outputs=ac_out)
            ac_query.submit(do_autocomplete, inputs=ac_query, outputs=ac_out)

        # ==================================================================
        # Tab 4: Analytics
        # ==================================================================
        with gr.TabItem("Click Analytics"):
            gr.Markdown("### Log search click events for analytics")
            with gr.Row():
                click_query = gr.Textbox(label="Search Query", placeholder="Original search query")
                click_product = gr.Textbox(label="Product ID", placeholder="Product ID that was clicked")
                click_position = gr.Number(label="Position", value=1, minimum=1)
            click_btn = gr.Button("Log Click", variant="primary")
            click_out = gr.Markdown("")
            click_btn.click(log_click, inputs=[click_query, click_product, click_position], outputs=click_out)

        # ==================================================================
        # Tab 5: Quick Tests
        # ==================================================================
        with gr.TabItem("Quick Tests"):
            gr.Markdown(
                "### Run 28 preset test queries\n"
                "Tests cover: exact/specific/vague intent, typo tolerance, all filter types, "
                "semantic boost tuning, pagination, error handling"
            )
            test_out = gr.Markdown("")
            test_btn = gr.Button("Run All Quick Tests", variant="primary", size="lg")
            test_btn.click(run_quick_tests, outputs=test_out)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Connecting to API at {API_URL}")
    print(f"Session ID: {_session_id}")
    print(f"Health: {check_health()}")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft(), css=CUSTOM_CSS)
