"""
Test queries that FAILED during manual review — focused regression test.

These are the specific queries (from the 254-query taxonomy review) where
the planner + exclusion pipeline produced bad results: wrong categories,
missing exclusions, price not applied, 0-result over-aggression, etc.

Usage:
    PYTHONPATH=src python scripts/test_exclusion_filters_html.py
    # Opens: scripts/exclusion_filter_results.html
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

# Each entry: (query, category, exclude_check)
# exclude_check is a dict of {result_field: set_of_bad_values}
# Any product matching a bad value is flagged as a VIOLATION.
#
# Special keys in exclude_check:
#   _name_contains: {substr, ...}    — flag if product name contains substr
#   _category_l1_not: {val, ...}     — flag if category_l1 is NOT in the set
#   _max_price: float                — flag if price > value
#   _min_results: int                — flag query-level if total results < N
#   _require_on_sale: True           — flag if product is NOT on sale

_QUERY_DATA = [
    # =================================================================
    # THEME 1: Sleeve/Coverage Exclusions Not Enforced (~6 queries)
    # Expected: no sleeveless/short-sleeve results when user wants coverage
    # =================================================================
    (
        "Top with longer sleeves",
        "T1: Sleeve Exclusion",
        {"sleeve_type": {"sleeveless", "spaghetti", "strapless"},
         "_category_l1_not": {"tops", "knitwear", "outerwear"}},
    ),
    (
        "Midi dress with sleeves",
        "T1: Sleeve Exclusion",
        {"sleeve_type": {"sleeveless", "spaghetti", "strapless"}},
    ),
    (
        "Long sleeves but lightweight",
        "T1: Sleeve Exclusion",
        {"sleeve_type": {"sleeveless", "short", "spaghetti", "strapless", "cap"}},
    ),
    (
        "Help me find a top that hides my arms",
        "T1: Sleeve / Arms",
        {"sleeve_type": {"sleeveless", "cap", "spaghetti", "strapless"},
         "neckline": {"off-shoulder", "one shoulder", "strapless"}},
    ),
    (
        "I need a dress that's flattering but I don't want to show my arms",
        "T1: Sleeve / Arms + Dress",
        {"sleeve_type": {"sleeveless", "cap", "spaghetti", "strapless"},
         "neckline": {"off-shoulder", "one shoulder", "strapless"}},
    ),
    (
        "Top that doesn't show bra straps",
        "T1: Bra Strap Coverage",
        {"neckline": {"strapless", "off-shoulder", "halter", "one shoulder"},
         "sleeve_type": {"sleeveless", "spaghetti"},
         "_name_contains": {"backless", "open back", "open-back"}},
    ),

    # =================================================================
    # THEME 2: Category Pollution (~8 queries)
    # Expected: correct product type, no random bottoms for top queries etc.
    # =================================================================
    (
        "Covers shoulders",
        "T2: Category Pollution",
        {"_category_l1_not": {"tops", "knitwear", "outerwear", "dresses"},
         "sleeve_type": {"sleeveless", "spaghetti", "strapless"}},
    ),
    (
        "No backless",
        "T2: Category Pollution",
        {"_name_contains": {"backless", "open back", "open-back"}},
    ),
    (
        "Not see through",
        "T2: Category Pollution",
        {},  # mainly checking we don't get random denim shorts/jeans
    ),
    (
        "Outfit to hide belly",
        "T2: Category Pollution",
        {"_name_contains": {"bikini", "swimsuit", "swimwear"}},
    ),
    (
        "Trousers with pleats",
        "T2: Category Pollution",
        {"_category_l1_not": {"bottoms", "trousers", "pants"}},
    ),

    # =================================================================
    # THEME 3: Price/Deal Filters Not Connected (~5 queries)
    # Expected: price constraints and on_sale_only actually applied
    # =================================================================
    (
        "Under $50 date night dress",
        "T3: Price Filter",
        {"_max_price": 50.0},
    ),
    (
        "Dress under $30",
        "T3: Price Filter",
        {"_max_price": 30.0},
    ),
    (
        "Affordable summer tops under 40",
        "T3: Price Filter",
        {"_max_price": 40.0},
    ),
    (
        "On sale coats",
        "T3: Sale Filter",
        {"_require_on_sale": True},
    ),
    (
        "Discounted dresses on sale",
        "T3: Sale Filter",
        {"_require_on_sale": True},
    ),

    # =================================================================
    # THEME 4: Vocabulary Translation (~5 queries)
    # Expected: planner maps user language to catalog terms
    # =================================================================
    (
        "Skirt with shorts underneath",
        "T4: Vocabulary",
        {"_name_should_contain": {"skort", "skirt"}},  # positive — at least one should be skort/skirt
    ),
    (
        "Butter yellow top",
        "T4: Color Vocab",
        {},  # check visually — should be yellow/gold items, not random colors
    ),
    (
        "Top that isn't clingy",
        "T4: Fit Vocab",
        {"fit_type": {"slim", "fitted", "bodycon"},
         "silhouette": {"bodycon"}},
    ),

    # =================================================================
    # THEME 5: Back/Body Coverage
    # =================================================================
    (
        "Outfits that cover my back",
        "T5: Back Coverage",
        {"_name_contains": {"backless", "open back", "open-back", "low back"}},
    ),

    # =================================================================
    # THEME 6: Non-filterable Attributes (semantic search dependent)
    # =================================================================
    (
        "No slit",
        "T6: Non-filterable",
        {"_name_contains": {"slit"}},  # best we can do — check product names
    ),

    # =================================================================
    # THEME 7: Over-Aggressive Exclusions -> 0 Results
    # Expected: progressive relaxation should kick in and return SOMETHING
    # =================================================================
    (
        "I'm going to a wedding and want something elegant not too revealing and not super expensive",
        "T7: 0-Result Recovery",
        {"_min_results": 1},  # the ONLY check: we must get results back
    ),
    (
        "Modest formal dress that isn't boring, under $100, with sleeves",
        "T7: 0-Result Recovery",
        {"_min_results": 1},
    ),

    # =================================================================
    # THEME 8: Degree/Nuance (aspirational — no strict check)
    # =================================================================
    (
        "Cropped top but not too cropped",
        "T8: Nuance",
        {},  # purely visual check — can't define strict violations for "degree"
    ),

    # =================================================================
    # BONUS: Modesty queries (regression — were passing before)
    # =================================================================
    (
        "Modest dress for a wedding",
        "Regression: Modesty",
        {"neckline": {"strapless", "off-shoulder", "halter", "sweetheart", "one shoulder", "v-neck", "deep v", "plunging"},
         "sleeve_type": {"sleeveless"},
         "length": {"mini", "cropped"}},
    ),
    (
        "Not too revealing date night outfit",
        "Regression: Modesty",
        {"neckline": {"strapless", "off-shoulder", "halter", "sweetheart", "deep v-neck", "plunging"},
         "_name_contains": {"backless", "open back", "open-back"}},
    ),
]

TOP_N = 10

# Result fields to show as tags in the product card
_TAG_FIELDS = [
    ("category_l1", "category_l1"),
    ("category_l2", "category_l2"),
    ("neckline", "neckline"),
    ("sleeve_type", "sleeve_type"),
    ("length", "length"),
    ("fit_type", "fit_type"),
    ("silhouette", "silhouette"),
    ("pattern", "pattern"),
    ("apparent_fabric", "apparent_fabric"),
    ("primary_color", "primary_color"),
    ("is_on_sale", "is_on_sale"),
]


def _check_violation(product, exclude_check):
    """Check if a product violates any exclusion rule. Return list of violated fields.

    Special keys:
      _name_contains: flag if product name DOES contain any substring (negative)
      _name_should_contain: flag if product name does NOT contain ANY substring (positive)
      _category_l1_not: flag if category_l1 is NOT in the expected set
      _max_price: flag if product price exceeds the value
      _require_on_sale: flag if product is not on sale
      _min_results: query-level check, not per-product (handled in run_tests)
    """
    violations = []
    for field, bad_vals in exclude_check.items():
        if field == "_name_contains":
            # Negative: flag if product name contains any of these substrings
            name = (getattr(product, "name", None) or "").lower()
            for substr in bad_vals:
                if substr in name:
                    violations.append(("name", f"contains '{substr}'"))
                    break
            continue
        if field == "_name_should_contain":
            # Positive: flag if product name does NOT contain ANY of these
            name = (getattr(product, "name", None) or "").lower()
            if not any(substr in name for substr in bad_vals):
                violations.append(("name", f"missing any of {bad_vals}"))
            continue
        if field == "_category_l1_not":
            # Flag if category_l1 is NOT in the expected set (wrong category)
            cat = (getattr(product, "category_l1", None) or "").lower()
            if cat and cat not in bad_vals:
                violations.append(("category_l1", f"'{cat}' not in {bad_vals}"))
            continue
        if field == "_max_price":
            # Flag if price exceeds the max
            price = getattr(product, "price", None)
            if price is not None and price > bad_vals:
                violations.append(("price", f"${price:.2f} > ${bad_vals:.2f}"))
            continue
        if field == "_require_on_sale":
            # Flag if product is not on sale
            if not getattr(product, "is_on_sale", False):
                violations.append(("is_on_sale", "not on sale"))
            continue
        if field == "_min_results":
            # Query-level, skip per-product
            continue
        val = getattr(product, field, None) or ""
        if val.lower() in bad_vals:
            violations.append((field, val))
    return violations


def run_tests():
    print("Initializing hybrid search service...")
    service = HybridSearchService()
    print("Service ready.\n")

    all_results = []
    total_start = time.time()

    for idx, (query, category, exclude_check) in enumerate(_QUERY_DATA):
        i = idx + 1
        print(f"[{i:2d}/{len(_QUERY_DATA)}] Searching: \"{query}\"...", end=" ", flush=True)

        request = HybridSearchRequest(
            query=query,
            page=1,
            page_size=30,
            sort_by=SortBy.RELEVANCE,
        )

        t_start = time.time()
        try:
            response = service.search(request)
            elapsed_ms = int((time.time() - t_start) * 1000)

            products = []
            total_violations = 0
            for p in response.results[:TOP_N]:
                viols = _check_violation(p, exclude_check)
                if viols:
                    total_violations += 1
                products.append({
                    "name": p.name,
                    "brand": p.brand,
                    "price": p.price,
                    "original_price": p.original_price,
                    "is_on_sale": p.is_on_sale,
                    "image_url": p.image_url,
                    "neckline": p.neckline,
                    "sleeve_type": p.sleeve_type,
                    "length": p.length,
                    "fit_type": p.fit_type,
                    "silhouette": p.silhouette,
                    "pattern": p.pattern,
                    "apparent_fabric": p.apparent_fabric,
                    "primary_color": p.primary_color,
                    "category_l1": p.category_l1,
                    "category_l2": p.category_l2,
                    "formality": p.formality,
                    "algolia_rank": p.algolia_rank,
                    "semantic_rank": p.semantic_rank,
                    "semantic_score": p.semantic_score,
                    "rrf_score": p.rrf_score,
                    "violations": viols,
                })

            # Query-level check: _min_results
            min_results = exclude_check.get("_min_results")
            total_count = response.pagination.total_results or len(response.results)
            if min_results and total_count < min_results:
                total_violations += 1  # count the whole query as a violation

            algolia_count = sum(1 for r in response.results if r.algolia_rank)
            semantic_count = sum(1 for r in response.results if r.semantic_rank)

            # Relaxation level from progressive relaxation
            relaxation = response.timing.get("relaxation_level")

            # Serialize exclude_check for HTML (handle non-set values)
            excl_serialized = {}
            for k, v in exclude_check.items():
                if isinstance(v, set):
                    excl_serialized[k] = sorted(v)
                elif isinstance(v, (list, tuple)):
                    excl_serialized[k] = list(v)
                else:
                    excl_serialized[k] = v

            all_results.append({
                "query": query,
                "category": category,
                "exclude_check": excl_serialized,
                "intent": response.intent,
                "total": total_count,
                "timing": response.timing,
                "elapsed_ms": elapsed_ms,
                "products": products,
                "algolia_count": algolia_count,
                "semantic_count": semantic_count,
                "total_violations": total_violations,
                "relaxation_level": relaxation,
                "success": True,
            })
            relax_str = f" [relaxed: {relaxation}]" if relaxation else ""
            v_str = f" ({total_violations} violations!)" if total_violations else ""
            print(f"{len(products)} results ({elapsed_ms}ms){relax_str}{v_str}")

        except Exception as e:
            elapsed_ms = int((time.time() - t_start) * 1000)
            all_results.append({
                "query": query,
                "category": category,
                "exclude_check": {k: (sorted(v) if isinstance(v, set) else v) for k, v in exclude_check.items()},
                "intent": "error",
                "total": 0,
                "timing": {},
                "elapsed_ms": elapsed_ms,
                "products": [],
                "algolia_count": 0,
                "semantic_count": 0,
                "total_violations": 0,
                "relaxation_level": None,
                "success": False,
                "error": str(e),
            })
            print(f"ERROR ({elapsed_ms}ms): {e}")

    total_elapsed = int(time.time() - total_start)
    generate_html(all_results, total_elapsed)


def generate_html(results, total_elapsed):
    succeeded = sum(1 for r in results if r["success"])
    with_results = sum(1 for r in results if r["total"] > 0)
    avg_time = sum(r["elapsed_ms"] for r in results) // max(len(results), 1)
    total_violations = sum(r["total_violations"] for r in results)
    total_checked = sum(len(r["products"]) for r in results if r["exclude_check"])
    queries_with_checks = sum(1 for r in results if r["exclude_check"])
    clean_queries = sum(1 for r in results if r["exclude_check"] and r["total_violations"] == 0)

    # Build query cards
    cards_html = ""
    for i, r in enumerate(results):
        q = html.escape(r["query"])
        intent_class = r["intent"]
        intent_color = {
            "exact": "#3b82f6", "specific": "#10b981",
            "vague": "#8b5cf6", "error": "#ef4444",
        }.get(intent_class, "#6b7280")

        # Timing pills
        timing = r.get("timing", {})
        timing_pills = ""
        for key in ["planner_ms", "algolia_ms", "semantic_ms", "total_ms"]:
            if key in timing:
                label = key.replace("_ms", "").replace("_", " ").title()
                timing_pills += f'<span class="timing-pill">{label}: {timing[key]}ms</span>'

        # Relaxation badge
        relaxation = r.get("relaxation_level")
        if relaxation:
            timing_pills += f'<span class="timing-pill" style="color:var(--orange);border-color:var(--orange)">Relaxed: {relaxation}</span>'

        # Exclusion rules display — handle special keys
        excl = r.get("exclude_check", {})
        excl_html = ""
        if excl:
            pills = []
            for field, vals in excl.items():
                if field == "_name_contains":
                    for v in vals:
                        pills.append(f'<span class="excl-pill">name NOT contains "{v}"</span>')
                elif field == "_name_should_contain":
                    expected = " or ".join(f'"{v}"' for v in vals)
                    pills.append(f'<span class="excl-pill" style="color:var(--green);border-color:rgba(16,185,129,0.3);background:rgba(16,185,129,0.1)">name SHOULD contain {expected}</span>')
                elif field == "_category_l1_not":
                    expected = ", ".join(sorted(vals)) if isinstance(vals, (list, set)) else str(vals)
                    pills.append(f'<span class="excl-pill" style="color:var(--orange);border-color:rgba(245,158,11,0.3);background:rgba(245,158,11,0.1)">category_l1 MUST be: {expected}</span>')
                elif field == "_max_price":
                    pills.append(f'<span class="excl-pill" style="color:var(--orange);border-color:rgba(245,158,11,0.3);background:rgba(245,158,11,0.1)">price &le; ${vals}</span>')
                elif field == "_require_on_sale":
                    pills.append(f'<span class="excl-pill" style="color:var(--orange);border-color:rgba(245,158,11,0.3);background:rgba(245,158,11,0.1)">must be on sale</span>')
                elif field == "_min_results":
                    pills.append(f'<span class="excl-pill" style="color:var(--blue);border-color:rgba(59,130,246,0.3);background:rgba(59,130,246,0.1)">min {vals} results</span>')
                elif isinstance(vals, (list, set)):
                    for v in vals:
                        pills.append(f'<span class="excl-pill">NOT {field}:{v}</span>')
                else:
                    pills.append(f'<span class="excl-pill">NOT {field}:{vals}</span>')
            excl_html = f'<div class="excl-row">Checks: {" ".join(pills)}</div>'
        else:
            excl_html = '<div class="excl-row" style="color:var(--text2)">No strict checks — visual review only</div>'

        # Violation summary
        v_count = r["total_violations"]
        p_count = len(r["products"])
        min_req = excl.get("_min_results")
        if min_req and r["total"] < min_req:
            viol_badge = f'<span class="viol-badge fail">0 RESULTS (need {min_req}+)</span>'
        elif excl:
            if v_count == 0:
                viol_badge = f'<span class="viol-badge pass">{v_count}/{p_count} violations</span>'
            else:
                viol_badge = f'<span class="viol-badge fail">{v_count}/{p_count} violations</span>'
        else:
            viol_badge = '<span class="viol-badge na">visual only</span>'

        # Source badge
        src_text = f'{r["algolia_count"]} Keyword / {r["semantic_count"]} Semantic'

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
                viols = p.get("violations", [])

                # Source indicator
                has_alg = p.get("algolia_rank") is not None
                has_sem = p.get("semantic_rank") is not None
                if has_alg and has_sem:
                    source_badge = '<span class="source-badge both">Both</span>'
                elif has_alg:
                    source_badge = '<span class="source-badge algolia">Keyword</span>'
                else:
                    source_badge = '<span class="source-badge semantic">Semantic</span>'

                # Price display
                if on_sale and orig and orig > price:
                    price_html = f'<span class="price-sale">${price:.2f}</span> <span class="price-original">${orig:.2f}</span>'
                else:
                    price_html = f'<span class="price">${price:.2f}</span>'

                # Build attribute tags — highlight violations in red
                tags_html = ""
                violated_fields = {f for f, _ in viols}
                for tag_field, attr_name in _TAG_FIELDS:
                    val = p.get(tag_field) or ""
                    if not val:
                        continue
                    is_violated = tag_field in violated_fields
                    cls = "tag violation-tag" if is_violated else "tag"
                    label = html.escape(f"{tag_field}: {val}")
                    tags_html += f'<span class="{cls}">{label}</span>'

                # Score
                rrf = p.get("rrf_score")
                score_html = f'<span class="score">RRF: {rrf:.4f}</span>' if rrf else ""

                # Card border
                card_cls = "product-card violation-card" if viols else "product-card"

                # Violation overlay
                viol_overlay = ""
                if viols:
                    viol_labels = ", ".join(f"{f}={v}" for f, v in viols)
                    viol_overlay = f'<div class="violation-overlay">VIOLATION: {html.escape(viol_labels)}</div>'

                products_html += f'''
                <div class="{card_cls}">
                    <div class="product-image-container">
                        <img src="{img}" alt="{name}" class="product-image" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 200 260%22><rect fill=%22%23f3f4f6%22 width=%22200%22 height=%22260%22/><text x=%2250%%22 y=%2250%%22 text-anchor=%22middle%22 fill=%22%239ca3af%22 font-size=%2214%22>No Image</text></svg>'"/>
                        {source_badge}
                        {viol_overlay}
                    </div>
                    <div class="product-info">
                        <p class="product-brand">{brand}</p>
                        <p class="product-name" title="{name}">{name}</p>
                        <div class="product-meta">
                            {price_html}
                        </div>
                        <div class="product-tags">
                            {tags_html}
                        </div>
                        {score_html}
                    </div>
                </div>'''
        else:
            products_html = '<div class="no-results">No results found</div>'

        # Plan details display (modes, attributes, avoid, queries)
        plan_html = ""
        plan_modes = timing.get("plan_modes", [])
        plan_attrs = timing.get("plan_attributes", {})
        plan_avoid = timing.get("plan_avoid", {})
        plan_alg_q = timing.get("plan_algolia_query", "")
        plan_sem_q = timing.get("plan_semantic_query", "")
        plan_applied = timing.get("plan_applied_filters", {})
        plan_excl = timing.get("plan_exclusions", {})

        if plan_modes or plan_attrs or plan_avoid or plan_alg_q:
            plan_parts = []
            if plan_modes:
                mode_pills = " ".join(
                    f'<span class="mode-pill">{html.escape(m)}</span>'
                    for m in plan_modes
                )
                plan_parts.append(f'<div class="plan-line"><span class="plan-label">Modes:</span> {mode_pills}</div>')
            if plan_attrs:
                attr_pills = " ".join(
                    f'<span class="attr-pill">{html.escape(k)}: {html.escape(", ".join(v) if isinstance(v, list) else str(v))}</span>'
                    for k, v in plan_attrs.items()
                )
                plan_parts.append(f'<div class="plan-line"><span class="plan-label">Attributes:</span> {attr_pills}</div>')
            if plan_avoid:
                avoid_pills = " ".join(
                    f'<span class="avoid-pill">{html.escape(k)}: {html.escape(", ".join(v) if isinstance(v, list) else str(v))}</span>'
                    for k, v in plan_avoid.items()
                )
                plan_parts.append(f'<div class="plan-line"><span class="plan-label">Avoid:</span> {avoid_pills}</div>')
            if plan_alg_q:
                plan_parts.append(f'<div class="plan-line"><span class="plan-label">Algolia Query:</span> <code>{html.escape(plan_alg_q)}</code></div>')
            if plan_sem_q:
                sem_display = plan_sem_q[:120] + ("..." if len(plan_sem_q) > 120 else "")
                plan_parts.append(f'<div class="plan-line"><span class="plan-label">Semantic Query:</span> <code>{html.escape(sem_display)}</code></div>')
            if plan_excl:
                excl_pills = " ".join(
                    f'<span class="excl-pill">{html.escape(k)}: {html.escape(", ".join(v) if isinstance(v, list) else str(v))}</span>'
                    for k, v in plan_excl.items()
                )
                plan_parts.append(f'<div class="plan-line"><span class="plan-label">Exclusions:</span> {excl_pills}</div>')
            plan_html = f'<div class="plan-details">{"".join(plan_parts)}</div>'

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
                    {viol_badge}
                    <span class="result-count">{r["total"]} results</span>
                    <span class="time-badge">{r["elapsed_ms"]}ms</span>
                </div>
            </div>
            {excl_html}
            <div class="timing-row">{timing_pills}</div>
            {plan_html}
            <div class="source-row">Sources: {src_text}</div>
            <div class="products-grid">{products_html}</div>
        </div>'''

    # Summary rows
    summary_rows = ""
    for i, r in enumerate(results):
        q = html.escape(r["query"])
        intent = r["intent"]
        color = {
            "exact": "#3b82f6", "specific": "#10b981",
            "vague": "#8b5cf6",
        }.get(intent, "#6b7280")
        count_class = "zero" if r["total"] == 0 else ("low" if r["total"] < 5 else "")
        v = r["total_violations"]
        has_check = bool(r.get("exclude_check"))
        if has_check:
            v_cls = "row-viol pass" if v == 0 else "row-viol fail"
            v_text = f"{v}"
        else:
            v_cls = "row-viol na"
            v_text = "-"
        summary_rows += f'''
        <tr onclick="document.getElementById('query-{i+1}').scrollIntoView({{behavior:'smooth'}})">
            <td class="row-num">{i+1}</td>
            <td class="row-query">{q}</td>
            <td><span class="intent-badge-sm" style="background:{color}">{intent}</span></td>
            <td class="{v_cls}">{v_text}</td>
            <td class="row-count {count_class}">{r["total"]}</td>
            <td class="row-time">{r["elapsed_ms"]}ms</td>
        </tr>'''

    page_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Failed Query Regression Test</title>
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
    background: linear-gradient(135deg, var(--red), var(--orange));
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
.stat-value.good {{ color: var(--green); }}
.stat-value.bad {{ color: var(--red); }}
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
.row-viol {{ text-align: center; font-weight: 700; }}
.row-viol.pass {{ color: var(--green); }}
.row-viol.fail {{ color: var(--red); }}
.row-viol.na {{ color: var(--text2); }}
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
.viol-badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
}}
.viol-badge.pass {{
    background: rgba(16,185,129,0.15);
    color: var(--green);
    border: 1px solid rgba(16,185,129,0.3);
}}
.viol-badge.fail {{
    background: rgba(239,68,68,0.15);
    color: var(--red);
    border: 1px solid rgba(239,68,68,0.3);
}}
.viol-badge.na {{
    background: var(--surface2);
    color: var(--text2);
}}
.excl-row {{
    display: flex;
    gap: 6px;
    margin-bottom: 8px;
    flex-wrap: wrap;
    font-size: 12px;
    color: var(--text2);
    align-items: center;
}}
.excl-pill {{
    font-size: 11px;
    font-weight: 600;
    color: var(--red);
    background: rgba(239,68,68,0.1);
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid rgba(239,68,68,0.2);
    font-family: 'SF Mono', 'Fira Code', monospace;
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
.plan-details {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
    font-size: 12px;
}}
.plan-line {{
    margin-bottom: 6px;
    line-height: 1.8;
}}
.plan-line:last-child {{ margin-bottom: 0; }}
.plan-label {{
    font-weight: 700;
    color: var(--text2);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-right: 6px;
}}
.plan-line code {{
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 11px;
    color: var(--text);
    background: var(--surface2);
    padding: 2px 6px;
    border-radius: 3px;
}}
.mode-pill {{
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    color: var(--accent);
    background: rgba(124,110,246,0.12);
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid rgba(124,110,246,0.25);
    margin: 1px 2px;
}}
.attr-pill {{
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    color: var(--green);
    background: rgba(16,185,129,0.1);
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid rgba(16,185,129,0.25);
    margin: 1px 2px;
    font-family: 'SF Mono', 'Fira Code', monospace;
}}
.avoid-pill {{
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    color: var(--orange);
    background: rgba(245,158,11,0.1);
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid rgba(245,158,11,0.25);
    margin: 1px 2px;
    font-family: 'SF Mono', 'Fira Code', monospace;
}}

/* Product grid */
.products-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
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
.product-card.violation-card {{
    border: 2px solid var(--red);
    box-shadow: 0 0 12px rgba(239,68,68,0.2);
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
.violation-overlay {{
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(239,68,68,0.9);
    color: white;
    font-size: 10px;
    font-weight: 700;
    padding: 4px 8px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}
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
.tag.violation-tag {{
    background: rgba(239,68,68,0.15);
    color: var(--red);
    border: 1px solid rgba(239,68,68,0.4);
    font-weight: 700;
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
        <h1>Failed Query Regression Test</h1>
        <p>{len(results)} previously-failing queries — mode-based planner architecture with gpt-4o</p>
        <p style="margin-top:4px;color:var(--text2);font-size:12px;">{datetime.now().strftime("%B %d, %Y at %H:%M")}</p>
    </div>

    <div class="stats-bar">
        <div class="stat-card">
            <div class="stat-value {'good' if total_violations == 0 else 'bad'}">{total_violations}</div>
            <div class="stat-label">Total Violations</div>
        </div>
        <div class="stat-card">
            <div class="stat-value good">{clean_queries}/{queries_with_checks}</div>
            <div class="stat-label">Clean Queries</div>
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
                    <th style="text-align:center">Violations</th>
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

    out_path = "scripts/exclusion_filter_results.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page_html)
    print(f"\nHTML report saved to: {out_path}")
    print(f"Total violations: {total_violations} across {total_checked} products checked")
    print(f"Clean queries: {clean_queries}/{queries_with_checks} (with exclusion checks)")


if __name__ == "__main__":
    run_tests()
