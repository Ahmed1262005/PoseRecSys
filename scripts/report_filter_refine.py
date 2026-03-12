"""
Filter Refinement Report: "athletic get together outfits" + Boohoo filter.

Runs:
1. Initial search (full pipeline)
2. Filter refinement with brands=["Boohoo"] (cached session)
3. Paginate (page 2, 3, ...) until semantic exhaustion
4. Print detailed report with counts, timings, facets.

Usage:
    PYTHONPATH=src python scripts/report_filter_refine.py
"""

import os
import sys
import time
import json
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

API_URL = os.getenv("API_URL", "http://localhost:8000")
SEARCH_URL = f"{API_URL}/api/search/hybrid"


def _make_token():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    import jwt as pyjwt
    secret = os.getenv("SUPABASE_JWT_SECRET")
    now = int(time.time())
    return pyjwt.encode({
        "sub": "report-filter-refine",
        "aud": "authenticated",
        "role": "authenticated",
        "email": "report@test.com",
        "aal": "aal1",
        "exp": now + 3600,
        "iat": now,
        "is_anonymous": False,
    }, secret, algorithm="HS256")


HEADERS = {
    "Authorization": f"Bearer {_make_token()}",
    "Content-Type": "application/json",
}

QUERY = "athletic get together outfits"
BRAND = "Boohoo"
PAGE_SIZE = 50


def search(body: dict) -> dict:
    r = requests.post(SEARCH_URL, json=body, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.json()


def get_brands(data):
    brands = {}
    for p in data.get("results", []):
        b = p.get("brand", "Unknown")
        brands[b] = brands.get(b, 0) + 1
    return brands


def get_categories(data):
    cats = {}
    for p in data.get("results", []):
        c = p.get("category_l1") or p.get("broad_category") or "Unknown"
        cats[c] = cats.get(c, 0) + 1
    return cats


def get_facet_summary(data, facet_name, top_n=10):
    facets = data.get("facets", {}).get(facet_name, [])
    items = []
    for fv in facets[:top_n]:
        v = fv.get("value") if isinstance(fv, dict) else getattr(fv, "value", "?")
        c = fv.get("count") if isinstance(fv, dict) else getattr(fv, "count", 0)
        items.append((v, c))
    return items


def fmt_table(rows, headers):
    """Format a simple ASCII table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    hdr = "|" + "|".join(f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)) + "|"

    lines = [sep, hdr, sep]
    for row in rows:
        line = "|" + "|".join(f" {str(c):<{col_widths[i]}} " for i, c in enumerate(row)) + "|"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def main():
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=" * 80)
    log("FILTER REFINEMENT REPORT")
    log(f"Query: \"{QUERY}\"")
    log(f"Brand filter: {BRAND}")
    log(f"Page size: {PAGE_SIZE}")
    log(f"Server: {API_URL}")
    log("=" * 80)

    # =========================================================================
    # PHASE 1: Initial Search (full pipeline)
    # =========================================================================
    log("\n--- PHASE 1: Initial Search (full pipeline) ---")

    t0 = time.time()
    p1 = search({"query": QUERY, "page_size": PAGE_SIZE})
    wall_p1 = int((time.time() - t0) * 1000)

    p1_results = len(p1.get("results", []))
    p1_total = p1.get("pagination", {}).get("total_results", 0)
    p1_intent = p1.get("intent", "?")
    p1_timing = p1.get("timing", {})
    p1_session = p1.get("search_session_id")
    p1_cursor = p1.get("cursor")
    p1_ids = {r["product_id"] for r in p1.get("results", [])}

    log(f"  Intent:          {p1_intent}")
    log(f"  Results:         {p1_results}")
    log(f"  total_results:   {p1_total}")
    log(f"  Session ID:      {p1_session}")
    log(f"  Cursor:          {'yes' if p1_cursor else 'no'}")
    log(f"  Wall time:       {wall_p1}ms")
    log(f"  Server timing:   {p1_timing.get('total_ms', '?')}ms")
    log(f"    planner:       {p1_timing.get('planner_ms', '?')}ms")
    log(f"    algolia:       {p1_timing.get('algolia_ms', '?')}ms")
    log(f"    semantic:      {p1_timing.get('semantic_ms', '?')}ms")
    log(f"    rrf+rerank:    {p1_timing.get('rerank_ms', '?')}ms")

    brands_p1 = get_brands(p1)
    cats_p1 = get_categories(p1)
    log(f"\n  Brands in results: {dict(sorted(brands_p1.items(), key=lambda x: -x[1]))}")
    log(f"  Categories:        {dict(sorted(cats_p1.items(), key=lambda x: -x[1]))}")

    # Show brand facets
    brand_facets = get_facet_summary(p1, "brand", top_n=15)
    if brand_facets:
        log(f"\n  Brand facets (top 15):")
        rows = [(v, c) for v, c in brand_facets]
        log("  " + fmt_table(rows, ["Brand", "Count"]).replace("\n", "\n  "))

    # =========================================================================
    # PHASE 2: Filter Refinement (Boohoo)
    # =========================================================================
    log(f"\n--- PHASE 2: Filter Refinement (brand={BRAND}) ---")

    if not p1_session:
        log("  ERROR: No session ID from page 1, cannot refine.")
        return

    t0 = time.time()
    refined = search({
        "query": QUERY,
        "page_size": PAGE_SIZE,
        "search_session_id": p1_session,
        "brands": [BRAND],
        # NO cursor = filter refinement signal
    })
    wall_refine = int((time.time() - t0) * 1000)

    r_results = len(refined.get("results", []))
    r_total = refined.get("pagination", {}).get("total_results", 0)
    r_timing = refined.get("timing", {})
    r_session = refined.get("search_session_id")
    r_cursor = refined.get("cursor")
    is_refine = r_timing.get("filter_refine", False)
    r_ids = {r["product_id"] for r in refined.get("results", [])}

    log(f"  Mode:            {'filter_refine' if is_refine else 'FULL PIPELINE (cache miss!)'}")
    log(f"  Results:         {r_results}")
    log(f"  total_results:   {r_total} (was {p1_total} before filter)")
    log(f"  New Session ID:  {r_session}")
    log(f"  Session changed: {r_session != p1_session}")
    log(f"  Cursor:          {'yes' if r_cursor else 'no'}")
    log(f"  Wall time:       {wall_refine}ms")
    log(f"  Server timing:   {r_timing.get('total_ms', '?')}ms")
    log(f"    algolia:       {r_timing.get('algolia_ms', '?')}ms")
    log(f"    semantic:      {r_timing.get('semantic_ms', '?')}ms")
    log(f"    enrich:        {r_timing.get('enrich_ms', '?')}ms")
    log(f"    rrf:           {r_timing.get('rrf_ms', '?')}ms")

    speedup = p1_timing.get("total_ms", 0) / max(r_timing.get("total_ms", 1), 1)
    log(f"\n  Speedup vs full pipeline: {speedup:.1f}x ({p1_timing.get('total_ms', '?')}ms -> {r_timing.get('total_ms', '?')}ms)")

    brands_r = get_brands(refined)
    cats_r = get_categories(refined)
    log(f"\n  Brands in results: {dict(sorted(brands_r.items(), key=lambda x: -x[1]))}")
    log(f"  Categories:        {dict(sorted(cats_r.items(), key=lambda x: -x[1]))}")

    # Overlap check
    overlap = p1_ids & r_ids
    log(f"\n  Overlap with original page 1: {len(overlap)} products")

    # Show facets after refinement
    brand_facets_r = get_facet_summary(refined, "brand", top_n=10)
    if brand_facets_r:
        log(f"\n  Brand facets after refinement (top 10):")
        rows = [(v, c) for v, c in brand_facets_r]
        log("  " + fmt_table(rows, ["Brand", "Count"]).replace("\n", "\n  "))

    cat_facets_r = get_facet_summary(refined, "category_l1", top_n=10)
    if cat_facets_r:
        log(f"\n  Category facets after refinement (top 10):")
        rows = [(v, c) for v, c in cat_facets_r]
        log("  " + fmt_table(rows, ["Category", "Count"]).replace("\n", "\n  "))

    # =========================================================================
    # PHASE 3: Paginate until exhaustion
    # =========================================================================
    log(f"\n--- PHASE 3: Endless Pagination (exhaust semantic) ---")

    if not r_session or not r_cursor:
        log("  No session/cursor from refinement, cannot paginate.")
        return

    all_ids = set(r_ids)
    page_stats = []
    current_session = r_session
    current_cursor = r_cursor
    page_num = 1  # refined was page 1

    # Record page 1
    page_stats.append({
        "page": 1,
        "results": r_results,
        "unique_total": len(all_ids),
        "overlap": 0,
        "wall_ms": wall_refine,
        "server_ms": r_timing.get("total_ms", 0),
        "mode": "filter_refine",
        "rounds": "-",
        "seen_ids": r_timing.get("seen_ids_total", "-"),
    })

    while current_cursor:
        page_num += 1
        body = {
            "query": QUERY,
            "page": page_num,
            "page_size": PAGE_SIZE,
            "search_session_id": current_session,
            "cursor": current_cursor,
            "brands": [BRAND],  # keep brand filter for consistency
        }

        t0 = time.time()
        try:
            pn = search(body)
        except Exception as e:
            log(f"  Page {page_num}: ERROR {e}")
            break
        wall_pn = int((time.time() - t0) * 1000)

        pn_results = pn.get("results", [])
        pn_ids = {r["product_id"] for r in pn_results}
        pn_timing = pn.get("timing", {})
        overlap_n = all_ids & pn_ids
        all_ids.update(pn_ids)

        is_endless = pn_timing.get("endless_semantic", False)
        is_extend = pn_timing.get("extend_search", False)
        mode = "endless_semantic" if is_endless else ("extend_search" if is_extend else "unknown")

        page_stats.append({
            "page": page_num,
            "results": len(pn_results),
            "unique_total": len(all_ids),
            "overlap": len(overlap_n),
            "wall_ms": wall_pn,
            "server_ms": pn_timing.get("total_ms", 0),
            "mode": mode,
            "rounds": pn_timing.get("rounds", "-"),
            "seen_ids": pn_timing.get("seen_ids_total", "-"),
        })

        current_cursor = pn.get("cursor")
        has_more = pn.get("pagination", {}).get("has_more", False)

        if not has_more or not current_cursor or len(pn_results) == 0:
            break

        # Safety cap
        if page_num >= 100:
            log("  Safety cap: stopped at page 100")
            break

    # Print pagination table
    rows = [
        (
            s["page"], s["results"], s["unique_total"], s["overlap"],
            s["wall_ms"], s["server_ms"], s["mode"], s["rounds"], s["seen_ids"],
        )
        for s in page_stats
    ]
    log("\n" + fmt_table(
        rows,
        ["Page", "Results", "Unique Total", "Overlap", "Wall ms", "Server ms", "Mode", "Rounds", "Seen IDs"],
    ))

    # Summary
    total_unique = len(all_ids)
    total_pages = len(page_stats)
    total_server_ms = sum(s["server_ms"] for s in page_stats)
    avg_page_ms = total_server_ms / max(total_pages - 1, 1) if total_pages > 1 else 0  # exclude refine page
    total_overlap = sum(s["overlap"] for s in page_stats)

    log(f"\n--- SUMMARY ---")
    log(f"  Query:              \"{QUERY}\"")
    log(f"  Brand filter:       {BRAND}")
    log(f"  Intent:             {p1_intent}")
    log(f"  Original total:     {p1_total}")
    log(f"  Filtered total:     {r_total}")
    log(f"  Total pages:        {total_pages}")
    log(f"  Total unique:       {total_unique}")
    log(f"  Total duplicates:   {total_overlap}")
    log(f"  Full pipeline P1:   {p1_timing.get('total_ms', '?')}ms")
    log(f"  Filter refine:      {r_timing.get('total_ms', '?')}ms")
    log(f"  Speedup:            {speedup:.1f}x")
    log(f"  Avg page 2+ ms:     {avg_page_ms:.0f}ms")
    log(f"  Last page results:  {page_stats[-1]['results']}")
    log(f"  Exhausted:          {'yes' if page_stats[-1]['results'] == 0 or not current_cursor else 'no'}")

    # Write report to file
    report_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'filter_refine_report.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    log(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
