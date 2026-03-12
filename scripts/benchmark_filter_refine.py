"""
Benchmark: Filter Refinement across multiple queries and filter types.

Runs each test case through:
1. Initial search (full pipeline) → page 1
2. Filter refinement (cached session) → page 1 with filters
3. Paginate until exhaustion

Reports per-case and aggregate stats: avg response time, total unique
results, speedup, exhaustion counts.

Usage:
    PYTHONPATH=src python scripts/benchmark_filter_refine.py
"""

import os
import sys
import time
import json
import requests
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

API_URL = os.getenv("API_URL", "http://localhost:8000")
SEARCH_URL = f"{API_URL}/api/search/hybrid"
PAGE_SIZE = 50


def _make_token():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    import jwt as pyjwt
    secret = os.getenv("SUPABASE_JWT_SECRET")
    now = int(time.time())
    return pyjwt.encode({
        "sub": "benchmark-filter-refine",
        "aud": "authenticated",
        "role": "authenticated",
        "email": "bench@test.com",
        "aal": "aal1",
        "exp": now + 3600,
        "iat": now,
        "is_anonymous": False,
    }, secret, algorithm="HS256")


HEADERS = {
    "Authorization": f"Bearer {_make_token()}",
    "Content-Type": "application/json",
}


# ============================================================================
# Test cases: (query, filter_dict, description)
# ============================================================================
TEST_CASES = [
    # Brand filters
    ("athletic get together outfits", {"brands": ["Boohoo"]}, "VAGUE + brand"),
    ("summer dress", {"brands": ["Forever 21"]}, "SPECIFIC + brand"),

    # Color filter
    ("midi dress", {"colors": ["Black"]}, "SPECIFIC + color"),

    # Price filter
    ("casual dresses", {"max_price": 30.0}, "SPECIFIC + max_price"),

    # Pattern filter
    ("party dress", {"patterns": ["Floral"]}, "SPECIFIC + pattern"),

    # Combined filters
    ("going out outfits", {"brands": ["Boohoo"], "max_price": 40.0}, "VAGUE + brand + price"),

    # On sale
    ("dresses", {"on_sale_only": True}, "EXACT + on_sale"),
]

# Max pages to paginate per case (avoid runaway exhaustion)
MAX_PAGES_PER_CASE = 10


def search(body: dict) -> dict:
    r = requests.post(SEARCH_URL, json=body, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.json()


def run_case(query: str, filters: dict, desc: str) -> Dict[str, Any]:
    """Run one test case: initial search → filter refine → exhaust."""
    result = {
        "query": query,
        "filters": filters,
        "desc": desc,
        "error": None,
    }

    try:
        # Phase 1: Initial search
        t0 = time.time()
        p1 = search({"query": query, "page_size": PAGE_SIZE})
        p1_wall = int((time.time() - t0) * 1000)

        p1_timing = p1.get("timing", {})
        p1_session = p1.get("search_session_id")
        p1_results = len(p1.get("results", []))
        p1_total = p1.get("pagination", {}).get("total_results", 0)
        p1_intent = p1.get("intent", "?")

        result["intent"] = p1_intent
        result["p1_results"] = p1_results
        result["p1_total"] = p1_total
        result["p1_server_ms"] = p1_timing.get("total_ms", p1_wall)
        result["p1_wall_ms"] = p1_wall

        if not p1_session:
            result["error"] = "No session ID from page 1"
            return result

        # Phase 2: Filter refinement
        refine_body = {
            "query": query,
            "page_size": PAGE_SIZE,
            "search_session_id": p1_session,
            **filters,
        }

        t0 = time.time()
        ref = search(refine_body)
        ref_wall = int((time.time() - t0) * 1000)

        ref_timing = ref.get("timing", {})
        ref_session = ref.get("search_session_id")
        ref_cursor = ref.get("cursor")
        ref_results = ref.get("results", [])
        ref_total = ref.get("pagination", {}).get("total_results", 0)
        is_refine = ref_timing.get("filter_refine", False)

        result["refine_mode"] = "filter_refine" if is_refine else "full_pipeline"
        result["refine_results"] = len(ref_results)
        result["refine_total"] = ref_total
        result["refine_server_ms"] = ref_timing.get("total_ms", ref_wall)
        result["refine_wall_ms"] = ref_wall
        result["refine_algolia_ms"] = ref_timing.get("algolia_ms", 0)
        result["refine_semantic_ms"] = ref_timing.get("semantic_ms", 0)

        if result["p1_server_ms"] and result["refine_server_ms"]:
            result["speedup"] = result["p1_server_ms"] / max(result["refine_server_ms"], 1)
        else:
            result["speedup"] = 0

        # Phase 3: Paginate to exhaustion
        all_ids = {r["product_id"] for r in ref_results}
        page_times = []
        current_session = ref_session
        current_cursor = ref_cursor
        page_num = 1

        while current_cursor:
            page_num += 1
            body = {
                "query": query,
                "page_size": PAGE_SIZE,
                "search_session_id": current_session,
                "cursor": current_cursor,
                **filters,
            }
            t0 = time.time()
            try:
                pn = search(body)
            except Exception:
                break
            pn_wall = int((time.time() - t0) * 1000)

            pn_results = pn.get("results", [])
            pn_timing = pn.get("timing", {})
            pn_ids = {r["product_id"] for r in pn_results}
            all_ids.update(pn_ids)

            page_times.append(pn_timing.get("total_ms", pn_wall))

            current_cursor = pn.get("cursor")
            has_more = pn.get("pagination", {}).get("has_more", False)
            if not has_more or not current_cursor or len(pn_results) == 0:
                break
            if page_num >= MAX_PAGES_PER_CASE:
                break

        result["total_pages"] = page_num
        result["total_unique"] = len(all_ids)
        result["avg_page2plus_ms"] = (
            sum(page_times) / len(page_times) if page_times else 0
        )
        result["exhausted"] = not current_cursor or page_num >= 2

    except Exception as e:
        result["error"] = str(e)

    return result


def fmt_table(rows, headers):
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
    output = []

    def log(msg=""):
        print(msg)
        output.append(msg)

    log("=" * 100)
    log("FILTER REFINEMENT BENCHMARK")
    log(f"Server: {API_URL}")
    log(f"Test cases: {len(TEST_CASES)}")
    log(f"Page size: {PAGE_SIZE}")
    log("=" * 100)

    results = []
    for i, (query, filters, desc) in enumerate(TEST_CASES):
        log(f"\n[{i+1}/{len(TEST_CASES)}] {desc}: \"{query}\" + {filters}")
        r = run_case(query, filters, desc)
        results.append(r)

        if r.get("error"):
            log(f"  ERROR: {r['error']}")
        else:
            log(f"  Intent: {r.get('intent', '?')} | "
                f"P1: {r.get('p1_results', 0)} results in {r.get('p1_server_ms', 0)}ms | "
                f"Refine: {r.get('refine_results', 0)} results in {r.get('refine_server_ms', 0)}ms ({r.get('refine_mode', '?')}) | "
                f"Speedup: {r.get('speedup', 0):.1f}x | "
                f"Total unique: {r.get('total_unique', 0)} across {r.get('total_pages', 0)} pages")

    # ========================================================================
    # Summary table
    # ========================================================================
    log("\n" + "=" * 100)
    log("RESULTS TABLE")
    log("=" * 100)

    headers = [
        "Case", "Intent", "P1 ms", "Refine ms", "Speedup",
        "P1 Results", "Refine Results", "Total Unique", "Pages",
        "Avg Pg2+ ms", "Mode",
    ]
    rows = []
    for r in results:
        if r.get("error"):
            rows.append([r["desc"], "ERR", "-", "-", "-", "-", "-", "-", "-", "-", r["error"][:30]])
        else:
            rows.append([
                r["desc"],
                r.get("intent", "?"),
                r.get("p1_server_ms", 0),
                r.get("refine_server_ms", 0),
                f"{r.get('speedup', 0):.1f}x",
                r.get("p1_results", 0),
                r.get("refine_results", 0),
                r.get("total_unique", 0),
                r.get("total_pages", 0),
                f"{r.get('avg_page2plus_ms', 0):.0f}",
                r.get("refine_mode", "?"),
            ])
    log(fmt_table(rows, headers))

    # ========================================================================
    # Aggregate stats
    # ========================================================================
    log("\n" + "=" * 100)
    log("AGGREGATE STATS")
    log("=" * 100)

    valid = [r for r in results if not r.get("error")]
    refine_hits = [r for r in valid if r.get("refine_mode") == "filter_refine"]
    full_fallbacks = [r for r in valid if r.get("refine_mode") != "filter_refine"]

    if valid:
        avg_p1 = sum(r["p1_server_ms"] for r in valid) / len(valid)
        avg_refine = sum(r["refine_server_ms"] for r in valid) / len(valid)
        avg_speedup = sum(r["speedup"] for r in valid) / len(valid)
        avg_unique = sum(r["total_unique"] for r in valid) / len(valid)
        avg_pages = sum(r["total_pages"] for r in valid) / len(valid)
        avg_refine_results = sum(r["refine_results"] for r in valid) / len(valid)

        page2_times = [r["avg_page2plus_ms"] for r in valid if r["avg_page2plus_ms"] > 0]
        avg_page2 = sum(page2_times) / len(page2_times) if page2_times else 0

        total_unique_all = sum(r["total_unique"] for r in valid)

        log(f"  Total test cases:              {len(TEST_CASES)}")
        log(f"  Successful:                    {len(valid)}")
        log(f"  Filter refine cache hits:      {len(refine_hits)}")
        log(f"  Full pipeline fallbacks:       {len(full_fallbacks)}")
        log(f"")
        log(f"  Avg initial search (P1):       {avg_p1:.0f}ms")
        log(f"  Avg filter refine:             {avg_refine:.0f}ms")
        log(f"  Avg speedup:                   {avg_speedup:.1f}x")
        log(f"  Avg page 2+ response:          {avg_page2:.0f}ms")
        log(f"")
        log(f"  Avg refine results (page 1):   {avg_refine_results:.1f}")
        log(f"  Avg total unique (exhausted):  {avg_unique:.1f}")
        log(f"  Avg pages to exhaustion:       {avg_pages:.1f}")
        log(f"  Total unique across all cases: {total_unique_all}")

        # Per-filter-type breakdown
        log(f"\n  --- By filter type ---")
        by_type = {}
        for r in valid:
            fkeys = sorted(r["filters"].keys())
            ftype = " + ".join(fkeys)
            by_type.setdefault(ftype, []).append(r)

        type_rows = []
        for ftype, cases in sorted(by_type.items()):
            n = len(cases)
            avg_ref = sum(c["refine_server_ms"] for c in cases) / n
            avg_sp = sum(c["speedup"] for c in cases) / n
            avg_u = sum(c["total_unique"] for c in cases) / n
            type_rows.append([ftype, n, f"{avg_ref:.0f}", f"{avg_sp:.1f}x", f"{avg_u:.0f}"])

        log(fmt_table(type_rows, ["Filter Type", "N", "Avg Refine ms", "Avg Speedup", "Avg Unique"]))

        # Min/Max
        log(f"\n  --- Extremes ---")
        fastest = min(valid, key=lambda r: r["refine_server_ms"])
        slowest = max(valid, key=lambda r: r["refine_server_ms"])
        most_results = max(valid, key=lambda r: r["total_unique"])
        fewest_results = min(valid, key=lambda r: r["total_unique"])

        log(f"  Fastest refine:   {fastest['refine_server_ms']}ms — \"{fastest['query']}\" + {fastest['filters']}")
        log(f"  Slowest refine:   {slowest['refine_server_ms']}ms — \"{slowest['query']}\" + {slowest['filters']}")
        log(f"  Most results:     {most_results['total_unique']} — \"{most_results['query']}\" + {most_results['filters']}")
        log(f"  Fewest results:   {fewest_results['total_unique']} — \"{fewest_results['query']}\" + {fewest_results['filters']}")

    # Save report
    report_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'filter_refine_benchmark.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write("\n".join(output))
    log(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
