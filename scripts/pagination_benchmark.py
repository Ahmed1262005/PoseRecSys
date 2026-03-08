#!/usr/bin/env python3
"""
Pagination Benchmark: Cache vs Re-run.

Compares two infinite-scroll strategies for hybrid search:
  A) CACHED:  Page 1 runs full pipeline + caches; pages 2-4 served from cache.
  B) RE-RUN:  Every page re-runs the full pipeline (current behaviour).

Outputs a terminal summary + HTML report.

Usage:
    PYTHONPATH=src python scripts/pagination_benchmark.py
"""

import os
import sys
import time
import json
from pathlib import Path

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dotenv import load_dotenv
load_dotenv()

from search.hybrid_search import HybridSearchService, get_hybrid_search_service
from search.models import HybridSearchRequest, SortBy
from search.session_cache import SearchSessionCache, decode_cursor

# ── Test queries (mix of intents) ──────────────────────────────────────────
TEST_QUERIES = [
    "summer dress",
    "black tops for work",
    "boho maxi dress",
    "Zara",
    "casual outfit ideas",
    "date night dress",
]

PAGES = 4
PAGE_SIZE = 50


def _fmt_ms(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def run_cached_benchmark(service: HybridSearchService, query: str) -> dict:
    """Run pages 1-4 using cached pagination."""
    pages = []

    # Page 1: full pipeline (returns search_session_id + cursor)
    req = HybridSearchRequest(query=query, page=1, page_size=PAGE_SIZE)
    t0 = time.time()
    resp = service.search(req)
    elapsed = (time.time() - t0) * 1000
    pages.append({
        "page": 1,
        "latency_ms": elapsed,
        "results": len(resp.results),
        "total_results": resp.pagination.total_results,
        "has_more": resp.pagination.has_more,
        "cache_hit": False,
    })

    session_id = resp.search_session_id
    cursor = resp.cursor

    # Pages 2-4: served from cache
    for page_num in range(2, PAGES + 1):
        if not cursor or not session_id:
            pages.append({
                "page": page_num,
                "latency_ms": 0,
                "results": 0,
                "total_results": 0,
                "has_more": False,
                "cache_hit": False,
                "skipped": True,
            })
            continue

        req2 = HybridSearchRequest(
            query=query,
            page=page_num,
            page_size=PAGE_SIZE,
            search_session_id=session_id,
            cursor=cursor,
        )
        t0 = time.time()
        resp2 = service.search(req2)
        elapsed2 = (time.time() - t0) * 1000
        pages.append({
            "page": page_num,
            "latency_ms": elapsed2,
            "results": len(resp2.results),
            "total_results": resp2.pagination.total_results,
            "has_more": resp2.pagination.has_more,
            "cache_hit": resp2.timing.get("cache_hit", False),
        })
        cursor = resp2.cursor
        session_id = resp2.search_session_id

    # Collect all product IDs across pages
    all_ids = set()
    for p in pages:
        all_ids.update(set())  # We don't have IDs in summary; just count

    return {
        "query": query,
        "mode": "cached",
        "pages": pages,
        "total_latency_ms": sum(p["latency_ms"] for p in pages),
        "total_results_shown": sum(p["results"] for p in pages),
    }


def run_rerun_benchmark(service: HybridSearchService, query: str) -> dict:
    """Run pages 1-4 by re-running the full pipeline each time (no cache)."""
    pages = []

    for page_num in range(1, PAGES + 1):
        # Force no cache by not passing search_session_id/cursor
        req = HybridSearchRequest(
            query=query, page=page_num, page_size=PAGE_SIZE
        )
        t0 = time.time()
        resp = service.search(req)
        elapsed = (time.time() - t0) * 1000
        pages.append({
            "page": page_num,
            "latency_ms": elapsed,
            "results": len(resp.results),
            "total_results": resp.pagination.total_results,
            "has_more": resp.pagination.has_more,
            "cache_hit": False,
        })

    return {
        "query": query,
        "mode": "rerun",
        "pages": pages,
        "total_latency_ms": sum(p["latency_ms"] for p in pages),
        "total_results_shown": sum(p["results"] for p in pages),
    }


def print_summary(results: list):
    """Print terminal summary table."""
    print("\n" + "=" * 90)
    print("PAGINATION BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'Query':<30} {'Mode':<8} {'P1':>8} {'P2':>8} {'P3':>8} {'P4':>8} {'Total':>10} {'Items':>6}")
    print("-" * 90)

    for r in results:
        pages = r["pages"]
        latencies = [_fmt_ms(p["latency_ms"]) for p in pages]
        while len(latencies) < 4:
            latencies.append("-")
        print(
            f"{r['query']:<30} {r['mode']:<8} "
            f"{latencies[0]:>8} {latencies[1]:>8} {latencies[2]:>8} {latencies[3]:>8} "
            f"{_fmt_ms(r['total_latency_ms']):>10} {r['total_results_shown']:>6}"
        )

    print("=" * 90)

    # Compute aggregate stats
    cached = [r for r in results if r["mode"] == "cached"]
    rerun = [r for r in results if r["mode"] == "rerun"]

    if cached and rerun:
        avg_cached_total = sum(r["total_latency_ms"] for r in cached) / len(cached)
        avg_rerun_total = sum(r["total_latency_ms"] for r in rerun) / len(rerun)
        avg_cached_p2 = sum(r["pages"][1]["latency_ms"] for r in cached if len(r["pages"]) > 1) / max(1, len(cached))
        avg_rerun_p2 = sum(r["pages"][1]["latency_ms"] for r in rerun if len(r["pages"]) > 1) / max(1, len(rerun))

        print(f"\nAVERAGES:")
        print(f"  Cached total (4 pages): {_fmt_ms(avg_cached_total)}")
        print(f"  Re-run total (4 pages): {_fmt_ms(avg_rerun_total)}")
        print(f"  Speedup factor:         {avg_rerun_total / max(1, avg_cached_total):.1f}x")
        print(f"  Cached page 2 avg:      {_fmt_ms(avg_cached_p2)}")
        print(f"  Re-run page 2 avg:      {_fmt_ms(avg_rerun_p2)}")
        print()


def generate_html_report(results: list, output_path: str):
    """Generate an HTML report of benchmark results."""
    cached = [r for r in results if r["mode"] == "cached"]
    rerun = [r for r in results if r["mode"] == "rerun"]

    # Build per-query comparison
    query_comparisons = []
    queries = list(dict.fromkeys(r["query"] for r in results))
    for q in queries:
        c = next((r for r in cached if r["query"] == q), None)
        rr = next((r for r in rerun if r["query"] == q), None)
        if c and rr:
            query_comparisons.append({"query": q, "cached": c, "rerun": rr})

    # Aggregate stats
    avg_cached_total = sum(r["total_latency_ms"] for r in cached) / max(1, len(cached))
    avg_rerun_total = sum(r["total_latency_ms"] for r in rerun) / max(1, len(rerun))
    speedup = avg_rerun_total / max(1, avg_cached_total)

    rows_html = ""
    for comp in query_comparisons:
        c, rr = comp["cached"], comp["rerun"]
        for mode_label, data, bg in [("Cached", c, "#e8f5e9"), ("Re-run", rr, "#fff3e0")]:
            pages = data["pages"]
            cells = "".join(
                f'<td style="text-align:right;">{_fmt_ms(p["latency_ms"])} <small>({p["results"]})</small></td>'
                for p in pages
            )
            rows_html += f"""
            <tr style="background:{bg};">
                <td><strong>{comp['query']}</strong></td>
                <td>{mode_label}</td>
                {cells}
                <td style="text-align:right;font-weight:bold;">{_fmt_ms(data['total_latency_ms'])}</td>
                <td style="text-align:right;">{data['total_results_shown']}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Pagination Benchmark</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; }}
h1 {{ color: #333; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ padding: 8px 12px; border: 1px solid #ddd; font-size: 14px; }}
th {{ background: #f5f5f5; text-align: left; }}
.metric {{ display: inline-block; padding: 16px 24px; margin: 8px; border-radius: 8px; text-align: center; }}
.metric .value {{ font-size: 28px; font-weight: bold; }}
.metric .label {{ font-size: 12px; color: #666; margin-top: 4px; }}
.green {{ background: #e8f5e9; }}
.orange {{ background: #fff3e0; }}
.blue {{ background: #e3f2fd; }}
</style></head><body>
<h1>Pagination Benchmark: Cached vs Re-run</h1>
<p>{len(queries)} queries, {PAGES} pages of {PAGE_SIZE} each</p>

<div>
    <div class="metric green"><div class="value">{_fmt_ms(avg_cached_total)}</div><div class="label">Avg Cached (4 pages)</div></div>
    <div class="metric orange"><div class="value">{_fmt_ms(avg_rerun_total)}</div><div class="label">Avg Re-run (4 pages)</div></div>
    <div class="metric blue"><div class="value">{speedup:.1f}x</div><div class="label">Speedup</div></div>
</div>

<h2>Per-Query Breakdown</h2>
<table>
<tr><th>Query</th><th>Mode</th><th>Page 1</th><th>Page 2</th><th>Page 3</th><th>Page 4</th><th>Total</th><th>Items</th></tr>
{rows_html}
</table>

<p><small>Generated {time.strftime('%Y-%m-%d %H:%M')}</small></p>
</body></html>"""

    Path(output_path).write_text(html)
    print(f"\nHTML report: {output_path}")


def main():
    print("Initializing HybridSearchService...")
    service = get_hybrid_search_service()

    all_results = []

    for i, query in enumerate(TEST_QUERIES):
        print(f"\n[{i+1}/{len(TEST_QUERIES)}] Query: {query!r}")

        # Run cached version
        print(f"  Running CACHED (pages 1-{PAGES})...")
        cached_result = run_cached_benchmark(service, query)
        all_results.append(cached_result)
        for p in cached_result["pages"]:
            hit = " [CACHE HIT]" if p.get("cache_hit") else ""
            skip = " [SKIPPED]" if p.get("skipped") else ""
            print(f"    Page {p['page']}: {_fmt_ms(p['latency_ms']):>8} — {p['results']} results{hit}{skip}")

        # Run re-run version
        print(f"  Running RE-RUN (pages 1-{PAGES})...")
        rerun_result = run_rerun_benchmark(service, query)
        all_results.append(rerun_result)
        for p in rerun_result["pages"]:
            print(f"    Page {p['page']}: {_fmt_ms(p['latency_ms']):>8} — {p['results']} results")

    print_summary(all_results)

    output_path = os.path.join(os.path.dirname(__file__), "pagination_benchmark.html")
    generate_html_report(all_results, output_path)


if __name__ == "__main__":
    main()
