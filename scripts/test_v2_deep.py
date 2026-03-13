#!/usr/bin/env python3
"""
Deep E2E test for v2 search pipeline.

Tests multiple query types, filters, pagination, and collects timing.

Usage:
    PYTHONPATH=src python scripts/test_v2_deep.py
"""

import json
import os
import sys
import time

import jwt
import requests
from dotenv import load_dotenv

load_dotenv()

BASE = os.environ.get("TEST_SERVER_URL", "http://localhost:8000")
V2_URL = f"{BASE}/api/search/v2/hybrid"
V1_URL = f"{BASE}/api/search/hybrid"

# Generate a fresh JWT
secret = os.environ["SUPABASE_JWT_SECRET"]
token = jwt.encode(
    {
        "sub": "e2e-deep-test-user",
        "aud": "authenticated",
        "role": "authenticated",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
    },
    secret,
    algorithm="HS256",
)
HEADERS = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# ── Formatting helpers ─────────────────────────────────────────────────

def fmt_ms(ms):
    if ms is None:
        return "  -  "
    return f"{ms:>5d}ms"


def print_timing(timing):
    parts = []
    for key in ["planner_ms", "semantic_ms", "algolia_ms", "rrf_ms", "total_ms"]:
        val = timing.get(key)
        if val is not None:
            parts.append(f"{key.replace('_ms','')}={val}ms")
    planner_src = timing.get("planner_source", "")
    if planner_src:
        parts.append(f"src={planner_src}")
    sem_q = timing.get("semantic_query_count")
    if sem_q:
        parts.append(f"sem_queries={sem_q}")
    print(f"  Timing: {' | '.join(parts)}")


def print_results(data, max_items=5):
    products = data.get("results", [])  # v2 uses "results" not "products"
    pag = data.get("pagination", {})
    total = pag.get("total_results", "?")
    has_more = pag.get("has_more", False)
    session_id = data.get("search_session_id", "")
    print(f"  Results: {len(products)} shown, {total} total, has_more={has_more}")
    if session_id:
        print(f"  Session: {session_id}")
    for i, p in enumerate(products[:max_items]):
        name = p.get("name", "?")[:45]
        brand = p.get("brand", "?")
        price = p.get("price", "?")
        cat = p.get("category_l1") or p.get("category") or "?"
        score = p.get("rrf_score") or p.get("semantic_score") or ""
        score_str = f" score={score:.4f}" if isinstance(score, float) else ""
        print(f"    {i+1}. [{brand}] {name} ${price} ({cat}){score_str}")
    if len(products) > max_items:
        print(f"    ... +{len(products) - max_items} more")


def check_enrichment(data):
    """Check if results have Gemini attributes (enrichment skip working)."""
    products = data.get("results", [])
    if not products:
        return
    sample = products[:5]
    has_attr = sum(1 for p in sample if p.get("category_l1"))
    print(f"  Attrs baked: {has_attr}/{len(sample)} have category_l1")


def search(url, payload, label=""):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Query: {payload.get('query', '(empty)')}")
    filters = {k: v for k, v in payload.items() if k not in ("query", "page", "page_size", "search_session_id", "cursor") and v}
    if filters:
        print(f"  Filters: {json.dumps(filters, default=str)}")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, headers=HEADERS, timeout=30)
    except requests.Timeout:
        print(f"  TIMEOUT after 30s")
        return None
    wall_ms = int((time.perf_counter() - t0) * 1000)

    if resp.status_code != 200:
        print(f"  ERROR {resp.status_code}: {resp.text[:200]}")
        return None

    data = resp.json()
    timing = data.get("timing", {})
    timing["wall_ms"] = wall_ms

    print_timing(timing)
    print(f"  Wall clock: {wall_ms}ms")
    print_results(data)
    check_enrichment(data)

    # Follow-ups
    follow_ups = data.get("follow_ups") or []
    if follow_ups:
        print(f"  Follow-ups: {len(follow_ups)}")
        for fu in follow_ups:
            dim = fu.get("dimension", "?")
            question = fu.get("question", "?")
            opts = fu.get("options", [])
            labels = [o.get("label", "?") for o in opts]
            print(f"    [{dim}] {question}")
            print(f"      -> {labels}")
    else:
        print(f"  Follow-ups: 0")

    return data


# ── Test cases ─────────────────────────────────────────────────────────

def main():
    print("Deep E2E Test — V2 Search Pipeline")
    print(f"Server: {BASE}")
    print(f"V2 endpoint: {V2_URL}")

    results_log = []

    def log(label, data, wall_ms=None):
        if data:
            t = data.get("timing", {})
            results_log.append({
                "label": label,
                "total_ms": t.get("total_ms"),
                "wall_ms": t.get("wall_ms", wall_ms),
                "planner_ms": t.get("planner_ms"),
                "semantic_ms": t.get("semantic_ms"),
                "algolia_ms": t.get("algolia_ms"),
                "planner_source": t.get("planner_source"),
                "results": len(data.get("results", [])),
                "total_results": data.get("pagination", {}).get("total_results"),
                "follow_ups": len(data.get("follow_ups") or []),
            })

    # ── 1. Vague queries ──────────────────────────────────────────────

    d = search(V2_URL, {"query": "good night out dress", "page_size": 20},
               "1. VAGUE: night out dress")
    log("vague: night out dress", d)

    d = search(V2_URL, {"query": "something cute for brunch", "page_size": 20},
               "2. VAGUE: cute for brunch")
    log("vague: cute for brunch", d)

    # ── 2. Specific queries ───────────────────────────────────────────

    d = search(V2_URL, {"query": "black midi dress", "page_size": 20},
               "3. SPECIFIC: black midi dress")
    log("specific: black midi dress", d)

    d = search(V2_URL, {"query": "white linen pants", "page_size": 20},
               "4. SPECIFIC: white linen pants")
    log("specific: white linen pants", d)

    # ── 3. Brand queries (heuristic bypass) ───────────────────────────

    d = search(V2_URL, {"query": "Boohoo", "page_size": 20},
               "5. EXACT/BRAND: Boohoo (heuristic bypass)")
    log("brand: Boohoo", d)

    d = search(V2_URL, {"query": "Princess Polly", "page_size": 20},
               "6. EXACT/BRAND: Princess Polly (heuristic bypass)")
    log("brand: Princess Polly", d)

    # ── 4. With filters ───────────────────────────────────────────────

    d = search(V2_URL, {
        "query": "summer dress",
        "brands": ["Boohoo", "Missguided"],
        "min_price": 20,
        "max_price": 60,
        "page_size": 20,
    }, "7. FILTERED: summer dress + brands + price range")
    log("filtered: summer dress", d)

    d = search(V2_URL, {
        "query": "office blouse",
        "colors": ["White", "Blue"],
        "page_size": 20,
    }, "8. FILTERED: office blouse + colors")
    log("filtered: office blouse", d)

    # ── 5. Sort modes ─────────────────────────────────────────────────

    d = search(V2_URL, {"query": "dress", "sort_by": "price_asc", "page_size": 10},
               "9. SORT: dress price_asc (Algolia fast path)")
    log("sort: price_asc", d)

    d = search(V2_URL, {"query": "dress", "sort_by": "price_desc", "page_size": 10},
               "10. SORT: dress price_desc (Algolia fast path)")
    log("sort: price_desc", d)

    # ── 6. Pagination (page 2 via cursor) ─────────────────────────────

    d1 = search(V2_URL, {"query": "floral maxi dress", "page_size": 10},
                "11. PAGINATION: floral maxi dress — page 1")
    log("pagination: page 1", d1)

    if d1 and d1.get("cursor") and d1.get("search_session_id"):
        d2 = search(V2_URL, {
            "query": "floral maxi dress",
            "page_size": 10,
            "search_session_id": d1["search_session_id"],
            "cursor": d1["cursor"],
        }, "12. PAGINATION: floral maxi dress — page 2 (cached)")
        log("pagination: page 2", d2)

        # Check no overlap
        if d1 and d2:
            ids1 = {p["product_id"] for p in d1.get("results", [])}
            ids2 = {p["product_id"] for p in d2.get("results", [])}
            overlap = ids1 & ids2
            print(f"  Page overlap: {len(overlap)} products (should be 0)")

        # Page 3
        if d2 and d2.get("cursor"):
            d3 = search(V2_URL, {
                "query": "floral maxi dress",
                "page_size": 10,
                "search_session_id": d1["search_session_id"],
                "cursor": d2["cursor"],
            }, "13. PAGINATION: floral maxi dress — page 3 (cached)")
            log("pagination: page 3", d3)
    else:
        print("\n  SKIP page 2/3 — no cursor or session_id in page 1 response")

    # ── 7. V1 vs V2 comparison ────────────────────────────────────────

    query = "casual summer top"
    print(f"\n{'='*70}")
    print(f"  V1 vs V2 COMPARISON: '{query}'")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    r1 = requests.post(V1_URL, json={"query": query, "page_size": 10}, headers=HEADERS, timeout=30)
    v1_wall = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    r2 = requests.post(V2_URL, json={"query": query, "page_size": 10}, headers=HEADERS, timeout=30)
    v2_wall = int((time.perf_counter() - t0) * 1000)

    if r1.status_code == 200 and r2.status_code == 200:
        d1_data = r1.json()
        d2_data = r2.json()
        t1 = d1_data.get("timing", {})
        t2 = d2_data.get("timing", {})
        print(f"  V1: {v1_wall}ms wall, {t1.get('total_ms')}ms server, "
              f"planner={t1.get('planner_ms')}ms, semantic={t1.get('semantic_ms')}ms, "
              f"algolia={t1.get('algolia_ms')}ms")
        print(f"  V2: {v2_wall}ms wall, {t2.get('total_ms')}ms server, "
              f"planner={t2.get('planner_ms')}ms, semantic={t2.get('semantic_ms')}ms, "
              f"algolia={t2.get('algolia_ms')}ms, src={t2.get('planner_source')}")

        ids1 = {p["product_id"] for p in d1_data.get("results", [])}
        ids2 = {p["product_id"] for p in d2_data.get("results", [])}
        overlap = ids1 & ids2
        print(f"  Result overlap: {len(overlap)}/{min(len(ids1), len(ids2))} products")

        speedup = v1_wall / v2_wall if v2_wall > 0 else 0
        print(f"  Speedup: {speedup:.1f}x")
    else:
        print(f"  V1: {r1.status_code}, V2: {r2.status_code}")

    # ── Summary table ─────────────────────────────────────────────────

    print(f"\n\n{'='*90}")
    print(f"  TIMING SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Label':<35} {'Wall':>7} {'Server':>7} {'Planner':>8} {'Semantic':>9} {'Algolia':>8} {'Source':>10} {'Results':>8} {'FU':>4}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*8} {'-'*9} {'-'*8} {'-'*10} {'-'*8} {'-'*4}")
    for r in results_log:
        print(f"  {r['label']:<35} {fmt_ms(r['wall_ms'])} {fmt_ms(r['total_ms'])} "
              f"{fmt_ms(r['planner_ms'])} {fmt_ms(r['semantic_ms'])} "
              f"{fmt_ms(r['algolia_ms'])} {r.get('planner_source', ''):<10} "
              f"{r['results']:>3}/{r.get('total_results', '?')}"
              f" {r.get('follow_ups', 0):>4}")

    # Calculate averages
    walls = [r["wall_ms"] for r in results_log if r["wall_ms"]]
    if walls:
        print(f"\n  Avg wall clock: {sum(walls)/len(walls):.0f}ms (n={len(walls)})")
        print(f"  Min: {min(walls)}ms, Max: {max(walls)}ms")


if __name__ == "__main__":
    main()
