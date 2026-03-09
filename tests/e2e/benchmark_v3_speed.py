#!/usr/bin/env python3
"""
V3 Feed Speed Benchmark — measures end-to-end latency.

Tests:
  1. First page (pool rebuild):  target ~400-700ms
  2. Subsequent pages (reuse):   target ~30-50ms
  3. Per-component breakdown
  4. Multiple scenarios (explore, sale, brand-filtered)

Run:
    source .venv/bin/activate
    PYTHONPATH=src python tests/e2e/benchmark_v3_speed.py
"""

import os
import sys
import time
import statistics
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.database import get_supabase_client
from recs.models import HardFilters
from recs.v3.eligibility import EligibilityFilter
from recs.v3.events import NoOpEventLogger
from recs.v3.feature_hydrator import FeatureHydrator
from recs.v3.mixer import CandidateMixer
from recs.v3.models import FeedRequest, SessionProfile
from recs.v3.orchestrator import FeedOrchestrator
from recs.v3.pool_manager import PoolManager
from recs.v3.ranker import FeedRanker
from recs.v3.reranker import V3Reranker
from recs.v3.session_store import InMemorySessionStore
from recs.v3.sources.exploration_source import ExplorationSource
from recs.v3.sources.merch_source import MerchSource
from recs.v3.sources.preference_source import PreferenceSource
from recs.v3.sources.session_source import SessionSource
from recs.v3.user_profile import UserProfileLoader


def build_orchestrator(supabase) -> FeedOrchestrator:
    """Build a real orchestrator with InMemory session store and NoOp events."""
    return FeedOrchestrator(
        session_store=InMemorySessionStore(),
        user_profile=UserProfileLoader(supabase),
        pool_manager=PoolManager(),
        sources={
            "preference": PreferenceSource(supabase),
            "session": SessionSource(supabase),
            "exploration": ExplorationSource(supabase),
            "merch": MerchSource(supabase),
        },
        mixer=CandidateMixer(),
        hydrator=FeatureHydrator(supabase),
        eligibility=EligibilityFilter(),
        ranker=FeedRanker(),
        reranker=V3Reranker(),
        events=NoOpEventLogger(),
    )


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms"


def run_scenario(
    name: str,
    orch: FeedOrchestrator,
    request: FeedRequest,
    pages: int = 5,
) -> Dict[str, Any]:
    """Run a scenario: first page (rebuild) + subsequent pages (reuse)."""
    print(f"\n{'='*60}")
    print(f"  Scenario: {name}")
    print(f"  Mode: {request.mode}  Pages: {pages}  PageSize: {request.page_size}")
    if request.hard_filters:
        hf = request.hard_filters
        extras = []
        if getattr(hf, "include_brands", None):
            extras.append(f"include_brands={hf.include_brands}")
        if getattr(hf, "categories", None):
            extras.append(f"categories={hf.categories}")
        if getattr(hf, "min_price", None) is not None:
            extras.append(f"min_price={hf.min_price}")
        if getattr(hf, "max_price", None) is not None:
            extras.append(f"max_price={hf.max_price}")
        if extras:
            print(f"  Filters: {', '.join(extras)}")
    print(f"{'='*60}")

    timings = []
    pool_sizes = []
    item_counts = []
    session_id = None

    for page_num in range(1, pages + 1):
        # For pages after the first, reuse session_id + cursor
        req = FeedRequest(
            user_id=request.user_id,
            session_id=session_id,
            page_size=request.page_size,
            mode=request.mode,
            hard_filters=request.hard_filters,
            cursor=None,  # Pool handles cursor internally via served_count
            debug=True,
        )

        t0 = time.time()
        response = orch.get_feed(req)
        elapsed = time.time() - t0

        d = response.to_dict()
        session_id = d["session_id"]

        items = len(d["results"])
        pool_size = d["pagination"].get("pool_size", 0)
        debug = d.get("debug", {})
        decision = debug.get("pool_decision", {}).get("action", "?") if debug else "?"
        elapsed_internal = debug.get("elapsed_ms", 0) if debug else 0

        timings.append(elapsed)
        pool_sizes.append(pool_size)
        item_counts.append(items)

        label = "REBUILD" if decision in ("rebuild", "top_up") else "REUSE"
        print(
            f"  Page {page_num}: {fmt_ms(elapsed):>7s} ({label:7s})  "
            f"items={items:>3d}  pool={pool_size:>4d}  "
            f"internal={elapsed_internal:.0f}ms"
        )

    # Stats
    first = timings[0]
    subsequent = timings[1:] if len(timings) > 1 else []
    avg_subsequent = statistics.mean(subsequent) if subsequent else 0

    result = {
        "name": name,
        "first_page_ms": first * 1000,
        "avg_subsequent_ms": avg_subsequent * 1000,
        "min_subsequent_ms": min(subsequent) * 1000 if subsequent else 0,
        "max_subsequent_ms": max(subsequent) * 1000 if subsequent else 0,
        "pool_size": pool_sizes[0] if pool_sizes else 0,
        "total_items_served": sum(item_counts),
        "pages": pages,
    }

    print(f"\n  Summary:")
    print(f"    First page (rebuild): {result['first_page_ms']:.0f}ms")
    if subsequent:
        print(
            f"    Subsequent (reuse):   {result['avg_subsequent_ms']:.0f}ms avg  "
            f"[{result['min_subsequent_ms']:.0f}-{result['max_subsequent_ms']:.0f}ms]"
        )
    print(f"    Pool size:            {result['pool_size']}")
    print(f"    Total items served:   {result['total_items_served']}")

    return result


def run_component_benchmark(supabase):
    """Measure individual component latencies."""
    print(f"\n{'='*60}")
    print(f"  Component-Level Benchmark")
    print(f"{'='*60}")

    from recs.v3.sources.preference_source import PreferenceSource, _assign_key_family

    # 1. Single RPC call (explore_key)
    t0 = time.time()
    result = supabase.rpc("v3_get_candidates_by_explore_key", {
        "p_key_family": "a", "p_gender": "female", "p_limit": 200,
    }).execute()
    t_rpc = time.time() - t0
    rows = result.data or []
    print(f"  explore_key RPC (200 items):   {fmt_ms(t_rpc):>7s}  ({len(rows)} rows)")

    # 2. Freshness RPC
    t0 = time.time()
    result = supabase.rpc("v3_get_candidates_by_freshness", {
        "p_days": 30, "p_limit": 75,
    }).execute()
    t_fresh = time.time() - t0
    fresh_rows = result.data or []
    print(f"  freshness RPC (75 items):      {fmt_ms(t_fresh):>7s}  ({len(fresh_rows)} rows)")

    # 3. Hydrate RPC (24 items — page size)
    ids_24 = [str(r["id"]) for r in rows[:24]]
    t0 = time.time()
    result = supabase.rpc("v3_hydrate_candidates", {"p_ids": ids_24}).execute()
    t_hydrate_24 = time.time() - t0
    print(f"  hydrate RPC (24 items):        {fmt_ms(t_hydrate_24):>7s}")

    # 4. Hydrate RPC (500 items — pool size)
    ids_500 = [str(r["id"]) for r in rows[:min(len(rows), 200)]]
    # Need more IDs — fetch another batch
    result2 = supabase.rpc("v3_get_candidates_by_explore_key", {
        "p_key_family": "b", "p_gender": "female", "p_limit": 300,
    }).execute()
    ids_500.extend([str(r["id"]) for r in (result2.data or [])[:300]])
    ids_500 = list(set(ids_500))[:500]

    t0 = time.time()
    result = supabase.rpc("v3_hydrate_candidates", {"p_ids": ids_500}).execute()
    t_hydrate_500 = time.time() - t0
    print(f"  hydrate RPC ({len(ids_500)} items):       {fmt_ms(t_hydrate_500):>7s}")

    # 5. Eligibility filter (500 items, CPU-only)
    from recs.v3.feature_hydrator import _row_to_candidate
    candidates = []
    for r in (result.data or []):
        try:
            candidates.append(_row_to_candidate(r))
        except Exception:
            pass

    eligibility = EligibilityFilter()
    t0 = time.time()
    passed, stats = eligibility.filter(candidates)
    t_elig = time.time() - t0
    print(f"  eligibility ({len(candidates)} -> {len(passed)}):  {fmt_ms(t_elig):>7s}")

    # 6. Ranker (CPU-only)
    ranker = FeedRanker()
    t0 = time.time()
    ranked = ranker.rank(passed)
    t_rank = time.time() - t0
    print(f"  ranker ({len(passed)} items):           {fmt_ms(t_rank):>7s}")

    # 7. Reranker (CPU-only)
    reranker = V3Reranker()
    t0 = time.time()
    reranked = reranker.rerank(ranked, target_size=500)
    t_rerank = time.time() - t0
    print(f"  reranker ({len(ranked)} -> {len(reranked)}):    {fmt_ms(t_rerank):>7s}")

    # 8. Mixer (CPU-only)
    from recs.v3.models import CandidateStub
    stubs = [
        CandidateStub(item_id=str(r["id"]), source="preference",
                       retrieval_score=float(r.get("explore_score", 0) or 0))
        for r in rows
    ]
    mixer = CandidateMixer()
    t0 = time.time()
    mixed = mixer.mix({"preference": stubs}, target_size=500)
    t_mix = time.time() - t0
    print(f"  mixer ({len(stubs)} -> {len(mixed)}):         {fmt_ms(t_mix):>7s}")

    print(f"\n  Estimated first-page breakdown:")
    # Sources run in parallel, so take the max of the RPCs as an estimate
    t_sources_est = max(t_rpc, t_fresh) * 1.2  # 4 sources, 2 RPCs each, parallelized
    total_est = t_sources_est + t_mix + t_hydrate_500 + t_elig + t_rank + t_rerank + t_hydrate_24
    print(f"    Sources (parallel):  ~{fmt_ms(t_sources_est)}")
    print(f"    Mix:                  {fmt_ms(t_mix)}")
    print(f"    Hydrate (pool):       {fmt_ms(t_hydrate_500)}")
    print(f"    Eligibility:          {fmt_ms(t_elig)}")
    print(f"    Rank:                 {fmt_ms(t_rank)}")
    print(f"    Rerank:               {fmt_ms(t_rerank)}")
    print(f"    Hydrate (page):       {fmt_ms(t_hydrate_24)}")
    print(f"    ─────────────────────────")
    print(f"    Estimated total:     ~{fmt_ms(total_est)}")


def main():
    print("V3 Feed Speed Benchmark")
    print("=" * 60)
    print("Connecting to Supabase...")
    supabase = get_supabase_client()
    print("Connected.\n")

    # Build orchestrator
    orch = build_orchestrator(supabase)

    all_results = []

    # Scenario 1: Default explore feed
    r = run_scenario(
        "Default Explore Feed",
        orch,
        FeedRequest(user_id="bench-user-1", mode="explore", page_size=24, debug=True),
        pages=5,
    )
    all_results.append(r)

    # Scenario 2: Brand-filtered feed (should be faster — exploration disabled)
    orch2 = build_orchestrator(supabase)
    r = run_scenario(
        "Brand Filter (Boohoo)",
        orch2,
        FeedRequest(
            user_id="bench-user-2", mode="explore", page_size=24, debug=True,
            hard_filters=HardFilters(include_brands=["Boohoo"]),
        ),
        pages=5,
    )
    all_results.append(r)

    # Scenario 3: Sale mode
    orch3 = build_orchestrator(supabase)
    r = run_scenario(
        "Sale Mode",
        orch3,
        FeedRequest(user_id="bench-user-3", mode="sale", page_size=24, debug=True),
        pages=3,
    )
    all_results.append(r)

    # Scenario 4: Category + price filter
    orch4 = build_orchestrator(supabase)
    r = run_scenario(
        "Category (dresses) + Price ($20-60)",
        orch4,
        FeedRequest(
            user_id="bench-user-4", mode="explore", page_size=24, debug=True,
            hard_filters=HardFilters(categories=["dresses"], min_price=20, max_price=60),
        ),
        pages=3,
    )
    all_results.append(r)

    # Scenario 5: Larger page size
    orch5 = build_orchestrator(supabase)
    r = run_scenario(
        "Large Page (48 items)",
        orch5,
        FeedRequest(user_id="bench-user-5", mode="explore", page_size=48, debug=True),
        pages=3,
    )
    all_results.append(r)

    # Component benchmark
    run_component_benchmark(supabase)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Scenario':<35s} {'First':>8s} {'Avg Reuse':>10s} {'Pool':>6s}")
    print(f"  {'─'*35} {'─'*8} {'─'*10} {'─'*6}")
    for r in all_results:
        print(
            f"  {r['name']:<35s} {r['first_page_ms']:>7.0f}ms {r['avg_subsequent_ms']:>9.0f}ms {r['pool_size']:>6d}"
        )

    # Pass/fail assessment
    print(f"\n  Targets:")
    print(f"    First page (rebuild): < 4000ms (ideal < 1000ms)")
    print(f"    Subsequent (reuse):   < 500ms  (ideal < 100ms)")

    first_ok = all(r["first_page_ms"] < 4000 for r in all_results)
    reuse_ok = all(r["avg_subsequent_ms"] < 500 for r in all_results if r["avg_subsequent_ms"] > 0)
    print(f"\n  First page target:  {'PASS' if first_ok else 'FAIL'}")
    print(f"  Reuse target:       {'PASS' if reuse_ok else 'FAIL'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
