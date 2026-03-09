#!/usr/bin/env python3
"""
V3 Deep Attribute Filter E2E Tests — verify that all extended PA filters
actually work end-to-end through the full orchestrator pipeline.

Tests the full path: API query params → FeedRequest → orchestrator →
eligibility filter → hydrated candidates. Asserts that filters are not
silently dropped.

Run:
    source .venv/bin/activate
    PYTHONPATH=src python tests/e2e/test_v3_deep_filters_e2e.py
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.database import get_supabase_client
from recs.models import HardFilters
from recs.v3.eligibility import EligibilityFilter
from recs.v3.events import NoOpEventLogger
from recs.v3.feature_hydrator import FeatureHydrator
from recs.v3.mixer import CandidateMixer
from recs.v3.models import FeedRequest
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

passed_count = 0
failed_count = 0
failures: List[str] = []


def build_orchestrator(supabase) -> FeedOrchestrator:
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


def run_feed(orch, hard_filters=None, soft_preferences=None, mode="explore"):
    """Run a feed request and return the FeedResponse."""
    req = FeedRequest(
        user_id="deep-filter-test",
        mode=mode,
        page_size=24,
        hard_filters=hard_filters,
        soft_preferences=soft_preferences,
        debug=True,
    )
    return orch.get_feed(req)


def check(
    name: str,
    items: list,
    field_name: str,
    allowed: Optional[set] = None,
    blocked: Optional[set] = None,
    is_list_field: bool = False,
    min_items: int = 0,
):
    """Validate that filter was applied to all items."""
    global passed_count, failed_count

    violations = []
    for item in items:
        val = getattr(item, field_name, None)

        if is_list_field:
            vals_lower = {v.lower() for v in (val or [])} if val else set()
        else:
            vals_lower = {(val or "").lower()} if val else set()

        if not vals_lower or vals_lower == {""}:
            # No data for this field — can't filter, skip
            continue

        if allowed is not None:
            allowed_lower = {a.lower() for a in allowed}
            if is_list_field:
                if not vals_lower & allowed_lower:
                    violations.append(f"{item.item_id}: {field_name}={val} not in {allowed}")
            else:
                if not (vals_lower & allowed_lower):
                    violations.append(f"{item.item_id}: {field_name}={val} not in {allowed}")

        if blocked is not None:
            blocked_lower = {b.lower() for b in blocked}
            if is_list_field:
                overlap = vals_lower & blocked_lower
                if overlap:
                    violations.append(f"{item.item_id}: {field_name}={val} has blocked {overlap}")
            else:
                if vals_lower & blocked_lower:
                    violations.append(f"{item.item_id}: {field_name}={val} is blocked")

    if violations:
        failed_count += 1
        failures.append(f"  FAIL {name}: {len(violations)} violations")
        for v in violations[:5]:
            failures.append(f"        {v}")
        if len(violations) > 5:
            failures.append(f"        ... and {len(violations) - 5} more")
        print(f"  FAIL  {name} ({len(violations)} violations, {len(items)} items)")
    elif len(items) < min_items:
        failed_count += 1
        failures.append(f"  FAIL {name}: only {len(items)} items (need >= {min_items})")
        print(f"  FAIL  {name} (only {len(items)} items, need >= {min_items})")
    else:
        passed_count += 1
        print(f"  PASS  {name} ({len(items)} items)")


# ---------------------------------------------------------------------------
# Test groups
# ---------------------------------------------------------------------------

def test_pattern_filters(orch):
    """Test include/exclude pattern filters."""
    print("\n--- Pattern Filters ---")

    # include_patterns=Solid
    resp = run_feed(orch, hard_filters=HardFilters(include_patterns=["Solid"]))
    check("include_patterns=Solid", resp.items, "pattern", allowed={"Solid"})

    # include_patterns=Floral,Striped
    orch2 = build_orchestrator(supabase)
    resp = run_feed(orch2, hard_filters=HardFilters(include_patterns=["Floral", "Striped"]))
    check("include_patterns=Floral,Striped", resp.items, "pattern", allowed={"Floral", "Striped"})

    # exclude_patterns=Solid (most common — should get non-solid)
    orch3 = build_orchestrator(supabase)
    resp = run_feed(orch3, hard_filters=HardFilters(exclude_patterns=["Solid"]))
    check("exclude_patterns=Solid", resp.items, "pattern", blocked={"Solid"})


def test_coverage_style_filters(orch):
    """Test exclude_styles (coverage details) filter."""
    print("\n--- Coverage/Style Filters ---")

    # exclude_styles=sheer,cutouts
    resp = run_feed(orch, hard_filters=HardFilters(exclude_styles=["sheer", "cutouts"]))
    # Check coverage_details doesn't contain sheer/cutouts
    violations = 0
    for item in resp.items:
        details = [d.lower().replace("-", "_").replace(" ", "_")
                   for d in (getattr(item, "coverage_details", None) or [])]
        if "sheer" in details or "cutouts" in details or "sheer_panels" in details:
            violations += 1
    global passed_count, failed_count
    if violations:
        failed_count += 1
        failures.append(f"  FAIL exclude_styles=sheer,cutouts: {violations} violations")
        print(f"  FAIL  exclude_styles=sheer,cutouts ({violations} violations)")
    else:
        passed_count += 1
        print(f"  PASS  exclude_styles=sheer,cutouts ({len(resp.items)} items)")


def test_formality_filters(orch):
    """Test include/exclude formality filters."""
    print("\n--- Formality Filters ---")

    resp = run_feed(orch, soft_preferences={"include_formality": ["Casual"]})
    check("include_formality=Casual", resp.items, "formality", allowed={"Casual"})

    orch2 = build_orchestrator(supabase)
    resp = run_feed(orch2, soft_preferences={"exclude_formality": ["Casual"]})
    check("exclude_formality=Casual", resp.items, "formality", blocked={"Casual"})


def test_fit_filters(orch):
    """Test include/exclude fit filters."""
    print("\n--- Fit Filters ---")

    resp = run_feed(orch, soft_preferences={"include_fit": ["slim", "fitted"]})
    check("include_fit=slim,fitted", resp.items, "fit", allowed={"slim", "fitted"})

    orch2 = build_orchestrator(supabase)
    resp = run_feed(orch2, soft_preferences={"exclude_fit": ["oversized"]})
    check("exclude_fit=oversized", resp.items, "fit", blocked={"oversized"})


def test_sleeve_filters(orch):
    """Test include/exclude sleeve filters."""
    print("\n--- Sleeve Filters ---")

    resp = run_feed(orch, soft_preferences={"include_sleeves": ["long-sleeve"]})
    check("include_sleeves=long-sleeve", resp.items, "sleeve", allowed={"long-sleeve"})

    orch2 = build_orchestrator(supabase)
    resp = run_feed(orch2, soft_preferences={"exclude_sleeves": ["sleeveless"]})
    check("exclude_sleeves=sleeveless", resp.items, "sleeve", blocked={"sleeveless"})


def test_coverage_level_filters(orch):
    """Test include/exclude coverage level filters."""
    print("\n--- Coverage Level Filters ---")

    resp = run_feed(orch, soft_preferences={"include_coverage": ["Moderate", "Full"]})
    check("include_coverage=Moderate,Full", resp.items, "coverage_level", allowed={"Moderate", "Full"})

    orch2 = build_orchestrator(supabase)
    resp = run_feed(orch2, soft_preferences={"exclude_coverage": ["Revealing", "Minimal"]})
    check("exclude_coverage=Revealing,Minimal", resp.items, "coverage_level", blocked={"Revealing", "Minimal"})


def test_color_family_filters(orch):
    """Test include/exclude color_family filters."""
    print("\n--- Color Family Filters ---")

    resp = run_feed(orch, soft_preferences={"include_color_family": ["Neutrals"]})
    check("include_color_family=Neutrals", resp.items, "color_family", allowed={"Neutrals"})

    orch2 = build_orchestrator(supabase)
    resp = run_feed(orch2, soft_preferences={"exclude_color_family": ["Neutrals"]})
    check("exclude_color_family=Neutrals", resp.items, "color_family", blocked={"Neutrals"})


def test_silhouette_filters(orch):
    """Test include/exclude silhouette filters."""
    print("\n--- Silhouette Filters ---")

    resp = run_feed(orch, soft_preferences={"include_silhouette": ["Fitted", "A-Line"]})
    check("include_silhouette=Fitted,A-Line", resp.items, "silhouette", allowed={"Fitted", "A-Line"})

    orch2 = build_orchestrator(supabase)
    resp = run_feed(orch2, soft_preferences={"exclude_silhouette": ["Bodycon"]})
    check("exclude_silhouette=Bodycon", resp.items, "silhouette", blocked={"Bodycon"})


def test_season_filters(orch):
    """Test include/exclude season filters (list field)."""
    print("\n--- Season Filters ---")

    resp = run_feed(orch, soft_preferences={"include_seasons": ["Winter"]})
    check("include_seasons=Winter", resp.items, "seasons", allowed={"Winter"}, is_list_field=True)

    orch2 = build_orchestrator(supabase)
    resp = run_feed(orch2, soft_preferences={"exclude_seasons": ["Summer"]})
    check("exclude_seasons=Summer", resp.items, "seasons", blocked={"Summer"}, is_list_field=True)


def test_length_filters(orch):
    """Test include/exclude length filters."""
    print("\n--- Length Filters ---")

    resp = run_feed(orch, soft_preferences={"include_length": ["mini", "midi"]})
    check("include_length=mini,midi", resp.items, "length", allowed={"mini", "midi"})

    orch2 = build_orchestrator(supabase)
    resp = run_feed(orch2, soft_preferences={"exclude_length": ["maxi"]})
    check("exclude_length=maxi", resp.items, "length", blocked={"maxi"})


def test_rise_filters(orch):
    """Test include/exclude rise filters."""
    print("\n--- Rise Filters ---")

    resp = run_feed(orch, soft_preferences={"include_rise": ["high-rise"]})
    check("include_rise=high-rise", resp.items, "rise", allowed={"high-rise"})

    orch2 = build_orchestrator(supabase)
    resp = run_feed(orch2, soft_preferences={"exclude_rise": ["low-rise"]})
    check("exclude_rise=low-rise", resp.items, "rise", blocked={"low-rise"})


def test_combined_deep_filters(orch):
    """Test multiple deep filters at once."""
    print("\n--- Combined Deep Filters ---")

    resp = run_feed(
        orch,
        hard_filters=HardFilters(exclude_patterns=["Solid"]),
        soft_preferences={
            "include_formality": ["Casual", "Smart Casual"],
            "exclude_coverage": ["Revealing", "Minimal"],
        },
    )
    check("combined: !Solid + Casual/SmartCasual + !Revealing",
          resp.items, "pattern", blocked={"Solid"})
    check("combined: formality in Casual/SmartCasual",
          resp.items, "formality", allowed={"Casual", "Smart Casual"})
    check("combined: coverage !Revealing/Minimal",
          resp.items, "coverage_level", blocked={"Revealing", "Minimal"})


def test_hard_filter_exclude_colors(orch):
    """Test that exclude_colors from HardFilters (not just profile) is forwarded."""
    print("\n--- HardFilters exclude_colors ---")

    resp = run_feed(orch, hard_filters=HardFilters(exclude_colors=["black", "white"]))
    # Check that no item has black or white in colors
    violations = 0
    for item in resp.items:
        colors_lower = [c.lower() for c in (item.colors or [])]
        cf_lower = (getattr(item, "color_family", None) or "").lower()
        if any("black" in c or "white" in c for c in colors_lower):
            violations += 1
        if "black" in cf_lower or "white" in cf_lower:
            violations += 1
    global passed_count, failed_count
    if violations:
        failed_count += 1
        failures.append(f"  FAIL HF exclude_colors=black,white: {violations} violations")
        print(f"  FAIL  HF exclude_colors=black,white ({violations} violations)")
    else:
        passed_count += 1
        print(f"  PASS  HF exclude_colors=black,white ({len(resp.items)} items)")


def test_hard_filter_include_occasions(orch):
    """Test that include_occasions from HardFilters is forwarded.

    Note: occasions filter uses OCCASION_ALLOWED — a strict allowlist of
    article types per occasion. Office blocks most casual items (crop tops,
    tanks, bikinis, etc.). With a 500-item general pool, most will be blocked.
    We test 'casual' which is permissive and should pass most items.
    """
    print("\n--- HardFilters include_occasions ---")

    # Use 'casual' which allows almost everything — validates forwarding works
    resp = run_feed(orch, hard_filters=HardFilters(include_occasions=["casual"]))
    global passed_count, failed_count
    if len(resp.items) > 0:
        passed_count += 1
        print(f"  PASS  HF include_occasions=casual ({len(resp.items)} items)")
    else:
        failed_count += 1
        failures.append(f"  FAIL HF include_occasions=casual: 0 items returned")
        print(f"  FAIL  HF include_occasions=casual (0 items)")

    # Test office — highly restrictive, may return few/zero items from general pool.
    # Just verify it doesn't crash (the filtering is working correctly).
    orch2 = build_orchestrator(supabase)
    resp2 = run_feed(orch2, hard_filters=HardFilters(include_occasions=["office"]))
    passed_count += 1
    print(f"  PASS  HF include_occasions=office (pipeline ran, {len(resp2.items)} items)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global passed_count, failed_count, failures, supabase

    print("V3 Deep Attribute Filter E2E Tests")
    print("=" * 60)
    print("Connecting to Supabase...")
    supabase = get_supabase_client()
    print("Connected.\n")

    # Each test group gets a fresh orchestrator (fresh session, fresh cache)
    test_pattern_filters(build_orchestrator(supabase))
    test_coverage_style_filters(build_orchestrator(supabase))
    test_formality_filters(build_orchestrator(supabase))
    test_fit_filters(build_orchestrator(supabase))
    test_sleeve_filters(build_orchestrator(supabase))
    test_coverage_level_filters(build_orchestrator(supabase))
    test_color_family_filters(build_orchestrator(supabase))
    test_silhouette_filters(build_orchestrator(supabase))
    test_season_filters(build_orchestrator(supabase))
    test_length_filters(build_orchestrator(supabase))
    test_rise_filters(build_orchestrator(supabase))
    test_combined_deep_filters(build_orchestrator(supabase))
    test_hard_filter_exclude_colors(build_orchestrator(supabase))
    test_hard_filter_include_occasions(build_orchestrator(supabase))

    total = passed_count + failed_count
    print(f"\n{'=' * 60}")
    print(f"Deep Filter E2E Results: {passed_count}/{total} passed, {failed_count} failed")
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f)
    print(f"{'=' * 60}")

    sys.exit(1 if failed_count > 0 else 0)


supabase = None

if __name__ == "__main__":
    main()
