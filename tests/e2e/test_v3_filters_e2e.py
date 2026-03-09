#!/usr/bin/env python3
"""
V3 Feed E2E Filter Tests — 51 scenarios against real Supabase RPCs.

Tests run against the deployed v3_get_candidates_by_explore_key,
v3_get_candidates_by_freshness, and v3_hydrate_candidates RPCs.

Uses DirectQuerySource (bypasses source classes, calls RPCs directly)
and DirectHydrator (calls v3_hydrate_candidates directly).

Run with:
    source .venv/bin/activate
    PYTHONPATH=src python tests/e2e/test_v3_filters_e2e.py

Expected: 51/51 scenarios pass, 0 brand leaks, avg ~3-4s response time.
"""

import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


# ---------------------------------------------------------------------------
# DirectQuerySource — call RPCs directly without going through source classes
# ---------------------------------------------------------------------------

class DirectQuerySource:
    """
    Bypasses PreferenceSource/SessionSource/etc and calls the RPCs directly.
    Used for E2E testing of filter behavior at the DB level.
    """

    def __init__(self, supabase_client: Any):
        self._supabase = supabase_client

    def query_explore_key(
        self,
        key_family: str = "a",
        gender: Optional[str] = None,
        categories: Optional[List[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        exclude_brands: Optional[List[str]] = None,
        include_brands: Optional[List[str]] = None,
        on_sale_only: bool = False,
        new_arrivals: bool = False,
        new_days: int = 30,
        preferred_brands: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Call v3_get_candidates_by_explore_key RPC directly."""
        params: Dict[str, Any] = {
            "p_key_family": key_family,
            "p_limit": limit,
        }
        if gender:
            params["p_gender"] = gender
        if categories:
            params["p_categories"] = categories
        if min_price is not None:
            params["p_min_price"] = min_price
        if max_price is not None:
            params["p_max_price"] = max_price
        if exclude_brands:
            params["p_exclude_brands"] = exclude_brands
        if include_brands:
            params["p_include_brands"] = include_brands
        if on_sale_only:
            params["p_on_sale_only"] = True
        if new_arrivals:
            params["p_new_arrivals"] = True
            params["p_new_days"] = new_days
        if preferred_brands:
            params["p_preferred_brands"] = preferred_brands

        result = self._supabase.rpc(
            "v3_get_candidates_by_explore_key", params
        ).execute()
        return result.data or []

    def query_freshness(
        self,
        gender: Optional[str] = None,
        categories: Optional[List[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        exclude_brands: Optional[List[str]] = None,
        include_brands: Optional[List[str]] = None,
        on_sale_only: bool = False,
        days: int = 30,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Call v3_get_candidates_by_freshness RPC directly."""
        params: Dict[str, Any] = {
            "p_days": days,
            "p_limit": limit,
        }
        if gender:
            params["p_gender"] = gender
        if categories:
            params["p_categories"] = categories
        if min_price is not None:
            params["p_min_price"] = min_price
        if max_price is not None:
            params["p_max_price"] = max_price
        if exclude_brands:
            params["p_exclude_brands"] = exclude_brands
        if include_brands:
            params["p_include_brands"] = include_brands
        if on_sale_only:
            params["p_on_sale_only"] = True

        result = self._supabase.rpc(
            "v3_get_candidates_by_freshness", params
        ).execute()
        return result.data or []


# ---------------------------------------------------------------------------
# DirectHydrator — call v3_hydrate_candidates directly
# ---------------------------------------------------------------------------

class DirectHydrator:
    """Calls v3_hydrate_candidates RPC directly for E2E testing."""

    def __init__(self, supabase_client: Any):
        self._supabase = supabase_client

    def hydrate(self, item_ids: List[str]) -> List[Dict[str, Any]]:
        """Hydrate a batch of item IDs, returning raw rows."""
        if not item_ids:
            return []
        result = self._supabase.rpc(
            "v3_hydrate_candidates", {"p_ids": item_ids}
        ).execute()
        return result.data or []


# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    name: str
    passed: bool
    duration_ms: float
    detail: str = ""
    brand_leak: bool = False


class E2ETestRunner:
    """Runs all 51 E2E scenarios and tracks results."""

    def __init__(self):
        self.results: List[ScenarioResult] = []
        self.source: Optional[DirectQuerySource] = None
        self.hydrator: Optional[DirectHydrator] = None

    def setup(self):
        """Initialize Supabase client and test helpers."""
        from config.database import get_supabase_client
        supabase = get_supabase_client()
        self.source = DirectQuerySource(supabase)
        self.hydrator = DirectHydrator(supabase)

    def run_scenario(self, name: str, fn):
        """Run a single scenario, tracking timing and pass/fail."""
        t0 = time.time()
        try:
            fn()
            elapsed = (time.time() - t0) * 1000
            self.results.append(ScenarioResult(
                name=name, passed=True, duration_ms=elapsed
            ))
            print(f"  PASS  {name} ({elapsed:.0f}ms)")
        except AssertionError as e:
            elapsed = (time.time() - t0) * 1000
            detail = str(e)[:200]
            brand_leak = "brand" in detail.lower() and "leak" in detail.lower()
            self.results.append(ScenarioResult(
                name=name, passed=False, duration_ms=elapsed,
                detail=detail, brand_leak=brand_leak,
            ))
            print(f"  FAIL  {name} ({elapsed:.0f}ms) — {detail}")
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            self.results.append(ScenarioResult(
                name=name, passed=False, duration_ms=elapsed,
                detail=f"ERROR: {e}"
            ))
            print(f"  ERROR {name} ({elapsed:.0f}ms) — {e}")

    def print_summary(self):
        """Print final summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        brand_leaks = sum(1 for r in self.results if r.brand_leak)
        avg_ms = sum(r.duration_ms for r in self.results) / max(total, 1)

        print("\n" + "=" * 70)
        print(f"E2E Results: {passed}/{total} passed, {failed} failed")
        print(f"Brand leaks: {brand_leaks}")
        print(f"Avg response time: {avg_ms:.1f}ms")
        print("=" * 70)

        if failed > 0:
            print("\nFailed scenarios:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.detail}")

        return failed == 0

    # -----------------------------------------------------------------------
    # Scenario definitions (51 total)
    # -----------------------------------------------------------------------

    def run_all(self):
        """Run all 51 E2E scenarios."""
        print("V3 Feed E2E Filter Tests")
        print("=" * 70)

        # Group 1: Basic RPC connectivity (3 scenarios)
        print("\n--- Group 1: RPC Connectivity ---")
        self.run_scenario("1.1 explore_key RPC returns data", self.test_explore_key_returns_data)
        self.run_scenario("1.2 freshness RPC returns data", self.test_freshness_returns_data)
        self.run_scenario("1.3 hydrate RPC returns data", self.test_hydrate_returns_data)

        # Group 2: Gender filter (3 scenarios)
        print("\n--- Group 2: Gender Filter ---")
        self.run_scenario("2.1 female gender filter", self.test_gender_female)
        self.run_scenario("2.2 male gender filter", self.test_gender_male)
        self.run_scenario("2.3 no gender filter returns more", self.test_gender_none_returns_more)

        # Group 3: Category filter (4 scenarios)
        print("\n--- Group 3: Category Filter ---")
        self.run_scenario("3.1 single category tops", self.test_category_tops)
        self.run_scenario("3.2 single category dresses", self.test_category_dresses)
        self.run_scenario("3.3 multiple categories", self.test_category_multiple)
        self.run_scenario("3.4 category bottoms", self.test_category_bottoms)

        # Group 4: Price filter (5 scenarios)
        print("\n--- Group 4: Price Filter ---")
        self.run_scenario("4.1 min price only", self.test_min_price)
        self.run_scenario("4.2 max price only", self.test_max_price)
        self.run_scenario("4.3 price range 20-50", self.test_price_range_20_50)
        self.run_scenario("4.4 price range 50-100", self.test_price_range_50_100)
        self.run_scenario("4.5 narrow price range", self.test_narrow_price_range)

        # Group 5: Brand exclusion (4 scenarios)
        print("\n--- Group 5: Brand Exclusion ---")
        self.run_scenario("5.1 exclude single brand", self.test_exclude_single_brand)
        self.run_scenario("5.2 exclude multiple brands", self.test_exclude_multiple_brands)
        self.run_scenario("5.3 exclude brand case insensitive", self.test_exclude_brand_case_insensitive)
        self.run_scenario("5.4 exclude brand no leak in freshness", self.test_exclude_brand_freshness)

        # Group 6: Brand inclusion (5 scenarios)
        print("\n--- Group 6: Brand Inclusion ---")
        self.run_scenario("6.1 include single brand", self.test_include_single_brand)
        self.run_scenario("6.2 include multiple brands", self.test_include_multiple_brands)
        self.run_scenario("6.3 include brand no leak", self.test_include_brand_no_leak)
        self.run_scenario("6.4 include brand freshness RPC", self.test_include_brand_freshness)
        self.run_scenario("6.5 include brand case insensitive", self.test_include_brand_case_insensitive)

        # Group 7: Sale filter (3 scenarios)
        print("\n--- Group 7: Sale Filter ---")
        self.run_scenario("7.1 on_sale_only explore_key", self.test_sale_only_explore)
        self.run_scenario("7.2 on_sale_only freshness", self.test_sale_only_freshness)
        self.run_scenario("7.3 sale items have discount", self.test_sale_items_have_discount)

        # Group 8: New arrivals filter (3 scenarios)
        print("\n--- Group 8: New Arrivals ---")
        self.run_scenario("8.1 new_arrivals flag", self.test_new_arrivals)
        self.run_scenario("8.2 new_arrivals with days=7", self.test_new_arrivals_7_days)
        self.run_scenario("8.3 freshness RPC has created_at", self.test_freshness_has_created_at)

        # Group 9: Key family (3 scenarios)
        print("\n--- Group 9: Key Family ---")
        self.run_scenario("9.1 key_family a returns data", self.test_key_family_a)
        self.run_scenario("9.2 key_family b returns data", self.test_key_family_b)
        self.run_scenario("9.3 key_family c returns data", self.test_key_family_c)

        # Group 10: Keyset pagination (3 scenarios)
        print("\n--- Group 10: Keyset Pagination ---")
        self.run_scenario("10.1 pagination first page", self.test_pagination_first_page)
        self.run_scenario("10.2 pagination second page no overlap", self.test_pagination_no_overlap)
        self.run_scenario("10.3 freshness pagination", self.test_freshness_pagination)

        # Group 11: Hydration (5 scenarios)
        print("\n--- Group 11: Hydration ---")
        self.run_scenario("11.1 hydrate single item", self.test_hydrate_single)
        self.run_scenario("11.2 hydrate batch of 24", self.test_hydrate_batch_24)
        self.run_scenario("11.3 hydrate has all MV columns", self.test_hydrate_all_columns)
        self.run_scenario("11.4 hydrate missing ID returns empty", self.test_hydrate_missing_id)
        self.run_scenario("11.5 hydrate empty array", self.test_hydrate_empty)

        # Group 12: Combined filters (5 scenarios)
        print("\n--- Group 12: Combined Filters ---")
        self.run_scenario("12.1 gender + category + price", self.test_combined_gender_category_price)
        self.run_scenario("12.2 include_brands + price range", self.test_combined_brands_price)
        self.run_scenario("12.3 exclude_brands + sale_only", self.test_combined_exclude_sale)
        self.run_scenario("12.4 category + exclude_brands + price", self.test_combined_category_exclude_price)
        self.run_scenario("12.5 all filters combined", self.test_combined_all_filters)

        # Group 13: Edge cases (5 scenarios)
        print("\n--- Group 13: Edge Cases ---")
        self.run_scenario("13.1 limit=1 returns exactly 1", self.test_limit_1)
        self.run_scenario("13.2 limit=200 returns up to 200", self.test_limit_200)
        self.run_scenario("13.3 impossible filter returns empty", self.test_impossible_filter)
        self.run_scenario("13.4 explore_key scores are ordered desc", self.test_scores_ordered_desc)
        self.run_scenario("13.5 freshness dates are ordered desc", self.test_freshness_dates_ordered)

        return self.print_summary()

    # -----------------------------------------------------------------------
    # Group 1: RPC Connectivity
    # -----------------------------------------------------------------------

    def test_explore_key_returns_data(self):
        rows = self.source.query_explore_key(limit=10)
        assert len(rows) > 0, "explore_key RPC returned no data"
        assert "id" in rows[0], "Missing 'id' column"
        assert "brand" in rows[0], "Missing 'brand' column"

    def test_freshness_returns_data(self):
        rows = self.source.query_freshness(days=90, limit=10)
        assert len(rows) > 0, "freshness RPC returned no data"
        assert "id" in rows[0], "Missing 'id' column"
        assert "created_at" in rows[0], "Missing 'created_at' column"

    def test_hydrate_returns_data(self):
        # Get some IDs first
        rows = self.source.query_explore_key(limit=3)
        assert len(rows) > 0, "Need IDs to hydrate"
        ids = [str(r["id"]) for r in rows]
        hydrated = self.hydrator.hydrate(ids)
        assert len(hydrated) > 0, "hydrate returned no data"
        assert "name" in hydrated[0], "Missing 'name' in hydrated row"

    # -----------------------------------------------------------------------
    # Group 2: Gender Filter
    # -----------------------------------------------------------------------

    def test_gender_female(self):
        rows = self.source.query_explore_key(gender="female", limit=20)
        assert len(rows) > 0, "No female results"

    def test_gender_male(self):
        rows = self.source.query_explore_key(gender="male", limit=20)
        # Male may have few or no results; just ensure no error
        assert isinstance(rows, list), "Expected list result"

    def test_gender_none_returns_more(self):
        female = self.source.query_explore_key(gender="female", limit=50)
        no_gender = self.source.query_explore_key(limit=50)
        assert len(no_gender) >= len(female), (
            f"No-gender ({len(no_gender)}) should be >= female-only ({len(female)})"
        )

    # -----------------------------------------------------------------------
    # Group 3: Category Filter
    # -----------------------------------------------------------------------

    def test_category_tops(self):
        rows = self.source.query_explore_key(categories=["tops"], limit=20)
        assert len(rows) > 0, "No tops results"
        # Hydrate to verify category
        ids = [str(r["id"]) for r in rows[:5]]
        hydrated = self.hydrator.hydrate(ids)
        for h in hydrated:
            cat = (h.get("category") or "").lower()
            broad = (h.get("broad_category") or "").lower()
            assert "top" in cat or "top" in broad or broad == "" or cat == "", (
                f"Item {h['id']} category={cat} broad={broad} not tops"
            )

    def test_category_dresses(self):
        rows = self.source.query_explore_key(categories=["dresses"], limit=20)
        assert len(rows) > 0, "No dresses results"

    def test_category_multiple(self):
        rows = self.source.query_explore_key(
            categories=["tops", "dresses"], limit=30
        )
        assert len(rows) > 0, "No results for tops+dresses"

    def test_category_bottoms(self):
        rows = self.source.query_explore_key(categories=["bottoms"], limit=20)
        assert len(rows) > 0, "No bottoms results"

    # -----------------------------------------------------------------------
    # Group 4: Price Filter
    # -----------------------------------------------------------------------

    def test_min_price(self):
        rows = self.source.query_explore_key(min_price=50, limit=20)
        assert len(rows) > 0, "No results with min_price=50"
        for r in rows:
            assert float(r["price"]) >= 50, (
                f"Item {r['id']} price={r['price']} below min_price=50"
            )

    def test_max_price(self):
        rows = self.source.query_explore_key(max_price=30, limit=20)
        assert len(rows) > 0, "No results with max_price=30"
        for r in rows:
            assert float(r["price"]) <= 30, (
                f"Item {r['id']} price={r['price']} above max_price=30"
            )

    def test_price_range_20_50(self):
        rows = self.source.query_explore_key(
            min_price=20, max_price=50, limit=20
        )
        assert len(rows) > 0, "No results in 20-50 range"
        for r in rows:
            p = float(r["price"])
            assert 20 <= p <= 50, f"Item {r['id']} price={p} outside 20-50"

    def test_price_range_50_100(self):
        rows = self.source.query_explore_key(
            min_price=50, max_price=100, limit=20
        )
        assert len(rows) > 0, "No results in 50-100 range"
        for r in rows:
            p = float(r["price"])
            assert 50 <= p <= 100, f"Item {r['id']} price={p} outside 50-100"

    def test_narrow_price_range(self):
        rows = self.source.query_explore_key(
            min_price=25, max_price=26, limit=10
        )
        # May or may not have results, but should not error
        for r in rows:
            p = float(r["price"])
            assert 25 <= p <= 26, f"Item {r['id']} price={p} outside 25-26"

    # -----------------------------------------------------------------------
    # Group 5: Brand Exclusion
    # -----------------------------------------------------------------------

    def test_exclude_single_brand(self):
        rows = self.source.query_explore_key(
            exclude_brands=["Boohoo"], limit=50
        )
        assert len(rows) > 0, "No results excluding Boohoo"
        for r in rows:
            assert (r.get("brand") or "").lower() != "boohoo", (
                f"Brand leak: {r['brand']} should be excluded"
            )

    def test_exclude_multiple_brands(self):
        excluded = ["Boohoo", "Missguided", "Forever 21"]
        rows = self.source.query_explore_key(
            exclude_brands=excluded, limit=50
        )
        assert len(rows) > 0, "No results excluding 3 brands"
        excluded_lower = {b.lower() for b in excluded}
        for r in rows:
            brand = (r.get("brand") or "").lower()
            assert brand not in excluded_lower, (
                f"Brand leak: {r['brand']} should be excluded"
            )

    def test_exclude_brand_case_insensitive(self):
        rows = self.source.query_explore_key(
            exclude_brands=["boohoo"], limit=50  # lowercase
        )
        assert len(rows) > 0, "No results"
        for r in rows:
            assert (r.get("brand") or "").lower() != "boohoo", (
                f"Brand leak (case): {r['brand']}"
            )

    def test_exclude_brand_freshness(self):
        rows = self.source.query_freshness(
            exclude_brands=["Boohoo"], days=90, limit=50
        )
        for r in rows:
            assert (r.get("brand") or "").lower() != "boohoo", (
                f"Brand leak in freshness: {r['brand']}"
            )

    # -----------------------------------------------------------------------
    # Group 6: Brand Inclusion
    # -----------------------------------------------------------------------

    def test_include_single_brand(self):
        rows = self.source.query_explore_key(
            include_brands=["Boohoo"], limit=50
        )
        assert len(rows) > 0, "No Boohoo results"
        for r in rows:
            assert (r.get("brand") or "").lower() == "boohoo", (
                f"Brand leak: got {r['brand']} instead of Boohoo"
            )

    def test_include_multiple_brands(self):
        included = ["Boohoo", "Missguided"]
        rows = self.source.query_explore_key(
            include_brands=included, limit=50
        )
        assert len(rows) > 0, "No results for Boohoo+Missguided"
        included_lower = {b.lower() for b in included}
        for r in rows:
            brand = (r.get("brand") or "").lower()
            assert brand in included_lower, (
                f"Brand leak: got {r['brand']}, expected one of {included}"
            )

    def test_include_brand_no_leak(self):
        rows = self.source.query_explore_key(
            include_brands=["Princess Polly"], limit=100
        )
        assert len(rows) > 0, "No Princess Polly results"
        brands = {(r.get("brand") or "").lower() for r in rows}
        assert brands <= {"princess polly"}, (
            f"Brand leak: got brands {brands}"
        )

    def test_include_brand_freshness(self):
        rows = self.source.query_freshness(
            include_brands=["Boohoo"], days=90, limit=50
        )
        for r in rows:
            assert (r.get("brand") or "").lower() == "boohoo", (
                f"Brand leak in freshness: {r['brand']}"
            )

    def test_include_brand_case_insensitive(self):
        rows = self.source.query_explore_key(
            include_brands=["boohoo"], limit=50  # lowercase
        )
        assert len(rows) > 0, "No results for lowercase boohoo"
        for r in rows:
            assert (r.get("brand") or "").lower() == "boohoo", (
                f"Brand leak (case): {r['brand']}"
            )

    # -----------------------------------------------------------------------
    # Group 7: Sale Filter
    # -----------------------------------------------------------------------

    def test_sale_only_explore(self):
        rows = self.source.query_explore_key(on_sale_only=True, limit=30)
        assert len(rows) > 0, "No sale items via explore_key"

    def test_sale_only_freshness(self):
        rows = self.source.query_freshness(on_sale_only=True, days=90, limit=30)
        assert len(rows) > 0, "No sale items via freshness"

    def test_sale_items_have_discount(self):
        rows = self.source.query_explore_key(on_sale_only=True, limit=10)
        assert len(rows) > 0, "No sale items"
        ids = [str(r["id"]) for r in rows[:5]]
        hydrated = self.hydrator.hydrate(ids)
        for h in hydrated:
            assert h.get("is_on_sale") is True, (
                f"Item {h['id']} on_sale_only but is_on_sale={h.get('is_on_sale')}"
            )
            orig = h.get("original_price")
            price = h.get("price")
            if orig and price:
                assert float(orig) > float(price), (
                    f"Item {h['id']} original_price={orig} not > price={price}"
                )

    # -----------------------------------------------------------------------
    # Group 8: New Arrivals
    # -----------------------------------------------------------------------

    def test_new_arrivals(self):
        rows = self.source.query_explore_key(
            new_arrivals=True, new_days=30, limit=20
        )
        # May return data or not depending on catalog freshness
        assert isinstance(rows, list)

    def test_new_arrivals_7_days(self):
        rows = self.source.query_explore_key(
            new_arrivals=True, new_days=7, limit=20
        )
        assert isinstance(rows, list)

    def test_freshness_has_created_at(self):
        rows = self.source.query_freshness(days=90, limit=5)
        if rows:
            assert "created_at" in rows[0], "Freshness RPC missing created_at"

    # -----------------------------------------------------------------------
    # Group 9: Key Family
    # -----------------------------------------------------------------------

    def test_key_family_a(self):
        rows = self.source.query_explore_key(key_family="a", limit=10)
        assert len(rows) > 0, "Key family a returned no data"
        assert rows[0].get("explore_score") is not None, "Missing explore_score"

    def test_key_family_b(self):
        rows = self.source.query_explore_key(key_family="b", limit=10)
        assert len(rows) > 0, "Key family b returned no data"

    def test_key_family_c(self):
        rows = self.source.query_explore_key(key_family="c", limit=10)
        assert len(rows) > 0, "Key family c returned no data"

    # -----------------------------------------------------------------------
    # Group 10: Keyset Pagination
    # -----------------------------------------------------------------------

    def test_pagination_first_page(self):
        rows = self.source.query_explore_key(limit=10)
        assert len(rows) == 10, f"Expected 10 rows, got {len(rows)}"

    def test_pagination_no_overlap(self):
        page1 = self.source.query_explore_key(key_family="a", limit=10)
        assert len(page1) == 10, "Need 10 items for page 1"

        # Use last item as cursor
        cursor_score = page1[-1]["explore_score"]
        cursor_id = page1[-1]["id"]

        params = {
            "p_key_family": "a",
            "p_cursor_score": cursor_score,
            "p_cursor_id": cursor_id,
            "p_limit": 10,
        }
        result = self.source._supabase.rpc(
            "v3_get_candidates_by_explore_key", params
        ).execute()
        page2 = result.data or []

        if page2:
            page1_ids = {str(r["id"]) for r in page1}
            page2_ids = {str(r["id"]) for r in page2}
            overlap = page1_ids & page2_ids
            assert len(overlap) == 0, (
                f"Page overlap: {len(overlap)} items appear in both pages"
            )

    def test_freshness_pagination(self):
        page1 = self.source.query_freshness(days=90, limit=10)
        assert len(page1) > 0, "Need items for freshness pagination"

        if len(page1) == 10:
            cursor_date = page1[-1]["created_at"]
            cursor_id = page1[-1]["id"]
            params = {
                "p_days": 90,
                "p_cursor_date": cursor_date,
                "p_cursor_id": cursor_id,
                "p_limit": 10,
            }
            result = self.source._supabase.rpc(
                "v3_get_candidates_by_freshness", params
            ).execute()
            page2 = result.data or []
            if page2:
                page1_ids = {str(r["id"]) for r in page1}
                page2_ids = {str(r["id"]) for r in page2}
                overlap = page1_ids & page2_ids
                assert len(overlap) == 0, (
                    f"Freshness page overlap: {len(overlap)} items"
                )

    # -----------------------------------------------------------------------
    # Group 11: Hydration
    # -----------------------------------------------------------------------

    def test_hydrate_single(self):
        rows = self.source.query_explore_key(limit=1)
        assert len(rows) == 1
        hydrated = self.hydrator.hydrate([str(rows[0]["id"])])
        assert len(hydrated) == 1, "Single hydration failed"

    def test_hydrate_batch_24(self):
        rows = self.source.query_explore_key(limit=24)
        assert len(rows) == 24, f"Need 24 items, got {len(rows)}"
        ids = [str(r["id"]) for r in rows]
        hydrated = self.hydrator.hydrate(ids)
        assert len(hydrated) == 24, (
            f"Hydrated {len(hydrated)}/24 items"
        )

    def test_hydrate_all_columns(self):
        rows = self.source.query_explore_key(limit=1)
        ids = [str(rows[0]["id"])]
        hydrated = self.hydrator.hydrate(ids)
        assert len(hydrated) == 1
        h = hydrated[0]

        expected_columns = [
            "id", "name", "brand", "category", "broad_category",
            "article_type", "colors", "materials", "price",
            "primary_image_url", "gender", "in_stock",
            "is_on_sale", "is_new",
            "pa_pattern", "pa_formality", "pa_color_family",
            "image_dedup_key",
        ]
        for col in expected_columns:
            assert col in h, f"Missing column '{col}' in hydrated row"

    def test_hydrate_missing_id(self):
        hydrated = self.hydrator.hydrate(["00000000-0000-0000-0000-000000000000"])
        assert len(hydrated) == 0, (
            f"Expected empty for fake ID, got {len(hydrated)}"
        )

    def test_hydrate_empty(self):
        hydrated = self.hydrator.hydrate([])
        assert len(hydrated) == 0, "Expected empty for empty input"

    # -----------------------------------------------------------------------
    # Group 12: Combined Filters
    # -----------------------------------------------------------------------

    def test_combined_gender_category_price(self):
        rows = self.source.query_explore_key(
            gender="female", categories=["dresses"],
            min_price=20, max_price=80, limit=20,
        )
        assert len(rows) > 0, "No female dresses in 20-80 range"
        for r in rows:
            p = float(r["price"])
            assert 20 <= p <= 80, f"Price {p} outside 20-80"

    def test_combined_brands_price(self):
        rows = self.source.query_explore_key(
            include_brands=["Boohoo"], min_price=10, max_price=40, limit=20,
        )
        assert len(rows) > 0, "No Boohoo items in 10-40 range"
        for r in rows:
            assert (r.get("brand") or "").lower() == "boohoo"
            p = float(r["price"])
            assert 10 <= p <= 40, f"Price {p} outside 10-40"

    def test_combined_exclude_sale(self):
        rows = self.source.query_explore_key(
            exclude_brands=["Boohoo"], on_sale_only=True, limit=30,
        )
        for r in rows:
            assert (r.get("brand") or "").lower() != "boohoo", (
                f"Brand leak: {r['brand']}"
            )

    def test_combined_category_exclude_price(self):
        rows = self.source.query_explore_key(
            categories=["tops"], exclude_brands=["Forever 21"],
            min_price=15, max_price=60, limit=20,
        )
        for r in rows:
            assert (r.get("brand") or "").lower() != "forever 21", (
                f"Brand leak: {r['brand']}"
            )
            p = float(r["price"])
            assert 15 <= p <= 60, f"Price {p} outside 15-60"

    def test_combined_all_filters(self):
        rows = self.source.query_explore_key(
            gender="female",
            categories=["tops", "dresses"],
            min_price=15,
            max_price=80,
            exclude_brands=["Missguided"],
            on_sale_only=False,
            limit=30,
        )
        assert len(rows) > 0, "No results with all filters"
        for r in rows:
            assert (r.get("brand") or "").lower() != "missguided"
            p = float(r["price"])
            assert 15 <= p <= 80

    # -----------------------------------------------------------------------
    # Group 13: Edge Cases
    # -----------------------------------------------------------------------

    def test_limit_1(self):
        rows = self.source.query_explore_key(limit=1)
        assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"

    def test_limit_200(self):
        rows = self.source.query_explore_key(limit=200)
        assert len(rows) <= 200, f"Got more than 200: {len(rows)}"
        assert len(rows) > 50, f"Expected many rows, got {len(rows)}"

    def test_impossible_filter(self):
        rows = self.source.query_explore_key(
            min_price=999999, max_price=999999, limit=10
        )
        assert len(rows) == 0, (
            f"Expected 0 results for impossible price, got {len(rows)}"
        )

    def test_scores_ordered_desc(self):
        rows = self.source.query_explore_key(key_family="a", limit=20)
        assert len(rows) > 1, "Need multiple rows to check ordering"
        scores = [r["explore_score"] for r in rows]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Scores not descending: {scores[i]} < {scores[i+1]} at position {i}"
            )

    def test_freshness_dates_ordered(self):
        rows = self.source.query_freshness(days=90, limit=20)
        if len(rows) > 1:
            dates = [r["created_at"] for r in rows]
            for i in range(len(dates) - 1):
                assert dates[i] >= dates[i + 1], (
                    f"Dates not descending: {dates[i]} < {dates[i+1]} at position {i}"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    runner = E2ETestRunner()

    print("Connecting to Supabase...")
    try:
        runner.setup()
        print("Connected.\n")
    except Exception as e:
        print(f"FATAL: Could not connect to Supabase: {e}")
        print("Ensure SUPABASE_URL and SUPABASE_KEY are set in .env")
        sys.exit(1)

    success = runner.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
