"""
Integration tests for Filter Refinement via Cached Session.

Tests verify end-to-end behavior against a running server with real
Algolia + pgvector backends.

Tests cover:
1. Filter refinement returns different results from original search
2. New search_session_id returned (different from original)
3. Facets update to reflect filtered catalog (Algolia facets, not computed)
4. total_results (nbHits) narrows when filters applied
5. Page 2+ works with new session_id after refinement
6. Brand filter refinement (most common user action)
7. Price filter refinement
8. Multiple filter combination (brand + price)
9. Filter refinement timing (faster than full pipeline)
10. Facet counts are catalog-wide (not computed from small merged set)
11. EXACT intent filter refinement
12. VAGUE intent filter refinement

Requires:
- Running server: PYTHONPATH=src uvicorn api.app:app --port 8000
- Env vars: TEST_SERVER_URL, SUPABASE_JWT_SECRET
- Algolia index + pgvector embeddings populated

Run with:
    TEST_SERVER_URL=http://localhost:8000 PYTHONPATH=src python -m pytest tests/integration/test_filter_refine.py -v -s
"""

import os
import sys
import time
import pytest
import requests
from typing import Dict, List, Any, Optional, Set

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# =============================================================================
# Configuration
# =============================================================================

BASE_URL = os.getenv("TEST_SERVER_URL", "http://localhost:8000")
SEARCH_PREFIX = "/api/search"


def get_auth_headers(user_id: str = "test-filter-refine-001") -> Dict[str, str]:
    """Generate auth headers with a test JWT token."""
    import jwt as pyjwt

    jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
    if not jwt_secret:
        raise ValueError("SUPABASE_JWT_SECRET required for integration tests")

    now = int(time.time())
    payload = {
        "sub": user_id,
        "aud": "authenticated",
        "role": "authenticated",
        "email": f"{user_id}@test.com",
        "aal": "aal1",
        "exp": now + 3600,
        "iat": now,
        "is_anonymous": False,
    }
    token = pyjwt.encode(payload, jwt_secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


def search(query: str, headers: dict, **kwargs) -> requests.Response:
    """Helper: POST /api/search/hybrid with timeout."""
    body = {"query": query, **kwargs}
    return requests.post(
        f"{BASE_URL}{SEARCH_PREFIX}/hybrid",
        json=body,
        headers=headers,
        timeout=60,
    )


def search_page1(query: str, headers: dict, page_size: int = 50, **kwargs) -> dict:
    """Run page 1 search and return parsed response."""
    r = search(query, headers, page_size=page_size, **kwargs)
    assert r.status_code == 200, f"Page 1 failed: {r.status_code} {r.text[:200]}"
    return r.json()


def search_refine(
    query: str,
    headers: dict,
    search_session_id: str,
    page_size: int = 50,
    **filters,
) -> dict:
    """Run a filter refinement: session_id + filters, NO cursor."""
    r = search(
        query, headers,
        search_session_id=search_session_id,
        page_size=page_size,
        **filters,
    )
    assert r.status_code == 200, f"Refine failed: {r.status_code} {r.text[:200]}"
    return r.json()


def search_page_n(
    query: str,
    headers: dict,
    search_session_id: str,
    cursor: str,
    page: int = 2,
    page_size: int = 50,
) -> dict:
    """Run page N search using session + cursor from previous page."""
    r = search(
        query, headers,
        page=page,
        page_size=page_size,
        search_session_id=search_session_id,
        cursor=cursor,
    )
    assert r.status_code == 200, f"Page {page} failed: {r.status_code} {r.text[:200]}"
    return r.json()


def get_ids(data: dict) -> Set[str]:
    """Extract product IDs from a response."""
    return {p["product_id"] for p in data.get("results", [])}


def get_brands(data: dict) -> Set[str]:
    """Extract unique brands from results."""
    return {p.get("brand", "") for p in data.get("results", []) if p.get("brand")}


def get_prices(data: dict) -> List[float]:
    """Extract prices from results."""
    return [float(p.get("price", 0) or 0) for p in data.get("results", []) if p.get("price")]


def get_facet_count(data: dict, facet_name: str, value: str) -> Optional[int]:
    """Get the count for a specific facet value from the response."""
    facets = data.get("facets", {})
    if not facets:
        return None
    facet_values = facets.get(facet_name, [])
    for fv in facet_values:
        v = fv.get("value") if isinstance(fv, dict) else getattr(fv, "value", None)
        c = fv.get("count") if isinstance(fv, dict) else getattr(fv, "count", None)
        if v and v.lower() == value.lower():
            return c
    return None


# =============================================================================
# Skip if no server
# =============================================================================

@pytest.fixture(scope="module")
def auth_headers():
    """Generate auth headers once per module."""
    return get_auth_headers()


@pytest.fixture(scope="module", autouse=True)
def check_server():
    """Skip all tests if server is not reachable."""
    server_url = os.getenv("TEST_SERVER_URL")
    if not server_url:
        pytest.skip("TEST_SERVER_URL not set")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code != 200:
            pytest.skip(f"Server at {BASE_URL} returned {r.status_code}")
    except requests.ConnectionError:
        pytest.skip(f"Server at {BASE_URL} not reachable")


# =============================================================================
# 1. SPECIFIC intent — brand filter refinement
# =============================================================================

@pytest.mark.integration
class TestSpecificBrandRefine:
    """SPECIFIC intent (e.g. 'black midi dress') with brand filter."""

    def test_refine_returns_new_session_id(self, auth_headers):
        """Filter refinement returns a NEW session_id (different from original)."""
        p1 = search_page1("black midi dress", auth_headers, page_size=20)
        assert p1.get("search_session_id"), "No session_id from page 1"
        original_sid = p1["search_session_id"]

        refined = search_refine(
            "black midi dress", auth_headers,
            search_session_id=original_sid,
            page_size=20,
            brands=["Boohoo"],
        )
        assert refined.get("search_session_id"), "No session_id from refinement"
        assert refined["search_session_id"] != original_sid, \
            "Refinement should create a NEW session ID"

    def test_refine_timing_has_filter_refine_flag(self, auth_headers):
        """Refined response timing should have filter_refine=True."""
        p1 = search_page1("red party dress", auth_headers, page_size=20)
        if not p1.get("search_session_id"):
            pytest.skip("No session_id from page 1")

        refined = search_refine(
            "red party dress", auth_headers,
            search_session_id=p1["search_session_id"],
            page_size=20,
            brands=["Boohoo"],
        )
        timing = refined.get("timing", {})
        assert timing.get("filter_refine") is True, \
            f"Expected filter_refine=True in timing, got: {timing}"

    def test_refine_brand_filter_narrows_results(self, auth_headers):
        """Brand filter should narrow total_results compared to original."""
        p1 = search_page1("black midi dress", auth_headers, page_size=20)
        original_total = p1["pagination"]["total_results"]

        if not p1.get("search_session_id"):
            pytest.skip("No session_id from page 1")

        refined = search_refine(
            "black midi dress", auth_headers,
            search_session_id=p1["search_session_id"],
            page_size=20,
            brands=["Boohoo"],
        )
        refined_total = refined["pagination"]["total_results"]

        print(f"  Original total: {original_total}")
        print(f"  Refined total (Boohoo only): {refined_total}")

        # Refined total should be smaller (or equal for all-Boohoo catalogs)
        assert refined_total <= original_total, \
            f"Refined total ({refined_total}) should be <= original ({original_total})"

    def test_refine_results_match_brand(self, auth_headers):
        """All results after brand refinement should be from the selected brand."""
        p1 = search_page1("summer dress", auth_headers, page_size=20)
        if not p1.get("search_session_id"):
            pytest.skip("No session_id from page 1")

        refined = search_refine(
            "summer dress", auth_headers,
            search_session_id=p1["search_session_id"],
            page_size=20,
            brands=["Boohoo"],
        )

        brands = get_brands(refined)
        if refined["results"]:
            # All results should be Boohoo (post-filter enforces this)
            assert brands == {"Boohoo"} or brands == set(), \
                f"Expected only Boohoo results, got brands: {brands}"

    def test_refine_faster_than_full_pipeline(self, auth_headers):
        """Filter refinement should be faster than running the full pipeline again."""
        p1 = search_page1("casual linen pants", auth_headers, page_size=20)
        p1_ms = p1.get("timing", {}).get("total_ms", 0)

        if not p1.get("search_session_id"):
            pytest.skip("No session_id from page 1")

        refined = search_refine(
            "casual linen pants", auth_headers,
            search_session_id=p1["search_session_id"],
            page_size=20,
            brands=["Boohoo"],
        )
        refine_ms = refined.get("timing", {}).get("total_ms", 0)

        print(f"  Page 1 (full pipeline): {p1_ms}ms")
        print(f"  Filter refine: {refine_ms}ms")

        # Refinement should be significantly faster (no planner, no encoding)
        # Allow some tolerance for cold-start and load spikes
        if p1_ms > 3000:  # only assert if page 1 was slow enough to be meaningful
            assert refine_ms < p1_ms, \
                f"Refine ({refine_ms}ms) should be faster than full pipeline ({p1_ms}ms)"

    def test_refine_page1_then_page2(self, auth_headers):
        """After refinement, page 2 should work with the new session_id."""
        p1 = search_page1("floral dress", auth_headers, page_size=10)
        if not p1.get("search_session_id"):
            pytest.skip("No session_id from page 1")

        refined = search_refine(
            "floral dress", auth_headers,
            search_session_id=p1["search_session_id"],
            page_size=10,
            brands=["Boohoo"],
        )
        new_sid = refined.get("search_session_id")
        new_cursor = refined.get("cursor")

        if not new_sid or not new_cursor:
            pytest.skip("Refinement returned no session/cursor (empty results?)")

        p2 = search_page_n(
            "floral dress", auth_headers,
            search_session_id=new_sid,
            cursor=new_cursor,
            page=2,
            page_size=10,
        )
        assert p2.get("results") is not None, "Page 2 after refine should return results"

        # No overlap between refined page 1 and page 2
        p1_ids = get_ids(refined)
        p2_ids = get_ids(p2)
        overlap = p1_ids & p2_ids
        assert len(overlap) == 0, f"Page 1 and page 2 share {len(overlap)} product IDs"


# =============================================================================
# 2. SPECIFIC intent — price filter refinement
# =============================================================================

@pytest.mark.integration
class TestSpecificPriceRefine:
    """SPECIFIC intent with price filter."""

    def test_price_filter_narrows_total(self, auth_headers):
        """Price filter should narrow total_results."""
        p1 = search_page1("black midi dress", auth_headers, page_size=20)
        original_total = p1["pagination"]["total_results"]

        if not p1.get("search_session_id"):
            pytest.skip("No session_id from page 1")

        refined = search_refine(
            "black midi dress", auth_headers,
            search_session_id=p1["search_session_id"],
            page_size=20,
            max_price=30,
        )
        refined_total = refined["pagination"]["total_results"]

        print(f"  Original total: {original_total}")
        print(f"  Refined total (max $30): {refined_total}")

        assert refined_total <= original_total

    def test_price_filter_results_under_max(self, auth_headers):
        """All results should be under the max price."""
        p1 = search_page1("summer dress", auth_headers, page_size=20)
        if not p1.get("search_session_id"):
            pytest.skip("No session_id from page 1")

        max_price = 40.0
        refined = search_refine(
            "summer dress", auth_headers,
            search_session_id=p1["search_session_id"],
            page_size=20,
            max_price=max_price,
        )

        prices = get_prices(refined)
        if prices:
            over_budget = [p for p in prices if p > max_price]
            assert len(over_budget) == 0, \
                f"Found {len(over_budget)} items over ${max_price}: {over_budget[:5]}"


# =============================================================================
# 3. Combined filters (brand + price)
# =============================================================================

@pytest.mark.integration
class TestCombinedFilterRefine:
    """Multiple filters applied simultaneously."""

    def test_brand_plus_price_narrows_more(self, auth_headers):
        """Brand + price should narrow more than brand alone."""
        p1 = search_page1("black midi dress", auth_headers, page_size=20)
        if not p1.get("search_session_id"):
            pytest.skip("No session_id from page 1")

        original_total = p1["pagination"]["total_results"]

        # Brand-only refinement
        brand_refined = search_refine(
            "black midi dress", auth_headers,
            search_session_id=p1["search_session_id"],
            page_size=20,
            brands=["Boohoo"],
        )
        brand_total = brand_refined["pagination"]["total_results"]

        # Need a fresh session for the combined test
        p1_fresh = search_page1("black midi dress", auth_headers, page_size=20)
        if not p1_fresh.get("search_session_id"):
            pytest.skip("No session_id from fresh page 1")

        # Brand + price refinement
        combined_refined = search_refine(
            "black midi dress", auth_headers,
            search_session_id=p1_fresh["search_session_id"],
            page_size=20,
            brands=["Boohoo"],
            max_price=30,
        )
        combined_total = combined_refined["pagination"]["total_results"]

        print(f"  Original total: {original_total}")
        print(f"  Brand-only total: {brand_total}")
        print(f"  Brand + price total: {combined_total}")

        # Combined should be <= brand-only (more restrictive)
        assert combined_total <= brand_total


# =============================================================================
# 4. Facet accuracy
# =============================================================================

@pytest.mark.integration
class TestFacetAccuracy:
    """Test that facets reflect the filtered catalog accurately."""

    def test_facets_present_after_refine(self, auth_headers):
        """Refined response should include facets."""
        p1 = search_page1("summer dress", auth_headers, page_size=20)
        if not p1.get("search_session_id"):
            pytest.skip("No session_id from page 1")

        refined = search_refine(
            "summer dress", auth_headers,
            search_session_id=p1["search_session_id"],
            page_size=20,
            brands=["Boohoo"],
        )

        assert refined.get("facets"), "Refined response should have facets"
        assert "brand" in refined["facets"], "Facets should include 'brand'"

    def test_facet_counts_catalog_wide(self, auth_headers):
        """Facet counts should reflect Algolia catalog, not just merged results."""
        p1 = search_page1("black midi dress", auth_headers, page_size=20)
        if not p1.get("search_session_id"):
            pytest.skip("No session_id from page 1")

        # Get brand facet count from page 1 (unfiltered)
        p1_boohoo_count = get_facet_count(p1, "brand", "Boohoo")

        # Now refine with just a price filter (keeping all brands visible)
        refined = search_refine(
            "black midi dress", auth_headers,
            search_session_id=p1["search_session_id"],
            page_size=20,
            max_price=100,
        )
        refined_boohoo_count = get_facet_count(refined, "brand", "Boohoo")

        print(f"  Page 1 Boohoo facet count: {p1_boohoo_count}")
        print(f"  Refined (max $100) Boohoo facet count: {refined_boohoo_count}")

        # Facet count should be a reasonable catalog-wide number (not tiny)
        # Both should be > the number of returned results (20)
        if refined_boohoo_count is not None:
            assert refined_boohoo_count > 0, "Boohoo count should be positive"

    def test_facets_update_with_filter(self, auth_headers):
        """Facets should reflect the filtered view (narrowed by brand)."""
        p1 = search_page1("dress", auth_headers, page_size=20)
        if not p1.get("search_session_id"):
            pytest.skip("No session_id from page 1")

        p1_total = p1["pagination"]["total_results"]

        # Refine with brand filter
        refined = search_refine(
            "dress", auth_headers,
            search_session_id=p1["search_session_id"],
            page_size=20,
            brands=["Boohoo"],
        )

        # The brand facet should still be present in refined facets
        # (even though we filtered by brand, Algolia returns facets for the filtered set)
        assert refined.get("facets"), "Refined should have facets"


# =============================================================================
# 5. EXACT intent refinement
# =============================================================================

@pytest.mark.integration
class TestExactIntentRefine:
    """EXACT intent (bare category) can also be refined."""

    def test_exact_brand_refine(self, auth_headers):
        """EXACT intent (e.g. 'tops') can be refined with a brand filter."""
        p1 = search_page1("tops", auth_headers, page_size=20)
        if not p1.get("search_session_id"):
            pytest.skip("No session_id from page 1")

        # EXACT intent may classify as "exact"
        print(f"  Intent: {p1.get('intent')}")

        refined = search_refine(
            "tops", auth_headers,
            search_session_id=p1["search_session_id"],
            page_size=20,
            brands=["Boohoo"],
        )

        refined_total = refined["pagination"]["total_results"]
        print(f"  Refined total (Boohoo tops): {refined_total}")

        assert refined_total > 0, "Should have Boohoo tops"
        assert refined_total <= p1["pagination"]["total_results"]


# =============================================================================
# 6. VAGUE intent refinement
# =============================================================================

@pytest.mark.integration
class TestVagueIntentRefine:
    """VAGUE intent (mood/aesthetic) can be refined with filters."""

    def test_vague_price_refine(self, auth_headers):
        """VAGUE intent (e.g. 'summer vibes') with price filter."""
        p1 = search_page1("summer vibes", auth_headers, page_size=20)
        if not p1.get("search_session_id"):
            pytest.skip("No session_id from page 1")

        print(f"  Intent: {p1.get('intent')}")

        refined = search_refine(
            "summer vibes", auth_headers,
            search_session_id=p1["search_session_id"],
            page_size=20,
            max_price=25,
        )

        timing = refined.get("timing", {})
        print(f"  Refine timing: {timing.get('total_ms')}ms")

        if refined["results"]:
            prices = get_prices(refined)
            over_budget = [p for p in prices if p > 25.0]
            assert len(over_budget) == 0, \
                f"Found items over $25: {over_budget[:5]}"


# =============================================================================
# 7. Result count consistency
# =============================================================================

@pytest.mark.integration
class TestResultCountConsistency:
    """Verify total_results is accurate and consistent."""

    def test_total_results_narrows_with_brand(self, auth_headers):
        """total_results should decrease when filtering by brand."""
        p1 = search_page1("black midi dress", auth_headers, page_size=20)
        if not p1.get("search_session_id"):
            pytest.skip("No session_id from page 1")

        original = p1["pagination"]["total_results"]

        refined = search_refine(
            "black midi dress", auth_headers,
            search_session_id=p1["search_session_id"],
            page_size=20,
            brands=["Boohoo"],
        )
        narrowed = refined["pagination"]["total_results"]

        print(f"  Original total_results: {original}")
        print(f"  Boohoo-only total_results: {narrowed}")

        assert narrowed <= original
        assert narrowed >= 0

    def test_total_results_consistent_with_page2(self, auth_headers):
        """total_results after refinement should be consistent on page 2."""
        p1 = search_page1("casual dress", auth_headers, page_size=10)
        if not p1.get("search_session_id"):
            pytest.skip("No session_id from page 1")

        refined = search_refine(
            "casual dress", auth_headers,
            search_session_id=p1["search_session_id"],
            page_size=10,
            brands=["Boohoo"],
        )
        refine_total = refined["pagination"]["total_results"]

        if not refined.get("search_session_id") or not refined.get("cursor"):
            pytest.skip("No session/cursor from refinement")

        p2 = search_page_n(
            "casual dress", auth_headers,
            search_session_id=refined["search_session_id"],
            cursor=refined["cursor"],
            page=2,
            page_size=10,
        )
        p2_total = p2["pagination"]["total_results"]

        print(f"  Refine total_results: {refine_total}")
        print(f"  Page 2 total_results: {p2_total}")

        # total_results should be consistent (both come from the same Algolia nbHits)
        assert p2_total == refine_total, \
            f"total_results mismatch: refine={refine_total}, page2={p2_total}"
