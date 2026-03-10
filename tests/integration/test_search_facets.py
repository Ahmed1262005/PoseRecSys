"""
Integration tests for facet computation in hybrid search.

Tests cover:
1. Facets returned on basic search (no user filters)
2. Facet structure matches Algolia pattern (count>1, >=2 values, sorted desc)
3. Facets recomputed from merged results when semantic contributes
4. Facets narrow when user brand filter applied (post-filter)
5. Facets narrow when user price filter applied (post-filter)
6. Facets reflect post-filtered results (brand counts match actual results)
7. Sorted search path returns Algolia-native facets
8. Facets include multi-value fields (occasions, seasons, etc.)

Requires:
- Running server: PYTHONPATH=src uvicorn api.app:app --port 8000
- Env vars: TEST_SERVER_URL, SUPABASE_JWT_SECRET
- Algolia index with products

Run with:
    TEST_SERVER_URL=http://localhost:8000 PYTHONPATH=src python -m pytest tests/integration/test_search_facets.py -v -s
"""

import os
import sys
import time
import pytest
import requests
from typing import Dict, List, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# =============================================================================
# Configuration
# =============================================================================

BASE_URL = os.getenv("TEST_SERVER_URL", "http://localhost:8000")
SEARCH_PREFIX = "/api/search"

# The 19 facet fields we expect
EXPECTED_FACET_FIELDS = {
    "brand", "category_l1", "broad_category", "article_type",
    "formality", "primary_color", "color_family", "pattern",
    "fit_type", "neckline", "sleeve_type", "length", "silhouette", "rise",
    "occasions", "seasons", "style_tags", "materials", "is_on_sale",
}


def get_auth_headers(user_id: str = "test-facet-user-001") -> Dict[str, str]:
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


def search(query: str, headers: dict, timeout: int = 120, **kwargs) -> requests.Response:
    """Helper: POST /api/search/hybrid."""
    body = {"query": query, **kwargs}
    return requests.post(
        f"{BASE_URL}{SEARCH_PREFIX}/hybrid",
        json=body,
        headers=headers,
        timeout=timeout,
    )


# =============================================================================
# Fixtures
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
# 1. Basic facet presence and structure
# =============================================================================

@pytest.mark.integration
class TestFacetBasic:
    """Facets should always be returned with valid structure."""

    def test_facets_present_on_basic_search(self, auth_headers):
        """A basic search should return facets in the response."""
        r = search("summer dress", auth_headers)
        assert r.status_code == 200
        data = r.json()
        assert "facets" in data
        assert data["facets"] is not None
        assert isinstance(data["facets"], dict)
        assert len(data["facets"]) > 0

    def test_facet_keys_are_known_fields(self, auth_headers):
        """All returned facet keys should be from the known 19 fields."""
        r = search("black top", auth_headers)
        assert r.status_code == 200
        facets = r.json()["facets"]
        if facets:
            for key in facets:
                assert key in EXPECTED_FACET_FIELDS, (
                    f"Unexpected facet key: {key}"
                )

    def test_facet_values_have_correct_shape(self, auth_headers):
        """Each facet should be a list of {value, count} objects."""
        r = search("floral dress", auth_headers)
        assert r.status_code == 200
        facets = r.json()["facets"]
        assert facets is not None
        for field_name, values in facets.items():
            assert isinstance(values, list), f"{field_name} should be a list"
            assert len(values) >= 2, (
                f"{field_name} should have >= 2 values (got {len(values)})"
            )
            for fv in values:
                assert "value" in fv, f"Missing 'value' in {field_name} facet"
                assert "count" in fv, f"Missing 'count' in {field_name} facet"
                assert isinstance(fv["value"], str)
                assert isinstance(fv["count"], int)
                assert fv["count"] > 1, (
                    f"{field_name}={fv['value']} has count={fv['count']}, "
                    "expected > 1"
                )

    def test_facet_values_sorted_by_count_descending(self, auth_headers):
        """Facet values should be sorted highest count first."""
        r = search("casual top", auth_headers)
        assert r.status_code == 200
        facets = r.json()["facets"]
        assert facets is not None
        for field_name, values in facets.items():
            counts = [fv["count"] for fv in values]
            assert counts == sorted(counts, reverse=True), (
                f"{field_name} facets not sorted descending: {counts}"
            )

    def test_facet_values_exclude_null_na(self, auth_headers):
        """Null-like values (N/A, null, none, '') should not appear."""
        r = search("dress", auth_headers)
        assert r.status_code == 200
        facets = r.json()["facets"]
        assert facets is not None
        null_values = {"null", "n/a", "none", ""}
        for field_name, values in facets.items():
            for fv in values:
                assert fv["value"].lower() not in null_values, (
                    f"{field_name} has null-like value: {fv['value']}"
                )

    def test_brand_facet_present_on_broad_query(self, auth_headers):
        """A broad query like 'dress' should have a brand facet."""
        r = search("dress", auth_headers)
        assert r.status_code == 200
        facets = r.json()["facets"]
        assert facets is not None
        assert "brand" in facets
        assert len(facets["brand"]) >= 2


# =============================================================================
# 2. Facets with user brand filter (post-filter architecture)
# =============================================================================

@pytest.mark.integration
class TestFacetWithBrandFilter:
    """When user selects a brand, facets should reflect the filtered set."""

    def test_brand_facet_narrows_to_selected_brand(self, auth_headers):
        """After filtering by brand, brand facet should only contain
        that brand (or be absent if only 1 value left)."""
        r = search("dress", auth_headers, brands=["Boohoo"])
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) > 0, "Expected results for Boohoo dress"

        facets = data["facets"]
        # Brand facet may be absent (only 1 brand → < 2 distinct values)
        # or present with just Boohoo-related counts
        if facets and "brand" in facets:
            brand_values = {fv["value"] for fv in facets["brand"]}
            # Should only contain Boohoo (user post-filtered)
            assert brand_values == {"Boohoo"} or len(brand_values) == 1

    def test_other_facets_reflect_brand_filtered_results(self, auth_headers):
        """Non-brand facets should show counts within the selected brand."""
        r = search("top", auth_headers, brands=["Boohoo"])
        assert r.status_code == 200
        data = r.json()
        facets = data.get("facets")
        results = data["results"]

        if not results or not facets:
            pytest.skip("No results or facets for Boohoo top")

        # All returned results should be Boohoo
        for item in results:
            assert item["brand"].lower() == "boohoo", (
                f"Result {item['product_id']} has brand {item['brand']}, "
                "expected Boohoo"
            )

        # If category_l1 facet exists, verify counts match results
        if "category_l1" in facets:
            facet_total = sum(fv["count"] for fv in facets["category_l1"])
            # Facet counts come from merged set (pre-pagination),
            # so total >= number of paginated results
            assert facet_total >= len(results)

    def test_all_results_match_brand_filter(self, auth_headers):
        """Fundamental: brand post-filter must actually work."""
        r = search("athleisure", auth_headers, brands=["Nike"])
        assert r.status_code == 200
        data = r.json()
        # Even if planner guesses wrong categories, Nike results
        # should come through via post-filtering
        for item in data["results"]:
            assert item["brand"].lower() == "nike", (
                f"Result {item['product_id']} has brand {item['brand']}, "
                "expected Nike (athleisure + Nike post-filter)"
            )


# =============================================================================
# 3. Facets with user price filter
# =============================================================================

@pytest.mark.integration
class TestFacetWithPriceFilter:
    """Facets should reflect the price-filtered result set."""

    def test_results_respect_price_range(self, auth_headers):
        """All results should be within the user's price range."""
        r = search("dress", auth_headers, min_price=20, max_price=50)
        assert r.status_code == 200
        data = r.json()
        for item in data["results"]:
            price = item.get("price", 0)
            assert 20 <= price <= 50, (
                f"Result {item['product_id']} price={price}, "
                "expected 20-50"
            )

    def test_facets_present_with_price_filter(self, auth_headers):
        """Facets should still be returned when price filter is active."""
        r = search("top", auth_headers, min_price=10, max_price=40)
        assert r.status_code == 200
        data = r.json()
        assert data["facets"] is not None
        assert isinstance(data["facets"], dict)


# =============================================================================
# 4. Facets with combined user filters
# =============================================================================

@pytest.mark.integration
class TestFacetWithCombinedFilters:
    """Facets should reflect the intersection of multiple user filters."""

    def test_brand_plus_price_narrows_facets(self, auth_headers):
        """Brand + price range: facets should reflect the intersection."""
        r = search(
            "dress", auth_headers,
            brands=["Boohoo"],
            min_price=20,
            max_price=60,
        )
        assert r.status_code == 200
        data = r.json()

        # All results must be Boohoo within price range
        for item in data["results"]:
            assert item["brand"].lower() == "boohoo"
            price = item.get("price", 0)
            assert 20 <= price <= 60

        # Facets should be non-null if there are results
        if data["results"]:
            assert data["facets"] is not None

    def test_brand_plus_color_narrows_facets(self, auth_headers):
        """Brand + color: facets should reflect both filters."""
        r = search(
            "top", auth_headers,
            brands=["Boohoo"],
            colors=["Black"],
        )
        assert r.status_code == 200
        data = r.json()

        for item in data["results"]:
            assert item["brand"].lower() == "boohoo"
            # Color could be in primary_color or colors list
            item_colors = [
                c.lower() for c in (item.get("colors") or [])
            ]
            primary = (item.get("primary_color") or "").lower()
            assert "black" in item_colors or primary == "black", (
                f"Result {item['product_id']} has no black color"
            )


# =============================================================================
# 5. Facet counts match result set
# =============================================================================

@pytest.mark.integration
class TestFacetCountAccuracy:
    """Facet counts should match what's actually in the result set."""

    def test_brand_facet_counts_match_results(self, auth_headers):
        """Brand facet counts should match the number of items per brand
        in the merged result set."""
        r = search("summer dress", auth_headers, page_size=40)
        assert r.status_code == 200
        data = r.json()
        facets = data.get("facets")
        results = data["results"]

        if not facets or "brand" not in facets:
            pytest.skip("No brand facet returned")

        # Count brands in actual results
        from collections import Counter
        result_brands = Counter(
            item["brand"] for item in results if item.get("brand")
        )

        # Facet counts come from the full merged set (before pagination),
        # which may be larger than the paginated results. But every brand
        # in the results should appear in the facets.
        facet_brands = {fv["value"]: fv["count"] for fv in facets["brand"]}
        for brand, count in result_brands.items():
            if count > 1:
                assert brand in facet_brands, (
                    f"Brand '{brand}' (count={count} in results) "
                    "missing from facets"
                )
                # Facet count >= result count (merged set >= paginated results)
                assert facet_brands[brand] >= count, (
                    f"Brand '{brand}' facet count ({facet_brands[brand]}) "
                    f"< result count ({count})"
                )

    def test_category_l1_facet_counts_match_results(self, auth_headers):
        """category_l1 facet counts should match results."""
        r = search("casual outfit", auth_headers, page_size=40)
        assert r.status_code == 200
        data = r.json()
        facets = data.get("facets")
        results = data["results"]

        if not facets or "category_l1" not in facets:
            pytest.skip("No category_l1 facet returned")

        from collections import Counter
        result_cats = Counter(
            item["category_l1"] for item in results
            if item.get("category_l1")
        )

        facet_cats = {fv["value"]: fv["count"] for fv in facets["category_l1"]}
        for cat, count in result_cats.items():
            if count > 1 and cat.lower() not in ("null", "n/a", "none", ""):
                assert cat in facet_cats, (
                    f"Category '{cat}' (count={count}) missing from facets"
                )


# =============================================================================
# 6. Multi-value facet fields
# =============================================================================

@pytest.mark.integration
class TestFacetMultiValueFields:
    """Multi-value fields (occasions, seasons, etc.) should be faceted."""

    def test_occasions_facet_present(self, auth_headers):
        """Occasions facet should appear for broad queries."""
        r = search("dress", auth_headers)
        assert r.status_code == 200
        facets = r.json()["facets"]
        if facets and "occasions" in facets:
            for fv in facets["occasions"]:
                assert fv["count"] > 1
                assert fv["value"].lower() not in ("null", "n/a", "none", "")

    def test_seasons_facet_present(self, auth_headers):
        """Seasons facet should appear for broad queries."""
        r = search("top", auth_headers)
        assert r.status_code == 200
        facets = r.json()["facets"]
        if facets and "seasons" in facets:
            for fv in facets["seasons"]:
                assert fv["count"] > 1

    def test_materials_facet_present(self, auth_headers):
        """Materials facet should appear when results have material data."""
        r = search("silk dress", auth_headers)
        assert r.status_code == 200
        facets = r.json()["facets"]
        if facets and "materials" in facets:
            for fv in facets["materials"]:
                assert fv["count"] > 1


# =============================================================================
# 7. Sorted search path (Algolia-only)
# =============================================================================

@pytest.mark.integration
class TestFacetSortedSearch:
    """Sorted search (price_asc, price_desc, trending) uses Algolia facets."""

    def test_price_asc_has_facets(self, auth_headers):
        """sort_by=price_asc should return facets from Algolia."""
        r = search("dress", auth_headers, sort_by="price_asc")
        assert r.status_code == 200
        data = r.json()
        assert data["sort_by"] == "price_asc"
        assert data["facets"] is not None
        assert isinstance(data["facets"], dict)

    def test_price_desc_has_facets(self, auth_headers):
        """sort_by=price_desc should return facets from Algolia."""
        r = search("top", auth_headers, sort_by="price_desc")
        assert r.status_code == 200
        data = r.json()
        assert data["sort_by"] == "price_desc"
        assert data["facets"] is not None

    def test_sorted_facets_have_valid_structure(self, auth_headers):
        """Sorted search facets should have the same structure."""
        r = search("dress", auth_headers, sort_by="price_asc")
        assert r.status_code == 200
        facets = r.json()["facets"]
        if facets:
            for field_name, values in facets.items():
                assert field_name in EXPECTED_FACET_FIELDS
                assert len(values) >= 2
                for fv in values:
                    assert fv["count"] > 1


# =============================================================================
# 8. The athleisure + brand scenario (regression)
# =============================================================================

@pytest.mark.integration
class TestFacetAthleisureRegression:
    """Regression: 'athleisure' + brand filter should return results
    with accurate facets, not 0 results due to category conflict."""

    def test_athleisure_boohoo_returns_results(self, auth_headers):
        """'athleisure' + Boohoo should return results and facets."""
        r = search("athleisure", auth_headers, brands=["Boohoo"])
        assert r.status_code == 200
        data = r.json()
        # Should have results — post-filter keeps Boohoo regardless
        # of planner's category guess
        assert len(data["results"]) > 0, (
            "Expected results for 'athleisure' + Boohoo"
        )
        # All results should be Boohoo
        for item in data["results"]:
            assert item["brand"].lower() == "boohoo"

        # Facets should be present and valid
        facets = data["facets"]
        if facets:
            for field_name, values in facets.items():
                assert field_name in EXPECTED_FACET_FIELDS
                for fv in values:
                    assert fv["count"] > 1

    def test_athleisure_without_brand_has_diverse_facets(self, auth_headers):
        """'athleisure' without brand filter should have multiple brands."""
        r = search("athleisure", auth_headers)
        assert r.status_code == 200
        data = r.json()
        facets = data.get("facets")

        if facets and "brand" in facets:
            brand_count = len(facets["brand"])
            assert brand_count >= 2, (
                f"Expected >= 2 brands for 'athleisure', got {brand_count}"
            )
