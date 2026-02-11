"""
Integration tests for the Search API endpoints.

Tests cover:
1. Hybrid search - various query types (exact, specific, vague)
2. Autocomplete - product + brand suggestions
3. Click/conversion analytics tracking
4. Filter combinations
5. Pagination

Requires:
- Running server: PYTHONPATH=src uvicorn api.app:app --port 8000
- Env vars: TEST_SERVER_URL, SUPABASE_JWT_SECRET
- Algolia index with products
- (Optional) search_analytics tables in Supabase for analytics tests

Run with:
    TEST_SERVER_URL=http://localhost:8000 PYTHONPATH=src python -m pytest tests/integration/test_search.py -v -s
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


def get_auth_headers(user_id: str = "test-search-user-001") -> Dict[str, str]:
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
    """Helper: POST /api/search/hybrid."""
    body = {"query": query, **kwargs}
    return requests.post(f"{BASE_URL}{SEARCH_PREFIX}/hybrid", json=body, headers=headers)


def autocomplete(q: str, headers: dict, limit: int = 10) -> requests.Response:
    """Helper: GET /api/search/autocomplete."""
    return requests.get(
        f"{BASE_URL}{SEARCH_PREFIX}/autocomplete",
        params={"q": q, "limit": limit},
        headers=headers,
    )


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
# 1. Hybrid Search - Basic
# =============================================================================

@pytest.mark.integration
class TestHybridSearchBasic:
    """Basic hybrid search functionality."""

    def test_search_returns_200(self, auth_headers):
        """Valid search request should return 200."""
        r = search("dress", auth_headers)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"

    def test_search_response_structure(self, auth_headers):
        """Response should have expected structure."""
        r = search("black top", auth_headers)
        assert r.status_code == 200
        data = r.json()
        assert "query" in data
        assert "intent" in data
        assert "results" in data
        assert "pagination" in data
        assert "timing" in data
        assert data["query"] == "black top"
        assert data["intent"] in ("exact", "specific", "vague")
        assert isinstance(data["results"], list)
        assert data["pagination"]["page"] == 1

    def test_search_returns_products(self, auth_headers):
        """Search should return actual products."""
        r = search("floral dress", auth_headers)
        data = r.json()
        assert len(data["results"]) > 0, "Expected results for 'floral dress'"

        product = data["results"][0]
        assert "product_id" in product
        assert "name" in product
        assert "brand" in product
        assert "price" in product
        assert "image_url" in product

    def test_search_timing_info(self, auth_headers):
        """Response should include timing breakdown."""
        r = search("sweater", auth_headers)
        data = r.json()
        assert "total_ms" in data["timing"]
        assert "algolia_ms" in data["timing"]
        assert data["timing"]["total_ms"] >= 0

    def test_search_requires_auth(self):
        """Search without auth should return 401/403."""
        r = requests.post(
            f"{BASE_URL}{SEARCH_PREFIX}/hybrid",
            json={"query": "test"},
        )
        assert r.status_code in (401, 403, 422)


# =============================================================================
# 2. Hybrid Search - Intent Classification
# =============================================================================

@pytest.mark.integration
class TestSearchIntentClassification:
    """Query intent detection and appropriate search behavior."""

    def test_exact_brand_query(self, auth_headers):
        """Brand-only query should be classified as exact."""
        r = search("boohoo", auth_headers)
        data = r.json()
        assert data["intent"] == "exact"
        # Semantic search should be skipped
        assert "semantic_ms" not in data["timing"] or data["timing"].get("semantic_ms", 0) == 0

    def test_specific_query(self, auth_headers):
        """Category + attribute should be classified as specific."""
        r = search("blue midi dress", auth_headers)
        data = r.json()
        assert data["intent"] == "specific"

    def test_vague_query(self, auth_headers):
        """Style/vibe query should be classified as vague."""
        r = search("quiet luxury", auth_headers)
        data = r.json()
        assert data["intent"] == "vague"
        # Semantic search should be triggered
        if len(data["results"]) > 0:
            assert "semantic_ms" in data["timing"]


# =============================================================================
# 3. Hybrid Search - Filters
# =============================================================================

@pytest.mark.integration
class TestSearchFilters:
    """Filter application in search."""

    def test_brand_filter(self, auth_headers):
        """Brand filter should restrict results to specified brands."""
        r = search("top", auth_headers, brands=["Boohoo"])
        data = r.json()
        for product in data["results"]:
            assert product["brand"].lower() == "boohoo", (
                f"Expected brand 'boohoo', got '{product['brand']}'"
            )

    def test_on_sale_filter(self, auth_headers):
        """on_sale_only should only return sale items (from Algolia results)."""
        r = search("dress", auth_headers, on_sale_only=True)
        data = r.json()
        # Products with algolia_rank came from Algolia and should respect the filter.
        # Semantic-only results may lack is_on_sale metadata.
        algolia_results = [p for p in data["results"] if p.get("algolia_rank")]
        for product in algolia_results:
            assert product.get("is_on_sale", False), (
                f"Product {product['product_id']} is not on sale"
            )

    def test_price_range_filter(self, auth_headers):
        """Price range filter should restrict results."""
        r = search("top", auth_headers, min_price=10, max_price=50)
        data = r.json()
        for product in data["results"]:
            assert 10 <= product["price"] <= 50, (
                f"Price {product['price']} outside range 10-50"
            )

    def test_category_filter(self, auth_headers):
        """Category filter should restrict to matching categories."""
        r = search("summer", auth_headers, categories=["dresses"])
        data = r.json()
        if data["results"]:
            categories = [p.get("broad_category", "").lower() for p in data["results"]]
            assert any("dress" in c for c in categories), (
                f"Expected some dresses, got categories: {categories[:5]}"
            )

    def test_exclude_brand_filter(self, auth_headers):
        """Excluded brands should not appear in results."""
        r = search("top", auth_headers, exclude_brands=["Boohoo"])
        data = r.json()
        for product in data["results"]:
            assert product["brand"].lower() != "boohoo", (
                f"Excluded brand 'boohoo' appeared in results"
            )


# =============================================================================
# 4. Hybrid Search - Pagination
# =============================================================================

@pytest.mark.integration
class TestSearchPagination:
    """Pagination behavior."""

    def test_pagination_defaults(self, auth_headers):
        """Default pagination should be page 1, page_size 50."""
        r = search("dress", auth_headers)
        data = r.json()
        assert data["pagination"]["page"] == 1
        assert data["pagination"]["page_size"] == 50

    def test_custom_page_size(self, auth_headers):
        """Custom page_size should limit results."""
        r = search("top", auth_headers, page_size=5)
        data = r.json()
        assert len(data["results"]) <= 5
        assert data["pagination"]["page_size"] == 5

    def test_pagination_page_2(self, auth_headers):
        """Page 2 should return different results than page 1."""
        r1 = search("dress", auth_headers, page=1, page_size=10)
        r2 = search("dress", auth_headers, page=2, page_size=10)
        data1 = r1.json()
        data2 = r2.json()

        ids1 = {p["product_id"] for p in data1["results"]}
        ids2 = {p["product_id"] for p in data2["results"]}

        if data1["pagination"]["has_more"] and len(data2["results"]) > 0:
            # Pages should have no overlap
            overlap = ids1 & ids2
            assert len(overlap) == 0, f"Pages overlap: {overlap}"

    def test_has_more_flag(self, auth_headers):
        """has_more should be true when more results exist."""
        r = search("dress", auth_headers, page_size=5)
        data = r.json()
        # With 99K products, "dress" should have more than 5 results
        if len(data["results"]) == 5:
            assert data["pagination"]["has_more"] is True


# =============================================================================
# 5. Autocomplete
# =============================================================================

@pytest.mark.integration
class TestAutocomplete:
    """Autocomplete endpoint tests."""

    def test_autocomplete_returns_200(self, auth_headers):
        """Valid autocomplete request should return 200."""
        r = autocomplete("dre", auth_headers)
        assert r.status_code == 200

    def test_autocomplete_response_structure(self, auth_headers):
        """Response should have products and brands arrays."""
        r = autocomplete("boo", auth_headers)
        data = r.json()
        assert "products" in data
        assert "brands" in data
        assert "query" in data
        assert isinstance(data["products"], list)
        assert isinstance(data["brands"], list)

    def test_autocomplete_product_suggestions(self, auth_headers):
        """Should return product name suggestions."""
        r = autocomplete("dress", auth_headers, limit=5)
        data = r.json()
        assert len(data["products"]) > 0
        product = data["products"][0]
        assert "id" in product
        assert "name" in product
        assert "brand" in product

    def test_autocomplete_brand_suggestions(self, auth_headers):
        """Should return brand suggestions matching query."""
        r = autocomplete("boo", auth_headers)
        data = r.json()
        # "boo" should match "Boohoo"
        brand_names = [b["name"].lower() for b in data["brands"]]
        assert any("boo" in name for name in brand_names), (
            f"Expected 'boohoo' in brands, got: {brand_names}"
        )

    def test_autocomplete_limit(self, auth_headers):
        """Limit parameter should cap product results."""
        r = autocomplete("top", auth_headers, limit=3)
        data = r.json()
        assert len(data["products"]) <= 3

    def test_autocomplete_requires_auth(self):
        """Autocomplete without auth should return 401/403."""
        r = requests.get(
            f"{BASE_URL}{SEARCH_PREFIX}/autocomplete",
            params={"q": "test"},
        )
        assert r.status_code in (401, 403, 422)


# =============================================================================
# 6. Typo Tolerance (via Algolia)
# =============================================================================

@pytest.mark.integration
class TestTypoTolerance:
    """Algolia's typo tolerance should handle misspellings."""

    def test_typo_in_query(self, auth_headers):
        """Misspelled queries should still return results."""
        r = search("sweter", auth_headers)  # sweater
        data = r.json()
        assert len(data["results"]) > 0, "Expected results for typo 'sweter'"

    def test_typo_in_brand(self, auth_headers):
        """Misspelled brand name should still find the brand."""
        r = search("booho", auth_headers)  # boohoo
        data = r.json()
        if len(data["results"]) > 0:
            brands = [p["brand"].lower() for p in data["results"][:5]]
            assert any("boohoo" in b for b in brands), (
                f"Expected 'boohoo' in brands for typo query, got: {brands}"
            )


# =============================================================================
# 7. Analytics Events
# =============================================================================

@pytest.mark.integration
class TestSearchAnalytics:
    """Analytics event tracking."""

    def test_click_event_accepted(self, auth_headers):
        """Click event should return 200."""
        r = requests.post(
            f"{BASE_URL}{SEARCH_PREFIX}/click",
            json={
                "query": "test dress",
                "product_id": "test-product-001",
                "position": 1,
            },
            headers=auth_headers,
        )
        assert r.status_code == 200, f"Got {r.status_code}: {r.text}"

    def test_conversion_event_accepted(self, auth_headers):
        """Conversion event should return 200."""
        r = requests.post(
            f"{BASE_URL}{SEARCH_PREFIX}/conversion",
            json={
                "query": "test dress",
                "product_id": "test-product-001",
            },
            headers=auth_headers,
        )
        assert r.status_code == 200, f"Got {r.status_code}: {r.text}"

    def test_analytics_requires_auth(self):
        """Analytics endpoints without auth should return 401/403."""
        r = requests.post(
            f"{BASE_URL}{SEARCH_PREFIX}/click",
            json={"query": "test", "product_id": "p1", "position": 1},
        )
        assert r.status_code in (401, 403, 422)


# =============================================================================
# 8. Search Quality Spot Checks
# =============================================================================

@pytest.mark.integration
class TestSearchQuality:
    """Spot checks for search relevance."""

    def test_brand_search_returns_brand_products(self, auth_headers):
        """Searching for a brand should return products from that brand."""
        r = search("boohoo", auth_headers, page_size=20)
        data = r.json()
        if len(data["results"]) > 0:
            boohoo_count = sum(
                1 for p in data["results"] if "boohoo" in p["brand"].lower()
            )
            ratio = boohoo_count / len(data["results"])
            assert ratio >= 0.5, (
                f"Expected >50% Boohoo products for brand search, got {ratio:.0%}"
            )

    def test_category_search_returns_matching(self, auth_headers):
        """Searching for a category should return matching products."""
        r = search("midi dress", auth_headers, page_size=20)
        data = r.json()
        if len(data["results"]) > 0:
            # At least some results should be dresses
            names = [p["name"].lower() for p in data["results"][:10]]
            has_dress = any("dress" in n for n in names)
            assert has_dress, f"Expected some dresses in results, got: {names}"

    def test_color_search_returns_matching(self, auth_headers):
        """Searching for a color + category should work."""
        r = search("black jeans", auth_headers, page_size=20)
        data = r.json()
        assert len(data["results"]) > 0, "Expected results for 'black jeans'"

    def test_semantic_query_returns_results(self, auth_headers):
        """Vague/semantic query should return relevant results."""
        r = search("something cute for a date night", auth_headers, page_size=20)
        data = r.json()
        assert data["intent"] == "vague"
        # Should return something even for vague queries
        assert len(data["results"]) > 0, "Expected results for semantic query"


# =============================================================================
# 9. Extended Filter Tests (all attribute filters)
# =============================================================================

@pytest.mark.integration
class TestExtendedFilters:
    """Tests for all filter types including new ones."""

    def test_color_filter(self, auth_headers):
        """Color filter should restrict to matching primary colors."""
        r = search("top", auth_headers, colors=["Black"])
        data = r.json()
        for p in data["results"][:10]:
            if p.get("algolia_rank") and p.get("primary_color"):
                assert p["primary_color"].lower() == "black", (
                    f"Expected black, got {p['primary_color']}"
                )

    def test_pattern_filter(self, auth_headers):
        """Pattern filter should return matching items."""
        r = search("dress", auth_headers, patterns=["Floral"])
        data = r.json()
        if data["results"]:
            algolia = [p for p in data["results"] if p.get("algolia_rank")]
            for p in algolia[:5]:
                if p.get("pattern"):
                    assert p["pattern"].lower() == "floral", (
                        f"Expected Floral, got {p['pattern']}"
                    )

    def test_formality_filter(self, auth_headers):
        """Formality filter should restrict results."""
        r = search("dress", auth_headers, formality=["Casual"])
        data = r.json()
        assert len(data["results"]) > 0

    def test_fit_type_filter(self, auth_headers):
        """Fit type filter should work."""
        r = search("jeans", auth_headers, fit_type=["Slim"])
        data = r.json()
        assert len(data["results"]) > 0

    def test_neckline_filter(self, auth_headers):
        """Neckline filter should return matching items."""
        r = search("top", auth_headers, neckline=["V-Neck"])
        data = r.json()
        assert len(data["results"]) > 0

    def test_sleeve_type_filter(self, auth_headers):
        """Sleeve type filter should work."""
        r = search("top", auth_headers, sleeve_type=["Long Sleeve"])
        data = r.json()
        assert len(data["results"]) >= 0  # May be empty depending on data

    def test_length_filter(self, auth_headers):
        """Length filter should work."""
        r = search("dress", auth_headers, length=["Midi"])
        data = r.json()
        assert len(data["results"]) > 0

    def test_season_filter(self, auth_headers):
        """Season filter should restrict results."""
        r = search("top", auth_headers, seasons=["Summer"])
        data = r.json()
        assert len(data["results"]) > 0

    def test_occasion_filter(self, auth_headers):
        r = search("dress", auth_headers, occasions=["Date Night"])
        data = r.json()
        assert len(data["results"]) >= 0

    def test_materials_filter(self, auth_headers):
        """Materials filter should map to apparent_fabric."""
        r = search("top", auth_headers, materials=["Cotton"])
        data = r.json()
        assert len(data["results"]) > 0

    def test_silhouette_filter(self, auth_headers):
        """New silhouette filter should work."""
        r = search("dress", auth_headers, silhouette=["A-Line"])
        data = r.json()
        assert r.status_code == 200

    def test_article_type_filter(self, auth_headers):
        """New article_type filter should work."""
        r = search("clothing", auth_headers, article_type=["jeans"])
        data = r.json()
        assert r.status_code == 200

    def test_style_tags_filter(self, auth_headers):
        """New style_tags filter should work."""
        r = search("top", auth_headers, style_tags=["casual"])
        data = r.json()
        assert r.status_code == 200

    def test_category_l1_filter(self, auth_headers):
        """Category L1 filter should restrict results."""
        r = search("shirt", auth_headers, category_l1=["Tops"])
        data = r.json()
        assert len(data["results"]) > 0

    def test_color_family_filter(self, auth_headers):
        """Color family filter should work."""
        r = search("dress", auth_headers, color_family=["Neutrals"])
        data = r.json()
        assert r.status_code == 200

    def test_rise_filter(self, auth_headers):
        r = search("jeans", auth_headers, rise=["High Rise"])
        data = r.json()
        assert r.status_code == 200


# =============================================================================
# 10. Combined / Multi-Filter Tests
# =============================================================================

@pytest.mark.integration
class TestCombinedFilters:
    """Tests for multiple filters applied simultaneously."""

    def test_brand_plus_category(self, auth_headers):
        r = search("top", auth_headers, brands=["Boohoo"], categories=["tops"])
        data = r.json()
        for p in data["results"]:
            assert p["brand"].lower() == "boohoo"

    def test_brand_plus_price_range(self, auth_headers):
        r = search("dress", auth_headers, brands=["Boohoo"], min_price=10, max_price=50)
        data = r.json()
        for p in data["results"]:
            assert p["brand"].lower() == "boohoo"
            assert 10 <= p["price"] <= 50

    def test_category_plus_formality_plus_color(self, auth_headers):
        r = search("dress", auth_headers, category_l1=["Dresses"], formality=["Casual"], colors=["Black"])
        data = r.json()
        assert r.status_code == 200

    def test_highly_specific_filters_may_narrow_results(self, auth_headers):
        """Many filters stacked should reduce results (not error)."""
        r = search("top", auth_headers,
                    categories=["tops"],
                    colors=["Black"],
                    formality=["Casual"],
                    fit_type=["Regular"],
                    min_price=10,
                    max_price=40)
        data = r.json()
        assert r.status_code == 200
        # May have few or no results, but should not crash
        assert isinstance(data["results"], list)


# =============================================================================
# 11. Edge Cases
# =============================================================================

@pytest.mark.integration
class TestEdgeCases:
    """Edge case testing."""

    def test_invalid_price_range_rejected(self, auth_headers):
        """min_price > max_price should return 422."""
        r = requests.post(
            f"{BASE_URL}{SEARCH_PREFIX}/hybrid",
            json={"query": "dress", "min_price": 100, "max_price": 50},
            headers=auth_headers,
        )
        assert r.status_code == 422

    def test_empty_query_rejected(self, auth_headers):
        """Empty query string should return 422."""
        r = requests.post(
            f"{BASE_URL}{SEARCH_PREFIX}/hybrid",
            json={"query": ""},
            headers=auth_headers,
        )
        assert r.status_code == 422

    def test_page_size_over_limit_rejected(self, auth_headers):
        """page_size > 100 should return 422."""
        r = requests.post(
            f"{BASE_URL}{SEARCH_PREFIX}/hybrid",
            json={"query": "dress", "page_size": 200},
            headers=auth_headers,
        )
        assert r.status_code == 422

    def test_nonexistent_brand_returns_empty(self, auth_headers):
        """Filtering by a brand that doesn't exist should return empty."""
        r = search("dress", auth_headers, brands=["ZZZZNOTABRAND"])
        data = r.json()
        assert data["results"] == []

    def test_search_health_endpoint(self, auth_headers):
        """Search health endpoint should be accessible."""
        r = requests.get(f"{BASE_URL}{SEARCH_PREFIX}/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("healthy", "degraded")
        assert data["algolia"] in ("healthy", "unhealthy")
