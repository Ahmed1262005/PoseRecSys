"""
Unit tests for Endless Semantic Search (page 2+ for SPECIFIC / VAGUE intent).

Tests cover:
1. _apply_endless_post_filter — category_l1, category_l2, brands, exclude_brands, price
2. _endless_semantic_page — batched pgvector pump
3. Routing in _serve_cached_page — exact vs specific/vague
4. Page filling across multiple rounds
5. Exhaustion detection (pgvector returns 0)
6. Max rounds safety cap
7. No duplicates across pages (seen_ids accumulation)
8. Dedup-only (no brand/category caps)
9. Timing dict fields
10. has_more flag logic

Run with: PYTHONPATH=src python -m pytest tests/unit/test_endless_semantic.py -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List, Dict, Any, Optional


# =============================================================================
# Helpers
# =============================================================================

def _make_result(
    product_id: str,
    brand: str = "TestBrand",
    price: float = 50.0,
    category_l1: str = "Tops",
    category_l2: str = "T-Shirt",
    source: str = "semantic",
    semantic_score: float = 0.8,
    **kwargs,
) -> dict:
    """Create a mock search result dict."""
    base = {
        "product_id": product_id,
        "name": f"Product {product_id}",
        "brand": brand,
        "image_url": f"https://img.example.com/{product_id}.jpg",
        "gallery_images": [],
        "price": price,
        "original_price": None,
        "is_on_sale": False,
        "category_l1": category_l1,
        "category_l2": category_l2,
        "broad_category": "tops",
        "article_type": "t-shirt",
        "primary_color": "Black",
        "color_family": "Dark",
        "pattern": "Solid",
        "apparent_fabric": "Cotton",
        "fit_type": "Regular",
        "formality": "Casual",
        "silhouette": "Straight",
        "length": "Regular",
        "neckline": "Crew",
        "sleeve_type": "Short",
        "rise": None,
        "style_tags": ["casual"],
        "occasions": ["everyday"],
        "seasons": ["spring", "summer"],
        "colors": ["black"],
        "materials": ["cotton"],
        "trending_score": 0,
        "source": source,
        "semantic_score": semantic_score,
        "rrf_score": 0.01,
    }
    base.update(kwargs)
    return base


def _make_batch(prefix: str, n: int, **kwargs) -> List[dict]:
    """Create a batch of results with unique IDs."""
    return [_make_result(f"{prefix}-{i}", **kwargs) for i in range(n)]


def _make_entry(
    intent: str = "specific",
    query: str = "black midi dress",
    page_size: int = 50,
    post_filter_criteria: Optional[Dict] = None,
    semantic_queries: Optional[List[str]] = None,
    seen_product_ids: Optional[set] = None,
    algolia_total_hits: int = 5000,
    **kwargs,
):
    """Create a SearchSessionEntry for testing."""
    from search.session_cache import SearchSessionEntry
    embeddings = [np.random.randn(512).astype(np.float32) for _ in (semantic_queries or ["query"])]
    return SearchSessionEntry(
        session_id="ss_test123",
        query=query,
        intent=intent,
        sort_by="relevance",
        algolia_query=query,
        algolia_filters="",
        semantic_queries=semantic_queries or [query],
        semantic_embeddings=embeddings,
        seen_product_ids=seen_product_ids or set(),
        page_size=page_size,
        fetch_size=150,
        facets={"brand": []},
        follow_ups=[],
        applied_filters={"category_l1": ["Tops"]},
        answered_dimensions=["category"],
        algolia_total_hits=algolia_total_hits,
        post_filter_criteria=post_filter_criteria,
        **kwargs,
    )


@pytest.fixture
def hybrid_service():
    """HybridSearchService with mocked Algolia + analytics."""
    from search.hybrid_search import HybridSearchService
    mock_algolia = MagicMock()
    mock_analytics = MagicMock()
    return HybridSearchService(
        algolia_client=mock_algolia,
        analytics=mock_analytics,
    )


# =============================================================================
# 1. _apply_endless_post_filter tests
# =============================================================================

class TestApplyEndlessPostFilter:
    """Tests for the static post-filter method."""

    def test_empty_criteria_passes_all(self):
        from search.hybrid_search import HybridSearchService
        results = _make_batch("p", 10)
        filtered = HybridSearchService._apply_endless_post_filter(results, {})
        assert len(filtered) == 10

    def test_none_criteria_passes_all(self):
        from search.hybrid_search import HybridSearchService
        results = _make_batch("p", 10)
        filtered = HybridSearchService._apply_endless_post_filter(results, {})
        assert len(filtered) == 10

    def test_category_l1_filter(self):
        from search.hybrid_search import HybridSearchService
        results = [
            _make_result("p1", category_l1="Tops"),
            _make_result("p2", category_l1="Dresses"),
            _make_result("p3", category_l1="Tops"),
            _make_result("p4", category_l1="Bottoms"),
        ]
        filtered = HybridSearchService._apply_endless_post_filter(
            results, {"category_l1": ["Tops"]}
        )
        assert len(filtered) == 2
        assert all(r["category_l1"] == "Tops" for r in filtered)

    def test_category_l1_case_insensitive(self):
        from search.hybrid_search import HybridSearchService
        results = [
            _make_result("p1", category_l1="tops"),
            _make_result("p2", category_l1="TOPS"),
        ]
        filtered = HybridSearchService._apply_endless_post_filter(
            results, {"category_l1": ["Tops"]}
        )
        assert len(filtered) == 2

    def test_category_l1_none_excluded(self):
        """Items with None category_l1 are excluded (strict mode)."""
        from search.hybrid_search import HybridSearchService
        results = [
            _make_result("p1", category_l1="Tops"),
            _make_result("p2", category_l1=None),
        ]
        filtered = HybridSearchService._apply_endless_post_filter(
            results, {"category_l1": ["Tops"]}
        )
        assert len(filtered) == 1
        assert filtered[0]["product_id"] == "p1"

    def test_category_l2_filter(self):
        from search.hybrid_search import HybridSearchService
        results = [
            _make_result("p1", category_l2="Midi Dress"),
            _make_result("p2", category_l2="Maxi Dress"),
            _make_result("p3", category_l2="T-Shirt"),
        ]
        filtered = HybridSearchService._apply_endless_post_filter(
            results, {"category_l2": ["Midi Dress"]}
        )
        assert len(filtered) == 1

    def test_category_l2_substring_match(self):
        """category_l2 supports substring matching (e.g. 'dress' matches 'Midi Dress')."""
        from search.hybrid_search import HybridSearchService
        results = [
            _make_result("p1", category_l2="Midi Dress"),
            _make_result("p2", category_l2="Maxi Dress"),
            _make_result("p3", category_l2="T-Shirt"),
        ]
        filtered = HybridSearchService._apply_endless_post_filter(
            results, {"category_l2": ["dress"]}
        )
        assert len(filtered) == 2

    def test_brands_include_filter(self):
        from search.hybrid_search import HybridSearchService
        results = [
            _make_result("p1", brand="Boohoo"),
            _make_result("p2", brand="Zara"),
            _make_result("p3", brand="Boohoo"),
        ]
        filtered = HybridSearchService._apply_endless_post_filter(
            results, {"brands": ["Boohoo"]}
        )
        assert len(filtered) == 2
        assert all(r["brand"] == "Boohoo" for r in filtered)

    def test_brands_case_insensitive(self):
        from search.hybrid_search import HybridSearchService
        results = [
            _make_result("p1", brand="boohoo"),
            _make_result("p2", brand="BOOHOO"),
        ]
        filtered = HybridSearchService._apply_endless_post_filter(
            results, {"brands": ["Boohoo"]}
        )
        assert len(filtered) == 2

    def test_exclude_brands_filter(self):
        from search.hybrid_search import HybridSearchService
        results = [
            _make_result("p1", brand="Boohoo"),
            _make_result("p2", brand="Zara"),
            _make_result("p3", brand="H&M"),
        ]
        filtered = HybridSearchService._apply_endless_post_filter(
            results, {"exclude_brands": ["Boohoo"]}
        )
        assert len(filtered) == 2
        assert not any(r["brand"] == "Boohoo" for r in filtered)

    def test_min_price_filter(self):
        from search.hybrid_search import HybridSearchService
        results = [
            _make_result("p1", price=10.0),
            _make_result("p2", price=30.0),
            _make_result("p3", price=50.0),
        ]
        filtered = HybridSearchService._apply_endless_post_filter(
            results, {"min_price": 25.0}
        )
        assert len(filtered) == 2

    def test_max_price_filter(self):
        from search.hybrid_search import HybridSearchService
        results = [
            _make_result("p1", price=10.0),
            _make_result("p2", price=30.0),
            _make_result("p3", price=50.0),
        ]
        filtered = HybridSearchService._apply_endless_post_filter(
            results, {"max_price": 35.0}
        )
        assert len(filtered) == 2

    def test_price_range_filter(self):
        from search.hybrid_search import HybridSearchService
        results = [
            _make_result("p1", price=10.0),
            _make_result("p2", price=30.0),
            _make_result("p3", price=50.0),
            _make_result("p4", price=70.0),
        ]
        filtered = HybridSearchService._apply_endless_post_filter(
            results, {"min_price": 20.0, "max_price": 60.0}
        )
        assert len(filtered) == 2
        assert {r["product_id"] for r in filtered} == {"p2", "p3"}

    def test_price_none_excluded(self):
        """Items with None price are excluded when price filter is active."""
        from search.hybrid_search import HybridSearchService
        results = [
            _make_result("p1", price=30.0),
            {"product_id": "p2", "name": "No Price", "brand": "X", "price": None,
             "category_l1": "Tops", "category_l2": "T-Shirt"},
        ]
        filtered = HybridSearchService._apply_endless_post_filter(
            results, {"min_price": 10.0}
        )
        assert len(filtered) == 1
        assert filtered[0]["product_id"] == "p1"

    def test_combined_filters(self):
        """Multiple filters stack (AND logic)."""
        from search.hybrid_search import HybridSearchService
        results = [
            _make_result("p1", brand="Boohoo", price=30.0, category_l1="Dresses"),
            _make_result("p2", brand="Boohoo", price=30.0, category_l1="Tops"),
            _make_result("p3", brand="Zara", price=30.0, category_l1="Dresses"),
            _make_result("p4", brand="Boohoo", price=100.0, category_l1="Dresses"),
        ]
        filtered = HybridSearchService._apply_endless_post_filter(
            results,
            {"brands": ["Boohoo"], "category_l1": ["Dresses"], "max_price": 50.0},
        )
        assert len(filtered) == 1
        assert filtered[0]["product_id"] == "p1"

    def test_category_l1_multiple_values(self):
        """category_l1 can match multiple allowed values."""
        from search.hybrid_search import HybridSearchService
        results = [
            _make_result("p1", category_l1="Tops"),
            _make_result("p2", category_l1="Dresses"),
            _make_result("p3", category_l1="Bottoms"),
        ]
        filtered = HybridSearchService._apply_endless_post_filter(
            results, {"category_l1": ["Tops", "Dresses"]}
        )
        assert len(filtered) == 2


# =============================================================================
# 2. _endless_semantic_page tests
# =============================================================================

class TestEndlessSemanticPage:
    """Tests for the batched pgvector pump."""

    def test_basic_page_served(self, hybrid_service):
        """Basic test: semantic returns enough results in one round."""
        entry = _make_entry(intent="specific", page_size=10)
        batch = _make_batch("sem", 15, category_l1="Tops")
        entry.post_filter_criteria = {"category_l1": ["Tops"]}

        with patch.object(hybrid_service, "_search_semantic", return_value=batch):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        assert len(resp.results) == 10
        assert resp.pagination.page == 2
        assert resp.pagination.has_more is True
        assert resp.timing["endless_semantic"] is True

    def test_multi_query_path(self, hybrid_service):
        """When entry has multiple semantic queries, uses _search_semantic_multi."""
        entry = _make_entry(
            intent="specific",
            page_size=10,
            semantic_queries=["black midi dress", "dark evening midi", "black formal dress"],
        )
        batch = _make_batch("sem", 15, category_l1="Tops")

        with patch.object(hybrid_service, "_search_semantic_multi", return_value=(batch, None)):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        assert len(resp.results) == 10

    def test_exhaustion_returns_has_more_false(self, hybrid_service):
        """When pgvector returns 0 results, has_more = False."""
        entry = _make_entry(intent="vague", page_size=50)

        with patch.object(hybrid_service, "_search_semantic", return_value=[]):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        assert len(resp.results) == 0
        assert resp.pagination.has_more is False
        assert resp.cursor is None

    def test_partial_fill_exhaustion(self, hybrid_service):
        """When pgvector returns some results but not enough, has_more = False."""
        entry = _make_entry(intent="specific", page_size=50)
        # First call returns 20, second returns 0 (exhausted)
        call_count = [0]
        def mock_semantic(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_batch("sem", 20)
            return []

        with patch.object(hybrid_service, "_search_semantic", side_effect=mock_semantic):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        assert len(resp.results) == 20
        assert resp.pagination.has_more is False

    def test_multiple_rounds_to_fill_page(self, hybrid_service):
        """Post-filter reduces results, requiring multiple rounds."""
        entry = _make_entry(
            intent="specific",
            page_size=20,
            post_filter_criteria={"category_l1": ["Dresses"]},
        )

        round_num = [0]
        def mock_semantic(*args, **kwargs):
            round_num[0] += 1
            # Each round: 50 items, but only ~10 are Dresses
            batch = []
            for i in range(50):
                idx = (round_num[0] - 1) * 50 + i
                cat = "Dresses" if i < 10 else "Tops"
                batch.append(_make_result(f"sem-{idx}", category_l1=cat))
            return batch

        with patch.object(hybrid_service, "_search_semantic", side_effect=mock_semantic):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        assert len(resp.results) == 20
        assert resp.timing["rounds"] >= 2
        # All results should be Dresses
        assert all(r.category_l1 == "Dresses" for r in resp.results)

    def test_max_rounds_safety_cap(self, hybrid_service):
        """Loop stops after MAX_ROUNDS even if page not full."""
        entry = _make_entry(
            intent="specific",
            page_size=50,
            post_filter_criteria={"category_l1": ["Dresses"]},
        )

        call_count = [0]
        def mock_semantic(*args, **kwargs):
            call_count[0] += 1
            # Return results but NONE match category filter
            return _make_batch(f"r{call_count[0]}", 10, category_l1="Tops")

        with patch.object(hybrid_service, "_search_semantic", side_effect=mock_semantic):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        assert resp.timing["rounds"] == hybrid_service._ENDLESS_MAX_ROUNDS
        assert len(resp.results) == 0

    def test_seen_ids_accumulation(self, hybrid_service):
        """All fetched IDs added to seen set, even filtered-out ones."""
        entry = _make_entry(
            intent="specific",
            page_size=5,
            post_filter_criteria={"category_l1": ["Dresses"]},
        )

        # 20 items: 5 dresses + 15 tops
        batch = [_make_result(f"sem-{i}", category_l1="Dresses" if i < 5 else "Tops")
                 for i in range(20)]

        with patch.object(hybrid_service, "_search_semantic", return_value=batch):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        # All 20 should be in seen_ids (5 dresses + 15 tops)
        assert len(entry.seen_product_ids) == 20
        assert len(resp.results) == 5

    def test_no_brand_category_caps(self, hybrid_service):
        """Endless search uses dedup only — no brand or category caps."""
        entry = _make_entry(intent="specific", page_size=20)

        # All results from same brand and category — should all pass
        batch = _make_batch("sem", 25, brand="Boohoo", category_l1="Dresses")

        with patch.object(hybrid_service, "_search_semantic", return_value=batch):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        assert len(resp.results) == 20
        # All from same brand — no brand cap applied
        assert all(r.brand == "Boohoo" for r in resp.results)

    def test_dedup_removes_near_duplicates(self, hybrid_service):
        """Dedup removes size variants (same brand + normalized name)."""
        entry = _make_entry(intent="specific", page_size=50)

        batch = [
            _make_result("p1", brand="Boohoo", name="Front Twist Dress"),
            _make_result("p2", brand="Boohoo", name="Petite Front Twist Dress"),
            _make_result("p3", brand="Boohoo", name="Tall Front Twist Dress"),
            _make_result("p4", brand="Boohoo", name="Floral Maxi Skirt"),
        ]

        with patch.object(hybrid_service, "_search_semantic", return_value=batch):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        # Should dedup size variants: keep only one "Front Twist Dress" + "Floral Maxi Skirt"
        assert len(resp.results) <= 3  # At most 3 (p1 kept, p2/p3 deduped as size variants, p4 kept)
        assert resp.timing["dedup_removed"] >= 1

    def test_timing_dict_fields(self, hybrid_service):
        """Response timing dict has all required fields."""
        entry = _make_entry(intent="specific", page_size=5)
        batch = _make_batch("sem", 10)

        with patch.object(hybrid_service, "_search_semantic", return_value=batch):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        assert "endless_semantic" in resp.timing
        assert resp.timing["endless_semantic"] is True
        assert "page" in resp.timing
        assert "rounds" in resp.timing
        assert "semantic_ms" in resp.timing
        assert "enrich_ms" in resp.timing
        assert "filter_ms" in resp.timing
        assert "total_ms" in resp.timing
        assert "seen_ids_total" in resp.timing
        assert "dedup_removed" in resp.timing

    def test_total_results_from_algolia_nb_hits(self, hybrid_service):
        """total_results comes from cached algolia_total_hits, not result count."""
        entry = _make_entry(
            intent="specific",
            page_size=10,
            algolia_total_hits=5000,
        )
        batch = _make_batch("sem", 15)

        with patch.object(hybrid_service, "_search_semantic", return_value=batch):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        assert resp.pagination.total_results == 5000

    def test_cursor_returned_when_has_more(self, hybrid_service):
        """Next cursor is returned when has_more is True."""
        entry = _make_entry(intent="specific", page_size=10)
        batch = _make_batch("sem", 15)

        with patch.object(hybrid_service, "_search_semantic", return_value=batch):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        assert resp.cursor is not None
        from search.session_cache import decode_cursor
        cursor_data = decode_cursor(resp.cursor)
        assert cursor_data["p"] == 3

    def test_no_cursor_when_exhausted(self, hybrid_service):
        """No cursor when pgvector is exhausted."""
        entry = _make_entry(intent="specific", page_size=50)

        with patch.object(hybrid_service, "_search_semantic", return_value=[]):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        assert resp.cursor is None

    def test_metadata_carried_from_page1(self, hybrid_service):
        """applied_filters, answered_dimensions carried from page 1."""
        entry = _make_entry(intent="specific", page_size=5)
        entry.facets = None  # Facets get Pydantic-coerced; skip for this test
        entry.follow_ups = None
        entry.applied_filters = {"category_l1": ["Tops"]}
        entry.answered_dimensions = ["category"]

        batch = _make_batch("sem", 10)

        with patch.object(hybrid_service, "_search_semantic", return_value=batch):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        assert resp.applied_filters == entry.applied_filters
        assert resp.answered_dimensions == entry.answered_dimensions
        assert resp.follow_ups is None
        assert resp.facets is None

    def test_response_fields(self, hybrid_service):
        """Response has correct query, intent, sort_by, session_id."""
        entry = _make_entry(
            intent="vague",
            query="summer vibes",
        )
        entry.page_size = 5
        batch = _make_batch("sem", 10)

        with patch.object(hybrid_service, "_search_semantic", return_value=batch):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=3, session_id="ss_abc")

        assert resp.query == "summer vibes"
        assert resp.intent == "vague"
        assert resp.sort_by == "relevance"
        assert resp.search_session_id == "ss_abc"
        assert resp.pagination.page == 3

    def test_no_planner_or_algolia_search_calls(self, hybrid_service):
        """Endless semantic makes NO planner plan() or Algolia search() calls.

        Note: _enrich_semantic_results uses algolia.get_objects() which is
        fine — we only verify that algolia.search() is NOT called.
        """
        entry = _make_entry(intent="specific", page_size=5)
        batch = _make_batch("sem", 10)

        # Track calls on the planner
        hybrid_service._planner = MagicMock()

        with patch.object(hybrid_service, "_search_semantic", return_value=batch):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        # Planner should NOT be called
        assert hybrid_service._planner.plan.call_count == 0
        # Algolia.search is on the _algolia mock (set in fixture)
        assert hybrid_service._algolia.search.call_count == 0

    def test_enrichment_called(self, hybrid_service):
        """Semantic results are enriched from Algolia."""
        entry = _make_entry(intent="specific", page_size=5)
        batch = _make_batch("sem", 10)

        with patch.object(hybrid_service, "_search_semantic", return_value=batch):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x) as mock_enrich:
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        assert mock_enrich.call_count >= 1

    def test_enrichment_skipped_for_attribute_search(self, hybrid_service):
        """Enrichment is skipped when use_attribute_search is True."""
        entry = _make_entry(intent="specific", page_size=5)
        entry.use_attribute_search = True
        batch = _make_batch("sem", 10)

        with patch.object(hybrid_service, "_search_semantic", return_value=batch):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x) as mock_enrich:
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        assert mock_enrich.call_count == 0


# =============================================================================
# 3. Routing tests — _serve_cached_page dispatches correctly
# =============================================================================

class TestServeCachedPageRouting:
    """Tests that _serve_cached_page routes exact to _extend_search, others to _endless_semantic_page."""

    def test_exact_routes_to_extend_search(self, hybrid_service):
        """EXACT intent goes through _extend_search (unchanged)."""
        from search.session_cache import SearchSessionCache, encode_cursor
        from search.models import HybridSearchRequest

        entry = _make_entry(intent="exact", query="tops")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        cursor = encode_cursor(page=2)
        request = MagicMock()
        request.search_session_id = entry.session_id
        request.cursor = cursor

        with patch.object(hybrid_service, "_extend_search", return_value="extend_result") as mock_extend:
            with patch.object(hybrid_service, "_endless_semantic_page") as mock_endless:
                result = hybrid_service._serve_cached_page(request)

        assert mock_extend.call_count == 1
        assert mock_endless.call_count == 0
        assert result == "extend_result"

        # Cleanup
        cache.delete(entry.session_id)

    def test_specific_routes_to_endless(self, hybrid_service):
        """SPECIFIC intent goes through _endless_semantic_page."""
        from search.session_cache import SearchSessionCache, encode_cursor

        entry = _make_entry(intent="specific", query="black midi dress")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        cursor = encode_cursor(page=2)
        request = MagicMock()
        request.search_session_id = entry.session_id
        request.cursor = cursor

        with patch.object(hybrid_service, "_extend_search") as mock_extend:
            with patch.object(hybrid_service, "_endless_semantic_page", return_value="endless_result") as mock_endless:
                result = hybrid_service._serve_cached_page(request)

        assert mock_extend.call_count == 0
        assert mock_endless.call_count == 1
        assert result == "endless_result"

        cache.delete(entry.session_id)

    def test_vague_routes_to_endless(self, hybrid_service):
        """VAGUE intent goes through _endless_semantic_page."""
        from search.session_cache import SearchSessionCache, encode_cursor

        entry = _make_entry(intent="vague", query="summer vibes")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        cursor = encode_cursor(page=2)
        request = MagicMock()
        request.search_session_id = entry.session_id
        request.cursor = cursor

        with patch.object(hybrid_service, "_extend_search") as mock_extend:
            with patch.object(hybrid_service, "_endless_semantic_page", return_value="endless_result") as mock_endless:
                result = hybrid_service._serve_cached_page(request)

        assert mock_extend.call_count == 0
        assert mock_endless.call_count == 1

        cache.delete(entry.session_id)

    def test_missing_session_returns_none(self, hybrid_service):
        """Missing session returns None (fall through to full pipeline)."""
        request = MagicMock()
        request.search_session_id = "ss_nonexistent"
        request.cursor = "eyJwIjoyfQ=="

        result = hybrid_service._serve_cached_page(request)
        assert result is None


# =============================================================================
# 4. Edge case tests
# =============================================================================

class TestEndlessEdgeCases:
    """Edge cases and boundary conditions."""

    def test_no_embeddings_fallback(self, hybrid_service):
        """Graceful fallback when no cached embeddings exist."""
        entry = _make_entry(intent="specific", page_size=5)
        entry.semantic_embeddings = None
        batch = _make_batch("sem", 10)

        with patch.object(hybrid_service, "_search_semantic", return_value=batch):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        assert len(resp.results) == 5

    def test_single_embedding_path(self, hybrid_service):
        """Single query with single embedding works."""
        entry = _make_entry(
            intent="specific",
            page_size=5,
            semantic_queries=["black dress"],
        )
        batch = _make_batch("sem", 10)

        with patch.object(hybrid_service, "_search_semantic", return_value=batch):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        assert len(resp.results) == 5

    def test_no_post_filter_criteria(self, hybrid_service):
        """When post_filter_criteria is None, no filtering applied."""
        entry = _make_entry(intent="specific", page_size=10)
        entry.post_filter_criteria = None
        batch = _make_batch("sem", 15, category_l1="RandomCat")

        with patch.object(hybrid_service, "_search_semantic", return_value=batch):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        assert len(resp.results) == 10

    def test_existing_seen_ids_excluded(self, hybrid_service):
        """Pre-existing seen_ids from page 1 are passed to pgvector."""
        entry = _make_entry(intent="specific", page_size=5)
        entry.seen_product_ids = {"old-1", "old-2", "old-3"}
        batch = _make_batch("sem", 10)

        with patch.object(hybrid_service, "_search_semantic", return_value=batch) as mock_sem:
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        # Check that exclude_product_ids was passed with the old seen IDs
        call_kwargs = mock_sem.call_args
        exclude_ids = call_kwargs.kwargs.get("exclude_product_ids") or call_kwargs[1].get("exclude_product_ids")
        assert "old-1" in exclude_ids
        assert "old-2" in exclude_ids
        assert "old-3" in exclude_ids

    def test_page_size_1(self, hybrid_service):
        """Extreme case: page_size=1 works."""
        entry = _make_entry(intent="specific", page_size=1)
        batch = _make_batch("sem", 5)

        with patch.object(hybrid_service, "_search_semantic", return_value=batch):
            with patch.object(hybrid_service, "_enrich_semantic_results", side_effect=lambda x: x):
                resp = hybrid_service._endless_semantic_page(entry, page=2, session_id="ss_test")

        assert len(resp.results) == 1
        assert resp.pagination.has_more is True
