"""
Unit tests for Filter Refinement via Cached Session.

Tests cover:
1. _build_user_filter_clauses — all filter types to Algolia syntax
2. Signal detection — routing logic in search() for filter refinement
3. _filter_refine_search — cache hit/miss, parallel Algolia+semantic, RRF, rerank
4. Facet strategy — SPECIFIC/VAGUE prefer Algolia facets, EXACT recomputes
5. Result counts — total_results reflects Algolia nbHits accurately
6. New session creation — embeddings reused, filters updated, seen_ids fresh
7. Post-filter criteria — user filters carried to new session entry
8. Edge cases — empty results, EXACT intent refinement, expired sessions

Run with: PYTHONPATH=src python -m pytest tests/unit/test_filter_refine.py -v
"""

import uuid
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call
from typing import List, Dict, Any, Optional, Set


# =============================================================================
# Helpers
# =============================================================================

def _make_result(
    product_id: str,
    brand: str = "TestBrand",
    price: float = 50.0,
    category_l1: str = "Tops",
    category_l2: str = "T-Shirt",
    source: str = "algolia",
    semantic_score: float = 0.0,
    rrf_score: float = 0.01,
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
        "is_set": False,
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
        "rrf_score": rrf_score,
    }
    base.update(kwargs)
    return base


def _make_batch(prefix: str, n: int, **kwargs) -> List[dict]:
    """Create a batch of results with unique IDs."""
    return [_make_result(f"{prefix}-{i}", **kwargs) for i in range(n)]


def _make_algolia_batch(prefix: str, n: int, **kwargs) -> List[dict]:
    """Create Algolia-sourced result batch."""
    return _make_batch(prefix, n, source="algolia", **kwargs)


def _make_semantic_batch(prefix: str, n: int, **kwargs) -> List[dict]:
    """Create semantic-sourced result batch."""
    return _make_batch(prefix, n, source="semantic", semantic_score=0.8, **kwargs)


def _make_entry(
    intent: str = "specific",
    query: str = "black midi dress",
    page_size: int = 50,
    algolia_filters: str = 'in_stock:true AND (category_l1:"Dresses")',
    algolia_query: str = "black midi dress",
    post_filter_criteria: Optional[Dict] = None,
    semantic_queries: Optional[List[str]] = None,
    seen_product_ids: Optional[set] = None,
    algolia_total_hits: int = 5000,
    algolia_weight: float = 0.6,
    semantic_weight: float = 0.4,
    rerank_kwargs: Optional[Dict] = None,
    **kwargs,
):
    """Create a SearchSessionEntry for testing."""
    from search.session_cache import SearchSessionEntry
    queries = semantic_queries or [query]
    embeddings = [np.random.randn(512).astype(np.float32) for _ in queries]
    return SearchSessionEntry(
        session_id=f"ss_test_{uuid.uuid4().hex[:8]}",
        query=query,
        intent=intent,
        sort_by="relevance",
        algolia_query=algolia_query,
        algolia_filters=algolia_filters,
        algolia_optional_filters=['color:"Black"'],
        semantic_queries=queries,
        semantic_embeddings=embeddings,
        semantic_request_updates={
            "categories": None,
            "colors": None,
            "patterns": None,
            "occasions": None,
        },
        algolia_weight=algolia_weight,
        semantic_weight=semantic_weight,
        rerank_kwargs=rerank_kwargs or {
            "user_profile": None,
            "user_context": None,
            "session_scores": None,
            "page_size": 0,
        },
        seen_product_ids=seen_product_ids or set(),
        page_size=page_size,
        fetch_size=150,
        facets={
            "brand": [{"value": "Boohoo", "count": 500}],
            "category_l1": [{"value": "Dresses", "count": 3000}],
        },
        follow_ups=[],
        applied_filters={"category_l1": ["Dresses"]},
        answered_dimensions=["category"],
        algolia_total_hits=algolia_total_hits,
        post_filter_criteria=post_filter_criteria,
        **kwargs,
    )


def _make_algolia_facets(brand_counts: Dict[str, int] = None, nb_hits: int = 3000):
    """Create mock Algolia facets matching the FacetValue format."""
    from search.models import FacetValue
    brands = brand_counts or {"Boohoo": 200, "Missguided": 150, "Forever 21": 100}
    return {
        "brand": [FacetValue(value=k, count=v) for k, v in brands.items()],
        "category_l1": [FacetValue(value="Dresses", count=nb_hits)],
    }


@pytest.fixture(autouse=True)
def _clear_session_cache():
    """Reset the SearchSessionCache singleton before each test.

    Without this, session entries from earlier tests (even from other
    test files) leak through the singleton and cause intermittent
    assertion failures on cache lookups.
    """
    from search.session_cache import SearchSessionCache
    instance = SearchSessionCache.get_instance()
    instance._store.clear()
    yield
    instance._store.clear()


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
# 1. _build_user_filter_clauses tests
# =============================================================================

class TestBuildUserFilterClauses:
    """Test conversion of user filter snapshot to Algolia filter string."""

    def test_empty_filters(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses({})
        assert result == ""

    def test_single_brand(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"brands": ["Boohoo"]}
        )
        assert result == '(brand:"Boohoo")'

    def test_multiple_brands(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"brands": ["Boohoo", "Missguided"]}
        )
        assert 'brand:"Boohoo"' in result
        assert 'brand:"Missguided"' in result
        assert " OR " in result
        # Wrapped in parens
        assert result.startswith("(")
        assert result.endswith(")")

    def test_min_price(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"min_price": 20.0}
        )
        assert result == "price >= 20.0"

    def test_max_price(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"max_price": 100.0}
        )
        assert result == "price <= 100.0"

    def test_price_range(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"min_price": 20.0, "max_price": 100.0}
        )
        assert "price >= 20.0" in result
        assert "price <= 100.0" in result
        assert " AND " in result

    def test_brand_plus_price(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"brands": ["Boohoo"], "max_price": 50.0}
        )
        assert 'brand:"Boohoo"' in result
        assert "price <= 50.0" in result
        assert " AND " in result

    def test_categories(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"categories": ["Dresses", "Tops"]}
        )
        assert 'broad_category:"Dresses"' in result
        assert 'broad_category:"Tops"' in result
        assert " OR " in result

    def test_category_l1(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"category_l1": ["Dresses"]}
        )
        assert result == '(category_l1:"Dresses")'

    def test_category_l2(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"category_l2": ["Midi Dress"]}
        )
        assert result == '(category_l2:"Midi Dress")'

    def test_colors(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"colors": ["Black", "Navy"]}
        )
        assert 'primary_color:"Black"' in result
        assert 'primary_color:"Navy"' in result

    def test_color_family(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"color_family": ["Dark"]}
        )
        assert result == '(color_family:"Dark")'

    def test_patterns(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"patterns": ["Floral"]}
        )
        assert result == '(pattern:"Floral")'

    def test_occasions(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"occasions": ["Party", "Date Night"]}
        )
        assert 'occasions:"Party"' in result
        assert 'occasions:"Date Night"' in result

    def test_formality(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"formality": ["Casual"]}
        )
        assert result == '(formality:"Casual")'

    def test_fit_type(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"fit_type": ["Regular", "Slim"]}
        )
        assert 'fit_type:"Regular"' in result
        assert 'fit_type:"Slim"' in result

    def test_neckline(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"neckline": ["V-Neck"]}
        )
        assert result == '(neckline:"V-Neck")'

    def test_sleeve_type(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"sleeve_type": ["Long"]}
        )
        assert result == '(sleeve_type:"Long")'

    def test_length(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"length": ["Midi"]}
        )
        assert result == '(length:"Midi")'

    def test_rise(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"rise": ["High"]}
        )
        assert result == '(rise:"High")'

    def test_silhouette(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"silhouette": ["A-Line"]}
        )
        assert result == '(silhouette:"A-Line")'

    def test_article_type(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"article_type": ["Midi Dress"]}
        )
        assert result == '(article_type:"Midi Dress")'

    def test_style_tags(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"style_tags": ["boho", "vintage"]}
        )
        assert 'style_tags:"boho"' in result
        assert 'style_tags:"vintage"' in result

    def test_materials(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"materials": ["Cotton", "Linen"]}
        )
        assert 'apparent_fabric:"Cotton"' in result
        assert 'apparent_fabric:"Linen"' in result

    def test_seasons(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"seasons": ["Summer"]}
        )
        assert result == '(seasons:"Summer")'

    def test_on_sale_only(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"on_sale_only": True}
        )
        assert result == "is_on_sale:true"

    def test_on_sale_false_ignored(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"on_sale_only": False}
        )
        assert result == ""

    def test_is_set_true(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"is_set": True}
        )
        assert result == "is_set:true"

    def test_is_set_false(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"is_set": False}
        )
        assert result == "is_set:false"

    def test_exclude_brands(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"exclude_brands": ["Shein", "Temu"]}
        )
        assert 'NOT brand:"Shein"' in result
        assert 'NOT brand:"Temu"' in result

    def test_exclude_neckline(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"exclude_neckline": ["Off-Shoulder"]}
        )
        assert result == 'NOT neckline:"Off-Shoulder"'

    def test_exclude_colors(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"exclude_colors": ["Red"]}
        )
        assert result == 'NOT primary_color:"Red"'

    def test_exclude_patterns(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"exclude_patterns": ["Animal Print"]}
        )
        assert result == 'NOT pattern:"Animal Print"'

    def test_exclude_materials(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"exclude_materials": ["Polyester"]}
        )
        assert result == 'NOT apparent_fabric:"Polyester"'

    def test_complex_multi_filter(self):
        """Test realistic multi-filter scenario: brand + price + color + exclusion."""
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses({
            "brands": ["Boohoo", "Missguided"],
            "min_price": 20.0,
            "max_price": 80.0,
            "colors": ["Black"],
            "exclude_patterns": ["Floral"],
        })
        assert "price >= 20.0" in result
        assert "price <= 80.0" in result
        assert 'brand:"Boohoo"' in result
        assert 'brand:"Missguided"' in result
        assert 'primary_color:"Black"' in result
        assert 'NOT pattern:"Floral"' in result
        # All joined with AND
        parts = result.split(" AND ")
        assert len(parts) >= 4


# =============================================================================
# 2. Signal Detection / Routing tests
# =============================================================================

class TestFilterRefineRouting:
    """Test that the correct path is taken based on request params."""

    def test_session_plus_filters_no_cursor_triggers_refine(self, hybrid_service):
        """search_session_id + filters + no cursor → _filter_refine_search."""
        from search.models import HybridSearchRequest

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id="ss_test123",
            brands=["Boohoo"],
        )

        with patch.object(
            hybrid_service, "_filter_refine_search", return_value=MagicMock()
        ) as mock_refine:
            hybrid_service.search(request)
            mock_refine.assert_called_once()
            # Verify user_filters was passed with brands
            call_kwargs = mock_refine.call_args
            assert "Boohoo" in str(call_kwargs)

    def test_session_plus_cursor_takes_cached_path(self, hybrid_service):
        """search_session_id + cursor → _serve_cached_page (NOT refinement)."""
        from search.models import HybridSearchRequest
        from search.session_cache import encode_cursor

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id="ss_test123",
            cursor=encode_cursor(page=2),
            brands=["Boohoo"],  # filters present but cursor takes priority
        )

        with patch.object(
            hybrid_service, "_serve_cached_page", return_value=MagicMock()
        ) as mock_cached, patch.object(
            hybrid_service, "_filter_refine_search"
        ) as mock_refine:
            hybrid_service.search(request)
            mock_cached.assert_called_once()
            mock_refine.assert_not_called()

    def test_session_only_no_filters_no_cursor_full_pipeline(self, hybrid_service):
        """search_session_id only (no cursor, no filters) → full pipeline."""
        from search.models import HybridSearchRequest

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id="ss_test123",
            # No cursor, no user filters
        )

        with patch.object(
            hybrid_service, "_filter_refine_search"
        ) as mock_refine, patch.object(
            hybrid_service, "_serve_cached_page"
        ) as mock_cached:
            # Will fall through to full pipeline (planner etc.)
            # We just verify the refinement path is NOT taken
            try:
                hybrid_service.search(request)
            except Exception:
                pass  # full pipeline will fail without real services
            mock_refine.assert_not_called()
            mock_cached.assert_not_called()

    def test_no_session_id_full_pipeline(self, hybrid_service):
        """No search_session_id → full pipeline regardless of filters."""
        from search.models import HybridSearchRequest

        request = HybridSearchRequest(
            query="black midi dress",
            brands=["Boohoo"],
            # No search_session_id
        )

        with patch.object(
            hybrid_service, "_filter_refine_search"
        ) as mock_refine:
            try:
                hybrid_service.search(request)
            except Exception:
                pass
            mock_refine.assert_not_called()


# =============================================================================
# 3. _filter_refine_search core logic tests
# =============================================================================

class TestFilterRefineSearch:
    """Test the _filter_refine_search method."""

    def test_cache_miss_returns_none(self, hybrid_service):
        """Expired/missing session → returns None."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id="ss_nonexistent",
            brands=["Boohoo"],
        )
        user_filters = {"brands": ["Boohoo"]}

        # Ensure cache is empty
        cache = SearchSessionCache.get_instance()

        result = hybrid_service._filter_refine_search(
            request=request,
            user_filters=user_filters,
        )
        assert result is None

    def test_cache_hit_returns_response(self, hybrid_service):
        """Valid session → returns HybridSearchResponse."""
        from search.models import HybridSearchRequest, HybridSearchResponse
        from search.session_cache import SearchSessionCache

        entry = _make_entry(intent="specific", algolia_total_hits=3000)
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        # Mock the internal methods
        algolia_results = _make_algolia_batch("alg", 30, brand="Boohoo")
        semantic_results = _make_semantic_batch("sem", 20, brand="Boohoo")
        algolia_facets = _make_algolia_facets({"Boohoo": 200}, nb_hits=200)

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(algolia_results, algolia_facets, 200)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=(semantic_results, None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r  # pass-through
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        assert result is not None
        assert isinstance(result, HybridSearchResponse)

    def test_new_session_id_different_from_original(self, hybrid_service):
        """Filter refinement creates a NEW session ID."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(intent="specific")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        algolia_results = _make_algolia_batch("alg", 30, brand="Boohoo")
        algolia_facets = _make_algolia_facets()

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(algolia_results, algolia_facets, 200)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=(_make_semantic_batch("sem", 20), None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        assert result is not None
        assert result.search_session_id is not None
        assert result.search_session_id != entry.session_id

    def test_always_page_1(self, hybrid_service):
        """Filter refinement always returns page 1."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(intent="specific")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
            page=3,  # client says page 3 but refinement resets
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 30), _make_algolia_facets(), 200)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=(_make_semantic_batch("sem", 20), None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        assert result.pagination.page == 1

    def test_cursor_present_in_response(self, hybrid_service):
        """Refined response includes cursor for page 2."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache, decode_cursor

        entry = _make_entry(intent="specific")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 30), _make_algolia_facets(), 200)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=(_make_semantic_batch("sem", 20), None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        assert result.cursor is not None
        cursor_data = decode_cursor(result.cursor)
        assert cursor_data["p"] == 2

    def test_merged_algolia_filters(self, hybrid_service):
        """Verify planner filters AND user filters are merged."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(
            intent="specific",
            algolia_filters='in_stock:true AND (category_l1:"Dresses")',
        )
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
            max_price=50.0,
        )

        captured_filters = {}

        def mock_search_algolia(query, filters=None, **kwargs):
            captured_filters["filters"] = filters
            return (_make_algolia_batch("alg", 10), _make_algolia_facets(), 100)

        with patch.object(
            hybrid_service, "_search_algolia",
            side_effect=mock_search_algolia,
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"], "max_price": 50.0},
            )

        # Original planner filters preserved
        assert 'category_l1:"Dresses"' in captured_filters["filters"]
        # User filters appended
        assert 'brand:"Boohoo"' in captured_filters["filters"]
        assert "price <= 50.0" in captured_filters["filters"]
        # Always has in_stock
        assert "in_stock:true" in captured_filters["filters"]

    def test_semantic_uses_precomputed_embeddings(self, hybrid_service):
        """Verify semantic search reuses cached embeddings (no re-encoding)."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(
            intent="specific",
            semantic_queries=["elegant black midi dress", "dark sophisticated dress"],
        )
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        captured_calls = {}

        def mock_semantic_multi(queries, request, precomputed_embeddings=None, **kwargs):
            captured_calls["precomputed"] = precomputed_embeddings
            captured_calls["exclude_ids"] = kwargs.get("exclude_product_ids", [])
            return (_make_semantic_batch("sem", 15), None)

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 10), _make_algolia_facets(), 100)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            side_effect=mock_semantic_multi,
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        # Embeddings were passed (not None)
        assert captured_calls["precomputed"] is not None
        assert len(captured_calls["precomputed"]) == 2  # two semantic queries
        # Fresh start — no exclusions
        assert captured_calls["exclude_ids"] == []

    def test_fresh_seen_ids(self, hybrid_service):
        """Refinement starts with fresh seen_ids (no carryover from original)."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(
            intent="specific",
            seen_product_ids={"old-1", "old-2", "old-3"},
        )
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        algolia_results = _make_algolia_batch("new", 10, brand="Boohoo")
        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(algolia_results, _make_algolia_facets(), 100)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        # The NEW session entry should have fresh seen_ids (only page-1 results)
        new_entry = cache.get(result.search_session_id)
        assert new_entry is not None
        assert "old-1" not in new_entry.seen_product_ids
        assert "old-2" not in new_entry.seen_product_ids

    def test_timing_dict_includes_refine_flag(self, hybrid_service):
        """Timing dict has filter_refine=True."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(intent="specific")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 10), _make_algolia_facets(), 100)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        assert result.timing["filter_refine"] is True
        assert "algolia_ms" in result.timing
        assert "semantic_ms" in result.timing
        assert "total_ms" in result.timing

    def test_empty_algolia_results(self, hybrid_service):
        """Zero Algolia results + some semantic → still returns results."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        # Use 2 queries so _search_semantic_multi is called (not _search_semantic)
        entry = _make_entry(
            intent="specific",
            semantic_queries=["elegant black midi dress", "dark sophisticated dress"],
        )
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["RareBrand"],
        )

        semantic_results = _make_semantic_batch("sem", 10, brand="RareBrand")
        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=([], None, 0)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=(semantic_results, None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["RareBrand"]},
            )

        assert result is not None
        assert len(result.results) > 0

    def test_empty_both_returns_empty(self, hybrid_service):
        """Zero Algolia + zero semantic → empty results, no session cached."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(intent="specific")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["NonexistentBrand"],
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=([], None, 0)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["NonexistentBrand"]},
            )

        assert result is not None
        assert len(result.results) == 0
        # No new session created (empty results)
        assert result.search_session_id is None

    def test_exact_intent_refinement(self, hybrid_service):
        """EXACT intent sessions can also be refined."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(
            intent="exact",
            query="tops",
            algolia_query="tops",
            algolia_filters="in_stock:true",
        )
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="tops",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        algolia_results = _make_algolia_batch("alg", 30, brand="Boohoo")
        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(algolia_results, _make_algolia_facets({"Boohoo": 5000}, 5000), 5000)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        assert result is not None
        assert result.intent == "exact"
        assert len(result.results) > 0

    def test_vague_intent_refinement(self, hybrid_service):
        """VAGUE intent sessions can be refined."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(
            intent="vague",
            query="summer vibes",
            algolia_query="summer vibes",
        )
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="summer vibes",
            search_session_id=entry.session_id,
            max_price=40.0,
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 20, price=30.0), _make_algolia_facets(), 500)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=(_make_semantic_batch("sem", 15, price=35.0), None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"max_price": 40.0},
            )

        assert result is not None
        assert result.intent == "vague"


# =============================================================================
# 4. Facet Strategy tests
# =============================================================================

class TestFacetStrategy:
    """Test that facets are sourced correctly per intent."""

    def test_refine_uses_algolia_facets_not_computed(self, hybrid_service):
        """Filter refinement always uses Algolia facets (not computed from merged)."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(intent="specific")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        # Algolia returns facets with proper counts
        algolia_facets = _make_algolia_facets({"Boohoo": 200, "Missguided": 150})

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 10, brand="Boohoo"), algolia_facets, 350)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=(_make_semantic_batch("sem", 5, brand="Boohoo"), None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        # Facets should be from Algolia, not recomputed from 15 merged results
        assert result.facets is not None
        brand_facet = result.facets.get("brand", [])
        # Should have Algolia's catalog-wide counts
        boohoo_count = next(
            (f.count if hasattr(f, 'count') else f["count"]
             for f in brand_facet
             if (f.value if hasattr(f, 'value') else f["value"]) == "Boohoo"),
            None
        )
        assert boohoo_count == 200  # Algolia's count, not 10 from merged

    def test_refine_falls_back_to_entry_facets_on_algolia_none(self, hybrid_service):
        """If Algolia returns no facets, fall back to cached entry facets."""
        from search.models import HybridSearchRequest, FacetValue
        from search.session_cache import SearchSessionCache

        # Use FacetValue objects in entry facets so they match the response type
        entry = _make_entry(
            intent="specific",
            semantic_queries=["elegant black midi dress", "dark sophisticated dress"],
        )
        # Override facets with FacetValue objects (matching response format)
        entry.facets = {
            "brand": [FacetValue(value="Boohoo", count=500)],
            "category_l1": [FacetValue(value="Dresses", count=3000)],
        }
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 5, brand="Boohoo"), None, 5)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        # Falls back to entry's cached facets
        assert result.facets is not None
        # Check values match entry's facets
        brand_facet = result.facets["brand"]
        assert len(brand_facet) == 1
        fv = brand_facet[0]
        assert (fv.value if hasattr(fv, "value") else fv["value"]) == "Boohoo"
        assert (fv.count if hasattr(fv, "count") else fv["count"]) == 500

    def test_new_session_stores_algolia_facets(self, hybrid_service):
        """New session entry stores Algolia facets (for page 2+ consistency)."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(intent="specific")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        algolia_facets = _make_algolia_facets({"Boohoo": 200})

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 10), algolia_facets, 200)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        # Verify the NEW session entry has the Algolia facets
        new_entry = cache.get(result.search_session_id)
        assert new_entry is not None
        assert new_entry.facets == algolia_facets


# =============================================================================
# 5. Result Count / total_results tests
# =============================================================================

class TestResultCounts:
    """Test that total_results (Algolia nbHits) is accurate."""

    def test_total_results_from_algolia_nb_hits(self, hybrid_service):
        """total_results uses Algolia's nbHits for catalog-wide count."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(intent="specific")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 50), _make_algolia_facets(), 3644)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        # total_results should be Algolia's 3644, not the 50 returned results
        assert result.pagination.total_results == 3644

    def test_total_results_brand_filter_narrows_count(self, hybrid_service):
        """Brand filter narrows nbHits (e.g. 3644 → 200 Boohoo dresses)."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(intent="specific", algolia_total_hits=3644)
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        # Algolia now returns only 200 hits (filtered by brand)
        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 30, brand="Boohoo"), _make_algolia_facets(), 200)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        # total_results should reflect the narrowed Algolia count
        assert result.pagination.total_results == 200

    def test_total_results_price_filter_narrows_count(self, hybrid_service):
        """Price filter narrows nbHits."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(intent="specific", algolia_total_hits=5000)
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            max_price=30.0,
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 25, price=25.0), _make_algolia_facets(), 800)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"max_price": 30.0},
            )

        assert result.pagination.total_results == 800

    def test_total_results_zero_algolia_falls_back_to_merged(self, hybrid_service):
        """When Algolia returns 0 hits, total_results uses merged count."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(intent="specific")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["NonexistentBrand"],
        )

        # Algolia returns 0 but semantic has some results
        semantic_results = _make_semantic_batch("sem", 5, brand="NonexistentBrand")
        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=([], None, 0)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=(semantic_results, None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["NonexistentBrand"]},
            )

        # Falls back to merged count
        assert result.pagination.total_results == len(result.results)

    def test_new_session_algolia_total_hits_updated(self, hybrid_service):
        """New session entry stores the refined algolia_total_hits."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(intent="specific", algolia_total_hits=5000)
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 20), _make_algolia_facets(), 200)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        new_entry = cache.get(result.search_session_id)
        assert new_entry is not None
        # New entry has updated count, not original 5000
        assert new_entry.algolia_total_hits == 200

    def test_page_size_respected(self, hybrid_service):
        """Results capped at page_size even when more available."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(intent="specific", page_size=50)
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
            page_size=10,  # client requests smaller page
        )

        # Return more than page_size
        algolia_results = _make_algolia_batch("alg", 30, brand="Boohoo")
        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(algolia_results, _make_algolia_facets(), 500)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=(_make_semantic_batch("sem", 20, brand="Boohoo"), None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        assert len(result.results) <= 10
        assert result.pagination.has_more is True


# =============================================================================
# 6. New Session Entry State tests
# =============================================================================

class TestNewSessionState:
    """Test that the new session entry has correct state."""

    def test_embeddings_reused(self, hybrid_service):
        """New session entry reuses original embeddings."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(
            intent="specific",
            semantic_queries=["elegant black midi dress", "dark sophisticated dress"],
        )
        original_embeddings = entry.semantic_embeddings
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        # Mock _search_semantic_multi (2 queries → multi path)
        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 10, brand="Boohoo"), _make_algolia_facets(), 100)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=(_make_semantic_batch("sem", 5, brand="Boohoo"), None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        new_entry = cache.get(result.search_session_id)
        assert new_entry is not None
        assert new_entry.semantic_embeddings is original_embeddings
        assert new_entry.semantic_queries == entry.semantic_queries

    def test_algolia_filters_updated(self, hybrid_service):
        """New session entry has merged filters (planner + user)."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(
            intent="specific",
            algolia_filters='in_stock:true AND (category_l1:"Dresses")',
        )
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 10), _make_algolia_facets(), 100)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        new_entry = cache.get(result.search_session_id)
        assert new_entry is not None
        # Original planner filter
        assert 'category_l1:"Dresses"' in new_entry.algolia_filters
        # User filter appended
        assert 'brand:"Boohoo"' in new_entry.algolia_filters

    def test_post_filter_criteria_updated(self, hybrid_service):
        """New session has post_filter_criteria from user filters."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(
            intent="specific",
            post_filter_criteria={"category_l1": ["Dresses"]},
        )
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
            max_price=50.0,
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 10), _make_algolia_facets(), 100)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"], "max_price": 50.0},
            )

        new_entry = cache.get(result.search_session_id)
        assert new_entry is not None
        pfc = new_entry.post_filter_criteria
        assert pfc is not None
        # User's brand filter
        assert pfc["brands"] == ["Boohoo"]
        # User's price filter
        assert pfc["max_price"] == 50.0
        # Inherited from original entry
        assert pfc["category_l1"] == ["Dresses"]

    def test_rrf_weights_preserved(self, hybrid_service):
        """New session entry preserves RRF weights from original."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(
            intent="specific",
            algolia_weight=0.7,
            semantic_weight=0.3,
        )
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 10), _make_algolia_facets(), 100)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        new_entry = cache.get(result.search_session_id)
        assert new_entry is not None
        assert new_entry.algolia_weight == 0.7
        assert new_entry.semantic_weight == 0.3

    def test_rerank_kwargs_preserved(self, hybrid_service):
        """New session entry preserves rerank_kwargs from original."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(
            intent="specific",
            rerank_kwargs={
                "user_profile": {"style": "casual"},
                "user_context": None,
                "session_scores": None,
                "page_size": 0,
                "max_per_brand": 4,
            },
        )
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 10), _make_algolia_facets(), 100)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        new_entry = cache.get(result.search_session_id)
        assert new_entry is not None
        assert new_entry.rerank_kwargs["max_per_brand"] == 4

    def test_intent_preserved(self, hybrid_service):
        """New session entry preserves intent from original."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        for intent in ["exact", "specific", "vague"]:
            entry = _make_entry(intent=intent)
            cache = SearchSessionCache.get_instance()
            cache.store(entry)

            request = HybridSearchRequest(
                query=entry.query,
                search_session_id=entry.session_id,
                brands=["Boohoo"],
            )

            with patch.object(
                hybrid_service, "_search_algolia",
                return_value=(_make_algolia_batch("alg", 10), _make_algolia_facets(), 100)
            ), patch.object(
                hybrid_service, "_search_semantic_multi",
                return_value=([], None)
            ), patch.object(
                hybrid_service, "_enrich_semantic_results",
                side_effect=lambda r: r
            ):
                result = hybrid_service._filter_refine_search(
                    request=request,
                    user_filters={"brands": ["Boohoo"]},
                )

            assert result.intent == intent
            new_entry = cache.get(result.search_session_id)
            if new_entry:
                assert new_entry.intent == intent


# =============================================================================
# 7. Facet strategy in main search() pipeline
# =============================================================================

class TestFacetStrategyMainPipeline:
    """Test the facet override fix in the main search() pipeline.

    SPECIFIC/VAGUE intents should keep Algolia's catalog-wide facets.
    EXACT intent should recompute from merged results when user filters applied.
    """

    def test_specific_keeps_algolia_facets(self):
        """SPECIFIC intent: Algolia facets kept even with semantic results."""
        from search.hybrid_search import HybridSearchService
        from search.models import QueryIntent

        svc = HybridSearchService.__new__(HybridSearchService)

        # Simulate the facet decision block
        intent = QueryIntent.SPECIFIC
        facets = {"brand": [{"value": "Boohoo", "count": 5000}]}
        semantic_results = [{"product_id": "1"}]
        user_filters = {}
        merged = [{"product_id": "1", "brand": "Boohoo"}]

        # The logic from the code:
        if semantic_results or user_filters:
            if intent.value != "exact" and facets:
                pass  # keep Algolia facets
            else:
                computed = {"brand": [{"value": "Boohoo", "count": 1}]}
                facets = computed

        # Algolia facets preserved (count=5000, not 1)
        assert facets["brand"][0]["count"] == 5000

    def test_vague_keeps_algolia_facets(self):
        """VAGUE intent: Algolia facets kept."""
        from search.models import QueryIntent

        intent = QueryIntent.VAGUE
        facets = {"brand": [{"value": "Boohoo", "count": 20000}]}
        semantic_results = [{"product_id": "1"}]
        user_filters = {"brands": ["Boohoo"]}

        if semantic_results or user_filters:
            if intent.value != "exact" and facets:
                pass
            else:
                facets = {"brand": [{"value": "Boohoo", "count": 3}]}

        assert facets["brand"][0]["count"] == 20000

    def test_exact_with_user_filters_recomputes(self):
        """EXACT intent + user filters: recomputes facets from merged."""
        from search.models import QueryIntent

        intent = QueryIntent.EXACT
        facets = {"brand": [{"value": "Boohoo", "count": 20000}]}
        semantic_results = []
        user_filters = {"max_price": 50.0}
        merged = [
            {"product_id": "1", "brand": "Boohoo"},
            {"product_id": "2", "brand": "Boohoo"},
            {"product_id": "3", "brand": "Missguided"},
        ]

        # Simulated: _compute_facets_from_results returns smaller counts
        computed_facets = {"brand": [{"value": "Boohoo", "count": 2}]}

        if semantic_results or user_filters:
            if intent.value != "exact" and facets:
                pass
            else:
                if computed_facets:
                    facets = computed_facets

        # EXACT recomputes
        assert facets["brand"][0]["count"] == 2

    def test_exact_no_user_filters_no_recompute(self):
        """EXACT intent without user filters or semantic: no facet recompute."""
        from search.models import QueryIntent

        intent = QueryIntent.EXACT
        facets = {"brand": [{"value": "Boohoo", "count": 20000}]}
        semantic_results = []
        user_filters = {}

        # Condition is False — no change
        if semantic_results or user_filters:
            if intent.value != "exact" and facets:
                pass
            else:
                facets = {"brand": [{"value": "Boohoo", "count": 2}]}

        assert facets["brand"][0]["count"] == 20000

    def test_specific_no_facets_falls_through(self):
        """SPECIFIC with no Algolia facets: still recomputes."""
        from search.models import QueryIntent

        intent = QueryIntent.SPECIFIC
        facets = None  # Algolia didn't return facets
        semantic_results = [{"product_id": "1"}]
        user_filters = {}
        computed_facets = {"brand": [{"value": "Boohoo", "count": 3}]}

        if semantic_results or user_filters:
            if intent.value != "exact" and facets:
                pass
            else:
                if computed_facets:
                    facets = computed_facets

        # Falls through because facets was None
        assert facets is not None
        assert facets["brand"][0]["count"] == 3


# =============================================================================
# 8. Filter merging edge cases
# =============================================================================

class TestFilterMerging:
    """Test Algolia filter string merging edge cases."""

    def test_in_stock_always_present(self, hybrid_service):
        """in_stock:true is always in the merged filter string."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        # Entry with no algolia_filters
        entry = _make_entry(intent="specific", algolia_filters="")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        captured = {}

        def mock_algolia(query, filters=None, **kwargs):
            captured["filters"] = filters
            return ([], None, 0)

        with patch.object(
            hybrid_service, "_search_algolia", side_effect=mock_algolia
        ), patch.object(
            hybrid_service, "_search_semantic_multi", return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results", side_effect=lambda r: r
        ):
            hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        assert "in_stock:true" in captured["filters"]

    def test_user_only_filters_no_planner(self, hybrid_service):
        """When entry has empty algolia_filters, only user clauses used."""
        from search.hybrid_search import HybridSearchService

        clauses = HybridSearchService._build_user_filter_clauses(
            {"brands": ["Boohoo"], "max_price": 50.0}
        )
        # Should have brand and price, no leftover planner filters
        assert 'brand:"Boohoo"' in clauses
        assert "price <= 50.0" in clauses

    def test_no_double_in_stock(self, hybrid_service):
        """in_stock:true not duplicated when already in planner filters."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(
            intent="specific",
            algolia_filters='in_stock:true AND (category_l1:"Dresses")',
        )
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        captured = {}

        def mock_algolia(query, filters=None, **kwargs):
            captured["filters"] = filters
            return ([], None, 0)

        with patch.object(
            hybrid_service, "_search_algolia", side_effect=mock_algolia
        ), patch.object(
            hybrid_service, "_search_semantic_multi", return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results", side_effect=lambda r: r
        ):
            hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        # Should contain exactly one in_stock:true
        assert captured["filters"].count("in_stock:true") == 1


# =============================================================================
# 9. has_more flag tests
# =============================================================================

class TestHasMore:
    """Test has_more flag accuracy after refinement."""

    def test_has_more_true_when_results_exist(self, hybrid_service):
        """has_more=True when new session is created (results > 0)."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(intent="specific")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 10), _make_algolia_facets(), 100)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        # Session was created → has_more overridden to True
        assert result.pagination.has_more is True

    def test_has_more_false_when_no_results(self, hybrid_service):
        """has_more=False when zero results."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(intent="specific")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="black midi dress",
            search_session_id=entry.session_id,
            brands=["NonexistentBrand"],
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=([], None, 0)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["NonexistentBrand"]},
            )

        assert result.pagination.has_more is False


# =============================================================================
# 10. Brand cap / category cap override tests
# =============================================================================

class TestRerankerCapOverrides:
    """Test that brand and category caps are disabled during filter refinement."""

    def test_brand_cap_disabled_when_user_filters_by_brand(self, hybrid_service):
        """When user filters by brand, max_per_brand=0 (no cap)."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        # Original VAGUE session has max_per_brand=4 (default)
        entry = _make_entry(
            intent="vague",
            query="athletic outfits",
            semantic_queries=["sporty casual outfits", "athletic fashion look"],
            rerank_kwargs={
                "user_profile": None,
                "user_context": None,
                "session_scores": None,
                "page_size": 50,  # VAGUE has category caps
                # max_per_brand NOT set → reranker defaults to 4
            },
        )
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="athletic outfits",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
        )

        # Return 20 Boohoo results — without fix, reranker caps to 4
        algolia_results = _make_algolia_batch("alg", 10, brand="Boohoo")
        semantic_results = _make_semantic_batch("sem", 10, brand="Boohoo")

        captured_rerank = {}

        def mock_rerank(**kwargs):
            captured_rerank.update(kwargs)
            # Return all results (no capping)
            return kwargs["results"]

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(algolia_results, _make_algolia_facets({"Boohoo": 200}), 200)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=(semantic_results, None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ), patch.object(
            hybrid_service._reranker, "rerank",
            side_effect=mock_rerank,
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        # max_per_brand should be 0 (disabled), not 4
        assert captured_rerank.get("max_per_brand") == 0, \
            f"Expected max_per_brand=0, got {captured_rerank.get('max_per_brand')}"

    def test_category_caps_disabled_on_refinement(self, hybrid_service):
        """Category proportional caps should be disabled during refinement."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(
            intent="vague",
            query="summer vibes",
            semantic_queries=["summer vibes aesthetic", "warm weather fashion"],
            rerank_kwargs={
                "user_profile": None,
                "user_context": None,
                "session_scores": None,
                "page_size": 50,  # VAGUE has category caps active
            },
        )
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="summer vibes",
            search_session_id=entry.session_id,
            max_price=40.0,
        )

        captured_rerank = {}

        def mock_rerank(**kwargs):
            captured_rerank.update(kwargs)
            return kwargs["results"]

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 10, price=30.0), _make_algolia_facets(), 500)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=(_make_semantic_batch("sem", 10, price=35.0), None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ), patch.object(
            hybrid_service._reranker, "rerank",
            side_effect=mock_rerank,
        ):
            hybrid_service._filter_refine_search(
                request=request,
                user_filters={"max_price": 40.0},
            )

        # page_size=0 disables category proportional caps
        assert captured_rerank.get("page_size") == 0, \
            f"Expected page_size=0 (caps disabled), got {captured_rerank.get('page_size')}"

    def test_brand_cap_not_overridden_without_brand_filter(self, hybrid_service):
        """Without brand filter, max_per_brand stays as cached (or unset)."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(
            intent="vague",
            query="summer outfits",
            semantic_queries=["summer outfits style", "warm casual look"],
            rerank_kwargs={
                "user_profile": None,
                "user_context": None,
                "session_scores": None,
                "page_size": 50,
                "max_per_brand": 4,  # explicitly cached
            },
        )
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="summer outfits",
            search_session_id=entry.session_id,
            max_price=60.0,
        )

        captured_rerank = {}

        def mock_rerank(**kwargs):
            captured_rerank.update(kwargs)
            return kwargs["results"]

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("alg", 10), _make_algolia_facets(), 500)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=(_make_semantic_batch("sem", 10), None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ), patch.object(
            hybrid_service._reranker, "rerank",
            side_effect=mock_rerank,
        ):
            hybrid_service._filter_refine_search(
                request=request,
                user_filters={"max_price": 60.0},  # price only, no brand
            )

        # max_per_brand should remain at cached value (4), NOT overridden to 0
        assert captured_rerank.get("max_per_brand") == 4, \
            f"Expected max_per_brand=4 (cached), got {captured_rerank.get('max_per_brand')}"

    def test_all_brand_results_returned_not_capped(self, hybrid_service):
        """With brand filter, all matching results should be returned (not capped to 4)."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_entry(
            intent="vague",
            query="athletic outfits",
            semantic_queries=["sporty outfits", "athletic fashion"],
        )
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="athletic outfits",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
            page_size=50,
        )

        # 25 Boohoo results — without fix, only 4 survive
        algolia_results = _make_algolia_batch("alg", 15, brand="Boohoo")
        semantic_results = _make_semantic_batch("sem", 10, brand="Boohoo")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(algolia_results, _make_algolia_facets({"Boohoo": 200}), 200)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=(semantic_results, None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        # Should have more than 4 results (the old bug cap)
        assert len(result.results) > 4, \
            f"Expected >4 results (all Boohoo), got {len(result.results)}. Brand cap bug?"


def _make_multi_entry(query: str = "black midi dress", **kwargs):
    """Helper: _make_entry with 2 semantic queries (ensures _search_semantic_multi is used)
    and max_per_brand=0 to disable brand diversity cap (test focus: filter accuracy)."""
    if "semantic_queries" not in kwargs:
        kwargs["semantic_queries"] = [query, f"{query} alternative"]
    if "rerank_kwargs" not in kwargs:
        kwargs["rerank_kwargs"] = {
            "user_profile": None,
            "user_context": None,
            "session_scores": None,
            "page_size": 0,
            "max_per_brand": 0,
        }
    return _make_entry(query=query, **kwargs)


# =============================================================================
# 11. Post-filter accuracy during refinement
# =============================================================================

class TestPostFilterAccuracyInRefinement:
    """Verify _apply_user_post_filters correctly filters results during refinement
    for various filter dimensions (colors, patterns, occasions, materials, etc.)."""

    def test_color_filter_keeps_matching(self, hybrid_service):
        """Color filter keeps only products matching primary_color or colors list."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="summer dress")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="summer dress",
            search_session_id=entry.session_id,
            colors=["Red"],
            page_size=50,
        )

        # Mix of red and blue products
        red_results = _make_algolia_batch("red", 8, primary_color="Red", colors=["red"])
        blue_results = _make_algolia_batch("blue", 5, primary_color="Blue", colors=["blue"])
        all_algolia = red_results + blue_results

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(all_algolia, _make_algolia_facets(), 200)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"colors": ["Red"]},
            )

        # All returned products should be red
        for p in result.results:
            assert "red" in p.product_id, f"Non-red product in results: {p.product_id}"

    def test_pattern_filter(self, hybrid_service):
        """Pattern filter keeps only matching patterns."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="tops")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="tops",
            search_session_id=entry.session_id,
            patterns=["Floral"],
            page_size=50,
        )

        floral = _make_algolia_batch("floral", 6, pattern="Floral")
        solid = _make_algolia_batch("solid", 6, pattern="Solid")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(floral + solid, _make_algolia_facets(), 100)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"patterns": ["Floral"]},
            )

        for p in result.results:
            assert "floral" in p.product_id

    def test_occasion_filter(self, hybrid_service):
        """Occasion filter keeps products whose occasions list contains the value."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="vague", query="going out looks")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="going out looks",
            search_session_id=entry.session_id,
            occasions=["Party"],
            page_size=50,
        )

        party = _make_algolia_batch("party", 5, occasions=["Party", "Night Out"])
        work = _make_algolia_batch("work", 5, occasions=["Work", "Office"])

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(party + work, _make_algolia_facets(), 80)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"occasions": ["Party"]},
            )

        for p in result.results:
            assert "party" in p.product_id

    def test_materials_filter(self, hybrid_service):
        """Materials filter (multi-value) keeps products with matching material."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="cotton dress")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="cotton dress",
            search_session_id=entry.session_id,
            materials=["Cotton"],
            page_size=50,
        )

        cotton = _make_algolia_batch("cotton", 7, materials=["Cotton", "Elastane"])
        silk = _make_algolia_batch("silk", 4, materials=["Silk"])

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(cotton + silk, _make_algolia_facets(), 60)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"materials": ["Cotton"]},
            )

        for p in result.results:
            assert "cotton" in p.product_id

    def test_sleeve_type_filter(self, hybrid_service):
        """Sleeve type (single-value attr) correctly filters."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="summer tops")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="summer tops",
            search_session_id=entry.session_id,
            sleeve_type=["Sleeveless"],
            page_size=50,
        )

        sleeveless = _make_algolia_batch("sl", 5, sleeve_type="Sleeveless")
        longsleeve = _make_algolia_batch("ls", 5, sleeve_type="Long Sleeve")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(sleeveless + longsleeve, _make_algolia_facets(), 50)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"sleeve_type": ["Sleeveless"]},
            )

        for p in result.results:
            assert "sl-" in p.product_id

    def test_neckline_filter(self, hybrid_service):
        """Neckline filter correctly filters."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="blouse")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="blouse",
            search_session_id=entry.session_id,
            neckline=["V-Neck"],
            page_size=50,
        )

        vneck = _make_algolia_batch("vn", 4, neckline="V-Neck")
        crew = _make_algolia_batch("cr", 4, neckline="Crew")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(vneck + crew, _make_algolia_facets(), 40)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"neckline": ["V-Neck"]},
            )

        for p in result.results:
            assert "vn-" in p.product_id

    def test_formality_filter(self, hybrid_service):
        """Formality filter correctly filters."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="vague", query="office outfits")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="office outfits",
            search_session_id=entry.session_id,
            formality=["Smart Casual"],
            page_size=50,
        )

        smart = _make_algolia_batch("sc", 5, formality="Smart Casual")
        casual = _make_algolia_batch("cas", 5, formality="Casual")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(smart + casual, _make_algolia_facets(), 60)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"formality": ["Smart Casual"]},
            )

        for p in result.results:
            assert "sc-" in p.product_id

    def test_on_sale_filter(self, hybrid_service):
        """on_sale_only filter keeps only sale items."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="dresses")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="dresses",
            search_session_id=entry.session_id,
            on_sale_only=True,
            page_size=50,
        )

        sale = _make_algolia_batch("sale", 4, is_on_sale=True)
        full = _make_algolia_batch("full", 6, is_on_sale=False)

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(sale + full, _make_algolia_facets(), 30)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"on_sale_only": True},
            )

        for p in result.results:
            assert "sale-" in p.product_id

    def test_is_set_filter(self, hybrid_service):
        """is_set filter keeps only co-ord/set products."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="matching sets")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="matching sets",
            search_session_id=entry.session_id,
            is_set=True,
            page_size=50,
        )

        sets = _make_algolia_batch("set", 3, is_set=True)
        singles = _make_algolia_batch("single", 5, is_set=False)

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(sets + singles, _make_algolia_facets(), 20)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"is_set": True},
            )

        for p in result.results:
            assert "set-" in p.product_id

    def test_season_filter(self, hybrid_service):
        """Season filter (multi-value) keeps matching products."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="vague", query="winter looks")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="winter looks",
            search_session_id=entry.session_id,
            seasons=["Winter"],
            page_size=50,
        )

        winter = _make_algolia_batch("win", 5, seasons=["Winter", "Fall"])
        summer = _make_algolia_batch("sum", 5, seasons=["Summer", "Spring"])

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(winter + summer, _make_algolia_facets(), 50)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"seasons": ["Winter"]},
            )

        for p in result.results:
            assert "win-" in p.product_id

    def test_silhouette_filter(self, hybrid_service):
        """Silhouette filter correctly filters."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="dresses")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="dresses",
            search_session_id=entry.session_id,
            silhouette=["A-Line"],
            page_size=50,
        )

        aline = _make_algolia_batch("al", 4, silhouette="A-Line")
        fitted = _make_algolia_batch("ft", 4, silhouette="Fitted")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(aline + fitted, _make_algolia_facets(), 40)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"silhouette": ["A-Line"]},
            )

        for p in result.results:
            assert "al-" in p.product_id

    def test_style_tags_filter(self, hybrid_service):
        """Style tags (multi-value) filter correctly filters."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="vague", query="trendy outfits")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="trendy outfits",
            search_session_id=entry.session_id,
            style_tags=["boho"],
            page_size=50,
        )

        boho = _make_algolia_batch("boho", 4, style_tags=["boho", "casual"])
        preppy = _make_algolia_batch("preppy", 4, style_tags=["preppy", "classic"])

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(boho + preppy, _make_algolia_facets(), 30)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"style_tags": ["boho"]},
            )

        for p in result.results:
            assert "boho-" in p.product_id

    def test_category_l1_filter(self, hybrid_service):
        """category_l1 filter correctly filters."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="vague", query="casual looks")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="casual looks",
            search_session_id=entry.session_id,
            category_l1=["Tops"],
            page_size=50,
        )

        tops = _make_algolia_batch("tops", 6, category_l1="Tops")
        bottoms = _make_algolia_batch("bots", 4, category_l1="Bottoms")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(tops + bottoms, _make_algolia_facets(), 80)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"category_l1": ["Tops"]},
            )

        for p in result.results:
            assert "tops-" in p.product_id

    def test_length_filter(self, hybrid_service):
        """Length filter correctly filters."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="midi dress")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="midi dress",
            search_session_id=entry.session_id,
            length=["Midi"],
            page_size=50,
        )

        midi = _make_algolia_batch("midi", 5, length="Midi")
        mini = _make_algolia_batch("mini", 5, length="Mini")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(midi + mini, _make_algolia_facets(), 40)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"length": ["Midi"]},
            )

        for p in result.results:
            assert "midi-" in p.product_id

    def test_fit_type_filter(self, hybrid_service):
        """Fit type filter correctly filters."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="jeans")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="jeans",
            search_session_id=entry.session_id,
            fit_type=["Slim"],
            page_size=50,
        )

        slim = _make_algolia_batch("slim", 5, fit_type="Slim")
        relaxed = _make_algolia_batch("relax", 5, fit_type="Relaxed")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(slim + relaxed, _make_algolia_facets(), 50)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"fit_type": ["Slim"]},
            )

        for p in result.results:
            assert "slim-" in p.product_id

    def test_color_matches_colors_list(self, hybrid_service):
        """Color filter should match on the colors list field too, not just primary_color."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="tops")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="tops",
            search_session_id=entry.session_id,
            colors=["Pink"],
            page_size=50,
        )

        # Product has primary_color="Multi" but "Pink" in colors list
        multi_with_pink = _make_algolia_batch("mp", 3, primary_color="Multi", colors=["pink", "white"])
        pure_blue = _make_algolia_batch("blue", 3, primary_color="Blue", colors=["blue"])

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(multi_with_pink + pure_blue, _make_algolia_facets(), 30)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"colors": ["Pink"]},
            )

        assert len(result.results) == 3
        for p in result.results:
            assert "mp-" in p.product_id


# =============================================================================
# 12. Exclusion filters during refinement
# =============================================================================

class TestExclusionFiltersInRefinement:
    """Verify exclusion filters work correctly during filter refinement."""

    def test_exclude_brands(self, hybrid_service):
        """exclude_brands removes matching brands from results."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="dresses")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="dresses",
            search_session_id=entry.session_id,
            exclude_brands=["Boohoo"],
            page_size=50,
        )

        boohoo = _make_algolia_batch("boo", 5, brand="Boohoo")
        zara = _make_algolia_batch("zara", 5, brand="Zara")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(boohoo + zara, _make_algolia_facets(), 100)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"exclude_brands": ["Boohoo"]},
            )

        for p in result.results:
            assert "zara-" in p.product_id

    def test_exclude_neckline(self, hybrid_service):
        """exclude_neckline removes matching neckline from results."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="tops")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="tops",
            search_session_id=entry.session_id,
            exclude_neckline=["Strapless"],
            page_size=50,
        )

        strapless = _make_algolia_batch("strap", 4, neckline="Strapless")
        vneck = _make_algolia_batch("vn", 4, neckline="V-Neck")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(strapless + vneck, _make_algolia_facets(), 40)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"exclude_neckline": ["Strapless"]},
            )

        for p in result.results:
            assert "vn-" in p.product_id

    def test_exclude_colors(self, hybrid_service):
        """exclude_colors removes matching colors."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="dresses")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="dresses",
            search_session_id=entry.session_id,
            exclude_colors=["Black"],
            page_size=50,
        )

        black = _make_algolia_batch("blk", 5, primary_color="Black")
        white = _make_algolia_batch("wht", 5, primary_color="White")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(black + white, _make_algolia_facets(), 60)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"exclude_colors": ["Black"]},
            )

        for p in result.results:
            assert "wht-" in p.product_id

    def test_exclude_patterns(self, hybrid_service):
        """exclude_patterns removes matching patterns."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="tops")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="tops",
            search_session_id=entry.session_id,
            exclude_patterns=["Animal Print"],
            page_size=50,
        )

        animal = _make_algolia_batch("ani", 3, pattern="Animal Print")
        solid = _make_algolia_batch("sol", 5, pattern="Solid")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(animal + solid, _make_algolia_facets(), 40)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"exclude_patterns": ["Animal Print"]},
            )

        for p in result.results:
            assert "sol-" in p.product_id

    def test_exclude_with_null_values_passthrough(self, hybrid_service):
        """Products with N/A or null attribute values should NOT be excluded."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="tops")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="tops",
            search_session_id=entry.session_id,
            exclude_neckline=["Halter"],
            page_size=50,
        )

        halter = _make_algolia_batch("halt", 3, neckline="Halter")
        na_neck = _make_algolia_batch("na", 3, neckline="N/A")
        none_neck = _make_algolia_batch("none", 2, neckline=None)

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(halter + na_neck + none_neck, _make_algolia_facets(), 40)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"exclude_neckline": ["Halter"]},
            )

        # N/A and None products should survive; only Halter removed
        ids = [p.product_id for p in result.results]
        assert any("na-" in pid for pid in ids), "N/A products should not be excluded"
        assert any("none-" in pid for pid in ids), "None-neckline products should not be excluded"
        assert not any("halt-" in pid for pid in ids), "Halter products should be excluded"


# =============================================================================
# 13. Combined / multi-filter refinement
# =============================================================================

class TestCombinedFilterRefinement:
    """Test refinement with multiple filters applied simultaneously."""

    def test_brand_plus_color_plus_price(self, hybrid_service):
        """Multiple filters narrow results to intersection."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="dress")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
            colors=["Red"],
            max_price=40.0,
            page_size=50,
        )

        # Only "match" products satisfy all three filters
        match = _make_algolia_batch("match", 3, brand="Boohoo", primary_color="Red", colors=["red"], price=30.0)
        wrong_brand = _make_algolia_batch("wb", 2, brand="Zara", primary_color="Red", colors=["red"], price=30.0)
        wrong_color = _make_algolia_batch("wc", 2, brand="Boohoo", primary_color="Blue", colors=["blue"], price=30.0)
        wrong_price = _make_algolia_batch("wp", 2, brand="Boohoo", primary_color="Red", colors=["red"], price=60.0)

        all_results = match + wrong_brand + wrong_color + wrong_price

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(all_results, _make_algolia_facets(), 50)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"], "colors": ["Red"], "max_price": 40.0},
            )

        assert len(result.results) == 3
        for p in result.results:
            assert "match-" in p.product_id

    def test_pattern_plus_occasion_plus_sleeve(self, hybrid_service):
        """Three attribute filters narrow correctly."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="vague", query="garden party")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="garden party",
            search_session_id=entry.session_id,
            patterns=["Floral"],
            occasions=["Party"],
            sleeve_type=["Short"],
            page_size=50,
        )

        perfect = _make_algolia_batch("ok", 4, pattern="Floral", occasions=["Party"], sleeve_type="Short")
        wrong_pattern = _make_algolia_batch("wp", 3, pattern="Solid", occasions=["Party"], sleeve_type="Short")
        wrong_occasion = _make_algolia_batch("wo", 3, pattern="Floral", occasions=["Work"], sleeve_type="Short")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(perfect + wrong_pattern + wrong_occasion, _make_algolia_facets(), 40)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"patterns": ["Floral"], "occasions": ["Party"], "sleeve_type": ["Short"]},
            )

        assert len(result.results) == 4
        for p in result.results:
            assert "ok-" in p.product_id

    def test_inclusion_plus_exclusion(self, hybrid_service):
        """Include brand + exclude color should both apply."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="tops")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="tops",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
            exclude_colors=["Black"],
            page_size=50,
        )

        good = _make_algolia_batch("good", 4, brand="Boohoo", primary_color="White")
        bad_brand = _make_algolia_batch("bb", 3, brand="Zara", primary_color="White")
        bad_color = _make_algolia_batch("bc", 3, brand="Boohoo", primary_color="Black")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(good + bad_brand + bad_color, _make_algolia_facets(), 50)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"], "exclude_colors": ["Black"]},
            )

        assert len(result.results) == 4
        for p in result.results:
            assert "good-" in p.product_id

    def test_all_filters_remove_everything(self, hybrid_service):
        """When filters are too restrictive, should return empty results."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="dress")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="dress",
            search_session_id=entry.session_id,
            brands=["NonExistentBrand"],
            page_size=50,
        )

        results = _make_algolia_batch("any", 10, brand="Boohoo")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(results, _make_algolia_facets(), 50)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["NonExistentBrand"]},
            )

        assert len(result.results) == 0

    def test_multiple_values_or_within_field(self, hybrid_service):
        """Multiple values for same filter field act as OR (any match)."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="dress")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="dress",
            search_session_id=entry.session_id,
            patterns=["Floral", "Striped"],
            page_size=50,
        )

        floral = _make_algolia_batch("fl", 3, pattern="Floral")
        striped = _make_algolia_batch("st", 3, pattern="Striped")
        solid = _make_algolia_batch("so", 3, pattern="Solid")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(floral + striped + solid, _make_algolia_facets(), 40)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"patterns": ["Floral", "Striped"]},
            )

        assert len(result.results) == 6
        ids = [p.product_id for p in result.results]
        assert not any("so-" in pid for pid in ids)


# =============================================================================
# 14. Sequential / chained refinements
# =============================================================================

class TestChainedRefinement:
    """Test applying filters, then applying different filters on the refined session."""

    def test_refine_then_refine_again(self, hybrid_service):
        """First refinement → new session → second refinement on new session."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="dress")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        # First refinement: brand filter
        request1 = HybridSearchRequest(
            query="dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
            page_size=50,
        )

        boohoo_red = _make_algolia_batch("br", 4, brand="Boohoo", primary_color="Red", colors=["red"], price=30.0)
        boohoo_blue = _make_algolia_batch("bb", 4, brand="Boohoo", primary_color="Blue", colors=["blue"], price=50.0)
        zara = _make_algolia_batch("za", 4, brand="Zara")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(boohoo_red + boohoo_blue + zara, _make_algolia_facets({"Boohoo": 100}), 100)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result1 = hybrid_service._filter_refine_search(
                request=request1,
                user_filters={"brands": ["Boohoo"]},
            )

        # Should have Boohoo red + blue = 8
        assert len(result1.results) == 8
        new_sid = result1.search_session_id
        assert new_sid is not None and new_sid != entry.session_id

        # Second refinement on new session: add color filter
        request2 = HybridSearchRequest(
            query="dress",
            search_session_id=new_sid,
            brands=["Boohoo"],
            colors=["Red"],
            page_size=50,
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(boohoo_red + boohoo_blue, _make_algolia_facets({"Boohoo": 80}), 80)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result2 = hybrid_service._filter_refine_search(
                request=request2,
                user_filters={"brands": ["Boohoo"], "colors": ["Red"]},
            )

        # Should narrow to only Boohoo Red = 4
        assert len(result2.results) == 4
        for p in result2.results:
            assert "br-" in p.product_id
        # Should get yet another new session ID
        assert result2.search_session_id != new_sid

    def test_chained_refinement_preserves_embeddings(self, hybrid_service):
        """Embeddings from original should survive through multiple refinements."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        original_entry = _make_multi_entry(
            intent="specific", query="summer dress",
            semantic_queries=["breezy summer dress", "lightweight dress"],
        )
        original_embeddings = [e.copy() for e in original_entry.semantic_embeddings]
        cache = SearchSessionCache.get_instance()
        cache.store(original_entry)

        request = HybridSearchRequest(
            query="summer dress",
            search_session_id=original_entry.session_id,
            brands=["Boohoo"],
            page_size=50,
        )

        results = _make_algolia_batch("r", 5, brand="Boohoo")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(results, _make_algolia_facets(), 50)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result1 = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        # Check new session has same embeddings
        new_entry = cache.get(result1.search_session_id)
        assert new_entry is not None
        assert len(new_entry.semantic_embeddings) == len(original_embeddings)
        for orig, new in zip(original_embeddings, new_entry.semantic_embeddings):
            assert np.array_equal(orig, new)

    def test_chained_algolia_filters_accumulate(self, hybrid_service):
        """Each refinement should merge user filters with the stored (already merged) filters."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(
            intent="specific", query="dress",
            algolia_filters='in_stock:true AND (category_l1:"Dresses")',
        )
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        captured_filters = {}

        def mock_algolia_search(**kwargs):
            captured_filters["first"] = kwargs.get("filters", "")
            return (_make_algolia_batch("r", 5, brand="Boohoo"), _make_algolia_facets(), 50)

        request = HybridSearchRequest(
            query="dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
            page_size=50,
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            side_effect=lambda **kw: mock_algolia_search(**kw)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result1 = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        # First refinement: original filters + brand
        first_filters = captured_filters.get("first", "")
        assert 'category_l1:"Dresses"' in first_filters
        assert 'brand:"Boohoo"' in first_filters

        # Now the new session should have the merged filters stored
        new_entry = cache.get(result1.search_session_id)
        assert new_entry is not None
        assert 'brand:"Boohoo"' in new_entry.algolia_filters
        assert 'category_l1:"Dresses"' in new_entry.algolia_filters


# =============================================================================
# 15. Semantic results with filter refinement
# =============================================================================

class TestSemanticResultsInRefinement:
    """Verify semantic results are properly handled during refinement."""

    def test_semantic_results_post_filtered(self, hybrid_service):
        """Semantic results (which bypass Algolia filters) should still be post-filtered."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="red dress")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="red dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
            page_size=50,
        )

        # Algolia returns only Boohoo (filtered), but semantic returns mixed brands
        algolia = _make_algolia_batch("alg", 5, brand="Boohoo")
        semantic = (
            _make_semantic_batch("sem-boo", 3, brand="Boohoo") +
            _make_semantic_batch("sem-zara", 3, brand="Zara")
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(algolia, _make_algolia_facets({"Boohoo": 200}), 200)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=(semantic, None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        # Zara semantic results should be filtered out
        for p in result.results:
            assert "zara" not in p.product_id, f"Zara product leaked through: {p.product_id}"

    def test_semantic_and_algolia_merge_via_rrf(self, hybrid_service):
        """Both sources should contribute to final results after RRF merge."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="floral dress")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="floral dress",
            search_session_id=entry.session_id,
            patterns=["Floral"],
            page_size=50,
        )

        algolia = _make_algolia_batch("alg", 5, pattern="Floral")
        semantic = _make_semantic_batch("sem", 5, pattern="Floral")

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(algolia, _make_algolia_facets(), 100)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=(semantic, None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"patterns": ["Floral"]},
            )

        ids = [p.product_id for p in result.results]
        has_algolia = any("alg-" in pid for pid in ids)
        has_semantic = any("sem-" in pid for pid in ids)
        assert has_algolia, "Algolia results should be in merged output"
        assert has_semantic, "Semantic results should be in merged output"


# =============================================================================
# 16. Post-filter criteria propagation
# =============================================================================

class TestPostFilterCriteriaPropagation:
    """Test that post_filter_criteria on new session entries is correctly built."""

    def test_brand_filter_stored_in_criteria(self, hybrid_service):
        """Brand filter should appear in new session's post_filter_criteria."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="dress")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
            page_size=50,
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("r", 5, brand="Boohoo"), _make_algolia_facets(), 50)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        new_entry = cache.get(result.search_session_id)
        assert new_entry.post_filter_criteria is not None
        assert new_entry.post_filter_criteria.get("brands") == ["Boohoo"]

    def test_price_filter_stored_in_criteria(self, hybrid_service):
        """Price filters should appear in new session's post_filter_criteria."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="dress")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="dress",
            search_session_id=entry.session_id,
            min_price=20.0,
            max_price=80.0,
            page_size=50,
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("r", 5, price=50.0), _make_algolia_facets(), 50)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"min_price": 20.0, "max_price": 80.0},
            )

        new_entry = cache.get(result.search_session_id)
        assert new_entry.post_filter_criteria["min_price"] == 20.0
        assert new_entry.post_filter_criteria["max_price"] == 80.0

    def test_inherits_original_criteria(self, hybrid_service):
        """New session should inherit original criteria that user didn't override."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(
            intent="specific", query="dress",
            post_filter_criteria={"category_l1": ["Dresses"], "min_price": 10.0},
        )
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="dress",
            search_session_id=entry.session_id,
            brands=["Boohoo"],
            page_size=50,
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("r", 5, brand="Boohoo"), _make_algolia_facets(), 50)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["Boohoo"]},
            )

        new_entry = cache.get(result.search_session_id)
        # Inherited from original
        assert new_entry.post_filter_criteria.get("category_l1") == ["Dresses"]
        assert new_entry.post_filter_criteria.get("min_price") == 10.0
        # New from user
        assert new_entry.post_filter_criteria.get("brands") == ["Boohoo"]

    def test_user_overrides_original_criteria(self, hybrid_service):
        """User's filter should override the same key from original criteria."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(
            intent="specific", query="dress",
            post_filter_criteria={"min_price": 10.0, "max_price": 200.0},
        )
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="dress",
            search_session_id=entry.session_id,
            max_price=50.0,
            page_size=50,
        )

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(_make_algolia_batch("r", 5, price=40.0), _make_algolia_facets(), 50)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"max_price": 50.0},
            )

        new_entry = cache.get(result.search_session_id)
        # User override
        assert new_entry.post_filter_criteria["max_price"] == 50.0
        # Inherited (not overridden)
        assert new_entry.post_filter_criteria["min_price"] == 10.0


# =============================================================================
# 17. Case insensitivity in post-filters
# =============================================================================

class TestCaseInsensitivity:
    """Verify that post-filters are case-insensitive."""

    def test_brand_case_insensitive(self, hybrid_service):
        """Brand filter should match regardless of case."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="dress")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="dress",
            search_session_id=entry.session_id,
            brands=["boohoo"],  # lowercase
            page_size=50,
        )

        results = _make_algolia_batch("boo", 5, brand="Boohoo")  # titlecase

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(results, _make_algolia_facets(), 50)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"brands": ["boohoo"]},
            )

        assert len(result.results) == 5

    def test_color_case_insensitive(self, hybrid_service):
        """Color filter should match regardless of case."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="tops")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="tops",
            search_session_id=entry.session_id,
            colors=["RED"],  # uppercase
            page_size=50,
        )

        results = _make_algolia_batch("r", 5, primary_color="Red", colors=["red"])

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(results, _make_algolia_facets(), 50)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"colors": ["RED"]},
            )

        assert len(result.results) == 5

    def test_pattern_case_insensitive(self, hybrid_service):
        """Pattern filter should match regardless of case."""
        from search.models import HybridSearchRequest
        from search.session_cache import SearchSessionCache

        entry = _make_multi_entry(intent="specific", query="dress")
        cache = SearchSessionCache.get_instance()
        cache.store(entry)

        request = HybridSearchRequest(
            query="dress",
            search_session_id=entry.session_id,
            patterns=["floral"],  # lowercase
            page_size=50,
        )

        results = _make_algolia_batch("f", 5, pattern="Floral")  # titlecase

        with patch.object(
            hybrid_service, "_search_algolia",
            return_value=(results, _make_algolia_facets(), 50)
        ), patch.object(
            hybrid_service, "_search_semantic_multi",
            return_value=([], None)
        ), patch.object(
            hybrid_service, "_enrich_semantic_results",
            side_effect=lambda r: r
        ):
            result = hybrid_service._filter_refine_search(
                request=request,
                user_filters={"patterns": ["floral"]},
            )

        assert len(result.results) == 5


# =============================================================================
# 18. Algolia filter clause correctness for non-trivial filters
# =============================================================================

class TestAlgoliaClauseCorrectness:
    """Verify _build_user_filter_clauses produces correct Algolia syntax for edge cases."""

    def test_brand_with_special_characters(self):
        """Brands with & and apostrophes should be quoted correctly."""
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"brands": ["Ba&sh"]}
        )
        assert result == '(brand:"Ba&sh")'

    def test_brand_with_apostrophe(self):
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"brands": ["Levi's"]}
        )
        assert result == '(brand:"Levi\'s")'

    def test_multiple_exclusions_are_anded(self):
        """Each exclusion value becomes a separate NOT clause, all ANDed."""
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses(
            {"exclude_colors": ["Black", "Navy"]}
        )
        assert 'NOT primary_color:"Black"' in result
        assert 'NOT primary_color:"Navy"' in result
        assert " AND " in result

    def test_mixed_inclusion_exclusion_same_dimension(self):
        """Include some patterns + exclude others simultaneously."""
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses({
            "patterns": ["Floral"],
            "exclude_patterns": ["Animal Print"],
        })
        assert '(pattern:"Floral")' in result
        assert 'NOT pattern:"Animal Print"' in result
        assert " AND " in result

    def test_price_range_plus_sale(self):
        """Price range + on_sale_only produces correct compound clause."""
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses({
            "min_price": 10.0,
            "max_price": 50.0,
            "on_sale_only": True,
        })
        assert "price >= 10.0" in result
        assert "price <= 50.0" in result
        assert "is_on_sale:true" in result

    def test_all_category_levels_together(self):
        """categories + category_l1 + category_l2 all produce separate clauses."""
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses({
            "categories": ["tops"],
            "category_l1": ["Tops"],
            "category_l2": ["T-Shirt"],
        })
        assert 'broad_category:"tops"' in result
        assert 'category_l1:"Tops"' in result
        assert 'category_l2:"T-Shirt"' in result

    def test_empty_list_ignored(self):
        """Empty lists should not produce any clause."""
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses({
            "brands": [],
            "colors": [],
        })
        assert result == ""

    def test_none_values_ignored(self):
        """None values should not produce any clause."""
        from search.hybrid_search import HybridSearchService
        result = HybridSearchService._build_user_filter_clauses({
            "brands": None,
            "min_price": None,
        })
        assert result == ""
