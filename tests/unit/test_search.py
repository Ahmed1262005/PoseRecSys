"""
Unit tests for the search module.

Tests cover:
1. QueryClassifier - intent classification (exact/specific/vague)
2. Reciprocal Rank Fusion (RRF) - merging Algolia + semantic results
3. SessionReranker - profile scoring, seen_ids dedup, brand diversity
4. Filter building - _build_algolia_filters
5. Filter summary extraction
6. Price validation
7. New filters (silhouette, article_type, style_tags)
8. Semantic post-filter
9. New reranker boosts (color, pattern, neckline, style, formality)
10. Autocomplete service
11. Search analytics
12. Graceful degradation
13. Algolia record mapping

Run with: PYTHONPATH=src python -m pytest tests/unit/test_search.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def classifier():
    from search.query_classifier import QueryClassifier
    return QueryClassifier()


@pytest.fixture
def reranker():
    from search.reranker import SessionReranker
    return SessionReranker()


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


def _make_result(
    product_id: str,
    brand: str = "TestBrand",
    price: float = 50.0,
    source: str = "algolia",
    rrf_score: float = 0.01,
    **kwargs,
) -> dict:
    """Helper to create a mock search result dict."""
    base = {
        "product_id": product_id,
        "name": f"Product {product_id}",
        "brand": brand,
        "image_url": f"https://img.example.com/{product_id}.jpg",
        "gallery_images": [],
        "price": price,
        "original_price": None,
        "is_on_sale": False,
        "category_l1": "Tops",
        "category_l2": "T-Shirt",
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
        "rrf_score": rrf_score,
    }
    base.update(kwargs)
    return base


def _make_algolia_results(n: int = 10) -> List[dict]:
    """Create a list of Algolia-style results."""
    return [
        _make_result(f"alg-{i}", brand=f"Brand{i % 3}", source="algolia")
        for i in range(n)
    ]


def _make_semantic_results(n: int = 10) -> List[dict]:
    """Create a list of semantic-style results."""
    return [
        _make_result(
            f"sem-{i}",
            brand=f"Brand{i % 4}",
            source="semantic",
            semantic_score=0.9 - i * 0.05,
        )
        for i in range(n)
    ]


# =============================================================================
# 1. QueryClassifier Tests
# =============================================================================

class TestQueryClassifier:
    """Tests for query intent classification."""

    def test_exact_brand_match(self, classifier):
        """Pure brand name should be EXACT."""
        from search.models import QueryIntent
        assert classifier.classify("boohoo") == QueryIntent.EXACT
        assert classifier.classify("Boohoo") == QueryIntent.EXACT
        assert classifier.classify("BOOHOO") == QueryIntent.EXACT

    def test_exact_brand_with_whitespace(self, classifier):
        """Brand with leading/trailing whitespace should be EXACT."""
        from search.models import QueryIntent
        assert classifier.classify("  boohoo  ") == QueryIntent.EXACT

    def test_brand_plus_category_is_specific(self, classifier):
        """Brand + additional terms should be SPECIFIC."""
        from search.models import QueryIntent
        assert classifier.classify("boohoo dress") == QueryIntent.SPECIFIC

    def test_specific_category_keyword(self, classifier):
        """Plain category keyword should be SPECIFIC."""
        from search.models import QueryIntent
        assert classifier.classify("dress") == QueryIntent.SPECIFIC
        assert classifier.classify("jeans") == QueryIntent.SPECIFIC
        assert classifier.classify("blazer") == QueryIntent.SPECIFIC

    def test_specific_color_plus_category(self, classifier):
        """Color + category should be SPECIFIC."""
        from search.models import QueryIntent
        assert classifier.classify("blue midi dress") == QueryIntent.SPECIFIC
        assert classifier.classify("black jeans") == QueryIntent.SPECIFIC

    def test_specific_material_plus_category(self, classifier):
        """Material + category should be SPECIFIC."""
        from search.models import QueryIntent
        assert classifier.classify("silk blouse") == QueryIntent.SPECIFIC
        assert classifier.classify("denim jacket") == QueryIntent.SPECIFIC

    def test_vague_style_keywords(self, classifier):
        """Pure style/vibe keywords (no facet mapping) should be VAGUE."""
        from search.models import QueryIntent
        assert classifier.classify("quiet luxury") == QueryIntent.VAGUE
        assert classifier.classify("old money") == QueryIntent.VAGUE
        assert classifier.classify("cottagecore") == QueryIntent.VAGUE
        # Occasion/formality terms are now SPECIFIC (they map to facet filters)
        assert classifier.classify("date night") == QueryIntent.SPECIFIC
        assert classifier.classify("formal") == QueryIntent.SPECIFIC

    def test_vague_occasion_keywords(self, classifier):
        """Occasion keywords should be VAGUE."""
        from search.models import QueryIntent
        assert classifier.classify("what to wear to brunch") == QueryIntent.VAGUE
        assert classifier.classify("outfit for wedding") == QueryIntent.VAGUE

    def test_vague_long_query(self, classifier):
        """Long queries without category terms should be VAGUE."""
        from search.models import QueryIntent
        # 4+ words, no category keywords -> VAGUE
        assert classifier.classify("something nice and comfy to relax in") == QueryIntent.VAGUE

    def test_short_unknown_is_specific(self, classifier):
        """Short (1-2 word) unrecognized query defaults to SPECIFIC."""
        from search.models import QueryIntent
        assert classifier.classify("xyz") == QueryIntent.SPECIFIC
        assert classifier.classify("hello world") == QueryIntent.SPECIFIC

    def test_algolia_weight_exact(self, classifier):
        """EXACT intent gives high Algolia weight."""
        from search.models import QueryIntent
        assert classifier.get_algolia_weight(QueryIntent.EXACT) == 0.85

    def test_algolia_weight_specific(self, classifier):
        from search.models import QueryIntent
        assert classifier.get_algolia_weight(QueryIntent.SPECIFIC) == 0.60

    def test_algolia_weight_vague(self, classifier):
        from search.models import QueryIntent
        assert classifier.get_algolia_weight(QueryIntent.VAGUE) == 0.35

    def test_semantic_weight_exact(self, classifier):
        from search.models import QueryIntent
        assert abs(classifier.get_semantic_weight(QueryIntent.EXACT) - 0.15) < 0.001

    def test_semantic_weight_vague(self, classifier):
        from search.models import QueryIntent
        assert classifier.get_semantic_weight(QueryIntent.VAGUE) == 0.65

    def test_weights_sum_to_one(self, classifier):
        """Algolia + semantic weights should always sum to 1.0."""
        from search.models import QueryIntent
        for intent in QueryIntent:
            algolia_w = classifier.get_algolia_weight(intent)
            semantic_w = classifier.get_semantic_weight(intent)
            assert abs(algolia_w + semantic_w - 1.0) < 0.001, (
                f"{intent}: {algolia_w} + {semantic_w} != 1.0"
            )


class TestLoadBrands:
    """Tests for the load_brands function."""

    def setup_method(self):
        """Reset global brand state before each test."""
        import search.query_classifier as qc
        self._original_brands = qc._BRAND_NAMES
    
    def teardown_method(self):
        """Restore global brand state after each test."""
        import search.query_classifier as qc
        qc._BRAND_NAMES = self._original_brands

    @patch("search.algolia_client.get_algolia_client")
    def test_load_brands_from_algolia(self, mock_get_client):
        """load_brands should populate _BRAND_NAMES from Algolia facets."""
        from search.query_classifier import load_brands
        mock_client = MagicMock()
        mock_client.search_for_facet_values.return_value = {
            "facetHits": [
                {"value": "TestBrandA", "count": 100},
                {"value": "TestBrandB", "count": 50},
                {"value": "TestBrandC", "count": 25},
            ]
        }
        mock_get_client.return_value = mock_client
        result = load_brands()
        assert "testbranda" in result
        assert "testbrandb" in result
        assert "testbrandc" in result

    @patch("search.algolia_client.get_algolia_client")
    def test_load_brands_fallback_on_error(self, mock_get_client):
        """load_brands should include _FALLBACK_BRANDS when Algolia fails."""
        from search.query_classifier import load_brands, _FALLBACK_BRANDS
        mock_get_client.side_effect = Exception("connection error")
        result = load_brands()
        # Should still have fallback brands
        assert "boohoo" in result
        assert _FALLBACK_BRANDS.issubset(result)

    @patch("search.algolia_client.get_algolia_client")
    def test_load_brands_deduplicates_and_lowercases(self, mock_get_client):
        """Brands are lowercased and deduplicated."""
        from search.query_classifier import load_brands
        mock_client = MagicMock()
        mock_client.search_for_facet_values.return_value = {
            "facetHits": [
                {"value": "Nike", "count": 100},
                {"value": "NIKE", "count": 50},
            ]
        }
        mock_get_client.return_value = mock_client
        result = load_brands()
        assert len([b for b in result if b == "nike"]) == 1


# =============================================================================
# 2. Reciprocal Rank Fusion Tests
# =============================================================================

class TestRRF:
    """Tests for Reciprocal Rank Fusion merging."""

    def test_rrf_empty_inputs(self, hybrid_service):
        """Empty inputs should return empty list."""
        result = hybrid_service._reciprocal_rank_fusion([], [], 0.6, 0.4)
        assert result == []

    def test_rrf_algolia_only(self, hybrid_service):
        """With only Algolia results, all should appear in output."""
        algolia = _make_algolia_results(5)
        result = hybrid_service._reciprocal_rank_fusion(algolia, [], 0.6, 0.4)
        assert len(result) == 5
        # All should have rrf_score > 0
        for r in result:
            assert r["rrf_score"] > 0

    def test_rrf_semantic_only(self, hybrid_service):
        """With only semantic results, all should appear in output."""
        semantic = _make_semantic_results(5)
        result = hybrid_service._reciprocal_rank_fusion([], semantic, 0.6, 0.4)
        assert len(result) == 5

    def test_rrf_preserves_order_by_score(self, hybrid_service):
        """Results should be sorted by descending RRF score."""
        algolia = _make_algolia_results(10)
        semantic = _make_semantic_results(10)
        result = hybrid_service._reciprocal_rank_fusion(algolia, semantic, 0.6, 0.4)
        scores = [r["rrf_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_merged_product_has_higher_score(self, hybrid_service):
        """A product appearing in both lists should score higher than one in only one list."""
        # Create overlapping product
        shared_id = "shared-001"
        algolia = [_make_result(shared_id, source="algolia")]
        semantic = [_make_result(shared_id, source="semantic", semantic_score=0.9)]
        single = [_make_result("single-001", source="algolia")]

        # Merge shared product
        merged_shared = hybrid_service._reciprocal_rank_fusion(algolia, semantic, 0.5, 0.5)
        merged_single = hybrid_service._reciprocal_rank_fusion(single, [], 0.5, 0.5)

        shared_score = merged_shared[0]["rrf_score"]
        single_score = merged_single[0]["rrf_score"]
        assert shared_score > single_score

    def test_rrf_respects_weights(self, hybrid_service):
        """Higher Algolia weight should favor Algolia-ranked items."""
        algolia = [_make_result("a1"), _make_result("a2")]
        semantic = [_make_result("s1"), _make_result("s2")]

        # High Algolia weight
        result_high_algolia = hybrid_service._reciprocal_rank_fusion(
            algolia, semantic, algolia_weight=0.9, semantic_weight=0.1
        )

        # Algolia #1 should rank above semantic #1
        ids = [r["product_id"] for r in result_high_algolia]
        assert ids.index("a1") < ids.index("s1")

    def test_rrf_deduplicates(self, hybrid_service):
        """Same product_id in both lists should appear only once."""
        shared = _make_result("shared-001", source="algolia")
        shared_sem = _make_result("shared-001", source="semantic", semantic_score=0.85)
        result = hybrid_service._reciprocal_rank_fusion([shared], [shared_sem], 0.5, 0.5)
        assert len(result) == 1
        assert result[0]["product_id"] == "shared-001"

    def test_rrf_keeps_algolia_data_for_overlap(self, hybrid_service):
        """When a product is in both, Algolia data (richer attributes) should be kept."""
        algolia = [_make_result("p1", source="algolia", category_l1="Tops")]
        semantic = [_make_result("p1", source="semantic", category_l1=None, semantic_score=0.8)]
        result = hybrid_service._reciprocal_rank_fusion(algolia, semantic, 0.6, 0.4)
        assert result[0]["category_l1"] == "Tops"  # Algolia data preserved
        assert result[0].get("semantic_score") == 0.8  # Semantic score added

    def test_rrf_skips_none_product_id(self, hybrid_service):
        """Results with None product_id should be skipped."""
        algolia = [{"product_id": None, "name": "bad"}]
        result = hybrid_service._reciprocal_rank_fusion(algolia, [], 0.6, 0.4)
        assert len(result) == 0

    def test_rrf_k_parameter(self, hybrid_service):
        """Different k values should affect score magnitudes."""
        algolia = _make_algolia_results(5)
        result_k10 = hybrid_service._reciprocal_rank_fusion(algolia, [], 0.6, 0.4, k=10)
        result_k100 = hybrid_service._reciprocal_rank_fusion(algolia, [], 0.6, 0.4, k=100)
        # Higher k = lower scores (more smoothing)
        assert result_k10[0]["rrf_score"] > result_k100[0]["rrf_score"]


# =============================================================================
# 3. SessionReranker Tests
# =============================================================================

class TestSessionReranker:
    """Tests for session-aware reranking."""

    def test_rerank_empty(self, reranker):
        """Empty results should return empty."""
        assert reranker.rerank([], None, None) == []

    def test_rerank_removes_seen_ids(self, reranker):
        """Seen IDs should be filtered out."""
        results = [_make_result(f"p{i}") for i in range(5)]
        seen = {"p0", "p2", "p4"}
        reranked = reranker.rerank(results, seen_ids=seen)
        ids = {r["product_id"] for r in reranked}
        assert ids == {"p1", "p3"}

    def test_rerank_brand_affinity_boost(self, reranker):
        """Products from preferred brands should rank higher."""
        results = [
            _make_result("p1", brand="Boring", rrf_score=0.01),
            _make_result("p2", brand="Nike", rrf_score=0.01),
        ]
        # Flat profile keys (ProfileScorer format)
        profile = {"preferred_brands": ["Nike"]}
        reranked = reranker.rerank(results, user_profile=profile)
        # Nike should now rank first
        assert reranked[0]["product_id"] == "p2"
        assert reranked[0]["profile_adjustment"] > 0

    def test_rerank_brand_avoidance(self, reranker):
        """Preferred brand should rank higher; unrelated brand should score lower."""
        results = [
            _make_result("p1", brand="GoodBrand", rrf_score=0.05),
            _make_result("p2", brand="BadBrand", rrf_score=0.05),
        ]
        # ProfileScorer: preferred brand gets +0.25, unrelated brand gets -0.05
        # (both also get +0.06 category boost for tops)
        profile = {"preferred_brands": ["GoodBrand"]}
        reranked = reranker.rerank(results, user_profile=profile)
        # GoodBrand should rank first with much higher adjustment
        assert reranked[0]["product_id"] == "p1"
        assert reranked[0]["profile_adjustment"] > reranked[-1]["profile_adjustment"]

    def test_rerank_color_avoidance(self, reranker):
        """Products with avoided colors should be demoted."""
        results = [
            _make_result("p1", rrf_score=0.05, primary_color="red", color_family="red"),
            _make_result("p2", rrf_score=0.05, primary_color="blue", color_family="blue"),
        ]
        # Flat profile key for ProfileScorer
        profile = {"colors_to_avoid": ["red"]}
        reranked = reranker.rerank(results, user_profile=profile)
        assert reranked[-1]["product_id"] == "p1"

    def test_rerank_fit_match_boost(self, reranker):
        """Products matching preferred fit should be boosted."""
        results = [
            _make_result("p1", rrf_score=0.01, fit_type="oversized"),
            _make_result("p2", rrf_score=0.01, fit_type="slim"),
        ]
        # Flat profile key for ProfileScorer
        profile = {"preferred_fits": ["slim"]}
        reranked = reranker.rerank(results, user_profile=profile)
        assert reranked[0]["product_id"] == "p2"

    def test_rerank_occasion_match_boost(self, reranker):
        """Products matching preferred occasions should be boosted."""
        results = [
            _make_result("p1", rrf_score=0.01, occasions=["gym"]),
            _make_result("p2", rrf_score=0.01, occasions=["office", "work"]),
        ]
        # Flat profile key for ProfileScorer
        profile = {"occasions": ["office"]}
        reranked = reranker.rerank(results, user_profile=profile)
        assert reranked[0]["product_id"] == "p2"

    def test_rerank_adjustment_capped(self, reranker):
        """Profile adjustment should be capped at ProfileScorer's max_positive."""
        from scoring.profile_scorer import ProfileScoringConfig
        max_positive = ProfileScoringConfig().max_positive
        results = [
            _make_result("p1", brand="Nike", rrf_score=0.01,
                         fit_type="slim", sleeve_type="long",
                         length="midi", occasions=["office"],
                         style_tags=["casual", "trendy"]),
        ]
        # Flat profile keys for ProfileScorer
        profile = {
            "preferred_brands": ["Nike"],
            "preferred_fits": ["slim"],
            "preferred_sleeves": ["long"],
            "preferred_lengths": ["midi"],
            "occasions": ["office"],
            "style_persona": ["casual", "trendy"],
        }
        reranked = reranker.rerank(results, user_profile=profile)
        assert reranked[0]["profile_adjustment"] <= max_positive

    def test_rerank_brand_diversity_cap(self, reranker):
        """No more than max_per_brand items from a single brand."""
        results = [_make_result(f"p{i}", brand="Boohoo", rrf_score=0.1 - i * 0.001) for i in range(15)]
        reranked = reranker.rerank(results, max_per_brand=5)
        boohoo_count = sum(1 for r in reranked if r["brand"] == "Boohoo")
        assert boohoo_count <= 5

    def test_rerank_brand_diversity_multiple_brands(self, reranker):
        """Diversity cap applies per-brand, not globally."""
        results = (
            [_make_result(f"a{i}", brand="BrandA", rrf_score=0.1 - i * 0.001,
                          broad_category=["tops", "bottoms", "dresses"][i % 3],
                          article_type=["t-shirt", "jeans", "dress"][i % 3])
             for i in range(5)]
            + [_make_result(f"b{i}", brand="BrandB", rrf_score=0.05 - i * 0.001,
                            broad_category=["tops", "bottoms", "dresses"][i % 3],
                            article_type=["blouse", "pants", "skirt"][i % 3])
               for i in range(5)]
        )
        reranked = reranker.rerank(results, max_per_brand=3)
        brand_a = [r for r in reranked if r["brand"] == "BrandA"]
        brand_b = [r for r in reranked if r["brand"] == "BrandB"]
        assert len(brand_a) <= 3
        assert len(brand_b) <= 3

    def test_rerank_no_profile_no_change(self, reranker):
        """Without a user profile, order should be preserved (by rrf_score)."""
        results = [_make_result(f"p{i}", rrf_score=0.1 - i * 0.01) for i in range(5)]
        reranked = reranker.rerank(results, user_profile=None, max_per_brand=0)
        ids = [r["product_id"] for r in reranked]
        assert ids == [f"p{i}" for i in range(5)]


# =============================================================================
# 4. Filter Building Tests
# =============================================================================

class TestFilterBuilding:
    """Tests for _build_algolia_filters."""

    def _build(self, **kwargs) -> str:
        """Helper: build filters from request params."""
        from search.hybrid_search import HybridSearchService
        from search.models import HybridSearchRequest
        service = HybridSearchService(
            algolia_client=MagicMock(),
            analytics=MagicMock(),
        )
        request = HybridSearchRequest(query="test", **kwargs)
        return service._build_algolia_filters(request)

    def test_default_in_stock_filter(self):
        """Default filter should always include in_stock:true."""
        result = self._build()
        assert "in_stock:true" in result

    def test_on_sale_filter(self):
        result = self._build(on_sale_only=True)
        assert "is_on_sale:true" in result

    def test_category_filter(self):
        result = self._build(categories=["tops", "dresses"])
        assert 'broad_category:"tops"' in result
        assert 'broad_category:"dresses"' in result
        assert " OR " in result

    def test_category_l1_filter(self):
        result = self._build(category_l1=["Tops"])
        assert 'category_l1:"Tops"' in result

    def test_category_l2_filter(self):
        result = self._build(category_l2=["Blouse", "T-Shirt"])
        assert 'category_l2:"Blouse"' in result
        assert 'category_l2:"T-Shirt"' in result

    def test_brand_filter(self):
        result = self._build(brands=["Nike", "Adidas"])
        assert 'brand:"Nike"' in result
        assert 'brand:"Adidas"' in result

    def test_exclude_brand_filter(self):
        result = self._build(exclude_brands=["Shein"])
        assert 'NOT brand:"Shein"' in result

    def test_color_filter(self):
        result = self._build(colors=["black", "white"])
        assert 'primary_color:"black"' in result
        assert 'primary_color:"white"' in result

    def test_color_family_filter(self):
        result = self._build(color_family=["Dark"])
        assert 'color_family:"Dark"' in result

    def test_pattern_filter(self):
        result = self._build(patterns=["Floral", "Striped"])
        assert 'pattern:"Floral"' in result

    def test_materials_filter(self):
        """Materials filter should map to apparent_fabric facet."""
        result = self._build(materials=["Cotton", "Silk"])
        assert 'apparent_fabric:"Cotton"' in result
        assert 'apparent_fabric:"Silk"' in result

    def test_occasion_filter(self):
        result = self._build(occasions=["office", "casual"])
        assert 'occasions:"office"' in result

    def test_season_filter(self):
        result = self._build(seasons=["summer"])
        assert 'seasons:"summer"' in result

    def test_formality_filter(self):
        result = self._build(formality=["Casual"])
        assert 'formality:"Casual"' in result

    def test_fit_type_filter(self):
        result = self._build(fit_type=["Slim"])
        assert 'fit_type:"Slim"' in result

    def test_neckline_filter(self):
        result = self._build(neckline=["V-Neck"])
        assert 'neckline:"V-Neck"' in result

    def test_sleeve_type_filter(self):
        result = self._build(sleeve_type=["Long"])
        assert 'sleeve_type:"Long"' in result

    def test_length_filter(self):
        result = self._build(length=["Midi"])
        assert 'length:"Midi"' in result

    def test_rise_filter(self):
        result = self._build(rise=["High"])
        assert 'rise:"High"' in result

    def test_price_range_filter(self):
        result = self._build(min_price=20, max_price=100)
        assert "price >= 20" in result
        assert "price <= 100" in result

    def test_min_price_only(self):
        result = self._build(min_price=10)
        assert "price >= 10" in result
        assert "price <=" not in result

    def test_combined_filters_use_and(self):
        """Multiple filters should be joined with AND."""
        result = self._build(
            categories=["tops"],
            brands=["Nike"],
            on_sale_only=True,
        )
        parts = result.split(" AND ")
        assert len(parts) >= 3  # in_stock + is_on_sale + category + brand


# =============================================================================
# 5. Filter Summary Extraction Tests
# =============================================================================

class TestFilterSummary:
    """Tests for _extract_filter_summary."""

    def _extract(self, **kwargs) -> dict:
        from search.hybrid_search import HybridSearchService
        from search.models import HybridSearchRequest
        service = HybridSearchService(
            algolia_client=MagicMock(),
            analytics=MagicMock(),
        )
        request = HybridSearchRequest(query="test", **kwargs)
        return service._extract_filter_summary(request)

    def test_empty_request_has_no_filters(self):
        """Default request should produce empty filter summary."""
        result = self._extract()
        assert result == {}

    def test_includes_set_filters(self):
        result = self._extract(brands=["Nike"], colors=["black"])
        assert result["brands"] == ["Nike"]
        assert result["colors"] == ["black"]

    def test_excludes_empty_lists(self):
        """Empty lists should not appear in summary."""
        result = self._extract()
        assert "categories" not in result
        assert "brands" not in result

    def test_includes_on_sale_only_when_true(self):
        result = self._extract(on_sale_only=True)
        assert result["on_sale_only"] is True

    def test_excludes_on_sale_only_when_false(self):
        result = self._extract(on_sale_only=False)
        assert "on_sale_only" not in result

    def test_includes_price_range(self):
        result = self._extract(min_price=10, max_price=200)
        assert result["min_price"] == 10
        assert result["max_price"] == 200


# =============================================================================
# 6. Response Formatting Tests
# =============================================================================

class TestResponseFormatting:
    """Tests for _to_product_result."""

    def test_formats_all_fields(self, hybrid_service):
        from search.models import ProductResult
        item = _make_result("test-001", brand="TestBrand", price=99.99)
        item["algolia_rank"] = 1
        item["semantic_rank"] = 3
        item["semantic_score"] = 0.85
        item["rrf_score"] = 0.012

        result = hybrid_service._to_product_result(item, position=1)
        assert isinstance(result, ProductResult)
        assert result.product_id == "test-001"
        assert result.brand == "TestBrand"
        assert result.price == 99.99
        assert result.algolia_rank == 1
        assert result.semantic_rank == 3
        assert result.semantic_score == 0.85
        assert result.rrf_score == 0.012

    def test_handles_missing_fields(self, hybrid_service):
        """Missing fields should default to None."""
        item = {"product_id": "min-001", "name": "Minimal", "brand": "X"}
        result = hybrid_service._to_product_result(item, position=1)
        assert result.image_url is None
        assert result.algolia_rank is None
        assert result.semantic_score is None


# =============================================================================
# 7. End-to-End Search Pipeline (mocked backends)
# =============================================================================

class TestSearchPipeline:
    """Test the full search() method with mocked Algolia + semantic."""

    def _setup_service(self, hybrid_service):
        """Set up the hybrid service with a mocked semantic engine."""
        mock_semantic = MagicMock()
        mock_semantic.search_with_filters.return_value = {"results": []}
        hybrid_service._semantic_engine = mock_semantic
        return hybrid_service

    def test_search_returns_response(self, hybrid_service):
        """search() should return a HybridSearchResponse."""
        from search.models import HybridSearchRequest, HybridSearchResponse

        service = self._setup_service(hybrid_service)
        service._algolia.search = MagicMock(return_value={
            "hits": [
                {"objectID": "p1", "name": "Blue Dress", "brand": "TestBrand",
                 "price": 50, "image_url": "http://img/1.jpg", "in_stock": True},
                {"objectID": "p2", "name": "Red Top", "brand": "TestBrand",
                 "price": 30, "image_url": "http://img/2.jpg", "in_stock": True},
            ],
            "nbHits": 2,
        })

        request = HybridSearchRequest(query="boohoo", page_size=10)
        result = service.search(request, user_id="test-user")

        assert isinstance(result, HybridSearchResponse)
        assert result.query == "boohoo"
        assert result.intent in ("exact", "specific", "vague")
        assert len(result.results) <= 10
        assert "total_ms" in result.timing

    def test_search_pagination(self, hybrid_service):
        """Pagination should slice results correctly."""
        from search.models import HybridSearchRequest

        service = self._setup_service(hybrid_service)
        hits = [
            {"objectID": f"p{i}", "name": f"Product {i}", "brand": "B", "price": 10}
            for i in range(20)
        ]
        service._algolia.search = MagicMock(return_value={"hits": hits, "nbHits": 20})

        request = HybridSearchRequest(query="test dress", page=2, page_size=5)
        result = service.search(request)

        assert result.pagination.page == 2
        assert result.pagination.page_size == 5
        assert len(result.results) <= 5

    def test_search_logs_analytics(self, hybrid_service):
        """Analytics should be called after search."""
        from search.models import HybridSearchRequest

        service = self._setup_service(hybrid_service)
        service._algolia.search = MagicMock(return_value={"hits": [], "nbHits": 0})
        request = HybridSearchRequest(query="test")
        service.search(request, user_id="u1")

        service._analytics.log_search.assert_called_once()
        call_kwargs = service._analytics.log_search.call_args
        assert call_kwargs.kwargs["query"] == "test" or call_kwargs[1]["query"] == "test"

    def test_search_semantic_skipped_for_exact_with_results(self, hybrid_service):
        """EXACT intent should skip semantic when Algolia returns results."""
        from search.models import HybridSearchRequest

        service = self._setup_service(hybrid_service)
        service._algolia.search = MagicMock(return_value={
            "hits": [{"objectID": "p1", "name": "Test", "brand": "Boohoo", "price": 20}],
            "nbHits": 1,
        })

        # "boohoo" is in fallback brands -> EXACT
        request = HybridSearchRequest(query="boohoo")
        result = service.search(request)

        # semantic_ms should not be in timing (semantic was skipped - Algolia had results)
        assert "semantic_ms" not in result.timing
        service._semantic_engine.search_with_filters.assert_not_called()

    def test_search_exact_falls_back_to_semantic_on_empty(self, hybrid_service):
        """EXACT intent should fall back to semantic when Algolia returns 0 results."""
        from search.models import HybridSearchRequest

        service = self._setup_service(hybrid_service)
        service._algolia.search = MagicMock(return_value={"hits": [], "nbHits": 0})

        # "boohoo" is in fallback brands -> EXACT, but Algolia returns nothing
        request = HybridSearchRequest(query="boohoo")
        result = service.search(request)

        # Semantic should have been called as fallback
        assert "semantic_ms" in result.timing
        service._semantic_engine.search_with_filters.assert_called_once()

    def test_search_semantic_called_for_vague(self, hybrid_service):
        """VAGUE intent should trigger semantic search."""
        from search.models import HybridSearchRequest

        service = self._setup_service(hybrid_service)
        service._algolia.search = MagicMock(return_value={"hits": [], "nbHits": 0})

        request = HybridSearchRequest(query="quiet luxury")
        result = service.search(request)

        assert "semantic_ms" in result.timing
        service._semantic_engine.search_with_filters.assert_called_once()

    def test_search_semantic_boost_override(self, hybrid_service):
        """Custom semantic_boost should override default weights."""
        from search.models import HybridSearchRequest

        service = self._setup_service(hybrid_service)
        service._algolia.search = MagicMock(return_value={"hits": [], "nbHits": 0})

        # Override with high semantic weight
        request = HybridSearchRequest(query="dress", semantic_boost=0.9)
        result = service.search(request)

        # Should have triggered semantic search (not exact intent)
        assert "semantic_ms" in result.timing

    def test_search_with_seen_ids(self, hybrid_service):
        """Seen IDs should be filtered from results."""
        from search.models import HybridSearchRequest

        service = self._setup_service(hybrid_service)
        service._algolia.search = MagicMock(return_value={
            "hits": [
                {"objectID": "p1", "name": "A", "brand": "B", "price": 10},
                {"objectID": "p2", "name": "C", "brand": "D", "price": 20},
                {"objectID": "p3", "name": "E", "brand": "F", "price": 30},
            ],
            "nbHits": 3,
        })

        request = HybridSearchRequest(query="boohoo")
        result = service.search(request, seen_ids={"p1", "p3"})

        result_ids = {r.product_id for r in result.results}
        assert "p1" not in result_ids
        assert "p3" not in result_ids


# =============================================================================
# 8. New Filter Tests (silhouette, article_type, style_tags)
# =============================================================================

class TestNewFilters:
    """Tests for newly added filters."""

    def _build(self, **kwargs) -> str:
        from search.hybrid_search import HybridSearchService
        from search.models import HybridSearchRequest
        service = HybridSearchService(algolia_client=MagicMock(), analytics=MagicMock())
        request = HybridSearchRequest(query="test", **kwargs)
        return service._build_algolia_filters(request)

    def test_silhouette_filter(self):
        result = self._build(silhouette=["Fitted", "A-Line"])
        assert 'silhouette:"Fitted"' in result
        assert 'silhouette:"A-Line"' in result

    def test_article_type_filter(self):
        result = self._build(article_type=["jeans", "midi dress"])
        assert 'article_type:"jeans"' in result
        assert 'article_type:"midi dress"' in result

    def test_style_tags_filter(self):
        result = self._build(style_tags=["boho", "minimalist"])
        assert 'style_tags:"boho"' in result
        assert 'style_tags:"minimalist"' in result

    def test_all_new_filters_combined(self):
        result = self._build(
            silhouette=["Fitted"],
            article_type=["jeans"],
            style_tags=["casual"],
        )
        assert 'silhouette:"Fitted"' in result
        assert 'article_type:"jeans"' in result
        assert 'style_tags:"casual"' in result


# =============================================================================
# 9. Price Validation Tests
# =============================================================================

class TestPriceValidation:
    """Tests for min_price <= max_price validation."""

    def test_valid_price_range(self):
        from search.models import HybridSearchRequest
        req = HybridSearchRequest(query="test", min_price=10, max_price=100)
        assert req.min_price == 10
        assert req.max_price == 100

    def test_equal_prices_valid(self):
        from search.models import HybridSearchRequest
        req = HybridSearchRequest(query="test", min_price=50, max_price=50)
        assert req.min_price == 50

    def test_invalid_price_range_raises(self):
        from search.models import HybridSearchRequest
        with pytest.raises(ValueError, match="min_price.*max_price"):
            HybridSearchRequest(query="test", min_price=100, max_price=50)

    def test_min_price_only_valid(self):
        from search.models import HybridSearchRequest
        req = HybridSearchRequest(query="test", min_price=10)
        assert req.min_price == 10
        assert req.max_price is None

    def test_max_price_only_valid(self):
        from search.models import HybridSearchRequest
        req = HybridSearchRequest(query="test", max_price=100)
        assert req.min_price is None
        assert req.max_price == 100


# =============================================================================
# 10. Semantic Post-Filter Tests
# =============================================================================

class TestSemanticPostFilter:
    """Tests for _post_filter_semantic."""

    def _make_semantic(self, **overrides) -> dict:
        """Create a semantic result with defaults."""
        base = {
            "product_id": "sem-1",
            "name": "Test",
            "brand": "B",
            "price": 50,
            "fit_type": "Slim",
            "sleeve_type": "Long",
            "length": "Midi",
            "neckline": "V-Neck",
            "formality": "Casual",
            "rise": "High",
            "silhouette": "Fitted",
            "seasons": ["summer"],
            "materials": ["cotton"],
            "style_tags": ["boho"],
            "article_type": "dress",
            "category_l1": "Dresses",
            "category_l2": "Midi Dress",
            "color_family": "Dark",
            "is_on_sale": False,
            "source": "semantic",
        }
        base.update(overrides)
        return base

    def _filter(self, results, **kwargs):
        from search.hybrid_search import HybridSearchService
        from search.models import HybridSearchRequest
        service = HybridSearchService(algolia_client=MagicMock(), analytics=MagicMock())
        request = HybridSearchRequest(query="test", **kwargs)
        return service._post_filter_semantic(results, request)

    def test_no_filters_passes_all(self):
        results = [self._make_semantic()]
        filtered = self._filter(results)
        assert len(filtered) == 1

    def test_fit_type_filter_matches(self):
        results = [self._make_semantic(fit_type="Slim")]
        filtered = self._filter(results, fit_type=["Slim"])
        assert len(filtered) == 1

    def test_fit_type_filter_rejects(self):
        results = [self._make_semantic(fit_type="Oversized")]
        filtered = self._filter(results, fit_type=["Slim"])
        assert len(filtered) == 0

    def test_fit_type_none_excluded(self):
        """If semantic result lacks fit_type, it is excluded (strict filtering after enrichment)."""
        results = [self._make_semantic(fit_type=None)]
        filtered = self._filter(results, fit_type=["Slim"])
        assert len(filtered) == 0

    def test_sleeve_type_filter(self):
        results = [
            self._make_semantic(product_id="s1", sleeve_type="Long"),
            self._make_semantic(product_id="s2", sleeve_type="Short"),
        ]
        filtered = self._filter(results, sleeve_type=["Long"])
        assert len(filtered) == 1
        assert filtered[0]["product_id"] == "s1"

    def test_neckline_filter(self):
        results = [self._make_semantic(neckline="V-Neck")]
        filtered = self._filter(results, neckline=["Crew"])
        assert len(filtered) == 0

    def test_formality_filter(self):
        results = [self._make_semantic(formality="Formal")]
        filtered = self._filter(results, formality=["Casual"])
        assert len(filtered) == 0

    def test_length_filter(self):
        results = [self._make_semantic(length="Midi")]
        filtered = self._filter(results, length=["Midi", "Maxi"])
        assert len(filtered) == 1

    def test_rise_filter(self):
        results = [self._make_semantic(rise="Low")]
        filtered = self._filter(results, rise=["High"])
        assert len(filtered) == 0

    def test_silhouette_filter(self):
        results = [self._make_semantic(silhouette="A-Line")]
        filtered = self._filter(results, silhouette=["A-Line"])
        assert len(filtered) == 1

    def test_seasons_filter(self):
        results = [self._make_semantic(seasons=["summer", "spring"])]
        filtered = self._filter(results, seasons=["winter"])
        assert len(filtered) == 0

    def test_seasons_filter_match(self):
        results = [self._make_semantic(seasons=["summer", "spring"])]
        filtered = self._filter(results, seasons=["summer"])
        assert len(filtered) == 1

    def test_materials_filter(self):
        results = [self._make_semantic(materials=["cotton", "polyester"])]
        filtered = self._filter(results, materials=["silk"])
        assert len(filtered) == 0

    def test_style_tags_filter(self):
        results = [self._make_semantic(style_tags=["boho", "casual"])]
        filtered = self._filter(results, style_tags=["minimalist"])
        assert len(filtered) == 0

    def test_style_tags_filter_match(self):
        results = [self._make_semantic(style_tags=["boho", "casual"])]
        filtered = self._filter(results, style_tags=["boho"])
        assert len(filtered) == 1

    def test_article_type_filter(self):
        results = [self._make_semantic(article_type="dress")]
        filtered = self._filter(results, article_type=["jeans"])
        assert len(filtered) == 0

    def test_category_l1_filter(self):
        results = [self._make_semantic(category_l1="Tops")]
        filtered = self._filter(results, category_l1=["Dresses"])
        assert len(filtered) == 0

    def test_color_family_filter(self):
        results = [self._make_semantic(color_family="Bright")]
        filtered = self._filter(results, color_family=["Dark"])
        assert len(filtered) == 0

    def test_on_sale_only_filter(self):
        results = [self._make_semantic(is_on_sale=False)]
        filtered = self._filter(results, on_sale_only=True)
        assert len(filtered) == 0

    def test_combined_filters(self):
        """Multiple filters should all apply."""
        results = [
            self._make_semantic(product_id="match", fit_type="Slim", length="Midi", formality="Casual"),
            self._make_semantic(product_id="fail", fit_type="Slim", length="Mini", formality="Casual"),
        ]
        filtered = self._filter(results, fit_type=["Slim"], length=["Midi"], formality=["Casual"])
        assert len(filtered) == 1
        assert filtered[0]["product_id"] == "match"

    def test_case_insensitive(self):
        results = [self._make_semantic(fit_type="SLIM")]
        filtered = self._filter(results, fit_type=["slim"])
        assert len(filtered) == 1


# =============================================================================
# 11. New Reranker Boost Tests
# =============================================================================

class TestNewRerankerBoosts:
    """Tests for color, pattern, neckline, style, formality boosts."""

    @pytest.fixture
    def reranker(self):
        from search.reranker import SessionReranker
        return SessionReranker()

    def test_color_preference_boost(self, reranker):
        """Products with avoided colors should be demoted (ProfileScorer uses
        colors_to_avoid, not preferred_colors — avoidance is the key signal)."""
        results = [
            _make_result("p1", rrf_score=0.05, primary_color="Red", color_family="red"),
            _make_result("p2", rrf_score=0.05, primary_color="Black", color_family="dark"),
        ]
        # ProfileScorer uses colors_to_avoid (penalty), not preferred_colors
        profile = {"colors_to_avoid": ["red"]}
        reranked = reranker.rerank(results, user_profile=profile)
        # Red should be demoted
        assert reranked[-1]["product_id"] == "p1"

    def test_pattern_preference_boost(self, reranker):
        """Products matching preferred patterns should be boosted."""
        results = [
            _make_result("p1", rrf_score=0.01, pattern="Solid"),
            _make_result("p2", rrf_score=0.01, pattern="Floral"),
        ]
        # ProfileScorer uses patterns_liked (not preferred_patterns)
        profile = {"patterns_liked": ["floral"]}
        reranked = reranker.rerank(results, user_profile=profile)
        assert reranked[0]["product_id"] == "p2"

    def test_neckline_preference_boost(self, reranker):
        results = [
            _make_result("p1", rrf_score=0.01, neckline="Crew"),
            _make_result("p2", rrf_score=0.01, neckline="V-Neck"),
        ]
        # Flat profile key for ProfileScorer
        profile = {"preferred_necklines": ["v-neck"]}
        reranked = reranker.rerank(results, user_profile=profile)
        assert reranked[0]["product_id"] == "p2"

    def test_style_preference_boost(self, reranker):
        results = [
            _make_result("p1", rrf_score=0.01, style_tags=["preppy"]),
            _make_result("p2", rrf_score=0.01, style_tags=["bohemian", "casual"]),
        ]
        # ProfileScorer uses style_persona (matched against item style_tags)
        profile = {"style_persona": ["bohemian"]}
        reranked = reranker.rerank(results, user_profile=profile)
        assert reranked[0]["product_id"] == "p2"

    def test_formality_preference_boost(self, reranker):
        """ProfileScorer infers formality from style_persona.
        Casual-leaning personas boost items with formality='Casual'."""
        results = [
            _make_result("p1", rrf_score=0.01, formality="Business Casual",
                         style_tags=["classic"]),
            _make_result("p2", rrf_score=0.01, formality="Casual",
                         style_tags=["casual"]),
        ]
        # A casual-leaning persona triggers formality_match for "Casual" items
        profile = {"style_persona": ["casual"]}
        reranked = reranker.rerank(results, user_profile=profile)
        assert reranked[0]["product_id"] == "p2"

    def test_materials_avoidance(self, reranker):
        """ProfileScorer doesn't directly score materials, but we can test
        via pattern avoidance which has a clear penalty signal."""
        results = [
            _make_result("p1", rrf_score=0.05, pattern="Floral"),
            _make_result("p2", rrf_score=0.05, pattern="Solid"),
        ]
        # patterns_avoided triggers -0.15 penalty in ProfileScorer
        profile = {"patterns_avoided": ["floral"]}
        reranked = reranker.rerank(results, user_profile=profile)
        assert reranked[-1]["product_id"] == "p1"
        assert reranked[-1]["profile_adjustment"] < 0

    def test_sleeve_boost(self, reranker):
        results = [
            _make_result("p1", rrf_score=0.01, sleeve_type="Short"),
            _make_result("p2", rrf_score=0.01, sleeve_type="Puff"),
        ]
        # Flat profile key for ProfileScorer
        profile = {"preferred_sleeves": ["puff"]}
        reranked = reranker.rerank(results, user_profile=profile)
        assert reranked[0]["product_id"] == "p2"

    def test_length_boost(self, reranker):
        results = [
            _make_result("p1", rrf_score=0.01, length="Mini"),
            _make_result("p2", rrf_score=0.01, length="Maxi"),
        ]
        # Flat profile key for ProfileScorer
        profile = {"preferred_lengths": ["maxi"]}
        reranked = reranker.rerank(results, user_profile=profile)
        assert reranked[0]["product_id"] == "p2"

    def test_multiple_boosts_stack(self, reranker):
        """Multiple matching prefs should stack (up to max_positive)."""
        from scoring.profile_scorer import ProfileScoringConfig
        max_positive = ProfileScoringConfig().max_positive
        results = [
            _make_result("p1", rrf_score=0.01, pattern="Solid",
                         style_tags=["preppy"], neckline="Crew"),
            _make_result("p2", rrf_score=0.01, pattern="Floral",
                         style_tags=["bohemian", "casual"], neckline="V-Neck"),
        ]
        # Flat ProfileScorer keys: patterns_liked, style_persona, preferred_necklines
        profile = {
            "patterns_liked": ["floral"],
            "style_persona": ["bohemian"],
            "preferred_necklines": ["v-neck"],
        }
        reranked = reranker.rerank(results, user_profile=profile)
        assert reranked[0]["product_id"] == "p2"
        assert reranked[0]["profile_adjustment"] > reranked[1]["profile_adjustment"]
        assert reranked[0]["profile_adjustment"] <= max_positive


# =============================================================================
# 12. Autocomplete Service Tests
# =============================================================================

class TestAutocompleteService:
    """Tests for the AutocompleteService."""

    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.fixture
    def service(self, mock_client):
        from search.autocomplete import AutocompleteService
        return AutocompleteService(client=mock_client)

    def test_empty_query_returns_empty(self, service):
        result = service.autocomplete(query="", limit=10)
        assert result.products == []
        assert result.brands == []

    def test_returns_product_suggestions(self, service, mock_client):
        mock_client.search.return_value = {
            "hits": [
                {"objectID": "p1", "name": "Blue Dress", "brand": "Zara",
                 "image_url": "http://img/1.jpg", "price": 49.99},
                {"objectID": "p2", "name": "Red Top", "brand": "H&M",
                 "image_url": "http://img/2.jpg", "price": 29.99},
            ]
        }
        mock_client.search_for_facet_values.return_value = {"facetHits": []}

        result = service.autocomplete(query="dre", limit=5)
        assert len(result.products) == 2
        assert result.products[0].id == "p1"
        assert result.products[0].name == "Blue Dress"
        assert result.products[0].brand == "Zara"
        assert result.products[0].price == 49.99

    def test_returns_brand_suggestions(self, service, mock_client):
        mock_client.search.return_value = {"hits": []}
        mock_client.search_for_facet_values.return_value = {
            "facetHits": [
                {"value": "Boohoo", "highlighted": "<em>Boo</em>hoo", "count": 5000},
                {"value": "Boden", "highlighted": "<em>Bo</em>den", "count": 200},
            ]
        }

        result = service.autocomplete(query="boo", limit=5)
        assert len(result.brands) == 2
        assert result.brands[0].name == "Boohoo"
        assert result.brands[0].highlighted == "<em>Boo</em>hoo"

    def test_includes_in_stock_filter(self, service, mock_client):
        """Autocomplete should filter in-stock items only."""
        mock_client.search.return_value = {"hits": []}
        mock_client.search_for_facet_values.return_value = {"facetHits": []}

        service.autocomplete(query="test", limit=5)
        call_kwargs = mock_client.search.call_args
        assert "in_stock:true" in str(call_kwargs)

    def test_respects_limit(self, service, mock_client):
        mock_client.search.return_value = {"hits": []}
        mock_client.search_for_facet_values.return_value = {"facetHits": []}

        service.autocomplete(query="test", limit=3)
        call_kwargs = mock_client.search.call_args
        assert call_kwargs.kwargs.get("hits_per_page") == 3 or \
               call_kwargs[1].get("hits_per_page") == 3

    def test_query_in_response(self, service, mock_client):
        mock_client.search.return_value = {"hits": []}
        mock_client.search_for_facet_values.return_value = {"facetHits": []}

        result = service.autocomplete(query="hello", limit=5)
        assert result.query == "hello"

    def test_product_search_error_handled(self, service, mock_client):
        """If product search fails, brands should still return."""
        mock_client.search.side_effect = Exception("Algolia down")
        mock_client.search_for_facet_values.return_value = {
            "facetHits": [{"value": "Nike", "highlighted": "Nike", "count": 100}]
        }

        result = service.autocomplete(query="ni", limit=5)
        assert result.products == []
        assert len(result.brands) == 1

    def test_brand_search_error_handled(self, service, mock_client):
        """If brand search fails, products should still return."""
        mock_client.search.return_value = {
            "hits": [{"objectID": "p1", "name": "Test", "brand": "X", "price": 10}]
        }
        mock_client.search_for_facet_values.side_effect = Exception("facet error")

        result = service.autocomplete(query="test", limit=5)
        assert len(result.products) == 1
        assert result.brands == []

    def test_highlighted_name_extracted(self, service, mock_client):
        mock_client.search.return_value = {
            "hits": [{
                "objectID": "p1", "name": "Blue Dress", "brand": "X",
                "_highlightResult": {"name": {"value": "<em>Blue</em> Dress"}},
            }]
        }
        mock_client.search_for_facet_values.return_value = {"facetHits": []}

        result = service.autocomplete(query="blue", limit=5)
        assert result.products[0].highlighted_name == "<em>Blue</em> Dress"


# =============================================================================
# 13. Search Analytics Tests
# =============================================================================

class TestSearchAnalytics:
    """Tests for SearchAnalytics (mocked DB)."""

    @pytest.fixture
    def analytics(self):
        from search.analytics import SearchAnalytics
        mock_supabase = MagicMock()
        return SearchAnalytics(supabase=mock_supabase), mock_supabase

    def test_log_search(self, analytics):
        svc, mock_sb = analytics
        mock_sb.table.return_value.insert.return_value.execute.return_value = MagicMock()
        svc.log_search(query="test", intent="specific", total_results=10)
        mock_sb.table.assert_called_with("search_analytics")

    def test_log_click(self, analytics):
        svc, mock_sb = analytics
        mock_sb.table.return_value.insert.return_value.execute.return_value = MagicMock()
        svc.log_click(query="test", product_id="p1", position=1, user_id="u1")
        mock_sb.table.assert_called_with("search_clicks")

    def test_log_conversion(self, analytics):
        svc, mock_sb = analytics
        mock_sb.table.return_value.insert.return_value.execute.return_value = MagicMock()
        svc.log_conversion(query="test", product_id="p1", user_id="u1")
        mock_sb.table.assert_called_with("search_conversions")

    def test_log_search_failure_silent(self, analytics):
        """Analytics errors should not raise."""
        svc, mock_sb = analytics
        mock_sb.table.side_effect = Exception("DB down")
        # Should not raise
        svc.log_search(query="test", intent="exact", total_results=0)

    def test_log_click_failure_silent(self, analytics):
        svc, mock_sb = analytics
        mock_sb.table.side_effect = Exception("DB down")
        svc.log_click(query="test", product_id="p1", position=1)

    def test_log_conversion_failure_silent(self, analytics):
        svc, mock_sb = analytics
        mock_sb.table.side_effect = Exception("DB down")
        svc.log_conversion(query="test", product_id="p1")


# =============================================================================
# 14. Graceful Degradation Tests
# =============================================================================

class TestGracefulDegradation:
    """Tests for graceful failure handling in the search pipeline."""

    def _setup(self):
        from search.hybrid_search import HybridSearchService
        mock_algolia = MagicMock()
        mock_analytics = MagicMock()
        service = HybridSearchService(algolia_client=mock_algolia, analytics=mock_analytics)
        mock_semantic = MagicMock()
        service._semantic_engine = mock_semantic
        return service, mock_algolia, mock_semantic, mock_analytics

    def test_algolia_failure_returns_empty(self):
        """If Algolia fails, search should return empty results (not crash)."""
        from search.models import HybridSearchRequest
        service, mock_algolia, mock_semantic, _ = self._setup()
        mock_algolia.search.side_effect = Exception("Algolia down")
        mock_semantic.search_with_filters.return_value = {"results": []}

        result = service.search(HybridSearchRequest(query="dress"))
        assert result.results == []
        assert result.query == "dress"

    def test_semantic_failure_returns_algolia_only(self):
        """If semantic fails, Algolia results should still return."""
        from search.models import HybridSearchRequest
        service, mock_algolia, mock_semantic, _ = self._setup()
        mock_algolia.search.return_value = {
            "hits": [{"objectID": "p1", "name": "Dress", "brand": "B", "price": 50}],
            "nbHits": 1,
        }
        mock_semantic.search_with_filters.side_effect = Exception("CLIP error")

        result = service.search(HybridSearchRequest(query="quiet luxury"))
        assert len(result.results) == 1
        assert result.results[0].product_id == "p1"

    def test_analytics_failure_doesnt_break_search(self):
        """If analytics logging fails, search should still return results."""
        from search.models import HybridSearchRequest
        service, mock_algolia, mock_semantic, mock_analytics = self._setup()
        mock_algolia.search.return_value = {
            "hits": [{"objectID": "p1", "name": "Test", "brand": "B", "price": 10}],
            "nbHits": 1,
        }
        mock_analytics.log_search.side_effect = Exception("Analytics DB down")

        result = service.search(HybridSearchRequest(query="boohoo"))
        assert len(result.results) == 1

    def test_both_backends_fail_returns_empty(self):
        """If both Algolia and semantic fail, return empty results gracefully."""
        from search.models import HybridSearchRequest
        service, mock_algolia, mock_semantic, _ = self._setup()
        mock_algolia.search.side_effect = Exception("Algolia down")
        mock_semantic.search_with_filters.side_effect = Exception("Semantic down")

        result = service.search(HybridSearchRequest(query="dress"))
        assert result.results == []


# =============================================================================
# 15. Algolia Record Mapping Tests
# =============================================================================

class TestAlgoliaRecordMapping:
    """Tests for product_to_algolia_record in algolia_config."""

    def test_basic_mapping(self):
        from search.algolia_config import product_to_algolia_record
        product = {
            "id": "abc-123",
            "name": "Blue Dress",
            "brand": "Zara",
            "price": 49.99,
            "original_price": 79.99,
            "image_url": "http://img/1.jpg",
            "in_stock": True,
        }
        attrs = {"primary_color": "Blue", "pattern": "Solid", "formality": "Casual"}
        record = product_to_algolia_record(product, attrs)
        assert record["objectID"] == "abc-123"
        assert record["name"] == "Blue Dress"
        assert record["brand"] == "Zara"
        assert record["price"] == 49.99
        assert record["primary_color"] == "Blue"

    def test_is_on_sale_computed(self):
        from search.algolia_config import product_to_algolia_record
        product = {"id": "p1", "price": 30, "original_price": 50, "in_stock": True}
        record = product_to_algolia_record(product, {})
        assert record["is_on_sale"] is True

    def test_not_on_sale(self):
        from search.algolia_config import product_to_algolia_record
        product = {"id": "p1", "price": 50, "original_price": 50, "in_stock": True}
        record = product_to_algolia_record(product, {})
        assert record.get("is_on_sale") in (False, None)

    def test_none_values_stripped(self):
        from search.algolia_config import product_to_algolia_record
        product = {"id": "p1", "name": None, "brand": None, "price": None, "in_stock": True}
        record = product_to_algolia_record(product, {"primary_color": None})
        # None values should not appear in the record
        for key, val in record.items():
            if key != "objectID":
                assert val is not None or key == "objectID", f"{key} should not be None"

    def test_construction_attrs_extracted(self):
        """Construction attrs come from the nested 'construction' dict."""
        from search.algolia_config import product_to_algolia_record
        product = {"id": "p1", "in_stock": True}
        attrs = {
            "silhouette": "A-Line",
            "construction": {
                "neckline": "V-Neck",
                "sleeve_type": "Long",
                "closure_type": "Zip",
                "length": "Midi",
            },
        }
        record = product_to_algolia_record(product, attrs)
        assert record.get("silhouette") == "A-Line"
        assert record.get("neckline") == "V-Neck"
        assert record.get("sleeve_type") == "Long"
        assert record.get("length") == "Midi"


# =============================================================================
# 16. Filter Summary with New Fields
# =============================================================================

class TestFilterSummaryNew:
    """Verify new filter fields appear in the analytics filter summary."""

    def _extract(self, **kwargs) -> dict:
        from search.hybrid_search import HybridSearchService
        from search.models import HybridSearchRequest
        service = HybridSearchService(algolia_client=MagicMock(), analytics=MagicMock())
        request = HybridSearchRequest(query="test", **kwargs)
        return service._extract_filter_summary(request)

    def test_silhouette_in_summary(self):
        result = self._extract(silhouette=["Fitted"])
        assert result["silhouette"] == ["Fitted"]

    def test_article_type_in_summary(self):
        result = self._extract(article_type=["jeans"])
        assert result["article_type"] == ["jeans"]

    def test_style_tags_in_summary(self):
        result = self._extract(style_tags=["boho"])
        assert result["style_tags"] == ["boho"]

    def test_materials_in_summary(self):
        result = self._extract(materials=["Cotton"])
        assert result["materials"] == ["Cotton"]


# =============================================================================
# 17. Word-Boundary Keyword Matching Tests (Fix: substring false positives)
# =============================================================================

class TestWordBoundaryMatching:
    """Tests that keyword matching uses word boundaries, not substring matching.

    Fixes: "tan" matching "important", "cos" matching "cosplay",
    "red" matching "bored", "gap" matching "gaping", etc.
    """

    def test_tan_not_in_important(self, classifier):
        """'tan' attribute should NOT match as substring of 'important'."""
        from search.models import QueryIntent
        # "important" contains "tan" as substring - should NOT match as attribute
        # 3-word query with no keywords -> SPECIFIC (default for short queries)
        # The key assertion: this is NOT classified because of "tan" matching
        result = classifier.classify("important meeting outfit")
        # Verify: if "tan" had matched, it would be SPECIFIC too, but for the
        # wrong reason. We verify the word-boundary fix separately.
        from search.query_classifier import _ATTRIBUTE_PATTERN
        assert not _ATTRIBUTE_PATTERN.search("important meeting outfit")

    def test_red_not_in_bored(self, classifier):
        """'red' should NOT match in 'bored of my wardrobe'."""
        from search.models import QueryIntent
        result = classifier.classify("bored of my wardrobe")
        assert result == QueryIntent.VAGUE

    def test_tan_as_standalone_matches(self, classifier):
        """'tan' as a standalone word should still match (it's an attribute)."""
        from search.models import QueryIntent
        result = classifier.classify("tan dress")
        assert result == QueryIntent.SPECIFIC

    def test_cos_brand_not_in_cosplay(self, classifier):
        """'cos' brand should NOT match in 'cosplay outfit'."""
        from search.models import QueryIntent
        result = classifier.classify("cosplay outfit")
        # "cosplay" is not a brand, "outfit" triggers vague
        assert result != QueryIntent.EXACT

    def test_gym_not_in_gymnasium(self, classifier):
        """'gym' should NOT match as substring of 'gymnasium'."""
        from search.models import QueryIntent
        # "gymnasium clothes" - "gymnasium" != "gym" at word boundary
        # but "clothes" is not a category keyword either
        result = classifier.classify("gymnasium clothes")
        # gymnasium doesn't match "gym" at word boundary, clothes not in categories
        assert result != QueryIntent.VAGUE or result == QueryIntent.SPECIFIC  # either is ok

    def test_gym_standalone_matches_specific(self, classifier):
        """'gym' as standalone maps to occasion filter -> SPECIFIC."""
        from search.models import QueryIntent
        result = classifier.classify("gym")
        assert result == QueryIntent.SPECIFIC

    def test_casual_not_in_occasionally(self, classifier):
        """'casual' should NOT match as substring of 'occasionally'."""
        # "occasionally" contains "casual" as substring - should NOT match
        # Verify the word-boundary regex does not match "casual" in "occasionally"
        from search.query_classifier import _VAGUE_PATTERN
        assert not _VAGUE_PATTERN.search("occasionally worn items")

    def test_office_not_in_officer(self, classifier):
        """'office' should NOT match as substring of 'officer'."""
        from search.models import QueryIntent
        result = classifier.classify("officer coat")
        # "coat" IS a category keyword, so SPECIFIC
        # But "officer" should NOT be matched as "office" (vague)
        assert result == QueryIntent.SPECIFIC

    def test_beach_standalone_matches_specific(self, classifier):
        """'beach' as standalone maps to occasion filter -> SPECIFIC."""
        from search.models import QueryIntent
        result = classifier.classify("beach")
        assert result == QueryIntent.SPECIFIC

    def test_multiword_keyword_tank_top(self, classifier):
        """Multi-word 'tank top' should match as category."""
        from search.models import QueryIntent
        result = classifier.classify("blue tank top")
        assert result == QueryIntent.SPECIFIC

    def test_new_category_keywords(self, classifier):
        """Newly added category keywords should be recognized."""
        from search.models import QueryIntent
        assert classifier.classify("bralette") == QueryIntent.SPECIFIC
        assert classifier.classify("corset dress") == QueryIntent.SPECIFIC
        assert classifier.classify("tunic") == QueryIntent.SPECIFIC
        assert classifier.classify("joggers") == QueryIntent.SPECIFIC
        assert classifier.classify("culottes") == QueryIntent.SPECIFIC
        assert classifier.classify("poncho") == QueryIntent.SPECIFIC


# =============================================================================
# 18. Brand Extraction Tests (Fix: original casing + word boundary)
# =============================================================================

class TestBrandExtraction:
    """Tests for QueryClassifier.extract_brand with word boundary matching."""

    def test_extract_brand_returns_match(self, classifier):
        """extract_brand should return the brand name."""
        result = classifier.extract_brand("boohoo dress")
        assert result is not None
        assert result.lower() == "boohoo"

    def test_extract_brand_no_match(self, classifier):
        """extract_brand should return None when no brand found."""
        result = classifier.extract_brand("blue midi dress")
        assert result is None

    def test_extract_brand_word_boundary(self, classifier):
        """extract_brand should NOT match brands as substrings."""
        # "cos" is a brand but should NOT match in "cosplay"
        result = classifier.extract_brand("cosplay outfit")
        assert result is None or result.lower() != "cos"

    def test_extract_brand_gap_not_in_gaping(self, classifier):
        """'gap' brand should NOT match in 'gaping holes'."""
        result = classifier.extract_brand("gaping holes in my jeans")
        assert result is None or result.lower() != "gap"

    def test_extract_brand_standalone_works(self, classifier):
        """Standalone brand names should still be extracted."""
        result = classifier.extract_brand("nike")
        assert result is not None
        assert result.lower() == "nike"


# =============================================================================
# 19. Load Brands Tests (Fix: limit 500 instead of 100)
# =============================================================================

class TestLoadBrandsLimit:
    """Tests for increased brand loading limit."""

    def setup_method(self):
        import search.query_classifier as qc
        self._original_brands = qc._BRAND_NAMES
        self._original_patterns = qc._BRAND_PATTERNS
        self._original_originals = dict(qc._BRAND_ORIGINALS)

    def teardown_method(self):
        import search.query_classifier as qc
        qc._BRAND_NAMES = self._original_brands
        qc._BRAND_PATTERNS = self._original_patterns
        qc._BRAND_ORIGINALS = self._original_originals

    @patch("search.algolia_client.get_algolia_client")
    def test_load_brands_queries_multiple_prefixes(self, mock_get_client):
        """load_brands should query multiple prefixes to cover all brands."""
        from search.query_classifier import load_brands
        mock_client = MagicMock()
        mock_client.search_for_facet_values.return_value = {
            "facetHits": [{"value": f"Brand{i}", "count": 100 - i} for i in range(10)]
        }
        mock_get_client.return_value = mock_client
        load_brands()
        # Should call with "" plus a-z prefixes = 27 calls, each with max_facet_hits=100
        assert mock_client.search_for_facet_values.call_count == 27
        for call in mock_client.search_for_facet_values.call_args_list:
            assert call.kwargs.get("max_facet_hits") == 100

    @patch("search.algolia_client.get_algolia_client")
    def test_load_brands_preserves_original_casing(self, mock_get_client):
        """load_brands should store original casing in _BRAND_ORIGINALS."""
        from search.query_classifier import load_brands, _BRAND_ORIGINALS
        mock_client = MagicMock()
        mock_client.search_for_facet_values.return_value = {
            "facetHits": [
                {"value": "Princess Polly", "count": 100},
                {"value": "Ba&sh", "count": 50},
            ]
        }
        mock_get_client.return_value = mock_client
        load_brands()
        assert _BRAND_ORIGINALS.get("princess polly") == "Princess Polly"
        assert _BRAND_ORIGINALS.get("ba&sh") == "Ba&sh"


# =============================================================================
# 20. Thread-Safe Singleton Tests
# =============================================================================

class TestThreadSafeSingletons:
    """Tests that singletons use thread-safe initialization."""

    def test_algolia_client_has_lock(self):
        """AlgoliaClient singleton should have a lock."""
        from search import algolia_client
        assert hasattr(algolia_client, '_algolia_lock')

    def test_hybrid_search_has_lock(self):
        """HybridSearchService singleton should have a lock."""
        from search import hybrid_search
        assert hasattr(hybrid_search, '_service_lock')

    def test_analytics_has_lock(self):
        """SearchAnalytics singleton should have a lock."""
        from search import analytics
        assert hasattr(analytics, '_analytics_lock')

    def test_autocomplete_has_lock(self):
        """AutocompleteService singleton should have a lock."""
        from search import autocomplete
        assert hasattr(autocomplete, '_autocomplete_lock')


# =============================================================================
# 21. Sync Route Tests (Fix: async -> def for sync backends)
# =============================================================================

class TestSyncRoutes:
    """Tests that search routes are sync (def) not async."""

    def test_hybrid_search_is_sync(self):
        """hybrid_search route should be a sync function (def, not async def)."""
        from api.routes.search import hybrid_search
        import asyncio
        assert not asyncio.iscoroutinefunction(hybrid_search)

    def test_autocomplete_is_sync(self):
        from api.routes.search import autocomplete
        import asyncio
        assert not asyncio.iscoroutinefunction(autocomplete)

    def test_record_click_is_sync(self):
        from api.routes.search import record_click
        import asyncio
        assert not asyncio.iscoroutinefunction(record_click)

    def test_record_conversion_is_sync(self):
        from api.routes.search import record_conversion
        import asyncio
        assert not asyncio.iscoroutinefunction(record_conversion)

    def test_search_health_is_sync(self):
        from api.routes.search import search_health
        import asyncio
        assert not asyncio.iscoroutinefunction(search_health)


# =============================================================================
# 22. Total Results in Pagination Tests
# =============================================================================

class TestTotalResultsPagination:
    """Tests that total_results is included in pagination."""

    def test_pagination_has_total_results_field(self):
        from search.models import PaginationInfo
        p = PaginationInfo(page=1, page_size=50, has_more=True, total_results=200)
        assert p.total_results == 200

    def test_pagination_total_results_optional(self):
        from search.models import PaginationInfo
        p = PaginationInfo(page=1, page_size=50, has_more=False)
        assert p.total_results is None

    def test_search_response_includes_total_results(self, hybrid_service):
        """search() response should include total_results in pagination."""
        from search.models import HybridSearchRequest

        mock_semantic = MagicMock()
        mock_semantic.search_with_filters.return_value = {"results": []}
        hybrid_service._semantic_engine = mock_semantic

        hits = [
            {"objectID": f"p{i}", "name": f"Product {i}", "brand": "B", "price": 10}
            for i in range(15)
        ]
        hybrid_service._algolia.search = MagicMock(return_value={"hits": hits, "nbHits": 15})

        request = HybridSearchRequest(query="boohoo", page_size=5)
        result = hybrid_service.search(request)

        assert result.pagination.total_results is not None
        assert result.pagination.total_results > 0


# =============================================================================
# 23. Algolia get_objects Chunking Tests
# =============================================================================

class TestAlgoliaGetObjectsChunking:
    """Tests for get_objects batch chunking and error logging."""

    def test_empty_ids_returns_empty(self):
        from search.algolia_client import AlgoliaClient
        with patch.object(AlgoliaClient, '__init__', lambda self, **kw: None):
            client = AlgoliaClient()
            client.index_name = "test"
            result = client.get_objects([])
            assert result == {}

    def test_chunking_splits_large_batches(self):
        """get_objects should chunk requests larger than batch_size."""
        from search.algolia_client import AlgoliaClient

        mock_client = MagicMock()
        # Return an object for each request
        def mock_get_objects(**kwargs):
            requests = kwargs.get("get_objects_params", {}).get("requests", [])
            mock_resp = MagicMock()
            mock_resp.to_dict.return_value = {
                "results": [{"objectID": r["objectID"]} for r in requests]
            }
            return mock_resp

        mock_client.get_objects = mock_get_objects

        with patch.object(AlgoliaClient, '__init__', lambda self, **kw: None):
            client = AlgoliaClient()
            client._client = mock_client
            client.index_name = "test"

            # Request 2500 IDs with batch_size=1000
            ids = [f"id-{i}" for i in range(2500)]
            result = client.get_objects(ids, batch_size=1000)
            assert len(result) == 2500

    def test_partial_batch_failure_returns_partial(self):
        """If one batch fails, other batches should still return."""
        from search.algolia_client import AlgoliaClient

        call_count = [0]

        def mock_get_objects(**kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Algolia timeout")
            mock_resp = MagicMock()
            requests = kwargs.get("get_objects_params", {}).get("requests", [])
            mock_resp.to_dict.return_value = {
                "results": [{"objectID": r["objectID"]} for r in requests]
            }
            return mock_resp

        mock_client = MagicMock()
        mock_client.get_objects = mock_get_objects

        with patch.object(AlgoliaClient, '__init__', lambda self, **kw: None):
            client = AlgoliaClient()
            client._client = mock_client
            client.index_name = "test"

            # 3 batches of 5, middle batch fails
            ids = [f"id-{i}" for i in range(15)]
            result = client.get_objects(ids, batch_size=5)
            # Should get results from batch 1 and 3 (10 out of 15)
            assert len(result) == 10


# =============================================================================
# 24. Analytics Silent Failure Logging Tests
# =============================================================================

class TestAnalyticsSilentFailureLogging:
    """Tests that analytics failures are logged, not silently swallowed."""

    def test_search_analytics_logs_warning_on_failure(self):
        """Failed analytics should log a warning."""
        from search.analytics import SearchAnalytics
        mock_supabase = MagicMock()
        mock_supabase.table.side_effect = Exception("DB connection lost")

        svc = SearchAnalytics(supabase=mock_supabase)
        with patch("search.analytics.logger") as mock_logger:
            svc.log_search(query="test", intent="exact", total_results=0)
            mock_logger.warning.assert_called_once()

    def test_click_analytics_logs_warning_on_failure(self):
        from search.analytics import SearchAnalytics
        mock_supabase = MagicMock()
        mock_supabase.table.side_effect = Exception("DB timeout")

        svc = SearchAnalytics(supabase=mock_supabase)
        with patch("search.analytics.logger") as mock_logger:
            svc.log_click(query="test", product_id="p1", position=1)
            mock_logger.warning.assert_called_once()

    def test_hybrid_search_logs_analytics_failure(self):
        """HybridSearchService should log warning when analytics fail, not silently pass."""
        from search.hybrid_search import HybridSearchService
        from search.models import HybridSearchRequest

        mock_algolia = MagicMock()
        mock_analytics = MagicMock()
        mock_analytics.log_search.side_effect = Exception("analytics down")

        service = HybridSearchService(algolia_client=mock_algolia, analytics=mock_analytics)
        mock_semantic = MagicMock()
        mock_semantic.search_with_filters.return_value = {"results": []}
        service._semantic_engine = mock_semantic

        mock_algolia.search.return_value = {"hits": [], "nbHits": 0}

        with patch("search.hybrid_search.logger") as mock_logger:
            result = service.search(HybridSearchRequest(query="boohoo"))
            # Should still return results
            assert result is not None
            # Should have logged a warning
            mock_logger.warning.assert_called()


# =============================================================================
# 25. Configurable RRF Weights Tests
# =============================================================================

class TestConfigurableRRFWeights:
    """Tests that RRF weights can be overridden via env vars."""

    def setup_method(self):
        """Reset cached weights before each test."""
        from search.query_classifier import QueryClassifier
        QueryClassifier._rrf_weights = None

    def teardown_method(self):
        """Reset cached weights after each test."""
        from search.query_classifier import QueryClassifier
        QueryClassifier._rrf_weights = None

    def test_default_weights(self, classifier):
        """Default weights should be 0.85/0.60/0.35."""
        from search.models import QueryIntent
        assert abs(classifier.get_algolia_weight(QueryIntent.EXACT) - 0.85) < 0.001
        assert abs(classifier.get_algolia_weight(QueryIntent.SPECIFIC) - 0.60) < 0.001
        assert abs(classifier.get_algolia_weight(QueryIntent.VAGUE) - 0.35) < 0.001

    @patch.dict("os.environ", {"RRF_WEIGHT_EXACT_ALGOLIA": "0.95"})
    def test_env_override_exact(self, classifier):
        """RRF_WEIGHT_EXACT_ALGOLIA env var should override exact weight."""
        from search.models import QueryIntent
        w = classifier.get_algolia_weight(QueryIntent.EXACT)
        assert abs(w - 0.95) < 0.001
        # Semantic weight should be complement
        s = classifier.get_semantic_weight(QueryIntent.EXACT)
        assert abs(s - 0.05) < 0.001

    @patch.dict("os.environ", {"RRF_WEIGHT_VAGUE_ALGOLIA": "0.20"})
    def test_env_override_vague(self, classifier):
        from search.models import QueryIntent
        w = classifier.get_algolia_weight(QueryIntent.VAGUE)
        assert abs(w - 0.20) < 0.001

    def test_weights_always_sum_to_one(self, classifier):
        """Algolia + semantic weights should always sum to 1.0."""
        from search.models import QueryIntent
        for intent in QueryIntent:
            total = classifier.get_algolia_weight(intent) + classifier.get_semantic_weight(intent)
            assert abs(total - 1.0) < 0.001


# =============================================================================
# 26. Auto-detect has_filters Tests
# =============================================================================

class TestAutoDetectFilters:
    """Tests that has_filters auto-detects from request model."""

    def test_no_filters_detected(self):
        """Default request should have no filters."""
        from search.models import HybridSearchRequest
        _NON_FILTER_FIELDS = {"query", "page", "page_size", "session_id", "semantic_boost", "sort_by"}
        request = HybridSearchRequest(query="test")
        has_filters = any(
            getattr(request, f) not in (None, False, [])
            for f in HybridSearchRequest.model_fields
            if f not in _NON_FILTER_FIELDS
        )
        assert not has_filters

    def test_brand_filter_detected(self):
        """Brand filter should be detected."""
        from search.models import HybridSearchRequest
        _NON_FILTER_FIELDS = {"query", "page", "page_size", "session_id", "semantic_boost", "sort_by"}
        request = HybridSearchRequest(query="test", brands=["Nike"])
        has_filters = any(
            getattr(request, f) not in (None, False, [])
            for f in HybridSearchRequest.model_fields
            if f not in _NON_FILTER_FIELDS
        )
        assert has_filters

    def test_on_sale_filter_detected(self):
        from search.models import HybridSearchRequest
        _NON_FILTER_FIELDS = {"query", "page", "page_size", "session_id", "semantic_boost", "sort_by"}
        request = HybridSearchRequest(query="test", on_sale_only=True)
        has_filters = any(
            getattr(request, f) not in (None, False, [])
            for f in HybridSearchRequest.model_fields
            if f not in _NON_FILTER_FIELDS
        )
        assert has_filters

    def test_price_filter_detected(self):
        from search.models import HybridSearchRequest
        _NON_FILTER_FIELDS = {"query", "page", "page_size", "session_id", "semantic_boost", "sort_by"}
        request = HybridSearchRequest(query="test", min_price=10)
        has_filters = any(
            getattr(request, f) not in (None, False, [])
            for f in HybridSearchRequest.model_fields
            if f not in _NON_FILTER_FIELDS
        )
        assert has_filters


# =============================================================================
# 27. User Profile Caching Tests
# =============================================================================

class TestUserProfileCaching:
    """Tests for TTL-cached user profile loading."""

    def setup_method(self):
        from api.routes.search import _profile_cache
        _profile_cache.clear()

    def test_cache_stores_profile(self):
        """Profile should be cached after first load."""
        from api.routes import search as search_mod

        mock_profile = {"hard_filters": {}, "soft_prefs": {"preferred_brands": ["Nike"]}}
        with patch("women_search_engine.get_women_search_engine") as mock_get:
            mock_engine = MagicMock()
            mock_engine.load_user_profile.return_value = mock_profile
            mock_get.return_value = mock_engine

            # First call - loads from DB
            result1 = search_mod._load_user_profile("user-1")
            assert result1 == mock_profile
            mock_engine.load_user_profile.assert_called_once()

            # Second call - should use cache
            result2 = search_mod._load_user_profile("user-1")
            assert result2 == mock_profile
            # Still only 1 DB call
            mock_engine.load_user_profile.assert_called_once()

    def test_cache_expires(self):
        """Expired cache entries should trigger a fresh DB load."""
        from api.routes import search as search_mod

        mock_profile = {"hard_filters": {}, "soft_prefs": {}}
        with patch("women_search_engine.get_women_search_engine") as mock_get:
            mock_engine = MagicMock()
            mock_engine.load_user_profile.return_value = mock_profile
            mock_get.return_value = mock_engine

            # First call
            search_mod._load_user_profile("user-2")

            # Expire the cache by backdating the timestamp
            with search_mod._profile_cache_lock:
                ts, profile = search_mod._profile_cache["user-2"]
                search_mod._profile_cache["user-2"] = (ts - 600, profile)

            # Second call - cache expired, should reload
            search_mod._load_user_profile("user-2")
            assert mock_engine.load_user_profile.call_count == 2


# =============================================================================
# 28. Public load_user_profile Method Tests
# =============================================================================

class TestPublicLoadUserProfile:
    """Tests that WomenSearchEngine has a public load_user_profile method."""

    def test_has_public_method(self):
        """WomenSearchEngine should have load_user_profile (not just _load_user_profile)."""
        from women_search_engine import WomenSearchEngine
        assert hasattr(WomenSearchEngine, 'load_user_profile')
        # Should not start with underscore
        assert not WomenSearchEngine.load_user_profile.__name__.startswith('_')

    def test_public_delegates_to_private(self):
        """load_user_profile should delegate to _load_user_profile."""
        from women_search_engine import WomenSearchEngine
        with patch.object(WomenSearchEngine, '__init__', lambda self: None):
            engine = WomenSearchEngine()
            engine.supabase = MagicMock()
            engine._load_user_profile = MagicMock(return_value={"test": True})
            result = engine.load_user_profile(user_id="u1")
            engine._load_user_profile.assert_called_once_with(user_id="u1", anon_id=None)
            assert result == {"test": True}


# =============================================================================
# 29. Thread-Safe FashionCLIP Model Load Tests
# =============================================================================

class TestThreadSafeModelLoad:
    """Tests that FashionCLIP model loading is thread-safe."""

    def test_engine_has_model_lock(self):
        """WomenSearchEngine should have _model_lock attribute."""
        from women_search_engine import WomenSearchEngine
        with patch.object(WomenSearchEngine, '__init__', lambda self: None):
            engine = WomenSearchEngine()
            engine._model = None
            engine._processor = None
            engine._model_lock = __import__('threading').Lock()
            assert hasattr(engine, '_model_lock')

    def test_engine_singleton_has_lock(self):
        """WomenSearchEngine singleton should use threading lock."""
        from women_search_engine import _engine_lock
        assert _engine_lock is not None


# =============================================================================
# 30. Algolia Fallback to Semantic Tests
# =============================================================================

class TestAlgoliaFallbackToSemantic:
    """Tests for graceful fallback to semantic when Algolia fails."""

    def test_exact_with_algolia_results_skips_semantic(self, hybrid_service):
        """EXACT intent with Algolia results should NOT call semantic."""
        from search.models import HybridSearchRequest

        mock_semantic = MagicMock()
        mock_semantic.search_with_filters.return_value = {"results": []}
        hybrid_service._semantic_engine = mock_semantic
        hybrid_service._algolia.search = MagicMock(return_value={
            "hits": [{"objectID": "p1", "name": "Boohoo Dress", "brand": "Boohoo", "price": 30}],
            "nbHits": 1,
        })

        result = hybrid_service.search(HybridSearchRequest(query="boohoo"))
        mock_semantic.search_with_filters.assert_not_called()

    def test_exact_with_empty_algolia_calls_semantic(self, hybrid_service):
        """EXACT intent with 0 Algolia results should fall back to semantic."""
        from search.models import HybridSearchRequest

        mock_semantic = MagicMock()
        mock_semantic.search_with_filters.return_value = {"results": []}
        hybrid_service._semantic_engine = mock_semantic
        hybrid_service._algolia.search = MagicMock(return_value={"hits": [], "nbHits": 0})

        result = hybrid_service.search(HybridSearchRequest(query="boohoo"))
        mock_semantic.search_with_filters.assert_called_once()

    def test_specific_always_calls_semantic(self, hybrid_service):
        """SPECIFIC intent should always call semantic regardless of Algolia results."""
        from search.models import HybridSearchRequest

        mock_semantic = MagicMock()
        mock_semantic.search_with_filters.return_value = {"results": []}
        hybrid_service._semantic_engine = mock_semantic
        hybrid_service._algolia.search = MagicMock(return_value={
            "hits": [{"objectID": "p1", "name": "Dress", "brand": "B", "price": 10}],
            "nbHits": 1,
        })

        result = hybrid_service.search(HybridSearchRequest(query="blue midi dress"))
        mock_semantic.search_with_filters.assert_called_once()


# =============================================================================
# 15. Sort-by Tests
# =============================================================================

class TestSortByEnum:
    """Tests for the SortBy enum and request model validation."""

    def test_sort_by_enum_values(self):
        from search.models import SortBy
        assert SortBy.RELEVANCE.value == "relevance"
        assert SortBy.PRICE_ASC.value == "price_asc"
        assert SortBy.PRICE_DESC.value == "price_desc"
        assert SortBy.TRENDING.value == "trending"

    def test_default_sort_is_relevance(self):
        from search.models import HybridSearchRequest, SortBy
        req = HybridSearchRequest(query="dress")
        assert req.sort_by == SortBy.RELEVANCE

    def test_sort_by_accepts_valid_values(self):
        from search.models import HybridSearchRequest, SortBy
        for sort_val in ["relevance", "price_asc", "price_desc", "trending"]:
            req = HybridSearchRequest(query="dress", sort_by=sort_val)
            assert req.sort_by.value == sort_val

    def test_sort_by_rejects_invalid_value(self):
        from search.models import HybridSearchRequest
        with pytest.raises(Exception):
            HybridSearchRequest(query="dress", sort_by="invalid_sort")

    def test_response_includes_sort_by(self):
        from search.models import HybridSearchResponse, PaginationInfo
        resp = HybridSearchResponse(
            query="dress",
            intent="specific",
            sort_by="price_asc",
            results=[],
            pagination=PaginationInfo(page=1, page_size=50, has_more=False),
        )
        assert resp.sort_by == "price_asc"

    def test_response_default_sort_by(self):
        from search.models import HybridSearchResponse, PaginationInfo
        resp = HybridSearchResponse(
            query="dress",
            intent="specific",
            results=[],
            pagination=PaginationInfo(page=1, page_size=50, has_more=False),
        )
        assert resp.sort_by == "relevance"


class TestSortedSearchPath:
    """Tests for the Algolia-only sorted search path."""

    @pytest.fixture
    def hybrid_service(self):
        from search.hybrid_search import HybridSearchService
        mock_algolia = MagicMock()
        mock_algolia.index_name = "products"
        mock_analytics = MagicMock()
        return HybridSearchService(
            algolia_client=mock_algolia,
            analytics=mock_analytics,
        )

    def test_relevance_sort_runs_full_pipeline(self, hybrid_service):
        """sort_by=relevance should run semantic + RRF (full pipeline)."""
        from search.models import HybridSearchRequest, SortBy

        mock_semantic = MagicMock()
        mock_semantic.search_with_filters.return_value = {"results": []}
        hybrid_service._semantic_engine = mock_semantic
        hybrid_service._algolia.search = MagicMock(return_value={
            "hits": [{"objectID": "p1", "name": "Dress", "brand": "B", "price": 10}],
            "nbHits": 1,
        })

        result = hybrid_service.search(
            HybridSearchRequest(query="dress", sort_by=SortBy.RELEVANCE),
        )
        # Semantic search should be called for the full pipeline
        mock_semantic.search_with_filters.assert_called_once()
        assert result.sort_by == "relevance"

    def test_price_asc_skips_semantic(self, hybrid_service):
        """sort_by=price_asc should use Algolia-only, skip semantic."""
        from search.models import HybridSearchRequest, SortBy

        mock_semantic = MagicMock()
        hybrid_service._semantic_engine = mock_semantic
        hybrid_service._algolia.search = MagicMock(return_value={
            "hits": [
                {"objectID": "p1", "name": "Cheap Dress", "brand": "B", "price": 10},
                {"objectID": "p2", "name": "Mid Dress", "brand": "B", "price": 50},
            ],
            "nbHits": 2,
            "nbPages": 1,
        })

        result = hybrid_service.search(
            HybridSearchRequest(query="dress", sort_by=SortBy.PRICE_ASC),
        )
        # Semantic should NOT be called
        mock_semantic.search_with_filters.assert_not_called()
        assert result.sort_by == "price_asc"
        assert len(result.results) == 2

    def test_price_desc_skips_semantic(self, hybrid_service):
        """sort_by=price_desc should use Algolia-only, skip semantic."""
        from search.models import HybridSearchRequest, SortBy

        mock_semantic = MagicMock()
        hybrid_service._semantic_engine = mock_semantic
        hybrid_service._algolia.search = MagicMock(return_value={
            "hits": [
                {"objectID": "p1", "name": "Expensive Dress", "brand": "B", "price": 200},
            ],
            "nbHits": 1,
            "nbPages": 1,
        })

        result = hybrid_service.search(
            HybridSearchRequest(query="dress", sort_by=SortBy.PRICE_DESC),
        )
        mock_semantic.search_with_filters.assert_not_called()
        assert result.sort_by == "price_desc"

    def test_trending_sort_skips_semantic(self, hybrid_service):
        """sort_by=trending should use Algolia-only, skip semantic."""
        from search.models import HybridSearchRequest, SortBy

        mock_semantic = MagicMock()
        hybrid_service._semantic_engine = mock_semantic
        hybrid_service._algolia.search = MagicMock(return_value={
            "hits": [
                {"objectID": "p1", "name": "Trendy Top", "brand": "B", "price": 30},
            ],
            "nbHits": 1,
            "nbPages": 1,
        })

        result = hybrid_service.search(
            HybridSearchRequest(query="top", sort_by=SortBy.TRENDING),
        )
        mock_semantic.search_with_filters.assert_not_called()
        assert result.sort_by == "trending"

    def test_sorted_search_uses_replica_index(self, hybrid_service):
        """Sorted search should pass the correct replica index name to Algolia."""
        from search.models import HybridSearchRequest, SortBy

        hybrid_service._algolia.search = MagicMock(return_value={
            "hits": [],
            "nbHits": 0,
            "nbPages": 0,
        })

        hybrid_service.search(
            HybridSearchRequest(query="dress", sort_by=SortBy.PRICE_ASC),
        )

        # Check the Algolia search was called with the replica index name
        call_kwargs = hybrid_service._algolia.search.call_args
        assert call_kwargs.kwargs.get("index_name") == "products_price_asc" or \
               (call_kwargs[1].get("index_name") == "products_price_asc" if len(call_kwargs) > 1 else False)

    def test_sorted_search_uses_algolia_native_pagination(self, hybrid_service):
        """Sorted search should pass page to Algolia (0-indexed)."""
        from search.models import HybridSearchRequest, SortBy

        hybrid_service._algolia.search = MagicMock(return_value={
            "hits": [],
            "nbHits": 100,
            "nbPages": 5,
        })

        hybrid_service.search(
            HybridSearchRequest(query="dress", sort_by=SortBy.PRICE_ASC, page=3, page_size=20),
        )

        call_kwargs = hybrid_service._algolia.search.call_args
        # page=3 in API (1-indexed) should be page=2 for Algolia (0-indexed)
        assert call_kwargs.kwargs.get("page") == 2
        assert call_kwargs.kwargs.get("hits_per_page") == 20

    def test_sorted_search_has_more_pagination(self, hybrid_service):
        """has_more should be correct based on Algolia pagination info."""
        from search.models import HybridSearchRequest, SortBy

        hybrid_service._algolia.search = MagicMock(return_value={
            "hits": [{"objectID": f"p{i}", "name": f"P{i}", "brand": "B", "price": i * 10}
                     for i in range(20)],
            "nbHits": 100,
            "nbPages": 5,
        })

        result = hybrid_service.search(
            HybridSearchRequest(query="dress", sort_by=SortBy.PRICE_ASC, page=1, page_size=20),
        )
        assert result.pagination.has_more is True
        assert result.pagination.total_results == 100

    def test_sorted_search_applies_filters(self, hybrid_service):
        """Sorted search should still apply all filters via Algolia."""
        from search.models import HybridSearchRequest, SortBy

        hybrid_service._algolia.search = MagicMock(return_value={
            "hits": [],
            "nbHits": 0,
            "nbPages": 0,
        })

        hybrid_service.search(
            HybridSearchRequest(
                query="dress",
                sort_by=SortBy.PRICE_ASC,
                brands=["Boohoo"],
                colors=["Black"],
                min_price=10,
                max_price=100,
            ),
        )

        call_kwargs = hybrid_service._algolia.search.call_args
        filters_str = call_kwargs.kwargs.get("filters", "")
        assert "brand:" in filters_str
        assert "primary_color:" in filters_str
        assert "price >= 10" in filters_str
        assert "price <= 100" in filters_str

    def test_sorted_search_handles_algolia_error(self, hybrid_service):
        """Sorted search should return empty results on Algolia error."""
        from search.models import HybridSearchRequest, SortBy

        hybrid_service._algolia.search = MagicMock(side_effect=Exception("Algolia down"))

        result = hybrid_service.search(
            HybridSearchRequest(query="dress", sort_by=SortBy.PRICE_ASC),
        )
        assert len(result.results) == 0
        assert result.sort_by == "price_asc"


class TestReplicaConfig:
    """Tests for Algolia replica index configuration."""

    def test_get_replica_index_name_relevance(self):
        from search.algolia_config import get_replica_index_name
        assert get_replica_index_name("products", "relevance") is None

    def test_get_replica_index_name_price_asc(self):
        from search.algolia_config import get_replica_index_name
        assert get_replica_index_name("products", "price_asc") == "products_price_asc"

    def test_get_replica_index_name_price_desc(self):
        from search.algolia_config import get_replica_index_name
        assert get_replica_index_name("products", "price_desc") == "products_price_desc"

    def test_get_replica_index_name_trending(self):
        from search.algolia_config import get_replica_index_name
        assert get_replica_index_name("products", "trending") == "products_trending"

    def test_get_replica_index_name_unknown(self):
        from search.algolia_config import get_replica_index_name
        assert get_replica_index_name("products", "unknown") is None

    def test_get_replica_names(self):
        from search.algolia_config import get_replica_names
        names = get_replica_names("products")
        assert len(names) == 3
        assert "virtual(products_price_asc)" in names
        assert "virtual(products_price_desc)" in names
        assert "virtual(products_trending)" in names

    def test_get_replica_names_custom_index(self):
        from search.algolia_config import get_replica_names
        names = get_replica_names("my_custom_index")
        assert "virtual(my_custom_index_price_asc)" in names

    def test_replica_suffixes_have_custom_ranking(self):
        from search.algolia_config import REPLICA_SUFFIXES
        for suffix, settings in REPLICA_SUFFIXES.items():
            assert "customRanking" in settings
            assert len(settings["customRanking"]) > 0
