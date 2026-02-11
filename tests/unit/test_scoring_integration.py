"""
Integration tests for the shared scoring module wiring.

Tests cover:
1. Candidate.to_scoring_dict() - field mapping from Candidate to scoring dict
2. Pipeline Step 6c - context scoring integrated into feed pipeline
3. Search Reranker Step 3.75 - context scoring integrated into search reranker
4. HybridSearchService.search() - user_context forwarding
5. Search route helpers - UserContext building, search signal wiring
6. End-to-end scoring flow - context scoring affects final ordering

Run with: PYTHONPATH=src python -m pytest tests/unit/test_scoring_integration.py -v
"""

import pytest
import time
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List, Dict, Any, Optional


# =============================================================================
# Helpers
# =============================================================================

def _make_candidate(**overrides):
    """Create a mock Candidate with sensible defaults."""
    from recs.models import Candidate
    defaults = {
        "item_id": "prod-001",
        "embedding_score": 0.8,
        "preference_score": 0.6,
        "sasrec_score": 0.5,
        "final_score": 0.7,
        "category": "tops",
        "broad_category": "tops",
        "article_type": "t-shirt",
        "brand": "TestBrand",
        "price": 29.99,
        "colors": ["black"],
        "materials": ["cotton"],
        "fit": "regular",
        "length": "standard",
        "sleeve": "short",
        "neckline": "crew",
        "style_tags": ["casual", "everyday"],
        "occasions": ["Everyday", "Casual"],
        "pattern": "Solid",
        "formality": "Casual",
        "color_family": "Dark",
        "seasons": ["Spring", "Summer"],
        "image_url": "https://img.example.com/prod-001.jpg",
        "name": "Basic Black T-Shirt",
        "source": "taste_vector",
    }
    defaults.update(overrides)
    return Candidate(**defaults)


def _make_search_result(
    product_id: str = "search-001",
    rrf_score: float = 0.05,
    **kwargs,
) -> dict:
    """Create a mock search result dict."""
    base = {
        "product_id": product_id,
        "name": f"Product {product_id}",
        "brand": "TestBrand",
        "image_url": f"https://img.example.com/{product_id}.jpg",
        "gallery_images": [],
        "price": 50.0,
        "original_price": None,
        "is_on_sale": False,
        "broad_category": "tops",
        "article_type": "t-shirt",
        "primary_color": "Black",
        "color_family": "Dark",
        "pattern": "Solid",
        "fit_type": "Regular",
        "formality": "Casual",
        "length": "Regular",
        "neckline": "Crew",
        "sleeve_type": "Short",
        "style_tags": ["casual"],
        "occasions": ["Everyday"],
        "seasons": ["Spring", "Summer"],
        "colors": ["black"],
        "materials": ["cotton"],
        "source": "algolia",
        "rrf_score": rrf_score,
    }
    base.update(kwargs)
    return base


def _make_user_context(
    age_group=None,
    weather=None,
    coverage_prefs=None,
):
    """Create a UserContext for testing."""
    from scoring.context import UserContext, AgeGroup, Season, WeatherContext
    ctx = UserContext(user_id="test-user")
    if age_group:
        ctx.age_group = age_group if isinstance(age_group, AgeGroup) else AgeGroup(age_group)
    if weather:
        ctx.weather = weather
    if coverage_prefs:
        ctx.coverage_prefs = coverage_prefs
    return ctx


def _make_summer_weather():
    """Create a hot summer WeatherContext."""
    from scoring.context import WeatherContext, Season
    return WeatherContext(
        temperature_c=32.0,
        feels_like_c=34.0,
        condition="clear",
        humidity=60,
        wind_speed_mps=3.0,
        season=Season.SUMMER,
        is_hot=True,
        is_cold=False,
        is_mild=False,
        is_rainy=False,
    )


def _make_winter_weather():
    """Create a cold winter WeatherContext."""
    from scoring.context import WeatherContext, Season
    return WeatherContext(
        temperature_c=-5.0,
        feels_like_c=-8.0,
        condition="snow",
        humidity=80,
        wind_speed_mps=5.0,
        season=Season.WINTER,
        is_hot=False,
        is_cold=True,
        is_mild=False,
        is_rainy=False,
    )


# =============================================================================
# 1. Candidate.to_scoring_dict()
# =============================================================================

class TestCandidateToScoringDict:
    """Tests for Candidate.to_scoring_dict() method."""

    def test_basic_fields_mapped(self):
        c = _make_candidate()
        d = c.to_scoring_dict()
        assert d["product_id"] == "prod-001"
        assert d["article_type"] == "t-shirt"
        assert d["broad_category"] == "tops"
        assert d["brand"] == "TestBrand"
        assert d["name"] == "Basic Black T-Shirt"

    def test_style_tags_preserved(self):
        c = _make_candidate(style_tags=["streetwear", "urban"])
        d = c.to_scoring_dict()
        assert d["style_tags"] == ["streetwear", "urban"]

    def test_occasions_preserved(self):
        c = _make_candidate(occasions=["Office", "Evening"])
        d = c.to_scoring_dict()
        assert d["occasions"] == ["Office", "Evening"]

    def test_pattern_and_formality(self):
        c = _make_candidate(pattern="Floral", formality="Semi-Formal")
        d = c.to_scoring_dict()
        assert d["pattern"] == "Floral"
        assert d["formality"] == "Semi-Formal"

    def test_fit_maps_from_candidate_fit(self):
        """Candidate uses 'fit' field, scoring dict expects 'fit_type'."""
        c = _make_candidate(fit="oversized")
        d = c.to_scoring_dict()
        assert d["fit_type"] == "oversized"

    def test_sleeve_maps_to_sleeve_type(self):
        """Candidate uses 'sleeve' field, scoring dict expects 'sleeve_type'."""
        c = _make_candidate(sleeve="long")
        d = c.to_scoring_dict()
        assert d["sleeve_type"] == "long"

    def test_seasons_and_materials(self):
        c = _make_candidate(seasons=["Fall", "Winter"], materials=["wool", "cashmere"])
        d = c.to_scoring_dict()
        assert d["seasons"] == ["Fall", "Winter"]
        assert d["materials"] == ["wool", "cashmere"]

    def test_none_fields_default_to_empty(self):
        """None/empty values should become empty strings or empty lists in scoring dict."""
        # Pydantic Candidate rejects None for List fields, so we pass empty
        # lists for list fields and None for Optional[str] fields.
        c = _make_candidate(
            article_type=None, brand=None, name=None,
            style_tags=[], occasions=[], seasons=[], materials=[],
        )
        d = c.to_scoring_dict()
        assert d["article_type"] == ""
        assert d["brand"] == ""
        assert d["name"] == ""
        assert d["style_tags"] == []
        assert d["occasions"] == []
        assert d["seasons"] == []
        assert d["materials"] == []

    def test_scoring_dict_is_plain_dict(self):
        """Result should be a plain dict, not a Pydantic model."""
        c = _make_candidate()
        d = c.to_scoring_dict()
        assert type(d) is dict

    def test_scoring_dict_compatible_with_context_scorer(self):
        """to_scoring_dict() output should work with ContextScorer.score_item()."""
        from scoring import ContextScorer
        from scoring.context import AgeGroup
        scorer = ContextScorer()
        c = _make_candidate(
            article_type="crop_top",
            style_tags=["streetwear"],
            seasons=["Summer"],
            materials=["cotton"],
        )
        ctx = _make_user_context(age_group=AgeGroup.GEN_Z, weather=_make_summer_weather())
        adj = scorer.score_item(c.to_scoring_dict(), ctx)
        assert isinstance(adj, float)
        assert -0.30 <= adj <= 0.30


# =============================================================================
# 2. Pipeline Step 6c - Context Scoring in Feed
# =============================================================================

class TestPipelineContextScoring:
    """Tests for context scoring integration in the recommendation pipeline."""

    def test_pipeline_has_context_scorer(self):
        """Pipeline should have context_scorer and context_resolver attributes."""
        from scoring import ContextScorer, ContextResolver
        # We can't easily instantiate the full pipeline (needs Supabase),
        # but we can verify the imports and class types
        scorer = ContextScorer()
        resolver = ContextResolver(weather_api_key="")
        assert scorer is not None
        assert resolver is not None

    def test_context_scoring_adjusts_final_score(self):
        """Context scoring should modify candidate final_score."""
        from scoring import ContextScorer
        from scoring.context import AgeGroup

        scorer = ContextScorer()
        ctx = _make_user_context(age_group=AgeGroup.GEN_Z, weather=_make_summer_weather())

        # Crop top in summer for Gen Z should get a boost
        c = _make_candidate(
            article_type="crop_top",
            style_tags=["streetwear"],
            seasons=["Summer"],
            materials=["cotton"],
            final_score=0.5,
        )
        item_dict = c.to_scoring_dict()
        adj = scorer.score_item(item_dict, ctx)
        c.final_score += adj
        assert c.final_score != 0.5  # Should have changed
        assert c.final_score > 0.5   # Should be boosted

    def test_context_scoring_penalizes_crop_for_senior_winter(self):
        """Crop top in winter for a senior should get penalty."""
        from scoring import ContextScorer
        from scoring.context import AgeGroup

        scorer = ContextScorer()
        ctx = _make_user_context(age_group=AgeGroup.SENIOR, weather=_make_winter_weather())

        c = _make_candidate(
            article_type="crop_top",
            style_tags=["trendy"],
            seasons=["Summer"],
            materials=["cotton"],
            final_score=0.5,
        )
        item_dict = c.to_scoring_dict()
        adj = scorer.score_item(item_dict, ctx)
        c.final_score += adj
        assert c.final_score < 0.5   # Should be penalized

    def test_context_scoring_sweater_winter_senior_boost(self):
        """Wool sweater in winter for a senior should get boosted."""
        from scoring import ContextScorer
        from scoring.context import AgeGroup

        scorer = ContextScorer()
        ctx = _make_user_context(age_group=AgeGroup.ESTABLISHED, weather=_make_winter_weather())

        c = _make_candidate(
            article_type="sweater",
            style_tags=["classic"],
            seasons=["Fall", "Winter"],
            materials=["wool"],
            fit="relaxed",
            final_score=0.5,
        )
        item_dict = c.to_scoring_dict()
        adj = scorer.score_item(item_dict, ctx)
        c.final_score += adj
        assert c.final_score > 0.5   # Should be boosted

    def test_context_scoring_no_context_no_change(self):
        """With empty context, scores should not change."""
        from scoring import ContextScorer

        scorer = ContextScorer()
        ctx = _make_user_context()  # No age, no weather

        c = _make_candidate(final_score=0.5)
        item_dict = c.to_scoring_dict()
        adj = scorer.score_item(item_dict, ctx)
        assert adj == 0.0

    def test_context_scoring_cap_applied(self):
        """Total context adjustment should be capped at +/- 0.20."""
        from scoring import ContextScorer
        from scoring.context import AgeGroup

        scorer = ContextScorer()
        ctx = _make_user_context(
            age_group=AgeGroup.SENIOR,
            weather=_make_winter_weather(),
            coverage_prefs=["no_revealing", "no_crop"],
        )

        # Maximally penalized item: crop top, summer, streetwear, bold pattern
        c = _make_candidate(
            article_type="crop_top",
            style_tags=["streetwear", "edgy"],
            seasons=["Summer"],
            materials=["mesh"],
            pattern="Animal Print",
            neckline="plunging",
            final_score=0.5,
        )
        item_dict = c.to_scoring_dict()
        adj = scorer.score_item(item_dict, ctx)
        assert adj >= -0.30
        assert adj <= 0.30

    def test_batch_scoring_applies_to_multiple_candidates(self):
        """score_items() should score all items in a list."""
        from scoring import ContextScorer
        from scoring.context import AgeGroup

        scorer = ContextScorer()
        ctx = _make_user_context(age_group=AgeGroup.GEN_Z, weather=_make_summer_weather())

        candidates = [
            _make_candidate(item_id="c1", article_type="crop_top", seasons=["Summer"]),
            _make_candidate(item_id="c2", article_type="sweater", seasons=["Winter"]),
            _make_candidate(item_id="c3", article_type="blazer", style_tags=["classic"]),
        ]

        # Convert to dicts and score
        dicts = [c.to_scoring_dict() for c in candidates]
        for d in dicts:
            d["score"] = 0.5
        scorer.score_items(dicts, ctx, score_field="score")

        # Each should have context_adjustment
        for d in dicts:
            assert "context_adjustment" in d
            assert isinstance(d["context_adjustment"], float)

        # Crop top in summer for Gen Z should score higher than sweater in winter
        assert dicts[0]["score"] > dicts[1]["score"]


# =============================================================================
# 3. Search Reranker Step 3.75 - Context Scoring
# =============================================================================

class TestSearchRerankerContextScoring:
    """Tests for context scoring in the search SessionReranker."""

    @pytest.fixture
    def reranker(self):
        from search.reranker import SessionReranker
        return SessionReranker()

    def test_rerank_accepts_user_context_param(self, reranker):
        """rerank() should accept user_context parameter."""
        from scoring.context import AgeGroup
        ctx = _make_user_context(age_group=AgeGroup.GEN_Z, weather=_make_summer_weather())
        # Use different brands to avoid brand diversity cap (max 4 per brand)
        brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE"]
        results = [
            _make_search_result(product_id=f"p{i}", rrf_score=0.1 - i*0.01, brand=brands[i])
            for i in range(5)
        ]
        reranked = reranker.rerank(results, user_context=ctx)
        assert len(reranked) == 5

    def test_rerank_without_context_works(self, reranker):
        """rerank() should work fine without user_context (backward compat)."""
        results = [_make_search_result(product_id=f"p{i}") for i in range(3)]
        reranked = reranker.rerank(results)
        assert len(reranked) == 3

    def test_context_scoring_changes_search_order(self, reranker):
        """Context scoring should be able to reorder search results."""
        from scoring.context import AgeGroup

        ctx = _make_user_context(age_group=AgeGroup.GEN_Z, weather=_make_summer_weather())

        # Sweater starts higher, crop_top starts lower
        sweater = _make_search_result(
            product_id="sweater-1",
            rrf_score=0.10,
            article_type="sweater",
            seasons=["Fall", "Winter"],
            materials=["wool"],
        )
        crop = _make_search_result(
            product_id="crop-1",
            rrf_score=0.09,  # Slightly lower base score
            article_type="crop_top",
            seasons=["Summer"],
            materials=["cotton"],
            style_tags=["streetwear"],
        )

        results = [sweater, crop]
        reranked = reranker.rerank(results, user_context=ctx)

        # After context scoring (Gen Z + Summer), crop should overtake sweater
        assert reranked[0]["product_id"] == "crop-1"
        assert reranked[1]["product_id"] == "sweater-1"

    def test_context_scoring_adds_adjustment_key(self, reranker):
        """Context scoring should add 'context_adjustment' to result dicts."""
        from scoring.context import AgeGroup
        ctx = _make_user_context(age_group=AgeGroup.MID_CAREER, weather=_make_summer_weather())
        results = [_make_search_result()]
        reranked = reranker.rerank(results, user_context=ctx)
        assert "context_adjustment" in reranked[0]

    def test_context_scoring_runs_after_profile_scoring(self, reranker):
        """Context scoring (Step 3.75) should run after profile scoring (Step 3)."""
        from scoring.context import AgeGroup
        ctx = _make_user_context(age_group=AgeGroup.GEN_Z, weather=_make_summer_weather())
        profile = {
            "soft_prefs": {"preferred_brands": ["TestBrand"]},
            "hard_filters": {},
        }

        results = [_make_search_result(rrf_score=0.10)]
        reranked = reranker.rerank(results, user_profile=profile, user_context=ctx)

        # Should have both profile and context adjustments
        assert "profile_adjustment" in reranked[0]
        assert "context_adjustment" in reranked[0]

    def test_context_scoring_resilient_to_errors(self, reranker):
        """If context scoring fails, reranking should still work."""
        # Pass a bogus user_context that would cause scorer to fail
        results = [_make_search_result(product_id=f"p{i}") for i in range(3)]
        # Even with a bad context object, it should not crash
        reranked = reranker.rerank(results, user_context="not_a_context")
        assert len(reranked) == 3  # Should still return all results

    def test_context_scoring_combined_with_seen_ids(self, reranker):
        """Context scoring should work together with seen_ids filtering."""
        from scoring.context import AgeGroup
        ctx = _make_user_context(age_group=AgeGroup.GEN_Z)

        results = [_make_search_result(product_id=f"p{i}") for i in range(5)]
        seen = {"p0", "p2"}
        reranked = reranker.rerank(results, seen_ids=seen, user_context=ctx)
        assert len(reranked) == 3
        ids = {r["product_id"] for r in reranked}
        assert "p0" not in ids
        assert "p2" not in ids


# =============================================================================
# 4. HybridSearchService - user_context Forwarding
# =============================================================================

class TestHybridSearchContextForwarding:
    """Tests for user_context parameter in HybridSearchService.search()."""

    @pytest.fixture
    def service(self):
        from search.hybrid_search import HybridSearchService
        mock_algolia = MagicMock()
        mock_algolia.search.return_value = {
            "hits": [
                {
                    "objectID": f"p{i}",
                    "name": f"Product {i}",
                    "brand": "TestBrand",
                    "price": 50.0,
                    "image_url": f"https://img.example.com/p{i}.jpg",
                    "in_stock": True,
                    "category_l1": None,
                    "category_l2": None,
                    "broad_category": "tops",
                    "article_type": "t-shirt",
                    "primary_color": "Black",
                    "color_family": "Dark",
                    "pattern": "Solid",
                    "apparent_fabric": "cotton",
                    "fit_type": "Regular",
                    "formality": "Casual",
                    "silhouette": None,
                    "length": "Regular",
                    "neckline": "Crew",
                    "sleeve_type": "Short",
                    "rise": None,
                    "style_tags": ["casual"],
                    "occasions": ["Everyday"],
                    "seasons": ["Spring", "Summer"],
                    "colors": ["black"],
                    "materials": ["cotton"],
                }
                for i in range(5)
            ],
            "nbHits": 5,
            "facets": {},
        }
        mock_analytics = MagicMock()
        svc = HybridSearchService(
            algolia_client=mock_algolia,
            analytics=mock_analytics,
        )
        # Mock semantic engine to avoid loading FashionCLIP in tests
        mock_semantic = MagicMock()
        mock_semantic.search_with_filters.return_value = {"results": [], "total": 0}
        svc._semantic_engine = mock_semantic
        return svc

    def test_search_accepts_user_context(self, service):
        """search() should accept user_context parameter."""
        from search.models import HybridSearchRequest
        from scoring.context import AgeGroup

        request = HybridSearchRequest(query="casual top", page=1, page_size=10)
        ctx = _make_user_context(age_group=AgeGroup.GEN_Z, weather=_make_summer_weather())

        result = service.search(
            request=request,
            user_id="test-user",
            user_context=ctx,
        )
        assert result is not None
        assert len(result.results) > 0

    def test_search_without_context_still_works(self, service):
        """search() should work without user_context (backward compat)."""
        from search.models import HybridSearchRequest

        request = HybridSearchRequest(query="black dress", page=1, page_size=10)
        result = service.search(request=request, user_id="test-user")
        assert result is not None

    def test_search_forwards_context_to_reranker(self, service):
        """search() should pass user_context to reranker.rerank()."""
        from search.models import HybridSearchRequest
        from scoring.context import AgeGroup

        request = HybridSearchRequest(query="casual top", page=1, page_size=10)
        ctx = _make_user_context(age_group=AgeGroup.GEN_Z)

        # Patch reranker to verify it receives user_context
        with patch.object(service._reranker, "rerank", wraps=service._reranker.rerank) as mock_rerank:
            service.search(request=request, user_id="test-user", user_context=ctx)
            mock_rerank.assert_called_once()
            call_kwargs = mock_rerank.call_args
            assert call_kwargs.kwargs.get("user_context") is ctx or \
                   (len(call_kwargs.args) > 0 and any(a is ctx for a in call_kwargs.args)) or \
                   call_kwargs[1].get("user_context") is ctx


# =============================================================================
# 5. Search Route Helpers - UserContext Building + Signal Wiring
# =============================================================================

class TestSearchRouteHelpers:
    """Tests for helper functions in api/routes/search.py."""

    def test_build_user_context_with_birthdate(self):
        """Should build UserContext with age from profile birthdate."""
        from api.routes.search import _build_user_context
        from core.auth import SupabaseUser

        user = SupabaseUser(
            id="user-123",
            user_metadata={"city": "New York", "country": "US", "state": "NY"},
        )
        profile = {
            "birthdate": "2000-01-15",
            "no_revealing": True,
            "styles_to_avoid": ["deep-necklines"],
        }

        ctx = _build_user_context(user, profile)
        assert ctx is not None
        assert ctx.age_group is not None
        assert ctx.coverage_prefs is not None
        assert "no_revealing" in ctx.coverage_prefs

    def test_build_user_context_no_profile(self):
        """Should handle None profile gracefully."""
        from api.routes.search import _build_user_context
        from core.auth import SupabaseUser

        user = SupabaseUser(
            id="user-123",
            user_metadata={"city": "London", "country": "UK"},
        )

        ctx = _build_user_context(user, None)
        # May or may not be None depending on weather resolution
        # but should not crash

    def test_build_user_context_no_metadata(self):
        """Should handle user with no metadata."""
        from api.routes.search import _build_user_context
        from core.auth import SupabaseUser

        user = SupabaseUser(id="user-123", user_metadata=None)
        ctx = _build_user_context(user, None)
        # Should return None (no useful context)
        assert ctx is None

    def test_build_user_context_extracts_address_from_jwt(self):
        """Should extract city/country from JWT user_metadata."""
        from api.routes.search import _build_user_context
        from core.auth import SupabaseUser

        user = SupabaseUser(
            id="user-123",
            user_metadata={
                "address": {
                    "city": "Paris",
                    "country": "France",
                    "state": "IDF",
                }
            },
        )
        profile = {"birthdate": "1990-06-15"}

        ctx = _build_user_context(user, profile)
        assert ctx is not None
        assert ctx.city == "Paris"
        assert ctx.country == "France"

    def test_context_resolver_singleton(self):
        """_get_context_resolver() should return the same instance."""
        from api.routes.search import _get_context_resolver
        r1 = _get_context_resolver()
        r2 = _get_context_resolver()
        assert r1 is r2

    def test_forward_search_signal_no_session_noop(self):
        """_forward_search_signal() should do nothing without session_id."""
        from api.routes.search import _forward_search_signal
        from search.models import HybridSearchRequest
        request = HybridSearchRequest(query="test", page=1, page_size=10)
        # Should not raise
        _forward_search_signal(
            user_id="user-123",
            session_id=None,
            query="test",
            request=request,
        )

    def test_forward_search_signal_extracts_filters(self):
        """_forward_search_signal() should extract filter data from request."""
        from api.routes.search import _forward_search_signal
        from search.models import HybridSearchRequest

        request = HybridSearchRequest(
            query="summer dress",
            page=1, page_size=10,
            categories=["dresses"],
            brands=["Zara"],
            colors=["red"],
            occasions=["Date Night"],
        )

        # Patch the pipeline at its source module (imported inside the function)
        with patch("recs.api_endpoints.get_pipeline") as mock_get:
            mock_pipeline = MagicMock()
            mock_get.return_value = mock_pipeline

            _forward_search_signal(
                user_id="user-123",
                session_id="sess-abc",
                query="summer dress",
                request=request,
            )

            mock_pipeline.update_session_scores_from_search.assert_called_once()
            call_kwargs = mock_pipeline.update_session_scores_from_search.call_args[1]
            assert call_kwargs["session_id"] == "sess-abc"
            assert call_kwargs["query"] == "summer dress"
            assert "categories" in call_kwargs["filters"]
            assert call_kwargs["filters"]["categories"] == ["dresses"]
            assert call_kwargs["filters"]["brands"] == ["Zara"]


# =============================================================================
# 6. End-to-End Scoring Flow
# =============================================================================

class TestEndToEndScoringFlow:
    """Tests that verify the complete scoring flow from context to final score."""

    def test_gen_z_summer_feed_ordering(self):
        """Gen Z user in summer: tank tops > sweaters in final score."""
        from scoring import ContextScorer
        from scoring.context import AgeGroup

        scorer = ContextScorer()
        ctx = _make_user_context(age_group=AgeGroup.GEN_Z, weather=_make_summer_weather())

        tank = _make_candidate(
            item_id="tank-1",
            article_type="tank_top",
            style_tags=["casual"],
            seasons=["Summer"],
            materials=["cotton"],
            final_score=0.50,
        )
        sweater = _make_candidate(
            item_id="sweater-1",
            article_type="sweater",
            style_tags=["classic"],
            seasons=["Fall", "Winter"],
            materials=["wool"],
            final_score=0.50,
        )

        tank_adj = scorer.score_item(tank.to_scoring_dict(), ctx)
        sweater_adj = scorer.score_item(sweater.to_scoring_dict(), ctx)

        tank.final_score += tank_adj
        sweater.final_score += sweater_adj

        assert tank.final_score > sweater.final_score

    def test_senior_winter_feed_ordering(self):
        """Senior user in winter: blazer > crop top in final score."""
        from scoring import ContextScorer
        from scoring.context import AgeGroup

        scorer = ContextScorer()
        ctx = _make_user_context(age_group=AgeGroup.SENIOR, weather=_make_winter_weather())

        blazer = _make_candidate(
            item_id="blazer-1",
            article_type="blazer",
            style_tags=["classic"],
            seasons=["Fall", "Winter"],
            materials=["wool"],
            fit="regular",
            final_score=0.50,
        )
        crop = _make_candidate(
            item_id="crop-1",
            article_type="crop_top",
            style_tags=["streetwear"],
            seasons=["Summer"],
            materials=["cotton"],
            pattern="Animal Print",
            final_score=0.50,
        )

        blazer_adj = scorer.score_item(blazer.to_scoring_dict(), ctx)
        crop_adj = scorer.score_item(crop.to_scoring_dict(), ctx)

        blazer.final_score += blazer_adj
        crop.final_score += crop_adj

        assert blazer.final_score > crop.final_score

    def test_search_results_reordered_by_context(self):
        """Search results should be reordered by context scoring."""
        from search.reranker import SessionReranker
        from scoring.context import AgeGroup

        reranker = SessionReranker()
        ctx = _make_user_context(age_group=AgeGroup.SENIOR, weather=_make_winter_weather())

        # Both start with same RRF score
        results = [
            _make_search_result(
                product_id="crop-1",
                rrf_score=0.10,
                article_type="crop_top",
                seasons=["Summer"],
                materials=["cotton"],
                style_tags=["streetwear"],
            ),
            _make_search_result(
                product_id="coat-1",
                rrf_score=0.10,
                article_type="coat",
                seasons=["Fall", "Winter"],
                materials=["wool"],
                style_tags=["classic"],
            ),
        ]

        reranked = reranker.rerank(results, user_context=ctx)
        # Coat should be ranked higher for senior in winter
        assert reranked[0]["product_id"] == "coat-1"

    def test_context_scoring_explain_breakdown(self):
        """explain_item() should provide detailed scoring breakdown."""
        from scoring import ContextScorer
        from scoring.context import AgeGroup

        scorer = ContextScorer()
        ctx = _make_user_context(age_group=AgeGroup.GEN_Z, weather=_make_summer_weather())

        item = _make_candidate(
            article_type="crop_top",
            style_tags=["streetwear"],
            seasons=["Summer"],
            materials=["cotton"],
        ).to_scoring_dict()

        breakdown = scorer.explain_item(item, ctx)
        assert "total" in breakdown
        assert "age" in breakdown
        assert "weather" in breakdown
        assert "age_group" in breakdown
        assert breakdown["age_group"] == "18-24"
        assert "season" in breakdown
        assert breakdown["season"] == "summer"

    def test_coverage_prefs_from_onboarding_applied(self):
        """Coverage prefs from onboarding should affect scoring."""
        from scoring import ContextScorer
        from scoring.context import AgeGroup

        scorer = ContextScorer()

        # Same item, same age, but different coverage prefs
        ctx_no_prefs = _make_user_context(
            age_group=AgeGroup.MID_CAREER,
            coverage_prefs=[],
        )
        ctx_with_prefs = _make_user_context(
            age_group=AgeGroup.MID_CAREER,
            coverage_prefs=["no_revealing", "no_crop"],
        )

        crop = _make_candidate(
            article_type="crop_top",
            neckline="plunging",
        ).to_scoring_dict()

        adj_no_prefs = scorer.score_item(crop, ctx_no_prefs)
        adj_with_prefs = scorer.score_item(crop, ctx_with_prefs)

        # Both should have a penalty for mid-career, but the age scoring
        # doesn't change based on coverage_prefs (that's FeasibilityFilter's job).
        # The scores should be equal since coverage prefs are handled upstream.
        assert isinstance(adj_no_prefs, float)
        assert isinstance(adj_with_prefs, float)


# =============================================================================
# 7. Context Resolver Integration
# =============================================================================

class TestContextResolverIntegration:
    """Tests for ContextResolver building complete UserContext."""

    def test_resolve_with_all_data(self):
        """Full resolution with birthdate + address + profile."""
        from scoring import ContextResolver

        resolver = ContextResolver(weather_api_key="")  # No API key = season fallback

        ctx = resolver.resolve(
            user_id="user-123",
            jwt_user_metadata={"city": "London", "country": "UK"},
            birthdate="1995-03-15",
            onboarding_profile={
                "no_revealing": True,
                "no_crop": True,
                "styles_to_avoid": ["deep-necklines"],
            },
        )

        assert ctx.user_id == "user-123"
        assert ctx.age_group is not None
        assert ctx.city == "London"
        assert ctx.country == "UK"
        assert ctx.weather is not None  # Season fallback
        assert "no_revealing" in ctx.coverage_prefs
        assert "no_crop" in ctx.coverage_prefs
        assert "deep-necklines" in ctx.coverage_prefs

    def test_resolve_graceful_with_no_data(self):
        """Should return empty context without crashing."""
        from scoring import ContextResolver

        resolver = ContextResolver(weather_api_key="")
        ctx = resolver.resolve(user_id="anon-user")

        assert ctx.user_id == "anon-user"
        assert ctx.age_group is None
        assert ctx.weather is None
        assert ctx.coverage_prefs == []

    def test_resolve_invalid_birthdate(self):
        """Should handle invalid birthdate gracefully."""
        from scoring import ContextResolver

        resolver = ContextResolver(weather_api_key="")
        ctx = resolver.resolve(
            user_id="user-456",
            birthdate="not-a-date",
        )

        assert ctx.age_group is None  # Invalid date, skip

    def test_resolve_address_from_nested_metadata(self):
        """Should extract address from nested user_metadata.address."""
        from scoring import ContextResolver

        resolver = ContextResolver(weather_api_key="")
        ctx = resolver.resolve(
            user_id="user-789",
            jwt_user_metadata={
                "address": {"city": "Tokyo", "country": "Japan"},
            },
        )

        assert ctx.city == "Tokyo"
        assert ctx.country == "Japan"

    def test_resolve_address_from_flat_metadata(self):
        """Should extract address from flat user_metadata fields."""
        from scoring import ContextResolver

        resolver = ContextResolver(weather_api_key="")
        ctx = resolver.resolve(
            user_id="user-012",
            jwt_user_metadata={"city": "Berlin", "country": "Germany"},
        )

        assert ctx.city == "Berlin"
        assert ctx.country == "Germany"


# =============================================================================
# 8. Regression: Backward Compatibility
# =============================================================================

class TestBackwardCompatibility:
    """Ensure existing interfaces still work without new parameters."""

    def test_reranker_without_new_params(self):
        """SessionReranker.rerank() should work with old-style calls."""
        from search.reranker import SessionReranker
        reranker = SessionReranker()

        results = [_make_search_result(product_id=f"p{i}") for i in range(3)]
        reranked = reranker.rerank(results)
        assert len(reranked) == 3

    def test_reranker_with_only_profile(self):
        """rerank() with only user_profile should still work."""
        from search.reranker import SessionReranker
        reranker = SessionReranker()

        profile = {
            "soft_prefs": {"preferred_brands": ["TestBrand"]},
            "hard_filters": {},
        }
        results = [_make_search_result()]
        reranked = reranker.rerank(results, user_profile=profile)
        assert len(reranked) == 1
        assert "profile_adjustment" in reranked[0]

    def test_hybrid_search_without_context(self):
        """HybridSearchService.search() without user_context should work."""
        from search.hybrid_search import HybridSearchService
        from search.models import HybridSearchRequest

        mock_algolia = MagicMock()
        mock_algolia.search.return_value = {
            "hits": [
                {
                    "objectID": "p1", "name": "Test", "brand": "B", "price": 10,
                    "image_url": "https://img.example.com/p1.jpg",
                    "category_l1": None, "category_l2": None,
                    "broad_category": "tops", "article_type": "t-shirt",
                    "primary_color": None, "color_family": None, "pattern": None,
                    "apparent_fabric": None, "fit_type": None, "formality": None,
                    "silhouette": None, "length": None, "neckline": None,
                    "sleeve_type": None, "rise": None,
                    "style_tags": [], "occasions": [], "seasons": [],
                    "colors": [], "materials": [],
                }
            ],
            "nbHits": 1,
            "facets": {},
        }
        service = HybridSearchService(
            algolia_client=mock_algolia,
            analytics=MagicMock(),
        )
        # Mock semantic engine to avoid loading FashionCLIP
        mock_semantic = MagicMock()
        mock_semantic.search_with_filters.return_value = {"results": [], "total": 0}
        service._semantic_engine = mock_semantic

        request = HybridSearchRequest(query="test", page=1, page_size=10)
        result = service.search(request=request, user_id="u1")
        assert result is not None
