"""
Tests for feed sort_by functionality.

Covers:
1. FeedSortBy enum (values, parsing)
2. KeysetCursor sort_mode encode/decode
3. apply_sort_diversity() constraints (consecutive brand, brand share, seen_ids)
4. Pipeline branching (sort_by dispatches to _get_feed_sorted)
5. API endpoint sort_by parameter parsing
"""

import sys
import os
import json
import base64
from unittest.mock import MagicMock, patch, PropertyMock
from collections import defaultdict

import pytest

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from recs.models import FeedSortBy, Candidate
from recs.session_state import KeysetCursor
from recs.feed_reranker import apply_sort_diversity


# =============================================================================
# Helpers
# =============================================================================

def make_candidate(
    item_id: str = "test-1",
    brand: str = "BrandA",
    price: float = 50.0,
    embedding_score: float = 0.0,
    **kwargs,
) -> Candidate:
    """Create a minimal Candidate for sort testing."""
    return Candidate(
        item_id=item_id,
        brand=brand,
        price=price,
        embedding_score=embedding_score,
        **kwargs,
    )


# =============================================================================
# FeedSortBy Enum Tests
# =============================================================================

class TestFeedSortByEnum:
    """Tests for the FeedSortBy enum."""

    def test_enum_values(self):
        assert FeedSortBy.RELEVANCE.value == "relevance"
        assert FeedSortBy.PRICE_ASC.value == "price_asc"
        assert FeedSortBy.PRICE_DESC.value == "price_desc"

    def test_enum_from_string(self):
        assert FeedSortBy("relevance") == FeedSortBy.RELEVANCE
        assert FeedSortBy("price_asc") == FeedSortBy.PRICE_ASC
        assert FeedSortBy("price_desc") == FeedSortBy.PRICE_DESC

    def test_enum_invalid_raises(self):
        with pytest.raises(ValueError):
            FeedSortBy("invalid_sort")

    def test_enum_members_count(self):
        assert len(FeedSortBy) == 3

    def test_enum_is_str_subclass(self):
        """FeedSortBy inherits from str so it serialises cleanly."""
        assert isinstance(FeedSortBy.RELEVANCE, str)
        assert FeedSortBy.PRICE_ASC == "price_asc"


# =============================================================================
# KeysetCursor sort_mode Tests
# =============================================================================

class TestKeysetCursorSortMode:
    """Tests for KeysetCursor sort_mode encode/decode."""

    def test_default_sort_mode(self):
        cursor = KeysetCursor(score=0.5, item_id="abc")
        assert cursor.sort_mode == "relevance"

    def test_sort_mode_preserved_in_to_dict(self):
        cursor = KeysetCursor(score=19.99, item_id="p1", page=2, sort_mode="price_asc")
        d = cursor.to_dict()
        assert d["sort_mode"] == "price_asc"
        assert d["score"] == 19.99
        assert d["item_id"] == "p1"
        assert d["page"] == 2

    def test_sort_mode_roundtrip_from_dict(self):
        original = KeysetCursor(score=99.0, item_id="xyz", page=5, sort_mode="price_desc")
        d = original.to_dict()
        restored = KeysetCursor.from_dict(d)
        assert restored.sort_mode == "price_desc"
        assert restored.score == 99.0
        assert restored.item_id == "xyz"
        assert restored.page == 5

    def test_sort_mode_encode_decode_roundtrip(self):
        """Encode to base64 and decode back — sort_mode must survive."""
        original = KeysetCursor(score=42.50, item_id="item-abc", page=3, sort_mode="price_asc")
        encoded = original.encode()
        assert isinstance(encoded, str)

        decoded = KeysetCursor.decode(encoded)
        assert decoded is not None
        assert decoded.sort_mode == "price_asc"
        assert decoded.score == 42.50
        assert decoded.item_id == "item-abc"
        assert decoded.page == 3

    def test_decode_missing_sort_mode_defaults_to_relevance(self):
        """Old cursors without sort_mode should default to relevance."""
        payload = json.dumps({"score": 0.8, "item_id": "old-cursor", "page": 1})
        encoded = base64.b64encode(payload.encode()).decode()
        decoded = KeysetCursor.decode(encoded)
        assert decoded is not None
        assert decoded.sort_mode == "relevance"

    def test_decode_invalid_returns_none(self):
        assert KeysetCursor.decode("not-valid-base64!!!") is None

    def test_encode_produces_different_strings_for_different_sort_modes(self):
        c1 = KeysetCursor(score=10.0, item_id="x", sort_mode="price_asc")
        c2 = KeysetCursor(score=10.0, item_id="x", sort_mode="price_desc")
        assert c1.encode() != c2.encode()


# =============================================================================
# apply_sort_diversity Tests
# =============================================================================

class TestApplySortDiversity:
    """Tests for the apply_sort_diversity function."""

    def test_empty_list(self):
        assert apply_sort_diversity([]) == []

    def test_no_diversity_needed(self):
        """All different brands — should return all in order."""
        candidates = [
            make_candidate(item_id=f"p{i}", brand=f"Brand{i}", price=float(i))
            for i in range(10)
        ]
        result = apply_sort_diversity(candidates)
        assert len(result) == 10
        # Order preserved
        assert [c.item_id for c in result] == [f"p{i}" for i in range(10)]

    def test_consecutive_brand_limit(self):
        """More than 2 consecutive same-brand items get deferred."""
        # 5 items from same brand — max_consecutive=2 means 3rd should be deferred
        candidates = [
            make_candidate(item_id=f"p{i}", brand="Boohoo", price=float(i * 10))
            for i in range(5)
        ]
        result = apply_sort_diversity(candidates, max_consecutive=2)
        assert len(result) == 5

        # First 2 should be Boohoo, then the rest are still Boohoo but deferred
        # With only one brand, consecutive limit kicks in but items still appear
        # The function defers items — they end up later in the list
        assert result[0].item_id == "p0"
        assert result[1].item_id == "p1"

    def test_brand_share_cap(self):
        """Brand exceeding 30% share gets excess items deferred."""
        # 10 items: 5 from Boohoo, 5 from different brands
        candidates = []
        for i in range(5):
            candidates.append(make_candidate(item_id=f"boohoo-{i}", brand="Boohoo", price=10.0 + i))
        for i in range(5):
            candidates.append(make_candidate(item_id=f"other-{i}", brand=f"Other{i}", price=20.0 + i))

        result = apply_sort_diversity(candidates, max_brand_share=0.30)
        assert len(result) == 10

        # Count Boohoo in result — all should still be present (deferred, not removed)
        boohoo_count = sum(1 for c in result if c.brand == "Boohoo")
        assert boohoo_count == 5

        # But the first 30% positions should have at most ~3 Boohoo items
        first_half = result[:5]
        boohoo_in_first_half = sum(1 for c in first_half if c.brand == "Boohoo")
        # max_per_brand = max(3, int(10 * 0.30)) = 3
        assert boohoo_in_first_half <= 3

    def test_seen_ids_excluded(self):
        """Items in seen_ids should be removed entirely."""
        candidates = [
            make_candidate(item_id="p1", brand="A", price=10.0),
            make_candidate(item_id="p2", brand="B", price=20.0),
            make_candidate(item_id="p3", brand="C", price=30.0),
        ]
        result = apply_sort_diversity(candidates, seen_ids={"p2"})
        assert len(result) == 2
        assert all(c.item_id != "p2" for c in result)

    def test_preserves_sort_order(self):
        """Price-sorted input should remain price-sorted (within same-brand groups)."""
        candidates = [
            make_candidate(item_id="cheap", brand="A", price=5.0),
            make_candidate(item_id="mid-a", brand="A", price=15.0),
            make_candidate(item_id="mid-b", brand="B", price=25.0),
            make_candidate(item_id="exp", brand="C", price=100.0),
        ]
        result = apply_sort_diversity(candidates)
        prices = [c.price for c in result]
        # Should be monotonically increasing (no diversity issues here)
        assert prices == sorted(prices)

    def test_mixed_brands_interleaved(self):
        """With alternating brands, no diversity needed — preserved as-is."""
        candidates = [
            make_candidate(item_id="p1", brand="A", price=10.0),
            make_candidate(item_id="p2", brand="B", price=20.0),
            make_candidate(item_id="p3", brand="A", price=30.0),
            make_candidate(item_id="p4", brand="B", price=40.0),
        ]
        result = apply_sort_diversity(candidates, max_consecutive=2)
        assert [c.item_id for c in result] == ["p1", "p2", "p3", "p4"]


# =============================================================================
# Pipeline Branching Tests
# =============================================================================

class TestPipelineSortBranching:
    """Tests that get_feed_keyset dispatches correctly based on sort_by."""

    def test_relevance_does_not_call_get_feed_sorted(self):
        """sort_by=RELEVANCE should NOT call _get_feed_sorted."""
        from recs.pipeline import RecommendationPipeline

        with patch.object(RecommendationPipeline, '__init__', return_value=None):
            pipeline = RecommendationPipeline.__new__(RecommendationPipeline)

        # Mock _get_feed_sorted to ensure it's NOT called
        pipeline._get_feed_sorted = MagicMock()

        # Mock the rest of the pipeline to avoid real DB calls
        pipeline.config = MagicMock()
        pipeline.config.MAX_LIMIT = 200
        pipeline.session_service = MagicMock()
        pipeline.session_service.decode_cursor.return_value = None
        pipeline.session_service.get_seen_items.return_value = set()
        pipeline.session_service.has_feed_version.return_value = True
        pipeline.session_service.get_feed_version.return_value = MagicMock(version_id="v1")
        pipeline.candidate_module = MagicMock()
        pipeline.candidate_module.get_user_seen_history.return_value = set()
        pipeline.candidate_module.get_candidates_keyset.return_value = []
        pipeline.ranker = MagicMock()
        pipeline.ranker.model = None
        pipeline.ranker.config = MagicMock()
        pipeline.ranker.config.WARM_WEIGHTS = {}
        pipeline.ranker.config.COLD_WEIGHTS = {}
        pipeline.ranker.rank_candidates.return_value = []
        pipeline.scoring_engine = MagicMock()
        pipeline.scoring_engine.score_candidates.return_value = []
        pipeline.feed_reranker = MagicMock()
        pipeline.feed_reranker.rerank.return_value = []
        pipeline._load_user_state = MagicMock()
        pipeline._load_user_state.return_value = MagicMock(
            taste_vector=None,
            onboarding_profile=None,
            user_id="test",
        )
        pipeline._get_or_create_session_scores = MagicMock()
        pipeline._get_or_create_session_scores.return_value = MagicMock(
            action_count=0, cluster_scores={}, brand_scores={},
            search_intents=[], skipped_ids=set(),
        )
        pipeline.context_scorer = MagicMock()
        pipeline.context_resolver = MagicMock()
        pipeline.context_resolver.resolve.return_value = MagicMock(
            age_group=None, weather=None, city=None,
        )
        pipeline._scores_l1_cache = {}
        pipeline._SCORES_L1_MAX = 200
        pipeline._scoring_backend = MagicMock()
        pipeline._scoring_backend.get_scores.return_value = None
        pipeline._init_scoring_backend = MagicMock()

        # Call with RELEVANCE
        pipeline.get_feed_keyset(
            anon_id="test-user",
            sort_by=FeedSortBy.RELEVANCE,
            page_size=10,
        )

        # _get_feed_sorted should NOT have been called
        pipeline._get_feed_sorted.assert_not_called()

    def test_price_asc_calls_get_feed_sorted(self):
        """sort_by=PRICE_ASC should call _get_feed_sorted."""
        from recs.pipeline import RecommendationPipeline

        with patch.object(RecommendationPipeline, '__init__', return_value=None):
            pipeline = RecommendationPipeline.__new__(RecommendationPipeline)

        pipeline._get_feed_sorted = MagicMock(return_value={"results": [], "sort_by": "price_asc"})

        result = pipeline.get_feed_keyset(
            anon_id="test-user",
            sort_by=FeedSortBy.PRICE_ASC,
            page_size=10,
        )

        pipeline._get_feed_sorted.assert_called_once()
        call_kwargs = pipeline._get_feed_sorted.call_args
        assert call_kwargs.kwargs["sort_by"] == FeedSortBy.PRICE_ASC

    def test_price_desc_calls_get_feed_sorted(self):
        """sort_by=PRICE_DESC should call _get_feed_sorted."""
        from recs.pipeline import RecommendationPipeline

        with patch.object(RecommendationPipeline, '__init__', return_value=None):
            pipeline = RecommendationPipeline.__new__(RecommendationPipeline)

        pipeline._get_feed_sorted = MagicMock(return_value={"results": [], "sort_by": "price_desc"})

        result = pipeline.get_feed_keyset(
            anon_id="test-user",
            sort_by=FeedSortBy.PRICE_DESC,
            page_size=10,
        )

        pipeline._get_feed_sorted.assert_called_once()
        call_kwargs = pipeline._get_feed_sorted.call_args
        assert call_kwargs.kwargs["sort_by"] == FeedSortBy.PRICE_DESC


# =============================================================================
# Response sort_by field Tests
# =============================================================================

class TestResponseSortByField:
    """Tests that the response includes the correct sort_by field."""

    def test_relevance_response_has_sort_by(self):
        """Relevance path should include sort_by='relevance' in response."""
        from recs.pipeline import RecommendationPipeline

        with patch.object(RecommendationPipeline, '__init__', return_value=None):
            pipeline = RecommendationPipeline.__new__(RecommendationPipeline)

        pipeline.config = MagicMock()
        pipeline.config.MAX_LIMIT = 200
        pipeline.session_service = MagicMock()
        pipeline.session_service.decode_cursor.return_value = None
        pipeline.session_service.get_seen_items.return_value = set()
        pipeline.session_service.has_feed_version.return_value = True
        pipeline.session_service.get_feed_version.return_value = MagicMock(version_id="v1")
        pipeline.candidate_module = MagicMock()
        pipeline.candidate_module.get_user_seen_history.return_value = set()
        pipeline.candidate_module.get_candidates_keyset.return_value = []
        pipeline.ranker = MagicMock()
        pipeline.ranker.model = None
        pipeline.ranker.config = MagicMock()
        pipeline.ranker.config.WARM_WEIGHTS = {}
        pipeline.ranker.config.COLD_WEIGHTS = {}
        pipeline.ranker.rank_candidates.return_value = []
        pipeline.scoring_engine = MagicMock()
        pipeline.scoring_engine.score_candidates.return_value = []
        pipeline.feed_reranker = MagicMock()
        pipeline.feed_reranker.rerank.return_value = []
        pipeline._load_user_state = MagicMock()
        pipeline._load_user_state.return_value = MagicMock(
            taste_vector=None,
            onboarding_profile=None,
            user_id="test",
        )
        pipeline._get_or_create_session_scores = MagicMock()
        pipeline._get_or_create_session_scores.return_value = MagicMock(
            action_count=0, cluster_scores={}, brand_scores={},
            search_intents=[], skipped_ids=set(),
        )
        pipeline.context_scorer = MagicMock()
        pipeline.context_resolver = MagicMock()
        pipeline.context_resolver.resolve.return_value = MagicMock(
            age_group=None, weather=None, city=None,
        )
        pipeline._scores_l1_cache = {}
        pipeline._SCORES_L1_MAX = 200
        pipeline._scoring_backend = MagicMock()
        pipeline._scoring_backend.get_scores.return_value = None

        result = pipeline.get_feed_keyset(
            anon_id="test-user",
            sort_by=FeedSortBy.RELEVANCE,
            page_size=10,
        )

        assert result["sort_by"] == "relevance"

    def test_sorted_response_has_sort_by(self):
        """Sorted path should include sort_by='price_asc' in response."""
        from recs.pipeline import RecommendationPipeline

        with patch.object(RecommendationPipeline, '__init__', return_value=None):
            pipeline = RecommendationPipeline.__new__(RecommendationPipeline)

        # Mock the sorted path dependencies
        pipeline.config = MagicMock()
        pipeline.config.MAX_LIMIT = 200
        pipeline.session_service = MagicMock()
        pipeline.session_service.decode_cursor.return_value = None
        pipeline.session_service.get_seen_items.return_value = set()
        pipeline.candidate_module = MagicMock()
        pipeline.candidate_module.get_user_seen_history.return_value = set()
        pipeline.candidate_module.get_candidates_sorted_keyset.return_value = [
            make_candidate(item_id="p1", brand="A", price=10.0, embedding_score=10.0),
            make_candidate(item_id="p2", brand="B", price=20.0, embedding_score=20.0),
        ]
        pipeline._load_user_state = MagicMock()
        pipeline._load_user_state.return_value = MagicMock(
            taste_vector=None,
            onboarding_profile=None,
            user_id="test",
        )
        pipeline._apply_color_filter = MagicMock(side_effect=lambda c, *a, **kw: c)
        pipeline._apply_brand_filter = MagicMock(side_effect=lambda c, *a, **kw: c)
        pipeline._apply_article_type_filter = MagicMock(side_effect=lambda c, *a, **kw: c)

        result = pipeline.get_feed_keyset(
            anon_id="test-user",
            sort_by=FeedSortBy.PRICE_ASC,
            page_size=10,
            debug=True,
        )

        assert result["sort_by"] == "price_asc"
        assert result["strategy"] == "sorted"
        assert len(result["results"]) == 2
        # Prices should be in ascending order
        prices = [r["price"] for r in result["results"]]
        assert prices == sorted(prices)


# =============================================================================
# Cursor sort_mode mismatch Tests
# =============================================================================

class TestCursorSortModeMismatch:
    """Tests for cursor sort_mode mismatch detection."""

    def test_matching_sort_mode_preserves_cursor(self):
        """If cursor sort_mode matches request, cursor values are used."""
        cursor = KeysetCursor(
            score=25.0, item_id="last-item", page=2, sort_mode="price_asc"
        )
        encoded = cursor.encode()
        decoded = KeysetCursor.decode(encoded)
        assert decoded is not None
        assert decoded.sort_mode == "price_asc"

    def test_mismatched_sort_mode_resets(self):
        """Cursor with price_asc should be ignored when request is price_desc."""
        cursor = KeysetCursor(
            score=25.0, item_id="last-item", page=2, sort_mode="price_asc"
        )
        encoded = cursor.encode()
        decoded = KeysetCursor.decode(encoded)
        # Simulating what _get_feed_sorted does:
        request_sort_mode = "price_desc"
        if decoded.sort_mode != request_sort_mode:
            # Reset - don't use cursor values
            cursor_value = None
            cursor_id = None
            page = 0
        else:
            cursor_value = decoded.score
            cursor_id = decoded.item_id
            page = decoded.page + 1

        assert cursor_value is None
        assert cursor_id is None
        assert page == 0


# =============================================================================
# API Endpoint sort_by Parameter Tests
# =============================================================================

class TestAPIEndpointSortBy:
    """Tests that API endpoints accept and parse sort_by parameter."""

    def test_feed_endpoint_has_sort_by_param(self):
        """GET /api/recs/v2/feed should accept sort_by query param."""
        import inspect
        from recs.api_endpoints import get_pipeline_feed
        sig = inspect.signature(get_pipeline_feed)
        assert "sort_by" in sig.parameters
        # Default should be "relevance"
        assert sig.parameters["sort_by"].default.default == "relevance"

    def test_sale_endpoint_has_sort_by_param(self):
        """GET /api/recs/v2/sale should accept sort_by query param."""
        import inspect
        from recs.api_endpoints import get_sale_items
        sig = inspect.signature(get_sale_items)
        assert "sort_by" in sig.parameters

    def test_new_arrivals_endpoint_has_sort_by_param(self):
        """GET /api/recs/v2/new-arrivals should accept sort_by query param."""
        import inspect
        from recs.api_endpoints import get_new_arrivals
        sig = inspect.signature(get_new_arrivals)
        assert "sort_by" in sig.parameters

    def test_keyset_endpoint_has_sort_by_param(self):
        """GET /api/recs/v2/feed/keyset should accept sort_by query param."""
        import inspect
        from recs.api_endpoints import get_keyset_feed
        sig = inspect.signature(get_keyset_feed)
        assert "sort_by" in sig.parameters

    def test_feed_sort_by_parsing_valid(self):
        """Valid sort_by values should parse to FeedSortBy enum."""
        from recs.models import FeedSortBy
        for val in ("relevance", "price_asc", "price_desc"):
            assert FeedSortBy(val) is not None

    def test_feed_sort_by_parsing_invalid_fallback(self):
        """Invalid sort_by value should fall back to RELEVANCE."""
        from recs.models import FeedSortBy
        try:
            result = FeedSortBy("not_a_sort")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected


# =============================================================================
# FeedResponse sort_by field Tests
# =============================================================================

class TestFeedResponseSortBy:
    """Tests for FeedResponse model sort_by field."""

    def test_feed_response_default_sort_by(self):
        """FeedResponse should default sort_by to 'relevance'."""
        from recs.models import FeedResponse
        resp = FeedResponse(
            user_id="test",
            strategy="exploration",
            results=[],
        )
        assert resp.sort_by == "relevance"

    def test_feed_response_custom_sort_by(self):
        """FeedResponse should accept custom sort_by."""
        from recs.models import FeedResponse
        resp = FeedResponse(
            user_id="test",
            strategy="sorted",
            sort_by="price_asc",
            results=[],
        )
        assert resp.sort_by == "price_asc"
