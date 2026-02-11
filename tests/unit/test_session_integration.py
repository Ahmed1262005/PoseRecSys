"""
Tests for Phase B: Redis backend for session scores + search reranker session scoring.

Tests cover:
1. InMemoryScoringBackend - save/get/delete/stats/eviction/TTL
2. ScoringRedisBackend - save/get/delete/stats (mocked Redis)
3. get_scoring_backend() factory - auto-select logic
4. Search reranker _apply_session_scoring() - brand/type/attr/intent/skip scoring
5. _get_item_attr_values() helper - attribute extraction
6. HybridSearchService session_scores forwarding
7. Search route _load_session_scores() helper
8. End-to-end: action -> score -> search reranker reflects session learning

Run with: PYTHONPATH=src python -m pytest tests/unit/test_session_integration.py -v
"""

import json
import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Dict, Set, Any


# =============================================================================
# Helpers
# =============================================================================

def _make_session_scores(**overrides):
    """Create a SessionScores instance with sensible defaults."""
    from recs.session_scoring import SessionScores, PreferenceState
    scores = SessionScores()
    # Auto-convert Dict[str, float] to Dict[str, PreferenceState]
    _pref_fields = {"brand_scores", "type_scores", "cluster_scores", "attr_scores", "search_intents"}
    for key, val in overrides.items():
        if key in _pref_fields and isinstance(val, dict):
            converted = {}
            for k, v in val.items():
                if isinstance(v, (int, float)):
                    # Seed both fast and slow with the value for test compatibility
                    converted[k] = PreferenceState(fast=v, slow=v, count=3)
                else:
                    converted[k] = v
            setattr(scores, key, converted)
        else:
            setattr(scores, key, val)
    return scores


def _make_search_result(
    product_id: str = "p-001",
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


# =============================================================================
# 1. InMemoryScoringBackend
# =============================================================================

class TestInMemoryScoringBackend:
    """Tests for InMemoryScoringBackend."""

    @pytest.fixture
    def backend(self):
        from recs.session_state import InMemoryScoringBackend
        return InMemoryScoringBackend(max_entries=10, ttl_seconds=3600)

    def test_save_and_get_roundtrip(self, backend):
        """save_scores() then get_scores() should return the same object."""
        scores = _make_session_scores(
            brand_scores={"zara": 1.5, "hm": -0.3},
            type_scores={"dresses": 0.8},
        )
        backend.save_scores("sess-1", scores)
        loaded = backend.get_scores("sess-1")
        assert loaded is not None
        assert loaded.brand_scores["zara"].fast == 1.5
        assert loaded.brand_scores["hm"].fast == -0.3
        assert loaded.type_scores["dresses"].fast == 0.8

    def test_get_returns_none_for_missing(self, backend):
        """get_scores() should return None for unknown session_id."""
        assert backend.get_scores("nonexistent") is None

    def test_delete_removes_scores(self, backend):
        """delete_scores() should remove the session."""
        scores = _make_session_scores()
        backend.save_scores("sess-del", scores)
        assert backend.get_scores("sess-del") is not None
        backend.delete_scores("sess-del")
        assert backend.get_scores("sess-del") is None

    def test_delete_nonexistent_is_noop(self, backend):
        """delete_scores() on unknown session should not raise."""
        backend.delete_scores("nonexistent")  # Should not raise

    def test_get_stats(self, backend):
        """get_stats() should report active sessions and backend type."""
        backend.save_scores("s1", _make_session_scores())
        backend.save_scores("s2", _make_session_scores())
        stats = backend.get_stats()
        assert stats["active_scoring_sessions"] == 2
        assert stats["backend"] == "memory"

    def test_eviction_when_over_capacity(self, backend):
        """Should evict oldest half when exceeding max_entries."""
        # backend has max_entries=10
        for i in range(12):
            s = _make_session_scores()
            s.last_updated = float(i)  # Older entries have lower timestamps
            backend.save_scores(f"s{i}", s)
        stats = backend.get_stats()
        # After eviction, should have roughly half
        assert stats["active_scoring_sessions"] <= 10

    def test_ttl_expiration(self):
        """Expired entries should return None on get."""
        from recs.session_state import InMemoryScoringBackend
        backend = InMemoryScoringBackend(ttl_seconds=1)
        scores = _make_session_scores()
        backend.save_scores("sess-ttl", scores)
        assert backend.get_scores("sess-ttl") is not None

        # Manually expire by backdating stored_at
        with backend._lock:
            backend._scores["sess-ttl"]["stored_at"] = time.time() - 10
        assert backend.get_scores("sess-ttl") is None

    def test_overwrite_existing(self, backend):
        """save_scores() with same session_id should overwrite."""
        s1 = _make_session_scores(brand_scores={"a": 1.0})
        s2 = _make_session_scores(brand_scores={"b": 2.0})
        backend.save_scores("sess-ow", s1)
        backend.save_scores("sess-ow", s2)
        loaded = backend.get_scores("sess-ow")
        assert loaded.brand_scores["b"].fast == 2.0


# =============================================================================
# 2. ScoringRedisBackend (mocked Redis)
# =============================================================================

class TestScoringRedisBackend:
    """Tests for ScoringRedisBackend with mocked Redis client."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = MagicMock()
        mock.ping.return_value = True
        return mock

    @pytest.fixture
    def backend(self, mock_redis):
        """Create ScoringRedisBackend with mocked Redis."""
        from recs.session_state import ScoringRedisBackend
        with patch("redis.from_url", return_value=mock_redis):
            b = ScoringRedisBackend(redis_url="redis://mock:6379/0")
        return b

    def test_save_scores_calls_setex(self, backend, mock_redis):
        """save_scores() should call redis.setex with JSON blob."""
        scores = _make_session_scores(brand_scores={"zara": 1.0})
        backend.save_scores("sess-r1", scores)
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "scoring:sess-r1"
        assert call_args[0][1] == 86400  # default TTL
        # Third arg should be valid JSON
        json_str = call_args[0][2]
        data = json.loads(json_str)
        assert data["brand_scores"]["zara"]["fast"] == 1.0

    def test_get_scores_returns_deserialized(self, backend, mock_redis):
        """get_scores() should deserialize JSON from Redis."""
        from recs.session_scoring import SessionScores
        scores = _make_session_scores(type_scores={"dresses": 0.5})
        mock_redis.get.return_value = scores.to_json()
        loaded = backend.get_scores("sess-r2")
        assert loaded is not None
        assert loaded.type_scores["dresses"].fast == 0.5
        mock_redis.get.assert_called_with("scoring:sess-r2")

    def test_get_scores_returns_none_for_missing(self, backend, mock_redis):
        """get_scores() should return None when Redis returns None."""
        mock_redis.get.return_value = None
        assert backend.get_scores("nonexistent") is None

    def test_get_scores_refreshes_ttl(self, backend, mock_redis):
        """get_scores() should refresh TTL on access."""
        from recs.session_scoring import SessionScores
        scores = SessionScores()
        mock_redis.get.return_value = scores.to_json()
        backend.get_scores("sess-r3")
        mock_redis.expire.assert_called_once_with("scoring:sess-r3", 86400)

    def test_delete_scores(self, backend, mock_redis):
        """delete_scores() should call redis.delete."""
        backend.delete_scores("sess-r4")
        mock_redis.delete.assert_called_once_with("scoring:sess-r4")

    def test_get_stats_counts_keys(self, backend, mock_redis):
        """get_stats() should count scoring:* keys."""
        mock_redis.scan_iter.return_value = iter(["scoring:s1", "scoring:s2", "scoring:s3"])
        stats = backend.get_stats()
        assert stats["active_scoring_sessions"] == 3
        assert stats["backend"] == "redis"


# =============================================================================
# 3. get_scoring_backend() Factory
# =============================================================================

class TestGetScoringBackend:
    """Tests for the get_scoring_backend factory function."""

    def test_auto_without_redis_url(self):
        """auto backend without REDIS_URL should use InMemory."""
        from recs.session_state import get_scoring_backend, InMemoryScoringBackend
        with patch.dict("os.environ", {}, clear=True):
            # Remove REDIS_URL if it exists
            import os
            os.environ.pop("REDIS_URL", None)
            backend = get_scoring_backend(backend="auto")
            assert isinstance(backend, InMemoryScoringBackend)

    def test_auto_with_redis_url_but_unavailable(self):
        """auto backend with REDIS_URL but Redis down should fall back to InMemory."""
        from recs.session_state import get_scoring_backend, InMemoryScoringBackend
        with patch.dict("os.environ", {"REDIS_URL": "redis://localhost:9999/0"}):
            backend = get_scoring_backend(backend="auto")
            assert isinstance(backend, InMemoryScoringBackend)

    def test_memory_backend_explicit(self):
        """Explicit 'memory' backend should always use InMemory."""
        from recs.session_state import get_scoring_backend, InMemoryScoringBackend
        backend = get_scoring_backend(backend="memory")
        assert isinstance(backend, InMemoryScoringBackend)


# =============================================================================
# 4. Search Reranker _apply_session_scoring()
# =============================================================================

class TestSearchRerankerSessionScoring:
    """Tests for session scoring in the search SessionReranker (Step 3.5)."""

    @pytest.fixture
    def reranker(self):
        from search.reranker import SessionReranker
        return SessionReranker()

    def test_rerank_accepts_session_scores_param(self, reranker):
        """rerank() should accept session_scores parameter."""
        scores = _make_session_scores(brand_scores={"testbrand": 1.0})
        results = [_make_search_result(product_id=f"p{i}", brand=f"Brand{i}") for i in range(3)]
        reranked = reranker.rerank(results, session_scores=scores)
        assert len(reranked) == 3

    def test_brand_affinity_boost(self, reranker):
        """Items with preferred brand should be boosted."""
        scores = _make_session_scores(brand_scores={"zara": 2.0})
        zara = _make_search_result(product_id="z1", brand="Zara", rrf_score=0.05)
        other = _make_search_result(product_id="o1", brand="Other", rrf_score=0.05)
        reranked = reranker.rerank([other, zara], session_scores=scores)
        # Zara should be boosted and rank higher
        assert reranked[0]["product_id"] == "z1"
        assert reranked[0].get("session_adjustment", 0) > 0

    def test_skipped_item_penalized(self, reranker):
        """Skipped items should get penalty."""
        scores = _make_session_scores(skipped_ids={"skip-1"})
        skipped = _make_search_result(product_id="skip-1", rrf_score=0.10)
        normal = _make_search_result(product_id="norm-1", rrf_score=0.10)
        reranked = reranker.rerank([skipped, normal], session_scores=scores)
        # Skipped item should be penalized
        skipped_result = next(r for r in reranked if r["product_id"] == "skip-1")
        assert skipped_result.get("session_adjustment", 0) < 0

    def test_type_affinity_boost(self, reranker):
        """Items matching preferred type should be boosted."""
        scores = _make_session_scores(type_scores={"dresses": 1.5})
        dress = _make_search_result(product_id="d1", article_type="dresses", rrf_score=0.05)
        top = _make_search_result(product_id="t1", article_type="t-shirt", rrf_score=0.05)
        reranked = reranker.rerank([top, dress], session_scores=scores)
        assert reranked[0]["product_id"] == "d1"

    def test_session_scoring_cap_applied(self, reranker):
        """Total session adjustment should be capped at MAX_SESSION_BOOST."""
        from search.reranker import MAX_SESSION_BOOST
        # Extremely high scores to test cap
        scores = _make_session_scores(
            brand_scores={"testbrand": 50.0},
            type_scores={"t-shirt": 50.0},
        )
        result = _make_search_result(product_id="p1", brand="TestBrand", article_type="t-shirt")
        reranked = reranker.rerank([result], session_scores=scores)
        adj = reranked[0].get("session_adjustment", 0)
        assert adj <= MAX_SESSION_BOOST
        assert adj >= -MAX_SESSION_BOOST

    def test_attr_scores_boost(self, reranker):
        """Attribute scores should boost matching items."""
        scores = _make_session_scores(attr_scores={
            "color:black": 1.0,
            "fit:slim": 0.8,
        })
        black_slim = _make_search_result(
            product_id="bs1",
            name="Black Slim Dress",
            brand="BrandA",
            primary_color="Black",
            colors=["black"],
            fit_type="slim",
            rrf_score=0.05,
        )
        white_regular = _make_search_result(
            product_id="wr1",
            name="White Regular Top",
            brand="BrandB",
            primary_color="White",
            colors=["white"],
            fit_type="Regular",
            rrf_score=0.05,
        )
        reranked = reranker.rerank([white_regular, black_slim], session_scores=scores)
        # black_slim matches both color:black and fit:slim -> should be boosted more
        bs_result = next(r for r in reranked if r["product_id"] == "bs1")
        wr_result = next(r for r in reranked if r["product_id"] == "wr1")
        assert bs_result["session_adjustment"] > wr_result["session_adjustment"]
        assert bs_result["rrf_score"] > wr_result["rrf_score"]

    def test_search_intent_boost(self, reranker):
        """Search intent signals should boost matching items."""
        scores = _make_session_scores(search_intents={
            "occasion:date night": 0.8,
            "category:dresses": 0.6,
        })
        date_dress = _make_search_result(
            product_id="dd1",
            occasions=["Date Night"],
            article_type="dresses",
            rrf_score=0.05,
        )
        casual_top = _make_search_result(
            product_id="ct1",
            occasions=["Everyday"],
            article_type="t-shirt",
            rrf_score=0.05,
        )
        reranked = reranker.rerank([casual_top, date_dress], session_scores=scores)
        assert reranked[0]["product_id"] == "dd1"

    def test_no_session_scores_no_change(self, reranker):
        """Without session_scores, results should not have session_adjustment."""
        results = [_make_search_result(product_id=f"p{i}") for i in range(3)]
        reranked = reranker.rerank(results)
        for r in reranked:
            assert "session_adjustment" not in r

    def test_empty_session_scores_no_change(self, reranker):
        """Empty session scores should produce zero adjustments."""
        scores = _make_session_scores()  # All empty
        results = [_make_search_result(product_id="p1", rrf_score=0.10)]
        reranked = reranker.rerank(results, session_scores=scores)
        assert reranked[0].get("session_adjustment", 0) == 0.0

    def test_resilient_to_malformed_scores(self, reranker):
        """Should not crash with a non-SessionScores object."""
        results = [_make_search_result(product_id="p1")]
        reranked = reranker.rerank(results, session_scores="not_a_scores_object")
        assert len(reranked) == 1

    def test_combined_with_profile_scoring(self, reranker):
        """Session scoring should work alongside profile scoring."""
        scores = _make_session_scores(brand_scores={"testbrand": 1.0})
        profile = {
            "soft_prefs": {"preferred_brands": ["TestBrand"]},
            "hard_filters": {},
        }
        results = [_make_search_result(rrf_score=0.10)]
        reranked = reranker.rerank(results, user_profile=profile, session_scores=scores)
        assert "profile_adjustment" in reranked[0]
        assert "session_adjustment" in reranked[0]

    def test_results_resorted_after_session_scoring(self, reranker):
        """Results should be re-sorted by rrf_score after session scoring."""
        scores = _make_session_scores(brand_scores={"boostme": 3.0})
        low_boost = _make_search_result(product_id="boosted", brand="BoostMe", rrf_score=0.01)
        high_base = _make_search_result(product_id="base", brand="Other", rrf_score=0.20)
        reranked = reranker.rerank([high_base, low_boost], session_scores=scores)
        # Verify sorted by rrf_score descending
        for i in range(len(reranked) - 1):
            assert reranked[i]["rrf_score"] >= reranked[i + 1]["rrf_score"]


# =============================================================================
# 5. _get_item_attr_values() Helper
# =============================================================================

class TestGetItemAttrValues:
    """Tests for the _get_item_attr_values() helper function."""

    def test_color_prefix(self):
        from search.reranker import _get_item_attr_values
        item = {"primary_color": "Black", "colors": ["blue", "navy"]}
        vals = _get_item_attr_values(item, "color")
        assert "black" in vals
        assert "blue" in vals
        assert "navy" in vals

    def test_fit_prefix(self):
        from search.reranker import _get_item_attr_values
        item = {"fit_type": "Slim", "fit": "Relaxed"}
        vals = _get_item_attr_values(item, "fit")
        assert "slim" in vals
        assert "relaxed" in vals

    def test_occasion_prefix(self):
        from search.reranker import _get_item_attr_values
        item = {"occasions": ["Date Night", "Evening"]}
        vals = _get_item_attr_values(item, "occasion")
        assert "date night" in vals
        assert "evening" in vals

    def test_style_prefix(self):
        from search.reranker import _get_item_attr_values
        item = {"style_tags": ["streetwear", "casual"]}
        vals = _get_item_attr_values(item, "style")
        assert "streetwear" in vals
        assert "casual" in vals

    def test_material_prefix(self):
        from search.reranker import _get_item_attr_values
        item = {"materials": ["cotton", "polyester"]}
        vals = _get_item_attr_values(item, "material")
        assert "cotton" in vals

    def test_category_prefix(self):
        from search.reranker import _get_item_attr_values
        item = {"article_type": "dress", "broad_category": "dresses"}
        vals = _get_item_attr_values(item, "category")
        assert "dress" in vals
        assert "dresses" in vals

    def test_unknown_prefix_returns_empty(self):
        from search.reranker import _get_item_attr_values
        item = {"brand": "Zara", "price": 50}
        vals = _get_item_attr_values(item, "unknown_prefix")
        assert vals == set()

    def test_none_values_skipped(self):
        from search.reranker import _get_item_attr_values
        item = {"primary_color": None, "colors": None}
        vals = _get_item_attr_values(item, "color")
        assert vals == set()

    def test_empty_strings_skipped(self):
        from search.reranker import _get_item_attr_values
        item = {"primary_color": "", "colors": ["", "blue"]}
        vals = _get_item_attr_values(item, "color")
        assert "" not in vals
        assert "blue" in vals

    def test_brand_prefix(self):
        from search.reranker import _get_item_attr_values
        item = {"brand": "Zara"}
        vals = _get_item_attr_values(item, "brand")
        assert "zara" in vals


# =============================================================================
# 6. HybridSearchService session_scores Forwarding
# =============================================================================

class TestHybridSearchSessionScoresForwarding:
    """Tests for session_scores parameter in HybridSearchService.search()."""

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
                    "category_l1": None, "category_l2": None,
                    "broad_category": "tops", "article_type": "t-shirt",
                    "primary_color": "Black", "color_family": "Dark",
                    "pattern": "Solid", "apparent_fabric": "cotton",
                    "fit_type": "Regular", "formality": "Casual",
                    "silhouette": None, "length": "Regular",
                    "neckline": "Crew", "sleeve_type": "Short", "rise": None,
                    "style_tags": ["casual"], "occasions": ["Everyday"],
                    "seasons": ["Spring"], "colors": ["black"],
                    "materials": ["cotton"],
                }
                for i in range(3)
            ],
            "nbHits": 3,
            "facets": {},
        }
        svc = HybridSearchService(
            algolia_client=mock_algolia,
            analytics=MagicMock(),
        )
        mock_semantic = MagicMock()
        mock_semantic.search_with_filters.return_value = {"results": [], "total": 0}
        svc._semantic_engine = mock_semantic
        return svc

    def test_search_accepts_session_scores(self, service):
        """search() should accept session_scores parameter."""
        from search.models import HybridSearchRequest
        scores = _make_session_scores(brand_scores={"testbrand": 1.0})
        request = HybridSearchRequest(query="casual top", page=1, page_size=10)
        result = service.search(
            request=request,
            user_id="test-user",
            session_scores=scores,
        )
        assert result is not None
        assert len(result.results) > 0

    def test_search_without_session_scores(self, service):
        """search() should work without session_scores (backward compat)."""
        from search.models import HybridSearchRequest
        request = HybridSearchRequest(query="test", page=1, page_size=10)
        result = service.search(request=request, user_id="u1")
        assert result is not None

    def test_search_forwards_session_scores_to_reranker(self, service):
        """search() should pass session_scores to reranker.rerank()."""
        from search.models import HybridSearchRequest
        scores = _make_session_scores(brand_scores={"zara": 1.0})
        request = HybridSearchRequest(query="dress", page=1, page_size=10)
        with patch.object(service._reranker, "rerank", wraps=service._reranker.rerank) as mock_rerank:
            service.search(request=request, user_id="u1", session_scores=scores)
            mock_rerank.assert_called_once()
            call_kwargs = mock_rerank.call_args
            # session_scores should be passed as kwarg
            assert call_kwargs.kwargs.get("session_scores") is scores


# =============================================================================
# 7. Search Route _load_session_scores() Helper
# =============================================================================

class TestLoadSessionScores:
    """Tests for _load_session_scores() in api/routes/search.py."""

    def test_returns_none_without_session_id(self):
        """Should return None when session_id is None."""
        from api.routes.search import _load_session_scores
        assert _load_session_scores(None) is None

    def test_returns_none_when_pipeline_unavailable(self):
        """Should return None when pipeline can't be loaded."""
        from api.routes.search import _load_session_scores
        with patch("recs.api_endpoints.get_pipeline", side_effect=RuntimeError("no pipeline")):
            assert _load_session_scores("sess-123") is None

    def test_returns_scores_when_available(self):
        """Should return scores when pipeline has them."""
        from api.routes.search import _load_session_scores
        mock_scores = _make_session_scores(brand_scores={"zara": 1.0})
        mock_pipeline = MagicMock()
        mock_pipeline.get_session_scores.return_value = mock_scores
        with patch("recs.api_endpoints.get_pipeline", return_value=mock_pipeline):
            result = _load_session_scores("sess-456")
            assert result is mock_scores

    def test_returns_none_when_no_scores_for_session(self):
        """Should return None when pipeline has no scores for this session."""
        from api.routes.search import _load_session_scores
        mock_pipeline = MagicMock()
        mock_pipeline.get_session_scores.return_value = None
        with patch("recs.api_endpoints.get_pipeline", return_value=mock_pipeline):
            assert _load_session_scores("sess-789") is None


# =============================================================================
# 8. End-to-End: Action -> Score -> Search Reranker
# =============================================================================

class TestEndToEndSessionScoring:
    """Tests verifying the full flow from user action to search reranking."""

    def test_action_updates_brand_score_and_reranker_uses_it(self):
        """
        Process a 'click' action for brand Zara, then verify the search
        reranker boosts Zara items using the updated session scores.
        """
        from recs.session_scoring import SessionScores, get_session_scoring_engine
        from search.reranker import SessionReranker

        engine = get_session_scoring_engine()
        scores = SessionScores()

        # Simulate user clicking a Zara dress
        engine.process_action(
            scores,
            action="click",
            product_id="zara-dress-1",
            brand="zara",
            item_type="dresses",
            attributes={"color": "black", "fit": "slim"},
            source="feed",
        )

        assert scores.get_score("brand", "zara") > 0
        assert scores.get_score("type", "dresses") > 0

        # Now use these scores in search reranker
        reranker = SessionReranker()
        zara_result = _make_search_result(
            product_id="z1", brand="Zara", article_type="dresses", rrf_score=0.05,
        )
        other_result = _make_search_result(
            product_id="o1", brand="Other", article_type="tops", rrf_score=0.05,
        )

        reranked = reranker.rerank([other_result, zara_result], session_scores=scores)
        # Zara dress should be boosted above Other tops
        assert reranked[0]["product_id"] == "z1"
        assert reranked[0]["session_adjustment"] > 0

    def test_search_signal_updates_intents_and_reranker_uses_them(self):
        """
        Process a search signal with 'occasions: Date Night' filter,
        then verify reranker boosts Date Night items.
        """
        from recs.session_scoring import SessionScores, get_session_scoring_engine
        from search.reranker import SessionReranker

        engine = get_session_scoring_engine()
        scores = SessionScores()

        # Simulate user searching with "Date Night" occasion filter
        engine.process_search_signal(
            scores,
            query="evening dress",
            filters={
                "occasions": ["Date Night"],
                "categories": ["dresses"],
            },
        )

        assert len(scores.search_intents) > 0

        # Use scores in reranker
        reranker = SessionReranker()
        date_dress = _make_search_result(
            product_id="dd1", occasions=["Date Night"], article_type="dresses", rrf_score=0.05,
        )
        casual_top = _make_search_result(
            product_id="ct1", occasions=["Everyday"], article_type="t-shirt", rrf_score=0.05,
        )

        reranked = reranker.rerank([casual_top, date_dress], session_scores=scores)
        # Date Night dress should rank higher
        assert reranked[0]["product_id"] == "dd1"

    def test_skip_action_penalizes_in_reranker(self):
        """
        Process a 'skip' action, then verify the reranker penalizes
        the skipped product in search results.
        """
        from recs.session_scoring import SessionScores, get_session_scoring_engine
        from search.reranker import SessionReranker

        engine = get_session_scoring_engine()
        scores = SessionScores()

        engine.process_action(
            scores,
            action="skip",
            product_id="skip-me-1",
            brand="somebrand",
            item_type="tops",
            attributes={},
            source="feed",
        )

        assert "skip-me-1" in scores.skipped_ids

        reranker = SessionReranker()
        skipped = _make_search_result(product_id="skip-me-1", rrf_score=0.10)
        normal = _make_search_result(product_id="normal-1", rrf_score=0.10)

        reranked = reranker.rerank([skipped, normal], session_scores=scores)
        skipped_r = next(r for r in reranked if r["product_id"] == "skip-me-1")
        normal_r = next(r for r in reranked if r["product_id"] == "normal-1")
        assert skipped_r["rrf_score"] < normal_r["rrf_score"]
        assert skipped_r["session_adjustment"] < 0

    def test_inmemory_backend_round_trip_with_scoring_engine(self):
        """
        Save scores to InMemoryScoringBackend, load them back,
        and verify they produce the same reranker behavior.
        """
        from recs.session_state import InMemoryScoringBackend
        from recs.session_scoring import SessionScores, get_session_scoring_engine
        from search.reranker import SessionReranker

        backend = InMemoryScoringBackend()
        engine = get_session_scoring_engine()
        scores = SessionScores()

        # Build up some scores
        engine.process_action(
            scores, action="click", product_id="p1",
            brand="zara", item_type="dresses",
            attributes={"color": "red"}, source="feed",
        )
        engine.process_search_signal(
            scores, query="office blazer",
            filters={"occasions": ["Work"]},
        )

        # Save to backend
        backend.save_scores("test-sess", scores)

        # Load back
        loaded = backend.get_scores("test-sess")
        assert loaded is not None
        assert loaded.brand_scores == scores.brand_scores
        assert loaded.type_scores == scores.type_scores
        assert loaded.search_intents == scores.search_intents

        # Both should produce the same reranking
        reranker = SessionReranker()
        result = _make_search_result(product_id="z1", brand="Zara", rrf_score=0.05)

        reranked_original = reranker.rerank([result.copy()], session_scores=scores)
        reranked_loaded = reranker.rerank([result.copy()], session_scores=loaded)

        assert reranked_original[0]["session_adjustment"] == reranked_loaded[0]["session_adjustment"]


# =============================================================================
# 9. Backward Compatibility
# =============================================================================

class TestBackwardCompatPhaseB:
    """Ensure existing interfaces still work with new parameters."""

    def test_reranker_without_session_scores(self):
        """rerank() without session_scores should work as before."""
        from search.reranker import SessionReranker
        reranker = SessionReranker()
        results = [_make_search_result(product_id=f"p{i}") for i in range(3)]
        reranked = reranker.rerank(results)
        assert len(reranked) == 3
        for r in reranked:
            assert "session_adjustment" not in r

    def test_reranker_with_all_params(self):
        """rerank() should accept all parameters together."""
        from search.reranker import SessionReranker
        from scoring.context import AgeGroup, UserContext, WeatherContext, Season

        reranker = SessionReranker()
        scores = _make_session_scores(brand_scores={"testbrand": 0.5})
        profile = {"soft_prefs": {"preferred_brands": ["TestBrand"]}, "hard_filters": {}}
        ctx = UserContext(user_id="u1")
        ctx.age_group = AgeGroup.GEN_Z

        results = [_make_search_result(rrf_score=0.10)]
        reranked = reranker.rerank(
            results,
            user_profile=profile,
            session_scores=scores,
            user_context=ctx,
        )
        assert len(reranked) == 1
        assert "profile_adjustment" in reranked[0]
        assert "session_adjustment" in reranked[0]
        assert "context_adjustment" in reranked[0]
