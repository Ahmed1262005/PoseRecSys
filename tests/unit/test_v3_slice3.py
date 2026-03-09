"""V3 Slice 3 tests: session_source, exploration_source, merch_source, mixer."""

from collections import Counter
from unittest.mock import MagicMock, patch

from recs.v3.models import CandidateStub, FeedRequest, SessionProfile
from recs.v3.sources.session_source import SessionSource, _extract_session_signals
from recs.v3.sources.exploration_source import (
    ExplorationSource,
    SAFE_SHARE,
    DISCOVERY_SHARE,
    SERENDIPITY_SHARE,
)
from recs.v3.sources.merch_source import MerchSource
from recs.v3.mixer import CandidateMixer, SOURCE_TARGETS, TARGET_POOL_SIZE


# =============================================================================
# Helpers
# =============================================================================


def _make_session(**overrides) -> SessionProfile:
    defaults = dict(session_id="sess1", user_id="user1")
    defaults.update(overrides)
    return SessionProfile(**defaults)


def _make_request(**overrides) -> FeedRequest:
    defaults = dict(user_id="user1", mode="explore")
    defaults.update(overrides)
    return FeedRequest(**defaults)


def _make_supabase_row(item_id: str, **overrides) -> dict:
    """Simulate a row returned by supabase RPC."""
    row = {
        "id": item_id,
        "brand": "TestBrand",
        "broad_category": "tops",
        "article_type": "t-shirt",
        "price": 29.99,
        "explore_score": 0.8,
        "image_dedup_key": None,
    }
    row.update(overrides)
    return row


def _mock_supabase(rows):
    """Create a MagicMock supabase client returning *rows* from rpc().execute()."""
    mock = MagicMock()
    mock.rpc.return_value.execute.return_value.data = rows
    return mock


def _make_stubs(n, source="preference", score_start=0.9, step=0.01):
    return [
        CandidateStub(
            item_id=f"id_{source}_{i}",
            source=source,
            retrieval_score=round(score_start - i * step, 4),
        )
        for i in range(n)
    ]


# =============================================================================
# SessionSource tests
# =============================================================================


class TestSessionSource:
    """Tests for SessionSource."""

    def test_init_stores_supabase_client(self):
        mock_sb = MagicMock()
        src = SessionSource(mock_sb)
        assert src._supabase is mock_sb

    def test_retrieve_no_session_signals_returns_empty(self):
        """No recent positive actions -> empty list."""
        src = SessionSource(_mock_supabase([]))
        session = _make_session(recent_actions=[])
        result = src.retrieve(None, session, _make_request(), set())
        assert result == []

    def test_retrieve_extracts_signals_from_recent_actions(self):
        """Positive actions are extracted into signals."""
        actions = [
            {"action": "click", "item_id": "p1", "brand": "Zara", "article_type": "dress"},
            {"action": "skip", "item_id": "p2", "brand": "H&M", "article_type": "top"},
            {"action": "save", "item_id": "p3", "brand": "Mango", "article_type": "skirt"},
        ]
        signals = _extract_session_signals(_make_session(recent_actions=actions))
        # skip is not a positive action, so only click + save
        brands = {s["brand"] for s in signals}
        assert "Zara" in brands
        assert "Mango" in brands
        assert "H&M" not in brands

    def test_retrieve_calls_supabase_rpc_with_correct_params(self):
        """RPC should be called with v3_get_candidates_by_explore_key."""
        rows = [_make_supabase_row("prod_1")]
        mock_sb = _mock_supabase(rows)
        src = SessionSource(mock_sb)
        session = _make_session(
            recent_actions=[{"action": "click", "item_id": "p1", "brand": "Boohoo", "article_type": "dress"}]
        )
        src.retrieve(None, session, _make_request(), set())
        mock_sb.rpc.assert_called_with("v3_get_candidates_by_explore_key", mock_sb.rpc.call_args[0][1])
        # Verify the RPC function name in the first positional arg
        assert mock_sb.rpc.call_args_list[0][0][0] == "v3_get_candidates_by_explore_key"

    def test_retrieve_respects_exclude_ids(self):
        """Items in exclude_ids must not appear in output."""
        rows = [_make_supabase_row("prod_1"), _make_supabase_row("prod_2")]
        mock_sb = _mock_supabase(rows)
        src = SessionSource(mock_sb)
        session = _make_session(
            recent_actions=[{"action": "click", "item_id": "x", "brand": "B", "article_type": "t"}]
        )
        result = src.retrieve(None, session, _make_request(), exclude_ids={"prod_1"})
        ids = {s.item_id for s in result}
        assert "prod_1" not in ids
        assert "prod_2" in ids

    def test_retrieve_returns_stubs_with_session_source(self):
        """All returned stubs should have source='session'."""
        rows = [_make_supabase_row("prod_1")]
        src = SessionSource(_mock_supabase(rows))
        session = _make_session(
            recent_actions=[{"action": "save", "item_id": "x", "brand": "B", "article_type": "t"}]
        )
        result = src.retrieve(None, session, _make_request(), set())
        assert all(s.source == "session" for s in result)

    def test_retrieve_with_limit_parameter(self):
        """Output should be capped at the limit."""
        rows = [_make_supabase_row(f"prod_{i}") for i in range(50)]
        src = SessionSource(_mock_supabase(rows))
        session = _make_session(
            recent_actions=[{"action": "click", "item_id": "x", "brand": "B", "article_type": "t"}]
        )
        result = src.retrieve(None, session, _make_request(), set(), limit=5)
        assert len(result) <= 5

    def test_retrieve_handles_supabase_error_gracefully(self):
        """Supabase RPC failure should return empty, not raise."""
        mock_sb = MagicMock()
        mock_sb.rpc.side_effect = Exception("connection timeout")
        src = SessionSource(mock_sb)
        session = _make_session(
            recent_actions=[{"action": "click", "item_id": "x", "brand": "B", "article_type": "t"}]
        )
        result = src.retrieve(None, session, _make_request(), set())
        assert result == []


# =============================================================================
# ExplorationSource tests
# =============================================================================


class TestExplorationSource:
    """Tests for ExplorationSource."""

    def test_init_stores_supabase_client(self):
        mock_sb = MagicMock()
        src = ExplorationSource(mock_sb)
        assert src._supabase is mock_sb

    @patch("recs.v3.sources.exploration_source.get_cluster_adjacent_brands", return_value=[])
    @patch("recs.v3.sources.exploration_source.CLUSTER_TO_BRANDS", {})
    def test_retrieve_returns_stubs_with_exploration_source(self, _mock_adj):
        """All returned stubs should have source='exploration'."""
        rows = [_make_supabase_row(f"prod_{i}") for i in range(5)]
        src = ExplorationSource(_mock_supabase(rows))
        session = _make_session(cluster_exposure=Counter({"A": 3}))
        result = src.retrieve(None, session, _make_request(), set())
        assert len(result) > 0
        assert all(s.source == "exploration" for s in result)

    def test_retrieve_has_three_tiers(self):
        """Tier shares should sum to 1.0 (safe + discovery + serendipity)."""
        assert abs(SAFE_SHARE + DISCOVERY_SHARE + SERENDIPITY_SHARE - 1.0) < 1e-9

    @patch("recs.v3.sources.exploration_source.get_cluster_adjacent_brands", return_value=[])
    @patch("recs.v3.sources.exploration_source.CLUSTER_TO_BRANDS", {"A": ["BrandX", "BrandY"]})
    def test_retrieve_respects_exclude_ids(self, _mock_adj):
        """Items in exclude_ids must not appear in output."""
        rows = [_make_supabase_row("prod_1"), _make_supabase_row("prod_2")]
        src = ExplorationSource(_mock_supabase(rows))
        session = _make_session(cluster_exposure=Counter({"A": 1}))
        result = src.retrieve(None, session, _make_request(), exclude_ids={"prod_1"})
        ids = {s.item_id for s in result}
        assert "prod_1" not in ids

    @patch("recs.v3.sources.exploration_source.get_cluster_adjacent_brands", return_value=set())
    @patch("recs.v3.sources.exploration_source.CLUSTER_TO_BRANDS", {})
    def test_retrieve_calls_supabase_rpc(self, _mock_adj):
        """RPC should be called with v3_get_candidates_by_explore_key."""
        rows = [_make_supabase_row("prod_1")]
        mock_sb = _mock_supabase(rows)
        src = ExplorationSource(mock_sb)
        session = _make_session()
        src.retrieve(None, session, _make_request(), set())
        # At minimum the serendipity tier calls the RPC
        rpc_names = [c[0][0] for c in mock_sb.rpc.call_args_list]
        assert "v3_get_candidates_by_explore_key" in rpc_names

    @patch("recs.v3.sources.exploration_source.get_cluster_adjacent_brands", return_value=set())
    @patch("recs.v3.sources.exploration_source.CLUSTER_TO_BRANDS", {})
    def test_retrieve_handles_empty_results(self, _mock_adj):
        """Empty RPC results should produce empty list."""
        src = ExplorationSource(_mock_supabase([]))
        session = _make_session()
        result = src.retrieve(None, session, _make_request(), set())
        assert result == []

    @patch("recs.v3.sources.exploration_source.get_cluster_adjacent_brands", return_value=set())
    @patch("recs.v3.sources.exploration_source.CLUSTER_TO_BRANDS", {})
    def test_retrieve_with_limit_parameter(self, _mock_adj):
        """Output should be capped at the limit."""
        rows = [_make_supabase_row(f"prod_{i}") for i in range(100)]
        src = ExplorationSource(_mock_supabase(rows))
        session = _make_session()
        result = src.retrieve(None, session, _make_request(), set(), limit=10)
        assert len(result) <= 10

    @patch("recs.v3.sources.exploration_source.get_cluster_adjacent_brands", return_value=[])
    @patch("recs.v3.sources.exploration_source.CLUSTER_TO_BRANDS", {})
    def test_retrieve_handles_supabase_error_gracefully(self, _mock_adj):
        """Supabase RPC failure should return empty, not raise."""
        mock_sb = MagicMock()
        mock_sb.rpc.side_effect = Exception("connection timeout")
        src = ExplorationSource(mock_sb)
        session = _make_session(cluster_exposure=Counter({"A": 3}))
        result = src.retrieve(None, session, _make_request(), set())
        assert result == []


# =============================================================================
# MerchSource tests
# =============================================================================


class TestMerchSource:
    """Tests for MerchSource."""

    def test_init_stores_supabase_client(self):
        mock_sb = MagicMock()
        src = MerchSource(mock_sb)
        assert src._supabase is mock_sb

    def test_retrieve_returns_stubs_with_merch_source(self):
        """All returned stubs should have source='merch'."""
        rows = [_make_supabase_row(f"prod_{i}") for i in range(3)]
        src = MerchSource(_mock_supabase(rows))
        session = _make_session()
        result = src.retrieve(None, session, _make_request(), set())
        assert len(result) == 3
        assert all(s.source == "merch" for s in result)

    def test_retrieve_calls_freshness_rpc(self):
        """RPC should be called with v3_get_candidates_by_freshness."""
        rows = [_make_supabase_row("prod_1")]
        mock_sb = _mock_supabase(rows)
        src = MerchSource(mock_sb)
        session = _make_session()
        src.retrieve(None, session, _make_request(), set())
        mock_sb.rpc.assert_called_once()
        assert mock_sb.rpc.call_args[0][0] == "v3_get_candidates_by_freshness"

    def test_retrieve_respects_exclude_ids(self):
        """Items in exclude_ids must not appear in output."""
        rows = [_make_supabase_row("prod_1"), _make_supabase_row("prod_2")]
        src = MerchSource(_mock_supabase(rows))
        session = _make_session()
        result = src.retrieve(None, session, _make_request(), exclude_ids={"prod_1"})
        ids = {s.item_id for s in result}
        assert "prod_1" not in ids
        assert "prod_2" in ids

    def test_retrieve_handles_include_brands_from_hard_filters(self):
        """include_brands from hard_filters should be forwarded to RPC params."""
        rows = [_make_supabase_row("prod_1")]
        mock_sb = _mock_supabase(rows)
        src = MerchSource(mock_sb)
        hf = MagicMock()
        hf.gender = None
        hf.categories = None
        hf.min_price = None
        hf.max_price = None
        hf.exclude_brands = None
        hf.include_brands = ["Zara", "Mango"]
        request = _make_request(hard_filters=hf)
        src.retrieve(None, _make_session(), request, set())
        call_params = mock_sb.rpc.call_args[0][1]
        assert call_params["p_include_brands"] == ["Zara", "Mango"]

    def test_retrieve_with_limit_parameter(self):
        """Output should be capped at the limit."""
        rows = [_make_supabase_row(f"prod_{i}") for i in range(50)]
        src = MerchSource(_mock_supabase(rows))
        session = _make_session()
        result = src.retrieve(None, session, _make_request(), set(), limit=5)
        assert len(result) <= 5

    def test_retrieve_handles_supabase_error_gracefully(self):
        """Supabase RPC failure should return empty, not raise."""
        mock_sb = MagicMock()
        mock_sb.rpc.side_effect = Exception("connection timeout")
        src = MerchSource(mock_sb)
        session = _make_session()
        result = src.retrieve(None, session, _make_request(), set())
        assert result == []


# =============================================================================
# CandidateMixer tests
# =============================================================================


class TestCandidateMixer:
    """Tests for CandidateMixer."""

    def test_default_quotas(self):
        """Default quotas should be preference=225, session=125, exploration=75, merch=75."""
        mixer = CandidateMixer()
        assert mixer.source_targets["preference"] == 225
        assert mixer.source_targets["session"] == 125
        assert mixer.source_targets["exploration"] == 75
        assert mixer.source_targets["merch"] == 75

    def test_mix_combines_sources_respecting_quotas(self):
        """Each source should contribute proportionally to its quota."""
        mixer = CandidateMixer()
        source_results = {
            "preference": _make_stubs(300, "preference"),
            "session": _make_stubs(200, "session"),
            "exploration": _make_stubs(100, "exploration"),
            "merch": _make_stubs(100, "merch"),
        }
        result = mixer.mix(source_results, target_size=500)
        sources = Counter(s.source for s in result)
        # preference should have the most items
        assert sources["preference"] >= sources["session"]
        assert sources["preference"] >= sources["exploration"]
        # Total should not exceed target
        assert len(result) <= 500

    def test_mix_deduplicates_by_item_id(self):
        """Duplicate item_ids across sources should be merged."""
        mixer = CandidateMixer()
        shared_stub = CandidateStub(item_id="shared_1", source="preference", retrieval_score=0.9)
        dup_stub = CandidateStub(item_id="shared_1", source="session", retrieval_score=0.5)
        source_results = {
            "preference": [shared_stub],
            "session": [dup_stub],
            "exploration": [],
            "merch": [],
        }
        result = mixer.mix(source_results, target_size=10)
        ids = [s.item_id for s in result]
        assert ids.count("shared_1") == 1
        # Should keep the higher-scored one
        matched = [s for s in result if s.item_id == "shared_1"][0]
        assert matched.retrieval_score == 0.9

    def test_mix_fallback_chains_when_source_empty(self):
        """When a source is empty, fallback chain fills its quota."""
        mixer = CandidateMixer()
        # session is empty -> its quota should be backfilled from preference/exploration
        source_results = {
            "preference": _make_stubs(400, "preference"),
            "session": [],
            "exploration": _make_stubs(100, "exploration"),
            "merch": _make_stubs(100, "merch"),
        }
        result = mixer.mix(source_results, target_size=500)
        # Even though session is empty, we should still get a healthy pool
        assert len(result) > 200

    def test_mix_target_size_limits_total_output(self):
        """Total output should not exceed target_size (after dedup)."""
        mixer = CandidateMixer()
        source_results = {
            "preference": _make_stubs(500, "preference"),
            "session": _make_stubs(500, "session"),
            "exploration": _make_stubs(500, "exploration"),
            "merch": _make_stubs(500, "merch"),
        }
        result = mixer.mix(source_results, target_size=100)
        assert len(result) <= 100

    def test_mix_single_source_fills_up_to_target(self):
        """With only one source, it should fill via backfill to approach target."""
        mixer = CandidateMixer()
        source_results = {
            "preference": _make_stubs(600, "preference"),
            "session": [],
            "exploration": [],
            "merch": [],
        }
        result = mixer.mix(source_results, target_size=500)
        # Backfill should draw from preference to fill up
        assert len(result) > 100

    def test_get_source_mix_returns_counts_per_source(self):
        """get_source_mix should return a dict of source -> count."""
        mixer = CandidateMixer()
        source_results = {
            "preference": _make_stubs(10, "preference"),
            "session": _make_stubs(5, "session"),
            "exploration": _make_stubs(3, "exploration"),
            "merch": _make_stubs(2, "merch"),
        }
        counts = mixer.get_source_mix(source_results)
        assert counts == {"preference": 10, "session": 5, "exploration": 3, "merch": 2}

    def test_mix_empty_source_results_returns_empty(self):
        """All-empty sources should produce empty output."""
        mixer = CandidateMixer()
        source_results = {
            "preference": [],
            "session": [],
            "exploration": [],
            "merch": [],
        }
        result = mixer.mix(source_results, target_size=500)
        assert result == []
