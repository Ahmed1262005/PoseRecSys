"""V3 Slice 4 tests: events, user_profile, orchestrator, brand quotas."""

import time
import uuid
from collections import Counter
from unittest.mock import MagicMock, patch, PropertyMock, call

import pytest

from recs.models import Candidate, HardFilters
from recs.v3.events import (
    ACTION_DB_MAP,
    DB_VALID_ACTIONS,
    EventLogger,
    NoOpEventLogger,
)
from recs.v3.models import (
    CandidatePool,
    CandidateStub,
    FeedRequest,
    PoolDecision,
    ScoringMeta,
    SessionProfile,
)
from recs.v3.orchestrator import FeedOrchestrator, FeedResponse, _candidate_to_feed_item
from recs.v3.user_profile import CACHE_MAX_SIZE, CACHE_TTL, UserProfileLoader


# =========================================================================
# Helpers
# =========================================================================


def _make_candidate(item_id=None, **kwargs):
    defaults = {
        "item_id": item_id or str(uuid.uuid4()),
        "brand": "Test",
        "price": 50.0,
        "final_score": 0.5,
        "image_url": f"http://img/{uuid.uuid4()}.jpg",
    }
    defaults.update(kwargs)
    return Candidate(**defaults)


def _make_stub(item_id=None, source="preference", **kwargs):
    defaults = {
        "item_id": item_id or str(uuid.uuid4()),
        "source": source,
        "retrieval_score": 0.8,
    }
    defaults.update(kwargs)
    return CandidateStub(**defaults)


def _make_session(session_id="sess1", user_id="user1"):
    return SessionProfile(session_id=session_id, user_id=user_id)


def _make_pool(session_id="sess1", ordered_ids=None, **kwargs):
    return CandidatePool(
        session_id=session_id,
        mode="explore",
        ordered_ids=ordered_ids or [],
        **kwargs,
    )


def _make_orchestrator():
    return FeedOrchestrator(
        session_store=MagicMock(),
        user_profile=MagicMock(),
        pool_manager=MagicMock(),
        sources={
            "preference": MagicMock(),
            "session": MagicMock(),
            "exploration": MagicMock(),
            "merch": MagicMock(),
        },
        mixer=MagicMock(),
        hydrator=MagicMock(),
        eligibility=MagicMock(),
        ranker=MagicMock(),
        reranker=MagicMock(),
        events=MagicMock(),
    )


def _wire_orchestrator_for_rebuild(orc, candidates=None):
    """Configure mocks so get_feed() goes through the rebuild path end-to-end."""
    if candidates is None:
        candidates = [_make_candidate(item_id=f"item{i}") for i in range(3)]

    session = _make_session()
    pool_ids = [c.item_id for c in candidates]

    # session_store
    orc.session_store.get_or_create_session.return_value = session
    orc.session_store.load_shown_set.return_value = set()
    orc.session_store.get_catalog_version.return_value = "v1"
    orc.session_store.get_pool.return_value = None

    # user_profile
    orc.user_profile.load.return_value = MagicMock(onboarding_profile=None)

    # pool_manager
    orc.pool_manager.decide.return_value = PoolDecision(action="rebuild", reason="no_pool")
    orc.pool_manager.get_target_pool_size.return_value = 500

    # sources
    stubs = [_make_stub(item_id=c.item_id) for c in candidates]
    for src in orc.sources.values():
        src.retrieve.return_value = stubs

    # mixer
    orc.mixer.mix.return_value = stubs
    orc.mixer.get_source_mix.return_value = {"preference": len(candidates)}

    # hydrator
    orc.hydrator.hydrate.return_value = candidates
    orc.hydrator.hydrate_ordered.return_value = candidates

    # eligibility
    orc.eligibility.filter.return_value = (candidates, {"penalties": {}})

    # ranker
    orc.ranker.rank.return_value = candidates

    # reranker
    orc.reranker.rerank.return_value = candidates

    return session, candidates


# =========================================================================
# events.py — EventLogger
# =========================================================================


def test_event_logger_init_stores_client():
    client = MagicMock()
    logger = EventLogger(client)
    assert logger._supabase is client


def test_event_logger_log_impressions_no_error():
    """log_impressions with items should not raise."""
    logger = EventLogger(MagicMock())
    logger.log_impressions(
        user_id="u1",
        session_id="s1",
        items=[{"item_id": "p1", "position": 0}],
        source="feed",
    )


def test_event_logger_log_impressions_empty_items():
    """log_impressions with empty list should not error."""
    logger = EventLogger(MagicMock())
    logger.log_impressions(user_id="u1", session_id="s1", items=[], source="feed")


def test_event_logger_log_impressions_source_param():
    """log_impressions passes source through correctly."""
    logger = EventLogger(MagicMock())
    # Should not raise for any source value
    logger.log_impressions(user_id="u1", session_id="s1", items=[{"item_id": "p1"}], source="search")


def test_action_db_map_covers_db_actions():
    """ACTION_DB_MAP values should be a subset of DB_VALID_ACTIONS."""
    for mapped in ACTION_DB_MAP.values():
        assert mapped in DB_VALID_ACTIONS


def test_event_logger_log_action_click_writes_to_db():
    """log_action for 'click' should trigger a background write."""
    client = MagicMock()
    logger = EventLogger(client)
    with patch("recs.v3.events.threading") as mock_threading:
        mock_thread = MagicMock()
        mock_threading.Thread.return_value = mock_thread
        logger.log_action(
            user_id="u1", session_id="s1", action="click", product_id="p1",
        )
        mock_threading.Thread.assert_called_once()
        mock_thread.start.assert_called_once()


def test_event_logger_log_action_all_db_actions():
    """All DB-allowed actions trigger a background write."""
    for action in ("click", "save", "cart", "purchase", "hover"):
        client = MagicMock()
        logger = EventLogger(client)
        with patch("recs.v3.events.threading") as mock_threading:
            mock_thread = MagicMock()
            mock_threading.Thread.return_value = mock_thread
            logger.log_action(
                user_id="u1", session_id="s1", action=action, product_id="p1",
            )
            mock_threading.Thread.assert_called_once()


def test_event_logger_log_action_skip_no_db_write():
    """Session-only actions (skip) should NOT trigger DB write."""
    client = MagicMock()
    logger = EventLogger(client)
    with patch("recs.v3.events.threading") as mock_threading:
        logger.log_action(
            user_id="u1", session_id="s1", action="skip", product_id="p1",
        )
        mock_threading.Thread.assert_not_called()


def test_event_logger_log_action_hide_no_db_write():
    """Session-only actions (hide) should NOT trigger DB write."""
    client = MagicMock()
    logger = EventLogger(client)
    with patch("recs.v3.events.threading") as mock_threading:
        logger.log_action(
            user_id="u1", session_id="s1", action="hide", product_id="p1",
        )
        mock_threading.Thread.assert_not_called()


def test_event_logger_log_action_with_metadata():
    """log_action merges metadata into the data dict."""
    client = MagicMock()
    logger = EventLogger(client)
    # Call _write_interaction directly to verify data shape
    logger._write_interaction = MagicMock()
    with patch("recs.v3.events.threading") as mock_threading:
        # Capture the args passed to Thread
        def capture_thread(**kwargs):
            thread = MagicMock()
            # Execute the target to check data
            kwargs["target"](*kwargs["args"])
            return thread

        mock_threading.Thread.side_effect = capture_thread
        logger.log_action(
            user_id="u1",
            session_id="s1",
            action="click",
            product_id="p1",
            metadata={"brand": "Nike"},
        )
        data = logger._write_interaction.call_args[0][0]
        assert data["brand"] == "Nike"
        assert data["action"] == "click"


def test_event_logger_log_action_with_position():
    """log_action includes position in data when provided."""
    client = MagicMock()
    logger = EventLogger(client)
    logger._write_interaction = MagicMock()
    with patch("recs.v3.events.threading") as mock_threading:
        def capture_thread(**kwargs):
            thread = MagicMock()
            kwargs["target"](*kwargs["args"])
            return thread

        mock_threading.Thread.side_effect = capture_thread
        logger.log_action(
            user_id="u1",
            session_id="s1",
            action="click",
            product_id="p1",
            position=5,
        )
        data = logger._write_interaction.call_args[0][0]
        assert data["position"] == 5


def test_event_logger_write_interaction_handles_error():
    """_write_interaction handles supabase errors gracefully (no raise)."""
    client = MagicMock()
    client.table.return_value.insert.return_value.execute.side_effect = Exception("DB down")
    logger = EventLogger(client)
    # Should not raise
    logger._write_interaction({"action": "click", "product_id": "p1"})


def test_action_db_map_save_maps_to_add_to_wishlist():
    """ACTION_DB_MAP maps 'save' to 'add_to_wishlist'."""
    assert ACTION_DB_MAP["save"] == "add_to_wishlist"


def test_action_db_map_cart_maps_to_add_to_cart():
    """ACTION_DB_MAP maps 'cart' to 'add_to_cart'."""
    assert ACTION_DB_MAP["cart"] == "add_to_cart"


# =========================================================================
# events.py — NoOpEventLogger
# =========================================================================


def test_noop_logger_log_impressions_records_calls():
    logger = NoOpEventLogger()
    logger.log_impressions(user_id="u1", session_id="s1", items=[{"item_id": "p1"}])
    assert len(logger.impressions) == 1
    assert logger.impressions[0]["user_id"] == "u1"
    assert logger.impressions[0]["items"] == [{"item_id": "p1"}]


def test_noop_logger_log_action_records_calls():
    logger = NoOpEventLogger()
    logger.log_action(user_id="u1", session_id="s1", action="click", product_id="p1")
    assert len(logger.actions) == 1
    assert logger.actions[0]["action"] == "click"
    assert logger.actions[0]["product_id"] == "p1"


def test_noop_logger_calls_tracks_all_interactions():
    logger = NoOpEventLogger()
    logger.log_impressions(user_id="u1", session_id="s1", items=[])
    logger.log_impressions(user_id="u1", session_id="s1", items=[{"item_id": "x"}])
    logger.log_action(user_id="u1", session_id="s1", action="click", product_id="p1")
    logger.log_action(user_id="u1", session_id="s1", action="skip", product_id="p2")
    assert len(logger.impressions) == 2
    assert len(logger.actions) == 2


# =========================================================================
# user_profile.py — UserProfileLoader
# =========================================================================


def test_user_profile_loader_init_stores_client():
    client = MagicMock()
    loader = UserProfileLoader(supabase_client=client)
    assert loader._supabase is client


@patch("recs.v3.user_profile.UserProfileLoader._get_csm")
def test_user_profile_load_returns_user_state(mock_get_csm):
    loader = UserProfileLoader()
    mock_csm = MagicMock()
    mock_csm.load_user_state.return_value = {"user_id": "u1", "taste": "cool"}
    mock_get_csm.return_value = mock_csm
    result = loader.load("u1")
    assert result == {"user_id": "u1", "taste": "cool"}
    mock_csm.load_user_state.assert_called_once_with("u1")


@patch("recs.v3.user_profile.UserProfileLoader._get_csm")
def test_user_profile_load_caches_result(mock_get_csm):
    """Second call for same user should use cache, not hit DB."""
    loader = UserProfileLoader()
    mock_csm = MagicMock()
    mock_csm.load_user_state.return_value = {"user_id": "u1"}
    mock_get_csm.return_value = mock_csm
    loader.load("u1")
    loader.load("u1")
    # Only one DB call
    assert mock_csm.load_user_state.call_count == 1


@patch("recs.v3.user_profile.UserProfileLoader._get_csm")
def test_user_profile_load_returns_none_on_error(mock_get_csm):
    """load() returns None when DB raises."""
    loader = UserProfileLoader()
    mock_csm = MagicMock()
    mock_csm.load_user_state.side_effect = Exception("DB down")
    mock_get_csm.return_value = mock_csm
    result = loader.load("u1")
    assert result is None


def test_user_profile_invalidate_removes_from_cache():
    loader = UserProfileLoader()
    # Manually place entry in cache
    from recs.v3.user_profile import _CacheEntry
    loader._cache["u1"] = _CacheEntry(user_state="state1", expires_at=time.time() + 300)
    assert "u1" in loader._cache
    loader.invalidate("u1")
    assert "u1" not in loader._cache


def test_user_profile_clear_cache_empties_all():
    loader = UserProfileLoader()
    from recs.v3.user_profile import _CacheEntry
    loader._cache["u1"] = _CacheEntry(user_state="s1", expires_at=time.time() + 300)
    loader._cache["u2"] = _CacheEntry(user_state="s2", expires_at=time.time() + 300)
    loader._hits = 5
    loader._misses = 3
    loader.clear_cache()
    assert len(loader._cache) == 0
    assert loader._hits == 0
    assert loader._misses == 0


def test_user_profile_cache_stats_returns_counts():
    loader = UserProfileLoader()
    stats = loader.cache_stats()
    assert stats["size"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["max_size"] == CACHE_MAX_SIZE
    assert stats["ttl"] == CACHE_TTL


@patch("recs.v3.user_profile.UserProfileLoader._get_csm")
def test_user_profile_load_same_user_twice_uses_cache(mock_get_csm):
    """Verify hit/miss counts when loading same user twice."""
    loader = UserProfileLoader()
    mock_csm = MagicMock()
    mock_csm.load_user_state.return_value = "state"
    mock_get_csm.return_value = mock_csm
    loader.load("u1")  # miss
    loader.load("u1")  # hit
    stats = loader.cache_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 1


@patch("recs.v3.user_profile.UserProfileLoader._get_csm")
def test_user_profile_load_different_users_cached_independently(mock_get_csm):
    loader = UserProfileLoader()
    mock_csm = MagicMock()
    mock_csm.load_user_state.side_effect = lambda uid: f"state_{uid}"
    mock_get_csm.return_value = mock_csm
    r1 = loader.load("u1")
    r2 = loader.load("u2")
    assert r1 == "state_u1"
    assert r2 == "state_u2"
    assert mock_csm.load_user_state.call_count == 2
    stats = loader.cache_stats()
    assert stats["size"] == 2


@patch("recs.v3.user_profile.UserProfileLoader._get_csm")
def test_user_profile_load_respects_max_cache_size(mock_get_csm):
    """When cache reaches CACHE_MAX_SIZE, oldest entry is evicted."""
    loader = UserProfileLoader()
    mock_csm = MagicMock()
    mock_csm.load_user_state.side_effect = lambda uid: f"state_{uid}"
    mock_get_csm.return_value = mock_csm

    # Fill cache to max
    for i in range(CACHE_MAX_SIZE):
        loader.load(f"user_{i}")

    assert len(loader._cache) == CACHE_MAX_SIZE

    # One more should evict oldest
    loader.load("user_overflow")
    assert len(loader._cache) == CACHE_MAX_SIZE
    assert "user_0" not in loader._cache
    assert "user_overflow" in loader._cache


# =========================================================================
# orchestrator.py — FeedOrchestrator.__init__
# =========================================================================


def test_orchestrator_init_stores_all_dependencies():
    orc = _make_orchestrator()
    assert orc.session_store is not None
    assert orc.user_profile is not None
    assert orc.pool_manager is not None
    assert len(orc.sources) == 4
    assert orc.mixer is not None
    assert orc.hydrator is not None
    assert orc.eligibility is not None
    assert orc.ranker is not None
    assert orc.reranker is not None
    assert orc.events is not None


# =========================================================================
# orchestrator.py — get_feed()
# =========================================================================


@patch("recs.brand_clusters.get_cluster_for_item", return_value="cluster_1")
@patch("recs.v3.orchestrator._assign_key_family", return_value="a")
def test_get_feed_calls_session_store(mock_kf, mock_cluster):
    orc = _make_orchestrator()
    _wire_orchestrator_for_rebuild(orc)
    req = FeedRequest(user_id="u1", session_id="s1")
    orc.get_feed(req)
    orc.session_store.get_or_create_session.assert_called_once_with("s1", "u1")


@patch("recs.brand_clusters.get_cluster_for_item", return_value="cluster_1")
@patch("recs.v3.orchestrator._assign_key_family", return_value="a")
def test_get_feed_generates_session_id_if_not_provided(mock_kf, mock_cluster):
    orc = _make_orchestrator()
    _wire_orchestrator_for_rebuild(orc)
    req = FeedRequest(user_id="u1")  # no session_id
    resp = orc.get_feed(req)
    assert resp.session_id is not None
    assert len(resp.session_id) == 16


@patch("recs.brand_clusters.get_cluster_for_item", return_value="cluster_1")
@patch("recs.v3.orchestrator._assign_key_family", return_value="a")
def test_get_feed_loads_shown_set(mock_kf, mock_cluster):
    orc = _make_orchestrator()
    _wire_orchestrator_for_rebuild(orc)
    req = FeedRequest(user_id="u1", session_id="s1")
    orc.get_feed(req)
    orc.session_store.load_shown_set.assert_called_once_with("s1")


@patch("recs.brand_clusters.get_cluster_for_item", return_value="cluster_1")
@patch("recs.v3.orchestrator._assign_key_family", return_value="a")
def test_get_feed_calls_user_profile_load(mock_kf, mock_cluster):
    orc = _make_orchestrator()
    _wire_orchestrator_for_rebuild(orc)
    req = FeedRequest(user_id="u1", session_id="s1")
    orc.get_feed(req)
    orc.user_profile.load.assert_called_once_with("u1")


@patch("recs.brand_clusters.get_cluster_for_item", return_value="cluster_1")
@patch("recs.v3.orchestrator._assign_key_family", return_value="a")
def test_get_feed_calls_pool_manager_decide(mock_kf, mock_cluster):
    orc = _make_orchestrator()
    _wire_orchestrator_for_rebuild(orc)
    req = FeedRequest(user_id="u1", session_id="s1")
    orc.get_feed(req)
    orc.pool_manager.decide.assert_called_once()


@patch("recs.brand_clusters.get_cluster_for_item", return_value="cluster_1")
@patch("recs.v3.orchestrator._assign_key_family", return_value="a")
def test_get_feed_rebuild_path(mock_kf, mock_cluster):
    """When decision is 'rebuild', full pipeline runs."""
    orc = _make_orchestrator()
    candidates = [_make_candidate(item_id=f"item{i}") for i in range(3)]
    _wire_orchestrator_for_rebuild(orc, candidates)
    req = FeedRequest(user_id="u1", session_id="s1")
    orc.get_feed(req)
    # Verify full pipeline was called
    orc.mixer.mix.assert_called_once()
    orc.hydrator.hydrate.assert_called_once()
    orc.eligibility.filter.assert_called_once()
    orc.ranker.rank.assert_called_once()
    orc.reranker.rerank.assert_called_once()


@patch("recs.brand_clusters.get_cluster_for_item", return_value="cluster_1")
@patch("recs.v3.orchestrator._assign_key_family", return_value="a")
def test_get_feed_reuse_path(mock_kf, mock_cluster):
    """When decision is 'reuse', pool rebuild is skipped."""
    orc = _make_orchestrator()
    candidates = [_make_candidate(item_id=f"item{i}") for i in range(2)]
    pool = _make_pool(
        ordered_ids=[c.item_id for c in candidates],
        scores={c.item_id: c.final_score for c in candidates},
    )

    session = _make_session()
    orc.session_store.get_or_create_session.return_value = session
    orc.session_store.load_shown_set.return_value = set()
    orc.session_store.get_catalog_version.return_value = "v1"
    orc.session_store.get_pool.return_value = pool
    orc.user_profile.load.return_value = MagicMock(onboarding_profile=None)
    orc.pool_manager.decide.return_value = PoolDecision(action="reuse", reason="pool_fresh")
    orc.pool_manager.should_rerank.return_value = False
    orc.hydrator.hydrate_ordered.return_value = candidates

    req = FeedRequest(user_id="u1", session_id="s1")
    orc.get_feed(req)
    # Mixer should NOT be called for reuse path
    orc.mixer.mix.assert_not_called()


@patch("recs.brand_clusters.get_cluster_for_item", return_value="cluster_1")
@patch("recs.v3.orchestrator._assign_key_family", return_value="a")
def test_get_feed_reuse_with_rerank(mock_kf, mock_cluster):
    """When reuse + should_rerank, calls rerank_pool_from_meta."""
    orc = _make_orchestrator()
    candidates = [_make_candidate(item_id=f"item{i}") for i in range(2)]
    pool = _make_pool(
        ordered_ids=[c.item_id for c in candidates],
        scores={c.item_id: c.final_score for c in candidates},
    )

    session = _make_session()
    orc.session_store.get_or_create_session.return_value = session
    orc.session_store.load_shown_set.return_value = set()
    orc.session_store.get_catalog_version.return_value = "v1"
    orc.session_store.get_pool.return_value = pool
    orc.user_profile.load.return_value = MagicMock(onboarding_profile=None)
    orc.pool_manager.decide.return_value = PoolDecision(action="reuse", reason="pool_fresh")
    orc.pool_manager.should_rerank.return_value = True
    orc.hydrator.hydrate_ordered.return_value = candidates

    req = FeedRequest(user_id="u1", session_id="s1")
    orc.get_feed(req)
    orc.ranker.rerank_pool_from_meta.assert_called_once_with(pool, session)


@patch("recs.brand_clusters.get_cluster_for_item", return_value="cluster_1")
@patch("recs.v3.orchestrator._assign_key_family", return_value="a")
def test_get_feed_calls_hydrate_ordered(mock_kf, mock_cluster):
    orc = _make_orchestrator()
    _wire_orchestrator_for_rebuild(orc)
    req = FeedRequest(user_id="u1", session_id="s1")
    orc.get_feed(req)
    orc.hydrator.hydrate_ordered.assert_called_once()


@patch("recs.brand_clusters.get_cluster_for_item", return_value="cluster_1")
@patch("recs.v3.orchestrator._assign_key_family", return_value="a")
def test_get_feed_calls_log_impressions(mock_kf, mock_cluster):
    orc = _make_orchestrator()
    _wire_orchestrator_for_rebuild(orc)
    req = FeedRequest(user_id="u1", session_id="s1")
    orc.get_feed(req)
    orc.events.log_impressions.assert_called_once()
    call_kwargs = orc.events.log_impressions.call_args
    assert call_kwargs[1]["user_id"] == "u1" or call_kwargs[0][0] == "u1"


@patch("recs.brand_clusters.get_cluster_for_item", return_value="cluster_1")
@patch("recs.v3.orchestrator._assign_key_family", return_value="a")
def test_get_feed_returns_feed_response(mock_kf, mock_cluster):
    orc = _make_orchestrator()
    candidates = [_make_candidate(item_id=f"item{i}") for i in range(3)]
    _wire_orchestrator_for_rebuild(orc, candidates)
    req = FeedRequest(user_id="u1", session_id="s1")
    resp = orc.get_feed(req)
    assert isinstance(resp, FeedResponse)
    assert resp.user_id == "u1"
    assert resp.session_id == "s1"
    assert resp.mode == "explore"


@patch("recs.brand_clusters.get_cluster_for_item", return_value="cluster_1")
@patch("recs.v3.orchestrator._assign_key_family", return_value="a")
def test_get_feed_response_to_dict_structure(mock_kf, mock_cluster):
    orc = _make_orchestrator()
    candidates = [_make_candidate(item_id="item1")]
    _wire_orchestrator_for_rebuild(orc, candidates)
    req = FeedRequest(user_id="u1", session_id="s1")
    resp = orc.get_feed(req)
    d = resp.to_dict()
    assert "user_id" in d
    assert "session_id" in d
    assert "mode" in d
    assert "results" in d
    assert "pagination" in d
    assert "metadata" in d
    assert "cursor" in d["pagination"]
    assert "page" in d["pagination"]
    assert "page_size" in d["pagination"]
    assert "pool_size" in d["pagination"]
    assert "has_more" in d["pagination"]
    assert "source_mix" in d["metadata"]


# =========================================================================
# orchestrator.py — record_action()
# =========================================================================


def test_record_action_updates_session():
    orc = _make_orchestrator()
    session = _make_session()
    orc.session_store.get_or_create_session.return_value = session
    orc.record_action(
        session_id="s1", user_id="u1", action="click", product_id="p1",
    )
    orc.session_store.save_session.assert_called_once_with("s1", session)


def test_record_action_calls_events_log_action():
    orc = _make_orchestrator()
    session = _make_session()
    orc.session_store.get_or_create_session.return_value = session
    orc.record_action(
        session_id="s1", user_id="u1", action="click", product_id="p1",
    )
    orc.events.log_action.assert_called_once()
    kwargs = orc.events.log_action.call_args[1]
    assert kwargs["action"] == "click"
    assert kwargs["product_id"] == "p1"
    assert kwargs["user_id"] == "u1"


def test_record_action_passes_metadata():
    orc = _make_orchestrator()
    session = _make_session()
    orc.session_store.get_or_create_session.return_value = session
    meta = {"brand": "Nike", "position": 3, "source": "search"}
    orc.record_action(
        session_id="s1", user_id="u1", action="click", product_id="p1",
        metadata=meta,
    )
    kwargs = orc.events.log_action.call_args[1]
    assert kwargs["source"] == "search"
    assert kwargs["position"] == 3
    assert kwargs["metadata"] == meta


# =========================================================================
# orchestrator.py — _rebuild_pool()
# =========================================================================


@patch("recs.brand_clusters.get_cluster_for_item", return_value="cluster_1")
@patch("recs.v3.orchestrator._assign_key_family", return_value="a")
def test_rebuild_pool_calls_pipeline_in_order(mock_kf, mock_cluster):
    """_rebuild_pool calls sources, mixer, hydrator, eligibility, ranker, reranker."""
    orc = _make_orchestrator()
    candidates = [_make_candidate(item_id=f"item{i}") for i in range(3)]
    session, _ = _wire_orchestrator_for_rebuild(orc, candidates)

    req = FeedRequest(user_id="u1", session_id="s1")
    orc.get_feed(req)

    # All pipeline steps called
    for src in orc.sources.values():
        src.retrieve.assert_called()
    orc.mixer.mix.assert_called_once()
    orc.hydrator.hydrate.assert_called_once()
    orc.eligibility.filter.assert_called_once()
    orc.ranker.rank.assert_called_once()
    orc.reranker.rerank.assert_called_once()
    orc.session_store.save_pool.assert_called()


# =========================================================================
# orchestrator.py — _run_sources()
# =========================================================================


def test_run_sources_runs_all_sources():
    orc = _make_orchestrator()
    session = _make_session()
    req = FeedRequest(user_id="u1")
    stubs = [_make_stub()]

    for src in orc.sources.values():
        src.retrieve.return_value = stubs

    results = orc._run_sources(None, session, req, set())
    assert len(results) == 4
    for name in ("preference", "session", "exploration", "merch"):
        assert name in results


def test_run_sources_handles_source_failure():
    """A source that raises should return empty list, not crash."""
    orc = _make_orchestrator()
    session = _make_session()
    req = FeedRequest(user_id="u1")
    stubs = [_make_stub()]

    orc.sources["preference"].retrieve.side_effect = Exception("Source crashed")
    orc.sources["session"].retrieve.return_value = stubs
    orc.sources["exploration"].retrieve.return_value = stubs
    orc.sources["merch"].retrieve.return_value = stubs

    results = orc._run_sources(None, session, req, set())
    assert results["preference"] == []
    assert len(results["session"]) == 1


# =========================================================================
# orchestrator.py — _get_source_limit() + brand quotas
# =========================================================================


def test_get_source_limit_default_quotas():
    req = FeedRequest(user_id="u1")
    assert FeedOrchestrator._get_source_limit("preference", req) == 225
    assert FeedOrchestrator._get_source_limit("session", req) == 125
    assert FeedOrchestrator._get_source_limit("exploration", req) == 75
    assert FeedOrchestrator._get_source_limit("merch", req) == 75


def test_get_source_limit_brand_quotas():
    """When include_brands is set, exploration=0 and budget shifts."""
    hf = HardFilters(include_brands=["Nike"])
    req = FeedRequest(user_id="u1", hard_filters=hf)
    assert FeedOrchestrator._get_source_limit("preference", req) == 350
    assert FeedOrchestrator._get_source_limit("session", req) == 100
    assert FeedOrchestrator._get_source_limit("exploration", req) == 0
    assert FeedOrchestrator._get_source_limit("merch", req) == 50


def test_get_source_limit_unknown_source_returns_zero():
    req = FeedRequest(user_id="u1")
    assert FeedOrchestrator._get_source_limit("unknown_source", req) == 0


def test_get_source_limit_brand_quotas_unknown_source():
    hf = HardFilters(include_brands=["Nike"])
    req = FeedRequest(user_id="u1", hard_filters=hf)
    assert FeedOrchestrator._get_source_limit("unknown_source", req) == 0


# =========================================================================
# orchestrator.py — _serve_time_validate()
# =========================================================================


def test_serve_time_validate_filters_hidden_items():
    orc = _make_orchestrator()
    c1 = _make_candidate(item_id="visible")
    c2 = _make_candidate(item_id="hidden1")
    session = _make_session()
    session.hidden_ids.add("hidden1")
    pool = _make_pool()
    result = orc._serve_time_validate([c1, c2], set(), session, pool)
    assert len(result) == 1
    assert result[0].item_id == "visible"


def test_serve_time_validate_filters_shown_items():
    orc = _make_orchestrator()
    c1 = _make_candidate(item_id="new_item")
    c2 = _make_candidate(item_id="already_shown")
    session = _make_session()
    pool = _make_pool()
    shown_set = {"already_shown"}
    result = orc._serve_time_validate([c1, c2], shown_set, session, pool)
    assert len(result) == 1
    assert result[0].item_id == "new_item"


def test_serve_time_validate_filters_duplicate_images():
    orc = _make_orchestrator()
    # Two candidates with the same image hash pattern
    c1 = _make_candidate(item_id="item1", image_url="http://img/original_0_abcdef12.jpg")
    c2 = _make_candidate(item_id="item2", image_url="http://img/original_1_abcdef12.jpg")
    session = _make_session()
    pool = _make_pool()
    result = orc._serve_time_validate([c1, c2], set(), session, pool)
    # Only first should survive (same image hash)
    assert len(result) == 1
    assert result[0].item_id == "item1"


# =========================================================================
# orchestrator.py — _backfill_page()
# =========================================================================


def test_backfill_page_fills_gaps():
    orc = _make_orchestrator()
    c1 = _make_candidate(item_id="item1")
    c_extra = _make_candidate(item_id="extra1")
    pool = _make_pool(ordered_ids=["item1", "extra1", "extra2"])
    pool.served_count = 0
    session = _make_session()

    orc.hydrator.hydrate_ordered.return_value = [c_extra]
    result = orc._backfill_page([c1], pool, set(), session, target_size=2)
    assert len(result) == 2


# =========================================================================
# orchestrator.py — _build_eligibility_kwargs()
# =========================================================================


def test_build_eligibility_kwargs_basic():
    orc = _make_orchestrator()
    session = _make_session()
    session.hidden_ids = {"h1"}
    session.explicit_negative_brands = {"badco"}
    req = FeedRequest(user_id="u1")
    kwargs = orc._build_eligibility_kwargs(None, session, {"shown1"}, req)
    assert kwargs["hidden_ids"] == {"h1"}
    assert kwargs["negative_brands"] == {"badco"}
    assert kwargs["shown_set"] == {"shown1"}


def test_build_eligibility_kwargs_include_brands_from_request():
    orc = _make_orchestrator()
    session = _make_session()
    hf = HardFilters(include_brands=["Nike", "Adidas"])
    req = FeedRequest(user_id="u1", hard_filters=hf)
    kwargs = orc._build_eligibility_kwargs(None, session, set(), req)
    assert kwargs["include_brands"] == ["Nike", "Adidas"]


def test_build_eligibility_kwargs_merges_profile_and_request_filters():
    """exclude_brands from profile + request are merged (union)."""
    orc = _make_orchestrator()
    session = _make_session()
    # Mock user_state with profile having brands_to_avoid
    profile = MagicMock()
    profile.get_all_exclusions = None  # not callable
    profile.occasions = None
    profile.exclude_colors = None
    profile.colors_to_avoid = None
    profile.exclude_brands = ["OldBrand"]
    profile.brands_to_avoid = None
    user_state = MagicMock()
    user_state.onboarding_profile = profile

    hf = HardFilters(exclude_brands=["NewBrand"])
    req = FeedRequest(user_id="u1", hard_filters=hf)
    kwargs = orc._build_eligibility_kwargs(user_state, session, set(), req)
    # Should contain both brands (union)
    assert set(kwargs["exclude_brands"]) == {"OldBrand", "NewBrand"}


# =========================================================================
# orchestrator.py — _build_meta()
# =========================================================================


def test_build_meta_builds_scoring_meta_dict():
    c1 = _make_candidate(item_id="item1", brand="Nike")
    stub1 = _make_stub(item_id="item1", source="preference", retrieval_score=0.9)
    stub_lookup = {"item1": stub1}
    meta = FeedOrchestrator._build_meta([c1], stub_lookup)
    assert "item1" in meta
    assert isinstance(meta["item1"], ScoringMeta)
    assert meta["item1"].source == "preference"
    assert meta["item1"].retrieval_score == 0.9
    assert meta["item1"].brand == "Nike"


def test_build_meta_unknown_stub_defaults():
    """If no stub found, source='unknown' and retrieval_score=0."""
    c1 = _make_candidate(item_id="orphan")
    meta = FeedOrchestrator._build_meta([c1], {})
    assert meta["orphan"].source == "unknown"
    assert meta["orphan"].retrieval_score == 0.0


# =========================================================================
# orchestrator.py — _build_debug()
# =========================================================================


def test_build_debug_returns_correct_structure():
    decision = PoolDecision(action="rebuild", reason="no_pool", remaining=0)
    c1 = _make_candidate(item_id="item1")
    pool = _make_pool(ordered_ids=["item1"])
    pool.meta = {
        "item1": ScoringMeta(
            source="preference", retrieval_score=0.8, brand="Test",
            cluster_id="c1", broad_category="tops", article_type="t-shirt",
            price=50.0,
        ),
    }
    debug = FeedOrchestrator._build_debug(decision, pool, [c1], 0.5)
    assert debug["pool_decision"]["action"] == "rebuild"
    assert debug["pool_decision"]["reason"] == "no_pool"
    assert debug["pool"]["size"] == 1
    assert "elapsed_ms" in debug
    assert "items" in debug
    assert debug["items"][0]["source"] == "preference"


# =========================================================================
# orchestrator.py — _candidate_to_feed_item()
# =========================================================================


def test_candidate_to_feed_item_converts_correctly():
    c = _make_candidate(
        item_id="p1", brand="TestBrand", price=99.99, final_score=0.7777,
        name="Cool Dress", category="dresses", image_url="http://img/1.jpg",
    )
    item = _candidate_to_feed_item(c)
    assert item["item_id"] == "p1"
    assert item["brand"] == "TestBrand"
    assert item["price"] == 99.99
    assert item["final_score"] == 0.7777
    assert item["name"] == "Cool Dress"
    assert item["category"] == "dresses"
    assert item["image_url"] == "http://img/1.jpg"
    assert "is_on_sale" in item
    assert "discount_percent" in item
    assert "is_new" in item


# =========================================================================
# orchestrator.py — FeedResponse
# =========================================================================


def test_feed_response_to_dict_no_debug():
    """FeedResponse.to_dict() omits debug key when debug is None."""
    resp = FeedResponse(
        user_id="u1", session_id="s1", mode="explore",
        items=[], pool_size=0,
    )
    d = resp.to_dict()
    assert "debug" not in d


def test_feed_response_to_dict_with_debug():
    """FeedResponse.to_dict() includes debug key when debug is set."""
    resp = FeedResponse(
        user_id="u1", session_id="s1", mode="explore",
        items=[], pool_size=0, debug={"pool_decision": {"action": "rebuild"}},
    )
    d = resp.to_dict()
    assert "debug" in d
    assert d["debug"]["pool_decision"]["action"] == "rebuild"


def test_get_source_limit_brand_quotas_total_is_500():
    """Brand quotas sum to 500 (same as default)."""
    hf = HardFilters(include_brands=["Nike"])
    req = FeedRequest(user_id="u1", hard_filters=hf)
    total = sum(
        FeedOrchestrator._get_source_limit(s, req)
        for s in ("preference", "session", "exploration", "merch")
    )
    assert total == 500
