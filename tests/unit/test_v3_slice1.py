"""V3 Slice 1 tests: models, session_store, pool_manager, preference_source."""

import json
import time
from collections import Counter
from unittest.mock import MagicMock, patch

import pytest

from recs.v3.models import (
    CandidatePool,
    CandidateStub,
    FeedRequest,
    PoolDecision,
    ScoringMeta,
    SessionProfile,
    compute_retrieval_signature,
    compute_ranking_signature,
)
from recs.v3.session_store import InMemorySessionStore, get_session_store
from recs.v3.pool_manager import PoolManager
from recs.v3.sources.preference_source import PreferenceSource, _assign_key_family


# =========================================================================
# models.py — FeedRequest
# =========================================================================


def test_feed_request_defaults():
    req = FeedRequest(user_id="u1")
    assert req.user_id == "u1"
    assert len(req.request_id) == 12
    assert req.session_id is None
    assert req.page_size == 24
    assert req.mode == "explore"
    assert req.hard_filters is None
    assert req.soft_preferences is None
    assert req.cursor is None
    assert req.context is None
    assert req.debug is False


def test_feed_request_custom_values():
    req = FeedRequest(
        user_id="u2",
        request_id="custom123",
        session_id="s1",
        page_size=48,
        mode="sale",
        hard_filters={"gender": "women"},
        soft_preferences={"style": "casual"},
        cursor="abc",
        context={"device": "mobile"},
        debug=True,
    )
    assert req.request_id == "custom123"
    assert req.session_id == "s1"
    assert req.page_size == 48
    assert req.mode == "sale"
    assert req.hard_filters == {"gender": "women"}
    assert req.soft_preferences == {"style": "casual"}
    assert req.cursor == "abc"
    assert req.context == {"device": "mobile"}
    assert req.debug is True


def test_feed_request_unique_request_ids():
    r1 = FeedRequest(user_id="u1")
    r2 = FeedRequest(user_id="u1")
    assert r1.request_id != r2.request_id


# =========================================================================
# models.py — CandidateStub
# =========================================================================


def test_candidate_stub_defaults():
    stub = CandidateStub(item_id="item1", source="preference")
    assert stub.item_id == "item1"
    assert stub.source == "preference"
    assert stub.retrieval_score == 0.0
    assert stub.brand is None
    assert stub.broad_category is None
    assert stub.cluster_id is None
    assert stub.article_type is None
    assert stub.price is None
    assert stub.image_dedup_key is None
    assert stub.embedding_score == 0.0
    assert stub.retrieval_key is None


def test_candidate_stub_with_all_fields():
    stub = CandidateStub(
        item_id="item2",
        source="semantic",
        retrieval_score=0.95,
        brand="Boohoo",
        broad_category="dresses",
        cluster_id="cl_42",
        article_type="midi_dress",
        price=59.99,
        image_dedup_key="dk123",
        embedding_score=0.88,
        retrieval_key="a",
    )
    assert stub.brand == "Boohoo"
    assert stub.price == 59.99
    assert stub.embedding_score == 0.88


# =========================================================================
# models.py — ScoringMeta
# =========================================================================


def _make_scoring_meta(**overrides):
    defaults = dict(
        source="preference",
        retrieval_score=0.9,
        brand="Zara",
        cluster_id="cl1",
        broad_category="tops",
        article_type="blouse",
        price=45.0,
        image_dedup_key="dk1",
    )
    defaults.update(overrides)
    return ScoringMeta(**defaults)


def test_scoring_meta_creation():
    sm = _make_scoring_meta()
    assert sm.source == "preference"
    assert sm.price == 45.0
    assert sm.image_dedup_key == "dk1"


def test_scoring_meta_to_dict():
    sm = _make_scoring_meta()
    d = sm.to_dict()
    assert d["s"] == "preference"
    assert d["rs"] == 0.9
    assert d["b"] == "Zara"
    assert d["c"] == "cl1"
    assert d["bc"] == "tops"
    assert d["at"] == "blouse"
    assert d["p"] == 45.0
    assert d["dk"] == "dk1"


def test_scoring_meta_from_dict():
    sm = _make_scoring_meta()
    d = sm.to_dict()
    rebuilt = ScoringMeta.from_dict(d)
    assert rebuilt.source == sm.source
    assert rebuilt.retrieval_score == sm.retrieval_score
    assert rebuilt.brand == sm.brand
    assert rebuilt.image_dedup_key == sm.image_dedup_key


def test_scoring_meta_roundtrip_none_dedup_key():
    sm = _make_scoring_meta(image_dedup_key=None)
    d = sm.to_dict()
    rebuilt = ScoringMeta.from_dict(d)
    assert rebuilt.image_dedup_key is None


# =========================================================================
# models.py — CandidatePool
# =========================================================================


def _make_pool(n_items=10, served=0, **overrides):
    ids = [f"item_{i}" for i in range(n_items)]
    scores = {id_: 1.0 - i * 0.01 for i, id_ in enumerate(ids)}
    meta = {ids[0]: _make_scoring_meta()}
    defaults = dict(
        session_id="sess1",
        mode="explore",
        ordered_ids=ids,
        scores=scores,
        meta=meta,
        served_count=served,
        retrieval_signature="sig123",
        catalog_version="v1",
    )
    defaults.update(overrides)
    return CandidatePool(**defaults)


def test_candidate_pool_creation():
    pool = _make_pool()
    assert pool.session_id == "sess1"
    assert pool.mode == "explore"
    assert len(pool.ordered_ids) == 10


def test_candidate_pool_remaining():
    pool = _make_pool(n_items=10, served=3)
    assert pool.remaining == 7


def test_candidate_pool_remaining_clamps_to_zero():
    pool = _make_pool(n_items=5, served=10)
    assert pool.remaining == 0


def test_candidate_pool_has_more():
    pool = _make_pool(n_items=10, served=5)
    assert pool.has_more is True


def test_candidate_pool_has_more_false():
    pool = _make_pool(n_items=5, served=5)
    assert pool.has_more is False


def test_candidate_pool_next_page_ids():
    pool = _make_pool(n_items=10, served=3)
    page = pool.next_page_ids(4)
    assert page == ["item_3", "item_4", "item_5", "item_6"]


def test_candidate_pool_next_page_ids_truncated():
    pool = _make_pool(n_items=5, served=3)
    page = pool.next_page_ids(10)
    assert page == ["item_3", "item_4"]


def test_candidate_pool_current_page_initial():
    pool = _make_pool(served=0)
    assert pool.current_page() == 1


def test_candidate_pool_current_page_after_serving():
    pool = _make_pool(served=48)
    assert pool.current_page() == 3  # 48 // 24 + 1


def test_candidate_pool_cursor_encode_decode_roundtrip():
    pool = _make_pool(n_items=100, served=24)
    cursor = pool.get_cursor()
    assert cursor is not None
    decoded = CandidatePool.decode_cursor(cursor)
    assert decoded == {"sc": 24, "m": "explore"}


def test_candidate_pool_cursor_none_when_exhausted():
    pool = _make_pool(n_items=5, served=5)
    assert pool.get_cursor() is None


def test_candidate_pool_decode_cursor_invalid():
    assert CandidatePool.decode_cursor("not-valid-base64!!!") is None


def test_candidate_pool_to_json_from_json_roundtrip():
    pool = _make_pool(n_items=5, served=2)
    json_str = pool.to_json()
    rebuilt = CandidatePool.from_json(json_str)
    assert rebuilt.session_id == pool.session_id
    assert rebuilt.mode == pool.mode
    assert rebuilt.ordered_ids == pool.ordered_ids
    assert rebuilt.served_count == pool.served_count
    assert rebuilt.retrieval_signature == pool.retrieval_signature
    assert rebuilt.catalog_version == pool.catalog_version
    # Verify meta survived
    assert "item_0" in rebuilt.meta
    assert rebuilt.meta["item_0"].brand == "Zara"


def test_candidate_pool_empty_pool():
    pool = CandidatePool(session_id="s1", mode="explore")
    assert pool.remaining == 0
    assert pool.has_more is False
    assert pool.next_page_ids(10) == []
    assert pool.get_cursor() is None


# =========================================================================
# models.py — SessionProfile
# =========================================================================


def test_session_profile_defaults():
    sp = SessionProfile(session_id="s1", user_id="u1")
    assert sp.session_id == "s1"
    assert sp.user_id == "u1"
    assert sp.shown_ids == set()
    assert sp.clicked_ids == set()
    assert sp.saved_ids == set()
    assert sp.skipped_ids == set()
    assert sp.hidden_ids == set()
    assert sp.explicit_negative_brands == set()
    assert isinstance(sp.brand_exposure, Counter)
    assert len(sp.brand_exposure) == 0
    assert isinstance(sp.cluster_exposure, Counter)
    assert sp.exploration_budget == 0.15
    assert sp.intent_strength == 0.0
    assert sp.action_seq == 0
    assert sp.recent_actions == []


def test_session_profile_record_click():
    sp = SessionProfile(session_id="s1", user_id="u1")
    sp.record_action("click", "item1", brand="Nike")
    assert "item1" in sp.clicked_ids
    assert "item1" not in sp.saved_ids
    assert sp.action_seq == 1
    assert sp.brand_exposure["Nike"] == 1


def test_session_profile_record_save():
    sp = SessionProfile(session_id="s1", user_id="u1")
    sp.record_action("save", "item2", brand="Zara", cluster_id="cl1")
    assert "item2" in sp.clicked_ids  # save also adds to clicked
    assert "item2" in sp.saved_ids
    assert sp.cluster_exposure["cl1"] == 1


def test_session_profile_record_skip():
    sp = SessionProfile(session_id="s1", user_id="u1")
    sp.record_action("skip", "item3")
    assert "item3" in sp.skipped_ids
    assert "item3" not in sp.clicked_ids
    assert sp.action_seq == 1


def test_session_profile_record_hide():
    sp = SessionProfile(session_id="s1", user_id="u1")
    sp.record_action("hide", "item4", brand="Shein")
    assert "item4" in sp.hidden_ids


def test_session_profile_hide_triggers_negative_brand():
    sp = SessionProfile(session_id="s1", user_id="u1")
    # First two hides build recent_actions, third check triggers negative
    sp.record_action("hide", "item1", brand="BadBrand")
    sp.record_action("hide", "item2", brand="BadBrand")
    sp.record_action("hide", "item3", brand="BadBrand")
    assert "BadBrand" in sp.explicit_negative_brands


def test_session_profile_hide_no_negative_brand_below_threshold():
    sp = SessionProfile(session_id="s1", user_id="u1")
    sp.record_action("hide", "item1", brand="OkBrand")
    sp.record_action("hide", "item2", brand="OkBrand")
    # Only 2 hides — the check on 2nd action sees 1 prior hide which is < 2
    assert "OkBrand" not in sp.explicit_negative_brands


def test_session_profile_record_action_updates_action_seq():
    sp = SessionProfile(session_id="s1", user_id="u1")
    sp.record_action("click", "a")
    sp.record_action("skip", "b")
    sp.record_action("save", "c")
    assert sp.action_seq == 3


def test_session_profile_record_action_updates_brand_exposure():
    sp = SessionProfile(session_id="s1", user_id="u1")
    sp.record_action("click", "a", brand="Nike")
    sp.record_action("click", "b", brand="Nike")
    sp.record_action("click", "c", brand="Adidas")
    assert sp.brand_exposure["Nike"] == 2
    assert sp.brand_exposure["Adidas"] == 1


def test_session_profile_record_action_updates_cluster_exposure():
    sp = SessionProfile(session_id="s1", user_id="u1")
    sp.record_action("click", "a", cluster_id="cl1")
    sp.record_action("click", "b", cluster_id="cl1")
    assert sp.cluster_exposure["cl1"] == 2


def test_session_profile_record_action_updates_category_exposure():
    sp = SessionProfile(session_id="s1", user_id="u1")
    sp.record_action("click", "a", article_type="dress")
    sp.record_action("click", "b", article_type="dress")
    sp.record_action("click", "c", article_type="top")
    assert sp.category_exposure["dress"] == 2
    assert sp.category_exposure["top"] == 1


def test_session_profile_recent_actions_capped():
    sp = SessionProfile(session_id="s1", user_id="u1")
    for i in range(60):
        sp.record_action("click", f"item_{i}")
    assert len(sp.recent_actions) == sp.MAX_RECENT_ACTIONS  # 50


def test_session_profile_purchase_adds_to_saved():
    sp = SessionProfile(session_id="s1", user_id="u1")
    sp.record_action("purchase", "item_p")
    assert "item_p" in sp.clicked_ids
    assert "item_p" in sp.saved_ids


def test_session_profile_to_dict_from_dict_roundtrip():
    sp = SessionProfile(session_id="s1", user_id="u1")
    sp.record_action("click", "a", brand="Nike", cluster_id="cl1")
    sp.record_action("save", "b", brand="Zara")
    sp.record_action("skip", "c")
    sp.record_action("hide", "d", brand="Bad")

    d = sp.to_dict()
    rebuilt = SessionProfile.from_dict(d)

    assert rebuilt.session_id == "s1"
    assert rebuilt.user_id == "u1"
    assert rebuilt.clicked_ids == sp.clicked_ids
    assert rebuilt.saved_ids == sp.saved_ids
    assert rebuilt.skipped_ids == sp.skipped_ids
    assert rebuilt.hidden_ids == sp.hidden_ids
    assert rebuilt.action_seq == sp.action_seq
    assert dict(rebuilt.brand_exposure) == dict(sp.brand_exposure)
    assert dict(rebuilt.cluster_exposure) == dict(sp.cluster_exposure)
    assert len(rebuilt.recent_actions) == len(sp.recent_actions)


def test_session_profile_to_json_from_json_roundtrip():
    sp = SessionProfile(session_id="s2", user_id="u2")
    sp.record_action("click", "x", brand="B1")
    json_str = sp.to_json()
    rebuilt = SessionProfile.from_json(json_str)
    assert rebuilt.session_id == "s2"
    assert rebuilt.clicked_ids == {"x"}


# =========================================================================
# models.py — PoolDecision
# =========================================================================


def test_pool_decision_creation():
    pd = PoolDecision(action="rebuild", reason="no pool exists")
    assert pd.action == "rebuild"
    assert pd.reason == "no pool exists"
    assert pd.remaining == 0


def test_pool_decision_with_remaining():
    pd = PoolDecision(action="reuse", reason="healthy", remaining=200)
    assert pd.remaining == 200


# =========================================================================
# models.py — compute_retrieval_signature
# =========================================================================


def test_compute_retrieval_signature_deterministic():
    sig1 = compute_retrieval_signature("explore", {"gender": "women"}, "a")
    sig2 = compute_retrieval_signature("explore", {"gender": "women"}, "a")
    assert sig1 == sig2
    assert len(sig1) == 16


def test_compute_retrieval_signature_changes_on_mode():
    sig1 = compute_retrieval_signature("explore", {}, "a")
    sig2 = compute_retrieval_signature("sale", {}, "a")
    assert sig1 != sig2


def test_compute_retrieval_signature_changes_on_filters():
    sig1 = compute_retrieval_signature("explore", {"gender": "women"}, "a")
    sig2 = compute_retrieval_signature("explore", {"gender": "men"}, "a")
    assert sig1 != sig2


def test_compute_retrieval_signature_changes_on_key_family():
    sig1 = compute_retrieval_signature("explore", {}, "a")
    sig2 = compute_retrieval_signature("explore", {}, "b")
    assert sig1 != sig2


def test_compute_retrieval_signature_changes_on_source_config():
    sig1 = compute_retrieval_signature("explore", {}, "a", "v1")
    sig2 = compute_retrieval_signature("explore", {}, "a", "v2")
    assert sig1 != sig2


def test_compute_ranking_signature_deterministic():
    sig1 = compute_ranking_signature({"w1": 0.5}, 6)
    sig2 = compute_ranking_signature({"w1": 0.5}, 6)
    assert sig1 == sig2


def test_compute_ranking_signature_groups_by_3():
    """action_seq 0,1,2 map to the same bucket (0); 3,4,5 map to 1."""
    sig_a = compute_ranking_signature({"w": 1}, 0)
    sig_b = compute_ranking_signature({"w": 1}, 2)
    sig_c = compute_ranking_signature({"w": 1}, 3)
    assert sig_a == sig_b  # same bucket
    assert sig_a != sig_c  # different bucket


# =========================================================================
# session_store.py — InMemorySessionStore
# =========================================================================


def test_inmemory_get_or_create_creates_new():
    store = InMemorySessionStore()
    session = store.get_or_create_session("s1", "u1")
    assert session.session_id == "s1"
    assert session.user_id == "u1"
    assert session.action_seq == 0


def test_inmemory_get_or_create_returns_existing():
    store = InMemorySessionStore()
    s1 = store.get_or_create_session("s1", "u1")
    s1.record_action("click", "item1")
    store.save_session("s1", s1)
    s2 = store.get_or_create_session("s1", "u1")
    assert s2.action_seq == 1
    assert "item1" in s2.clicked_ids


def test_inmemory_save_session_roundtrip():
    store = InMemorySessionStore()
    session = SessionProfile(session_id="s1", user_id="u1")
    session.record_action("save", "it1", brand="B")
    store.save_session("s1", session)
    loaded = store.get_or_create_session("s1", "u1")
    assert "it1" in loaded.saved_ids


def test_inmemory_load_shown_set_empty():
    store = InMemorySessionStore()
    shown = store.load_shown_set("s1")
    assert shown == set()


def test_inmemory_add_shown_and_load():
    store = InMemorySessionStore()
    store.add_shown("s1", {"a", "b", "c"})
    shown = store.load_shown_set("s1")
    assert shown == {"a", "b", "c"}


def test_inmemory_add_shown_accumulates():
    store = InMemorySessionStore()
    store.add_shown("s1", {"a", "b"})
    store.add_shown("s1", {"c", "d"})
    assert store.load_shown_set("s1") == {"a", "b", "c", "d"}


def test_inmemory_add_shown_empty_is_noop():
    store = InMemorySessionStore()
    store.add_shown("s1", set())
    assert store.load_shown_set("s1") == set()


def test_inmemory_get_shown_count():
    store = InMemorySessionStore()
    store.add_shown("s1", {"x", "y", "z"})
    assert store.get_shown_count("s1") == 3


def test_inmemory_get_shown_count_zero():
    store = InMemorySessionStore()
    assert store.get_shown_count("nonexistent") == 0


def test_inmemory_save_pool_get_pool_roundtrip():
    store = InMemorySessionStore()
    pool = _make_pool(n_items=5)
    store.save_pool("sess1", pool)
    loaded = store.get_pool("sess1", "explore")
    assert loaded is not None
    assert loaded.session_id == "sess1"
    assert loaded.ordered_ids == pool.ordered_ids


def test_inmemory_get_pool_returns_none_for_unknown_mode():
    store = InMemorySessionStore()
    pool = _make_pool(n_items=5, mode="explore")
    store.save_pool("sess1", pool)
    assert store.get_pool("sess1", "sale") is None


def test_inmemory_get_pool_returns_none_for_unknown_session():
    store = InMemorySessionStore()
    assert store.get_pool("nonexistent", "explore") is None


def test_inmemory_delete_session_clears_everything():
    store = InMemorySessionStore()
    store.get_or_create_session("s1", "u1")
    store.add_shown("s1", {"a", "b"})
    pool = _make_pool(session_id="s1")
    store.save_pool("s1", pool)

    store.delete_session("s1")

    # Session re-created as fresh
    fresh = store.get_or_create_session("s1", "u1")
    assert fresh.action_seq == 0
    assert store.load_shown_set("s1") == set()
    assert store.get_pool("s1", "explore") is None


def test_inmemory_delete_pool():
    store = InMemorySessionStore()
    pool = _make_pool(session_id="s1")
    store.save_pool("s1", pool)
    assert store.get_pool("s1", "explore") is not None
    store.delete_pool("s1", "explore")
    assert store.get_pool("s1", "explore") is None


def test_inmemory_delete_pool_nonexistent_is_noop():
    store = InMemorySessionStore()
    store.delete_pool("s1", "explore")  # should not raise


def test_inmemory_get_catalog_version_default():
    store = InMemorySessionStore()
    assert store.get_catalog_version() == ""


def test_inmemory_set_and_get_catalog_version():
    store = InMemorySessionStore()
    store.set_catalog_version("2026-03-09")
    assert store.get_catalog_version() == "2026-03-09"


def test_inmemory_get_stats():
    store = InMemorySessionStore()
    store.get_or_create_session("s1", "u1")
    pool = _make_pool(session_id="s1")
    store.save_pool("s1", pool)
    stats = store.get_stats()
    assert stats["sessions"] == 1
    assert stats["pools"] == 1
    assert stats["backend"] == "memory"


def test_get_session_store_memory():
    store = get_session_store("memory")
    assert isinstance(store, InMemorySessionStore)


def test_inmemory_multiple_sessions_independent():
    store = InMemorySessionStore()
    store.get_or_create_session("s1", "u1")
    store.get_or_create_session("s2", "u2")
    store.add_shown("s1", {"a"})
    store.add_shown("s2", {"b"})
    assert store.load_shown_set("s1") == {"a"}
    assert store.load_shown_set("s2") == {"b"}


# =========================================================================
# pool_manager.py — PoolManager
# =========================================================================


def test_pool_manager_defaults():
    pm = PoolManager()
    assert pm.reuse_threshold == 48
    assert pm.top_up_threshold == 120
    assert pm.target_pool_size == 500
    assert pm.rerank_action_delta == 3


def test_pool_manager_decide_rebuild_when_no_pool():
    pm = PoolManager()
    req = FeedRequest(user_id="u1")
    session = SessionProfile(session_id="s1", user_id="u1")
    decision = pm.decide(None, req, session, "v1", "sig1")
    assert decision.action == "rebuild"
    assert "no pool" in decision.reason


def test_pool_manager_decide_rebuild_when_catalog_version_changes():
    pm = PoolManager()
    pool = _make_pool(n_items=200, served=0, catalog_version="v1", retrieval_signature="sig1")
    req = FeedRequest(user_id="u1")
    session = SessionProfile(session_id="s1", user_id="u1")
    decision = pm.decide(pool, req, session, "v2", "sig1")
    assert decision.action == "rebuild"
    assert "catalog" in decision.reason


def test_pool_manager_decide_rebuild_when_retrieval_signature_changes():
    pm = PoolManager()
    pool = _make_pool(n_items=200, served=0, catalog_version="v1", retrieval_signature="old_sig")
    req = FeedRequest(user_id="u1")
    session = SessionProfile(session_id="s1", user_id="u1")
    decision = pm.decide(pool, req, session, "v1", "new_sig")
    assert decision.action == "rebuild"
    assert "signature" in decision.reason


def test_pool_manager_decide_rebuild_when_nearly_exhausted():
    pm = PoolManager()
    # 50 items, 10 remaining (< 48 reuse_threshold)
    pool = _make_pool(n_items=50, served=41, catalog_version="v1", retrieval_signature="sig1")
    req = FeedRequest(user_id="u1")
    session = SessionProfile(session_id="s1", user_id="u1")
    decision = pm.decide(pool, req, session, "v1", "sig1")
    assert decision.action == "rebuild"
    assert "exhausted" in decision.reason


def test_pool_manager_decide_reuse_when_pool_healthy():
    pm = PoolManager()
    pool = _make_pool(n_items=300, served=0, catalog_version="v1", retrieval_signature="sig1")
    req = FeedRequest(user_id="u1")
    session = SessionProfile(session_id="s1", user_id="u1")
    decision = pm.decide(pool, req, session, "v1", "sig1")
    assert decision.action == "reuse"
    assert decision.remaining == 300


def test_pool_manager_decide_top_up_when_getting_low():
    pm = PoolManager()
    # 100 items remaining — below top_up_threshold (120) but above reuse_threshold (48)
    pool = _make_pool(n_items=200, served=100, catalog_version="v1", retrieval_signature="sig1")
    req = FeedRequest(user_id="u1")
    session = SessionProfile(session_id="s1", user_id="u1")
    decision = pm.decide(pool, req, session, "v1", "sig1")
    assert decision.action == "top_up"
    assert decision.remaining == 100


def test_pool_manager_decide_reuse_at_exact_threshold():
    pm = PoolManager()
    # Exactly 120 remaining == top_up_threshold, so NOT < threshold, should reuse
    pool = _make_pool(n_items=200, served=80, catalog_version="v1", retrieval_signature="sig1")
    req = FeedRequest(user_id="u1")
    session = SessionProfile(session_id="s1", user_id="u1")
    decision = pm.decide(pool, req, session, "v1", "sig1")
    assert decision.action == "reuse"


def test_pool_manager_should_rerank_true():
    pm = PoolManager()
    session = SessionProfile(session_id="s1", user_id="u1")
    session.action_seq = 5
    pool = _make_pool()
    pool.last_rerank_action_seq = 2
    assert pm.should_rerank(session, pool) is True  # delta=3 >= 3


def test_pool_manager_should_rerank_false():
    pm = PoolManager()
    session = SessionProfile(session_id="s1", user_id="u1")
    session.action_seq = 4
    pool = _make_pool()
    pool.last_rerank_action_seq = 2
    assert pm.should_rerank(session, pool) is False  # delta=2 < 3


def test_pool_manager_should_rerank_exact_boundary():
    pm = PoolManager()
    session = SessionProfile(session_id="s1", user_id="u1")
    session.action_seq = 6
    pool = _make_pool()
    pool.last_rerank_action_seq = 3
    assert pm.should_rerank(session, pool) is True  # delta=3 >= 3


def test_pool_manager_get_target_pool_size_explore():
    pm = PoolManager()
    assert pm.get_target_pool_size("explore") == 500


def test_pool_manager_get_target_pool_size_sale():
    pm = PoolManager()
    assert pm.get_target_pool_size("sale") == 300


def test_pool_manager_get_target_pool_size_unknown():
    pm = PoolManager()
    assert pm.get_target_pool_size("unknown_mode") == 500  # falls back to default


# =========================================================================
# preference_source.py — _assign_key_family
# =========================================================================


def test_assign_key_family_deterministic():
    f1 = _assign_key_family("user_123")
    f2 = _assign_key_family("user_123")
    assert f1 == f2


def test_assign_key_family_returns_valid_values():
    family = _assign_key_family("test_user")
    assert family in ("a", "b", "c")


def test_assign_key_family_different_users_can_differ():
    # Try many users; with MD5 modulo 3 we should see at least 2 different families
    families = set()
    for i in range(100):
        families.add(_assign_key_family(f"user_{i}"))
    assert len(families) >= 2  # probabilistically should get all 3


def test_assign_key_family_covers_all_families():
    """With 100 users we expect all 3 families represented."""
    families = {_assign_key_family(f"u_{i}") for i in range(100)}
    assert families == {"a", "b", "c"}


# =========================================================================
# preference_source.py — PreferenceSource
# =========================================================================


def test_preference_source_init_stores_client():
    mock_client = MagicMock()
    source = PreferenceSource(mock_client)
    assert source._supabase is mock_client


def test_preference_source_retrieve_zero_limit_returns_empty():
    mock_client = MagicMock()
    source = PreferenceSource(mock_client)
    session = SessionProfile(session_id="s1", user_id="u1")
    request = FeedRequest(user_id="u1")
    result = source.retrieve(None, session, request, set(), limit=0)
    assert result == []
    mock_client.rpc.assert_not_called()


def test_preference_source_retrieve_negative_limit_returns_empty():
    mock_client = MagicMock()
    source = PreferenceSource(mock_client)
    session = SessionProfile(session_id="s1", user_id="u1")
    request = FeedRequest(user_id="u1")
    result = source.retrieve(None, session, request, set(), limit=-5)
    assert result == []


def test_preference_source_retrieve_general_query():
    """Without preferred brands, should make a single general query."""
    mock_execute = MagicMock()
    mock_execute.data = [
        {"id": 1, "brand": "B1", "broad_category": "tops", "article_type": "blouse",
         "price": 30, "explore_score": 0.8, "image_dedup_key": None},
        {"id": 2, "brand": "B2", "broad_category": "dresses", "article_type": "midi",
         "price": 50, "explore_score": 0.7, "image_dedup_key": None},
    ]
    mock_client = MagicMock()
    mock_client.rpc.return_value.execute.return_value = mock_execute

    source = PreferenceSource(mock_client)
    session = SessionProfile(session_id="s1", user_id="u1")
    request = FeedRequest(user_id="u1")
    result = source.retrieve(None, session, request, set(), limit=10)
    assert len(result) == 2
    assert result[0].item_id == "1"
    assert result[0].source == "preference"
    assert result[1].item_id == "2"


def test_preference_source_retrieve_excludes_ids():
    mock_execute = MagicMock()
    mock_execute.data = [
        {"id": 1, "brand": "B1", "broad_category": "tops", "article_type": "blouse",
         "price": 30, "explore_score": 0.8, "image_dedup_key": None},
        {"id": 2, "brand": "B2", "broad_category": "dresses", "article_type": "midi",
         "price": 50, "explore_score": 0.7, "image_dedup_key": None},
    ]
    mock_client = MagicMock()
    mock_client.rpc.return_value.execute.return_value = mock_execute

    source = PreferenceSource(mock_client)
    session = SessionProfile(session_id="s1", user_id="u1")
    request = FeedRequest(user_id="u1")
    result = source.retrieve(None, session, request, {"1"}, limit=10)
    assert len(result) == 1
    assert result[0].item_id == "2"


def test_preference_source_retrieve_handles_rpc_error():
    mock_client = MagicMock()
    mock_client.rpc.return_value.execute.side_effect = Exception("DB error")

    source = PreferenceSource(mock_client)
    session = SessionProfile(session_id="s1", user_id="u1")
    request = FeedRequest(user_id="u1")
    result = source.retrieve(None, session, request, set(), limit=10)
    assert result == []


def test_preference_source_build_params_sale_mode():
    mock_client = MagicMock()
    source = PreferenceSource(mock_client)
    request = FeedRequest(user_id="u1", mode="sale")
    key_family = _assign_key_family("u1")
    params = source._build_params(request, key_family)
    assert params["p_on_sale_only"] is True
    assert "p_new_arrivals" not in params


def test_preference_source_build_params_new_arrivals_mode():
    mock_client = MagicMock()
    source = PreferenceSource(mock_client)
    request = FeedRequest(user_id="u1", mode="new_arrivals")
    key_family = _assign_key_family("u1")
    params = source._build_params(request, key_family)
    assert params["p_new_arrivals"] is True
    assert "p_on_sale_only" not in params


def test_preference_source_build_params_with_hard_filters():
    mock_client = MagicMock()
    source = PreferenceSource(mock_client)
    hf = MagicMock()
    hf.gender = "women"
    hf.categories = ["dresses", "tops"]
    hf.min_price = 20
    hf.max_price = 100
    hf.exclude_brands = ["BadBrand"]
    hf.include_brands = None
    request = FeedRequest(user_id="u1", hard_filters=hf)
    key_family = "a"
    params = source._build_params(request, key_family)
    assert params["p_gender"] == "women"
    assert params["p_categories"] == ["dresses", "tops"]
    assert params["p_min_price"] == 20
    assert params["p_max_price"] == 100
    assert params["p_exclude_brands"] == ["BadBrand"]
