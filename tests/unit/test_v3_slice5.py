"""V3 Slice 5 tests: API endpoints, route registration, auth."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.auth import SupabaseUser, require_auth
from recs.v3.api import (
    VALID_ACTIONS,
    V3ActionRequest,
    V3ActionResponse,
    _build_hard_filters,
    _parse_csv,
    _get_orchestrator_or_503,
    get_orchestrator,
    router,
)


# =========================================================================
# Fixtures
# =========================================================================

MOCK_USER = SupabaseUser(id="test-user-123", email="test@test.com")


async def _mock_auth():
    return MOCK_USER


def _make_mock_feed_response(**overrides):
    """Build a mock FeedResponse.to_dict() return value."""
    base = {
        "user_id": "test-user-123",
        "session_id": "abc123",
        "mode": "explore",
        "results": [],
        "pagination": {
            "cursor": None,
            "page": 1,
            "page_size": 0,
            "pool_size": 0,
            "has_more": False,
        },
        "metadata": {"source_mix": {}},
    }
    base.update(overrides)
    return base


@pytest.fixture
def mock_orchestrator():
    mock_orch = MagicMock()
    mock_response = MagicMock()
    mock_response.to_dict.return_value = _make_mock_feed_response()
    mock_orch.get_feed.return_value = mock_response
    mock_orch.record_action.return_value = None
    # session_store for health / session endpoints
    mock_orch.session_store = MagicMock()
    mock_orch.session_store.get_stats.return_value = {"active_sessions": 3}
    mock_orch.session_store.delete_session.return_value = None
    # get_or_create_session returns a mock session profile
    mock_session = MagicMock()
    mock_session.session_id = "sess-1"
    mock_session.user_id = "test-user-123"
    mock_session.action_seq = 5
    mock_session.clicked_ids = {"p1", "p2"}
    mock_session.saved_ids = {"p1"}
    mock_session.skipped_ids = set()
    mock_session.hidden_ids = set()
    mock_session.explicit_negative_brands = set()
    mock_session.brand_exposure = {}
    mock_session.category_exposure = {}
    mock_session.cluster_exposure = {}
    mock_session.exploration_budget = 0.15
    mock_session.intent_strength = 0.3
    mock_session.recent_actions = [{"action": "click", "item_id": "p1"}]
    mock_orch.session_store.get_or_create_session.return_value = mock_session
    mock_orch.session_store.get_shown_count.return_value = 48
    mock_orch.session_store.get_pool.return_value = None
    return mock_orch


@pytest.fixture
def client(mock_orchestrator):
    """TestClient with mocked auth and orchestrator."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_auth] = _mock_auth

    with patch("recs.v3.api._get_orchestrator_or_503", return_value=mock_orchestrator):
        with patch("recs.v3.api.get_orchestrator", return_value=mock_orchestrator):
            yield TestClient(app)


@pytest.fixture
def client_with_orch(mock_orchestrator):
    """Return (client, mock_orchestrator) so tests can inspect calls."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_auth] = _mock_auth

    with patch("recs.v3.api._get_orchestrator_or_503", return_value=mock_orchestrator):
        with patch("recs.v3.api.get_orchestrator", return_value=mock_orchestrator):
            yield TestClient(app), mock_orchestrator


# =========================================================================
# API Helpers — _parse_csv
# =========================================================================


def test_parse_csv_none_returns_none():
    assert _parse_csv(None) is None


def test_parse_csv_empty_string_returns_none():
    assert _parse_csv("") is None


def test_parse_csv_basic_list():
    assert _parse_csv("a,b,c") == ["a", "b", "c"]


def test_parse_csv_strips_whitespace():
    assert _parse_csv("a, b , c") == ["a", "b", "c"]


def test_parse_csv_single_value():
    assert _parse_csv("single") == ["single"]


# =========================================================================
# API Helpers — _build_hard_filters
# =========================================================================


def test_build_hard_filters_all_none():
    hf = _build_hard_filters(
        gender=None, categories=None, article_types=None,
        exclude_colors=None, include_colors=None,
        exclude_brands=None, include_brands=None,
        min_price=None, max_price=None,
        exclude_styles=None, include_occasions=None,
        include_patterns=None, exclude_patterns=None,
    )
    assert hf.gender is None
    assert hf.categories is None
    assert hf.min_price is None
    assert hf.max_price is None
    assert hf.include_brands is None
    assert hf.exclude_brands is None


def test_build_hard_filters_some_values():
    hf = _build_hard_filters(
        gender="female", categories="tops,dresses", article_types=None,
        exclude_colors="red,blue", include_colors=None,
        exclude_brands=None, include_brands=None,
        min_price=10.0, max_price=200.0,
        exclude_styles=None, include_occasions=None,
        include_patterns=None, exclude_patterns=None,
    )
    assert hf.gender == "female"
    assert hf.categories == ["tops", "dresses"]
    assert hf.exclude_colors == ["red", "blue"]
    assert hf.min_price == 10.0
    assert hf.max_price == 200.0


def test_build_hard_filters_include_brands_parses_csv():
    hf = _build_hard_filters(
        gender=None, categories=None, article_types=None,
        exclude_colors=None, include_colors=None,
        exclude_brands=None, include_brands="Boohoo, Zara, H&M",
        min_price=None, max_price=None,
        exclude_styles=None, include_occasions=None,
        include_patterns=None, exclude_patterns=None,
    )
    assert hf.include_brands == ["Boohoo", "Zara", "H&M"]


# =========================================================================
# API Helpers — V3ActionRequest / V3ActionResponse / VALID_ACTIONS
# =========================================================================


def test_v3_action_request_required_fields():
    req = V3ActionRequest(session_id="s1", product_id="p1", action="click")
    assert req.session_id == "s1"
    assert req.product_id == "p1"
    assert req.action == "click"
    assert req.source == "feed"
    assert req.position is None
    assert req.brand is None
    assert req.item_type is None
    assert req.cluster_id is None
    assert req.attributes is None


def test_v3_action_response_structure():
    resp = V3ActionResponse(status="success", action_seq=5)
    assert resp.status == "success"
    assert resp.action_seq == 5

    resp_min = V3ActionResponse(status="ok")
    assert resp_min.action_seq is None


def test_valid_actions_contains_expected():
    expected = {"click", "save", "wishlist", "cart", "add_to_cart",
                "purchase", "skip", "hide", "dislike", "hover"}
    assert VALID_ACTIONS == expected


# =========================================================================
# Route Registration
# =========================================================================


def test_router_prefix():
    assert router.prefix == "/api/recs/v3"


def test_router_tags():
    assert "Recommendations V3 (Pool-based)" in router.tags


def test_router_has_five_routes():
    paths = [route.path for route in router.routes]
    prefix = router.prefix
    assert f"{prefix}/feed" in paths
    assert f"{prefix}/feed/action" in paths
    assert f"{prefix}/feed/search-signal" in paths
    assert f"{prefix}/feed/health" in paths
    assert f"{prefix}/feed/session/{{session_id}}" in paths
    # DELETE and GET on same path count as separate routes
    # Check we have at least 6 routes
    assert len(paths) >= 6


def test_router_feed_is_get():
    prefix = router.prefix
    for route in router.routes:
        if route.path == f"{prefix}/feed":
            assert "GET" in route.methods
            break
    else:
        pytest.fail("No /feed route found")


def test_router_action_is_post():
    prefix = router.prefix
    for route in router.routes:
        if route.path == f"{prefix}/feed/action":
            assert "POST" in route.methods
            break
    else:
        pytest.fail("No /feed/action route found")


def test_router_session_has_get_and_delete():
    prefix = router.prefix
    methods_found = set()
    for route in router.routes:
        if route.path == f"{prefix}/feed/session/{{session_id}}":
            methods_found.update(route.methods)
    assert "GET" in methods_found
    assert "DELETE" in methods_found


# =========================================================================
# Singleton / 503 helper
# =========================================================================


def test_get_orchestrator_returns_singleton():
    """get_orchestrator caches globally; calling twice returns same object."""
    mock_orch = MagicMock()
    with patch("recs.v3.api._orchestrator", mock_orch):
        result = get_orchestrator()
        assert result is mock_orch


def test_get_orchestrator_or_503_raises_on_failure():
    with patch("recs.v3.api.get_orchestrator", side_effect=RuntimeError("boom")):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _get_orchestrator_or_503()
        assert exc_info.value.status_code == 503
        assert "unavailable" in exc_info.value.detail.lower()


# =========================================================================
# GET /feed — Basic Requests
# =========================================================================


def test_feed_basic_returns_200(client):
    resp = client.get("/api/recs/v3/feed")
    assert resp.status_code == 200


def test_feed_response_structure(client):
    resp = client.get("/api/recs/v3/feed")
    data = resp.json()
    assert "user_id" in data
    assert "session_id" in data
    assert "mode" in data
    assert "results" in data
    assert "pagination" in data
    assert "metadata" in data


def test_feed_response_pagination_keys(client):
    resp = client.get("/api/recs/v3/feed")
    pag = resp.json()["pagination"]
    assert "cursor" in pag
    assert "page" in pag
    assert "page_size" in pag
    assert "pool_size" in pag
    assert "has_more" in pag


def test_feed_response_metadata_has_source_mix(client):
    resp = client.get("/api/recs/v3/feed")
    meta = resp.json()["metadata"]
    assert "source_mix" in meta


def test_feed_with_session_id(client_with_orch):
    client, orch = client_with_orch
    resp = client.get("/api/recs/v3/feed?session_id=my-session")
    assert resp.status_code == 200
    call_args = orch.get_feed.call_args[0][0]
    assert call_args.session_id == "my-session"


def test_feed_mode_sale(client_with_orch):
    client, orch = client_with_orch
    resp = client.get("/api/recs/v3/feed?mode=sale")
    assert resp.status_code == 200
    call_args = orch.get_feed.call_args[0][0]
    assert call_args.mode == "sale"


def test_feed_mode_new_arrivals(client_with_orch):
    client, orch = client_with_orch
    resp = client.get("/api/recs/v3/feed?mode=new_arrivals")
    assert resp.status_code == 200
    call_args = orch.get_feed.call_args[0][0]
    assert call_args.mode == "new_arrivals"


def test_feed_page_size_param(client_with_orch):
    client, orch = client_with_orch
    resp = client.get("/api/recs/v3/feed?page_size=48")
    assert resp.status_code == 200
    call_args = orch.get_feed.call_args[0][0]
    assert call_args.page_size == 48


def test_feed_cursor_param(client_with_orch):
    client, orch = client_with_orch
    resp = client.get("/api/recs/v3/feed?cursor=eyJzYyI6IDI0fQ==")
    assert resp.status_code == 200
    call_args = orch.get_feed.call_args[0][0]
    assert call_args.cursor == "eyJzYyI6IDI0fQ=="


def test_feed_debug_true(client_with_orch):
    client, orch = client_with_orch
    mock_resp = MagicMock()
    mock_resp.to_dict.return_value = _make_mock_feed_response(debug={"elapsed_ms": 100})
    orch.get_feed.return_value = mock_resp
    resp = client.get("/api/recs/v3/feed?debug=true")
    assert resp.status_code == 200
    call_args = orch.get_feed.call_args[0][0]
    assert call_args.debug is True


def test_feed_debug_false_no_debug_key(client):
    resp = client.get("/api/recs/v3/feed")
    data = resp.json()
    assert "debug" not in data


# =========================================================================
# GET /feed — Filter Parameters
# =========================================================================


def test_feed_gender_filter(client_with_orch):
    client, orch = client_with_orch
    resp = client.get("/api/recs/v3/feed?gender=male")
    assert resp.status_code == 200
    call_args = orch.get_feed.call_args[0][0]
    assert call_args.hard_filters.gender == "male"


def test_feed_categories_csv(client_with_orch):
    client, orch = client_with_orch
    resp = client.get("/api/recs/v3/feed?categories=tops,dresses,outerwear")
    assert resp.status_code == 200
    call_args = orch.get_feed.call_args[0][0]
    assert call_args.hard_filters.categories == ["tops", "dresses", "outerwear"]


def test_feed_include_brands_filter(client_with_orch):
    client, orch = client_with_orch
    resp = client.get("/api/recs/v3/feed?include_brands=Zara,Boohoo")
    assert resp.status_code == 200
    call_args = orch.get_feed.call_args[0][0]
    assert call_args.hard_filters.include_brands == ["Zara", "Boohoo"]


def test_feed_exclude_brands_filter(client_with_orch):
    client, orch = client_with_orch
    resp = client.get("/api/recs/v3/feed?exclude_brands=Shein")
    assert resp.status_code == 200
    call_args = orch.get_feed.call_args[0][0]
    assert call_args.hard_filters.exclude_brands == ["Shein"]


def test_feed_min_max_price(client_with_orch):
    client, orch = client_with_orch
    resp = client.get("/api/recs/v3/feed?min_price=20&max_price=150")
    assert resp.status_code == 200
    call_args = orch.get_feed.call_args[0][0]
    assert call_args.hard_filters.min_price == 20.0
    assert call_args.hard_filters.max_price == 150.0


def test_feed_exclude_styles(client_with_orch):
    client, orch = client_with_orch
    resp = client.get("/api/recs/v3/feed?exclude_styles=sheer,backless")
    assert resp.status_code == 200
    call_args = orch.get_feed.call_args[0][0]
    assert call_args.hard_filters.exclude_styles == ["sheer", "backless"]


def test_feed_include_occasions(client_with_orch):
    client, orch = client_with_orch
    resp = client.get("/api/recs/v3/feed?include_occasions=casual,office")
    assert resp.status_code == 200
    call_args = orch.get_feed.call_args[0][0]
    assert call_args.hard_filters.include_occasions == ["casual", "office"]


def test_feed_include_patterns(client_with_orch):
    client, orch = client_with_orch
    resp = client.get("/api/recs/v3/feed?include_patterns=floral,stripes")
    assert resp.status_code == 200
    call_args = orch.get_feed.call_args[0][0]
    assert call_args.hard_filters.include_patterns == ["floral", "stripes"]


def test_feed_exclude_patterns(client_with_orch):
    client, orch = client_with_orch
    resp = client.get("/api/recs/v3/feed?exclude_patterns=animal")
    assert resp.status_code == 200
    call_args = orch.get_feed.call_args[0][0]
    assert call_args.hard_filters.exclude_patterns == ["animal"]


# =========================================================================
# GET /feed — Extended PA Filters (soft preferences)
# =========================================================================


def test_feed_extended_pa_filters(client_with_orch):
    client, orch = client_with_orch
    resp = client.get(
        "/api/recs/v3/feed"
        "?include_formality=casual,smart+casual"
        "&include_seasons=spring,summer"
        "&include_fit=slim,regular"
    )
    assert resp.status_code == 200
    call_args = orch.get_feed.call_args[0][0]
    sp = call_args.soft_preferences
    assert sp is not None
    assert "include_formality" in sp
    assert "include_seasons" in sp
    assert "include_fit" in sp


def test_feed_on_sale_only_switches_mode(client_with_orch):
    client, orch = client_with_orch
    resp = client.get("/api/recs/v3/feed?on_sale_only=true")
    assert resp.status_code == 200
    call_args = orch.get_feed.call_args[0][0]
    assert call_args.mode == "sale"


def test_feed_on_sale_only_does_not_override_explicit_mode(client_with_orch):
    """on_sale_only only switches from explore to sale, not from new_arrivals."""
    client, orch = client_with_orch
    resp = client.get("/api/recs/v3/feed?mode=new_arrivals&on_sale_only=true")
    assert resp.status_code == 200
    call_args = orch.get_feed.call_args[0][0]
    assert call_args.mode == "new_arrivals"


# =========================================================================
# GET /feed — Error / Auth
# =========================================================================


def test_feed_error_returns_500(mock_orchestrator):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_auth] = _mock_auth

    mock_orchestrator.get_feed.side_effect = RuntimeError("DB connection lost")
    with patch("recs.v3.api._get_orchestrator_or_503", return_value=mock_orchestrator):
        client = TestClient(app)
        resp = client.get("/api/recs/v3/feed")
    assert resp.status_code == 500
    assert "Feed error" in resp.json()["detail"]


def test_feed_without_auth_returns_401():
    """No auth override → require_auth raises 401."""
    app = FastAPI()
    app.include_router(router)
    # No dependency override — real require_auth runs
    client = TestClient(app)
    resp = client.get("/api/recs/v3/feed")
    assert resp.status_code in (401, 403)


def test_feed_user_id_from_auth(client_with_orch):
    """Verify user_id in the FeedRequest comes from authenticated user."""
    client, orch = client_with_orch
    resp = client.get("/api/recs/v3/feed")
    assert resp.status_code == 200
    call_args = orch.get_feed.call_args[0][0]
    assert call_args.user_id == "test-user-123"


# =========================================================================
# POST /feed/action — Happy Paths
# =========================================================================


def test_action_valid_returns_200(client):
    resp = client.post("/api/recs/v3/feed/action", json={
        "session_id": "s1",
        "product_id": "p1",
        "action": "click",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_action_invalid_returns_400(client):
    resp = client.post("/api/recs/v3/feed/action", json={
        "session_id": "s1",
        "product_id": "p1",
        "action": "invalid_action",
    })
    assert resp.status_code == 400
    assert "Invalid action" in resp.json()["detail"]


def test_action_with_all_metadata(client):
    resp = client.post("/api/recs/v3/feed/action", json={
        "session_id": "s1",
        "product_id": "p1",
        "action": "save",
        "source": "search",
        "position": 3,
        "brand": "Zara",
        "item_type": "dress",
        "cluster_id": "cl_1",
        "attributes": {"color_family": "Neutrals", "pattern": "Solid"},
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_action_minimal_fields(client):
    resp = client.post("/api/recs/v3/feed/action", json={
        "session_id": "s1",
        "product_id": "p1",
        "action": "hover",
    })
    assert resp.status_code == 200


def test_action_calls_orchestrator_record_action(client_with_orch):
    client, orch = client_with_orch
    resp = client.post("/api/recs/v3/feed/action", json={
        "session_id": "s1",
        "product_id": "p1",
        "action": "click",
    })
    assert resp.status_code == 200
    orch.record_action.assert_called_once()
    kwargs = orch.record_action.call_args[1]
    assert kwargs["session_id"] == "s1"
    assert kwargs["product_id"] == "p1"
    assert kwargs["action"] == "click"
    assert kwargs["user_id"] == "test-user-123"


def test_action_passes_metadata_to_orchestrator(client_with_orch):
    client, orch = client_with_orch
    client.post("/api/recs/v3/feed/action", json={
        "session_id": "s1",
        "product_id": "p1",
        "action": "cart",
        "brand": "Boohoo",
        "position": 5,
    })
    kwargs = orch.record_action.call_args[1]
    meta = kwargs["metadata"]
    assert meta["brand"] == "Boohoo"
    assert meta["position"] == 5


def test_action_all_valid_actions_accepted(client):
    for action in VALID_ACTIONS:
        resp = client.post("/api/recs/v3/feed/action", json={
            "session_id": "s1",
            "product_id": "p1",
            "action": action,
        })
        assert resp.status_code == 200, f"Action '{action}' returned {resp.status_code}"


def test_action_missing_required_field(client):
    """Missing product_id should return 422 (validation error)."""
    resp = client.post("/api/recs/v3/feed/action", json={
        "session_id": "s1",
        "action": "click",
    })
    assert resp.status_code == 422


def test_action_error_returns_500(mock_orchestrator):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_auth] = _mock_auth

    mock_orchestrator.record_action.side_effect = RuntimeError("DB error")
    with patch("recs.v3.api._get_orchestrator_or_503", return_value=mock_orchestrator):
        client = TestClient(app)
        resp = client.post("/api/recs/v3/feed/action", json={
            "session_id": "s1",
            "product_id": "p1",
            "action": "click",
        })
    assert resp.status_code == 500
    assert "Feed error" in resp.json()["detail"]


# =========================================================================
# GET /feed/health
# =========================================================================


def test_health_returns_healthy(client):
    resp = client.get("/api/recs/v3/feed/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "v3-feed"
    assert "session_store" in data


def test_health_returns_degraded_on_error():
    app = FastAPI()
    app.include_router(router)
    with patch("recs.v3.api.get_orchestrator", side_effect=RuntimeError("init failed")):
        client = TestClient(app)
        resp = client.get("/api/recs/v3/feed/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"
    assert "error" in data


def test_health_no_auth_required():
    """Health endpoint should not require auth."""
    app = FastAPI()
    app.include_router(router)
    # No auth override, no auth header — should still work (no Depends(require_auth))
    mock_orch = MagicMock()
    mock_orch.session_store.get_stats.return_value = {}
    with patch("recs.v3.api.get_orchestrator", return_value=mock_orch):
        client = TestClient(app)
        resp = client.get("/api/recs/v3/feed/health")
    assert resp.status_code == 200


# =========================================================================
# GET /feed/session/{session_id}
# =========================================================================


def test_session_info_returns_dict(client):
    resp = client.get("/api/recs/v3/feed/session/sess-1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess-1"
    assert data["user_id"] == "test-user-123"
    assert "action_seq" in data
    assert "shown_count" in data
    assert "clicked_count" in data
    assert "saved_count" in data


def test_session_info_pool_info(client_with_orch):
    """When a pool exists for a mode, it should be in the pools dict."""
    client, orch = client_with_orch
    mock_pool = MagicMock()
    mock_pool.ordered_ids = ["p1", "p2", "p3"]
    mock_pool.served_count = 1
    mock_pool.remaining = 2
    mock_pool.source_mix = {"preference": 2, "exploration": 1}
    mock_pool.last_rerank_action_seq = 3

    # Return pool for "explore" mode, None for others
    def side_get_pool(sid, mode):
        if mode == "explore":
            return mock_pool
        return None
    orch.session_store.get_pool.side_effect = side_get_pool

    resp = client.get("/api/recs/v3/feed/session/sess-1")
    assert resp.status_code == 200
    data = resp.json()
    assert "pools" in data
    assert "explore" in data["pools"]
    assert data["pools"]["explore"]["pool_size"] == 3
    assert data["pools"]["explore"]["served_count"] == 1


def test_session_info_includes_exposure_data(client):
    resp = client.get("/api/recs/v3/feed/session/sess-1")
    data = resp.json()
    assert "brand_exposure" in data
    assert "category_exposure" in data
    assert "cluster_exposure" in data
    assert "exploration_budget" in data
    assert "intent_strength" in data


def test_session_info_includes_action_counts(client):
    resp = client.get("/api/recs/v3/feed/session/sess-1")
    data = resp.json()
    assert data["clicked_count"] == 2
    assert data["saved_count"] == 1
    assert data["skipped_count"] == 0
    assert data["hidden_count"] == 0


# =========================================================================
# DELETE /feed/session/{session_id}
# =========================================================================


def test_delete_session_returns_success(client):
    resp = client.delete("/api/recs/v3/feed/session/sess-1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["session_id"] == "sess-1"
    assert "cleared" in data["message"].lower() or "fresh" in data["message"].lower()


def test_delete_session_calls_store(client_with_orch):
    client, orch = client_with_orch
    resp = client.delete("/api/recs/v3/feed/session/my-session")
    assert resp.status_code == 200
    orch.session_store.delete_session.assert_called_once_with("my-session")


def test_delete_session_error_returns_500(mock_orchestrator):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_auth] = _mock_auth

    mock_orchestrator.session_store.delete_session.side_effect = RuntimeError("Redis down")
    with patch("recs.v3.api._get_orchestrator_or_503", return_value=mock_orchestrator):
        client = TestClient(app)
        resp = client.delete("/api/recs/v3/feed/session/sess-1")
    assert resp.status_code == 500
    assert "Failed to clear session" in resp.json()["detail"]
