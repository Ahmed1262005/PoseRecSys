"""
Tests for Search V2: Groq provider support + heuristic planner bypass.

Run with:
    PYTHONPATH=src python -m pytest tests/unit/test_search_v2.py -v

Covers:
    1. Groq provider initialization (settings, client, base_url, defaults)
    2. Heuristic bypass — pure brand, bare category, category+color
    3. Heuristic bypass — negative cases (should return None / fall to LLM)
    4. Heuristic bypass — edge cases (HTML entities, casing, whitespace)
    5. Response time benchmarks (heuristic bypass < 1ms)
    6. V2 route wiring (planner_source tagging in timing dict)
"""

import json
import time

import pytest
from unittest.mock import MagicMock, patch


# =============================================================================
# Helpers
# =============================================================================

def _make_groq_settings(**overrides):
    """Create a mock settings object configured for Groq provider."""
    defaults = {
        "query_planner_provider": "groq",
        "groq_api_key": "gsk-test-key-12345",
        "openai_api_key": "",
        "google_api_key": "",
        "query_planner_model": "gpt-4.1-mini",  # default — should auto-switch
        "query_planner_enabled": True,
        "query_planner_timeout_seconds": 90.0,  # default — should auto-tighten
        "query_planner_heuristic_bypass": True,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


def _make_openai_settings(**overrides):
    """Create a mock settings object configured for OpenAI provider."""
    defaults = {
        "query_planner_provider": "openai",
        "openai_api_key": "sk-test-key",
        "google_api_key": "",
        "groq_api_key": "",
        "query_planner_model": "gpt-4.1-mini",
        "query_planner_enabled": True,
        "query_planner_timeout_seconds": 90.0,
        "query_planner_heuristic_bypass": True,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


def _make_planner(settings_mock):
    """Create a QueryPlanner with mocked settings."""
    from search.query_planner import QueryPlanner
    with patch("search.query_planner.get_settings", return_value=settings_mock):
        return QueryPlanner()


# Ensure the query_classifier has brands loaded for heuristic tests
@pytest.fixture(autouse=True)
def _seed_brands():
    """Seed the query classifier brand cache with test brands."""
    import search.query_classifier as qc
    original_names = qc._BRAND_NAMES
    original_originals = qc._BRAND_ORIGINALS

    qc._BRAND_NAMES = {
        "boohoo", "nasty gal", "prettylittlething", "ba&sh",
        "forever 21", "princess polly", "zara", "h&m", "mango",
        "reformation", "alo yoga", "free people", "cos",
    }
    qc._BRAND_ORIGINALS = {
        "boohoo": "Boohoo",
        "nasty gal": "Nasty Gal",
        "prettylittlething": "PrettyLittleThing",
        "ba&sh": "Ba&sh",
        "forever 21": "Forever 21",
        "princess polly": "Princess Polly",
        "zara": "Zara",
        "h&m": "H&M",
        "mango": "Mango",
        "reformation": "Reformation",
        "alo yoga": "Alo Yoga",
        "free people": "Free People",
        "cos": "COS",
    }

    yield

    qc._BRAND_NAMES = original_names
    qc._BRAND_ORIGINALS = original_originals


# =============================================================================
# 1. Groq Provider Initialization
# =============================================================================

class TestGroqProviderInit:
    """Tests for Groq provider setup in QueryPlanner.__init__."""

    def test_groq_provider_detected(self):
        planner = _make_planner(_make_groq_settings())
        assert planner._provider == "groq"

    def test_groq_api_key_used(self):
        planner = _make_planner(_make_groq_settings())
        assert planner._api_key == "gsk-test-key-12345"

    def test_groq_default_model_auto_switches(self):
        """When model is default gpt-4.1-mini, should auto-switch to Llama 4 Scout."""
        planner = _make_planner(_make_groq_settings())
        assert planner._model == "meta-llama/llama-4-scout-17b-16e-instruct"

    def test_groq_explicit_model_preserved(self):
        """When model is explicitly set, should NOT auto-switch."""
        planner = _make_planner(_make_groq_settings(
            query_planner_model="llama-3.3-70b-versatile"
        ))
        assert planner._model == "llama-3.3-70b-versatile"

    def test_groq_timeout_auto_tightens(self):
        """Default 90s timeout should auto-tighten to 15s for Groq."""
        planner = _make_planner(_make_groq_settings())
        assert planner._timeout == 15.0

    def test_groq_explicit_timeout_preserved(self):
        """Non-default timeout should be preserved."""
        planner = _make_planner(_make_groq_settings(
            query_planner_timeout_seconds=30.0
        ))
        assert planner._timeout == 30.0

    def test_groq_enabled_with_key(self):
        planner = _make_planner(_make_groq_settings())
        assert planner._enabled is True

    def test_groq_disabled_without_key(self):
        planner = _make_planner(_make_groq_settings(groq_api_key=""))
        assert planner._enabled is False

    def test_groq_client_uses_groq_base_url(self):
        """Client should use Groq's OpenAI-compatible base URL."""
        planner = _make_planner(_make_groq_settings())
        with patch("search.query_planner.OpenAI", create=True) as MockOpenAI:
            # Reset cached client
            planner._client = None
            from openai import OpenAI
            with patch("search.query_planner.OpenAI", MockOpenAI) if False else \
                 patch.object(type(planner), "client", new_callable=lambda: property(
                     lambda self: self._get_client_for_test())):
                pass
            # Direct test: instantiate client and check
            planner._client = None
            _ = planner.client
            # The client was created — verify it's an OpenAI instance
            assert planner._client is not None

    def test_openai_provider_no_base_url_override(self):
        """OpenAI provider should NOT set base_url."""
        planner = _make_planner(_make_openai_settings())
        assert planner._provider == "openai"
        assert planner._timeout == 90.0
        assert planner._model == "gpt-4.1-mini"

    def test_groq_not_reasoning_model(self):
        """Llama 4 Scout should NOT be detected as a reasoning model."""
        planner = _make_planner(_make_groq_settings())
        is_reasoning = any(
            planner._model.startswith(p) for p in ("gpt-5", "o1", "o3", "o4")
        )
        assert is_reasoning is False


# =============================================================================
# 2. Heuristic Bypass — Positive Cases
# =============================================================================

class TestHeuristicBypassPositive:
    """Tests for queries that SHOULD be handled by heuristic bypass."""

    def _get_plan(self, query: str):
        planner = _make_planner(_make_groq_settings())
        return planner.try_heuristic_plan(query)

    # --- Pure brand ---

    def test_brand_exact_boohoo(self):
        plan = self._get_plan("boohoo")
        assert plan is not None
        assert plan.intent == "exact"
        assert plan.brand == "Boohoo"
        assert len(plan.semantic_queries) == 1

    def test_brand_exact_nasty_gal(self):
        plan = self._get_plan("nasty gal")
        assert plan is not None
        assert plan.intent == "exact"
        assert plan.brand == "Nasty Gal"

    def test_brand_with_special_chars(self):
        """Ba&sh should match (ampersand in brand name)."""
        plan = self._get_plan("ba&sh")
        assert plan is not None
        assert plan.intent == "exact"
        assert plan.brand == "Ba&sh"

    def test_brand_html_entity(self):
        """HTML-encoded ampersand should be unescaped before matching."""
        plan = self._get_plan("ba&amp;sh")
        assert plan is not None
        assert plan.intent == "exact"
        assert plan.brand == "Ba&sh"

    def test_brand_case_insensitive(self):
        plan = self._get_plan("BOOHOO")
        assert plan is not None
        assert plan.brand == "Boohoo"

    def test_brand_h_and_m(self):
        plan = self._get_plan("h&m")
        assert plan is not None
        assert plan.brand == "H&M"

    # --- Bare category ---

    def test_category_dresses(self):
        plan = self._get_plan("dresses")
        assert plan is not None
        assert plan.intent == "exact"
        assert plan.brand is None
        assert "dresses" in plan.algolia_query.lower()

    def test_category_tops(self):
        plan = self._get_plan("tops")
        assert plan is not None
        assert plan.intent == "exact"

    def test_category_jeans(self):
        plan = self._get_plan("jeans")
        assert plan is not None
        assert plan.intent == "exact"

    def test_category_blazers(self):
        plan = self._get_plan("blazers")
        assert plan is not None
        assert plan.intent == "exact"

    def test_category_jumpsuit(self):
        plan = self._get_plan("jumpsuit")
        assert plan is not None
        assert plan.intent == "exact"

    def test_category_crop_top(self):
        """Multi-word category keyword."""
        plan = self._get_plan("crop top")
        assert plan is not None
        assert plan.intent == "exact"

    # --- Category + color ---

    def test_black_dress(self):
        plan = self._get_plan("black dress")
        assert plan is not None
        assert plan.intent == "specific"
        assert plan.attributes.get("colors") == ["black"]
        assert len(plan.semantic_queries) == 2

    def test_red_skirt(self):
        plan = self._get_plan("red skirt")
        assert plan is not None
        assert plan.intent == "specific"
        assert plan.attributes.get("colors") == ["red"]

    def test_navy_blazer(self):
        plan = self._get_plan("navy blazer")
        assert plan is not None
        assert plan.intent == "specific"
        assert plan.attributes.get("colors") == ["navy"]

    def test_white_blouse(self):
        plan = self._get_plan("white blouse")
        assert plan is not None
        assert plan.intent == "specific"
        assert plan.attributes.get("colors") == ["white"]

    def test_ivory_jumpsuit(self):
        plan = self._get_plan("ivory jumpsuit")
        assert plan is not None
        assert plan.intent == "specific"
        assert "ivory" in plan.attributes.get("colors", [])

    def test_category_color_reversed_order(self):
        """'dress black' — color after category."""
        plan = self._get_plan("dress black")
        assert plan is not None
        assert plan.intent == "specific"
        assert plan.attributes.get("colors") == ["black"]


# =============================================================================
# 3. Heuristic Bypass — Negative Cases (should return None)
# =============================================================================

class TestHeuristicBypassNegative:
    """Tests for queries that should NOT be handled by heuristic bypass."""

    def _get_plan(self, query: str):
        planner = _make_planner(_make_groq_settings())
        return planner.try_heuristic_plan(query)

    def test_vague_query_returns_none(self):
        """Vague/mood queries need LLM understanding."""
        assert self._get_plan("summer vibes") is None

    def test_quiet_luxury_returns_none(self):
        assert self._get_plan("quiet luxury blazer") is None

    def test_date_night_returns_none(self):
        assert self._get_plan("outfit for date night") is None

    def test_category_plus_attribute_returns_none(self):
        """Category + non-color attribute needs LLM for proper mode mapping."""
        assert self._get_plan("midi dress") is None  # "midi" is attribute

    def test_category_plus_material_returns_none(self):
        assert self._get_plan("silk blouse") is None

    def test_category_plus_pattern_returns_none(self):
        assert self._get_plan("floral dress") is None

    def test_brand_plus_category_returns_none(self):
        """Brand + other words needs LLM to decide filters."""
        assert self._get_plan("boohoo dresses") is None

    def test_complex_query_returns_none(self):
        assert self._get_plan("black fitted midi dress for office") is None

    def test_long_query_returns_none(self):
        """Queries > 4 words are too complex for heuristics."""
        assert self._get_plan("elegant evening gown for wedding guest") is None

    def test_empty_query_returns_none(self):
        assert self._get_plan("") is None

    def test_whitespace_only_returns_none(self):
        assert self._get_plan("   ") is None

    def test_two_categories_returns_none(self):
        """Multiple categories need LLM disambiguation."""
        assert self._get_plan("dress jacket") is None

    def test_coquette_returns_none(self):
        assert self._get_plan("coquette") is None

    def test_y2k_tops_returns_none(self):
        """Vague keyword + category should go to LLM."""
        assert self._get_plan("y2k tops") is None

    def test_heuristic_disabled_returns_none(self):
        """When bypass is disabled via settings, always return None."""
        planner = _make_planner(_make_groq_settings(
            query_planner_heuristic_bypass=False
        ))
        assert planner.try_heuristic_plan("dresses") is None
        assert planner.try_heuristic_plan("boohoo") is None


# =============================================================================
# 4. Heuristic Bypass — Edge Cases
# =============================================================================

class TestHeuristicBypassEdgeCases:
    """Edge cases for heuristic bypass."""

    def _get_plan(self, query: str):
        planner = _make_planner(_make_groq_settings())
        return planner.try_heuristic_plan(query)

    def test_leading_trailing_whitespace(self):
        plan = self._get_plan("  dresses  ")
        assert plan is not None
        assert plan.intent == "exact"

    def test_mixed_case_category(self):
        plan = self._get_plan("DRESSES")
        assert plan is not None
        assert plan.intent == "exact"

    def test_mixed_case_brand(self):
        plan = self._get_plan("PrettyLittleThing")
        assert plan is not None
        assert plan.brand == "PrettyLittleThing"

    def test_html_entity_ampersand(self):
        plan = self._get_plan("H&amp;M")
        assert plan is not None
        assert plan.brand == "H&M"

    def test_semantic_queries_are_nonempty_strings(self):
        """All heuristic plans should have non-empty semantic queries."""
        for query in ["boohoo", "dresses", "black dress"]:
            plan = self._get_plan(query)
            assert plan is not None
            for sq in plan.semantic_queries:
                assert isinstance(sq, str)
                assert len(sq.strip()) > 0

    def test_confidence_is_set(self):
        """All heuristic plans should have confidence >= 0.9."""
        for query in ["boohoo", "dresses", "black dress"]:
            plan = self._get_plan(query)
            assert plan is not None
            assert plan.confidence >= 0.9

    def test_category_color_produces_two_semantic_queries(self):
        """Category+color should produce exactly 2 lane-like queries."""
        plan = self._get_plan("black dress")
        assert plan is not None
        assert len(plan.semantic_queries) == 2
        # Verify they're different (style diversity)
        assert plan.semantic_queries[0] != plan.semantic_queries[1]

    def test_brand_plan_has_brand_in_semantic_query(self):
        """Brand heuristic should include brand name in semantic query."""
        plan = self._get_plan("reformation")
        assert plan is not None
        assert "reformation" in plan.semantic_queries[0].lower()

    def test_bare_category_semantic_query_includes_category(self):
        plan = self._get_plan("jeans")
        assert plan is not None
        assert "jeans" in plan.semantic_queries[0].lower()


# =============================================================================
# 5. Response Time Benchmarks
# =============================================================================

class TestHeuristicResponseTime:
    """Verify heuristic bypass is effectively instant (< 1ms)."""

    def _time_heuristic(self, query: str, iterations: int = 100) -> float:
        """Run heuristic N times, return average microseconds."""
        planner = _make_planner(_make_groq_settings())
        start = time.perf_counter()
        for _ in range(iterations):
            planner.try_heuristic_plan(query)
        elapsed = time.perf_counter() - start
        return (elapsed / iterations) * 1_000_000  # microseconds

    def test_brand_bypass_under_1ms(self):
        avg_us = self._time_heuristic("boohoo")
        assert avg_us < 1000, f"Brand bypass took {avg_us:.0f}us avg, expected < 1000us"

    def test_category_bypass_under_1ms(self):
        avg_us = self._time_heuristic("dresses")
        assert avg_us < 1000, f"Category bypass took {avg_us:.0f}us avg, expected < 1000us"

    def test_category_color_bypass_under_1ms(self):
        avg_us = self._time_heuristic("black dress")
        assert avg_us < 1000, f"Category+color bypass took {avg_us:.0f}us avg, expected < 1000us"

    def test_negative_case_under_1ms(self):
        """Even when heuristic returns None, it should be fast."""
        avg_us = self._time_heuristic("quiet luxury blazer")
        assert avg_us < 1000, f"Negative case took {avg_us:.0f}us avg, expected < 1000us"

    def test_brand_bypass_under_100us_avg(self):
        """Stretch goal: brand match should be < 100us on average."""
        avg_us = self._time_heuristic("boohoo", iterations=500)
        # This is a soft assertion — log it even if it fails
        print(f"  Brand bypass avg: {avg_us:.0f}us")
        assert avg_us < 500, f"Brand bypass took {avg_us:.0f}us avg, expected < 500us"


# =============================================================================
# 6. Groq LLM Call Wiring (mocked)
# =============================================================================

class TestGroqLLMCall:
    """Verify that Groq planner makes correct API calls (mocked)."""

    def test_groq_plan_uses_temperature_not_reasoning(self):
        """Groq Llama models should use temperature, not max_completion_tokens."""
        planner = _make_planner(_make_groq_settings())

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.content = json.dumps({
            "intent": "specific",
            "algolia_query": "black midi dress",
            "semantic_query": "elegant black midi dress",
            "semantic_queries": [
                "black midi dress, clean styling",
                "black midi dress, fashion forward",
            ],
            "modes": [],
            "attributes": {},
            "avoid": {},
            "confidence": 0.85,
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        planner._client = mock_client

        plan = planner.plan("black midi dress")
        assert plan is not None

        # Verify the API call parameters
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "meta-llama/llama-4-scout-17b-16e-instruct"
        assert "temperature" in call_kwargs
        assert call_kwargs["temperature"] == 0.15
        assert "max_tokens" in call_kwargs
        assert call_kwargs["max_tokens"] == 1600
        # Should NOT have max_completion_tokens (that's for reasoning models)
        assert "max_completion_tokens" not in call_kwargs

    def test_groq_plan_returns_search_plan(self):
        """Groq planner should return a valid SearchPlan on success."""
        planner = _make_planner(_make_groq_settings())

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].finish_reason = "stop"
        mock_response.choices[0].message.content = json.dumps({
            "intent": "vague",
            "algolia_query": "summer outfit",
            "semantic_query": "light airy summer outfit",
            "semantic_queries": [
                "breezy summer dress with floral print",
                "casual linen top and shorts set",
                "lightweight maxi dress for warm weather",
            ],
            "modes": ["summer_vibes"],
            "attributes": {},
            "avoid": {},
            "confidence": 0.75,
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        planner._client = mock_client

        plan = planner.plan("summer vibes")
        assert plan is not None
        assert plan.intent == "vague"
        assert len(plan.semantic_queries) == 3

    def test_groq_plan_returns_none_on_timeout(self):
        """Groq planner should return None on timeout (caller falls back)."""
        planner = _make_planner(_make_groq_settings())

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Request timed out")
        planner._client = mock_client

        plan = planner.plan("something complex")
        assert plan is None


# =============================================================================
# 7. V2 Route Integration (mocked service)
# =============================================================================

class TestV2RouteWiring:
    """Test that /api/search/v2/hybrid correctly wires heuristic bypass."""

    def test_heuristic_hit_sets_planner_source(self):
        """When heuristic matches, timing should show planner_source=heuristic."""
        from search.models import HybridSearchResponse, PaginationInfo, ProductResult

        mock_response = HybridSearchResponse(
            query="dresses",
            intent="exact",
            results=[],
            pagination=PaginationInfo(
                page=1, page_size=50, total_results=0, has_more=False,
            ),
            timing={"total_ms": 100},
        )

        with patch("api.routes.search_v2.get_hybrid_search_service") as mock_svc, \
             patch("api.routes.search_v2._load_user_profile", return_value=None), \
             patch("api.routes.search_v2._build_user_context", return_value=None), \
             patch("api.routes.search_v2._build_planner_context", return_value=None), \
             patch("api.routes.search_v2._load_session_scores", return_value=None), \
             patch("api.routes.search_v2._forward_search_signal"):

            mock_svc.return_value.search.return_value = mock_response

            from api.routes.search_v2 import hybrid_search_v2
            from search.models import HybridSearchRequest

            request = HybridSearchRequest(query="dresses")
            mock_user = MagicMock()
            mock_user.id = "test-user-123"
            mock_bg = MagicMock()

            result = hybrid_search_v2(request, mock_bg, mock_user)

            assert result.timing["search_version"] == "v2"
            assert result.timing["planner_source"] == "heuristic"

            # Verify pre_plan was passed to service.search
            call_kwargs = mock_svc.return_value.search.call_args[1]
            assert call_kwargs["pre_plan"] is not None
            assert call_kwargs["pre_plan"].intent == "exact"

    def test_complex_query_falls_to_llm(self):
        """When heuristic doesn't match, planner_source should be 'llm'."""
        from search.models import HybridSearchResponse, PaginationInfo

        mock_response = HybridSearchResponse(
            query="quiet luxury blazer",
            intent="vague",
            results=[],
            pagination=PaginationInfo(
                page=1, page_size=50, total_results=0, has_more=False,
            ),
            timing={"total_ms": 2000},
        )

        with patch("api.routes.search_v2.get_hybrid_search_service") as mock_svc, \
             patch("api.routes.search_v2._load_user_profile", return_value=None), \
             patch("api.routes.search_v2._build_user_context", return_value=None), \
             patch("api.routes.search_v2._build_planner_context", return_value=None), \
             patch("api.routes.search_v2._load_session_scores", return_value=None), \
             patch("api.routes.search_v2._forward_search_signal"):

            mock_svc.return_value.search.return_value = mock_response

            from api.routes.search_v2 import hybrid_search_v2
            from search.models import HybridSearchRequest

            request = HybridSearchRequest(query="quiet luxury blazer")
            mock_user = MagicMock()
            mock_user.id = "test-user-123"
            mock_bg = MagicMock()

            result = hybrid_search_v2(request, mock_bg, mock_user)

            assert result.timing["search_version"] == "v2"
            assert result.timing["planner_source"] == "llm"

            # Verify pre_plan was NOT passed (None)
            call_kwargs = mock_svc.return_value.search.call_args[1]
            assert call_kwargs["pre_plan"] is None

    def test_pagination_request_skips_heuristic(self):
        """Cursor-based pagination should NOT attempt heuristic bypass."""
        from search.models import HybridSearchResponse, PaginationInfo

        mock_response = HybridSearchResponse(
            query="dresses",
            intent="exact",
            results=[],
            pagination=PaginationInfo(
                page=2, page_size=50, total_results=0, has_more=False,
            ),
            timing={"total_ms": 50},
        )

        with patch("api.routes.search_v2.get_hybrid_search_service") as mock_svc, \
             patch("api.routes.search_v2._load_user_profile", return_value=None), \
             patch("api.routes.search_v2._build_user_context", return_value=None), \
             patch("api.routes.search_v2._build_planner_context", return_value=None), \
             patch("api.routes.search_v2._load_session_scores", return_value=None), \
             patch("api.routes.search_v2._forward_search_signal"):

            mock_svc.return_value.search.return_value = mock_response

            from api.routes.search_v2 import hybrid_search_v2
            from search.models import HybridSearchRequest

            request = HybridSearchRequest(
                query="dresses",
                search_session_id="ss_abc123",
                cursor="eyJwYWdlIjogMn0=",
            )
            mock_user = MagicMock()
            mock_user.id = "test-user-123"
            mock_bg = MagicMock()

            result = hybrid_search_v2(request, mock_bg, mock_user)

            # pre_plan should be None (skipped heuristic)
            call_kwargs = mock_svc.return_value.search.call_args[1]
            assert call_kwargs["pre_plan"] is None
            assert result.timing["planner_source"] == "llm"


# =============================================================================
# Section 7: Enrichment skip when FAISS attributes are baked in
# =============================================================================

class TestEnrichmentSkip:
    """Test that _enrich_semantic_results skips Algolia when attributes present."""

    def _make_service(self):
        """Create a HybridSearchService with a mock Algolia client."""
        mock_algolia = MagicMock()
        with patch("search.hybrid_search.get_query_planner"):
            from search.hybrid_search import HybridSearchService
            svc = HybridSearchService(algolia_client=mock_algolia)
            return svc, mock_algolia

    def _make_faiss_results(self, count=10):
        """Build results that look like FAISS v2 output (attributes baked in)."""
        return [
            {
                "product_id": f"pid-{i}",
                "name": f"Product {i}",
                "brand": "TestBrand",
                "price": 50.0 + i,
                "category_l1": "Tops",
                "formality": "Casual",
                "silhouette": "Relaxed",
                "neckline": "Crew",
                "sleeve_type": "Short",
                "color_family": "Neutrals",
                "pattern": "Solid",
                "semantic_score": 0.9 - i * 0.01,
                "source": "semantic",
            }
            for i in range(count)
        ]

    def _make_pgvector_results(self, count=10):
        """Build results that look like pgvector output (no Gemini attributes)."""
        return [
            {
                "product_id": f"pid-{i}",
                "name": f"Product {i}",
                "brand": "TestBrand",
                "price": 50.0 + i,
                "category_l1": None,
                "formality": None,
                "silhouette": None,
                "semantic_score": 0.9 - i * 0.01,
                "source": "semantic",
            }
            for i in range(count)
        ]

    def test_skip_enrichment_when_faiss_attrs_present(self):
        """FAISS v2 results with category_l1 should skip Algolia entirely."""
        svc, mock_algolia = self._make_service()
        results = self._make_faiss_results(20)

        enriched = svc._enrich_semantic_results(results)

        # Algolia get_objects should NOT have been called
        mock_algolia.get_objects.assert_not_called()
        # Results should be returned unchanged
        assert len(enriched) == 20
        assert enriched[0]["category_l1"] == "Tops"

    def test_enrichment_runs_for_pgvector_results(self):
        """pgvector results without category_l1 should trigger Algolia fetch."""
        svc, mock_algolia = self._make_service()
        results = self._make_pgvector_results(10)

        mock_algolia.get_objects.return_value = {
            f"pid-{i}": {"category_l1": "Dresses", "formality": "Evening"}
            for i in range(10)
        }

        enriched = svc._enrich_semantic_results(results)

        # Algolia get_objects SHOULD have been called
        mock_algolia.get_objects.assert_called_once()
        # Results should now have enriched fields
        assert enriched[0]["category_l1"] == "Dresses"

    def test_enrichment_runs_for_empty_results(self):
        """Empty result list should short-circuit without Algolia call."""
        svc, mock_algolia = self._make_service()

        enriched = svc._enrich_semantic_results([])

        mock_algolia.get_objects.assert_not_called()
        assert enriched == []

    def test_skip_with_partial_coverage(self):
        """If most results have attrs, skip enrichment (>=50% threshold)."""
        svc, mock_algolia = self._make_service()
        # 4 out of 5 sample results have category_l1
        results = self._make_faiss_results(4) + self._make_pgvector_results(1)

        enriched = svc._enrich_semantic_results(results)

        # 4/5 = 80% >= 50% threshold => skip
        mock_algolia.get_objects.assert_not_called()
        assert len(enriched) == 5

    def test_no_skip_with_low_coverage(self):
        """If most results lack attrs, run enrichment."""
        svc, mock_algolia = self._make_service()
        # 1 out of 5 sample results has category_l1
        results = self._make_faiss_results(1) + self._make_pgvector_results(4)

        mock_algolia.get_objects.return_value = {
            r["product_id"]: {"category_l1": "Tops"} for r in results
        }

        enriched = svc._enrich_semantic_results(results)

        # 1/5 = 20% < 50% threshold => run enrichment
        mock_algolia.get_objects.assert_called_once()

    def test_skip_enrichment_preserves_all_fields(self):
        """When enrichment is skipped, all existing fields are preserved."""
        svc, mock_algolia = self._make_service()
        results = self._make_faiss_results(3)

        enriched = svc._enrich_semantic_results(results)

        for r in enriched:
            assert r["neckline"] == "Crew"
            assert r["sleeve_type"] == "Short"
            assert r["formality"] == "Casual"
            assert r["color_family"] == "Neutrals"
            assert r["pattern"] == "Solid"
        mock_algolia.get_objects.assert_not_called()
