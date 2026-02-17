"""
Unit tests for the LLM-based query planner.

Tests cover:
1. SearchPlan model validation
2. plan_to_request_updates conversion
3. Planner disabled/fallback behavior
4. Mock LLM responses for various query types
5. Integration with post-filter expanded_filters

Run with: PYTHONPATH=src python -m pytest tests/unit/test_query_planner.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from search.query_planner import QueryPlanner, SearchPlan, get_query_planner
from search.models import QueryIntent


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_plan_floral_jacket():
    """Sample SearchPlan for 'a jacket with floral leaves'."""
    return SearchPlan(
        intent="specific",
        algolia_query="leaf print jacket",
        semantic_query="a jacket with botanical leaf and floral print pattern",
        filters={
            "category_l2": ["Jacket", "Jackets"],
            "patterns": ["Floral"],
        },
        expanded_filters={
            "category_l2": [
                "Jacket", "Jackets", "Bomber Jacket", "Denim Jacket",
                "Leather Jacket", "Puffer Jacket", "Fleece Jacket",
            ],
            "patterns": ["Floral", "Tropical", "Abstract"],
        },
        brand=None,
        matched_terms=["jacket", "floral", "leaves"],
        confidence=0.9,
    )


@pytest.fixture
def sample_plan_brand_exact():
    """Sample SearchPlan for 'Zara'."""
    return SearchPlan(
        intent="exact",
        algolia_query="",
        semantic_query="Zara fashion clothing",
        filters={"brands": ["Zara"]},
        expanded_filters={},
        brand="Zara",
        matched_terms=["zara"],
        confidence=0.95,
    )


@pytest.fixture
def sample_plan_vague():
    """Sample SearchPlan for 'quiet luxury date night'."""
    return SearchPlan(
        intent="vague",
        algolia_query="",
        semantic_query="elegant quiet luxury outfit for a romantic date night, understated sophistication",
        filters={
            "occasions": ["Date Night"],
            "formality": ["Semi-Formal", "Smart Casual"],
        },
        expanded_filters={
            "occasions": ["Date Night", "Party", "Night Out"],
        },
        brand=None,
        matched_terms=["quiet luxury", "date night"],
        confidence=0.85,
    )


@pytest.fixture
def sample_plan_simple():
    """Sample SearchPlan for 'red midi dress'."""
    return SearchPlan(
        intent="specific",
        algolia_query="midi dress",
        semantic_query="a red midi length dress",
        filters={
            "colors": ["Red"],
            "length": ["Midi"],
            "category_l1": ["Dresses"],
        },
        expanded_filters={},
        brand=None,
        matched_terms=["red", "midi", "dress"],
        confidence=0.95,
    )


# =============================================================================
# 1. SearchPlan Model Tests
# =============================================================================

class TestSearchPlanModel:
    """Tests for the SearchPlan pydantic model."""

    def test_valid_plan(self, sample_plan_floral_jacket):
        """Valid plan should parse without errors."""
        plan = sample_plan_floral_jacket
        assert plan.intent == "specific"
        assert plan.algolia_query == "leaf print jacket"
        assert "Floral" in plan.filters["patterns"]
        assert "Tropical" in plan.expanded_filters["patterns"]

    def test_minimal_plan(self):
        """Plan with only required fields should work."""
        plan = SearchPlan(intent="specific")
        assert plan.algolia_query == ""
        assert plan.filters == {}
        assert plan.expanded_filters == {}
        assert plan.brand is None
        assert plan.confidence == 0.8

    def test_plan_from_json(self):
        """Plan should parse from JSON (as returned by LLM)."""
        raw = json.dumps({
            "intent": "specific",
            "algolia_query": "floral jacket",
            "semantic_query": "a jacket with floral print",
            "filters": {"patterns": ["Floral"]},
            "expanded_filters": {"patterns": ["Floral", "Tropical"]},
            "brand": None,
            "matched_terms": ["floral", "jacket"],
            "confidence": 0.9,
        })
        plan = SearchPlan(**json.loads(raw))
        assert plan.intent == "specific"
        assert plan.filters["patterns"] == ["Floral"]

    def test_plan_handles_extra_fields(self):
        """Plan should ignore unknown fields from LLM response."""
        plan = SearchPlan(
            intent="specific",
            algolia_query="test",
        )
        assert plan.intent == "specific"


# =============================================================================
# 2. plan_to_request_updates Tests
# =============================================================================

class TestPlanToRequestUpdates:
    """Tests for converting SearchPlan to request updates."""

    def test_floral_jacket_updates(self, sample_plan_floral_jacket):
        """Floral jacket plan should produce correct request updates."""
        planner = QueryPlanner.__new__(QueryPlanner)
        planner._enabled = False  # Don't need real API

        updates, expanded, excludes, matched, algolia_q, semantic_q, intent_str = (
            planner.plan_to_request_updates(sample_plan_floral_jacket)
        )

        assert "category_l2" in updates
        assert "Jacket" in updates["category_l2"]
        assert "patterns" in updates
        assert "Floral" in updates["patterns"]
        assert "Tropical" in expanded.get("patterns", [])
        assert algolia_q == "leaf print jacket"
        assert "botanical" in semantic_q
        assert intent_str == "specific"

    def test_brand_plan_updates(self, sample_plan_brand_exact):
        """Brand plan should inject brand filter."""
        planner = QueryPlanner.__new__(QueryPlanner)
        planner._enabled = False

        updates, expanded, excludes, matched, algolia_q, semantic_q, intent_str = (
            planner.plan_to_request_updates(sample_plan_brand_exact)
        )

        assert updates["brands"] == ["Zara"]
        assert intent_str == "exact"
        assert algolia_q == ""

    def test_vague_plan_updates(self, sample_plan_vague):
        """Vague plan should produce occasion/formality filters."""
        planner = QueryPlanner.__new__(QueryPlanner)
        planner._enabled = False

        updates, expanded, excludes, matched, algolia_q, semantic_q, intent_str = (
            planner.plan_to_request_updates(sample_plan_vague)
        )

        assert "occasions" in updates
        assert "Date Night" in updates["occasions"]
        assert intent_str == "vague"
        assert "Night Out" in expanded.get("occasions", [])

    def test_invalid_filter_fields_ignored(self):
        """Unknown filter field names from LLM should be ignored."""
        planner = QueryPlanner.__new__(QueryPlanner)
        planner._enabled = False

        plan = SearchPlan(
            intent="specific",
            filters={"invalid_field": ["value"], "patterns": ["Floral"]},
        )
        updates, _, _, _, _, _, _ = planner.plan_to_request_updates(plan)

        assert "invalid_field" not in updates
        assert "patterns" in updates

    def test_empty_filter_values_ignored(self):
        """Empty filter values should not be included."""
        planner = QueryPlanner.__new__(QueryPlanner)
        planner._enabled = False

        plan = SearchPlan(
            intent="specific",
            filters={"patterns": [], "colors": ["Red"]},
        )
        updates, _, _, _, _, _, _ = planner.plan_to_request_updates(plan)

        assert "patterns" not in updates
        assert "colors" in updates


# =============================================================================
# 3. Planner Disabled/Fallback Tests
# =============================================================================

class TestPlannerFallback:
    """Tests for planner disabled/error scenarios."""

    def test_planner_disabled_without_api_key(self):
        """Planner should be disabled when no API key is set."""
        with patch("search.query_planner.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                openai_api_key="",
                query_planner_model="gpt-4o-mini",
                query_planner_enabled=True,
                query_planner_timeout_seconds=2.0,
            )
            planner = QueryPlanner()
            assert not planner.enabled
            assert planner.plan("test query") is None

    def test_planner_disabled_by_flag(self):
        """Planner should be disabled when feature flag is off."""
        with patch("search.query_planner.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                openai_api_key="sk-test",
                query_planner_model="gpt-4o-mini",
                query_planner_enabled=False,
                query_planner_timeout_seconds=2.0,
            )
            planner = QueryPlanner()
            assert not planner.enabled

    def test_planner_returns_none_on_error(self):
        """Planner should return None on API error (caller falls back)."""
        with patch("search.query_planner.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                openai_api_key="sk-test",
                query_planner_model="gpt-4o-mini",
                query_planner_enabled=True,
                query_planner_timeout_seconds=2.0,
            )
            planner = QueryPlanner()

            # Mock the client to raise an exception
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("API timeout")
            planner._client = mock_client

            result = planner.plan("floral jacket")
            assert result is None

    def test_planner_returns_none_on_invalid_json(self):
        """Planner should return None when LLM returns invalid JSON."""
        with patch("search.query_planner.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                openai_api_key="sk-test",
                query_planner_model="gpt-4o-mini",
                query_planner_enabled=True,
                query_planner_timeout_seconds=2.0,
            )
            planner = QueryPlanner()

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "not valid json {{"

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            planner._client = mock_client

            result = planner.plan("floral jacket")
            assert result is None


# =============================================================================
# 4. Mock LLM Response Tests
# =============================================================================

class TestMockLLMResponses:
    """Tests with mocked LLM responses to verify end-to-end planning."""

    def _make_planner_with_response(self, response_dict: dict) -> QueryPlanner:
        """Create a planner that returns a mocked LLM response."""
        with patch("search.query_planner.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                openai_api_key="sk-test",
                query_planner_model="gpt-4o-mini",
                query_planner_enabled=True,
                query_planner_timeout_seconds=2.0,
            )
            planner = QueryPlanner()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(response_dict)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        planner._client = mock_client

        return planner

    def test_floral_leaves_jacket(self):
        """'a jacket with floral leaves' should get proper decomposition."""
        planner = self._make_planner_with_response({
            "intent": "specific",
            "algolia_query": "leaf print jacket",
            "semantic_query": "a jacket with botanical leaf and floral print pattern",
            "filters": {
                "category_l2": ["Jacket", "Jackets"],
                "patterns": ["Floral"],
            },
            "expanded_filters": {
                "category_l2": ["Jacket", "Jackets", "Bomber Jacket", "Denim Jacket"],
                "patterns": ["Floral", "Tropical", "Abstract"],
            },
            "brand": None,
            "matched_terms": ["jacket", "floral", "leaves"],
            "confidence": 0.9,
        })

        plan = planner.plan("a jacket with floral leaves")
        assert plan is not None
        assert plan.intent == "specific"
        assert plan.algolia_query == "leaf print jacket"
        assert "Tropical" in plan.expanded_filters.get("patterns", [])

    def test_red_midi_dress(self):
        """'red midi dress' should get simple clean extraction."""
        planner = self._make_planner_with_response({
            "intent": "specific",
            "algolia_query": "midi dress",
            "semantic_query": "a red midi length dress",
            "filters": {
                "colors": ["Red"],
                "length": ["Midi"],
                "category_l1": ["Dresses"],
            },
            "expanded_filters": {},
            "brand": None,
            "matched_terms": ["red", "midi", "dress"],
            "confidence": 0.95,
        })

        plan = planner.plan("red midi dress")
        assert plan is not None
        assert plan.filters["colors"] == ["Red"]
        assert plan.algolia_query == "midi dress"

    def test_boohoo_brand_search(self):
        """'boohoo' should be classified as exact brand search."""
        planner = self._make_planner_with_response({
            "intent": "exact",
            "algolia_query": "",
            "semantic_query": "Boohoo fashion clothing",
            "filters": {"brands": ["Boohoo"]},
            "expanded_filters": {},
            "brand": "Boohoo",
            "matched_terms": ["boohoo"],
            "confidence": 0.95,
        })

        plan = planner.plan("boohoo")
        assert plan is not None
        assert plan.intent == "exact"
        assert plan.brand == "Boohoo"

    def test_quiet_luxury_vague(self):
        """'quiet luxury' should be classified as vague."""
        planner = self._make_planner_with_response({
            "intent": "vague",
            "algolia_query": "",
            "semantic_query": "elegant understated quiet luxury fashion, neutral tones, high quality fabrics",
            "filters": {
                "formality": ["Smart Casual", "Semi-Formal"],
            },
            "expanded_filters": {},
            "brand": None,
            "matched_terms": ["quiet luxury"],
            "confidence": 0.8,
        })

        plan = planner.plan("quiet luxury")
        assert plan is not None
        assert plan.intent == "vague"
        assert plan.semantic_query != ""


# =============================================================================
# 5. Post-Filter Expanded Filters Integration Tests
# =============================================================================

class TestPostFilterWithExpansion:
    """Tests that expanded_filters properly relaxes post-filtering."""

    def _make_result(self, product_id, **kwargs):
        base = {
            "product_id": product_id,
            "name": f"Product {product_id}",
            "brand": "TestBrand",
            "price": 50.0,
            "is_on_sale": False,
            "category_l1": "Outerwear",
            "category_l2": "Jacket",
            "broad_category": "outerwear",
            "pattern": "Floral",
            "primary_color": "Green",
            "occasions": [],
            "seasons": [],
            "materials": [],
            "style_tags": [],
            "colors": ["Green"],
        }
        base.update(kwargs)
        return base

    def test_expanded_pattern_boosts_matching(self):
        """With expanded_filters, matching patterns get boosted (soft scoring)."""
        from search.hybrid_search import HybridSearchService
        from search.models import HybridSearchRequest

        service = HybridSearchService(
            algolia_client=MagicMock(),
            analytics=MagicMock(),
        )

        results = [
            self._make_result("p1", pattern="Floral", rrf_score=0.01),
            self._make_result("p2", pattern="Tropical", rrf_score=0.01),
            self._make_result("p3", pattern="Solid", rrf_score=0.01),
        ]

        request = HybridSearchRequest(
            query="floral jacket",
            patterns=["Floral"],
        )

        # Without expansion: strict -- only Floral passes
        strict = service._post_filter_semantic(results, request, expanded_filters={})
        assert len(strict) == 1
        assert strict[0]["product_id"] == "p1"

        # With expansion (soft scoring): all 3 kept, but Floral/Tropical boosted
        expanded = service._post_filter_semantic(
            [self._make_result("p1", pattern="Floral", rrf_score=0.01),
             self._make_result("p2", pattern="Tropical", rrf_score=0.01),
             self._make_result("p3", pattern="Solid", rrf_score=0.01)],
            request,
            expanded_filters={"patterns": ["Floral", "Tropical", "Abstract"]},
        )
        # All items kept (soft scoring doesn't drop)
        assert len(expanded) == 3
        # Matching items (Floral, Tropical) should be boosted to the top
        assert expanded[0]["product_id"] in ("p1", "p2")
        assert expanded[1]["product_id"] in ("p1", "p2")
        # Non-matching item (Solid) should be last
        assert expanded[2]["product_id"] == "p3"

    def test_expanded_category_l2_allows_subtypes(self):
        """Bomber Jacket should pass when expanded category_l2 includes it."""
        from search.hybrid_search import HybridSearchService
        from search.models import HybridSearchRequest

        service = HybridSearchService(
            algolia_client=MagicMock(),
            analytics=MagicMock(),
        )

        results = [
            self._make_result("p1", category_l2="Jacket"),
            self._make_result("p2", category_l2="Bomber Jacket"),
            self._make_result("p3", category_l2="T-Shirt"),
        ]

        request = HybridSearchRequest(
            query="jacket",
            category_l2=["Jacket", "Jackets"],
        )

        # Without expansion: only exact "Jacket" passes
        strict = service._post_filter_semantic(results, request, expanded_filters={})
        assert len(strict) == 1
        assert strict[0]["product_id"] == "p1"

        # With expansion: Bomber Jacket passes too (substring match)
        expanded = service._post_filter_semantic(
            results, request,
            expanded_filters={"category_l2": [
                "Jacket", "Jackets", "Bomber Jacket", "Denim Jacket",
            ]},
        )
        assert len(expanded) == 2
        pids = {r["product_id"] for r in expanded}
        assert "p1" in pids
        assert "p2" in pids
        assert "p3" not in pids

    def test_no_expansion_is_strict(self):
        """Without expanded_filters, post-filter should behave strictly."""
        from search.hybrid_search import HybridSearchService
        from search.models import HybridSearchRequest

        service = HybridSearchService(
            algolia_client=MagicMock(),
            analytics=MagicMock(),
        )

        results = [
            self._make_result("p1", pattern="Floral"),
            self._make_result("p2", pattern="Tropical"),
        ]

        request = HybridSearchRequest(
            query="floral jacket",
            patterns=["Floral"],
        )

        # Default (no expansion): strict
        filtered = service._post_filter_semantic(results, request)
        assert len(filtered) == 1
        assert filtered[0]["product_id"] == "p1"
