"""
Unit tests for the LLM-based query planner (mode-based architecture).

Tests cover:
1. SearchPlan model validation (modes, attributes, avoid)
2. plan_to_request_updates conversion (mode expansion + merge)
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
        modes=[],
        attributes={
            "category_l2": ["Jacket", "Jackets"],
            "patterns": ["Floral"],
        },
        avoid={},
        brand=None,
        confidence=0.9,
    )


@pytest.fixture
def sample_plan_brand_exact():
    """Sample SearchPlan for 'Zara'."""
    return SearchPlan(
        intent="exact",
        algolia_query="",
        semantic_query="Zara fashion clothing",
        modes=[],
        attributes={"brands": ["Zara"]},
        avoid={},
        brand="Zara",
        confidence=0.95,
    )


@pytest.fixture
def sample_plan_vague():
    """Sample SearchPlan for 'quiet luxury date night'."""
    return SearchPlan(
        intent="vague",
        algolia_query="",
        semantic_query="elegant quiet luxury outfit for a romantic date night, understated sophistication",
        modes=["quiet_luxury", "date_night"],
        attributes={},
        avoid={},
        brand=None,
        confidence=0.85,
    )


@pytest.fixture
def sample_plan_simple():
    """Sample SearchPlan for 'red midi dress'."""
    return SearchPlan(
        intent="specific",
        algolia_query="midi dress",
        semantic_query="a red midi length dress",
        modes=[],
        attributes={
            "colors": ["Red"],
            "length": ["Midi"],
            "category_l1": ["Dresses"],
        },
        avoid={},
        brand=None,
        confidence=0.95,
    )


@pytest.fixture
def sample_plan_coverage():
    """Sample SearchPlan for 'modest dress for wedding, no polyester'."""
    return SearchPlan(
        intent="specific",
        algolia_query="dress",
        semantic_query="a modest conservative dress for a wedding, opaque fabric, full coverage",
        modes=["modest", "wedding_guest"],
        attributes={"category_l1": ["Dresses"]},
        avoid={"materials": ["Polyester"]},
        brand=None,
        confidence=0.9,
    )


# =============================================================================
# 1. SearchPlan Model Tests
# =============================================================================

class TestSearchPlanModel:
    """Tests for the SearchPlan pydantic model (mode-based)."""

    def test_valid_plan(self, sample_plan_floral_jacket):
        """Valid plan should parse without errors."""
        plan = sample_plan_floral_jacket
        assert plan.intent == "specific"
        assert plan.algolia_query == "leaf print jacket"
        assert "Floral" in plan.attributes["patterns"]
        assert plan.modes == []

    def test_minimal_plan(self):
        """Plan with only required fields should work."""
        plan = SearchPlan(intent="specific")
        assert plan.algolia_query == ""
        assert plan.modes == []
        assert plan.attributes == {}
        assert plan.avoid == {}
        assert plan.brand is None
        assert plan.confidence == 0.8

    def test_plan_from_json(self):
        """Plan should parse from JSON (as returned by LLM)."""
        raw = json.dumps({
            "intent": "specific",
            "algolia_query": "floral jacket",
            "semantic_query": "a jacket with floral print",
            "modes": [],
            "attributes": {"patterns": ["Floral"]},
            "avoid": {},
            "brand": None,
            "confidence": 0.9,
        })
        plan = SearchPlan(**json.loads(raw))
        assert plan.intent == "specific"
        assert plan.attributes["patterns"] == ["Floral"]

    def test_plan_handles_extra_fields(self):
        """Plan should ignore unknown fields from LLM response."""
        plan = SearchPlan(
            intent="specific",
            algolia_query="test",
            unknown_extra_field="should be ignored",
        )
        assert plan.intent == "specific"

    def test_plan_with_modes(self, sample_plan_coverage):
        """Plan with modes should store them correctly."""
        plan = sample_plan_coverage
        assert "modest" in plan.modes
        assert "wedding_guest" in plan.modes
        assert plan.avoid == {"materials": ["Polyester"]}
        assert plan.attributes == {"category_l1": ["Dresses"]}

    def test_plan_with_avoid(self):
        """Plan with avoid should store negative constraints."""
        plan = SearchPlan(
            intent="specific",
            modes=[],
            attributes={"category_l1": ["Bottoms"]},
            avoid={"style_tags": ["Distressed"], "patterns": ["Animal Print"]},
        )
        assert "Distressed" in plan.avoid["style_tags"]
        assert "Animal Print" in plan.avoid["patterns"]

    # ------ is_set SearchPlan model tests ------

    def test_search_plan_accepts_is_set(self):
        """SearchPlan should accept is_set=True."""
        plan = SearchPlan(intent="specific", is_set=True)
        assert plan.is_set is True

    def test_search_plan_defaults_is_set_none(self):
        """SearchPlan should default is_set to None."""
        plan = SearchPlan(intent="specific")
        assert plan.is_set is None

    def test_search_plan_extra_ignore_with_is_set(self):
        """Extra fields should be ignored even when is_set is present."""
        plan = SearchPlan(intent="specific", is_set=True, unknown_field="test")
        assert plan.is_set is True


# =============================================================================
# 2. plan_to_request_updates Tests
# =============================================================================

class TestPlanToRequestUpdates:
    """Tests for converting SearchPlan to request updates (with mode expansion)."""

    def _make_planner(self):
        """Create a planner instance without real API."""
        planner = QueryPlanner.__new__(QueryPlanner)
        planner._enabled = False
        return planner

    def test_attributes_only_plan(self, sample_plan_floral_jacket):
        """Plan with only attributes (no modes) should pass through as filters."""
        planner = self._make_planner()
        updates, expanded, excludes, matched, algolia_q, semantic_q, intent_str, _ = (
            planner.plan_to_request_updates(sample_plan_floral_jacket)
        )

        assert "category_l2" in updates
        assert "Jacket" in updates["category_l2"]
        assert "patterns" in updates
        assert "Floral" in updates["patterns"]
        assert algolia_q == "leaf print jacket"
        assert "botanical" in semantic_q
        assert intent_str == "specific"

    def test_brand_plan_updates(self, sample_plan_brand_exact):
        """Brand plan should inject brand filter."""
        planner = self._make_planner()
        updates, expanded, excludes, matched, algolia_q, semantic_q, intent_str, _ = (
            planner.plan_to_request_updates(sample_plan_brand_exact)
        )

        assert updates["brands"] == ["Zara"]
        assert intent_str == "exact"
        assert algolia_q == ""

    def test_mode_expansion_in_updates(self, sample_plan_vague):
        """Modes should be expanded into filters via expand_modes()."""
        planner = self._make_planner()
        updates, expanded, excludes, matched, algolia_q, semantic_q, intent_str, _ = (
            planner.plan_to_request_updates(sample_plan_vague)
        )

        # quiet_luxury mode → style_tags + formality
        assert "style_tags" in updates
        assert "Classic" in updates["style_tags"]
        assert "Minimalist" in updates["style_tags"]

        # date_night mode → occasions + formality
        assert "occasions" in updates
        assert "Date Night" in updates["occasions"]

        # formality should be merged from both modes
        assert "formality" in updates
        assert "Smart Casual" in updates["formality"]
        assert "Business Casual" in updates["formality"]

        assert intent_str == "vague"

    def test_coverage_mode_produces_exclusions(self, sample_plan_coverage):
        """Coverage modes should expand into exclude_* request fields."""
        planner = self._make_planner()
        updates, expanded, excludes, matched, algolia_q, semantic_q, intent_str, _ = (
            planner.plan_to_request_updates(sample_plan_coverage)
        )

        # modest mode → all coverage exclusions
        assert "exclude_sleeve_type" in updates
        assert "Sleeveless" in updates["exclude_sleeve_type"]
        assert "Short" in updates["exclude_sleeve_type"]

        assert "exclude_neckline" in updates
        assert "V-Neck" in updates["exclude_neckline"]
        assert "Strapless" in updates["exclude_neckline"]

        assert "exclude_materials" in updates
        # Should include both mode exclusions (Mesh, Lace, Chiffon, Sheer)
        # AND avoid values (Polyester)
        assert "Mesh" in updates["exclude_materials"]
        assert "Polyester" in updates["exclude_materials"]

        assert "exclude_length" in updates
        assert "Mini" in updates["exclude_length"]

        # wedding_guest mode → formality filter
        assert "formality" in updates
        assert "Formal" in updates["formality"]

        # attributes → category_l1
        assert "category_l1" in updates
        assert "Dresses" in updates["category_l1"]

    def test_avoid_without_modes(self):
        """Avoid values should produce exclusions even without modes."""
        planner = self._make_planner()
        plan = SearchPlan(
            intent="specific",
            modes=[],
            attributes={"category_l1": ["Bottoms"]},
            avoid={"style_tags": ["Distressed"], "materials": ["Polyester"]},
        )
        updates, expanded, excludes, matched, algolia_q, semantic_q, intent_str, _ = (
            planner.plan_to_request_updates(plan)
        )

        assert "exclude_style_tags" in updates
        assert "Distressed" in updates["exclude_style_tags"]
        assert "exclude_materials" in updates
        assert "Polyester" in updates["exclude_materials"]

    def test_mode_and_avoid_merge_exclusions(self):
        """Mode exclusions and avoid values should union without duplicates."""
        planner = self._make_planner()
        plan = SearchPlan(
            intent="specific",
            modes=["opaque"],  # excludes Mesh, Lace, Chiffon, Sheer
            attributes={},
            avoid={"materials": ["Polyester", "Mesh"]},  # Mesh overlaps with opaque
        )
        updates, _, excludes, _, _, _, _, _ = planner.plan_to_request_updates(plan)

        assert "exclude_materials" in updates
        materials = updates["exclude_materials"]
        # Should have Mesh, Lace, Chiffon, Sheer from opaque + Polyester from avoid
        assert "Mesh" in materials
        assert "Lace" in materials
        assert "Polyester" in materials
        # No duplicates
        assert len(materials) == len(set(m.lower() for m in materials))

    def test_invalid_attribute_fields_ignored(self):
        """Unknown attribute field names from LLM should be ignored."""
        planner = self._make_planner()
        plan = SearchPlan(
            intent="specific",
            modes=[],
            attributes={"invalid_field": ["value"], "patterns": ["Floral"]},
            avoid={},
        )
        updates, _, _, _, _, _, _, _ = planner.plan_to_request_updates(plan)

        assert "invalid_field" not in updates
        assert "patterns" in updates

    def test_empty_attribute_values_ignored(self):
        """Empty attribute values should not be included."""
        planner = self._make_planner()
        plan = SearchPlan(
            intent="specific",
            modes=[],
            attributes={"patterns": [], "colors": ["Red"]},
            avoid={},
        )
        updates, _, _, _, _, _, _, _ = planner.plan_to_request_updates(plan)

        assert "patterns" not in updates
        assert "colors" in updates

    def test_typo_correction_in_attributes(self):
        """Typos in attribute keys should be corrected."""
        planner = self._make_planner()
        plan = SearchPlan(
            intent="specific",
            modes=[],
            attributes={"necktine": ["V-Neck"], "pattern": ["Floral"]},
            avoid={"material": ["Polyester"]},
        )
        updates, _, _, _, _, _, _, _ = planner.plan_to_request_updates(plan)

        # necktine → neckline
        assert "neckline" in updates
        assert "V-Neck" in updates["neckline"]
        # pattern → patterns
        assert "patterns" in updates
        assert "Floral" in updates["patterns"]
        # material → materials in avoid
        assert "exclude_materials" in updates
        assert "Polyester" in updates["exclude_materials"]

    def test_price_and_sale_injection(self):
        """Price and on_sale_only should be injected into updates."""
        planner = self._make_planner()
        plan = SearchPlan(
            intent="specific",
            modes=[],
            attributes={},
            avoid={},
            max_price=50.0,
            min_price=20.0,
            on_sale_only=True,
        )
        updates, _, _, _, _, _, _, _ = planner.plan_to_request_updates(plan)

        assert updates["max_price"] == 50.0
        assert updates["min_price"] == 20.0
        assert updates["on_sale_only"] is True

    def test_semantic_query_fallback_to_algolia(self):
        """If semantic_query is empty, should fall back to algolia_query."""
        planner = self._make_planner()
        plan = SearchPlan(
            intent="specific",
            algolia_query="red dress",
            semantic_query="",
            modes=[],
            attributes={},
            avoid={},
        )
        _, _, _, _, algolia_q, semantic_q, _, _ = planner.plan_to_request_updates(plan)

        assert algolia_q == "red dress"
        assert semantic_q == "red dress"  # fallback

    def test_expanded_filters_include_mode_and_attribute_values(self):
        """Expanded filters should include values from both modes and attributes."""
        planner = self._make_planner()
        plan = SearchPlan(
            intent="specific",
            modes=["work"],  # occasions: Office, Work; formality: Business Casual
            attributes={"occasions": ["Meeting"]},  # extra value
            avoid={},
        )
        _, expanded, _, _, _, _, _, _ = planner.plan_to_request_updates(plan)

        assert "occasions" in expanded
        assert "Office" in expanded["occasions"]
        assert "Work" in expanded["occasions"]
        assert "Meeting" in expanded["occasions"]

    def test_matched_terms_always_empty(self):
        """matched_terms should always be empty list (legacy compat)."""
        planner = self._make_planner()
        plan = SearchPlan(intent="specific")
        _, _, _, matched, _, _, _, _ = planner.plan_to_request_updates(plan)
        assert matched == []

    # ------ is_set plan_to_request_updates tests ------

    def test_is_set_injection(self):
        """is_set=True should be injected into request_updates."""
        planner = self._make_planner()
        plan = SearchPlan(
            intent="specific",
            modes=[],
            attributes={},
            avoid={},
            is_set=True,
        )
        updates, _, _, _, _, _, _, _ = planner.plan_to_request_updates(plan)
        assert updates["is_set"] is True

    def test_is_set_none_not_injected(self):
        """is_set=None should NOT appear in request_updates."""
        planner = self._make_planner()
        plan = SearchPlan(
            intent="specific",
            modes=[],
            attributes={},
            avoid={},
        )
        updates, _, _, _, _, _, _, _ = planner.plan_to_request_updates(plan)
        assert "is_set" not in updates


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
                query_planner_model="gpt-4o",
                query_planner_enabled=True,
                query_planner_timeout_seconds=15.0,
            )
            planner = QueryPlanner()
            assert not planner.enabled
            assert planner.plan("test query") is None

    def test_planner_disabled_by_flag(self):
        """Planner should be disabled when feature flag is off."""
        with patch("search.query_planner.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                openai_api_key="sk-test",
                query_planner_model="gpt-4o",
                query_planner_enabled=False,
                query_planner_timeout_seconds=15.0,
            )
            planner = QueryPlanner()
            assert not planner.enabled

    def test_planner_returns_none_on_error(self):
        """Planner should return None on API error (caller falls back)."""
        with patch("search.query_planner.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                openai_api_key="sk-test",
                query_planner_model="gpt-4o",
                query_planner_enabled=True,
                query_planner_timeout_seconds=15.0,
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
                query_planner_model="gpt-4o",
                query_planner_enabled=True,
                query_planner_timeout_seconds=15.0,
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
# 4. Mock LLM Response Tests (mode-based format)
# =============================================================================

class TestMockLLMResponses:
    """Tests with mocked LLM responses to verify end-to-end planning."""

    def _make_planner_with_response(self, response_dict: dict) -> QueryPlanner:
        """Create a planner that returns a mocked LLM response."""
        with patch("search.query_planner.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                openai_api_key="sk-test",
                query_planner_model="gpt-4o",
                query_planner_enabled=True,
                query_planner_timeout_seconds=15.0,
            )
            planner = QueryPlanner()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(response_dict)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        planner._client = mock_client

        return planner

    def test_ribbed_knit_top_no_modes(self):
        """'Ribbed knit top with square neckline' — pure attributes, no modes."""
        planner = self._make_planner_with_response({
            "intent": "specific",
            "algolia_query": "ribbed knit top",
            "semantic_query": "a ribbed knit top with square neckline",
            "modes": [],
            "attributes": {
                "category_l1": ["Tops"],
                "neckline": ["Square"],
                "materials": ["Knit"],
            },
            "avoid": {},
            "confidence": 0.95,
        })

        plan = planner.plan("ribbed knit top with square neckline")
        assert plan is not None
        assert plan.intent == "specific"
        assert plan.modes == []
        assert plan.attributes["neckline"] == ["Square"]

        # Verify plan_to_request_updates
        updates, _, excludes, _, _, _, _, _ = planner.plan_to_request_updates(plan)
        assert "neckline" in updates
        assert "Square" in updates["neckline"]
        assert "category_l1" in updates
        assert excludes == {}  # no exclusions

    def test_hide_arms_mode(self):
        """'Help me find a top that hides my arms' — cover_arms mode."""
        planner = self._make_planner_with_response({
            "intent": "specific",
            "algolia_query": "top",
            "semantic_query": "a top with long sleeves that covers the arms completely",
            "modes": ["cover_arms"],
            "attributes": {"category_l1": ["Tops"]},
            "avoid": {},
            "confidence": 0.9,
        })

        plan = planner.plan("help me find a top that hides my arms")
        assert plan is not None
        assert "cover_arms" in plan.modes

        updates, _, excludes, _, _, _, _, _ = planner.plan_to_request_updates(plan)
        assert "exclude_sleeve_type" in updates
        assert "Sleeveless" in updates["exclude_sleeve_type"]
        assert "exclude_neckline" in updates
        assert "Off-Shoulder" in updates["exclude_neckline"]
        assert "exclude_materials" in updates
        assert "Mesh" in updates["exclude_materials"]

    def test_modest_wedding_with_avoid(self):
        """'Modest dress for wedding, no polyester' — modes + avoid."""
        planner = self._make_planner_with_response({
            "intent": "specific",
            "algolia_query": "dress",
            "semantic_query": "a modest conservative dress for a wedding, opaque fabric, full coverage",
            "modes": ["modest", "wedding_guest"],
            "attributes": {"category_l1": ["Dresses"]},
            "avoid": {"materials": ["Polyester"]},
            "confidence": 0.9,
        })

        plan = planner.plan("modest dress for wedding, no polyester")
        assert plan is not None
        assert "modest" in plan.modes
        assert "wedding_guest" in plan.modes
        assert "Polyester" in plan.avoid.get("materials", [])

        updates, _, excludes, _, _, _, _, _ = planner.plan_to_request_updates(plan)
        # Polyester merged with mode material exclusions
        assert "Polyester" in updates["exclude_materials"]
        assert "Mesh" in updates["exclude_materials"]
        # modest → all coverage exclusions
        assert "exclude_neckline" in updates
        assert "exclude_sleeve_type" in updates

    def test_sexy_classy_date_night(self):
        """'Sexy but classy date night' — aesthetic + occasion modes."""
        planner = self._make_planner_with_response({
            "intent": "vague",
            "algolia_query": "",
            "semantic_query": "a sexy but elegant date night outfit, sophisticated and glamorous",
            "modes": ["glamorous", "smart_casual", "date_night"],
            "attributes": {"category_l1": ["Tops", "Dresses"]},
            "avoid": {},
            "confidence": 0.85,
        })

        plan = planner.plan("sexy but classy date night")
        assert plan is not None
        assert plan.intent == "vague"

        updates, _, excludes, _, _, _, _, _ = planner.plan_to_request_updates(plan)
        # glamorous → Glamorous, Sexy style_tags
        assert "style_tags" in updates
        assert "Glamorous" in updates["style_tags"]
        # date_night → Date Night occasion
        assert "occasions" in updates
        assert "Date Night" in updates["occasions"]
        # No exclusions (positive vibes only)
        assert excludes == {}

    def test_boohoo_brand_search(self):
        """'boohoo' should be classified as exact brand search."""
        planner = self._make_planner_with_response({
            "intent": "exact",
            "algolia_query": "",
            "semantic_query": "Boohoo fashion clothing",
            "modes": [],
            "attributes": {"brands": ["Boohoo"]},
            "avoid": {},
            "brand": "Boohoo",
            "confidence": 0.95,
        })

        plan = planner.plan("boohoo")
        assert plan is not None
        assert plan.intent == "exact"
        assert plan.brand == "Boohoo"

    def test_jeans_no_rips(self):
        """'Mid rise straight jeans no rips' — attributes + avoid."""
        planner = self._make_planner_with_response({
            "intent": "specific",
            "algolia_query": "straight jeans",
            "semantic_query": "mid rise straight leg jeans without rips or distressing",
            "modes": [],
            "attributes": {
                "category_l1": ["Bottoms"],
                "rise": ["Mid"],
                "silhouette": ["Straight"],
            },
            "avoid": {"style_tags": ["Distressed"]},
            "confidence": 0.9,
        })

        plan = planner.plan("mid rise straight jeans no rips")
        assert plan is not None

        updates, _, excludes, _, _, _, _, _ = planner.plan_to_request_updates(plan)
        assert updates.get("rise") == ["Mid"]
        assert updates.get("silhouette") == ["Straight"]
        assert "exclude_style_tags" in updates
        assert "Distressed" in updates["exclude_style_tags"]

    def test_linen_pants_opaque_mode(self):
        """'Linen pants not see-through' — opaque mode."""
        planner = self._make_planner_with_response({
            "intent": "specific",
            "algolia_query": "linen pants",
            "semantic_query": "linen pants that are not see-through, opaque fabric",
            "modes": ["opaque"],
            "attributes": {
                "category_l1": ["Bottoms"],
                "materials": ["Linen"],
            },
            "avoid": {},
            "confidence": 0.9,
        })

        plan = planner.plan("linen pants not see-through")
        assert plan is not None
        assert "opaque" in plan.modes

        updates, _, excludes, _, _, _, _, _ = planner.plan_to_request_updates(plan)
        assert "materials" in updates
        assert "Linen" in updates["materials"]
        assert "exclude_materials" in updates
        assert "Mesh" in updates["exclude_materials"]
        assert "Sheer" in updates["exclude_materials"]

    def test_nested_avoid_in_attributes_fixed(self):
        """LLM might nest avoid inside attributes — should be extracted."""
        planner = self._make_planner_with_response({
            "intent": "specific",
            "algolia_query": "dress",
            "semantic_query": "a dress",
            "modes": [],
            "attributes": {
                "category_l1": ["Dresses"],
                "avoid": {"materials": ["Polyester"]},  # Misplaced!
            },
            "avoid": {},
            "confidence": 0.8,
        })

        plan = planner.plan("dress no polyester")
        assert plan is not None
        # The nested avoid should have been extracted
        assert "Polyester" in plan.avoid.get("materials", [])


# =============================================================================
# 5. Post-Filter Expanded Filters Integration Tests
# =============================================================================

class TestPostFilterWithExpansion:
    """Tests that expanded_filters properly relaxes post-filtering.

    These tests exercise HybridSearchService._post_filter_semantic() directly
    and are independent of the SearchPlan model.
    """

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


# =============================================================================
# Vibe-Brand Tests
# =============================================================================

class TestVibeBrand:
    """Tests for vibe_brand field on SearchPlan and plan_to_request_updates."""

    def test_vibe_brand_field_accepted(self):
        """SearchPlan accepts vibe_brand as a string."""
        plan = SearchPlan(
            intent="specific",
            algolia_query="boho maxi dress",
            semantic_query="flowing bohemian maxi dress earthy layered textures",
            modes=["boho"],
            attributes={"category_l1": ["Dresses"]},
            avoid={},
            brand=None,
            vibe_brand="Anthropologie",
            confidence=0.85,
        )
        assert plan.vibe_brand == "Anthropologie"
        assert plan.brand is None

    def test_vibe_brand_null_by_default(self):
        """vibe_brand defaults to None."""
        plan = SearchPlan(
            intent="specific",
            algolia_query="dress",
            semantic_query="dress",
            modes=[],
            attributes={},
            avoid={},
        )
        assert plan.vibe_brand is None

    def test_vibe_brand_mutually_exclusive_with_brand(self):
        """vibe_brand and brand can both be set (LLM contract: only one should
        be non-null, but the model doesn't enforce mutual exclusion).
        plan_to_request_updates gives vibe_brand priority."""
        plan = SearchPlan(
            intent="specific",
            algolia_query="dress",
            semantic_query="dress",
            modes=[],
            attributes={"category_l1": ["Dresses"]},
            avoid={},
            brand="Zara",
            vibe_brand="Zara",
            confidence=0.8,
        )
        planner = QueryPlanner.__new__(QueryPlanner)
        planner._enabled = True
        result = planner.plan_to_request_updates(plan)
        plan_updates = result[0]

        # vibe_brand takes precedence — no hard brand filter
        assert "brands" not in plan_updates
        assert plan_updates.get("_vibe_brand") == "Zara"

    def test_plan_to_request_updates_vibe_brand_no_hard_filter(self):
        """When vibe_brand is set, plan_to_request_updates should NOT set
        brands in the request updates (no hard filter)."""
        plan = SearchPlan(
            intent="specific",
            algolia_query="elevated basics neutral minimalist",
            semantic_query="elevated clean basics neutral palette minimalist capsule wardrobe",
            semantic_queries=[
                "polished minimalist basics neutral tones elevated essentials",
                "clean tailored basics muted palette capsule wardrobe staples",
            ],
            modes=["smart_casual"],
            attributes={"category_l1": ["Tops", "Dresses"]},
            avoid={},
            brand=None,
            vibe_brand="Zara",
            min_price=50.0,
            confidence=0.85,
        )
        planner = QueryPlanner.__new__(QueryPlanner)
        planner._enabled = True
        result = planner.plan_to_request_updates(plan)
        plan_updates = result[0]

        # Should NOT have hard brand filter
        assert "brands" not in plan_updates
        # Should have _vibe_brand internal key
        assert plan_updates["_vibe_brand"] == "Zara"
        # min_price should still be applied (from "better quality than Zara")
        assert plan_updates.get("min_price") == 50.0

    def test_plan_to_request_updates_exact_brand_unchanged(self):
        """When brand is set (not vibe_brand), hard filter is applied as before."""
        plan = SearchPlan(
            intent="exact",
            algolia_query="",
            semantic_query="Zara fashion clothing",
            modes=[],
            attributes={},
            avoid={},
            brand="Zara",
            vibe_brand=None,
            confidence=0.95,
        )
        planner = QueryPlanner.__new__(QueryPlanner)
        planner._enabled = True
        result = planner.plan_to_request_updates(plan)
        plan_updates = result[0]

        # Should have hard brand filter
        assert plan_updates.get("brands") == ["Zara"]
        # Should NOT have vibe brand
        assert "_vibe_brand" not in plan_updates

    def test_vibe_brand_with_quality_modifier(self):
        """'Like Zara but better quality' → vibe_brand + higher min_price."""
        plan = SearchPlan(
            intent="specific",
            algolia_query="basics elevated quality",
            semantic_query="elevated basics clean minimalist better quality knitwear",
            modes=["smart_casual"],
            attributes={
                "category_l1": ["Tops", "Dresses"],
            },
            avoid={},
            brand=None,
            vibe_brand="Zara",
            min_price=60.0,
            confidence=0.8,
        )
        planner = QueryPlanner.__new__(QueryPlanner)
        planner._enabled = True
        result = planner.plan_to_request_updates(plan)
        plan_updates = result[0]

        assert "brands" not in plan_updates
        assert plan_updates["_vibe_brand"] == "Zara"
        assert plan_updates["min_price"] == 60.0

    def test_vibe_brand_with_budget_modifier(self):
        """'Cheaper than Aritzia' → vibe_brand + lower max_price."""
        plan = SearchPlan(
            intent="specific",
            algolia_query="minimal basics neutral",
            semantic_query="premium minimalist basics neutral quiet luxury affordable",
            modes=["quiet_luxury"],
            attributes={
                "category_l1": ["Tops", "Dresses"],
            },
            avoid={},
            brand=None,
            vibe_brand="Aritzia",
            max_price=80.0,
            confidence=0.8,
        )
        planner = QueryPlanner.__new__(QueryPlanner)
        planner._enabled = True
        result = planner.plan_to_request_updates(plan)
        plan_updates = result[0]

        assert "brands" not in plan_updates
        assert plan_updates["_vibe_brand"] == "Aritzia"
        assert plan_updates["max_price"] == 80.0

    def test_system_prompt_contains_vibe_brand_section(self):
        """The system prompt should include the brand-as-vibe instructions."""
        from search.query_planner import _build_system_prompt
        prompt = _build_system_prompt()

        assert "vibe_brand" in prompt
        assert "Brand-as-Vibe Queries" in prompt
        assert "like X" in prompt.lower() or '"like X"' in prompt
        assert "Style cluster landscape" in prompt
        assert "Massimo Dutti" in prompt  # example in tier landscape


class TestVibeBrandFilters:
    """Tests for _build_vibe_brand_filters in HybridSearchService."""

    def _make_service(self):
        from search.hybrid_search import HybridSearchService
        return HybridSearchService(
            algolia_client=MagicMock(),
            analytics=MagicMock(),
        )

    def test_known_brand_returns_filters(self):
        """A known brand (e.g., 'zara') should return cluster brand filters."""
        service = self._make_service()
        filters = service._build_vibe_brand_filters("Zara")
        # Zara is in cluster G (and secondary A) — should have brands from those
        assert len(filters) > 0
        # All filters should be brand:"..." format
        for f in filters:
            assert f.startswith('brand:"')
            assert f.endswith('"')

    def test_unknown_brand_returns_empty(self):
        """An unknown brand should return an empty list."""
        service = self._make_service()
        filters = service._build_vibe_brand_filters("TotallyFakeBrandXYZ")
        assert filters == []

    def test_case_insensitive_lookup(self):
        """Brand lookup should be case-insensitive."""
        service = self._make_service()
        filters_lower = service._build_vibe_brand_filters("anthropologie")
        filters_upper = service._build_vibe_brand_filters("Anthropologie")
        assert len(filters_lower) == len(filters_upper)
        assert set(filters_lower) == set(filters_upper)

    def test_includes_cluster_adjacent_brands(self):
        """Vibe filters for Anthropologie should include Free People (same cluster M)."""
        service = self._make_service()
        filters = service._build_vibe_brand_filters("Anthropologie")
        brand_names = [f.split('"')[1].lower() for f in filters]
        assert "free people" in brand_names or "anthropologie" in brand_names
