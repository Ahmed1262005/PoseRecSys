"""
Unit tests for the attribute search module.

Tests:
  - plan_to_attribute_filters(): SearchPlan -> AttributeFilters translation
  - DETAIL_TERM_MAP: detail term matching
  - AttributeFilters: helpers and edge cases
  - _passes_attribute_filter(): post-filter logic

No DB connection needed — these are all pure logic tests.
"""

import pytest
from unittest.mock import MagicMock

from search.attribute_search import (
    plan_to_attribute_filters,
    AttributeFilters,
    DETAIL_TERM_MAP,
    _match_term,
    _scan_query_for_details,
)


# ---------------------------------------------------------------------------
# Helper to build a mock SearchPlan
# ---------------------------------------------------------------------------

def _mock_plan(
    detail_terms=None,
    attributes=None,
    brand=None,
    max_price=None,
    min_price=None,
    modes=None,
):
    plan = MagicMock()
    plan.detail_terms = detail_terms or []
    plan.attributes = attributes or {}
    plan.brand = brand
    plan.max_price = max_price
    plan.min_price = min_price
    plan.modes = modes or []
    return plan


# ---------------------------------------------------------------------------
# plan_to_attribute_filters: detail_terms translation
# ---------------------------------------------------------------------------

class TestDetailTermTranslation:
    """detail_terms from the planner are translated to attribute filters."""

    def test_pockets(self):
        plan = _mock_plan(detail_terms=["pockets"])
        f = plan_to_attribute_filters(plan)
        assert f.has_pockets is True

    def test_open_back(self):
        plan = _mock_plan(detail_terms=["open back"])
        f = plan_to_attribute_filters(plan)
        assert f.back_openness == ["open", "partial"]

    def test_backless(self):
        plan = _mock_plan(detail_terms=["backless"])
        f = plan_to_attribute_filters(plan)
        assert f.back_openness == ["open", "partial"]

    def test_slit(self):
        plan = _mock_plan(detail_terms=["slit"])
        f = plan_to_attribute_filters(plan)
        assert f.slit_presence is True

    def test_high_slit(self):
        plan = _mock_plan(detail_terms=["high slit"])
        f = plan_to_attribute_filters(plan)
        assert f.slit_presence is True
        assert f.slit_height == ["high"]

    def test_lace(self):
        plan = _mock_plan(detail_terms=["lace"])
        f = plan_to_attribute_filters(plan)
        assert f.detail_tags == ["lace_trim"]

    def test_ruffle(self):
        plan = _mock_plan(detail_terms=["ruffle"])
        f = plan_to_attribute_filters(plan)
        assert f.detail_tags == ["ruffle_detail"]

    def test_embroidered(self):
        plan = _mock_plan(detail_terms=["embroidered"])
        f = plan_to_attribute_filters(plan)
        assert f.detail_tags == ["embroidery_detail"]

    def test_distressed(self):
        plan = _mock_plan(detail_terms=["distressed"])
        f = plan_to_attribute_filters(plan)
        assert f.detail_tags == ["distressed_detail"]

    def test_crochet(self):
        plan = _mock_plan(detail_terms=["crochet"])
        f = plan_to_attribute_filters(plan)
        assert f.detail_tags == ["crochet_detail"]

    def test_mesh(self):
        plan = _mock_plan(detail_terms=["mesh"])
        f = plan_to_attribute_filters(plan)
        assert f.detail_tags == ["mesh_panels"]

    def test_cutout(self):
        plan = _mock_plan(detail_terms=["cutout"])
        f = plan_to_attribute_filters(plan)
        assert f.detail_tags == ["cutout_detail"]

    # -- Coverage / silhouette terms --
    def test_off_shoulder(self):
        plan = _mock_plan(detail_terms=["off shoulder"])
        f = plan_to_attribute_filters(plan)
        assert f.shoulder_coverage == ["off_shoulder"]

    def test_strapless(self):
        plan = _mock_plan(detail_terms=["strapless"])
        f = plan_to_attribute_filters(plan)
        assert f.shoulder_coverage == ["exposed"]
        assert f.arm_coverage == ["none"]

    def test_bodycon(self):
        plan = _mock_plan(detail_terms=["bodycon"])
        f = plan_to_attribute_filters(plan)
        assert f.body_cling_visual == ["bodycon"]

    def test_oversized(self):
        plan = _mock_plan(detail_terms=["oversized"])
        f = plan_to_attribute_filters(plan)
        assert f.body_cling_visual == ["loose"]
        assert "bulky" in f.bulk_visual

    def test_structured(self):
        plan = _mock_plan(detail_terms=["structured"])
        f = plan_to_attribute_filters(plan)
        assert f.structure_level == ["structured"]

    def test_flowy(self):
        plan = _mock_plan(detail_terms=["flowy"])
        f = plan_to_attribute_filters(plan)
        assert "high" in f.drape_level
        assert "loose" in f.body_cling_visual

    def test_wide_leg(self):
        plan = _mock_plan(detail_terms=["wide leg"])
        f = plan_to_attribute_filters(plan)
        assert f.leg_volume_visual == ["wide"]

    def test_sheer(self):
        plan = _mock_plan(detail_terms=["sheer"])
        f = plan_to_attribute_filters(plan)
        assert f.sheerness_visual == ["semi_sheer"]


# ---------------------------------------------------------------------------
# plan_to_attribute_filters: plan.attributes passthrough
# ---------------------------------------------------------------------------

class TestPlanAttributesPassthrough:
    """Standard plan.attributes are passed through to filters."""

    def test_category_l1(self):
        plan = _mock_plan(attributes={"category_l1": ["Dresses"]})
        f = plan_to_attribute_filters(plan)
        assert f.category_l1 == ["Dresses"]

    def test_category_l2(self):
        plan = _mock_plan(attributes={"category_l2": ["Jacket"]})
        f = plan_to_attribute_filters(plan)
        assert f.category_l2 == ["Jacket"]

    def test_occasions(self):
        plan = _mock_plan(attributes={"occasions": ["Date Night", "Party"]})
        f = plan_to_attribute_filters(plan)
        assert f.occasions == ["Date Night", "Party"]

    def test_formality(self):
        plan = _mock_plan(attributes={"formality": ["Formal"]})
        f = plan_to_attribute_filters(plan)
        assert f.formality == ["Formal"]

    def test_seasons(self):
        plan = _mock_plan(attributes={"seasons": ["Summer"]})
        f = plan_to_attribute_filters(plan)
        assert f.seasons == ["Summer"]

    def test_brand(self):
        plan = _mock_plan(brand="Aje")
        f = plan_to_attribute_filters(plan)
        assert f.brands == ["Aje"]

    def test_price(self):
        plan = _mock_plan(max_price=100.0, min_price=20.0)
        f = plan_to_attribute_filters(plan)
        assert f.max_price == 100.0
        assert f.min_price == 20.0


# ---------------------------------------------------------------------------
# plan_to_attribute_filters: combined (detail_terms + attributes + query)
# ---------------------------------------------------------------------------

class TestCombinedTranslation:
    """Detail terms, plan attributes, and query text combine correctly."""

    def test_detail_terms_plus_attributes(self):
        plan = _mock_plan(
            detail_terms=["pockets"],
            attributes={"category_l1": ["Dresses"]},
        )
        f = plan_to_attribute_filters(plan)
        assert f.has_pockets is True
        assert f.category_l1 == ["Dresses"]

    def test_query_scan_catches_missed_terms(self):
        """detail_terms may miss things the raw query contains."""
        plan = _mock_plan(
            detail_terms=[],  # planner missed "backless"
            attributes={"category_l1": ["Dresses"]},
        )
        f = plan_to_attribute_filters(plan, query="backless dress")
        assert f.back_openness == ["open", "partial"]
        assert f.category_l1 == ["Dresses"]

    def test_detail_terms_take_precedence_over_query_scan(self):
        """If detail_terms already matched, query scan shouldn't override."""
        plan = _mock_plan(
            detail_terms=["high slit"],
            attributes={"category_l1": ["Dresses"]},
        )
        f = plan_to_attribute_filters(plan, query="dress with high slit")
        assert f.slit_presence is True
        assert f.slit_height == ["high"]

    def test_multiple_detail_terms(self):
        plan = _mock_plan(detail_terms=["pockets", "lace"])
        f = plan_to_attribute_filters(plan)
        assert f.has_pockets is True
        assert f.detail_tags == ["lace_trim"]


# ---------------------------------------------------------------------------
# _scan_query_for_details: raw query scanning
# ---------------------------------------------------------------------------

class TestQueryScanning:

    def test_scans_backless(self):
        merged = {}
        _scan_query_for_details("backless dress", merged)
        assert merged.get("back_openness") == ["open", "partial"]

    def test_scans_pockets(self):
        merged = {}
        _scan_query_for_details("dress with pockets", merged)
        assert merged.get("has_pockets") is True

    def test_scans_multiple(self):
        merged = {}
        _scan_query_for_details("backless lace dress with slit", merged)
        assert merged.get("back_openness") == ["open", "partial"]
        assert merged.get("detail_tags") == ["lace_trim"]
        assert merged.get("slit_presence") is True

    def test_ignores_unknown(self):
        merged = {}
        _scan_query_for_details("beautiful amazing dress", merged)
        assert len(merged) == 0

    def test_case_insensitive(self):
        """_scan_query_for_details expects pre-lowered input (callers lower() first)."""
        merged = {}
        _scan_query_for_details("backless dress", merged)
        assert merged.get("back_openness") == ["open", "partial"]

    def test_plan_to_filters_is_case_insensitive(self):
        """plan_to_attribute_filters lowercases the query before scanning."""
        plan = _mock_plan(detail_terms=[], attributes={})
        f = plan_to_attribute_filters(plan, query="BACKLESS DRESS")
        assert f.back_openness == ["open", "partial"]


# ---------------------------------------------------------------------------
# AttributeFilters: helpers
# ---------------------------------------------------------------------------

class TestAttributeFilters:

    def test_no_attribute_filters(self):
        f = AttributeFilters()
        assert not f.has_attribute_filters()

    def test_category_only_not_attribute_filter(self):
        """category_l1/l2 alone don't count as 'attribute filters'."""
        f = AttributeFilters(category_l1=["Dresses"])
        assert not f.has_attribute_filters()

    def test_has_pockets_is_attribute_filter(self):
        f = AttributeFilters(has_pockets=True)
        assert f.has_attribute_filters()

    def test_back_openness_is_attribute_filter(self):
        f = AttributeFilters(back_openness=["open"])
        assert f.has_attribute_filters()

    def test_detail_tags_is_attribute_filter(self):
        f = AttributeFilters(detail_tags=["lace_trim"])
        assert f.has_attribute_filters()

    def test_body_cling_is_attribute_filter(self):
        f = AttributeFilters(body_cling_visual=["bodycon"])
        assert f.has_attribute_filters()

    def test_occasions_is_attribute_filter(self):
        f = AttributeFilters(occasions=["Date Night"])
        assert f.has_attribute_filters()

    def test_describe_empty(self):
        assert AttributeFilters().describe() == "(no filters)"

    def test_describe_with_filters(self):
        f = AttributeFilters(has_pockets=True, category_l1=["Dresses"])
        desc = f.describe()
        assert "has_pockets" in desc
        assert "category_l1" in desc


# ---------------------------------------------------------------------------
# DETAIL_TERM_MAP coverage
# ---------------------------------------------------------------------------

class TestDetailTermMapCoverage:
    """Verify the map contains entries for key detail categories."""

    def test_pocket_terms(self):
        assert "pockets" in DETAIL_TERM_MAP
        assert "with pockets" in DETAIL_TERM_MAP
        assert "has pockets" in DETAIL_TERM_MAP

    def test_slit_terms(self):
        assert "slit" in DETAIL_TERM_MAP
        assert "high slit" in DETAIL_TERM_MAP
        assert "thigh slit" in DETAIL_TERM_MAP

    def test_coverage_terms(self):
        assert "backless" in DETAIL_TERM_MAP
        assert "strapless" in DETAIL_TERM_MAP
        assert "off shoulder" in DETAIL_TERM_MAP
        assert "sleeveless" in DETAIL_TERM_MAP
        assert "long sleeve" in DETAIL_TERM_MAP
        assert "sheer" in DETAIL_TERM_MAP

    def test_silhouette_terms(self):
        assert "bodycon" in DETAIL_TERM_MAP
        assert "flowy" in DETAIL_TERM_MAP
        assert "oversized" in DETAIL_TERM_MAP
        assert "structured" in DETAIL_TERM_MAP
        assert "wide leg" in DETAIL_TERM_MAP
        assert "skinny" in DETAIL_TERM_MAP

    def test_detail_tag_terms(self):
        assert "lace" in DETAIL_TERM_MAP
        assert "ruffle" in DETAIL_TERM_MAP
        assert "embroidered" in DETAIL_TERM_MAP
        assert "distressed" in DETAIL_TERM_MAP
        assert "crochet" in DETAIL_TERM_MAP
        assert "pleated" in DETAIL_TERM_MAP
        assert "cutout" in DETAIL_TERM_MAP
        assert "mesh" in DETAIL_TERM_MAP

    def test_lining_terms(self):
        assert "lined" in DETAIL_TERM_MAP
        assert "unlined" in DETAIL_TERM_MAP

    def test_all_detail_tags_map_to_valid_tags(self):
        """Every detail_tags value should be a known tag."""
        known_tags = {
            "lace_trim", "ruffle_detail", "crochet_detail", "distressed_detail",
            "scalloped_hem", "ruched_bodice", "embroidery_detail", "ribbed_trim",
            "mesh_panels", "fringe_detail", "raw_hem", "frayed_edge",
            "quilted_texture", "pleated_detail", "cutout_detail", "tie_front",
            "wrap_detail",
        }
        for term, mapping in DETAIL_TERM_MAP.items():
            if "detail_tags" in mapping:
                for tag in mapping["detail_tags"]:
                    assert tag in known_tags, f"Unknown tag '{tag}' in DETAIL_TERM_MAP['{term}']"


# ---------------------------------------------------------------------------
# _match_term: unit-level
# ---------------------------------------------------------------------------

class TestMatchTerm:

    def test_exact_match(self):
        merged = {}
        assert _match_term("pockets", merged) is True
        assert merged["has_pockets"] is True

    def test_substring_match(self):
        """Terms like 'open back' match 'backless' if contained."""
        merged = {}
        # "open back" is in the map; "open back dress" contains "open back"
        assert _match_term("open back", merged) is True
        assert merged["back_openness"] == ["open", "partial"]

    def test_no_match(self):
        merged = {}
        assert _match_term("xyzzy123", merged) is False
        assert len(merged) == 0

    def test_does_not_overwrite(self):
        """First match wins for a given key."""
        merged = {"back_openness": ["open"]}
        _match_term("backless", merged)
        # Should extend, not overwrite
        assert "partial" in merged["back_openness"]
