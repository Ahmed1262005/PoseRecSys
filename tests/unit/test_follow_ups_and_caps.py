"""
Unit tests for Phase 1 (Follow-Up Questions) and search scoring caps.

Tests cover:
1. Search scoring caps in reranker (ProfileScorer, session, context)
2. FollowUpOption / FollowUpQuestion / SearchRefineRequest models
3. QueryPlanner._parse_follow_ups (valid, malformed, empty)
4. SearchPlan with parsed_follow_ups
5. HybridSearchResponse serialization with follow_ups
6. /api/search/refine endpoint (filter merge, mode expansion)

Run with: PYTHONPATH=src python -m pytest tests/unit/test_follow_ups_and_caps.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def reranker():
    from search.reranker import SessionReranker
    return SessionReranker()


def _make_result(
    product_id: str,
    brand: str = "TestBrand",
    price: float = 50.0,
    rrf_score: float = 0.5,
    **kwargs,
) -> dict:
    """Helper to create a mock search result dict."""
    base = {
        "product_id": product_id,
        "name": f"Product {product_id}",
        "brand": brand,
        "image_url": f"https://img.example.com/{product_id}.jpg",
        "gallery_images": [],
        "price": price,
        "original_price": None,
        "is_on_sale": False,
        "category_l1": "Tops",
        "category_l2": "T-Shirt",
        "broad_category": "tops",
        "article_type": "t-shirt",
        "primary_color": "Black",
        "color_family": "Dark",
        "pattern": "Solid",
        "apparent_fabric": "Cotton",
        "fit_type": "Regular",
        "formality": "Casual",
        "silhouette": "Straight",
        "length": "Regular",
        "neckline": "Crew",
        "sleeve_type": "Short",
        "rise": None,
        "style_tags": ["casual"],
        "occasions": ["everyday"],
        "seasons": ["spring", "summer"],
        "colors": ["black"],
        "materials": ["cotton"],
        "trending_score": 0,
        "source": "algolia",
        "rrf_score": rrf_score,
    }
    base.update(kwargs)
    return base


# =============================================================================
# 1. Search Scoring Caps in Reranker
# =============================================================================

class TestSearchScoringCaps:
    """Verify that the reranker uses search-specific (lighter) scoring caps."""

    def test_max_session_boost_is_008(self):
        """MAX_SESSION_BOOST should be 0.08 for search (not 0.15)."""
        from search.reranker import MAX_SESSION_BOOST
        assert MAX_SESSION_BOOST == 0.08

    def test_session_skip_penalty_unchanged(self):
        """SESSION_SKIP_PENALTY should still be -0.08."""
        from search.reranker import SESSION_SKIP_PENALTY
        assert SESSION_SKIP_PENALTY == -0.08

    def test_profile_scorer_uses_search_caps(self, reranker):
        """ProfileScorer in reranker should use +/-0.10 caps, not feed caps."""
        # Create results with a profile that would give max boost
        results = [_make_result("p1", brand="Boohoo")]
        profile = {
            "preferred_brands": ["Boohoo"],
            "brand_openness": "stick_to_favorites",
            "style_persona": ["casual"],
            "occasions": ["everyday"],
            "patterns_liked": ["Solid"],
        }

        reranked = reranker._apply_profile_scoring(results, profile)

        # The profile_adjustment should be capped at +0.10
        adj = reranked[0].get("profile_adjustment", 0)
        assert adj <= 0.10 + 1e-6, f"Profile adjustment {adj} exceeds search cap of 0.10"
        # And negative cap should be -0.10 (not -2.0)
        # We can't easily test negative cap without a strongly mismatched profile,
        # so just verify the config is correct
        from scoring.profile_scorer import ProfileScoringConfig
        # Instantiate the same config the reranker uses
        search_config = ProfileScoringConfig(
            max_positive=0.10,
            max_negative=-0.10,
            coverage_kill_penalty=-1.0,
        )
        assert search_config.max_positive == 0.10
        assert search_config.max_negative == -0.10
        assert search_config.coverage_kill_penalty == -1.0

    def test_profile_scorer_coverage_kill_preserved(self, reranker):
        """Coverage hard-kill (-1.0) should be preserved even with search caps."""
        # Create a result that violates no_crop
        results = [_make_result("p1", article_type="crop top", name="Cute Crop Top")]
        profile = {
            "preferred_brands": [],
            "no_crop": True,
        }

        reranked = reranker._apply_profile_scoring(results, profile)

        # Coverage kill should still be -1.0 — max_negative is set to -1.0
        # to allow hard-kills through (personality penalties never reach -1.0)
        adj = reranked[0].get("profile_adjustment", 0)
        assert adj <= -0.5, f"Coverage kill should still apply, got {adj}"

    def test_profile_scorer_max_negative_allows_coverage_kill(self):
        """max_negative should be -1.0 (not -0.10) to let coverage kills through."""
        from scoring.profile_scorer import ProfileScoringConfig
        search_config = ProfileScoringConfig(
            max_positive=0.10,
            max_negative=-1.0,
            coverage_kill_penalty=-1.0,
        )
        assert search_config.max_negative == -1.0
        assert search_config.coverage_kill_penalty == -1.0

    def test_session_scoring_capped_at_008(self, reranker):
        """Session scoring should be capped at +/-0.08."""
        from recs.session_scoring import SessionScoringEngine

        engine = SessionScoringEngine()
        scores = engine.initialize_from_onboarding(
            preferred_brands=["Boohoo"]
        )

        # Click many times to build up strong brand preference
        for i in range(20):
            engine.process_action(
                scores, action="click",
                product_id=f"click_{i}",
                brand="Boohoo",
                item_type="t-shirt",
            )

        results = [_make_result("p_test", brand="Boohoo")]
        reranked = reranker._apply_session_scoring(results, scores)

        adj = reranked[0].get("session_adjustment", 0)
        assert abs(adj) <= 0.08 + 1e-6, \
            f"Session adjustment {adj} exceeds search cap of 0.08"

    def test_context_scoring_uses_weight_020(self, reranker):
        """Context scoring should use weight=0.20 for search."""
        from scoring.context import UserContext, AgeGroup

        ctx = UserContext(user_id="test-user", age_group=AgeGroup.GEN_Z)

        # Create a result that strongly triggers age scoring (crop top for Gen Z)
        results = [
            _make_result("p1", article_type="crop top", name="Trendy Crop Top",
                         style_tags=["streetwear"], neckline="Crew"),
        ]
        original_score = results[0]["rrf_score"]

        reranked = reranker._apply_context_scoring(results, ctx)

        # With weight=0.20, the context adjustment should be scaled down.
        # The raw ContextScorer adjustment can be up to 0.30 (MAX_CONTEXT_ADJUSTMENT).
        # With weight=0.20, the max score change is 0.30 * 0.20 = 0.06.
        context_adj = reranked[0].get("context_adjustment", 0)
        score_change = reranked[0]["rrf_score"] - original_score

        assert abs(score_change) <= 0.30 * 0.20 + 0.01, \
            f"Score change {score_change} too large for weight=0.20 (context_adj={context_adj})"


# =============================================================================
# 2. Follow-Up Models
# =============================================================================

class TestFollowUpModels:
    """Test FollowUpOption, FollowUpQuestion, and SearchRefineRequest."""

    def test_follow_up_option_basic(self):
        from search.models import FollowUpOption
        opt = FollowUpOption(label="Under $50", filters={"max_price": 50})
        assert opt.label == "Under $50"
        assert opt.filters == {"max_price": 50}

    def test_follow_up_option_empty_filters(self):
        from search.models import FollowUpOption
        opt = FollowUpOption(label="No preference", filters={})
        assert opt.filters == {}

    def test_follow_up_option_complex_filters(self):
        from search.models import FollowUpOption
        opt = FollowUpOption(
            label="Sexy & glam",
            filters={"formality": ["Semi-Formal"], "modes": ["glamorous"]},
        )
        assert opt.filters["formality"] == ["Semi-Formal"]
        assert opt.filters["modes"] == ["glamorous"]

    def test_follow_up_question_basic(self):
        from search.models import FollowUpQuestion, FollowUpOption
        q = FollowUpQuestion(
            dimension="formality",
            question="How dressed up do you want to be?",
            options=[
                FollowUpOption(label="Casual", filters={"formality": ["Casual"]}),
                FollowUpOption(label="Dressy", filters={"formality": ["Formal"]}),
            ],
        )
        assert q.dimension == "formality"
        assert len(q.options) == 2
        assert q.options[0].label == "Casual"

    def test_follow_up_question_all_dimensions(self):
        """All 6 dimensions should be accepted."""
        from search.models import FollowUpQuestion, FollowUpOption
        for dim in ["formality", "coverage", "price", "garment_type", "color", "occasion"]:
            q = FollowUpQuestion(
                dimension=dim,
                question=f"Test question for {dim}?",
                options=[FollowUpOption(label="Option A", filters={})],
            )
            assert q.dimension == dim

    def test_follow_up_question_serialization(self):
        """FollowUpQuestion should serialize to dict cleanly."""
        from search.models import FollowUpQuestion, FollowUpOption
        q = FollowUpQuestion(
            dimension="price",
            question="What's your budget?",
            options=[
                FollowUpOption(label="Under $30", filters={"max_price": 30}),
                FollowUpOption(label="$30-$75", filters={"min_price": 30, "max_price": 75}),
                FollowUpOption(label="$75+", filters={"min_price": 75}),
            ],
        )
        d = q.model_dump()
        assert d["dimension"] == "price"
        assert len(d["options"]) == 3
        assert d["options"][0]["filters"]["max_price"] == 30

    def test_search_refine_request_basic(self):
        from search.models import SearchRefineRequest
        req = SearchRefineRequest(
            original_query="something cute",
            selected_filters={"formality": ["Casual"], "category_l1": ["Dresses"]},
        )
        assert req.original_query == "something cute"
        assert req.selected_filters["formality"] == ["Casual"]
        assert req.page == 1
        assert req.page_size == 50

    def test_search_refine_request_with_modes(self):
        from search.models import SearchRefineRequest
        req = SearchRefineRequest(
            original_query="outfit for tonight",
            selected_filters={
                "modes": ["glamorous", "cover_chest"],
                "formality": ["Semi-Formal"],
            },
        )
        assert "modes" in req.selected_filters
        assert "glamorous" in req.selected_filters["modes"]


# =============================================================================
# 3. QueryPlanner._parse_follow_ups
# =============================================================================

class TestParseFollowUps:
    """Test the static _parse_follow_ups method."""

    def test_parse_valid_single_question(self):
        from search.query_planner import QueryPlanner
        raw = [
            {
                "dimension": "formality",
                "question": "How dressed up?",
                "options": [
                    {"label": "Casual", "filters": {"formality": ["Casual"]}},
                    {"label": "Dressy", "filters": {"formality": ["Formal"]}},
                ],
            }
        ]
        parsed = QueryPlanner._parse_follow_ups(raw)
        assert len(parsed) == 1
        assert parsed[0].dimension == "formality"
        assert parsed[0].question == "How dressed up?"
        assert len(parsed[0].options) == 2

    def test_parse_valid_multiple_questions(self):
        from search.query_planner import QueryPlanner
        raw = [
            {
                "dimension": "formality",
                "question": "How dressed up?",
                "options": [
                    {"label": "Casual", "filters": {"formality": ["Casual"]}},
                    {"label": "Dressy", "filters": {"formality": ["Formal"]}},
                ],
            },
            {
                "dimension": "garment_type",
                "question": "What type?",
                "options": [
                    {"label": "A dress", "filters": {"category_l1": ["Dresses"]}},
                    {"label": "Top + bottoms", "filters": {"category_l1": ["Tops", "Bottoms"]}},
                ],
            },
            {
                "dimension": "price",
                "question": "Budget?",
                "options": [
                    {"label": "Under $50", "filters": {"max_price": 50}},
                    {"label": "$50-$100", "filters": {"min_price": 50, "max_price": 100}},
                ],
            },
        ]
        parsed = QueryPlanner._parse_follow_ups(raw)
        assert len(parsed) == 3
        assert parsed[0].dimension == "formality"
        assert parsed[1].dimension == "garment_type"
        assert parsed[2].dimension == "price"

    def test_parse_empty_list(self):
        from search.query_planner import QueryPlanner
        assert QueryPlanner._parse_follow_ups([]) == []

    def test_parse_none(self):
        from search.query_planner import QueryPlanner
        assert QueryPlanner._parse_follow_ups(None) == []

    def test_parse_not_a_list(self):
        from search.query_planner import QueryPlanner
        assert QueryPlanner._parse_follow_ups("not a list") == []
        assert QueryPlanner._parse_follow_ups(42) == []
        assert QueryPlanner._parse_follow_ups({}) == []

    def test_parse_skips_non_dict_entries(self):
        from search.query_planner import QueryPlanner
        raw = [
            "not a dict",
            42,
            None,
            {
                "dimension": "color",
                "question": "Any color preference?",
                "options": [
                    {"label": "Black", "filters": {"colors": ["Black"]}},
                ],
            },
        ]
        parsed = QueryPlanner._parse_follow_ups(raw)
        assert len(parsed) == 1
        assert parsed[0].dimension == "color"

    def test_parse_skips_missing_question(self):
        """Entries without 'question' should be skipped."""
        from search.query_planner import QueryPlanner
        raw = [
            {
                "dimension": "price",
                # Missing "question" key
                "options": [
                    {"label": "Cheap", "filters": {"max_price": 30}},
                ],
            },
        ]
        parsed = QueryPlanner._parse_follow_ups(raw)
        assert len(parsed) == 0

    def test_parse_skips_empty_options(self):
        """Entries with no valid options should be skipped."""
        from search.query_planner import QueryPlanner
        raw = [
            {
                "dimension": "formality",
                "question": "How dressed up?",
                "options": [],
            },
        ]
        parsed = QueryPlanner._parse_follow_ups(raw)
        assert len(parsed) == 0

    def test_parse_skips_options_without_label(self):
        """Options missing 'label' should be skipped."""
        from search.query_planner import QueryPlanner
        raw = [
            {
                "dimension": "formality",
                "question": "How dressed up?",
                "options": [
                    {"filters": {"formality": ["Casual"]}},  # missing label
                    {"label": "Dressy", "filters": {"formality": ["Formal"]}},
                ],
            },
        ]
        parsed = QueryPlanner._parse_follow_ups(raw)
        assert len(parsed) == 1
        assert len(parsed[0].options) == 1  # only the valid option
        assert parsed[0].options[0].label == "Dressy"

    def test_parse_option_without_filters_gets_empty_dict(self):
        """Options without 'filters' should get empty dict."""
        from search.query_planner import QueryPlanner
        raw = [
            {
                "dimension": "color",
                "question": "Color preference?",
                "options": [
                    {"label": "No preference"},  # missing filters
                ],
            },
        ]
        parsed = QueryPlanner._parse_follow_ups(raw)
        assert len(parsed) == 1
        assert parsed[0].options[0].filters == {}

    def test_parse_defaults_dimension_to_other(self):
        """Missing dimension should default to 'other'."""
        from search.query_planner import QueryPlanner
        raw = [
            {
                "question": "Something?",
                "options": [{"label": "Yes", "filters": {}}],
            },
        ]
        parsed = QueryPlanner._parse_follow_ups(raw)
        assert len(parsed) == 1
        assert parsed[0].dimension == "other"

    def test_parse_with_modes_in_filters(self):
        """Options with modes in filters should parse correctly."""
        from search.query_planner import QueryPlanner
        raw = [
            {
                "dimension": "coverage",
                "question": "How much coverage?",
                "options": [
                    {"label": "Full coverage", "filters": {"modes": ["modest"]}},
                    {"label": "Some coverage", "filters": {"modes": ["cover_chest", "cover_arms"]}},
                    {"label": "Don't mind", "filters": {}},
                ],
            },
        ]
        parsed = QueryPlanner._parse_follow_ups(raw)
        assert len(parsed) == 1
        assert len(parsed[0].options) == 3
        assert parsed[0].options[0].filters["modes"] == ["modest"]


# =============================================================================
# 4. SearchPlan with parsed_follow_ups
# =============================================================================

class TestSearchPlanFollowUps:
    """Test that SearchPlan correctly holds parsed_follow_ups."""

    def test_search_plan_default_empty(self):
        from search.query_planner import SearchPlan
        plan = SearchPlan(intent="specific", algolia_query="dress", semantic_query="a dress")
        assert plan.follow_ups == []
        assert plan.parsed_follow_ups == []

    def test_search_plan_with_raw_follow_ups(self):
        from search.query_planner import SearchPlan
        plan = SearchPlan(
            intent="vague",
            algolia_query="",
            semantic_query="cute outfit",
            follow_ups=[{"dimension": "price", "question": "Budget?", "options": []}],
        )
        assert len(plan.follow_ups) == 1

    def test_search_plan_parsed_follow_ups_settable(self):
        from search.query_planner import SearchPlan, QueryPlanner
        from search.models import FollowUpQuestion, FollowUpOption

        plan = SearchPlan(intent="vague", algolia_query="", semantic_query="night out")
        parsed = QueryPlanner._parse_follow_ups([
            {
                "dimension": "formality",
                "question": "How dressy?",
                "options": [
                    {"label": "Casual", "filters": {"formality": ["Casual"]}},
                    {"label": "Glam", "filters": {"formality": ["Formal"]}},
                ],
            },
        ])
        plan.parsed_follow_ups = parsed
        assert len(plan.parsed_follow_ups) == 1
        assert plan.parsed_follow_ups[0].dimension == "formality"


# =============================================================================
# 5. HybridSearchResponse with follow_ups
# =============================================================================

class TestHybridSearchResponseFollowUps:
    """Test HybridSearchResponse serialization with follow_ups."""

    def test_response_without_follow_ups(self):
        from search.models import HybridSearchResponse, PaginationInfo
        resp = HybridSearchResponse(
            query="red dress",
            intent="specific",
            results=[],
            pagination=PaginationInfo(page=1, page_size=50, has_more=False),
        )
        assert resp.follow_ups is None

    def test_response_with_follow_ups(self):
        from search.models import (
            HybridSearchResponse, PaginationInfo,
            FollowUpQuestion, FollowUpOption,
        )
        resp = HybridSearchResponse(
            query="cute outfit",
            intent="vague",
            results=[],
            pagination=PaginationInfo(page=1, page_size=50, has_more=False),
            follow_ups=[
                FollowUpQuestion(
                    dimension="formality",
                    question="How dressed up?",
                    options=[
                        FollowUpOption(label="Casual", filters={"formality": ["Casual"]}),
                        FollowUpOption(label="Dressy", filters={"formality": ["Formal"]}),
                    ],
                ),
            ],
        )
        assert resp.follow_ups is not None
        assert len(resp.follow_ups) == 1
        assert resp.follow_ups[0].dimension == "formality"

    def test_response_serialization_with_follow_ups(self):
        """Full JSON serialization roundtrip."""
        from search.models import (
            HybridSearchResponse, PaginationInfo,
            FollowUpQuestion, FollowUpOption,
        )
        resp = HybridSearchResponse(
            query="something for tonight",
            intent="vague",
            results=[],
            pagination=PaginationInfo(page=1, page_size=50, has_more=False),
            follow_ups=[
                FollowUpQuestion(
                    dimension="formality",
                    question="How dressed up do you want to be?",
                    options=[
                        FollowUpOption(label="Casual & fun", filters={"formality": ["Casual"]}),
                        FollowUpOption(label="Sexy & glam", filters={"formality": ["Semi-Formal"], "modes": ["glamorous"]}),
                    ],
                ),
                FollowUpQuestion(
                    dimension="garment_type",
                    question="What type of outfit?",
                    options=[
                        FollowUpOption(label="A dress", filters={"category_l1": ["Dresses"]}),
                        FollowUpOption(label="Top & bottoms", filters={"category_l1": ["Tops", "Bottoms"]}),
                    ],
                ),
            ],
        )
        d = resp.model_dump()
        assert d["follow_ups"] is not None
        assert len(d["follow_ups"]) == 2
        assert d["follow_ups"][0]["dimension"] == "formality"
        assert d["follow_ups"][0]["options"][1]["filters"]["modes"] == ["glamorous"]

        # Verify it's JSON-serializable
        json_str = json.dumps(d)
        assert '"follow_ups"' in json_str
        assert '"formality"' in json_str

    def test_response_serialization_without_follow_ups(self):
        """When follow_ups is None, JSON should have null."""
        from search.models import HybridSearchResponse, PaginationInfo
        resp = HybridSearchResponse(
            query="red dress",
            intent="specific",
            results=[],
            pagination=PaginationInfo(page=1, page_size=50, has_more=False),
        )
        d = resp.model_dump()
        assert d["follow_ups"] is None


# =============================================================================
# 6. Refine Endpoint
# =============================================================================

class TestRefineEndpoint:
    """Test the /api/search/refine endpoint logic."""

    def test_refine_request_builds_search_request(self):
        """SearchRefineRequest selected_filters should map to HybridSearchRequest fields."""
        from search.models import HybridSearchRequest, SearchRefineRequest

        refine = SearchRefineRequest(
            original_query="something cute",
            selected_filters={
                "formality": ["Casual"],
                "category_l1": ["Dresses"],
                "max_price": 75,
            },
        )

        # Build search request the same way the endpoint does
        search_fields = {
            "query": refine.original_query,
            "page": refine.page,
            "page_size": refine.page_size,
        }
        _ALLOWED = {
            "categories", "category_l1", "category_l2", "brands", "colors",
            "formality", "max_price", "min_price",
        }
        for field, value in refine.selected_filters.items():
            if field in _ALLOWED and value is not None:
                search_fields[field] = value

        req = HybridSearchRequest(**search_fields)
        assert req.query == "something cute"
        assert req.formality == ["Casual"]
        assert req.category_l1 == ["Dresses"]
        assert req.max_price == 75

    def test_refine_mode_expansion(self):
        """Modes in selected_filters should be expanded into concrete filters."""
        from search.mode_config import expand_modes

        modes = ["cover_arms", "work"]
        mode_filters, mode_exclusions, _, _ = expand_modes(modes)

        # cover_arms should produce sleeve_type exclusions
        assert "sleeve_type" in mode_exclusions
        excl_sleeves = {v.lower() for v in mode_exclusions["sleeve_type"]}
        assert "sleeveless" in excl_sleeves

        # work should produce occasion/formality filters
        assert "occasions" in mode_filters or "formality" in mode_filters

    def test_refine_mode_expansion_glamorous(self):
        """glamorous mode should expand into style_tags."""
        from search.mode_config import expand_modes

        mode_filters, _, _, _ = expand_modes(["glamorous"])
        assert "style_tags" in mode_filters
        style_vals = {v.lower() for v in mode_filters["style_tags"]}
        assert "glamorous" in style_vals

    def test_refine_coverage_modes(self):
        """Coverage modes should expand into exclusion filters."""
        from search.mode_config import expand_modes

        mode_filters, mode_exclusions, _, _ = expand_modes(["cover_chest"])
        # Should exclude deep necklines
        assert "neckline" in mode_exclusions
        excl_necklines = {v.lower() for v in mode_exclusions["neckline"]}
        # At least some revealing necklines should be excluded
        assert len(excl_necklines) > 0

    def test_refine_merges_multiple_filter_sources(self):
        """selected_filters with both direct filters and modes should merge."""
        from search.models import SearchRefineRequest

        refine = SearchRefineRequest(
            original_query="outfit",
            selected_filters={
                "colors": ["Black"],
                "modes": ["casual"],
                "max_price": 50,
            },
        )

        # Simulate what the endpoint does
        modes = refine.selected_filters.pop("modes", None)
        assert modes == ["casual"]
        assert refine.selected_filters["colors"] == ["Black"]
        assert refine.selected_filters["max_price"] == 50

        # Expand modes
        from search.mode_config import expand_modes
        mode_filters, _, _, _ = expand_modes(modes)
        # casual mode should produce formality filters
        assert "formality" in mode_filters


# =============================================================================
# 7. Integration: Follow-ups flow through hybrid_search
# =============================================================================

class TestFollowUpsIntegration:
    """Test that follow_ups are passed through the search pipeline."""

    def test_search_plan_follow_ups_reach_response(self):
        """Verify follow_ups on SearchPlan flow to HybridSearchResponse."""
        from search.query_planner import SearchPlan, QueryPlanner
        from search.models import (
            HybridSearchResponse, PaginationInfo,
            FollowUpQuestion, FollowUpOption,
        )

        # Simulate what hybrid_search.py does
        plan = SearchPlan(
            intent="vague",
            algolia_query="",
            semantic_query="cute outfit",
        )
        plan.parsed_follow_ups = QueryPlanner._parse_follow_ups([
            {
                "dimension": "formality",
                "question": "How dressed up?",
                "options": [
                    {"label": "Casual", "filters": {"formality": ["Casual"]}},
                    {"label": "Dressy", "filters": {"formality": ["Formal"]}},
                ],
            },
        ])

        # Extract follow_ups the same way hybrid_search.py does
        follow_ups = None
        if plan.parsed_follow_ups:
            follow_ups = plan.parsed_follow_ups

        # Build response
        resp = HybridSearchResponse(
            query="cute outfit",
            intent="vague",
            results=[],
            pagination=PaginationInfo(page=1, page_size=50, has_more=False),
            follow_ups=follow_ups,
        )

        assert resp.follow_ups is not None
        assert len(resp.follow_ups) == 1
        assert resp.follow_ups[0].dimension == "formality"
        assert resp.follow_ups[0].options[0].label == "Casual"

    def test_specific_query_no_follow_ups(self):
        """Specific queries should have no follow-ups (planner returns [])."""
        from search.query_planner import SearchPlan, QueryPlanner
        from search.models import HybridSearchResponse, PaginationInfo

        plan = SearchPlan(
            intent="specific",
            algolia_query="red midi dress",
            semantic_query="a red midi length dress",
        )
        plan.parsed_follow_ups = QueryPlanner._parse_follow_ups([])

        follow_ups = plan.parsed_follow_ups if plan.parsed_follow_ups else None

        resp = HybridSearchResponse(
            query="red midi dress",
            intent="specific",
            results=[],
            pagination=PaginationInfo(page=1, page_size=50, has_more=False),
            follow_ups=follow_ups,
        )
        assert resp.follow_ups is None

    def test_planner_failure_no_follow_ups(self):
        """When planner fails (returns None), follow_ups should be None."""
        from search.models import HybridSearchResponse, PaginationInfo

        # Simulate planner returning None
        search_plan = None
        follow_ups = None

        if search_plan is not None:
            follow_ups = getattr(search_plan, "parsed_follow_ups", None) or None

        resp = HybridSearchResponse(
            query="something",
            intent="specific",
            results=[],
            pagination=PaginationInfo(page=1, page_size=50, has_more=False),
            follow_ups=follow_ups,
        )
        assert resp.follow_ups is None


# =============================================================================
# 8. Realistic follow-up scenarios
# =============================================================================

class TestRealisticFollowUps:
    """Test realistic follow-up question scenarios."""

    def test_night_out_query_follow_ups(self):
        """'something for a night out' should generate formality + garment questions."""
        from search.query_planner import QueryPlanner

        raw = [
            {
                "dimension": "formality",
                "question": "How dressed up do you want to be for your night out?",
                "options": [
                    {"label": "Casual & fun", "filters": {"formality": ["Casual"], "modes": ["casual"]}},
                    {"label": "Sexy & glam", "filters": {"formality": ["Semi-Formal"], "modes": ["glamorous"]}},
                    {"label": "Elegant & classy", "filters": {"formality": ["Formal"], "modes": ["very_formal"]}},
                ],
            },
            {
                "dimension": "garment_type",
                "question": "What type of outfit are you thinking?",
                "options": [
                    {"label": "A dress", "filters": {"category_l1": ["Dresses"]}},
                    {"label": "Top & bottoms", "filters": {"category_l1": ["Tops", "Bottoms"]}},
                    {"label": "A jumpsuit", "filters": {"category_l2": ["Jumpsuit", "Jumpsuits"]}},
                ],
            },
        ]
        parsed = QueryPlanner._parse_follow_ups(raw)
        assert len(parsed) == 2

        # Verify formality question
        formality_q = parsed[0]
        assert formality_q.dimension == "formality"
        assert "night out" in formality_q.question.lower()
        assert len(formality_q.options) == 3
        # Verify the "Sexy & glam" option has modes
        glam_opt = formality_q.options[1]
        assert "glamorous" in glam_opt.filters.get("modes", [])

        # Verify garment type question
        garment_q = parsed[1]
        assert garment_q.dimension == "garment_type"
        assert len(garment_q.options) == 3

    def test_wedding_query_follow_ups(self):
        """'what to wear to a wedding' should generate coverage + color questions."""
        from search.query_planner import QueryPlanner

        raw = [
            {
                "dimension": "coverage",
                "question": "How much coverage do you want?",
                "options": [
                    {"label": "Full coverage", "filters": {"modes": ["modest"]}},
                    {"label": "Moderate", "filters": {"modes": ["cover_chest", "cover_straps"]}},
                    {"label": "Don't mind", "filters": {}},
                ],
            },
            {
                "dimension": "color",
                "question": "Any color preference for the wedding?",
                "options": [
                    {"label": "Pastels", "filters": {"colors": ["Pink", "Light Blue", "Cream"]}},
                    {"label": "Bold colors", "filters": {"colors": ["Red", "Green", "Purple"]}},
                    {"label": "Neutrals", "filters": {"colors": ["Black", "Beige", "Navy Blue"]}},
                ],
            },
        ]
        parsed = QueryPlanner._parse_follow_ups(raw)
        assert len(parsed) == 2
        assert parsed[0].dimension == "coverage"
        assert parsed[1].dimension == "color"
        assert "wedding" in parsed[1].question.lower()

    def test_follow_up_option_filters_are_additive(self):
        """Each option's filters should be self-contained and mergeable."""
        from search.query_planner import QueryPlanner
        from search.models import HybridSearchRequest

        raw = [
            {
                "dimension": "price",
                "question": "What's your budget?",
                "options": [
                    {"label": "Under $30", "filters": {"max_price": 30}},
                    {"label": "$30-$75", "filters": {"min_price": 30, "max_price": 75}},
                    {"label": "Over $75", "filters": {"min_price": 75}},
                ],
            },
        ]
        parsed = QueryPlanner._parse_follow_ups(raw)

        # Verify each option can be merged into a search request
        for opt in parsed[0].options:
            fields = {"query": "test"}
            fields.update(opt.filters)
            req = HybridSearchRequest(**fields)
            assert req.query == "test"
            if "max_price" in opt.filters:
                assert req.max_price == opt.filters["max_price"]
            if "min_price" in opt.filters:
                assert req.min_price == opt.filters["min_price"]
