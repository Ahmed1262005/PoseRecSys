"""
Tests for comprehensive feed attribute filters (include/exclude).

Tests the _apply_attribute_filters() method in pipeline.py which provides
hard filtering on all product_attributes dimensions:
- Single-value: formality, color_family, silhouette, fit, length, sleeve,
                neckline, rise, coverage, pattern
- Multi-value:  seasons, style_tags, occasions, materials

Each filter supports both include (positive) and exclude (negative) modes.
"""

import sys
import os
import pytest

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from recs.models import Candidate


# =============================================================================
# Helpers
# =============================================================================

def make_candidate(
    item_id: str = "test-1",
    formality: str = None,
    color_family: str = None,
    silhouette: str = None,
    fit: str = None,
    length: str = None,
    sleeve: str = None,
    neckline: str = None,
    rise: str = None,
    coverage_level: str = None,
    pattern: str = None,
    seasons: list = None,
    style_tags: list = None,
    occasions: list = None,
    materials: list = None,
    **kwargs,
) -> Candidate:
    """Create a Candidate with specified attributes for testing."""
    return Candidate(
        item_id=item_id,
        formality=formality,
        color_family=color_family,
        silhouette=silhouette,
        fit=fit,
        length=length,
        sleeve=sleeve,
        neckline=neckline,
        rise=rise,
        coverage_level=coverage_level,
        pattern=pattern,
        seasons=seasons or [],
        style_tags=style_tags or [],
        occasions=occasions or [],
        materials=materials or [],
        **kwargs,
    )


class FakeReranker:
    def rerank(self, candidates, **kwargs):
        return candidates


class FakeRanker:
    model = None
    config = type('C', (), {
        'WARM_WEIGHTS': {},
        'COLD_WEIGHTS': {},
        'MIN_SEQUENCE_FOR_SASREC': 5,
    })()
    def rank_candidates(self, user_state, candidates):
        return candidates
    def get_model_info(self):
        return {"model_loaded": False, "vocab_size": 0}


def get_filter_method():
    """
    Get the _apply_attribute_filters method from the pipeline class
    without initializing the full pipeline (which requires Supabase).
    """
    from recs.pipeline import RecommendationPipeline
    # Access the unbound method directly from the class
    return RecommendationPipeline._apply_attribute_filters


# We test the method directly by calling it with self=None (it only uses self for the class-level dicts)
# Actually, it references self.SINGLE_VALUE_FILTERS etc, so we need a minimal instance.
# Instead, we'll instantiate a minimal mock.

class MinimalPipeline:
    """Minimal pipeline mock that only has the filter method and class-level dicts."""
    pass

# Copy class-level dicts and method
from recs.pipeline import RecommendationPipeline
MinimalPipeline.SINGLE_VALUE_FILTERS = RecommendationPipeline.SINGLE_VALUE_FILTERS
MinimalPipeline.MULTI_VALUE_FILTERS = RecommendationPipeline.MULTI_VALUE_FILTERS
MinimalPipeline._apply_attribute_filters = RecommendationPipeline._apply_attribute_filters


@pytest.fixture
def pipeline():
    """Return a minimal pipeline instance for testing filters."""
    return MinimalPipeline()


# =============================================================================
# Single-Value Include Tests
# =============================================================================

class TestSingleValueInclude:
    """Test include filters on single-value attributes."""

    def test_include_formality_passes(self, pipeline):
        candidates = [
            make_candidate("c1", formality="Casual"),
            make_candidate("c2", formality="Formal"),
            make_candidate("c3", formality="Smart Casual"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_formality": ["Casual", "Smart Casual"]
        })
        assert len(result) == 2
        assert {c.item_id for c in result} == {"c1", "c3"}

    def test_include_formality_case_insensitive(self, pipeline):
        candidates = [
            make_candidate("c1", formality="casual"),
            make_candidate("c2", formality="FORMAL"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_formality": ["Casual"]
        })
        assert len(result) == 1
        assert result[0].item_id == "c1"

    def test_include_formality_excludes_none(self, pipeline):
        """Items with None formality should be excluded when include is active."""
        candidates = [
            make_candidate("c1", formality="Casual"),
            make_candidate("c2", formality=None),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_formality": ["Casual"]
        })
        assert len(result) == 1
        assert result[0].item_id == "c1"

    def test_include_silhouette(self, pipeline):
        candidates = [
            make_candidate("c1", silhouette="Wide Leg"),
            make_candidate("c2", silhouette="Skinny"),
            make_candidate("c3", silhouette="Straight"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_silhouette": ["Wide Leg", "Straight"]
        })
        assert len(result) == 2
        assert {c.item_id for c in result} == {"c1", "c3"}

    def test_include_fit(self, pipeline):
        candidates = [
            make_candidate("c1", fit="Regular"),
            make_candidate("c2", fit="Slim"),
            make_candidate("c3", fit="Oversized"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_fit": ["regular", "oversized"]
        })
        assert len(result) == 2
        assert {c.item_id for c in result} == {"c1", "c3"}

    def test_include_length(self, pipeline):
        candidates = [
            make_candidate("c1", length="Mini"),
            make_candidate("c2", length="Midi"),
            make_candidate("c3", length="Maxi"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_length": ["mini", "midi"]
        })
        assert len(result) == 2

    def test_include_sleeves(self, pipeline):
        """'sleeves' filter maps to Candidate.sleeve field."""
        candidates = [
            make_candidate("c1", sleeve="Long"),
            make_candidate("c2", sleeve="Short"),
            make_candidate("c3", sleeve="Sleeveless"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_sleeves": ["long"]
        })
        assert len(result) == 1
        assert result[0].item_id == "c1"

    def test_include_neckline(self, pipeline):
        candidates = [
            make_candidate("c1", neckline="V-Neck"),
            make_candidate("c2", neckline="Crew"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_neckline": ["v-neck"]
        })
        assert len(result) == 1
        assert result[0].item_id == "c1"

    def test_include_rise(self, pipeline):
        candidates = [
            make_candidate("c1", rise="High"),
            make_candidate("c2", rise="Mid"),
            make_candidate("c3", rise="Low"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_rise": ["high", "mid"]
        })
        assert len(result) == 2

    def test_include_coverage(self, pipeline):
        """'coverage' filter maps to Candidate.coverage_level."""
        candidates = [
            make_candidate("c1", coverage_level="Full"),
            make_candidate("c2", coverage_level="Moderate"),
            make_candidate("c3", coverage_level="Minimal"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_coverage": ["full", "moderate"]
        })
        assert len(result) == 2
        assert {c.item_id for c in result} == {"c1", "c2"}

    def test_include_pattern(self, pipeline):
        candidates = [
            make_candidate("c1", pattern="Solid"),
            make_candidate("c2", pattern="Floral"),
            make_candidate("c3", pattern="Striped"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_pattern": ["solid", "striped"]
        })
        assert len(result) == 2
        assert {c.item_id for c in result} == {"c1", "c3"}

    def test_include_color_family(self, pipeline):
        candidates = [
            make_candidate("c1", color_family="Neutrals"),
            make_candidate("c2", color_family="Blues"),
            make_candidate("c3", color_family="Greens"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_color_family": ["neutrals", "blues"]
        })
        assert len(result) == 2


# =============================================================================
# Single-Value Exclude Tests
# =============================================================================

class TestSingleValueExclude:
    """Test exclude filters on single-value attributes."""

    def test_exclude_formality(self, pipeline):
        candidates = [
            make_candidate("c1", formality="Casual"),
            make_candidate("c2", formality="Formal"),
            make_candidate("c3", formality="Smart Casual"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "exclude_formality": ["Formal"]
        })
        assert len(result) == 2
        assert {c.item_id for c in result} == {"c1", "c3"}

    def test_exclude_formality_case_insensitive(self, pipeline):
        candidates = [
            make_candidate("c1", formality="FORMAL"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "exclude_formality": ["formal"]
        })
        assert len(result) == 0

    def test_exclude_none_passes(self, pipeline):
        """Items with None attribute should PASS the exclude filter."""
        candidates = [
            make_candidate("c1", formality=None),
            make_candidate("c2", formality="Formal"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "exclude_formality": ["Formal"]
        })
        assert len(result) == 1
        assert result[0].item_id == "c1"

    def test_exclude_silhouette(self, pipeline):
        candidates = [
            make_candidate("c1", silhouette="Skinny"),
            make_candidate("c2", silhouette="Wide Leg"),
            make_candidate("c3", silhouette=None),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "exclude_silhouette": ["skinny"]
        })
        assert len(result) == 2
        assert {c.item_id for c in result} == {"c2", "c3"}

    def test_exclude_pattern(self, pipeline):
        candidates = [
            make_candidate("c1", pattern="Solid"),
            make_candidate("c2", pattern="Animal Print"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "exclude_pattern": ["animal print"]
        })
        assert len(result) == 1
        assert result[0].item_id == "c1"


# =============================================================================
# Multi-Value Include Tests
# =============================================================================

class TestMultiValueInclude:
    """Test include filters on multi-value attributes."""

    def test_include_seasons(self, pipeline):
        candidates = [
            make_candidate("c1", seasons=["Spring", "Summer"]),
            make_candidate("c2", seasons=["Fall", "Winter"]),
            make_candidate("c3", seasons=["Spring", "Fall"]),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_seasons": ["Spring"]
        })
        assert len(result) == 2
        assert {c.item_id for c in result} == {"c1", "c3"}

    def test_include_seasons_case_insensitive(self, pipeline):
        candidates = [
            make_candidate("c1", seasons=["spring", "summer"]),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_seasons": ["Spring"]
        })
        assert len(result) == 1

    def test_include_seasons_excludes_empty(self, pipeline):
        """Items with empty seasons list should be excluded when include is active."""
        candidates = [
            make_candidate("c1", seasons=["Spring"]),
            make_candidate("c2", seasons=[]),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_seasons": ["Spring"]
        })
        assert len(result) == 1
        assert result[0].item_id == "c1"

    def test_include_style_tags(self, pipeline):
        candidates = [
            make_candidate("c1", style_tags=["Classic", "Minimal"]),
            make_candidate("c2", style_tags=["Bold", "Street"]),
            make_candidate("c3", style_tags=["Trendy", "Classic"]),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_style_tags": ["Classic"]
        })
        assert len(result) == 2
        assert {c.item_id for c in result} == {"c1", "c3"}

    def test_include_occasions(self, pipeline):
        candidates = [
            make_candidate("c1", occasions=["Office", "Everyday"]),
            make_candidate("c2", occasions=["Party", "Date Night"]),
            make_candidate("c3", occasions=["Everyday"]),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_occasions": ["Office"]
        })
        assert len(result) == 1
        assert result[0].item_id == "c1"

    def test_include_materials(self, pipeline):
        candidates = [
            make_candidate("c1", materials=["Cotton", "Polyester"]),
            make_candidate("c2", materials=["Silk", "Linen"]),
            make_candidate("c3", materials=["Cotton", "Linen"]),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_materials": ["cotton"]
        })
        assert len(result) == 2
        assert {c.item_id for c in result} == {"c1", "c3"}

    def test_include_multiple_values_any_match(self, pipeline):
        """Include should pass if ANY value in the list matches ANY include value."""
        candidates = [
            make_candidate("c1", seasons=["Fall", "Winter"]),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_seasons": ["Spring", "Summer", "Fall"]
        })
        assert len(result) == 1


# =============================================================================
# Multi-Value Exclude Tests
# =============================================================================

class TestMultiValueExclude:
    """Test exclude filters on multi-value attributes."""

    def test_exclude_seasons(self, pipeline):
        candidates = [
            make_candidate("c1", seasons=["Spring", "Summer"]),
            make_candidate("c2", seasons=["Fall", "Winter"]),
            make_candidate("c3", seasons=["Spring", "Winter"]),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "exclude_seasons": ["Winter"]
        })
        assert len(result) == 1
        assert result[0].item_id == "c1"

    def test_exclude_seasons_empty_passes(self, pipeline):
        """Items with empty seasons list should PASS the exclude filter."""
        candidates = [
            make_candidate("c1", seasons=[]),
            make_candidate("c2", seasons=["Winter"]),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "exclude_seasons": ["Winter"]
        })
        assert len(result) == 1
        assert result[0].item_id == "c1"

    def test_exclude_style_tags(self, pipeline):
        candidates = [
            make_candidate("c1", style_tags=["Classic", "Minimal"]),
            make_candidate("c2", style_tags=["Bold", "Street"]),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "exclude_style_tags": ["Bold"]
        })
        assert len(result) == 1
        assert result[0].item_id == "c1"

    def test_exclude_occasions(self, pipeline):
        candidates = [
            make_candidate("c1", occasions=["Office", "Everyday"]),
            make_candidate("c2", occasions=["Party"]),
            make_candidate("c3", occasions=[]),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "exclude_occasions": ["Party"]
        })
        assert len(result) == 2
        assert {c.item_id for c in result} == {"c1", "c3"}

    def test_exclude_materials(self, pipeline):
        candidates = [
            make_candidate("c1", materials=["Cotton"]),
            make_candidate("c2", materials=["Polyester", "Nylon"]),
            make_candidate("c3", materials=[]),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "exclude_materials": ["polyester"]
        })
        assert len(result) == 2
        assert {c.item_id for c in result} == {"c1", "c3"}


# =============================================================================
# Combined Filter Tests
# =============================================================================

class TestCombinedFilters:
    """Test multiple filters applied simultaneously."""

    def test_include_and_exclude_same_dimension(self, pipeline):
        """Include + exclude on same attribute: include takes priority for what's kept,
        exclude removes from that subset."""
        candidates = [
            make_candidate("c1", formality="Casual"),
            make_candidate("c2", formality="Smart Casual"),
            make_candidate("c3", formality="Formal"),
            make_candidate("c4", formality="Semi-Formal"),
        ]
        # Include Casual+Smart Casual+Semi-Formal, then exclude Semi-Formal
        result = pipeline._apply_attribute_filters(candidates, {
            "include_formality": ["Casual", "Smart Casual", "Semi-Formal"],
            "exclude_formality": ["Semi-Formal"],
        })
        assert len(result) == 2
        assert {c.item_id for c in result} == {"c1", "c2"}

    def test_multiple_dimensions(self, pipeline):
        """Multiple different attribute filters applied together (AND logic)."""
        candidates = [
            make_candidate("c1", formality="Casual", seasons=["Spring", "Summer"]),
            make_candidate("c2", formality="Casual", seasons=["Winter"]),
            make_candidate("c3", formality="Formal", seasons=["Spring"]),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_formality": ["Casual"],
            "include_seasons": ["Spring", "Summer"],
        })
        # Only c1: Casual AND has Spring/Summer
        assert len(result) == 1
        assert result[0].item_id == "c1"

    def test_three_dimensions(self, pipeline):
        candidates = [
            make_candidate("c1", formality="Casual", fit="Regular", pattern="Solid"),
            make_candidate("c2", formality="Casual", fit="Slim", pattern="Solid"),
            make_candidate("c3", formality="Formal", fit="Regular", pattern="Solid"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_formality": ["Casual"],
            "include_fit": ["regular"],
            "include_pattern": ["solid"],
        })
        assert len(result) == 1
        assert result[0].item_id == "c1"

    def test_mix_include_exclude_across_dimensions(self, pipeline):
        candidates = [
            make_candidate("c1", formality="Casual", silhouette="Wide Leg"),
            make_candidate("c2", formality="Casual", silhouette="Skinny"),
            make_candidate("c3", formality="Formal", silhouette="Straight"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_formality": ["Casual"],
            "exclude_silhouette": ["Skinny"],
        })
        assert len(result) == 1
        assert result[0].item_id == "c1"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_filters_returns_all(self, pipeline):
        candidates = [make_candidate(f"c{i}") for i in range(5)]
        result = pipeline._apply_attribute_filters(candidates, {})
        assert len(result) == 5

    def test_none_filters_returns_all(self, pipeline):
        candidates = [make_candidate(f"c{i}") for i in range(5)]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_formality": None,
            "exclude_seasons": None,
        })
        # None values are skipped
        assert len(result) == 5

    def test_empty_candidate_list(self, pipeline):
        result = pipeline._apply_attribute_filters([], {
            "include_formality": ["Casual"]
        })
        assert len(result) == 0

    def test_all_filtered_out(self, pipeline):
        candidates = [
            make_candidate("c1", formality="Formal"),
            make_candidate("c2", formality="Formal"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_formality": ["Casual"]
        })
        assert len(result) == 0

    def test_all_none_attributes_with_include(self, pipeline):
        """All candidates have None attribute -> all excluded by include."""
        candidates = [
            make_candidate("c1", formality=None),
            make_candidate("c2", formality=None),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_formality": ["Casual"]
        })
        assert len(result) == 0

    def test_all_none_attributes_with_exclude(self, pipeline):
        """All candidates have None attribute -> all pass exclude."""
        candidates = [
            make_candidate("c1", formality=None),
            make_candidate("c2", formality=None),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "exclude_formality": ["Formal"]
        })
        assert len(result) == 2

    def test_mixed_none_and_values(self, pipeline):
        candidates = [
            make_candidate("c1", formality="Casual"),
            make_candidate("c2", formality=None),
            make_candidate("c3", formality="Formal"),
        ]
        # Include: c1 passes, c2 excluded (None), c3 excluded (not in list)
        result = pipeline._apply_attribute_filters(candidates, {
            "include_formality": ["Casual"]
        })
        assert len(result) == 1
        assert result[0].item_id == "c1"

    def test_single_item_passes(self, pipeline):
        candidates = [make_candidate("c1", formality="Casual")]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_formality": ["Casual"]
        })
        assert len(result) == 1

    def test_large_include_list(self, pipeline):
        """Many values in include list."""
        candidates = [make_candidate("c1", formality="Casual")]
        include_list = [f"Value{i}" for i in range(100)] + ["Casual"]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_formality": include_list
        })
        assert len(result) == 1


# =============================================================================
# Filter Mapping Tests
# =============================================================================

class TestFilterMapping:
    """Test that filter names correctly map to Candidate fields."""

    def test_sleeves_maps_to_sleeve(self, pipeline):
        """API param 'sleeves' should map to Candidate.sleeve field."""
        candidates = [
            make_candidate("c1", sleeve="Long"),
            make_candidate("c2", sleeve="Short"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_sleeves": ["long"]
        })
        assert len(result) == 1
        assert result[0].item_id == "c1"

    def test_coverage_maps_to_coverage_level(self, pipeline):
        """API param 'coverage' should map to Candidate.coverage_level field."""
        candidates = [
            make_candidate("c1", coverage_level="Full"),
            make_candidate("c2", coverage_level="Minimal"),
        ]
        result = pipeline._apply_attribute_filters(candidates, {
            "include_coverage": ["full"]
        })
        assert len(result) == 1
        assert result[0].item_id == "c1"

    def test_all_single_value_filters_exist(self, pipeline):
        """Verify all SINGLE_VALUE_FILTERS map to valid Candidate fields."""
        c = make_candidate("test")
        for filter_name, field_name in pipeline.SINGLE_VALUE_FILTERS.items():
            assert hasattr(c, field_name), f"Candidate has no field '{field_name}' for filter '{filter_name}'"

    def test_all_multi_value_filters_exist(self, pipeline):
        """Verify all MULTI_VALUE_FILTERS map to valid Candidate fields."""
        c = make_candidate("test")
        for filter_name, field_name in pipeline.MULTI_VALUE_FILTERS.items():
            assert hasattr(c, field_name), f"Candidate has no field '{field_name}' for filter '{filter_name}'"


# =============================================================================
# on_sale_only Tests (via pipeline params)
# =============================================================================

class TestOnSaleFilter:
    """Test on_sale_only parameter is plumbed through."""

    def test_on_sale_only_is_accepted(self):
        """Verify pipeline.get_feed_keyset accepts on_sale_only parameter."""
        from recs.pipeline import RecommendationPipeline
        import inspect
        sig = inspect.signature(RecommendationPipeline.get_feed_keyset)
        assert 'on_sale_only' in sig.parameters

    def test_all_new_filters_accepted(self):
        """Verify pipeline.get_feed_keyset accepts all new filter parameters."""
        from recs.pipeline import RecommendationPipeline
        import inspect
        sig = inspect.signature(RecommendationPipeline.get_feed_keyset)
        expected_params = [
            'include_formality', 'exclude_formality',
            'include_seasons', 'exclude_seasons',
            'include_style_tags', 'exclude_style_tags',
            'include_color_family', 'exclude_color_family',
            'include_silhouette', 'exclude_silhouette',
            'include_fit', 'exclude_fit',
            'include_length', 'exclude_length',
            'include_sleeves', 'exclude_sleeves',
            'include_neckline', 'exclude_neckline',
            'include_rise', 'exclude_rise',
            'include_coverage', 'exclude_coverage',
            'include_materials', 'exclude_materials',
            'exclude_occasions',
        ]
        for param in expected_params:
            assert param in sig.parameters, f"Missing parameter: {param}"


# =============================================================================
# API Endpoint Tests (signature verification)
# =============================================================================

class TestAPIEndpointSignatures:
    """Verify API endpoints accept all filter query params."""

    def test_feed_endpoint_has_all_filters(self):
        """Verify /feed endpoint function accepts all new filter params."""
        from recs.api_endpoints import get_pipeline_feed
        import inspect
        sig = inspect.signature(get_pipeline_feed)
        expected = [
            'include_formality', 'exclude_formality',
            'include_seasons', 'exclude_seasons',
            'include_style_tags', 'exclude_style_tags',
            'include_color_family', 'exclude_color_family',
            'include_silhouette', 'exclude_silhouette',
            'include_fit', 'exclude_fit',
            'include_length', 'exclude_length',
            'include_sleeves', 'exclude_sleeves',
            'include_neckline', 'exclude_neckline',
            'include_rise', 'exclude_rise',
            'include_coverage', 'exclude_coverage',
            'include_materials',
            'exclude_occasions',
            'on_sale_only',
        ]
        for param in expected:
            assert param in sig.parameters, f"/feed missing param: {param}"

    def test_keyset_endpoint_has_all_filters(self):
        """Verify /feed/keyset endpoint function accepts all filter params."""
        from recs.api_endpoints import get_keyset_feed
        import inspect
        sig = inspect.signature(get_keyset_feed)
        expected = [
            'include_formality', 'exclude_formality',
            'include_seasons', 'exclude_seasons',
            'include_style_tags', 'exclude_style_tags',
            'include_color_family', 'exclude_color_family',
            'include_silhouette', 'exclude_silhouette',
            'include_fit', 'exclude_fit',
            'include_length', 'exclude_length',
            'include_sleeves', 'exclude_sleeves',
            'include_neckline', 'exclude_neckline',
            'include_rise', 'exclude_rise',
            'include_coverage', 'exclude_coverage',
            'include_materials',
            'exclude_occasions',
            'on_sale_only',
            'categories', 'article_types', 'min_price', 'max_price',
            'exclude_brands', 'include_brands',
            'exclude_colors', 'include_colors',
            'include_patterns', 'exclude_patterns',
        ]
        for param in expected:
            assert param in sig.parameters, f"/feed/keyset missing param: {param}"

    def test_keyset_has_parity_with_feed(self):
        """Verify /feed/keyset has the same filter params as /feed."""
        from recs.api_endpoints import get_pipeline_feed, get_keyset_feed
        import inspect

        feed_sig = inspect.signature(get_pipeline_feed)
        keyset_sig = inspect.signature(get_keyset_feed)

        # Params that should be in both (ignoring background_tasks, user, etc.)
        filter_params = [
            'categories', 'article_types', 'gender',
            'exclude_styles', 'include_occasions',
            'min_price', 'max_price',
            'exclude_brands', 'include_brands',
            'exclude_colors', 'include_colors',
            'include_patterns', 'exclude_patterns',
            'include_formality', 'exclude_formality',
            'include_seasons', 'exclude_seasons',
            'include_style_tags', 'exclude_style_tags',
            'include_color_family', 'exclude_color_family',
            'include_silhouette', 'exclude_silhouette',
            'include_fit', 'exclude_fit',
            'include_length', 'exclude_length',
            'include_sleeves', 'exclude_sleeves',
            'include_neckline', 'exclude_neckline',
            'include_rise', 'exclude_rise',
            'include_coverage', 'exclude_coverage',
            'include_materials',
            'exclude_occasions',
            'on_sale_only',
            'cursor', 'page_size', 'session_id',
        ]
        for param in filter_params:
            assert param in feed_sig.parameters, f"/feed missing: {param}"
            assert param in keyset_sig.parameters, f"/feed/keyset missing: {param}"


# =============================================================================
# Candidate Selection include_materials Tests
# =============================================================================

class TestCandidateSelectionMaterials:
    """Test that include_materials parameter is accepted by candidate selection."""

    def test_get_candidates_keyset_accepts_include_materials(self):
        """Verify get_candidates_keyset accepts include_materials parameter."""
        from recs.candidate_selection import CandidateSelectionModule
        import inspect
        sig = inspect.signature(CandidateSelectionModule.get_candidates_keyset)
        assert 'include_materials' in sig.parameters

    def test_retrieve_exploration_keyset_accepts_include_materials(self):
        """Verify _retrieve_exploration_keyset accepts include_materials parameter."""
        from recs.candidate_selection import CandidateSelectionModule
        import inspect
        sig = inspect.signature(CandidateSelectionModule._retrieve_exploration_keyset)
        assert 'include_materials' in sig.parameters
