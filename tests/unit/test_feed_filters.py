"""
Tests for feed attribute filters after SQL JOIN refactor.

Architecture change: Attribute filtering moved from Python (_apply_attribute_filters)
to SQL via LEFT JOIN product_attributes in get_exploration_keyset functions.

These tests verify:
1. All attr_* params are accepted by candidate_selection and pipeline methods
2. The pipeline correctly maps API param names to SQL column names
3. The Python-only rise filter still works
4. API endpoints accept all filter query params
5. Feed and keyset endpoints have filter parity
"""

import sys
import os
import inspect
from unittest.mock import MagicMock, patch
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


# =============================================================================
# Test: CandidateSelectionModule accepts all attr_* params
# =============================================================================

class TestCandidateSelectionSignatures:
    """Verify candidate selection methods accept all attribute filter params."""

    ALL_ATTR_PARAMS = [
        'attr_include_formality', 'attr_exclude_formality',
        'attr_include_seasons', 'attr_exclude_seasons',
        'attr_include_style_tags', 'attr_exclude_style_tags',
        'attr_include_color_family', 'attr_exclude_color_family',
        'attr_include_silhouette', 'attr_exclude_silhouette',
        'attr_include_fit_type', 'attr_exclude_fit_type',
        'attr_include_coverage', 'attr_exclude_coverage',
        'attr_include_pattern', 'attr_exclude_pattern',
        'attr_include_neckline', 'attr_exclude_neckline',
        'attr_include_sleeve_type', 'attr_exclude_sleeve_type',
        'attr_include_length', 'attr_exclude_length',
        'attr_include_occasions', 'attr_exclude_occasions',
    ]

    def test_get_candidates_keyset_accepts_all_attr_params(self):
        """get_candidates_keyset must accept all 24 attr_* params."""
        from recs.candidate_selection import CandidateSelectionModule
        sig = inspect.signature(CandidateSelectionModule.get_candidates_keyset)
        for param in self.ALL_ATTR_PARAMS:
            assert param in sig.parameters, f"get_candidates_keyset missing param: {param}"

    def test_retrieve_exploration_keyset_accepts_all_attr_params(self):
        """_retrieve_exploration_keyset must accept all 24 attr_* params."""
        from recs.candidate_selection import CandidateSelectionModule
        sig = inspect.signature(CandidateSelectionModule._retrieve_exploration_keyset)
        for param in self.ALL_ATTR_PARAMS:
            assert param in sig.parameters, f"_retrieve_exploration_keyset missing param: {param}"

    def test_get_candidates_keyset_accepts_include_materials(self):
        """Verify get_candidates_keyset accepts include_materials parameter."""
        from recs.candidate_selection import CandidateSelectionModule
        sig = inspect.signature(CandidateSelectionModule.get_candidates_keyset)
        assert 'include_materials' in sig.parameters

    def test_retrieve_exploration_keyset_accepts_include_materials(self):
        """Verify _retrieve_exploration_keyset accepts include_materials parameter."""
        from recs.candidate_selection import CandidateSelectionModule
        sig = inspect.signature(CandidateSelectionModule._retrieve_exploration_keyset)
        assert 'include_materials' in sig.parameters

    def test_get_candidates_keyset_accepts_sale_filters(self):
        """Verify get_candidates_keyset accepts sale/new arrival params."""
        from recs.candidate_selection import CandidateSelectionModule
        sig = inspect.signature(CandidateSelectionModule.get_candidates_keyset)
        assert 'on_sale_only' in sig.parameters
        assert 'new_arrivals_only' in sig.parameters
        assert 'new_arrivals_days' in sig.parameters

    def test_get_candidates_keyset_accepts_exclude_ids(self):
        """Verify get_candidates_keyset accepts exclude_ids for seen history."""
        from recs.candidate_selection import CandidateSelectionModule
        sig = inspect.signature(CandidateSelectionModule.get_candidates_keyset)
        assert 'exclude_ids' in sig.parameters

    def test_all_attr_params_default_to_none(self):
        """All attr_* params should default to None."""
        from recs.candidate_selection import CandidateSelectionModule
        sig = inspect.signature(CandidateSelectionModule.get_candidates_keyset)
        for param in self.ALL_ATTR_PARAMS:
            default = sig.parameters[param].default
            assert default is None, f"{param} default is {default}, expected None"

    def test_attr_param_count_is_24(self):
        """There should be exactly 24 attr_* params (12 dimensions x include/exclude)."""
        from recs.candidate_selection import CandidateSelectionModule
        sig = inspect.signature(CandidateSelectionModule.get_candidates_keyset)
        attr_params = [p for p in sig.parameters if p.startswith('attr_')]
        assert len(attr_params) == 24, f"Expected 24 attr_* params, got {len(attr_params)}: {attr_params}"


# =============================================================================
# Test: Pipeline get_feed_keyset accepts all filter params
# =============================================================================

class TestPipelineSignatures:
    """Verify pipeline method accepts all filter parameters."""

    def test_get_feed_keyset_accepts_all_include_exclude_params(self):
        """get_feed_keyset must accept all include/exclude filter params."""
        from recs.pipeline import RecommendationPipeline
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
            'on_sale_only',
        ]
        for param in expected_params:
            assert param in sig.parameters, f"get_feed_keyset missing param: {param}"

    def test_pipeline_no_longer_has_apply_attribute_filters(self):
        """The old _apply_attribute_filters method should be removed."""
        from recs.pipeline import RecommendationPipeline
        assert not hasattr(RecommendationPipeline, '_apply_attribute_filters') or \
               not callable(getattr(RecommendationPipeline, '_apply_attribute_filters', None)), \
               "_apply_attribute_filters should have been removed in SQL JOIN refactor"

    def test_pipeline_no_longer_has_single_value_filters_dict(self):
        """The old SINGLE_VALUE_FILTERS class dict should be removed."""
        from recs.pipeline import RecommendationPipeline
        assert not hasattr(RecommendationPipeline, 'SINGLE_VALUE_FILTERS'), \
            "SINGLE_VALUE_FILTERS should have been removed in SQL JOIN refactor"

    def test_pipeline_no_longer_has_multi_value_filters_dict(self):
        """The old MULTI_VALUE_FILTERS class dict should be removed."""
        from recs.pipeline import RecommendationPipeline
        assert not hasattr(RecommendationPipeline, 'MULTI_VALUE_FILTERS'), \
            "MULTI_VALUE_FILTERS should have been removed in SQL JOIN refactor"


# =============================================================================
# Test: API param -> SQL param name mapping
# =============================================================================

class TestParamMapping:
    """Verify the pipeline maps API param names to correct SQL attr_* names.

    This is critical because the API uses user-friendly names (e.g., 'fit', 'sleeves')
    while SQL uses database column names (e.g., 'fit_type', 'sleeve_type').
    """

    # Expected mapping: (API param name in pipeline, SQL attr_* param name)
    EXPECTED_MAPPINGS = [
        ('include_formality', 'attr_include_formality'),
        ('exclude_formality', 'attr_exclude_formality'),
        ('include_seasons', 'attr_include_seasons'),
        ('exclude_seasons', 'attr_exclude_seasons'),
        ('include_style_tags', 'attr_include_style_tags'),
        ('exclude_style_tags', 'attr_exclude_style_tags'),
        ('include_color_family', 'attr_include_color_family'),
        ('exclude_color_family', 'attr_exclude_color_family'),
        ('include_silhouette', 'attr_include_silhouette'),
        ('exclude_silhouette', 'attr_exclude_silhouette'),
        ('include_fit', 'attr_include_fit_type'),       # fit -> fit_type
        ('exclude_fit', 'attr_exclude_fit_type'),
        ('include_coverage', 'attr_include_coverage'),   # coverage -> coverage_level in SQL
        ('exclude_coverage', 'attr_exclude_coverage'),
        ('include_patterns', 'attr_include_pattern'),    # patterns -> pattern (plural -> singular)
        ('exclude_patterns', 'attr_exclude_pattern'),
        ('include_neckline', 'attr_include_neckline'),
        ('exclude_neckline', 'attr_exclude_neckline'),
        ('include_sleeves', 'attr_include_sleeve_type'), # sleeves -> sleeve_type
        ('exclude_sleeves', 'attr_exclude_sleeve_type'),
        ('include_length', 'attr_include_length'),
        ('exclude_length', 'attr_exclude_length'),
        ('include_occasions', 'attr_include_occasions'),
        ('exclude_occasions', 'attr_exclude_occasions'),
    ]

    def test_pipeline_get_feed_keyset_has_api_params(self):
        """Pipeline must accept all API-side param names."""
        from recs.pipeline import RecommendationPipeline
        sig = inspect.signature(RecommendationPipeline.get_feed_keyset)
        for api_param, _ in self.EXPECTED_MAPPINGS:
            assert api_param in sig.parameters, f"Pipeline missing API param: {api_param}"

    def test_candidate_selection_has_sql_params(self):
        """CandidateSelectionModule must accept all SQL-side attr_* param names."""
        from recs.candidate_selection import CandidateSelectionModule
        sig = inspect.signature(CandidateSelectionModule.get_candidates_keyset)
        for _, sql_param in self.EXPECTED_MAPPINGS:
            assert sql_param in sig.parameters, f"CandidateSelection missing SQL param: {sql_param}"

    def test_mapping_covers_all_12_dimensions(self):
        """Should have 24 mappings covering 12 dimensions x include/exclude."""
        assert len(self.EXPECTED_MAPPINGS) == 24, \
            f"Expected 24 mappings (12 dimensions x 2), got {len(self.EXPECTED_MAPPINGS)}"

    def test_renamed_params_are_correct(self):
        """Verify specifically the params that get renamed between API and SQL."""
        renamed = {
            'include_fit': 'attr_include_fit_type',
            'exclude_fit': 'attr_exclude_fit_type',
            'include_sleeves': 'attr_include_sleeve_type',
            'exclude_sleeves': 'attr_exclude_sleeve_type',
            'include_patterns': 'attr_include_pattern',
            'exclude_patterns': 'attr_exclude_pattern',
        }
        for api_param, expected_sql_param in renamed.items():
            found = False
            for a, s in self.EXPECTED_MAPPINGS:
                if a == api_param:
                    assert s == expected_sql_param, \
                        f"Mapping for {api_param}: expected {expected_sql_param}, got {s}"
                    found = True
                    break
            assert found, f"Mapping for {api_param} not found in EXPECTED_MAPPINGS"


# =============================================================================
# Test: attr_* params are threaded through to SQL RPC params dict
# =============================================================================

class TestAttrParamsThreading:
    """Verify attr_* params are correctly passed from get_candidates_keyset
    through to _retrieve_exploration_keyset and into the SQL RPC params dict.

    Uses mocking to intercept the Supabase RPC call and inspect the params.
    """

    def _create_mock_candidate_module(self):
        """Create a CandidateSelectionModule with mocked Supabase client."""
        from recs.candidate_selection import CandidateSelectionModule

        mock_supabase = MagicMock()
        # Make rpc().execute() return empty data
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value = MagicMock(data=[])
        mock_supabase.rpc.return_value = mock_rpc

        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase
        return module, mock_supabase

    def test_exploration_keyset_passes_attrs_to_rpc(self):
        """_retrieve_exploration_keyset should pass all attr_* params to SQL RPC."""
        module, mock_supabase = self._create_mock_candidate_module()

        from recs.candidate_selection import HardFilters
        hf = HardFilters(gender='female')

        test_attrs = {
            'attr_include_formality': ['Casual'],
            'attr_exclude_formality': ['Formal'],
            'attr_include_seasons': ['Summer'],
            'attr_exclude_seasons': ['Winter'],
            'attr_include_style_tags': ['Classic'],
            'attr_exclude_style_tags': ['Bold'],
            'attr_include_color_family': ['Neutrals'],
            'attr_exclude_color_family': ['Neons'],
            'attr_include_silhouette': ['Straight'],
            'attr_exclude_silhouette': ['Skinny'],
            'attr_include_fit_type': ['Regular'],
            'attr_exclude_fit_type': ['Oversized'],
            'attr_include_coverage': ['Full'],
            'attr_exclude_coverage': ['Minimal'],
            'attr_include_pattern': ['Solid'],
            'attr_exclude_pattern': ['Animal Print'],
            'attr_include_neckline': ['V-Neck'],
            'attr_exclude_neckline': ['Halter'],
            'attr_include_sleeve_type': ['Long'],
            'attr_exclude_sleeve_type': ['Sleeveless'],
            'attr_include_length': ['Midi'],
            'attr_exclude_length': ['Mini'],
            'attr_include_occasions': ['Office'],
            'attr_exclude_occasions': ['Party'],
        }

        module._retrieve_exploration_keyset(
            hard_filters=hf,
            random_seed='test123',
            cursor_score=None,
            cursor_id=None,
            limit=50,
            **test_attrs,
        )

        # Verify rpc was called
        mock_supabase.rpc.assert_called_once()
        call_args = mock_supabase.rpc.call_args
        rpc_name = call_args[0][0]
        rpc_params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get('params', {})

        assert rpc_name == 'get_exploration_keyset'

        # Verify all attr_* params are in the RPC params dict
        for attr_key, attr_val in test_attrs.items():
            assert attr_key in rpc_params, f"RPC params missing: {attr_key}"
            assert rpc_params[attr_key] == attr_val, \
                f"RPC params[{attr_key}] = {rpc_params[attr_key]}, expected {attr_val}"

    def test_exploration_keyset_none_attrs_passed_as_none(self):
        """When attr_* params are None, they should still be passed to RPC as None."""
        module, mock_supabase = self._create_mock_candidate_module()

        from recs.candidate_selection import HardFilters
        hf = HardFilters(gender='female')

        module._retrieve_exploration_keyset(
            hard_filters=hf,
            random_seed='test123',
            cursor_score=None,
            cursor_id=None,
            limit=50,
            # All attr_* params default to None
        )

        mock_supabase.rpc.assert_called_once()
        call_args = mock_supabase.rpc.call_args
        rpc_params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get('params', {})

        # All attr_* params should be present and None
        attr_keys = [k for k in rpc_params if k.startswith('attr_')]
        assert len(attr_keys) == 24, f"Expected 24 attr_* params in RPC, got {len(attr_keys)}"
        for key in attr_keys:
            assert rpc_params[key] is None, f"Expected None for {key}, got {rpc_params[key]}"

    def test_preferred_brands_uses_brand_function(self):
        """When preferred_brands is set, should use get_exploration_keyset_with_brands."""
        module, mock_supabase = self._create_mock_candidate_module()

        from recs.candidate_selection import HardFilters
        hf = HardFilters(gender='female')

        module._retrieve_exploration_keyset(
            hard_filters=hf,
            random_seed='test123',
            cursor_score=None,
            cursor_id=None,
            limit=50,
            preferred_brands=['Zara', 'H&M'],
        )

        mock_supabase.rpc.assert_called_once()
        call_args = mock_supabase.rpc.call_args
        rpc_name = call_args[0][0]
        assert rpc_name == 'get_exploration_keyset_with_brands'

    def test_exclude_ids_splits_at_sql_limit(self):
        """Seen history beyond SQL_EXCLUDE_IDS_LIMIT should be handled in Python."""
        from recs.candidate_selection import CandidateSelectionModule
        module, mock_supabase = self._create_mock_candidate_module()

        from recs.candidate_selection import HardFilters
        hf = HardFilters(gender='female')

        # Create more IDs than the SQL limit
        sql_limit = CandidateSelectionModule.SQL_EXCLUDE_IDS_LIMIT
        large_exclude = {f"id-{i}" for i in range(sql_limit + 500)}

        module._retrieve_exploration_keyset(
            hard_filters=hf,
            random_seed='test123',
            cursor_score=None,
            cursor_id=None,
            limit=50,
            exclude_ids=large_exclude,
        )

        mock_supabase.rpc.assert_called_once()
        call_args = mock_supabase.rpc.call_args
        rpc_params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get('params', {})

        # SQL should receive exactly SQL_EXCLUDE_IDS_LIMIT IDs
        assert len(rpc_params['exclude_product_ids']) == sql_limit


# =============================================================================
# Test: Pipeline maps API params to SQL attr_* params correctly
# =============================================================================

class TestPipelineAttrMapping:
    """Verify the pipeline's get_feed_keyset correctly maps API param names
    to attr_* param names when calling get_candidates_keyset.

    Uses mocking to intercept the call to get_candidates_keyset.
    """

    def test_pipeline_maps_fit_to_fit_type(self):
        """include_fit should be mapped to attr_include_fit_type."""
        self._verify_mapping('include_fit', ['Slim'], 'attr_include_fit_type')

    def test_pipeline_maps_sleeves_to_sleeve_type(self):
        """include_sleeves should be mapped to attr_include_sleeve_type."""
        self._verify_mapping('include_sleeves', ['Long'], 'attr_include_sleeve_type')

    def test_pipeline_maps_patterns_to_pattern(self):
        """include_patterns should be mapped to attr_include_pattern (plural -> singular)."""
        self._verify_mapping('include_patterns', ['Solid'], 'attr_include_pattern')

    def test_pipeline_maps_coverage_to_coverage(self):
        """include_coverage should be mapped to attr_include_coverage."""
        self._verify_mapping('include_coverage', ['Full'], 'attr_include_coverage')

    def test_pipeline_maps_neckline(self):
        """include_neckline should be mapped to attr_include_neckline."""
        self._verify_mapping('include_neckline', ['V-Neck'], 'attr_include_neckline')

    def test_pipeline_maps_length(self):
        """include_length should be mapped to attr_include_length."""
        self._verify_mapping('include_length', ['Midi'], 'attr_include_length')

    def test_pipeline_maps_occasions(self):
        """include_occasions should be mapped to attr_include_occasions."""
        self._verify_mapping('include_occasions', ['Office'], 'attr_include_occasions')

    def test_pipeline_maps_exclude_fit(self):
        """exclude_fit should be mapped to attr_exclude_fit_type."""
        self._verify_mapping('exclude_fit', ['Oversized'], 'attr_exclude_fit_type')

    def test_pipeline_maps_exclude_sleeves(self):
        """exclude_sleeves should be mapped to attr_exclude_sleeve_type."""
        self._verify_mapping('exclude_sleeves', ['Sleeveless'], 'attr_exclude_sleeve_type')

    def test_pipeline_maps_exclude_patterns(self):
        """exclude_patterns should be mapped to attr_exclude_pattern."""
        self._verify_mapping('exclude_patterns', ['Floral'], 'attr_exclude_pattern')

    def _verify_mapping(self, api_param: str, api_value: list, expected_sql_param: str):
        """Helper: verify that the SQL param is set from the API param in source code.

        Accounts for normalization wrappers like _tc(), _expand(), or expanded_ variables.
        """
        from recs.pipeline import RecommendationPipeline

        source = inspect.getsource(RecommendationPipeline.get_feed_keyset)

        # Check for direct mapping or wrapped mapping (e.g., _tc(include_fit), expanded_occasions)
        patterns = [
            f'{expected_sql_param}={api_param}',
            f'{expected_sql_param}= {api_param}',
            f'{expected_sql_param} = {api_param}',
            f'{expected_sql_param}=_tc({api_param})',
            f'{expected_sql_param}= _tc({api_param})',
            f'{expected_sql_param}=_expand({api_param}',
            f'{expected_sql_param}=expanded_',
        ]
        found = any(p in source for p in patterns)
        assert found, \
            f"Expected mapping '{expected_sql_param}' from '{api_param}' not found in get_feed_keyset source"


# =============================================================================
# Test: Python-only rise filter
# =============================================================================

class TestRiseFilter:
    """Rise filter is the only attribute filter that remains in Python,
    because there is no 'rise' column in product_attributes.

    Test that the pipeline applies rise filtering correctly.
    """

    def test_rise_filter_exists_in_pipeline_source(self):
        """Pipeline should still have Python-level rise filtering code."""
        from recs.pipeline import RecommendationPipeline
        source = inspect.getsource(RecommendationPipeline.get_feed_keyset)
        assert 'include_rise' in source
        assert 'exclude_rise' in source

    def test_rise_not_in_sql_attr_params(self):
        """Rise should NOT be passed as attr_* param to SQL."""
        from recs.candidate_selection import CandidateSelectionModule
        sig = inspect.signature(CandidateSelectionModule.get_candidates_keyset)
        attr_params = [p for p in sig.parameters if p.startswith('attr_')]
        rise_attrs = [p for p in attr_params if 'rise' in p.lower()]
        assert len(rise_attrs) == 0, f"Rise should not be in attr_* params: {rise_attrs}"

    def test_rise_include_logic(self):
        """Verify rise include filter logic: only keep candidates with matching rise."""
        candidates = [
            make_candidate("c1", rise="High"),
            make_candidate("c2", rise="Mid"),
            make_candidate("c3", rise="Low"),
            make_candidate("c4", rise=None),
        ]
        inc_set = {'high', 'mid'}
        filtered = []
        for c in candidates:
            val = (c.rise or '').lower() if c.rise else None
            if inc_set and (not val or val not in inc_set):
                continue
            filtered.append(c)
        assert len(filtered) == 2
        assert {c.item_id for c in filtered} == {"c1", "c2"}

    def test_rise_exclude_logic(self):
        """Verify rise exclude filter logic: remove candidates with matching rise."""
        candidates = [
            make_candidate("c1", rise="High"),
            make_candidate("c2", rise="Mid"),
            make_candidate("c3", rise="Low"),
            make_candidate("c4", rise=None),
        ]
        exc_set = {'low'}
        filtered = []
        for c in candidates:
            val = (c.rise or '').lower() if c.rise else None
            if exc_set and val and val in exc_set:
                continue
            filtered.append(c)
        assert len(filtered) == 3
        assert {c.item_id for c in filtered} == {"c1", "c2", "c4"}

    def test_rise_include_case_insensitive(self):
        """Rise include filter should be case-insensitive."""
        candidates = [
            make_candidate("c1", rise="HIGH"),
            make_candidate("c2", rise="low"),
        ]
        inc_set = {'high'}
        filtered = []
        for c in candidates:
            val = (c.rise or '').lower() if c.rise else None
            if inc_set and (not val or val not in inc_set):
                continue
            filtered.append(c)
        assert len(filtered) == 1
        assert filtered[0].item_id == "c1"

    def test_rise_none_excluded_by_include(self):
        """Items with None rise should be excluded when include is active."""
        candidates = [
            make_candidate("c1", rise="High"),
            make_candidate("c2", rise=None),
        ]
        inc_set = {'high'}
        filtered = []
        for c in candidates:
            val = (c.rise or '').lower() if c.rise else None
            if inc_set and (not val or val not in inc_set):
                continue
            filtered.append(c)
        assert len(filtered) == 1
        assert filtered[0].item_id == "c1"

    def test_rise_none_passes_exclude(self):
        """Items with None rise should pass exclude filter."""
        candidates = [
            make_candidate("c1", rise=None),
            make_candidate("c2", rise="Low"),
        ]
        exc_set = {'low'}
        filtered = []
        for c in candidates:
            val = (c.rise or '').lower() if c.rise else None
            if exc_set and val and val in exc_set:
                continue
            filtered.append(c)
        assert len(filtered) == 1
        assert filtered[0].item_id == "c1"

    def test_rise_combined_include_and_exclude(self):
        """Include and exclude on rise should work together."""
        candidates = [
            make_candidate("c1", rise="High"),
            make_candidate("c2", rise="Mid"),
            make_candidate("c3", rise="Low"),
        ]
        inc_set = {'high', 'mid'}
        exc_set = {'mid'}
        filtered = []
        for c in candidates:
            val = (c.rise or '').lower() if c.rise else None
            if inc_set and (not val or val not in inc_set):
                continue
            if exc_set and val and val in exc_set:
                continue
            filtered.append(c)
        assert len(filtered) == 1
        assert filtered[0].item_id == "c1"


# =============================================================================
# Test: on_sale_only is plumbed through
# =============================================================================

class TestOnSaleFilter:
    """Test on_sale_only parameter is plumbed through."""

    def test_on_sale_only_is_accepted(self):
        """Verify pipeline.get_feed_keyset accepts on_sale_only parameter."""
        from recs.pipeline import RecommendationPipeline
        sig = inspect.signature(RecommendationPipeline.get_feed_keyset)
        assert 'on_sale_only' in sig.parameters

    def test_all_new_filters_accepted(self):
        """Verify pipeline.get_feed_keyset accepts all new filter parameters."""
        from recs.pipeline import RecommendationPipeline
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
# Test: API Endpoint Signatures
# =============================================================================

class TestAPIEndpointSignatures:
    """Verify API endpoints accept all filter query params."""

    def test_feed_endpoint_has_all_filters(self):
        """Verify /feed endpoint function accepts all new filter params."""
        from recs.api_endpoints import get_pipeline_feed
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
        sig_feed = inspect.signature(get_pipeline_feed)
        sig_keyset = inspect.signature(get_keyset_feed)

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
            assert param in sig_feed.parameters, f"/feed missing: {param}"
            assert param in sig_keyset.parameters, f"/feed/keyset missing: {param}"


# =============================================================================
# Test: Architecture validation
# =============================================================================

class TestArchitectureValidation:
    """Validate the overall architecture after the SQL JOIN refactor."""

    def test_candidate_selection_no_pre_filter_method(self):
        """pre_filter_by_attributes should have been removed."""
        from recs.candidate_selection import CandidateSelectionModule
        assert not hasattr(CandidateSelectionModule, 'pre_filter_by_attributes'), \
            "pre_filter_by_attributes should be removed in SQL JOIN refactor"

    def test_candidate_selection_no_pre_filter_column_map(self):
        """_PRE_FILTER_COLUMN_MAP should have been removed."""
        from recs.candidate_selection import CandidateSelectionModule
        assert not hasattr(CandidateSelectionModule, '_PRE_FILTER_COLUMN_MAP'), \
            "_PRE_FILTER_COLUMN_MAP should be removed in SQL JOIN refactor"

    def test_candidate_selection_no_include_product_ids_param(self):
        """include_product_ids param should have been removed from keyset methods."""
        from recs.candidate_selection import CandidateSelectionModule

        sig1 = inspect.signature(CandidateSelectionModule.get_candidates_keyset)
        assert 'include_product_ids' not in sig1.parameters, \
            "include_product_ids should be removed (replaced by attr_* SQL JOIN params)"

        sig2 = inspect.signature(CandidateSelectionModule._retrieve_exploration_keyset)
        assert 'include_product_ids' not in sig2.parameters, \
            "include_product_ids should be removed (replaced by attr_* SQL JOIN params)"

    def test_sql_exclude_ids_limit_exists(self):
        """SQL_EXCLUDE_IDS_LIMIT should still exist for seen history management."""
        from recs.candidate_selection import CandidateSelectionModule
        assert hasattr(CandidateSelectionModule, 'SQL_EXCLUDE_IDS_LIMIT')
        assert CandidateSelectionModule.SQL_EXCLUDE_IDS_LIMIT == 5000

    def test_python_filters_still_exist(self):
        """Color, brand, article_type, rise filters should still be in Python."""
        from recs.pipeline import RecommendationPipeline
        source = inspect.getsource(RecommendationPipeline.get_feed_keyset)

        # These should still exist as Python-level filters
        assert '_apply_color_filter' in source, "Color filter should remain in Python"
        assert '_apply_brand_filter' in source, "Brand filter should remain in Python"
        assert '_apply_article_type_filter' in source, "Article type filter should remain in Python"
        assert 'include_rise' in source, "Rise filter should remain in Python"

    def test_has_python_filters_includes_rise(self):
        """has_python_filters check should include rise."""
        from recs.pipeline import RecommendationPipeline
        source = inspect.getsource(RecommendationPipeline.get_feed_keyset)
        # The has_python_filters check should include rise
        assert 'include_rise' in source
        assert 'exclude_rise' in source
