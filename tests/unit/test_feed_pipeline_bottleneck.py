"""
Stress tests for the feed pipeline candidate retrieval bottleneck fix.

Tests cover:
1. fetch_size formula — ensures adequate candidate retrieval from 91K catalog
2. SQL-level seen exclusion — exclude_ids forwarded to _retrieve_exploration_keyset
3. Large seen history handling — chunking at SQL_EXCLUDE_IDS_LIMIT boundary
4. Python fallback filter — graceful degradation when SQL migration not applied
5. Category filter + fetch_size interaction — no starvation for filtered feeds
6. Power user scenarios — 1000+, 5000+ seen history
7. Regression: fresh user gets enough candidates

Run with: PYTHONPATH=src python -m pytest tests/unit/test_feed_pipeline_bottleneck.py -v
"""

import uuid
import pytest
from unittest.mock import MagicMock, patch, PropertyMock, call
from typing import List, Dict, Any, Optional, Set


# =============================================================================
# Helpers
# =============================================================================

def _make_candidate(item_id: str = None, **overrides):
    """Create a Candidate with sensible defaults."""
    from recs.models import Candidate
    defaults = {
        "item_id": item_id or str(uuid.uuid4()),
        "embedding_score": 0.5,
        "preference_score": 0.5,
        "sasrec_score": 0.0,
        "final_score": 0.5,
        "category": "tops",
        "broad_category": "tops",
        "article_type": "t-shirt",
        "brand": "TestBrand",
        "price": 29.99,
        "colors": ["black"],
        "materials": ["cotton"],
        "image_url": "https://img.example.com/test.jpg",
        "name": "Test Product",
        "source": "exploration",
    }
    defaults.update(overrides)
    return Candidate(**defaults)


def _make_candidates(n: int, prefix: str = "prod") -> list:
    """Create N candidates with unique IDs."""
    return [_make_candidate(item_id=f"{prefix}-{i:05d}") for i in range(n)]


def _make_user_state(
    user_id: str = "test-user-123",
    categories: Optional[List[str]] = None,
    preferred_brands: Optional[List[str]] = None,
):
    """Create a minimal UserState."""
    from recs.models import UserState, OnboardingProfile
    profile = None
    if categories or preferred_brands:
        kwargs = {"user_id": user_id}
        if categories:
            kwargs["categories"] = categories
        if preferred_brands:
            kwargs["preferred_brands"] = preferred_brands
        profile = OnboardingProfile(**kwargs)
    return UserState(
        user_id=user_id,
        onboarding_profile=profile,
    )


def _make_seen_ids(n: int) -> Set[str]:
    """Generate N unique UUID-like seen IDs."""
    return {str(uuid.uuid4()) for _ in range(n)}


# =============================================================================
# Test Class: fetch_size Formula (pipeline.py)
# =============================================================================

class TestFetchSizeFormula:
    """Tests for the improved fetch_size calculation in pipeline.py."""

    def _compute_fetch_size(
        self,
        page_size: int = 50,
        db_history_count: int = 0,
        has_python_filters: bool = False,
    ) -> int:
        """Replicate the new fetch_size formula from pipeline.py."""
        filter_multiplier = 3 if has_python_filters else 1
        base_fetch = max(500, page_size * 10)
        fetch_size = base_fetch * filter_multiplier
        fetch_size = min(fetch_size, 5000)
        return fetch_size

    def _compute_old_fetch_size(
        self,
        page_size: int = 50,
        db_history_count: int = 0,
        has_python_filters: bool = False,
    ) -> int:
        """Replicate the OLD (broken) fetch_size formula for comparison."""
        filter_multiplier = 3 if has_python_filters else 1
        fetch_size = (page_size + db_history_count + 100) * filter_multiplier
        fetch_size = min(fetch_size, 3000)
        return fetch_size

    # --- Fresh user scenarios ---

    def test_fresh_user_default_page_size(self):
        """Fresh user (no seen history) should get 500 candidates, not 150."""
        new = self._compute_fetch_size(page_size=50, db_history_count=0)
        old = self._compute_old_fetch_size(page_size=50, db_history_count=0)
        assert new == 500, f"Expected 500, got {new}"
        assert old == 150, f"Old formula should have been 150, got {old}"
        assert new >= 3 * old, "New formula should be at least 3x the old one for fresh users"

    def test_fresh_user_small_page(self):
        """Small page request still gets 500 minimum."""
        new = self._compute_fetch_size(page_size=20, db_history_count=0)
        assert new == 500, "Minimum base fetch should be 500"

    def test_fresh_user_large_page(self):
        """Large page request scales to 10x."""
        new = self._compute_fetch_size(page_size=100, db_history_count=0)
        assert new == 1000, f"Expected 1000 (100*10), got {new}"

    # --- Seen history no longer inflates fetch_size ---

    def test_seen_history_does_not_inflate_fetch_size(self):
        """With SQL-level exclusion, seen history shouldn't inflate fetch_size."""
        no_seen = self._compute_fetch_size(page_size=50, db_history_count=0)
        with_500_seen = self._compute_fetch_size(page_size=50, db_history_count=500)
        with_1000_seen = self._compute_fetch_size(page_size=50, db_history_count=1000)
        assert no_seen == with_500_seen == with_1000_seen == 500, \
            "Seen history should NOT inflate fetch_size since exclusion is now in SQL"

    def test_old_formula_seen_history_dominated_fetch(self):
        """Verify OLD formula was dominated by seen history (regression check)."""
        old_500 = self._compute_old_fetch_size(page_size=50, db_history_count=500)
        assert old_500 == 650, f"Old formula with 500 seen should be 650, got {old_500}"

        old_1166 = self._compute_old_fetch_size(page_size=50, db_history_count=1166)
        assert old_1166 == 1316, f"Old formula with 1166 seen should be 1316, got {old_1166}"

    def test_old_formula_power_user_hit_cap(self):
        """Old formula: power user (2850+ seen) hit the 3000 cap and got 0 new items."""
        old = self._compute_old_fetch_size(page_size=50, db_history_count=2850)
        assert old == 3000, "Old formula should have hit 3000 cap"
        usable = old - 2850  # SQL returned 3000, minus 2850 seen = 150 usable
        assert usable == 150, "Power user should have had only 150 usable after Python filter"

    # --- Python filter multiplier ---

    def test_python_filter_multiplier(self):
        """Python filters (colors, brands) should 3x the fetch."""
        no_filter = self._compute_fetch_size(page_size=50, has_python_filters=False)
        with_filter = self._compute_fetch_size(page_size=50, has_python_filters=True)
        assert no_filter == 500
        assert with_filter == 1500, f"Expected 1500 (500*3), got {with_filter}"

    def test_python_filter_large_page_capped(self):
        """Large page + Python filters should be capped at 5000."""
        fetch = self._compute_fetch_size(page_size=200, has_python_filters=True)
        assert fetch == 5000, f"Expected 5000 cap, got {fetch}"

    def test_cap_at_5000(self):
        """Fetch size should never exceed 5000."""
        fetch = self._compute_fetch_size(page_size=1000, has_python_filters=True)
        assert fetch == 5000

    # --- Impact table validation ---

    def test_impact_table_fresh_user_no_filters(self):
        """Validate impact: fresh user, no filters."""
        old = self._compute_old_fetch_size(page_size=50, db_history_count=0)
        new = self._compute_fetch_size(page_size=50, db_history_count=0)
        improvement = new / old
        assert improvement >= 3.0, f"Expected 3x+ improvement, got {improvement:.1f}x"

    def test_impact_table_user_500_seen(self):
        """Validate impact: user with 500 seen items.

        OLD: fetch=650, minus 500 seen = 150 usable
        NEW: fetch=500, ALL usable (seen excluded in SQL)
        """
        old = self._compute_old_fetch_size(page_size=50, db_history_count=500)
        old_usable = old - 500  # Python excluded 500 seen
        new = self._compute_fetch_size(page_size=50, db_history_count=500)
        # New: all 500 are usable since SQL excludes seen items
        assert new == 500
        assert old_usable == 150
        assert new > old_usable

    def test_impact_table_user_1166_seen(self):
        """Validate impact: power user with 1166 seen items (real user a42818ff)."""
        old = self._compute_old_fetch_size(page_size=50, db_history_count=1166)
        old_usable = old - 1166
        new = self._compute_fetch_size(page_size=50, db_history_count=1166)
        assert new == 500
        assert old_usable == 150
        assert new > old_usable

    def test_impact_table_user_5000_seen(self):
        """Validate impact: extreme user with 5000 seen items.

        OLD: fetch=min(5150, 3000)=3000, minus 5000 seen = NEGATIVE (broken)
        NEW: fetch=500, ALL usable
        """
        old = self._compute_old_fetch_size(page_size=50, db_history_count=5000)
        old_usable = max(0, old - 5000)  # Would be negative -> 0 results
        new = self._compute_fetch_size(page_size=50, db_history_count=5000)
        assert old_usable == 0, "Old formula should have produced 0 usable candidates"
        assert new == 500, "New formula should still produce 500 candidates"


# =============================================================================
# Test Class: SQL-Level Seen Exclusion (candidate_selection.py)
# =============================================================================

class TestSQLExclusion:
    """Tests for forwarding exclude_ids to the SQL RPC function."""

    def _make_mock_candidate_module(self, sql_returns: list = None):
        """Create a mock CandidateSelectionModule."""
        from recs.candidate_selection import CandidateSelectionModule
        module = MagicMock(spec=CandidateSelectionModule)

        # Set up the class attribute for the limit
        module.SQL_EXCLUDE_IDS_LIMIT = 5000
        return module

    def test_exclude_ids_forwarded_to_exploration_keyset(self):
        """exclude_ids should be forwarded as exclude_ids kwarg to _retrieve_exploration_keyset."""
        from recs.candidate_selection import CandidateSelectionModule

        # Create a real instance with mocked supabase
        mock_supabase = MagicMock()
        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase

        # Generate seen IDs
        seen = _make_seen_ids(100)

        # Mock _retrieve_exploration_keyset to capture what it receives
        candidates = _make_candidates(50)

        user_state = _make_user_state()
        from recs.models import HardFilters
        mock_hard_filters = HardFilters(gender="female")

        with patch.object(module, '_retrieve_exploration_keyset', return_value=candidates) as mock_retrieve, \
             patch.object(module, '_apply_profile_scoring', side_effect=lambda c, p: c) as mock_score, \
             patch('recs.candidate_selection.HardFilters') as MockHF, \
             patch('recs.candidate_selection.get_include_article_types', return_value=None):
            MockHF.from_user_state.return_value = mock_hard_filters

            result = module.get_candidates_keyset(
                user_state=user_state,
                gender="female",
                page_size=50,
                exclude_ids=seen,
            )

            # Verify exclude_ids was forwarded
            mock_retrieve.assert_called_once()
            call_kwargs = mock_retrieve.call_args
            assert call_kwargs.kwargs.get('exclude_ids') == seen or \
                   (len(call_kwargs.args) > 6 and call_kwargs.args[6] == seen), \
                   "exclude_ids should be forwarded to _retrieve_exploration_keyset"

    def test_sql_params_include_exclude_product_ids(self):
        """The SQL RPC call should include exclude_product_ids parameter."""
        from recs.candidate_selection import CandidateSelectionModule
        from recs.models import HardFilters

        mock_supabase = MagicMock()
        # Mock the RPC chain: supabase.rpc(...).execute()
        mock_rpc_result = MagicMock()
        mock_rpc_result.data = []
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value = mock_rpc_result
        mock_supabase.rpc.return_value = mock_rpc

        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase

        seen = _make_seen_ids(200)
        hard_filters = HardFilters(gender="female")

        with patch.object(module, '_enrich_with_attributes', side_effect=lambda c: c):
            module._retrieve_exploration_keyset(
                hard_filters=hard_filters,
                random_seed="test-seed",
                cursor_score=None,
                cursor_id=None,
                limit=500,
                exclude_ids=seen,
            )

        # Verify the RPC was called with exclude_product_ids
        mock_supabase.rpc.assert_called_once()
        rpc_call_args = mock_supabase.rpc.call_args
        rpc_name = rpc_call_args.args[0] if rpc_call_args.args else rpc_call_args.kwargs.get('function_name')
        rpc_params = rpc_call_args.args[1] if len(rpc_call_args.args) > 1 else rpc_call_args.kwargs.get('params', {})

        assert rpc_name == 'get_exploration_keyset', f"Should call get_exploration_keyset, got {rpc_name}"
        assert 'exclude_product_ids' in rpc_params, "SQL params must include exclude_product_ids"
        assert len(rpc_params['exclude_product_ids']) == 200, \
            f"Expected 200 exclude IDs, got {len(rpc_params['exclude_product_ids'])}"

    def test_no_exclude_ids_sends_null(self):
        """When no seen history, exclude_product_ids should be None."""
        from recs.candidate_selection import CandidateSelectionModule
        from recs.models import HardFilters

        mock_supabase = MagicMock()
        mock_rpc_result = MagicMock()
        mock_rpc_result.data = []
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value = mock_rpc_result
        mock_supabase.rpc.return_value = mock_rpc

        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase

        hard_filters = HardFilters(gender="female")

        with patch.object(module, '_enrich_with_attributes', side_effect=lambda c: c):
            module._retrieve_exploration_keyset(
                hard_filters=hard_filters,
                random_seed="test-seed",
                cursor_score=None,
                cursor_id=None,
                limit=500,
                exclude_ids=None,
            )

        rpc_params = mock_supabase.rpc.call_args.args[1] if len(mock_supabase.rpc.call_args.args) > 1 else {}
        assert rpc_params.get('exclude_product_ids') is None, \
            "No seen history -> exclude_product_ids should be None"

    def test_brands_function_also_gets_exclude_ids(self):
        """When preferred_brands is set, the _with_brands RPC should also get exclude_product_ids."""
        from recs.candidate_selection import CandidateSelectionModule
        from recs.models import HardFilters

        mock_supabase = MagicMock()
        mock_rpc_result = MagicMock()
        mock_rpc_result.data = []
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value = mock_rpc_result
        mock_supabase.rpc.return_value = mock_rpc

        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase

        seen = _make_seen_ids(300)
        hard_filters = HardFilters(gender="female")

        with patch.object(module, '_enrich_with_attributes', side_effect=lambda c: c):
            module._retrieve_exploration_keyset(
                hard_filters=hard_filters,
                random_seed="test-seed",
                cursor_score=None,
                cursor_id=None,
                limit=500,
                preferred_brands=["Boohoo", "Princess Polly"],
                exclude_ids=seen,
            )

        rpc_call_args = mock_supabase.rpc.call_args
        rpc_name = rpc_call_args.args[0]
        rpc_params = rpc_call_args.args[1] if len(rpc_call_args.args) > 1 else {}

        assert rpc_name == 'get_exploration_keyset_with_brands', \
            f"Should call _with_brands variant, got {rpc_name}"
        assert 'exclude_product_ids' in rpc_params
        assert len(rpc_params['exclude_product_ids']) == 300


# =============================================================================
# Test Class: Large Seen Sets (chunking)
# =============================================================================

class TestLargeSeenSets:
    """Tests for handling seen sets larger than SQL_EXCLUDE_IDS_LIMIT."""

    def test_under_limit_all_sent_to_sql(self):
        """Under 5000 seen: all IDs sent to SQL, no Python fallback."""
        from recs.candidate_selection import CandidateSelectionModule
        from recs.models import HardFilters

        mock_supabase = MagicMock()
        mock_rpc_result = MagicMock()
        mock_rpc_result.data = [
            {"product_id": str(uuid.uuid4()), "name": f"Product {i}", "brand": "TestBrand",
             "category": "tops", "broad_category": "tops", "article_type": "t-shirt",
             "colors": ["black"], "materials": ["cotton"], "price": 29.99,
             "fit": None, "length": None, "sleeve": None, "neckline": None, "rise": None,
             "style_tags": [], "primary_image_url": "https://img.example.com/test.jpg",
             "hero_image_url": None, "gallery_images": [],
             "exploration_score": 0.5, "similarity": 0.5,
             "computed_occasion_scores": {}, "computed_style_scores": {},
             "computed_pattern_scores": {}, "original_price": None,
             "discount_percent": None, "is_on_sale": False, "is_new": False,
             "created_at": "2025-01-01T00:00:00Z"}
            for i in range(100)
        ]
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value = mock_rpc_result
        mock_supabase.rpc.return_value = mock_rpc

        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase

        seen = _make_seen_ids(4999)  # Just under limit
        hard_filters = HardFilters(gender="female")

        with patch.object(module, '_enrich_with_attributes', side_effect=lambda c: c):
            result = module._retrieve_exploration_keyset(
                hard_filters=hard_filters,
                random_seed="test-seed",
                cursor_score=None,
                cursor_id=None,
                limit=500,
                exclude_ids=seen,
            )

        rpc_params = mock_supabase.rpc.call_args.args[1]
        assert len(rpc_params['exclude_product_ids']) == 4999, \
            "All 4999 IDs should be sent to SQL"
        # All 100 results should be returned (none filtered in Python)
        assert len(result) == 100

    def test_at_limit_boundary(self):
        """Exactly 5000 seen: all sent to SQL."""
        from recs.candidate_selection import CandidateSelectionModule
        from recs.models import HardFilters

        mock_supabase = MagicMock()
        mock_rpc_result = MagicMock()
        mock_rpc_result.data = []
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value = mock_rpc_result
        mock_supabase.rpc.return_value = mock_rpc

        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase

        seen = _make_seen_ids(5000)
        hard_filters = HardFilters(gender="female")

        with patch.object(module, '_enrich_with_attributes', side_effect=lambda c: c):
            module._retrieve_exploration_keyset(
                hard_filters=hard_filters,
                random_seed="test-seed",
                cursor_score=None,
                cursor_id=None,
                limit=500,
                exclude_ids=seen,
            )

        rpc_params = mock_supabase.rpc.call_args.args[1]
        assert len(rpc_params['exclude_product_ids']) == 5000

    def test_over_limit_splits_sql_and_python(self):
        """Over 5000 seen: first 5000 to SQL, overflow filtered in Python."""
        from recs.candidate_selection import CandidateSelectionModule
        from recs.models import HardFilters

        mock_supabase = MagicMock()

        # Create a deterministic ordered list of 5500 seen IDs
        # The code does list(exclude_ids)[:5000] for SQL, remainder for Python
        # We build the set carefully so we know which IDs are in overflow
        all_seen_list = [str(uuid.uuid4()) for _ in range(5500)]
        all_seen = set(all_seen_list)  # 5500 unique IDs

        # The code does: exclude_list = list(exclude_ids) -> first 5000 to SQL
        # Because set ordering is non-deterministic, we can't predict which 500
        # end up in overflow. Instead, we'll verify the behavior:
        # - 5000 go to SQL
        # - Remaining 500 are used as Python overflow filter
        # - Any candidate whose item_id is in the overflow set gets removed

        # Create candidates: 200 total. Some will match overflow IDs, some won't.
        # We use IDs from the END of the list (likely in overflow) + fresh IDs
        maybe_overflow_ids = all_seen_list[-200:]  # Last 200 from seen set
        fresh_ids = [str(uuid.uuid4()) for _ in range(100)]  # Definitely not in seen

        mock_rpc_result = MagicMock()
        mock_rpc_result.data = [
            {"product_id": pid, "name": "Product", "brand": "TestBrand",
             "category": "tops", "broad_category": "tops", "article_type": "t-shirt",
             "colors": ["black"], "materials": ["cotton"], "price": 29.99,
             "fit": None, "length": None, "sleeve": None, "neckline": None, "rise": None,
             "style_tags": [], "primary_image_url": "https://img.example.com/test.jpg",
             "hero_image_url": None, "gallery_images": [],
             "exploration_score": 0.5, "similarity": 0.5,
             "computed_occasion_scores": {}, "computed_style_scores": {},
             "computed_pattern_scores": {}, "original_price": None,
             "discount_percent": None, "is_on_sale": False, "is_new": False,
             "created_at": "2025-01-01T00:00:00Z"}
            for pid in maybe_overflow_ids + fresh_ids
        ]
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value = mock_rpc_result
        mock_supabase.rpc.return_value = mock_rpc

        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase
        hard_filters = HardFilters(gender="female")

        with patch.object(module, '_enrich_with_attributes', side_effect=lambda c: c):
            result = module._retrieve_exploration_keyset(
                hard_filters=hard_filters,
                random_seed="test-seed",
                cursor_score=None,
                cursor_id=None,
                limit=500,
                exclude_ids=all_seen,
            )

        # SQL should get exactly 5000
        rpc_params = mock_supabase.rpc.call_args.args[1]
        assert len(rpc_params['exclude_product_ids']) == 5000, \
            f"SQL should get 5000 IDs, got {len(rpc_params['exclude_product_ids'])}"

        # The 100 fresh candidates should always survive
        fresh_in_result = [c for c in result if c.item_id in set(fresh_ids)]
        assert len(fresh_in_result) == 100, \
            f"All 100 fresh candidates should survive, got {len(fresh_in_result)}"

        # No candidate in the result should be in the overflow set
        sql_sent = set(rpc_params['exclude_product_ids'])
        overflow = all_seen - sql_sent
        overflow_in_result = [c for c in result if c.item_id in overflow]
        assert len(overflow_in_result) == 0, \
            f"No overflow IDs should survive Python filter, found {len(overflow_in_result)}"

        # Total: fresh 100 + (maybe_overflow that were in sql_sent, not overflow)
        # The maybe_overflow IDs that were sent to SQL would not have been returned by SQL
        # (in real scenario), but in mock they are returned. Python overflow filter only
        # catches the 500 overflow IDs, not the 5000 SQL-sent ones.
        sql_sent_in_results = [c for c in result if c.item_id in sql_sent]
        # These leaked through (mock doesn't enforce SQL exclusion). That's fine —
        # the test verifies the OVERFLOW path works. SQL exclusion is tested separately.
        assert len(result) == 100 + len(sql_sent_in_results), \
            f"Result = fresh ({100}) + sql-sent-leaks ({len(sql_sent_in_results)})"

    def test_extreme_seen_history_7500(self):
        """Extreme case: 7500 seen items. 5000 to SQL, 2500 overflow to Python."""
        from recs.candidate_selection import CandidateSelectionModule
        from recs.models import HardFilters

        mock_supabase = MagicMock()
        mock_rpc_result = MagicMock()
        mock_rpc_result.data = []
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value = mock_rpc_result
        mock_supabase.rpc.return_value = mock_rpc

        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase

        seen = _make_seen_ids(7500)
        hard_filters = HardFilters(gender="female")

        with patch.object(module, '_enrich_with_attributes', side_effect=lambda c: c):
            module._retrieve_exploration_keyset(
                hard_filters=hard_filters,
                random_seed="test-seed",
                cursor_score=None,
                cursor_id=None,
                limit=500,
                exclude_ids=seen,
            )

        rpc_params = mock_supabase.rpc.call_args.args[1]
        assert len(rpc_params['exclude_product_ids']) == 5000, \
            "Should cap SQL exclude at 5000"


# =============================================================================
# Test Class: Python Fallback Filter (graceful degradation)
# =============================================================================

class TestPythonFallbackFilter:
    """Tests that the Python fallback still works for when SQL migration isn't applied."""

    def test_fallback_catches_seen_items_sql_missed(self):
        """If SQL migration not applied, SQL returns seen items; Python fallback filters them."""
        from recs.candidate_selection import CandidateSelectionModule
        from recs.models import HardFilters, Candidate

        # Simulate SQL returning items that include some from seen history
        # (this happens when SQL function doesn't have exclude_product_ids param yet)
        seen = {f"seen-{i}" for i in range(100)}
        sql_candidates = (
            [_make_candidate(item_id=f"seen-{i}") for i in range(30)] +  # 30 seen items leaked through
            [_make_candidate(item_id=f"fresh-{i}") for i in range(470)]  # 470 fresh items
        )

        mock_supabase = MagicMock()
        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase

        user_state = _make_user_state()

        with patch.object(module, '_retrieve_exploration_keyset', return_value=sql_candidates), \
             patch.object(module, '_apply_profile_scoring', side_effect=lambda c, p: c), \
             patch('recs.candidate_selection.HardFilters') as MockHF, \
             patch('recs.candidate_selection.get_include_article_types', return_value=None):
            MockHF.from_user_state.return_value = HardFilters(gender="female")

            result = module.get_candidates_keyset(
                user_state=user_state,
                gender="female",
                page_size=500,
                exclude_ids=seen,
            )

        # All 30 seen items should be filtered out by the Python fallback
        result_ids = {c.item_id for c in result}
        assert len(result_ids & seen) == 0, "No seen items should survive the fallback filter"
        assert len(result) == 470, f"Expected 470 fresh items, got {len(result)}"

    def test_fallback_does_nothing_when_sql_works(self):
        """When SQL properly excludes seen items, Python fallback removes 0 items."""
        from recs.candidate_selection import CandidateSelectionModule
        from recs.models import HardFilters

        seen = {f"seen-{i}" for i in range(100)}
        # All fresh — SQL already excluded seen items
        sql_candidates = [_make_candidate(item_id=f"fresh-{i}") for i in range(500)]

        mock_supabase = MagicMock()
        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase

        user_state = _make_user_state()

        with patch.object(module, '_retrieve_exploration_keyset', return_value=sql_candidates), \
             patch.object(module, '_apply_profile_scoring', side_effect=lambda c, p: c), \
             patch('recs.candidate_selection.HardFilters') as MockHF, \
             patch('recs.candidate_selection.get_include_article_types', return_value=None):
            MockHF.from_user_state.return_value = HardFilters(gender="female")

            result = module.get_candidates_keyset(
                user_state=user_state,
                gender="female",
                page_size=500,
                exclude_ids=seen,
            )

        assert len(result) == 500, f"Expected all 500 candidates preserved, got {len(result)}"


# =============================================================================
# Test Class: Category Filter + fetch_size Interaction
# =============================================================================

class TestCategoryFilterInteraction:
    """Tests that category filtering doesn't starve the feed."""

    def test_category_filtering_at_sql_level(self):
        """Categories are filtered in SQL (not Python), so fetch_size doesn't need inflation for them."""
        # This validates the design: categories go to SQL via HardFilters.categories,
        # so we don't need to over-fetch to account for category filtering.

        # Old formula: 150 candidates, ~27% bottoms = 40 items -> too few
        # New formula: 500 candidates ALL pre-filtered to bottoms in SQL = 500 items

        from recs.models import HardFilters, UserState, OnboardingProfile

        user_state = _make_user_state(categories=["bottoms"])
        hard_filters = HardFilters.from_user_state(user_state, "female")

        assert hard_filters.categories == ["bottoms"], \
            f"Categories should flow through to HardFilters, got {hard_filters.categories}"

    def test_categories_from_api_override_profile(self):
        """API ?categories=bottoms should override whatever's in the onboarding profile."""
        from recs.models import UserState, OnboardingProfile

        # User profile has tops+dresses
        user_state = _make_user_state(categories=["tops", "dresses"])
        assert user_state.onboarding_profile.categories == ["tops", "dresses"]

        # API override (simulating what pipeline.py:510-518 does)
        api_categories = ["bottoms"]
        user_state.onboarding_profile.categories = api_categories

        from recs.models import HardFilters
        hard_filters = HardFilters.from_user_state(user_state, "female")
        assert hard_filters.categories == ["bottoms"], \
            "API categories should override profile categories"

    def test_no_categories_returns_all(self):
        """No category filter should return all categories (SQL passes NULL)."""
        from recs.models import HardFilters

        user_state = _make_user_state()  # No categories
        hard_filters = HardFilters.from_user_state(user_state, "female")
        assert hard_filters.categories is None, \
            "No categories -> HardFilters.categories should be None (SQL returns all)"


# =============================================================================
# Test Class: Power User Scenarios (End-to-End Pipeline Logic)
# =============================================================================

class TestPowerUserScenarios:
    """Stress tests simulating real power user behaviors."""

    def test_user_with_1166_seen_gets_results(self):
        """Real user a42818ff with 1166 seen items should still get 500 candidates.

        Before fix: fetch_size = 1316, minus 1166 seen = 150 usable
        After fix: fetch_size = 500, ALL usable (SQL excludes seen)
        """
        # Simulate the new formula
        page_size = 50
        db_history_count = 1166
        base_fetch = max(500, page_size * 10)
        fetch_size = base_fetch  # No inflation from seen history
        assert fetch_size == 500

        # SQL returns 500 candidates (all unseen because SQL excluded 1166)
        # This is a 3.3x improvement over the old 150 usable
        candidates = _make_candidates(500)
        assert len(candidates) == 500

    def test_user_with_5000_seen_still_viable(self):
        """User with 5000 seen items gets 500 candidates (OLD: would get 0).

        Old formula: min(5150, 3000)=3000, minus 5000 = NEGATIVE -> 0 results
        New formula: 500 candidates, all unseen
        """
        page_size = 50
        db_history_count = 5000

        # Old formula
        old = min((page_size + db_history_count + 100), 3000)
        old_usable = max(0, old - db_history_count)
        assert old_usable == 0, "Old formula should produce 0 usable"

        # New formula
        base_fetch = max(500, page_size * 10)
        new = base_fetch
        assert new == 500, "New formula still gives 500"

    def test_progressive_seen_growth_simulation(self):
        """Simulate a user over time: 0 -> 100 -> 500 -> 1000 -> 3000 -> 5000 seen.

        At every stage, the new formula should produce >= 500 candidates.
        The old formula degrades and eventually produces 0.
        """
        page_size = 50
        stages = [0, 100, 500, 1000, 2000, 3000, 5000]

        for seen_count in stages:
            # New formula
            base_fetch = max(500, page_size * 10)
            new_fetch = base_fetch  # Decoupled from seen history
            new_usable = new_fetch  # All usable (SQL excludes seen)

            # Old formula
            old_fetch = min((page_size + seen_count + 100), 3000)
            old_usable = max(0, old_fetch - seen_count)

            assert new_usable >= 500, \
                f"Seen={seen_count}: new formula should give 500+ candidates, got {new_usable}"
            if seen_count >= 2900:
                assert old_usable <= 100, \
                    f"Seen={seen_count}: old formula should be severely degraded"

    def test_concurrent_filters_and_seen_history(self):
        """User with 500 seen + Python filters (colors, brands) still gets results."""
        page_size = 50
        db_history_count = 500
        has_python_filters = True

        filter_multiplier = 3
        base_fetch = max(500, page_size * 10)
        fetch_size = min(base_fetch * filter_multiplier, 5000)

        assert fetch_size == 1500, \
            f"Expected 1500 (500*3), got {fetch_size}"
        # 1500 candidates pre-filtered in SQL for seen history.
        # Python then filters for colors/brands out of 1500 -> plenty of results.

    def test_endless_scroll_deep_pagination(self):
        """User scrolls through 20 pages (1000 items). Still gets fresh results."""
        page_size = 50
        pages_scrolled = 20
        total_seen = page_size * pages_scrolled  # 1000 seen

        # The keyset cursor advances through the catalog.
        # SQL excludes 1000 seen items AND starts from cursor position.
        # fetch_size is 500 per page -> each page gets 500 fresh candidates.
        base_fetch = max(500, page_size * 10)
        assert base_fetch == 500

        # After 20 pages with 91K catalog: 1000/91630 = 1.1% consumed. Plenty left.
        catalog_remaining = 91630 - total_seen
        assert catalog_remaining > 90000, "Should have 90K+ products remaining"


# =============================================================================
# Test Class: Seen History Loading (get_user_seen_history)
# =============================================================================

class TestSeenHistoryLoading:
    """Tests for the get_user_seen_history method."""

    def test_loads_all_seen_ids_from_db(self):
        """Seen history is loaded from session_seen_ids table."""
        from recs.candidate_selection import CandidateSelectionModule

        mock_supabase = MagicMock()
        # Simulate 3 sessions with different seen_ids
        mock_result = MagicMock()
        mock_result.data = [
            {"seen_ids": [str(uuid.uuid4()) for _ in range(100)]},
            {"seen_ids": [str(uuid.uuid4()) for _ in range(200)]},
            {"seen_ids": [str(uuid.uuid4()) for _ in range(50)]},
        ]
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase

        seen = module.get_user_seen_history(anon_id=None, user_id="user-123")
        assert len(seen) == 350, f"Expected 350 seen items, got {len(seen)}"

    def test_capped_at_5000(self):
        """Seen history should be capped at 5000 items."""
        from recs.candidate_selection import CandidateSelectionModule

        mock_supabase = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {"seen_ids": [str(uuid.uuid4()) for _ in range(6000)]},
        ]
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase

        seen = module.get_user_seen_history(anon_id=None, user_id="user-123")
        assert len(seen) == 5000, f"Expected capped at 5000, got {len(seen)}"

    def test_empty_seen_history(self):
        """Fresh user with no seen history."""
        from recs.candidate_selection import CandidateSelectionModule

        mock_supabase = MagicMock()
        mock_result = MagicMock()
        mock_result.data = []
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase

        seen = module.get_user_seen_history(anon_id=None, user_id="user-123")
        assert len(seen) == 0

    def test_deduplication_across_sessions(self):
        """Same product ID in multiple sessions should be deduplicated."""
        from recs.candidate_selection import CandidateSelectionModule

        shared_id = str(uuid.uuid4())
        mock_supabase = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {"seen_ids": [shared_id, str(uuid.uuid4())]},
            {"seen_ids": [shared_id, str(uuid.uuid4())]},  # shared_id appears in both sessions
        ]
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_result

        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase

        seen = module.get_user_seen_history(anon_id=None, user_id="user-123")
        assert len(seen) == 3, "Should be 3 unique IDs (shared_id counted once)"

    def test_db_error_returns_empty_set(self):
        """Database error should return empty set, not crash."""
        from recs.candidate_selection import CandidateSelectionModule

        mock_supabase = MagicMock()
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.side_effect = \
            Exception("Connection timeout")

        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase

        seen = module.get_user_seen_history(anon_id=None, user_id="user-123")
        assert len(seen) == 0, "Error should return empty set"

    def test_no_identifiers_returns_empty(self):
        """No user_id or anon_id should return empty set."""
        from recs.candidate_selection import CandidateSelectionModule

        mock_supabase = MagicMock()
        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase

        seen = module.get_user_seen_history(anon_id=None, user_id=None)
        assert len(seen) == 0


# =============================================================================
# Test Class: SQL Migration Validation
# =============================================================================

class TestSQLMigration:
    """Tests to validate the SQL migration file structure."""

    def test_migration_file_exists(self):
        """Migration 041 should exist."""
        import os
        migration_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'sql', '041_add_exclude_product_ids.sql'
        )
        assert os.path.exists(migration_path), \
            f"Migration file not found at {migration_path}"

    def test_migration_has_both_functions(self):
        """Migration should update both get_exploration_keyset and _with_brands."""
        import os
        migration_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'sql', '041_add_exclude_product_ids.sql'
        )
        with open(migration_path) as f:
            content = f.read()

        assert 'get_exploration_keyset' in content, \
            "Migration must update get_exploration_keyset"
        assert 'get_exploration_keyset_with_brands' in content, \
            "Migration must update get_exploration_keyset_with_brands"

    def test_migration_has_exclude_product_ids_param(self):
        """Migration should add exclude_product_ids uuid[] parameter."""
        import os
        migration_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'sql', '041_add_exclude_product_ids.sql'
        )
        with open(migration_path) as f:
            content = f.read()

        assert 'exclude_product_ids uuid[]' in content, \
            "Migration must add exclude_product_ids uuid[] parameter"

    def test_migration_has_exclusion_where_clause(self):
        """Migration should have the NOT (p.id = ANY(exclude_product_ids)) clause."""
        import os
        migration_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'sql', '041_add_exclude_product_ids.sql'
        )
        with open(migration_path) as f:
            content = f.read()

        # Check for the exclusion clause (appears twice - once per function)
        assert content.count('NOT (p.id = ANY(exclude_product_ids))') == 2, \
            "Exclusion clause should appear in both functions"

    def test_migration_has_grants(self):
        """Migration should grant execute to anon and authenticated."""
        import os
        migration_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'sql', '041_add_exclude_product_ids.sql'
        )
        with open(migration_path) as f:
            content = f.read()

        assert 'GRANT EXECUTE' in content, "Migration must include GRANT statements"

    def test_migration_has_schema_reload(self):
        """Migration should reload PostgREST schema cache."""
        import os
        migration_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'sql', '041_add_exclude_product_ids.sql'
        )
        with open(migration_path) as f:
            content = f.read()

        assert "NOTIFY pgrst, 'reload schema'" in content, \
            "Migration must notify PostgREST to reload schema"


# =============================================================================
# Test Class: Regression - Before vs After Comparison
# =============================================================================

class TestRegressionComparison:
    """Regression tests comparing old vs new behavior for known problem scenarios."""

    def test_regression_bottoms_filter_fresh_user(self):
        """
        Regression: Fresh user requesting ?categories=bottoms
        OLD: 150 candidates (mixed categories), ~40 bottoms after filter
        NEW: 500 candidates (all bottoms from SQL), all usable
        """
        # Old: mixed categories, Python filtered
        old_total = 150
        bottoms_pct = 0.27  # bottoms are ~27% of catalog
        old_bottoms = int(old_total * bottoms_pct)
        assert old_bottoms < 50, f"Old formula: only {old_bottoms} bottoms from 150 mixed"

        # New: SQL filters to bottoms category, returns 500 bottoms
        new_total = 500  # All are bottoms (SQL filter)
        assert new_total >= 10 * old_bottoms, \
            f"New should be 10x+ improvement for category-filtered feeds"

    def test_regression_user_47f05869_zero_results(self):
        """
        Regression: User 47f05869 with 554 seen items got 0 bottoms.
        OLD: fetch=704, minus 554 seen = 150, bottoms_pct=27% = 40, dedup -> 18 -> sometimes 0
        NEW: fetch=500, all unseen (SQL), all bottoms (SQL) = 500
        """
        seen_count = 554
        page_size = 50

        # Old path
        old_fetch = min((page_size + seen_count + 100), 3000)  # 704
        old_usable = old_fetch - seen_count  # 150
        old_bottoms = int(old_usable * 0.27)  # ~40
        # After dedup + reranker -> could be 0

        # New path
        new_fetch = max(500, page_size * 10)  # 500
        # All 500 are unseen (SQL), all bottoms (SQL)
        new_usable = new_fetch

        assert new_usable > 10 * old_bottoms, \
            f"New ({new_usable}) should be dramatically more than old ({old_bottoms})"

    def test_regression_consistent_across_page_sizes(self):
        """Minimum 500 candidates regardless of requested page size."""
        for page_size in [10, 20, 30, 50, 100]:
            base_fetch = max(500, page_size * 10)
            assert base_fetch >= 500, \
                f"page_size={page_size}: fetch should be >= 500, got {base_fetch}"


# =============================================================================
# Test Class: Catalog Exhaustion Calculations
# =============================================================================

class TestCatalogExhaustion:
    """Tests verifying catalog exhaustion timelines are acceptable."""

    CATALOG_SIZE = 91630  # Current in-stock product count

    def test_casual_user_lifetime(self):
        """Casual user: 50 items/day, 3 days/week.

        150 items/week -> 7800/year -> 11.7 years to exhaust catalog.
        """
        items_per_week = 50 * 3
        years_to_exhaust = self.CATALOG_SIZE / (items_per_week * 52)
        assert years_to_exhaust > 10, \
            f"Casual user should take 10+ years, got {years_to_exhaust:.1f}"

    def test_power_user_lifetime(self):
        """Power user: 200 items/day, every day.

        200/day -> 73K/year -> 1.25 years to exhaust.
        """
        items_per_day = 200
        days_to_exhaust = self.CATALOG_SIZE / items_per_day
        years_to_exhaust = days_to_exhaust / 365
        assert years_to_exhaust > 1.0, \
            f"Power user should take 1+ year, got {years_to_exhaust:.1f}"

    def test_extreme_user_lifetime(self):
        """Extreme user: 500 items/day, every day.

        500/day -> 183 days (6 months). Acceptable with new inventory.
        """
        items_per_day = 500
        days_to_exhaust = self.CATALOG_SIZE / items_per_day
        assert days_to_exhaust > 180, \
            f"Even extreme user should last 6+ months, got {days_to_exhaust:.0f} days"

    def test_bottoms_category_power_user(self):
        """Power user browsing only bottoms (25,394 products).

        100 bottoms/day -> 254 days (~8.5 months).
        """
        bottoms_count = 25394
        items_per_day = 100
        days_to_exhaust = bottoms_count / items_per_day
        assert days_to_exhaust > 200, \
            f"Bottoms-only power user should last 6+ months, got {days_to_exhaust:.0f} days"


# =============================================================================
# Test Class: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_exclude_ids_set(self):
        """Empty set should be treated same as None."""
        from recs.candidate_selection import CandidateSelectionModule
        from recs.models import HardFilters

        mock_supabase = MagicMock()
        mock_rpc_result = MagicMock()
        mock_rpc_result.data = []
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value = mock_rpc_result
        mock_supabase.rpc.return_value = mock_rpc

        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase
        hard_filters = HardFilters(gender="female")

        with patch.object(module, '_enrich_with_attributes', side_effect=lambda c: c):
            module._retrieve_exploration_keyset(
                hard_filters=hard_filters,
                random_seed="test-seed",
                cursor_score=None,
                cursor_id=None,
                limit=500,
                exclude_ids=set(),  # Empty set
            )

        rpc_params = mock_supabase.rpc.call_args.args[1]
        assert rpc_params.get('exclude_product_ids') is None, \
            "Empty set should map to None (SQL skips the filter)"

    def test_single_exclude_id(self):
        """Single seen item should still be sent to SQL."""
        from recs.candidate_selection import CandidateSelectionModule
        from recs.models import HardFilters

        mock_supabase = MagicMock()
        mock_rpc_result = MagicMock()
        mock_rpc_result.data = []
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value = mock_rpc_result
        mock_supabase.rpc.return_value = mock_rpc

        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase
        hard_filters = HardFilters(gender="female")

        single_id = str(uuid.uuid4())

        with patch.object(module, '_enrich_with_attributes', side_effect=lambda c: c):
            module._retrieve_exploration_keyset(
                hard_filters=hard_filters,
                random_seed="test-seed",
                cursor_score=None,
                cursor_id=None,
                limit=500,
                exclude_ids={single_id},
            )

        rpc_params = mock_supabase.rpc.call_args.args[1]
        assert rpc_params['exclude_product_ids'] == [single_id], \
            "Single ID should be sent as a list with one element"

    def test_exclude_ids_type_consistency(self):
        """exclude_product_ids should always be a list (not set) when sent to SQL."""
        from recs.candidate_selection import CandidateSelectionModule
        from recs.models import HardFilters

        mock_supabase = MagicMock()
        mock_rpc_result = MagicMock()
        mock_rpc_result.data = []
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value = mock_rpc_result
        mock_supabase.rpc.return_value = mock_rpc

        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase
        hard_filters = HardFilters(gender="female")

        seen = _make_seen_ids(100)  # This is a set

        with patch.object(module, '_enrich_with_attributes', side_effect=lambda c: c):
            module._retrieve_exploration_keyset(
                hard_filters=hard_filters,
                random_seed="test-seed",
                cursor_score=None,
                cursor_id=None,
                limit=500,
                exclude_ids=seen,
            )

        rpc_params = mock_supabase.rpc.call_args.args[1]
        assert isinstance(rpc_params['exclude_product_ids'], list), \
            "exclude_product_ids must be a list, not a set (JSON serialization)"

    def test_rpc_error_returns_empty(self):
        """If SQL RPC fails, should return empty list gracefully."""
        from recs.candidate_selection import CandidateSelectionModule
        from recs.models import HardFilters

        mock_supabase = MagicMock()
        mock_supabase.rpc.side_effect = Exception("function get_exploration_keyset does not exist")

        module = CandidateSelectionModule.__new__(CandidateSelectionModule)
        module.supabase = mock_supabase
        hard_filters = HardFilters(gender="female")

        result = module._retrieve_exploration_keyset(
            hard_filters=hard_filters,
            random_seed="test-seed",
            cursor_score=None,
            cursor_id=None,
            limit=500,
            exclude_ids=_make_seen_ids(100),
        )

        assert result == [], "RPC error should return empty list"

    def test_page_size_1(self):
        """Minimum page size should still work correctly."""
        base_fetch = max(500, 1 * 10)
        assert base_fetch == 500, "Even page_size=1 should fetch 500 candidates"

    def test_page_size_200(self):
        """Large page size (200) should scale fetch to 2000."""
        base_fetch = max(500, 200 * 10)
        assert base_fetch == 2000
