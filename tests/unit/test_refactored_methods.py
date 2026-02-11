"""
Integration tests for refactored pipeline and search engine methods.

These tests verify that the refactored methods work correctly after
the unification of filter utilities.

Run with: pytest src/tests/test_refactored_methods.py -v
"""

import pytest
from typing import List
from unittest.mock import MagicMock, patch

from recs.models import (
    Candidate,
    UserState,
    UserStateType,
    OnboardingProfile,
)
from recs.filter_utils import (
    deduplicate_candidates,
    apply_diversity_candidates,
    get_diversity_limit,
    DiversityConfig,
)
from recs.candidate_factory import candidate_from_dict


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def cold_user_state() -> UserState:
    """Cold start user with no taste vector."""
    return UserState(
        user_id='test_cold_user',
        state_type=UserStateType.COLD_START,
        taste_vector=None,
        onboarding_profile=OnboardingProfile(
            user_id='test_cold_user',
            categories=['tops', 'bottoms'],
        ),
    )


@pytest.fixture
def warm_user_state() -> UserState:
    """Warm user with taste vector."""
    return UserState(
        user_id='test_warm_user',
        state_type=UserStateType.TINDER_COMPLETE,
        taste_vector=[0.1] * 512,  # Mock 512-dim vector
        onboarding_profile=OnboardingProfile(
            user_id='test_warm_user',
            categories=['tops', 'bottoms', 'dresses', 'outerwear'],
        ),
    )


@pytest.fixture
def candidates_with_duplicates() -> List[Candidate]:
    """Candidates with image hash and name+brand duplicates."""
    return [
        # Original items
        Candidate(
            item_id='1',
            image_url='https://cdn.example.com/original_0_abc12345.jpg',
            name='Blue Cotton T-Shirt',
            brand='Nike',
            broad_category='tops',
            final_score=0.95,
        ),
        # Same image hash (cross-brand duplicate like Boohoo/Nasty Gal)
        Candidate(
            item_id='2',
            image_url='https://cdn.example.com/original_0_abc12345.jpg',
            name='Navy Cotton Tee',
            brand='Adidas',
            broad_category='tops',
            final_score=0.92,
        ),
        # Different item
        Candidate(
            item_id='3',
            image_url='https://cdn.example.com/original_0_def67890.jpg',
            name='Red Summer Dress',
            brand='Zara',
            broad_category='dresses',
            final_score=0.90,
        ),
        # Same name+brand (same product scraped multiple times)
        Candidate(
            item_id='4',
            image_url='https://cdn.example.com/product_different.jpg',
            name='Blue Cotton T-Shirt',
            brand='Nike',
            broad_category='tops',
            final_score=0.88,
        ),
        # Another unique item
        Candidate(
            item_id='5',
            image_url='https://cdn.example.com/original_0_fff11111.jpg',
            name='Black Skinny Jeans',
            brand="Levi's",
            broad_category='bottoms',
            final_score=0.85,
        ),
    ]


@pytest.fixture
def candidates_for_diversity() -> List[Candidate]:
    """Candidates for testing diversity constraints."""
    candidates = []
    # 15 tops
    for i in range(15):
        candidates.append(Candidate(
            item_id=f'top_{i}',
            broad_category='tops',
            category='t-shirts',
            final_score=0.9 - i * 0.01,
        ))
    # 15 bottoms
    for i in range(15):
        candidates.append(Candidate(
            item_id=f'bottom_{i}',
            broad_category='bottoms',
            category='jeans',
            final_score=0.9 - i * 0.01,
        ))
    # 15 dresses
    for i in range(15):
        candidates.append(Candidate(
            item_id=f'dress_{i}',
            broad_category='dresses',
            category='midi-dresses',
            final_score=0.9 - i * 0.01,
        ))
    return candidates


# =============================================================================
# Pipeline Deduplication Tests
# =============================================================================

class TestPipelineDeduplication:
    """Tests for pipeline's _deduplicate_by_image using shared utilities."""

    def test_removes_image_hash_duplicates(self, candidates_with_duplicates):
        """Test that candidates with same image hash are deduplicated."""
        result = deduplicate_candidates(candidates_with_duplicates)

        # Should have 3 unique items (items 1, 3, 5)
        assert len(result) == 3

        # First occurrence should be kept
        item_ids = [c.item_id for c in result]
        assert '1' in item_ids  # Original with hash abc12345
        assert '2' not in item_ids  # Duplicate hash
        assert '3' in item_ids  # Different hash
        assert '4' not in item_ids  # Same name+brand as item 1
        assert '5' in item_ids  # Unique item

    def test_preserves_ranking_order(self, candidates_with_duplicates):
        """Test that deduplication preserves original ranking order."""
        result = deduplicate_candidates(candidates_with_duplicates)

        # Scores should be in descending order
        scores = [c.final_score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_list(self):
        """Test deduplication of empty list."""
        result = deduplicate_candidates([])
        assert result == []

    def test_all_duplicates(self):
        """Test when all items are duplicates."""
        candidates = [
            Candidate(
                item_id=str(i),
                image_url='https://ex.com/original_0_same.jpg',
                name='Same Shirt',
                brand='Brand',
            )
            for i in range(5)
        ]
        result = deduplicate_candidates(candidates)
        assert len(result) == 1
        assert result[0].item_id == '0'

    def test_no_image_url(self):
        """Test items without image URLs are handled correctly."""
        candidates = [
            Candidate(item_id='1', name='Shirt A', brand='Nike'),
            Candidate(item_id='2', name='Shirt A', brand='Nike'),  # Same name+brand
            Candidate(item_id='3', name='Pants', brand='Zara'),
        ]
        result = deduplicate_candidates(candidates)
        assert len(result) == 2  # Item 2 removed as name+brand duplicate


# =============================================================================
# Pipeline Diversity Tests
# =============================================================================

class TestPipelineDiversity:
    """Tests for pipeline's _apply_diversity using shared utilities."""

    def test_cold_user_single_category(self, cold_user_state, candidates_for_diversity):
        """Test diversity limit for cold user with few categories."""
        # Modify to single category
        cold_user_state.onboarding_profile.categories = ['tops']

        limit = get_diversity_limit(
            user_categories=cold_user_state.onboarding_profile.categories,
            has_taste_vector=cold_user_state.taste_vector is not None,
        )
        assert limit == 50  # Single category gets high limit

        result = apply_diversity_candidates(candidates_for_diversity, max_per_category=limit)

        # Should include all tops (15) since limit is 50
        tops_count = sum(1 for c in result if c.broad_category == 'tops')
        assert tops_count == 15

    def test_cold_user_four_categories(self, cold_user_state, candidates_for_diversity):
        """Test diversity limit for cold user with 4+ categories."""
        cold_user_state.onboarding_profile.categories = ['tops', 'bottoms', 'dresses', 'outerwear']

        limit = get_diversity_limit(
            user_categories=cold_user_state.onboarding_profile.categories,
            has_taste_vector=False,
        )
        assert limit == 8  # Default limit for 4+ categories

        result = apply_diversity_candidates(candidates_for_diversity, max_per_category=limit)

        # Each category should have at most 8
        by_cat = {}
        for c in result:
            by_cat[c.broad_category] = by_cat.get(c.broad_category, 0) + 1

        assert all(count <= 8 for count in by_cat.values())

    def test_warm_user_gets_lenient_limit(self, warm_user_state, candidates_for_diversity):
        """Test that warm users (with taste_vector) get lenient diversity limits."""
        limit = get_diversity_limit(
            user_categories=warm_user_state.onboarding_profile.categories,
            has_taste_vector=True,
        )
        assert limit == 50  # Warm user limit

    def test_two_categories(self, cold_user_state, candidates_for_diversity):
        """Test diversity with exactly 2 categories."""
        cold_user_state.onboarding_profile.categories = ['tops', 'bottoms']

        limit = get_diversity_limit(
            user_categories=cold_user_state.onboarding_profile.categories,
            has_taste_vector=False,
        )
        assert limit == 25

        result = apply_diversity_candidates(candidates_for_diversity, max_per_category=limit)

        tops = sum(1 for c in result if c.broad_category == 'tops')
        bottoms = sum(1 for c in result if c.broad_category == 'bottoms')
        dresses = sum(1 for c in result if c.broad_category == 'dresses')

        assert tops == 15  # All tops (less than 25 limit)
        assert bottoms == 15  # All bottoms
        assert dresses == 15  # All dresses (also under limit)


# =============================================================================
# Search Engine Deduplication Tests
# =============================================================================

class TestSearchEngineDeduplication:
    """Tests for search engine's _deduplicate_results using shared utilities."""

    def test_dict_deduplication(self):
        """Test deduplication of search result dicts."""
        from recs.filter_utils import deduplicate_dicts

        results = [
            {'product_id': '1', 'image_url': 'https://ex.com/original_0_aaa.jpg', 'name': 'Shirt', 'brand': 'Nike', 'similarity': 0.9},
            {'product_id': '2', 'image_url': 'https://ex.com/original_0_aaa.jpg', 'name': 'Tee', 'brand': 'Adidas', 'similarity': 0.85},  # Same hash
            {'product_id': '3', 'image_url': 'https://ex.com/original_0_bbb.jpg', 'name': 'Dress', 'brand': 'Zara', 'similarity': 0.8},
        ]

        deduped = deduplicate_dicts(results)
        assert len(deduped) == 2
        assert deduped[0]['product_id'] == '1'
        assert deduped[1]['product_id'] == '3'

    def test_with_limit(self):
        """Test deduplication with limit parameter."""
        from recs.filter_utils import deduplicate_dicts

        results = [
            {'product_id': str(i), 'image_url': f'https://ex.com/original_0_{i:08x}.jpg', 'name': f'Item {i}', 'brand': 'Brand'}
            for i in range(100)
        ]

        deduped = deduplicate_dicts(results, limit=10)
        assert len(deduped) == 10

    def test_primary_image_url_field(self):
        """Test that primary_image_url field is also handled."""
        from recs.filter_utils import deduplicate_dicts

        results = [
            {'product_id': '1', 'primary_image_url': 'https://ex.com/original_0_abc.jpg', 'name': 'A', 'brand': 'X'},
            {'product_id': '2', 'primary_image_url': 'https://ex.com/original_0_abc.jpg', 'name': 'B', 'brand': 'Y'},
        ]

        deduped = deduplicate_dicts(results)
        assert len(deduped) == 1


# =============================================================================
# Search Engine Diversity Tests
# =============================================================================

class TestSearchEngineDiversity:
    """Tests for search engine's _apply_diversity using shared utilities."""

    def test_dict_diversity(self):
        """Test diversity constraints on search result dicts."""
        from recs.filter_utils import apply_diversity_dicts

        results = [
            {'product_id': str(i), 'broad_category': 'tops'} for i in range(20)
        ] + [
            {'product_id': str(i + 20), 'broad_category': 'bottoms'} for i in range(20)
        ]

        diverse = apply_diversity_dicts(results, max_per_category=5)

        tops = sum(1 for r in diverse if r['broad_category'] == 'tops')
        bottoms = sum(1 for r in diverse if r['broad_category'] == 'bottoms')

        assert tops == 5
        assert bottoms == 5
        assert len(diverse) == 10


# =============================================================================
# Search Engine Soft Scoring Tests
# =============================================================================

class TestSearchEngineSoftScoring:
    """Tests for search engine's _apply_soft_scoring using shared utilities."""

    def test_boost_for_fit_match(self):
        """Test that fit preferences boost matching items."""
        from recs.filter_utils import apply_soft_scoring

        results = [
            {'product_id': '1', 'similarity': 0.30, 'base_similarity': 0.30, 'fit': 'slim', 'colors': [], 'materials': [], 'broad_category': 'tops'},
            {'product_id': '2', 'similarity': 0.28, 'base_similarity': 0.28, 'fit': 'regular', 'colors': [], 'materials': [], 'broad_category': 'tops'},
        ]
        soft_prefs = {'preferred_fits': ['regular']}

        scored = apply_soft_scoring(results, soft_prefs, {})

        # Item 2 should be boosted above item 1
        assert scored[0]['product_id'] == '2'
        assert scored[0]['preference_boost'] > 0
        assert 'fit' in scored[0]['preference_matches']

    def test_demote_for_avoided_color(self):
        """Test that avoided colors demote items."""
        from recs.filter_utils import apply_soft_scoring

        results = [
            {'product_id': '1', 'similarity': 0.35, 'base_similarity': 0.35, 'colors': ['pink', 'white'], 'materials': [], 'broad_category': 'tops'},
            {'product_id': '2', 'similarity': 0.32, 'base_similarity': 0.32, 'colors': ['blue', 'white'], 'materials': [], 'broad_category': 'tops'},
        ]
        hard_filters = {'exclude_colors': ['pink']}

        scored = apply_soft_scoring(results, {}, {}, hard_filters)

        # Item 1 should be demoted below item 2
        assert scored[0]['product_id'] == '2'
        assert 'color_avoid' in scored[1]['preference_demotes']

    def test_combined_boost_and_demote(self):
        """Test combined boost and demote effects."""
        from recs.filter_utils import apply_soft_scoring

        results = [
            {'product_id': '1', 'similarity': 0.32, 'base_similarity': 0.32, 'fit': 'regular', 'brand': 'Nike', 'colors': ['pink'], 'materials': [], 'broad_category': 'tops'},
            {'product_id': '2', 'similarity': 0.30, 'base_similarity': 0.30, 'fit': 'regular', 'brand': 'Nike', 'colors': ['blue'], 'materials': [], 'broad_category': 'tops'},
        ]
        soft_prefs = {'preferred_fits': ['regular'], 'preferred_brands': ['Nike']}
        hard_filters = {'exclude_colors': ['pink']}

        scored = apply_soft_scoring(results, soft_prefs, {}, hard_filters)

        # Item 2 should win: same boosts, but item 1 has color demote
        assert scored[0]['product_id'] == '2'


# =============================================================================
# Candidate Factory Tests for Occasion Gate
# =============================================================================

class TestCandidateFactoryForOccasionFiltering:
    """Tests for candidate_from_dict used in occasion filtering."""

    def test_conversion_preserves_occasions(self):
        """Test that occasions array is preserved in conversion."""
        data = {
            'product_id': '123',
            'name': 'Office Blouse',
            'brand': 'Ann Taylor',
            'occasions': ['Office', 'Everyday'],
            'style_tags': ['Classic', 'Professional'],
        }

        candidate = candidate_from_dict(data)

        assert 'Office' in candidate.occasions
        assert 'Everyday' in candidate.occasions
        assert 'Classic' in candidate.style_tags

    def test_pattern_field_preserved(self):
        """Test that pattern field is preserved."""
        data = {
            'product_id': '123',
            'pattern': 'Floral',
        }

        candidate = candidate_from_dict(data)

        assert candidate.pattern == 'Floral'

    def test_none_fields_default_to_empty(self):
        """Test that None product_attributes default to empty."""
        data = {
            'product_id': '123',
            'occasions': None,
            'style_tags': None,
            'pattern': None,
        }

        candidate = candidate_from_dict(data)

        assert candidate.occasions == []
        assert candidate.style_tags == []
        assert candidate.pattern is None

    def test_all_fields_for_occasion_filter(self):
        """Test that all fields needed for occasion filtering are preserved."""
        data = {
            'product_id': '123',
            'name': 'Evening Gown',
            'brand': 'Designer',
            'article_type': 'gown',
            'sleeve': 'sleeveless',
            'length': 'maxi',
            'fit': 'fitted',
            'occasions': ['Evening Event', 'Party', 'Date Night'],
            'formality': 'Semi-Formal',
        }

        candidate = candidate_from_dict(data)

        assert candidate.name == 'Evening Gown'
        assert candidate.brand == 'Designer'
        assert candidate.article_type == 'gown'
        assert candidate.sleeve == 'sleeveless'
        assert candidate.length == 'maxi'
        assert candidate.fit == 'fitted'
        assert 'Evening Event' in candidate.occasions
        assert candidate.formality == 'Semi-Formal'


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Edge case tests for the refactored utilities."""

    def test_unicode_in_names(self):
        """Test handling of Unicode characters in names."""
        candidates = [
            Candidate(item_id='1', name='Été Summer Dress', brand='Côte d\'Azur'),
            Candidate(item_id='2', name='été summer dress', brand='côte d\'azur'),  # Same, lowercase
        ]
        result = deduplicate_candidates(candidates)
        assert len(result) == 1

    def test_whitespace_in_names(self):
        """Test handling of whitespace variations."""
        candidates = [
            Candidate(item_id='1', name='  Blue Shirt  ', brand='  Nike  '),
            Candidate(item_id='2', name='Blue Shirt', brand='Nike'),  # Same after strip
        ]
        result = deduplicate_candidates(candidates)
        assert len(result) == 1

    def test_very_long_results_list(self):
        """Test performance with large result sets."""
        from recs.filter_utils import deduplicate_dicts, apply_diversity_dicts

        # Create 10,000 results
        results = [
            {'product_id': str(i), 'image_url': f'https://ex.com/original_0_{i:08x}.jpg', 'name': f'Item {i}', 'brand': 'Brand', 'broad_category': 'tops' if i % 2 == 0 else 'bottoms'}
            for i in range(10000)
        ]

        # Should complete without timeout
        deduped = deduplicate_dicts(results, limit=1000)
        assert len(deduped) == 1000

        diverse = apply_diversity_dicts(deduped, max_per_category=100)
        assert len(diverse) <= 200  # Max 100 per 2 categories

    def test_special_characters_in_hash(self):
        """Test that only valid hex characters are extracted."""
        from recs.filter_utils import extract_image_hash

        # Valid hex
        assert extract_image_hash('https://ex.com/original_0_abcdef12.jpg') == 'abcdef12'

        # Invalid hex characters (g, h, etc.)
        assert extract_image_hash('https://ex.com/original_0_ghijkl12.jpg') is None

    def test_empty_user_profile(self):
        """Test with completely empty user profile."""
        limit = get_diversity_limit(
            user_categories=None,
            has_taste_vector=False,
        )
        # Should fall back to single category limit
        assert limit == 50

    def test_null_values_in_dict(self):
        """Test handling of null values in dict conversion."""
        data = {
            'product_id': '1',
            'name': None,
            'brand': None,
            'colors': None,
            'materials': None,
            'fit': None,
        }

        candidate = candidate_from_dict(data)

        assert candidate.name == ''
        assert candidate.brand == ''
        assert candidate.colors == []
        assert candidate.materials == []
        assert candidate.fit is None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
