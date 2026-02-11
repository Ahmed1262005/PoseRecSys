"""
Comprehensive unit tests for filter_utils and candidate_factory modules.

Tests cover:
1. Image hash extraction - various URL formats and edge cases
2. Deduplication - Candidates, Dicts, edge cases, seen hash persistence
3. Diversity constraints - dynamic limits, category handling
4. Soft scoring - boosts, demotes, caps, semantic floor
5. Candidate factory - conversions, JSON parsing, missing fields

Run with: pytest src/tests/test_filter_utils.py -v
"""

import pytest
import json
from typing import List, Dict, Any

# Import modules under test
from recs.filter_utils import (
    extract_image_hash,
    IMAGE_HASH_PATTERN,
    deduplicate_items,
    deduplicate_candidates,
    deduplicate_dicts,
    DiversityConfig,
    DEFAULT_DIVERSITY_CONFIG,
    get_diversity_limit,
    apply_diversity_candidates,
    apply_diversity_dicts,
    SoftScoringWeights,
    DEFAULT_SOFT_WEIGHTS,
    compute_soft_score_boost,
    apply_soft_scoring,
)
from recs.candidate_factory import (
    candidate_from_dict,
    candidate_to_dict,
    candidates_from_dicts,
    candidates_to_dicts,
    merge_candidate_with_dict,
)
from recs.models import Candidate


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_candidates() -> List[Candidate]:
    """Sample Candidate objects for testing."""
    return [
        Candidate(
            item_id='1',
            image_url='https://example.com/original_0_abc12345.jpg',
            name='Blue T-Shirt',
            brand='Nike',
            broad_category='tops',
            category='t-shirts',
            price=29.99,
            colors=['blue', 'white'],
            fit='regular',
            final_score=0.95,
        ),
        Candidate(
            item_id='2',
            image_url='https://example.com/original_0_def67890.jpg',
            name='Red Dress',
            brand='Zara',
            broad_category='dresses',
            category='midi-dresses',
            price=79.99,
            colors=['red'],
            fit='fitted',
            final_score=0.90,
        ),
        Candidate(
            item_id='3',
            image_url='https://example.com/original_0_ghi11111.jpg',
            name='Black Jeans',
            brand='Levi\'s',
            broad_category='bottoms',
            category='jeans',
            price=59.99,
            colors=['black'],
            fit='slim',
            final_score=0.85,
        ),
    ]


@pytest.fixture
def sample_dicts() -> List[Dict[str, Any]]:
    """Sample Dict results for testing."""
    return [
        {
            'product_id': '1',
            'image_url': 'https://example.com/original_0_abc12345.jpg',
            'name': 'Blue T-Shirt',
            'brand': 'Nike',
            'broad_category': 'tops',
            'category': 't-shirts',
            'price': 29.99,
            'colors': ['blue', 'white'],
            'materials': ['cotton'],
            'fit': 'regular',
            'similarity': 0.35,
            'base_similarity': 0.35,
        },
        {
            'product_id': '2',
            'image_url': 'https://example.com/original_0_def67890.jpg',
            'name': 'Red Dress',
            'brand': 'Zara',
            'broad_category': 'dresses',
            'category': 'midi-dresses',
            'price': 79.99,
            'colors': ['red'],
            'materials': ['polyester'],
            'fit': 'fitted',
            'similarity': 0.32,
            'base_similarity': 0.32,
        },
    ]


# =============================================================================
# Image Hash Extraction Tests
# =============================================================================

class TestExtractImageHash:
    """Tests for extract_image_hash function."""

    def test_standard_url_format(self):
        """Test standard URL format with hash."""
        url = 'https://example.com/images/original_0_85a218f8.jpg'
        assert extract_image_hash(url) == '85a218f8'

    def test_different_index_numbers(self):
        """Test URLs with different index numbers."""
        assert extract_image_hash('https://ex.com/original_0_abc123.jpg') == 'abc123'
        assert extract_image_hash('https://ex.com/original_1_def456.jpg') == 'def456'
        assert extract_image_hash('https://ex.com/original_99_aaa999.jpg') == 'aaa999'
        assert extract_image_hash('https://ex.com/original_123_bbb012.jpg') == 'bbb012'

    def test_different_file_extensions(self):
        """Test URLs with different file extensions."""
        assert extract_image_hash('https://ex.com/original_0_abc123.jpg') == 'abc123'
        assert extract_image_hash('https://ex.com/original_0_abc123.jpeg') == 'abc123'
        assert extract_image_hash('https://ex.com/original_0_abc123.png') == 'abc123'
        assert extract_image_hash('https://ex.com/original_0_abc123.webp') == 'abc123'

    def test_long_hash(self):
        """Test URLs with longer hashes."""
        url = 'https://ex.com/original_0_abc123def456789.jpg'
        assert extract_image_hash(url) == 'abc123def456789'

    def test_hash_in_path(self):
        """Test URLs with hash embedded in longer path."""
        url = 'https://cdn.example.com/products/fashion/2024/original_0_abc123.jpg?quality=80'
        assert extract_image_hash(url) == 'abc123'

    def test_no_hash_in_url(self):
        """Test URLs without the expected hash format."""
        assert extract_image_hash('https://ex.com/product_123.jpg') is None
        assert extract_image_hash('https://ex.com/image.jpg') is None
        assert extract_image_hash('https://ex.com/original_abc123.jpg') is None  # Missing _0_

    def test_none_url(self):
        """Test None URL input."""
        assert extract_image_hash(None) is None

    def test_empty_url(self):
        """Test empty URL input."""
        assert extract_image_hash('') is None

    def test_uppercase_in_hash(self):
        """Test that uppercase hex is not matched (pattern uses [a-f0-9])."""
        # Pattern only matches lowercase hex
        url = 'https://ex.com/original_0_ABC123.jpg'
        assert extract_image_hash(url) is None

    def test_mixed_case_hash(self):
        """Test mixed case - pattern requires hash followed directly by dot."""
        # Pattern is: original_\d+_([a-f0-9]+)\.
        # The hash must be followed immediately by '.' for the pattern to match
        # So 'abc123ABC.jpg' won't match because ABC comes before the dot

        # This won't match because ABC is between hash and dot
        url = 'https://ex.com/original_0_abc123ABC.jpg'
        assert extract_image_hash(url) is None

        # This will match - lowercase hash followed directly by dot
        url2 = 'https://ex.com/original_0_abc123.jpg'
        assert extract_image_hash(url2) == 'abc123'


# =============================================================================
# Deduplication Tests
# =============================================================================

class TestDeduplicateCandidates:
    """Tests for deduplicate_candidates function."""

    def test_no_duplicates(self, sample_candidates):
        """Test with no duplicates - all items kept."""
        result = deduplicate_candidates(sample_candidates)
        assert len(result) == 3
        assert [c.item_id for c in result] == ['1', '2', '3']

    def test_duplicate_image_hash(self):
        """Test removing duplicates with same image hash."""
        candidates = [
            Candidate(item_id='1', image_url='https://ex.com/original_0_abc123.jpg', name='Shirt A', brand='Nike'),
            Candidate(item_id='2', image_url='https://ex.com/original_0_abc123.jpg', name='Shirt B', brand='Adidas'),
            Candidate(item_id='3', image_url='https://ex.com/original_0_def456.jpg', name='Pants', brand='Zara'),
        ]
        result = deduplicate_candidates(candidates)
        assert len(result) == 2
        assert result[0].item_id == '1'
        assert result[1].item_id == '3'

    def test_duplicate_name_brand(self):
        """Test removing duplicates with same name+brand."""
        candidates = [
            Candidate(item_id='1', image_url='https://ex.com/img1.jpg', name='Blue Shirt', brand='Nike'),
            Candidate(item_id='2', image_url='https://ex.com/img2.jpg', name='Blue Shirt', brand='Nike'),  # Same name+brand
            Candidate(item_id='3', image_url='https://ex.com/img3.jpg', name='Blue Shirt', brand='Adidas'),  # Different brand
        ]
        result = deduplicate_candidates(candidates)
        assert len(result) == 2
        assert result[0].item_id == '1'
        assert result[1].item_id == '3'

    def test_case_insensitive_name_brand(self):
        """Test that name+brand matching is case-insensitive."""
        candidates = [
            Candidate(item_id='1', image_url='https://ex.com/img1.jpg', name='Blue Shirt', brand='Nike'),
            Candidate(item_id='2', image_url='https://ex.com/img2.jpg', name='BLUE SHIRT', brand='NIKE'),  # Same, diff case
        ]
        result = deduplicate_candidates(candidates)
        assert len(result) == 1
        assert result[0].item_id == '1'

    def test_empty_name_brand_not_matched(self):
        """Test that items with empty name or brand are not matched."""
        candidates = [
            Candidate(item_id='1', image_url='https://ex.com/img1.jpg', name='', brand='Nike'),
            Candidate(item_id='2', image_url='https://ex.com/img2.jpg', name='', brand='Nike'),  # Both empty name
            Candidate(item_id='3', image_url='https://ex.com/img3.jpg', name='Shirt', brand=''),  # Empty brand
        ]
        result = deduplicate_candidates(candidates)
        # All should be kept because empty name/brand doesn't form valid key
        assert len(result) == 3

    def test_seen_hashes_persistence(self):
        """Test that seen_hashes set is mutated and persisted."""
        seen_hashes = set()
        candidates1 = [
            Candidate(item_id='1', image_url='https://ex.com/original_0_abc123.jpg', name='A', brand='X'),
        ]
        deduplicate_candidates(candidates1, seen_hashes=seen_hashes)
        assert 'abc123' in seen_hashes

        # Second batch should respect seen hashes
        candidates2 = [
            Candidate(item_id='2', image_url='https://ex.com/original_0_abc123.jpg', name='B', brand='Y'),  # Same hash
            Candidate(item_id='3', image_url='https://ex.com/original_0_def456.jpg', name='C', brand='Z'),
        ]
        result = deduplicate_candidates(candidates2, seen_hashes=seen_hashes)
        assert len(result) == 1
        assert result[0].item_id == '3'

    def test_empty_list(self):
        """Test with empty list."""
        result = deduplicate_candidates([])
        assert result == []

    def test_single_item(self):
        """Test with single item."""
        candidates = [Candidate(item_id='1', name='Shirt', brand='Nike')]
        result = deduplicate_candidates(candidates)
        assert len(result) == 1


class TestDeduplicateDicts:
    """Tests for deduplicate_dicts function."""

    def test_basic_deduplication(self, sample_dicts):
        """Test basic deduplication with no duplicates."""
        result = deduplicate_dicts(sample_dicts)
        assert len(result) == 2

    def test_duplicate_image_hash(self):
        """Test removing duplicates with same image hash."""
        results = [
            {'product_id': '1', 'image_url': 'https://ex.com/original_0_abc123.jpg', 'name': 'A', 'brand': 'X'},
            {'product_id': '2', 'image_url': 'https://ex.com/original_0_abc123.jpg', 'name': 'B', 'brand': 'Y'},
        ]
        deduped = deduplicate_dicts(results)
        assert len(deduped) == 1
        assert deduped[0]['product_id'] == '1'

    def test_primary_image_url_field(self):
        """Test that primary_image_url field is also checked."""
        results = [
            {'product_id': '1', 'primary_image_url': 'https://ex.com/original_0_abc123.jpg', 'name': 'A', 'brand': 'X'},
            {'product_id': '2', 'primary_image_url': 'https://ex.com/original_0_abc123.jpg', 'name': 'B', 'brand': 'Y'},
        ]
        deduped = deduplicate_dicts(results)
        assert len(deduped) == 1

    def test_limit_parameter(self):
        """Test that limit parameter stops early."""
        results = [
            {'product_id': str(i), 'image_url': f'https://ex.com/original_0_{i:06x}.jpg', 'name': f'Item {i}', 'brand': 'Brand'}
            for i in range(10)
        ]
        deduped = deduplicate_dicts(results, limit=5)
        assert len(deduped) == 5

    def test_limit_with_duplicates(self):
        """Test limit works correctly when duplicates are filtered."""
        results = [
            {'product_id': '1', 'image_url': 'https://ex.com/original_0_aaa.jpg', 'name': 'A', 'brand': 'X'},
            {'product_id': '2', 'image_url': 'https://ex.com/original_0_aaa.jpg', 'name': 'B', 'brand': 'Y'},  # Dup
            {'product_id': '3', 'image_url': 'https://ex.com/original_0_bbb.jpg', 'name': 'C', 'brand': 'Z'},
            {'product_id': '4', 'image_url': 'https://ex.com/original_0_ccc.jpg', 'name': 'D', 'brand': 'W'},
        ]
        deduped = deduplicate_dicts(results, limit=2)
        assert len(deduped) == 2
        assert deduped[0]['product_id'] == '1'
        assert deduped[1]['product_id'] == '3'


# =============================================================================
# Diversity Constraints Tests
# =============================================================================

class TestDiversityConfig:
    """Tests for DiversityConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DiversityConfig()
        assert config.default_limit == 8
        assert config.single_category_limit == 50
        assert config.two_category_limit == 25
        assert config.three_category_limit == 16
        assert config.warm_user_limit == 50

    def test_custom_config(self):
        """Test custom configuration."""
        config = DiversityConfig(
            default_limit=10,
            single_category_limit=100,
            warm_user_limit=75,
        )
        assert config.default_limit == 10
        assert config.single_category_limit == 100
        assert config.warm_user_limit == 75


class TestGetDiversityLimit:
    """Tests for get_diversity_limit function."""

    def test_no_categories(self):
        """Test with no categories selected."""
        limit = get_diversity_limit(user_categories=None, has_taste_vector=False)
        assert limit == 50  # single_category_limit (0 <= 1)

    def test_empty_categories(self):
        """Test with empty categories list."""
        limit = get_diversity_limit(user_categories=[], has_taste_vector=False)
        assert limit == 50  # single_category_limit

    def test_single_category(self):
        """Test with single category."""
        limit = get_diversity_limit(user_categories=['tops'], has_taste_vector=False)
        assert limit == 50

    def test_two_categories(self):
        """Test with two categories."""
        limit = get_diversity_limit(user_categories=['tops', 'bottoms'], has_taste_vector=False)
        assert limit == 25

    def test_three_categories(self):
        """Test with three categories."""
        limit = get_diversity_limit(user_categories=['tops', 'bottoms', 'dresses'], has_taste_vector=False)
        assert limit == 16

    def test_four_or_more_categories(self):
        """Test with four or more categories."""
        limit = get_diversity_limit(
            user_categories=['tops', 'bottoms', 'dresses', 'outerwear'],
            has_taste_vector=False
        )
        assert limit == 8  # default_limit

    def test_warm_user_overrides(self):
        """Test that warm user (has_taste_vector) gets higher limit."""
        limit = get_diversity_limit(
            user_categories=['tops', 'bottoms', 'dresses', 'outerwear'],
            has_taste_vector=True
        )
        assert limit == 50  # warm_user_limit

    def test_custom_config(self):
        """Test with custom config."""
        config = DiversityConfig(
            default_limit=5,
            single_category_limit=30,
            warm_user_limit=100,
        )
        limit = get_diversity_limit(
            user_categories=['tops'],
            has_taste_vector=False,
            config=config
        )
        assert limit == 30


class TestApplyDiversityCandidates:
    """Tests for apply_diversity_candidates function."""

    def test_basic_diversity(self):
        """Test basic diversity constraint application."""
        candidates = [
            Candidate(item_id=str(i), broad_category='tops') for i in range(10)
        ]
        result = apply_diversity_candidates(candidates, max_per_category=5)
        assert len(result) == 5

    def test_multiple_categories(self):
        """Test diversity across multiple categories."""
        candidates = [
            Candidate(item_id=str(i), broad_category='tops') for i in range(10)
        ] + [
            Candidate(item_id=str(i), broad_category='bottoms') for i in range(10, 20)
        ] + [
            Candidate(item_id=str(i), broad_category='dresses') for i in range(20, 30)
        ]
        result = apply_diversity_candidates(candidates, max_per_category=3)
        assert len(result) == 9  # 3 per category

        # Verify distribution
        by_cat = {}
        for c in result:
            by_cat[c.broad_category] = by_cat.get(c.broad_category, 0) + 1
        assert by_cat['tops'] == 3
        assert by_cat['bottoms'] == 3
        assert by_cat['dresses'] == 3

    def test_fallback_to_category(self):
        """Test fallback to category field when broad_category is missing."""
        candidates = [
            Candidate(item_id='1', category='t-shirts'),  # No broad_category
            Candidate(item_id='2', category='t-shirts'),
            Candidate(item_id='3', category='jeans'),
        ]
        result = apply_diversity_candidates(candidates, max_per_category=1)
        assert len(result) == 2
        assert result[0].item_id == '1'
        assert result[1].item_id == '3'

    def test_unknown_category(self):
        """Test handling of items with no category."""
        candidates = [
            Candidate(item_id=str(i)) for i in range(5)  # No category
        ]
        result = apply_diversity_candidates(candidates, max_per_category=3)
        assert len(result) == 3  # All grouped under 'unknown'

    def test_preserves_order(self):
        """Test that original order is preserved."""
        candidates = [
            Candidate(item_id='A', broad_category='tops', final_score=0.9),
            Candidate(item_id='B', broad_category='bottoms', final_score=0.8),
            Candidate(item_id='C', broad_category='tops', final_score=0.7),
            Candidate(item_id='D', broad_category='bottoms', final_score=0.6),
        ]
        result = apply_diversity_candidates(candidates, max_per_category=10)
        assert [c.item_id for c in result] == ['A', 'B', 'C', 'D']


class TestApplyDiversityDicts:
    """Tests for apply_diversity_dicts function."""

    def test_basic_diversity(self):
        """Test basic diversity for dicts."""
        results = [
            {'product_id': str(i), 'broad_category': 'tops'} for i in range(10)
        ]
        diverse = apply_diversity_dicts(results, max_per_category=3)
        assert len(diverse) == 3

    def test_category_fallback(self):
        """Test fallback to category field."""
        results = [
            {'product_id': '1', 'category': 'jeans'},
            {'product_id': '2', 'category': 'jeans'},
            {'product_id': '3', 'category': 't-shirts'},
        ]
        diverse = apply_diversity_dicts(results, max_per_category=1)
        assert len(diverse) == 2


# =============================================================================
# Soft Scoring Tests
# =============================================================================

class TestSoftScoringWeights:
    """Tests for SoftScoringWeights dataclass."""

    def test_default_weights(self):
        """Test default weight values."""
        weights = SoftScoringWeights()
        assert weights.max_total_boost == 0.15
        assert weights.semantic_floor == 0.25
        assert weights.fit_boost == 0.03
        assert weights.brand_boost == 0.05
        assert weights.color_demote == -0.15

    def test_custom_weights(self):
        """Test custom weight configuration."""
        weights = SoftScoringWeights(
            max_total_boost=0.20,
            fit_boost=0.10,
            brand_demote=-0.25,
        )
        assert weights.max_total_boost == 0.20
        assert weights.fit_boost == 0.10
        assert weights.brand_demote == -0.25


class TestComputeSoftScoreBoost:
    """Tests for compute_soft_score_boost function."""

    def test_fit_boost(self):
        """Test boost for matching fit preference."""
        item = {'fit': 'regular', 'base_similarity': 0.30, 'colors': [], 'materials': [], 'broad_category': 'tops'}
        soft_prefs = {'preferred_fits': ['regular']}
        type_prefs = {}

        boost, matches, demotes = compute_soft_score_boost(item, soft_prefs, type_prefs)
        assert boost > 0
        assert 'fit' in matches
        assert len(demotes) == 0

    def test_multiple_boosts(self):
        """Test multiple attribute boosts."""
        item = {
            'fit': 'regular',
            'sleeve': 'long',
            'brand': 'Nike',
            'base_similarity': 0.30,
            'colors': [],
            'materials': [],
            'broad_category': 'tops',
        }
        soft_prefs = {
            'preferred_fits': ['regular'],
            'preferred_sleeves': ['long'],
            'preferred_brands': ['Nike'],
        }
        type_prefs = {}

        boost, matches, demotes = compute_soft_score_boost(item, soft_prefs, type_prefs)
        assert 'fit' in matches
        assert 'sleeve' in matches
        assert 'brand' in matches
        assert len(matches) == 3

    def test_boost_capped_at_max(self):
        """Test that total boost is capped at max_total_boost."""
        item = {
            'fit': 'regular',
            'sleeve': 'long',
            'length': 'regular',
            'rise': 'high',
            'brand': 'Nike',
            'article_type': 't-shirts',
            'base_similarity': 0.30,
            'colors': [],
            'materials': [],
            'broad_category': 'tops',
        }
        soft_prefs = {
            'preferred_fits': ['regular'],
            'preferred_sleeves': ['long'],
            'preferred_lengths': ['regular'],
            'preferred_rises': ['high'],
            'preferred_brands': ['Nike'],
        }
        type_prefs = {'top_types': ['t-shirts']}

        boost, matches, demotes = compute_soft_score_boost(item, soft_prefs, type_prefs)
        # Total should be capped at 0.15
        assert boost <= DEFAULT_SOFT_WEIGHTS.max_total_boost

    def test_semantic_floor(self):
        """Test that boosts are not applied below semantic floor."""
        item = {
            'fit': 'regular',
            'base_similarity': 0.20,  # Below floor of 0.25
            'colors': [],
            'materials': [],
            'broad_category': 'tops',
        }
        soft_prefs = {'preferred_fits': ['regular']}
        type_prefs = {}

        boost, matches, demotes = compute_soft_score_boost(item, soft_prefs, type_prefs)
        # No positive boost should be applied below floor
        assert 'fit' not in matches

    def test_color_demote(self):
        """Test demotion for avoided colors."""
        item = {
            'fit': 'regular',
            'base_similarity': 0.30,
            'colors': ['red', 'pink'],
            'materials': [],
            'broad_category': 'tops',
        }
        soft_prefs = {}
        type_prefs = {}
        hard_filters = {'exclude_colors': ['pink']}

        boost, matches, demotes = compute_soft_score_boost(item, soft_prefs, type_prefs, hard_filters)
        assert boost < 0
        assert 'color_avoid' in demotes

    def test_partial_color_match(self):
        """Test partial color matching (e.g., 'red' matches 'Wine Red')."""
        item = {
            'base_similarity': 0.30,
            'colors': ['Wine Red', 'Cherry'],
            'materials': [],
            'broad_category': 'tops',
        }
        soft_prefs = {}
        type_prefs = {}
        hard_filters = {'exclude_colors': ['red']}

        boost, matches, demotes = compute_soft_score_boost(item, soft_prefs, type_prefs, hard_filters)
        assert 'color_avoid' in demotes

    def test_material_demote(self):
        """Test demotion for avoided materials."""
        item = {
            'base_similarity': 0.30,
            'colors': [],
            'materials': ['100% Polyester'],
            'broad_category': 'tops',
        }
        soft_prefs = {}
        type_prefs = {}
        hard_filters = {'exclude_materials': ['polyester']}

        boost, matches, demotes = compute_soft_score_boost(item, soft_prefs, type_prefs, hard_filters)
        assert 'material_avoid' in demotes

    def test_brand_demote(self):
        """Test demotion for avoided brands."""
        item = {
            'base_similarity': 0.30,
            'brand': 'FastFashion',
            'colors': [],
            'materials': [],
            'broad_category': 'tops',
        }
        soft_prefs = {}
        type_prefs = {}
        hard_filters = {'exclude_brands': ['fastfashion']}

        boost, matches, demotes = compute_soft_score_boost(item, soft_prefs, type_prefs, hard_filters)
        assert 'brand_avoid' in demotes

    def test_demotes_applied_below_floor(self):
        """Test that demotes are applied even below semantic floor."""
        item = {
            'base_similarity': 0.20,  # Below floor
            'colors': ['pink'],
            'materials': [],
            'broad_category': 'tops',
        }
        soft_prefs = {}
        type_prefs = {}
        hard_filters = {'exclude_colors': ['pink']}

        boost, matches, demotes = compute_soft_score_boost(item, soft_prefs, type_prefs, hard_filters)
        assert boost < 0
        assert 'color_avoid' in demotes

    def test_dress_length_matching(self):
        """Test that dress lengths use preferred_lengths_dresses."""
        item = {
            'length': 'midi',
            'base_similarity': 0.30,
            'colors': [],
            'materials': [],
            'broad_category': 'dresses',
        }
        soft_prefs = {
            'preferred_lengths': ['cropped'],  # For tops/bottoms
            'preferred_lengths_dresses': ['midi'],  # For dresses
        }
        type_prefs = {}

        boost, matches, demotes = compute_soft_score_boost(item, soft_prefs, type_prefs)
        assert 'length' in matches


class TestApplySoftScoring:
    """Tests for apply_soft_scoring function."""

    def test_reorders_by_score(self):
        """Test that results are reordered by boosted score."""
        results = [
            {'product_id': '1', 'similarity': 0.30, 'base_similarity': 0.30, 'fit': 'slim', 'brand': 'X', 'colors': [], 'materials': [], 'broad_category': 'tops'},
            {'product_id': '2', 'similarity': 0.28, 'base_similarity': 0.28, 'fit': 'regular', 'brand': 'Nike', 'colors': [], 'materials': [], 'broad_category': 'tops'},
        ]
        soft_prefs = {'preferred_fits': ['regular'], 'preferred_brands': ['Nike']}
        type_prefs = {}

        scored = apply_soft_scoring(results, soft_prefs, type_prefs)

        # Item 2 should now be first due to boosts
        assert scored[0]['product_id'] == '2'
        assert scored[0]['similarity'] > 0.28

    def test_adds_preference_fields(self):
        """Test that preference_boost and preference_matches are added."""
        results = [
            {'product_id': '1', 'similarity': 0.30, 'base_similarity': 0.30, 'fit': 'regular', 'colors': [], 'materials': [], 'broad_category': 'tops'},
        ]
        soft_prefs = {'preferred_fits': ['regular']}
        type_prefs = {}

        scored = apply_soft_scoring(results, soft_prefs, type_prefs)

        assert 'preference_boost' in scored[0]
        assert 'preference_matches' in scored[0]
        assert 'preference_demotes' in scored[0]

    def test_empty_prefs_returns_unchanged(self):
        """Test that empty preferences returns results unchanged."""
        results = [
            {'product_id': '1', 'similarity': 0.30},
            {'product_id': '2', 'similarity': 0.28},
        ]

        scored = apply_soft_scoring(results, {}, {}, None)

        # Order should be unchanged
        assert scored[0]['product_id'] == '1'
        assert scored[1]['product_id'] == '2'

    def test_similarity_clamped_to_range(self):
        """Test that similarity is clamped to 0-1 range."""
        results = [
            {'product_id': '1', 'similarity': 0.95, 'base_similarity': 0.95, 'fit': 'regular', 'brand': 'Nike', 'colors': [], 'materials': [], 'broad_category': 'tops'},
        ]
        soft_prefs = {'preferred_fits': ['regular'], 'preferred_brands': ['Nike']}
        type_prefs = {}

        scored = apply_soft_scoring(results, soft_prefs, type_prefs)

        # Even with boosts, should not exceed 1.0
        assert scored[0]['similarity'] <= 1.0


# =============================================================================
# Candidate Factory Tests
# =============================================================================

class TestCandidateFromDict:
    """Tests for candidate_from_dict function."""

    def test_basic_conversion(self, sample_dicts):
        """Test basic dict to Candidate conversion."""
        candidate = candidate_from_dict(sample_dicts[0])

        assert candidate.item_id == '1'
        assert candidate.name == 'Blue T-Shirt'
        assert candidate.brand == 'Nike'
        assert candidate.category == 't-shirts'
        assert candidate.broad_category == 'tops'
        assert candidate.price == 29.99
        assert candidate.colors == ['blue', 'white']
        assert candidate.fit == 'regular'

    def test_product_id_vs_item_id(self):
        """Test that both product_id and item_id fields are handled."""
        dict_with_product_id = {'product_id': '123', 'name': 'Test'}
        dict_with_item_id = {'item_id': '456', 'name': 'Test'}

        c1 = candidate_from_dict(dict_with_product_id)
        c2 = candidate_from_dict(dict_with_item_id)

        assert c1.item_id == '123'
        assert c2.item_id == '456'

    def test_image_url_variations(self):
        """Test that both image_url and primary_image_url are handled."""
        dict1 = {'product_id': '1', 'image_url': 'https://ex.com/img1.jpg'}
        dict2 = {'product_id': '2', 'primary_image_url': 'https://ex.com/img2.jpg'}

        c1 = candidate_from_dict(dict1)
        c2 = candidate_from_dict(dict2)

        assert c1.image_url == 'https://ex.com/img1.jpg'
        assert c2.image_url == 'https://ex.com/img2.jpg'

    def test_missing_fields_default_to_empty(self):
        """Test that missing fields get default values."""
        minimal_dict = {'product_id': '1'}
        candidate = candidate_from_dict(minimal_dict)

        assert candidate.item_id == '1'
        assert candidate.name == ''
        assert candidate.brand == ''
        assert candidate.colors == []
        assert candidate.occasions == []

    def test_occasions_array(self):
        """Test handling of occasions array from product_attributes."""
        dict_with_occasions = {
            'product_id': '1',
            'occasions': ['Office', 'Everyday', 'Date Night'],
        }
        candidate = candidate_from_dict(dict_with_occasions)

        assert candidate.occasions == ['Office', 'Everyday', 'Date Night']

    def test_pattern_field(self):
        """Test handling of pattern field from product_attributes."""
        dict_with_pattern = {
            'product_id': '1',
            'pattern': 'Floral',
        }
        candidate = candidate_from_dict(dict_with_pattern)

        assert candidate.pattern == 'Floral'

    def test_none_product_attributes(self):
        """Test handling of None product_attributes fields."""
        dict_with_none = {
            'product_id': '1',
            'occasions': None,
            'pattern': None,
            'formality': None,
        }
        candidate = candidate_from_dict(dict_with_none)

        assert candidate.occasions == []
        assert candidate.pattern is None
        assert candidate.formality is None

    def test_score_fields(self):
        """Test conversion of score fields."""
        dict_with_scores = {
            'product_id': '1',
            'embedding_score': 0.85,
            'similarity': 0.90,
            'sasrec_score': 0.75,
            'final_score': 0.88,
        }
        candidate = candidate_from_dict(dict_with_scores)

        assert candidate.embedding_score == 0.85
        assert candidate.sasrec_score == 0.75
        assert candidate.final_score == 0.88


class TestCandidateToDict:
    """Tests for candidate_to_dict function."""

    def test_basic_conversion(self, sample_candidates):
        """Test basic Candidate to dict conversion."""
        result = candidate_to_dict(sample_candidates[0])

        assert result['product_id'] == '1'
        assert result['item_id'] == '1'
        assert result['name'] == 'Blue T-Shirt'
        assert result['brand'] == 'Nike'
        assert result['image_url'] == 'https://example.com/original_0_abc12345.jpg'
        assert result['primary_image_url'] == result['image_url']

    def test_round_trip(self, sample_dicts):
        """Test dict -> Candidate -> dict round trip."""
        original = sample_dicts[0]
        candidate = candidate_from_dict(original)
        result = candidate_to_dict(candidate)

        # Key fields should match
        assert result['product_id'] == original['product_id']
        assert result['name'] == original['name']
        assert result['brand'] == original['brand']


class TestCandidatesFromDicts:
    """Tests for batch conversion functions."""

    def test_batch_from_dicts(self, sample_dicts):
        """Test batch dict to Candidate conversion."""
        candidates = candidates_from_dicts(sample_dicts)

        assert len(candidates) == 2
        assert all(isinstance(c, Candidate) for c in candidates)
        assert candidates[0].item_id == '1'
        assert candidates[1].item_id == '2'

    def test_batch_to_dicts(self, sample_candidates):
        """Test batch Candidate to dict conversion."""
        dicts = candidates_to_dicts(sample_candidates)

        assert len(dicts) == 3
        assert all(isinstance(d, dict) for d in dicts)
        assert dicts[0]['product_id'] == '1'


class TestMergeCandidateWithDict:
    """Tests for merge_candidate_with_dict function."""

    def test_merge_adds_extra_fields(self, sample_candidates):
        """Test that extra fields are added to candidate dict."""
        extra = {
            'keyword_match': True,
            'brand_match': True,
            'custom_field': 'value',
        }
        result = merge_candidate_with_dict(sample_candidates[0], extra)

        assert result['product_id'] == '1'
        assert result['keyword_match'] is True
        assert result['brand_match'] is True
        assert result['custom_field'] == 'value'

    def test_extra_fields_override(self, sample_candidates):
        """Test that extra fields can override candidate fields."""
        extra = {'name': 'Overridden Name'}
        result = merge_candidate_with_dict(sample_candidates[0], extra)

        assert result['name'] == 'Overridden Name'


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple utilities."""

    def test_full_search_pipeline(self, sample_dicts):
        """Test typical search result processing pipeline."""
        # 1. Start with search results
        results = sample_dicts * 3  # Duplicate to test dedup

        # 2. Deduplicate
        deduped = deduplicate_dicts(results)
        assert len(deduped) == 2

        # 3. Apply soft scoring
        soft_prefs = {'preferred_fits': ['regular']}
        scored = apply_soft_scoring(deduped, soft_prefs, {})

        # 4. Apply diversity
        diverse = apply_diversity_dicts(scored, max_per_category=10)

        # 5. Convert to candidates for occasion gate
        candidates = candidates_from_dicts(diverse)
        assert len(candidates) == 2
        assert all(isinstance(c, Candidate) for c in candidates)

    def test_full_feed_pipeline(self, sample_candidates):
        """Test typical feed processing pipeline."""
        # 1. Start with candidates
        candidates = sample_candidates * 4  # Duplicate to test dedup

        # 2. Deduplicate
        deduped = deduplicate_candidates(candidates)
        assert len(deduped) == 3

        # 3. Apply diversity
        diverse = apply_diversity_candidates(deduped, max_per_category=2)

        # Each category should have at most 2
        by_cat = {}
        for c in diverse:
            by_cat[c.broad_category] = by_cat.get(c.broad_category, 0) + 1
        assert all(count <= 2 for count in by_cat.values())

    def test_seen_hashes_across_pages(self):
        """Test that seen hashes persist across pagination."""
        # Page 1
        page1 = [
            {'product_id': '1', 'image_url': 'https://ex.com/original_0_aaa.jpg', 'name': 'A', 'brand': 'X'},
            {'product_id': '2', 'image_url': 'https://ex.com/original_0_bbb.jpg', 'name': 'B', 'brand': 'Y'},
        ]
        seen_hashes = set()
        seen_name_brand = set()

        result1 = deduplicate_dicts(page1, seen_hashes=seen_hashes, seen_name_brand=seen_name_brand)
        assert len(result1) == 2
        assert 'aaa' in seen_hashes
        assert 'bbb' in seen_hashes

        # Page 2 - should skip items with same hashes
        page2 = [
            {'product_id': '3', 'image_url': 'https://ex.com/original_0_aaa.jpg', 'name': 'C', 'brand': 'Z'},  # Same hash as item 1
            {'product_id': '4', 'image_url': 'https://ex.com/original_0_ccc.jpg', 'name': 'D', 'brand': 'W'},
        ]

        result2 = deduplicate_dicts(page2, seen_hashes=seen_hashes, seen_name_brand=seen_name_brand)
        assert len(result2) == 1
        assert result2[0]['product_id'] == '4'


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
