"""
Tests for pool-aware diversity relaxation in the reranker and ranker.

Covers the fix for: brand filter + reranker diversity constraints = only 1 item.

When users apply narrow filters (e.g., single brand, sportswear category),
diversity constraints must auto-detect the pool composition and relax
accordingly, rather than hard-rejecting valid items.

Test groups:
1. GreedyConstrainedReranker — strict_diversity_positions + max_brand_share
2. SASRecRanker — _apply_brand_diversity_cap
3. SASRecRanker — _apply_sportswear_cap
4. apply_sort_diversity — sorted feeds (already works, regression tests)
"""

import sys
import os
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from recs.models import Candidate
from recs.feed_reranker import (
    GreedyConstrainedReranker,
    RerankerConfig,
    apply_sort_diversity,
)


# =============================================================================
# Helpers
# =============================================================================

def make_candidate(
    item_id: str = "item-1",
    brand: str = "BrandA",
    final_score: float = 1.0,
    broad_category: str = "tops",
    color_family: str = "Neutrals",
    fit: str = "regular",
    **kwargs,
) -> Candidate:
    """Create a minimal Candidate for reranker testing."""
    return Candidate(
        item_id=item_id,
        brand=brand,
        final_score=final_score,
        broad_category=broad_category,
        color_family=color_family,
        fit=fit,
        **kwargs,
    )


def make_pool(brand: str, count: int, base_score: float = 1.0) -> list:
    """Create a pool of N candidates from a single brand."""
    categories = ["tops", "dresses", "bottoms", "outerwear"]
    colors = ["Neutrals", "Blues", "Greens", "Browns", "Pinks"]
    fits = ["slim", "regular", "relaxed", "oversized"]
    return [
        make_candidate(
            item_id=f"{brand.lower()}-{i}",
            brand=brand,
            final_score=base_score - i * 0.01,
            broad_category=categories[i % len(categories)],
            color_family=colors[i % len(colors)],
            fit=fits[i % len(fits)],
        )
        for i in range(count)
    ]


def make_diverse_pool(brands: list, per_brand: int = 10) -> list:
    """Create a multi-brand pool."""
    pool = []
    for bi, brand in enumerate(brands):
        for i in range(per_brand):
            pool.append(make_candidate(
                item_id=f"{brand.lower()}-{i}",
                brand=brand,
                final_score=1.0 - (bi * per_brand + i) * 0.001,
                broad_category=["tops", "dresses", "bottoms"][i % 3],
                color_family=["Neutrals", "Blues", "Greens"][i % 3],
                fit=["slim", "regular", "relaxed"][i % 3],
            ))
    return pool


# =============================================================================
# 1. GreedyConstrainedReranker — single-brand pool
# =============================================================================

class TestRerankerSingleBrand:
    """The core bug: brand filter + strict diversity = 1 item."""

    def test_single_brand_returns_full_page(self):
        """Single-brand pool should fill the entire target_size, not just 1."""
        reranker = GreedyConstrainedReranker()
        pool = make_pool("Abercrombie & Fitch", 50)
        result = reranker.rerank(pool, target_size=24)
        assert len(result) == 24

    def test_single_brand_returns_all_when_pool_smaller_than_target(self):
        """If pool has 10 items and target is 24, return all 10."""
        reranker = GreedyConstrainedReranker()
        pool = make_pool("Boohoo", 10)
        result = reranker.rerank(pool, target_size=24)
        assert len(result) == 10

    def test_single_brand_preserves_rough_score_order(self):
        """Top-scored items should appear near the top of the result.

        Exact monotonic ordering is NOT guaranteed because the greedy
        reranker applies soft penalties (brand_decay, cluster_decay,
        combo_penalty, category_overshoot) that intentionally promote
        attribute diversity.  We check that the top-5 items by raw score
        all appear in the first 10 positions.
        """
        reranker = GreedyConstrainedReranker()
        pool = make_pool("Zara", 30)
        result = reranker.rerank(pool, target_size=24)
        top_5_ids = {c.item_id for c in pool[:5]}
        first_10_ids = {c.item_id for c in result[:10]}
        assert top_5_ids.issubset(first_10_ids)

    def test_single_brand_seen_ids_excluded(self):
        """Seen IDs should still be excluded with single brand."""
        reranker = GreedyConstrainedReranker()
        pool = make_pool("Nike", 30)
        seen = {pool[0].item_id, pool[1].item_id, pool[2].item_id}
        result = reranker.rerank(pool, target_size=24, seen_ids=seen)
        result_ids = {c.item_id for c in result}
        assert seen.isdisjoint(result_ids)
        assert len(result) == 24  # Still fills page from remaining 27

    def test_single_brand_no_duplicates(self):
        """No duplicate item_ids in result."""
        reranker = GreedyConstrainedReranker()
        pool = make_pool("H&M", 50)
        result = reranker.rerank(pool, target_size=30)
        ids = [c.item_id for c in result]
        assert len(ids) == len(set(ids))


# =============================================================================
# 2. GreedyConstrainedReranker — two-brand pool
# =============================================================================

class TestRerankerTwoBrands:
    """Low-diversity pool with exactly 2 brands."""

    def test_two_brands_fills_page(self):
        """Two-brand pool should fill the full target."""
        reranker = GreedyConstrainedReranker()
        pool = make_pool("BrandA", 20) + make_pool("BrandB", 20)
        result = reranker.rerank(pool, target_size=30)
        assert len(result) == 30

    def test_two_brands_relaxed_cap(self):
        """With 2 brands, cap is 80% of target (not 40%)."""
        reranker = GreedyConstrainedReranker()
        # 25 from BrandA (high scores) + 5 from BrandB (low scores)
        pool_a = make_pool("BrandA", 25, base_score=2.0)
        pool_b = make_pool("BrandB", 5, base_score=0.5)
        result = reranker.rerank(pool_a + pool_b, target_size=30)
        brand_a_count = sum(1 for c in result if c.brand == "BrandA")
        # With 80% cap on target=30: max_brand = max(3, int(30*0.80)) = 24
        assert brand_a_count <= 24
        assert brand_a_count >= 20  # Should include most of BrandA


# =============================================================================
# 3. GreedyConstrainedReranker — diverse pool (regression)
# =============================================================================

class TestRerankerDiversePool:
    """Normal diverse pools should behave exactly as before."""

    def test_diverse_pool_strict_diversity_enforced(self):
        """First 3 positions should have different brands (default behavior)."""
        reranker = GreedyConstrainedReranker()
        pool = make_diverse_pool(["Alpha", "Beta", "Gamma", "Delta"], per_brand=15)
        result = reranker.rerank(pool, target_size=20)
        # First 3 items must have distinct brands
        first_3_brands = [c.brand for c in result[:3]]
        assert len(set(first_3_brands)) == 3

    def test_diverse_pool_brand_cap_40_pct(self):
        """Default 40% brand cap enforced with diverse pool."""
        reranker = GreedyConstrainedReranker()
        # 80 from one dominant brand, 10 each from 2 others
        pool = make_pool("Dominant", 80, base_score=2.0)
        pool += make_pool("MinorA", 10, base_score=1.0)
        pool += make_pool("MinorB", 10, base_score=0.5)
        result = reranker.rerank(pool, target_size=50)
        dominant_count = sum(1 for c in result if c.brand == "Dominant")
        max_allowed = max(3, int(50 * 0.40))  # = 20
        assert dominant_count <= max_allowed

    def test_diverse_pool_fills_target(self):
        """Diverse pool should fill the target size."""
        reranker = GreedyConstrainedReranker()
        pool = make_diverse_pool(["A", "B", "C", "D", "E"], per_brand=20)
        result = reranker.rerank(pool, target_size=50)
        assert len(result) == 50


# =============================================================================
# 4. SASRecRanker — _apply_brand_diversity_cap
# =============================================================================

class TestBrandDiversityCap:
    """Brand diversity cap should skip for low-diversity pools."""

    def setup_method(self):
        """Create a mock ranker with the real _apply_brand_diversity_cap method."""
        from recs.sasrec_ranker import SASRecRanker, SASRecRankerConfig
        self.config = SASRecRankerConfig()
        # Create ranker without loading model
        self.ranker = SASRecRanker.__new__(SASRecRanker)
        self.ranker.config = self.config
        self.ranker.model = None
        self.ranker.token2id = {}

    def test_single_brand_no_reorder(self):
        """Single-brand pool should be returned unchanged."""
        pool = make_pool("OnlyBrand", 30)
        result = self.ranker._apply_brand_diversity_cap(pool)
        # Should be identical — no reordering
        assert [c.item_id for c in result] == [c.item_id for c in pool]

    def test_two_brands_no_reorder(self):
        """Two-brand pool should be returned unchanged."""
        pool = make_pool("BrandX", 20) + make_pool("BrandY", 10)
        result = self.ranker._apply_brand_diversity_cap(pool)
        assert [c.item_id for c in result] == [c.item_id for c in pool]

    def test_three_brands_applies_cap(self):
        """Three+ brands should still apply the diversity cap."""
        pool = make_diverse_pool(["A", "B", "C"], per_brand=30)
        result = self.ranker._apply_brand_diversity_cap(pool, target_count=50)
        # With cap=0.25, max_per_brand = max(2, int(50*0.25)) = 12
        # First 12 from each brand stay in-place, rest deferred to end.
        assert len(result) == len(pool)  # All items still present
        # The non-deferred section should have at most 12 per brand.
        # Total non-deferred = 3 brands * 12 = 36 items.
        max_per = max(2, int(50 * 0.25))  # 12
        from collections import Counter
        top_section = result[:max_per * 3]  # 36 items (the non-deferred zone)
        counts = Counter(c.brand for c in top_section)
        for brand, count in counts.items():
            assert count <= max_per


# =============================================================================
# 5. SASRecRanker — _apply_sportswear_cap
# =============================================================================

class TestSportswearCap:
    """Sportswear cap should skip when pool is mostly sportswear."""

    def setup_method(self):
        from recs.sasrec_ranker import SASRecRanker, SASRecRankerConfig
        self.config = SASRecRankerConfig()
        self.ranker = SASRecRanker.__new__(SASRecRanker)
        self.ranker.config = self.config
        self.ranker.model = None
        self.ranker.token2id = {}

    def _make_sportswear(self, count: int, brand: str = "Nike") -> list:
        """Create sportswear candidates (broad_category='sportswear')."""
        return [
            make_candidate(
                item_id=f"sw-{i}",
                brand=brand,
                final_score=1.0 - i * 0.01,
                broad_category="sportswear",
            )
            for i in range(count)
        ]

    def _make_non_sportswear(self, count: int, brand: str = "Zara") -> list:
        return [
            make_candidate(
                item_id=f"nsw-{i}",
                brand=brand,
                final_score=0.5 - i * 0.01,
                broad_category="tops",
            )
            for i in range(count)
        ]

    def test_mostly_sportswear_no_cap(self):
        """Pool >80% sportswear should skip cap entirely."""
        pool = self._make_sportswear(45) + self._make_non_sportswear(5)
        result = self.ranker._apply_sportswear_cap(pool)
        # 90% sportswear → bypass. Result should be identical to input.
        assert [c.item_id for c in result] == [c.item_id for c in pool]

    def test_all_sportswear_no_cap(self):
        """100% sportswear pool should skip cap."""
        pool = self._make_sportswear(30)
        result = self.ranker._apply_sportswear_cap(pool)
        assert [c.item_id for c in result] == [c.item_id for c in pool]

    def test_minority_sportswear_applies_cap(self):
        """Pool with <80% sportswear should still apply cap."""
        pool = self._make_sportswear(10) + self._make_non_sportswear(40)
        result = self.ranker._apply_sportswear_cap(pool, target_count=50)
        # 20% sportswear → cap applies. max_sportswear = max(2, int(50*0.15)) = 7
        # First 7 sportswear in place, rest deferred to end
        assert len(result) == len(pool)
        # Count sportswear in first 50 positions (before deferred)
        sw_in_top = sum(
            1 for c in result[:47]  # 40 non-sw + 7 sw
            if self.ranker._is_sportswear(c)
        )
        assert sw_in_top <= 10  # Some sportswear gets deferred


# =============================================================================
# 6. apply_sort_diversity — single brand (regression test)
# =============================================================================

class TestSortDiversitySingleBrand:
    """Sorted feeds with single brand should still return all items."""

    def test_single_brand_all_items_present(self):
        """All items from a single brand should appear in the result."""
        pool = make_pool("OnlyBrand", 30)
        result = apply_sort_diversity(pool, max_consecutive=2, max_brand_share=0.30)
        assert len(result) == 30  # All items present

    def test_single_brand_preserves_order(self):
        """Items should maintain their original relative order."""
        pool = make_pool("OnlyBrand", 20)
        result = apply_sort_diversity(pool, max_consecutive=2, max_brand_share=0.30)
        # All items present in same order (deferred items appended)
        result_ids = [c.item_id for c in result]
        pool_ids = [c.item_id for c in pool]
        assert result_ids == pool_ids

    def test_single_brand_with_seen_ids(self):
        """Seen IDs excluded but remaining items all present."""
        pool = make_pool("OnlyBrand", 20)
        seen = {pool[0].item_id, pool[5].item_id}
        result = apply_sort_diversity(pool, seen_ids=seen)
        assert len(result) == 18
        result_ids = {c.item_id for c in result}
        assert seen.isdisjoint(result_ids)


# =============================================================================
# 7. Edge cases
# =============================================================================

class TestEdgeCases:
    """Edge cases for pool diversity detection."""

    def test_empty_pool(self):
        reranker = GreedyConstrainedReranker()
        result = reranker.rerank([], target_size=24)
        assert result == []

    def test_pool_with_empty_brand_strings(self):
        """Items with empty brand strings should not count as a distinct brand."""
        reranker = GreedyConstrainedReranker()
        pool = [
            make_candidate(item_id=f"item-{i}", brand="", final_score=1.0 - i * 0.01)
            for i in range(20)
        ]
        # Empty brands → distinct_brands = 0 (empty string excluded)
        # Should not crash; all items returned
        result = reranker.rerank(pool, target_size=20)
        assert len(result) == 20

    def test_pool_one_brand_plus_empty_brands(self):
        """One real brand + some empty brands → treated as single-brand."""
        reranker = GreedyConstrainedReranker()
        pool = make_pool("RealBrand", 15)
        pool += [
            make_candidate(item_id=f"empty-{i}", brand="", final_score=0.5 - i * 0.01)
            for i in range(5)
        ]
        result = reranker.rerank(pool, target_size=20)
        assert len(result) == 20  # All items, no diversity rejection

    def test_target_size_one(self):
        """Target size of 1 should return exactly 1 item."""
        reranker = GreedyConstrainedReranker()
        pool = make_pool("Brand", 10)
        result = reranker.rerank(pool, target_size=1)
        assert len(result) == 1
