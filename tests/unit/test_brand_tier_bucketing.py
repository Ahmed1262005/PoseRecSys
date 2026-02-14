"""
Tests for 3-tier brand bucketing in the recommendation pipeline.

The bucketing logic matches the Gradio demo's approach:
  Tier 1 (60%): Preferred brands -- user explicitly chose these
  Tier 2 (30%): Cluster-adjacent brands -- same style cluster, not preferred
  Tier 3 (10%): Everything else -- discovery / diversity

Tests cover:
1. brand_clusters.py: CLUSTER_TO_BRANDS reverse mapping
2. brand_clusters.py: get_cluster_adjacent_brands() helper
3. pipeline.py: _apply_brand_tier_bucketing() method
4. Edge cases: no preferred brands, empty clusters, backfill, small pools
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from recs.brand_clusters import (
    BRAND_CLUSTER_MAP,
    CLUSTER_TO_BRANDS,
    DEFAULT_CLUSTER,
    get_cluster_adjacent_brands,
    get_cluster_for_item,
    get_brand_cluster,
)
from recs.models import Candidate


# =============================================================================
# Helpers
# =============================================================================

def make_candidate(item_id: str, brand: str = "", **kwargs) -> Candidate:
    """Create a minimal Candidate for testing."""
    return Candidate(item_id=item_id, brand=brand, **kwargs)


def make_pool(brand_counts: dict) -> list:
    """
    Create a candidate pool from a dict of {brand: count}.

    Example: make_pool({"Boohoo": 50, "Zara": 30}) -> 80 candidates
    """
    pool = []
    idx = 0
    for brand, count in brand_counts.items():
        for i in range(count):
            pool.append(make_candidate(item_id=f"{brand.lower()}-{i}", brand=brand))
            idx += 1
    return pool


# =============================================================================
# Tests: CLUSTER_TO_BRANDS reverse mapping
# =============================================================================

class TestClusterToBrands:
    """Verify the reverse mapping built at import time."""

    def test_reverse_mapping_populated(self):
        """CLUSTER_TO_BRANDS should have entries for all clusters in BRAND_CLUSTER_MAP."""
        cluster_ids = {cid for cid, _ in BRAND_CLUSTER_MAP.values()}
        for cid in cluster_ids:
            assert cid in CLUSTER_TO_BRANDS, f"Cluster {cid} missing from CLUSTER_TO_BRANDS"
            assert len(CLUSTER_TO_BRANDS[cid]) > 0, f"Cluster {cid} has no brands"

    def test_every_brand_in_reverse_map(self):
        """Every brand in BRAND_CLUSTER_MAP should appear in the reverse mapping."""
        for brand, (cid, _) in BRAND_CLUSTER_MAP.items():
            assert brand in CLUSTER_TO_BRANDS[cid], (
                f"Brand '{brand}' not in CLUSTER_TO_BRANDS['{cid}']"
            )

    def test_known_cluster_contents(self):
        """Spot-check a few known clusters."""
        # Cluster H = Ultra-Fast Fashion
        assert "boohoo" in CLUSTER_TO_BRANDS["H"]
        assert "missguided" in CLUSTER_TO_BRANDS["H"]
        assert "prettylittlething" in CLUSTER_TO_BRANDS["H"]
        # Cluster A = Modern Classics
        assert "reformation" in CLUSTER_TO_BRANDS["A"]
        assert "ba&sh" in CLUSTER_TO_BRANDS["A"]
        # Cluster E = Athletic Heritage
        assert "nike" in CLUSTER_TO_BRANDS["E"]
        assert "adidas" in CLUSTER_TO_BRANDS["E"]

    def test_secondary_clusters_included(self):
        """Brands with secondary clusters should appear in both cluster sets."""
        # Zara: primary G, secondary A
        assert "zara" in CLUSTER_TO_BRANDS["G"]
        assert "zara" in CLUSTER_TO_BRANDS["A"]
        # Free People: primary M, secondary Q
        assert "free people" in CLUSTER_TO_BRANDS["M"]
        assert "free people" in CLUSTER_TO_BRANDS["Q"]


# =============================================================================
# Tests: get_cluster_adjacent_brands()
# =============================================================================

class TestGetClusterAdjacentBrands:
    """Verify cluster-adjacent brand discovery."""

    def test_finds_adjacent_brands(self):
        """Preferred 'boohoo' (cluster H) should find other H brands."""
        adjacent = get_cluster_adjacent_brands(["Boohoo"])
        # missguided, prettylittlething, shein, etc. are in cluster H
        assert "missguided" in adjacent
        assert "prettylittlething" in adjacent
        # But boohoo itself should NOT be in the adjacent set
        assert "boohoo" not in adjacent

    def test_excludes_preferred_brands(self):
        """Adjacent set must not contain the preferred brands themselves."""
        prefs = ["Boohoo", "Missguided"]
        adjacent = get_cluster_adjacent_brands(prefs)
        assert "boohoo" not in adjacent
        assert "missguided" not in adjacent
        # But other H brands should still be there
        assert "prettylittlething" in adjacent

    def test_multi_cluster_expansion(self):
        """Preferred brands from different clusters expand both clusters."""
        # Boohoo = cluster H, Nike = cluster E
        adjacent = get_cluster_adjacent_brands(["Boohoo", "Nike"])
        # Cluster H adjacents
        assert "missguided" in adjacent
        # Cluster E adjacents
        assert "adidas" in adjacent
        # Neither preferred brand should be included
        assert "boohoo" not in adjacent
        assert "nike" not in adjacent

    def test_empty_preferred(self):
        """Empty preferred brands should return empty set."""
        assert get_cluster_adjacent_brands([]) == set()

    def test_unknown_brand(self):
        """Unknown brands (not in cluster map) should return empty set."""
        adjacent = get_cluster_adjacent_brands(["TotallyFakeBrand123"])
        assert adjacent == set()

    def test_secondary_cluster_expansion(self):
        """Brands with secondary clusters should expand both clusters."""
        # Zara: primary G (Affordable Essentials), secondary A (Modern Classics)
        adjacent = get_cluster_adjacent_brands(["Zara"])
        # Should include G brands (excluding zara)
        assert "gap" in adjacent or "old navy" in adjacent
        # Should also include A brands (from secondary cluster)
        assert "reformation" in adjacent or "ba&sh" in adjacent

    def test_case_insensitive(self):
        """Should work regardless of input casing."""
        adj_lower = get_cluster_adjacent_brands(["boohoo"])
        adj_upper = get_cluster_adjacent_brands(["BOOHOO"])
        adj_mixed = get_cluster_adjacent_brands(["Boohoo"])
        assert adj_lower == adj_upper == adj_mixed


# =============================================================================
# Tests: _apply_brand_tier_bucketing() (pipeline method)
# =============================================================================

class TestBrandTierBucketing:
    """Test the 3-tier bucketing applied in the pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create a minimal pipeline mock with just the bucketing method."""
        from unittest.mock import MagicMock
        from recs.pipeline import RecommendationPipeline

        # Create a mock pipeline but bind the real method
        mock = MagicMock(spec=RecommendationPipeline)
        mock._apply_brand_tier_bucketing = RecommendationPipeline._apply_brand_tier_bucketing.__get__(mock)
        return mock

    def test_basic_60_30_10_allocation(self, pipeline):
        """With enough items in all tiers, should approximate 60/30/10."""
        # Boohoo (H), Missguided (H) = preferred brands
        # PrettyLittleThing (H), Nasty Gal (H) = cluster-adjacent
        # Zara (G), Gap (G) = discovery
        pool = (
            make_pool({"Boohoo": 80, "Missguided": 40}) +      # tier 1: preferred
            make_pool({"PrettyLittleThing": 50, "Nasty Gal": 30}) +  # tier 2: cluster H adj
            make_pool({"Zara": 40, "Gap": 30, "Nike": 20})     # tier 3: discovery
        )

        result = pipeline._apply_brand_tier_bucketing(
            candidates=pool,
            preferred_brands=["Boohoo", "Missguided"],
            target_size=100,
            seed="test-user",
        )

        assert len(result) == 100

        # Count tiers in result
        pref_set = {"boohoo", "missguided"}
        from recs.brand_clusters import get_cluster_adjacent_brands
        adj_set = get_cluster_adjacent_brands(["Boohoo", "Missguided"])

        t1 = sum(1 for c in result if (c.brand or "").lower() in pref_set)
        t2 = sum(1 for c in result if (c.brand or "").lower() in adj_set)
        t3 = len(result) - t1 - t2

        # Allow some tolerance due to rounding
        assert t1 >= 55, f"Tier 1 too low: {t1} (expected ~60)"
        assert t1 <= 65, f"Tier 1 too high: {t1} (expected ~60)"
        assert t2 >= 25, f"Tier 2 too low: {t2} (expected ~30)"
        assert t2 <= 35, f"Tier 2 too high: {t2} (expected ~30)"
        assert t3 >= 5, f"Tier 3 too low: {t3} (expected ~10)"

    def test_no_preferred_brands_returns_unchanged(self, pipeline):
        """If no preferred brands, bucketing should return all candidates unchanged."""
        pool = make_pool({"Boohoo": 20, "Zara": 20})
        result = pipeline._apply_brand_tier_bucketing(
            candidates=pool,
            preferred_brands=[],
            target_size=100,
            seed="test",
        )
        assert len(result) == len(pool)

    def test_small_pool_keeps_everything(self, pipeline):
        """If pool is smaller than target, keep all items."""
        pool = make_pool({"Boohoo": 10, "Zara": 10})
        result = pipeline._apply_brand_tier_bucketing(
            candidates=pool,
            preferred_brands=["Boohoo"],
            target_size=100,
            seed="test",
        )
        # Pool is 20, target is 100 -> keep all 20
        assert len(result) == 20

    def test_backfill_when_tier1_short(self, pipeline):
        """If preferred brands have few items, backfill from other tiers."""
        pool = (
            make_pool({"Boohoo": 5}) +                # tier 1: only 5
            make_pool({"Missguided": 40}) +            # tier 2: cluster H adjacent
            make_pool({"Zara": 60, "Nike": 30})        # tier 3: discovery
        )

        result = pipeline._apply_brand_tier_bucketing(
            candidates=pool,
            preferred_brands=["Boohoo"],
            target_size=50,
            seed="test",
        )

        assert len(result) == 50
        # Even though tier1 only had 5, we should still get 50 total via backfill
        t1 = sum(1 for c in result if (c.brand or "").lower() == "boohoo")
        assert t1 == 5  # All preferred brand items should be included

    def test_backfill_when_tier2_empty(self, pipeline):
        """If no cluster-adjacent brands exist, fill with preferred + discovery."""
        # Use a brand with no cluster mapping (unknown brand)
        # Actually, let's use brands where all cluster-adjacent are also preferred
        pool = (
            make_pool({"Boohoo": 40, "Missguided": 40, "PrettyLittleThing": 40}) +
            make_pool({"Zara": 30, "Nike": 20})
        )

        # If all cluster H brands are preferred, tier 2 is empty
        result = pipeline._apply_brand_tier_bucketing(
            candidates=pool,
            preferred_brands=["Boohoo", "Missguided", "PrettyLittleThing",
                              "Nasty Gal", "Shein", "Fashion Nova", "Meshki",
                              "Oh Polly", "White Fox", "PLT"],
            target_size=100,
            seed="test",
        )

        assert len(result) == 100

    def test_deterministic_with_same_seed(self, pipeline):
        """Same seed should produce same ordering."""
        pool = make_pool({"Boohoo": 50, "Zara": 50, "Nike": 50})

        result1 = pipeline._apply_brand_tier_bucketing(
            candidates=pool,
            preferred_brands=["Boohoo"],
            target_size=80,
            seed="user-123",
        )
        result2 = pipeline._apply_brand_tier_bucketing(
            candidates=pool,
            preferred_brands=["Boohoo"],
            target_size=80,
            seed="user-123",
        )

        ids1 = [c.item_id for c in result1]
        ids2 = [c.item_id for c in result2]
        assert ids1 == ids2

    def test_different_seeds_differ(self, pipeline):
        """Different seeds should produce different orderings."""
        pool = make_pool({"Boohoo": 50, "Zara": 50, "Nike": 50})

        result1 = pipeline._apply_brand_tier_bucketing(
            candidates=pool,
            preferred_brands=["Boohoo"],
            target_size=80,
            seed="user-111",
        )
        result2 = pipeline._apply_brand_tier_bucketing(
            candidates=pool,
            preferred_brands=["Boohoo"],
            target_size=80,
            seed="user-222",
        )

        ids1 = [c.item_id for c in result1]
        ids2 = [c.item_id for c in result2]
        # Very unlikely to be identical with different seeds
        assert ids1 != ids2

    def test_brandless_candidates_go_to_tier3(self, pipeline):
        """Candidates with no brand should end up in tier 3 (discovery)."""
        pool = (
            make_pool({"Boohoo": 30}) +
            [make_candidate(item_id=f"nobrand-{i}", brand="") for i in range(20)]
        )

        result = pipeline._apply_brand_tier_bucketing(
            candidates=pool,
            preferred_brands=["Boohoo"],
            target_size=40,
            seed="test",
        )

        assert len(result) == 40
        # Brand-less items should be present (in tier 3)
        brandless_count = sum(1 for c in result if not c.brand)
        assert brandless_count > 0
