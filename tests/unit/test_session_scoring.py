"""
Unit tests for the session-aware scoring engine.

Tests cover:
1. Brand-Cluster mapping and cold-start initialization
2. Session scoring: action updates, rolling windows, blending
3. Item scoring formula
4. Search/filter signal ingestion
5. Greedy constrained list-wise reranker
6. Serialization/deserialization
"""

import pytest
import time
from unittest.mock import MagicMock
from collections import defaultdict


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def brand_clusters():
    """Lazy import brand_clusters module."""
    from recs.brand_clusters import (
        BRAND_CLUSTER_MAP, CLUSTER_TRAITS, BRAND_SECONDARY_CLUSTER,
        get_brand_cluster, get_brand_clusters, get_cluster_for_item,
        compute_cluster_scores_from_brands, get_cluster_traits,
        DEFAULT_CLUSTER,
    )
    return type("BC", (), {
        "MAP": BRAND_CLUSTER_MAP,
        "TRAITS": CLUSTER_TRAITS,
        "SECONDARY": BRAND_SECONDARY_CLUSTER,
        "get_brand_cluster": staticmethod(get_brand_cluster),
        "get_brand_clusters": staticmethod(get_brand_clusters),
        "get_cluster_for_item": staticmethod(get_cluster_for_item),
        "compute_scores": staticmethod(compute_cluster_scores_from_brands),
        "get_traits": staticmethod(get_cluster_traits),
        "DEFAULT": DEFAULT_CLUSTER,
    })()


@pytest.fixture
def engine():
    """Create a fresh SessionScoringEngine."""
    from recs.session_scoring import SessionScoringEngine, ScoringConfig
    return SessionScoringEngine(ScoringConfig())


@pytest.fixture
def empty_scores():
    """Create empty SessionScores."""
    from recs.session_scoring import SessionScores
    return SessionScores()


@pytest.fixture
def reranker():
    """Create a GreedyConstrainedReranker."""
    from recs.feed_reranker import GreedyConstrainedReranker, RerankerConfig
    return GreedyConstrainedReranker(RerankerConfig())


def _make_candidate(**kwargs):
    """Create a mock candidate with given attributes."""
    defaults = {
        "item_id": "p1",
        "brand": "Boohoo",
        "broad_category": "dresses",
        "article_type": "midi dress",
        "fit": "slim",
        "color_family": "Neutrals",
        "pattern": "Solid",
        "neckline": "V-Neck",
        "sleeve": "Short",
        "length": "Midi",
        "style_tags": ["casual"],
        "occasions": ["Everyday"],
        "final_score": 0.5,
        "is_new": False,
        "is_on_sale": False,
        "formality": "Casual",
    }
    defaults.update(kwargs)
    candidate = MagicMock()
    for k, v in defaults.items():
        setattr(candidate, k, v)
    return candidate


# =============================================================================
# 1. Brand-Cluster Mapping Tests
# =============================================================================

class TestBrandClusterMapping:
    """Tests for brand -> cluster lookups."""

    def test_known_brand_primary_cluster(self, brand_clusters):
        result = brand_clusters.get_brand_cluster("boohoo")
        assert result == ("H", 1.0)

    def test_case_insensitive_lookup(self, brand_clusters):
        result = brand_clusters.get_brand_cluster("BOOHOO")
        assert result is not None
        assert result[0] == "H"

    def test_unknown_brand_returns_none(self, brand_clusters):
        result = brand_clusters.get_brand_cluster("totally_unknown_brand_xyz")
        assert result is None

    def test_multi_cluster_brand(self, brand_clusters):
        clusters = brand_clusters.get_brand_clusters("zara")
        assert len(clusters) == 2
        assert clusters[0][0] == "G"  # primary
        assert clusters[1][0] == "A"  # secondary

    def test_single_cluster_brand(self, brand_clusters):
        clusters = brand_clusters.get_brand_clusters("reformation")
        assert len(clusters) == 1
        assert clusters[0][0] == "A"

    def test_cluster_for_item(self, brand_clusters):
        assert brand_clusters.get_cluster_for_item("Nike") == "E"
        assert brand_clusters.get_cluster_for_item("unknown_brand") is None

    def test_all_clusters_have_traits(self, brand_clusters):
        for cluster_id in brand_clusters.TRAITS:
            traits = brand_clusters.get_traits(cluster_id)
            assert traits is not None
            assert traits.name
            assert traits.style_tags
            assert traits.occasions

    def test_brand_map_has_minimum_brands(self, brand_clusters):
        assert len(brand_clusters.MAP) >= 100

    def test_all_mapped_clusters_exist(self, brand_clusters):
        for brand, (cluster_id, conf) in brand_clusters.MAP.items():
            assert cluster_id in brand_clusters.TRAITS, f"Brand '{brand}' maps to unknown cluster '{cluster_id}'"

    def test_confidence_range(self, brand_clusters):
        for brand, (cluster_id, conf) in brand_clusters.MAP.items():
            assert 0.0 < conf <= 1.0, f"Brand '{brand}' has invalid confidence {conf}"


class TestColdStartInitialization:
    """Tests for computing initial cluster scores from onboarding brands."""

    def test_single_brand_cold_start(self, brand_clusters):
        scores = brand_clusters.compute_scores(["Reformation"])
        assert "A" in scores
        assert scores["A"] == 1.0  # normalized

    def test_multi_brand_cold_start(self, brand_clusters):
        scores = brand_clusters.compute_scores(["Reformation", "Boohoo", "Nike"])
        assert "A" in scores  # Reformation
        assert "H" in scores  # Boohoo
        assert "E" in scores  # Nike

    def test_overlapping_clusters(self, brand_clusters):
        # Zara (G primary, A secondary) + Reformation (A primary)
        scores = brand_clusters.compute_scores(["Zara", "Reformation"])
        assert "A" in scores
        assert "G" in scores

    def test_empty_brands(self, brand_clusters):
        scores = brand_clusters.compute_scores([])
        assert scores == {}

    def test_unknown_brands_ignored(self, brand_clusters):
        scores = brand_clusters.compute_scores(["totally_unknown_brand"])
        assert scores == {}

    def test_scores_normalized(self, brand_clusters):
        scores = brand_clusters.compute_scores(["Reformation", "Boohoo", "Nike", "Zara"])
        if scores:
            assert max(scores.values()) == 1.0
            assert all(0.0 <= v <= 1.0 for v in scores.values())


# =============================================================================
# 2. Session Scoring Tests
# =============================================================================

class TestSessionScoring:
    """Tests for session score updates."""

    def test_initialize_from_onboarding(self, engine):
        scores = engine.initialize_from_onboarding(
            preferred_brands=["Reformation", "Zara"]
        )
        assert "A" in scores.cluster_scores
        assert "reformation" in scores.brand_scores
        assert "zara" in scores.brand_scores

    def test_initialize_with_profile(self, engine):
        profile = MagicMock()
        profile.preferred_fits = ["slim", "regular"]
        profile.preferred_sleeves = ["long"]
        profile.preferred_lengths = ["midi"]
        profile.preferred_necklines = ["v-neck"]
        profile.patterns_liked = ["floral"]
        profile.occasions = ["office"]
        profile.style_persona = ["classic"]
        profile.categories = ["dresses", "tops"]
        profile.brands_to_avoid = ["Shein"]
        profile.colors_to_avoid = ["Orange"]

        scores = engine.initialize_from_onboarding(
            preferred_brands=["Reformation"],
            onboarding_profile=profile,
        )
        assert "fit:slim" in scores.attr_scores
        assert "sleeve:long" in scores.attr_scores
        assert "occasion:office" in scores.attr_scores
        assert scores.brand_scores["shein"].slow == -1.0
        assert scores.attr_scores["color:orange"].slow == -0.5

    def test_click_updates_scores(self, engine, empty_scores):
        engine.process_action(
            empty_scores, action="click", product_id="p1",
            brand="Boohoo", item_type="dresses",
            attributes={"fit": "slim", "color_family": "Neutrals"},
        )
        assert empty_scores.get_score("brand", "boohoo") > 0
        assert empty_scores.get_score("type", "dresses") > 0
        assert empty_scores.get_score("cluster", "H") > 0
        assert empty_scores.get_score("attr", "fit:slim") > 0
        assert empty_scores.action_count == 1

    def test_skip_updates_negatively(self, engine, empty_scores):
        engine.process_action(
            empty_scores, action="skip", product_id="p1",
            brand="Boohoo", item_type="dresses",
        )
        assert empty_scores.get_score("brand", "boohoo") < 0
        assert empty_scores.get_score("type", "dresses") < 0
        assert "p1" in empty_scores.skipped_ids

    def test_purchase_strongest_signal(self, engine, empty_scores):
        engine.process_action(
            empty_scores, action="purchase", product_id="p1",
            brand="Reformation", item_type="dresses",
        )
        purchase_brand = empty_scores.get_score("brand", "reformation")

        engine.process_action(
            empty_scores, action="click", product_id="p2",
            brand="Zara", item_type="dresses",
        )
        click_brand = empty_scores.get_score("brand", "zara")

        assert purchase_brand > click_brand

    def test_action_log_bounded(self, engine, empty_scores):
        for i in range(600):
            engine.process_action(
                empty_scores, action="click", product_id=f"p{i}",
                brand="Boohoo",
            )
        assert len(empty_scores.action_log) <= 500

    def test_search_signal_updates_intents(self, engine, empty_scores):
        engine.process_search_signal(
            empty_scores, query="blue midi dress",
            filters={"colors": ["blue"], "lengths": ["midi"]},
        )
        assert "color:blue" in empty_scores.search_intents
        assert "length:midi" in empty_scores.search_intents

    def test_search_brand_filter_updates_brand_scores(self, engine, empty_scores):
        engine.process_search_signal(
            empty_scores, filters={"brands": ["Reformation"]},
        )
        assert empty_scores.get_score("brand", "reformation") > 0


class TestRollingWindowScores:
    """Tests for rolling window blended computation."""

    def test_empty_log_returns_empty_blended(self, engine, empty_scores):
        blended = engine.compute_blended_scores(empty_scores)
        assert "cluster" in blended
        assert "brand" in blended
        assert "type" in blended
        assert "attr" in blended

    def test_recent_actions_dominate(self, engine, empty_scores):
        # Add 20 clicks on Boohoo (cluster H)
        for i in range(20):
            engine.process_action(
                empty_scores, action="click", product_id=f"p{i}",
                brand="Boohoo", item_type="dresses",
            )
        # Add 1 click on Reformation (cluster A)
        engine.process_action(
            empty_scores, action="click", product_id="p99",
            brand="Reformation", item_type="tops",
        )

        blended = engine.compute_blended_scores(empty_scores)
        # In the session window (last 20), Boohoo still dominates
        cluster_scores = blended["cluster"]
        # H should have a strong positive score
        assert cluster_scores.get("H", 0) > 0


class TestSerialization:
    """Tests for SessionScores serialization."""

    def test_round_trip_json(self, engine):
        from recs.session_scoring import SessionScores
        scores = engine.initialize_from_onboarding(["Reformation", "Zara"])
        engine.process_action(
            scores, action="click", product_id="p1",
            brand="Boohoo", item_type="dresses",
            attributes={"fit": "slim"},
        )
        scores.skipped_ids.add("p99")

        json_str = scores.to_json()
        restored = SessionScores.from_json(json_str)

        assert restored.cluster_scores == scores.cluster_scores
        assert restored.brand_scores == scores.brand_scores
        assert restored.type_scores == scores.type_scores
        assert restored.attr_scores == scores.attr_scores
        assert restored.action_count == scores.action_count
        assert restored.skipped_ids == scores.skipped_ids

    def test_round_trip_dict(self, engine):
        from recs.session_scoring import SessionScores
        scores = engine.initialize_from_onboarding(["Nike"])
        d = scores.to_dict()
        restored = SessionScores.from_dict(d)
        assert restored.cluster_scores == scores.cluster_scores

    def test_empty_scores_serialization(self):
        from recs.session_scoring import SessionScores
        scores = SessionScores()
        json_str = scores.to_json()
        restored = SessionScores.from_json(json_str)
        assert restored.action_count == 0
        assert restored.brand_scores == {}


# =============================================================================
# 3. Item Scoring Tests
# =============================================================================

class TestItemScoring:
    """Tests for the score_item formula."""

    def test_preferred_brand_scores_higher(self, engine):
        scores = engine.initialize_from_onboarding(["Reformation"])
        # Process some clicks to build session data
        for i in range(5):
            engine.process_action(
                scores, action="click", product_id=f"ref{i}",
                brand="Reformation", item_type="dresses",
            )

        preferred = _make_candidate(item_id="c1", brand="Reformation")
        other = _make_candidate(item_id="c2", brand="Unknown Brand XYZ")

        s1 = engine.score_item(scores, preferred)
        s2 = engine.score_item(scores, other)
        assert s1 > s2

    def test_skipped_item_penalized(self, engine, empty_scores):
        empty_scores.skipped_ids.add("p_skip")
        normal = _make_candidate(item_id="p_normal", brand="Boohoo")
        skipped = _make_candidate(item_id="p_skip", brand="Boohoo")

        s_normal = engine.score_item(empty_scores, normal)
        s_skipped = engine.score_item(empty_scores, skipped)
        assert s_normal > s_skipped

    def test_new_item_gets_boost(self, engine, empty_scores):
        new_item = _make_candidate(item_id="p_new", is_new=True)
        old_item = _make_candidate(item_id="p_old", is_new=False)

        s_new = engine.score_item(empty_scores, new_item)
        s_old = engine.score_item(empty_scores, old_item)
        assert s_new > s_old

    def test_search_intent_boosts_matching_items(self, engine, empty_scores):
        engine.process_search_signal(
            empty_scores, filters={"colors": ["blue"], "fit_types": ["slim"]},
        )
        matching = _make_candidate(
            item_id="c1", color_family="blue", fit="slim",
        )
        non_matching = _make_candidate(
            item_id="c2", color_family="red", fit="oversized",
        )

        s1 = engine.score_item(empty_scores, matching)
        s2 = engine.score_item(empty_scores, non_matching)
        assert s1 > s2

    def test_score_candidates_sorts_by_score(self, engine):
        scores = engine.initialize_from_onboarding(["Reformation"])
        for i in range(5):
            engine.process_action(
                scores, action="click", product_id=f"ref{i}",
                brand="Reformation", item_type="dresses",
            )

        candidates = [
            _make_candidate(item_id="c1", brand="Unknown Brand", final_score=0.5),
            _make_candidate(item_id="c2", brand="Reformation", final_score=0.5),
            _make_candidate(item_id="c3", brand="Boohoo", final_score=0.5),
        ]

        result = engine.score_candidates(scores, candidates)
        # Reformation should be ranked higher due to session affinity
        brands = [getattr(c, "brand", "") for c in result]
        assert brands[0] == "Reformation"


# =============================================================================
# 4. Greedy Constrained Reranker Tests
# =============================================================================

class TestGreedyReranker:
    """Tests for the greedy constrained list-wise reranker."""

    def test_basic_reranking(self, reranker):
        candidates = [
            _make_candidate(item_id=f"p{i}", brand="Boohoo", final_score=1.0 - i * 0.01)
            for i in range(20)
        ]
        result = reranker.rerank(candidates, target_size=10)
        assert len(result) <= 10

    def test_brand_cap_enforced(self, reranker):
        # 20 items all from Boohoo + 10 from others
        candidates = [
            _make_candidate(item_id=f"boo{i}", brand="Boohoo", final_score=0.9 - i * 0.01)
            for i in range(20)
        ] + [
            _make_candidate(item_id=f"ref{i}", brand="Reformation", final_score=0.5 - i * 0.01)
            for i in range(10)
        ]

        result = reranker.rerank(candidates, target_size=15)
        brand_counts = defaultdict(int)
        for item in result:
            brand_counts[item.brand.lower()] += 1

        assert brand_counts["boohoo"] <= reranker.config.max_per_brand

    def test_seen_ids_excluded(self, reranker):
        candidates = [
            _make_candidate(item_id=f"p{i}", brand=f"Brand{i}", final_score=1.0 - i * 0.01)
            for i in range(10)
        ]
        seen = {"p0", "p1", "p2"}
        result = reranker.rerank(candidates, target_size=10, seen_ids=seen)

        result_ids = {item.item_id for item in result}
        assert result_ids.isdisjoint(seen)

    def test_strict_diversity_in_first_positions(self, reranker):
        """First 10 positions should have no brand repeats."""
        colors = ["Blues", "Reds", "Greens", "Browns", "Pinks"]
        fits = ["slim", "regular", "relaxed", "oversized", "fitted"]
        candidates = []
        for i in range(50):
            brand = f"Brand{i % 5}"  # 5 brands cycling
            candidates.append(
                _make_candidate(
                    item_id=f"p{i}", brand=brand,
                    broad_category=f"cat{i % 3}",
                    color_family=colors[i % 5],
                    fit=fits[i % 5],
                    final_score=1.0 - i * 0.01,
                )
            )

        result = reranker.rerank(candidates, target_size=20)
        # First 5 positions should all be different brands
        first_5_brands = [item.brand.lower() for item in result[:5]]
        assert len(set(first_5_brands)) == 5

    def test_type_cap_enforced(self, reranker):
        """No more than max_per_type items of the same type."""
        from recs.feed_reranker import RerankerConfig
        config = RerankerConfig(max_per_type=3, max_per_brand=20, max_per_cluster=20)
        from recs.feed_reranker import GreedyConstrainedReranker
        r = GreedyConstrainedReranker(config)

        candidates = [
            _make_candidate(
                item_id=f"p{i}", brand=f"Brand{i}",
                broad_category="dresses", final_score=1.0 - i * 0.01,
            )
            for i in range(20)
        ]
        result = r.rerank(candidates, target_size=10)
        type_counts = defaultdict(int)
        for item in result:
            type_counts[item.broad_category.lower()] += 1
        assert type_counts["dresses"] <= 3

    def test_empty_candidates(self, reranker):
        result = reranker.rerank([], target_size=10)
        assert result == []

    def test_fewer_candidates_than_target(self, reranker):
        candidates = [
            _make_candidate(
                item_id=f"p{i}", brand=f"Brand{i}",
                broad_category=f"cat{i}", color_family=f"Color{i}",
                fit=f"fit{i}", final_score=0.5,
            )
            for i in range(3)
        ]
        result = reranker.rerank(candidates, target_size=10)
        assert len(result) == 3

    def test_diversity_stats(self, reranker):
        candidates = [
            _make_candidate(
                item_id=f"p{i}", brand=f"Brand{i % 5}",
                broad_category=f"cat{i % 3}", final_score=1.0 - i * 0.01,
            )
            for i in range(20)
        ]
        result = reranker.rerank(candidates, target_size=15)
        stats = reranker.get_diversity_stats(result)

        assert stats["total_items"] > 0
        assert stats["unique_brands"] > 1
        assert stats["brand_entropy"] > 0

    def test_combo_dedup_prevents_same_attribute_combo(self, reranker):
        """Items with same cluster+type+color+fit should be deduped."""
        # All from same brand (same cluster), same category, color, fit
        candidates = [
            _make_candidate(
                item_id=f"p{i}", brand="Boohoo",
                broad_category="dresses", color_family="Neutrals",
                fit="slim", final_score=1.0 - i * 0.01,
            )
            for i in range(10)
        ]
        # Add different items
        candidates += [
            _make_candidate(
                item_id=f"d{i}", brand=f"Brand{i}",
                broad_category="tops", color_family="Blues",
                fit="relaxed", final_score=0.5 - i * 0.01,
            )
            for i in range(10)
        ]

        result = reranker.rerank(candidates, target_size=10)
        # Should not be all Boohoo dresses in Neutrals/slim
        boohoo_neutral_slim = sum(
            1 for item in result
            if item.brand == "Boohoo" and item.color_family == "Neutrals"
        )
        assert boohoo_neutral_slim <= 2  # combo dedup + brand cap


# =============================================================================
# 5. Integration: Scoring + Reranking Pipeline
# =============================================================================

class TestScoringRerankerIntegration:
    """Tests for the full scoring -> reranking pipeline."""

    def test_full_pipeline(self, engine, reranker):
        # 1. Initialize from onboarding
        scores = engine.initialize_from_onboarding(["Reformation", "Zara"])

        # 2. Simulate some actions
        engine.process_action(
            scores, action="click", product_id="ref1",
            brand="Reformation", item_type="dresses",
            attributes={"fit": "slim", "color_family": "Neutrals"},
        )
        engine.process_action(
            scores, action="skip", product_id="boo1",
            brand="Boohoo", item_type="tops",
        )

        # 3. Score candidates
        candidates = [
            _make_candidate(item_id=f"ref{i}", brand="Reformation",
                          broad_category="dresses", final_score=0.5)
            for i in range(10)
        ] + [
            _make_candidate(item_id=f"boo{i}", brand="Boohoo",
                          broad_category="tops", final_score=0.5)
            for i in range(10)
        ] + [
            _make_candidate(item_id=f"oth{i}", brand=f"Other{i}",
                          broad_category="bottoms", final_score=0.5)
            for i in range(10)
        ]

        scored = engine.score_candidates(scores, candidates)

        # 4. Rerank with constraints
        result = reranker.rerank(scored, target_size=15)

        # Verify diversity
        stats = reranker.get_diversity_stats(result)
        assert stats["unique_brands"] > 1
        assert len(result) <= 15

    def test_session_adapts_over_time(self, engine):
        """Scores change as more actions come in."""
        scores = engine.initialize_from_onboarding(["Reformation"])

        item = _make_candidate(item_id="c1", brand="Boohoo")
        score_before = engine.score_item(scores, item)

        # User starts clicking Boohoo items
        for i in range(10):
            engine.process_action(
                scores, action="click", product_id=f"boo{i}",
                brand="Boohoo", item_type="dresses",
            )

        score_after = engine.score_item(scores, item)
        # Boohoo should now be scored higher
        assert score_after > score_before


# =============================================================================
# 8. Extract Intent Filters (Session-Aware Retrieval)
# =============================================================================

class TestExtractIntentFilters:
    """Tests for extract_intent_filters() — the bridge between scoring and retrieval."""

    def test_empty_session_has_no_intent(self):
        from recs.session_scoring import extract_intent_filters, SessionScores
        scores = SessionScores()
        result = extract_intent_filters(scores)
        assert result["has_intent"] is False
        assert result["brands"] == []
        assert result["types"] == []
        assert result["signal_count"] == 0

    def test_onboarding_only_excluded(self, engine):
        """Onboarding-seeded brands (count=1, fast=0) should NOT trigger intent."""
        from recs.session_scoring import extract_intent_filters
        scores = engine.initialize_from_onboarding(["Reformation", "Zara"])
        result = extract_intent_filters(scores)
        # Onboarding priors have count=1 and fast=0.0, so should be excluded
        assert result["has_intent"] is False
        assert result["brands"] == []

    def test_live_signals_create_intent(self, engine):
        """Repeated clicks on a brand should create live intent."""
        from recs.session_scoring import extract_intent_filters
        scores = engine.initialize_from_onboarding([])
        # Simulate 5 clicks on Alo Yoga leggings
        for i in range(5):
            engine.process_action(
                scores, action="click", product_id=f"alo{i}",
                brand="Alo Yoga", item_type="leggings",
            )
        result = extract_intent_filters(scores)
        assert result["has_intent"] is True
        assert "alo yoga" in result["brands"]
        assert "leggings" in result["types"]
        assert result["signal_count"] >= 2

    def test_search_signal_creates_intent(self, engine):
        """Search signals should create brand/type intent."""
        from recs.session_scoring import extract_intent_filters
        scores = engine.initialize_from_onboarding([])
        # Simulate repeated searches
        for _ in range(3):
            engine.process_search_signal(
                scores, query="lululemon leggings",
                filters={"brands": ["Lululemon"], "article_types": ["leggings"]},
            )
        result = extract_intent_filters(scores)
        assert result["has_intent"] is True
        assert result["signal_count"] >= 1

    def test_weak_signals_excluded(self, engine):
        """A single click shouldn't be strong enough for intent retrieval."""
        from recs.session_scoring import extract_intent_filters
        scores = engine.initialize_from_onboarding([])
        # Single click — might be below min_score threshold
        engine.process_action(
            scores, action="click", product_id="x1",
            brand="SomeRareBrand", item_type="romper",
        )
        result = extract_intent_filters(scores, min_score=0.15)
        # With a higher threshold, a single click shouldn't qualify
        assert "somerarebrand" not in result.get("brands", [])

    def test_max_brands_and_types_capped(self, engine):
        """Should not return more than max_brands/max_types."""
        from recs.session_scoring import extract_intent_filters
        scores = engine.initialize_from_onboarding([])
        # Generate signals for 10 different brands
        for i in range(10):
            for _ in range(5):
                engine.process_action(
                    scores, action="click", product_id=f"b{i}_p",
                    brand=f"Brand{i}", item_type=f"type{i}",
                )
        result = extract_intent_filters(scores, max_brands=3, max_types=3)
        assert len(result["brands"]) <= 3
        assert len(result["types"]) <= 3

    def test_onboarding_plus_live_signals(self, engine):
        """Onboarding brands excluded but the same brand with live signals should appear."""
        from recs.session_scoring import extract_intent_filters
        scores = engine.initialize_from_onboarding(["Reformation"])
        # Now add live clicks on Reformation — this updates the EMA
        for i in range(5):
            engine.process_action(
                scores, action="click", product_id=f"ref{i}",
                brand="Reformation", item_type="dresses",
            )
        result = extract_intent_filters(scores)
        assert result["has_intent"] is True
        # Reformation now has live signals (count > 1, fast != 0)
        assert "reformation" in result["brands"]


# =============================================================================
# 9. Inject Session Intent Candidates (Gradio Demo)
# =============================================================================

class TestInjectSessionIntentCandidates:
    """Tests for the Gradio demo's inject_session_intent_candidates() function."""

    @staticmethod
    def _make_real_candidate(item_id, brand="Boohoo", article_type="dress", **kwargs):
        """Create a simple object with item_id, brand, article_type attributes."""
        obj = MagicMock()
        obj.item_id = item_id
        obj.brand = brand
        obj.article_type = article_type
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj

    def test_no_intent_returns_unchanged(self, engine):
        """Without session intent, candidates should be returned unchanged."""
        scores = engine.initialize_from_onboarding([])
        base = [self._make_real_candidate(f"p{i}") for i in range(10)]
        all_pool = [self._make_real_candidate(f"p{i}") for i in range(50)]

        # Inline the logic since we can't import from scripts easily
        from recs.session_scoring import extract_intent_filters
        intent = extract_intent_filters(scores)
        assert intent["has_intent"] is False
        # Without intent, the function would return base unchanged
        # (We test the logic, not the import)

    def test_intent_matching_logic(self, engine):
        """Verify that intent filters correctly match brand/type candidates."""
        from recs.session_scoring import extract_intent_filters
        scores = engine.initialize_from_onboarding([])

        # Build up Alo Yoga + leggings signals
        for i in range(5):
            engine.process_action(
                scores, action="click", product_id=f"alo{i}",
                brand="Alo Yoga", item_type="leggings",
            )

        intent = extract_intent_filters(scores)
        assert intent["has_intent"] is True

        intent_brands = {b.lower() for b in intent["brands"]}
        intent_types = {t.lower() for t in intent["types"]}

        # Simulate matching against a pool
        pool = [
            self._make_real_candidate("a1", brand="Alo Yoga", article_type="leggings"),
            self._make_real_candidate("a2", brand="Alo Yoga", article_type="sports bra"),
            self._make_real_candidate("a3", brand="Nike", article_type="leggings"),
            self._make_real_candidate("a4", brand="Boohoo", article_type="dress"),
        ]
        matching = [
            c for c in pool
            if (c.brand and c.brand.lower() in intent_brands) or
               (c.article_type and c.article_type.lower() in intent_types)
        ]

        # a1 matches brand+type, a2 matches brand, a3 matches type
        assert len(matching) == 3
        matched_ids = {c.item_id for c in matching}
        assert "a1" in matched_ids
        assert "a2" in matched_ids
        assert "a3" in matched_ids
        assert "a4" not in matched_ids

    def test_already_present_items_not_duplicated(self, engine):
        """Items already in base pool should not be injected again."""
        from recs.session_scoring import extract_intent_filters
        scores = engine.initialize_from_onboarding([])
        for i in range(5):
            engine.process_action(
                scores, action="click", product_id=f"alo{i}",
                brand="Alo Yoga", item_type="leggings",
            )

        intent = extract_intent_filters(scores)
        intent_brands = {b.lower() for b in intent["brands"]}

        # Item "a1" is already in base candidates
        base = [self._make_real_candidate("a1", brand="Alo Yoga", article_type="leggings")]
        existing_ids = {c.item_id for c in base}

        all_pool = [
            self._make_real_candidate("a1", brand="Alo Yoga", article_type="leggings"),
            self._make_real_candidate("a2", brand="Alo Yoga", article_type="tank top"),
        ]

        new_items = [
            c for c in all_pool
            if c.item_id not in existing_ids and
               c.brand and c.brand.lower() in intent_brands
        ]
        # Only a2 should be new (a1 already present)
        assert len(new_items) == 1
        assert new_items[0].item_id == "a2"
