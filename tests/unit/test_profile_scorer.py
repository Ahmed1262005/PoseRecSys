"""
Unit tests for the ProfileScorer (attribute-driven profile preference scoring).

Tests cover all 10 scoring dimensions:
1. Brand preference (exact, cluster-adjacent, unrelated, openness)
2. Style tag matching (direct persona -> style_tags)
3. Formality alignment (inferred from persona)
4. Attribute preferences (fit, sleeve, length, neckline, rise) with category mapping
5. Type preferences (article type per category)
6. Pattern scoring (liked / avoided)
7. Occasion matching (with expansion)
8. Color avoid
9. Price range (derived from brands)
10. Coverage hard-kills

Run with: PYTHONPATH=src python -m pytest tests/unit/test_profile_scorer.py -v
"""

import pytest
from scoring.profile_scorer import (
    ProfileScorer,
    ProfileScoringConfig,
    _ProfileFields,
    _expand_occasions,
    _infer_formality_preference,
    _attribute_applies,
)
from scoring.constants.brand_data import (
    BRAND_TO_CLUSTER,
    BRAND_CLUSTERS,
    derive_price_range,
    get_brand_cluster,
    get_brand_avg_price,
    get_user_brand_clusters,
)


# =============================================================================
# Test helpers
# =============================================================================

def _make_item(**overrides) -> dict:
    """Create a minimal item dict with sensible defaults."""
    defaults = {
        "product_id": "prod-001",
        "article_type": "t-shirt",
        "broad_category": "tops",
        "brand": "TestBrand",
        "style_tags": ["casual"],
        "occasions": ["Everyday"],
        "pattern": "Solid",
        "formality": "Casual",
        "fit_type": "regular",
        "neckline": "crew",
        "sleeve_type": "short",
        "length": "standard",
        "color_family": "Neutrals",
        "primary_color": "black",
        "price": 30.0,
        "name": "Test T-Shirt",
    }
    defaults.update(overrides)
    return defaults


def _make_profile(**overrides) -> dict:
    """Create a minimal profile dict."""
    defaults = {
        "preferred_brands": [],
        "brand_openness": None,
        "style_persona": [],
        "style_directions": [],
        "preferred_fits": [],
        "fit_category_mapping": [],
        "preferred_sleeves": [],
        "sleeve_category_mapping": [],
        "preferred_lengths": [],
        "length_category_mapping": [],
        "preferred_lengths_dresses": [],
        "length_dresses_category_mapping": [],
        "preferred_necklines": [],
        "preferred_rises": [],
        "top_types": [],
        "bottom_types": [],
        "dress_types": [],
        "outerwear_types": [],
        "patterns_liked": [],
        "patterns_avoided": [],
        "occasions": [],
        "colors_to_avoid": [],
        "global_min_price": None,
        "global_max_price": None,
        "no_crop": False,
        "no_revealing": False,
        "no_deep_necklines": False,
        "no_sleeveless": False,
        "no_tanks": False,
        "styles_to_avoid": [],
    }
    defaults.update(overrides)
    return defaults


# =============================================================================
# Brand data tests
# =============================================================================

class TestBrandData:
    """Tests for brand_data.py constants and helpers."""

    def test_brand_to_cluster_populated(self):
        assert len(BRAND_TO_CLUSTER) > 100

    def test_brand_to_cluster_case_insensitive(self):
        assert get_brand_cluster("Zara") == "G"
        assert get_brand_cluster("zara") == "G"
        assert get_brand_cluster("ZARA") == "G"  # get_brand_cluster lowercases input

    def test_get_brand_avg_price(self):
        assert get_brand_avg_price("Zara") == 30
        assert get_brand_avg_price("Unknown Brand") == 0.0

    def test_derive_price_range(self):
        lo, hi = derive_price_range(["Zara", "H&M"])
        assert lo > 0
        assert hi > lo
        # Zara=30, H&M=18, min*0.4 = 7.2, max*2.0 = 60
        assert lo == pytest.approx(18 * 0.4, rel=0.01)
        assert hi == pytest.approx(30 * 2.0, rel=0.01)

    def test_derive_price_range_unknown_brands(self):
        lo, hi = derive_price_range(["UnknownBrand1", "UnknownBrand2"])
        assert lo == 0.0
        assert hi == 0.0

    def test_get_user_brand_clusters(self):
        clusters = get_user_brand_clusters(["Zara", "Mango", "Paige"])
        assert "G" in clusters  # Zara, Mango
        assert "B" in clusters  # Paige

    def test_cluster_completeness(self):
        """Every brand in BRAND_CLUSTERS should be in BRAND_TO_CLUSTER."""
        for cid, brands in BRAND_CLUSTERS.items():
            for brand in brands:
                assert brand.lower() in BRAND_TO_CLUSTER, f"{brand} missing from reverse lookup"
                assert BRAND_TO_CLUSTER[brand.lower()] == cid


# =============================================================================
# Occasion expansion tests
# =============================================================================

class TestOccasionExpansion:

    def test_expand_casual(self):
        expanded = _expand_occasions(["casual"])
        assert "everyday" in expanded
        assert "casual" in expanded

    def test_expand_office(self):
        expanded = _expand_occasions(["office"])
        assert "office" in expanded
        assert "work" in expanded

    def test_expand_evening(self):
        expanded = _expand_occasions(["evening"])
        assert "date night" in expanded
        assert "party" in expanded

    def test_expand_unknown_passthrough(self):
        expanded = _expand_occasions(["Garden Party"])
        assert "garden party" in expanded

    def test_expand_empty(self):
        assert _expand_occasions([]) == set()


# =============================================================================
# Formality inference tests
# =============================================================================

class TestFormalityInference:

    def test_classic_is_formal(self):
        assert _infer_formality_preference(["classic"]) == "formal"

    def test_elegant_is_formal(self):
        assert _infer_formality_preference(["elegant"]) == "formal"

    def test_casual_is_casual(self):
        assert _infer_formality_preference(["casual"]) == "casual"

    def test_sporty_is_casual(self):
        assert _infer_formality_preference(["sporty"]) == "casual"

    def test_mixed_is_none(self):
        assert _infer_formality_preference(["classic", "casual"]) is None

    def test_trendy_is_none(self):
        # "trendy" is in neither set
        assert _infer_formality_preference(["trendy"]) is None

    def test_empty_is_none(self):
        assert _infer_formality_preference([]) is None


# =============================================================================
# ProfileScorer: Brand scoring
# =============================================================================

class TestBrandScoring:
    """Test brand preference scoring dimension."""

    def setup_method(self):
        self.scorer = ProfileScorer()

    def test_no_brand_prefs_neutral(self):
        """No brand preferences -> 0.0 adjustment."""
        item = _make_item(brand="Zara")
        profile = _make_profile()
        adj = self.scorer.score_item(item, profile)
        # No brand prefs -> no brand contribution
        # (may have other contributions, so test the dimension directly)
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["brand"] == 0.0

    def test_preferred_brand_boost(self):
        item = _make_item(brand="Zara")
        profile = _make_profile(preferred_brands=["Zara"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["brand"] == pytest.approx(0.25, abs=0.01)

    def test_preferred_brand_case_insensitive(self):
        item = _make_item(brand="zara")
        profile = _make_profile(preferred_brands=["ZARA"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["brand"] == pytest.approx(0.25, abs=0.01)

    def test_cluster_adjacent_boost(self):
        """Mango is in cluster G with Zara -> +0.10."""
        item = _make_item(brand="Mango")
        profile = _make_profile(preferred_brands=["Zara"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["brand"] == pytest.approx(0.10, abs=0.01)

    def test_unrelated_brand_penalty(self):
        """Nike (cluster E) is unrelated to Zara (cluster G) -> -0.05."""
        item = _make_item(brand="Nike")
        profile = _make_profile(preferred_brands=["Zara"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["brand"] == pytest.approx(-0.05, abs=0.01)

    def test_brand_openness_stick_to_favorites(self):
        """stick_to_favorites multiplies brand_preferred by 2.0."""
        item = _make_item(brand="Zara")
        profile = _make_profile(
            preferred_brands=["Zara"],
            brand_openness="stick_to_favorites",
        )
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["brand"] == pytest.approx(0.50, abs=0.01)

    def test_brand_openness_discover_new(self):
        """discover_new multiplies brand_preferred by 0.7."""
        item = _make_item(brand="Zara")
        profile = _make_profile(
            preferred_brands=["Zara"],
            brand_openness="discover_new",
        )
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["brand"] == pytest.approx(0.175, abs=0.01)

    def test_unknown_brand_no_cluster(self):
        """Unknown brand when user has brand prefs -> unrelated penalty."""
        item = _make_item(brand="SomeTotallyUnknownBrand")
        profile = _make_profile(preferred_brands=["Zara"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["brand"] == pytest.approx(-0.05, abs=0.01)

    def test_no_brand_on_item_neutral(self):
        item = _make_item(brand="")
        profile = _make_profile(preferred_brands=["Zara"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["brand"] == 0.0


# =============================================================================
# ProfileScorer: Style tag scoring
# =============================================================================

class TestStyleScoring:

    def setup_method(self):
        self.scorer = ProfileScorer()

    def test_no_style_prefs_neutral(self):
        item = _make_item(style_tags=["casual", "trendy"])
        profile = _make_profile()
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["style"] == 0.0

    def test_direct_style_match(self):
        """style_persona='casual' matches item's style_tags containing 'casual'."""
        item = _make_item(style_tags=["casual", "modern"])
        profile = _make_profile(style_persona=["casual"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["style"] == pytest.approx(0.15, abs=0.01)

    def test_multiple_style_hits(self):
        """Two hits -> 2 * 0.15 = 0.30."""
        item = _make_item(style_tags=["casual", "modern", "trendy"])
        profile = _make_profile(style_persona=["casual", "trendy"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["style"] == pytest.approx(0.30, abs=0.01)

    def test_style_cap(self):
        """Style score capped at 0.35."""
        item = _make_item(style_tags=["casual", "trendy", "modern", "chic", "edgy"])
        profile = _make_profile(style_persona=["casual", "trendy", "modern", "chic"])
        breakdown = self.scorer.explain_item(item, profile)
        # 4 hits * 0.15 = 0.60 -> capped at 0.35
        assert breakdown["style"] == pytest.approx(0.35, abs=0.01)

    def test_no_match_penalty(self):
        """Items with zero style overlap get mismatch penalty."""
        item = _make_item(style_tags=["glamorous"])
        profile = _make_profile(style_persona=["casual"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["style"] == pytest.approx(-0.08, abs=0.01)

    def test_style_directions_fallback(self):
        """Falls back to style_directions when style_persona is empty."""
        item = _make_item(style_tags=["casual"])
        profile = _make_profile(style_persona=[], style_directions=["casual"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["style"] == pytest.approx(0.15, abs=0.01)

    def test_alias_boho_matches_bohemian(self):
        """'boho' persona matches item's 'bohemian' style_tag via alias."""
        item = _make_item(style_tags=["bohemian", "romantic"])
        profile = _make_profile(style_persona=["boho"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["style"] >= 0.15  # At least one hit

    def test_alias_glam_matches_glamorous(self):
        item = _make_item(style_tags=["glamorous", "chic"])
        profile = _make_profile(style_persona=["glam"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["style"] >= 0.15

    def test_alias_minimal_matches_minimalist(self):
        item = _make_item(style_tags=["minimalist"])
        profile = _make_profile(style_persona=["minimal"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["style"] >= 0.15

    def test_style_spread_is_significant(self):
        """Match vs mismatch spread must be > 0.20 to dominate base score."""
        match_item = _make_item(style_tags=["bohemian"])
        miss_item = _make_item(style_tags=["casual"])
        profile = _make_profile(style_persona=["boho"])
        match_score = self.scorer.score_item(match_item, profile)
        miss_score = self.scorer.score_item(miss_item, profile)
        assert match_score - miss_score >= 0.20


# =============================================================================
# ProfileScorer: Formality scoring
# =============================================================================

class TestFormalityScoring:

    def setup_method(self):
        self.scorer = ProfileScorer()

    def test_classic_persona_boosts_business_casual(self):
        item = _make_item(formality="Business Casual")
        profile = _make_profile(style_persona=["classic"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["formality"] == pytest.approx(0.10, abs=0.01)

    def test_casual_persona_boosts_casual_items(self):
        item = _make_item(formality="Casual")
        profile = _make_profile(style_persona=["sporty"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["formality"] == pytest.approx(0.10, abs=0.01)

    def test_no_formality_on_item_neutral(self):
        item = _make_item(formality="")
        profile = _make_profile(style_persona=["classic"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["formality"] == 0.0

    def test_mixed_personas_no_formality_boost(self):
        """Mixed formal + casual personas -> no formality direction."""
        item = _make_item(formality="Casual")
        profile = _make_profile(style_persona=["classic", "casual"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["formality"] == 0.0


# =============================================================================
# ProfileScorer: Attribute preferences (fit, sleeve, length, neckline, rise)
# =============================================================================

class TestAttributeScoring:

    def setup_method(self):
        self.scorer = ProfileScorer()

    def test_fit_match(self):
        item = _make_item(fit_type="regular", broad_category="tops")
        profile = _make_profile(preferred_fits=["regular"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["attributes"] >= 0.10

    def test_fit_with_category_mapping(self):
        """'fitted' only applies to 'tops', not 'bottoms'."""
        mapping = [{"fitId": "fitted", "categories": ["tops"]}]
        item_tops = _make_item(fit_type="fitted", broad_category="tops")
        item_bottoms = _make_item(fit_type="fitted", broad_category="bottoms")
        profile = _make_profile(
            preferred_fits=["fitted"],
            fit_category_mapping=mapping,
        )
        bd_tops = self.scorer.explain_item(item_tops, profile)
        bd_bottoms = self.scorer.explain_item(item_bottoms, profile)
        assert bd_tops["attributes"] > bd_bottoms["attributes"]

    def test_sleeve_match(self):
        item = _make_item(sleeve_type="short")
        profile = _make_profile(preferred_sleeves=["short"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["attributes"] >= 0.08

    def test_neckline_match(self):
        item = _make_item(neckline="v-neck")
        profile = _make_profile(preferred_necklines=["v-neck"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["attributes"] >= 0.08

    def test_length_match(self):
        item = _make_item(length="midi", broad_category="dresses")
        profile = _make_profile(preferred_lengths_dresses=["midi"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["attributes"] >= 0.08

    def test_rise_match_bottoms_only(self):
        item_bottoms = _make_item(rise="high", broad_category="bottoms")
        item_tops = _make_item(rise="high", broad_category="tops")
        profile = _make_profile(preferred_rises=["high"])
        bd_bottoms = self.scorer.explain_item(item_bottoms, profile)
        bd_tops = self.scorer.explain_item(item_tops, profile)
        assert bd_bottoms["attributes"] > bd_tops["attributes"]

    def test_multiple_attribute_hits_stack(self):
        """Multiple attribute matches stack additively."""
        item = _make_item(
            fit_type="regular",
            sleeve_type="short",
            neckline="crew",
            broad_category="tops",
        )
        profile = _make_profile(
            preferred_fits=["regular"],
            preferred_sleeves=["short"],
            preferred_necklines=["crew"],
        )
        breakdown = self.scorer.explain_item(item, profile)
        # fit(0.10) + sleeve(0.08) + neckline(0.08) = 0.26
        assert breakdown["attributes"] >= 0.25

    def test_no_attribute_prefs_neutral(self):
        item = _make_item(fit_type="regular")
        profile = _make_profile()
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["attributes"] == 0.0


# =============================================================================
# ProfileScorer: Type preferences
# =============================================================================

class TestTypeScoring:

    def setup_method(self):
        self.scorer = ProfileScorer()

    def test_top_type_match(self):
        item = _make_item(article_type="blouse", broad_category="tops")
        profile = _make_profile(top_types=["blouse"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["type"] == pytest.approx(0.15, abs=0.01)

    def test_bottom_type_match(self):
        item = _make_item(article_type="jeans", broad_category="bottoms")
        profile = _make_profile(bottom_types=["jeans"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["type"] == pytest.approx(0.15, abs=0.01)

    def test_dress_type_match(self):
        item = _make_item(article_type="wrap-dress", broad_category="dresses")
        profile = _make_profile(dress_types=["wrap-dress"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["type"] == pytest.approx(0.15, abs=0.01)

    def test_no_type_prefs_neutral(self):
        item = _make_item(article_type="blouse", broad_category="tops")
        profile = _make_profile()
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["type"] == 0.0

    def test_no_match_zero(self):
        item = _make_item(article_type="blouse", broad_category="tops")
        profile = _make_profile(top_types=["sweater", "cardigan"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["type"] == 0.0


# =============================================================================
# ProfileScorer: Pattern scoring
# =============================================================================

class TestPatternScoring:

    def setup_method(self):
        self.scorer = ProfileScorer()

    def test_liked_pattern_boost(self):
        item = _make_item(pattern="Floral")
        profile = _make_profile(patterns_liked=["Floral"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["pattern"] == pytest.approx(0.10, abs=0.01)

    def test_avoided_pattern_penalty(self):
        item = _make_item(pattern="Animal Print")
        profile = _make_profile(patterns_avoided=["Animal Print"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["pattern"] == pytest.approx(-0.15, abs=0.01)

    def test_no_pattern_neutral(self):
        item = _make_item(pattern="")
        profile = _make_profile(patterns_liked=["Floral"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["pattern"] == 0.0

    def test_pattern_case_insensitive(self):
        item = _make_item(pattern="floral")
        profile = _make_profile(patterns_liked=["Floral"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["pattern"] == pytest.approx(0.10, abs=0.01)


# =============================================================================
# ProfileScorer: Occasion scoring
# =============================================================================

class TestOccasionScoring:

    def setup_method(self):
        self.scorer = ProfileScorer()

    def test_occasion_match(self):
        item = _make_item(occasions=["Everyday", "Weekend"])
        profile = _make_profile(occasions=["casual"])  # expands to Everyday, Casual, Weekend, Lounging
        breakdown = self.scorer.explain_item(item, profile)
        # 2 hits (everyday + weekend) -> occasion_match + occasion_multi_bonus
        assert breakdown["occasions"] >= 0.10

    def test_multi_occasion_bonus(self):
        item = _make_item(occasions=["Office", "Work", "Everyday"])
        profile = _make_profile(occasions=["office", "casual"])
        breakdown = self.scorer.explain_item(item, profile)
        # Office + Work + Everyday = 3 hits -> base + multi bonus
        assert breakdown["occasions"] == pytest.approx(0.15, abs=0.01)

    def test_no_occasion_match(self):
        item = _make_item(occasions=["Party"])
        profile = _make_profile(occasions=["office"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["occasions"] == 0.0

    def test_no_occasion_prefs_neutral(self):
        item = _make_item(occasions=["Office"])
        profile = _make_profile()
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["occasions"] == 0.0


# =============================================================================
# ProfileScorer: Color avoid scoring
# =============================================================================

class TestColorAvoidScoring:

    def setup_method(self):
        self.scorer = ProfileScorer()

    def test_color_family_avoid(self):
        item = _make_item(color_family="Reds")
        profile = _make_profile(colors_to_avoid=["Reds"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["color_avoid"] == pytest.approx(-0.15, abs=0.01)

    def test_primary_color_avoid(self):
        item = _make_item(primary_color="red", color_family="Warm")
        profile = _make_profile(colors_to_avoid=["red"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["color_avoid"] == pytest.approx(-0.15, abs=0.01)

    def test_no_color_avoid_neutral(self):
        item = _make_item(color_family="Blues")
        profile = _make_profile(colors_to_avoid=["Reds"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["color_avoid"] == 0.0


# =============================================================================
# ProfileScorer: Price scoring
# =============================================================================

class TestPriceScoring:

    def setup_method(self):
        self.scorer = ProfileScorer()

    def test_price_in_range_boost(self):
        """Item in derived price range -> +0.05."""
        item = _make_item(price=35.0)
        profile = _make_profile(
            preferred_brands=["Zara"],  # avg=30, range ~12 - 60
        )
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["price"] == pytest.approx(0.05, abs=0.01)

    def test_price_way_over_penalty(self):
        """Item >1.5x max -> -0.10."""
        item = _make_item(price=200.0)
        profile = _make_profile(
            preferred_brands=["Zara"],  # max ~60, 200 > 60*1.5=90
        )
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["price"] == pytest.approx(-0.10, abs=0.01)

    def test_price_slightly_over(self):
        """Item > max but < 1.5x max -> -0.05."""
        item = _make_item(price=75.0)
        profile = _make_profile(
            preferred_brands=["Zara"],  # max ~60, 75 < 60*1.5=90
        )
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["price"] == pytest.approx(-0.05, abs=0.01)

    def test_explicit_price_range(self):
        """Explicit global_min_price/max_price takes precedence."""
        item = _make_item(price=50.0)
        profile = _make_profile(
            global_min_price=20.0,
            global_max_price=100.0,
        )
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["price"] == pytest.approx(0.05, abs=0.01)

    def test_no_price_range_neutral(self):
        item = _make_item(price=50.0)
        profile = _make_profile()
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["price"] == 0.0


# =============================================================================
# ProfileScorer: Coverage hard-kills
# =============================================================================

class TestCoverageScoring:

    def setup_method(self):
        self.scorer = ProfileScorer()

    def test_no_crop_kills_crop_top(self):
        item = _make_item(article_type="crop top", name="Cute Crop Top")
        profile = _make_profile(no_crop=True)
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["coverage"] <= -1.0

    def test_no_revealing_kills_mini_dress(self):
        item = _make_item(article_type="mini dress", length="mini")
        profile = _make_profile(no_revealing=True)
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["coverage"] <= -1.0

    def test_no_deep_necklines_kills_plunging(self):
        item = _make_item(neckline="plunging")
        profile = _make_profile(no_deep_necklines=True)
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["coverage"] <= -1.0

    def test_no_sleeveless_kills_sleeveless(self):
        item = _make_item(sleeve_type="sleeveless")
        profile = _make_profile(no_sleeveless=True)
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["coverage"] <= -1.0

    def test_styles_to_avoid_sheer(self):
        """styles_to_avoid=['sheer'] maps to no_revealing."""
        item = _make_item(style_tags=["sheer"])
        profile = _make_profile(styles_to_avoid=["sheer"])
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["coverage"] <= -1.0

    def test_no_coverage_prefs_neutral(self):
        item = _make_item(article_type="crop top")
        profile = _make_profile()
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["coverage"] == 0.0

    def test_normal_item_no_penalty(self):
        item = _make_item(
            article_type="t-shirt",
            neckline="crew",
            sleeve_type="short",
        )
        profile = _make_profile(no_crop=True, no_revealing=True)
        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["coverage"] == 0.0


# =============================================================================
# ProfileScorer: End-to-end scoring
# =============================================================================

class TestEndToEnd:

    def setup_method(self):
        self.scorer = ProfileScorer()

    def test_empty_profile_neutral(self):
        """Empty profile -> only category boost (no preference signals)."""
        item = _make_item(broad_category="bottoms")  # bottoms has no boost
        profile = _make_profile()
        adj = self.scorer.score_item(item, profile)
        assert adj == pytest.approx(0.0, abs=0.01)

    def test_perfect_match_high_positive(self):
        """Item matching all preferences -> high positive score."""
        item = _make_item(
            brand="Zara",
            style_tags=["casual", "modern"],
            formality="Casual",
            fit_type="regular",
            sleeve_type="short",
            neckline="crew",
            pattern="Solid",
            occasions=["Everyday"],
            price=25.0,
        )
        profile = _make_profile(
            preferred_brands=["Zara"],
            style_persona=["casual"],
            preferred_fits=["regular"],
            preferred_sleeves=["short"],
            preferred_necklines=["crew"],
            patterns_liked=["Solid"],
            occasions=["casual"],
        )
        adj = self.scorer.score_item(item, profile)
        # brand(0.25) + style(0.08) + formality(0.08) + fit(0.10) +
        # sleeve(0.08) + neckline(0.08) + pattern(0.10) + occasion(0.10) +
        # price(0.05) = 0.92 -> capped at 0.50
        assert adj == pytest.approx(0.50, abs=0.01)

    def test_coverage_kill_dominates(self):
        """Coverage kill should result in very negative score."""
        item = _make_item(
            brand="Zara",
            article_type="crop top",
            name="Crop Top",
        )
        profile = _make_profile(
            preferred_brands=["Zara"],
            no_crop=True,
        )
        adj = self.scorer.score_item(item, profile)
        assert adj < 0  # Coverage kill (-1.0) should dominate brand boost (+0.25)

    def test_max_negative_cap(self):
        """Total score can't go below -2.0."""
        item = _make_item(
            article_type="crop top",
            style_tags=["sheer"],
            neckline="plunging",
            name="Sheer Crop",
        )
        profile = _make_profile(
            no_crop=True,
            no_revealing=True,
            no_deep_necklines=True,
        )
        adj = self.scorer.score_item(item, profile)
        assert adj == pytest.approx(-2.0, abs=0.01)

    def test_score_items_batch(self):
        """Batch scoring modifies items in-place."""
        items = [
            _make_item(brand="Zara"),
            _make_item(brand="Nike"),
        ]
        profile = _make_profile(preferred_brands=["Zara"])

        self.scorer.score_items(items, profile, score_field="score")

        assert "profile_adjustment" in items[0]
        assert "profile_adjustment" in items[1]
        assert items[0]["profile_adjustment"] > items[1]["profile_adjustment"]

    def test_explain_item_has_all_keys(self):
        """explain_item returns all 10 dimensions + total."""
        item = _make_item()
        profile = _make_profile()
        breakdown = self.scorer.explain_item(item, profile)
        expected_keys = {
            "brand", "style", "formality", "attributes", "type",
            "pattern", "occasions", "color_avoid", "price", "coverage",
            "total",
        }
        assert set(breakdown.keys()) == expected_keys


# =============================================================================
# ProfileScorer: Works with OnboardingProfile objects
# =============================================================================

class TestWithOnboardingProfile:

    def setup_method(self):
        self.scorer = ProfileScorer()

    def test_works_with_pydantic_model(self):
        """ProfileScorer accepts OnboardingProfile (not just dicts)."""
        from recs.models import OnboardingProfile

        profile = OnboardingProfile(
            user_id="test-user",
            preferred_brands=["Zara", "Mango"],
            style_persona=["casual"],
            preferred_fits=["regular"],
            occasions=["casual"],
        )

        item = _make_item(
            brand="Zara",
            style_tags=["casual"],
            formality="Casual",
            fit_type="regular",
            occasions=["Everyday"],
            price=25.0,
        )

        adj = self.scorer.score_item(item, profile)
        assert adj > 0.3  # Should get significant positive boost

    def test_explain_with_pydantic(self):
        from recs.models import OnboardingProfile

        profile = OnboardingProfile(
            user_id="test-user",
            style_persona=["classic"],
        )

        item = _make_item(
            style_tags=["classic", "chic"],
            formality="Business Casual",
        )

        breakdown = self.scorer.explain_item(item, profile)
        assert breakdown["style"] > 0  # classic matches
        assert breakdown["formality"] > 0  # classic -> formal -> business casual match


# =============================================================================
# _attribute_applies helper
# =============================================================================

class TestAttributeApplies:

    def test_no_mapping_applies_globally(self):
        assert _attribute_applies("regular", "tops", {"regular"}, [], "fitId")

    def test_mapping_matches_category(self):
        mapping = [{"fitId": "fitted", "categories": ["tops"]}]
        assert _attribute_applies("fitted", "tops", {"fitted"}, mapping, "fitId")

    def test_mapping_rejects_wrong_category(self):
        mapping = [{"fitId": "fitted", "categories": ["tops"]}]
        assert not _attribute_applies("fitted", "bottoms", {"fitted"}, mapping, "fitId")

    def test_value_not_in_preferences(self):
        assert not _attribute_applies("oversized", "tops", {"regular"}, [], "fitId")

    def test_unmapped_value_applies_globally(self):
        """Preference value not in mapping -> applies to all categories."""
        mapping = [{"fitId": "fitted", "categories": ["tops"]}]
        # "regular" is in prefs but NOT in the mapping -> global
        assert _attribute_applies("regular", "bottoms", {"regular", "fitted"}, mapping, "fitId")


# =============================================================================
# Custom config tests
# =============================================================================

class TestCustomConfig:

    def test_custom_weights(self):
        """Custom config with higher brand weight."""
        config = ProfileScoringConfig(brand_preferred=0.40)
        scorer = ProfileScorer(config)
        item = _make_item(brand="Zara")
        profile = _make_profile(preferred_brands=["Zara"])
        breakdown = scorer.explain_item(item, profile)
        assert breakdown["brand"] == pytest.approx(0.40, abs=0.01)

    def test_custom_cap(self):
        config = ProfileScoringConfig(max_positive=0.30)
        scorer = ProfileScorer(config)
        item = _make_item(
            brand="Zara",
            style_tags=["casual"],
            formality="Casual",
            fit_type="regular",
        )
        profile = _make_profile(
            preferred_brands=["Zara"],
            style_persona=["casual"],
            preferred_fits=["regular"],
        )
        adj = scorer.score_item(item, profile)
        assert adj <= 0.30
