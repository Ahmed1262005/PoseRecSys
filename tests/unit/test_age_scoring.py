"""
Unit tests for the Age-Affinity Scoring Engine.

Tests cover:
1. Weight constants verification
2. Each of the 7 scoring dimensions in isolation
3. Bidirectional coverage scoring (boost + penalty)
4. Color scoring (new dimension)
5. Cap enforcement (output within [-0.25, +0.25])
6. Graceful degradation with missing/empty data
7. Combined extremes (Gen Z ideal vs Senior ideal)
8. Regression: sensible items for each age group

Run with: PYTHONPATH=src python -m pytest tests/unit/test_age_scoring.py -v
"""

import pytest
from scoring.age_scorer import (
    AgeScorer,
    MAX_AGE_ADJUSTMENT,
    ITEM_FREQ_WEIGHT,
    STYLE_AFFINITY_WEIGHT,
    OCCASION_AFFINITY_WEIGHT,
    COVERAGE_WEIGHT,
    FIT_AFFINITY_WEIGHT,
    PATTERN_WEIGHT,
    COLOR_WEIGHT,
)
from scoring.context import AgeGroup


# =============================================================================
# Helpers
# =============================================================================

def _item(**overrides) -> dict:
    """Create a product dict with sensible defaults."""
    base = {
        "article_type": "tshirt",
        "broad_category": "tops",
        "style_tags": ["casual"],
        "occasions": ["Everyday"],
        "pattern": "Solid",
        "fit_type": "regular",
        "neckline": "crew",
        "sleeve_type": "short",
        "length": "standard",
        "color_family": "",
        "primary_color": "",
        "seasons": ["Spring", "Summer"],
        "materials": ["cotton"],
        "name": "Test Product",
    }
    base.update(overrides)
    return base


@pytest.fixture
def scorer():
    return AgeScorer()


# =============================================================================
# 1. Weight Constants Verification
# =============================================================================

class TestWeightConstants:
    """Verify weight constants match the Phase 1 overhaul."""

    def test_max_age_adjustment(self):
        assert MAX_AGE_ADJUSTMENT == 0.25

    def test_item_freq_weight(self):
        assert ITEM_FREQ_WEIGHT == 0.07

    def test_style_affinity_weight(self):
        assert STYLE_AFFINITY_WEIGHT == 0.06

    def test_occasion_affinity_weight(self):
        assert OCCASION_AFFINITY_WEIGHT == 0.04

    def test_coverage_weight(self):
        assert COVERAGE_WEIGHT == 0.12

    def test_fit_affinity_weight(self):
        assert FIT_AFFINITY_WEIGHT == 0.05

    def test_pattern_weight(self):
        assert PATTERN_WEIGHT == 0.04

    def test_color_weight(self):
        assert COLOR_WEIGHT == 0.04


# =============================================================================
# 2. Item Type Frequency
# =============================================================================

class TestItemFrequency:
    """Test _score_item_frequency dimension."""

    def test_crop_top_gen_z_boost(self, scorer):
        """Crop top is very common (0.9) for Gen Z -> positive."""
        item = _item(article_type="crop_top")
        score = scorer._score_item_frequency(item, AgeGroup.GEN_Z)
        assert score > 0

    def test_crop_top_senior_penalty(self, scorer):
        """Crop top is rare (0.05) for Senior -> negative."""
        item = _item(article_type="crop_top")
        score = scorer._score_item_frequency(item, AgeGroup.SENIOR)
        assert score < 0

    def test_tshirt_universal_neutral(self, scorer):
        """T-shirt is common (1.0) across all ages -> positive."""
        item = _item(article_type="tshirt")
        for age in AgeGroup:
            score = scorer._score_item_frequency(item, age)
            assert score >= 0

    def test_unknown_article_type_zero(self, scorer):
        """Unknown article type should return 0."""
        item = _item(article_type="space_suit")
        score = scorer._score_item_frequency(item, AgeGroup.GEN_Z)
        # Unknown type gets default 0.5 -> 0.0 adjustment
        assert score == 0.0

    def test_empty_article_type_zero(self, scorer):
        """Empty article type should return 0."""
        item = _item(article_type="")
        score = scorer._score_item_frequency(item, AgeGroup.GEN_Z)
        assert score == 0.0


# =============================================================================
# 3. Style Affinity
# =============================================================================

class TestStyleAffinity:
    """Test _score_style_affinity dimension."""

    def test_streetwear_gen_z_boost(self, scorer):
        """Streetwear is 1.0 for Gen Z -> positive."""
        item = _item(style_tags=["streetwear"])
        score = scorer._score_style_affinity(item, AgeGroup.GEN_Z)
        assert score > 0

    def test_streetwear_senior_penalty(self, scorer):
        """Streetwear is 0.2 for Senior -> negative."""
        item = _item(style_tags=["streetwear"])
        score = scorer._score_style_affinity(item, AgeGroup.SENIOR)
        assert score < 0

    def test_classic_established_boost(self, scorer):
        """Classic is 1.0 for Established -> positive."""
        item = _item(style_tags=["classic"])
        score = scorer._score_style_affinity(item, AgeGroup.ESTABLISHED)
        assert score > 0

    def test_multiple_tags_averaged(self, scorer):
        """Multiple tags should be averaged."""
        item = _item(style_tags=["streetwear", "classic"])
        score = scorer._score_style_affinity(item, AgeGroup.GEN_Z)
        # streetwear=1.0, classic=0.4 -> avg=0.7 -> (0.7-0.5)*2*0.06 = +0.024
        assert score > 0

    def test_unknown_tags_ignored(self, scorer):
        """Unknown style tags should be skipped."""
        item = _item(style_tags=["alien_fashion"])
        score = scorer._score_style_affinity(item, AgeGroup.GEN_Z)
        assert score == 0.0

    def test_empty_tags_zero(self, scorer):
        """No tags -> zero."""
        item = _item(style_tags=[])
        score = scorer._score_style_affinity(item, AgeGroup.GEN_Z)
        assert score == 0.0

    def test_string_tag_handled(self, scorer):
        """String tag (not list) should be handled."""
        item = _item(style_tags="streetwear")
        score = scorer._score_style_affinity(item, AgeGroup.GEN_Z)
        assert score > 0

    def test_tag_aliases_work(self, scorer):
        """Alias tags (e.g., 'grunge' -> 'streetwear') should resolve."""
        item = _item(style_tags=["grunge"])
        score = scorer._score_style_affinity(item, AgeGroup.GEN_Z)
        assert score > 0  # grunge -> streetwear -> 1.0 for Gen Z


# =============================================================================
# 4. Occasion Affinity
# =============================================================================

class TestOccasionAffinity:
    """Test _score_occasion_affinity dimension."""

    def test_evenings_gen_z_boost(self, scorer):
        """Evenings is 1.0 for Gen Z -> positive."""
        item = _item(occasions=["Date Night"])
        score = scorer._score_occasion_affinity(item, AgeGroup.GEN_Z)
        assert score > 0

    def test_office_senior_penalty(self, scorer):
        """Office is 0.4 for Senior -> negative."""
        item = _item(occasions=["Office"])
        score = scorer._score_occasion_affinity(item, AgeGroup.SENIOR)
        assert score < 0

    def test_casual_universal_boost(self, scorer):
        """Casual is 1.0 for all ages -> always positive."""
        item = _item(occasions=["Casual"])
        for age in AgeGroup:
            score = scorer._score_occasion_affinity(item, age)
            assert score > 0

    def test_empty_occasions_zero(self, scorer):
        """No occasions -> zero."""
        item = _item(occasions=[])
        score = scorer._score_occasion_affinity(item, AgeGroup.GEN_Z)
        assert score == 0.0


# =============================================================================
# 5. Coverage Tolerance (Bidirectional)
# =============================================================================

class TestCoverageBidirectional:
    """Test the new bidirectional _score_coverage dimension."""

    def test_crop_gen_z_positive_boost(self, scorer):
        """Gen Z + crop_top (tolerance 0.9) -> POSITIVE boost, not penalty."""
        item = _item(article_type="crop_top")
        score = scorer._score_coverage(item, AgeGroup.GEN_Z, None)
        # tolerance=0.9 -> (0.9-0.5)*2*0.12 = +0.096
        assert score > 0
        assert abs(score - 0.096) < 0.01

    def test_crop_senior_negative_penalty(self, scorer):
        """Senior + crop_top (tolerance 0.05) -> negative penalty."""
        item = _item(article_type="crop_top")
        score = scorer._score_coverage(item, AgeGroup.SENIOR, None)
        # tolerance=0.05 -> (0.05-0.5)*2*0.12 = -0.108
        assert score < 0
        assert abs(score - (-0.108)) < 0.01

    def test_bodycon_gen_z_boost(self, scorer):
        """Gen Z + bodycon (tolerance 0.9) -> positive."""
        item = _item(article_type="bodycon_dress")
        score = scorer._score_coverage(item, AgeGroup.GEN_Z, None)
        assert score > 0

    def test_bodycon_established_penalty(self, scorer):
        """Established + bodycon (tolerance 0.2) -> negative."""
        item = _item(article_type="bodycon_dress")
        score = scorer._score_coverage(item, AgeGroup.ESTABLISHED, None)
        assert score < 0

    def test_plunging_neckline_triggers_coverage(self, scorer):
        """Plunging neckline should trigger deep_necklines coverage."""
        item = _item(neckline="plunging")
        score_genz = scorer._score_coverage(item, AgeGroup.GEN_Z, None)
        score_senior = scorer._score_coverage(item, AgeGroup.SENIOR, None)
        assert score_genz > score_senior

    def test_multiple_coverage_dims_stack(self, scorer):
        """Item with crop + strapless should stack both dimensions."""
        item = _item(article_type="crop_top", neckline="strapless")
        score = scorer._score_coverage(item, AgeGroup.GEN_Z, None)
        # crop=0.9, strapless=0.8 -> two boosts
        assert score > 0.05  # More than a single dimension

    def test_non_revealing_item_zero(self, scorer):
        """Regular t-shirt with crew neck -> no coverage dims -> 0."""
        item = _item(article_type="tshirt", neckline="crew")
        score = scorer._score_coverage(item, AgeGroup.GEN_Z, None)
        assert score == 0.0

    def test_mid_career_moderate(self, scorer):
        """Mid-career coverage scores should be smaller in magnitude."""
        item = _item(article_type="crop_top")
        score_genz = scorer._score_coverage(item, AgeGroup.GEN_Z, None)
        score_mid = scorer._score_coverage(item, AgeGroup.MID_CAREER, None)
        score_senior = scorer._score_coverage(item, AgeGroup.SENIOR, None)
        # Gen Z > Mid-Career > Senior
        assert score_genz > score_mid > score_senior

    def test_sleeveless_detected_from_sleeve_type(self, scorer):
        """Sleeveless sleeve_type should trigger 'sleeveless' coverage."""
        item = _item(sleeve_type="sleeveless")
        score_genz = scorer._score_coverage(item, AgeGroup.GEN_Z, None)
        score_senior = scorer._score_coverage(item, AgeGroup.SENIOR, None)
        assert score_genz > 0  # Gen Z tolerates sleeveless
        assert score_senior < 0  # Senior penalized

    def test_shorts_trigger_mini_coverage(self, scorer):
        """Shorts article_type should trigger 'mini' coverage dimension."""
        item = _item(article_type="shorts", broad_category="bottoms")
        score_established = scorer._score_coverage(item, AgeGroup.ESTABLISHED, None)
        assert score_established < 0  # Established penalized for shorts

    def test_micro_in_name_triggers_mini(self, scorer):
        """'Micro' in product name should trigger 'mini' coverage."""
        item = _item(article_type="shorts", name="Denim Micro Short")
        score_established = scorer._score_coverage(item, AgeGroup.ESTABLISHED, None)
        assert score_established < 0


# =============================================================================
# 6. Fit Preference
# =============================================================================

class TestFitPreference:
    """Test _score_fit dimension."""

    def test_oversized_gen_z_tops_boost(self, scorer):
        """Oversized tops are 0.9 for Gen Z -> positive."""
        item = _item(article_type="tshirt", fit_type="oversized")
        score = scorer._score_fit(item, AgeGroup.GEN_Z)
        assert score > 0

    def test_skinny_senior_bottoms_penalty(self, scorer):
        """Skinny bottoms are 0.1 for Senior -> negative."""
        item = _item(article_type="jeans", broad_category="bottoms", fit_type="skinny")
        score = scorer._score_fit(item, AgeGroup.SENIOR)
        assert score < 0

    def test_regular_fit_universal_neutral(self, scorer):
        """Regular fit should be neutral-to-positive for all ages."""
        item = _item(fit_type="regular")
        for age in AgeGroup:
            score = scorer._score_fit(item, age)
            # Regular is typically 0.5-0.9 -> should be >= 0
            assert score >= 0

    def test_no_fit_data_zero(self, scorer):
        """Missing fit type -> 0."""
        item = _item(fit_type="")
        score = scorer._score_fit(item, AgeGroup.GEN_Z)
        assert score == 0.0


# =============================================================================
# 7. Pattern Loudness
# =============================================================================

class TestPatternLoudness:
    """Test _score_pattern dimension."""

    def test_animal_print_gen_z_boost(self, scorer):
        """Bold pattern for Gen Z -> positive."""
        item = _item(pattern="animal_print")
        score = scorer._score_pattern(item, AgeGroup.GEN_Z)
        assert score > 0

    def test_animal_print_senior_penalty(self, scorer):
        """Bold pattern for Senior -> negative."""
        item = _item(pattern="animal_print")
        score = scorer._score_pattern(item, AgeGroup.SENIOR)
        assert score < 0

    def test_classic_pattern_established_boost(self, scorer):
        """Classic pattern (floral) for Established -> positive."""
        item = _item(pattern="floral")
        score = scorer._score_pattern(item, AgeGroup.ESTABLISHED)
        assert score > 0

    def test_solid_universal_positive(self, scorer):
        """Solid is 0.7-0.9 across all ages -> positive or neutral."""
        item = _item(pattern="solid")
        for age in AgeGroup:
            score = scorer._score_pattern(item, age)
            assert score >= 0

    def test_unknown_pattern_zero(self, scorer):
        """Unknown pattern -> 0."""
        item = _item(pattern="holographic_sparkle")
        score = scorer._score_pattern(item, AgeGroup.GEN_Z)
        assert score == 0.0


# =============================================================================
# 8. Color Boldness (NEW)
# =============================================================================

class TestColorBoldness:
    """Test the new _score_color dimension."""

    def test_bold_color_gen_z_boost(self, scorer):
        """Bold color (neon) for Gen Z (boldness=0.9) -> positive."""
        item = _item(color_family="neon")
        score = scorer._score_color(item, AgeGroup.GEN_Z)
        # (0.9-0.5)*2*0.04 = +0.032
        assert score > 0
        assert abs(score - 0.032) < 0.005

    def test_bold_color_established_penalty(self, scorer):
        """Bold color (fuchsia) for Established (boldness=0.4) -> negative."""
        item = _item(color_family="fuchsia")
        score = scorer._score_color(item, AgeGroup.ESTABLISHED)
        # (0.4-0.5)*2*0.04 = -0.008
        assert score < 0

    def test_neutral_color_conservative_boost(self, scorer):
        """Neutral color for Established (low boldness) -> mild boost."""
        item = _item(color_family="black")
        score = scorer._score_color(item, AgeGroup.ESTABLISHED)
        # (0.5-0.4)*0.5*0.04 = +0.002
        assert score > 0

    def test_neutral_color_gen_z_mild_penalty(self, scorer):
        """Neutral color for Gen Z (high boldness) -> mild penalty."""
        item = _item(color_family="beige")
        score = scorer._score_color(item, AgeGroup.GEN_Z)
        # (0.5-0.9)*0.5*0.04 = -0.008
        assert score < 0

    def test_jewel_tone_established_boost(self, scorer):
        """Jewel tones boost for established/senior."""
        item = _item(color_family="emerald")
        score = scorer._score_color(item, AgeGroup.ESTABLISHED)
        # 0.04 * 0.5 = +0.02
        assert score > 0
        assert abs(score - 0.02) < 0.005

    def test_jewel_tone_gen_z_neutral(self, scorer):
        """Jewel tones are neutral for Gen Z."""
        item = _item(color_family="burgundy")
        score = scorer._score_color(item, AgeGroup.GEN_Z)
        assert score == 0.0

    def test_pastel_universal_boost(self, scorer):
        """Pastels give a small universal boost."""
        item = _item(color_family="lavender")
        for age in AgeGroup:
            score = scorer._score_color(item, age)
            assert score > 0  # Always positive

    def test_no_color_data_zero(self, scorer):
        """Missing color -> 0."""
        item = _item(color_family="", primary_color="")
        score = scorer._score_color(item, AgeGroup.GEN_Z)
        assert score == 0.0

    def test_primary_color_fallback(self, scorer):
        """Should fallback to primary_color when color_family is empty."""
        item = _item(color_family="", primary_color="neon")
        score = scorer._score_color(item, AgeGroup.GEN_Z)
        assert score > 0

    def test_unknown_color_zero(self, scorer):
        """Unknown color family -> 0."""
        item = _item(color_family="rainbow")
        score = scorer._score_color(item, AgeGroup.GEN_Z)
        assert score == 0.0


# =============================================================================
# 9. Cap Enforcement
# =============================================================================

class TestCapEnforcement:
    """Test that total score is capped at +/- MAX_AGE_ADJUSTMENT."""

    def test_positive_cap(self, scorer):
        """Max-boosted item should be capped at +0.25."""
        # Gen Z ideal: crop top, streetwear, casual, bold pattern, bold color, oversized
        item = _item(
            article_type="crop_top",
            style_tags=["streetwear"],
            occasions=["Date Night"],
            pattern="animal_print",
            fit_type="oversized",
            color_family="neon",
            neckline="off shoulder",
        )
        score = scorer.score(item, AgeGroup.GEN_Z)
        assert score <= MAX_AGE_ADJUSTMENT
        assert score == MAX_AGE_ADJUSTMENT  # Should hit the cap

    def test_negative_cap(self, scorer):
        """Max-penalized item should be capped at -0.25."""
        # Senior worst: crop top, streetwear, evenings, bold pattern, bold color
        item = _item(
            article_type="crop_top",
            style_tags=["streetwear"],
            occasions=["Date Night"],
            pattern="animal_print",
            fit_type="oversized",
            color_family="neon",
            neckline="off shoulder",
        )
        score = scorer.score(item, AgeGroup.SENIOR)
        assert score >= -MAX_AGE_ADJUSTMENT
        assert score == -MAX_AGE_ADJUSTMENT  # Should hit the negative cap

    def test_output_always_within_bounds(self, scorer):
        """Score should always be within [-0.25, +0.25]."""
        items = [
            _item(article_type="crop_top", style_tags=["streetwear"], pattern="animal_print"),
            _item(article_type="blazer", style_tags=["classic"], pattern="solid"),
            _item(article_type="sweater", style_tags=["minimal"], pattern="floral"),
            _item(),  # defaults
        ]
        for item in items:
            for age in AgeGroup:
                score = scorer.score(item, age)
                assert -MAX_AGE_ADJUSTMENT <= score <= MAX_AGE_ADJUSTMENT


# =============================================================================
# 10. Graceful Degradation
# =============================================================================

class TestGracefulDegradation:
    """Test that scorer handles missing/empty data without crashing."""

    def test_empty_item(self, scorer):
        """Completely empty item should return 0."""
        score = scorer.score({}, AgeGroup.GEN_Z)
        assert score == 0.0

    def test_none_fields(self, scorer):
        """None values for all fields should return 0."""
        item = {
            "article_type": None,
            "style_tags": None,
            "occasions": None,
            "pattern": None,
            "fit_type": None,
            "color_family": None,
            "primary_color": None,
            "neckline": None,
            "name": None,
        }
        score = scorer.score(item, AgeGroup.GEN_Z)
        assert score == 0.0

    def test_partial_data(self, scorer):
        """Item with only some fields should score without errors."""
        item = {"article_type": "blazer", "style_tags": ["classic"]}
        score = scorer.score(item, AgeGroup.ESTABLISHED)
        assert isinstance(score, float)
        assert score > 0  # Blazer + classic for Established should boost


# =============================================================================
# 11. Combined Extremes - Visible Feed Impact
# =============================================================================

class TestCombinedExtremes:
    """Test that age scoring creates visible differences between age groups."""

    def test_gen_z_ideal_vs_senior_huge_swing(self, scorer):
        """The same Gen Z item should score very differently for Gen Z vs Senior."""
        gen_z_item = _item(
            article_type="crop_top",
            style_tags=["streetwear", "trendy"],
            occasions=["Date Night"],
            pattern="animal_print",
            fit_type="cropped",
            color_family="neon",
            sleeve_type="sleeveless",
        )
        gen_z_score = scorer.score(gen_z_item, AgeGroup.GEN_Z)
        senior_score = scorer.score(gen_z_item, AgeGroup.SENIOR)

        # Should be a massive swing (both hit caps)
        swing = gen_z_score - senior_score
        assert swing >= 0.45, f"Swing {swing:.3f} too small, expected >= 0.45"

    def test_senior_ideal_vs_gen_z(self, scorer):
        """A conservative item should score higher for Senior than Gen Z."""
        senior_item = _item(
            article_type="blazer",
            style_tags=["classic", "elegant"],
            occasions=["Office"],
            pattern="solid",
            fit_type="regular",
            color_family="navy",
        )
        gen_z_score = scorer.score(senior_item, AgeGroup.GEN_Z)
        senior_score = scorer.score(senior_item, AgeGroup.SENIOR)

        # Senior should like it more (or dislike it less) than Gen Z
        # But since "classic" is still somewhat positive for Gen Z (0.4),
        # the main differentiator is office occasion and navy color
        assert senior_score > gen_z_score

    def test_established_prefers_midi_over_mini(self, scorer):
        """Established user should prefer midi dress over mini dress."""
        midi = _item(article_type="midi_dress", style_tags=["classic"])
        mini = _item(article_type="mini_dress", style_tags=["trendy"])

        midi_score = scorer.score(midi, AgeGroup.ESTABLISHED)
        mini_score = scorer.score(mini, AgeGroup.ESTABLISHED)

        assert midi_score > mini_score

    def test_gen_z_prefers_mini_over_midi(self, scorer):
        """Gen Z should prefer mini dress over midi dress."""
        midi = _item(article_type="midi_dress", style_tags=["classic"])
        mini = _item(article_type="mini_dress", style_tags=["trendy"])

        midi_score = scorer.score(midi, AgeGroup.GEN_Z)
        mini_score = scorer.score(mini, AgeGroup.GEN_Z)

        assert mini_score > midi_score


# =============================================================================
# 12. Regression: Sensible Items for Each Age
# =============================================================================

class TestRegressionSensibleItems:
    """Verify that age-appropriate items score positively for their target age."""

    def test_gen_z_crop_streetwear_positive(self, scorer):
        """Crop top + streetwear should be positive for Gen Z."""
        item = _item(article_type="crop_top", style_tags=["streetwear"])
        score = scorer.score(item, AgeGroup.GEN_Z)
        assert score > 0

    def test_young_adult_blazer_minimal_positive(self, scorer):
        """Blazer + minimal should be positive for Young Adult."""
        item = _item(
            article_type="blazer", broad_category="outerwear",
            style_tags=["minimal"], occasions=["Office"],
        )
        score = scorer.score(item, AgeGroup.YOUNG_ADULT)
        assert score > 0

    def test_mid_career_wrap_dress_classic_positive(self, scorer):
        """Wrap dress + classic should be positive for Mid-Career."""
        item = _item(
            article_type="wrap_dress", broad_category="dresses",
            style_tags=["classic"], occasions=["Office"],
            fit_type="regular",
        )
        score = scorer.score(item, AgeGroup.MID_CAREER)
        assert score > 0

    def test_established_sweater_classic_solid_positive(self, scorer):
        """Sweater + classic + solid should be positive for Established."""
        item = _item(
            article_type="sweater", style_tags=["classic"],
            pattern="solid", fit_type="relaxed",
            color_family="navy",
        )
        score = scorer.score(item, AgeGroup.ESTABLISHED)
        assert score > 0

    def test_senior_cardigan_relaxed_positive(self, scorer):
        """Cardigan + relaxed should be positive for Senior."""
        item = _item(
            article_type="cardigan", style_tags=["classic"],
            pattern="solid", fit_type="relaxed",
        )
        score = scorer.score(item, AgeGroup.SENIOR)
        assert score > 0
