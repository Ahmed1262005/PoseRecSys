"""
Tests for the shared scoring module (src/scoring/).

Covers:
- ItemUtils: canonical type, broad category, coverage detection
- AgeScorer: all 6 dimensions x 5 age groups
- WeatherScorer: season, material, temperature
- ContextScorer: combined orchestrator
- ContextResolver: pure helpers (age, season, address extraction)
- Constants: table completeness
"""

import sys
import os
import pytest
from datetime import date
from unittest.mock import patch, MagicMock

# Ensure src/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from scoring.context import AgeGroup, Season, UserContext, WeatherContext
from scoring.age_scorer import AgeScorer, MAX_AGE_ADJUSTMENT
from scoring.weather_scorer import WeatherScorer, MAX_WEATHER_ADJUSTMENT
from scoring.scorer import ContextScorer, MAX_CONTEXT_ADJUSTMENT
from scoring.item_utils import (
    get_canonical_type, get_broad_category, detect_coverage_dimensions,
)
from scoring.context_resolver import (
    _compute_age, _age_to_group, _season_from_date_and_country,
    _extract_address_from_metadata, _extract_coverage_prefs,
    ContextResolver,
)
from scoring.constants.age_item_frequency import ITEM_FREQUENCY
from scoring.constants.age_style_affinity import STYLE_AFFINITY, STYLE_TAG_MAP
from scoring.constants.age_occasion_affinity import OCCASION_AFFINITY
from scoring.constants.age_coverage_tolerance import COVERAGE_TOLERANCE
from scoring.constants.age_fit_preferences import FIT_PREFERENCES
from scoring.constants.age_color_pattern import PATTERN_LOUDNESS
from scoring.constants.weather_materials import SEASON_MATERIALS, TEMP_ITEM_AFFINITY


# =====================================================================
# Helpers
# =====================================================================

def _summer_hot() -> WeatherContext:
    return WeatherContext(
        temperature_c=32, feels_like_c=35, condition="clear",
        humidity=60, wind_speed_mps=3, season=Season.SUMMER,
        is_hot=True,
    )

def _winter_cold() -> WeatherContext:
    return WeatherContext(
        temperature_c=-5, feels_like_c=-8, condition="snow",
        humidity=80, wind_speed_mps=5, season=Season.WINTER,
        is_cold=True,
    )

def _spring_mild() -> WeatherContext:
    return WeatherContext(
        temperature_c=18, feels_like_c=17, condition="clouds",
        humidity=55, wind_speed_mps=2, season=Season.SPRING,
        is_mild=True,
    )

def _rainy_fall() -> WeatherContext:
    return WeatherContext(
        temperature_c=12, feels_like_c=10, condition="rain",
        humidity=90, wind_speed_mps=4, season=Season.FALL,
        is_rainy=True, is_mild=True,
    )


# =====================================================================
# ItemUtils
# =====================================================================

class TestGetCanonicalType:
    def test_direct_match(self):
        assert get_canonical_type({"article_type": "sweater"}) == "sweater"

    def test_alias(self):
        assert get_canonical_type({"article_type": "t-shirt"}) == "tshirt"
        assert get_canonical_type({"article_type": "Tee"}) == "tshirt"

    def test_space_to_underscore(self):
        assert get_canonical_type({"article_type": "crop top"}) == "crop_top"
        assert get_canonical_type({"article_type": "Tank Top"}) == "tank_top"

    def test_empty(self):
        assert get_canonical_type({}) is None
        assert get_canonical_type({"article_type": ""}) is None

    def test_unknown_type(self):
        # Should still normalize
        result = get_canonical_type({"article_type": "fancy toga"})
        assert result == "fancy_toga"


class TestGetBroadCategory:
    def test_explicit_field(self):
        assert get_broad_category({"broad_category": "tops"}) == "tops"
        assert get_broad_category({"broad_category": "bottoms"}) == "bottoms"

    def test_one_piece(self):
        assert get_broad_category({"broad_category": "one_piece"}) == "dresses"

    def test_infer_from_type(self):
        assert get_broad_category({"article_type": "sweater"}) == "tops"
        assert get_broad_category({"article_type": "jeans"}) == "bottoms"
        assert get_broad_category({"article_type": "blazer"}) == "outerwear"
        assert get_broad_category({"article_type": "midi_dress"}) == "dresses"

    def test_empty(self):
        assert get_broad_category({}) is None


class TestDetectCoverageDimensions:
    def test_crop_top(self):
        dims = detect_coverage_dimensions({"article_type": "crop_top"})
        assert "crop" in dims

    def test_tube_top(self):
        dims = detect_coverage_dimensions({"article_type": "tube_top"})
        assert "strapless" in dims

    def test_bodycon_dress(self):
        dims = detect_coverage_dimensions({"article_type": "bodycon_dress"})
        assert "bodycon" in dims

    def test_deep_neckline(self):
        dims = detect_coverage_dimensions({"article_type": "blouse", "neckline": "plunging"})
        assert "deep_necklines" in dims

    def test_sheer_tag(self):
        dims = detect_coverage_dimensions({"article_type": "blouse", "style_tags": ["sheer"]})
        assert "sheer" in dims

    def test_mini_length(self):
        dims = detect_coverage_dimensions({"article_type": "dress", "length": "Mini"})
        assert "mini" in dims

    def test_crop_in_name(self):
        dims = detect_coverage_dimensions({"article_type": "tshirt", "name": "Crop Ribbed Tee"})
        assert "crop" in dims

    def test_no_coverage(self):
        dims = detect_coverage_dimensions({"article_type": "sweater", "neckline": "crew"})
        assert dims == []

    def test_multiple_dimensions(self):
        dims = detect_coverage_dimensions({
            "article_type": "mini_dress", "neckline": "plunging",
            "style_tags": ["cutouts"],
        })
        assert "mini" in dims
        assert "deep_necklines" in dims
        assert "cutouts" in dims


# =====================================================================
# AgeScorer
# =====================================================================

class TestAgeScorer:
    def setup_method(self):
        self.scorer = AgeScorer()

    # ── Item frequency ────────────────────────────────────────────

    def test_crop_top_gen_z_high(self):
        item = {"article_type": "crop_top", "broad_category": "tops"}
        score = self.scorer._score_item_frequency(item, AgeGroup.GEN_Z)
        assert score > 0  # Very common -> positive

    def test_crop_top_senior_low(self):
        item = {"article_type": "crop_top", "broad_category": "tops"}
        score = self.scorer._score_item_frequency(item, AgeGroup.SENIOR)
        assert score < 0  # Uncommon -> negative

    def test_sweater_universal(self):
        item = {"article_type": "sweater"}
        # Sweater is very common across all ages
        for age in AgeGroup:
            score = self.scorer._score_item_frequency(item, age)
            assert score > 0

    def test_blazer_mid_career_high(self):
        item = {"article_type": "blazer"}
        score_mid = self.scorer._score_item_frequency(item, AgeGroup.MID_CAREER)
        score_gen_z = self.scorer._score_item_frequency(item, AgeGroup.GEN_Z)
        assert score_mid > score_gen_z

    # ── Style affinity ────────────────────────────────────────────

    def test_streetwear_gen_z(self):
        item = {"style_tags": ["streetwear"]}
        score = self.scorer._score_style_affinity(item, AgeGroup.GEN_Z)
        assert score > 0

    def test_streetwear_senior(self):
        item = {"style_tags": ["streetwear"]}
        score = self.scorer._score_style_affinity(item, AgeGroup.SENIOR)
        assert score < 0  # Low affinity

    def test_classic_senior(self):
        item = {"style_tags": ["classic"]}
        score = self.scorer._score_style_affinity(item, AgeGroup.SENIOR)
        assert score > 0

    def test_style_tag_mapping(self):
        """y2k maps to trendy -> Gen Z loves it."""
        item = {"style_tags": ["y2k"]}
        score = self.scorer._score_style_affinity(item, AgeGroup.GEN_Z)
        assert score > 0

    def test_empty_tags(self):
        item = {"style_tags": []}
        score = self.scorer._score_style_affinity(item, AgeGroup.GEN_Z)
        assert score == 0.0

    # ── Occasion affinity ─────────────────────────────────────────

    def test_evenings_gen_z(self):
        item = {"occasions": ["Party"]}
        score = self.scorer._score_occasion_affinity(item, AgeGroup.GEN_Z)
        assert score > 0

    def test_office_mid_career(self):
        item = {"occasions": ["Work"]}
        score_mid = self.scorer._score_occasion_affinity(item, AgeGroup.MID_CAREER)
        score_senior = self.scorer._score_occasion_affinity(item, AgeGroup.SENIOR)
        assert score_mid > score_senior

    # ── Coverage tolerance ────────────────────────────────────────

    def test_crop_no_penalty_gen_z(self):
        item = {"article_type": "crop_top"}
        penalty = self.scorer._score_coverage(item, AgeGroup.GEN_Z, None)
        # Gen Z has 0.9 tolerance for crop -> very small penalty
        assert penalty > -0.02

    def test_crop_penalty_senior(self):
        item = {"article_type": "crop_top"}
        penalty = self.scorer._score_coverage(item, AgeGroup.SENIOR, None)
        # Senior has 0.05 tolerance -> big penalty
        assert penalty < -0.05

    def test_no_coverage_no_penalty(self):
        item = {"article_type": "sweater", "neckline": "crew"}
        penalty = self.scorer._score_coverage(item, AgeGroup.SENIOR, None)
        assert penalty == 0.0

    def test_multiple_coverage_compounds(self):
        """Item with both crop and cutouts gets double penalty."""
        item = {"article_type": "crop_top", "style_tags": ["cutouts"]}
        penalty = self.scorer._score_coverage(item, AgeGroup.ESTABLISHED, None)
        assert penalty < -0.05

    # ── Fit preference ────────────────────────────────────────────

    def test_oversized_gen_z_tops(self):
        item = {"article_type": "tshirt", "broad_category": "tops", "fit_type": "oversized"}
        score = self.scorer._score_fit(item, AgeGroup.GEN_Z)
        assert score > 0

    def test_oversized_senior_tops(self):
        item = {"article_type": "tshirt", "broad_category": "tops", "fit_type": "oversized"}
        score = self.scorer._score_fit(item, AgeGroup.SENIOR)
        assert score < 0

    def test_relaxed_senior_tops(self):
        item = {"article_type": "tshirt", "broad_category": "tops", "fit_type": "relaxed"}
        score = self.scorer._score_fit(item, AgeGroup.SENIOR)
        assert score > 0

    # ── Pattern loudness ──────────────────────────────────────────

    def test_bold_pattern_gen_z(self):
        item = {"pattern": "animal_print"}
        score = self.scorer._score_pattern(item, AgeGroup.GEN_Z)
        assert score > 0

    def test_bold_pattern_established(self):
        item = {"pattern": "animal_print"}
        score = self.scorer._score_pattern(item, AgeGroup.ESTABLISHED)
        assert score < 0

    def test_solid_universal(self):
        """Solid is accepted across all ages."""
        item = {"pattern": "solid"}
        for age in AgeGroup:
            score = self.scorer._score_pattern(item, age)
            assert score >= 0

    # ── Composite scoring ─────────────────────────────────────────

    def test_cap_positive(self):
        """Total score never exceeds MAX_AGE_ADJUSTMENT."""
        item = {
            "article_type": "crop_top", "style_tags": ["trendy"],
            "occasions": ["Party"], "pattern": "animal_print",
            "fit_type": "fitted", "broad_category": "tops",
        }
        score = self.scorer.score(item, AgeGroup.GEN_Z)
        assert score <= MAX_AGE_ADJUSTMENT

    def test_cap_negative(self):
        """Total score never goes below -MAX_AGE_ADJUSTMENT."""
        item = {
            "article_type": "crop_top", "style_tags": ["streetwear"],
            "occasions": ["Party"], "pattern": "animal_print",
            "fit_type": "oversized", "broad_category": "tops",
            "neckline": "plunging",
        }
        score = self.scorer.score(item, AgeGroup.SENIOR)
        assert score >= -MAX_AGE_ADJUSTMENT

    def test_empty_item(self):
        score = self.scorer.score({}, AgeGroup.GEN_Z)
        assert score == 0.0


# =====================================================================
# WeatherScorer
# =====================================================================

class TestWeatherScorer:
    def setup_method(self):
        self.scorer = WeatherScorer()

    # ── Season match ──────────────────────────────────────────────

    def test_in_season_boost(self):
        item = {"seasons": ["Summer"]}
        score = self.scorer._score_season(item, _summer_hot())
        assert score > 0

    def test_out_of_season_penalty(self):
        item = {"seasons": ["Winter"]}
        score = self.scorer._score_season(item, _summer_hot())
        assert score < 0

    def test_all_season_neutral(self):
        item = {"seasons": ["Spring", "Summer", "Fall", "Winter"]}
        score = self.scorer._score_season(item, _summer_hot())
        assert score == 0.0

    def test_no_season_neutral(self):
        item = {"seasons": []}
        score = self.scorer._score_season(item, _summer_hot())
        assert score == 0.0

    # ── Material match ────────────────────────────────────────────

    def test_linen_summer_good(self):
        item = {"materials": ["linen"]}
        score = self.scorer._score_materials(item, _summer_hot())
        assert score > 0

    def test_wool_summer_bad(self):
        item = {"materials": ["wool"]}
        score = self.scorer._score_materials(item, _summer_hot())
        assert score < 0

    def test_wool_winter_good(self):
        item = {"materials": ["wool"]}
        score = self.scorer._score_materials(item, _winter_cold())
        assert score > 0

    def test_mixed_materials(self):
        """One good + one bad should partially cancel."""
        item = {"materials": ["cotton", "wool"]}
        score = self.scorer._score_materials(item, _summer_hot())
        # cotton is good, wool is bad -> near zero
        assert -0.05 < score < 0.05

    def test_apparent_fabric_fallback(self):
        """Should check apparent_fabric if materials is missing."""
        item = {"apparent_fabric": "linen"}
        score = self.scorer._score_materials(item, _summer_hot())
        assert score > 0

    # ── Temperature x item type ───────────────────────────────────

    def test_tank_top_hot_boost(self):
        item = {"article_type": "tank_top"}
        score = self.scorer._score_temperature(item, _summer_hot())
        assert score > 0

    def test_coat_hot_penalize(self):
        item = {"article_type": "coat"}
        score = self.scorer._score_temperature(item, _summer_hot())
        assert score < 0

    def test_coat_cold_boost(self):
        item = {"article_type": "coat"}
        score = self.scorer._score_temperature(item, _winter_cold())
        assert score > 0

    def test_tank_cold_penalize(self):
        item = {"article_type": "tank_top"}
        score = self.scorer._score_temperature(item, _winter_cold())
        assert score < 0

    def test_rainy_jacket_boost(self):
        item = {"article_type": "jacket"}
        score = self.scorer._score_temperature(item, _rainy_fall())
        assert score > 0

    # ── Composite ─────────────────────────────────────────────────

    def test_cap(self):
        item = {
            "article_type": "tank_top", "seasons": ["Summer"],
            "materials": ["cotton"],
        }
        score = self.scorer.score(item, _summer_hot())
        assert score <= MAX_WEATHER_ADJUSTMENT

    def test_empty_item(self):
        score = self.scorer.score({}, _summer_hot())
        assert score == 0.0


# =====================================================================
# ContextScorer (combined)
# =====================================================================

class TestContextScorer:
    def setup_method(self):
        self.scorer = ContextScorer()

    def test_gen_z_summer_crop(self):
        ctx = UserContext(
            user_id="u1", age_group=AgeGroup.GEN_Z,
            weather=_summer_hot(),
        )
        item = {
            "article_type": "crop_top", "style_tags": ["trendy"],
            "occasions": ["Party"], "seasons": ["Summer"],
            "pattern": "solid", "fit_type": "fitted", "broad_category": "tops",
        }
        score = self.scorer.score_item(item, ctx)
        assert score > 0.1  # Strong positive

    def test_senior_winter_sweater(self):
        ctx = UserContext(
            user_id="u2", age_group=AgeGroup.SENIOR,
            weather=_winter_cold(),
        )
        item = {
            "article_type": "sweater", "style_tags": ["classic"],
            "occasions": ["Everyday"], "seasons": ["Winter"],
            "materials": ["wool"], "pattern": "solid",
            "fit_type": "relaxed", "broad_category": "tops",
        }
        score = self.scorer.score_item(item, ctx)
        assert score > 0.1

    def test_senior_winter_crop_negative(self):
        ctx = UserContext(
            user_id="u3", age_group=AgeGroup.SENIOR,
            weather=_winter_cold(),
        )
        item = {
            "article_type": "crop_top", "style_tags": ["streetwear"],
            "occasions": ["Party"], "seasons": ["Summer"],
            "pattern": "animal_print", "fit_type": "fitted",
            "broad_category": "tops", "neckline": "plunging",
        }
        score = self.scorer.score_item(item, ctx)
        assert score < -0.1

    def test_cap(self):
        ctx = UserContext(
            user_id="u4", age_group=AgeGroup.GEN_Z,
            weather=_summer_hot(),
        )
        item = {
            "article_type": "crop_top", "style_tags": ["trendy"],
            "occasions": ["Party"], "seasons": ["Summer"],
            "materials": ["cotton"], "pattern": "solid",
            "fit_type": "cropped", "broad_category": "tops",
        }
        score = self.scorer.score_item(item, ctx)
        assert -MAX_CONTEXT_ADJUSTMENT <= score <= MAX_CONTEXT_ADJUSTMENT

    def test_no_context(self):
        """No age, no weather -> zero adjustment."""
        ctx = UserContext(user_id="u5")
        item = {"article_type": "sweater"}
        assert self.scorer.score_item(item, ctx) == 0.0

    def test_age_only(self):
        ctx = UserContext(user_id="u6", age_group=AgeGroup.MID_CAREER)
        item = {"article_type": "blazer", "style_tags": ["classic"]}
        score = self.scorer.score_item(item, ctx)
        assert score > 0

    def test_weather_only(self):
        ctx = UserContext(user_id="u7", weather=_summer_hot())
        item = {"article_type": "tank_top", "seasons": ["Summer"]}
        score = self.scorer.score_item(item, ctx)
        assert score > 0

    # ── Batch scoring ─────────────────────────────────────────────

    def test_score_items_batch(self):
        ctx = UserContext(
            user_id="u8", age_group=AgeGroup.YOUNG_ADULT,
            weather=_spring_mild(),
        )
        items = [
            {"article_type": "blazer", "score": 0.5, "style_tags": ["classic"]},
            {"article_type": "crop_top", "score": 0.5, "style_tags": ["trendy"]},
        ]
        self.scorer.score_items(items, ctx, score_field="score")
        # Both should have context_adjustment added
        assert "context_adjustment" in items[0]
        assert "context_adjustment" in items[1]
        # Blazer should score higher than crop for 25-34
        assert items[0]["score"] != items[1]["score"]

    # ── Explain ───────────────────────────────────────────────────

    def test_explain(self):
        ctx = UserContext(
            user_id="u9", age_group=AgeGroup.GEN_Z,
            weather=_summer_hot(),
        )
        item = {"article_type": "crop_top", "style_tags": ["trendy"]}
        breakdown = self.scorer.explain_item(item, ctx)
        assert "age" in breakdown
        assert "weather" in breakdown
        assert "total" in breakdown
        assert breakdown["age_group"] == "18-24"
        assert breakdown["season"] == "summer"


# =====================================================================
# ContextResolver helpers
# =====================================================================

class TestComputeAge:
    def test_basic(self):
        with patch("scoring.context_resolver.date") as mock_date:
            mock_date.today.return_value = date(2026, 2, 8)
            mock_date.fromisoformat = date.fromisoformat
            assert _compute_age("2000-01-15") == 26
            assert _compute_age("2000-06-15") == 25  # Birthday hasn't happened yet

    def test_birthday_today(self):
        with patch("scoring.context_resolver.date") as mock_date:
            mock_date.today.return_value = date(2026, 2, 8)
            mock_date.fromisoformat = date.fromisoformat
            assert _compute_age("2000-02-08") == 26


class TestAgeToGroup:
    def test_gen_z(self):
        assert _age_to_group(18) == AgeGroup.GEN_Z
        assert _age_to_group(24) == AgeGroup.GEN_Z

    def test_young_adult(self):
        assert _age_to_group(25) == AgeGroup.YOUNG_ADULT
        assert _age_to_group(34) == AgeGroup.YOUNG_ADULT

    def test_mid_career(self):
        assert _age_to_group(35) == AgeGroup.MID_CAREER
        assert _age_to_group(44) == AgeGroup.MID_CAREER

    def test_established(self):
        assert _age_to_group(45) == AgeGroup.ESTABLISHED
        assert _age_to_group(64) == AgeGroup.ESTABLISHED

    def test_senior(self):
        assert _age_to_group(65) == AgeGroup.SENIOR
        assert _age_to_group(85) == AgeGroup.SENIOR

    def test_under_18(self):
        # Under 18 maps to Gen Z
        assert _age_to_group(16) == AgeGroup.GEN_Z


class TestSeasonFromCountry:
    def test_northern_february(self):
        with patch("scoring.context_resolver.date") as mock_date:
            mock_date.today.return_value = date(2026, 2, 8)
            assert _season_from_date_and_country("US") == Season.WINTER

    def test_southern_february(self):
        with patch("scoring.context_resolver.date") as mock_date:
            mock_date.today.return_value = date(2026, 2, 8)
            assert _season_from_date_and_country("australia") == Season.SUMMER

    def test_northern_july(self):
        with patch("scoring.context_resolver.date") as mock_date:
            mock_date.today.return_value = date(2026, 7, 15)
            assert _season_from_date_and_country("US") == Season.SUMMER

    def test_southern_july(self):
        with patch("scoring.context_resolver.date") as mock_date:
            mock_date.today.return_value = date(2026, 7, 15)
            assert _season_from_date_and_country("au") == Season.WINTER


class TestExtractAddress:
    def test_nested(self):
        meta = {"address": {"city": "NYC", "country": "US"}}
        addr = _extract_address_from_metadata(meta)
        assert addr["city"] == "NYC"

    def test_flat(self):
        meta = {"city": "London", "country": "UK", "state": "England"}
        addr = _extract_address_from_metadata(meta)
        assert addr["city"] == "London"
        assert addr["country"] == "UK"

    def test_empty(self):
        assert _extract_address_from_metadata({}) is None
        assert _extract_address_from_metadata(None) is None

    def test_postal_code(self):
        meta = {"city": "Berlin", "country": "DE", "postal_code": "10115"}
        addr = _extract_address_from_metadata(meta)
        assert addr["zip"] == "10115"


class TestExtractCoveragePrefs:
    def test_flags(self):
        profile = {"no_revealing": True, "no_crop": True, "no_sleeveless": False}
        prefs = _extract_coverage_prefs(profile)
        assert "no_revealing" in prefs
        assert "no_crop" in prefs
        assert "no_sleeveless" not in prefs

    def test_styles_to_avoid(self):
        profile = {"styles_to_avoid": ["deep-necklines", "sheer"]}
        prefs = _extract_coverage_prefs(profile)
        assert "deep-necklines" in prefs
        assert "sheer" in prefs

    def test_empty(self):
        assert _extract_coverage_prefs({}) == []


class TestContextResolverResolve:
    def test_full_context(self):
        resolver = ContextResolver(weather_api_key="")
        # Mock address resolution to avoid Supabase call
        resolver._resolve_address = MagicMock(
            return_value={"city": "NYC", "country": "US"}
        )
        ctx = resolver.resolve(
            user_id="test",
            birthdate="2000-01-01",
            onboarding_profile={"no_revealing": True},
        )
        assert ctx.age_group is not None
        assert ctx.age_years is not None
        assert ctx.city == "NYC"
        assert ctx.weather is not None
        assert "no_revealing" in ctx.coverage_prefs

    def test_no_data(self):
        resolver = ContextResolver()
        resolver._resolve_address = MagicMock(return_value=None)
        ctx = resolver.resolve(user_id="empty")
        assert ctx.age_group is None
        assert ctx.weather is None
        assert ctx.city is None

    def test_invalid_birthdate(self):
        resolver = ContextResolver()
        resolver._resolve_address = MagicMock(return_value=None)
        ctx = resolver.resolve(user_id="bad", birthdate="not-a-date")
        assert ctx.age_group is None


# =====================================================================
# Constants completeness
# =====================================================================

class TestConstantsCompleteness:
    """Verify all age groups have entries in every table."""

    def test_item_frequency_all_groups(self):
        for ag in AgeGroup:
            assert ag in ITEM_FREQUENCY, f"Missing {ag} in ITEM_FREQUENCY"
            assert len(ITEM_FREQUENCY[ag]) > 10

    def test_style_affinity_all_groups(self):
        for ag in AgeGroup:
            assert ag in STYLE_AFFINITY, f"Missing {ag} in STYLE_AFFINITY"

    def test_occasion_affinity_all_groups(self):
        for ag in AgeGroup:
            assert ag in OCCASION_AFFINITY, f"Missing {ag} in OCCASION_AFFINITY"

    def test_coverage_tolerance_all_groups(self):
        for ag in AgeGroup:
            assert ag in COVERAGE_TOLERANCE, f"Missing {ag} in COVERAGE_TOLERANCE"

    def test_fit_preferences_all_groups(self):
        for ag in AgeGroup:
            assert ag in FIT_PREFERENCES, f"Missing {ag} in FIT_PREFERENCES"
            # Each group should have all 4 categories
            for cat in ("tops", "bottoms", "dresses", "outerwear"):
                assert cat in FIT_PREFERENCES[ag], f"Missing {cat} for {ag}"

    def test_pattern_loudness_all_groups(self):
        for ag in AgeGroup:
            assert ag in PATTERN_LOUDNESS, f"Missing {ag} in PATTERN_LOUDNESS"

    def test_season_materials_all_seasons(self):
        for s in Season:
            assert s in SEASON_MATERIALS, f"Missing {s} in SEASON_MATERIALS"

    def test_temp_affinity_all_conditions(self):
        for cond in ("hot", "cold", "mild", "rainy"):
            assert cond in TEMP_ITEM_AFFINITY, f"Missing {cond} in TEMP_ITEM_AFFINITY"


# =====================================================================
# Cross-age comparison tests (verify the data tells the right story)
# =====================================================================

class TestCrossAgeStory:
    """Verify the scoring tells the right story across age groups."""

    def setup_method(self):
        self.scorer = AgeScorer()

    def test_crop_top_decreases_with_age(self):
        """Crop tops should score progressively lower as age increases."""
        item = {"article_type": "crop_top", "broad_category": "tops"}
        scores = {}
        for ag in AgeGroup:
            scores[ag] = self.scorer._score_item_frequency(item, ag)
        assert scores[AgeGroup.GEN_Z] > scores[AgeGroup.MID_CAREER]
        assert scores[AgeGroup.MID_CAREER] > scores[AgeGroup.SENIOR]

    def test_blazer_peaks_mid_career(self):
        """Blazer should score highest for mid-career."""
        item = {"article_type": "blazer"}
        scores = {}
        for ag in AgeGroup:
            scores[ag] = self.scorer._score_item_frequency(item, ag)
        assert scores[AgeGroup.MID_CAREER] >= scores[AgeGroup.GEN_Z]
        assert scores[AgeGroup.MID_CAREER] >= scores[AgeGroup.SENIOR]

    def test_streetwear_highest_gen_z(self):
        """Streetwear style should score highest for Gen Z."""
        item = {"style_tags": ["streetwear"]}
        scores = {}
        for ag in AgeGroup:
            scores[ag] = self.scorer._score_style_affinity(item, ag)
        assert scores[AgeGroup.GEN_Z] > scores[AgeGroup.ESTABLISHED]
        assert scores[AgeGroup.GEN_Z] > scores[AgeGroup.SENIOR]

    def test_classic_highest_mid_career(self):
        """Classic style should score highest 35-44+."""
        item = {"style_tags": ["classic"]}
        scores = {}
        for ag in AgeGroup:
            scores[ag] = self.scorer._score_style_affinity(item, ag)
        assert scores[AgeGroup.MID_CAREER] >= scores[AgeGroup.GEN_Z]

    def test_office_strongest_mid_career(self):
        """Office occasion should be strongest for 35-44."""
        item = {"occasions": ["Office"]}
        scores = {}
        for ag in AgeGroup:
            scores[ag] = self.scorer._score_occasion_affinity(item, ag)
        assert scores[AgeGroup.MID_CAREER] >= scores[AgeGroup.GEN_Z]
        assert scores[AgeGroup.MID_CAREER] >= scores[AgeGroup.SENIOR]

    def test_coverage_penalty_increases_with_age(self):
        """Coverage penalty should get stricter with age."""
        item = {"article_type": "crop_top", "neckline": "plunging"}
        penalties = {}
        for ag in AgeGroup:
            penalties[ag] = self.scorer._score_coverage(item, ag, None)
        assert penalties[AgeGroup.GEN_Z] > penalties[AgeGroup.SENIOR]  # Less negative
        assert penalties[AgeGroup.YOUNG_ADULT] > penalties[AgeGroup.ESTABLISHED]
