"""
Weather & Season Scoring Engine.

Scores items based on:
1. Product season tags vs current season
2. Material-weather appropriateness
3. Temperature-based item type affinity (with magnitude scaling)
4. Coverage/exposure penalty (sleeveless in cold, covered in hot)

Temperature scaling:
- Extreme cold (<0C) or extreme heat (>35C) doubles the penalty
- Moderate mismatch (10-25C) applies base penalty
- The further from comfortable (15-22C), the stronger the signal

Degrades gracefully:
- No weather data -> skip entirely (return 0)
- No season tags on product -> neutral
- No materials on product -> neutral
"""

from typing import Optional

from scoring.context import WeatherContext
from scoring.constants.weather_materials import (
    SEASON_MATERIALS,
    SEASON_PRODUCT_MAP,
    TEMP_ITEM_AFFINITY,
)
from scoring.item_utils import get_canonical_type, detect_coverage_dimensions

# ── Weights ───────────────────────────────────────────────────────
MAX_WEATHER_ADJUSTMENT = 0.25
SEASON_MATCH_WEIGHT = 0.08
MATERIAL_WEIGHT = 0.04
TEMP_ITEM_WEIGHT = 0.07
COVERAGE_PENALTY_WEIGHT = 0.08

# ── Exposure types that are bad in cold weather ────────────────────
_COLD_EXPOSED = frozenset({
    "crop", "strapless", "mini", "backless", "sleeveless",
})

# ── Coverage types that are bad in hot weather ─────────────────────
_HOT_COVERED = frozenset({
    "long_sleeve", "turtleneck", "high_neck",
})


class WeatherScorer:
    """Score items based on weather/season context."""

    def score(self, item: dict, weather: WeatherContext) -> float:
        """
        Compute weather-based adjustment for a single item.

        Returns float in ``[-MAX_WEATHER_ADJUSTMENT, +MAX_WEATHER_ADJUSTMENT]``.
        """
        adj = 0.0
        adj += self._score_season(item, weather)
        adj += self._score_materials(item, weather)
        adj += self._score_temperature(item, weather)
        adj += self._score_coverage(item, weather)
        return max(-MAX_WEATHER_ADJUSTMENT, min(MAX_WEATHER_ADJUSTMENT, adj))

    # ── Temperature intensity multiplier ──────────────────────────

    @staticmethod
    def _temp_intensity(weather: WeatherContext) -> float:
        """
        Compute a multiplier (1.0 - 2.0) based on temperature extremity.

        Comfortable zone: 15-22C -> 1.0x
        Moderate: 5-15C or 22-30C -> 1.0-1.5x
        Extreme: <0C or >35C -> 2.0x
        """
        t = weather.temperature_c
        if t is None:
            return 1.0

        if weather.is_cold:
            # 10C -> 1.0, 0C -> 1.5, -10C -> 2.0
            if t <= -10:
                return 2.0
            elif t <= 0:
                return 1.5 + (0 - t) * 0.05  # 0C=1.5, -10C=2.0
            else:
                return 1.0 + max(0, (10 - t)) * 0.05  # 10C=1.0, 0C=1.5
        elif weather.is_hot:
            # 25C -> 1.0, 32C -> 1.5, 40C -> 2.0
            if t >= 40:
                return 2.0
            elif t >= 32:
                return 1.5 + (t - 32) * 0.0625  # 32C=1.5, 40C=2.0
            else:
                return 1.0 + max(0, (t - 25)) * 0.0714  # 25C=1.0, 32C=1.5
        return 1.0

    # ── 1. Season tag match ───────────────────────────────────────

    def _score_season(self, item: dict, weather: WeatherContext) -> float:
        """Boost items tagged for current season, penalize out-of-season."""
        item_seasons = item.get("seasons") or []
        if isinstance(item_seasons, str):
            item_seasons = [item_seasons]
        if not item_seasons:
            return 0.0  # No season data = neutral

        # All-season items are neutral
        if len(item_seasons) >= 4:
            return 0.0

        current = SEASON_PRODUCT_MAP.get(weather.season, "")
        intensity = self._temp_intensity(weather)

        if current in item_seasons:
            return SEASON_MATCH_WEIGHT  # In-season boost
        else:
            # Full out-of-season penalty, scaled by temperature intensity
            return -SEASON_MATCH_WEIGHT * intensity

    # ── 2. Material appropriateness ───────────────────────────────

    def _score_materials(self, item: dict, weather: WeatherContext) -> float:
        """Boost weather-appropriate materials, penalize bad ones."""
        materials = item.get("materials") or item.get("apparent_fabric") or []
        if isinstance(materials, str):
            materials = [materials]
        materials = [m.lower().strip() for m in materials if m]
        if not materials:
            return 0.0

        season_mats = SEASON_MATERIALS.get(weather.season, {})
        good = season_mats.get("good", frozenset())
        bad = season_mats.get("bad", frozenset())

        total = 0.0
        for mat in materials:
            if mat in good:
                total += MATERIAL_WEIGHT
            elif mat in bad:
                total -= MATERIAL_WEIGHT
        # Average across materials
        return total / max(len(materials), 1)

    # ── 3. Temperature x item type ────────────────────────────────

    def _score_temperature(self, item: dict, weather: WeatherContext) -> float:
        """Boost/penalize items based on current temperature with magnitude scaling."""
        canon = get_canonical_type(item)
        if not canon:
            return 0.0

        intensity = self._temp_intensity(weather)

        # Determine temperature bucket
        if weather.is_rainy:
            temp_rules = TEMP_ITEM_AFFINITY.get("rainy", {})
            score = self._check_rules(canon, temp_rules, intensity)
            if score != 0.0:
                return score
            # Fall through to temperature-based rules

        if weather.is_hot:
            temp_rules = TEMP_ITEM_AFFINITY.get("hot", {})
        elif weather.is_cold:
            temp_rules = TEMP_ITEM_AFFINITY.get("cold", {})
        else:
            temp_rules = TEMP_ITEM_AFFINITY.get("mild", {})

        return self._check_rules(canon, temp_rules, intensity)

    @staticmethod
    def _check_rules(canon: str, rules: dict, intensity: float = 1.0) -> float:
        """Check if canonical type is boosted or penalized by rules."""
        if canon in rules.get("boost", frozenset()):
            return TEMP_ITEM_WEIGHT * intensity
        elif canon in rules.get("penalize", frozenset()):
            return -TEMP_ITEM_WEIGHT * intensity
        return 0.0

    # ── 4. Coverage / exposure penalty ────────────────────────────

    def _score_coverage(self, item: dict, weather: WeatherContext) -> float:
        """
        Penalize exposed items in cold weather, and heavily covered in hot.

        Uses detect_coverage_dimensions() to check for sleeveless, crop,
        mini, strapless, etc. — dimensions the season/temp scorer misses
        because they're about *how much skin is exposed*, not article type.
        """
        dims = detect_coverage_dimensions(item)
        if not dims:
            return 0.0

        dims_set = set(dims)
        intensity = self._temp_intensity(weather)

        if weather.is_cold:
            # Penalize exposed items in cold
            exposed = dims_set & _COLD_EXPOSED
            if exposed:
                # More exposed dimensions = worse. Cap at 2 hits.
                n = min(len(exposed), 2)
                return -COVERAGE_PENALTY_WEIGHT * n * intensity

        elif weather.is_hot:
            # Penalize heavily covered items in hot weather
            covered = dims_set & _HOT_COVERED
            if covered:
                n = min(len(covered), 2)
                return -COVERAGE_PENALTY_WEIGHT * n * intensity

        return 0.0
