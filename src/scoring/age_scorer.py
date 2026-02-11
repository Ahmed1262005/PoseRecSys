"""
Age-Affinity Scoring Engine.

Scores items based on how well they match the user's age-group preferences
across seven dimensions:

1. Item type frequency   -- Are tank tops common for this age?
2. Style affinity        -- Is streetwear popular for this age?
3. Occasion affinity     -- Is "evenings out" common for this age?
4. Coverage tolerance    -- Bidirectional: boost crop tops for Gen Z, penalize for seniors
5. Fit preference        -- Oversized vs fitted vs relaxed by age x category
6. Pattern loudness      -- Bold patterns vs classic vs solid
7. Color boldness        -- Bold/bright colors vs neutrals/jewel tones by age

All scores are additive adjustments.  The total is capped at
+/- MAX_AGE_ADJUSTMENT to prevent age from dominating relevance.

IMPORTANT: These are soft priors.  A 50-year-old who actively clicks
crop tops (captured by session scoring) will override the age penalty.
User's explicit onboarding preferences (no_revealing, styles_to_avoid)
are hard filters handled upstream by FeasibilityFilter.
"""

from typing import List, Optional

from scoring.context import AgeGroup
from scoring.constants.age_item_frequency import ITEM_FREQUENCY
from scoring.constants.age_style_affinity import STYLE_AFFINITY, STYLE_TAG_MAP
from scoring.constants.age_occasion_affinity import OCCASION_AFFINITY, OCCASION_MAP
from scoring.constants.age_coverage_tolerance import COVERAGE_TOLERANCE
from scoring.constants.age_fit_preferences import FIT_PREFERENCES
from scoring.constants.age_color_pattern import (
    PATTERN_LOUDNESS, PATTERN_TO_LOUDNESS,
    COLOR_BOLDNESS, BOLD_COLOR_FAMILIES, NEUTRAL_COLOR_FAMILIES,
    JEWEL_COLOR_FAMILIES, PASTEL_COLOR_FAMILIES,
)
from scoring.item_utils import get_canonical_type, get_broad_category, detect_coverage_dimensions

# ── Weights ───────────────────────────────────────────────────────
MAX_AGE_ADJUSTMENT = 0.25
ITEM_FREQ_WEIGHT = 0.07         # item type is highly age-differentiating
STYLE_AFFINITY_WEIGHT = 0.06
OCCASION_AFFINITY_WEIGHT = 0.04 # reduced: "casual" is 1.0 for all ages, dilutes signal
COVERAGE_WEIGHT = 0.12          # bidirectional: strongest age signal (sleeveless, crop, mini)
FIT_AFFINITY_WEIGHT = 0.05
PATTERN_WEIGHT = 0.04
COLOR_WEIGHT = 0.04             # activates COLOR_BOLDNESS scoring


class AgeScorer:
    """Score items based on age-group affinity."""

    def score(
        self,
        item: dict,
        age_group: AgeGroup,
        coverage_prefs: Optional[List[str]] = None,
    ) -> float:
        """
        Compute age-affinity adjustment for a single item.

        Args:
            item: Product dict with article_type, style_tags, occasions, etc.
            age_group: User's age bracket.
            coverage_prefs: User's explicit coverage preferences from onboarding
                (e.g. ``["no_revealing", "no_crop"]``).  When present these
                override age-based coverage scoring -- if the user hasn't set
                any coverage prefs, the age default applies.

        Returns:
            Float adjustment in ``[-MAX_AGE_ADJUSTMENT, +MAX_AGE_ADJUSTMENT]``.
        """
        adj = 0.0
        adj += self._score_item_frequency(item, age_group)
        adj += self._score_style_affinity(item, age_group)
        adj += self._score_occasion_affinity(item, age_group)
        adj += self._score_coverage(item, age_group, coverage_prefs)
        adj += self._score_fit(item, age_group)
        adj += self._score_pattern(item, age_group)
        adj += self._score_color(item, age_group)
        return max(-MAX_AGE_ADJUSTMENT, min(MAX_AGE_ADJUSTMENT, adj))

    # ── 1. Item type frequency ────────────────────────────────────

    def _score_item_frequency(self, item: dict, age_group: AgeGroup) -> float:
        """Boost items common for this age, penalize uncommon ones."""
        canon = get_canonical_type(item)
        if not canon:
            return 0.0
        freq_table = ITEM_FREQUENCY.get(age_group, {})
        freq = freq_table.get(canon, 0.5)
        # 0-1 -> -1..+1 range, then scaled
        return (freq - 0.5) * 2 * ITEM_FREQ_WEIGHT

    # ── 2. Style affinity ─────────────────────────────────────────

    def _score_style_affinity(self, item: dict, age_group: AgeGroup) -> float:
        """Boost styles popular for this age group."""
        tags = item.get("style_tags") or []
        if isinstance(tags, str):
            tags = [tags]
        affinity = STYLE_AFFINITY.get(age_group, {})
        if not tags or not affinity:
            return 0.0

        scores = []
        for tag in tags:
            canonical = STYLE_TAG_MAP.get(tag.lower().strip())
            if canonical and canonical in affinity:
                scores.append(affinity[canonical])
        if not scores:
            return 0.0
        avg = sum(scores) / len(scores)
        return (avg - 0.5) * 2 * STYLE_AFFINITY_WEIGHT

    # ── 3. Occasion affinity ──────────────────────────────────────

    def _score_occasion_affinity(self, item: dict, age_group: AgeGroup) -> float:
        """Boost occasions popular for this age group."""
        occasions = item.get("occasions") or []
        if isinstance(occasions, str):
            occasions = [occasions]
        affinity = OCCASION_AFFINITY.get(age_group, {})
        if not occasions or not affinity:
            return 0.0

        scores = []
        for occ in occasions:
            canonical = OCCASION_MAP.get(occ.strip())
            if canonical and canonical in affinity:
                scores.append(affinity[canonical])
            else:
                # Try lowercase
                canonical = OCCASION_MAP.get(occ.lower().strip())
                if canonical and canonical in affinity:
                    scores.append(affinity[canonical])
        if not scores:
            return 0.0
        avg = sum(scores) / len(scores)
        return (avg - 0.5) * 2 * OCCASION_AFFINITY_WEIGHT

    # ── 4. Coverage tolerance (bidirectional) ─────────────────────

    def _score_coverage(
        self,
        item: dict,
        age_group: AgeGroup,
        user_prefs: Optional[List[str]],
    ) -> float:
        """
        Score items based on coverage appropriateness for the age group.

        Bidirectional: high tolerance (Gen Z + crop = 0.9) gives a boost,
        low tolerance (Senior + crop = 0.05) gives a penalty.

        User explicit prefs (``no_revealing``, etc.) are handled by
        FeasibilityFilter upstream; this is the soft age-based signal.
        """
        tolerance_table = COVERAGE_TOLERANCE.get(age_group, {})
        if not tolerance_table:
            return 0.0

        triggered = detect_coverage_dimensions(item)
        if not triggered:
            return 0.0

        total = 0.0
        for dim in triggered:
            tolerance = tolerance_table.get(dim, 0.5)
            # Bidirectional: tolerance 0.9 -> +0.064, tolerance 0.1 -> -0.064
            total += (tolerance - 0.5) * 2 * COVERAGE_WEIGHT

        return total

    # ── 5. Fit preference ─────────────────────────────────────────

    def _score_fit(self, item: dict, age_group: AgeGroup) -> float:
        """Boost fits common for this age x category."""
        broad = get_broad_category(item)
        fit = (item.get("fit_type") or item.get("fit") or "").lower().strip()
        if not broad or not fit:
            return 0.0

        fit_table = FIT_PREFERENCES.get(age_group, {}).get(broad, {})
        if not fit_table:
            return 0.0

        affinity = fit_table.get(fit, 0.5)
        return (affinity - 0.5) * 2 * FIT_AFFINITY_WEIGHT

    # ── 6. Pattern loudness ───────────────────────────────────────

    def _score_pattern(self, item: dict, age_group: AgeGroup) -> float:
        """Score pattern based on age-appropriate loudness."""
        pattern = (item.get("pattern") or "").lower().strip()
        if not pattern:
            return 0.0

        loudness_cat = PATTERN_TO_LOUDNESS.get(pattern)
        if not loudness_cat:
            return 0.0

        loudness_table = PATTERN_LOUDNESS.get(age_group, {})
        tolerance = loudness_table.get(loudness_cat, 0.5)
        return (tolerance - 0.5) * 2 * PATTERN_WEIGHT

    # ── 7. Color boldness ─────────────────────────────────────────

    def _score_color(self, item: dict, age_group: AgeGroup) -> float:
        """
        Score color based on age-group boldness preferences.

        Bold colors (neon, fuchsia, coral) boosted for Gen Z, penalized for older.
        Neutrals get a mild inverse boost (conservative ages prefer them).
        Jewel tones (emerald, burgundy, navy) boost established/senior.
        Pastels are universally mildly positive.
        """
        color_family = (
            item.get("color_family") or item.get("primary_color") or ""
        ).lower().strip()
        if not color_family:
            return 0.0

        age_boldness = COLOR_BOLDNESS.get(age_group, 0.5)

        if color_family in BOLD_COLOR_FAMILIES:
            # Bold color: boost for high-boldness ages, penalize for low
            return (age_boldness - 0.5) * 2 * COLOR_WEIGHT
        elif color_family in NEUTRAL_COLOR_FAMILIES:
            # Neutrals: mild boost for conservative ages, mild penalty for adventurous
            return (0.5 - age_boldness) * 0.5 * COLOR_WEIGHT
        elif color_family in PASTEL_COLOR_FAMILIES:
            # Pastels: small universal boost
            return COLOR_WEIGHT * 0.3
        elif color_family in JEWEL_COLOR_FAMILIES:
            # Jewel tones: boost for established/senior (sophisticated)
            if age_group in (AgeGroup.ESTABLISHED, AgeGroup.SENIOR, AgeGroup.MID_CAREER):
                return COLOR_WEIGHT * 0.5
            return 0.0

        return 0.0
