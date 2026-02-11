"""
ProfileScorer -- attribute-driven profile preference scoring.

Scores items against the user's onboarding profile by directly matching
Gemini attributes (style_tags, formality, fit, sleeve, length, neckline,
occasions, pattern, etc.) to stated preferences.  No persona abstraction
— the structured attributes ARE the persona.

Replaces:
  - Demo's PERSONA_STYLE_MAP + compute_profile_adjustment()
  - Production's _apply_soft_scoring() in candidate_selection.py

Design principles:
  1. Works with plain dicts (like ContextScorer) so both feed pipeline
     and search reranker can use it.
  2. Accepts an OnboardingProfile (or simplified dict) for preferences.
  3. Every onboarding dimension scores independently and additively.
  4. Brand cluster adjacency for discovery.
  5. Coverage violations are hard-kills (-1.0).

Usage::

    from scoring.profile_scorer import ProfileScorer

    scorer = ProfileScorer()

    # From production pipeline (Candidate objects)
    adj = scorer.score_item(candidate.to_scoring_dict(), profile)
    candidate.preference_score = adj

    # From search reranker (plain dicts)
    adj = scorer.score_item(item_dict, profile)
    item_dict["profile_adjustment"] = adj
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from scoring.item_utils import (
    detect_coverage_dimensions,
    get_broad_category,
    get_canonical_type,
)
from scoring.constants.brand_data import (
    BRAND_TO_CLUSTER,
    derive_price_range,
    get_user_brand_clusters,
)


# ── Occasion expansion (frontend names -> DB values) ──────────────
OCCASION_MAP: Dict[str, List[str]] = {
    "office": ["Office", "Work"],
    "casual": ["Everyday", "Casual", "Weekend", "Lounging"],
    "evening": ["Date Night", "Party", "Evening Event"],
    "smart-casual": ["Brunch", "Vacation"],
    "beach": ["Vacation", "Beach"],
    "active": ["Workout"],
    "everyday": ["Everyday"],
    "date night": ["Date Night"],
    "party": ["Party"],
    "brunch": ["Brunch"],
    "vacation": ["Vacation"],
    "workout": ["Workout"],
}


def _expand_occasions(user_occasions: List[str]) -> Set[str]:
    """Expand frontend occasion names to DB values (case-insensitive set)."""
    expanded: Set[str] = set()
    for occ in user_occasions:
        key = occ.lower().strip()
        if key in OCCASION_MAP:
            expanded.update(v.lower() for v in OCCASION_MAP[key])
        else:
            expanded.add(key)
    return expanded


# ── Style persona synonym normalization ────────────────────────────
# The frontend uses user-friendly names that don't always match the
# exact style_tag values in product_attributes.  These are pure
# synonyms (not editorial expansion) — just mapping to the DB form.
STYLE_ALIASES: Dict[str, str] = {
    "boho": "bohemian",
    "minimal": "minimalist",
    "glam": "glamorous",
    "business casual": "classic",      # closest DB style_tag
    "business-casual": "classic",
    "athleisure": "sporty",            # DB has "sporty", not "athleisure"
}


# ── Formality inference from style persona ────────────────────────
# Maps user-facing style_persona values to expected formality levels
# in the product_attributes.formality field.
_FORMAL_LEANING_PERSONAS: Set[str] = {
    "classic", "elegant", "preppy", "business casual", "business-casual",
}
_CASUAL_LEANING_PERSONAS: Set[str] = {
    "casual", "sporty", "athleisure", "streetwear", "boho", "bohemian",
}


def _infer_formality_preference(style_personas: List[str]) -> Optional[str]:
    """
    Infer a formality direction from the user's style personas.

    Returns "formal" if any persona leans formal, "casual" if all lean
    casual, or None if mixed/unknown.
    """
    personas_lower = {p.lower() for p in style_personas}
    has_formal = bool(personas_lower & _FORMAL_LEANING_PERSONAS)
    has_casual = bool(personas_lower & _CASUAL_LEANING_PERSONAS)

    if has_formal and not has_casual:
        return "formal"
    if has_casual and not has_formal:
        return "casual"
    return None  # mixed or unknown


# ── Coverage dimension -> exclusion flag mapping ──────────────────
_COVERAGE_KILLS: Dict[str, Set[str]] = {
    "no_crop": {"crop"},
    "no_revealing": {"mini", "bodycon", "sheer", "cutouts", "high_slit", "open_back"},
    "no_deep_necklines": {"deep_necklines", "strapless"},
    "no_sleeveless": {"sleeveless"},
}


# ── Scoring configuration ────────────────────────────────────────

@dataclass
class ProfileScoringConfig:
    """
    Weights and caps for profile-based preference scoring.

    Each weight represents the additive adjustment for a matching
    dimension.  All are applied independently.
    """

    # Brand scoring
    brand_preferred: float = 0.25
    brand_cluster_adjacent: float = 0.10
    brand_unrelated_penalty: float = -0.05

    # Brand openness multiplier applied to brand_preferred
    brand_openness_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "stick_to_favorites": 2.0,
        "stick-to-favorites": 2.0,
        "mix": 1.2,
        "mix-favorites-new": 1.0,
        "discover_new": 0.7,
        "discover-new": 0.7,
    })

    # Style tags: direct match of style_persona against item's style_tags
    style_match: float = 0.15       # per hit (strong — must dominate feed)
    style_match_cap: float = 0.35   # max from all style hits
    style_mismatch_penalty: float = -0.08  # items with ZERO style overlap

    # Formality alignment
    formality_match: float = 0.10

    # Attribute preferences (V3 category-aware matching)
    fit_match: float = 0.10
    sleeve_match: float = 0.08
    length_match: float = 0.08
    neckline_match: float = 0.08
    rise_match: float = 0.08

    # Type preferences (article type matching)
    type_match: float = 0.15

    # Pattern scoring
    pattern_match: float = 0.10
    pattern_avoid_penalty: float = -0.15

    # Occasion scoring
    occasion_match: float = 0.10    # first hit
    occasion_multi_bonus: float = 0.05  # additional for 2+ hits

    # Color scoring
    color_avoid_penalty: float = -0.15

    # Price scoring (derived from brand avg prices)
    price_in_range: float = 0.05
    price_over_penalty: float = -0.05
    price_way_over_penalty: float = -0.10
    price_way_over_threshold: float = 1.5  # multiplier on max_price

    # Coverage violation (hard kill)
    coverage_kill_penalty: float = -1.0

    # Category-level soft boosts / penalties
    # Keys: broad_category keywords (e.g. "top", "bottom", "dress", "outer")
    # Values: additive adjustment (positive = boost, negative = penalty)
    category_boosts: Dict[str, float] = field(default_factory=lambda: {
        "top": 0.06,          # slight boost for tops
        "dress": -0.03,       # mild demotion for dresses
        "outer": -0.03,       # mild demotion for outerwear
    })

    # Caps
    max_positive: float = 0.50
    max_negative: float = -2.0


# ── Main scorer ───────────────────────────────────────────────────

class ProfileScorer:
    """
    Scores items against the user's onboarding profile using direct
    attribute matching.

    Stateless — safe to share across threads / reuse across requests.

    Accepts any dict-like profile with the following optional keys
    (matching OnboardingProfile field names):

        preferred_brands, brand_openness,
        style_persona, style_directions,
        preferred_fits, fit_category_mapping,
        preferred_sleeves, sleeve_category_mapping,
        preferred_lengths, length_category_mapping,
        preferred_lengths_dresses, length_dresses_category_mapping,
        preferred_necklines, preferred_rises,
        top_types, bottom_types, dress_types, outerwear_types,
        patterns_liked, patterns_avoided,
        occasions,
        colors_to_avoid,
        no_crop, no_revealing, no_deep_necklines, no_sleeveless, no_tanks,
        styles_to_avoid,
        global_min_price, global_max_price,
    """

    def __init__(self, config: Optional[ProfileScoringConfig] = None) -> None:
        self.config = config or ProfileScoringConfig()

    # ── Public API ────────────────────────────────────────────────

    def score_item(self, item: dict, profile: Any) -> float:
        """
        Compute profile-preference adjustment for one item.

        Args:
            item: Dict with Gemini attributes (from Candidate.to_scoring_dict()
                  or search result dict).
            profile: OnboardingProfile instance or dict with profile fields.

        Returns:
            Float adjustment, clamped to [max_negative, max_positive].
        """
        cfg = self.config
        adj = 0.0

        # ── Precompute profile fields once ────────────────────────
        pf = _ProfileFields(profile)

        # ── 1. Brand scoring ──────────────────────────────────────
        adj += self._score_brand(item, pf, cfg)

        # ── 2. Style tag matching ─────────────────────────────────
        adj += self._score_style(item, pf, cfg)

        # ── 3. Formality alignment ────────────────────────────────
        adj += self._score_formality(item, pf, cfg)

        # ── 4. Attribute preferences (fit, sleeve, length, neckline, rise)
        adj += self._score_attributes(item, pf, cfg)

        # ── 5. Type preferences ───────────────────────────────────
        adj += self._score_type(item, pf, cfg)

        # ── 6. Pattern scoring ────────────────────────────────────
        adj += self._score_pattern(item, pf, cfg)

        # ── 7. Occasion scoring ───────────────────────────────────
        adj += self._score_occasions(item, pf, cfg)

        # ── 8. Color avoid scoring ────────────────────────────────
        adj += self._score_color_avoid(item, pf, cfg)

        # ── 9. Price range scoring ────────────────────────────────
        adj += self._score_price(item, pf, cfg)

        # ── 10. Coverage hard-kills ───────────────────────────────
        adj += self._score_coverage(item, pf, cfg)

        # ── 11. Category-level soft boost ─────────────────────────
        adj += self._score_category_boost(item, cfg)

        return max(cfg.max_negative, min(cfg.max_positive, adj))

    def score_items(
        self,
        items: list,
        profile: Any,
        score_field: str = "preference_score",
    ) -> list:
        """
        Batch-score items in-place.

        Adds ``profile_adjustment`` key to each item dict and updates
        ``score_field``.
        """
        for item in items:
            adj = self.score_item(item, profile)
            item["profile_adjustment"] = adj
            current = item.get(score_field, 0)
            if isinstance(current, (int, float)):
                item[score_field] = current + adj
        return items

    def explain_item(self, item: dict, profile: Any) -> dict:
        """Return a detailed breakdown for debugging / admin UI."""
        cfg = self.config
        pf = _ProfileFields(profile)

        breakdown = {
            "brand": round(self._score_brand(item, pf, cfg), 4),
            "style": round(self._score_style(item, pf, cfg), 4),
            "formality": round(self._score_formality(item, pf, cfg), 4),
            "attributes": round(self._score_attributes(item, pf, cfg), 4),
            "type": round(self._score_type(item, pf, cfg), 4),
            "pattern": round(self._score_pattern(item, pf, cfg), 4),
            "occasions": round(self._score_occasions(item, pf, cfg), 4),
            "color_avoid": round(self._score_color_avoid(item, pf, cfg), 4),
            "price": round(self._score_price(item, pf, cfg), 4),
            "coverage": round(self._score_coverage(item, pf, cfg), 4),
        }
        raw = sum(breakdown.values())
        breakdown["total"] = round(
            max(cfg.max_negative, min(cfg.max_positive, raw)), 4
        )
        return breakdown

    # ── Scoring dimensions (private) ──────────────────────────────

    def _score_brand(self, item: dict, pf: "_ProfileFields", cfg: ProfileScoringConfig) -> float:
        brand_lower = (item.get("brand") or "").lower()
        if not brand_lower:
            return 0.0

        if not pf.preferred_brands_lower:
            return 0.0

        # Exact match
        if brand_lower in pf.preferred_brands_lower:
            base = cfg.brand_preferred
            return base * pf.brand_openness_mult

        # Cluster-adjacent
        item_cluster = BRAND_TO_CLUSTER.get(brand_lower, "")
        if item_cluster and item_cluster in pf.user_clusters:
            return cfg.brand_cluster_adjacent

        # Unrelated brand
        return cfg.brand_unrelated_penalty

    def _score_style(self, item: dict, pf: "_ProfileFields", cfg: ProfileScoringConfig) -> float:
        """Direct match: style_persona values against item's style_tags.

        Alias-normalized persona values are compared against item style_tags.
        Matching items get boosted; items with ZERO overlap get penalized
        so that style selection visibly reshapes the feed.
        """
        if not pf.style_prefs_lower:
            return 0.0

        item_tags = item.get("style_tags") or []
        if isinstance(item_tags, str):
            item_tags = [item_tags]
        item_tags_lower = {t.lower().strip() for t in item_tags if t}

        if not item_tags_lower:
            # No style tags on item -> mismatch penalty
            return cfg.style_mismatch_penalty

        hits = len(pf.style_prefs_lower & item_tags_lower)
        if hits == 0:
            return cfg.style_mismatch_penalty

        return min(hits * cfg.style_match, cfg.style_match_cap)

    def _score_formality(self, item: dict, pf: "_ProfileFields", cfg: ProfileScoringConfig) -> float:
        """Score formality alignment inferred from style personas."""
        if not pf.formality_direction:
            return 0.0

        item_formality = (item.get("formality") or "").lower().strip()
        if not item_formality:
            return 0.0

        # Map DB formality values to direction
        if pf.formality_direction == "formal":
            # User leans formal: boost business casual / smart casual / semi-formal / formal
            if item_formality in ("business casual", "smart casual", "semi-formal", "formal"):
                return cfg.formality_match
        elif pf.formality_direction == "casual":
            # User leans casual: boost casual items
            if item_formality == "casual":
                return cfg.formality_match

        return 0.0

    def _score_attributes(self, item: dict, pf: "_ProfileFields", cfg: ProfileScoringConfig) -> float:
        """Score fit, sleeve, length, neckline, rise using V3 category-aware matching."""
        adj = 0.0
        item_category = (
            item.get("broad_category")
            or item.get("category")
            or ""
        ).lower().strip()

        # If no broad_category, try inferring
        if not item_category:
            item_category = (get_broad_category(item) or "").lower()

        # Fit
        item_fit = (item.get("fit_type") or item.get("fit") or "").lower().strip()
        if item_fit and pf.preferred_fits:
            if _attribute_applies(
                item_fit, item_category,
                pf.preferred_fits, pf.fit_category_mapping, "fitId"
            ):
                adj += cfg.fit_match

        # Sleeve
        item_sleeve = (item.get("sleeve_type") or item.get("sleeve") or "").lower().strip()
        if item_sleeve and pf.preferred_sleeves:
            if _attribute_applies(
                item_sleeve, item_category,
                pf.preferred_sleeves, pf.sleeve_category_mapping, "sleeveId"
            ):
                adj += cfg.sleeve_match

        # Length
        item_length = (item.get("length") or "").lower().strip()
        if item_length:
            # Standard lengths (for tops/bottoms)
            if pf.preferred_lengths:
                if _attribute_applies(
                    item_length, item_category,
                    pf.preferred_lengths, pf.length_category_mapping, "lengthId"
                ):
                    adj += cfg.length_match

            # Dress/skirt lengths (mini, midi, maxi)
            if pf.preferred_lengths_dresses and item_category in ("dresses", "skirts"):
                if _attribute_applies(
                    item_length, item_category,
                    pf.preferred_lengths_dresses,
                    pf.length_dresses_category_mapping, "lengthId"
                ):
                    adj += cfg.length_match

        # Neckline
        item_neckline = (item.get("neckline") or "").lower().strip()
        if item_neckline and pf.preferred_necklines:
            if item_neckline in pf.preferred_necklines:
                adj += cfg.neckline_match

        # Rise (bottoms only)
        item_rise = (item.get("rise") or "").lower().strip()
        if item_rise and pf.preferred_rises and item_category in ("bottoms",):
            if item_rise in pf.preferred_rises:
                adj += cfg.rise_match

        return adj

    def _score_type(self, item: dict, pf: "_ProfileFields", cfg: ProfileScoringConfig) -> float:
        """Score article type against user's type preferences per category."""
        article_type = (item.get("article_type") or "").lower().strip()
        if not article_type:
            return 0.0

        item_category = (
            item.get("broad_category")
            or item.get("category")
            or ""
        ).lower().strip()
        if not item_category:
            item_category = (get_broad_category(item) or "").lower()

        # Pick the right type list based on category
        types_list: Optional[set] = None
        if any(k in item_category for k in ("top", "knit", "woven")):
            types_list = pf.top_types
        elif any(k in item_category for k in ("bottom", "trouser", "pant", "skirt")):
            types_list = pf.bottom_types
        elif any(k in item_category for k in ("dress", "jumpsuit", "romper")):
            types_list = pf.dress_types
        elif any(k in item_category for k in ("outer", "jacket", "coat")):
            types_list = pf.outerwear_types

        if not types_list:
            return 0.0  # No type prefs for this category -> neutral

        # Normalize article type for comparison
        canonical = get_canonical_type(item) or article_type
        at_variants = {
            article_type,
            article_type.replace("-", " "),
            article_type.replace(" ", "-"),
            canonical,
        }

        if at_variants & types_list:
            return cfg.type_match

        return 0.0

    def _score_pattern(self, item: dict, pf: "_ProfileFields", cfg: ProfileScoringConfig) -> float:
        item_pattern = (item.get("pattern") or "").lower().strip()
        if not item_pattern:
            return 0.0

        adj = 0.0
        if pf.patterns_liked and item_pattern in pf.patterns_liked:
            adj += cfg.pattern_match
        if pf.patterns_avoided and item_pattern in pf.patterns_avoided:
            adj += cfg.pattern_avoid_penalty

        return adj

    def _score_occasions(self, item: dict, pf: "_ProfileFields", cfg: ProfileScoringConfig) -> float:
        if not pf.occasions_expanded:
            return 0.0

        item_occasions = item.get("occasions") or []
        if isinstance(item_occasions, str):
            item_occasions = [item_occasions]
        item_occ_lower = {o.lower().strip() for o in item_occasions if o}

        if not item_occ_lower:
            return 0.0

        hits = len(pf.occasions_expanded & item_occ_lower)
        if hits == 0:
            return 0.0

        adj = cfg.occasion_match
        if hits >= 2:
            adj += cfg.occasion_multi_bonus
        return adj

    def _score_color_avoid(self, item: dict, pf: "_ProfileFields", cfg: ProfileScoringConfig) -> float:
        if not pf.colors_to_avoid:
            return 0.0

        # Check color_family and primary_color
        color_family = (item.get("color_family") or "").lower().strip()
        primary_color = (item.get("primary_color") or "").lower().strip()

        if color_family and color_family in pf.colors_to_avoid:
            return cfg.color_avoid_penalty
        if primary_color and primary_color in pf.colors_to_avoid:
            return cfg.color_avoid_penalty

        return 0.0

    def _score_price(self, item: dict, pf: "_ProfileFields", cfg: ProfileScoringConfig) -> float:
        item_price = float(item.get("price") or 0)
        if item_price <= 0:
            return 0.0

        min_p, max_p = pf.price_range
        if max_p <= 0:
            return 0.0  # No price range available

        if min_p <= item_price <= max_p:
            return cfg.price_in_range
        elif item_price > max_p * cfg.price_way_over_threshold:
            return cfg.price_way_over_penalty
        elif item_price > max_p:
            return cfg.price_over_penalty
        elif item_price < min_p * 0.5:
            # Unusually cheap — might signal low quality for premium-brand users
            return cfg.price_over_penalty

        return 0.0

    def _score_coverage(self, item: dict, pf: "_ProfileFields", cfg: ProfileScoringConfig) -> float:
        if not pf.coverage_exclusions:
            return 0.0

        # detect_coverage_dimensions now uses structured Gemini data when available
        item_dims = set(detect_coverage_dimensions(item))
        if not item_dims:
            return 0.0

        penalty = 0.0
        for exclusion_key, blocked_dims in _COVERAGE_KILLS.items():
            if exclusion_key in pf.coverage_exclusions and (item_dims & blocked_dims):
                penalty += cfg.coverage_kill_penalty

        return penalty

    @staticmethod
    def _score_category_boost(item: dict, cfg: ProfileScoringConfig) -> float:
        """Apply a small additive boost/penalty based on broad category.

        This shifts the scoring baseline so that tops score slightly higher
        and dresses/outerwear score slightly lower, ensuring the best tops
        naturally rise within their proportional allocation.
        """
        if not cfg.category_boosts:
            return 0.0

        item_category = (
            item.get("broad_category")
            or item.get("category")
            or ""
        ).lower().strip()
        if not item_category:
            return 0.0

        for keyword, boost in cfg.category_boosts.items():
            if keyword in item_category:
                return boost

        return 0.0


# ── Profile field pre-computation ─────────────────────────────────

class _ProfileFields:
    """
    Extract and normalize profile fields once per request.

    Accepts OnboardingProfile or a dict with the same field names.
    """

    __slots__ = (
        "preferred_brands_lower", "user_clusters", "brand_openness_mult",
        "style_prefs_lower", "formality_direction",
        "preferred_fits", "fit_category_mapping",
        "preferred_sleeves", "sleeve_category_mapping",
        "preferred_lengths", "length_category_mapping",
        "preferred_lengths_dresses", "length_dresses_category_mapping",
        "preferred_necklines", "preferred_rises",
        "top_types", "bottom_types", "dress_types", "outerwear_types",
        "patterns_liked", "patterns_avoided",
        "occasions_expanded",
        "colors_to_avoid",
        "price_range",
        "coverage_exclusions",
    )

    def __init__(self, profile: Any) -> None:
        g = _getter(profile)

        # Brand
        pref_brands = g("preferred_brands", [])
        self.preferred_brands_lower: Set[str] = {
            b.lower() for b in pref_brands
        } if pref_brands else set()
        self.user_clusters: Set[str] = (
            get_user_brand_clusters(pref_brands) if pref_brands else set()
        )
        openness = (g("brand_openness", None) or "").lower()
        self.brand_openness_mult: float = ProfileScoringConfig().brand_openness_multipliers.get(
            openness, 1.0
        )

        # Style: direct persona -> style_tags matching (with alias normalization)
        style_persona = g("style_persona", [])
        style_directions = g("style_directions", [])
        style_list = style_persona if style_persona else style_directions
        if style_list:
            normalized: Set[str] = set()
            for s in style_list:
                s_low = s.lower().strip()
                # Apply alias: "boho" -> "bohemian", "glam" -> "glamorous", etc.
                canonical = STYLE_ALIASES.get(s_low, s_low)
                normalized.add(canonical)
                # Also keep the original so both forms match
                if canonical != s_low:
                    normalized.add(s_low)
            self.style_prefs_lower: Set[str] = normalized
        else:
            self.style_prefs_lower: Set[str] = set()

        # Formality direction inferred from personas
        self.formality_direction: Optional[str] = (
            _infer_formality_preference(style_list) if style_list else None
        )

        # V3 Attribute preferences with category mappings
        self.preferred_fits: Set[str] = _lower_set(g("preferred_fits", []))
        self.fit_category_mapping: List[Dict] = g("fit_category_mapping", [])

        self.preferred_sleeves: Set[str] = _lower_set(g("preferred_sleeves", []))
        self.sleeve_category_mapping: List[Dict] = g("sleeve_category_mapping", [])

        self.preferred_lengths: Set[str] = _lower_set(g("preferred_lengths", []))
        self.length_category_mapping: List[Dict] = g("length_category_mapping", [])

        self.preferred_lengths_dresses: Set[str] = _lower_set(g("preferred_lengths_dresses", []))
        self.length_dresses_category_mapping: List[Dict] = g("length_dresses_category_mapping", [])

        self.preferred_necklines: Set[str] = _lower_set(g("preferred_necklines", []))
        self.preferred_rises: Set[str] = _lower_set(g("preferred_rises", []))

        # Type preferences
        self.top_types: Set[str] = _lower_set(g("top_types", []))
        self.bottom_types: Set[str] = _lower_set(g("bottom_types", []))
        self.dress_types: Set[str] = _lower_set(g("dress_types", []))
        self.outerwear_types: Set[str] = _lower_set(g("outerwear_types", []))

        # Patterns
        self.patterns_liked: Set[str] = _lower_set(
            g("patterns_liked", []) or g("patterns_preferred", [])
        )
        self.patterns_avoided: Set[str] = _lower_set(
            g("patterns_avoided", []) or g("patterns_to_avoid", [])
        )

        # Occasions (expanded)
        occasions_raw = g("occasions", [])
        self.occasions_expanded: Set[str] = (
            _expand_occasions(occasions_raw) if occasions_raw else set()
        )

        # Colors to avoid
        self.colors_to_avoid: Set[str] = _lower_set(g("colors_to_avoid", []))

        # Price range: use explicit if set, otherwise derive from brands
        min_p = g("global_min_price", None)
        max_p = g("global_max_price", None)
        if min_p and max_p and max_p > 0:
            self.price_range: Tuple[float, float] = (float(min_p), float(max_p))
        elif pref_brands:
            self.price_range = derive_price_range(pref_brands)
        else:
            self.price_range = (0.0, 0.0)

        # Coverage exclusions: from boolean flags + styles_to_avoid
        exclusions: Set[str] = set()
        if g("no_crop", False):
            exclusions.add("no_crop")
        if g("no_revealing", False):
            exclusions.add("no_revealing")
        if g("no_deep_necklines", False):
            exclusions.add("no_deep_necklines")
        if g("no_sleeveless", False):
            exclusions.add("no_sleeveless")
        if g("no_tanks", False):
            exclusions.add("no_sleeveless")  # tanks are a subset of sleeveless

        # Also handle styles_to_avoid list
        for s in (g("styles_to_avoid", []) or []):
            s_lower = s.lower().replace("-", "_")
            # Map string values to exclusion flags
            if s_lower in ("deep_necklines", "deep necklines"):
                exclusions.add("no_deep_necklines")
            elif s_lower in ("sheer", "cutouts", "backless", "strapless"):
                exclusions.add("no_revealing")
            elif s_lower == "crop":
                exclusions.add("no_crop")
            elif s_lower == "sleeveless":
                exclusions.add("no_sleeveless")

        self.coverage_exclusions: Set[str] = exclusions


# ── Helpers ───────────────────────────────────────────────────────

def _getter(obj: Any):
    """Return a getter function that works with both objects and dicts."""
    if isinstance(obj, dict):
        return lambda key, default=None: obj.get(key, default)
    return lambda key, default=None: getattr(obj, key, default)


def _lower_set(values: list) -> Set[str]:
    """Convert a list of strings to a lowercase set."""
    if not values:
        return set()
    return {v.lower().strip() for v in values if v}


def _attribute_applies(
    item_value: str,
    item_category: str,
    preferences: Set[str],
    category_mapping: List[Dict[str, Any]],
    mapping_key: str,
) -> bool:
    """
    Check if an attribute preference applies to the given item/category.

    Uses V3 category mappings to scope preferences (e.g., "fitted" only
    applies to "tops" if the user mapped it that way).

    If no mapping exists, assumes the preference applies globally.
    """
    if item_value not in preferences:
        return False

    # No mapping -> applies globally
    if not category_mapping:
        return True

    # Find the mapping for this specific preference value
    for mapping in category_mapping:
        mapped_val = (mapping.get(mapping_key) or "").lower()
        if mapped_val == item_value:
            categories = mapping.get("categories", [])
            if any(
                item_category in cat.lower() or cat.lower() in item_category
                for cat in categories
            ):
                return True
            return False  # Mapping exists but item category not in it

    # No mapping found for this value -> applies globally
    return True
