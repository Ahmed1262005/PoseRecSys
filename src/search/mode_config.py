"""
Mode-based configuration for the fashion search query planner.

Modes are high-level intent labels that the LLM picks from a menu.
Deterministic code expands them into concrete filters and exclusions.
This replaces the old approach where the LLM constructed filter value
lists directly (which it often got wrong).

Architecture:
    LLM picks modes[]  -->  expand_modes()  -->  filters + exclusions
    LLM picks attributes{}  -->  merged with mode filters  -->  request
    LLM picks avoid{}  -->  merged with mode exclusions  -->  request

Three sections in the LLM output:
    1. modes[]      - abstract/subjective intent (coverage, vibe, occasion)
    2. attributes{} - concrete positive values user explicitly stated
    3. avoid{}      - concrete negative values user explicitly said NO to
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mode configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModeConfig:
    """Configuration for a single mode.

    filters:            positive attribute values to boost/require
    exclusions:         attribute values to exclude (hard drops)
    name_exclusions:    substrings to exclude from product names (lowercase).
                        Products whose name contains any of these substrings
                        are dropped. This catches data quality gaps where the
                        product name says "backless" but style_tags is empty.
    """
    filters: Dict[str, List[str]] = field(default_factory=dict)
    exclusions: Dict[str, List[str]] = field(default_factory=dict)
    name_exclusions: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# All ~40 modes
# ---------------------------------------------------------------------------

MODE_CONFIGS: Dict[str, ModeConfig] = {

    # === COVERAGE modes (composable, each maps to exclusion filters) ===

    "cover_arms": ModeConfig(
        exclusions={
            "sleeve_type": ["Short", "Cap", "Spaghetti Strap", "Sleeveless"],
            "neckline": ["Off-Shoulder", "One Shoulder", "Strapless"],
            "materials": ["Mesh", "Lace", "Chiffon"],
        },
        name_exclusions=(
            "sheer", "see through", "see-through", "mesh",
            "lace sleeve", "lace long sleeve", "lace top",
        ),
    ),
    "cover_chest": ModeConfig(
        exclusions={
            "neckline": [
                "V-Neck", "Deep V-Neck", "Sweetheart", "Halter",
                "Off-Shoulder", "One Shoulder", "Strapless", "Plunging",
            ],
        },
    ),
    "cover_back": ModeConfig(
        exclusions={
            "style_tags": ["Backless", "Open Back", "Low Back"],
        },
        name_exclusions=(
            "backless", "open back", "open-back", "low back", "low-back",
            "cowl back", "cowl-back",
            "keyhole back", "cut out back", "cutout back",
            "scoop back", "scoopback", "scoop-back",
            "lace up back", "lace-up back", "tie back",
            "v-back", "v back", "racerback", "racer back",
        ),
    ),
    "cover_legs": ModeConfig(
        exclusions={
            "length": ["Mini", "Cropped", "Micro"],
        },
    ),
    "cover_straps": ModeConfig(
        exclusions={
            "neckline": ["Strapless", "Off-Shoulder", "Halter", "One Shoulder"],
            "sleeve_type": ["Sleeveless", "Spaghetti Strap"],
            "style_tags": ["Backless", "Open Back", "Low Back"],
        },
        name_exclusions=(
            "backless", "open back", "open-back", "low back", "low-back",
            "halter", "strapless", "racerback", "racer back",
            "keyhole back", "scoop back", "scoopback",
            "cami", "camisole",
        ),
    ),
    "cover_stomach": ModeConfig(
        exclusions={
            "fit_type": ["Fitted", "Slim"],
            "silhouette": ["Bodycon"],
            "length": ["Cropped"],
        },
    ),
    "opaque": ModeConfig(
        exclusions={
            "materials": ["Mesh", "Lace", "Chiffon", "Sheer"],
        },
        name_exclusions=(
            "sheer", "see through", "see-through", "mesh",
            "transparent",
        ),
    ),

    # "modest" is a composite — handled via MODE_IMPLIES (all coverage + opaque)

    "modest": ModeConfig(
        # No direct filters/exclusions — everything comes from implied modes
    ),

    # === FIT modes ===

    "relaxed_fit": ModeConfig(
        exclusions={
            "fit_type": ["Fitted", "Slim"],
            "silhouette": ["Bodycon"],
        },
    ),
    "not_oversized": ModeConfig(
        exclusions={
            "fit_type": ["Oversized", "Relaxed"],
        },
    ),

    # === OCCASION modes (positive occasion + formality filters) ===

    "work": ModeConfig(
        filters={
            "occasions": ["Office", "Work"],
            "formality": ["Business Casual"],
        },
    ),
    "date_night": ModeConfig(
        filters={
            "occasions": ["Date Night"],
            "formality": ["Smart Casual"],
        },
    ),
    "wedding_guest": ModeConfig(
        filters={
            "occasions": ["Wedding Guest"],
            "formality": ["Formal", "Semi-Formal"],
        },
    ),
    "party": ModeConfig(
        filters={
            "occasions": ["Party", "Night Out"],
        },
    ),
    "brunch": ModeConfig(
        filters={
            "occasions": ["Brunch", "Weekend"],
            "formality": ["Casual", "Smart Casual"],
        },
    ),
    "vacation": ModeConfig(
        filters={
            "occasions": ["Vacation", "Beach"],
        },
    ),
    "formal_event": ModeConfig(
        filters={
            "formality": ["Formal"],
        },
    ),
    "interview": ModeConfig(
        filters={
            "occasions": ["Office"],
            "formality": ["Business Casual"],
        },
    ),
    "funeral": ModeConfig(
        filters={
            "formality": ["Formal"],
        },
        # implies modest (handled by MODE_IMPLIES)
        # color hint: Black/Navy/Gray handled by LLM in attributes
    ),
    "religious_event": ModeConfig(
        filters={
            "formality": ["Formal", "Semi-Formal"],
        },
        # implies modest (handled by MODE_IMPLIES)
    ),
    "graduation": ModeConfig(
        filters={
            "formality": ["Smart Casual", "Semi-Formal"],
        },
    ),
    "birthday": ModeConfig(
        filters={
            "occasions": ["Party"],
        },
    ),
    "family_gathering": ModeConfig(
        filters={
            "formality": ["Smart Casual"],
        },
    ),

    # === FORMALITY modes (single-select in practice) ===

    "very_formal": ModeConfig(
        filters={"formality": ["Formal"]},
    ),
    "formal": ModeConfig(
        filters={"formality": ["Formal", "Semi-Formal"]},
    ),
    "smart_casual": ModeConfig(
        filters={"formality": ["Smart Casual"]},
    ),
    "casual": ModeConfig(
        filters={"formality": ["Casual"]},
    ),

    # === AESTHETIC modes (multi-select, maps to style_tags + sometimes formality) ===

    "quiet_luxury": ModeConfig(
        filters={
            "style_tags": ["Classic", "Minimalist", "Modern"],
            "formality": ["Smart Casual", "Business Casual"],
        },
    ),
    "clean_minimal": ModeConfig(
        filters={
            "style_tags": ["Minimalist", "Modern"],
        },
    ),
    "french_chic": ModeConfig(
        filters={
            "style_tags": ["Classic", "Romantic"],
            "formality": ["Smart Casual"],
        },
    ),
    "bohemian": ModeConfig(
        filters={
            "style_tags": ["Bohemian", "Romantic"],
        },
    ),
    "edgy": ModeConfig(
        filters={
            "style_tags": ["Edgy", "Streetwear"],
        },
    ),
    "romantic": ModeConfig(
        filters={
            "style_tags": ["Romantic", "Glamorous"],
        },
    ),
    "glamorous": ModeConfig(
        filters={
            "style_tags": ["Glamorous", "Sexy"],
        },
    ),
    "sporty": ModeConfig(
        filters={
            "style_tags": ["Sporty"],
        },
    ),
    "western": ModeConfig(
        filters={
            "style_tags": ["Western"],
        },
    ),
    "dark_academia": ModeConfig(
        filters={
            "style_tags": ["Classic", "Vintage"],
        },
        # color hint: dark colors — handled by LLM in attributes/semantic_query
    ),
    "y2k": ModeConfig(
        filters={
            "style_tags": ["Streetwear", "Modern"],
        },
    ),
    "coastal": ModeConfig(
        filters={
            "style_tags": ["Classic", "Bohemian"],
        },
        # color hint: light colors — handled by LLM in attributes/semantic_query
    ),
    "preppy": ModeConfig(
        filters={
            "style_tags": ["Preppy", "Classic"],
        },
    ),

    # === WEATHER modes ===

    "hot_weather": ModeConfig(
        filters={
            "seasons": ["Summer"],
        },
    ),
    "cold_weather": ModeConfig(
        filters={
            "seasons": ["Winter", "Fall"],
        },
    ),
    "transitional": ModeConfig(
        filters={
            "seasons": ["Spring", "Fall"],
        },
    ),
}


# ---------------------------------------------------------------------------
# Mode implications (code-side automatic expansion)
# ---------------------------------------------------------------------------

MODE_IMPLIES: Dict[str, List[str]] = {
    "funeral": ["modest"],
    "religious_event": ["modest"],
    "modest": [
        "cover_arms", "cover_chest", "cover_back",
        "cover_legs", "cover_straps", "cover_stomach",
        "opaque",
    ],
}


# ---------------------------------------------------------------------------
# Mode conflicts (warn, don't crash)
# ---------------------------------------------------------------------------

MODE_CONFLICTS: Set[FrozenSet[str]] = {
    frozenset({"relaxed_fit", "not_oversized"}),
}


# ---------------------------------------------------------------------------
# Valid mode names (for validation)
# ---------------------------------------------------------------------------

VALID_MODES: Set[str] = set(MODE_CONFIGS.keys())


# ---------------------------------------------------------------------------
# expand_modes() — the core expansion function
# ---------------------------------------------------------------------------

def expand_modes(
    modes: List[str],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]], List[str]]:
    """Expand a list of mode names into concrete filters and exclusions.

    Args:
        modes: List of mode names selected by the LLM.

    Returns:
        A 4-tuple of:
            mode_filters:       Dict[str, List[str]] — positive attribute filters
            mode_exclusions:    Dict[str, List[str]] — attribute values to exclude
            expanded_filters:   Dict[str, List[str]] — broadened values for
                                lenient semantic post-filtering (superset of
                                mode_filters values)
            name_exclusions:    List[str] — lowercase substrings to exclude
                                from product names (catches data quality gaps
                                where product name says "backless" but
                                style_tags is empty)
    """
    if not modes:
        return {}, {}, {}, []

    # 1. Validate and warn about unknown modes
    valid_modes: List[str] = []
    for mode in modes:
        if mode in VALID_MODES:
            valid_modes.append(mode)
        else:
            logger.warning("Unknown mode '%s' — ignoring", mode)

    if not valid_modes:
        return {}, {}, {}, []

    # 2. Resolve implications (transitive)
    resolved: Set[str] = set()
    stack = list(valid_modes)
    while stack:
        m = stack.pop()
        if m in resolved:
            continue
        resolved.add(m)
        for implied in MODE_IMPLIES.get(m, []):
            if implied not in resolved:
                stack.append(implied)

    # 3. Check for conflicts (log warning, don't crash)
    resolved_list = sorted(resolved)  # deterministic order
    for conflict_pair in MODE_CONFLICTS:
        if conflict_pair.issubset(resolved):
            pair_str = " + ".join(sorted(conflict_pair))
            logger.warning(
                "Conflicting modes detected: %s — both will be applied, "
                "results may be empty",
                pair_str,
            )

    # 4. Collect filters, exclusions, and name exclusions from all resolved modes
    mode_filters: Dict[str, List[str]] = {}
    mode_exclusions: Dict[str, List[str]] = {}
    name_exclusion_set: Set[str] = set()

    for mode_name in resolved_list:
        config = MODE_CONFIGS.get(mode_name)
        if not config:
            continue

        # Union filters
        for key, values in config.filters.items():
            if key not in mode_filters:
                mode_filters[key] = []
            for v in values:
                if v not in mode_filters[key]:
                    mode_filters[key].append(v)

        # Union exclusions
        for key, values in config.exclusions.items():
            if key not in mode_exclusions:
                mode_exclusions[key] = []
            for v in values:
                if v not in mode_exclusions[key]:
                    mode_exclusions[key].append(v)

        # Collect name exclusions
        if config.name_exclusions:
            name_exclusion_set.update(config.name_exclusions)

    # 5. Build expanded_filters for lenient semantic post-filtering.
    #    These are the same as mode_filters — the mode system provides
    #    exact values, and the post-filter allows nulls as lenient fallback.
    expanded_filters = {k: list(v) for k, v in mode_filters.items()}

    return mode_filters, mode_exclusions, expanded_filters, sorted(name_exclusion_set)


# ---------------------------------------------------------------------------
# RRF weight lookup (moved from QueryClassifier)
# ---------------------------------------------------------------------------

# Cache loaded weights
_rrf_weights: Optional[Dict[str, Tuple[float, float]]] = None


def get_rrf_weights(intent: str) -> Tuple[float, float]:
    """Get (algolia_weight, semantic_weight) for a given intent string.

    Supports env-var overrides: RRF_WEIGHT_EXACT_ALGOLIA, etc.

    Args:
        intent: One of "exact", "specific", "vague".

    Returns:
        Tuple of (algolia_weight, semantic_weight) that sums to 1.0.
    """
    global _rrf_weights

    if _rrf_weights is None:
        def _load(intent_name: str, default_algolia: float) -> Tuple[float, float]:
            env_key = f"RRF_WEIGHT_{intent_name.upper()}_ALGOLIA"
            try:
                algolia_w = float(os.getenv(env_key, str(default_algolia)))
            except ValueError:
                algolia_w = default_algolia
            return (algolia_w, 1.0 - algolia_w)

        _rrf_weights = {
            "exact": _load("exact", 0.85),
            "specific": _load("specific", 0.60),
            "vague": _load("vague", 0.35),
        }

    return _rrf_weights.get(intent, (0.60, 0.40))


# ---------------------------------------------------------------------------
# Mode menu text for the LLM prompt
# ---------------------------------------------------------------------------

def get_mode_menu_text() -> str:
    """Generate the mode menu section for the LLM system prompt.

    Returns a formatted string listing all available modes grouped
    by category with short descriptions.
    """
    return """Available modes (pick zero or more):

COVERAGE (composable — each excludes revealing attributes):
- cover_arms: Hide upper arms (excludes short/cap/spaghetti/sleeveless sleeves, off-shoulder/strapless necklines, sheer materials)
- cover_chest: Hide cleavage/chest (excludes V-neck/sweetheart/halter/plunging/off-shoulder/strapless necklines)
- cover_back: Hide back (excludes backless/open-back/low-back styles)
- cover_legs: Hide legs (excludes mini/cropped/micro lengths)
- cover_straps: Hide bra straps (excludes strapless/off-shoulder/halter/one-shoulder necklines, sleeveless/spaghetti sleeves, back-exposing styles). This is STRUCTURAL coverage — a chiffon or lace blouse with proper neckline and sleeves still hides bra straps. Do NOT combine with opaque unless user explicitly mentions see-through/sheer concerns.
- cover_stomach: Hide stomach area (excludes fitted/slim fit, bodycon silhouette, cropped length)
- opaque: Non-see-through only (excludes mesh/lace/chiffon/sheer materials). Only use when user explicitly mentions see-through, sheer, or transparent concerns — NOT for general coverage requests.
- modest: Full coverage (activates ALL coverage modes + opaque)

FIT:
- relaxed_fit: Loose/comfortable fit (excludes fitted/slim fit, bodycon)
- not_oversized: Not oversized or baggy (excludes oversized/relaxed fit)

OCCASION (sets occasion + formality filters):
- work: Office/workplace appropriate
- date_night: Date night outfit
- wedding_guest: Wedding guest attire
- party: Party/night out
- brunch: Weekend brunch
- vacation: Vacation/beach/resort
- formal_event: Formal event
- interview: Job interview
- funeral: Funeral/memorial (implies modest, hint dark colors in attributes)
- religious_event: Church/temple/mosque (implies modest)
- graduation: Graduation ceremony
- birthday: Birthday celebration
- family_gathering: Family event

FORMALITY (pick at most one):
- very_formal: Black-tie/gala level
- formal: Formal/semi-formal
- smart_casual: Smart casual / elevated casual
- casual: Casual everyday

AESTHETIC (multi-select):
- quiet_luxury: Understated elegance, "looks expensive", stealth wealth
- clean_minimal: Clean lines, minimalist, modern
- french_chic: Effortless French style, classic romantic
- bohemian: Boho, free-spirited, flowy
- edgy: Edgy, streetwear, punk-inspired
- romantic: Soft, feminine, coquette, balletcore
- glamorous: Bold, sexy, dramatic, mob wife
- sporty: Athletic, athleisure
- western: Western, cowgirl
- dark_academia: Scholarly, vintage, dark tones
- y2k: Y2K, early 2000s, trendy
- coastal: Coastal grandmother, light/breezy
- preppy: Preppy, collegiate, polished

WEATHER:
- hot_weather: Summer/warm weather appropriate
- cold_weather: Winter/fall layering
- transitional: Spring/fall transitional weather"""
