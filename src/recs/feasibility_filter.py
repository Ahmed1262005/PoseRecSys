"""
Feasibility Filter Module

Hard constraint-based filtering that runs BEFORE any ranking or scoring.
User preferences are treated as authoritative constraints, not soft signals.

Philosophy:
- Onboarding preferences = hard constraints (user expects ZERO violations)
- Occasion selection = defines allowed article types (structural gate)
- CLIP/SASRec = only ranks items that ALREADY passed feasibility
- Items either pass or fail - no "almost passes"

Architecture:
1. CANONICALIZE first - handle messy taxonomy before filtering
2. Structural constraints (article_type, sleeve, neckline) = Hard filters
3. Stylistic attributes (color, pattern, vibe) = Soft scoring (later stage)
4. Occasion = Controls which structures are allowed

Key Design Decisions:
- Allowed lists > Blocked lists (safer, easier to reason about)
- Unknown sleeve = penalize, not block (no empty feeds)
- Per-occasion-per-category allowed types for precision

Usage:
    from recs.feasibility_filter import FeasibilityFilter, FilterResult, canonicalize_article_type

    filter = FeasibilityFilter()
    result = filter.check(candidate, user_profile)

    if not result.passes:
        print(f"Blocked: {result.reason}")
        print(f"Failed rules: {result.failed_rules}")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum


# =============================================================================
# Type Definitions
# =============================================================================

class ConstraintType(Enum):
    """Types of constraints."""
    STRUCTURAL = "structural"      # article_type, sleeve, neckline - hard filter
    STYLISTIC = "stylistic"        # color, pattern - soft scoring
    CONTEXTUAL = "contextual"      # occasion - controls allowed structures


@dataclass
class FilterResult:
    """Result of feasibility check with full explainability."""
    passes: bool
    reason: Optional[str] = None
    failed_rules: List[str] = field(default_factory=list)
    passed_rules: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    penalty: float = 0.0  # Soft penalty for unknown attributes (0-1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passes": self.passes,
            "reason": self.reason,
            "failed_rules": self.failed_rules,
            "passed_rules": self.passed_rules,
            "warnings": self.warnings,
            "penalty": self.penalty,
        }


# =============================================================================
# Article Type Canonicalization
# =============================================================================

# Canonical article type mapping - normalizes messy taxonomy
ARTICLE_TYPE_CANON: Dict[str, str] = {
    # Tank variations -> tank_top
    "tank top": "tank_top",
    "tank tops": "tank_top",
    "tank": "tank_top",
    "tank tops & camis": "tank_top",
    "ribbed tank": "tank_top",
    "racerback tank": "tank_top",
    "muscle tank": "tank_top",

    # Camisole variations -> cami
    "camisole": "cami",
    "camisoles": "cami",
    "cami": "cami",
    "silk cami": "cami",
    "lace cami": "cami",

    # Crop top variations -> crop_top
    "crop top": "crop_top",
    "crop tops": "crop_top",
    "cropped top": "crop_top",
    "cropped tee": "crop_top",
    "cropped tank": "crop_top",

    # Blouse variations -> blouse
    "blouse": "blouse",
    "blouses": "blouse",
    "button-down": "blouse",
    "button down": "blouse",
    "button-up": "blouse",
    "button up": "blouse",
    "dress shirt": "blouse",
    "oxford shirt": "blouse",

    # T-shirt variations -> tshirt
    "t-shirt": "tshirt",
    "t-shirts": "tshirt",
    "tee": "tshirt",
    "tees": "tshirt",
    "tshirt": "tshirt",
    "graphic tee": "tshirt",

    # Sweater variations -> sweater
    "sweater": "sweater",
    "sweaters": "sweater",
    "pullover": "sweater",
    "pullovers": "sweater",
    "jumper": "sweater",
    "knit top": "sweater",
    "knit tops": "sweater",
    "knitwear": "sweater",

    # Cardigan variations -> cardigan
    "cardigan": "cardigan",
    "cardigans": "cardigan",
    "cardi": "cardigan",

    # Turtleneck variations -> turtleneck
    "turtleneck": "turtleneck",
    "turtlenecks": "turtleneck",
    "turtle neck": "turtleneck",
    "mock neck": "turtleneck",
    "mock turtleneck": "turtleneck",

    # Shell (professional sleeveless) -> shell
    "shell": "shell",
    "shells": "shell",
    "sleeveless blouse": "shell",

    # Polo -> polo
    "polo": "polo",
    "polos": "polo",
    "polo shirt": "polo",

    # Hoodie/Sweatshirt variations -> hoodie/sweatshirt
    "hoodie": "hoodie",
    "hoodies": "hoodie",
    "hoody": "hoodie",
    "sweatshirt": "sweatshirt",
    "sweatshirts": "sweatshirt",
    "fleece": "sweatshirt",

    # Athletic tops -> athletic_top
    "sports bra": "sports_bra",
    "sports bras": "sports_bra",
    "sport bra": "sports_bra",
    "athletic top": "athletic_top",
    "workout top": "athletic_top",

    # Bodysuits -> bodysuit
    "bodysuit": "bodysuit",
    "bodysuits": "bodysuit",
    "body suit": "bodysuit",

    # Revealing tops -> specific types
    "bralette": "bralette",
    "bralettes": "bralette",
    "bandeau": "bandeau",
    "tube top": "tube_top",
    "halter": "halter_top",
    "halter top": "halter_top",

    # Silk/Evening tops -> silk_top
    "silk top": "silk_top",
    "satin top": "silk_top",
    "sequin top": "sequin_top",
    "sequined top": "sequin_top",
    "beaded top": "sequin_top",

    # Dresses -> specific dress types
    "dress": "dress",
    "dresses": "dress",
    "mini dress": "mini_dress",
    "mini dresses": "mini_dress",
    "midi dress": "midi_dress",
    "midi dresses": "midi_dress",
    "maxi dress": "maxi_dress",
    "maxi dresses": "maxi_dress",
    "cocktail dress": "cocktail_dress",
    "evening dress": "evening_dress",
    "gown": "gown",
    "evening gown": "gown",
    "sheath dress": "sheath_dress",
    "shift dress": "shift_dress",
    "wrap dress": "wrap_dress",
    "sundress": "sundress",
    "slip dress": "slip_dress",
    "bodycon dress": "bodycon_dress",
    "a-line dress": "aline_dress",
    "shirt dress": "shirt_dress",

    # Swimwear -> specific types
    "bikini": "bikini",
    "bikinis": "bikini",
    "bikini top": "bikini",
    "swimsuit": "swimsuit",
    "swimsuits": "swimsuit",
    "swimwear": "swimsuit",
    "one-piece": "swimsuit",
    "cover-up": "coverup",
    "cover up": "coverup",
    "coverup": "coverup",
    "sarong": "sarong",
    "kaftan": "kaftan",
    "caftan": "kaftan",

    # Bottoms -> specific types
    "jeans": "jeans",
    "denim": "jeans",
    "pants": "pants",
    "trousers": "pants",
    "slacks": "pants",
    "dress pants": "dress_pants",
    "wide leg pants": "wide_leg_pants",
    "palazzo pants": "wide_leg_pants",
    "shorts": "shorts",
    "short": "shorts",
    "denim shorts": "shorts",
    "athletic shorts": "athletic_shorts",
    "running shorts": "athletic_shorts",
    "bike shorts": "bike_shorts",
    "leggings": "leggings",
    "yoga pants": "leggings",
    "joggers": "joggers",
    "sweatpants": "sweatpants",
    "track pants": "sweatpants",
    "chinos": "chinos",
    "linen pants": "linen_pants",

    # Skirts -> specific types
    "skirt": "skirt",
    "skirts": "skirt",
    "mini skirt": "mini_skirt",
    "mini skirts": "mini_skirt",
    "midi skirt": "midi_skirt",
    "midi skirts": "midi_skirt",
    "maxi skirt": "maxi_skirt",
    "maxi skirts": "maxi_skirt",
    "pencil skirt": "pencil_skirt",
    "a-line skirt": "aline_skirt",
    "pleated skirt": "pleated_skirt",
    "wrap skirt": "wrap_skirt",

    # One-pieces -> specific types
    "jumpsuit": "jumpsuit",
    "jumpsuits": "jumpsuit",
    "romper": "romper",
    "rompers": "romper",
    "overalls": "overalls",

    # Outerwear -> specific types
    "blazer": "blazer",
    "blazers": "blazer",
    "jacket": "jacket",
    "jackets": "jacket",
    "coat": "coat",
    "coats": "coat",
    "trench coat": "trench_coat",
    "trench": "trench_coat",
    "puffer": "puffer",
    "puffer jacket": "puffer",
    "puffers": "puffer",
    "vest": "vest",
    "vests": "vest",
    "windbreaker": "windbreaker",
    "athletic jacket": "athletic_jacket",
    "linen jacket": "linen_jacket",
    "kimono": "kimono",
    "evening jacket": "evening_jacket",
    "structured jacket": "structured_jacket",
}


def canonicalize_article_type(raw: str) -> str:
    """
    Normalize messy article types to canonical values.

    Args:
        raw: Raw article type string from database

    Returns:
        Canonical article type string (e.g., "tank_top", "blouse")
    """
    if not raw:
        return "unknown"

    normalized = raw.lower().strip()

    # Direct lookup
    if normalized in ARTICLE_TYPE_CANON:
        return ARTICLE_TYPE_CANON[normalized]

    # Priority patterns - check more specific patterns FIRST
    # These override the general ARTICLE_TYPE_CANON substring matching
    priority_patterns = [
        ("cropped tank", "crop_top"),
        ("crop top", "crop_top"),
        ("cropped top", "crop_top"),
        ("cropped tee", "crop_top"),
        ("sports bra", "sports_bra"),
        ("sport bra", "sports_bra"),
        ("silk cami", "cami"),
        ("lace cami", "cami"),
        ("mini dress", "mini_dress"),
        ("midi dress", "midi_dress"),
        ("maxi dress", "maxi_dress"),
        ("cocktail dress", "cocktail_dress"),
        ("evening dress", "evening_dress"),
        ("sheath dress", "sheath_dress"),
        ("shift dress", "shift_dress"),
        ("wrap dress", "wrap_dress"),
        ("slip dress", "slip_dress"),
        ("bodycon dress", "bodycon_dress"),
        ("a-line dress", "aline_dress"),
        ("shirt dress", "shirt_dress"),
        ("athletic shorts", "athletic_shorts"),
        ("running shorts", "athletic_shorts"),
        ("bike shorts", "bike_shorts"),
        ("wide leg pants", "wide_leg_pants"),
        ("palazzo pants", "wide_leg_pants"),
        ("dress pants", "dress_pants"),
        ("track pants", "sweatpants"),
        ("linen pants", "linen_pants"),
        ("mini skirt", "mini_skirt"),
        ("midi skirt", "midi_skirt"),
        ("maxi skirt", "maxi_skirt"),
        ("pencil skirt", "pencil_skirt"),
        ("a-line skirt", "aline_skirt"),
        ("pleated skirt", "pleated_skirt"),
        ("wrap skirt", "wrap_skirt"),
        ("trench coat", "trench_coat"),
        ("puffer jacket", "puffer"),
        ("athletic jacket", "athletic_jacket"),
        ("linen jacket", "linen_jacket"),
        ("evening jacket", "evening_jacket"),
        ("structured jacket", "structured_jacket"),
    ]

    for pattern, canonical in priority_patterns:
        if pattern in normalized:
            return canonical

    # Substring matching for compound names (e.g., "Ribbed Tank Top" contains "tank")
    for pattern, canonical in ARTICLE_TYPE_CANON.items():
        if pattern in normalized:
            return canonical

    return "unknown"


def canonicalize_name(name: str) -> str:
    """
    Extract canonical type from item name.

    Args:
        name: Product name (e.g., "Casual Ribbed Tank Top")

    Returns:
        Canonical article type or "unknown"
    """
    if not name:
        return "unknown"

    name_lower = name.lower()

    # Check for specific patterns in order of specificity
    # More specific patterns first
    specific_patterns = [
        ("crop top", "crop_top"),
        ("cropped top", "crop_top"),
        ("cropped tank", "crop_top"),
        ("tank top", "tank_top"),
        ("tank", "tank_top"),
        ("camisole", "cami"),
        ("cami", "cami"),
        ("sports bra", "sports_bra"),
        ("sport bra", "sports_bra"),
        ("bralette", "bralette"),
        ("halter", "halter_top"),
        ("bandeau", "bandeau"),
        ("tube top", "tube_top"),
        ("bodysuit", "bodysuit"),
        ("blouse", "blouse"),
        ("button-down", "blouse"),
        ("button down", "blouse"),
        ("turtleneck", "turtleneck"),
        ("mock neck", "turtleneck"),
        ("cardigan", "cardigan"),
        ("sweater", "sweater"),
        ("pullover", "sweater"),
        ("hoodie", "hoodie"),
        ("sweatshirt", "sweatshirt"),
        ("t-shirt", "tshirt"),
        ("tee", "tshirt"),
        ("polo", "polo"),
        ("silk top", "silk_top"),
        ("satin top", "silk_top"),
        ("sequin", "sequin_top"),
        ("bikini", "bikini"),
        ("swimsuit", "swimsuit"),
        ("cover-up", "coverup"),
        ("coverup", "coverup"),
        ("legging", "leggings"),
        ("jogger", "joggers"),
        ("sweatpant", "sweatpants"),
        ("jean", "jeans"),
        ("trouser", "pants"),
        ("pant", "pants"),
        ("short", "shorts"),
        ("skirt", "skirt"),
        ("dress", "dress"),
        ("jumpsuit", "jumpsuit"),
        ("romper", "romper"),
        ("blazer", "blazer"),
        ("jacket", "jacket"),
        ("coat", "coat"),
    ]

    for pattern, canonical in specific_patterns:
        if pattern in name_lower:
            return canonical

    return "unknown"


# =============================================================================
# Occasion -> Allowed Types (Allowed List Approach)
# =============================================================================

# Allowed canonical types PER OCCASION PER BROAD CATEGORY
# This is the primary filtering mechanism - items must be in the allowed list
OCCASION_ALLOWED: Dict[str, Dict[str, Set[str]]] = {
    "office": {
        "tops": {
            "blouse", "shell", "turtleneck", "sweater", "cardigan",
            "polo", "tshirt",  # Basic tees OK for business casual
        },
        "bottoms": {
            "pants", "dress_pants", "chinos", "pencil_skirt", "midi_skirt",
            "aline_skirt", "wide_leg_pants", "jeans",  # Dark jeans OK for business casual
        },
        "one_piece": {
            "dress", "sheath_dress", "shift_dress", "midi_dress",
            "aline_dress", "wrap_dress", "shirt_dress", "jumpsuit",
        },
        "outerwear": {
            "blazer", "structured_jacket", "coat", "trench_coat", "vest", "cardigan",
        },
    },

    "smart-casual": {
        "tops": {
            "blouse", "shell", "turtleneck", "sweater", "cardigan",
            "polo", "tshirt", "cami",  # Cami OK with blazer
        },
        "bottoms": {
            "pants", "dress_pants", "chinos", "jeans", "pencil_skirt",
            "midi_skirt", "aline_skirt", "wide_leg_pants", "pleated_skirt",
        },
        "one_piece": {
            "dress", "sheath_dress", "shift_dress", "midi_dress", "mini_dress",
            "aline_dress", "wrap_dress", "shirt_dress", "jumpsuit", "slip_dress",
        },
        "outerwear": {
            "blazer", "jacket", "coat", "trench_coat", "vest", "cardigan",
        },
    },

    "casual": {
        # Almost everything allowed for casual
        "tops": {
            "blouse", "tshirt", "tank_top", "cami", "crop_top", "sweater",
            "cardigan", "turtleneck", "polo", "hoodie", "sweatshirt",
            "bodysuit", "halter_top",
        },
        "bottoms": {
            "jeans", "pants", "shorts", "leggings", "joggers", "sweatpants",
            "skirt", "mini_skirt", "midi_skirt", "maxi_skirt", "aline_skirt",
            "pleated_skirt", "chinos", "linen_pants", "wide_leg_pants",
        },
        "one_piece": {
            "dress", "mini_dress", "midi_dress", "maxi_dress", "sundress",
            "wrap_dress", "slip_dress", "bodycon_dress", "aline_dress",
            "jumpsuit", "romper", "overalls",
        },
        "outerwear": {
            "jacket", "blazer", "coat", "hoodie", "cardigan", "vest",
            "puffer", "windbreaker", "kimono",
        },
    },

    "beach": {
        "tops": {
            "tank_top", "cami", "crop_top", "bikini", "halter_top",
            "bandeau", "coverup", "blouse",  # Linen blouse OK
        },
        "bottoms": {
            "shorts", "linen_pants", "maxi_skirt", "sarong", "wide_leg_pants",
        },
        "one_piece": {
            "sundress", "maxi_dress", "coverup", "swimsuit", "bikini",
            "kaftan", "romper",
        },
        "outerwear": {
            "linen_jacket", "kimono", "coverup",
        },
    },

    "active": {
        "tops": {
            "sports_bra", "athletic_top", "tank_top", "tshirt", "hoodie",
            "sweatshirt", "crop_top",
        },
        "bottoms": {
            "leggings", "athletic_shorts", "bike_shorts", "joggers",
            "sweatpants", "shorts",
        },
        "one_piece": set(),  # No dresses for active
        "outerwear": {
            "athletic_jacket", "hoodie", "windbreaker", "puffer",
        },
    },

    "evening": {
        "tops": {
            "silk_top", "cami", "blouse", "sequin_top", "shell",
            "halter_top",  # Dressy halter OK for evening
        },
        "bottoms": {
            "wide_leg_pants", "dress_pants", "midi_skirt", "maxi_skirt",
            "pencil_skirt", "pleated_skirt",
        },
        "one_piece": {
            "cocktail_dress", "evening_dress", "gown", "maxi_dress",
            "midi_dress", "slip_dress", "bodycon_dress", "wrap_dress",
            "jumpsuit",
        },
        "outerwear": {
            "blazer", "evening_jacket", "structured_jacket", "coat",
        },
    },

    "events": {
        "tops": {
            "blouse", "silk_top", "sequin_top", "cami", "shell",
        },
        "bottoms": {
            "wide_leg_pants", "dress_pants", "midi_skirt", "maxi_skirt",
            "pencil_skirt", "pleated_skirt",
        },
        "one_piece": {
            "cocktail_dress", "dress", "midi_dress", "maxi_dress",
            "wrap_dress", "jumpsuit", "romper", "slip_dress",
        },
        "outerwear": {
            "blazer", "jacket", "coat",
        },
    },
}

# Universal blocks (applies to ALL occasions except where explicitly allowed)
# These are items that should NEVER appear for certain occasions
UNIVERSAL_BLOCKS: Dict[str, Set[str]] = {
    "office": {
        "tank_top", "crop_top", "bralette", "bandeau", "tube_top",
        "halter_top", "bikini", "swimsuit", "sports_bra",
        "leggings", "joggers", "sweatpants", "athletic_shorts",
        "hoodie", "sweatshirt", "coverup", "sarong", "kaftan",
    },
    "smart-casual": {
        "bralette", "bandeau", "tube_top", "bikini", "swimsuit",
        "sports_bra", "athletic_shorts", "coverup", "sarong", "kaftan",
    },
    "casual": {
        "gown", "evening_dress", "cocktail_dress",
    },
    "beach": {
        "blazer", "dress_pants", "pencil_skirt", "turtleneck",
        "evening_dress", "cocktail_dress", "gown",
    },
    "active": {
        "blouse", "dress", "skirt", "jeans", "blazer", "heels",
        "cocktail_dress", "evening_dress", "gown", "silk_top",
    },
    "evening": {
        "tshirt", "tank_top", "crop_top", "hoodie", "sweatshirt",
        "leggings", "joggers", "sweatpants", "athletic_shorts",
        "shorts", "bikini", "swimsuit", "sports_bra", "coverup",
    },
    "events": {
        "tshirt", "tank_top", "hoodie", "sweatshirt",
        "leggings", "joggers", "sweatpants", "athletic_shorts",
        "shorts", "bikini", "swimsuit", "sports_bra", "coverup",
    },
}


# =============================================================================
# User Exclusion Mapping
# =============================================================================

# Map user preference flags to canonical types to block
USER_EXCLUSION_MAPPING: Dict[str, Set[str]] = {
    "no_tanks": {"tank_top"},
    "no_crop": {"crop_top"},
    "no_sleeveless": {
        "tank_top", "cami", "halter_top", "bandeau", "tube_top",
        "strapless", "sports_bra", "bralette",
    },
    "no_athletic": {
        "sports_bra", "athletic_top", "leggings", "joggers",
        "sweatpants", "athletic_shorts", "bike_shorts",
    },
    "no_revealing": {
        "crop_top", "bralette", "bandeau", "tube_top",
        "bikini", "bodycon_dress",
    },
    "no_deep_v": set(),  # Handled by neckline check, not article type
    "no_off_shoulder": set(),  # Handled by style check, not article type
}


# Athletic brands - blocked for office/formal occasions
ATHLETIC_BRANDS: Set[str] = {
    "alo", "alo yoga",
    "lululemon",
    "athleta",
    "fabletics",
    "nike",
    "adidas",
    "puma",
    "under armour",
    "reebok",
    "gymshark",
    "sweaty betty",
    "outdoor voices",
    "beyond yoga",
    "vuori",
    "girlfriend collective",
    "free people movement",
    "varley",
    "splits59",
    "year of ours",
}

OCCASIONS_BLOCKING_ATHLETIC_BRANDS: Set[str] = {
    "office", "smart-casual", "evening", "events"
}


# =============================================================================
# Broad Category Mapping
# =============================================================================

def get_broad_category(canonical_type: str) -> str:
    """Map canonical type to broad category for allowed list lookup."""
    tops = {
        "blouse", "shell", "turtleneck", "sweater", "cardigan", "polo",
        "tshirt", "tank_top", "cami", "crop_top", "hoodie", "sweatshirt",
        "bodysuit", "halter_top", "bandeau", "tube_top", "bralette",
        "silk_top", "sequin_top", "sports_bra", "athletic_top",
    }
    bottoms = {
        "pants", "dress_pants", "chinos", "jeans", "shorts", "leggings",
        "joggers", "sweatpants", "athletic_shorts", "bike_shorts",
        "linen_pants", "wide_leg_pants",
        "skirt", "mini_skirt", "midi_skirt", "maxi_skirt", "pencil_skirt",
        "aline_skirt", "pleated_skirt", "wrap_skirt", "sarong",
    }
    one_piece = {
        "dress", "mini_dress", "midi_dress", "maxi_dress", "cocktail_dress",
        "evening_dress", "gown", "sheath_dress", "shift_dress", "wrap_dress",
        "sundress", "slip_dress", "bodycon_dress", "aline_dress", "shirt_dress",
        "jumpsuit", "romper", "overalls", "swimsuit", "bikini", "coverup", "kaftan",
    }
    outerwear = {
        "blazer", "jacket", "coat", "trench_coat", "puffer", "vest",
        "windbreaker", "athletic_jacket", "linen_jacket", "kimono",
        "evening_jacket", "structured_jacket",
    }

    if canonical_type in tops:
        return "tops"
    elif canonical_type in bottoms:
        return "bottoms"
    elif canonical_type in one_piece:
        return "one_piece"
    elif canonical_type in outerwear:
        return "outerwear"
    else:
        return "unknown"


# =============================================================================
# Feasibility Filter Class
# =============================================================================

class FeasibilityFilter:
    """
    Hard constraint filter that determines item eligibility.

    This runs BEFORE any neural ranking (SASRec, etc.).
    Items that fail feasibility are NEVER shown to the user.

    Key features:
    - String-based canonicalization for article type detection
    - Allowed lists per occasion per category (safer than blocked lists)
    - Missing data = penalize, not block (prevents empty feeds)
    - Full explainability for debugging

    Detection methods:
    1. String canonicalization (article_type field)
    2. Name parsing (product name keywords)

    Note: CLIP-based visual analysis was removed. The system now relies
    on string-based detection which is simpler and more predictable.
    """

    def __init__(self):
        self.occasion_allowed = OCCASION_ALLOWED
        self.universal_blocks = UNIVERSAL_BLOCKS
        self.user_exclusion_mapping = USER_EXCLUSION_MAPPING
        self.athletic_brands = ATHLETIC_BRANDS

    def _detect_via_clip(
        self,
        candidate: "Candidate",
        feature: str,
        verbose: bool = False,
    ) -> Tuple[bool, float, str]:
        """
        Detect a structural feature.

        Note: CLIP-based detection has been removed. This method now always
        returns "no_clip_scores" to trigger the string-based fallback logic.

        Args:
            candidate: Candidate to check
            feature: Feature key (e.g., "sleeveless", "crop-tops")

        Returns:
            (detected, score, method) - always returns (False, 0.0, "no_clip_scores")
        """
        # CLIP scores have been removed - always use string-based fallback
        return (False, 0.0, "no_clip_scores")

    def check(
        self,
        candidate: "Candidate",
        occasions: Optional[List[str]] = None,
        user_exclusions: Optional[List[str]] = None,
        excluded_article_types: Optional[Set[str]] = None,
        verbose: bool = False,
    ) -> FilterResult:
        """
        Check if candidate passes all feasibility constraints.

        Uses CLIP visual analysis as PRIMARY detection method,
        with string-based canonicalization as FALLBACK.

        Args:
            candidate: The item to check
            occasions: User-selected occasions (e.g., ["office"])
            user_exclusions: User preference exclusions (e.g., ["no_tanks", "no_crop"])
            excluded_article_types: Explicit article types to exclude
            verbose: If True, print detailed reasoning

        Returns:
            FilterResult with pass/fail and full explainability
        """
        passed_rules = []
        failed_rules = []
        warnings = []
        penalty = 0.0

        # =================================================================
        # Step 1: Extract candidate info + canonicalize (FALLBACK method)
        # =================================================================
        article_type_raw = (candidate.article_type or "").lower().strip()
        name = (candidate.name or "").lower()
        brand = (candidate.brand or "").lower()
        sleeve = (candidate.sleeve or "").lower()

        # Get canonical type - try article_type first, then name
        canonical_type = canonicalize_article_type(article_type_raw)
        if canonical_type == "unknown":
            canonical_type = canonicalize_name(name)

        # Store on candidate for later use
        if hasattr(candidate, 'canonical_type'):
            candidate.canonical_type = canonical_type

        if verbose:
            print(f"[FeasibilityFilter] Item: {name[:40]}")
            print(f"  article_type_raw: {article_type_raw}")
            print(f"  canonical_type: {canonical_type}")

        # =================================================================
        # Check 1: User explicit exclusions - CLIP PRIMARY, string FALLBACK
        # =================================================================
        if user_exclusions:
            # --- no_sleeveless check ---
            if "no_sleeveless" in user_exclusions:
                # PRIMARY: CLIP detection
                clip_sleeveless, clip_score, clip_method = self._detect_via_clip(
                    candidate, "sleeveless", verbose
                )

                if clip_method == "clip" and clip_sleeveless:
                    failed_rules.append(f"clip_sleeveless:{clip_score:.3f}")
                    return FilterResult(
                        passes=False,
                        reason=f"CLIP detected sleeveless (score={clip_score:.3f})",
                        failed_rules=failed_rules,
                        passed_rules=passed_rules,
                    )

                # FALLBACK: String-based detection (if CLIP didn't trigger)
                if not clip_sleeveless:
                    sleeveless_types = {"tank_top", "cami", "halter_top", "bandeau", "tube_top", "sports_bra", "bralette"}
                    if canonical_type in sleeveless_types:
                        failed_rules.append(f"canonical_sleeveless:{canonical_type}")
                        return FilterResult(
                            passes=False,
                            reason=f"Canonical type '{canonical_type}' is sleeveless",
                            failed_rules=failed_rules,
                            passed_rules=passed_rules,
                        )

                    # Check sleeve field
                    if sleeve in {"sleeveless", "strapless"}:
                        failed_rules.append(f"sleeve_field:{sleeve}")
                        return FilterResult(
                            passes=False,
                            reason=f"Sleeve field indicates '{sleeve}'",
                            failed_rules=failed_rules,
                            passed_rules=passed_rules,
                        )

                # If CLIP scores missing, add penalty (don't block)
                if clip_method == "no_clip_scores":
                    warnings.append("no_clip_scores_for_sleeveless_check")
                    penalty += 0.2

                passed_rules.append("sleeveless_check")

            # --- no_crop check ---
            if "no_crop" in user_exclusions:
                # PRIMARY: CLIP detection
                clip_crop, clip_score, clip_method = self._detect_via_clip(
                    candidate, "crop-tops", verbose
                )

                if clip_method == "clip" and clip_crop:
                    failed_rules.append(f"clip_crop:{clip_score:.3f}")
                    return FilterResult(
                        passes=False,
                        reason=f"CLIP detected crop top (score={clip_score:.3f})",
                        failed_rules=failed_rules,
                        passed_rules=passed_rules,
                    )

                # FALLBACK: String-based detection
                if not clip_crop:
                    if canonical_type == "crop_top":
                        failed_rules.append(f"canonical_crop:{canonical_type}")
                        return FilterResult(
                            passes=False,
                            reason=f"Canonical type is crop_top",
                            failed_rules=failed_rules,
                            passed_rules=passed_rules,
                        )

                passed_rules.append("crop_check")

            # --- no_tanks check (string-based, CLIP sleeveless is broader) ---
            if "no_tanks" in user_exclusions:
                if canonical_type == "tank_top":
                    failed_rules.append(f"canonical_tank:{canonical_type}")
                    return FilterResult(
                        passes=False,
                        reason=f"Canonical type is tank_top",
                        failed_rules=failed_rules,
                        passed_rules=passed_rules,
                    )
                passed_rules.append("tank_check")

            # --- no_revealing check (CLIP-based) ---
            if "no_revealing" in user_exclusions:
                revealing_signals = []
                revealing_scores = {}

                for feature in ["deep-necklines", "cutouts", "sheer", "open-back"]:
                    detected, score, method = self._detect_via_clip(candidate, feature, verbose)
                    revealing_scores[feature] = score
                    if detected:
                        revealing_signals.append(feature)

                # Block if any single revealing feature detected
                if revealing_signals:
                    failed_rules.append(f"clip_revealing:{revealing_signals}")
                    return FilterResult(
                        passes=False,
                        reason=f"CLIP detected revealing features: {revealing_signals}",
                        failed_rules=failed_rules,
                        passed_rules=passed_rules,
                    )

                passed_rules.append("revealing_check")

            # --- no_athletic check ---
            if "no_athletic" in user_exclusions:
                athletic_types = {"sports_bra", "athletic_top", "leggings", "joggers", "sweatpants", "athletic_shorts", "bike_shorts"}
                if canonical_type in athletic_types:
                    failed_rules.append(f"canonical_athletic:{canonical_type}")
                    return FilterResult(
                        passes=False,
                        reason=f"Canonical type '{canonical_type}' is athletic",
                        failed_rules=failed_rules,
                        passed_rules=passed_rules,
                    )
                passed_rules.append("athletic_check")

            # --- Legacy: check blocked types from mapping (for backward compat) ---
            for exclusion_key in user_exclusions:
                if exclusion_key in ["no_sleeveless", "no_crop", "no_tanks", "no_revealing", "no_athletic"]:
                    continue  # Already handled above

                blocked_types = self.user_exclusion_mapping.get(exclusion_key, set())
                if canonical_type in blocked_types:
                    failed_rules.append(f"user_exclusion:{exclusion_key}:{canonical_type}")
                    return FilterResult(
                        passes=False,
                        reason=f"User preference '{exclusion_key}' excludes '{canonical_type}'",
                        failed_rules=failed_rules,
                        passed_rules=passed_rules,
                    )

                # Also check name for blocked keywords (backup)
                for blocked in blocked_types:
                    blocked_readable = blocked.replace("_", " ")
                    if blocked_readable in name and canonical_type == "unknown":
                        failed_rules.append(f"user_exclusion_name:{exclusion_key}:{blocked}")
                        return FilterResult(
                            passes=False,
                            reason=f"User preference '{exclusion_key}' excludes '{blocked}' (from name)",
                            failed_rules=failed_rules,
                            passed_rules=passed_rules,
                        )

            passed_rules.append("user_exclusions_check")

        # =================================================================
        # Check 2: Handle sleeve constraint with missing data tolerance
        # =================================================================
        if user_exclusions and "no_sleeveless" in user_exclusions:
            if not sleeve or sleeve == "unknown":
                # Unknown sleeve with strict preference -> penalize, don't block
                warnings.append("sleeve_unknown_with_no_sleeveless_preference")
                penalty += 0.3  # Apply scoring penalty later

                # BUT if canonical_type is known sleeveless, block it
                if canonical_type in {"tank_top", "cami", "halter_top", "bandeau", "tube_top", "sports_bra", "bralette"}:
                    failed_rules.append(f"canonical_sleeveless:{canonical_type}")
                    return FilterResult(
                        passes=False,
                        reason=f"User excluded sleeveless items, '{canonical_type}' is inherently sleeveless",
                        failed_rules=failed_rules,
                        passed_rules=passed_rules,
                    )
            elif sleeve in {"sleeveless", "strapless"}:
                failed_rules.append(f"sleeve_blocked:{sleeve}")
                return FilterResult(
                    passes=False,
                    reason="User excluded sleeveless items",
                    failed_rules=failed_rules,
                    passed_rules=passed_rules,
                )

            passed_rules.append("sleeve_constraint_check")

        # =================================================================
        # Check 3: Explicit article type exclusions
        # =================================================================
        if excluded_article_types:
            for excluded in excluded_article_types:
                excluded_lower = excluded.lower().replace(" ", "_")
                if canonical_type == excluded_lower:
                    failed_rules.append(f"explicit_exclusion:{excluded}")
                    return FilterResult(
                        passes=False,
                        reason=f"Article type '{excluded}' explicitly excluded",
                        failed_rules=failed_rules,
                        passed_rules=passed_rules,
                    )
            passed_rules.append("explicit_exclusions_check")

        # =================================================================
        # Check 4: Occasion-based constraints (MAIN FILTER)
        # =================================================================
        if occasions:
            for occasion in occasions:
                occasion_lower = occasion.lower()

                # Check universal blocks first (hard exclusion)
                blocked_types = self.universal_blocks.get(occasion_lower, set())
                if canonical_type in blocked_types:
                    failed_rules.append(f"universal_block:{occasion}:{canonical_type}")
                    return FilterResult(
                        passes=False,
                        reason=f"'{canonical_type}' universally blocked for {occasion}",
                        failed_rules=failed_rules,
                        passed_rules=passed_rules,
                    )

                # Check if in allowed list for this occasion
                occasion_allowed = self.occasion_allowed.get(occasion_lower, {})
                if occasion_allowed and canonical_type != "unknown":
                    broad_category = get_broad_category(canonical_type)

                    if broad_category in occasion_allowed:
                        allowed_types = occasion_allowed[broad_category]
                        if canonical_type not in allowed_types:
                            failed_rules.append(f"not_in_allowed_list:{occasion}:{broad_category}:{canonical_type}")
                            return FilterResult(
                                passes=False,
                                reason=f"'{canonical_type}' not in allowed {broad_category} for {occasion}",
                                failed_rules=failed_rules,
                                passed_rules=passed_rules,
                            )
                        else:
                            passed_rules.append(f"allowed:{occasion}:{canonical_type}")
                    else:
                        # Unknown broad category - allow with warning
                        warnings.append(f"unknown_broad_category:{broad_category}:{canonical_type}")

                # If canonical_type is unknown, check name for blocked keywords
                if canonical_type == "unknown":
                    for blocked in blocked_types:
                        blocked_readable = blocked.replace("_", " ")
                        if blocked_readable in name:
                            failed_rules.append(f"name_contains_blocked:{occasion}:{blocked}")
                            return FilterResult(
                                passes=False,
                                reason=f"Item name contains '{blocked}' which is blocked for {occasion}",
                                failed_rules=failed_rules,
                                passed_rules=passed_rules,
                            )
                    # Unknown type - penalize but don't block
                    warnings.append(f"unknown_canonical_type_for_occasion:{occasion}")
                    penalty += 0.2

                passed_rules.append(f"occasion_check:{occasion}")

        # =================================================================
        # Check 5: Athletic brand constraints
        # =================================================================
        if occasions:
            for occasion in occasions:
                if occasion.lower() in OCCASIONS_BLOCKING_ATHLETIC_BRANDS:
                    for athletic_brand in self.athletic_brands:
                        if athletic_brand in brand:
                            failed_rules.append(f"athletic_brand:{athletic_brand}:{occasion}")
                            return FilterResult(
                                passes=False,
                                reason=f"Athletic brand '{athletic_brand}' not allowed for {occasion}",
                                failed_rules=failed_rules,
                                passed_rules=passed_rules,
                            )
            passed_rules.append("brand_check")

        # All checks passed
        return FilterResult(
            passes=True,
            reason=None,
            failed_rules=[],
            passed_rules=passed_rules,
            warnings=warnings,
            penalty=penalty,
        )

    def filter_candidates(
        self,
        candidates: List["Candidate"],
        occasions: Optional[List[str]] = None,
        user_exclusions: Optional[List[str]] = None,
        excluded_article_types: Optional[Set[str]] = None,
        verbose: bool = False,
    ) -> Tuple[List["Candidate"], Dict[str, Any]]:
        """
        Filter a list of candidates through feasibility checks.

        Returns:
            (passed_candidates, stats) where stats includes detailed breakdown
        """
        passed = []
        blocked_reasons: Dict[str, int] = {}
        blocked_details: List[Dict] = []
        all_warnings: List[str] = []
        total_penalty = 0.0

        for candidate in candidates:
            result = self.check(
                candidate,
                occasions=occasions,
                user_exclusions=user_exclusions,
                excluded_article_types=excluded_article_types,
                verbose=verbose,
            )

            if result.passes:
                passed.append(candidate)
                total_penalty += result.penalty
                all_warnings.extend(result.warnings)
            else:
                reason = result.reason or "unknown"
                blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1

                if verbose or len(blocked_details) < 10:
                    blocked_details.append({
                        "item_id": candidate.item_id,
                        "name": candidate.name,
                        "article_type": candidate.article_type,
                        "canonical_type": canonicalize_article_type(candidate.article_type or "") or canonicalize_name(candidate.name or ""),
                        "reason": reason,
                        "failed_rules": result.failed_rules,
                    })

        # Deduplicate warnings
        unique_warnings = list(set(all_warnings))

        stats = {
            "total_candidates": len(candidates),
            "passed": len(passed),
            "blocked": len(candidates) - len(passed),
            "blocked_reasons": blocked_reasons,
            "sample_blocked": blocked_details[:10],
            "warnings": unique_warnings[:20],  # Limit warnings
            "avg_penalty": total_penalty / len(passed) if passed else 0,
            "filter_config": {
                "occasions": occasions,
                "user_exclusions": user_exclusions,
                "excluded_article_types": list(excluded_article_types) if excluded_article_types else None,
            }
        }

        if verbose:
            print(f"[FeasibilityFilter] {len(candidates)} -> {len(passed)} candidates")
            print(f"[FeasibilityFilter] Blocked reasons:")
            for reason, count in sorted(blocked_reasons.items(), key=lambda x: -x[1])[:10]:
                print(f"  {count:4d} - {reason}")
            if unique_warnings:
                print(f"[FeasibilityFilter] Warnings: {unique_warnings[:5]}")

        return passed, stats


# =============================================================================
# Convenience Functions
# =============================================================================

def check_feasibility(
    candidate: "Candidate",
    occasions: Optional[List[str]] = None,
    user_exclusions: Optional[List[str]] = None,
    excluded_article_types: Optional[Set[str]] = None,
) -> FilterResult:
    """Convenience function for single-item feasibility check."""
    filter_obj = FeasibilityFilter()
    return filter_obj.check(
        candidate,
        occasions=occasions,
        user_exclusions=user_exclusions,
        excluded_article_types=excluded_article_types,
    )


def filter_by_feasibility(
    candidates: List["Candidate"],
    occasions: Optional[List[str]] = None,
    user_exclusions: Optional[List[str]] = None,
    excluded_article_types: Optional[Set[str]] = None,
    verbose: bool = False,
) -> Tuple[List["Candidate"], Dict[str, Any]]:
    """Convenience function for filtering candidate list."""
    filter_obj = FeasibilityFilter()
    return filter_obj.filter_candidates(
        candidates,
        occasions=occasions,
        user_exclusions=user_exclusions,
        excluded_article_types=excluded_article_types,
        verbose=verbose,
    )


# =============================================================================
# Testing
# =============================================================================

def test_canonicalization():
    """Test the canonicalization function."""
    print("=" * 70)
    print("Testing Canonicalization")
    print("=" * 70)

    test_cases = [
        ("Tank Tops & Camis", "tank_top"),
        ("Ribbed Tank", "tank_top"),
        ("Button-Down Blouse", "blouse"),
        ("Silk Camisole", "cami"),
        ("Cropped Tank Top", "crop_top"),
        ("Sports Bra", "sports_bra"),
        ("Oversized Sweater", "sweater"),
        ("Midi Dress", "midi_dress"),
        ("Athletic Shorts", "athletic_shorts"),
        ("", "unknown"),
        ("Random Unknown Item", "unknown"),
    ]

    for raw, expected in test_cases:
        result = canonicalize_article_type(raw)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: '{raw}' -> '{result}' (expected: '{expected}')")

    print("\nCanonicalization tests complete!")


def test_feasibility_filter():
    """Test the feasibility filter."""
    from recs.models import Candidate

    print("=" * 70)
    print("Testing Feasibility Filter")
    print("=" * 70)

    filter_obj = FeasibilityFilter()

    # Test 1: Tank top for office (should FAIL)
    print("\n1. Tank top for office (should FAIL)...")
    tank = Candidate(
        item_id="test-tank-1",
        name="Casual Tank Top",
        article_type="tank top",
        brand="Generic Brand",
        sleeve="sleeveless",
    )
    result = filter_obj.check(tank, occasions=["office"])
    print(f"   Result: {'PASS' if result.passes else f'FAIL - {result.reason}'}")
    assert not result.passes, "Tank top should fail office filter"

    # Test 2: Blouse for office (should PASS)
    print("\n2. Silk blouse for office (should PASS)...")
    blouse = Candidate(
        item_id="test-blouse-1",
        name="Professional Silk Blouse",
        article_type="blouse",
        brand="Theory",
        sleeve="long-sleeve",
    )
    result = filter_obj.check(blouse, occasions=["office"])
    print(f"   Result: {'PASS' if result.passes else f'FAIL - {result.reason}'}")
    assert result.passes, f"Blouse should pass office filter: {result.reason}"

    # Test 3: Lululemon item for office (should FAIL - athletic brand)
    print("\n3. Lululemon item for office (should FAIL)...")
    lulu = Candidate(
        item_id="test-lulu-1",
        name="Define Jacket",
        article_type="jacket",
        brand="Lululemon",
        sleeve="long-sleeve",
    )
    result = filter_obj.check(lulu, occasions=["office"])
    print(f"   Result: {'PASS' if result.passes else f'FAIL - {result.reason}'}")
    assert not result.passes, "Lululemon should fail office filter"

    # Test 4: User explicitly excludes tanks
    print("\n4. Tank with user 'no_tanks' preference (should FAIL)...")
    result = filter_obj.check(tank, user_exclusions=["no_tanks"])
    print(f"   Result: {'PASS' if result.passes else f'FAIL - {result.reason}'}")
    assert not result.passes, "Tank should fail with no_tanks preference"

    # Test 5: Crop top for office (should FAIL)
    print("\n5. Crop top for office (should FAIL)...")
    crop = Candidate(
        item_id="test-crop-1",
        name="Cute Crop Top",
        article_type="crop top",
        brand="Zara",
        sleeve="short-sleeve",
    )
    result = filter_obj.check(crop, occasions=["office"])
    print(f"   Result: {'PASS' if result.passes else f'FAIL - {result.reason}'}")
    assert not result.passes, "Crop top should fail office filter"

    # Test 6: Crop top for casual (should PASS)
    print("\n6. Crop top for casual (should PASS)...")
    result = filter_obj.check(crop, occasions=["casual"])
    print(f"   Result: {'PASS' if result.passes else f'FAIL - {result.reason}'}")
    assert result.passes, f"Crop top should pass casual filter: {result.reason}"

    # Test 7: Cami for evening (should PASS - silk camis OK for evening)
    print("\n7. Cami for evening (should PASS)...")
    cami = Candidate(
        item_id="test-cami-1",
        name="Silk Camisole Top",
        article_type="camisole",
        brand="Reformation",
        sleeve="sleeveless",
    )
    result = filter_obj.check(cami, occasions=["evening"])
    print(f"   Result: {'PASS' if result.passes else f'FAIL - {result.reason}'}")
    assert result.passes, f"Cami should pass evening filter: {result.reason}"

    # Test 8: Bikini for office (should FAIL)
    print("\n8. Bikini for office (should FAIL)...")
    bikini = Candidate(
        item_id="test-bikini-1",
        name="String Bikini Top",
        article_type="bikini",
        brand="Seafolly",
    )
    result = filter_obj.check(bikini, occasions=["office"])
    print(f"   Result: {'PASS' if result.passes else f'FAIL - {result.reason}'}")
    assert not result.passes, "Bikini should fail office filter"

    # Test 9: Bikini for beach (should PASS)
    print("\n9. Bikini for beach (should PASS)...")
    result = filter_obj.check(bikini, occasions=["beach"])
    print(f"   Result: {'PASS' if result.passes else f'FAIL - {result.reason}'}")
    assert result.passes, f"Bikini should pass beach filter: {result.reason}"

    # Test 10: Unknown sleeve with no_sleeveless (should PASS with penalty)
    print("\n10. Unknown sleeve with no_sleeveless (should PASS with warning)...")
    unknown_sleeve = Candidate(
        item_id="test-unknown-1",
        name="Basic Top",
        article_type="top",
        brand="Generic",
        sleeve=None,  # Unknown sleeve
    )
    result = filter_obj.check(unknown_sleeve, user_exclusions=["no_sleeveless"])
    print(f"   Result: {'PASS' if result.passes else f'FAIL - {result.reason}'}")
    print(f"   Warnings: {result.warnings}")
    print(f"   Penalty: {result.penalty}")
    assert result.passes, "Unknown sleeve should pass with penalty, not block"
    assert result.penalty > 0, "Should have penalty for unknown sleeve"

    print("\n" + "=" * 70)
    print("All feasibility filter tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_canonicalization()
    print()
    test_feasibility_filter()
