"""
Application constants and algorithm configuration.

These are values that don't change based on environment but may need
to be tuned or referenced across the codebase.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set


# =============================================================================
# Pipeline Configuration
# =============================================================================

@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the recommendation pipeline."""
    
    # Candidate retrieval limits
    PRIMARY_CANDIDATES: int = 300
    CONTEXTUAL_CANDIDATES: int = 100
    EXPLORATION_CANDIDATES: int = 50
    
    # Diversity limits
    MAX_PER_CATEGORY: int = 8
    
    # Exploration rate for injecting variety
    EXPLORATION_RATE: float = 0.10
    
    # Feed limits
    DEFAULT_LIMIT: int = 50
    MAX_LIMIT: int = 200


# Default pipeline config instance
DEFAULT_PIPELINE_CONFIG = PipelineConfig()


# =============================================================================
# SASRec Ranker Configuration
# =============================================================================

@dataclass(frozen=True)
class SASRecConfig:
    """Configuration for SASRec ranking model."""
    
    # Scoring weights for warm users (with interaction history)
    WARM_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "sasrec": 0.40,
        "embedding": 0.35,
        "preference": 0.25,
    })
    
    # Scoring weights for cold users (no interaction history)
    COLD_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "embedding": 0.40,
        "preference": 0.60,
    })
    
    # Sequence settings
    MAX_SEQ_LENGTH: int = 50
    MIN_SEQUENCE_FOR_SASREC: int = 5
    
    # Diversity caps
    BRAND_DIVERSITY_CAP: float = 0.25
    SPORTSWEAR_FREQUENCY_CAP: float = 0.15


DEFAULT_SASREC_CONFIG = SASRecConfig()


# =============================================================================
# Candidate Selection Configuration
# =============================================================================

@dataclass(frozen=True)
class CandidateSelectionConfig:
    """Configuration for candidate selection module."""
    
    # Retrieval limits
    PRIMARY_CANDIDATES: int = 300
    CONTEXTUAL_CANDIDATES: int = 100
    EXPLORATION_CANDIDATES: int = 50
    
    # Brand diversity
    BRAND_DIVERSITY_CAP: float = 0.25
    
    # Brand openness multipliers
    BRAND_OPENNESS_MULTIPLIERS: Dict[str, float] = field(default_factory=lambda: {
        "loyal": 1.2,
        "open": 0.8,
        "neutral": 1.0,
    })


DEFAULT_CANDIDATE_CONFIG = CandidateSelectionConfig()


# =============================================================================
# Diversity Configuration
# =============================================================================

@dataclass(frozen=True)
class DiversityConfig:
    """Configuration for diversity constraints."""
    
    default_limit: int = 8
    single_category_limit: int = 50
    two_category_limit: int = 25
    three_category_limit: int = 16
    warm_user_limit: int = 50


DEFAULT_DIVERSITY_CONFIG = DiversityConfig()


# =============================================================================
# Soft Scoring Weights
# =============================================================================

@dataclass(frozen=True)
class SoftScoringWeights:
    """Weights for soft preference scoring."""
    
    # Maximum total boost (positive or negative)
    max_total_boost: float = 0.15
    
    # Semantic floor - don't boost items below this similarity
    semantic_floor: float = 0.25
    
    # Positive boosts
    fit_boost: float = 0.03
    sleeve_boost: float = 0.02
    length_boost: float = 0.02
    rise_boost: float = 0.02
    brand_boost: float = 0.05
    article_type_boost: float = 0.03
    
    # Negative demotes
    color_demote: float = -0.15
    material_demote: float = -0.10
    brand_demote: float = -0.20


DEFAULT_SOFT_WEIGHTS = SoftScoringWeights()


# =============================================================================
# Category Mappings
# =============================================================================

# Women's fashion categories
WOMEN_CATEGORIES: List[str] = [
    "tops_knitwear",
    "tops_woven", 
    "tops_sleeveless",
    "tops_special",
    "dresses",
    "bottoms_trousers",
    "bottoms_skorts",
    "outerwear",
    "sportswear",
]

# Broad category mappings
BROAD_CATEGORY_MAP: Dict[str, str] = {
    "tops_knitwear": "tops",
    "tops_woven": "tops",
    "tops_sleeveless": "tops",
    "tops_special": "tops",
    "dresses": "dresses",
    "bottoms_trousers": "bottoms",
    "bottoms_skorts": "bottoms",
    "outerwear": "outerwear",
    "sportswear": "sportswear",
}


# =============================================================================
# Sportswear Detection
# =============================================================================

SPORTSWEAR_BROAD_CATEGORIES: Set[str] = {
    "sportswear",
    "activewear",
    "athletic",
}

SPORTSWEAR_ARTICLE_TYPES: Set[str] = {
    "leggings",
    "sports bra",
    "sports top",
    "joggers",
    "track pants",
    "athletic shorts",
    "yoga pants",
    "running shorts",
}

SPORTSWEAR_BRANDS: Set[str] = {
    "nike",
    "adidas",
    "puma",
    "under armour",
    "lululemon",
    "athleta",
    "fabletics",
    "alo yoga",
    "gymshark",
}

SPORTSWEAR_NAME_KEYWORDS: Set[str] = {
    "athletic",
    "sport",
    "yoga",
    "running",
    "workout",
    "gym",
    "training",
    "active",
    "performance",
}


# =============================================================================
# Attribute Values
# =============================================================================

# Pattern types
PATTERN_TYPES: List[str] = [
    "solid",
    "striped",
    "floral",
    "geometric",
    "abstract",
    "animal_print",
    "checkered",
    "polka_dot",
]

# Style types
STYLE_TYPES: List[str] = [
    "casual",
    "office",
    "evening",
    "bohemian",
    "minimalist",
    "streetwear",
    "classic",
    "romantic",
]

# Color families
COLOR_FAMILIES: List[str] = [
    "neutral",
    "bright",
    "cool",
    "warm",
    "pastel",
    "dark",
    "earth",
]

# Fit types
FIT_TYPES: List[str] = [
    "fitted",
    "regular",
    "relaxed",
    "oversized",
    "cropped",
]

# Occasion types
OCCASION_TYPES: List[str] = [
    "everyday",
    "work",
    "date_night",
    "party",
    "formal",
    "vacation",
    "weekend",
]


# =============================================================================
# Price Range Defaults
# =============================================================================

DEFAULT_PRICE_RANGES: Dict[str, tuple] = {
    "t-shirts": (10, 50),
    "polos": (30, 100),
    "sweaters": (50, 200),
    "hoodies": (40, 120),
    "shirts": (40, 150),
    "dresses": (50, 300),
    "jeans": (50, 200),
    "pants": (40, 150),
    "shorts": (30, 100),
    "skirts": (40, 150),
    "jackets": (80, 400),
    "coats": (100, 600),
}
