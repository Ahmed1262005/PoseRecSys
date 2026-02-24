"""
TATTOO-Inspired Outfit Engine  v2
==================================

Production service for "Complete the Fit" outfit recommendations and
TATTOO-scored similar items.  8-dimension rule-based compatibility
scoring inspired by the TATTOO paper (arXiv:2509.23242) with
fashion-expert rules for occasion, style, fabric, silhouette, color,
seasonality, pattern, and price.

Dimensions (8):
  1. Occasion & Formality  - formality ladder + occasion overlap + conflict penalties
  2. Style                 - adjacency matrix + style strength + bridge items
  3. Fabric                - contrast principle + pair table + weight compatibility
  4. Silhouette            - category-aware balance rules + waist logic
  5. Color                 - temperature-aware harmony + saturation + same-mat penalty
  6. Seasonality           - temp band + layer role + seasonal fabric logic
  7. Pattern               - solid/subtle/bold clash detection
  8. Price                 - price coherence ratio
"""

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from supabase import Client

logger = logging.getLogger(__name__)


# =============================================================================
# 1. AESTHETIC PROFILE
# =============================================================================

@dataclass
class AestheticProfile:
    """Structured aesthetic profile extracted from Gemini product_attributes."""

    # Core attributes (from DB)
    color_family: Optional[str] = None
    primary_color: Optional[str] = None
    secondary_colors: List[str] = field(default_factory=list)
    style_tags: List[str] = field(default_factory=list)
    occasions: List[str] = field(default_factory=list)
    seasons: List[str] = field(default_factory=list)
    apparent_fabric: Optional[str] = None
    texture: Optional[str] = None
    formality: Optional[str] = None
    pattern: Optional[str] = None
    fit_type: Optional[str] = None
    silhouette: Optional[str] = None
    neckline: Optional[str] = None
    sleeve_type: Optional[str] = None
    length: Optional[str] = None
    coverage_level: Optional[str] = None
    sheen: Optional[str] = None         # NEW: matte/satin/shiny/shimmer/metallic
    rise: Optional[str] = None          # NEW: high/mid/low
    leg_shape: Optional[str] = None     # NEW: skinny/straight/wide/flare
    stretch: Optional[str] = None       # NEW: no stretch/slight/stretchy/very stretchy

    # Derived fields (computed at build time, not from DB)
    formality_level: int = 2            # 1-5 scale
    is_bridge: bool = False             # blazer, trench, leather jacket, etc.
    primary_style: Optional[str] = None # dominant style tag
    style_strength: float = 0.4         # 0-1: basic tee=0.2, sequin corset=0.9
    material_family: Optional[str] = None  # from _get_fabric_family()
    texture_intensity: Optional[str] = None  # smooth/medium/strong
    shine_level: Optional[str] = None   # matte/slight/shiny
    fabric_weight: Optional[str] = None # light/mid/heavy
    layer_role: Optional[str] = None    # base/midlayer/outer
    temp_band: Optional[str] = None     # hot/mild/cold/any
    color_saturation: Optional[str] = None  # muted/medium/bright

    # Product metadata
    product_id: Optional[str] = None
    name: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    broad_category: Optional[str] = None
    price: float = 0.0
    image_url: Optional[str] = None
    similarity: float = 0.0  # pgvector cosine similarity

    # Gemini-derived true category (authoritative)
    gemini_category_l1: Optional[str] = None
    gemini_category_l2: Optional[str] = None

    @classmethod
    def from_product_and_attrs(
        cls, product: Dict, attrs: Optional[Dict] = None
    ) -> "AestheticProfile":
        """Build profile from a products row + product_attributes row."""
        p = product
        a = attrs or {}

        construction = a.get("construction") or {}
        if isinstance(construction, str):
            try:
                construction = json.loads(construction)
            except (json.JSONDecodeError, TypeError):
                construction = {}

        profile = cls(
            product_id=p.get("id") or p.get("product_id"),
            name=p.get("name", ""),
            brand=p.get("brand", ""),
            category=p.get("category", ""),
            broad_category=p.get("broad_category", ""),
            price=float(p.get("price", 0) or 0),
            image_url=p.get("primary_image_url", ""),
            color_family=a.get("color_family") or p.get("base_color"),
            primary_color=a.get("primary_color") or p.get("base_color"),
            secondary_colors=_to_list(a.get("secondary_colors")),
            style_tags=_to_list(a.get("style_tags") or p.get("style_tags")),
            occasions=_to_list(a.get("occasions")),
            seasons=_to_list(a.get("seasons")),
            apparent_fabric=a.get("apparent_fabric"),
            texture=a.get("texture"),
            formality=a.get("formality"),
            pattern=a.get("pattern"),
            fit_type=a.get("fit_type") or p.get("fit"),
            silhouette=a.get("silhouette"),
            neckline=construction.get("neckline"),
            sleeve_type=construction.get("sleeve_type"),
            length=construction.get("length"),
            coverage_level=a.get("coverage_level"),
            sheen=a.get("sheen"),
            rise=a.get("rise"),
            leg_shape=a.get("leg_shape"),
            stretch=a.get("stretch"),
            gemini_category_l1=a.get("category_l1"),
            gemini_category_l2=a.get("category_l2"),
        )
        # Compute derived fields
        _derive_fields(profile)
        return profile

    def to_api_dict(self) -> Dict[str, Any]:
        """Return aesthetic profile fields for API response."""
        return {
            "formality": self.formality,
            "color_family": self.color_family,
            "primary_color": self.primary_color,
            "pattern": self.pattern,
            "texture": self.texture,
            "apparent_fabric": self.apparent_fabric,
            "silhouette": self.silhouette,
            "coverage_level": self.coverage_level,
            "length": self.length,
            "fit_type": self.fit_type,
            "occasions": self.occasions[:5] if self.occasions else [],
            "seasons": self.seasons[:4] if self.seasons else [],
            "style_tags": self.style_tags[:5] if self.style_tags else [],
        }


def _to_list(val) -> List[str]:
    """Safely convert a value to a list of strings."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(v) for v in val if v]
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v]
        except (json.JSONDecodeError, TypeError):
            pass
        return [val] if val else []
    return []


# ---------------------------------------------------------------------------
# Derived field computation
# ---------------------------------------------------------------------------

_FORMALITY_TO_LEVEL = {
    "casual": 1, "smart casual": 2, "business casual": 3,
    "semi-formal": 4, "formal": 5,
}

_BRIDGE_L2 = {
    "blazer", "trench", "leather jacket", "denim jacket",
    "shirt", "button-down", "oxford", "t-shirt", "tee",
    "cardigan", "fine knit",
}

_BRIDGE_NAME_RE = re.compile(
    r'\b(blazer|trench|leather jacket|denim jacket|white sneaker|button.?down)\b',
    re.IGNORECASE,
)

_HIGH_STRENGTH_SIGNALS = {
    "sequin", "sequined", "beaded", "embellished", "corset", "bustier",
    "feather", "fringe", "rhinestone", "crystal",
}

_MID_STRENGTH_PATTERNS = {"floral", "animal", "leopard", "zebra", "tie-dye",
                          "graphic", "geometric", "abstract", "tropical"}

_TEXTURE_TO_INTENSITY = {
    "smooth": "smooth", "matte": "smooth",
    "ribbed": "medium", "pleated": "medium", "textured": "medium",
    "cable knit": "strong", "quilted": "strong", "terry": "strong",
    "waffle": "medium", "crochet": "strong",
}

_SHEEN_TO_SHINE = {
    "matte": "matte", "satin": "slight", "shimmer": "slight",
    "shiny": "shiny", "metallic": "shiny",
}

_FABRIC_WEIGHT_MAP = {
    "silk": "light", "chiffon": "light", "sheer": "light", "satin": "light",
    "crochet": "light", "linen": "light",
    "cotton": "mid", "denim": "mid", "synthetic_woven": "mid",
    "synthetic_stretch": "mid", "knit": "mid", "corduroy": "mid",
    "sequin": "mid", "velvet": "mid",
    "wool": "heavy", "leather": "heavy", "technical": "heavy",
}

_BASE_LAYER_L2 = {
    "tank top", "camisole", "bandeau", "tube top", "bodysuit",
    "t-shirt", "tee", "crop top", "bustier", "corset",
}
_MID_LAYER_L2 = {
    "hoodie", "sweatshirt", "sweater", "cardigan", "vest",
    "blazer", "shirt", "blouse",
}

_BRIGHT_SIGNALS = {"neon", "bright", "electric", "hot", "vivid", "fluorescent"}
_MUTED_SIGNALS = {"dusty", "muted", "pastel", "soft", "pale", "washed", "faded"}


def _derive_fields(p: "AestheticProfile") -> None:
    """Compute all derived fields from raw DB attributes."""
    # formality_level
    f = (p.formality or "").lower().strip()
    p.formality_level = _FORMALITY_TO_LEVEL.get(f, 2)
    if p.formality_level == 2 and f:
        for key, val in _FORMALITY_TO_LEVEL.items():
            if key in f:
                p.formality_level = val
                break

    # is_bridge
    l2 = (p.gemini_category_l2 or "").lower().strip()
    p.is_bridge = l2 in _BRIDGE_L2 or bool(_BRIDGE_NAME_RE.search(p.name or ""))

    # primary_style
    tags_lower = [t.lower().strip() for t in p.style_tags if t]
    p.primary_style = tags_lower[0] if tags_lower else None

    # style_strength (0-1)
    strength = 0.35  # default for plain items
    name_l = (p.name or "").lower()
    pat_l = (p.pattern or "").lower().strip()
    fab_l = (p.apparent_fabric or "").lower()
    if any(sig in name_l or sig in fab_l for sig in _HIGH_STRENGTH_SIGNALS):
        strength = 0.90
    elif pat_l in _MID_STRENGTH_PATTERNS:
        strength = 0.65
    elif any(t in {"glamorous", "edgy", "party", "statement"}
             for t in tags_lower):
        strength = 0.70
    elif any(t in {"casual", "minimalist"} for t in tags_lower):
        strength = 0.20
    elif pat_l in {"solid", "plain", "none", ""}:
        strength = 0.25
    p.style_strength = strength

    # material_family
    p.material_family = _get_fabric_family(p.apparent_fabric)

    # texture_intensity
    tex = (p.texture or "").lower().strip()
    p.texture_intensity = _TEXTURE_TO_INTENSITY.get(tex)
    if not p.texture_intensity and tex:
        p.texture_intensity = "medium"  # unknown texture = medium

    # shine_level
    sh = (p.sheen or "").lower().strip()
    p.shine_level = _SHEEN_TO_SHINE.get(sh, "matte" if sh else None)

    # fabric_weight
    if p.material_family:
        p.fabric_weight = _FABRIC_WEIGHT_MAP.get(p.material_family, "mid")

    # layer_role
    l1 = (p.gemini_category_l1 or "").lower().strip()
    if l1 == "outerwear":
        p.layer_role = "outer"
    elif l2 in _BASE_LAYER_L2:
        p.layer_role = "base"
    elif l2 in _MID_LAYER_L2:
        p.layer_role = "midlayer"
    else:
        p.layer_role = "base"  # default tops/bottoms/dresses = base

    # temp_band (from seasons)
    seasons_lower = {s.lower().strip() for s in p.seasons if s}
    if seasons_lower == {"summer"}:
        p.temp_band = "hot"
    elif seasons_lower == {"winter"}:
        p.temp_band = "cold"
    elif seasons_lower <= {"spring", "fall"} and seasons_lower:
        p.temp_band = "mild"
    elif len(seasons_lower) >= 3:
        p.temp_band = "any"
    elif "summer" in seasons_lower and "winter" not in seasons_lower:
        p.temp_band = "hot"
    elif "winter" in seasons_lower and "summer" not in seasons_lower:
        p.temp_band = "cold"
    else:
        p.temp_band = "mild" if seasons_lower else None

    # color_saturation
    color_str = (p.color_family or p.primary_color or "").lower()
    if any(s in color_str for s in _BRIGHT_SIGNALS):
        p.color_saturation = "bright"
    elif any(s in color_str for s in _MUTED_SIGNALS):
        p.color_saturation = "muted"
    elif color_str:
        p.color_saturation = "medium"


# =============================================================================
# 2. COLOR HARMONY
# =============================================================================

NEUTRAL_COLORS = {
    "black", "white", "cream", "grey", "gray", "beige", "camel", "taupe",
    "ivory", "charcoal", "navy", "browns", "neutrals", "off-white", "khaki",
    "nude", "tan", "oatmeal", "stone", "mushroom",
}

ANALOGOUS_GROUPS = [
    {"red", "orange", "coral", "salmon", "peach", "rust", "terracotta", "burnt orange"},
    {"orange", "yellow", "gold", "amber", "mustard", "honey", "saffron"},
    {"yellow", "green", "lime", "chartreuse", "olive"},
    {"green", "teal", "mint", "sage", "emerald", "forest", "jade", "khaki", "olive"},
    {"teal", "blue", "aqua", "turquoise", "cyan"},
    {"blue", "navy", "indigo", "cobalt", "sky blue", "powder blue", "denim", "steel blue"},
    {"indigo", "purple", "violet", "plum", "grape", "eggplant", "amethyst"},
    {"purple", "pink", "magenta", "fuchsia", "mauve", "lilac", "lavender", "orchid"},
    {"pink", "red", "rose", "blush", "dusty pink", "hot pink", "berry", "raspberry"},
]

COMPLEMENTARY_PAIRS = [
    ({"red", "berry", "burgundy", "wine", "crimson", "maroon"},
     {"green", "teal", "emerald", "forest", "sage", "olive"}),
    ({"blue", "navy", "cobalt", "indigo", "denim"},
     {"orange", "rust", "terracotta", "burnt orange", "amber"}),
    ({"purple", "plum", "violet", "eggplant", "amethyst"},
     {"yellow", "gold", "mustard", "amber", "honey"}),
    ({"pink", "blush", "rose", "dusty pink", "mauve"},
     {"green", "mint", "sage", "olive", "lime"}),
]

_TRUE_NEUTRALS = {
    "black", "white", "cream", "grey", "gray", "charcoal", "ivory",
    "off-white", "silver",
}
_WARM_NEUTRALS = {
    "brown", "camel", "taupe", "beige", "tan", "nude", "oatmeal",
    "stone", "mushroom", "khaki", "cognac", "chocolate", "rust",
    "terracotta", "browns", "copper",
}
_COOL_NEUTRALS = {"navy", "slate", "steel", "gunmetal"}
_COOL_SATURATED = {
    "purple", "violet", "plum", "grape", "eggplant", "amethyst",
    "fuchsia", "magenta", "lilac", "lavender", "orchid",
    "cobalt", "electric blue", "royal blue", "neon",
}

# Denim best-practice boost colors
_DENIM_BOOST_COLORS = {
    "white", "cream", "ivory", "off-white", "black", "grey", "gray",
    "charcoal", "navy", "camel", "beige", "tan", "cognac",
}


def _normalize_color(color: Optional[str]) -> str:
    if not color:
        return ""
    return color.lower().strip()


def _neutral_type(color: str) -> Optional[str]:
    c = _normalize_color(color)
    if not c:
        return None
    if any(n in c for n in _TRUE_NEUTRALS):
        return "true"
    if any(n in c for n in _WARM_NEUTRALS):
        return "warm"
    if any(n in c for n in _COOL_NEUTRALS):
        return "cool"
    return None


def _is_cool_saturated(color: str) -> bool:
    c = _normalize_color(color)
    return any(cs in c for cs in _COOL_SATURATED)


def _base_color_harmony(source_color: Optional[str], candidate_color: Optional[str]) -> float:
    """Temperature-aware color compatibility (base logic, no cross-dim). Returns 0.0-1.0."""
    sc = _normalize_color(source_color)
    cc = _normalize_color(candidate_color)
    if not sc or not cc:
        return 0.5

    sc_neutral = _neutral_type(sc)
    cc_neutral = _neutral_type(cc)

    # Both neutrals
    if sc_neutral and cc_neutral:
        if sc_neutral == cc_neutral:
            return 0.85
        if "true" in (sc_neutral, cc_neutral):
            return 0.80
        return 0.65

    # One neutral, one chromatic — neutrals always get minimum 0.75
    if sc_neutral or cc_neutral:
        neutral_type = sc_neutral if sc_neutral else cc_neutral
        chromatic_side = cc if sc_neutral else sc
        if neutral_type == "true":
            return 0.80
        if neutral_type == "warm":
            if _is_cool_saturated(chromatic_side):
                return 0.35
            for group in ANALOGOUS_GROUPS:
                if any(c in chromatic_side for c in group) and any(
                    c in chromatic_side
                    for c in {
                        "red", "orange", "coral", "salmon", "peach", "rust",
                        "terracotta", "gold", "amber", "mustard", "olive",
                        "sage", "green", "pink", "rose", "blush", "dusty pink",
                    }
                ):
                    return 0.75
            return 0.60
        if neutral_type == "cool":
            return 0.75

    # Both chromatic
    if sc == cc:
        return 0.85
    if sc in cc or cc in sc:
        return 0.80

    for group in ANALOGOUS_GROUPS:
        if any(c in sc for c in group) and any(c in cc for c in group):
            return 0.75

    for group_a, group_b in COMPLEMENTARY_PAIRS:
        sc_in_a = any(c in sc for c in group_a)
        cc_in_b = any(c in cc for c in group_b)
        sc_in_b = any(c in sc for c in group_b)
        cc_in_a = any(c in cc for c in group_a)
        if (sc_in_a and cc_in_b) or (sc_in_b and cc_in_a):
            return 0.70

    return 0.35


# =============================================================================
# 3. SCORING FUNCTIONS (8 dimensions — v2 rule-based)
# =============================================================================

# --- Weight dictionaries (8 dims, all sum to 1.0) ---
# Dimension keys: occasion_formality, style, fabric, silhouette,
#                 color, seasonality, pattern, price

CATEGORY_PAIR_WEIGHTS: Dict[Tuple[str, str], Dict[str, float]] = {
    ("tops", "bottoms"): {
        "occasion_formality": 0.20, "style": 0.14, "fabric": 0.14,
        "silhouette": 0.18, "color": 0.12, "seasonality": 0.10,
        "pattern": 0.06, "price": 0.06,
    },
    ("bottoms", "tops"): {
        "occasion_formality": 0.20, "style": 0.14, "fabric": 0.14,
        "silhouette": 0.18, "color": 0.12, "seasonality": 0.10,
        "pattern": 0.06, "price": 0.06,
    },
    ("dresses", "outerwear"): {
        "occasion_formality": 0.18, "style": 0.12, "fabric": 0.16,
        "silhouette": 0.12, "color": 0.12, "seasonality": 0.16,
        "pattern": 0.06, "price": 0.06,
    },
    ("outerwear", "tops"): {
        "occasion_formality": 0.18, "style": 0.14, "fabric": 0.14,
        "silhouette": 0.14, "color": 0.12, "seasonality": 0.14,
        "pattern": 0.06, "price": 0.08,
    },
    ("outerwear", "bottoms"): {
        "occasion_formality": 0.18, "style": 0.14, "fabric": 0.14,
        "silhouette": 0.14, "color": 0.12, "seasonality": 0.14,
        "pattern": 0.06, "price": 0.08,
    },
    ("outerwear", "dresses"): {
        "occasion_formality": 0.18, "style": 0.12, "fabric": 0.16,
        "silhouette": 0.12, "color": 0.12, "seasonality": 0.16,
        "pattern": 0.06, "price": 0.06,
    },
    ("tops", "outerwear"): {
        "occasion_formality": 0.18, "style": 0.14, "fabric": 0.14,
        "silhouette": 0.14, "color": 0.12, "seasonality": 0.14,
        "pattern": 0.06, "price": 0.08,
    },
    ("bottoms", "outerwear"): {
        "occasion_formality": 0.18, "style": 0.14, "fabric": 0.14,
        "silhouette": 0.14, "color": 0.12, "seasonality": 0.14,
        "pattern": 0.06, "price": 0.08,
    },
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "occasion_formality": 0.22, "style": 0.14, "fabric": 0.14,
    "silhouette": 0.16, "color": 0.12, "seasonality": 0.10,
    "pattern": 0.06, "price": 0.06,
}


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa = {s.lower().strip() for s in a if s}
    sb = {s.lower().strip() for s in b if s}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# DIM 1: Occasion & Formality (merged)
# ---------------------------------------------------------------------------

_OCCASION_CONFLICTS = [
    ({"workout", "gym", "exercise", "training"}, {"formal event", "wedding guest", "office", "work"}),
    ({"lounging"}, {"office", "work", "formal event", "wedding guest"}),
    ({"night out", "party", "club"}, {"workout", "gym", "exercise"}),
]

_DAY_OCCASIONS = {"everyday", "work", "office", "brunch", "school", "beach", "weekend"}
_NIGHT_OCCASIONS = {"night out", "party", "club", "date night", "formal event"}


def _score_occasion_formality(source: AestheticProfile, cand: AestheticProfile) -> float:
    """Combined occasion + formality scoring with conflict detection."""
    # Sub-score A: Formality distance ladder (weight 0.55)
    delta = abs(source.formality_level - cand.formality_level)
    if delta == 0:
        form_score = 1.0
    elif delta == 1:
        form_score = 0.85
    elif delta == 2:
        # Bridge rule: soften penalty if one item is a bridge piece
        form_score = 0.70 if (source.is_bridge or cand.is_bridge) else 0.55
    else:
        form_score = 0.10

    # Sub-score B: Occasion overlap + conflicts (weight 0.30)
    s_occ = {o.lower().strip() for o in source.occasions if o}
    c_occ = {o.lower().strip() for o in cand.occasions if o}
    if s_occ and c_occ:
        # Check hard conflicts first
        for group_a, group_b in _OCCASION_CONFLICTS:
            if (s_occ & group_a and c_occ & group_b) or (s_occ & group_b and c_occ & group_a):
                occ_score = 0.05
                break
        else:
            overlap = len(s_occ & c_occ)
            if overlap > 0:
                jaccard = overlap / len(s_occ | c_occ)
                occ_score = 0.4 + jaccard * 0.6
            else:
                # No overlap but no conflict — check if formality is close
                occ_score = 0.30 if delta <= 1 else 0.15
    elif not s_occ and not c_occ:
        occ_score = 0.50
    else:
        occ_score = 0.35

    # Sub-score C: Time context compatibility (weight 0.15)
    s_day = bool(s_occ & _DAY_OCCASIONS)
    s_night = bool(s_occ & _NIGHT_OCCASIONS)
    c_day = bool(c_occ & _DAY_OCCASIONS)
    c_night = bool(c_occ & _NIGHT_OCCASIONS)
    if (s_day and c_day) or (s_night and c_night):
        time_score = 1.0
    elif (s_day and c_night and not s_night) or (s_night and c_day and not c_night):
        time_score = 0.50
    else:
        time_score = 0.60  # unknown / mixed

    return 0.55 * form_score + 0.30 * occ_score + 0.15 * time_score


# ---------------------------------------------------------------------------
# DIM 2: Style (adjacency matrix + strength + bridge)
# ---------------------------------------------------------------------------

# 15 style tags from our DB, mapped to adjacency tiers:
# 1.0 = same, 0.75 = neighbor, 0.40 = weak neighbor, 0.10 = clash
_STYLE_NEIGHBORS: Dict[str, Set[str]] = {
    "casual":     {"minimalist", "streetwear", "sporty", "trendy", "chic"},
    "minimalist": {"casual", "classic", "chic", "modern"},
    "classic":    {"minimalist", "chic", "preppy", "modern"},
    "chic":       {"classic", "minimalist", "modern", "romantic", "glamorous"},
    "modern":     {"minimalist", "chic", "trendy", "classic"},
    "trendy":     {"modern", "streetwear", "casual", "edgy", "party"},
    "streetwear": {"casual", "trendy", "edgy", "sporty"},
    "edgy":       {"streetwear", "trendy", "glamorous", "party"},
    "romantic":   {"chic", "bohemian", "glamorous", "vintage"},
    "glamorous":  {"romantic", "chic", "party", "edgy"},
    "bohemian":   {"romantic", "vintage", "casual"},
    "sporty":     {"casual", "streetwear"},
    "party":      {"glamorous", "edgy", "trendy"},
    "vintage":    {"romantic", "bohemian", "classic", "preppy"},
    "preppy":     {"classic", "vintage", "chic"},
}

_STYLE_WEAK_NEIGHBORS: Dict[str, Set[str]] = {
    "casual":     {"classic", "bohemian", "vintage"},
    "minimalist": {"trendy"},
    "classic":    {"glamorous", "romantic"},
    "streetwear": {"bohemian"},
    "romantic":   {"classic", "party"},
    "glamorous":  {"classic", "modern"},
    "edgy":       {"sporty", "bohemian"},
    "sporty":     {"trendy", "edgy"},
    "trendy":     {"glamorous", "romantic"},
    "party":      {"casual", "streetwear"},
}

# Explicit clashes (score 0.10)
_STYLE_CLASHES: Dict[str, Set[str]] = {
    "romantic":   {"streetwear", "sporty"},
    "preppy":     {"edgy", "streetwear"},
    "sporty":     {"romantic", "glamorous", "preppy"},
    "glamorous":  {"sporty", "casual"},
    "bohemian":   {"preppy", "glamorous"},
}

# Style bridge items that reduce mismatch penalties
_STYLE_BRIDGE_TAGS = {"casual", "minimalist", "classic", "chic", "modern"}


def _style_adjacency(s1: str, s2: str) -> float:
    """Score two style tags using the adjacency matrix."""
    if s1 == s2:
        return 1.0
    if s2 in _STYLE_NEIGHBORS.get(s1, set()) or s1 in _STYLE_NEIGHBORS.get(s2, set()):
        return 0.75
    if s2 in _STYLE_WEAK_NEIGHBORS.get(s1, set()) or s1 in _STYLE_WEAK_NEIGHBORS.get(s2, set()):
        return 0.40
    if s2 in _STYLE_CLASHES.get(s1, set()) or s1 in _STYLE_CLASHES.get(s2, set()):
        return 0.10
    return 0.35  # unrelated but not clashing


def _score_style(source: AestheticProfile, cand: AestheticProfile) -> float:
    """Style compatibility: adjacency matrix + strength balance + bridge boost."""
    s_tags = [t.lower().strip() for t in source.style_tags if t]
    c_tags = [t.lower().strip() for t in cand.style_tags if t]
    if not s_tags or not c_tags:
        return 0.40  # unknown

    # Sub-score A: Best adjacency score between primary styles (weight 0.60)
    # Compare primary vs primary, and check secondary crossovers
    best_adj = 0.0
    for st in s_tags[:3]:
        for ct in c_tags[:3]:
            best_adj = max(best_adj, _style_adjacency(st, ct))
    adj_score = best_adj

    # Sub-score B: Style strength balance (weight 0.25)
    ss = source.style_strength
    cs = cand.style_strength
    if ss >= 0.7 and cs >= 0.7:
        strength_score = 0.30  # two statement pieces — too busy
    elif (ss >= 0.6 and cs <= 0.35) or (cs >= 0.6 and ss <= 0.35):
        strength_score = 0.90  # one statement + one supporting — ideal
    elif ss <= 0.3 and cs <= 0.3:
        strength_score = 0.70  # both basic — safe but bland
    else:
        strength_score = 0.65  # moderate mix

    # Sub-score C: Style bridge boost (weight 0.15)
    s_bridge = bool(set(s_tags) & _STYLE_BRIDGE_TAGS) or source.is_bridge
    c_bridge = bool(set(c_tags) & _STYLE_BRIDGE_TAGS) or cand.is_bridge
    if adj_score < 0.50 and (s_bridge or c_bridge):
        bridge_score = min(1.0, adj_score + 0.30)  # bridge softens clash
    else:
        bridge_score = adj_score  # no help needed

    return 0.60 * adj_score + 0.25 * strength_score + 0.15 * bridge_score


# ---------------------------------------------------------------------------
# DIM 3: Fabric (contrast + pair table + weight)
# ---------------------------------------------------------------------------

_FABRIC_FAMILIES: Dict[str, Set[str]] = {
    "cotton": {"cotton", "cotton blend", "100% cotton", "organic cotton", "cotton jersey",
               "cotton twill", "cotton poplin", "cotton sateen", "brushed cotton",
               "washed cotton", "cotton linen", "cotton silk"},
    "denim": {"denim", "chambray", "cotton denim", "stretch denim", "rigid denim", "raw denim"},
    "silk": {"silk", "silk blend", "100% silk", "silk satin", "silk chiffon",
             "silk crepe", "charmeuse", "silk georgette", "dupioni"},
    "linen": {"linen", "linen blend", "100% linen", "cotton linen", "ramie"},
    "wool": {"wool", "wool blend", "merino", "cashmere", "mohair", "alpaca",
             "virgin wool", "boiled wool", "felted wool", "tweed"},
    "knit": {"knit", "jersey", "rib knit", "sweater knit", "fine knit", "chunky knit",
             "cable knit", "waffle knit", "french terry", "terry", "bouclé", "boucle"},
    "synthetic_woven": {"polyester", "poly blend", "nylon", "rayon", "viscose", "modal",
                        "lyocell", "tencel", "cupro", "acetate", "acetate blend",
                        "rayon blend", "viscose blend"},
    "synthetic_stretch": {"spandex", "elastane", "lycra", "stretch", "scuba",
                          "neoprene", "ponte", "power mesh"},
    "sheer": {"chiffon", "organza", "tulle", "mesh", "lace", "net", "voile",
              "georgette", "sheer"},
    "leather": {"leather", "faux leather", "vegan leather", "pu leather", "pleather",
                "patent leather", "suede", "faux suede", "nubuck", "microsuede"},
    "satin": {"satin", "sateen", "charmeuse", "duchess satin"},
    "velvet": {"velvet", "velour", "crushed velvet", "stretch velvet"},
    "corduroy": {"corduroy", "cord", "wide wale", "fine wale"},
    "technical": {"technical", "performance", "moisture-wicking", "waterproof",
                  "windproof", "softshell", "hardshell", "gore-tex", "ripstop"},
    "crochet": {"crochet", "macramé", "macrame", "open knit"},
    "sequin": {"sequin", "sequined", "beaded", "embellished", "rhinestone"},
}

_FABRIC_TO_FAMILY: Dict[str, str] = {}
for _fam, _members in _FABRIC_FAMILIES.items():
    for _m in _members:
        _FABRIC_TO_FAMILY[_m] = _fam


def _get_fabric_family(fabric: Optional[str]) -> Optional[str]:
    if not fabric:
        return None
    f = fabric.lower().strip()
    if f in _FABRIC_TO_FAMILY:
        return _FABRIC_TO_FAMILY[f]
    for member, fam in _FABRIC_TO_FAMILY.items():
        if member in f or f in member:
            return fam
    return None


# Fabric family pair compatibility (full 16x16 matrix)
_FABRIC_FAMILY_COMPAT: Dict[Tuple[str, str], float] = {
    ("cotton", "denim"): 0.85, ("cotton", "silk"): 0.70, ("cotton", "linen"): 0.80,
    ("cotton", "wool"): 0.75, ("cotton", "knit"): 0.80, ("cotton", "synthetic_woven"): 0.75,
    ("cotton", "synthetic_stretch"): 0.70, ("cotton", "sheer"): 0.60,
    ("cotton", "leather"): 0.80, ("cotton", "satin"): 0.60, ("cotton", "velvet"): 0.55,
    ("cotton", "corduroy"): 0.80, ("cotton", "technical"): 0.55,
    ("cotton", "crochet"): 0.70, ("cotton", "sequin"): 0.45,
    ("denim", "silk"): 0.75, ("denim", "linen"): 0.70, ("denim", "wool"): 0.70,
    ("denim", "knit"): 0.85, ("denim", "synthetic_woven"): 0.70,
    ("denim", "synthetic_stretch"): 0.65, ("denim", "sheer"): 0.65,
    ("denim", "leather"): 0.80, ("denim", "satin"): 0.60, ("denim", "velvet"): 0.55,
    ("denim", "corduroy"): 0.60, ("denim", "technical"): 0.50,
    ("denim", "crochet"): 0.65, ("denim", "sequin"): 0.55,
    ("silk", "linen"): 0.65, ("silk", "wool"): 0.70, ("silk", "knit"): 0.55,
    ("silk", "synthetic_woven"): 0.65, ("silk", "synthetic_stretch"): 0.45,
    ("silk", "sheer"): 0.70, ("silk", "leather"): 0.70, ("silk", "satin"): 0.80,
    ("silk", "velvet"): 0.75, ("silk", "corduroy"): 0.40, ("silk", "technical"): 0.30,
    ("silk", "crochet"): 0.45, ("silk", "sequin"): 0.70,
    ("linen", "wool"): 0.45, ("linen", "knit"): 0.55, ("linen", "synthetic_woven"): 0.65,
    ("linen", "synthetic_stretch"): 0.50, ("linen", "sheer"): 0.65,
    ("linen", "leather"): 0.50, ("linen", "satin"): 0.45, ("linen", "velvet"): 0.30,
    ("linen", "corduroy"): 0.40, ("linen", "technical"): 0.50,
    ("linen", "crochet"): 0.70, ("linen", "sequin"): 0.30,
    ("wool", "knit"): 0.85, ("wool", "synthetic_woven"): 0.60,
    ("wool", "synthetic_stretch"): 0.50, ("wool", "sheer"): 0.40,
    ("wool", "leather"): 0.80, ("wool", "satin"): 0.55, ("wool", "velvet"): 0.70,
    ("wool", "corduroy"): 0.75, ("wool", "technical"): 0.55,
    ("wool", "crochet"): 0.60, ("wool", "sequin"): 0.40,
    ("knit", "synthetic_woven"): 0.65, ("knit", "synthetic_stretch"): 0.65,
    ("knit", "sheer"): 0.45, ("knit", "leather"): 0.75, ("knit", "satin"): 0.80,
    ("knit", "velvet"): 0.55, ("knit", "corduroy"): 0.70, ("knit", "technical"): 0.50,
    ("knit", "crochet"): 0.70, ("knit", "sequin"): 0.35,
    ("synthetic_woven", "synthetic_stretch"): 0.70, ("synthetic_woven", "sheer"): 0.65,
    ("synthetic_woven", "leather"): 0.70, ("synthetic_woven", "satin"): 0.70,
    ("synthetic_woven", "velvet"): 0.60, ("synthetic_woven", "corduroy"): 0.60,
    ("synthetic_woven", "technical"): 0.65, ("synthetic_woven", "crochet"): 0.50,
    ("synthetic_woven", "sequin"): 0.60,
    ("synthetic_stretch", "sheer"): 0.50, ("synthetic_stretch", "leather"): 0.65,
    ("synthetic_stretch", "satin"): 0.45, ("synthetic_stretch", "velvet"): 0.45,
    ("synthetic_stretch", "corduroy"): 0.40, ("synthetic_stretch", "technical"): 0.80,
    ("synthetic_stretch", "crochet"): 0.35, ("synthetic_stretch", "sequin"): 0.40,
    ("sheer", "leather"): 0.65, ("sheer", "satin"): 0.75, ("sheer", "velvet"): 0.65,
    ("sheer", "corduroy"): 0.35, ("sheer", "technical"): 0.30,
    ("sheer", "crochet"): 0.60, ("sheer", "sequin"): 0.65,
    ("leather", "satin"): 0.60, ("leather", "velvet"): 0.65,
    ("leather", "corduroy"): 0.60, ("leather", "technical"): 0.55,
    ("leather", "crochet"): 0.40, ("leather", "sequin"): 0.55,
    ("satin", "velvet"): 0.75, ("satin", "corduroy"): 0.35,
    ("satin", "technical"): 0.25, ("satin", "crochet"): 0.35, ("satin", "sequin"): 0.75,
    ("velvet", "corduroy"): 0.55, ("velvet", "technical"): 0.25,
    ("velvet", "crochet"): 0.40, ("velvet", "sequin"): 0.65,
    ("corduroy", "technical"): 0.35, ("corduroy", "crochet"): 0.50,
    ("corduroy", "sequin"): 0.30,
    ("technical", "crochet"): 0.20, ("technical", "sequin"): 0.20,
    ("crochet", "sequin"): 0.30,
}
# Same-family default
for _fam in _FABRIC_FAMILIES:
    _FABRIC_FAMILY_COMPAT[(_fam, _fam)] = 0.75

# Same-material penalties (denim+denim, leather+leather → bad default)
_SAME_MATERIAL_PENALTY = {"denim": 0.25, "leather": 0.25, "sequin": 0.30}


def _lookup_symmetric(
    table: Dict[Tuple[str, str], float], a: str, b: str, default: float = 0.55
) -> float:
    key = (a.lower().strip(), b.lower().strip())
    if key in table:
        return table[key]
    rev = (key[1], key[0])
    if rev in table:
        return table[rev]
    return default


def _score_fabric(source: AestheticProfile, cand: AestheticProfile) -> float:
    """Fabric compatibility: contrast principle + pair table + weight."""
    fam1 = source.material_family
    fam2 = cand.material_family

    # Sub-score A: Contrast principle (weight 0.40)
    contrast_score = 0.60  # neutral default
    ti1 = source.texture_intensity
    ti2 = cand.texture_intensity
    sh1 = source.shine_level
    sh2 = cand.shine_level
    if ti1 and ti2:
        if ti1 == "strong" and ti2 == "smooth":
            contrast_score = 0.85
        elif ti1 == "smooth" and ti2 == "strong":
            contrast_score = 0.85
        elif ti1 == "strong" and ti2 == "strong":
            contrast_score = 0.40
        elif ti1 == "smooth" and ti2 == "smooth":
            contrast_score = 0.70
        else:
            contrast_score = 0.65  # medium + anything
    if sh1 and sh2:
        if sh1 == "shiny" and sh2 == "matte":
            contrast_score = max(contrast_score, 0.80)
        elif sh1 == "matte" and sh2 == "shiny":
            contrast_score = max(contrast_score, 0.80)
        elif sh1 == "shiny" and sh2 == "shiny":
            contrast_score = min(contrast_score, 0.30)  # both shiny = bad

    # Sub-score B: Fabric pair table (weight 0.45)
    if fam1 and fam2:
        if fam1 == fam2 and fam1 in _SAME_MATERIAL_PENALTY:
            pair_score = _SAME_MATERIAL_PENALTY[fam1]
        else:
            pair_score = _lookup_symmetric(_FABRIC_FAMILY_COMPAT, fam1, fam2, 0.55)
    else:
        pair_score = 0.50

    # Sub-score C: Weight compatibility (weight 0.15)
    w1 = source.fabric_weight
    w2 = cand.fabric_weight
    if w1 and w2:
        if w1 == w2:
            weight_score = 0.80 if w1 == "mid" else 0.70 if w1 == "light" else 0.50
        elif {w1, w2} == {"light", "mid"} or {w1, w2} == {"mid", "heavy"}:
            weight_score = 0.75
        else:  # light + heavy
            weight_score = 0.45
    else:
        weight_score = 0.55

    return 0.40 * contrast_score + 0.45 * pair_score + 0.15 * weight_score


# ---------------------------------------------------------------------------
# DIM 4: Silhouette (category-aware balance rules)
# ---------------------------------------------------------------------------

_SILHOUETTE_COMPAT: Dict[Tuple[str, str], float] = {
    ("fitted", "relaxed"): 0.85, ("fitted", "oversized"): 0.85,
    ("fitted", "wide leg"): 0.90, ("fitted", "a-line"): 0.80,
    ("fitted", "flared"): 0.85, ("fitted", "straight"): 0.80,
    ("fitted", "regular"): 0.75, ("fitted", "slim"): 0.60,
    ("fitted", "bodycon"): 0.55, ("fitted", "fitted"): 0.55,
    ("slim", "relaxed"): 0.85, ("slim", "oversized"): 0.85,
    ("slim", "wide leg"): 0.85, ("slim", "a-line"): 0.80,
    ("slim", "flared"): 0.80, ("slim", "straight"): 0.75,
    ("slim", "regular"): 0.75, ("slim", "bodycon"): 0.55, ("slim", "slim"): 0.60,
    ("bodycon", "relaxed"): 0.80, ("bodycon", "oversized"): 0.80,
    ("bodycon", "wide leg"): 0.80, ("bodycon", "a-line"): 0.75,
    ("bodycon", "flared"): 0.75, ("bodycon", "straight"): 0.70,
    ("bodycon", "regular"): 0.70, ("bodycon", "bodycon"): 0.35,
    ("relaxed", "a-line"): 0.65, ("relaxed", "flared"): 0.60,
    ("relaxed", "wide leg"): 0.45, ("relaxed", "oversized"): 0.40,
    ("relaxed", "straight"): 0.70, ("relaxed", "regular"): 0.70,
    ("relaxed", "relaxed"): 0.45,
    ("oversized", "a-line"): 0.55, ("oversized", "flared"): 0.50,
    ("oversized", "wide leg"): 0.25, ("oversized", "straight"): 0.65,
    ("oversized", "regular"): 0.65, ("oversized", "oversized"): 0.25,
    ("a-line", "straight"): 0.70, ("a-line", "regular"): 0.70,
    ("a-line", "wide leg"): 0.50, ("a-line", "flared"): 0.50, ("a-line", "a-line"): 0.55,
    ("flared", "straight"): 0.70, ("flared", "regular"): 0.70,
    ("flared", "wide leg"): 0.45, ("flared", "flared"): 0.45,
    ("straight", "wide leg"): 0.60, ("straight", "regular"): 0.75,
    ("straight", "straight"): 0.65,
    ("wide leg", "regular"): 0.65, ("wide leg", "wide leg"): 0.35,
    ("regular", "regular"): 0.65,
}

_COVERAGE_COMPAT: Dict[Tuple[str, str], float] = {
    ("full", "full"): 0.75, ("full", "moderate"): 0.80,
    ("full", "partial"): 0.70, ("full", "minimal"): 0.55, ("full", "revealing"): 0.50,
    ("moderate", "moderate"): 0.80, ("moderate", "partial"): 0.75,
    ("moderate", "minimal"): 0.65, ("moderate", "revealing"): 0.55,
    ("partial", "partial"): 0.70, ("partial", "minimal"): 0.60,
    ("partial", "revealing"): 0.50,
    ("minimal", "minimal"): 0.50, ("minimal", "revealing"): 0.45,
    ("revealing", "revealing"): 0.30,
}

_LENGTH_BALANCE: Dict[Tuple[str, str], float] = {
    ("cropped", "regular"): 0.80, ("cropped", "midi"): 0.75,
    ("cropped", "maxi"): 0.70, ("cropped", "mini"): 0.70,
    ("cropped", "cropped"): 0.40, ("cropped", "floor-length"): 0.65,
    ("regular", "regular"): 0.75, ("regular", "midi"): 0.75,
    ("regular", "maxi"): 0.65, ("regular", "mini"): 0.75,
    ("regular", "floor-length"): 0.60,
    ("mini", "regular"): 0.75, ("mini", "midi"): 0.65, ("mini", "maxi"): 0.55,
    ("mini", "mini"): 0.40, ("mini", "floor-length"): 0.50,
    ("midi", "regular"): 0.75, ("midi", "midi"): 0.55, ("midi", "maxi"): 0.45,
    ("midi", "floor-length"): 0.40,
    ("maxi", "regular"): 0.65, ("maxi", "maxi"): 0.35,
    ("maxi", "floor-length"): 0.30,
    ("floor-length", "floor-length"): 0.25,
}

# Volume categories for the balance rule
_WIDE_BOTTOM_SILS = {"wide leg", "flared", "a-line"}
_FITTED_TOP_SILS = {"fitted", "slim", "bodycon"}
_SKINNY_BOTTOM_SILS = {"slim", "skinny", "fitted", "bodycon", "straight"}
_RELAXED_TOP_SILS = {"relaxed", "oversized", "regular"}


def _score_silhouette(source: AestheticProfile, cand: AestheticProfile) -> float:
    """Category-aware silhouette scoring with balance rules."""
    s_sil = (source.silhouette or "").strip().lower()
    c_sil = (cand.silhouette or "").strip().lower()
    s_len = (source.length or "").strip().lower()
    c_len = (cand.length or "").strip().lower()
    s_cov = (source.coverage_level or "").strip().lower()
    c_cov = (cand.coverage_level or "").strip().lower()
    s_rise = (cand.rise or "").strip().lower()   # candidate bottom rise

    s_broad = (source.broad_category or "").lower()
    c_broad = (cand.broad_category or "").lower()

    scores: List[float] = []
    weights: List[float] = []

    # --- Silhouette balance (weight 0.50) ---
    if s_sil and c_sil:
        # Category-aware overrides
        is_top_bottom = (s_broad == "tops" and c_broad == "bottoms") or \
                        (s_broad == "bottoms" and c_broad == "tops")
        is_outerwear = s_broad == "outerwear" or c_broad == "outerwear"

        if is_top_bottom:
            top_sil = s_sil if s_broad == "tops" else c_sil
            bot_sil = c_sil if s_broad == "tops" else s_sil
            top_len = s_len if s_broad == "tops" else c_len
            bot_rise = s_rise if s_broad == "bottoms" else (source.rise or "").strip().lower()

            # Balance rule: wide bottom → fitted top
            if bot_sil in _WIDE_BOTTOM_SILS and top_sil in _FITTED_TOP_SILS:
                sil_score = 0.90
            elif bot_sil in _SKINNY_BOTTOM_SILS and top_sil in _RELAXED_TOP_SILS:
                sil_score = 0.85
            elif top_sil == "oversized" and bot_sil in _WIDE_BOTTOM_SILS:
                # Oversized top + wide bottom = bad (unless streetwear)
                s_tags = {t.lower() for t in source.style_tags}
                c_tags = {t.lower() for t in cand.style_tags}
                if "streetwear" in s_tags or "streetwear" in c_tags:
                    sil_score = 0.60
                else:
                    sil_score = 0.25
            else:
                sil_score = _lookup_symmetric(_SILHOUETTE_COMPAT, top_sil, bot_sil, 0.55)

            # Waist logic: cropped top + high-rise = boost
            if top_len == "cropped" and bot_rise == "high":
                sil_score = max(sil_score, 0.90)
            elif top_len == "cropped" and bot_rise == "low":
                s_tags = {t.lower() for t in source.style_tags}
                c_tags = {t.lower() for t in cand.style_tags}
                if "trendy" in s_tags or "trendy" in c_tags:
                    sil_score = max(sil_score, 0.60)
                else:
                    sil_score = min(sil_score, 0.40)

        elif is_outerwear:
            outer_sil = s_sil if s_broad == "outerwear" else c_sil
            inner_sil = c_sil if s_broad == "outerwear" else s_sil
            outer_len = s_len if s_broad == "outerwear" else c_len

            # Outerwear balance rules
            if outer_len == "cropped" and inner_sil in _WIDE_BOTTOM_SILS:
                sil_score = 0.90
            elif outer_len in {"regular", "midi", "maxi"} and inner_sil in _SKINNY_BOTTOM_SILS:
                sil_score = 0.85
            elif outer_sil == "oversized" and inner_sil == "oversized":
                sil_score = 0.30
            else:
                sil_score = _lookup_symmetric(_SILHOUETTE_COMPAT, outer_sil, inner_sil, 0.55)
        else:
            sil_score = _lookup_symmetric(_SILHOUETTE_COMPAT, s_sil, c_sil, 0.55)

        scores.append(sil_score)
        weights.append(0.50)

    # --- Coverage compat (weight 0.20) ---
    if s_cov and c_cov:
        scores.append(_lookup_symmetric(_COVERAGE_COMPAT, s_cov, c_cov, 0.60))
        weights.append(0.20)

    # --- Length balance (weight 0.30) ---
    if s_len and c_len:
        scores.append(_lookup_symmetric(_LENGTH_BALANCE, s_len, c_len, 0.60))
        weights.append(0.30)

    if not scores:
        return 0.50
    total_w = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_w


# ---------------------------------------------------------------------------
# DIM 5: Color Harmony (upgraded with saturation + same-mat + denim rules)
# ---------------------------------------------------------------------------

def _score_color(source: AestheticProfile, cand: AestheticProfile) -> float:
    """Full color scoring: base harmony + saturation + same-material penalty + denim."""
    sc = source.color_family or source.primary_color
    cc = cand.color_family or cand.primary_color
    base = _base_color_harmony(sc, cc)

    # Saturation adjustment
    s_sat = source.color_saturation
    c_sat = cand.color_saturation
    if s_sat == "bright" and c_sat == "bright":
        sc_norm = _normalize_color(sc)
        cc_norm = _normalize_color(cc)
        # Two different bright colors fighting for attention
        if sc_norm != cc_norm and base < 0.70:
            base -= 0.10

    # Same-family + same-material penalty (denim-on-denim, leather-on-leather)
    sc_norm = _normalize_color(sc)
    cc_norm = _normalize_color(cc)
    fam1 = source.material_family
    fam2 = cand.material_family
    if fam1 and fam2 and fam1 == fam2 and fam1 in _SAME_MATERIAL_PENALTY:
        sc_fam = _normalize_color(source.color_family)
        cc_fam = _normalize_color(cand.color_family)
        if sc_fam and cc_fam:
            sc_nt = _neutral_type(sc_fam)
            cc_nt = _neutral_type(cc_fam)
            # Same neutral type + same material = bad (e.g. blue denim + blue denim)
            if not sc_nt and not cc_nt:
                # Both chromatic same-family same-material → strong penalty
                for group in ANALOGOUS_GROUPS:
                    if any(c in sc_fam for c in group) and any(c in cc_fam for c in group):
                        base = min(base, 0.30)
                        break

    # Denim best practice: blue denim + white/cream/black/grey/navy/camel = boost
    if fam1 == "denim" or fam2 == "denim":
        other_color = cc_norm if fam1 == "denim" else sc_norm
        if any(bc in other_color for bc in _DENIM_BOOST_COLORS):
            base = max(base, 0.80)

    return max(0.0, min(1.0, base))


# ---------------------------------------------------------------------------
# DIM 6: Seasonality (temp band + layer role + seasonal fabric)
# ---------------------------------------------------------------------------

_SUMMER_FABRICS = {"linen", "cotton", "sheer", "crochet", "satin", "silk"}
_WINTER_FABRICS = {"wool", "leather", "velvet", "corduroy", "technical", "knit"}

_LAYER_COMPAT: Dict[Tuple[str, str], float] = {
    ("base", "midlayer"): 0.85, ("base", "outer"): 0.80,
    ("midlayer", "outer"): 0.75, ("midlayer", "base"): 0.85,
    ("outer", "base"): 0.80, ("outer", "midlayer"): 0.75,
    ("base", "base"): 0.60, ("midlayer", "midlayer"): 0.50,
    ("outer", "outer"): 0.30,
}


def _score_seasonality(source: AestheticProfile, cand: AestheticProfile) -> float:
    """Seasonality: temperature + layering + seasonal fabric compatibility."""
    # Sub-score A: Temperature compatibility (weight 0.45)
    t1 = source.temp_band
    t2 = cand.temp_band
    if t1 and t2:
        if t1 == t2 or "any" in (t1, t2):
            temp_score = 1.0
        elif {t1, t2} in [{"hot", "mild"}, {"mild", "cold"}]:
            temp_score = 0.70
        elif {t1, t2} == {"hot", "cold"}:
            # Check if layering makes sense (outer + base = ok)
            roles = {source.layer_role, cand.layer_role}
            temp_score = 0.35 if "outer" in roles else 0.15
        else:
            temp_score = 0.60
    else:
        temp_score = 0.55  # unknown

    # Sub-score B: Layer role compatibility (weight 0.30)
    r1 = source.layer_role
    r2 = cand.layer_role
    if r1 and r2:
        layer_score = _LAYER_COMPAT.get((r1, r2), 0.55)
    else:
        layer_score = 0.55

    # Sub-score C: Seasonal fabric logic (weight 0.25)
    fam1 = source.material_family
    fam2 = cand.material_family
    fab_score = 0.55  # default
    if fam1 and fam2:
        s1_summer = fam1 in _SUMMER_FABRICS
        s2_summer = fam2 in _SUMMER_FABRICS
        s1_winter = fam1 in _WINTER_FABRICS
        s2_winter = fam2 in _WINTER_FABRICS
        if s1_summer and s2_summer:
            fab_score = 0.80
        elif s1_winter and s2_winter:
            fab_score = 0.80
        elif (s1_summer and s2_winter) or (s1_winter and s2_summer):
            # Cross-season fabric — bad unless layering
            roles = {source.layer_role, cand.layer_role}
            fab_score = 0.45 if "outer" in roles else 0.20
        else:
            fab_score = 0.65  # mid-weight fabrics are versatile

    return 0.45 * temp_score + 0.30 * layer_score + 0.25 * fab_score


# ---------------------------------------------------------------------------
# DIM 7: Pattern (kept from v1 — solid/subtle/bold)
# ---------------------------------------------------------------------------

def _pattern_compatibility(p1: Optional[str], p2: Optional[str]) -> float:
    if not p1 or not p2:
        return 0.5
    p1_l, p2_l = p1.lower().strip(), p2.lower().strip()
    solids = {"solid", "plain", "none"}
    subtle = {
        "pinstripe", "herringbone", "tweed", "textured", "tonal", "ribbed",
        "knit", "lace", "crochet", "cable knit", "quilted", "embroidered",
        "woven", "mesh", "broderie", "eyelet", "ruched",
    }
    bold = {
        "floral", "tropical", "graphic", "geometric", "abstract", "animal",
        "leopard", "zebra", "paisley", "tie-dye", "camouflage", "plaid",
        "checked", "polka dot", "striped", "gingham",
    }
    p1_solid = p1_l in solids
    p2_solid = p2_l in solids
    p1_subtle = p1_l in subtle
    p2_subtle = p2_l in subtle
    p1_bold = p1_l in bold or (not p1_solid and not p1_subtle)
    p2_bold = p2_l in bold or (not p2_solid and not p2_subtle)

    if p1_solid and p2_solid:
        return 0.8
    if p1_solid or p2_solid:
        return 0.85
    if p1_subtle and p2_subtle:
        return 0.65
    if p1_bold and p2_bold:
        return 0.45 if p1_l == p2_l else 0.2
    return 0.55


# ---------------------------------------------------------------------------
# DIM 8: Price (kept from v1)
# ---------------------------------------------------------------------------

def _price_coherence(price1: float, price2: float) -> float:
    if price1 <= 0 or price2 <= 0:
        return 0.5
    ratio = max(price1, price2) / min(price1, price2)
    if ratio <= 1.5:
        return 1.0
    elif ratio <= 2.0:
        return 0.85
    elif ratio <= 3.0:
        return 0.6
    elif ratio <= 4.0:
        return 0.4
    else:
        return 0.2


# ---------------------------------------------------------------------------
# MAIN SCORER + CROSS-DIMENSION GATES
# ---------------------------------------------------------------------------

def compute_compatibility_score(
    source: AestheticProfile, candidate: AestheticProfile,
) -> Tuple[float, Dict[str, float]]:
    """Compute 8-dimension compatibility score with cross-dimension gates.
    Returns (total, dim_scores).
    """
    pair_key = (
        (source.broad_category or "").lower(),
        (candidate.broad_category or "").lower(),
    )
    weights = CATEGORY_PAIR_WEIGHTS.get(pair_key, DEFAULT_WEIGHTS)
    dim_scores: Dict[str, float] = {}

    # 1. Occasion & Formality (merged)
    dim_scores["occasion_formality"] = _score_occasion_formality(source, candidate)

    # 2. Style
    dim_scores["style"] = _score_style(source, candidate)

    # 3. Fabric
    dim_scores["fabric"] = _score_fabric(source, candidate)

    # 4. Silhouette
    dim_scores["silhouette"] = _score_silhouette(source, candidate)

    # 5. Color
    dim_scores["color"] = _score_color(source, candidate)

    # 6. Seasonality
    dim_scores["seasonality"] = _score_seasonality(source, candidate)

    # 7. Pattern
    dim_scores["pattern"] = _pattern_compatibility(source.pattern, candidate.pattern)

    # 8. Price
    dim_scores["price"] = _price_coherence(source.price, candidate.price)

    # Weighted sum
    total = sum(weights.get(dim, 0) * score for dim, score in dim_scores.items())

    # --- Cross-dimension gates (post-hoc penalties) ---

    # Gate 1: Formality hard gate — if occasion_formality < 0.25, cap total
    if dim_scores["occasion_formality"] < 0.25:
        total = min(total, 0.35)

    # Gate 2: Color contribution halved when formality clashes
    if dim_scores["occasion_formality"] < 0.50:
        color_w = weights.get("color", 0)
        # Reduce effective color contribution by 50%
        excess = color_w * dim_scores["color"] * 0.50
        total = max(0.0, total - excess)

    # Gate 3: Oversized + oversized hard cap
    if dim_scores["silhouette"] < 0.30:
        total = min(total, 0.40)

    # Gate 4: Same-material same-color cross-penalty
    fam1 = source.material_family
    fam2 = candidate.material_family
    if fam1 and fam2 and fam1 == fam2 and fam1 in _SAME_MATERIAL_PENALTY:
        sc = _normalize_color(source.color_family)
        cc = _normalize_color(candidate.color_family)
        if sc and cc:
            sc_nt = _neutral_type(sc)
            cc_nt = _neutral_type(cc)
            if not sc_nt and not cc_nt:
                # Both chromatic + same material → extra penalty
                total = max(0.0, total - 0.12)

    return max(0.0, min(1.0, total)), dim_scores


# ---------------------------------------------------------------------------
# NOVELTY / CONTRAST SCORE  (used only for complement search, not similar)
# ---------------------------------------------------------------------------

def compute_novelty_score(
    source: AestheticProfile, candidate: AestheticProfile,
) -> float:
    """Reward candidates that bring something *new* to the outfit.

    High novelty = different color family, different fabric, different pattern
    tier, different visual feel.  This is NOT random difference — it's the
    kind of contrast that ChatGPT's rules describe as good outfit building
    (fabric contrast, color harmony through difference, visual interest).

    Returns 0.0-1.0 where 1.0 = maximally novel complement.
    """
    signals = []

    # 1. Color family difference (biggest driver of visual novelty)
    sc = _normalize_color(source.color_family)
    cc = _normalize_color(candidate.color_family)
    if sc and cc:
        if sc == cc:
            signals.append(0.1)   # same color = no novelty
        elif _neutral_type(sc) or _neutral_type(cc):
            signals.append(0.5)   # neutral + chromatic = moderate novelty
        else:
            # Check if same analogous group
            in_same_group = False
            for group in ANALOGOUS_GROUPS:
                if sc in group and cc in group:
                    in_same_group = True
                    break
            signals.append(0.55 if in_same_group else 0.85)
    else:
        signals.append(0.4)

    # 2. Fabric family difference
    sf = (source.material_family or "").lower()
    cf = (candidate.material_family or "").lower()
    if sf and cf:
        if sf == cf:
            signals.append(0.05)  # same fabric = no novelty
        else:
            # Different fabric families = good contrast
            signals.append(0.80)
    else:
        signals.append(0.4)

    # 3. Texture intensity contrast
    st = source.texture_intensity
    ct = candidate.texture_intensity
    if st and ct:
        if st == ct:
            signals.append(0.2)
        elif {st, ct} == {"smooth", "strong"}:
            signals.append(0.90)  # best contrast
        else:
            signals.append(0.55)
    else:
        signals.append(0.4)

    # 4. Pattern tier difference
    sp = (source.pattern or "").lower()
    cp = (candidate.pattern or "").lower()
    _SOLIDS = {"solid", "plain", "none", ""}
    sp_solid = sp in _SOLIDS
    cp_solid = cp in _SOLIDS
    if sp_solid and cp_solid:
        signals.append(0.15)  # both solid = no pattern interest
    elif sp_solid != cp_solid:
        signals.append(0.75)  # one print + one solid = good
    elif sp == cp:
        signals.append(0.10)  # same pattern = low novelty
    else:
        signals.append(0.50)  # different prints

    # 5. Silhouette contrast (volume difference = visual interest)
    _VOLUME = {"oversized": 3, "relaxed": 2, "regular": 1, "fitted": 0, "slim": 0, "bodycon": 0}
    sv = _VOLUME.get((source.silhouette or "").lower(), 1)
    cv = _VOLUME.get((candidate.silhouette or "").lower(), 1)
    vol_diff = abs(sv - cv)
    if vol_diff >= 2:
        signals.append(0.80)
    elif vol_diff == 1:
        signals.append(0.55)
    else:
        signals.append(0.15)

    return sum(signals) / len(signals) if signals else 0.4


# ---------------------------------------------------------------------------
# GREEDY DIVERSE SELECTION (MMR-style)
# ---------------------------------------------------------------------------

def _item_signature(entry: Dict) -> Dict[str, Optional[str]]:
    """Extract key visual attributes for diversity comparison."""
    p: AestheticProfile = entry["profile"]
    return {
        "color": _normalize_color(p.color_family) or _normalize_color(p.primary_color),
        "fabric": (p.material_family or "").lower() or None,
        "pattern": (p.pattern or "").lower() or None,
        "silhouette": (p.silhouette or "").lower() or None,
        "brand": (p.brand or "").lower() or None,
    }


def _sig_overlap(a: Dict[str, Optional[str]], b: Dict[str, Optional[str]]) -> float:
    """Fraction of shared attributes between two signatures (0-1).
    Higher = more similar = less diverse.
    """
    keys = ["color", "fabric", "pattern", "silhouette", "brand"]
    matches = 0
    compared = 0
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if va and vb:
            compared += 1
            if va == vb:
                matches += 1
    if compared == 0:
        return 0.0
    return matches / compared


def _diverse_select(
    scored: List[Dict],
    diversity_lambda: float = 0.25,
) -> List[Dict]:
    """Greedy MMR-style reranking for intra-result diversity.

    Picks items one-by-one.  For each slot after the first, the score is:
        adjusted = (1 - λ) * tattoo  -  λ * max_overlap_with_selected

    where max_overlap is the worst-case attribute overlap with any item
    already in the result list.  This pushes each next pick to be
    different in color, fabric, pattern, silhouette, and brand.

    With λ=0.40, a candidate with 0.65 tattoo but 0.0 overlap beats
    a candidate with 0.70 tattoo but 0.60 overlap:
      0.60*0.65 - 0.40*0.0 = 0.39  vs  0.60*0.70 - 0.40*0.60 = 0.18
    """
    if len(scored) <= 1:
        return scored

    selected: List[Dict] = []
    selected_sigs: List[Dict[str, Optional[str]]] = []
    remaining = list(scored)

    # First pick: highest tattoo score (no diversity penalty)
    selected.append(remaining.pop(0))
    selected_sigs.append(_item_signature(selected[0]))

    while remaining:
        best_idx = 0
        best_adjusted = -999.0

        for i, cand in enumerate(remaining):
            cand_sig = _item_signature(cand)
            # Max overlap with any already-selected item
            max_overlap = max(
                _sig_overlap(cand_sig, sel_sig)
                for sel_sig in selected_sigs
            )
            adjusted = (1.0 - diversity_lambda) * cand["tattoo"] - diversity_lambda * max_overlap
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_idx = i

        picked = remaining.pop(best_idx)
        selected.append(picked)
        selected_sigs.append(_item_signature(picked))

    return selected


# =============================================================================
# 4. CATEGORY LOGIC & STATE MACHINE
# =============================================================================

COMPLEMENTARY_CATEGORIES: Dict[str, List[str]] = {
    "tops": ["bottoms", "outerwear"],
    "bottoms": ["tops", "outerwear"],
    "dresses": ["outerwear"],
    "outerwear": ["tops", "bottoms", "dresses"],
}

BROAD_TO_CATEGORIES: Dict[str, List[str]] = {
    "tops": ["tops"], "bottoms": ["bottoms"],
    "dresses": ["dresses"], "outerwear": ["outerwear"],
}

_GEMINI_L1_TO_BROAD: Dict[str, str] = {
    "tops": "tops", "top": "tops",
    "bottoms": "bottoms", "bottom": "bottoms",
    "dresses": "dresses", "dress": "dresses",
    "outerwear": "outerwear",
    "jackets & coats": "outerwear", "coats & jackets": "outerwear",
    "activewear": "tops", "swimwear": "tops",
    "jumpsuits & rompers": "dresses", "jumpsuit": "dresses",
}

_SET_L2_KEYWORDS = {
    "co-ord", "coord", "set", "two-piece", "2-piece",
    "matching set", "coord set", "co-ord set",
}

_ACTIVEWEAR_L2_TO_BROAD: Dict[str, str] = {
    "sports bra": "tops", "sport bra": "tops", "bra": "tops",
    "short": "bottoms", "bike short": "bottoms", "legging": "bottoms",
    "leotard": "dresses", "bodysuit": "dresses",
}
_ACTIVEWEAR_OCCASIONS = {"workout", "gym", "exercise", "training"}
_ACTIVEWEAR_STYLE_TAGS = {"activewear", "athletic"}

# Tops L2 types that already function as an outer layer — skip outerwear recs
_OUTER_LAYER_TOP_L2 = {
    "hoodie", "sweatshirt", "sweater", "cardigan", "vest",
}

# Outerwear L2 types that are really waistcoats/vests — exclude from outerwear results
_OUTERWEAR_WAISTCOAT_L2 = {"vest", "hoodie"}

_NON_OUTFIT_L1 = {"intimates", "swimwear", "accessories", "shoes", "other"}
_NON_OUTFIT_L2 = {
    "pajama", "sleepwear", "nightgown", "robe", "underwear", "lingerie",
    "bra", "slip", "negligee", "bikini", "one-piece swimsuit", "surf suit",
    "swim brief", "swim trunk", "cover-up", "hat", "bag", "belt", "scarf",
    "glove", "beanie", "headband", "fascinator", "veil", "clutch", "pouch",
}

_SET_KEYWORDS_RE = re.compile(
    r'\b(co-?ord|coord|coordinates?|set\b|two[\s-]?piece|2[\s-]?piece)',
    re.IGNORECASE,
)
_SET_MULTI_PIECE_RE = re.compile(
    r'(?:&|\band\b|\+|\bwith\b)\s+(?:matching\s+)?'
    r'(?:wide\s+leg\s+|high\s+waist(?:ed)?\s+|straight\s+leg\s+)?'
    r'(?:maxi\s+|mini\s+|midi\s+)?'
    r'(?:trouser|pants?|shorts?|skirt|top|blouse|shirt|blazer|jacket|cardigan|dress|bag)\b',
    re.IGNORECASE,
)

_SIZE_RE = re.compile(
    r'\b(petite|plus\s*size?|tall|curve|curvy|regular|short|long|mini|midi|maxi)\b',
    re.IGNORECASE,
)
_CATEGORY_WORDS_RE = re.compile(
    r'\b(dress|top|blouse|shirt|skirt|pants?|trousers?|jeans?|shorts?'
    r'|jacket|coat|blazer|cardigan|sweater|hoodie|vest|bodysuit'
    r'|jumpsuit|romper|playsuit|overalls?|boilersuit)\b',
    re.IGNORECASE,
)

_SISTER_BRANDS: Dict[str, str] = {}
for _canonical, _aliases in [
    ("boohoo", [
        "boohoo", "nasty gal", "prettylittlething", "plt", "missguided",
        "karen millen", "coast", "oasis", "debenhams", "dorothy perkins",
        "wallis", "burton",
    ]),
]:
    for _alias in _aliases:
        _SISTER_BRANDS[_alias] = _canonical


def _gemini_broad(gemini_l1: Optional[str]) -> Optional[str]:
    if not gemini_l1:
        return None
    return _GEMINI_L1_TO_BROAD.get(gemini_l1.lower().strip())


def _is_non_outfit_product(profile: AestheticProfile) -> bool:
    l1 = (profile.gemini_category_l1 or "").lower().strip()
    l2 = (profile.gemini_category_l2 or "").lower().strip()
    return l1 in _NON_OUTFIT_L1 or l2 in _NON_OUTFIT_L2


def _is_set_product(name: str) -> bool:
    return bool(_SET_KEYWORDS_RE.search(name) or _SET_MULTI_PIECE_RE.search(name))


def _is_source_set(profile: AestheticProfile) -> bool:
    if _is_set_product(profile.name or ""):
        return True
    l2 = (profile.gemini_category_l2 or "").lower().strip()
    return l2 in _SET_L2_KEYWORDS


def _is_activewear_compatible(profile: AestheticProfile) -> bool:
    occ = {o.lower().strip() for o in (profile.occasions or []) if o}
    sty = {s.lower().strip() for s in (profile.style_tags or []) if s}
    return bool(occ & _ACTIVEWEAR_OCCASIONS or sty & _ACTIVEWEAR_STYLE_TAGS)


def _is_outer_layer_top(profile: AestheticProfile) -> bool:
    """True if this top already functions as an outer layer (hoodie, sweater, etc.)."""
    l2 = (profile.gemini_category_l2 or "").lower().strip()
    if l2 in _OUTER_LAYER_TOP_L2:
        return True
    # Fallback: check product name for hoodie/sweatshirt/sweater/cardigan
    name = (profile.name or "").lower()
    return bool(re.search(r'\b(hoodie|sweatshirt|sweater|cardigan)\b', name))


def get_complementary_targets(
    source_broad: str, source_profile: AestheticProfile,
) -> Tuple[List[str], str]:
    """Return (target_categories, status). Status: ok/set/activewear/blocked."""
    if _is_non_outfit_product(source_profile):
        return [], "blocked"
    if _is_source_set(source_profile):
        return ["outerwear"], "set"
    l1 = (source_profile.gemini_category_l1 or "").lower().strip()
    if l1 == "activewear":
        l2 = (source_profile.gemini_category_l2 or "").lower().strip()
        equiv = _ACTIVEWEAR_L2_TO_BROAD.get(l2)
        targets = COMPLEMENTARY_CATEGORIES.get(
            equiv, ["tops", "bottoms", "outerwear"]
        ) if equiv else ["tops", "bottoms", "outerwear"]
        return targets, "activewear"
    targets = list(COMPLEMENTARY_CATEGORIES.get(
        source_broad, ["tops", "bottoms", "outerwear"]
    ))
    # Tops that already function as outer layers don't need outerwear recs
    if source_broad == "tops" and _is_outer_layer_top(source_profile):
        targets = [t for t in targets if t != "outerwear"]
    return targets, "ok"


# =============================================================================
# 5. FILTERING & DEDUP
# =============================================================================

def _normalize_name(name: str) -> str:
    n = _SIZE_RE.sub("", name)
    return re.sub(r"\s+", " ", n).strip().lower()


def _canon_brand(brand: str) -> str:
    return _SISTER_BRANDS.get(brand.lower().strip(), brand.lower().strip())


def _name_core(name: str) -> str:
    n = _normalize_name(name)
    n = _CATEGORY_WORDS_RE.sub("", n)
    return re.sub(r"\s+", " ", n).strip()


def _name_word_overlap(n1: str, n2: str) -> float:
    w1 = set(n1.split()) - {"the", "a", "an", "in", "of", "and", "&", "-", "x", "by"}
    w2 = set(n2.split()) - {"the", "a", "an", "in", "of", "and", "&", "-", "x", "by"}
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def _filter_by_gemini_category(
    source: AestheticProfile,
    candidates: List[AestheticProfile],
    target_broad: str,
    similarity_ceiling: float = 0.92,
) -> List[AestheticProfile]:
    """Hard-gate candidates using Gemini category_l1."""
    source_gemini_broad = _gemini_broad(source.gemini_category_l1) or (source.category or "").lower()
    filtered = []
    for cand in candidates:
        if cand.product_id == source.product_id:
            continue
        if cand.similarity > similarity_ceiling:
            continue
        cand_broad = _gemini_broad(cand.gemini_category_l1)
        if cand_broad and cand_broad != target_broad:
            continue
        # Exclude waistcoats/vests/hoodies from outerwear results
        if target_broad == "outerwear":
            cand_l2 = (cand.gemini_category_l2 or "").lower().strip()
            if cand_l2 in _OUTERWEAR_WAISTCOAT_L2:
                continue
        filtered.append(cand)
    return filtered


def _deduplicate(candidates: List[AestheticProfile]) -> List[AestheticProfile]:
    """Remove cross-brand dupes, size variants, same-image dupes."""
    seen_keys: Set[str] = set()
    seen_images: Set[str] = set()
    deduped: List[AestheticProfile] = []
    for cand in sorted(candidates, key=lambda p: p.similarity, reverse=True):
        name_key = f"{_canon_brand(cand.brand or '')}|{_normalize_name(cand.name or '')}"
        img_key = (cand.image_url or "").strip()
        if name_key in seen_keys or (img_key and img_key in seen_images):
            continue
        deduped.append(cand)
        seen_keys.add(name_key)
        if img_key:
            seen_images.add(img_key)
    return deduped


def _remove_sets_and_non_outfit(
    source: AestheticProfile, candidates: List[AestheticProfile],
) -> List[AestheticProfile]:
    """Remove set/co-ord pieces, non-outfit items, and matchy-matchy."""
    src_core = _name_core(source.name or "")
    src_canon = _canon_brand(source.brand or "")
    src_color = (source.color_family or source.primary_color or "").lower()
    src_pattern = (source.pattern or "").lower()
    src_fabric = (source.apparent_fabric or "").lower()

    filtered = []
    for cand in candidates:
        if _is_non_outfit_product(cand):
            continue
        if _is_set_product(cand.name or ""):
            continue
        cand_canon = _canon_brand(cand.brand or "")
        cand_core = _name_core(cand.name or "")
        cand_color = (cand.color_family or cand.primary_color or "").lower()
        cand_pattern = (cand.pattern or "").lower()
        cand_fabric = (cand.apparent_fabric or "").lower()

        # Co-ord detection
        if src_canon == cand_canon and src_core and cand_core:
            overlap = _name_word_overlap(src_core, cand_core)
            if overlap >= 0.6:
                same_color = src_color and cand_color and src_color == cand_color
                same_pattern = src_pattern and cand_pattern and src_pattern == cand_pattern
                if same_color or same_pattern:
                    continue
        # Matchy-matchy
        if src_canon == cand_canon:
            if (src_color and cand_color and src_color == cand_color
                    and src_pattern and cand_pattern and src_pattern == cand_pattern
                    and src_fabric and cand_fabric and src_fabric == cand_fabric):
                continue
        filtered.append(cand)
    return filtered


def _filter_activewear(candidates: List[AestheticProfile]) -> List[AestheticProfile]:
    return [c for c in candidates if _is_activewear_compatible(c)]


# =============================================================================
# 6. OUTFIT ENGINE (production service)
# =============================================================================

# Attribute columns fetched from product_attributes
_ATTRS_SELECT = (
    "sku_id, category_l1, category_l2, "
    "occasions, style_tags, pattern, formality, fit_type, "
    "color_family, primary_color, secondary_colors, seasons, silhouette, "
    "construction, apparent_fabric, texture, coverage_level, "
    "sheen, rise, leg_shape, stretch"
)

_PRODUCT_SELECT = (
    "id, name, brand, category, broad_category, price, "
    "primary_image_url, base_color, colors, materials, style_tags, fit"
)


class OutfitEngine:
    """
    Production TATTOO outfit engine. Provides:
    - build_outfit(): Complete the Fit (cross-category complement scoring)
    - get_similar_scored(): Same-category similar items with 9-dim scoring
    """

    def __init__(self, supabase: Client):
        self.supabase = supabase
        self._clip_model = None
        self._clip_processor = None
        self._clip_lock = threading.Lock()

    # ------------------------------------------------------------------
    # FashionCLIP text encoder (lazy-loaded)
    # ------------------------------------------------------------------

    def _load_clip(self):
        """Lazy-load FashionCLIP for text prompt encoding (thread-safe)."""
        if self._clip_model is None:
            with self._clip_lock:
                if self._clip_model is None:
                    import torch
                    from transformers import CLIPProcessor, CLIPModel
                    logger.info("OutfitEngine: loading FashionCLIP model...")
                    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
                    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
                    if torch.cuda.is_available():
                        model = model.cuda()
                    model.eval()
                    self._clip_processor = processor
                    self._clip_model = model
                    logger.info("OutfitEngine: FashionCLIP model loaded")

    def _encode_text(self, text: str) -> str:
        """Encode text to a pgvector-compatible embedding string."""
        import torch
        self._load_clip()
        with torch.no_grad():
            inputs = self._clip_processor(
                text=[text], return_tensors="pt",
                padding=True, truncation=True, max_length=77,
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            emb = self._clip_model.get_text_features(**inputs)
            if isinstance(emb, torch.Tensor):
                pass
            elif hasattr(emb, "pooler_output") and emb.pooler_output is not None:
                emb = emb.pooler_output
            elif hasattr(emb, "text_embeds") and emb.text_embeds is not None:
                emb = emb.text_embeds
            emb = emb / emb.norm(dim=-1, keepdim=True)
            vec = emb.cpu().numpy().flatten().astype("float32").tolist()
        return f"[{','.join(map(str, vec))}]"

    # ------------------------------------------------------------------
    # Diverse prompt generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_complement_prompts(
        source: "AestheticProfile",
        source_broad: str,
        target_broad: str,
    ) -> List[str]:
        """
        Generate 3-5 diverse text prompts that describe complementary items
        for the source product in the target category.  Each prompt targets
        a different style / fabric / formality neighborhood so pgvector
        returns candidates from distinct regions of the embedding space.
        """
        src_color = source.primary_color or source.color_family or ""
        src_fabric = source.material_family or source.apparent_fabric or ""
        src_style = source.primary_style or ""
        src_formality = source.formality or ""
        src_pattern = source.pattern or ""
        src_name = (source.name or "").lower()

        # Target garment noun for prompts
        _TARGET_NOUNS = {
            "tops":      ["top", "blouse", "shirt", "sweater", "tee"],
            "bottoms":   ["trousers", "jeans", "skirt", "pants", "shorts"],
            "dresses":   ["dress", "midi dress", "mini dress", "maxi dress"],
            "outerwear": ["jacket", "blazer", "coat", "cardigan"],
        }
        nouns = _TARGET_NOUNS.get(target_broad, [target_broad])

        # Contrasting fabric families for texture play
        _FABRIC_CONTRAST = {
            "denim":  ["silk", "knit", "cotton", "leather"],
            "knit":   ["denim", "leather", "satin", "cotton"],
            "silk":   ["denim", "knit", "wool", "cotton"],
            "satin":  ["knit", "cotton", "wool", "denim"],
            "leather": ["silk", "chiffon", "knit", "cotton"],
            "linen":  ["silk", "denim", "knit", "cotton"],
            "wool":   ["silk", "satin", "cotton", "leather"],
            "cotton": ["silk", "leather", "satin", "denim"],
        }

        # Complementary color suggestions
        _COLOR_COMPLEMENTS = {
            "blue":   ["white", "camel", "black", "cream"],
            "black":  ["white", "red", "cream", "camel"],
            "white":  ["navy", "black", "beige", "denim blue"],
            "red":    ["black", "white", "navy", "cream"],
            "pink":   ["grey", "black", "navy", "cream"],
            "green":  ["white", "beige", "brown", "cream"],
            "navy":   ["white", "cream", "camel", "blush"],
            "brown":  ["cream", "white", "navy", "olive"],
            "beige":  ["navy", "black", "white", "burgundy"],
            "cream":  ["navy", "black", "brown", "burgundy"],
            "grey":   ["pink", "white", "black", "navy"],
        }

        # Style neighbors to suggest variety
        _STYLE_VARIETY = {
            "casual":     ["classic", "minimalist", "streetwear"],
            "classic":    ["minimalist", "chic", "preppy"],
            "minimalist": ["classic", "modern", "casual"],
            "trendy":     ["modern", "streetwear", "edgy"],
            "streetwear": ["casual", "edgy", "sporty"],
            "romantic":   ["chic", "bohemian", "classic"],
            "chic":       ["classic", "modern", "romantic"],
            "edgy":       ["streetwear", "trendy", "modern"],
            "bohemian":   ["romantic", "casual", "vintage"],
            "glamorous":  ["chic", "romantic", "party"],
            "sporty":     ["casual", "streetwear", "modern"],
        }

        prompts = []

        # --- Prompt 1: Neutral safe complement (always included) ---
        safe_noun = nouns[0]
        prompts.append(f"women's {safe_noun} to wear with {src_color} {src_fabric}".strip())

        # --- Prompt 2: Fabric contrast ---
        contrast_fabrics = _FABRIC_CONTRAST.get(
            (src_fabric or "").lower().split()[0] if src_fabric else "",
            ["cotton", "knit", "silk"],
        )
        prompts.append(
            f"women's {contrast_fabrics[0]} {nouns[1 % len(nouns)]} "
            f"{src_formality.lower() if src_formality else 'casual'}".strip()
        )

        # --- Prompt 3: Color complement ---
        color_key = (src_color or "").lower().split()[0] if src_color else ""
        comp_colors = _COLOR_COMPLEMENTS.get(color_key, ["white", "black", "cream"])
        prompts.append(
            f"{comp_colors[0]} {nouns[2 % len(nouns)]} "
            f"{contrast_fabrics[1] if len(contrast_fabrics) > 1 else 'cotton'}"
        )

        # --- Prompt 4: Style neighbor (different vibe) ---
        style_key = (src_style or "").lower()
        alt_styles = _STYLE_VARIETY.get(style_key, ["classic", "casual", "modern"])
        prompts.append(
            f"{alt_styles[0]} {nouns[3 % len(nouns)]} for women "
            f"{'solid' if src_pattern and src_pattern.lower() not in ('solid', 'plain', 'none', '') else 'print'}"
        )

        # --- Prompt 5: Elevated / dressy version (if source is casual) ---
        if source.formality_level <= 2:
            prompts.append(
                f"elegant {comp_colors[1] if len(comp_colors) > 1 else 'black'} "
                f"{nouns[0]} smart casual women"
            )
        # --- Prompt 5 alt: Relaxed version (if source is formal) ---
        elif source.formality_level >= 4:
            prompts.append(
                f"relaxed {contrast_fabrics[0]} {nouns[1 % len(nouns)]} casual women"
            )

        return prompts[:5]

    # ------------------------------------------------------------------
    # Text-based candidate retrieval
    # ------------------------------------------------------------------

    def _retrieve_text_candidates(
        self,
        prompts: List[str],
        target_category: str,
        per_prompt: int = 15,
    ) -> List[Dict]:
        """
        Run each prompt through FashionCLIP → text_search_products and
        merge results.  Deduplicates by product_id across prompts.
        """
        seen_ids: Set[str] = set()
        all_results: List[Dict] = []

        for prompt in prompts:
            try:
                vec_str = self._encode_text(prompt)
                result = self.supabase.rpc("text_search_products", {
                    "query_embedding": vec_str,
                    "match_count": per_prompt,
                    "match_offset": 0,
                    "filter_category": target_category,
                }).execute()
                for row in (result.data or []):
                    pid = str(row.get("product_id", ""))
                    if pid and pid not in seen_ids:
                        seen_ids.add(pid)
                        all_results.append(row)
            except Exception as e:
                logger.warning("Text search prompt failed (%s): %s", prompt[:40], e)
        return all_results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_outfit(
        self,
        product_id: str,
        items_per_category: int = 4,
        fusion_weight: float = 0.55,
        target_category: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Build a complete outfit from a source product.

        Carousel mode (default): returns top N items per complementary category.
        Feed mode: set target_category for paginated single-category results.

        Returns dict with: source_product, recommendations, status,
        scoring_info, complete_outfit.
        """
        # 1. Fetch source
        source = self._fetch_product_with_attrs(product_id)
        if not source:
            return {
                "error": f"Product {product_id} not found",
                "source_product": None,
                "recommendations": {},
            }

        # 2. Determine broad category from Gemini
        source_broad = _gemini_broad(source.gemini_category_l1) or (source.category or "").lower()
        source.broad_category = source_broad

        # 3. State machine
        all_targets, status = get_complementary_targets(source_broad, source)

        if status == "blocked":
            l1 = source.gemini_category_l1 or source.category or "unknown"
            return {
                "error": f"Complete the Fit is not available for {l1} products.",
                "source_product": self._format_source(source),
                "recommendations": {},
                "status": "blocked",
            }

        # Filter to requested category if in feed mode
        if target_category:
            target_broads = [target_category] if target_category in all_targets else [target_category]
        else:
            target_broads = all_targets

        effective_limit = limit if (target_category and limit) else items_per_category

        # 4. Retrieve, enrich, filter, score per category
        recommendations: Dict[str, Any] = {}
        all_top_picks = []

        for target_broad in target_broads:
            scored = self._score_category(
                source, source_broad, target_broad, status, fusion_weight,
            )

            # Pagination
            if target_category:
                page = scored[offset: offset + effective_limit]
                has_more = len(scored) > offset + effective_limit
            else:
                page = scored[:effective_limit]
                has_more = len(scored) > effective_limit

            items = []
            for i, entry in enumerate(page):
                rank = (offset + i + 1) if target_category else (i + 1)
                items.append(self._format_item(entry, rank))

            recommendations[target_broad] = {
                "items": items,
                "pagination": {
                    "offset": offset if target_category else 0,
                    "limit": effective_limit,
                    "returned": len(items),
                    "has_more": has_more,
                },
            }
            if items:
                all_top_picks.append(items[0])

        # 5. Build response
        source_fmt = self._format_source(source)
        total_price = source.price + sum(
            float(p.get("price", 0) or 0) for p in all_top_picks
        )

        return {
            "source_product": source_fmt,
            "recommendations": recommendations,
            "status": status,
            "scoring_info": {
                "dimensions": 8,
                "fusion": "0.70*compat + 0.08*novelty + 0.22*cosine",
                "engine": "tattoo_v2.1",
            },
            "complete_outfit": {
                "items": [product_id] + [p["product_id"] for p in all_top_picks],
                "total_price": round(total_price, 2),
                "item_count": 1 + len(all_top_picks),
            },
        }

    def get_similar_scored(
        self,
        product_id: str,
        limit: int = 20,
        offset: int = 0,
        fusion_weight: float = 0.55,
    ) -> Dict[str, Any]:
        """
        Same-category similar items with 9-dimension TATTOO scoring.

        Returns dict with: product_id, results, pagination.
        """
        source = self._fetch_product_with_attrs(product_id)
        if not source:
            return None  # Let caller handle 404

        source_broad = _gemini_broad(source.gemini_category_l1) or (source.category or "").lower()
        source.broad_category = source_broad

        # Retrieve same-category candidates via pgvector
        fetch_limit = offset + limit + 40  # Extra for dedup/filtering
        raw = self._retrieve_candidates(product_id, [source_broad], fetch_limit)
        if not raw:
            return {
                "product_id": product_id,
                "results": [],
                "pagination": {"offset": offset, "limit": limit, "returned": 0, "has_more": False},
            }

        # Enrich + filter (keep same category, dedup)
        profiles = self._enrich_candidates(raw)
        profiles = _filter_by_gemini_category(source, profiles, source_broad)
        profiles = _deduplicate(profiles)

        # Score
        scored = []
        for cand in profiles:
            cand.broad_category = _gemini_broad(cand.gemini_category_l1) or source_broad
            compat, dims = compute_compatibility_score(source, cand)
            tattoo = fusion_weight * compat + (1 - fusion_weight) * cand.similarity
            scored.append({"profile": cand, "compat": compat, "tattoo": tattoo,
                           "cosine": cand.similarity, "dims": dims})

        scored.sort(key=lambda x: x["tattoo"], reverse=True)

        # Paginate
        page = scored[offset: offset + limit]
        has_more = len(scored) > offset + limit

        results = []
        for i, entry in enumerate(page):
            results.append(self._format_similar_item(entry, offset + i + 1))

        return {
            "product_id": product_id,
            "results": results,
            "pagination": {
                "offset": offset,
                "limit": limit,
                "returned": len(results),
                "has_more": has_more,
            },
        }

    # ------------------------------------------------------------------
    # Internal: DB operations
    # ------------------------------------------------------------------

    def _fetch_product_with_attrs(self, product_id: str) -> Optional[AestheticProfile]:
        try:
            result = self.supabase.table("products").select(
                _PRODUCT_SELECT
            ).eq("id", product_id).limit(1).execute()
            if not result.data:
                return None
            product = result.data[0]

            attrs_result = self.supabase.table("product_attributes").select(
                _ATTRS_SELECT
            ).eq("sku_id", product_id).limit(1).execute()
            attrs = attrs_result.data[0] if attrs_result.data else {}
            return AestheticProfile.from_product_and_attrs(product, attrs)
        except Exception as e:
            logger.error("Failed to fetch product %s: %s", product_id, e)
            return None

    def _retrieve_candidates(
        self, source_id: str, target_categories: List[str], limit: int = 60,
    ) -> List[Dict]:
        """Retrieve candidates via pgvector similarity search.

        Uses get_similar_products_v2 RPC per target category.
        """
        all_results: List[Dict] = []
        per_cat = max(10, limit // len(target_categories)) if target_categories else limit
        for cat in target_categories:
            try:
                result = self.supabase.rpc("get_similar_products_v2", {
                    "source_product_id": source_id,
                    "match_count": per_cat,
                    "match_offset": 0,
                    "filter_category": cat,
                }).execute()
                all_results.extend(result.data or [])
            except Exception as e:
                logger.warning("pgvector retrieval failed for %s: %s", cat, e)
        return all_results

    def _enrich_candidates(self, candidates: List[Dict]) -> List[AestheticProfile]:
        """Batch-fetch Gemini attributes and build AestheticProfiles."""
        if not candidates:
            return []

        ids = [c.get("product_id") or c.get("id") for c in candidates if c.get("product_id") or c.get("id")]
        attrs_by_id: Dict[str, Dict] = {}
        for i in range(0, len(ids), 500):
            batch = ids[i: i + 500]
            try:
                result = self.supabase.table("product_attributes").select(
                    _ATTRS_SELECT
                ).in_("sku_id", batch).execute()
                for row in (result.data or []):
                    attrs_by_id[row["sku_id"]] = row
            except Exception as e:
                logger.warning("Attribute batch fetch failed: %s", e)

        profiles = []
        for c in candidates:
            pid = c.get("product_id") or c.get("id")
            product_row = {
                "id": pid, "name": c.get("name", ""), "brand": c.get("brand", ""),
                "category": c.get("category", ""), "broad_category": c.get("broad_category", ""),
                "price": c.get("price", 0), "primary_image_url": c.get("primary_image_url", ""),
                "base_color": c.get("base_color", ""), "style_tags": c.get("style_tags"),
                "fit": c.get("fit"),
            }
            attrs = attrs_by_id.get(pid, {})
            profile = AestheticProfile.from_product_and_attrs(product_row, attrs)
            profile.similarity = float(c.get("similarity", 0))
            profiles.append(profile)
        return profiles

    # ------------------------------------------------------------------
    # Internal: scoring pipeline for one category
    # ------------------------------------------------------------------

    def _score_category(
        self,
        source: AestheticProfile,
        source_broad: str,
        target_broad: str,
        status: str,
        fusion_weight: float,
    ) -> List[Dict]:
        """Retrieve, filter, score candidates for one target category.

        Uses two retrieval strategies merged together:
        1. Product-to-product pgvector similarity (original pool)
        2. FashionCLIP text prompts describing diverse complements (diversity pool)
        Deduplicates by product_id before scoring.
        """
        target_db_cats = BROAD_TO_CATEGORIES.get(target_broad, [target_broad])

        # --- Pool 1: product-to-product pgvector similarity ---
        raw = self._retrieve_candidates(source.product_id, target_db_cats, limit=60)

        # --- Pool 2: diverse text prompt retrieval ---
        try:
            prompts = self._generate_complement_prompts(source, source_broad, target_broad)
            text_raw = self._retrieve_text_candidates(
                prompts, target_category=target_db_cats[0], per_prompt=15,
            )
            # Merge: deduplicate by product_id
            existing_ids = {
                str(r.get("product_id") or r.get("id", ""))
                for r in raw
            }
            for row in text_raw:
                pid = str(row.get("product_id", ""))
                if pid and pid not in existing_ids:
                    existing_ids.add(pid)
                    raw.append(row)
            logger.info(
                "Diverse retrieval for %s: %d pgvector + %d text-prompt = %d merged",
                target_broad, len(raw) - len(text_raw), len(text_raw), len(raw),
            )
        except Exception as e:
            logger.warning("Text-prompt retrieval failed for %s, using pgvector only: %s", target_broad, e)

        if not raw:
            return []

        profiles = self._enrich_candidates(raw)
        profiles = _filter_by_gemini_category(source, profiles, target_broad)
        profiles = _deduplicate(profiles)
        profiles = _remove_sets_and_non_outfit(source, profiles)

        if status == "activewear":
            profiles = _filter_activewear(profiles)

        # --- Complement fusion ---
        # For outfit building the scoring favors TATTOO rules + novelty over
        # raw cosine similarity.  Cosine rewards items that *look like* the
        # source — the opposite of what we want for complementary outfits.
        #
        # Formula:  tattoo = 0.70 * compat + 0.15 * novelty + 0.15 * cosine
        #
        # The TATTOO dimensions already encode contrast rules (fabric contrast
        # principle, color harmony, silhouette balance) so giving them 70%
        # weight lets the fashion rules decide.  The novelty bonus (15%)
        # explicitly rewards items that bring something visually different
        # (different color, fabric, texture, pattern).  Cosine (15%) is kept
        # as a light tie-breaker for visual coherence.
        W_COMPAT = 0.70
        W_NOVELTY = 0.08
        W_COSINE = 0.22

        scored = []
        for cand in profiles:
            cand.broad_category = _gemini_broad(cand.gemini_category_l1) or target_broad
            compat, dims = compute_compatibility_score(source, cand)
            novelty = compute_novelty_score(source, cand)
            cosine = cand.similarity
            tattoo = W_COMPAT * compat + W_NOVELTY * novelty + W_COSINE * cosine
            scored.append({
                "profile": cand, "compat": compat, "tattoo": tattoo,
                "cosine": cosine, "novelty": novelty, "dims": dims,
            })

        scored.sort(key=lambda x: x["tattoo"], reverse=True)

        # --- Greedy diverse selection (MMR-style) ---
        # Ensures the final list has visual variety: each next pick must
        # be sufficiently different from items already selected.
        scored = _diverse_select(scored)

        return scored

    # ------------------------------------------------------------------
    # Internal: response formatting
    # ------------------------------------------------------------------

    def _format_source(self, source: AestheticProfile) -> Dict[str, Any]:
        return {
            "product_id": source.product_id,
            "name": source.name,
            "brand": source.brand,
            "category": source.gemini_category_l1 or source.category,
            "price": source.price,
            "image_url": source.image_url,
            "base_color": source.color_family or source.primary_color,
            "aesthetic_profile": source.to_api_dict(),
        }

    def _format_item(self, entry: Dict, rank: int) -> Dict[str, Any]:
        p = entry["profile"]
        result = {
            "product_id": p.product_id,
            "name": p.name,
            "brand": p.brand,
            "category": p.gemini_category_l1 or p.category,
            "price": p.price,
            "image_url": p.image_url,
            "base_color": p.color_family or p.primary_color,
            "rank": rank,
            "tattoo_score": round(entry["tattoo"], 4),
            "compatibility_score": round(entry["compat"], 4),
            "cosine_similarity": round(entry["cosine"], 4),
            "dimension_scores": {k: round(v, 4) for k, v in entry["dims"].items()},
        }
        if "novelty" in entry:
            result["novelty_score"] = round(entry["novelty"], 4)
        return result

    def _format_similar_item(self, entry: Dict, rank: int) -> Dict[str, Any]:
        p = entry["profile"]
        return {
            "product_id": p.product_id,
            "name": p.name,
            "brand": p.brand,
            "category": p.gemini_category_l1 or p.category,
            "price": p.price,
            "primary_image_url": p.image_url,
            "base_color": p.color_family or p.primary_color,
            "rank": rank,
            "similarity": round(entry["cosine"], 4),
            "tattoo_score": round(entry["tattoo"], 4),
            "compatibility_score": round(entry["compat"], 4),
            "dimension_scores": {k: round(v, 4) for k, v in entry["dims"].items()},
        }


# =============================================================================
# 7. SINGLETON
# =============================================================================

_engine: Optional[OutfitEngine] = None
_engine_lock = threading.Lock()


def get_outfit_engine() -> OutfitEngine:
    """Get or create OutfitEngine singleton (thread-safe)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from config.database import get_supabase_client
                _engine = OutfitEngine(get_supabase_client())
    return _engine
