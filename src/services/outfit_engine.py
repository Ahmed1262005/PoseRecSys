"""
TATTOO-Inspired Outfit Engine
==============================

Production service for "Complete the Fit" outfit recommendations and
TATTOO-scored similar items. Uses 9-dimension aesthetic compatibility
scoring based on the TATTOO paper (arXiv:2509.23242).

Dimensions (6 TATTOO + 3 extensions):
  1. Style       - style_tags overlap
  2. Occasion    - occasions overlap
  3. Color       - temperature-aware color harmony
  4. Season      - season overlap
  5. Material    - fabric family + texture compatibility  (TATTOO dim 5)
  6. Balance     - silhouette contrast + coverage + length (TATTOO dim 6)
  7. Formality   - formality distance
  8. Pattern     - pattern clash detection
  9. Price       - price coherence ratio
"""

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from supabase import Client

logger = logging.getLogger(__name__)


# =============================================================================
# 1. AESTHETIC PROFILE
# =============================================================================

@dataclass
class AestheticProfile:
    """Structured aesthetic profile extracted from Gemini product_attributes."""

    # Core TATTOO dimensions
    color_family: Optional[str] = None
    primary_color: Optional[str] = None
    secondary_colors: List[str] = field(default_factory=list)
    style_tags: List[str] = field(default_factory=list)
    occasions: List[str] = field(default_factory=list)
    seasons: List[str] = field(default_factory=list)
    apparent_fabric: Optional[str] = None
    texture: Optional[str] = None
    formality: Optional[str] = None

    # Extended fashion dimensions
    pattern: Optional[str] = None
    fit_type: Optional[str] = None
    silhouette: Optional[str] = None
    neckline: Optional[str] = None
    sleeve_type: Optional[str] = None
    length: Optional[str] = None
    coverage_level: Optional[str] = None

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

        return cls(
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
            gemini_category_l1=a.get("category_l1"),
            gemini_category_l2=a.get("category_l2"),
        )

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


def score_color_harmony(source_color: Optional[str], candidate_color: Optional[str]) -> float:
    """Temperature-aware color compatibility scoring. Returns 0.0-1.0."""
    sc = _normalize_color(source_color)
    cc = _normalize_color(candidate_color)
    if not sc or not cc:
        return 0.5

    sc_neutral = _neutral_type(sc)
    cc_neutral = _neutral_type(cc)

    if sc_neutral and cc_neutral:
        if sc_neutral == cc_neutral:
            return 0.85
        if "true" in (sc_neutral, cc_neutral):
            return 0.80
        return 0.65

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
# 3. SCORING FUNCTIONS (9 dimensions)
# =============================================================================

FORMALITY_LEVELS = {
    "very casual": 0, "casual": 1, "smart casual": 2,
    "semi-formal": 3, "business casual": 3,
    "formal": 4, "very formal": 5, "black tie": 6,
}

# --- Weight dictionaries (9 dims, all sum to 1.0) ---

CATEGORY_PAIR_WEIGHTS: Dict[Tuple[str, str], Dict[str, float]] = {
    ("tops", "bottoms"): {
        "style": 0.18, "occasion": 0.14, "color": 0.14,
        "formality": 0.12, "season": 0.10, "material": 0.08, "balance": 0.08,
        "price": 0.08, "pattern": 0.08,
    },
    ("bottoms", "tops"): {
        "style": 0.18, "color": 0.16, "occasion": 0.14,
        "formality": 0.10, "season": 0.10, "material": 0.08, "balance": 0.08,
        "price": 0.08, "pattern": 0.08,
    },
    ("dresses", "outerwear"): {
        "style": 0.16, "season": 0.16, "color": 0.14,
        "material": 0.12, "formality": 0.10, "occasion": 0.10, "balance": 0.06,
        "price": 0.08, "pattern": 0.08,
    },
    ("outerwear", "tops"): {
        "style": 0.18, "season": 0.14, "color": 0.14,
        "material": 0.10, "formality": 0.10, "occasion": 0.10, "balance": 0.08,
        "price": 0.08, "pattern": 0.08,
    },
    ("outerwear", "bottoms"): {
        "style": 0.18, "season": 0.14, "color": 0.14,
        "formality": 0.10, "occasion": 0.10, "material": 0.10, "balance": 0.08,
        "price": 0.08, "pattern": 0.08,
    },
    ("outerwear", "dresses"): {
        "style": 0.16, "season": 0.16, "color": 0.14,
        "material": 0.12, "formality": 0.10, "occasion": 0.10, "balance": 0.06,
        "price": 0.08, "pattern": 0.08,
    },
    ("tops", "outerwear"): {
        "style": 0.18, "season": 0.14, "color": 0.14,
        "material": 0.10, "formality": 0.10, "occasion": 0.10, "balance": 0.08,
        "price": 0.08, "pattern": 0.08,
    },
    ("bottoms", "outerwear"): {
        "style": 0.18, "season": 0.14, "color": 0.14,
        "formality": 0.10, "occasion": 0.10, "material": 0.10, "balance": 0.08,
        "price": 0.08, "pattern": 0.08,
    },
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "style": 0.18, "occasion": 0.14, "color": 0.14,
    "formality": 0.10, "season": 0.10, "material": 0.10, "balance": 0.08,
    "price": 0.08, "pattern": 0.08,
}


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa = {s.lower().strip() for s in a if s}
    sb = {s.lower().strip() for s in b if s}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _formality_distance(f1: Optional[str], f2: Optional[str]) -> float:
    if not f1 or not f2:
        return 0.5
    level1 = FORMALITY_LEVELS.get(f1.lower().strip())
    level2 = FORMALITY_LEVELS.get(f2.lower().strip())
    if level1 is None:
        for key, val in FORMALITY_LEVELS.items():
            if key in f1.lower().strip():
                level1 = val
                break
    if level2 is None:
        for key, val in FORMALITY_LEVELS.items():
            if key in f2.lower().strip():
                level2 = val
                break
    if level1 is None or level2 is None:
        return 0.5
    distance = abs(level1 - level2)
    if distance == 0:
        return 1.0
    elif distance == 1:
        return 0.75
    elif distance == 2:
        return 0.45
    else:
        return 0.15


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


# --- Material / Texture (TATTOO dimension 5) ---

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


_TEXTURE_COMPAT: Dict[Tuple[str, str], float] = {
    ("smooth", "smooth"): 0.75, ("smooth", "textured"): 0.85,
    ("smooth", "ribbed"): 0.85, ("smooth", "pleated"): 0.80,
    ("smooth", "cable knit"): 0.80, ("smooth", "waffle"): 0.75,
    ("smooth", "quilted"): 0.75, ("smooth", "terry"): 0.65,
    ("textured", "textured"): 0.55, ("textured", "ribbed"): 0.50,
    ("textured", "pleated"): 0.45, ("textured", "cable knit"): 0.45,
    ("textured", "waffle"): 0.50, ("textured", "quilted"): 0.50,
    ("textured", "terry"): 0.45,
    ("ribbed", "ribbed"): 0.55, ("ribbed", "pleated"): 0.50,
    ("ribbed", "cable knit"): 0.60, ("ribbed", "waffle"): 0.60,
    ("ribbed", "quilted"): 0.55, ("ribbed", "terry"): 0.55,
    ("pleated", "pleated"): 0.40, ("pleated", "cable knit"): 0.45,
    ("pleated", "waffle"): 0.45, ("pleated", "quilted"): 0.40,
    ("pleated", "terry"): 0.40,
    ("cable knit", "cable knit"): 0.50, ("cable knit", "waffle"): 0.55,
    ("cable knit", "quilted"): 0.60, ("cable knit", "terry"): 0.50,
    ("waffle", "waffle"): 0.50, ("waffle", "quilted"): 0.55,
    ("waffle", "terry"): 0.55,
    ("quilted", "quilted"): 0.40, ("quilted", "terry"): 0.45,
    ("terry", "terry"): 0.55,
}

# Fabric family compatibility (major pairings only for brevity -- full matrix)
_FABRIC_FAMILY_COMPAT: Dict[Tuple[str, str], float] = {
    ("cotton", "denim"): 0.85, ("cotton", "silk"): 0.70, ("cotton", "linen"): 0.80,
    ("cotton", "wool"): 0.75, ("cotton", "knit"): 0.80, ("cotton", "synthetic_woven"): 0.75,
    ("cotton", "synthetic_stretch"): 0.70, ("cotton", "sheer"): 0.60,
    ("cotton", "leather"): 0.80, ("cotton", "satin"): 0.60, ("cotton", "velvet"): 0.55,
    ("cotton", "corduroy"): 0.80, ("cotton", "technical"): 0.55,
    ("cotton", "crochet"): 0.70, ("cotton", "sequin"): 0.45,
    ("denim", "silk"): 0.75, ("denim", "linen"): 0.70, ("denim", "wool"): 0.70,
    ("denim", "knit"): 0.80, ("denim", "synthetic_woven"): 0.70,
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
    ("knit", "sheer"): 0.45, ("knit", "leather"): 0.75, ("knit", "satin"): 0.45,
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
for _fam in _FABRIC_FAMILIES:
    _FABRIC_FAMILY_COMPAT[(_fam, _fam)] = 0.75


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


def _material_compatibility(source: AestheticProfile, candidate: AestheticProfile) -> float:
    """Score material/texture compatibility (TATTOO dim 5). Returns 0.0-1.0."""
    tex_score = None
    fab_score = None

    t1 = (source.texture or "").strip().lower()
    t2 = (candidate.texture or "").strip().lower()
    if t1 and t2:
        tex_score = _lookup_symmetric(_TEXTURE_COMPAT, t1, t2, 0.55)

    fam1 = _get_fabric_family(source.apparent_fabric)
    fam2 = _get_fabric_family(candidate.apparent_fabric)
    if fam1 and fam2:
        fab_score = _lookup_symmetric(_FABRIC_FAMILY_COMPAT, fam1, fam2, 0.55)

    if tex_score is not None and fab_score is not None:
        return 0.40 * tex_score + 0.60 * fab_score
    elif fab_score is not None:
        return fab_score
    elif tex_score is not None:
        return tex_score
    return 0.50


# --- Balance / Proportion (TATTOO dimension 6) ---

_SILHOUETTE_COMPAT: Dict[Tuple[str, str], float] = {
    ("fitted", "relaxed"): 0.85, ("fitted", "oversized"): 0.85,
    ("fitted", "wide leg"): 0.85, ("fitted", "a-line"): 0.80,
    ("fitted", "flared"): 0.80, ("fitted", "straight"): 0.80,
    ("fitted", "regular"): 0.75, ("fitted", "slim"): 0.60,
    ("fitted", "bodycon"): 0.55, ("fitted", "fitted"): 0.55,
    ("slim", "relaxed"): 0.85, ("slim", "oversized"): 0.85,
    ("slim", "wide leg"): 0.80, ("slim", "a-line"): 0.80,
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
    ("oversized", "wide leg"): 0.30, ("oversized", "straight"): 0.65,
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


def _balance_compatibility(source: AestheticProfile, candidate: AestheticProfile) -> float:
    """Score balance/proportion compatibility (TATTOO dim 6). Returns 0.0-1.0."""
    scores: List[float] = []
    weights: List[float] = []

    s1 = (source.silhouette or "").strip().lower()
    s2 = (candidate.silhouette or "").strip().lower()
    if s1 and s2:
        scores.append(_lookup_symmetric(_SILHOUETTE_COMPAT, s1, s2, 0.55))
        weights.append(0.45)

    c1 = (source.coverage_level or "").strip().lower()
    c2 = (candidate.coverage_level or "").strip().lower()
    if c1 and c2:
        scores.append(_lookup_symmetric(_COVERAGE_COMPAT, c1, c2, 0.60))
        weights.append(0.25)

    l1 = (source.length or "").strip().lower()
    l2 = (candidate.length or "").strip().lower()
    if l1 and l2:
        scores.append(_lookup_symmetric(_LENGTH_BALANCE, l1, l2, 0.60))
        weights.append(0.30)

    if not scores:
        return 0.50
    total_w = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_w


# --- Main scorer ---

def compute_compatibility_score(
    source: AestheticProfile, candidate: AestheticProfile,
) -> Tuple[float, Dict[str, float]]:
    """Compute 9-dimension TATTOO compatibility score. Returns (total, dim_scores)."""
    pair_key = (
        (source.broad_category or "").lower(),
        (candidate.broad_category or "").lower(),
    )
    weights = CATEGORY_PAIR_WEIGHTS.get(pair_key, DEFAULT_WEIGHTS)
    dim_scores: Dict[str, float] = {}

    # 1. Formality
    dim_scores["formality"] = _formality_distance(source.formality, candidate.formality)

    # 2. Occasion
    raw_occ = _jaccard(source.occasions, candidate.occasions)
    if raw_occ > 0:
        dim_scores["occasion"] = 0.4 + raw_occ * 0.6
    elif source.occasions and candidate.occasions:
        dim_scores["occasion"] = 0.15
    else:
        dim_scores["occasion"] = 0.35

    # 3. Color
    dim_scores["color"] = score_color_harmony(
        source.color_family or source.primary_color,
        candidate.color_family or candidate.primary_color,
    )

    # 4. Style
    raw_style = _jaccard(source.style_tags, candidate.style_tags)
    if raw_style > 0:
        dim_scores["style"] = 0.3 + raw_style * 0.7
    elif source.style_tags and candidate.style_tags:
        dim_scores["style"] = 0.10
    else:
        dim_scores["style"] = 0.35

    # 5. Season
    raw_season = _jaccard(source.seasons, candidate.seasons)
    if raw_season > 0:
        dim_scores["season"] = 0.4 + raw_season * 0.6
    elif source.seasons and candidate.seasons:
        dim_scores["season"] = 0.10
    elif not source.seasons and not candidate.seasons:
        dim_scores["season"] = 0.5
    else:
        dim_scores["season"] = 0.40

    # 6. Price
    dim_scores["price"] = _price_coherence(source.price, candidate.price)

    # 7. Pattern
    dim_scores["pattern"] = _pattern_compatibility(source.pattern, candidate.pattern)

    # 8. Material (TATTOO dim 5)
    dim_scores["material"] = _material_compatibility(source, candidate)

    # 9. Balance (TATTOO dim 6)
    dim_scores["balance"] = _balance_compatibility(source, candidate)

    total = sum(weights.get(dim, 0) * score for dim, score in dim_scores.items())
    return total, dim_scores


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
    "construction, apparent_fabric, texture, coverage_level"
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
                "dimensions": 9,
                "fusion_weight": fusion_weight,
                "engine": "tattoo_v1",
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
        """Retrieve candidates via pgvector RPC with fallback."""
        try:
            result = self.supabase.rpc("get_complementary_products", {
                "source_product_id": source_id,
                "target_categories": target_categories,
                "match_count": limit,
            }).execute()
            return result.data or []
        except Exception:
            # Fallback to per-category retrieval
            all_results = []
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
                except Exception as e2:
                    logger.warning("pgvector fallback failed for %s: %s", cat, e2)
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
        """Retrieve, filter, score candidates for one target category."""
        target_db_cats = BROAD_TO_CATEGORIES.get(target_broad, [target_broad])
        raw = self._retrieve_candidates(source.product_id, target_db_cats, limit=60)
        if not raw:
            return []

        profiles = self._enrich_candidates(raw)
        profiles = _filter_by_gemini_category(source, profiles, target_broad)
        profiles = _deduplicate(profiles)
        profiles = _remove_sets_and_non_outfit(source, profiles)

        if status == "activewear":
            profiles = _filter_activewear(profiles)

        scored = []
        for cand in profiles:
            cand.broad_category = _gemini_broad(cand.gemini_category_l1) or target_broad
            compat, dims = compute_compatibility_score(source, cand)
            tattoo = fusion_weight * compat + (1 - fusion_weight) * cand.similarity
            scored.append({
                "profile": cand, "compat": compat, "tattoo": tattoo,
                "cosine": cand.similarity, "dims": dims,
            })

        scored.sort(key=lambda x: x["tattoo"], reverse=True)
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
        return {
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
