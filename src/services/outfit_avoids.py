"""Outfit avoids system — hard filters and soft penalties for complete-the-fit.

Two-layer system:
1. Hard filters: Remove truly absurd combos before scoring (HF1-HF4).
2. Soft penalties: Additive negative adjustments in the fusion formula,
   capped at PENALTY_CAP total.  User style overrides reduce specific
   penalties to 30 % of their base value.

Integration point in outfit_engine._score_category():
    profiles = filter_hard_avoids(source, profiles, user_styles)
    ...
    avoid_adj, _ = compute_avoid_penalties(source, cand, user_styles)
    tattoo += avoid_adj
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

if TYPE_CHECKING:
    from services.outfit_engine import AestheticProfile

# ============================================================================
# L2 type-bucket constants
# ============================================================================

_SPORTY_CASUAL_L2: Set[str] = {
    "hoodie", "sweatshirt", "jogger", "joggers", "sweatpants", "fleece",
    "track pants", "track top", "graphic tee",
}

_FORMAL_EVENING_L2: Set[str] = {
    "cocktail dress", "evening dress", "gown", "formal jumpsuit",
    "formal dress", "ball gown",
}

_GYM_PIECE_L2: Set[str] = {
    "track pants", "windbreaker", "track jacket", "track top",
    "gym hoodie", "running shorts", "sports bra", "performance legging",
    "performance leggings",
}

_TAILORED_L2: Set[str] = {
    "blazer", "suit trousers", "pencil skirt", "tailored trousers",
    "dress pants", "suit jacket", "waistcoat",
}

_ACTIVEWEAR_L2: Set[str] = {
    "performance legging", "performance leggings", "sports bra",
    "running shorts", "bike shorts", "athletic top", "training top",
}

_CASUAL_OUTERWEAR_L2: Set[str] = {
    "denim jacket", "bomber", "bomber jacket", "utility jacket",
    "cargo jacket", "anorak",
}

_DELICATE_LAYER_L2: Set[str] = {
    "camisole", "bustier", "corset", "bralette", "slip dress",
}

_HEAVY_BASE_L2: Set[str] = {
    "hoodie", "sweatshirt", "chunky knit", "thick sweater", "fleece",
}

_SWEAT_L2: Set[str] = {
    "hoodie", "sweatshirt", "fleece", "jogger", "joggers", "sweatpants",
}

# Statement materials that look "too matchy" when repeated
_STATEMENT_MATERIALS: Set[str] = {"denim", "leather", "sequin"}

# Party / evening fabrics
_PARTY_MATERIALS: Set[str] = {"sequin"}
_PARTY_SHEENS: Set[str] = {"metallic", "shiny"}

# Competing statement prints
_STATEMENT_PRINTS: Set[str] = {
    "leopard", "zebra", "animal", "tropical", "tie-dye",
    "camouflage", "camo",
}

_NEON_SIGNALS: Set[str] = {"neon", "electric", "fluorescent", "hot pink", "lime"}

# ============================================================================
# Style-override mappings
# ============================================================================

# style-tag → set of rule IDs whose penalty is reduced to 30 %
_STYLE_OVERRIDES: Dict[str, Set[str]] = {
    "streetwear":    {"E1", "E3", "HF1"},
    "edgy":          {"F3", "B1", "HF2"},
    "punk":          {"F3", "B1", "HF2"},
    "grunge":        {"F3", "B1", "HF2"},
    "athleisure":    {"C1", "C2", "HF3"},
    "sporty chic":   {"C1", "C2", "HF3"},
    "avant-garde":   {"E1", "E2", "E3"},
    "experimental":  {"E1", "E2", "E3"},
    "eclectic":      {"J1", "B1", "HF1"},
    "maximalist":    {"J1", "B1", "HF1"},
    "monochrome":    {"D1"},
    "high-low":      {"B1", "HF1", "HF2"},
}

_OVERRIDE_MULTIPLIER: float = 0.3   # penalty × 0.3 when overridden

PENALTY_CAP: float = -0.25          # floor on total soft penalty

# ============================================================================
# Helpers — safe attribute access
# ============================================================================

def _l2(p: "AestheticProfile") -> str:
    return (getattr(p, "gemini_category_l2", None) or "").lower().strip()


def _style_set(p: "AestheticProfile") -> Set[str]:
    return {t.lower().strip() for t in (getattr(p, "style_tags", None) or []) if t}


def _occ_set(p: "AestheticProfile") -> Set[str]:
    return {o.lower().strip() for o in (getattr(p, "occasions", None) or []) if o}


# ============================================================================
# Type-bucket classifiers
# ============================================================================

def _is_formal_evening(p: "AestheticProfile") -> bool:
    if getattr(p, "formality_level", 2) >= 4:
        return True
    return _l2(p) in _FORMAL_EVENING_L2


def _is_sporty_casual(p: "AestheticProfile") -> bool:
    if _l2(p) in _SPORTY_CASUAL_L2:
        return True
    mat = getattr(p, "material_family", None)
    return mat == "technical" and bool(
        _style_set(p) & {"sporty", "athletic", "activewear"}
    )


def _is_party_fabric(p: "AestheticProfile") -> bool:
    mat = getattr(p, "material_family", None)
    if mat in _PARTY_MATERIALS:
        return True
    sheen = (getattr(p, "sheen", None) or "").lower()
    if sheen in _PARTY_SHEENS and getattr(p, "formality_level", 2) >= 3:
        return True
    return (
        getattr(p, "shine_level", None) == "shiny"
        and getattr(p, "formality_level", 2) >= 4
    )


def _is_tailored(p: "AestheticProfile") -> bool:
    if getattr(p, "formality_level", 2) >= 3 and _l2(p) in _TAILORED_L2:
        return True
    return (
        getattr(p, "formality_level", 2) >= 3
        and getattr(p, "material_family", None)
        in {"wool", "cotton", "synthetic_woven"}
    )


def _is_gym_piece(p: "AestheticProfile") -> bool:
    if _l2(p) in _GYM_PIECE_L2:
        return True
    return getattr(p, "material_family", None) == "technical" and bool(
        _style_set(p) & {"sporty", "athletic", "activewear"}
    )


def _is_activewear(p: "AestheticProfile") -> bool:
    return _l2(p) in _ACTIVEWEAR_L2


def _is_formal_piece(p: "AestheticProfile") -> bool:
    return (
        getattr(p, "formality_level", 2) >= 4
        and getattr(p, "material_family", None)
        in {"silk", "satin", "sequin", "velvet"}
    )


# ============================================================================
# Hard-filter rules (HF1-HF4)
# ============================================================================

def _get_override_rules(user_styles: Set[str]) -> Set[str]:
    """Collect rule IDs that are overridden by the user's style tags."""
    overridden: Set[str] = set()
    for style in user_styles:
        rules = _STYLE_OVERRIDES.get(style)
        if rules:
            overridden |= rules
    return overridden


def filter_hard_avoids(
    source: "AestheticProfile",
    candidates: List["AestheticProfile"],
    user_styles: Optional[Set[str]] = None,
) -> List["AestheticProfile"]:
    """Remove candidates that form absurd combos with *source*.

    Hard-filter rules (HF1-HF4) exclude clearly ridiculous pairings.
    User style overrides can skip specific rules.
    """
    styles = user_styles or set()
    overridden = _get_override_rules(styles)
    return [c for c in candidates if not _should_hard_filter(source, c, overridden)]


def _should_hard_filter(
    source: "AestheticProfile",
    candidate: "AestheticProfile",
    overridden: Set[str],
) -> bool:
    # HF1: Formal evening  ↔  sporty casual
    if "HF1" not in overridden:
        if _is_formal_evening(source) and _is_sporty_casual(candidate):
            return True
        if _is_sporty_casual(source) and _is_formal_evening(candidate):
            return True

    # HF2: Sporty casual  ↔  party fabric
    if "HF2" not in overridden:
        if _is_sporty_casual(source) and _is_party_fabric(candidate):
            return True
        if _is_party_fabric(source) and _is_sporty_casual(candidate):
            return True

    # HF3: Tailored  ↔  gym piece
    if "HF3" not in overridden:
        if _is_tailored(source) and _is_gym_piece(candidate):
            return True
        if _is_gym_piece(source) and _is_tailored(candidate):
            return True

    # HF4: Performance activewear  ↔  formal piece  (no override)
    if _is_activewear(source) and _is_formal_piece(candidate):
        return True
    if _is_formal_piece(source) and _is_activewear(candidate):
        return True

    return False


# ============================================================================
# Soft-penalty rule functions
# Each returns 0.0 (no penalty) or a negative float.
# ============================================================================

# --- A: Layering / warmth ------------------------------------------------

def _check_A1(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Light midlayer/outer over heavy base."""
    def _is_heavy_base(p: "AestheticProfile") -> bool:
        return getattr(p, "fabric_weight", None) == "heavy" or _l2(p) in _HEAVY_BASE_L2

    def _is_light_layer(p: "AestheticProfile") -> bool:
        return (
            getattr(p, "fabric_weight", None) == "light"
            and getattr(p, "layer_role", None) in ("midlayer", "outer")
        )

    if _is_heavy_base(src) and _is_light_layer(cand):
        return -0.10
    if _is_heavy_base(cand) and _is_light_layer(src):
        return -0.10
    return 0.0


def _check_A2(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Cold outerwear + hot-weather anchor."""
    if (
        getattr(src, "temp_band", None) == "hot"
        and getattr(cand, "temp_band", None) == "cold"
        and getattr(cand, "layer_role", None) == "outer"
    ):
        return -0.12
    if (
        getattr(cand, "temp_band", None) == "hot"
        and getattr(src, "temp_band", None) == "cold"
        and getattr(src, "layer_role", None) == "outer"
    ):
        return -0.12
    return 0.0


def _check_A3(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Hot-weather layer + cold-weather anchor."""
    if (
        getattr(src, "temp_band", None) == "cold"
        and getattr(cand, "temp_band", None) == "hot"
        and getattr(cand, "layer_role", None) in ("midlayer", "outer")
    ):
        return -0.08
    if (
        getattr(cand, "temp_band", None) == "cold"
        and getattr(src, "temp_band", None) == "hot"
        and getattr(src, "layer_role", None) in ("midlayer", "outer")
    ):
        return -0.08
    return 0.0


# --- B: Formality distance -----------------------------------------------

def _check_B1(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Formality distance ≥ 3  →  graduated penalty."""
    delta = abs(getattr(src, "formality_level", 2) - getattr(cand, "formality_level", 2))
    if delta >= 4:
        return -0.15
    if delta >= 3:
        return -0.08
    return 0.0


# --- C: Sporty ↔ tailored ------------------------------------------------

def _check_C1(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Technical/activewear material + tailored material."""
    sporty_mats = {"technical", "synthetic_stretch"}
    tailored_mats = {"wool", "silk", "cotton", "synthetic_woven"}

    def _sporty_mat(p: "AestheticProfile") -> bool:
        return (
            getattr(p, "material_family", None) in sporty_mats
            and bool(_style_set(p) & {"sporty", "athletic", "activewear"})
        )

    def _tailored_mat(p: "AestheticProfile") -> bool:
        return (
            getattr(p, "material_family", None) in tailored_mats
            and getattr(p, "formality_level", 2) >= 3
        )

    if _sporty_mat(src) and _tailored_mat(cand):
        return -0.10
    if _sporty_mat(cand) and _tailored_mat(src):
        return -0.10
    return 0.0


def _check_C2(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Sporty L2 type + formal L2 type."""
    sporty_l2s = {
        "track top", "windbreaker", "jogger", "joggers",
        "track pants", "track jacket",
    }
    formal_l2s = {
        "blazer", "pencil skirt", "dress shirt",
        "suit trousers", "tailored trousers",
    }
    if _l2(src) in sporty_l2s and _l2(cand) in formal_l2s:
        return -0.10
    if _l2(cand) in sporty_l2s and _l2(src) in formal_l2s:
        return -0.10
    return 0.0


# --- D: Same-material matchy ----------------------------------------------

def _check_D1(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Same statement material in both items."""
    fam_s = getattr(src, "material_family", None)
    fam_c = getattr(cand, "material_family", None)
    if not fam_s or not fam_c or fam_s != fam_c:
        return 0.0

    # Direct statement materials
    if fam_s in _STATEMENT_MATERIALS:
        return -0.10

    # Extended: wool or velvet with statement texture
    if fam_s in ("wool", "velvet"):
        statement_tex = {"tweed", "boucle", "textured", "cable knit"}
        src_tex = (getattr(src, "texture", None) or "").lower()
        cand_tex = (getattr(cand, "texture", None) or "").lower()
        if src_tex in statement_tex or cand_tex in statement_tex:
            return -0.10

    return 0.0


# --- E: Silhouette / proportion -------------------------------------------

_OVERSIZED_SILS: Set[str] = {"oversized", "relaxed", "baggy", "loose"}
_WIDE_BOTTOM_SILS: Set[str] = {"wide leg", "wide-leg", "flared", "palazzo"}


def _check_E1(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Both oversized/relaxed, neither is outerwear."""
    src_broad = (getattr(src, "broad_category", None) or "").lower()
    cand_broad = (getattr(cand, "broad_category", None) or "").lower()
    if src_broad == "outerwear" or cand_broad == "outerwear":
        return 0.0

    src_sil = (getattr(src, "silhouette", None) or "").lower()
    cand_sil = (getattr(cand, "silhouette", None) or "").lower()
    src_fit = (getattr(src, "fit_type", None) or "").lower()
    cand_fit = (getattr(cand, "fit_type", None) or "").lower()

    src_oversized = src_sil in _OVERSIZED_SILS or src_fit in _OVERSIZED_SILS
    cand_oversized = cand_sil in _OVERSIZED_SILS or cand_fit in _OVERSIZED_SILS

    if src_oversized and cand_oversized:
        return -0.08
    return 0.0


def _check_E2(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Cropped outer over longline bulky base."""
    def _cropped_outer_over_longline(outer: "AestheticProfile", inner: "AestheticProfile") -> bool:
        return (
            getattr(outer, "layer_role", None) == "outer"
            and (getattr(outer, "length", None) or "").lower() == "cropped"
            and getattr(inner, "fabric_weight", None) == "heavy"
            and (getattr(inner, "length", None) or "").lower() in {"long", "maxi", "midi"}
        )

    if _cropped_outer_over_longline(cand, src):
        return -0.08
    if _cropped_outer_over_longline(src, cand):
        return -0.08
    return 0.0


def _check_E3(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Bulky top + bulky wide-leg bottom."""
    src_broad = (getattr(src, "broad_category", None) or "").lower()
    cand_broad = (getattr(cand, "broad_category", None) or "").lower()

    def _bulky_top(p: "AestheticProfile") -> bool:
        return (
            (getattr(p, "silhouette", None) or "").lower() in _OVERSIZED_SILS
            and getattr(p, "fabric_weight", None) == "heavy"
        )

    def _wide_bottom(p: "AestheticProfile") -> bool:
        return (getattr(p, "silhouette", None) or "").lower() in _WIDE_BOTTOM_SILS

    if src_broad == "tops" and cand_broad == "bottoms":
        if _bulky_top(src) and _wide_bottom(cand):
            return -0.06
    if cand_broad == "tops" and src_broad == "bottoms":
        if _bulky_top(cand) and _wide_bottom(src):
            return -0.06
    return 0.0


# --- F: Texture / material clash ------------------------------------------

def _check_F1(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Party fabric + casual utilitarian."""
    casual_util_l2 = {
        "hoodie", "sweatshirt", "jogger", "joggers", "sweatpants", "fleece",
    }

    def _party(p: "AestheticProfile") -> bool:
        mat = getattr(p, "material_family", None)
        shine = getattr(p, "shine_level", None)
        sheen = (getattr(p, "sheen", None) or "").lower()
        return mat in _PARTY_MATERIALS or shine == "shiny" or sheen in _PARTY_SHEENS

    def _casual_util(p: "AestheticProfile") -> bool:
        return _l2(p) in casual_util_l2 or getattr(p, "material_family", None) == "technical"

    if _party(src) and _casual_util(cand):
        return -0.12
    if _party(cand) and _casual_util(src):
        return -0.12
    return 0.0


def _check_F2(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Satin/silk + fleece/sweats."""
    delicate_mats = {"satin", "silk"}
    if getattr(src, "material_family", None) in delicate_mats and _l2(cand) in _SWEAT_L2:
        return -0.10
    if getattr(cand, "material_family", None) in delicate_mats and _l2(src) in _SWEAT_L2:
        return -0.10
    return 0.0


def _check_F3(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Heavy distressed + delicate lingerie-coded."""
    def _is_distressed(p: "AestheticProfile") -> bool:
        name = (getattr(p, "name", None) or "").lower()
        tex = (getattr(p, "texture", None) or "").lower()
        return any(kw in name or kw in tex for kw in ("distressed", "destroyed", "ripped"))

    def _is_lingerie_coded(p: "AestheticProfile") -> bool:
        return (
            _l2(p) in _DELICATE_LAYER_L2
            and getattr(p, "material_family", None) in {"silk", "satin", "sheer"}
        )

    if _is_distressed(src) and _is_lingerie_coded(cand):
        return -0.08
    if _is_distressed(cand) and _is_lingerie_coded(src):
        return -0.08
    return 0.0


# --- G: Occasion mismatch ------------------------------------------------

def _check_G1(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Beach/summer occasion + heavy knit/wool."""
    beach_occs = {"beach", "vacation", "resort", "pool", "tropical"}
    heavy_mats = {"wool", "leather", "velvet"}

    if _occ_set(src) & beach_occs:
        if getattr(cand, "material_family", None) in heavy_mats and getattr(cand, "fabric_weight", None) == "heavy":
            return -0.10
    if _occ_set(cand) & beach_occs:
        if getattr(src, "material_family", None) in heavy_mats and getattr(src, "fabric_weight", None) == "heavy":
            return -0.10
    return 0.0


def _check_G2(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Office source + clubwear candidate."""
    office_occs = {"office", "work", "business", "professional"}

    if _occ_set(src) & office_occs:
        cov = (getattr(cand, "coverage_level", None) or "").lower()
        if cov in {"minimal", "revealing"} and getattr(cand, "formality_level", 2) <= 1:
            return -0.08
    if _occ_set(cand) & office_occs:
        cov = (getattr(src, "coverage_level", None) or "").lower()
        if cov in {"minimal", "revealing"} and getattr(src, "formality_level", 2) <= 1:
            return -0.08
    return 0.0


def _check_G3(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Formal source + casual outerwear candidate."""
    if getattr(src, "formality_level", 2) >= 4 and _l2(cand) in _CASUAL_OUTERWEAR_L2:
        return -0.10
    if getattr(cand, "formality_level", 2) >= 4 and _l2(src) in _CASUAL_OUTERWEAR_L2:
        return -0.10
    return 0.0


# --- I: Category-specific hard avoids ------------------------------------

def _check_I1(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Hoodie/sweatshirt → delicate/dressy candidate."""
    casual_base_l2 = {"hoodie", "sweatshirt"}

    def _is_delicate_dressy(p: "AestheticProfile") -> bool:
        return (
            getattr(p, "material_family", None) in {"silk", "satin", "sheer"}
            and getattr(p, "shine_level", None) in {"slight", "shiny"}
        )

    if _l2(src) in casual_base_l2 and _is_delicate_dressy(cand):
        return -0.10
    if _l2(cand) in casual_base_l2 and _is_delicate_dressy(src):
        return -0.10
    return 0.0


def _check_I2(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Blazer/tailored → gym-adjacent candidate."""
    tailored_l2 = {"blazer", "suit trousers", "tailored trousers", "pencil skirt"}
    gym_l2 = {"track top", "track jacket", "windbreaker", "gym hoodie"}

    def _is_gym_adj(p: "AestheticProfile") -> bool:
        if _l2(p) in gym_l2:
            return True
        return (
            getattr(p, "material_family", None) == "technical"
            and bool(_style_set(p) & {"sporty", "athletic"})
        )

    if _l2(src) in tailored_l2 and _is_gym_adj(cand):
        return -0.10
    if _l2(cand) in tailored_l2 and _is_gym_adj(src):
        return -0.10
    return 0.0


def _check_I3(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Formal dress → casual outerwear."""
    formal_dress_l2 = {
        "cocktail dress", "evening dress", "formal dress",
        "gown", "formal jumpsuit",
    }
    casual_outer_l2 = {"denim jacket", "hoodie", "bomber", "bomber jacket"}

    if _l2(src) in formal_dress_l2 and _l2(cand) in casual_outer_l2:
        return -0.10
    if _l2(cand) in formal_dress_l2 and _l2(src) in casual_outer_l2:
        return -0.10
    return 0.0


def _check_I4(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Performance leggings → formal tops."""
    perf_l2 = {
        "legging", "leggings", "performance legging", "performance leggings",
    }

    def _is_perf_legging(p: "AestheticProfile") -> bool:
        return (
            _l2(p) in perf_l2
            and getattr(p, "material_family", None) in {"synthetic_stretch", "technical"}
        )

    def _is_formal_top(p: "AestheticProfile") -> bool:
        return (
            getattr(p, "formality_level", 2) >= 4
            and (getattr(p, "broad_category", None) or "").lower() == "tops"
        )

    if _is_perf_legging(src) and _is_formal_top(cand):
        return -0.10
    if _is_perf_legging(cand) and _is_formal_top(src):
        return -0.10
    return 0.0


# --- J: Color / print clash -----------------------------------------------

def _check_J1(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Two competing statement prints."""
    src_pat = (getattr(src, "pattern", None) or "").lower().strip()
    cand_pat = (getattr(cand, "pattern", None) or "").lower().strip()
    if src_pat in _STATEMENT_PRINTS and cand_pat in _STATEMENT_PRINTS and src_pat != cand_pat:
        return -0.08
    return 0.0


def _check_J2(src: "AestheticProfile", cand: "AestheticProfile") -> float:
    """Neon + neon."""
    def _is_neon(p: "AestheticProfile") -> bool:
        color = (
            getattr(p, "color_family", None) or getattr(p, "primary_color", None) or ""
        ).lower()
        return (
            getattr(p, "color_saturation", None) == "bright"
            and any(sig in color for sig in _NEON_SIGNALS)
        )

    if _is_neon(src) and _is_neon(cand):
        return -0.06
    return 0.0


# ============================================================================
# Soft-rule registry
# ============================================================================

_SOFT_RULES: List[Tuple[str, Callable]] = [
    # A: Layering / warmth
    ("A1", _check_A1),
    ("A2", _check_A2),
    ("A3", _check_A3),
    # B: Formality distance
    ("B1", _check_B1),
    # C: Sporty ↔ tailored
    ("C1", _check_C1),
    ("C2", _check_C2),
    # D: Same-material matchy
    ("D1", _check_D1),
    # E: Silhouette / proportion
    ("E1", _check_E1),
    ("E2", _check_E2),
    ("E3", _check_E3),
    # F: Texture / material clash
    ("F1", _check_F1),
    ("F2", _check_F2),
    ("F3", _check_F3),
    # G: Occasion mismatch
    ("G1", _check_G1),
    ("G2", _check_G2),
    ("G3", _check_G3),
    # I: Category-specific
    ("I1", _check_I1),
    ("I2", _check_I2),
    ("I3", _check_I3),
    ("I4", _check_I4),
    # J: Color / print
    ("J1", _check_J1),
    ("J2", _check_J2),
]


# ============================================================================
# Public API
# ============================================================================

def compute_avoid_penalties(
    source: "AestheticProfile",
    candidate: "AestheticProfile",
    user_styles: Optional[Set[str]] = None,
) -> Tuple[float, List[str]]:
    """Return *(total_penalty, triggered_rule_ids)*.

    *total_penalty* is ≤ 0, capped at :data:`PENALTY_CAP`.
    """
    styles = user_styles or set()
    overridden = _get_override_rules(styles)

    total = 0.0
    triggered: List[str] = []

    for rule_id, check_fn in _SOFT_RULES:
        penalty = check_fn(source, candidate)
        if penalty < 0:
            if rule_id in overridden:
                penalty *= _OVERRIDE_MULTIPLIER
            total += penalty
            triggered.append(rule_id)

    return max(PENALTY_CAP, total), triggered
