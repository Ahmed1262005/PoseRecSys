"""
Translate follow-up filter patches from the LLM's look-focused schema
into the native HybridSearchRequest / plan attribute keys.

The LLM follow-up prompt uses a human-friendly vocabulary:
    product_types, dress_code, vibe, coverage{}, silhouette{},
    color_palette, pattern, fabric_vibe, layering, setting

This translator converts those into our canonical pipeline keys:
    category_l1, formality, style_tags, neckline, sleeve_type, length,
    fit_type, rise, colors, patterns, materials, modes, occasions
"""

from typing import Any, Dict, List

from core.logging import get_logger

logger = get_logger(__name__)


# =====================================================================
# Product type → category_l1
# =====================================================================
_PRODUCT_TYPE_MAP: Dict[str, List[str]] = {
    "dress": ["Dresses"],
    "dresses": ["Dresses"],
    "top": ["Tops"],
    "tops": ["Tops"],
    "bottom": ["Bottoms"],
    "bottoms": ["Bottoms"],
    "outerwear": ["Outerwear"],
    "set": ["Tops", "Bottoms"],  # Matching sets span categories
    "jumpsuit": ["Dresses"],  # Jumpsuits live under Dresses in our catalog
    "activewear": ["Activewear"],
    "swimwear": ["Swimwear"],
}


# =====================================================================
# Dress code → formality
# =====================================================================
_DRESS_CODE_MAP: Dict[str, str] = {
    "casual": "Casual",
    "smart_casual": "Smart Casual",
    "cocktail": "Semi-Formal",
    "formal": "Formal",
    "semi-formal": "Semi-Formal",
    "semi_formal": "Semi-Formal",
    "black_tie": "Formal",
    "business_casual": "Business Casual",
}


# =====================================================================
# Vibe → style_tags
# =====================================================================
_VIBE_MAP: Dict[str, str] = {
    "classic": "Classic",
    "romantic": "Romantic",
    "minimal": "Minimalist",
    "minimalist": "Minimalist",
    "trendy": "Modern",
    "edgy": "Edgy",
    "boho": "Bohemian",
    "bohemian": "Bohemian",
    "sporty": "Sporty",
    "glamorous": "Glamorous",
    "vintage": "Vintage",
    "preppy": "Preppy",
    "sexy": "Sexy",
    "western": "Western",
    "utility": "Utility",
    "streetwear": "Streetwear",
    "modern": "Modern",
}


# =====================================================================
# Coverage sub-fields → flat filter keys
# =====================================================================

# Neckline abstraction → concrete neckline values
_NECKLINE_MAP: Dict[str, List[str]] = {
    "high": ["Turtleneck", "Mock", "Crew", "Collared"],
    "mid": ["Crew", "Scoop", "Square", "Boat"],
    "low": ["V-Neck", "Deep V-Neck", "Plunging", "Sweetheart"],
    "strapless_ok": [],  # No filter — allow everything
}

# Sleeve abstraction → concrete sleeve_type values
_SLEEVE_MAP: Dict[str, List[str]] = {
    "long": ["Long", "3/4"],
    "short": ["Short", "Cap"],
    "sleeveless_ok": [],  # No filter — allow everything
}

# Hem abstraction → concrete length values
_HEM_MAP: Dict[str, List[str]] = {
    "mini_ok": [],  # No filter — allow everything
    "mini": ["Mini"],
    "midi": ["Midi"],
    "maxi": ["Maxi"],
    "floor": ["Floor-length"],
}


# =====================================================================
# Silhouette sub-fields → flat filter keys
# =====================================================================

_FIT_MAP: Dict[str, List[str]] = {
    "relaxed": ["Relaxed", "Loose"],
    "regular": ["Regular"],
    "fitted": ["Fitted", "Slim"],
    "oversized": ["Oversized"],
}

_RISE_MAP: Dict[str, List[str]] = {
    "high": ["High"],
    "mid": ["Mid"],
    "low": ["Low"],
}


# =====================================================================
# Color palette (abstract groups) → concrete colors from our taxonomy
# =====================================================================
_COLOR_PALETTE_MAP: Dict[str, List[str]] = {
    "neutrals": ["Black", "White", "Beige", "Cream", "Gray", "Taupe", "Off White"],
    "pastels": ["Pink", "Light Blue", "Cream", "Beige"],
    "jewel_tones": ["Burgundy", "Purple", "Green", "Navy Blue", "Blue"],
    "brights": ["Red", "Orange", "Yellow", "Pink", "Blue", "Green"],
    "dark": ["Black", "Navy Blue", "Burgundy", "Brown", "Olive"],
    "warm": ["Red", "Orange", "Yellow", "Brown", "Beige", "Burgundy"],
    "cool": ["Blue", "Navy Blue", "Purple", "Green", "Light Blue", "Gray"],
    "earth_tones": ["Brown", "Beige", "Olive", "Taupe", "Cream"],
}


# =====================================================================
# Pattern → patterns (simple rename + title-case)
# =====================================================================
_PATTERN_MAP: Dict[str, str] = {
    "solid": "Solid",
    "floral": "Floral",
    "stripe": "Striped",
    "striped": "Striped",
    "stripes": "Striped",
    "polka_dot": "Polka Dot",
    "polka dot": "Polka Dot",
    "abstract": "Abstract",
    "geometric": "Geometric",
    "animal_print": "Animal Print",
    "animal print": "Animal Print",
    "plaid": "Plaid",
    "tie_dye": "Tie Dye",
    "camo": "Camo",
    "colorblock": "Colorblock",
    "tropical": "Tropical",
    "no_preference": None,  # Drop — no filter
}


# =====================================================================
# Fabric vibe → materials (simple rename + title-case)
# =====================================================================
_FABRIC_MAP: Dict[str, str] = {
    "linen": "Linen",
    "cotton": "Cotton",
    "knit": "Knit",
    "satin": "Satin",
    "silk": "Silk",
    "denim": "Denim",
    "leather": "Faux Leather",
    "faux leather": "Faux Leather",
    "wool": "Wool",
    "velvet": "Velvet",
    "chiffon": "Chiffon",
    "lace": "Lace",
    "jersey": "Jersey",
    "fleece": "Fleece",
    "mesh": "Mesh",
    "no_preference": None,  # Drop
}


# =====================================================================
# Layering → modes
# =====================================================================
_LAYERING_MAP: Dict[str, List[str]] = {
    "light_layering": ["hot_weather"],
    "medium_layering": [],  # No mode needed — default
    "warm_layering": ["cold_weather"],
    "no_preference": [],
}


# =====================================================================
# Main translation function
# =====================================================================

def translate_follow_up_filters(raw_filters: Dict[str, Any]) -> Dict[str, Any]:
    """Translate an LLM follow-up filter patch into our native schema.

    Handles all keys from the look-focused follow-up prompt:
        product_types, dress_code, vibe, coverage, silhouette,
        color_palette, pattern, fabric_vibe, layering, setting

    Keys already in our native schema (category_l1, formality, etc.)
    are passed through unchanged.

    Returns a new dict with only our native keys.
    """
    if not raw_filters:
        return {}

    out: Dict[str, Any] = {}

    for key, value in raw_filters.items():
        # ----------------------------------------------------------
        # product_types → category_l1
        # ----------------------------------------------------------
        if key == "product_types":
            cats: List[str] = []
            seen = set()
            for pt in (value if isinstance(value, list) else [value]):
                for c in _PRODUCT_TYPE_MAP.get(str(pt).lower(), []):
                    if c not in seen:
                        cats.append(c)
                        seen.add(c)
            if cats:
                _merge_list(out, "category_l1", cats)

        # ----------------------------------------------------------
        # dress_code → formality
        # ----------------------------------------------------------
        elif key == "dress_code":
            mapped = _DRESS_CODE_MAP.get(str(value).lower())
            if mapped:
                _merge_list(out, "formality", [mapped])

        # ----------------------------------------------------------
        # vibe → style_tags
        # ----------------------------------------------------------
        elif key == "vibe":
            tags: List[str] = []
            for v in (value if isinstance(value, list) else [value]):
                mapped = _VIBE_MAP.get(str(v).lower())
                if mapped and mapped not in tags:
                    tags.append(mapped)
            if tags:
                _merge_list(out, "style_tags", tags)

        # ----------------------------------------------------------
        # coverage (nested) → neckline, sleeve_type, length, modes
        # ----------------------------------------------------------
        elif key == "coverage":
            if isinstance(value, dict):
                neck = value.get("neckline")
                if neck:
                    vals = _NECKLINE_MAP.get(str(neck).lower(), [])
                    if vals:
                        _merge_list(out, "neckline", vals)

                sleeves = value.get("sleeves")
                if sleeves:
                    vals = _SLEEVE_MAP.get(str(sleeves).lower(), [])
                    if vals:
                        _merge_list(out, "sleeve_type", vals)

                hem = value.get("hem")
                if hem:
                    vals = _HEM_MAP.get(str(hem).lower(), [])
                    if vals:
                        _merge_list(out, "length", vals)

        # ----------------------------------------------------------
        # silhouette (nested) → fit_type, rise
        # ----------------------------------------------------------
        elif key == "silhouette":
            if isinstance(value, dict):
                fit = value.get("fit")
                if fit:
                    vals = _FIT_MAP.get(str(fit).lower(), [])
                    if vals:
                        _merge_list(out, "fit_type", vals)

                rise = value.get("rise")
                if rise:
                    vals = _RISE_MAP.get(str(rise).lower(), [])
                    if vals:
                        _merge_list(out, "rise", vals)

        # ----------------------------------------------------------
        # color_palette → colors
        # ----------------------------------------------------------
        elif key == "color_palette":
            all_colors: List[str] = []
            seen_colors: set = set()
            for cp in (value if isinstance(value, list) else [value]):
                for c in _COLOR_PALETTE_MAP.get(str(cp).lower(), []):
                    if c not in seen_colors:
                        all_colors.append(c)
                        seen_colors.add(c)
            if all_colors:
                _merge_list(out, "colors", all_colors)

        # ----------------------------------------------------------
        # pattern → patterns
        # ----------------------------------------------------------
        elif key == "pattern":
            mapped_patterns: List[str] = []
            for p in (value if isinstance(value, list) else [value]):
                mapped = _PATTERN_MAP.get(str(p).lower())
                if mapped and mapped not in mapped_patterns:
                    mapped_patterns.append(mapped)
            if mapped_patterns:
                _merge_list(out, "patterns", mapped_patterns)

        # ----------------------------------------------------------
        # fabric_vibe → materials
        # ----------------------------------------------------------
        elif key == "fabric_vibe":
            mapped_mats: List[str] = []
            for fv in (value if isinstance(value, list) else [value]):
                mapped = _FABRIC_MAP.get(str(fv).lower())
                if mapped and mapped not in mapped_mats:
                    mapped_mats.append(mapped)
            if mapped_mats:
                _merge_list(out, "materials", mapped_mats)

        # ----------------------------------------------------------
        # layering → modes
        # ----------------------------------------------------------
        elif key == "layering":
            for lv in (value if isinstance(value, list) else [value]):
                modes = _LAYERING_MAP.get(str(lv).lower(), [])
                if modes:
                    _merge_list(out, "modes", modes)

        # ----------------------------------------------------------
        # Native keys — pass through unchanged
        # ----------------------------------------------------------
        elif key in _NATIVE_KEYS:
            out[key] = value

        # ----------------------------------------------------------
        # Unknown keys — log and pass through (safety net)
        # ----------------------------------------------------------
        else:
            logger.debug(
                "Unknown follow-up filter key, passing through",
                key=key, value=value,
            )
            out[key] = value

    return out


# Native keys that need no translation (already in our schema)
_NATIVE_KEYS = {
    "category_l1", "category_l2", "colors", "color_family",
    "formality", "fit_type", "neckline", "sleeve_type", "length",
    "rise", "silhouette", "materials", "patterns", "style_tags",
    "occasions", "seasons", "modes", "brands", "exclude_brands",
    "min_price", "max_price", "on_sale_only",
    "exclude_neckline", "exclude_sleeve_type", "exclude_length",
    "exclude_fit_type", "exclude_silhouette", "exclude_patterns",
    "exclude_colors", "exclude_materials", "exclude_occasions",
}


def _merge_list(out: Dict[str, Any], key: str, values: List[str]) -> None:
    """Merge values into an existing list in out, deduplicating."""
    existing = out.get(key, [])
    if not isinstance(existing, list):
        existing = [existing]
    seen = set(existing)
    for v in values:
        if v not in seen:
            existing.append(v)
            seen.add(v)
    out[key] = existing
