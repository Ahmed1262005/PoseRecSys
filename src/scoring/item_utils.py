"""
Shared item-attribute helpers for scoring modules.

Both AgeScorer and WeatherScorer need to extract canonical article types,
broad categories, and coverage dimensions from item dicts.  These helpers
work with both search result dicts (from Algolia/pgvector) and
Candidate.to_scoring_dict() output from the feed pipeline.
"""

from typing import List, Optional, Set

from scoring.constants.age_coverage_tolerance import (
    ARTICLE_TYPE_COVERAGE,
    NECKLINE_COVERAGE,
    SLEEVE_TYPE_COVERAGE,
    STYLE_TAG_COVERAGE,
)


# ── Canonical article type aliases ────────────────────────────────
# Maps common variants / Gemini labels to our canonical keys used in
# the item-frequency and temperature-affinity tables.
_ARTICLE_TYPE_ALIASES: dict = {
    # Tops
    "t-shirt": "tshirt", "t shirt": "tshirt", "tee": "tshirt",
    "tank": "tank_top", "tank top": "tank_top",
    "camisole": "cami",
    "body suit": "bodysuit", "body-suit": "bodysuit",
    "sweatshirt": "sweatshirt", "hoodie": "hoodie",
    "pullover": "sweater", "jumper": "sweater", "knit": "sweater",
    "cardigan top": "cardigan",
    "crop top": "crop_top", "crop": "crop_top",
    "tube top": "tube_top", "bandeau top": "bandeau",
    "halter": "halter_top", "halter top": "halter_top",
    "sports bra": "sports_bra",
    "athletic top": "athletic_top",
    "polo shirt": "polo",
    "turtle neck": "turtleneck", "mock neck": "turtleneck",
    # Bottoms
    "jean": "jeans", "denim": "jeans",
    "trouser": "pants", "trousers": "pants", "slacks": "pants",
    "dress pants": "dress_pants", "dress_pants": "dress_pants",
    "wide leg": "wide_leg_pants", "wide-leg pants": "wide_leg_pants",
    "wide leg pants": "wide_leg_pants",
    "short": "shorts",
    "legging": "leggings", "yoga pants": "leggings",
    "jogger": "joggers", "sweatpant": "sweatpants",
    "athletic shorts": "athletic_shorts",
    "bike short": "bike_shorts", "bike shorts": "bike_shorts",
    "chino": "chinos", "linen pants": "linen_pants",
    "mini skirt": "mini_skirt", "midi skirt": "midi_skirt",
    "maxi skirt": "maxi_skirt", "pencil skirt": "pencil_skirt",
    "a-line skirt": "aline_skirt", "pleated skirt": "pleated_skirt",
    # One-piece
    "mini dress": "mini_dress", "midi dress": "midi_dress",
    "maxi dress": "maxi_dress", "cocktail dress": "cocktail_dress",
    "evening dress": "evening_dress", "gown": "gown",
    "sheath dress": "sheath_dress", "shift dress": "shift_dress",
    "wrap dress": "wrap_dress", "sun dress": "sundress",
    "slip dress": "slip_dress", "bodycon dress": "bodycon_dress",
    "a-line dress": "aline_dress", "shirt dress": "shirt_dress",
    "overall": "overalls", "swim suit": "swimsuit",
    "cover up": "coverup", "cover-up": "coverup",
    # Outerwear
    "trench": "trench_coat", "trench coat": "trench_coat",
    "puffer jacket": "puffer", "puffer coat": "puffer",
    "wind breaker": "windbreaker",
    "leather jacket": "jacket", "denim jacket": "jacket",
    "bomber jacket": "jacket", "varsity jacket": "jacket",
    "structured jacket": "structured_jacket",
    "evening jacket": "evening_jacket",
    "linen jacket": "linen_jacket",
}

# Broad category inference from canonical article type
_TYPE_TO_BROAD: dict = {
    # Tops
    "tank_top": "tops", "cami": "tops", "tshirt": "tops",
    "blouse": "tops", "tube_top": "tops", "sweater": "tops",
    "cardigan": "tops", "bodysuit": "tops", "hoodie": "tops",
    "sweatshirt": "tops", "crop_top": "tops", "halter_top": "tops",
    "polo": "tops", "turtleneck": "tops", "bralette": "tops",
    "bandeau": "tops", "sports_bra": "tops", "athletic_top": "tops",
    "silk_top": "tops", "sequin_top": "tops", "shell": "tops",
    # Bottoms
    "jeans": "bottoms", "pants": "bottoms", "dress_pants": "bottoms",
    "wide_leg_pants": "bottoms", "shorts": "bottoms",
    "athletic_shorts": "bottoms", "bike_shorts": "bottoms",
    "leggings": "bottoms", "joggers": "bottoms", "sweatpants": "bottoms",
    "chinos": "bottoms", "linen_pants": "bottoms",
    "skirt": "bottoms", "mini_skirt": "bottoms", "midi_skirt": "bottoms",
    "maxi_skirt": "bottoms", "pencil_skirt": "bottoms",
    "aline_skirt": "bottoms", "pleated_skirt": "bottoms",
    "wrap_skirt": "bottoms", "sarong": "bottoms",
    # Dresses / one-piece
    "dress": "dresses", "mini_dress": "dresses", "midi_dress": "dresses",
    "maxi_dress": "dresses", "cocktail_dress": "dresses",
    "evening_dress": "dresses", "gown": "dresses",
    "sheath_dress": "dresses", "shift_dress": "dresses",
    "wrap_dress": "dresses", "sundress": "dresses",
    "slip_dress": "dresses", "bodycon_dress": "dresses",
    "aline_dress": "dresses", "shirt_dress": "dresses",
    "jumpsuit": "dresses", "romper": "dresses", "overalls": "dresses",
    "swimsuit": "dresses", "bikini": "dresses",
    "coverup": "dresses", "kaftan": "dresses",
    # Outerwear
    "blazer": "outerwear", "jacket": "outerwear", "coat": "outerwear",
    "trench_coat": "outerwear", "puffer": "outerwear", "vest": "outerwear",
    "windbreaker": "outerwear", "athletic_jacket": "outerwear",
    "linen_jacket": "outerwear", "kimono": "outerwear",
    "evening_jacket": "outerwear", "structured_jacket": "outerwear",
}


def get_canonical_type(item: dict) -> Optional[str]:
    """
    Get canonical article type from an item dict.

    Resolution order:
    1. ``article_type`` field (normalized via alias map)
    2. Raw value with spaces/hyphens replaced
    3. None if nothing found
    """
    raw = (item.get("article_type") or "").strip()
    if not raw:
        return None

    # Lowercase, replace spaces with underscores
    key = raw.lower().replace("-", " ")

    # Check alias map first
    if key in _ARTICLE_TYPE_ALIASES:
        return _ARTICLE_TYPE_ALIASES[key]

    # Normalize directly
    canonical = key.replace(" ", "_")
    # If it's a known canonical type, return it
    if canonical in _TYPE_TO_BROAD:
        return canonical

    return canonical  # Return best-effort normalization


def get_broad_category(item: dict) -> Optional[str]:
    """
    Get broad category: ``tops``, ``bottoms``, ``dresses``, ``outerwear``.

    Tries explicit ``broad_category`` field first, then infers from
    canonical article type.
    """
    # Explicit field
    cat = (item.get("broad_category") or item.get("category") or "").lower().strip()
    if cat in ("tops", "bottoms", "outerwear"):
        return cat
    if cat in ("dresses", "one_piece", "one-piece"):
        return "dresses"

    # Infer from article type
    canon = get_canonical_type(item)
    if canon and canon in _TYPE_TO_BROAD:
        return _TYPE_TO_BROAD[canon]

    return None


# ── Mapping from Gemini coverage_details tags to our canonical dims ──
_COVERAGE_DETAIL_MAP: dict = {
    "backless": "open_back",
    "open_back": "open_back",
    "sheer_panels": "sheer",
    "sheer": "sheer",
    "see_through": "sheer",
    "high_slit": "high_slit",
    "cutouts": "cutouts",
    "cut_outs": "cutouts",
    "midriff_exposed": "crop",
    "crop": "crop",
    "strapless": "strapless",
    "off_shoulder": "strapless",
    "deep_v": "deep_necklines",
    "plunging": "deep_necklines",
    "deep_neckline": "deep_necklines",
    "mini": "mini",
    "micro": "mini",
    "sleeveless": "sleeveless",
    "spaghetti_straps": "sleeveless",
    "bodycon": "bodycon",
    "low_back": "open_back",
}


def detect_coverage_dimensions(item: dict) -> List[str]:
    """
    Detect which coverage dimensions an item triggers.

    If structured Gemini coverage data is present (coverage_level,
    skin_exposure, coverage_details), uses that directly.  Otherwise
    falls back to heuristic inference from article type, neckline, etc.

    Returns:
        List of dimension strings, e.g. ["crop", "strapless", "mini"].
    """
    dims: Set[str] = set()

    # ── Structured coverage (from Gemini Vision) ──────────────────
    coverage_details = item.get("coverage_details") or []
    if isinstance(coverage_details, str):
        coverage_details = [coverage_details]

    if coverage_details:
        for detail in coverage_details:
            key = detail.lower().strip().replace("-", "_").replace(" ", "_")
            if key in _COVERAGE_DETAIL_MAP:
                dims.add(_COVERAGE_DETAIL_MAP[key])

    # Also infer from coverage_level / skin_exposure when they signal
    # something strong, even without specific details
    coverage_level = (item.get("coverage_level") or "").strip()
    skin_exposure = (item.get("skin_exposure") or "").strip()

    if coverage_level in ("Minimal", "Revealing") and "mini" not in dims:
        # Minimal/Revealing coverage is a strong signal, but we don't know
        # WHICH dimension — let the details list handle specifics.
        # Only add a generic flag if no details were found.
        if not dims:
            dims.add("mini")
    if skin_exposure == "High" and not dims:
        dims.add("mini")

    # ── Heuristic fallback (text-based inference) ─────────────────
    # Always run heuristics to catch what structured data might miss

    # 1. Article type
    canon = get_canonical_type(item)
    if canon and canon in ARTICLE_TYPE_COVERAGE:
        dims.add(ARTICLE_TYPE_COVERAGE[canon])

    # 2. Neckline
    neckline = (item.get("neckline") or "").lower().strip()
    if neckline in NECKLINE_COVERAGE:
        dims.add(NECKLINE_COVERAGE[neckline])

    # 3. Style tags
    style_tags = item.get("style_tags") or []
    if isinstance(style_tags, str):
        style_tags = [style_tags]
    for tag in style_tags:
        tag_lower = tag.lower().strip()
        if tag_lower in STYLE_TAG_COVERAGE:
            dims.add(STYLE_TAG_COVERAGE[tag_lower])

    # 4. Sleeve type
    sleeve = (item.get("sleeve_type") or item.get("sleeve") or "").lower().strip()
    if sleeve in SLEEVE_TYPE_COVERAGE:
        dims.add(SLEEVE_TYPE_COVERAGE[sleeve])

    # 5. Length check for "mini"
    length = (item.get("length") or "").lower()
    if "mini" in length and "mini" not in dims:
        dims.add("mini")

    # 6. Check for crop / micro in name as fallback
    name = (item.get("name") or "").lower()
    if "crop" in name and "crop" not in dims:
        dims.add("crop")
    if "micro" in name and "mini" not in dims:
        dims.add("mini")

    return list(dims)
