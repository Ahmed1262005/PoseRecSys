"""
Coverage tolerance by age group.

Scale: 0.0 (generally avoided) to 1.0 (very accepted / normalized).
Used bidirectionally: boost for high-tolerance combos, penalize for low.

Dimensions:
- deep_necklines: plunging, deep-V, sweetheart
- open_back: backless, open-back
- sheer: see-through, mesh
- cutouts: cut-outs, side cutouts
- high_slit: thigh-high slits
- crop: crop tops, midriff-baring
- mini: mini skirts, mini dresses, short shorts, micro shorts
- bodycon: bodycon / body-hugging fit
- strapless: tube tops, strapless dresses, off-shoulder
- sleeveless: tank tops, sleeveless dresses/tops (NEW)

IMPORTANT: These are priors only. User's explicit preferences
(no_revealing, styles_to_avoid) are hard filters handled by
FeasibilityFilter and always override these soft scores.
"""

from scoring.context import AgeGroup

# fmt: off
COVERAGE_TOLERANCE: dict = {
    AgeGroup.GEN_Z: {
        "deep_necklines": 0.9, "open_back": 0.9, "sheer": 0.7,
        "cutouts": 0.9, "high_slit": 0.7, "crop": 0.9,
        "mini": 0.9, "bodycon": 0.9, "strapless": 0.8,
        "sleeveless": 0.9,
    },
    AgeGroup.YOUNG_ADULT: {
        "deep_necklines": 0.7, "open_back": 0.5, "sheer": 0.5,
        "cutouts": 0.5, "high_slit": 0.8, "crop": 0.6,
        "mini": 0.6, "bodycon": 0.5, "strapless": 0.6,
        "sleeveless": 0.8,
    },
    AgeGroup.MID_CAREER: {
        "deep_necklines": 0.5, "open_back": 0.3, "sheer": 0.3,
        "cutouts": 0.3, "high_slit": 0.6, "crop": 0.3,
        "mini": 0.3, "bodycon": 0.3, "strapless": 0.4,
        "sleeveless": 0.5,
    },
    AgeGroup.ESTABLISHED: {
        "deep_necklines": 0.3, "open_back": 0.2, "sheer": 0.2,
        "cutouts": 0.2, "high_slit": 0.5, "crop": 0.1,
        "mini": 0.2, "bodycon": 0.2, "strapless": 0.2,
        "sleeveless": 0.35,
    },
    AgeGroup.SENIOR: {
        "deep_necklines": 0.2, "open_back": 0.1, "sheer": 0.1,
        "cutouts": 0.1, "high_slit": 0.3, "crop": 0.05,
        "mini": 0.1, "bodycon": 0.1, "strapless": 0.1,
        "sleeveless": 0.25,
    },
}
# fmt: on


# ── Detection maps: product attributes -> coverage dimensions ────

# Article types that inherently trigger a coverage dimension
ARTICLE_TYPE_COVERAGE: dict = {
    "crop_top": "crop",
    "tube_top": "strapless",
    "bandeau": "strapless",
    "bralette": "crop",
    "bodycon_dress": "bodycon",
    "mini_dress": "mini",
    "mini_skirt": "mini",
    # Shorts — trigger "mini" because they expose legs like mini skirts
    "shorts": "mini",
    "athletic_shorts": "mini",
    "bike_shorts": "mini",
    # Tank tops — trigger "sleeveless"
    "tank_top": "sleeveless",
    "cami": "sleeveless",
    "halter_top": "sleeveless",
}

# Neckline values that trigger a coverage dimension
NECKLINE_COVERAGE: dict = {
    "deep v": "deep_necklines",
    "deep-v": "deep_necklines",
    "plunging": "deep_necklines",
    "sweetheart": "deep_necklines",
    "halter": "strapless",
    "off shoulder": "strapless",
    "off_shoulder": "strapless",
    "off-shoulder": "strapless",
    "strapless": "strapless",
}

# Sleeve type values that trigger "sleeveless" coverage dimension (NEW)
SLEEVE_TYPE_COVERAGE: dict = {
    "sleeveless": "sleeveless",
    "strapless": "strapless",
    "spaghetti": "sleeveless",
    "spaghetti strap": "sleeveless",
    "spaghetti-strap": "sleeveless",
}

# Style tags that trigger coverage dimensions
STYLE_TAG_COVERAGE: dict = {
    "cutouts": "cutouts",
    "cut-outs": "cutouts",
    "sheer": "sheer",
    "see-through": "sheer",
    "backless": "open_back",
    "open back": "open_back",
    "open-back": "open_back",
}
