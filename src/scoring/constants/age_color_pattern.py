"""
Color/pattern preferences by age group.

Pattern loudness: how bold/experimental the patterns are.
Color boldness: how bright/varied the color palette is.
"""

from scoring.context import AgeGroup

# ── Pattern loudness tolerance by age ─────────────────────────────
# Higher value = more accepting of that loudness category.
# fmt: off
PATTERN_LOUDNESS: dict = {
    AgeGroup.GEN_Z: {
        "bold": 0.9,        # checker, neon, graphic, animal print, tie dye
        "playful": 0.9,     # stripes, polka dots, novelty
        "classic": 0.7,     # florals, subtle stripes
        "solid": 0.7,       # still major (black/white)
    },
    AgeGroup.YOUNG_ADULT: {
        "bold": 0.5,
        "playful": 0.6,
        "classic": 0.8,
        "solid": 0.9,
    },
    AgeGroup.MID_CAREER: {
        "bold": 0.3,
        "playful": 0.4,
        "classic": 0.9,
        "solid": 0.9,
    },
    AgeGroup.ESTABLISHED: {
        "bold": 0.2,
        "playful": 0.3,
        "classic": 0.9,
        "solid": 0.9,
    },
    AgeGroup.SENIOR: {
        "bold": 0.2,
        "playful": 0.3,
        "classic": 0.8,
        "solid": 0.9,
    },
}
# fmt: on

# ── Map pattern values to loudness categories ────────────────────
PATTERN_TO_LOUDNESS: dict = {
    # Bold
    "animal_print": "bold", "animal print": "bold", "leopard": "bold",
    "neon": "bold", "tie_dye": "bold", "tie dye": "bold",
    "camo": "bold", "camouflage": "bold",
    "abstract": "bold", "graphic": "bold",
    "tropical": "bold", "snake": "bold", "zebra": "bold",
    # Playful
    "polka_dots": "playful", "polka dot": "playful", "polka dots": "playful",
    "checkered": "playful", "checker": "playful", "gingham": "playful",
    "stripes": "playful", "striped": "playful",
    "plaid": "playful", "tartan": "playful",
    "novelty": "playful", "emoji": "playful",
    # Classic
    "floral": "classic", "geometric": "classic",
    "paisley": "classic", "houndstooth": "classic",
    "herringbone": "classic", "damask": "classic",
    "toile": "classic", "ikat": "classic",
    "pinstripe": "classic", "argyle": "classic",
    # Solid
    "solid": "solid",
}

# ── Color boldness by age ─────────────────────────────────────────
# Higher = more open to bright/varied colors.
COLOR_BOLDNESS: dict = {
    AgeGroup.GEN_Z: 0.9,
    AgeGroup.YOUNG_ADULT: 0.6,
    AgeGroup.MID_CAREER: 0.5,
    AgeGroup.ESTABLISHED: 0.4,
    AgeGroup.SENIOR: 0.5,       # uplifting accents welcomed
}

# Color family classifications
BOLD_COLOR_FAMILIES = frozenset({
    "neon", "bright", "hot pink", "electric blue", "lime",
    "fuchsia", "coral", "orange",
})
NEUTRAL_COLOR_FAMILIES = frozenset({
    "neutrals", "black", "white", "cream", "beige", "camel",
    "grey", "gray", "browns", "taupe", "ivory",
})
JEWEL_COLOR_FAMILIES = frozenset({
    "emerald", "ruby", "sapphire", "burgundy", "navy",
    "deep purple", "teal", "plum", "garnet",
})
PASTEL_COLOR_FAMILIES = frozenset({
    "pastels", "blush", "lavender", "mint", "baby blue",
    "soft pink", "lilac", "periwinkle",
})
