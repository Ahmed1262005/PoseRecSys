"""
Style popularity by age group.

Source rank 1-5 normalized: rank / 5 -> 0.2 .. 1.0
Maps system style_tags and style_persona values to canonical style keys.
"""

from scoring.context import AgeGroup

# fmt: off
STYLE_AFFINITY: dict = {
    AgeGroup.GEN_Z: {
        "trendy": 1.0, "streetwear": 1.0, "sporty": 0.8,
        "romantic": 0.8, "boho": 0.6, "minimal": 0.6,
        "classic": 0.4, "elegant": 0.4,
    },
    AgeGroup.YOUNG_ADULT: {
        "minimal": 1.0, "sporty": 1.0, "trendy": 0.8,
        "classic": 0.8, "streetwear": 0.8, "elegant": 0.8,
        "boho": 0.6, "romantic": 0.6,
    },
    AgeGroup.MID_CAREER: {
        "classic": 1.0, "elegant": 1.0, "minimal": 0.8,
        "sporty": 0.8, "trendy": 0.6, "boho": 0.6,
        "streetwear": 0.6, "romantic": 0.6,
    },
    AgeGroup.ESTABLISHED: {
        "classic": 1.0, "elegant": 1.0, "minimal": 0.8,
        "sporty": 0.8, "boho": 0.6, "romantic": 0.6,
        "trendy": 0.4, "streetwear": 0.4,
    },
    AgeGroup.SENIOR: {
        "sporty": 0.8, "classic": 0.8, "elegant": 0.8,
        "minimal": 0.6, "boho": 0.4, "romantic": 0.4,
        "trendy": 0.2, "streetwear": 0.2,
    },
}
# fmt: on

# ── Map system values -> canonical style keys ─────────────────────
STYLE_TAG_MAP: dict = {
    # Direct
    "trendy": "trendy", "classic": "classic", "elegant": "elegant",
    "minimal": "minimal", "minimalist": "minimal", "clean": "minimal",
    "streetwear": "streetwear", "sporty": "sporty", "athletic": "sporty",
    "romantic": "romantic", "feminine": "romantic", "boho": "boho",
    "bohemian": "boho",
    # Extended / subculture
    "casual": "minimal",
    "statement": "trendy",
    "preppy": "classic",
    "coquette": "romantic",
    "cottagecore": "boho",
    "dark academia": "classic",
    "quiet luxury": "classic",
    "old money": "classic",
    "clean girl": "minimal",
    "coastal": "boho",
    "y2k": "trendy",
    "edgy": "streetwear",
    "grunge": "streetwear",
    "mod": "classic",
    "retro": "trendy",
    "vintage": "trendy",
    "glam": "elegant",
    "resort": "boho",
    # Brand cluster style_tags -> canonical
    "tailored": "classic", "knitwear": "classic", "blazers": "classic",
    "denim": "streetwear", "basics": "minimal", "logo": "streetwear",
    "leggings": "sporty", "performance": "sporty",
    "micro-trends": "trendy", "party": "trendy", "crop": "trendy",
    "cutouts": "trendy", "bodycon": "trendy",
    "western": "boho", "prints": "boho", "linen": "boho",
    "investment-pieces": "classic", "tailoring": "classic",
    "sneakers": "sporty", "joggers": "sporty",
    "mini-dresses": "trendy", "corset": "romantic",
    "sets": "minimal", "dresses": "romantic", "flattering": "classic",
    "impeccable-basics": "classic",
}
