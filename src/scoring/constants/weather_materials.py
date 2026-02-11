"""
Material-season/weather mapping and temperature-based rules.

Used by WeatherScorer to boost weather-appropriate items and
penalize items that don't match current conditions.
"""

from scoring.context import Season

# ── Materials suited for each season ──────────────────────────────

SEASON_MATERIALS: dict = {
    Season.SUMMER: {
        "good": frozenset({
            "linen", "cotton", "silk", "chiffon", "rayon", "chambray",
            "seersucker", "mesh", "jersey", "bamboo", "gauze",
            "muslin", "poplin", "lawn", "voile",
        }),
        "bad": frozenset({
            "wool", "cashmere", "fleece", "velvet", "corduroy",
            "sherpa", "down", "heavy knit", "tweed", "mohair",
        }),
    },
    Season.WINTER: {
        "good": frozenset({
            "wool", "cashmere", "fleece", "velvet", "corduroy",
            "sherpa", "down", "heavy knit", "leather", "suede",
            "faux fur", "thermal", "tweed", "mohair", "merino",
            "flannel",
        }),
        "bad": frozenset({
            "linen", "chiffon", "seersucker", "mesh", "gauze",
            "voile", "lawn",
        }),
    },
    Season.SPRING: {
        "good": frozenset({
            "cotton", "linen", "denim", "jersey", "silk",
            "rayon", "chambray", "light knit", "poplin",
            "chiffon", "crepe",
        }),
        "bad": frozenset({
            "heavy knit", "sherpa", "down", "faux fur", "thermal",
            "mohair",
        }),
    },
    Season.FALL: {
        "good": frozenset({
            "wool", "cashmere", "denim", "corduroy", "suede",
            "leather", "flannel", "knit", "jersey", "tweed",
            "velvet", "crepe",
        }),
        "bad": frozenset({
            "linen", "seersucker", "mesh", "chiffon", "gauze",
            "voile",
        }),
    },
}


# ── Temperature-based item type rules ─────────────────────────────
# Keys use canonical article types from feasibility_filter.py.

TEMP_ITEM_AFFINITY: dict = {
    "hot": {  # feels_like > 25C
        "boost": frozenset({
            "tank_top", "cami", "shorts", "sundress", "tube_top",
            "crop_top", "mini_dress", "mini_skirt", "bralette",
            "bandeau", "halter_top", "athletic_shorts", "romper",
        }),
        "penalize": frozenset({
            "coat", "puffer", "sweater", "turtleneck", "hoodie",
            "sweatshirt", "trench_coat", "heavy knit",
        }),
    },
    "cold": {  # feels_like < 10C
        "boost": frozenset({
            "coat", "puffer", "sweater", "cardigan", "turtleneck",
            "hoodie", "sweatshirt", "vest", "jacket", "blazer",
            "pants", "jeans", "trench_coat",
        }),
        "penalize": frozenset({
            "tank_top", "tube_top", "crop_top", "shorts",
            "mini_dress", "mini_skirt", "sundress", "cami",
            "bralette", "bandeau", "halter_top", "romper",
            "athletic_shorts", "bike_shorts",
        }),
    },
    "mild": {  # 10-25C
        "boost": frozenset({
            "jacket", "blazer", "cardigan", "jeans", "pants",
            "dress", "blouse", "tshirt", "midi_dress",
        }),
        "penalize": frozenset({
            "puffer", "trench_coat",
        }),
    },
    "rainy": {
        "boost": frozenset({
            "jacket", "coat", "pants", "jeans", "windbreaker",
            "trench_coat",
        }),
        "penalize": frozenset({
            # Suede and open footwear don't do well in rain
        }),
    },
}

# Map Season enum to the product seasons field values
SEASON_PRODUCT_MAP: dict = {
    Season.SPRING: "Spring",
    Season.SUMMER: "Summer",
    Season.FALL: "Fall",
    Season.WINTER: "Winter",
}
