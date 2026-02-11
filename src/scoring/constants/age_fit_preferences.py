"""
Fit preferences by age group, per broad category.

Scale: 0.0 (actively avoided) to 1.0 (dominant / most-worn fit).
"""

from scoring.context import AgeGroup

# fmt: off
FIT_PREFERENCES: dict = {
    # ── 18-24 ─────────────────────────────────────────────────────
    AgeGroup.GEN_Z: {
        "tops": {
            "cropped": 0.9, "oversized": 0.9, "fitted": 0.7,
            "relaxed": 0.6, "regular": 0.5, "slim": 0.4, "loose": 0.7,
        },
        "bottoms": {
            "wide-leg": 0.9, "baggy": 0.9, "relaxed": 0.8,
            "straight": 0.7, "regular": 0.5, "slim": 0.3,
            "skinny": 0.3, "loose": 0.8, "flare": 0.6,
        },
        "dresses": {
            "bodycon": 0.8, "fitted": 0.7, "relaxed": 0.5,
            "regular": 0.5, "loose": 0.4, "oversized": 0.3,
        },
        "outerwear": {
            "oversized": 0.9, "regular": 0.5, "fitted": 0.4, "relaxed": 0.6,
        },
    },

    # ── 25-34 ─────────────────────────────────────────────────────
    AgeGroup.YOUNG_ADULT: {
        "tops": {
            "fitted": 0.8, "regular": 0.8, "relaxed": 0.6,
            "slim": 0.6, "oversized": 0.4, "cropped": 0.4, "loose": 0.4,
        },
        "bottoms": {
            "straight": 0.8, "wide-leg": 0.8, "regular": 0.7,
            "slim": 0.5, "skinny": 0.4, "relaxed": 0.5,
            "loose": 0.4, "flare": 0.5,
        },
        "dresses": {
            "regular": 0.7, "fitted": 0.7, "relaxed": 0.5,
            "loose": 0.4, "oversized": 0.3, "bodycon": 0.3,
        },
        "outerwear": {
            "regular": 0.8, "fitted": 0.7, "oversized": 0.4, "relaxed": 0.5,
        },
    },

    # ── 35-44 ─────────────────────────────────────────────────────
    AgeGroup.MID_CAREER: {
        "tops": {
            "regular": 0.9, "fitted": 0.7, "relaxed": 0.6,
            "slim": 0.5, "oversized": 0.3, "cropped": 0.2, "loose": 0.4,
        },
        "bottoms": {
            "straight": 0.9, "wide-leg": 0.7, "regular": 0.8,
            "slim": 0.5, "skinny": 0.3, "relaxed": 0.5,
            "loose": 0.3, "flare": 0.4,
        },
        "dresses": {
            "regular": 0.7, "fitted": 0.7, "relaxed": 0.5,
            "loose": 0.4, "oversized": 0.2, "bodycon": 0.2,
        },
        "outerwear": {
            "regular": 0.9, "fitted": 0.7, "oversized": 0.3, "relaxed": 0.4,
        },
    },

    # ── 45-64 ─────────────────────────────────────────────────────
    AgeGroup.ESTABLISHED: {
        "tops": {
            "regular": 0.9, "relaxed": 0.8, "fitted": 0.5,
            "slim": 0.4, "oversized": 0.2, "cropped": 0.1, "loose": 0.5,
        },
        "bottoms": {
            "straight": 0.9, "regular": 0.8, "wide-leg": 0.6,
            "slim": 0.4, "skinny": 0.2, "relaxed": 0.6,
            "loose": 0.4, "flare": 0.3,
        },
        "dresses": {
            "regular": 0.8, "relaxed": 0.6, "fitted": 0.5,
            "loose": 0.5, "oversized": 0.2, "bodycon": 0.1,
        },
        "outerwear": {
            "regular": 0.9, "fitted": 0.6, "oversized": 0.2, "relaxed": 0.5,
        },
    },

    # ── 65+ ───────────────────────────────────────────────────────
    AgeGroup.SENIOR: {
        "tops": {
            "relaxed": 0.9, "regular": 0.9, "loose": 0.7,
            "fitted": 0.3, "slim": 0.2, "oversized": 0.2, "cropped": 0.05,
        },
        "bottoms": {
            "straight": 0.9, "relaxed": 0.9, "regular": 0.8,
            "wide-leg": 0.5, "slim": 0.3, "skinny": 0.1,
            "loose": 0.6, "flare": 0.2,
        },
        "dresses": {
            "relaxed": 0.9, "regular": 0.8, "loose": 0.7,
            "fitted": 0.3, "oversized": 0.2, "bodycon": 0.05,
        },
        "outerwear": {
            "regular": 0.9, "relaxed": 0.7, "fitted": 0.3, "oversized": 0.2,
        },
    },
}
# fmt: on
