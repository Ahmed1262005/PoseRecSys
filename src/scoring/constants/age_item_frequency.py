"""
Item frequency weights by age group.

Source: Domain expert fashion data.
Scale: 0.0 (never worn) -> 1.0 (very common / wardrobe staple).

Mapping:
    Very common  = 1.0
    Common       = 0.6
    Uncommon     = 0.2
    Transitional = 0.4 (e.g. "Uncommon -> Common")

Keys use canonical article types from feasibility_filter.py.
"""

from scoring.context import AgeGroup

# fmt: off
ITEM_FREQUENCY: dict = {
    # ── 18-24 (Gen Z) ─────────────────────────────────────────────
    AgeGroup.GEN_Z: {
        # Tops
        "tank_top": 1.0, "cami": 0.6, "tshirt": 1.0, "blouse": 0.6,
        "tube_top": 0.8, "sweater": 1.0, "cardigan": 0.6, "bodysuit": 0.6,
        "hoodie": 0.8, "sweatshirt": 0.8, "crop_top": 0.9, "halter_top": 0.6,
        "polo": 0.2, "turtleneck": 0.4, "bralette": 0.6, "bandeau": 0.6,
        "sports_bra": 0.6, "athletic_top": 0.6,
        # Bottoms
        "pants": 0.6, "jeans": 1.0, "shorts": 0.6, "leggings": 0.6,
        "skirt": 0.6, "mini_skirt": 0.8, "midi_skirt": 0.4, "maxi_skirt": 0.2,
        "joggers": 0.6, "sweatpants": 0.6, "wide_leg_pants": 0.6,
        "athletic_shorts": 0.6, "bike_shorts": 0.6,
        # One-piece
        "dress": 0.6, "mini_dress": 0.8, "midi_dress": 0.4, "maxi_dress": 0.2,
        "bodycon_dress": 0.8, "slip_dress": 0.6, "sundress": 0.6,
        "romper": 0.2, "jumpsuit": 0.4, "overalls": 0.4,
        # Outerwear
        "coat": 0.6, "jacket": 1.0, "vest": 0.6, "blazer": 0.6,
        "puffer": 0.8, "windbreaker": 0.4, "trench_coat": 0.4,
    },

    # ── 25-34 (Young Adult / Career Building) ─────────────────────
    AgeGroup.YOUNG_ADULT: {
        "tank_top": 1.0, "cami": 0.6, "tshirt": 1.0, "blouse": 1.0,
        "tube_top": 0.4, "sweater": 1.0, "cardigan": 1.0, "bodysuit": 0.6,
        "hoodie": 0.6, "sweatshirt": 0.6, "crop_top": 0.4, "halter_top": 0.4,
        "polo": 0.4, "turtleneck": 0.6, "bralette": 0.4, "bandeau": 0.2,
        "sports_bra": 0.6, "athletic_top": 0.6,
        "pants": 1.0, "jeans": 1.0, "shorts": 0.6, "leggings": 1.0,
        "skirt": 0.6, "mini_skirt": 0.4, "midi_skirt": 0.6, "maxi_skirt": 0.4,
        "joggers": 0.6, "sweatpants": 0.4, "wide_leg_pants": 0.8,
        "athletic_shorts": 0.6, "bike_shorts": 0.4,
        "dress": 1.0, "mini_dress": 0.4, "midi_dress": 0.8, "maxi_dress": 0.4,
        "bodycon_dress": 0.4, "slip_dress": 0.6, "sundress": 0.6,
        "wrap_dress": 0.8, "shirt_dress": 0.6,
        "romper": 0.4, "jumpsuit": 0.6, "overalls": 0.2,
        "coat": 1.0, "jacket": 1.0, "vest": 0.6, "blazer": 1.0,
        "puffer": 0.6, "windbreaker": 0.4, "trench_coat": 0.8,
    },

    # ── 35-44 (Mid-Career / Peak Polish) ──────────────────────────
    AgeGroup.MID_CAREER: {
        "tank_top": 0.8, "cami": 0.6, "tshirt": 1.0, "blouse": 1.0,
        "tube_top": 0.2, "sweater": 1.0, "cardigan": 1.0, "bodysuit": 0.4,
        "hoodie": 0.4, "sweatshirt": 0.4, "crop_top": 0.2, "halter_top": 0.2,
        "polo": 0.4, "turtleneck": 0.8, "bralette": 0.2, "bandeau": 0.1,
        "sports_bra": 0.4, "athletic_top": 0.4,
        "pants": 1.0, "jeans": 1.0, "shorts": 0.4, "leggings": 0.6,
        "skirt": 0.6, "mini_skirt": 0.2, "midi_skirt": 0.6, "maxi_skirt": 0.4,
        "joggers": 0.4, "sweatpants": 0.2, "wide_leg_pants": 0.8,
        "athletic_shorts": 0.4, "bike_shorts": 0.2,
        "dress": 1.0, "mini_dress": 0.2, "midi_dress": 1.0, "maxi_dress": 0.6,
        "bodycon_dress": 0.2, "slip_dress": 0.4, "sundress": 0.4,
        "wrap_dress": 1.0, "shirt_dress": 0.8, "sheath_dress": 0.8,
        "romper": 0.2, "jumpsuit": 0.6, "overalls": 0.1,
        "coat": 1.0, "jacket": 1.0, "vest": 0.6, "blazer": 1.0,
        "puffer": 0.6, "windbreaker": 0.4, "trench_coat": 0.8,
    },

    # ── 45-64 (Established / Comfort + Refined) ──────────────────
    AgeGroup.ESTABLISHED: {
        "tank_top": 0.6, "cami": 0.6, "tshirt": 1.0, "blouse": 1.0,
        "tube_top": 0.2, "sweater": 1.0, "cardigan": 1.0, "bodysuit": 0.2,
        "hoodie": 0.2, "sweatshirt": 0.2, "crop_top": 0.1, "halter_top": 0.1,
        "polo": 0.4, "turtleneck": 0.8, "bralette": 0.1, "bandeau": 0.05,
        "sports_bra": 0.4, "athletic_top": 0.4,
        "pants": 1.0, "jeans": 1.0, "shorts": 0.4, "leggings": 0.6,
        "skirt": 0.6, "mini_skirt": 0.1, "midi_skirt": 0.6, "maxi_skirt": 0.6,
        "joggers": 0.4, "sweatpants": 0.2, "wide_leg_pants": 0.6,
        "athletic_shorts": 0.4, "bike_shorts": 0.2,
        "dress": 1.0, "mini_dress": 0.1, "midi_dress": 1.0, "maxi_dress": 0.8,
        "bodycon_dress": 0.1, "slip_dress": 0.2, "sundress": 0.4,
        "wrap_dress": 0.8, "shirt_dress": 0.8, "sheath_dress": 0.8,
        "shift_dress": 0.8,
        "romper": 0.2, "jumpsuit": 0.4, "overalls": 0.1,
        "coat": 1.0, "jacket": 1.0, "vest": 0.6, "blazer": 1.0,
        "puffer": 0.6, "windbreaker": 0.4, "trench_coat": 0.8,
    },

    # ── 65+ (Senior / Ease + Timelessness) ────────────────────────
    AgeGroup.SENIOR: {
        "tank_top": 0.6, "cami": 0.6, "tshirt": 1.0, "blouse": 1.0,
        "tube_top": 0.2, "sweater": 1.0, "cardigan": 1.0, "bodysuit": 0.2,
        "hoodie": 0.2, "sweatshirt": 0.2, "crop_top": 0.05, "halter_top": 0.05,
        "polo": 0.4, "turtleneck": 0.8, "bralette": 0.05, "bandeau": 0.05,
        "sports_bra": 0.2, "athletic_top": 0.2,
        "pants": 1.0, "jeans": 0.6, "shorts": 0.4, "leggings": 0.6,
        "skirt": 0.6, "mini_skirt": 0.05, "midi_skirt": 0.6, "maxi_skirt": 0.6,
        "joggers": 0.4, "sweatpants": 0.4, "wide_leg_pants": 0.6,
        "athletic_shorts": 0.2, "bike_shorts": 0.1,
        "dress": 0.8, "mini_dress": 0.05, "midi_dress": 0.8, "maxi_dress": 0.8,
        "bodycon_dress": 0.05, "slip_dress": 0.1, "sundress": 0.4,
        "wrap_dress": 0.6, "shirt_dress": 0.6, "sheath_dress": 0.4,
        "shift_dress": 0.8,
        "romper": 0.2, "jumpsuit": 0.2, "overalls": 0.1,
        "coat": 1.0, "jacket": 1.0, "vest": 1.0, "blazer": 0.6,
        "puffer": 0.8, "windbreaker": 0.4, "trench_coat": 0.6,
    },
}
# fmt: on
