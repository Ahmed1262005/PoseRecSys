"""
Occasion popularity by age group.

Source rank 1-5 normalized to 0.2 .. 1.0.
"""

from scoring.context import AgeGroup

# fmt: off
OCCASION_AFFINITY: dict = {
    AgeGroup.GEN_Z: {
        "casual": 1.0, "evenings": 1.0, "events": 0.8,
        "sporty": 0.8, "smart_casual": 0.6, "office": 0.5,
    },
    AgeGroup.YOUNG_ADULT: {
        "casual": 1.0, "smart_casual": 1.0, "sporty": 1.0,
        "office": 0.8, "events": 0.8, "evenings": 0.8,
    },
    AgeGroup.MID_CAREER: {
        "casual": 1.0, "office": 1.0, "smart_casual": 1.0,
        "sporty": 0.8, "events": 0.8, "evenings": 0.6,
    },
    AgeGroup.ESTABLISHED: {
        "casual": 1.0, "smart_casual": 1.0,
        "office": 0.8, "sporty": 0.8, "events": 0.8,
        "evenings": 0.6,
    },
    AgeGroup.SENIOR: {
        "casual": 1.0, "sporty": 0.8,
        "smart_casual": 0.8, "events": 0.6,
        "office": 0.4, "evenings": 0.4,
    },
}
# fmt: on

# ── Map system occasion values -> canonical keys ──────────────────
OCCASION_MAP: dict = {
    # Casual
    "casual": "casual", "everyday": "casual", "weekend": "casual",
    "lounging": "casual", "errands": "casual",
    "Everyday": "casual", "Casual": "casual", "Weekend": "casual",
    "Lounging": "casual",
    # Office
    "office": "office", "work": "office",
    "Office": "office", "Work": "office",
    # Smart casual
    "smart-casual": "smart_casual", "smart casual": "smart_casual",
    "brunch": "smart_casual",
    "Brunch": "smart_casual",
    # Evenings
    "evening": "evenings", "date night": "evenings", "date_night": "evenings",
    "party": "evenings", "nights-out": "evenings", "going-out": "evenings",
    "Date Night": "evenings", "Party": "evenings",
    "Evening Event": "evenings",
    # Events
    "events": "events", "wedding": "events", "formal": "events",
    "galas": "events", "weddings-guest": "events",
    # Sporty
    "workout": "sporty", "active": "sporty", "gym": "sporty",
    "athleisure": "sporty", "sportswear": "sporty",
    "Workout": "sporty",
    # Vacation -> casual (cross-age)
    "beach": "casual", "vacation": "casual", "resort": "casual",
    "Vacation": "casual", "Beach": "casual",
}
