"""
UI Engines for style discovery.

- SwipeEngine: Base Tinder-style binary choice
- PredictiveFourEngine: Predictive four-choice with learning

Factory functions:
- get_engine: Get engine by gender
- get_women_engine: Get women's fashion engine
- get_men_engine: Get men's fashion engine
- get_search_engine: Get search engine singleton
"""
from .swipe_engine import SwipeEngine, UserPreferences, SwipeAction
from .predictive_four_engine import PredictiveFourEngine, PredictivePreferences
from .factory import (
    get_engine,
    get_women_engine,
    get_men_engine,
    get_image_url,
    get_search_engine,
    normalize_gender,
)

__all__ = [
    # Engines
    'SwipeEngine', 'UserPreferences', 'SwipeAction',
    'PredictiveFourEngine', 'PredictivePreferences',
    # Factory functions
    'get_engine',
    'get_women_engine',
    'get_men_engine',
    'get_image_url',
    'get_search_engine',
    'normalize_gender',
]
