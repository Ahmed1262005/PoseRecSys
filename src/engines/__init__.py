"""
UI Engines for style discovery.

- SwipeEngine: Base Tinder-style binary choice
- FourChoiceEngine: Four-choice selection
- RankingEngine: Drag-to-rank interaction
- AttributeTestEngine: Attribute preference testing
- PredictiveFourEngine: Predictive four-choice with learning
"""
from .swipe_engine import SwipeEngine, UserPreferences, SwipeAction
from .four_choice_engine import FourChoiceEngine, FourChoicePreferences
from .ranking_engine import RankingEngine, RankingPreferences
from .attribute_test_engine import AttributeTestEngine, AttributeTestPreferences, ATTRIBUTE_TEST_PHASES
from .predictive_four_engine import PredictiveFourEngine, PredictivePreferences

__all__ = [
    'SwipeEngine', 'UserPreferences', 'SwipeAction',
    'FourChoiceEngine', 'FourChoicePreferences',
    'RankingEngine', 'RankingPreferences',
    'AttributeTestEngine', 'AttributeTestPreferences', 'ATTRIBUTE_TEST_PHASES',
    'PredictiveFourEngine', 'PredictivePreferences',
]
