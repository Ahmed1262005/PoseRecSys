"""
Shared Scoring Module.

Context-aware scoring components used by both the feed pipeline
(``recs/``) and search reranker (``search/``).

Quick start::

    from scoring import ContextScorer, ContextResolver, UserContext, ProfileScorer

    resolver = ContextResolver(weather_api_key="...")
    context_scorer = ContextScorer()
    profile_scorer = ProfileScorer()

    ctx = resolver.resolve(
        user_id="abc",
        jwt_user_metadata=user.user_metadata,
        birthdate="2000-01-15",
        onboarding_profile=profile_dict,
    )

    context_adj = context_scorer.score_item(item_dict, ctx)
    profile_adj = profile_scorer.score_item(item_dict, onboarding_profile)
"""

from scoring.context import AgeGroup, Season, UserContext, WeatherContext
from scoring.scorer import ContextScorer
from scoring.context_resolver import ContextResolver
from scoring.profile_scorer import ProfileScorer, ProfileScoringConfig

__all__ = [
    "AgeGroup",
    "Season",
    "UserContext",
    "WeatherContext",
    "ContextScorer",
    "ContextResolver",
    "ProfileScorer",
    "ProfileScoringConfig",
]
