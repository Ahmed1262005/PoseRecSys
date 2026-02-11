"""
ContextScorer -- the shared scoring orchestrator.

Combines age-affinity and weather/season scoring into a single
adjustment that both the feed pipeline and search reranker can use.

Usage::

    from scoring.scorer import ContextScorer
    from scoring.context_resolver import ContextResolver

    resolver = ContextResolver(weather_api_key="...")
    scorer = ContextScorer()

    ctx = resolver.resolve(user_id, jwt_metadata, birthdate, profile)

    # Single item
    adj = scorer.score_item(item_dict, ctx)
    item["score"] += adj

    # Batch (modifies in-place)
    scorer.score_items(items, ctx, score_field="rrf_score")
"""

from typing import List, Optional

from scoring.context import UserContext
from scoring.age_scorer import AgeScorer
from scoring.weather_scorer import WeatherScorer

MAX_CONTEXT_ADJUSTMENT = 0.30


class ContextScorer:
    """
    Orchestrates all context-aware scoring signals.

    Stateless — safe to share across threads / reuse across requests.
    """

    def __init__(self) -> None:
        self._age_scorer = AgeScorer()
        self._weather_scorer = WeatherScorer()

    def score_item(self, item: dict, ctx: UserContext) -> float:
        """
        Compute total context-aware adjustment for one item.

        Returns float in ``[-MAX_CONTEXT_ADJUSTMENT, +MAX_CONTEXT_ADJUSTMENT]``.
        """
        adj = 0.0

        if ctx.age_group:
            adj += self._age_scorer.score(
                item, ctx.age_group, ctx.coverage_prefs,
            )

        if ctx.weather:
            adj += self._weather_scorer.score(item, ctx.weather)

        return max(-MAX_CONTEXT_ADJUSTMENT, min(MAX_CONTEXT_ADJUSTMENT, adj))

    def score_items(
        self,
        items: list,
        ctx: UserContext,
        score_field: str = "score",
        weight: float = 1.0,
    ) -> list:
        """
        Batch score items **in-place**.

        Adds ``context_adjustment`` key to each item dict and modifies
        ``score_field`` by adding ``adjustment * weight``.
        """
        for item in items:
            adj = self.score_item(item, ctx)
            item["context_adjustment"] = adj
            current = item.get(score_field, 0)
            if isinstance(current, (int, float)):
                item[score_field] = current + (adj * weight)
        return items

    def explain_item(self, item: dict, ctx: UserContext) -> dict:
        """
        Return detailed breakdown of scoring for debugging / admin UI.
        """
        breakdown: dict = {"total": 0.0}

        if ctx.age_group:
            age_adj = self._age_scorer.score(
                item, ctx.age_group, ctx.coverage_prefs,
            )
            breakdown["age"] = round(age_adj, 4)
            breakdown["age_group"] = ctx.age_group.value
            breakdown["total"] += age_adj

        if ctx.weather:
            weather_adj = self._weather_scorer.score(item, ctx.weather)
            breakdown["weather"] = round(weather_adj, 4)
            breakdown["season"] = ctx.weather.season.value
            breakdown["temperature_c"] = ctx.weather.temperature_c
            breakdown["is_hot"] = ctx.weather.is_hot
            breakdown["is_cold"] = ctx.weather.is_cold
            breakdown["is_rainy"] = ctx.weather.is_rainy
            breakdown["total"] += weather_adj

        breakdown["total"] = round(
            max(-MAX_CONTEXT_ADJUSTMENT, min(MAX_CONTEXT_ADJUSTMENT, breakdown["total"])),
            4,
        )
        return breakdown
