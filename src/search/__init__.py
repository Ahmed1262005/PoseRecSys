"""
Hybrid Search Module: Algolia (lexical) + FashionCLIP (semantic).

Provides:
- AlgoliaClient: Index management, search, autocomplete
- HybridSearchService: Combined Algolia + FashionCLIP search
- QueryPlanner: LLM-based query understanding (mode-based architecture)
- SearchAnalytics: Query/click/conversion tracking
- Mode system: Deterministic mode expansion for filters/exclusions
"""

from search.algolia_client import get_algolia_client, AlgoliaClient
from search.hybrid_search import HybridSearchService, get_hybrid_search_service
from search.models import QueryIntent
from search.mode_config import expand_modes, get_rrf_weights
from search.query_planner import QueryPlanner, SearchPlan, get_query_planner
from search.autocomplete import AutocompleteService
from search.analytics import SearchAnalytics

__all__ = [
    "AlgoliaClient",
    "get_algolia_client",
    "HybridSearchService",
    "get_hybrid_search_service",
    "QueryIntent",
    "QueryPlanner",
    "SearchPlan",
    "get_query_planner",
    "expand_modes",
    "get_rrf_weights",
    "AutocompleteService",
    "SearchAnalytics",
]
