"""
Hybrid Search Module: Algolia (lexical) + FashionCLIP (semantic).

Provides:
- AlgoliaClient: Index management, search, autocomplete
- HybridSearchService: Combined Algolia + FashionCLIP search
- QueryClassifier: Intent classification (exact/specific/vague)
- QueryPlanner: LLM-based query understanding (agentic search)
- SearchAnalytics: Query/click/conversion tracking
"""

from search.algolia_client import get_algolia_client, AlgoliaClient
from search.hybrid_search import HybridSearchService, get_hybrid_search_service
from search.query_classifier import QueryClassifier, QueryIntent
from search.query_planner import QueryPlanner, SearchPlan, get_query_planner
from search.autocomplete import AutocompleteService
from search.analytics import SearchAnalytics

__all__ = [
    "AlgoliaClient",
    "get_algolia_client",
    "HybridSearchService",
    "get_hybrid_search_service",
    "QueryClassifier",
    "QueryIntent",
    "QueryPlanner",
    "SearchPlan",
    "get_query_planner",
    "AutocompleteService",
    "SearchAnalytics",
]
