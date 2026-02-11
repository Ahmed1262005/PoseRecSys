"""
Hybrid Search Module: Algolia (lexical) + FashionCLIP (semantic).

Provides:
- AlgoliaClient: Index management, search, autocomplete
- HybridSearchService: Combined Algolia + FashionCLIP search
- QueryClassifier: Intent classification (exact/specific/vague)
- SearchAnalytics: Query/click/conversion tracking
"""

from search.algolia_client import get_algolia_client, AlgoliaClient
from search.hybrid_search import HybridSearchService, get_hybrid_search_service
from search.query_classifier import QueryClassifier, QueryIntent
from search.autocomplete import AutocompleteService
from search.analytics import SearchAnalytics

__all__ = [
    "AlgoliaClient",
    "get_algolia_client",
    "HybridSearchService",
    "get_hybrid_search_service",
    "QueryClassifier",
    "QueryIntent",
    "AutocompleteService",
    "SearchAnalytics",
]
