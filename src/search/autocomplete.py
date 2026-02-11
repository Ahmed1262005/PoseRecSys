"""
Autocomplete Service.

Returns product name suggestions first, then brand suggestions.
Powered by Algolia's search_single_index and search_for_facet_values (v4 API).
"""

import threading
from typing import Optional

from core.logging import get_logger
from search.algolia_client import AlgoliaClient, get_algolia_client
from search.models import (
    AutocompleteResponse,
    AutocompleteProductSuggestion,
    AutocompleteBrandSuggestion,
)

logger = get_logger(__name__)


class AutocompleteService:
    """
    Autocomplete: products first, then brands.

    Uses Algolia v4 SearchClientSync under the hood:
    - search_single_index() for product suggestions
    - search_for_facet_values() for brand suggestions
    """

    def __init__(self, client: Optional[AlgoliaClient] = None):
        self._client = client

    @property
    def client(self) -> AlgoliaClient:
        if self._client is None:
            self._client = get_algolia_client()
        return self._client

    def autocomplete(
        self,
        query: str,
        limit: int = 10,
        brand_limit: int = 5,
    ) -> AutocompleteResponse:
        """
        Get autocomplete suggestions.

        Products appear first, then brand suggestions.

        Args:
            query: Partial search text.
            limit: Max product suggestions.
            brand_limit: Max brand suggestions.

        Returns:
            AutocompleteResponse with products first, brands second.
        """
        if len(query) < 1:
            return AutocompleteResponse(products=[], brands=[], query=query)

        # 1. Search for product name matches
        products = []
        try:
            product_resp = self.client.search(
                query=query,
                hits_per_page=limit,
                filters="in_stock:true",
                attributes_to_retrieve=[
                    "objectID", "name", "brand", "image_url", "price",
                ],
            )

            for hit in product_resp.get("hits", []):
                highlight_result = hit.get("_highlightResult", {})
                highlighted_name = (
                    highlight_result.get("name", {}).get("value")
                    if isinstance(highlight_result, dict) else None
                )
                products.append(AutocompleteProductSuggestion(
                    id=hit.get("objectID", ""),
                    name=hit.get("name", ""),
                    brand=hit.get("brand", ""),
                    image_url=hit.get("image_url"),
                    price=hit.get("price"),
                    highlighted_name=highlighted_name,
                ))
        except Exception as e:
            logger.error("Autocomplete product search failed", error=str(e))

        # 2. Search for brand facet matches
        brands = []
        try:
            brand_resp = self.client.search_for_facet_values(
                facet_name="brand",
                facet_query=query,
                max_facet_hits=brand_limit,
            )

            for facet in brand_resp.get("facetHits", []):
                brands.append(AutocompleteBrandSuggestion(
                    name=facet.get("value", ""),
                    highlighted=facet.get("highlighted"),
                ))
        except Exception as e:
            logger.error("Autocomplete brand search failed", error=str(e))

        return AutocompleteResponse(
            products=products,
            brands=brands,
            query=query,
        )


# =============================================================================
# Singleton
# =============================================================================

_autocomplete: Optional[AutocompleteService] = None
_autocomplete_lock = threading.Lock()


def get_autocomplete_service() -> AutocompleteService:
    """Get or create the AutocompleteService singleton (thread-safe)."""
    global _autocomplete
    if _autocomplete is None:
        with _autocomplete_lock:
            if _autocomplete is None:
                _autocomplete = AutocompleteService()
    return _autocomplete
