"""
Algolia Client Singleton.

Uses algoliasearch v4 (SearchClientSync) for index management, search,
facet search, and object operations.

API reference verified against algoliasearch==4.36.0:
- SearchClientSync(app_id, api_key)
- search_single_index(index_name, search_params={...})
- save_objects(index_name, objects=[...])
- set_settings(index_name, index_settings={...})
- save_synonyms(index_name, synonym_hit=[...], replace_existing_synonyms=True)
- search_for_facet_values(index_name, facet_name, {"facetQuery": ...})
- clear_objects(index_name)
- delete_object(index_name, object_id)

Responses are pydantic models; use .to_dict() for plain dicts.
Hit objects use extra='allow' so product fields are accessible via .to_dict().
"""

import threading
from typing import Any, Dict, List, Optional

from algoliasearch.search.client import SearchClientSync

from config.settings import get_settings
from core.logging import get_logger
from search.algolia_config import (
    ALGOLIA_INDEX_SETTINGS,
    ALGOLIA_SYNONYMS,
    REPLICA_SUFFIXES,
    get_replica_names,
)

logger = get_logger(__name__)


class AlgoliaClient:
    """
    Singleton wrapper around the Algolia SearchClientSync (v4).

    Handles index configuration, object operations, and search.
    """

    def __init__(
        self,
        app_id: Optional[str] = None,
        write_key: Optional[str] = None,
        search_key: Optional[str] = None,
        index_name: Optional[str] = None,
    ):
        settings = get_settings()
        self.app_id = app_id or settings.algolia_app_id
        self.write_key = write_key or settings.algolia_write_key
        self.search_key = search_key or settings.algolia_search_key
        self.index_name = index_name or settings.algolia_index_name

        if not self.app_id:
            raise ValueError("ALGOLIA_APP_ID is required")

        # Use write key for indexing operations, search key for queries
        api_key = self.write_key or self.search_key
        if not api_key:
            raise ValueError("ALGOLIA_WRITE_KEY or ALGOLIA_SEARCH_KEY is required")

        self._client = SearchClientSync(self.app_id, api_key)

    def close(self):
        """Close the underlying HTTP client."""
        self._client.close()

    # =========================================================================
    # Index Configuration
    # =========================================================================

    def configure_index(self) -> dict:
        """
        Apply index settings (searchable attributes, facets, ranking).

        Returns:
            Response dict with taskID.
        """
        resp = self._client.set_settings(
            index_name=self.index_name,
            index_settings=ALGOLIA_INDEX_SETTINGS,
        )
        return resp.to_dict()

    def configure_synonyms(self) -> dict:
        """
        Upload synonyms to the index, replacing existing ones.

        Returns:
            Response dict with taskID.
        """
        resp = self._client.save_synonyms(
            index_name=self.index_name,
            synonym_hit=ALGOLIA_SYNONYMS,
            replace_existing_synonyms=True,
        )
        return resp.to_dict()

    def configure_replicas(self) -> List[dict]:
        """
        Declare virtual replicas on the primary index and configure each
        replica's ranking.

        This is a two-step process:
        1. Set the `replicas` list on the primary index settings so Algolia
           knows which virtual replicas exist.
        2. Apply custom ranking settings to each virtual replica index.

        Virtual replicas share the primary index's data and settings
        (searchableAttributes, attributesForFaceting, etc.) — only
        customRanking is overridden.

        Returns:
            List of response dicts (one per replica).
        """
        # Step 1: Declare replicas on the primary index and wait
        replica_declarations = get_replica_names(self.index_name)
        resp = self._client.set_settings(
            index_name=self.index_name,
            index_settings={"replicas": replica_declarations},
        )
        task_id = resp.to_dict().get("taskID")
        if task_id:
            self._client.wait_for_task(
                index_name=self.index_name, task_id=task_id,
            )
        logger.info(
            "Declared virtual replicas on primary index",
            primary=self.index_name,
            replicas=replica_declarations,
        )

        # Step 2: Configure each replica's ranking and wait for each
        responses = []
        for suffix, settings in REPLICA_SUFFIXES.items():
            replica_name = f"{self.index_name}{suffix}"
            resp = self._client.set_settings(
                index_name=replica_name,
                index_settings=settings,
            )
            resp_dict = resp.to_dict()
            task_id = resp_dict.get("taskID")
            if task_id:
                self._client.wait_for_task(
                    index_name=replica_name, task_id=task_id,
                )
            responses.append(resp_dict)
            logger.info(
                "Configured virtual replica",
                replica=replica_name,
                settings=settings,
            )
        return responses

    def get_settings(self, index_name: Optional[str] = None) -> dict:
        """Get current index settings."""
        resp = self._client.get_settings(
            index_name=index_name or self.index_name,
        )
        return resp.to_dict()

    # =========================================================================
    # Object Operations
    # =========================================================================

    def save_objects(
        self,
        records: List[dict],
        wait: bool = False,
        batch_size: int = 1000,
    ) -> List:
        """
        Save (upsert) a batch of records to the index.

        Args:
            records: List of dicts, each must have 'objectID'.
            wait: If True, wait for task completion.
            batch_size: Records per batch (default 1000).

        Returns:
            List of BatchResponse objects.
        """
        return self._client.save_objects(
            index_name=self.index_name,
            objects=records,
            wait_for_tasks=wait,
            batch_size=batch_size,
        )

    def delete_object(self, object_id: str) -> dict:
        """Delete a single record by objectID."""
        resp = self._client.delete_object(
            index_name=self.index_name,
            object_id=object_id,
        )
        return resp.to_dict()

    def clear_objects(self) -> dict:
        """Delete all objects from the index (keeps settings)."""
        resp = self._client.clear_objects(index_name=self.index_name)
        return resp.to_dict()

    # =========================================================================
    # Search
    # =========================================================================

    def search(
        self,
        query: str,
        filters: Optional[str] = None,
        optional_filters: Optional[List[str]] = None,
        facet_filters: Optional[List] = None,
        hits_per_page: int = 50,
        page: int = 0,
        attributes_to_retrieve: Optional[List[str]] = None,
        facets: Optional[List[str]] = None,
        index_name: Optional[str] = None,
    ) -> dict:
        """
        Search the index (or a replica).

        Args:
            query: Search query text.
            filters: Algolia filter string (e.g. 'brand:"Boohoo" AND price < 100').
                     Hard constraint -- non-matching results are excluded.
            optional_filters: Algolia optional filter strings. Each string
                     boosts matching results' ranking but does NOT exclude
                     non-matching results. Format: ['color:"Red"', 'neckline:"Turtleneck"'].
                     Use for soft attribute preferences when the LLM planner
                     is active, so descriptive keywords (like "ribbed") in the
                     query still have room to match via keyword search.
            facet_filters: Facet filter list.
            hits_per_page: Number of results per page.
            page: Page number (0-indexed).
            attributes_to_retrieve: Specific attributes to return.
            facets: List of facet attribute names to return counts for.
            index_name: Override index name (e.g. for virtual replicas).
                        Defaults to the primary index.

        Returns:
            Dict with keys: hits (list of dicts), nbHits, page, nbPages,
            hitsPerPage, query, params, processingTimeMS, facets (if requested), etc.
        """
        params: Dict[str, Any] = {
            "query": query,
            "hitsPerPage": hits_per_page,
            "page": page,
        }
        if filters:
            params["filters"] = filters
        if optional_filters:
            params["optionalFilters"] = optional_filters
        if facet_filters:
            params["facetFilters"] = facet_filters
        if attributes_to_retrieve:
            params["attributesToRetrieve"] = attributes_to_retrieve
        if facets:
            params["facets"] = facets

        resp = self._client.search_single_index(
            index_name=index_name or self.index_name,
            search_params=params,
        )
        return resp.to_dict()

    def search_for_facet_values(
        self,
        facet_name: str,
        facet_query: str,
        max_facet_hits: int = 10,
    ) -> dict:
        """
        Search within facet values (e.g. brand autocomplete).

        The facet must be declared as searchable(facetName) in attributesForFaceting.

        Args:
            facet_name: Facet attribute name (e.g. 'brand').
            facet_query: Text to search within facet values.
            max_facet_hits: Maximum number of facet hits.

        Returns:
            Dict with 'facetHits' list (each has value, highlighted, count).
        """
        resp = self._client.search_for_facet_values(
            index_name=self.index_name,
            facet_name=facet_name,
            search_for_facet_values_request={
                "facetQuery": facet_query,
                "maxFacetHits": max_facet_hits,
            },
        )
        return resp.to_dict()

    def get_objects(
        self,
        object_ids: List[str],
        attributes_to_retrieve: Optional[List[str]] = None,
        batch_size: int = 1000,
    ) -> Dict[str, dict]:
        """
        Fetch multiple objects by their objectIDs.

        Automatically chunks requests into batches of `batch_size` to stay
        within Algolia's getObjects limit.

        Args:
            object_ids: List of objectIDs to fetch.
            attributes_to_retrieve: Optional list of attributes to return.
            batch_size: Max objects per batch request (default 1000).

        Returns:
            Dict mapping objectID -> record dict. Missing IDs are omitted.
        """
        if not object_ids:
            return {}

        result: Dict[str, dict] = {}

        # Chunk into batches to respect Algolia limits
        for i in range(0, len(object_ids), batch_size):
            chunk = object_ids[i : i + batch_size]
            requests = []
            for oid in chunk:
                req: Dict[str, Any] = {"objectID": oid, "indexName": self.index_name}
                if attributes_to_retrieve:
                    req["attributesToRetrieve"] = attributes_to_retrieve
                requests.append(req)

            try:
                resp = self._client.get_objects(
                    get_objects_params={"requests": requests},
                )
                for item in resp.to_dict().get("results", []):
                    if item and item.get("objectID"):
                        result[item["objectID"]] = item
            except Exception as e:
                logger.warning(
                    "Failed to fetch objects from Algolia",
                    batch_start=i,
                    batch_size=len(chunk),
                    total_ids=len(object_ids),
                    error=str(e),
                )

        return result

    # =========================================================================
    # Utilities
    # =========================================================================

    def index_exists(self) -> bool:
        """Check if the index exists."""
        return self._client.index_exists(index_name=self.index_name)

    def wait_for_task(self, task_id: int):
        """Wait for an indexing task to complete."""
        self._client.wait_for_task(
            index_name=self.index_name,
            task_id=task_id,
        )


# =============================================================================
# Singleton
# =============================================================================

_algolia_client: Optional[AlgoliaClient] = None
_algolia_lock = threading.Lock()


def get_algolia_client() -> AlgoliaClient:
    """Get or create the AlgoliaClient singleton (thread-safe)."""
    global _algolia_client
    if _algolia_client is None:
        with _algolia_lock:
            if _algolia_client is None:
                _algolia_client = AlgoliaClient()
    return _algolia_client
