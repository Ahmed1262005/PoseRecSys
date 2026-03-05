"""
Search Analytics Tracking.

Logs search queries, clicks, and conversions to Supabase
for later analysis and search quality improvement.
"""

import threading
from typing import Any, Dict, List, Optional

from supabase import Client

from core.logging import get_logger

logger = get_logger(__name__)


class SearchAnalytics:
    """
    Track search events for analysis.

    Tables:
    - search_analytics: Every search query with timing/results
    - search_clicks: When a user clicks a search result
    - search_conversions: When a user converts from search
    """

    def __init__(self, supabase: Optional[Client] = None):
        if supabase is None:
            from config.database import get_supabase_client
            supabase = get_supabase_client()
        self._supabase = supabase

    # =========================================================================
    # Search Events
    # =========================================================================

    def log_search(
        self,
        query: str,
        intent: str,
        total_results: int,
        algolia_results: int = 0,
        semantic_results: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        latency_ms: int = 0,
        algolia_latency_ms: int = 0,
        semantic_latency_ms: int = 0,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Log a search event."""
        try:
            self._supabase.table("search_analytics").insert({
                "query": query,
                "query_normalized": query.lower().strip(),
                "user_id": user_id,
                "session_id": session_id,
                "intent": intent,
                "total_results": total_results,
                "algolia_results": algolia_results,
                "semantic_results": semantic_results,
                "filters": filters or {},
                "latency_ms": latency_ms,
                "algolia_latency_ms": algolia_latency_ms,
                "semantic_latency_ms": semantic_latency_ms,
            }).execute()
        except Exception as e:
            # Don't let analytics failures break search
            logger.warning("Failed to log search analytics", error=str(e))

    # =========================================================================
    # Click Events
    # =========================================================================

    def log_click(
        self,
        query: str,
        product_id: str,
        position: int,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Log when a user clicks a search result."""
        try:
            self._supabase.table("search_clicks").insert({
                "query": query,
                "product_id": product_id,
                "position": position,
                "user_id": user_id,
                "session_id": session_id,
            }).execute()
        except Exception as e:
            logger.warning("Failed to log search click", error=str(e))

    # =========================================================================
    # Conversion Events
    # =========================================================================

    def log_conversion(
        self,
        query: str,
        product_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Log when a user converts (add to cart / purchase) from search."""
        try:
            self._supabase.table("search_conversions").insert({
                "query": query,
                "product_id": product_id,
                "user_id": user_id,
                "session_id": session_id,
            }).execute()
        except Exception as e:
            logger.warning("Failed to log search conversion", error=str(e))

    # =========================================================================
    # Impression Tracking
    # =========================================================================

    def log_impressions(
        self,
        query: str,
        product_ids: List[str],
        page: int = 1,
        user_id: Optional[str] = None,
        anon_id: Optional[str] = None,
    ):
        """Log which products were shown in a search result page.

        Called as a background task after every search response so it
        never adds latency to the user-facing request.
        """
        if not product_ids:
            return
        try:
            self._supabase.table("search_impressions").insert({
                "query": query,
                "product_ids": product_ids,
                "page": page,
                "user_id": user_id,
                "anon_id": anon_id,
            }).execute()
        except Exception as e:
            logger.warning("Failed to log search impressions", error=str(e))

    def load_impression_counts(
        self,
        user_id: Optional[str] = None,
        anon_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """Load per-product impression counts for a user (last 7 days).

        Returns a dict of {product_id: count}. Used by the reranker to
        soft-demote over-exposed products.

        Returns an empty dict on failure (non-fatal).
        """
        if not user_id and not anon_id:
            return {}
        try:
            params: Dict[str, Any] = {}
            if user_id:
                params["p_user_id"] = user_id
            if anon_id:
                params["p_anon_id"] = anon_id

            result = self._supabase.rpc(
                "get_user_search_impression_counts", params
            ).execute()

            if not result.data:
                return {}

            return {
                str(row["product_id"]): int(row["impression_count"])
                for row in result.data
            }
        except Exception as e:
            logger.warning(
                "Failed to load impression counts",
                user_id=user_id,
                error=str(e),
            )
            return {}


# =============================================================================
# Singleton
# =============================================================================

_analytics: Optional[SearchAnalytics] = None
_analytics_lock = threading.Lock()


def get_search_analytics() -> SearchAnalytics:
    """Get or create the SearchAnalytics singleton (thread-safe)."""
    global _analytics
    if _analytics is None:
        with _analytics_lock:
            if _analytics is None:
                _analytics = SearchAnalytics()
    return _analytics
