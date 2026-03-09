"""
V3 Feed Event Logger — async writes to Supabase user_interactions.

DB CHECK constraint only allows: click, hover, add_to_wishlist, add_to_cart, purchase.
Session-only actions (skip, hide, search) are NOT written to DB.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Actions the DB accepts (CHECK constraint on user_interactions)
DB_VALID_ACTIONS = {"click", "add_to_wishlist", "add_to_cart", "purchase", "hover"}

# Map V3 action names to DB action names
ACTION_DB_MAP = {
    "click": "click",
    "save": "add_to_wishlist",
    "cart": "add_to_cart",
    "purchase": "purchase",
    "hover": "hover",
    # These are session-only, NOT written to DB:
    # "skip", "hide", "search"
}


class EventLogger:
    """
    Async event logger for V3 feed.

    Writes to Supabase ``user_interactions`` in a background thread
    so the request path is never blocked.

    Impression events are batched per page. Action events are single.
    """

    def __init__(self, supabase_client: Any) -> None:
        self._supabase = supabase_client

    def log_impressions(
        self,
        user_id: str,
        session_id: str,
        items: List[Dict[str, Any]],
        source: str = "feed",
    ) -> None:
        logger.info(
            "Impressions: session=%s user=%s items=%d source=%s",
            session_id, user_id, len(items), source,
        )

    def log_action(
        self,
        user_id: str,
        session_id: str,
        action: str,
        product_id: str,
        source: str = "feed",
        position: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        db_action = ACTION_DB_MAP.get(action)
        if db_action is None:
            # Session-only action (skip, hide, search) — not written to DB
            logger.info(
                "Action %s (session-only): session=%s product=%s",
                action, session_id, product_id,
            )
            return

        data = {
            "user_id": user_id,
            "session_id": session_id,
            "action": db_action,
            "product_id": product_id,
            "source": source,
        }
        if position is not None:
            data["position"] = position
        if metadata:
            data.update(metadata)

        # Write in background thread
        thread = threading.Thread(
            target=self._write_interaction, args=(data,), daemon=True
        )
        thread.start()

    def _write_interaction(self, data: Dict[str, Any]) -> None:
        """Background thread: write one interaction row to Supabase."""
        try:
            self._supabase.table("user_interactions").insert(data).execute()
        except Exception as e:
            logger.error(
                "Failed to write interaction: action=%s product=%s error=%s",
                data.get("action"), data.get("product_id"), e,
            )


class NoOpEventLogger:
    """
    No-op event logger for testing. Records calls for assertions.
    """

    def __init__(self) -> None:
        self.impressions: List[Dict[str, Any]] = []
        self.actions: List[Dict[str, Any]] = []

    def log_impressions(
        self,
        user_id: str,
        session_id: str,
        items: List[Dict[str, Any]],
        source: str = "feed",
    ) -> None:
        self.impressions.append({
            "user_id": user_id,
            "session_id": session_id,
            "items": items,
            "source": source,
        })

    def log_action(
        self,
        user_id: str,
        session_id: str,
        action: str,
        product_id: str,
        source: str = "feed",
        position: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.actions.append({
            "user_id": user_id,
            "session_id": session_id,
            "action": action,
            "product_id": product_id,
            "source": source,
            "position": position,
            "metadata": metadata,
        })
