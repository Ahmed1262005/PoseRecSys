"""
V3 Pool Manager — decides whether to reuse, top-up, or rebuild the pool.

Pure logic, no I/O. All decisions based on data passed in.

See docs/V3_FEED_ARCHITECTURE_PLAN.md §9.2 for full specification.
"""

from dataclasses import dataclass
from typing import Optional

from recs.v3.models import (
    CandidatePool,
    FeedRequest,
    PoolDecision,
    SessionProfile,
    compute_retrieval_signature,
)


# ---------------------------------------------------------------------------
# Thresholds & defaults
# ---------------------------------------------------------------------------

REUSE_THRESHOLD = 48  # Items remaining: healthy enough to reuse
TOP_UP_THRESHOLD = 120  # Items remaining: getting low, could top up
TARGET_POOL_SIZE = 500
RERANK_ACTION_DELTA = 3  # Re-rank every 3 actions

MODE_POOL_SIZES = {
    "explore": 500,
    "sale": 300,
    "new_arrivals": 300,
}


# ---------------------------------------------------------------------------
# Pool Manager
# ---------------------------------------------------------------------------


@dataclass
class PoolManager:
    """
    Decides what to do with the candidate pool for each request.

    Pure logic — no Redis calls, no DB calls. Takes data in, returns
    a decision. The orchestrator acts on the decision.
    """

    reuse_threshold: int = REUSE_THRESHOLD
    top_up_threshold: int = TOP_UP_THRESHOLD
    target_pool_size: int = TARGET_POOL_SIZE
    rerank_action_delta: int = RERANK_ACTION_DELTA

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def get_target_pool_size(self, mode: str) -> int:
        """Return the target pool size for the given feed mode."""
        return MODE_POOL_SIZES.get(mode, self.target_pool_size)

    def decide(
        self,
        pool: Optional[CandidatePool],
        request: FeedRequest,
        session: SessionProfile,
        current_catalog_version: str,
        retrieval_signature: str,
    ) -> PoolDecision:
        """
        Decide whether to rebuild, top-up, or reuse the candidate pool.

        Decision hierarchy (first match wins):
        1. No pool at all          → rebuild
        2. Retrieval sig changed   → rebuild  (filters / preferences changed)
        3. Catalog version drifted → rebuild  (new products indexed)
        4. Nearly exhausted        → rebuild  (< reuse_threshold items left)
        5. Getting low             → top_up   (< top_up_threshold items left)
        6. Otherwise               → reuse    (pool is healthy)

        Parameters
        ----------
        pool : CandidatePool | None
            Current candidate pool, if any.
        request : FeedRequest
            Incoming feed request.
        session : SessionProfile
            Current session state.
        current_catalog_version : str
            Latest catalog version from the data layer.
        retrieval_signature : str
            Hash of the retrieval parameters (filters, prefs, etc.).

        Returns
        -------
        PoolDecision
        """
        # 1. No pool
        if pool is None:
            return PoolDecision(
                action="rebuild",
                reason="no pool exists",
            )

        # 2. Retrieval signature changed
        if pool.retrieval_signature != retrieval_signature:
            return PoolDecision(
                action="rebuild",
                reason="retrieval signature changed",
            )

        # 3. Catalog version drifted
        if pool.catalog_version != current_catalog_version:
            return PoolDecision(
                action="rebuild",
                reason="catalog version drifted",
            )

        # Capture remaining count once
        remaining = pool.remaining

        # 4. Nearly exhausted
        if remaining < self.reuse_threshold:
            return PoolDecision(
                action="rebuild",
                reason=f"pool nearly exhausted ({remaining} remaining)",
                remaining=remaining,
            )

        # 5. Getting low — top up
        if remaining < self.top_up_threshold:
            return PoolDecision(
                action="top_up",
                reason=f"pool getting low ({remaining} remaining)",
                remaining=remaining,
            )

        # 6. Healthy — reuse as-is
        return PoolDecision(
            action="reuse",
            reason=f"pool healthy ({remaining} remaining)",
            remaining=remaining,
        )

    # ------------------------------------------------------------------ #
    # Re-rank trigger
    # ------------------------------------------------------------------ #

    def should_rerank(
        self,
        session: SessionProfile,
        pool: CandidatePool,
    ) -> bool:
        """
        Determine whether the pool should be re-ranked.

        Uses monotonic action_seq, not set lengths. Re-rank fires
        every RERANK_ACTION_DELTA actions.
        """
        delta = session.action_seq - pool.last_rerank_action_seq
        return delta >= self.rerank_action_delta
