"""
V3 Candidate Mixer — combines candidates from multiple sources with
quota allocation, fallback chains, and deduplication.

Source quotas: preference=225, session=125, exploration=75, merch=75
When include_brands active: preference=350, session=100, exploration=0, merch=50
"""

import logging
from typing import Any, Dict, List, Optional, Set

from recs.v3.models import CandidateStub

logger = logging.getLogger(__name__)


# Default source targets (total = 500)
SOURCE_TARGETS: Dict[str, int] = {
    "preference": 225,
    "session": 125,
    "exploration": 75,
    "merch": 75,
}

# Fallback chains: if a source is weak, overflow to these sources
FALLBACK_CHAIN: Dict[str, List[str]] = {
    "preference": ["exploration", "merch"],
    "session": ["preference", "exploration"],
    "exploration": ["preference", "merch"],
    "merch": ["preference", "exploration"],
}

# Minimum pool fraction before triggering backfill
MIN_POOL_FRACTION = 0.8

# Default target pool size
TARGET_POOL_SIZE = 500


class CandidateMixer:
    """
    Combine candidates from multiple sources with quota allocation,
    fallback chains, and deduplication.
    """

    def __init__(
        self,
        source_targets: Optional[Dict[str, int]] = None,
        fallback_chain: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.source_targets = source_targets or dict(SOURCE_TARGETS)
        self.fallback_chain = fallback_chain or dict(FALLBACK_CHAIN)

    def mix(
        self,
        source_results: Dict[str, List[CandidateStub]],
        target_size: int = TARGET_POOL_SIZE,
    ) -> List[CandidateStub]:
        """
        Mix candidates from all sources respecting quotas and fallbacks.

        Returns deduplicated list of CandidateStubs.
        """
        # Scale quotas to target size
        total_quota = sum(self.source_targets.values()) or 1
        scaled_quotas = {
            name: max(1, int(target * target_size / total_quota))
            for name, target in self.source_targets.items()
        }

        # Phase 1: Fill each source up to its quota
        stubs: List[CandidateStub] = []
        remaining: Dict[str, List[CandidateStub]] = {}

        for name, quota in scaled_quotas.items():
            source_stubs = source_results.get(name, [])
            taken = source_stubs[:quota]
            leftover = source_stubs[quota:]
            stubs.extend(taken)
            remaining[name] = leftover

        # Phase 2: Fallback — fill under-quota sources from fallback chains
        for name, quota in scaled_quotas.items():
            source_stubs = source_results.get(name, [])
            taken_count = min(len(source_stubs), quota)
            deficit = quota - taken_count

            if deficit > 0:
                chain = self.fallback_chain.get(name, [])
                for fallback_name in chain:
                    if deficit <= 0:
                        break
                    fb_remaining = remaining.get(fallback_name, [])
                    take = fb_remaining[:deficit]
                    stubs.extend(take)
                    remaining[fallback_name] = fb_remaining[len(take):]
                    deficit -= len(take)

        # Phase 3: Dedup
        stubs = self._dedup_by_item_id(stubs)
        stubs = self._dedup_by_image_key(stubs)

        # Phase 4: Backfill if pool is too small
        min_size = int(target_size * MIN_POOL_FRACTION)
        if len(stubs) < min_size:
            stubs = self._backfill(stubs, source_results, target_size)
            # Re-dedup after backfill
            stubs = self._dedup_by_item_id(stubs)
            stubs = self._dedup_by_image_key(stubs)

        logger.info(
            "Mixer: %d stubs from %s (target=%d)",
            len(stubs),
            {n: len(r) for n, r in source_results.items()},
            target_size,
        )

        return stubs

    def get_source_mix(
        self, source_results: Dict[str, List[CandidateStub]]
    ) -> Dict[str, int]:
        """Return the source mix without actually mixing. For debug/stats."""
        return {name: len(stubs) for name, stubs in source_results.items()}

    @staticmethod
    def _dedup_by_item_id(stubs: List[CandidateStub]) -> List[CandidateStub]:
        """Deduplicate by item_id, keeping the stub with highest retrieval_score."""
        seen: Dict[str, CandidateStub] = {}
        for s in stubs:
            existing = seen.get(s.item_id)
            if existing is None or s.retrieval_score > existing.retrieval_score:
                seen[s.item_id] = s
        return list(seen.values())

    @staticmethod
    def _dedup_by_image_key(stubs: List[CandidateStub]) -> List[CandidateStub]:
        """Deduplicate by image_dedup_key, keeping first occurrence."""
        seen_keys: Set[str] = set()
        result: List[CandidateStub] = []
        for s in stubs:
            key = s.image_dedup_key
            if key is None or key not in seen_keys:
                result.append(s)
                if key is not None:
                    seen_keys.add(key)
        return result

    def _backfill(
        self,
        current: List[CandidateStub],
        source_results: Dict[str, List[CandidateStub]],
        target_size: int,
    ) -> List[CandidateStub]:
        """
        Top up from preference + exploration when pool is too small.

        Appends stubs not already in the pool.
        """
        existing_ids = {s.item_id for s in current}
        result = list(current)

        # Backfill order: preference first, then exploration
        for source_name in ["preference", "exploration", "merch", "session"]:
            if len(result) >= target_size:
                break
            for stub in source_results.get(source_name, []):
                if stub.item_id not in existing_ids:
                    result.append(stub)
                    existing_ids.add(stub.item_id)
                    if len(result) >= target_size:
                        break

        return result
