"""
V3 Candidate Source Protocol — defines the retrieval source interface.
"""

from typing import Any, List, Protocol, Set

from recs.v3.models import CandidateStub, FeedRequest, SessionProfile


class CandidateSource(Protocol):
    """Protocol defining the retrieval source interface."""

    def retrieve(
        self,
        user_state: Any,
        session: SessionProfile,
        request: FeedRequest,
        exclude_ids: Set[str],
        limit: int = 200,
    ) -> List[CandidateStub]:
        ...
