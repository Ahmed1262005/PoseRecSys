"""
V3 Feed Orchestrator — request lifecycle coordinator.

Ties together all V3 components into a single ``get_feed()`` call:

    Session → Profile → Pool decision → [Rebuild|Reuse] → Serve → Events

First page (rebuild):  ~400–700ms
Subsequent pages:      ~30–50ms

See docs/V3_FEED_ARCHITECTURE_PLAN.md §4.2 + §Slice 5.
"""

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

from recs.filter_utils import deduplicate_candidates, extract_image_hash
from recs.models import Candidate
from recs.v3.models import (
    CandidatePool,
    CandidateStub,
    FeedRequest,
    PoolDecision,
    ScoringMeta,
    SessionProfile,
    compute_retrieval_signature,
)
from recs.v3.sources.preference_source import _assign_key_family

logger = logging.getLogger(__name__)


class FeedResponse:
    """
    V3 feed response returned to the API layer.

    Not a Pydantic BaseModel — a plain class to avoid coupling.
    The API layer converts this to the appropriate HTTP response.
    """

    __slots__ = (
        "user_id",
        "session_id",
        "mode",
        "items",
        "cursor",
        "page",
        "pool_size",
        "source_mix",
        "debug",
    )

    def __init__(
        self,
        user_id: str,
        session_id: str,
        mode: str,
        items: List[Candidate],
        cursor: Optional[str] = None,
        page: int = 1,
        pool_size: int = 0,
        source_mix: Optional[Dict[str, int]] = None,
        debug: Optional[Any] = None,
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.mode = mode
        self.items = items
        self.cursor = cursor
        self.page = page
        self.pool_size = pool_size
        self.source_mix = source_mix
        self.debug = debug

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "mode": self.mode,
            "results": [_candidate_to_feed_item(c) for c in self.items],
            "pagination": {
                "cursor": self.cursor,
                "page": self.page,
                "page_size": len(self.items),
                "pool_size": self.pool_size,
                "has_more": self.pool_size > 0,
            },
            "metadata": {
                "source_mix": self.source_mix,
            },
        }
        if self.debug:
            d["debug"] = self.debug
        return d


def _candidate_to_feed_item(c: Candidate) -> Dict[str, Any]:
    """Convert Candidate to API-level feed item dict."""
    return {
        "item_id": c.item_id,
        "name": c.name,
        "brand": c.brand,
        "price": c.price,
        "image_url": c.image_url,
        "gallery_images": c.gallery_images,
        "colors": c.colors,
        "category": c.category,
        "broad_category": c.broad_category,
        "article_type": c.article_type,
        "original_price": c.original_price,
        "is_on_sale": c.is_on_sale,
        "discount_percent": getattr(c, "discount_percent", 0) or 0,
        "is_new": c.is_new,
        "final_score": round(c.final_score, 4),
    }


class FeedOrchestrator:
    """
    V3 request lifecycle coordinator.

    Composes:
    - session_store: Redis session + pool persistence
    - user_profile: UserProfileLoader (cached user state from DB)
    - pool_manager: Decide reuse/rebuild/top-up
    - sources: Dict of CandidateSource instances
    - mixer: CandidateMixer
    - hydrator: FeatureHydrator
    - eligibility: EligibilityFilter
    - ranker: FeedRanker
    - reranker: V3Reranker
    - events: EventLogger

    All dependencies are injected so the orchestrator is testable with mocks.
    """

    def __init__(
        self,
        session_store: Any,
        user_profile: Any,
        pool_manager: Any,
        sources: Dict[str, Any],
        mixer: Any,
        hydrator: Any,
        eligibility: Any,
        ranker: Any,
        reranker: Any,
        events: Any,
    ):
        self.session_store = session_store
        self.user_profile = user_profile
        self.pool_manager = pool_manager
        self.sources = sources
        self.mixer = mixer
        self.hydrator = hydrator
        self.eligibility = eligibility
        self.ranker = ranker
        self.reranker = reranker
        self.events = events

    def get_feed(self, request: FeedRequest) -> FeedResponse:
        """
        Full feed lifecycle.

        Steps:
            1. Load session from Redis (or create)
            2. Load shown_ids from Redis SET into Python set
            3. Load user profile (DB, cached ~5min)
            4. Get catalog_version
            5. Pool decision (rebuild / reuse / top-up)
            6. REBUILD path: sources → mix → hydrate → eligibility → rank → rerank → save
            7. REUSE path: optional re-rank from meta
            8. Serve page from pool (IDs only)
            9. Hydrate page items from MV
            10. Serve-time validation (stock, hidden, shown, image dedup)
            11. Backfill if validation removed items
            12. Update session (Redis)
            13. Build response with source provenance in debug
            14. Enqueue impressions

        Returns:
            FeedResponse with items, cursor, pagination, optional debug.
        """
        t0 = time.time()

        # 1. Session
        session_id = request.session_id or uuid.uuid4().hex[:16]
        session = self.session_store.get_or_create_session(session_id, request.user_id)

        # 2. Shown set
        shown_set = self.session_store.load_shown_set(session_id)

        # 3. User profile
        user_state = self.user_profile.load(request.user_id)

        # 4. Catalog version
        catalog_version = self.session_store.get_catalog_version()

        # 5. Pool decision
        existing_pool = self.session_store.get_pool(session_id, request.mode)
        key_family = _assign_key_family(request.user_id)

        # Build retrieval signature
        hf_hashable = (
            dict(request.hard_filters.model_dump())
            if request.hard_filters and hasattr(request.hard_filters, "model_dump")
            else str(request.hard_filters)
        )

        retrieval_sig = compute_retrieval_signature(
            mode=request.mode,
            hard_filters_hashable=hf_hashable,
            key_family=key_family,
        )

        decision = self.pool_manager.decide(
            pool=existing_pool,
            request=request,
            session=session,
            current_catalog_version=catalog_version,
            retrieval_signature=retrieval_sig,
        )

        # 6/7. Rebuild or reuse
        if decision.action in ("rebuild", "top_up"):
            pool = self._rebuild_pool(
                request=request,
                session=session,
                user_state=user_state,
                shown_set=shown_set,
                catalog_version=catalog_version,
                retrieval_sig=retrieval_sig,
                key_family=key_family,
            )
        else:
            # reuse
            pool = existing_pool
            if self.pool_manager.should_rerank(session, pool):
                self.ranker.rerank_pool_from_meta(pool, session)
                self.session_store.save_pool(session_id, pool)

        # 8. Serve page (ID-only)
        page_ids = pool.next_page_ids(request.page_size)

        # 9. Hydrate page items
        page_candidates = self.hydrator.hydrate_ordered(page_ids)

        # 10. Serve-time validation
        # Pool items are exempt from shown_set: they were deduped at build time.
        # shown_set should only block items from *previous* pools, not same-pool pages.
        pool_id_set = set(pool.ordered_ids)
        validated = self._serve_time_validate(page_candidates, shown_set, session, pool, pool_id_set)

        # 11. Backfill if needed
        if len(validated) < request.page_size and pool.remaining > 0:
            validated = self._backfill_page(validated, pool, shown_set, session, request.page_size, pool_id_set)

        # 11b. Advance pool served_count so next page gets different items
        pool.served_count += len(page_ids)
        self.session_store.save_pool(session_id, pool)

        # 12. Update session
        served_ids = set()
        for c in validated:
            served_ids.add(c.item_id)
            brand_lower = (c.brand or "").lower()
            session.brand_exposure[brand_lower] += 1
            broad = c.broad_category or c.category or ""
            session.category_exposure[broad] += 1
            from recs.brand_clusters import get_cluster_for_item
            cluster = get_cluster_for_item(c.brand or "") or "unknown"
            session.cluster_exposure[cluster] += 1

        self.session_store.add_shown(session_id, served_ids)
        self.session_store.save_session(session_id, session)

        # 13. Build response
        debug_info = self._build_debug(decision, pool, validated, time.time() - t0) if request.debug else None
        cursor = pool.get_cursor()

        response = FeedResponse(
            user_id=request.user_id,
            session_id=session_id,
            mode=request.mode,
            items=validated,
            cursor=cursor,
            page=pool.current_page,
            pool_size=len(pool.ordered_ids),
            source_mix=pool.source_mix,
            debug=debug_info,
        )

        # 14. Enqueue impressions
        impression_items = [
            {"item_id": c.item_id, "position": i}
            for i, c in enumerate(validated)
        ]
        self.events.log_impressions(
            user_id=request.user_id,
            session_id=session_id,
            items=impression_items,
            source="feed",
        )

        elapsed = (time.time() - t0) * 1000
        logger.info(
            "V3 feed: user=%s session=%s mode=%s decision=%s page=%d items=%d pool=%d elapsed=%.0fms",
            request.user_id,
            session_id,
            request.mode,
            decision.action,
            pool.current_page,
            len(validated),
            len(pool.ordered_ids),
            elapsed,
        )

        return response

    def record_action(
        self,
        session_id: str,
        user_id: str,
        action: str,
        product_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a user action: update session + log event.

        Args:
            session_id: V3 session ID.
            user_id: Authenticated user ID.
            action: Action type (click, save, cart, purchase, skip, hide, etc.).
            product_id: Product UUID.
            metadata: Extra context (brand, cluster_id, article_type, position, etc.).
        """
        meta = metadata or {}
        session = self.session_store.get_or_create_session(session_id, user_id)
        session.record_action(
            action=action,
            item_id=product_id,
            brand=meta.get("brand"),
            cluster_id=meta.get("cluster_id"),
            article_type=meta.get("article_type"),
            metadata=meta,
        )
        self.session_store.save_session(session_id, session)
        self.events.log_action(
            user_id=user_id,
            session_id=session_id,
            action=action,
            product_id=product_id,
            source=meta.get("source", "feed"),
            position=meta.get("position"),
            metadata=meta,
        )

    def _rebuild_pool(
        self,
        request: FeedRequest,
        session: SessionProfile,
        user_state: Any,
        shown_set: Set[str],
        catalog_version: str,
        retrieval_sig: str,
        key_family: str,
    ) -> CandidatePool:
        """
        Full pool rebuild: sources → mix → hydrate → eligibility → rank → rerank → save.

        ~400–700ms on first page.
        """
        t0 = time.time()

        # 1. Run sources in parallel
        source_results = self._run_sources(user_state, session, request, shown_set)
        t_sources = time.time()

        # 2. Mix
        target_size = self.pool_manager.get_target_pool_size(request.mode)
        mixed_stubs = self.mixer.mix(source_results, target_size=target_size)
        t_mix = time.time()

        # 3. Hydrate
        stub_ids = [s.item_id for s in mixed_stubs]
        hydrated = self.hydrator.hydrate(stub_ids)
        t_hydrate = time.time()

        # 4. Eligibility
        stub_lookup = {s.item_id: s for s in mixed_stubs}
        elig_kwargs = self._build_eligibility_kwargs(user_state, session, shown_set, request)
        passed, elig_stats = self.eligibility.filter(hydrated, **elig_kwargs)
        t_elig = time.time()

        # 5. Rank
        is_warm = len(session.clicked_ids) >= 5
        onboarding = getattr(user_state, "onboarding_profile", None) if user_state else None
        ranked = self.ranker.rank(
            candidates=passed,
            user_profile=onboarding,
            session=session,
            is_warm=is_warm,
            eligibility_penalties=elig_stats.get("penalties"),
        )
        t_rank = time.time()

        # 6. Rerank
        reranked = self.reranker.rerank(
            candidates=ranked,
            target_size=target_size,
            seen_ids=shown_set,
        )
        t_rerank = time.time()

        # 7. Build pool
        pool = CandidatePool(
            session_id=request.session_id or session.session_id,
            mode=request.mode,
            ordered_ids=[c.item_id for c in reranked],
            scores={c.item_id: c.final_score for c in reranked},
            meta=self._build_meta(reranked, stub_lookup),
            served_count=0,
            retrieval_signature=retrieval_sig,
            catalog_version=catalog_version,
            source_mix=self.mixer.get_source_mix(source_results),
            last_rerank_action_seq=session.action_seq,
        )

        self.session_store.save_pool(session.session_id, pool)

        t_total = time.time()
        logger.info(
            "Pool rebuild: %d items (sources=%.0fms mix=%.0fms hydrate=%.0fms elig=%.0fms rank=%.0fms rerank=%.0fms total=%.0fms)",
            len(pool.ordered_ids),
            (t_sources - t0) * 1000,
            (t_mix - t_sources) * 1000,
            (t_hydrate - t_mix) * 1000,
            (t_elig - t_hydrate) * 1000,
            (t_rank - t_elig) * 1000,
            (t_rerank - t_rank) * 1000,
            (t_total - t0) * 1000,
        )

        return pool

    def _run_sources(
        self,
        user_state: Any,
        session: SessionProfile,
        request: FeedRequest,
        shown_set: Set[str],
    ) -> Dict[str, List[CandidateStub]]:
        """
        Run all retrieval sources in parallel using ThreadPoolExecutor.

        Returns dict mapping source name → list of CandidateStubs.
        """
        results: Dict[str, List[CandidateStub]] = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for name, source in self.sources.items():
                limit = self._get_source_limit(name, request)
                if limit > 0:
                    futures[executor.submit(source.retrieve, user_state, session, request, shown_set, limit)] = name

            for future in as_completed(futures, timeout=10):
                name = futures[future]
                try:
                    stubs = future.result()
                    results[name] = stubs
                except Exception as e:
                    logger.error("Source %s failed: %s", name, e)
                    results[name] = []

        return results

    @staticmethod
    def _get_source_limit(source_name: str, request: FeedRequest) -> int:
        """
        Retrieval limit per source, adjusted for brand-filtered requests.

        When include_brands is active (1-3 brands), exploration is pointless
        (can't discover other brands when the user explicitly wants specific
        brands). Redirect that budget to preference + session so we fill the
        pool with type/style-relevant items from the requested brand(s).
        """
        hf = getattr(request, "hard_filters", None)
        include_brands = getattr(hf, "include_brands", None) if hf else None

        if include_brands:
            brand_limits = dict(zip(
                ("preference", "session", "exploration", "merch"),
                (350, 100, 0, 50),
            ))
            return brand_limits.get(source_name, 0)

        limits = dict(zip(
            ("preference", "session", "exploration", "merch"),
            (225, 125, 75, 75),
        ))
        return limits.get(source_name, 0)

    def _serve_time_validate(
        self,
        candidates: List[Candidate],
        shown_set: Set[str],
        session: SessionProfile,
        pool: CandidatePool,
        pool_id_set: Optional[Set[str]] = None,
    ) -> List[Candidate]:
        """
        Revalidate page items before serving.

        Checks:
        1. Still in stock (item present in hydration result — missing = out of stock)
        2. Not hidden since pool build (check session.hidden_ids)
        3. Not shown in a *previous* pool (check shown_set, but exempt current pool items)
        4. No duplicate images in this page (image hash dedup)

        Items in pool_id_set bypass the shown_set check — they were already
        deduplicated during pool build. This prevents pages 2-N from being
        empty due to shown_set growing as pages are served.
        """
        validated = []
        page_image_hashes: Set[str] = set()
        _pool_ids = pool_id_set or set()
        neg_brands = session.explicit_negative_brands

        for c in candidates:
            # Check hidden
            if c.item_id in session.hidden_ids:
                continue
            # Check shown — but exempt items from the current pool
            if c.item_id in shown_set and c.item_id not in _pool_ids:
                continue
            # Check negative brands (formed mid-session from repeated hides)
            if neg_brands and c.brand:
                brand_lower = c.brand.lower()
                if any(nb.lower() in brand_lower or brand_lower in nb.lower() for nb in neg_brands):
                    continue
            # Image dedup within page
            img_hash = extract_image_hash(c.image_url)
            if img_hash and img_hash in page_image_hashes:
                continue
            if img_hash:
                page_image_hashes.add(img_hash)
            validated.append(c)

        return validated

    def _backfill_page(
        self,
        current: List[Candidate],
        pool: CandidatePool,
        shown_set: Set[str],
        session: SessionProfile,
        target_size: int,
        pool_id_set: Optional[Set[str]] = None,
    ) -> List[Candidate]:
        """
        When serve-time validation removes items, fill from next pool items.

        Fetches additional IDs from the pool beyond the current page,
        hydrates them, validates, and appends until we hit target_size.
        """
        current_ids = {c.item_id for c in current}
        needed = target_size - len(current)
        if needed <= 0:
            return current

        start = pool.served_count
        max_lookahead = min(needed * 3, len(pool.ordered_ids) - start)
        if max_lookahead <= 0:
            return current

        _pool_ids = pool_id_set or set()
        extra_ids = []
        for iid in pool.ordered_ids[start : start + max_lookahead]:
            if iid not in current_ids:
                # Only skip if shown in a previous pool (not this pool)
                if iid in shown_set and iid not in _pool_ids:
                    continue
                extra_ids.append(iid)
            if len(extra_ids) >= needed * 3:
                break

        if not extra_ids:
            return current

        candidate_ids = extra_ids
        extra_candidates = self.hydrator.hydrate_ordered(candidate_ids)
        extra_validated = self._serve_time_validate(extra_candidates, shown_set, session, pool, _pool_ids)

        # Merge with existing page image hashes
        page_image_hashes: Set[str] = set()
        for c in current:
            h = extract_image_hash(c.image_url)
            if h:
                page_image_hashes.add(h)

        for c in extra_validated:
            if len(current) >= target_size:
                break
            h = extract_image_hash(c.image_url)
            if h and h in page_image_hashes:
                continue
            if h:
                page_image_hashes.add(h)
            current.append(c)

        return current

    def _build_eligibility_kwargs(
        self,
        user_state: Any,
        session: SessionProfile,
        shown_set: Set[str],
        request: FeedRequest,
    ) -> Dict[str, Any]:
        """Extract eligibility filter parameters from user state, session, and request.

        Merges onboarding profile constraints with request-level hard filters.
        Request hard_filters (query params) take precedence: if both profile
        and request specify exclude_brands, the lists are merged (union).

        Also forwards soft_preferences (extended PA filters) directly to eligibility.
        """
        kwargs: Dict[str, Any] = {
            "hidden_ids": session.hidden_ids,
            "negative_brands": session.explicit_negative_brands,
            "shown_set": shown_set,
        }

        # User profile exclusions
        profile = getattr(user_state, "onboarding_profile", None) if user_state else None
        if profile:
            # Try get_all_exclusions method if available
            if callable(getattr(profile, "get_all_exclusions", None)):
                exclusions = profile.get_all_exclusions()
                kwargs["user_exclusions"] = list(exclusions)

            occasions = getattr(profile, "occasions", None)
            if occasions:
                kwargs["occasions"] = occasions

            exclude_colors = getattr(profile, "exclude_colors", None) or getattr(
                profile, "colors_to_avoid", None
            )
            if exclude_colors:
                kwargs["exclude_colors"] = exclude_colors

            exclude_brands = getattr(profile, "exclude_brands", None) or getattr(
                profile, "brands_to_avoid", None
            )
            if exclude_brands:
                kwargs["exclude_brands"] = exclude_brands

        # Request hard filters (take precedence / merge)
        hf = getattr(request, "hard_filters", None)
        if hf:
            # Merge exclude_brands (union of profile + request)
            req_exclude = getattr(hf, "exclude_brands", None)
            if req_exclude:
                existing = kwargs.get("exclude_brands") or []
                merged = list(set(existing) | set(req_exclude))
                kwargs["exclude_brands"] = merged

            # Include brands (whitelist)
            req_include = getattr(hf, "include_brands", None)
            if req_include:
                kwargs["include_brands"] = req_include

            # Merge exclude_colors (union of profile + request)
            req_exc_colors = getattr(hf, "exclude_colors", None)
            if req_exc_colors:
                existing = kwargs.get("exclude_colors") or []
                merged = list(set(existing) | set(req_exc_colors))
                kwargs["exclude_colors"] = merged

            # Forward core HardFilters fields to eligibility
            _hf_direct = {
                "exclude_styles": getattr(hf, "exclude_styles", None),
                "include_patterns": getattr(hf, "include_patterns", None),
                "exclude_patterns": getattr(hf, "exclude_patterns", None),
            }
            for k, v in _hf_direct.items():
                if v:
                    kwargs[k] = v

            # Forward include_occasions from HF (merge with profile occasions)
            req_occasions = getattr(hf, "include_occasions", None)
            if req_occasions:
                existing = kwargs.get("occasions") or []
                merged = list(set(existing) | set(req_occasions))
                kwargs["occasions"] = merged

        # Forward soft_preferences (extended PA filters) directly to eligibility
        # These are include_/exclude_ pairs parsed from query params by the API layer.
        soft_prefs = getattr(request, "soft_preferences", None)
        if soft_prefs and isinstance(soft_prefs, dict):
            for key, val in soft_prefs.items():
                if val:
                    kwargs[key] = val

        return kwargs

    @staticmethod
    def _build_meta(
        candidates: List[Candidate],
        stub_lookup: Dict[str, CandidateStub],
    ) -> Dict[str, ScoringMeta]:
        """Build ScoringMeta dict for pool storage."""
        meta: Dict[str, ScoringMeta] = {}
        for c in candidates:
            stub = stub_lookup.get(c.item_id)
            source = stub.source if stub else "unknown"
            retrieval_score = stub.retrieval_score if stub else 0.0
            cluster_id = stub.cluster_id if stub else ""
            meta[c.item_id] = ScoringMeta(
                source=source,
                retrieval_score=retrieval_score,
                brand=c.brand or "",
                cluster_id=cluster_id,
                broad_category=c.broad_category or "",
                article_type=c.article_type or "",
                price=c.price,
                image_dedup_key=getattr(c, "image_dedup_key", ""),
            )
        return meta

    @staticmethod
    def _build_debug(
        decision: PoolDecision,
        pool: CandidatePool,
        items: List[Candidate],
        elapsed: float,
    ) -> Dict[str, Any]:
        """Build debug payload for the response."""
        item_debug = []
        for i, c in enumerate(items):
            meta = pool.meta.get(c.item_id)
            item_debug.append({
                "item_id": c.item_id,
                "source": meta.source if meta else "unknown",
                "retrieval_score": round(meta.retrieval_score, 4) if meta else 0.0,
                "final_score": round(c.final_score, 4),
                "rank_position": pool.ordered_ids.index(c.item_id) if c.item_id in pool.ordered_ids else -1,
            })

        return {
            "pool_decision": {
                "action": decision.action,
                "reason": decision.reason,
                "remaining": decision.remaining,
            },
            "pool": {
                "size": len(pool.ordered_ids),
                "served_count": pool.served_count,
                "remaining": pool.remaining,
                "source_mix": pool.source_mix,
            },
            "elapsed_ms": round(elapsed * 1000, 1),
            "items": item_debug,
        }
