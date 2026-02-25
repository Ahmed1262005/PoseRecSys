"""
Hybrid Search Service: Algolia (lexical) + FashionCLIP (semantic).

Pipeline:
1. Classify query intent (exact / specific / vague)
2. Run Algolia search (always - handles typos, synonyms, exact matches)
3. Run FashionCLIP semantic search (skip for exact intent)
4. Merge results with Reciprocal Rank Fusion (RRF)
5. Apply session-aware reranking (profile boosts, dedup, diversity)
6. Log analytics
7. Return results
"""

import html as html_mod
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from core.logging import get_logger
from core.utils import filter_gallery_images
from config.settings import get_settings
from search.algolia_client import AlgoliaClient, get_algolia_client
from search.mode_config import get_rrf_weights
from search.query_planner import QueryPlanner, SearchPlan, get_query_planner
from search.reranker import SessionReranker
from search.analytics import SearchAnalytics, get_search_analytics
from search.algolia_config import get_replica_index_name
from search.models import (
    FacetValue,
    FollowUpQuestion,
    HybridSearchRequest,
    HybridSearchResponse,
    ProductResult,
    PaginationInfo,
    QueryIntent,
    SortBy,
)

logger = get_logger(__name__)


_FACET_FIELDS = [
    "brand", "category_l1", "broad_category", "article_type",
    "formality", "primary_color", "color_family", "pattern",
    "fit_type", "neckline", "sleeve_type", "length", "silhouette", "rise",
    "occasions", "seasons", "style_tags", "materials", "is_on_sale",
]


class HybridSearchService:
    """
    Main search service combining Algolia + FashionCLIP.
    """

    def __init__(
        self,
        algolia_client: Optional[AlgoliaClient] = None,
        analytics: Optional[SearchAnalytics] = None,
    ):
        self._algolia = algolia_client
        self._analytics = analytics
        self._reranker = SessionReranker()
        self._planner = get_query_planner()

        # Lazy-load FashionCLIP search engine
        self._semantic_engine = None

    @property
    def algolia(self) -> AlgoliaClient:
        if self._algolia is None:
            self._algolia = get_algolia_client()
        return self._algolia

    @property
    def analytics(self) -> SearchAnalytics:
        if self._analytics is None:
            self._analytics = get_search_analytics()
        return self._analytics

    @property
    def semantic_engine(self):
        """Lazy-load WomenSearchEngine for FashionCLIP semantic search."""
        if self._semantic_engine is None:
            from women_search_engine import get_women_search_engine
            self._semantic_engine = get_women_search_engine()
        return self._semantic_engine

    # =========================================================================
    # Main Search
    # =========================================================================

    def search(
        self,
        request: HybridSearchRequest,
        user_id: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        seen_ids: Optional[Set[str]] = None,
        user_context: Optional[Any] = None,
        session_scores: Optional[Any] = None,
        skip_planner: bool = False,
        planner_context: Optional[Dict[str, Any]] = None,
    ) -> HybridSearchResponse:
        """
        Execute hybrid search.

        Args:
            request: Search request with query, filters, pagination.
            user_id: Authenticated user ID.
            user_profile: User's onboarding profile (for reranking).
            seen_ids: Already-shown product IDs (for session dedup).
            user_context: UserContext for age-affinity + weather scoring.
            session_scores: Live SessionScores for session-aware reranking.
            skip_planner: If True, skip the LLM planner (filters pre-resolved).
            planner_context: Optional compact dict with user profile info for
                personalized follow-ups (passed through to QueryPlanner.plan()).

        Returns:
            HybridSearchResponse with results and metadata.
        """
        t_start = time.time()
        timing: Dict[str, Any] = {}

        # Step 0: Normalize query (decode HTML entities, collapse whitespace)
        clean_query = html_mod.unescape(request.query).strip()
        if clean_query != request.query:
            request = request.model_copy(update={"query": clean_query})

        # -----------------------------------------------------------------
        # SORT-MODE FAST PATH
        # When sort_by is not "relevance", use Algolia-only via a virtual
        # replica index. Skip semantic search, RRF merge, and reranker —
        # they would break the deterministic sort order.
        # -----------------------------------------------------------------
        if request.sort_by != SortBy.RELEVANCE:
            return self._search_sorted(request, user_id)

        # =====================================================================
        # STEP 1: Query Understanding (LLM Planner with regex fallback)
        # =====================================================================
        # The LLM planner decomposes the query into:
        # - Structured filters (hard Algolia facets)
        # - Optimized Algolia keyword query
        # - Rich semantic query for FashionCLIP
        # - Expanded filter values for lenient post-filtering
        # Falls back to regex-based extraction if planner is disabled/fails.
        # =====================================================================

        search_plan: Optional[SearchPlan] = None
        expanded_filters: Dict[str, List[str]] = {}
        semantic_query_override: Optional[str] = None
        semantic_queries: Optional[List[str]] = None
        name_exclusions: List[str] = []
        follow_ups: Optional[List[FollowUpQuestion]] = None

        t_plan = time.time()
        if not skip_planner:
            search_plan = self._planner.plan(
                request.query,
                user_context=planner_context,
            )
        timing["planner_ms"] = int((time.time() - t_plan) * 1000)

        if search_plan is not None:
            # --- LLM planner succeeded ---
            (
                plan_updates,
                expanded_filters,
                exclude_filters_from_plan,
                matched_terms,
                algolia_query,
                semantic_query_override,
                intent_str,
                semantic_queries,
            ) = self._planner.plan_to_request_updates(search_plan)

            # Map intent string to enum
            _INTENT_MAP = {"exact": QueryIntent.EXACT, "specific": QueryIntent.SPECIFIC, "vague": QueryIntent.VAGUE}
            intent = _INTENT_MAP.get(intent_str, QueryIntent.SPECIFIC)

            # Get RRF weights for the intent
            algolia_weight, semantic_weight = get_rrf_weights(intent_str)

            # Extract internal pipeline keys (not real request fields)
            name_exclusions = plan_updates.pop("_name_exclusions", [])
            mode_excl_keys = plan_updates.pop("_mode_excl_keys", [])

            # ---------------------------------------------------------
            # VAGUE queries: skip ALL planner attribute filters.
            #
            # SPECIFIC with modes: strip mode-derived exclude_* fields.
            # Composite modes like "modest" generate 20+ exclusion
            # filters that together eliminate nearly all products on
            # fast-fashion catalogs.  The semantic queries already
            # encode the intent ("modest midi dress with long sleeves")
            # so hard-excluding V-Neck/Mini/Fitted/etc is redundant
            # and destructive.  Name exclusions still catch egregious
            # violations (e.g. "backless" in product name).
            #
            # EXACT: apply filters normally.
            # ---------------------------------------------------------
            if intent_str == "vague":
                updates = {}
                logger.info(
                    "Vague query — skipping planner attribute filters (follow-ups will narrow)",
                    query=request.query,
                    skipped_filters=plan_updates,
                    semantic_queries=semantic_queries,
                )
            else:
                # For SPECIFIC intent, strip mode-derived exclusion filters
                if intent_str == "specific" and mode_excl_keys:
                    stripped = {}
                    for key in mode_excl_keys:
                        if key in plan_updates:
                            stripped[key] = plan_updates.pop(key)
                    if stripped:
                        logger.info(
                            "SPECIFIC intent: stripped mode exclusion filters",
                            query=request.query,
                            stripped_filters=list(stripped.keys()),
                        )

                # Apply remaining planner filters
                updates = {}
                for field, values in plan_updates.items():
                    if not getattr(request, field, None):
                        updates[field] = values
                if updates:
                    request = request.model_copy(update=updates)
                    logger.info(
                        "LLM planner applied filters",
                        query=request.query,
                        filters=updates,
                        exclude_filters=exclude_filters_from_plan,
                        expanded=expanded_filters,
                        algolia_query=algolia_query,
                        semantic_query=semantic_query_override,
                    )

            # Extract follow-up questions from the plan (if any)
            if search_plan.parsed_follow_ups:
                follow_ups = search_plan.parsed_follow_ups

            # Stash plan details in timing for debugging/HTML reports
            timing["plan_modes"] = search_plan.modes
            timing["plan_attributes"] = search_plan.attributes
            timing["plan_avoid"] = search_plan.avoid
            timing["plan_algolia_query"] = algolia_query
            timing["plan_semantic_query"] = semantic_query_override
            timing["plan_semantic_queries"] = semantic_queries
            timing["plan_applied_filters"] = updates
            timing["plan_exclusions"] = exclude_filters_from_plan
            if name_exclusions:
                timing["plan_name_exclusions"] = name_exclusions
        else:
            # --- Fallback: basic search (no regex, no filter extraction) ---
            logger.info("Planner unavailable, using basic search", query=request.query)

            intent = QueryIntent.SPECIFIC
            algolia_weight, semantic_weight = get_rrf_weights("specific")

            # Use raw query for Algolia (clean meta-terms only, no extracted terms)
            algolia_query = self._clean_query_for_algolia(request.query)

        # Allow request-level override of semantic weight.
        _SEMANTIC_BOOST_DEFAULT = 0.4
        if abs(request.semantic_boost - _SEMANTIC_BOOST_DEFAULT) > 1e-9:
            semantic_weight = request.semantic_boost
            algolia_weight = 1.0 - semantic_weight

        # Step 2c: Build Algolia filter strings.
        # When the planner is active, split into hard filters (brand, price,
        # stock, category) and optional filters (neckline, color, fit, etc.)
        # so Algolia returns a wider candidate pool where keyword terms like
        # "ribbed" can match, while still boosting attribute-matching results.
        algolia_optional_filters: Optional[List[str]] = None
        if search_plan is not None:
            algolia_filters, algolia_optional_filters = self._build_algolia_filters_split(request)
            if algolia_optional_filters:
                logger.info(
                    "Using optionalFilters for Algolia (planner active)",
                    hard_filters=algolia_filters,
                    optional_count=len(algolia_optional_filters),
                )
        else:
            algolia_filters = self._build_algolia_filters(request)

        # Step 3+4: Run Algolia + Semantic search
        # For SPECIFIC/VAGUE intent, both searches are independent so we run
        # them in parallel.  For EXACT intent, semantic only runs if Algolia
        # returns 0, so we keep those sequential.
        #
        # Auto-detect active filters from the request model.
        _NON_FILTER_FIELDS = {"query", "page", "page_size", "session_id", "semantic_boost", "sort_by", "planner_context"}
        has_filters = any(
            getattr(request, field_name) not in (None, False, [])
            for field_name in HybridSearchRequest.model_fields
            if field_name not in _NON_FILTER_FIELDS
        )
        # Fetch extra semantic candidates when filters are active,
        # since strict post-filtering will drop non-matching results
        fetch_multiplier = 5 if has_filters else 3
        fetch_size = request.page_size * fetch_multiplier

        # Pre-build the semantic search parameters (needed for both paths)
        semantic_results = []
        clip_query = semantic_query_override if semantic_query_override else request.query
        _queries_to_run = (
            semantic_queries
            if semantic_queries and len(semantic_queries) > 1
            else [clip_query]
        )

        # Build a relaxed request for semantic search when planner is active.
        if search_plan is not None:
            semantic_request = request.model_copy(update={
                "categories": None,
                "colors": None,
                "patterns": None,
                "occasions": None,
            })
            logger.info(
                "Using relaxed filters for semantic search (planner active)",
                kept=["price", "brands", "exclude_brands", "category_l1", "category_l2"],
                removed=["categories", "colors", "patterns", "occasions"],
            )
        else:
            semantic_request = request

        def _run_algolia() -> tuple:
            """Run Algolia search, return (results, facets, elapsed_ms)."""
            t0 = time.time()
            results, fcts = self._search_algolia(
                query=algolia_query,
                filters=algolia_filters,
                hits_per_page=fetch_size,
                facets=_FACET_FIELDS,
                optional_filters=algolia_optional_filters,
            )
            return results, fcts, int((time.time() - t0) * 1000)

        def _run_semantic() -> tuple:
            """Run semantic search, return (results, query_count, elapsed_ms)."""
            t0 = time.time()
            if len(_queries_to_run) > 1:
                results = self._search_semantic_multi(
                    queries=_queries_to_run,
                    request=semantic_request,
                    limit_per_query=max(fetch_size // len(_queries_to_run), 30),
                )
                qcount = len(_queries_to_run)
            else:
                results = self._search_semantic(
                    query=_queries_to_run[0],
                    request=semantic_request,
                    limit=fetch_size,
                )
                qcount = 1
            return results, qcount, int((time.time() - t0) * 1000)

        if intent == QueryIntent.EXACT:
            # EXACT: run Algolia first; only run semantic if Algolia returns 0
            algolia_results, facets, algolia_ms = _run_algolia()
            timing["algolia_ms"] = algolia_ms

            if len(algolia_results) == 0:
                semantic_results, sq_count, semantic_ms = _run_semantic()
                timing["semantic_ms"] = semantic_ms
                timing["semantic_query_count"] = sq_count
                if semantic_results:
                    logger.info(
                        "Algolia returned 0 results for EXACT query, fell back to semantic",
                        query=request.query,
                        semantic_count=len(semantic_results),
                    )
        else:
            # SPECIFIC / VAGUE: run Algolia + Semantic in parallel
            from concurrent.futures import ThreadPoolExecutor, as_completed

            algolia_results, facets = [], {}
            algolia_ms = 0
            semantic_ms = 0
            sq_count = 1

            with ThreadPoolExecutor(max_workers=2) as executor:
                future_algolia = executor.submit(_run_algolia)
                future_semantic = executor.submit(_run_semantic)

                try:
                    algolia_results, facets, algolia_ms = future_algolia.result()
                except Exception as e:
                    logger.warning("Algolia search failed in parallel", error=str(e))

                try:
                    semantic_results, sq_count, semantic_ms = future_semantic.result()
                except Exception as e:
                    logger.warning("Semantic search failed in parallel", error=str(e))

            timing["algolia_ms"] = algolia_ms
            timing["semantic_ms"] = semantic_ms
            timing["semantic_query_count"] = sq_count

            if len(algolia_results) == 0 and semantic_results:
                logger.info(
                    "Algolia returned 0 results, falling back to semantic-only",
                    query=request.query,
                    semantic_count=len(semantic_results),
                )

        # Step 4b: Enrich semantic results with Gemini attributes from Algolia
        if semantic_results:
            semantic_results = self._enrich_semantic_results(semantic_results)

        # Step 4c: Post-filter semantic results to enforce filters Algolia handles.
        # When the LLM planner provided expanded_filters, use lenient matching
        # so visually-correct results aren't dropped due to label mismatches
        # (e.g. "Tropical" pattern on a leaf-print jacket when filter is "Floral").
        #
        # Progressive relaxation: if post-filtering drops ALL semantic results
        # to 0, retry with progressively looser exclusion settings:
        #   1. Strict (default): exclusion filters drop matching + null items
        #   2. Lenient nulls: exclusion filters keep null/N/A items
        #   3. No exclusions: skip exclusion filters entirely
        # Uses saved copy of enriched results to avoid re-fetching from DB.
        relaxation_level = None
        if semantic_results:
            # Save enriched results before filtering so we can retry without
            # re-fetching from pgvector (expensive network call).
            enriched_snapshot = list(semantic_results)
            pre_filter_count = len(enriched_snapshot)

            semantic_results = self._post_filter_semantic(
                enriched_snapshot, request, expanded_filters=expanded_filters,
            )

            # Check if we have active exclusion filters that could be relaxed
            _EXCLUDE_FIELDS = (
                "exclude_neckline", "exclude_sleeve_type", "exclude_length",
                "exclude_fit_type", "exclude_silhouette", "exclude_patterns",
                "exclude_colors", "exclude_materials", "exclude_occasions",
                "exclude_seasons", "exclude_formality", "exclude_rise",
                "exclude_style_tags",
            )
            has_exclusions = any(getattr(request, f, None) for f in _EXCLUDE_FIELDS)

            if len(semantic_results) == 0 and has_exclusions and pre_filter_count > 0:
                # Retry 1: Keep exclusion filters but stop dropping null-valued items.
                # Many products have missing attribute data (N/A); strict mode
                # drops these, which can eliminate too many candidates.
                logger.info(
                    "Progressive relaxation: retry with lenient nulls",
                    pre_filter_count=pre_filter_count,
                )
                semantic_results = self._post_filter_semantic(
                    list(enriched_snapshot), request,
                    expanded_filters=expanded_filters,
                    drop_nulls=False,
                )
                relaxation_level = "lenient_nulls"

            if len(semantic_results) == 0 and has_exclusions and pre_filter_count > 0:
                # Retry 2: Skip exclusion filters entirely — the user's query
                # was too restrictive for the catalog. Return relevant results
                # even if some excluded attributes might be present.
                logger.info(
                    "Progressive relaxation: retry without exclusion filters",
                    pre_filter_count=pre_filter_count,
                )
                relaxed_request = request.model_copy(update={
                    f: None for f in _EXCLUDE_FIELDS
                })
                semantic_results = self._post_filter_semantic(
                    list(enriched_snapshot), relaxed_request,
                    expanded_filters=expanded_filters,
                )
                relaxation_level = "no_exclusions"

            if relaxation_level:
                logger.info(
                    "Progressive relaxation recovered results",
                    relaxation_level=relaxation_level,
                    recovered=len(semantic_results),
                    pre_filter_count=pre_filter_count,
                )

        # Record relaxation in timing metadata so callers can see it
        if relaxation_level:
            timing["relaxation_level"] = relaxation_level

        # Step 5: Merge with RRF
        merged = self._reciprocal_rank_fusion(
            algolia_results=algolia_results,
            semantic_results=semantic_results,
            algolia_weight=algolia_weight,
            semantic_weight=semantic_weight,
        )

        # Step 5b: Apply exclusion filters on ALL merged results.
        # Semantic results were already filtered in step 4c, but Algolia results
        # only had NOT clauses in the filter string — which can miss products
        # with null/missing attributes (Algolia NOT clauses don't exclude nulls).
        # Use drop_nulls=True: if the user explicitly excludes certain attribute
        # values, products with unknown/missing attributes are suspect and should
        # be excluded from the final results.
        has_any_exclusion = any(
            getattr(request, f, None)
            for f in (
                "exclude_neckline", "exclude_sleeve_type", "exclude_length",
                "exclude_fit_type", "exclude_silhouette", "exclude_patterns",
                "exclude_colors", "exclude_materials", "exclude_occasions",
                "exclude_seasons", "exclude_formality", "exclude_rise",
                "exclude_style_tags",
            )
        )
        if has_any_exclusion and merged:
            pre_count = len(merged)
            strict_merged = self._apply_exclusion_filters(merged, request, drop_nulls=True)
            # If strict filtering removes too many results (below page_size),
            # fall back to lenient mode that keeps null-attribute products
            min_needed = request.page_size
            if len(strict_merged) >= min_needed:
                merged = strict_merged
                if len(merged) < pre_count:
                    logger.info(
                        "Post-merge exclusion filter (strict) caught items",
                        before=pre_count,
                        after=len(merged),
                        dropped=pre_count - len(merged),
                    )
            else:
                lenient_merged = self._apply_exclusion_filters(merged, request, drop_nulls=False)
                if len(lenient_merged) >= min_needed:
                    merged = lenient_merged
                    logger.info(
                        "Post-merge exclusion filter fell back to lenient mode",
                        strict_count=len(strict_merged),
                        lenient_count=len(lenient_merged),
                        before=pre_count,
                    )
                else:
                    # Even lenient mode has too few — use strict anyway
                    # (better to show fewer correct results than wrong ones)
                    merged = strict_merged
                    logger.info(
                        "Post-merge exclusion filter: strict mode low but lenient not better",
                        strict_count=len(strict_merged),
                        lenient_count=len(lenient_merged),
                        before=pre_count,
                    )

        # Step 5c: Name-based exclusion filter.
        # Catches products where the product name says "backless" or "open back"
        # but the style_tags attribute is empty/null (data quality gap).
        # Applied AFTER attribute-based exclusion, so it's a safety net.
        if name_exclusions and merged:
            pre_name_count = len(merged)
            merged = [
                r for r in merged
                if not any(
                    excl in (r.get("name") or "").lower()
                    for excl in name_exclusions
                )
            ]
            if len(merged) < pre_name_count:
                logger.info(
                    "Name-based exclusion filter caught items",
                    before=pre_name_count,
                    after=len(merged),
                    dropped=pre_name_count - len(merged),
                    patterns=name_exclusions,
                )

        # Step 6: Rerank with session/profile + context scoring
        # For EXACT brand queries, disable brand diversity cap — the user
        # explicitly searched for that brand and expects all results from it.
        brand_cap = 0 if intent == QueryIntent.EXACT else None
        rerank_kwargs: Dict[str, Any] = dict(
            results=merged,
            user_profile=user_profile,
            seen_ids=seen_ids,
            user_context=user_context,
            session_scores=session_scores,
        )
        if brand_cap is not None:
            rerank_kwargs["max_per_brand"] = brand_cap
        merged = self._reranker.rerank(**rerank_kwargs)

        # Step 6b: Category diversity for VAGUE queries (post-reranker).
        # Semantic search tends to cluster results in one category (e.g. all
        # pants) because FashionCLIP matches visual features, not garment type.
        # For vague queries where no hard category filter was set, interleave
        # results round-robin by category_l1 so the top results show a mix
        # of garment types (dresses, tops, bottoms, etc.) rather than a wall
        # of one category.  Within each category, items keep their reranked
        # order.  Applied AFTER the reranker so scoring doesn't undo diversity.
        if intent == QueryIntent.VAGUE and not request.category_l1 and merged:
            cat_buckets: Dict[str, List[dict]] = {}
            for item in merged:
                cat = item.get("category_l1") or item.get("broad_category") or "Other"
                cat_buckets.setdefault(cat, []).append(item)

            if len(cat_buckets) > 1:
                # Sort buckets: largest first so round-robin starts with
                # the most populated category (feels natural).
                sorted_cats = sorted(cat_buckets.keys(), key=lambda c: -len(cat_buckets[c]))
                interleaved: List[dict] = []
                max_bucket = max(len(b) for b in cat_buckets.values())
                for pos in range(max_bucket):
                    for cat in sorted_cats:
                        if pos < len(cat_buckets[cat]):
                            interleaved.append(cat_buckets[cat][pos])

                before_cats = {c: len(b) for c, b in cat_buckets.items()}
                logger.info(
                    "Category diversity interleave applied (vague query)",
                    categories=before_cats,
                    total=len(interleaved),
                )
                merged = interleaved

        # Step 7: Paginate
        start_idx = (request.page - 1) * request.page_size
        page_results = merged[start_idx : start_idx + request.page_size]
        has_more = len(merged) > start_idx + request.page_size

        # Step 8: Format response
        products = [self._to_product_result(r, idx + start_idx + 1) for idx, r in enumerate(page_results)]

        timing["total_ms"] = int((time.time() - t_start) * 1000)

        # Step 9: Log analytics (fire-and-forget)
        try:
            self.analytics.log_search(
                query=request.query,
                intent=intent.value,
                total_results=len(merged),
                algolia_results=len(algolia_results),
                semantic_results=len(semantic_results),
                filters=self._extract_filter_summary(request),
                latency_ms=timing.get("total_ms", 0),
                algolia_latency_ms=timing.get("algolia_ms", 0),
                semantic_latency_ms=timing.get("semantic_ms", 0),
                user_id=user_id,
                session_id=request.session_id,
            )
        except Exception as e:
            logger.warning("Failed to log search analytics", error=str(e))

        return HybridSearchResponse(
            query=request.query,
            intent=intent.value,
            sort_by=request.sort_by.value,
            results=products,
            pagination=PaginationInfo(
                page=request.page,
                page_size=request.page_size,
                has_more=has_more,
                total_results=len(merged),
            ),
            timing=timing,
            facets=facets,
            follow_ups=follow_ups,
        )

    # =========================================================================
    # Sorted Search (Algolia-only via virtual replicas)
    # =========================================================================

    def _search_sorted(
        self,
        request: HybridSearchRequest,
        user_id: Optional[str] = None,
    ) -> HybridSearchResponse:
        """
        Algolia-only search path for non-relevance sort orders.

        Uses a virtual replica index that sorts by the requested field
        (price, trending, etc.). Skips semantic search, RRF merge, and
        reranker to preserve the deterministic sort order.

        Algolia handles pagination natively — no in-memory slicing needed.
        """
        t_start = time.time()
        timing: Dict[str, int] = {}

        # Resolve the replica index name
        replica_index = get_replica_index_name(
            self.algolia.index_name, request.sort_by.value,
        )
        if not replica_index:
            # Should never happen — relevance is handled by the main path
            logger.warning("No replica for sort_by=%s, falling back to relevance", request.sort_by)
            replica_index = self.algolia.index_name

        # Use LLM planner for query understanding (brand, filters, query optimization)
        intent = QueryIntent.SPECIFIC  # default
        algolia_query = request.query

        t_plan = time.time()
        search_plan = self._planner.plan(request.query)
        timing["planner_ms"] = int((time.time() - t_plan) * 1000)

        if search_plan is not None:
            (
                plan_updates, _, _, _,
                algolia_query_from_plan, _, intent_str, _,
            ) = self._planner.plan_to_request_updates(search_plan)

            _INTENT_MAP = {"exact": QueryIntent.EXACT, "specific": QueryIntent.SPECIFIC, "vague": QueryIntent.VAGUE}
            intent = _INTENT_MAP.get(intent_str, QueryIntent.SPECIFIC)

            # Apply planner filters (only fields the user hasn't set explicitly)
            updates = {}
            for field, values in plan_updates.items():
                if not getattr(request, field, None):
                    updates[field] = values
            if updates:
                request = request.model_copy(update=updates)
                logger.info(
                    "LLM planner applied filters for sorted search",
                    query=request.query,
                    filters=updates,
                    sort_by=request.sort_by.value,
                )

            if algolia_query_from_plan:
                algolia_query = algolia_query_from_plan
        else:
            # Planner failed — use raw query with meta-term cleaning only
            algolia_query = self._clean_query_for_algolia(request.query)

        algolia_filters = self._build_algolia_filters(request)

        # Search the replica with Algolia-native pagination
        t_algolia = time.time()
        try:
            resp = self.algolia.search(
                query=algolia_query,
                filters=algolia_filters,
                hits_per_page=request.page_size,
                page=request.page - 1,  # Algolia is 0-indexed, API is 1-indexed
                facets=_FACET_FIELDS,
                index_name=replica_index,
            )
        except Exception as e:
            logger.error(
                "Sorted search failed",
                replica=replica_index,
                sort_by=request.sort_by.value,
                error=str(e),
            )
            resp = {"hits": [], "nbHits": 0, "nbPages": 0}
        timing["algolia_ms"] = int((time.time() - t_algolia) * 1000)

        # Normalize hits to result dicts (same format as _search_algolia)
        results = []
        for hit in resp.get("hits", []):
            results.append({
                "product_id": hit.get("objectID"),
                "name": hit.get("name", ""),
                "brand": hit.get("brand", ""),
                "image_url": hit.get("image_url"),
                "gallery_images": filter_gallery_images(hit.get("gallery_images") or []),
                "price": float(hit.get("price", 0) or 0),
                "original_price": hit.get("original_price"),
                "is_on_sale": hit.get("is_on_sale", False),
                "category_l1": hit.get("category_l1"),
                "category_l2": hit.get("category_l2"),
                "broad_category": hit.get("broad_category"),
                "article_type": hit.get("article_type"),
                "primary_color": hit.get("primary_color"),
                "color_family": hit.get("color_family"),
                "pattern": hit.get("pattern"),
                "apparent_fabric": hit.get("apparent_fabric"),
                "fit_type": hit.get("fit_type"),
                "formality": hit.get("formality"),
                "silhouette": hit.get("silhouette"),
                "length": hit.get("length"),
                "neckline": hit.get("neckline"),
                "sleeve_type": hit.get("sleeve_type"),
                "rise": hit.get("rise"),
                "style_tags": hit.get("style_tags") or [],
                "occasions": hit.get("occasions") or [],
                "seasons": hit.get("seasons") or [],
                "source": "algolia",
            })

        # Parse facets (same logic as _search_algolia)
        parsed_facets: Optional[Dict[str, List[FacetValue]]] = None
        raw_facets = resp.get("facets")
        if raw_facets:
            parsed_facets = {}
            for facet_name, value_counts in raw_facets.items():
                values = [
                    FacetValue(value=val, count=cnt)
                    for val, cnt in sorted(value_counts.items(), key=lambda x: -x[1])
                    if cnt > 1 and val and val.lower() not in ("null", "n/a", "none", "")
                ]
                if len(values) >= 2:
                    parsed_facets[facet_name] = values

        # Build response
        total_hits = resp.get("nbHits", len(results))
        nb_pages = resp.get("nbPages", 1)
        has_more = request.page < nb_pages

        products = [self._to_product_result(r, idx + 1) for idx, r in enumerate(results)]
        timing["total_ms"] = int((time.time() - t_start) * 1000)

        # Analytics (fire-and-forget)
        try:
            self.analytics.log_search(
                query=request.query,
                intent=intent.value,
                total_results=total_hits,
                algolia_results=len(results),
                semantic_results=0,
                filters=self._extract_filter_summary(request),
                latency_ms=timing.get("total_ms", 0),
                algolia_latency_ms=timing.get("algolia_ms", 0),
                semantic_latency_ms=0,
                user_id=user_id,
                session_id=request.session_id,
            )
        except Exception as e:
            logger.warning("Failed to log sorted search analytics", error=str(e))

        return HybridSearchResponse(
            query=request.query,
            intent=intent.value,
            sort_by=request.sort_by.value,
            results=products,
            pagination=PaginationInfo(
                page=request.page,
                page_size=request.page_size,
                has_more=has_more,
                total_results=total_hits,
            ),
            timing=timing,
            facets=parsed_facets,
        )

    # =========================================================================
    # Algolia Search
    # =========================================================================

    def _search_algolia(
        self,
        query: str,
        filters: Optional[str] = None,
        hits_per_page: int = 100,
        facets: Optional[List[str]] = None,
        optional_filters: Optional[List[str]] = None,
    ) -> Tuple[List[dict], Optional[Dict[str, List[FacetValue]]]]:
        """Search Algolia and return normalized results + facet counts.

        Returns:
            Tuple of (results list, facets dict or None).
        """
        try:
            resp = self.algolia.search(
                query=query,
                filters=filters,
                optional_filters=optional_filters,
                hits_per_page=hits_per_page,
                page=0,
                facets=facets,
            )
            results = []
            for hit in resp.get("hits", []):
                results.append({
                    "product_id": hit.get("objectID"),
                    "name": hit.get("name", ""),
                    "brand": hit.get("brand", ""),
                    "image_url": hit.get("image_url"),
                    "gallery_images": filter_gallery_images(hit.get("gallery_images") or []),
                    "price": float(hit.get("price", 0) or 0),
                    "original_price": hit.get("original_price"),
                    "is_on_sale": hit.get("is_on_sale", False),
                    "category_l1": hit.get("category_l1"),
                    "category_l2": hit.get("category_l2"),
                    "broad_category": hit.get("broad_category"),
                    "article_type": hit.get("article_type"),
                    "primary_color": hit.get("primary_color"),
                    "color_family": hit.get("color_family"),
                    "pattern": hit.get("pattern"),
                    "apparent_fabric": hit.get("apparent_fabric"),
                    "fit_type": hit.get("fit_type"),
                    "formality": hit.get("formality"),
                    "silhouette": hit.get("silhouette"),
                    "length": hit.get("length"),
                    "neckline": hit.get("neckline"),
                    "sleeve_type": hit.get("sleeve_type"),
                    "rise": hit.get("rise"),
                    "style_tags": hit.get("style_tags") or [],
                    "occasions": hit.get("occasions") or [],
                    "seasons": hit.get("seasons") or [],
                    "colors": hit.get("colors") or [],
                    "materials": hit.get("materials") or [],
                    "trending_score": hit.get("trending_score", 0),
                    "source": "algolia",
                })

            # Parse facet counts.
            # Only include a facet if it has 2+ distinct valid values
            # (a facet with only 1 value means every result is the same —
            # not useful as a filter option for the frontend).
            parsed_facets: Optional[Dict[str, List[FacetValue]]] = None
            raw_facets = resp.get("facets")
            if raw_facets:
                parsed_facets = {}
                for facet_name, value_counts in raw_facets.items():
                    values = [
                        FacetValue(value=val, count=cnt)
                        for val, cnt in sorted(value_counts.items(), key=lambda x: -x[1])
                        if cnt > 1 and val and val.lower() not in ("null", "n/a", "none", "")
                    ]
                    # Need at least 2 values to be a useful filter
                    if len(values) >= 2:
                        parsed_facets[facet_name] = values

            return results, parsed_facets
        except Exception as e:
            logger.error("Algolia search failed", error=str(e))
            return [], None

    # =========================================================================
    # FashionCLIP Semantic Search
    # =========================================================================

    def _search_semantic(
        self,
        query: str,
        request: HybridSearchRequest,
        limit: int = 100,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[dict]:
        """
        Search using FashionCLIP + pgvector and return normalized results.

        Tries multimodal embeddings first (combined image + text vectors),
        which can match descriptive terms like "ribbed" or "quilted" that
        exist in product text but not in images. Falls back to image-only
        embeddings if multimodal search returns no results or is disabled.

        Args:
            query: Text query string.
            request: Full search request with filters.
            limit: Max results.
            query_embedding: Pre-computed FashionCLIP embedding (skips encode_text).
        """
        settings = get_settings()

        # Try multimodal search first
        if settings.multimodal_search_enabled:
            multimodal_results = self._search_multimodal(
                query=query, request=request, limit=limit,
                version=settings.multimodal_embedding_version,
                query_embedding=query_embedding,
            )
            if multimodal_results:
                return multimodal_results
            # Fall through to image-only if multimodal returned nothing
            logger.info(
                "Multimodal search returned 0 results, falling back to image-only",
                query=query,
            )

        # Image-only fallback (original path)
        return self._search_image_only(
            query=query, request=request, limit=limit,
            query_embedding=query_embedding,
        )

    def _search_semantic_multi(
        self,
        queries: List[str],
        request: HybridSearchRequest,
        limit_per_query: int = 50,
    ) -> List[dict]:
        """
        Run multiple diverse semantic queries in parallel and merge results.

        Each query targets a different visual angle (garment type, style,
        silhouette, etc.) to produce wider variety than a single query which
        tends to return visually-clustered results.

        Batch-encodes all query embeddings upfront in a single forward pass,
        then runs only the pgvector RPC calls in parallel (skipping redundant
        per-thread FashionCLIP encode_text calls).

        Dedup by product_id: if the same product appears in multiple query
        results, keep the one with the highest semantic_score.

        Returns a single merged list, interleaved round-robin by query to
        ensure diversity in the top results, then sorted by score.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time as _time

        # Batch-encode ALL query embeddings in a single forward pass.
        # This replaces N separate encode_text() calls with one batch call.
        t0 = _time.perf_counter()
        embeddings = self.semantic_engine.encode_text_batch(queries)
        encode_ms = (_time.perf_counter() - t0) * 1000
        logger.info(
            "Batch-encoded semantic queries",
            query_count=len(queries),
            encode_ms=round(encode_ms, 1),
        )

        results_per_query: List[List[dict]] = [[] for _ in queries]

        def _run_query(idx: int, query: str) -> tuple:
            results = self._search_semantic(
                query=query, request=request, limit=limit_per_query,
                query_embedding=embeddings[idx],
            )
            return idx, results

        # Run all pgvector RPC queries in parallel (embeddings already computed)
        with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as executor:
            futures = {
                executor.submit(_run_query, i, q): i
                for i, q in enumerate(queries)
            }
            for future in as_completed(futures):
                try:
                    idx, results = future.result()
                    results_per_query[idx] = results
                except Exception as e:
                    logger.warning(
                        "Multi-query semantic search failed for one query",
                        error=str(e),
                    )

        # Log per-query counts
        counts = [len(r) for r in results_per_query]
        logger.info(
            "Multi-query semantic search completed",
            query_count=len(queries),
            per_query_counts=counts,
            total_before_dedup=sum(counts),
        )

        # Merge + dedup: keep highest semantic_score per product_id.
        # Track which query each product came from for interleaving.
        seen: Dict[str, dict] = {}  # product_id -> best result dict
        query_assignments: Dict[str, int] = {}  # product_id -> query index

        for q_idx, results in enumerate(results_per_query):
            for item in results:
                pid = item.get("product_id")
                if not pid:
                    continue
                existing = seen.get(pid)
                if existing is None:
                    seen[pid] = item
                    query_assignments[pid] = q_idx
                else:
                    # Keep the one with the higher semantic score
                    if (item.get("semantic_score", 0) or 0) > (existing.get("semantic_score", 0) or 0):
                        seen[pid] = item
                        query_assignments[pid] = q_idx

        if not seen:
            return []

        # Interleave round-robin from each query's results to ensure
        # the top results show diversity rather than all coming from
        # one query's cluster.
        per_query_ordered: List[List[str]] = [[] for _ in queries]
        for q_idx, results in enumerate(results_per_query):
            for item in results:
                pid = item.get("product_id")
                if pid and pid in seen and query_assignments.get(pid) == q_idx:
                    per_query_ordered[q_idx].append(pid)

        merged_ids: List[str] = []
        merged_set: set = set()
        max_len = max(len(q) for q in per_query_ordered) if per_query_ordered else 0
        for pos in range(max_len):
            for q_idx in range(len(queries)):
                if pos < len(per_query_ordered[q_idx]):
                    pid = per_query_ordered[q_idx][pos]
                    if pid not in merged_set:
                        merged_ids.append(pid)
                        merged_set.add(pid)

        merged = [seen[pid] for pid in merged_ids if pid in seen]

        logger.info(
            "Multi-query semantic dedup",
            before=sum(counts),
            after=len(merged),
            duplicates_removed=sum(counts) - len(merged),
        )
        return merged

    def _search_multimodal(
        self,
        query: str,
        request: HybridSearchRequest,
        limit: int = 100,
        version: int = 1,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[dict]:
        """Search using multimodal embeddings (combined image + text vectors)."""
        try:
            # Skip category filter at RPC level — the search_multimodal SQL
            # function filters on p.broad_category which is NULL for all
            # products.  The actual category data lives in p.category
            # (lowercase).  Until the SQL is fixed, category filtering is
            # handled by _post_filter_semantic() using enriched Gemini
            # attributes (category_l1).  We pass None here so the RPC
            # returns results across all categories.
            categories = None

            resp = self.semantic_engine.search_multimodal(
                query=query,
                limit=limit,
                embedding_version=version,
                categories=categories,
                exclude_brands=request.exclude_brands,
                include_brands=request.brands,
                min_price=request.min_price,
                max_price=request.max_price,
                query_embedding=query_embedding,
            )

            results = []
            for item in resp.get("results", []):
                colors = item.get("colors") or []
                results.append({
                    "product_id": item.get("product_id"),
                    "name": item.get("name", ""),
                    "brand": item.get("brand", ""),
                    "image_url": item.get("image_url"),
                    "gallery_images": filter_gallery_images(item.get("gallery_images") or []),
                    "price": float(item.get("price", 0) or 0),
                    "original_price": item.get("original_price"),
                    "is_on_sale": item.get("is_on_sale", False),
                    "category_l1": None,  # enriched from Algolia
                    "category_l2": None,  # enriched from Algolia
                    "broad_category": item.get("broad_category") or item.get("category"),
                    "article_type": item.get("article_type"),
                    "primary_color": colors[0] if colors else None,
                    "color_family": None,  # enriched from Algolia
                    "pattern": item.get("pattern"),
                    "apparent_fabric": None,  # enriched from Algolia
                    "fit_type": item.get("fit") or item.get("fit_type"),
                    "formality": None,  # enriched from Algolia
                    "silhouette": None,  # enriched from Algolia
                    "length": item.get("length"),
                    "neckline": None,  # enriched from Algolia
                    "sleeve_type": item.get("sleeve") or item.get("sleeve_type"),
                    "rise": None,  # enriched from Algolia
                    "style_tags": item.get("style_tags") or [],
                    "occasions": item.get("occasions") or [],
                    "seasons": [],  # enriched from Algolia
                    "colors": colors,
                    "materials": item.get("materials") or [],
                    "semantic_score": item.get("similarity", 0),
                    "source": "semantic",
                })

            if results:
                logger.info(
                    "Multimodal semantic search returned results",
                    query=query,
                    count=len(results),
                    version=version,
                )
            return results

        except Exception as e:
            logger.warning(
                "Multimodal search failed, will fall back to image-only",
                error=str(e),
                query=query,
            )
            return []

    def _search_image_only(
        self,
        query: str,
        request: HybridSearchRequest,
        limit: int = 100,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[dict]:
        """Search using FashionCLIP image-only embeddings (original path)."""
        try:
            categories = request.categories or request.category_l1
            if categories:
                categories = [c.lower() for c in categories]
            resp = self.semantic_engine.search_with_filters(
                query=query,
                page=1,
                page_size=limit,
                categories=categories,
                exclude_brands=request.exclude_brands,
                include_brands=request.brands,
                include_colors=request.colors,
                min_price=request.min_price,
                max_price=request.max_price,
                occasions=request.occasions,
                patterns=request.patterns,
                use_hybrid_search=False,  # Pure semantic - Algolia handles keyword
                query_embedding=query_embedding,
            )
            results = []
            for item in resp.get("results", []):
                # Map pgvector fields; enrichment fills remaining Nones later
                colors = item.get("colors") or []
                results.append({
                    "product_id": item.get("product_id"),
                    "name": item.get("name", ""),
                    "brand": item.get("brand", ""),
                    "image_url": item.get("image_url"),
                    "gallery_images": filter_gallery_images(item.get("gallery_images") or []),
                    "price": float(item.get("price", 0) or 0),
                    "original_price": item.get("original_price"),
                    "is_on_sale": item.get("is_on_sale", False),
                    "category_l1": None,  # enriched from Algolia
                    "category_l2": None,  # enriched from Algolia
                    "broad_category": item.get("broad_category") or item.get("category"),
                    "article_type": item.get("article_type"),
                    "primary_color": colors[0] if colors else None,
                    "color_family": None,  # enriched from Algolia
                    "pattern": item.get("pattern"),
                    "apparent_fabric": None,  # enriched from Algolia
                    "fit_type": item.get("fit") or item.get("fit_type"),
                    "formality": None,  # enriched from Algolia
                    "silhouette": None,  # enriched from Algolia
                    "length": item.get("length"),
                    "neckline": None,  # enriched from Algolia
                    "sleeve_type": item.get("sleeve") or item.get("sleeve_type"),
                    "rise": None,  # enriched from Algolia
                    "style_tags": item.get("style_tags") or [],
                    "occasions": item.get("occasions") or [],
                    "seasons": [],  # enriched from Algolia
                    "colors": colors,
                    "materials": item.get("materials") or [],
                    "semantic_score": item.get("similarity", 0),
                    "source": "semantic",
                })
            return results
        except Exception as e:
            logger.error("Image-only semantic search failed", error=str(e))
            return []

    # =========================================================================
    # Enrich Semantic Results
    # =========================================================================

    _ENRICH_FIELDS = [
        "category_l1", "category_l2", "primary_color", "color_family",
        "formality", "silhouette", "neckline", "rise", "apparent_fabric",
        "seasons", "fit_type", "sleeve_type", "length", "pattern",
        "style_tags", "occasions", "article_type", "broad_category",
        "is_on_sale", "original_price", "trending_score",
    ]

    def _enrich_semantic_results(self, results: List[dict]) -> List[dict]:
        """
        Batch-fetch Gemini Vision attributes from Algolia for semantic results.

        pgvector results lack many attributes (category_l1, formality, neckline,
        etc.) that are only stored in Algolia. This method fetches them in a
        single batch call and merges them into the semantic results.
        """
        product_ids: List[str] = [r["product_id"] for r in results if r.get("product_id")]
        if not product_ids:
            return results

        try:
            algolia_records = self.algolia.get_objects(product_ids)
        except Exception as e:
            logger.warning("Failed to enrich semantic results from Algolia", error=str(e))
            return results

        if not algolia_records:
            return results

        enriched_count = 0
        for item in results:
            pid = item.get("product_id")
            if not pid:
                continue
            record = algolia_records.get(pid)
            if not record:
                continue
            for field in self._ENRICH_FIELDS:
                if item.get(field) is None:
                    val = record.get(field)
                    if val is not None:
                        item[field] = val
            enriched_count += 1

        logger.info(
            "Enriched semantic results from Algolia",
            total=len(results),
            enriched=enriched_count,
            algolia_found=len(algolia_records),
        )
        return results

    # =========================================================================
    # Reciprocal Rank Fusion
    # =========================================================================

    def _reciprocal_rank_fusion(
        self,
        algolia_results: List[dict],
        semantic_results: List[dict],
        algolia_weight: float = 0.6,
        semantic_weight: float = 0.4,
        k: int = 60,
    ) -> List[dict]:
        """
        Merge two result lists using Reciprocal Rank Fusion.

        RRF score = sum(weight / (k + rank))

        Args:
            algolia_results: Results from Algolia.
            semantic_results: Results from FashionCLIP.
            algolia_weight: Weight for Algolia results.
            semantic_weight: Weight for semantic results.
            k: RRF smoothing constant.

        Returns:
            Merged and scored results, sorted by RRF score.
        """
        scores: Dict[str, float] = defaultdict(float)
        product_data: Dict[str, dict] = {}

        # Score Algolia results
        for rank, item in enumerate(algolia_results, 1):
            pid = item.get("product_id")
            if not pid:
                continue
            scores[pid] += algolia_weight / (k + rank)
            product_data[pid] = item
            product_data[pid]["algolia_rank"] = rank

        # Score semantic results
        for rank, item in enumerate(semantic_results, 1):
            pid = item.get("product_id")
            if not pid:
                continue
            scores[pid] += semantic_weight / (k + rank)
            if pid not in product_data:
                product_data[pid] = item
            else:
                # Merge: keep Algolia data but add semantic info
                product_data[pid]["semantic_score"] = item.get("semantic_score", 0)
            product_data[pid]["semantic_rank"] = rank

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return [
            {**product_data[pid], "rrf_score": scores[pid]}
            for pid in sorted_ids
            if pid in product_data
        ]

    # =========================================================================
    # Query Cleaning
    # =========================================================================

    # Meta-terms that describe user intent but don't appear in product text.
    # Algolia requires ALL query words to match, so these cause 0 results.
    # Example: "date night outfit" → "date night" (Algolia), full query → FashionCLIP.
    _META_TERMS = re.compile(
        r'\b(?:outfit|outfits|look|looks|wear|clothes|clothing|attire|'
        r'garment|garments|ensemble|apparel|ideas?|inspo|inspiration|'
        r'for me|for women|for her|for petite|for tall|womens|women\'s)\b',
        re.IGNORECASE,
    )

    def _clean_query_for_algolia(
        self, query: str, extracted_terms: Optional[List[str]] = None,
    ) -> str:
        """Clean query for Algolia keyword search.

        Strips two types of words that hurt Algolia's keyword matching:

        1. **Meta-terms** ("outfit", "look", "wear") — describe user intent
           but don't appear in product text. Algolia requires ALL words to
           match, so these cause 0 results.

        2. **Extracted attribute terms** ("formal", "floral", "silk") — these
           have been converted to facet filters. Keeping them as text AND
           filters over-restricts results (e.g. "formal shirt" → Algolia
           requires "formal" in product name AND formality:Formal filter,
           but most formal shirts don't have "formal" in their name).

        The original unmodified query is still used for FashionCLIP semantic
        search, which understands all these terms naturally.

        Returns:
            Cleaned query. Falls back to original if cleaning would empty it.
        """
        cleaned = query

        # Strip meta-terms
        cleaned = self._META_TERMS.sub("", cleaned)

        # Strip extracted attribute terms (longest first to avoid partial matches)
        if extracted_terms:
            for term in sorted(extracted_terms, key=len, reverse=True):
                cleaned = re.sub(
                    r'\b' + re.escape(term) + r'\b', '', cleaned, flags=re.IGNORECASE,
                )

        # Collapse multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # If all words were extracted to filters, send empty string to Algolia.
        # Algolia with empty query + facet filters returns all matching products,
        # which is exactly what we want (e.g. "formal shirt" → empty query +
        # formality:Formal + category_l2:Shirt returns all formal shirts).
        # Only fall back to original if there were NO extracted terms
        # (cleaning removed nothing meaningful).
        if not cleaned and not extracted_terms:
            return query
        if cleaned != query:
            logger.info(
                "Cleaned query for Algolia",
                original=query,
                algolia_query=cleaned or "(empty - filter only)",
            )
        return cleaned

    # =========================================================================
    # Filter Building
    # =========================================================================

    # Attribute fields that become optionalFilters when the planner is active.
    # These are subjective attributes where the user's description may not
    # exactly match DB labels. Making them optional in Algolia preserves
    # keyword matching (e.g. "ribbed" in product names) while still boosting
    # results that match the extracted attributes.
    _OPTIONAL_WHEN_PLANNED = {
        "colors", "color_family", "patterns", "occasions", "seasons",
        "formality", "fit_type", "neckline", "sleeve_type", "length",
        "rise", "silhouette", "article_type", "style_tags", "materials",
    }

    # Map request field -> Algolia facet attribute name
    _FIELD_TO_ALGOLIA_FACET = {
        "colors": "primary_color",
        "color_family": "color_family",
        "patterns": "pattern",
        "occasions": "occasions",
        "seasons": "seasons",
        "formality": "formality",
        "fit_type": "fit_type",
        "neckline": "neckline",
        "sleeve_type": "sleeve_type",
        "length": "length",
        "rise": "rise",
        "silhouette": "silhouette",
        "article_type": "article_type",
        "style_tags": "style_tags",
        "materials": "apparent_fabric",
    }

    # Map exclude_* request fields -> Algolia facet attribute name (for NOT clauses)
    _EXCLUDE_FIELD_TO_ALGOLIA_FACET = {
        "exclude_neckline": "neckline",
        "exclude_sleeve_type": "sleeve_type",
        "exclude_length": "length",
        "exclude_fit_type": "fit_type",
        "exclude_silhouette": "silhouette",
        "exclude_patterns": "pattern",
        "exclude_colors": "primary_color",
        "exclude_materials": "apparent_fabric",
        "exclude_occasions": "occasions",
        "exclude_seasons": "seasons",
        "exclude_formality": "formality",
        "exclude_rise": "rise",
        "exclude_style_tags": "style_tags",
    }

    def _build_algolia_filters_split(
        self, request: HybridSearchRequest,
    ) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Build Algolia filters split into hard and optional.

        When the LLM planner is active, attribute filters (neckline, color,
        fit, etc.) become optionalFilters -- they boost matching results'
        ranking but don't exclude non-matching ones. This preserves keyword
        matching for descriptive terms like "ribbed" that don't map to any
        filter.

        Hard filters (always enforced): in_stock, brand, price, category,
        exclude_brands, on_sale_only.

        Returns:
            Tuple of (hard_filter_string, optional_filter_list).
        """
        hard_parts = []
        optional_parts = []

        # --- Always hard ---
        hard_parts.append("in_stock:true")

        if request.on_sale_only:
            hard_parts.append("is_on_sale:true")

        if request.categories:
            f = " OR ".join(f'broad_category:"{c}"' for c in request.categories)
            hard_parts.append(f"({f})")

        if request.category_l1:
            f = " OR ".join(f'category_l1:"{c}"' for c in request.category_l1)
            hard_parts.append(f"({f})")

        if request.category_l2:
            f = " OR ".join(f'category_l2:"{c}"' for c in request.category_l2)
            hard_parts.append(f"({f})")

        if request.brands:
            f = " OR ".join(f'brand:"{b}"' for b in request.brands)
            hard_parts.append(f"({f})")

        if request.exclude_brands:
            for b in request.exclude_brands:
                hard_parts.append(f'NOT brand:"{b}"')

        if request.min_price is not None:
            hard_parts.append(f"price >= {request.min_price}")

        if request.max_price is not None:
            hard_parts.append(f"price <= {request.max_price}")

        # --- Exclusion filters (always hard — NOT clauses) ---
        for req_field, algolia_facet in self._EXCLUDE_FIELD_TO_ALGOLIA_FACET.items():
            req_vals = getattr(request, req_field, None)
            if not req_vals:
                continue
            for val in req_vals:
                hard_parts.append(f'NOT {algolia_facet}:"{val}"')

        # --- Attribute filters become optional ---
        for req_field in self._OPTIONAL_WHEN_PLANNED:
            req_vals = getattr(request, req_field, None)
            if not req_vals:
                continue
            algolia_facet = self._FIELD_TO_ALGOLIA_FACET.get(req_field, req_field)
            for val in req_vals:
                optional_parts.append(f'{algolia_facet}:"{val}"')

        hard_str = " AND ".join(hard_parts) if hard_parts else None
        opt_list = optional_parts if optional_parts else None

        return hard_str, opt_list

    def _build_algolia_filters(self, request: HybridSearchRequest) -> Optional[str]:
        """Build Algolia filter string from request parameters."""
        parts = []

        # Always filter in-stock
        parts.append("in_stock:true")

        if request.on_sale_only:
            parts.append("is_on_sale:true")

        if request.categories:
            cat_filter = " OR ".join(f'broad_category:"{c}"' for c in request.categories)
            parts.append(f"({cat_filter})")

        if request.category_l1:
            f = " OR ".join(f'category_l1:"{c}"' for c in request.category_l1)
            parts.append(f"({f})")

        if request.category_l2:
            f = " OR ".join(f'category_l2:"{c}"' for c in request.category_l2)
            parts.append(f"({f})")

        if request.brands:
            f = " OR ".join(f'brand:"{b}"' for b in request.brands)
            parts.append(f"({f})")

        if request.exclude_brands:
            for b in request.exclude_brands:
                parts.append(f'NOT brand:"{b}"')

        if request.colors:
            f = " OR ".join(f'primary_color:"{c}"' for c in request.colors)
            parts.append(f"({f})")

        if request.color_family:
            f = " OR ".join(f'color_family:"{c}"' for c in request.color_family)
            parts.append(f"({f})")

        if request.patterns:
            f = " OR ".join(f'pattern:"{p}"' for p in request.patterns)
            parts.append(f"({f})")

        if request.occasions:
            f = " OR ".join(f'occasions:"{o}"' for o in request.occasions)
            parts.append(f"({f})")

        if request.seasons:
            f = " OR ".join(f'seasons:"{s}"' for s in request.seasons)
            parts.append(f"({f})")

        if request.formality:
            f = " OR ".join(f'formality:"{fm}"' for fm in request.formality)
            parts.append(f"({f})")

        if request.fit_type:
            f = " OR ".join(f'fit_type:"{ft}"' for ft in request.fit_type)
            parts.append(f"({f})")

        if request.neckline:
            f = " OR ".join(f'neckline:"{n}"' for n in request.neckline)
            parts.append(f"({f})")

        if request.sleeve_type:
            f = " OR ".join(f'sleeve_type:"{s}"' for s in request.sleeve_type)
            parts.append(f"({f})")

        if request.length:
            f = " OR ".join(f'length:"{l}"' for l in request.length)
            parts.append(f"({f})")

        if request.rise:
            f = " OR ".join(f'rise:"{r}"' for r in request.rise)
            parts.append(f"({f})")

        if request.materials:
            f = " OR ".join(f'apparent_fabric:"{m}"' for m in request.materials)
            parts.append(f"({f})")

        if request.silhouette:
            f = " OR ".join(f'silhouette:"{s}"' for s in request.silhouette)
            parts.append(f"({f})")

        if request.article_type:
            f = " OR ".join(f'article_type:"{a}"' for a in request.article_type)
            parts.append(f"({f})")

        if request.style_tags:
            f = " OR ".join(f'style_tags:"{s}"' for s in request.style_tags)
            parts.append(f"({f})")

        if request.min_price is not None:
            parts.append(f"price >= {request.min_price}")

        if request.max_price is not None:
            parts.append(f"price <= {request.max_price}")

        # Exclusion filters (NOT clauses)
        for req_field, algolia_facet in self._EXCLUDE_FIELD_TO_ALGOLIA_FACET.items():
            req_vals = getattr(request, req_field, None)
            if not req_vals:
                continue
            for val in req_vals:
                parts.append(f'NOT {algolia_facet}:"{val}"')

        if not parts:
            return None
        return " AND ".join(parts)

    # =========================================================================
    # Response Formatting
    # =========================================================================

    def _to_product_result(self, item: dict, position: int) -> ProductResult:
        """Convert a merged result dict to a ProductResult."""
        return ProductResult(
            product_id=item.get("product_id", ""),
            name=item.get("name", ""),
            brand=item.get("brand", ""),
            image_url=item.get("image_url"),
            gallery_images=item.get("gallery_images"),
            price=item.get("price", 0),
            original_price=item.get("original_price"),
            is_on_sale=item.get("is_on_sale", False),
            category_l1=item.get("category_l1"),
            category_l2=item.get("category_l2"),
            broad_category=item.get("broad_category"),
            article_type=item.get("article_type"),
            primary_color=item.get("primary_color"),
            color_family=item.get("color_family"),
            pattern=item.get("pattern"),
            apparent_fabric=item.get("apparent_fabric"),
            fit_type=item.get("fit_type"),
            formality=item.get("formality"),
            silhouette=item.get("silhouette"),
            length=item.get("length"),
            neckline=item.get("neckline"),
            sleeve_type=item.get("sleeve_type"),
            rise=item.get("rise"),
            style_tags=item.get("style_tags"),
            occasions=item.get("occasions"),
            seasons=item.get("seasons"),
            algolia_rank=item.get("algolia_rank"),
            semantic_rank=item.get("semantic_rank"),
            semantic_score=item.get("semantic_score"),
            rrf_score=item.get("rrf_score"),
        )

    # =========================================================================
    # Semantic Post-Filter
    # =========================================================================

    # Filters that are ALWAYS hard (drop non-matching) regardless of planner.
    # These represent objective constraints the user explicitly specified.
    _HARD_FILTER_FIELDS = {
        "brands", "exclude_brands", "categories",
        "category_l1", "category_l2",
        "min_price", "max_price", "on_sale_only",
    }

    # Filters that become SOFT (score boost/penalty) when the LLM planner is
    # active. These are subjective attributes where DB labels may not match
    # the user's language (e.g. "Sheer" isn't a valid material, "Y2K" isn't
    # a valid style_tag, occasion labels are sparse).
    _SOFT_WHEN_PLANNED = {
        "formality", "fit_type", "neckline", "sleeve_type", "length",
        "rise", "silhouette", "article_type", "color_family",
        "occasions", "seasons", "materials", "style_tags",
        "patterns", "colors",
    }

    # Score adjustments for soft filtering
    _SOFT_MATCH_BOOST = 0.05    # Boost for matching a soft filter
    _SOFT_MISMATCH_PENALTY = 0.0  # No penalty -- just don't boost

    def _post_filter_semantic(
        self,
        results: List[dict],
        request: HybridSearchRequest,
        expanded_filters: Optional[Dict[str, List[str]]] = None,
        drop_nulls: bool = True,
    ) -> List[dict]:
        """
        Post-filter on semantic results after Algolia enrichment.

        Two modes:
        1. **Strict mode** (no expanded_filters / regex fallback): All filters
           are hard -- non-matching results are dropped. This is the original
           behavior.
        2. **Relaxed mode** (expanded_filters from LLM planner): Only hard
           filters (brand, category, price) drop results. Attribute filters
           (occasions, materials, style_tags, formality, etc.) become soft
           scoring boosts -- matching items rank higher, but non-matching items
           aren't dropped. This prevents 0-result scenarios when DB labels
           don't match the user's language.

        When drop_nulls=False (progressive relaxation), exclusion filters keep
        items with null/N/A attribute values instead of dropping them.
        """
        if expanded_filters is None:
            expanded_filters = {}

        use_soft_scoring = bool(expanded_filters)
        filtered = results

        def _get_allowed_vals(field_name: str, strict_vals: List[str]) -> set:
            """Get allowed values: use expanded set if available, else strict."""
            expanded = expanded_filters.get(field_name)
            if expanded:
                all_vals = set(v.lower() for v in strict_vals)
                all_vals.update(v.lower() for v in expanded)
                return all_vals
            return {v.lower() for v in strict_vals}

        # =====================================================================
        # HARD FILTERS (always enforced)
        # =====================================================================

        # --- Sale filter ---
        if request.on_sale_only:
            filtered = [r for r in filtered if r.get("is_on_sale")]

        # --- Price filters ---
        if request.min_price is not None:
            filtered = [r for r in filtered
                        if r.get("price") is not None and r["price"] >= request.min_price]
        if request.max_price is not None:
            filtered = [r for r in filtered
                        if r.get("price") is not None and r["price"] <= request.max_price]

        # --- Brand filters ---
        if request.brands:
            vals = _get_allowed_vals("brands", request.brands)
            filtered = [r for r in filtered
                        if r.get("brand") and r["brand"].lower() in vals]
        if request.exclude_brands:
            vals = {v.lower() for v in request.exclude_brands}
            filtered = [r for r in filtered
                        if not r.get("brand") or r["brand"].lower() not in vals]

        # --- Exclusion filters (always hard, regardless of planner mode) ---
        # These remove products matching any excluded attribute value.
        # When drop_nulls=False (progressive relaxation), keep items with
        # null/N/A values instead of dropping them.
        filtered = self._apply_exclusion_filters(filtered, request, drop_nulls=drop_nulls)

        # --- Category filters (broad) ---
        if request.categories:
            vals = _get_allowed_vals("categories", request.categories)
            filtered = [r for r in filtered
                        if r.get("broad_category") and r["broad_category"].lower() in vals]

        # --- Category L1 (hard -- structural) ---
        if request.category_l1:
            vals = _get_allowed_vals("category_l1", request.category_l1)
            filtered = [r for r in filtered
                        if r.get("category_l1") and r["category_l1"].lower() in vals]

        # --- Category L2 (hard -- structural, with substring match for expanded) ---
        if request.category_l2:
            vals = _get_allowed_vals("category_l2", request.category_l2)
            if expanded_filters.get("category_l2"):
                filtered = [r for r in filtered
                            if r.get("category_l2") and (
                                r["category_l2"].lower() in vals
                                or any(v in r["category_l2"].lower() for v in vals)
                            )]
            else:
                filtered = [r for r in filtered
                            if r.get("category_l2") and r["category_l2"].lower() in vals]

        # =====================================================================
        # SOFT vs HARD ATTRIBUTE FILTERS
        # =====================================================================
        if use_soft_scoring:
            # --- SOFT MODE (LLM planner active) ---
            # Attribute filters boost matching items' scores instead of dropping
            # non-matching ones. This prevents 0-result failures when DB labels
            # don't match the user's language.
            filtered = self._apply_soft_attribute_scoring(filtered, request, expanded_filters)
        else:
            # --- STRICT MODE (regex fallback) ---
            # All attribute filters are hard drops (original behavior).

            # Colors
            if request.colors:
                vals = _get_allowed_vals("colors", request.colors)
                filtered = [r for r in filtered
                            if (r.get("primary_color") and r["primary_color"].lower() in vals)
                            or (r.get("colors") and any(c.lower() in vals for c in r["colors"]))]

            # Patterns
            if request.patterns:
                vals = _get_allowed_vals("patterns", request.patterns)
                filtered = [r for r in filtered
                            if r.get("pattern") and r["pattern"].lower() in vals]

            # Occasions
            if request.occasions:
                vals = _get_allowed_vals("occasions", request.occasions)
                filtered = [r for r in filtered
                            if r.get("occasions") and any(o.lower() in vals for o in r["occasions"])]

            # Single-value Gemini attribute filters
            _STRICT_SINGLE = [
                ("formality", "formality"),
                ("fit_type", "fit_type"),
                ("neckline", "neckline"),
                ("sleeve_type", "sleeve_type"),
                ("length", "length"),
                ("rise", "rise"),
                ("silhouette", "silhouette"),
                ("article_type", "article_type"),
                ("color_family", "color_family"),
            ]
            for req_field, data_field in _STRICT_SINGLE:
                req_vals = getattr(request, req_field, None)
                if req_vals:
                    vals = {v.lower() for v in req_vals}
                    filtered = [r for r in filtered
                                if r.get(data_field) and r[data_field].lower() in vals]

            # Multi-value Gemini attribute filters
            _STRICT_MULTI = [
                ("seasons", "seasons"),
                ("materials", "materials"),
                ("style_tags", "style_tags"),
            ]
            for req_field, data_field in _STRICT_MULTI:
                req_vals = getattr(request, req_field, None)
                if req_vals:
                    vals = {v.lower() for v in req_vals}
                    filtered = [r for r in filtered
                                if r.get(data_field) and any(v.lower() in vals for v in r[data_field])]

        return filtered

    # Map exclude_* request fields -> result dict field for post-filtering
    _EXCLUDE_TO_RESULT_FIELD: Dict[str, Tuple[str, str]] = {
        # (exclude_request_field, result_dict_field, type)
        # "single" = scalar field, "multi" = list field
    }

    _EXCLUDE_SINGLE_FIELDS = [
        ("exclude_neckline", "neckline"),
        ("exclude_sleeve_type", "sleeve_type"),
        ("exclude_length", "length"),
        ("exclude_fit_type", "fit_type"),
        ("exclude_silhouette", "silhouette"),
        ("exclude_patterns", "pattern"),
        ("exclude_colors", "primary_color"),
        ("exclude_formality", "formality"),
        ("exclude_rise", "rise"),
        ("exclude_materials", "apparent_fabric"),  # single-value Gemini attribute
    ]

    _EXCLUDE_MULTI_FIELDS = [
        ("exclude_occasions", "occasions"),
        ("exclude_seasons", "seasons"),
        ("exclude_style_tags", "style_tags"),
    ]

    # Values treated as "missing data" — not a real attribute value
    _NULL_VALUES = {"n/a", "none", "null", "unknown", "-", ""}

    def _apply_exclusion_filters(
        self,
        results: List[dict],
        request: HybridSearchRequest,
        drop_nulls: bool = True,
    ) -> List[dict]:
        """
        Apply exclusion filters as HARD drops on results.

        Exclusion filters are always enforced regardless of planner mode.
        Any product whose attribute matches an excluded value is removed.

        For single-value fields (neckline, pattern, etc.): drop if value is
        in the exclude set.  When drop_nulls=True (default), also drops items
        where the value is N/A/null since we can't confirm they don't match
        the excluded attribute.  When drop_nulls=False, items with null/N/A
        values are kept (used during progressive relaxation).
        For multi-value fields (occasions, materials, etc.): drop if ANY
        value in the product's list matches the exclude set.
        """
        filtered = results

        def _is_null(val) -> bool:
            """Check if a value is missing/null-like."""
            if not val:
                return True
            return str(val).strip().lower() in self._NULL_VALUES

        # Single-value exclusions
        for req_field, data_field in self._EXCLUDE_SINGLE_FIELDS:
            exclude_vals = getattr(request, req_field, None)
            if not exclude_vals:
                continue
            vals = {v.lower() for v in exclude_vals}
            if drop_nulls:
                filtered = [
                    r for r in filtered
                    if not _is_null(r.get(data_field))
                    and r[data_field].lower() not in vals
                ]
            else:
                # Lenient: keep items with null/missing values
                filtered = [
                    r for r in filtered
                    if _is_null(r.get(data_field))
                    or r[data_field].lower() not in vals
                ]

        # Multi-value exclusions
        # For multi-value fields (style_tags, occasions, materials, seasons),
        # an empty list [] means "no values" — NOT "unknown". A product with
        # style_tags=[] does NOT contain "Backless", so it should PASS the
        # exclusion filter. Only treat None as truly unknown/null.
        for req_field, data_field in self._EXCLUDE_MULTI_FIELDS:
            exclude_vals = getattr(request, req_field, None)
            if not exclude_vals:
                continue
            vals = {v.lower() for v in exclude_vals}
            if drop_nulls:
                filtered = [
                    r for r in filtered
                    if (
                        # None means truly unknown — drop when drop_nulls=True
                        r.get(data_field) is not None
                        # Empty list [] means no tags — passes exclusion
                        and not any(v.lower() in vals for v in r[data_field])
                    )
                ]
            else:
                # Lenient: keep items with null/missing values
                filtered = [
                    r for r in filtered
                    if r.get(data_field) is None
                    or not any(v.lower() in vals for v in r[data_field])
                ]

        # Also check exclude_colors against the "colors" array (multi-value)
        if request.exclude_colors:
            vals = {v.lower() for v in request.exclude_colors}
            filtered = [
                r for r in filtered
                if not (r.get("colors") and any(c.lower() in vals for c in r["colors"]))
            ]

        if len(filtered) < len(results):
            logger.info(
                "Exclusion filters removed semantic results",
                before=len(results),
                after=len(filtered),
                dropped=len(results) - len(filtered),
                drop_nulls=drop_nulls,
            )

        return filtered

    def _apply_soft_attribute_scoring(
        self,
        results: List[dict],
        request: HybridSearchRequest,
        expanded_filters: Dict[str, List[str]],
    ) -> List[dict]:
        """
        Apply soft scoring boosts for attribute filter matches.

        Instead of dropping items that don't match attribute filters, boost
        matching items' RRF scores so they rank higher. This is used when
        the LLM planner is active to prevent 0-result failures.

        Boost per matching filter: +0.05 to rrf_score (or semantic_score).
        """
        if not results:
            return results

        def _get_expanded_vals(field_name: str, strict_vals: List[str]) -> set:
            expanded = expanded_filters.get(field_name)
            if expanded:
                all_vals = set(v.lower() for v in strict_vals)
                all_vals.update(v.lower() for v in expanded)
                return all_vals
            return {v.lower() for v in strict_vals}

        # Build all active soft filters
        soft_checks = []

        # Single-value fields
        _SINGLE_FIELDS = [
            ("colors", "primary_color"),
            ("patterns", "pattern"),
            ("formality", "formality"),
            ("fit_type", "fit_type"),
            ("neckline", "neckline"),
            ("sleeve_type", "sleeve_type"),
            ("length", "length"),
            ("rise", "rise"),
            ("silhouette", "silhouette"),
            ("article_type", "article_type"),
            ("color_family", "color_family"),
        ]
        for req_field, data_field in _SINGLE_FIELDS:
            req_vals = getattr(request, req_field, None)
            if req_vals:
                vals = _get_expanded_vals(req_field, req_vals)
                soft_checks.append(("single", data_field, vals))

        # Multi-value fields
        _MULTI_FIELDS = [
            ("occasions", "occasions"),
            ("seasons", "seasons"),
            ("materials", "materials"),
            ("style_tags", "style_tags"),
        ]
        for req_field, data_field in _MULTI_FIELDS:
            req_vals = getattr(request, req_field, None)
            if req_vals:
                vals = _get_expanded_vals(req_field, req_vals)
                soft_checks.append(("multi", data_field, vals))

        # Also check colors in the colors array (multi-value)
        if request.colors:
            vals = _get_expanded_vals("colors", request.colors)
            soft_checks.append(("multi", "colors", vals))

        if not soft_checks:
            return results

        # Score each result
        for item in results:
            soft_boost = 0.0
            matches = 0
            for check_type, data_field, vals in soft_checks:
                if check_type == "single":
                    val = item.get(data_field)
                    if val and val.lower() in vals:
                        matches += 1
                elif check_type == "multi":
                    arr = item.get(data_field)
                    if arr and any(v.lower() in vals for v in arr):
                        matches += 1

            if matches > 0:
                soft_boost = matches * self._SOFT_MATCH_BOOST
                item["rrf_score"] = item.get("rrf_score", 0) + soft_boost
                item["_soft_matches"] = matches

        # Re-sort by boosted rrf_score
        results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        return results

    def _extract_filter_summary(self, request: HybridSearchRequest) -> dict:
        """Extract a summary of applied filters for analytics."""
        filters = {}
        for field_name in [
            "categories", "category_l1", "category_l2", "brands",
            "exclude_brands", "colors", "color_family", "patterns",
            "occasions", "seasons", "formality", "fit_type", "neckline",
            "sleeve_type", "length", "rise", "silhouette", "article_type",
            "style_tags", "materials", "min_price", "max_price",
            "on_sale_only",
            # Exclusion filters
            "exclude_neckline", "exclude_sleeve_type", "exclude_length",
            "exclude_fit_type", "exclude_silhouette", "exclude_patterns",
            "exclude_colors", "exclude_materials", "exclude_occasions",
            "exclude_seasons", "exclude_formality", "exclude_rise",
            "exclude_style_tags",
        ]:
            val = getattr(request, field_name, None)
            if val is not None and val != [] and val is not False:
                filters[field_name] = val
        return filters


# =============================================================================
# Singleton
# =============================================================================

import threading as _threading

_service: Optional[HybridSearchService] = None
_service_lock = _threading.Lock()


def get_hybrid_search_service() -> HybridSearchService:
    """Get or create the HybridSearchService singleton (thread-safe)."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = HybridSearchService()
    return _service
