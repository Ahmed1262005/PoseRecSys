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
from search.attribute_search import (
    AttributeFilters,
    AttributeSearchEngine,
    get_attribute_search_engine,
    plan_to_attribute_filters,
)
from search.mode_config import get_rrf_weights
from search.query_classifier import QueryClassifier
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
from search.session_cache import (
    SearchSessionCache,
    SearchSessionEntry,
    decode_cursor,
    encode_cursor,
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

        # Lazy-load attribute search engine (pgvector + attribute filters)
        self._attribute_engine = None

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

    @property
    def attribute_engine(self) -> AttributeSearchEngine:
        """Lazy-load AttributeSearchEngine for pgvector + attribute-filtered search."""
        if self._attribute_engine is None:
            self._attribute_engine = get_attribute_search_engine()
        return self._attribute_engine

    # =========================================================================
    # User-filter / planner-filter separation
    # =========================================================================

    # Fields on HybridSearchRequest that are NOT user-selectable filters.
    # These are control/pagination/metadata fields — never snapshot them.
    _NON_FILTER_SNAPSHOT_FIELDS = frozenset({
        "query", "page", "page_size", "session_id", "search_session_id",
        "cursor", "sort_by", "semantic_boost", "planner_context",
        "selected_filters", "selection_labels",
    })

    # Request field -> result dict key mapping for user post-filtering.
    # Some request fields map to differently-named keys in result dicts.
    _USER_FILTER_FIELD_MAP: Dict[str, str] = {
        # Category fields
        "categories": "broad_category",      # List[str] match
        "category_l1": "category_l1",        # List[str] match
        "category_l2": "category_l2",        # List[str] match
        # Brand
        "brands": "brand",                   # List[str] match
        "exclude_brands": "brand",           # List[str] NOT match
        # Appearance — single-value in result dict
        "colors": "primary_color",           # Also check "colors" list
        "color_family": "color_family",
        "patterns": "pattern",
        "formality": "formality",
        "fit_type": "fit_type",
        "neckline": "neckline",
        "sleeve_type": "sleeve_type",
        "length": "length",
        "rise": "rise",
        "silhouette": "silhouette",
        "article_type": "article_type",
        "style_tags": "style_tags",          # Multi-value in result dict
        # Multi-value in result dict
        "materials": "materials",
        "occasions": "occasions",
        "seasons": "seasons",
    }

    # Multi-value result dict fields (list of strings instead of scalar)
    _USER_FILTER_MULTI_FIELDS = frozenset({
        "style_tags", "materials", "occasions", "seasons",
    })

    # "colors" needs special handling: check both "primary_color" (scalar)
    # and "colors" (list) in result dicts.
    _USER_FILTER_DUAL_FIELDS = frozenset({"colors"})

    # Multi-value facet fields — result dict stores these as lists.
    # Each list element is counted individually (same as Algolia).
    _MULTI_VALUE_FACET_FIELDS = frozenset({
        "occasions", "seasons", "style_tags", "materials",
    })

    # Null-like values excluded from facet counts.
    _FACET_NULL_VALUES = frozenset({"null", "n/a", "none", ""})

    def _compute_facets_from_results(
        self,
        results: List[dict],
    ) -> Optional[Dict[str, List["FacetValue"]]]:
        """Compute facet counts from merged results.

        Mirrors Algolia's facet processing logic:
        1. For each field in _FACET_FIELDS, count value occurrences
        2. Multi-value fields (occasions, seasons, etc.): each list
           element counted individually
        3. is_on_sale (bool) → "true"/"false" strings
        4. Filter: count > 1, not null/N/A, field needs >= 2 distinct values
        5. Sort values by count descending

        Used when semantic results contributed to the merge (so Algolia
        facets alone don't represent the full result set), or when user
        UI filters narrowed the results post-merge.
        """
        if not results:
            return None

        from collections import Counter

        computed: Dict[str, List[FacetValue]] = {}

        for facet_field in _FACET_FIELDS:
            counter: Counter = Counter()

            if facet_field == "is_on_sale":
                # Boolean → string
                for item in results:
                    val = item.get("is_on_sale")
                    if val is not None:
                        counter[str(val).lower()] += 1
            elif facet_field in self._MULTI_VALUE_FACET_FIELDS:
                # List field — count each element
                for item in results:
                    vals = item.get(facet_field)
                    if vals and isinstance(vals, list):
                        for v in vals:
                            if v and str(v).lower() not in self._FACET_NULL_VALUES:
                                counter[v] += 1
            else:
                # Scalar field
                for item in results:
                    val = item.get(facet_field)
                    if val is not None:
                        s = str(val)
                        if s.lower() not in self._FACET_NULL_VALUES:
                            counter[s] += 1

            # Apply Algolia's 3 filtering rules:
            # 1. count > 1
            # 2. null/junk already excluded above
            # 3. field needs >= 2 distinct valid values
            values = [
                FacetValue(value=val, count=cnt)
                for val, cnt in sorted(counter.items(), key=lambda x: -x[1])
                if cnt > 1
            ]
            if len(values) >= 2:
                computed[facet_field] = values

        return computed if computed else None

    @staticmethod
    def _snapshot_user_filters(
        request: HybridSearchRequest,
    ) -> Dict[str, Any]:
        """Capture user-set filters BEFORE the planner mutates the request.

        A filter is considered "user-set" if it is non-None (and non-default
        for special fields like on_sale_only which defaults to False).

        Returns a dict of {field_name: value} for every user-set filter.
        This snapshot is used later to post-filter merged results.
        """
        snapshot: Dict[str, Any] = {}
        for field_name in HybridSearchRequest.model_fields:
            if field_name in HybridSearchService._NON_FILTER_SNAPSHOT_FIELDS:
                continue
            val = getattr(request, field_name, None)
            # on_sale_only defaults to False; True = user set it
            if field_name == "on_sale_only":
                if val is True:
                    snapshot[field_name] = val
                continue
            # is_set defaults to None; non-None = user set it
            if field_name == "is_set":
                if val is not None:
                    snapshot[field_name] = val
                continue
            # All other filters default to None; non-None = user set
            if val is not None:
                snapshot[field_name] = val
        return snapshot

    @staticmethod
    def _strip_filters_from_request(
        request: HybridSearchRequest,
        filter_names: Set[str],
    ) -> HybridSearchRequest:
        """Return a copy of *request* with the named filter fields reset.

        Used to remove user-set UI filters from the request before passing
        it to Algolia / semantic search.  The planner-derived filters remain.
        """
        if not filter_names:
            return request
        resets: Dict[str, Any] = {}
        for field in filter_names:
            if field == "on_sale_only":
                resets[field] = False
            elif field == "is_set":
                resets[field] = None
            elif field in ("min_price", "max_price"):
                resets[field] = None
            else:
                resets[field] = None
        return request.model_copy(update=resets)

    def _apply_user_post_filters(
        self,
        results: List[dict],
        user_filters: Dict[str, Any],
    ) -> List[dict]:
        """Apply user UI filters on merged results (post-RRF, pre-rerank).

        This is the core of the architectural fix: user-set filters from
        dropdowns/UI (brand, color, price, etc.) are NOT sent to Algolia
        or semantic search as hard pre-filters. Instead, they are applied
        here on the full merged result set — similar to facet filtering.

        This prevents conflicts between the planner's structural guesses
        (e.g. category_l1=Activewear) and the user's brand selection
        (e.g. brand=Nike) which may not overlap.
        """
        if not user_filters:
            return results

        filtered = results

        # --- Price filters ---
        if "min_price" in user_filters:
            min_p = user_filters["min_price"]
            filtered = [
                r for r in filtered
                if r.get("price") is not None and float(r["price"]) >= min_p
            ]
        if "max_price" in user_filters:
            max_p = user_filters["max_price"]
            filtered = [
                r for r in filtered
                if r.get("price") is not None and float(r["price"]) <= max_p
            ]

        # --- Sale filter ---
        if user_filters.get("on_sale_only"):
            filtered = [r for r in filtered if r.get("is_on_sale")]

        # --- Set / co-ord filter ---
        if "is_set" in user_filters:
            is_set_val = user_filters["is_set"]
            filtered = [r for r in filtered if bool(r.get("is_set")) == is_set_val]

        # --- Brand inclusion ---
        if "brands" in user_filters:
            brand_vals = {b.lower() for b in user_filters["brands"]}
            filtered = [
                r for r in filtered
                if r.get("brand") and r["brand"].lower() in brand_vals
            ]

        # --- Brand exclusion ---
        if "exclude_brands" in user_filters:
            excl_vals = {b.lower() for b in user_filters["exclude_brands"]}
            filtered = [
                r for r in filtered
                if not r.get("brand") or r["brand"].lower() not in excl_vals
            ]

        # --- Category filters (broad_category, category_l1, category_l2) ---
        for cat_field, result_key in [
            ("categories", "broad_category"),
            ("category_l1", "category_l1"),
            ("category_l2", "category_l2"),
        ]:
            if cat_field in user_filters:
                vals = {v.lower() for v in user_filters[cat_field]}
                filtered = [
                    r for r in filtered
                    if r.get(result_key) and r[result_key].lower() in vals
                ]

        # --- Colors (dual: check primary_color scalar AND colors list) ---
        if "colors" in user_filters:
            vals = {v.lower() for v in user_filters["colors"]}
            filtered = [
                r for r in filtered
                if (r.get("primary_color") and r["primary_color"].lower() in vals)
                or (r.get("colors") and any(c.lower() in vals for c in r["colors"]))
            ]

        # --- Color family ---
        if "color_family" in user_filters:
            vals = {v.lower() for v in user_filters["color_family"]}
            filtered = [
                r for r in filtered
                if r.get("color_family") and r["color_family"].lower() in vals
            ]

        # --- Single-value attribute filters ---
        _SINGLE_ATTRS = [
            ("patterns", "pattern"),
            ("formality", "formality"),
            ("fit_type", "fit_type"),
            ("neckline", "neckline"),
            ("sleeve_type", "sleeve_type"),
            ("length", "length"),
            ("rise", "rise"),
            ("silhouette", "silhouette"),
            ("article_type", "article_type"),
        ]
        for req_field, data_field in _SINGLE_ATTRS:
            if req_field in user_filters:
                vals = {v.lower() for v in user_filters[req_field]}
                filtered = [
                    r for r in filtered
                    if r.get(data_field) and r[data_field].lower() in vals
                ]

        # --- Multi-value attribute filters ---
        _MULTI_ATTRS = [
            ("materials", "materials"),
            ("occasions", "occasions"),
            ("seasons", "seasons"),
            ("style_tags", "style_tags"),
        ]
        for req_field, data_field in _MULTI_ATTRS:
            if req_field in user_filters:
                vals = {v.lower() for v in user_filters[req_field]}
                filtered = [
                    r for r in filtered
                    if r.get(data_field) and any(
                        v.lower() in vals for v in r[data_field]
                    )
                ]

        # --- Exclusion filters (exclude_neckline, exclude_colors, etc.) ---
        _NULL_VALUES = {"n/a", "none", "null", "unknown", "-", ""}
        for excl_field, data_field in self._EXCLUDE_SINGLE_FIELDS:
            if excl_field in user_filters:
                vals = {v.lower() for v in user_filters[excl_field]}
                filtered = [
                    r for r in filtered
                    if not (
                        r.get(data_field)
                        and r[data_field].lower() not in _NULL_VALUES
                        and r[data_field].lower() in vals
                    )
                ]
        for excl_field, data_field in self._EXCLUDE_MULTI_FIELDS:
            if excl_field in user_filters:
                vals = {v.lower() for v in user_filters[excl_field]}
                filtered = [
                    r for r in filtered
                    if not (
                        r.get(data_field)
                        and any(
                            v.lower() in vals
                            for v in r[data_field]
                            if v.lower() not in _NULL_VALUES
                        )
                    )
                ]

        if len(filtered) < len(results):
            logger.info(
                "User post-filter applied on merged results",
                before=len(results),
                after=len(filtered),
                dropped=len(results) - len(filtered),
                user_filters=list(user_filters.keys()),
            )

        return filtered

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
        pre_plan: Optional[Any] = None,
        selected_filters: Optional[Dict[str, Any]] = None,
        selection_labels: Optional[List[str]] = None,
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
            selected_filters: Optional follow-up filter selections. When
                present, the planner runs in REFINEMENT mode (Section 11).
            selection_labels: Optional human-readable labels of selected
                follow-up options.
            pre_plan: Optional pre-computed SearchPlan (from refinement planner).
                When provided, skip the planner call and use this plan directly.

        Returns:
            HybridSearchResponse with results and metadata.
        """
        t_start = time.time()
        timing: Dict[str, Any] = {}

        # Step 0: Normalize query (decode HTML entities, collapse whitespace)
        clean_query = html_mod.unescape(request.query).strip()
        if clean_query != request.query:
            request = request.model_copy(update={"query": clean_query})

        # Step 0b: Snapshot user-set UI filters BEFORE the planner mutates
        # the request.  These will be applied as post-filters on merged
        # results instead of being sent as Algolia/semantic pre-filters.
        # This prevents the "athleisure + Nike brand → 0 results" bug where
        # the planner's category guess AND the user's brand filter conflict.
        user_filters = self._snapshot_user_filters(request)
        if user_filters:
            logger.info(
                "Snapshotted user UI filters for post-filtering",
                user_filters=list(user_filters.keys()),
            )

        # -----------------------------------------------------------------
        # SORT-MODE: handled via post-sort after hybrid merge+rerank.
        # The full pipeline runs normally (planner + Algolia + semantic +
        # RRF + reranker), then results are sorted by price/trending
        # before pagination. This gives rich hybrid candidates instead
        # of the Algolia-only path which over-filters on vague queries.
        # _search_sorted() is kept as a legacy fallback but not routed to.
        # -----------------------------------------------------------------

        # -----------------------------------------------------------------
        # CACHED PAGINATION FAST PATH
        # When the client passes back a search_session_id + cursor from a
        # previous response, serve the next page directly from the in-memory
        # cache (~1ms) instead of re-running the full pipeline (~15s).
        # -----------------------------------------------------------------
        if request.search_session_id and request.cursor:
            cached = self._serve_cached_page(request)
            if cached is not None:
                return cached
            # Cache miss (expired or invalid) — fall through to full pipeline
            logger.info(
                "Search session cache miss, running full pipeline",
                search_session_id=request.search_session_id,
            )

        # -----------------------------------------------------------------
        # FILTER REFINEMENT FAST PATH
        # When the client passes a search_session_id + user filters but NO
        # cursor, this signals "apply new facet filters, reset to page 1".
        # Reuses cached FashionCLIP embeddings + planner state (~3-5s)
        # instead of re-running the full pipeline (~12-15s).
        # -----------------------------------------------------------------
        if request.search_session_id and not request.cursor and user_filters:
            refined = self._filter_refine_search(
                request=request,
                user_filters=user_filters,
                user_id=user_id,
                user_profile=user_profile,
                seen_ids=seen_ids,
                user_context=user_context,
                session_scores=session_scores,
            )
            if refined is not None:
                return refined
            # Cache miss — fall through to full pipeline
            logger.info(
                "Filter refinement cache miss, running full pipeline",
                search_session_id=request.search_session_id,
            )

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
        vibe_brand: Optional[str] = None

        t_plan = time.time()
        if pre_plan is not None:
            # Use the pre-computed plan (from refinement planner)
            search_plan = pre_plan
            logger.info(
                "Using pre-computed search plan (refinement)",
                query=request.query,
                intent=search_plan.intent,
            )
        elif not skip_planner:
            search_plan = self._planner.plan(
                request.query,
                user_context=planner_context,
                selected_filters=selected_filters,
                selection_labels=selection_labels,
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
            vibe_brand = plan_updates.pop("_vibe_brand", None)

            # ---------------------------------------------------------
            # Vibe-brand hard filter: when the LLM identifies a brand as
            # a style reference (not a purchase target), hard-filter to
            # brands from the same cluster(s).  This ensures cluster
            # brands dominate results instead of being drowned by high-
            # volume brands in semantic search.
            #
            # Cluster selection depends on the price direction:
            # - "better quality" (min_price set, no max_price):
            #   UPGRADE — exclude the brand's primary cluster (its own
            #   tier) and keep only secondary/higher clusters.  Also
            #   exclude the vibe brand itself.
            # - "cheaper" (max_price set, no min_price):
            #   DOWNGRADE — exclude secondary cluster, keep primary.
            # - no price modifier:
            #   SAME TIER — include all clusters, exclude vibe brand.
            # ---------------------------------------------------------
            if vibe_brand and not request.brands:
                try:
                    from recs.brand_clusters import get_brand_clusters
                    from scoring.constants.brand_data import BRAND_CLUSTERS

                    all_clusters = get_brand_clusters(vibe_brand.lower())
                    # all_clusters is [(cluster_id, confidence), ...]
                    # First entry = primary, rest = secondary

                    # Detect price direction from the search plan
                    _has_min = search_plan.min_price is not None
                    _has_max = search_plan.max_price is not None
                    _upgrade = _has_min and not _has_max
                    _downgrade = _has_max and not _has_min

                    # Select which clusters to include
                    if _upgrade and len(all_clusters) > 1:
                        # Upgrade: drop primary (brand's own tier), keep
                        # secondary+ (aspirational/higher tier)
                        vibe_cluster_ids = {cid for cid, _ in all_clusters[1:]}
                        direction = "upgrade"
                    elif _downgrade and len(all_clusters) > 1:
                        # Downgrade: keep primary (affordable tier), drop
                        # secondary (premium tier)
                        vibe_cluster_ids = {all_clusters[0][0]}
                        direction = "downgrade"
                    else:
                        # Same tier or single cluster: include all
                        vibe_cluster_ids = {cid for cid, _ in all_clusters}
                        direction = "same_tier"

                    if vibe_cluster_ids:
                        cluster_brands: List[str] = []
                        seen_lower: set = set()
                        vibe_lower = vibe_brand.lower()
                        for cid in vibe_cluster_ids:
                            for brand_name in BRAND_CLUSTERS.get(cid, []):
                                key = brand_name.lower()
                                # Always exclude the vibe brand itself —
                                # user wants alternatives, not the brand
                                if key == vibe_lower:
                                    continue
                                if key not in seen_lower:
                                    seen_lower.add(key)
                                    cluster_brands.append(brand_name)
                        if cluster_brands:
                            request = request.model_copy(
                                update={"brands": cluster_brands}
                            )
                            logger.info(
                                "Vibe-brand hard filter applied",
                                vibe_brand=vibe_brand,
                                direction=direction,
                                clusters=sorted(vibe_cluster_ids),
                                brand_count=len(cluster_brands),
                            )
                except Exception:
                    pass  # Non-fatal — fall through to soft boost

            # ---------------------------------------------------------
            # Filter application strategy per intent:
            #
            # EXACT: Only keep brand + price filters. The LLM often
            #   guesses category/style for the brand (e.g. "Outdoor
            #   Voices" → Activewear) which becomes a hard Algolia
            #   filter and over-restricts results.
            #
            # VAGUE: Keep structural filters (category, formality,
            #   occasions) and mode exclusions.  Strip subjective
            #   visual attributes (colors, patterns, materials, etc.)
            #   — those are carried by the semantic queries.
            #
            # SPECIFIC: Apply ALL filters including mode exclusions.
            #   Mode exclusions (modest, work, etc.) are important
            #   safety/appropriateness constraints.
            # ---------------------------------------------------------

            # Subjective visual attributes to strip for VAGUE queries.
            # These are appearance-based filters that the LLM may guess
            # imprecisely — semantic queries handle them better.
            _VAGUE_STRIP_KEYS = {
                "colors", "patterns", "materials", "style_tags",
                "fit_type", "neckline", "sleeve_type", "length",
                "silhouette", "aesthetics", "age_groups", "body_types",
                "care_instructions", "sustainability_ratings",
                "versatility_scores",
            }

            # Keys to KEEP for EXACT intent (pure brand search).
            # Everything else is the LLM guessing what the brand
            # sells — those guesses become hard Algolia filters
            # that over-restrict results (e.g. "Outdoor Voices" →
            # category_l1=Activewear → only 2 hits).
            _EXACT_KEEP_KEYS = {
                "brands", "exclude_brands",
                "min_price", "max_price", "on_sale_only",
            }

            if intent_str == "exact":
                # Pure brand search — only apply brand + price filters.
                preserved = {}
                skipped = {}
                for k, v in plan_updates.items():
                    if k in _EXACT_KEEP_KEYS:
                        if not getattr(request, k, None):
                            preserved[k] = v
                    else:
                        skipped[k] = v
                updates = preserved
                if updates:
                    request = request.model_copy(update=updates)
                if skipped:
                    logger.info(
                        "Exact brand query — stripped non-brand filters",
                        query=request.query,
                        preserved_filters=list(preserved.keys()),
                        skipped_filters=list(skipped.keys()),
                    )
            elif intent_str == "vague":
                # Keep structural + safety filters; strip subjective visuals
                preserved = {}
                skipped = {}
                for k, v in plan_updates.items():
                    if k in _VAGUE_STRIP_KEYS:
                        skipped[k] = v
                    elif not getattr(request, k, None):
                        preserved[k] = v
                updates = preserved
                if updates:
                    request = request.model_copy(update=updates)
                logger.info(
                    "Vague query — keeping structural + exclusion filters",
                    query=request.query,
                    preserved_filters=list(preserved.keys()),
                    skipped_filters=list(skipped.keys()),
                    semantic_queries=semantic_queries,
                )
            else:
                # SPECIFIC: apply ALL planner filters including
                # mode-derived exclusions (no longer stripped).
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
            if vibe_brand:
                timing["plan_vibe_brand"] = vibe_brand

            # Compute v1.0.0.2 attribute filters from the search plan.
            # These are used for attribute-filtered semantic search (pgvector
            # + WHERE clauses) replacing the old vision reranker approach.
            timing["plan_detail_terms"] = search_plan.detail_terms
        else:
            # --- Fallback: regex-based classification (no LLM planner) ---
            intent = QueryClassifier.classify(request.query)
            algolia_weight, semantic_weight = get_rrf_weights(intent.value)
            logger.info(
                "Planner unavailable, using regex classifier fallback",
                query=request.query,
                intent=intent.value,
            )

            # Use raw query for Algolia (clean meta-terms only, no extracted terms)
            algolia_query = self._clean_query_for_algolia(request.query)

        # Allow request-level override of semantic weight.
        _SEMANTIC_BOOST_DEFAULT = 0.4
        if abs(request.semantic_boost - _SEMANTIC_BOOST_DEFAULT) > 1e-9:
            semantic_weight = request.semantic_boost
            algolia_weight = 1.0 - semantic_weight

        # -----------------------------------------------------------------
        # Step 2b½: Strip user UI filters from the request.
        #
        # The request now has BOTH user-set filters AND planner-derived
        # filters merged together. We need the planner-derived filters
        # for Algolia (structural: category_l1, mode exclusions) but NOT
        # the user's UI selections (brand, color, price dropdowns).
        #
        # User filters will be applied post-RRF as computational filters
        # on the full merged result set (like Algolia facets).
        #
        # We build a "pipeline request" with user filters stripped — this
        # is what Algolia and semantic search will use.  The user_filters
        # snapshot (captured in Step 0b) is applied after RRF merge.
        # -----------------------------------------------------------------
        if user_filters:
            pipeline_request = self._strip_filters_from_request(
                request, set(user_filters.keys()),
            )
            logger.info(
                "Stripped user UI filters from pipeline request",
                stripped=list(user_filters.keys()),
                remaining_filters={
                    f: getattr(pipeline_request, f)
                    for f in HybridSearchRequest.model_fields
                    if f not in self._NON_FILTER_SNAPSHOT_FIELDS
                    and getattr(pipeline_request, f) not in (None, False, [])
                },
            )
        else:
            pipeline_request = request

        # Step 2c: Build Algolia filter strings.
        # When the planner is active OR skip_planner (refine path), split into
        # hard filters (brand, price, stock, category) and optional filters
        # (neckline, color, fit, etc.) so Algolia returns a wider candidate
        # pool.  Subjective attributes become optionalFilters (boost, don't
        # exclude) — this prevents 0-result failures when the user selects
        # multiple follow-up filters that are too restrictive together.
        algolia_optional_filters: Optional[List[str]] = None
        use_split_filters = search_plan is not None or skip_planner
        if use_split_filters:
            algolia_filters, algolia_optional_filters = self._build_algolia_filters_split(pipeline_request)
            if algolia_optional_filters:
                logger.info(
                    "Using optionalFilters for Algolia",
                    source="planner" if search_plan else "refine",
                    hard_filters=algolia_filters,
                    optional_count=len(algolia_optional_filters),
                )
        else:
            algolia_filters = self._build_algolia_filters(pipeline_request)

        # Step 2d: Boost cluster brands in Algolia retrieval.
        # Two sources of brand boosting:
        #   (a) Vibe brand: when the LLM identified a brand-as-style-reference
        #       (e.g., "like Zara but better"), boost brands from the same
        #       style cluster(s) as the referenced brand.
        #   (b) User profile: when the user has preferred brands from
        #       onboarding, boost their cluster-adjacent brands.
        # Skip when pipeline already has brand constraints (vibe-brand or planner).
        # User UI brand filters are post-applied, not on the pipeline request.
        if not pipeline_request.brands:
            # (a) Vibe brand cluster boosting (query-level)
            if vibe_brand:
                vibe_filters = self._build_vibe_brand_filters(vibe_brand)
                if vibe_filters:
                    if algolia_optional_filters is None:
                        algolia_optional_filters = []
                    algolia_optional_filters.extend(vibe_filters)
                    logger.info(
                        "Added vibe-brand cluster optionalFilters",
                        vibe_brand=vibe_brand,
                        cluster_brands=len(vibe_filters),
                        total_optional=len(algolia_optional_filters),
                    )
                    timing["vibe_brand"] = vibe_brand
                    timing["vibe_brand_filters"] = len(vibe_filters)

            # (b) User profile cluster boosting (session-level)
            if user_profile:
                cluster_brand_filters = self._build_cluster_brand_filters(user_profile)
                if cluster_brand_filters:
                    if algolia_optional_filters is None:
                        algolia_optional_filters = []
                    algolia_optional_filters.extend(cluster_brand_filters)
                    logger.info(
                        "Added cluster brand optionalFilters",
                        cluster_brands=len(cluster_brand_filters),
                        total_optional=len(algolia_optional_filters),
                    )

        # =====================================================================
        # Compute v1.0.0.2 attribute filters from the search plan.
        # These replace the old detail_mode + vision reranker approach.
        # When attribute filters are present, the semantic search uses
        # pgvector + attribute WHERE clauses (search_semantic_with_attributes
        # RPC) for high-precision results on detail queries.
        # =====================================================================
        attribute_filters: Optional[AttributeFilters] = None
        if search_plan is not None:
            try:
                attribute_filters = plan_to_attribute_filters(
                    search_plan, query=request.query,
                )
                if attribute_filters.has_attribute_filters():
                    timing["attribute_filters"] = attribute_filters.describe()
                    logger.info(
                        "Attribute filters computed from search plan",
                        query=request.query,
                        filters=attribute_filters.describe(),
                    )
            except Exception as e:
                logger.warning(
                    "Failed to compute attribute filters, continuing without",
                    error=str(e),
                )
                attribute_filters = None

        # =====================================================================
        # NORMAL MODE: Algolia + FashionCLIP semantic search
        # (with optional attribute-filtered semantic search)
        #
        # When attribute_filters are active, _search_semantic_with_attributes()
        # replaces standard _search_semantic() — uses pgvector + attribute
        # WHERE clauses for high-precision results on detail queries like
        # "dress with pockets", "backless dress", "lace midi dress".
        # =====================================================================

        # Step 3+4: Run Algolia + Semantic search
        # For SPECIFIC/VAGUE intent, both searches are independent so we
        # run them in parallel.  For EXACT intent, semantic only runs if
        # Algolia returns 0, so we keep those sequential.
        #
        # Auto-detect active filters — either planner-derived (on
        # pipeline_request) or user UI filters (applied post-merge).
        # Both cause result drops, so we fetch more candidates.
        has_pipeline_filters = any(
            getattr(pipeline_request, field_name) not in (None, False, [])
            for field_name in HybridSearchRequest.model_fields
            if field_name not in self._NON_FILTER_SNAPSHOT_FIELDS
        )
        has_filters = has_pipeline_filters or bool(user_filters)
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

        # Build a relaxed request for semantic search when planner is active
        # or when filters are pre-resolved (refine path).  Subjective filters
        # (colors, patterns, occasions) are removed so pgvector returns more
        # candidates — the post-filter soft scoring will boost matches.
        # Uses pipeline_request (user UI filters already stripped).
        if search_plan is not None or skip_planner:
            semantic_request = pipeline_request.model_copy(update={
                "categories": None,
                "colors": None,
                "patterns": None,
                "occasions": None,
            })
            logger.info(
                "Using relaxed filters for semantic search",
                source="planner" if search_plan else "refine",
                kept=["price", "brands", "exclude_brands", "category_l1", "category_l2"],
                removed=["categories", "colors", "patterns", "occasions"],
            )
        else:
            semantic_request = pipeline_request

        # Check if attribute-filtered semantic search should be used.
        # When v1.0.0.2 attribute filters are active, use the attribute
        # search engine (pgvector + WHERE clauses) for the semantic path
        # instead of the standard multimodal search.
        _use_attribute_search = (
            attribute_filters is not None
            and attribute_filters.has_detail_attribute_filters()
        )

        def _run_algolia() -> tuple:
            """Run Algolia search, return (results, facets, elapsed_ms, nb_hits)."""
            t0 = time.time()
            results, fcts, nb_hits = self._search_algolia(
                query=algolia_query,
                filters=algolia_filters,
                hits_per_page=fetch_size,
                facets=_FACET_FIELDS,
                optional_filters=algolia_optional_filters,
            )
            return results, fcts, int((time.time() - t0) * 1000), nb_hits

        # Mutable container to capture semantic embeddings from _run_semantic().
        # Populated by _search_semantic_multi() when it batch-encodes queries.
        # Used later to cache in SearchSessionEntry for extend-search pagination.
        _captured_embeddings: List[Optional[List[np.ndarray]]] = [None]

        def _run_semantic() -> tuple:
            """Run semantic search, return (results, query_count, elapsed_ms).

            When attribute_filters are active, uses the attribute search
            engine (pgvector + attribute WHERE clauses) for high-precision
            results on detail queries like "dress with pockets".
            """
            t0 = time.time()

            if _use_attribute_search:
                # Attribute-filtered semantic search path
                attr_engine = self.attribute_engine
                if len(_queries_to_run) > 1:
                    attr_results, _ = attr_engine.search_multi_semantic(
                        queries=_queries_to_run,
                        filters=attribute_filters,
                        per_query_limit=max(fetch_size // len(_queries_to_run), 30),
                    )
                    qcount = len(_queries_to_run)
                else:
                    attr_results, _ = attr_engine.search(
                        query=_queries_to_run[0],
                        filters=attribute_filters,
                        semantic_query=_queries_to_run[0],
                    )
                    qcount = 1

                # Convert AttributeSearchEngine results to the dict format
                # expected by the rest of the pipeline.
                results = self._convert_attribute_results(attr_results)
                logger.info(
                    "Attribute-filtered semantic search",
                    query_count=qcount,
                    results=len(results),
                    filters=attribute_filters.describe(),
                )
                return results, qcount, int((time.time() - t0) * 1000)

            # Standard semantic search path (no attribute filters)
            if len(_queries_to_run) > 1:
                results, embeddings = self._search_semantic_multi(
                    queries=_queries_to_run,
                    request=semantic_request,
                    limit_per_query=max(fetch_size // len(_queries_to_run), 30),
                    user_id=user_id,
                )
                _captured_embeddings[0] = embeddings
                qcount = len(_queries_to_run)
            else:
                results = self._search_semantic(
                    query=_queries_to_run[0],
                    request=semantic_request,
                    limit=fetch_size,
                    user_id=user_id,
                )
                qcount = 1
            return results, qcount, int((time.time() - t0) * 1000)

        # Skip Algolia entirely when the query is empty/whitespace AND
        # no brand hard-filter is active.  Empty queries with no brand
        # constraint return arbitrary results (all trending/popularity
        # scores are 0.0, so tiebreaker is document insertion order)
        # which poison RRF merge.  When a brand filter IS set, the
        # results are scoped to that brand and remain useful even
        # without text matching.  Applies to ALL intents.
        _empty_query = not algolia_query or not algolia_query.strip()
        _has_brand_filter = bool(pipeline_request.brands)
        _skip_algolia = _empty_query and not _has_brand_filter
        if _skip_algolia:
            logger.info(
                "Skipping Algolia — empty query with no brand filter would return arbitrary results",
                intent=intent.value,
                original_query=request.query,
            )
        elif _empty_query and _has_brand_filter:
            logger.info(
                "Empty query but brand filter active — keeping Algolia for filtered results",
                intent=intent.value,
                brands=(request.brands or [])[:5],
            )

        # Algolia's total hit count (nbHits) — the full catalog count for
        # the query+filters, independent of our fetch_size limit.  Surfaced
        # as total_results in the response so the client can display "47K
        # results" instead of just the merged-pool size.
        algolia_nb_hits: int = 0

        if intent == QueryIntent.EXACT:
            # EXACT: run Algolia first; only run semantic if Algolia returns 0
            if _skip_algolia:
                algolia_results, facets = [], {}
                timing["algolia_ms"] = 0
                # Always run semantic when Algolia is skipped
                semantic_results, sq_count, semantic_ms = _run_semantic()
                timing["semantic_ms"] = semantic_ms
                timing["semantic_query_count"] = sq_count
            else:
                algolia_results, facets, algolia_ms, algolia_nb_hits = _run_algolia()
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

            if _skip_algolia:
                # Algolia skipped — run semantic only
                semantic_results, sq_count, semantic_ms = _run_semantic()
                timing["semantic_ms"] = semantic_ms
                timing["semantic_query_count"] = sq_count
            else:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_algolia = executor.submit(_run_algolia)
                    future_semantic = executor.submit(_run_semantic)

                    try:
                        algolia_results, facets, algolia_ms, algolia_nb_hits = future_algolia.result()
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

        # Step 4b: Enrich semantic results with Gemini attributes from Algolia.
        # Skip enrichment when attribute-filtered search was used — those
        # results already have attribute data from product_attributes table.
        if semantic_results and not _use_attribute_search:
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

            # Use pipeline_request (user UI filters stripped) for semantic
            # post-filtering — only planner-derived filters apply here.
            # User filters are applied post-RRF on all merged results.
            semantic_results = self._post_filter_semantic(
                enriched_snapshot, pipeline_request,
                expanded_filters=expanded_filters,
                force_soft=skip_planner,
            )

            # Check if we have active exclusion filters that could be relaxed
            _EXCLUDE_FIELDS = (
                "exclude_neckline", "exclude_sleeve_type", "exclude_length",
                "exclude_fit_type", "exclude_silhouette", "exclude_patterns",
                "exclude_colors", "exclude_materials", "exclude_occasions",
                "exclude_seasons", "exclude_formality", "exclude_rise",
                "exclude_style_tags",
            )
            has_exclusions = any(getattr(pipeline_request, f, None) for f in _EXCLUDE_FIELDS)

            if len(semantic_results) == 0 and has_exclusions and pre_filter_count > 0:
                # Retry 1: Keep exclusion filters but stop dropping null-valued items.
                # Many products have missing attribute data (N/A); strict mode
                # drops these, which can eliminate too many candidates.
                logger.info(
                    "Progressive relaxation: retry with lenient nulls",
                    pre_filter_count=pre_filter_count,
                )
                semantic_results = self._post_filter_semantic(
                    list(enriched_snapshot), pipeline_request,
                    expanded_filters=expanded_filters,
                    drop_nulls=False,
                    force_soft=skip_planner,
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
                relaxed_request = pipeline_request.model_copy(update={
                    f: None for f in _EXCLUDE_FIELDS
                })
                semantic_results = self._post_filter_semantic(
                    list(enriched_snapshot), relaxed_request,
                    expanded_filters=expanded_filters,
                    force_soft=skip_planner,
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
        # When the planner returned an empty algolia_query for a VAGUE query,
        # Algolia fell back to customRanking (trending/popularity) — the same
        # popular items for every query.  Discard those results so they don't
        # pollute RRF.  For EXACT/SPECIFIC, Algolia has meaningful filters
        # (brand, category, occasions) so its results are relevant even
        # without a text query — keep them.
        _algolia_for_rrf = algolia_results
        if intent == QueryIntent.VAGUE and (not algolia_query or not algolia_query.strip()):
            # Only discard Algolia's generic results if semantic actually
            # returned something.  When semantic fails or returns nothing,
            # Algolia is the only source — keep it as a fallback.
            if algolia_results and semantic_results:
                logger.info(
                    "VAGUE + empty algolia_query — discarding Algolia results "
                    "from RRF (keeping facets only)",
                    discarded_count=len(algolia_results),
                    semantic_count=len(semantic_results),
                )
                _algolia_for_rrf = []

        merged = self._reciprocal_rank_fusion(
            algolia_results=_algolia_for_rrf,
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
        # Post-merge exclusion filters use pipeline_request (planner-derived
        # exclusions only). User UI exclusions are handled in _apply_user_post_filters.
        has_any_exclusion = any(
            getattr(pipeline_request, f, None)
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
            strict_merged = self._apply_exclusion_filters(merged, pipeline_request, drop_nulls=True)
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
                lenient_merged = self._apply_exclusion_filters(merged, pipeline_request, drop_nulls=False)
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

        # Step 5d: Apply user UI filters on ALL merged results.
        # This is the core architectural fix: user-set filters (brand, color,
        # price dropdowns) are NOT sent to Algolia/semantic as pre-filters.
        # Instead they are applied here on the full merged result set.
        # This prevents "athleisure + Nike → 0" where the planner's category
        # guess (Activewear) AND the user's brand (Nike) conflict.
        if user_filters and merged:
            merged = self._apply_user_post_filters(merged, user_filters)

        # Step 5e: Facet strategy.
        # For SPECIFIC/VAGUE intents, prefer Algolia's catalog-wide facets.
        # The merged set is too small (45-200 items) to give meaningful
        # facet counts — e.g. "Boohoo: 3" when catalog has 20K matches.
        # Algolia facets reflect the full filtered catalog and are correct.
        #
        # For EXACT intent with user UI filters, recompute from merged
        # results since Algolia won't reflect post-filter narrowing.
        # For EXACT without user filters, Algolia facets are already correct.
        if semantic_results or user_filters:
            if intent.value != "exact" and facets:
                # Keep Algolia's catalog-wide facets for SPECIFIC/VAGUE
                pass
            else:
                computed_facets = self._compute_facets_from_results(merged)
                if computed_facets:
                    facets = computed_facets

        # Step 5.5: Load impression counts for soft demotion
        impression_counts: Optional[Dict[str, int]] = None
        if user_id:
            try:
                _raw_counts = self.analytics.load_impression_counts(
                    user_id=user_id,
                )
                if isinstance(_raw_counts, dict) and _raw_counts:
                    impression_counts = _raw_counts
            except Exception:
                pass  # Non-fatal — skip demotion if load fails

        # Step 6: Rerank with session/profile + context scoring
        # Disable brand diversity cap when the user explicitly requested a
        # brand (via EXACT intent or a brands filter).  Otherwise the
        # reranker's MAX_PER_BRAND=4 cap kills almost all results when
        # every result is from the same brand.
        # Exception: vibe-brand queries set request.brands programmatically
        # to cluster brands — we WANT the cap there to equalize across brands.
        # Use user_filters snapshot (not request.brands) because user UI
        # brands are now post-filters, not on the pipeline request.
        _user_set_brands = user_filters.get("brands") and not vibe_brand
        brand_cap = 0 if intent == QueryIntent.EXACT or _user_set_brands else None

        # Compute vibe-brand cluster set for reranker boost
        vibe_clusters: Optional[Set[str]] = None
        if vibe_brand:
            try:
                from recs.brand_clusters import get_brand_clusters
                # get_brand_clusters returns List[(cluster_id, confidence)]
                vibe_clusters = {cid for cid, _ in get_brand_clusters(vibe_brand)}
                if vibe_clusters:
                    logger.info(
                        "Passing vibe-brand clusters to reranker",
                        vibe_brand=vibe_brand,
                        clusters=sorted(vibe_clusters),
                    )
            except Exception:
                pass  # Non-fatal — skip cluster boost if lookup fails

        # Only enforce category proportional caps for VAGUE queries where
        # we're mixing multiple categories.  For SPECIFIC/EXACT the user
        # targeted a category (e.g., "midi skirt") — capping bottoms to
        # 25% would artificially suppress the results they asked for.
        _cat_cap_page_size = request.page_size if intent == QueryIntent.VAGUE else 0
        rerank_kwargs: Dict[str, Any] = dict(
            results=merged,
            user_profile=user_profile,
            seen_ids=seen_ids,
            user_context=user_context,
            session_scores=session_scores,
            impression_counts=impression_counts,
            page_size=_cat_cap_page_size,
        )
        if brand_cap is not None:
            rerank_kwargs["max_per_brand"] = brand_cap
        if vibe_clusters:
            rerank_kwargs["vibe_brand_clusters"] = vibe_clusters
        merged = self._reranker.rerank(**rerank_kwargs)

        # Step 6b: Score-gated category diversity for VAGUE queries.
        # Only interleave categories whose top item scores within 25% of
        # the best category's top score.  Weak categories (e.g. Outerwear
        # at 0.005 when Tops is at 0.068) are appended at the end instead
        # of being forced into prominent positions via round-robin.
        _INTERLEAVE_SCORE_RATIO = 0.25

        if intent == QueryIntent.VAGUE and merged:
            cat_buckets: Dict[str, List[dict]] = {}
            for item in merged:
                cat = item.get("category_l1") or item.get("broad_category") or "Other"
                cat_buckets.setdefault(cat, []).append(item)

            if len(cat_buckets) > 1:
                # Find each category's top score (first item, already sorted)
                cat_top_scores = {
                    cat: items[0].get("rrf_score", 0)
                    for cat, items in cat_buckets.items()
                    if items
                }
                best_score = max(cat_top_scores.values()) if cat_top_scores else 0
                threshold = best_score * _INTERLEAVE_SCORE_RATIO

                strong_cats = [
                    c for c, s in cat_top_scores.items() if s >= threshold
                ]
                weak_cats = [
                    c for c, s in cat_top_scores.items() if s < threshold
                ]

                # Round-robin only strong categories
                strong_cats.sort(key=lambda c: -len(cat_buckets[c]))
                interleaved: List[dict] = []
                if strong_cats:
                    max_bucket = max(len(cat_buckets[c]) for c in strong_cats)
                    for pos in range(max_bucket):
                        for cat in strong_cats:
                            if pos < len(cat_buckets[cat]):
                                interleaved.append(cat_buckets[cat][pos])

                # Append weak categories at the end (in score order)
                for cat in weak_cats:
                    interleaved.extend(cat_buckets[cat])

                before_cats = {c: len(b) for c, b in cat_buckets.items()}
                logger.info(
                    "Category diversity interleave (score-gated)",
                    categories=before_cats,
                    strong=strong_cats,
                    weak=weak_cats,
                    threshold=round(threshold, 6),
                    total=len(interleaved),
                )
                merged = interleaved

        # Step 7a: Post-sort by price/trending if requested
        if request.sort_by == SortBy.PRICE_ASC:
            merged.sort(key=lambda r: float(r.get("price", 0) or 0))
        elif request.sort_by == SortBy.PRICE_DESC:
            merged.sort(key=lambda r: float(r.get("price", 0) or 0), reverse=True)
        elif request.sort_by == SortBy.TRENDING:
            merged.sort(key=lambda r: float(r.get("trending_score", 0) or 0), reverse=True)

        # Step 7b: Paginate
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

        # Compute refinement state for client (accumulated across rounds)
        _applied_filters: Optional[Dict[str, Any]] = None
        _answered_dims: Optional[List[str]] = None
        if selected_filters:
            _applied_filters = selected_filters
            from search.query_planner import QueryPlanner
            _answered_dims = QueryPlanner._infer_answered_dimensions(
                selected_filters
            )

        # Step 10: Cache plan state for extend-search pagination.
        # Store the plan state (embeddings, filters, seen_ids, reranker config)
        # so page 2+ can re-run Algolia (native page=N) + semantic (reuse
        # cached embeddings + exclude seen_ids) in ~2-3s instead of re-running
        # the full pipeline with LLM planner (~12-15s).
        _search_session_id: Optional[str] = None
        _cursor: Optional[str] = None
        if len(merged) > 0:
            try:
                cache = SearchSessionCache.get_instance()
                _search_session_id = cache.generate_session_id()

                # Collect page-1 product IDs as the initial seen set
                _page1_ids: Set[str] = {
                    r["product_id"] for r in merged
                    if r.get("product_id")
                }

                # Build the semantic_request_updates dict so extend-search
                # can reconstruct the relaxed semantic request.
                _semantic_request_updates: Optional[Dict[str, Any]] = None
                if search_plan is not None or skip_planner:
                    _semantic_request_updates = {
                        "categories": None,
                        "colors": None,
                        "patterns": None,
                        "occasions": None,
                    }

                cache.store(SearchSessionEntry(
                    session_id=_search_session_id,
                    query=request.query,
                    intent=intent.value,
                    sort_by=request.sort_by.value,
                    # Algolia state
                    algolia_query=algolia_query or "",
                    algolia_filters=algolia_filters or "",
                    algolia_optional_filters=algolia_optional_filters,
                    # Semantic state
                    semantic_queries=_queries_to_run,
                    semantic_embeddings=_captured_embeddings[0],
                    semantic_request_updates=_semantic_request_updates,
                    # RRF weights
                    algolia_weight=algolia_weight,
                    semantic_weight=semantic_weight,
                    # Reranker config
                    rerank_kwargs={
                        "user_profile": user_profile,
                        "user_context": user_context,
                        "session_scores": session_scores,
                        "page_size": _cat_cap_page_size,
                        **({"max_per_brand": brand_cap} if brand_cap is not None else {}),
                        **({"vibe_brand_clusters": vibe_clusters} if vibe_clusters else {}),
                    },
                    # Pagination tracking
                    seen_product_ids=_page1_ids,
                    algolia_page=0,  # page 1 used Algolia page 0
                    page_size=request.page_size,
                    fetch_size=fetch_size,
                    # Response metadata (returned on all pages)
                    facets=facets,
                    follow_ups=follow_ups,
                    applied_filters=_applied_filters,
                    answered_dimensions=_answered_dims,
                    # Algolia catalog count
                    algolia_total_hits=algolia_nb_hits,
                    # Post-filter criteria for endless semantic (page 2+)
                    post_filter_criteria={
                        k: getattr(pipeline_request, k, None)
                        for k in (
                            "category_l1", "category_l2", "brands",
                            "exclude_brands", "min_price", "max_price",
                        )
                        if getattr(pipeline_request, k, None) is not None
                    } or None,
                    # Flags
                    skip_algolia=_skip_algolia,
                    use_attribute_search=_use_attribute_search,
                    attribute_filters=attribute_filters,
                ))
                _cursor = encode_cursor(page=2)
                # Extend-search fetches FRESH candidates from 96K products,
                # so even if page 1 has fewer than page_size results, there
                # are likely more to find. Override has_more to True.
                has_more = True
            except Exception as exc:
                logger.warning("Failed to cache search session", error=str(exc))

        return HybridSearchResponse(
            query=request.query,
            intent=intent.value,
            sort_by=request.sort_by.value,
            results=products,
            pagination=PaginationInfo(
                page=request.page,
                page_size=request.page_size,
                has_more=has_more,
                # Surface Algolia's full catalog count (nbHits) so the client
                # can show "47K results" instead of the merged-pool size.
                # Falls back to merged count when Algolia was skipped.
                total_results=algolia_nb_hits if algolia_nb_hits > 0 else len(merged),
            ),
            search_session_id=_search_session_id,
            cursor=_cursor,
            timing=timing,
            facets=facets,
            follow_ups=follow_ups,
            applied_filters=_applied_filters,
            answered_dimensions=_answered_dims,
        )

    # =========================================================================
    # Extend-Search Pagination (page 2+)
    # =========================================================================

    def _serve_cached_page(
        self, request: HybridSearchRequest
    ) -> Optional[HybridSearchResponse]:
        """Extend search for page 2+ using cached plan state.

        Instead of re-running the full pipeline (~12-15s), reuses the
        cached plan state from page 1:
        - Algolia: native page=N pagination (same query/filters) ~200ms
        - Semantic: reuse cached embeddings + exclude seen IDs ~1-2s
        - RRF merge + rerank on fresh candidates ~500ms
        - LLM planner: SKIP entirely (saves ~5-8s)

        Returns a HybridSearchResponse if the cache hit succeeds, or None
        if the session is expired / missing / cursor is invalid (caller
        should fall through to the full pipeline).
        """
        cache = SearchSessionCache.get_instance()
        sid = request.search_session_id
        assert sid is not None  # caller checks before calling
        entry = cache.get(sid)
        if entry is None:
            return None

        cur = request.cursor
        assert cur is not None  # caller checks before calling
        try:
            cursor_data = decode_cursor(cur)
            page = cursor_data["p"]
        except (ValueError, KeyError):
            logger.warning(
                "Invalid search cursor, cache miss",
                search_session_id=sid,
            )
            return None

        # EXACT intent: Algolia-native pagination (unchanged).
        # SPECIFIC / VAGUE intent: endless semantic pump (pgvector only).
        if entry.intent == "exact":
            return self._extend_search(entry, page, sid)
        else:
            return self._endless_semantic_page(entry, page, sid)

    def _extend_search(
        self,
        entry: SearchSessionEntry,
        page: int,
        session_id: str,
    ) -> HybridSearchResponse:
        """Core extend-search logic: run Algolia + semantic with cached plan.

        Args:
            entry: Cached plan state from page 1.
            page: The requested page number (2+).
            session_id: The search session ID (for response + logging).

        Returns:
            HybridSearchResponse with fresh results for the requested page.
        """
        from concurrent.futures import ThreadPoolExecutor

        t_start = time.time()
        timing: Dict[str, Any] = {"extend_search": True, "page": page}

        seen_ids_list = list(entry.seen_product_ids)

        # -----------------------------------------------------------------
        # Algolia: native page=N pagination (same query/filters)
        # -----------------------------------------------------------------
        algolia_results: List[dict] = []
        algolia_facets = None

        def _run_algolia_extend():
            if entry.skip_algolia:
                return [], None, 0
            t0 = time.time()
            next_page = entry.next_algolia_page()
            # Call the raw Algolia client with the correct page offset.
            # Unlike _search_algolia() which always uses page=0, we need
            # native Algolia pagination for extend-search.
            try:
                resp = self.algolia.search(
                    query=entry.algolia_query,
                    filters=entry.algolia_filters or None,
                    optional_filters=entry.algolia_optional_filters,
                    hits_per_page=entry.fetch_size,
                    page=next_page,
                    facets=_FACET_FIELDS,
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
                        "is_set": hit.get("is_set", False),
                        "set_role": hit.get("set_role"),
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
                fcts = None
                raw_facets = resp.get("facets")
                if raw_facets:
                    fcts = {}
                    for facet_name, value_counts in raw_facets.items():
                        values = [
                            FacetValue(value=val, count=cnt)
                            for val, cnt in sorted(value_counts.items(), key=lambda x: -x[1])
                            if cnt > 1 and val and val.lower() not in ("null", "n/a", "none", "")
                        ]
                        if len(values) >= 2:
                            fcts[facet_name] = values
            except Exception as e:
                logger.warning("Extend-search Algolia page=%d failed", next_page, error=str(e))
                results, fcts = [], None

            return results, fcts, int((time.time() - t0) * 1000)

        # -----------------------------------------------------------------
        # Semantic: reuse cached embeddings + exclude seen IDs
        # -----------------------------------------------------------------
        def _run_semantic_extend():
            t0 = time.time()

            if entry.use_attribute_search:
                # Attribute-filtered path — no embedding reuse yet
                # (AttributeSearchEngine doesn't support exclude_product_ids)
                # Fall back to standard semantic with embeddings
                pass

            queries = entry.semantic_queries or [entry.query]

            # Build relaxed request for semantic search
            base_request = HybridSearchRequest(query=entry.query)
            if entry.semantic_request_updates:
                base_request = base_request.model_copy(
                    update=entry.semantic_request_updates
                )

            if len(queries) > 1 and entry.semantic_embeddings is not None:
                results, _ = self._search_semantic_multi(
                    queries=queries,
                    request=base_request,
                    limit_per_query=max(entry.fetch_size // len(queries), 30),
                    precomputed_embeddings=entry.semantic_embeddings,
                    exclude_product_ids=seen_ids_list,
                )
            elif entry.semantic_embeddings and len(entry.semantic_embeddings) >= 1:
                # Single query with precomputed embedding
                results = self._search_semantic(
                    query=queries[0],
                    request=base_request,
                    limit=entry.fetch_size,
                    query_embedding=entry.semantic_embeddings[0],
                    exclude_product_ids=seen_ids_list,
                )
            else:
                # No cached embeddings — re-encode (shouldn't happen normally)
                logger.warning("Extend search: no cached embeddings, re-encoding")
                results = self._search_semantic(
                    query=queries[0],
                    request=base_request,
                    limit=entry.fetch_size,
                    exclude_product_ids=seen_ids_list,
                )

            return results, int((time.time() - t0) * 1000)

        # -----------------------------------------------------------------
        # Run Algolia + Semantic in parallel
        # -----------------------------------------------------------------
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_algolia = executor.submit(_run_algolia_extend)
            future_semantic = executor.submit(_run_semantic_extend)

            try:
                algolia_results, algolia_facets, algolia_ms = future_algolia.result()
                timing["algolia_ms"] = algolia_ms
            except Exception as e:
                logger.warning("Extend-search Algolia failed", error=str(e))
                algolia_results, algolia_facets = [], None
                timing["algolia_ms"] = 0

            try:
                semantic_results, semantic_ms = future_semantic.result()
                timing["semantic_ms"] = semantic_ms
            except Exception as e:
                logger.warning("Extend-search semantic failed", error=str(e))
                semantic_results = []
                timing["semantic_ms"] = 0

        # Enrich semantic results from Algolia (same as page 1)
        if semantic_results and not entry.use_attribute_search:
            semantic_results = self._enrich_semantic_results(semantic_results)

        # -----------------------------------------------------------------
        # RRF merge fresh candidates
        # -----------------------------------------------------------------
        merged = self._reciprocal_rank_fusion(
            algolia_results=algolia_results,
            semantic_results=semantic_results,
            algolia_weight=entry.algolia_weight,
            semantic_weight=entry.semantic_weight,
        )

        # Remove already-seen products from merged results
        merged = [
            r for r in merged
            if r.get("product_id") not in entry.seen_product_ids
        ]

        # -----------------------------------------------------------------
        # Rerank with cached config
        # -----------------------------------------------------------------
        if merged and entry.rerank_kwargs:
            # Load fresh impression counts if user_id available
            rerank_args = dict(entry.rerank_kwargs)
            rerank_args["results"] = merged
            # Use cumulative seen_ids for session dedup
            rerank_args["seen_ids"] = entry.seen_product_ids
            merged = self._reranker.rerank(**rerank_args)

        # -----------------------------------------------------------------
        # Post-sort by price/trending if requested
        # -----------------------------------------------------------------
        if entry.sort_by == SortBy.PRICE_ASC.value:
            merged.sort(key=lambda r: float(r.get("price", 0) or 0))
        elif entry.sort_by == SortBy.PRICE_DESC.value:
            merged.sort(key=lambda r: float(r.get("price", 0) or 0), reverse=True)
        elif entry.sort_by == SortBy.TRENDING.value:
            merged.sort(key=lambda r: float(r.get("trending_score", 0) or 0), reverse=True)

        # -----------------------------------------------------------------
        # Paginate + format
        # -----------------------------------------------------------------
        page_results = merged[:entry.page_size]

        # Determine has_more:
        # - If we got fewer than page_size results, both sources are likely
        #   exhausted — signal no more results.
        # - If we got 0 results, definitely exhausted.
        # - If we got >= page_size, there are probably more.
        has_more = len(page_results) >= entry.page_size

        products = [
            self._to_product_result(r, idx + 1)
            for idx, r in enumerate(page_results)
        ]

        # Update seen IDs with this page's results
        new_ids: List[str] = [
            r["product_id"] for r in page_results if r.get("product_id")
        ]
        entry.add_seen_ids(new_ids)

        # Build next cursor (None when exhausted)
        next_cursor = encode_cursor(page=page + 1) if has_more else None

        timing["total_ms"] = int((time.time() - t_start) * 1000)
        timing["seen_ids_total"] = len(entry.seen_product_ids)

        logger.info(
            "Extend-search page served",
            search_session_id=session_id,
            page=page,
            results=len(products),
            has_more=has_more,
            algolia_fresh=len(algolia_results),
            semantic_fresh=len(semantic_results),
            merged_after_dedup=len(merged),
            total_ms=timing["total_ms"],
        )

        return HybridSearchResponse(
            query=entry.query,
            intent=entry.intent,
            sort_by=entry.sort_by,
            results=products,
            pagination=PaginationInfo(
                page=page,
                page_size=entry.page_size,
                has_more=has_more,
                # Carry forward Algolia's catalog count from page 1.
                total_results=entry.algolia_total_hits if entry.algolia_total_hits > 0 else len(products),
            ),
            search_session_id=session_id,
            cursor=next_cursor,
            timing=timing,
            facets=entry.facets or algolia_facets,
            follow_ups=entry.follow_ups,
            applied_filters=entry.applied_filters,
            answered_dimensions=entry.answered_dimensions,
        )

    # =========================================================================
    # Endless Semantic Search (page 2+ for SPECIFIC / VAGUE intent)
    # =========================================================================

    # Max rounds of pgvector fetch-filter-accumulate before giving up.
    _ENDLESS_MAX_ROUNDS = 10
    # Per-query limit per round (each semantic query fetches this many).
    _ENDLESS_BATCH_SIZE = 100

    @staticmethod
    def _apply_endless_post_filter(
        results: List[dict],
        criteria: Dict[str, Any],
    ) -> List[dict]:
        """Apply cached structural filters to semantic results.

        Only enforces hard structural filters from the page-1 planner output:
        category_l1, category_l2, brands, exclude_brands, min_price, max_price.

        Items with None/missing values for a filter field are EXCLUDED
        (strict mode — same as page-1 post-filter behaviour).

        Args:
            results: Enriched semantic result dicts.
            criteria: Dict with optional keys: category_l1, category_l2,
                brands, exclude_brands, min_price, max_price.

        Returns:
            Filtered list (may be shorter than input).
        """
        if not criteria:
            return results

        filtered = results

        # --- Category L1 ---
        cat_l1 = criteria.get("category_l1")
        if cat_l1:
            vals = {v.lower() for v in cat_l1} if isinstance(cat_l1, list) else {cat_l1.lower()}
            filtered = [
                r for r in filtered
                if r.get("category_l1") and r["category_l1"].lower() in vals
            ]

        # --- Category L2 ---
        cat_l2 = criteria.get("category_l2")
        if cat_l2:
            vals = {v.lower() for v in cat_l2} if isinstance(cat_l2, list) else {cat_l2.lower()}
            filtered = [
                r for r in filtered
                if r.get("category_l2") and (
                    r["category_l2"].lower() in vals
                    or any(v in r["category_l2"].lower() for v in vals)
                )
            ]

        # --- Brand include ---
        brands = criteria.get("brands")
        if brands:
            vals = {v.lower() for v in brands}
            filtered = [
                r for r in filtered
                if r.get("brand") and r["brand"].lower() in vals
            ]

        # --- Brand exclude ---
        exclude_brands = criteria.get("exclude_brands")
        if exclude_brands:
            vals = {v.lower() for v in exclude_brands}
            filtered = [
                r for r in filtered
                if not r.get("brand") or r["brand"].lower() not in vals
            ]

        # --- Price range ---
        min_price = criteria.get("min_price")
        if min_price is not None:
            filtered = [
                r for r in filtered
                if r.get("price") is not None and r["price"] >= min_price
            ]
        max_price = criteria.get("max_price")
        if max_price is not None:
            filtered = [
                r for r in filtered
                if r.get("price") is not None and r["price"] <= max_price
            ]

        return filtered

    def _endless_semantic_page(
        self,
        entry: SearchSessionEntry,
        page: int,
        session_id: str,
    ) -> HybridSearchResponse:
        """Serve page 2+ for SPECIFIC/VAGUE intent via batched pgvector pump.

        Instead of re-running the full hybrid pipeline (planner → Algolia →
        semantic → RRF → reranker), this method:
        1. Queries pgvector with cached embeddings, excluding all seen IDs.
        2. Enriches results from Algolia (batch get_objects).
        3. Applies structural post-filters from the page-1 planner output.
        4. Accumulates survivors until page_size is filled (or exhausted).
        5. Deduplicates (near-dupe only — NO brand/category caps).

        Properties:
        - No planner, no follow-ups, no Algolia search, no RRF.
        - Dedup only (via SessionReranker._deduplicate).
        - Typical latency: 2-3 rounds × ~700ms = ~1.5-2s per page.
        - has_more = False when pgvector returns 0 new candidates in a round.

        Args:
            entry: Cached plan state from page 1.
            page: The requested page number (2+).
            session_id: The search session ID.

        Returns:
            HybridSearchResponse with fresh semantic results.
        """
        t_start = time.time()
        timing: Dict[str, Any] = {
            "endless_semantic": True,
            "page": page,
            "rounds": 0,
            "semantic_ms": 0,
            "enrich_ms": 0,
            "filter_ms": 0,
        }

        criteria = entry.post_filter_criteria or {}
        accumulated: List[dict] = []
        exhausted = False
        round_num = 0

        queries = entry.semantic_queries or [entry.query]
        embeddings = entry.semantic_embeddings

        # Build a minimal request for _search_semantic / _search_semantic_multi.
        # We DON'T pass brand/category filters here — those are applied in
        # post-filter after enrichment (pgvector doesn't have Gemini attrs).
        # We DO pass price + brand filters that pgvector SQL supports natively
        # to reduce the fetch set.
        base_request = HybridSearchRequest(query=entry.query)
        if entry.semantic_request_updates:
            base_request = base_request.model_copy(
                update=entry.semantic_request_updates
            )

        while len(accumulated) < entry.page_size and round_num < self._ENDLESS_MAX_ROUNDS:
            round_num += 1
            seen_ids_list = list(entry.seen_product_ids)

            # --- Fetch from pgvector ---
            t_sem = time.time()
            per_query_limit = max(self._ENDLESS_BATCH_SIZE // max(len(queries), 1), 30)

            if len(queries) > 1 and embeddings is not None:
                batch_results, _ = self._search_semantic_multi(
                    queries=queries,
                    request=base_request,
                    limit_per_query=per_query_limit,
                    precomputed_embeddings=embeddings,
                    exclude_product_ids=seen_ids_list,
                )
            elif embeddings and len(embeddings) >= 1:
                batch_results = self._search_semantic(
                    query=queries[0],
                    request=base_request,
                    limit=self._ENDLESS_BATCH_SIZE,
                    query_embedding=embeddings[0],
                    exclude_product_ids=seen_ids_list,
                )
            else:
                # No cached embeddings — shouldn't happen, but handle gracefully
                logger.warning("Endless semantic: no cached embeddings, re-encoding")
                batch_results = self._search_semantic(
                    query=queries[0],
                    request=base_request,
                    limit=self._ENDLESS_BATCH_SIZE,
                    exclude_product_ids=seen_ids_list,
                )
            timing["semantic_ms"] += int((time.time() - t_sem) * 1000)

            if not batch_results:
                exhausted = True
                break

            # Add ALL fetched IDs to seen set (even ones that will be
            # filtered out) so we never re-fetch them in the next round.
            batch_ids = [r["product_id"] for r in batch_results if r.get("product_id")]
            entry.add_seen_ids(batch_ids)

            # --- Enrich from Algolia ---
            t_enr = time.time()
            if not entry.use_attribute_search:
                batch_results = self._enrich_semantic_results(batch_results)
            timing["enrich_ms"] += int((time.time() - t_enr) * 1000)

            # --- Post-filter ---
            t_flt = time.time()
            survivors = self._apply_endless_post_filter(batch_results, criteria)
            timing["filter_ms"] += int((time.time() - t_flt) * 1000)

            accumulated.extend(survivors)

            logger.info(
                "Endless semantic round",
                round=round_num,
                fetched=len(batch_results),
                survived_filter=len(survivors),
                accumulated=len(accumulated),
                target=entry.page_size,
            )

        timing["rounds"] = round_num

        # --- Deduplicate (near-dupe only, NO brand/category caps) ---
        deduped = self._reranker._deduplicate(accumulated)

        # Take page_size results
        page_results = deduped[:entry.page_size]

        # has_more: False only when pgvector is fully exhausted
        has_more = not exhausted and len(page_results) >= entry.page_size

        products = [
            self._to_product_result(r, idx + 1)
            for idx, r in enumerate(page_results)
        ]

        # Build next cursor
        next_cursor = encode_cursor(page=page + 1) if has_more else None

        timing["total_ms"] = int((time.time() - t_start) * 1000)
        timing["seen_ids_total"] = len(entry.seen_product_ids)
        timing["dedup_removed"] = len(accumulated) - len(deduped)

        logger.info(
            "Endless semantic page served",
            search_session_id=session_id,
            page=page,
            results=len(products),
            has_more=has_more,
            rounds=round_num,
            accumulated_before_dedup=len(accumulated),
            deduped=len(deduped),
            exhausted=exhausted,
            total_ms=timing["total_ms"],
        )

        return HybridSearchResponse(
            query=entry.query,
            intent=entry.intent,
            sort_by=entry.sort_by,
            results=products,
            pagination=PaginationInfo(
                page=page,
                page_size=entry.page_size,
                has_more=has_more,
                total_results=entry.algolia_total_hits if entry.algolia_total_hits > 0 else len(products),
            ),
            search_session_id=session_id,
            cursor=next_cursor,
            timing=timing,
            facets=entry.facets,
            follow_ups=entry.follow_ups,
            applied_filters=entry.applied_filters,
            answered_dimensions=entry.answered_dimensions,
        )

    # =========================================================================
    # Filter Refinement (facet filters applied mid-session)
    # =========================================================================

    # Mapping from user filter snapshot keys to Algolia facet names.
    # Used by _build_user_filter_clauses() to convert a dict of user-selected
    # facet values into an Algolia filter string.
    _USER_FILTER_ALGOLIA_MAP: Dict[str, str] = {
        # Inclusion filters  → Algolia facet attribute
        "brands": "brand",
        "categories": "broad_category",
        "category_l1": "category_l1",
        "category_l2": "category_l2",
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

    # Exclusion filters → Algolia facet attribute (reuses class-level mapping)
    _USER_EXCLUDE_FILTER_ALGOLIA_MAP: Dict[str, str] = {
        "exclude_brands": "brand",
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

    @classmethod
    def _build_user_filter_clauses(cls, user_filters: Dict[str, Any]) -> str:
        """Convert user UI filter snapshot into Algolia filter string clauses.

        Takes the dict produced by ``_snapshot_user_filters()`` and returns
        an Algolia-compatible filter string (AND-joined).  This is merged
        with the cached planner filter string during filter refinement so
        that Algolia returns facets reflecting the user's selection.

        Returns:
            Algolia filter string, or empty string if no filters apply.
        """
        parts: List[str] = []

        # --- Price ---
        min_price = user_filters.get("min_price")
        if min_price is not None:
            parts.append(f"price >= {min_price}")
        max_price = user_filters.get("max_price")
        if max_price is not None:
            parts.append(f"price <= {max_price}")

        # --- Sale / Set ---
        if user_filters.get("on_sale_only") is True:
            parts.append("is_on_sale:true")
        is_set = user_filters.get("is_set")
        if is_set is not None:
            parts.append(f"is_set:{str(is_set).lower()}")

        # --- Inclusion filters (OR within field, AND across fields) ---
        for key, algolia_facet in cls._USER_FILTER_ALGOLIA_MAP.items():
            vals = user_filters.get(key)
            if not vals:
                continue
            if isinstance(vals, list):
                clause = " OR ".join(f'{algolia_facet}:"{v}"' for v in vals)
                parts.append(f"({clause})")
            else:
                parts.append(f'{algolia_facet}:"{vals}"')

        # --- Exclusion filters (NOT per value) ---
        for key, algolia_facet in cls._USER_EXCLUDE_FILTER_ALGOLIA_MAP.items():
            vals = user_filters.get(key)
            if not vals:
                continue
            if isinstance(vals, list):
                for v in vals:
                    parts.append(f'NOT {algolia_facet}:"{v}"')
            else:
                parts.append(f'NOT {algolia_facet}:"{vals}"')

        return " AND ".join(parts)

    def _filter_refine_search(
        self,
        request: HybridSearchRequest,
        user_filters: Dict[str, Any],
        user_id: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        seen_ids: Optional[Set[str]] = None,
        user_context: Optional[Any] = None,
        session_scores: Optional[Any] = None,
    ) -> Optional[HybridSearchResponse]:
        """Apply new facet filters to an existing search session.

        Reuses cached FashionCLIP embeddings and planner state from the
        original search.  Runs Algolia (fresh facets) + semantic (cached
        embeddings, fresh start) + RRF merge + rerank.

        Expected latency: ~3-5s (vs 12-15s full pipeline).

        Returns:
            HybridSearchResponse with new session ID, or None on cache miss.
        """
        from concurrent.futures import ThreadPoolExecutor

        t_start = time.time()
        timing: Dict[str, Any] = {"filter_refine": True}

        # 1. Look up cached session
        cache = SearchSessionCache.get_instance()
        sid = request.search_session_id
        assert sid is not None
        entry = cache.get(sid)
        if entry is None:
            return None

        logger.info(
            "Filter refinement: reusing cached session",
            original_session_id=sid,
            intent=entry.intent,
            user_filters=list(user_filters.keys()),
        )

        # 2. Build merged Algolia filter string:
        #    planner filters (from page 1) AND user UI filter clauses.
        user_clauses = self._build_user_filter_clauses(user_filters)
        if entry.algolia_filters and user_clauses:
            merged_filters = f"{entry.algolia_filters} AND {user_clauses}"
        elif user_clauses:
            merged_filters = user_clauses
        else:
            merged_filters = entry.algolia_filters or ""

        # Ensure in_stock:true is present (defensive)
        if "in_stock:true" not in merged_filters:
            merged_filters = f"in_stock:true AND {merged_filters}" if merged_filters else "in_stock:true"

        logger.info(
            "Filter refinement: merged Algolia filters",
            original_filters=entry.algolia_filters,
            user_clauses=user_clauses,
            merged=merged_filters,
        )

        # 3. Run Algolia + Semantic in parallel
        algolia_results: List[dict] = []
        algolia_facets = None
        algolia_nb_hits = 0
        semantic_results: List[dict] = []

        fetch_size = entry.fetch_size or 150

        def _run_algolia_refine():
            t0 = time.time()
            results, fcts, nb_hits = self._search_algolia(
                query=entry.algolia_query,
                filters=merged_filters,
                hits_per_page=fetch_size,
                facets=_FACET_FIELDS,
                optional_filters=entry.algolia_optional_filters,
            )
            elapsed = int((time.time() - t0) * 1000)
            return results, fcts, elapsed, nb_hits

        def _run_semantic_refine():
            if entry.skip_algolia and not entry.semantic_queries:
                return [], 0
            queries = entry.semantic_queries or [entry.query]
            embeddings = entry.semantic_embeddings

            # Build relaxed request for semantic (same as page 1)
            sem_request = HybridSearchRequest(query=entry.query)
            if entry.semantic_request_updates:
                sem_request = sem_request.model_copy(
                    update=entry.semantic_request_updates
                )

            t0 = time.time()
            if embeddings is not None and len(queries) > 1:
                results, _ = self._search_semantic_multi(
                    queries=queries,
                    request=sem_request,
                    limit_per_query=max(fetch_size // max(len(queries), 1), 30),
                    precomputed_embeddings=embeddings,
                    exclude_product_ids=[],  # fresh start — no exclusions
                )
            elif embeddings is not None and len(embeddings) >= 1:
                results = self._search_semantic(
                    query=queries[0],
                    request=sem_request,
                    limit=fetch_size,
                    query_embedding=embeddings[0],
                    exclude_product_ids=[],  # fresh start
                )
            else:
                # No cached embeddings — shouldn't happen, but handle gracefully
                logger.warning("Filter refine: no cached embeddings, re-encoding")
                results = self._search_semantic(
                    query=queries[0],
                    request=sem_request,
                    limit=fetch_size,
                    exclude_product_ids=[],
                )
            elapsed = int((time.time() - t0) * 1000)
            return results, elapsed

        with ThreadPoolExecutor(max_workers=2) as executor:
            algolia_future = executor.submit(_run_algolia_refine)
            semantic_future = executor.submit(_run_semantic_refine)

            algolia_results, algolia_facets, algolia_ms, algolia_nb_hits = algolia_future.result()
            semantic_results, semantic_ms = semantic_future.result()

        timing["algolia_ms"] = algolia_ms
        timing["semantic_ms"] = semantic_ms
        timing["algolia_results"] = len(algolia_results)
        timing["semantic_results_raw"] = len(semantic_results)

        # 4. Enrich semantic results (batch fetch Gemini attrs from Algolia)
        if semantic_results and not entry.use_attribute_search:
            t0 = time.time()
            semantic_results = self._enrich_semantic_results(semantic_results)
            timing["enrich_ms"] = int((time.time() - t0) * 1000)

        # 5. RRF merge
        t0 = time.time()
        merged = self._reciprocal_rank_fusion(
            algolia_results=algolia_results,
            semantic_results=semantic_results,
            algolia_weight=entry.algolia_weight,
            semantic_weight=entry.semantic_weight,
        )
        timing["rrf_ms"] = int((time.time() - t0) * 1000)
        timing["merged_total"] = len(merged)

        # 6. Apply user post-filters on merged results
        if user_filters and merged:
            merged = self._apply_user_post_filters(merged, user_filters)
            timing["post_filter_results"] = len(merged)

        # 7. Rerank using cached rerank config
        #
        # Override caps when the user explicitly filtered by brand:
        # - max_per_brand=0 → skip brand diversity cap (the user WANTS
        #   all results from that brand; capping to 4 is wrong).
        # - page_size=0 → skip category proportional caps (the original
        #   VAGUE category distribution doesn't apply when the user is
        #   actively narrowing by brand/price/etc.).
        _user_set_brands = bool(user_filters.get("brands"))

        if merged and entry.rerank_kwargs:
            rk = entry.rerank_kwargs.copy()
            rk["results"] = merged
            rk["seen_ids"] = seen_ids
            # Restore runtime-only params that aren't in cached kwargs
            if "user_profile" not in rk:
                rk["user_profile"] = user_profile
            if "user_context" not in rk:
                rk["user_context"] = user_context
            if "session_scores" not in rk:
                rk["session_scores"] = session_scores
            # Disable brand cap when user explicitly filters by brand
            if _user_set_brands:
                rk["max_per_brand"] = 0
            # Disable category proportional caps — user's explicit filter
            # intent overrides the original VAGUE category distribution
            rk["page_size"] = 0
            merged = self._reranker.rerank(**rk)
        elif merged:
            # No cached rerank kwargs — just deduplicate
            merged = self._reranker._deduplicate(merged)

        # 8. Paginate (always page 1 for refinement)
        page_results = merged[:request.page_size]
        has_more = len(merged) > request.page_size

        products = [
            self._to_product_result(r, idx + 1)
            for idx, r in enumerate(page_results)
        ]

        # 9. Use Algolia facets (catalog-wide, not computed from small merged set)
        # For SPECIFIC/VAGUE, Algolia facets are far more accurate than
        # computing from the 45-200 item merged set.
        final_facets = algolia_facets if algolia_facets else entry.facets

        # 10. Create NEW session entry with updated state
        new_session_id: Optional[str] = None
        new_cursor: Optional[str] = None
        if len(merged) > 0:
            try:
                new_session_id = cache.generate_session_id()
                _page1_ids: Set[str] = {
                    r["product_id"] for r in merged if r.get("product_id")
                }

                # Build updated post_filter_criteria from user filters
                _post_criteria_keys = (
                    "category_l1", "category_l2", "brands",
                    "exclude_brands", "min_price", "max_price",
                )
                new_post_criteria = {
                    k: user_filters[k]
                    for k in _post_criteria_keys
                    if k in user_filters and user_filters[k] is not None
                }
                # Also inherit criteria from original entry that user didn't override
                if entry.post_filter_criteria:
                    for k in _post_criteria_keys:
                        if k not in new_post_criteria and k in entry.post_filter_criteria:
                            new_post_criteria[k] = entry.post_filter_criteria[k]

                cache.store(SearchSessionEntry(
                    session_id=new_session_id,
                    query=entry.query,
                    intent=entry.intent,
                    sort_by=entry.sort_by,
                    # Algolia state — use MERGED filters for page 2+
                    algolia_query=entry.algolia_query,
                    algolia_filters=merged_filters,
                    algolia_optional_filters=entry.algolia_optional_filters,
                    # Semantic state — REUSE from original
                    semantic_queries=entry.semantic_queries,
                    semantic_embeddings=entry.semantic_embeddings,
                    semantic_request_updates=entry.semantic_request_updates,
                    # RRF weights — REUSE from original
                    algolia_weight=entry.algolia_weight,
                    semantic_weight=entry.semantic_weight,
                    # Reranker config — REUSE from original
                    rerank_kwargs=entry.rerank_kwargs,
                    # Pagination tracking — FRESH start
                    seen_product_ids=_page1_ids,
                    algolia_page=0,
                    page_size=request.page_size,
                    fetch_size=fetch_size,
                    # Response metadata — UPDATED
                    facets=final_facets,
                    follow_ups=entry.follow_ups,
                    applied_filters=entry.applied_filters,
                    answered_dimensions=entry.answered_dimensions,
                    # Algolia catalog count — UPDATED
                    algolia_total_hits=algolia_nb_hits,
                    # Post-filter criteria — UPDATED with user filters
                    post_filter_criteria=new_post_criteria or None,
                    # Flags — REUSE from original
                    skip_algolia=entry.skip_algolia,
                    use_attribute_search=entry.use_attribute_search,
                    attribute_filters=entry.attribute_filters,
                ))
                new_cursor = encode_cursor(page=2)
                has_more = True
            except Exception as exc:
                logger.warning(
                    "Failed to cache refined search session",
                    error=str(exc),
                )

        timing["total_ms"] = int((time.time() - t_start) * 1000)

        logger.info(
            "Filter refinement complete",
            original_session_id=sid,
            new_session_id=new_session_id,
            algolia_hits=algolia_nb_hits,
            merged=len(merged),
            returned=len(products),
            total_ms=timing["total_ms"],
        )

        return HybridSearchResponse(
            query=entry.query,
            intent=entry.intent,
            sort_by=entry.sort_by,
            results=products,
            pagination=PaginationInfo(
                page=1,
                page_size=request.page_size,
                has_more=has_more,
                total_results=algolia_nb_hits if algolia_nb_hits > 0 else len(merged),
            ),
            search_session_id=new_session_id,
            cursor=new_cursor,
            timing=timing,
            facets=final_facets,
            follow_ups=entry.follow_ups,
            applied_filters=entry.applied_filters,
            answered_dimensions=entry.answered_dimensions,
        )

    # =========================================================================
    # Attribute Search Result Conversion
    # =========================================================================

    @staticmethod
    def _convert_attribute_results(attr_results: list) -> List[dict]:
        """Convert AttributeSearchEngine SearchResult objects to pipeline dicts.

        The rest of the hybrid search pipeline expects results as plain dicts
        with specific keys (product_id, name, brand, semantic_score, source,
        etc.).  This method bridges the two formats.
        """
        results = []
        for r in attr_results:
            results.append({
                "product_id": r.product_id,
                "name": r.name,
                "brand": r.brand,
                "image_url": r.image_url,
                "gallery_images": r.gallery_images or [],
                "price": r.price,
                "original_price": r.original_price,
                "is_on_sale": False,
                "category_l1": r.category_l1,
                "category_l2": r.category_l2,
                "broad_category": None,
                "article_type": None,
                "primary_color": None,
                "color_family": None,
                "pattern": None,
                "apparent_fabric": None,
                "fit_type": None,
                "formality": r.formality,
                "silhouette": None,
                "length": None,
                "neckline": None,
                "sleeve_type": None,
                "rise": None,
                "style_tags": r.style_tags or [],
                "occasions": r.occasions or [],
                "seasons": [],
                "colors": [],
                "materials": [],
                "semantic_score": r.similarity,
                "source": "semantic",
            })
        return results

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
                "is_set": hit.get("is_set", False),
                "set_role": hit.get("set_role"),
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
    ) -> Tuple[List[dict], Optional[Dict[str, List[FacetValue]]], int]:
        """Search Algolia and return normalized results + facet counts + total hits.

        Returns:
            Tuple of (results list, facets dict or None, nbHits from Algolia).
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
                    "is_set": hit.get("is_set", False),
                    "set_role": hit.get("set_role"),
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

            nb_hits = resp.get("nbHits", len(results))
            return results, parsed_facets, nb_hits
        except Exception as e:
            logger.error("Algolia search failed", error=str(e))
            return [], None, 0

    # =========================================================================
    # FashionCLIP Semantic Search
    # =========================================================================

    # Embedding perturbation for cross-user retrieval diversity.
    # Small Gaussian noise seeded per user_id causes different users to
    # get slightly different nearest-neighbor results from pgvector.
    _PERTURBATION_SIGMA = 0.02

    def _perturb_embedding(
        self,
        embedding: np.ndarray,
        user_id: Optional[str],
    ) -> np.ndarray:
        """Add small user-seeded noise to a query embedding for diversity.

        The perturbation is deterministic per user so the same user gets
        consistent results within a session, but different users diverge.
        Returns the original embedding unchanged if no user_id.
        """
        if not user_id:
            return embedding
        seed = hash(user_id) % (2**32)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, self._PERTURBATION_SIGMA, embedding.shape)
        perturbed = embedding + noise
        # Re-normalize to unit length (cosine similarity requires it)
        norm = np.linalg.norm(perturbed)
        if norm > 0:
            perturbed = perturbed / norm
        return perturbed.astype(embedding.dtype)

    def _search_semantic(
        self,
        query: str,
        request: HybridSearchRequest,
        limit: int = 100,
        query_embedding: Optional[np.ndarray] = None,
        user_id: Optional[str] = None,
        exclude_product_ids: Optional[List[str]] = None,
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
            user_id: User ID for seeding embedding perturbation (diversity).
        """
        settings = get_settings()

        # Apply per-user embedding perturbation for retrieval diversity.
        # If no pre-computed embedding, encode first so we can perturb.
        if user_id:
            if query_embedding is None:
                query_embedding = self.semantic_engine.encode_text(query)
            query_embedding = self._perturb_embedding(query_embedding, user_id)

        # Try multimodal search first
        if settings.multimodal_search_enabled:
            multimodal_results = self._search_multimodal(
                query=query, request=request, limit=limit,
                version=settings.multimodal_embedding_version,
                query_embedding=query_embedding,
                exclude_product_ids=exclude_product_ids,
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
            exclude_product_ids=exclude_product_ids,
        )

    def _search_semantic_multi(
        self,
        queries: List[str],
        request: HybridSearchRequest,
        limit_per_query: int = 50,
        user_id: Optional[str] = None,
        exclude_product_ids: Optional[List[str]] = None,
        precomputed_embeddings: Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[dict], Optional[List[np.ndarray]]]:
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

        Returns:
            Tuple of (merged_results, embeddings). The embeddings are the
            batch-encoded query vectors (or precomputed ones) — returned so
            callers can cache them for extend-search pagination.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time as _time

        # Batch-encode ALL query embeddings in a single forward pass,
        # OR reuse precomputed embeddings from a cached session (page 2+).
        if precomputed_embeddings is not None:
            embeddings = precomputed_embeddings
            encode_ms = 0.0
            logger.info(
                "Reusing precomputed semantic embeddings (extend search)",
                query_count=len(queries),
            )
        else:
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
                user_id=user_id,
                exclude_product_ids=exclude_product_ids,
            )
            return idx, results

        # Run all pgvector RPC queries in parallel (embeddings already computed)
        with ThreadPoolExecutor(max_workers=min(len(queries), 6)) as executor:
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
        # Also count how many queries each product appeared in — products
        # found by multiple overlapping queries are more likely to truly
        # match the specific detail the user asked for.
        seen: Dict[str, dict] = {}  # product_id -> best result dict
        query_assignments: Dict[str, int] = {}  # product_id -> query index
        overlap_counts: Dict[str, int] = defaultdict(int)  # product_id -> # queries

        for q_idx, results in enumerate(results_per_query):
            for item in results:
                pid = item.get("product_id")
                if not pid:
                    continue
                overlap_counts[pid] += 1
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
            return [], embeddings

        # Multi-query overlap boost: products found by 2+ queries get
        # a semantic_score bonus.  When all semantic queries describe
        # the same detail (e.g., "zipped pockets" from 3 angles), items
        # matching multiple phrasings are far more likely to actually
        # have that detail.  Boost = 0.03 per extra query hit.
        _OVERLAP_BOOST_PER_HIT = 0.03
        num_queries = len(queries)
        overlap_boosted = 0
        for pid, count in overlap_counts.items():
            if count >= 2 and pid in seen:
                boost = (count - 1) * _OVERLAP_BOOST_PER_HIT
                item = seen[pid]
                old_score = item.get("semantic_score", 0) or 0
                item["semantic_score"] = old_score + boost
                item["overlap_count"] = count
                overlap_boosted += 1

        if overlap_boosted:
            logger.info(
                "Multi-query overlap boost applied",
                boosted_items=overlap_boosted,
                num_queries=num_queries,
                boost_per_hit=_OVERLAP_BOOST_PER_HIT,
            )

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
        return merged, embeddings

    def _search_multimodal(
        self,
        query: str,
        request: HybridSearchRequest,
        limit: int = 100,
        version: int = 1,
        query_embedding: Optional[np.ndarray] = None,
        exclude_product_ids: Optional[List[str]] = None,
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
                exclude_product_ids=exclude_product_ids,
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
        exclude_product_ids: Optional[List[str]] = None,
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
                exclude_product_ids=exclude_product_ids,
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
        # Core attributes
        "category_l1", "category_l2", "category_l3", "primary_color", "color_family",
        "formality", "silhouette", "neckline", "rise", "apparent_fabric",
        "seasons", "fit_type", "sleeve_type", "length", "pattern", "pattern_scale",
        "style_tags", "occasions", "article_type", "broad_category",
        "is_on_sale", "original_price", "trending_score",
        "is_set", "set_role",
        # Coverage (v1.0.0.2)
        "arm_coverage", "shoulder_coverage", "neckline_depth",
        "midriff_exposure", "back_openness", "sheerness_visual",
        # Shape / Silhouette (v1.0.0.2)
        "body_cling_visual", "structure_level", "drape_level",
        "cropped_degree", "waist_definition_visual", "leg_volume_visual",
        # Details (v1.0.0.2)
        "has_pockets", "slit_presence", "slit_height",
        "detail_tags", "lining_status_likely", "pocket_types",
        # Metadata
        "vibe_tags", "coverage_level", "styling_role",
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

        if request.is_set is not None:
            hard_parts.append(f"is_set:{str(request.is_set).lower()}")

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

    def _build_cluster_brand_filters(
        self,
        user_profile: Dict[str, Any],
    ) -> List[str]:
        """Build Algolia optionalFilters to boost the user's cluster brands.

        Looks up which clusters the user's preferred brands belong to, then
        collects all properly-cased brand names from those clusters.  These
        become optionalFilters — Algolia boosts matching items' ranking
        without excluding non-matching ones.

        Returns:
            List of filter strings like ``'brand:"Ba&sh"'``.
            Empty list if the user has no preferred brands or cluster data.
        """
        preferred_brands = user_profile.get("preferred_brands") or []
        if not preferred_brands:
            return []

        try:
            from recs.brand_clusters import get_brand_clusters
            from scoring.constants.brand_data import BRAND_CLUSTERS
        except ImportError:
            return []

        # Find all clusters the user's preferred brands belong to
        user_cluster_ids: set = set()
        for brand in preferred_brands:
            for cid, _conf in get_brand_clusters(brand):
                user_cluster_ids.add(cid)

        if not user_cluster_ids:
            return []

        # Collect all properly-cased brand names from those clusters
        seen: set = set()
        filters: List[str] = []
        for cid in user_cluster_ids:
            for brand in BRAND_CLUSTERS.get(cid, []):
                key = brand.lower()
                if key not in seen:
                    seen.add(key)
                    filters.append(f'brand:"{brand}"')

        return filters

    def _build_vibe_brand_filters(self, vibe_brand: str) -> List[str]:
        """Build Algolia optionalFilters from a vibe brand's clusters.

        When the LLM identifies a brand as a style reference (not a purchase
        target), look up which cluster(s) that brand belongs to and collect
        all in-inventory brands from those clusters.  These become
        optionalFilters — Algolia boosts matching items without excluding
        non-matching ones.

        The LLM handles brand detection (including typo correction), so
        ``vibe_brand`` is the LLM-normalized brand name (e.g.,
        "Anthropologie" even if the user typed "Antropologie").

        Args:
            vibe_brand: Brand name as output by the LLM (may be properly cased
                or lowercase).

        Returns:
            List of filter strings like ``'brand:"Ba&sh"'``.
            Empty list if the brand is unknown or has no cluster data.
        """
        try:
            from recs.brand_clusters import get_brand_clusters
            from scoring.constants.brand_data import BRAND_CLUSTERS
        except ImportError:
            logger.debug("Brand cluster modules not available for vibe brand")
            return []

        vibe_lower = vibe_brand.lower()
        cluster_ids: set = set()
        for cid, _conf in get_brand_clusters(vibe_lower):
            cluster_ids.add(cid)

        if not cluster_ids:
            logger.info(
                "Vibe brand not found in cluster map — no optionalFilters",
                vibe_brand=vibe_brand,
            )
            return []

        # Collect all properly-cased brand names from the vibe brand's clusters
        seen: set = set()
        filters: List[str] = []
        for cid in cluster_ids:
            for brand in BRAND_CLUSTERS.get(cid, []):
                key = brand.lower()
                if key not in seen:
                    seen.add(key)
                    filters.append(f'brand:"{brand}"')

        logger.info(
            "Built vibe-brand cluster filters",
            vibe_brand=vibe_brand,
            clusters=sorted(cluster_ids),
            brand_count=len(filters),
        )
        return filters

    def _build_algolia_filters(self, request: HybridSearchRequest) -> Optional[str]:
        """Build Algolia filter string from request parameters."""
        parts = []

        # Always filter in-stock
        parts.append("in_stock:true")

        if request.on_sale_only:
            parts.append("is_on_sale:true")

        if request.is_set is not None:
            parts.append(f"is_set:{str(request.is_set).lower()}")

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
            is_set=item.get("is_set"),
            set_role=item.get("set_role"),
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
        force_soft: bool = False,
    ) -> List[dict]:
        """
        Post-filter on semantic results after Algolia enrichment.

        Two modes:
        1. **Strict mode** (no expanded_filters / regex fallback): All filters
           are hard -- non-matching results are dropped. This is the original
           behavior.
        2. **Relaxed mode** (expanded_filters from LLM planner OR force_soft
           from refine path): Only hard filters (brand, category, price) drop
           results. Attribute filters (occasions, materials, style_tags,
           formality, etc.) become soft scoring boosts -- matching items rank
           higher, but non-matching items aren't dropped. This prevents
           0-result scenarios when DB labels don't match the user's language
           or when multiple follow-up selections are too restrictive together.

        When drop_nulls=False (progressive relaxation), exclusion filters keep
        items with null/N/A attribute values instead of dropping them.
        """
        if expanded_filters is None:
            expanded_filters = {}

        use_soft_scoring = bool(expanded_filters) or force_soft
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

        # --- Set / co-ord filter ---
        if request.is_set is not None:
            filtered = [r for r in filtered if bool(r.get("is_set")) == request.is_set]

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
            "on_sale_only", "is_set",
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
