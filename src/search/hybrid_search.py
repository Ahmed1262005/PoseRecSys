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

from core.logging import get_logger
from core.utils import filter_gallery_images
from search.algolia_client import AlgoliaClient, get_algolia_client
from search.query_classifier import QueryClassifier, QueryIntent
from search.reranker import SessionReranker
from search.analytics import SearchAnalytics, get_search_analytics
from search.models import (
    FacetValue,
    HybridSearchRequest,
    HybridSearchResponse,
    ProductResult,
    PaginationInfo,
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
        self._classifier = QueryClassifier()

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

        Returns:
            HybridSearchResponse with results and metadata.
        """
        t_start = time.time()
        timing: Dict[str, int] = {}

        # Step 0: Normalize query (decode HTML entities, collapse whitespace)
        clean_query = html_mod.unescape(request.query).strip()
        if clean_query != request.query:
            request = request.model_copy(update={"query": clean_query})

        # Step 1: Classify query intent
        intent = self._classifier.classify(request.query)
        algolia_weight = self._classifier.get_algolia_weight(intent)
        semantic_weight = self._classifier.get_semantic_weight(intent)

        # Allow request-level override of semantic weight.
        # Use field info to detect if user explicitly set semantic_boost
        # (avoids fragile float equality comparison with default 0.4).
        _SEMANTIC_BOOST_DEFAULT = 0.4
        if abs(request.semantic_boost - _SEMANTIC_BOOST_DEFAULT) > 1e-9:
            semantic_weight = request.semantic_boost
            algolia_weight = 1.0 - semantic_weight

        # Step 2: For exact brand matches, inject a brand filter so Algolia
        # doesn't misinterpret special chars (e.g. "Ba&sh" splitting on &)
        matched_brand = None
        if intent == QueryIntent.EXACT:
            matched_brand = self._classifier.extract_brand(request.query)
            if matched_brand and not request.brands:
                request = request.model_copy(update={"brands": [matched_brand]})

        # Step 2a: Extract structured attribute filters from the query.
        # Maps natural language terms to Algolia facet values so "formal"
        # becomes formality:"Formal", "floral midi dress" becomes
        # pattern:"Floral" + length:"Midi", etc.
        # Only inject filters the user hasn't already set explicitly.
        # Also returns the matched terms so we can strip them from the
        # Algolia text query (they're now handled by facet filters).
        extracted, matched_terms = self._classifier.extract_attributes(request.query)
        if extracted:
            updates = {}
            for field, values in extracted.items():
                if not getattr(request, field, None):
                    updates[field] = values
            if updates:
                request = request.model_copy(update=updates)
                logger.info(
                    "Auto-extracted attribute filters from query",
                    query=request.query,
                    extracted=updates,
                )

        # Step 2b: Clean query for Algolia — strip:
        # 1. Meta-terms ("outfit", "look", "clothes") that describe intent
        # 2. Attribute terms that were converted to facet filters ("formal",
        #    "floral", "silk") — keeping them as text AND filters causes
        #    Algolia to over-restrict (e.g. "formal shirt" returns 1 hit
        #    because few products have "formal" in name + formality:Formal).
        algolia_query = self._clean_query_for_algolia(request.query, matched_terms)

        # Step 2c: Build Algolia filter string
        algolia_filters = self._build_algolia_filters(request)

        # Step 3: Run Algolia search
        # Auto-detect active filters from the request model.
        # Checks all Optional fields that are not pagination/query/session/boost.
        _NON_FILTER_FIELDS = {"query", "page", "page_size", "session_id", "semantic_boost"}
        has_filters = any(
            getattr(request, field_name) not in (None, False, [])
            for field_name in HybridSearchRequest.model_fields
            if field_name not in _NON_FILTER_FIELDS
        )
        # Fetch extra semantic candidates when filters are active,
        # since strict post-filtering will drop non-matching results
        fetch_multiplier = 5 if has_filters else 3
        fetch_size = request.page_size * fetch_multiplier
        t_algolia = time.time()
        algolia_results, facets = self._search_algolia(
            query=algolia_query,
            filters=algolia_filters,
            hits_per_page=fetch_size,
            facets=_FACET_FIELDS,
        )
        timing["algolia_ms"] = int((time.time() - t_algolia) * 1000)

        # Step 4: Run semantic search
        # - Skip for EXACT brand matches (Algolia is authoritative)
        # - Force semantic if Algolia returned nothing (graceful fallback)
        semantic_results = []
        algolia_failed = len(algolia_results) == 0
        run_semantic = (intent != QueryIntent.EXACT) or algolia_failed
        if run_semantic:
            t_semantic = time.time()
            semantic_results = self._search_semantic(
                query=request.query,
                request=request,
                limit=fetch_size,
            )
            timing["semantic_ms"] = int((time.time() - t_semantic) * 1000)
            if algolia_failed and semantic_results:
                logger.info(
                    "Algolia returned 0 results, falling back to semantic-only",
                    query=request.query,
                    semantic_count=len(semantic_results),
                )

        # Step 4b: Enrich semantic results with Gemini attributes from Algolia
        if semantic_results:
            semantic_results = self._enrich_semantic_results(semantic_results)

        # Step 4c: Post-filter semantic results to enforce filters Algolia handles
        if semantic_results:
            semantic_results = self._post_filter_semantic(semantic_results, request)

        # Step 5: Merge with RRF
        merged = self._reciprocal_rank_fusion(
            algolia_results=algolia_results,
            semantic_results=semantic_results,
            algolia_weight=algolia_weight,
            semantic_weight=semantic_weight,
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
            results=products,
            pagination=PaginationInfo(
                page=request.page,
                page_size=request.page_size,
                has_more=has_more,
                total_results=len(merged),
            ),
            timing=timing,
            facets=facets,
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
    ) -> Tuple[List[dict], Optional[Dict[str, List[FacetValue]]]]:
        """Search Algolia and return normalized results + facet counts.

        Returns:
            Tuple of (results list, facets dict or None).
        """
        try:
            resp = self.algolia.search(
                query=query,
                filters=filters,
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
    ) -> List[dict]:
        """Search using FashionCLIP + pgvector and return normalized results."""
        try:
            resp = self.semantic_engine.search_with_filters(
                query=query,
                page=1,
                page_size=limit,
                categories=request.categories,
                exclude_brands=request.exclude_brands,
                include_brands=request.brands,
                include_colors=request.colors,
                min_price=request.min_price,
                max_price=request.max_price,
                occasions=request.occasions,
                patterns=request.patterns,
                use_hybrid_search=False,  # Pure semantic - Algolia handles keyword
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
            logger.error("Semantic search failed", error=str(e))
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

    def _post_filter_semantic(
        self,
        results: List[dict],
        request: HybridSearchRequest,
    ) -> List[dict]:
        """
        Hard post-filter on semantic results after Algolia enrichment.

        This runs AFTER _enrich_semantic_results, so most products now have
        Gemini attributes. We use strict filtering: if a filter is active and
        the product lacks that attribute, it is excluded. This prevents
        un-enriched products from leaking through as false positives.

        Filters already applied natively by the pgvector RPC (categories,
        brands, colors, price, occasions, patterns) are re-checked here to
        catch any that slipped through or were only partially applied.
        """
        filtered = results

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
            vals = {v.lower() for v in request.brands}
            filtered = [r for r in filtered
                        if r.get("brand") and r["brand"].lower() in vals]
        if request.exclude_brands:
            vals = {v.lower() for v in request.exclude_brands}
            filtered = [r for r in filtered
                        if not r.get("brand") or r["brand"].lower() not in vals]

        # --- Category filters (broad) ---
        if request.categories:
            vals = {v.lower() for v in request.categories}
            filtered = [r for r in filtered
                        if r.get("broad_category") and r["broad_category"].lower() in vals]

        # --- Color filters ---
        if request.colors:
            vals = {v.lower() for v in request.colors}
            filtered = [r for r in filtered
                        if (r.get("primary_color") and r["primary_color"].lower() in vals)
                        or (r.get("colors") and any(c.lower() in vals for c in r["colors"]))]

        # --- Pattern filter ---
        if request.patterns:
            vals = {v.lower() for v in request.patterns}
            filtered = [r for r in filtered
                        if r.get("pattern") and r["pattern"].lower() in vals]

        # --- Occasion filter ---
        if request.occasions:
            vals = {v.lower() for v in request.occasions}
            filtered = [r for r in filtered
                        if r.get("occasions") and any(o.lower() in vals for o in r["occasions"])]

        # --- Strict Gemini attribute filters ---
        # After enrichment, products should have these fields. If they don't
        # (not in Algolia), they are excluded when that filter is active.
        _STRICT_SINGLE = [
            ("category_l1", "category_l1"),
            ("category_l2", "category_l2"),
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
