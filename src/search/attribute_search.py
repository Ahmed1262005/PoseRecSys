"""
Attribute-aware semantic search engine.

Architecture:
  User query
    -> Query planner (gpt-4.1-mini) -> SearchPlan
    -> FashionCLIP encodes semantic_queries -> embedding
    -> plan_to_attribute_filters() translates detail_terms -> AttributeFilters
    -> Supabase RPC: pgvector similarity + attribute WHERE clauses
    -> Results ranked by semantic similarity, filtered by attributes

Semantic search is ALWAYS the primary retrieval.  Attribute filters
are applied only when the planner detects specific visual details
(pockets, backless, lace trim, etc.) — replacing the vision reranker.

When no attribute filters are active, the RPC degrades to a standard
multimodal pgvector search across ALL products.

Requires:
  - sql/060_search_with_attributes.sql deployed to Supabase
  - FashionCLIP model (patrickjohncyh/fashion-clip) — lazy-loaded
  - Optionally: OpenAI API key for query planner (gpt-4.1-mini)
"""

from __future__ import annotations

import logging
import re
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional

from config.database import get_supabase_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AttributeFilters dataclass
# ---------------------------------------------------------------------------

@dataclass
class AttributeFilters:
    """
    Structured attribute filters for the search RPC.

    These map 1:1 to the SQL parameters of search_semantic_with_attributes().
    Only non-None fields become WHERE clauses.
    """

    # -- Basic / taxonomy --
    category_l1: list[str] | None = None
    category_l2: list[str] | None = None
    brands: list[str] | None = None
    exclude_brands: list[str] | None = None
    min_price: float | None = None
    max_price: float | None = None

    # -- Coverage (Layer 4A) --
    arm_coverage: list[str] | None = None
    shoulder_coverage: list[str] | None = None
    neckline_depth: list[str] | None = None
    midriff_exposure: list[str] | None = None
    back_openness: list[str] | None = None
    sheerness_visual: list[str] | None = None

    # -- Shape / silhouette (Layer 4E) --
    body_cling_visual: list[str] | None = None
    structure_level: list[str] | None = None
    drape_level: list[str] | None = None
    bulk_visual: list[str] | None = None
    cropped_degree: list[str] | None = None
    waist_definition_visual: list[str] | None = None
    leg_volume_visual: list[str] | None = None

    # -- Details (Layer 4 + 5) --
    has_pockets: bool | None = None
    pocket_types: list[str] | None = None       # e.g. ["Cargo", "Patch", "Zip"]
    pocket_has_zip: bool | None = None           # True = zippered pockets
    slit_presence: bool | None = None
    slit_height: list[str] | None = None
    detail_tags: list[str] | None = None
    lining: list[str] | None = None

    # -- Occasions / style --
    occasions: list[str] | None = None
    formality: list[str] | None = None
    seasons: list[str] | None = None

    # -- Pagination --
    limit: int = 60
    offset: int = 0

    def has_attribute_filters(self) -> bool:
        """True if any attribute filter (beyond basic category/price/brand) is set."""
        for f in [
            self.arm_coverage, self.shoulder_coverage, self.neckline_depth,
            self.midriff_exposure, self.back_openness, self.sheerness_visual,
            self.body_cling_visual, self.structure_level, self.drape_level,
            self.bulk_visual, self.cropped_degree, self.waist_definition_visual,
            self.leg_volume_visual,
            self.slit_height, self.detail_tags, self.lining,
            self.occasions, self.formality, self.seasons,
        ]:
            if f:
                return True
        if self.has_pockets is not None or self.slit_presence is not None:
            return True
        return False

    def has_detail_attribute_filters(self) -> bool:
        """True only if v1.0.0.2-specific visual/detail attributes are set.

        This excludes standard planner fields (formality, occasions, seasons)
        which are present on most queries. Use this to gate whether the
        attribute search engine should replace the standard semantic path —
        we only want that for true detail queries like "dress with pockets"
        or "backless top", not for vague queries that merely have formality.
        """
        # Coverage attributes
        for f in [
            self.arm_coverage, self.shoulder_coverage, self.neckline_depth,
            self.midriff_exposure, self.back_openness, self.sheerness_visual,
        ]:
            if f:
                return True
        # Shape / silhouette attributes
        for f in [
            self.body_cling_visual, self.structure_level, self.drape_level,
            self.bulk_visual, self.cropped_degree, self.waist_definition_visual,
            self.leg_volume_visual,
        ]:
            if f:
                return True
        # Detail attributes
        if self.has_pockets is not None or self.slit_presence is not None:
            return True
        if self.pocket_has_zip is not None:
            return True
        for f in [self.pocket_types, self.slit_height, self.detail_tags, self.lining]:
            if f:
                return True
        return False

    def describe(self) -> str:
        """Human-readable summary of active filters."""
        parts = []
        for fname in sorted(vars(self)):
            val = getattr(self, fname)
            if fname in ("limit", "offset"):
                continue
            if val is None:
                continue
            if isinstance(val, list) and not val:
                continue
            parts.append(f"{fname}={val}")
        return ", ".join(parts) if parts else "(no filters)"


# ---------------------------------------------------------------------------
# Detail term -> Attribute filter translation
#
# This is the core mapping that replaces the vision reranker.
# The query planner outputs detail_terms like "pockets", "lace trim",
# "backless".  This map converts those to structured attribute filters
# that can be enforced at the SQL level.
# ---------------------------------------------------------------------------

DETAIL_TERM_MAP: dict[str, dict[str, Any]] = {
    # ---- Coverage / exposure ----
    "backless":         {"back_openness": ["open", "partial"]},
    "open back":        {"back_openness": ["open", "partial"]},
    "off shoulder":     {"shoulder_coverage": ["off_shoulder"]},
    "off-shoulder":     {"shoulder_coverage": ["off_shoulder"]},
    "one shoulder":     {"shoulder_coverage": ["one_shoulder"]},
    "strapless":        {"shoulder_coverage": ["exposed"], "arm_coverage": ["none"]},
    "sleeveless":       {"arm_coverage": ["none"]},
    "long sleeve":      {"arm_coverage": ["full"]},
    "long sleeves":     {"arm_coverage": ["full"]},
    "short sleeve":     {"arm_coverage": ["half", "short"]},
    "short sleeves":    {"arm_coverage": ["half", "short"]},
    "cropped":          {"cropped_degree": ["moderate", "very", "slightly"]},
    "crop":             {"cropped_degree": ["moderate", "very", "slightly"]},
    "sheer":            {"sheerness_visual": ["semi_sheer"]},
    "see through":      {"sheerness_visual": ["semi_sheer"]},
    "see-through":      {"sheerness_visual": ["semi_sheer"]},
    "low cut":          {"neckline_depth": ["low", "deep"]},
    "low-cut":          {"neckline_depth": ["low", "deep"]},
    "plunging":         {"neckline_depth": ["deep"]},
    "plunging neckline": {"neckline_depth": ["deep"]},
    "deep v":           {"neckline_depth": ["deep"]},
    "high neck":        {"neckline_depth": ["high"]},
    "high-neck":        {"neckline_depth": ["high"]},
    "midriff-baring":   {"midriff_exposure": ["exposed"]},
    "exposed midriff":  {"midriff_exposure": ["exposed", "partial"]},

    # ---- Silhouette / fit ----
    "bodycon":       {"body_cling_visual": ["bodycon"]},
    "body-con":      {"body_cling_visual": ["bodycon"]},
    "fitted":        {"body_cling_visual": ["bodycon", "skim"]},
    "tight":         {"body_cling_visual": ["bodycon"]},
    "slim fit":      {"body_cling_visual": ["skim", "slim"]},
    "slim-fit":      {"body_cling_visual": ["skim", "slim"]},
    "loose":         {"body_cling_visual": ["loose", "relaxed"]},
    "loose fit":     {"body_cling_visual": ["loose", "relaxed"]},
    "oversized":     {"body_cling_visual": ["loose"], "bulk_visual": ["moderate", "bulky"]},
    "relaxed fit":   {"body_cling_visual": ["relaxed", "loose"]},
    "structured":    {"structure_level": ["structured"]},
    "unstructured":  {"structure_level": ["unstructured", "soft"]},
    "flowy":         {"drape_level": ["high", "moderate"], "body_cling_visual": ["loose", "relaxed"]},
    "flowing":       {"drape_level": ["high", "moderate"]},
    "draped":        {"drape_level": ["high"]},
    "wide leg":      {"leg_volume_visual": ["wide"]},
    "wide-leg":      {"leg_volume_visual": ["wide"]},
    "flared":        {"leg_volume_visual": ["flared"]},
    "flare":         {"leg_volume_visual": ["flared"]},
    "bell bottom":   {"leg_volume_visual": ["flared"]},
    "skinny":        {"leg_volume_visual": ["skinny"]},
    "straight leg":  {"leg_volume_visual": ["straight"]},
    "straight-leg":  {"leg_volume_visual": ["straight"]},
    "balloon":       {"leg_volume_visual": ["balloon"]},
    "cinched waist": {"waist_definition_visual": ["cinched"]},
    "cinched":       {"waist_definition_visual": ["cinched"]},
    "defined waist": {"waist_definition_visual": ["defined", "cinched"]},
    "drop shoulder": {"shoulder_shape_visual": ["dropped"]},
    "drop-shoulder": {"shoulder_shape_visual": ["dropped"]},
    "sleek":         {"bulk_visual": ["sleek"]},

    # ---- Details (Layer 4 + 5) ----
    "pockets":           {"has_pockets": True},
    "with pockets":      {"has_pockets": True},
    "has pockets":       {"has_pockets": True},
    "zipped pockets":    {"has_pockets": True, "pocket_has_zip": True},
    "zip pockets":       {"has_pockets": True, "pocket_has_zip": True},
    "zipper pockets":    {"has_pockets": True, "pocket_has_zip": True},
    "zippered pockets":  {"has_pockets": True, "pocket_has_zip": True},
    "zip pocket":        {"has_pockets": True, "pocket_has_zip": True},
    "zipper pocket":     {"has_pockets": True, "pocket_has_zip": True},
    "cargo pockets":     {"has_pockets": True, "pocket_types": ["cargo"]},
    "cargo pocket":      {"has_pockets": True, "pocket_types": ["cargo"]},
    "patch pockets":     {"has_pockets": True, "pocket_types": ["patch"]},
    "patch pocket":      {"has_pockets": True, "pocket_types": ["patch"]},
    "welt pockets":      {"has_pockets": True, "pocket_types": ["welt", "jetted"]},
    "welt pocket":       {"has_pockets": True, "pocket_types": ["welt", "jetted"]},
    "flap pockets":      {"has_pockets": True, "pocket_types": ["flap"]},
    "flap pocket":       {"has_pockets": True, "pocket_types": ["flap"]},
    "kangaroo pocket":   {"has_pockets": True, "pocket_types": ["kangaroo"]},
    "slash pockets":     {"has_pockets": True, "pocket_types": ["slash", "seam"]},
    "slash pocket":      {"has_pockets": True, "pocket_types": ["slash", "seam"]},
    "inseam pockets":    {"has_pockets": True, "pocket_types": ["inseam"]},
    "inseam pocket":     {"has_pockets": True, "pocket_types": ["inseam"]},
    "slit":           {"slit_presence": True},
    "with slit":      {"slit_presence": True},
    "high slit":      {"slit_presence": True, "slit_height": ["high"]},
    "thigh slit":     {"slit_presence": True, "slit_height": ["high", "mid"]},
    "lace":           {"detail_tags": ["lace_trim"]},
    "lace trim":      {"detail_tags": ["lace_trim"]},
    "ruffle":         {"detail_tags": ["ruffle_detail"]},
    "ruffled":        {"detail_tags": ["ruffle_detail"]},
    "ruffle detail":  {"detail_tags": ["ruffle_detail"]},
    "distressed":     {"detail_tags": ["distressed_detail"]},
    "ripped":         {"detail_tags": ["distressed_detail"]},
    "embroidered":    {"detail_tags": ["embroidery_detail"]},
    "embroidery":     {"detail_tags": ["embroidery_detail"]},
    "crochet":        {"detail_tags": ["crochet_detail"]},
    "scalloped":      {"detail_tags": ["scalloped_hem"]},
    "scallop hem":    {"detail_tags": ["scalloped_hem"]},
    "ribbed":         {"detail_tags": ["ribbed_trim"]},
    "mesh":           {"detail_tags": ["mesh_panels"]},
    "fringe":         {"detail_tags": ["fringe_detail"]},
    "raw hem":        {"detail_tags": ["raw_hem"]},
    "frayed":         {"detail_tags": ["frayed_edge"]},
    "ruched":         {"detail_tags": ["ruched_bodice"]},
    "quilted":        {"detail_tags": ["quilted_texture"]},
    "pleated":        {"detail_tags": ["pleated_detail"]},
    "cutout":         {"detail_tags": ["cutout_detail"]},
    "cut-out":        {"detail_tags": ["cutout_detail"]},
    "cutouts":        {"detail_tags": ["cutout_detail"]},

    # ---- Lining ----
    "lined":     {"lining": ["lined", "partially_lined"]},
    "unlined":   {"lining": ["unlined"]},
}

# Pre-sort by phrase length descending so longer phrases match first
_SORTED_TERMS = sorted(DETAIL_TERM_MAP.keys(), key=len, reverse=True)


def plan_to_attribute_filters(
    plan: Any,  # SearchPlan — Any to avoid circular import
    *,
    query: str = "",
) -> AttributeFilters:
    """
    Convert a SearchPlan into AttributeFilters.

    Sources:
      1. plan.detail_terms  → matched against DETAIL_TERM_MAP
      2. plan.attributes    → category_l1, brand, price, occasions, etc.
      3. plan.modes         → some modes map to attribute filters
      4. Raw query text     → fallback matching against DETAIL_TERM_MAP

    Returns filters with only the fields the planner detected.
    """
    merged: dict[str, Any] = {}

    # ---- 1. Translate detail_terms via the map ----
    if hasattr(plan, "detail_terms"):
        for term in (plan.detail_terms or []):
            _match_term(term.lower(), merged)

    # ---- 2. Also scan the raw query for detail terms the planner missed ----
    if query:
        _scan_query_for_details(query.lower(), merged)

    # ---- 3. Pull standard attributes from the plan ----
    attrs = getattr(plan, "attributes", {}) or {}

    # Standard taxonomy / occasion keys
    for key in ("category_l1", "category_l2", "occasions", "formality", "seasons"):
        if key in attrs:
            merged.setdefault(key, attrs[key])

    # v1.0.0.2 detail attribute keys — pass through directly from planner
    _V1002_PASSTHROUGH = {
        "back_openness", "shoulder_coverage", "arm_coverage",
        "neckline_depth", "midriff_exposure", "sheerness_visual",
        "body_cling_visual", "structure_level", "drape_level",
        "cropped_degree", "waist_definition_visual", "leg_volume_visual",
        "bulk_visual", "has_pockets", "pocket_types", "pocket_has_zip",
        "slit_presence", "slit_height", "detail_tags", "lining_status_likely",
    }
    for key in _V1002_PASSTHROUGH:
        if key in attrs:
            # lining_status_likely in planner maps to lining in AttributeFilters
            target = "lining" if key == "lining_status_likely" else key
            merged.setdefault(target, attrs[key])

    # ---- 4. Brand / price from plan ----
    brand = getattr(plan, "brand", None)
    if brand:
        merged.setdefault("brands", [brand])

    max_price = getattr(plan, "max_price", None)
    if max_price is not None:
        merged["max_price"] = max_price
    min_price = getattr(plan, "min_price", None)
    if min_price is not None:
        merged["min_price"] = min_price

    return AttributeFilters(**merged)


def _match_term(term: str, merged: dict) -> bool:
    """Try to match a detail term against the map. Returns True if matched."""
    term_lower = term.strip()
    for phrase in _SORTED_TERMS:
        if phrase == term_lower or phrase in term_lower:
            for k, v in DETAIL_TERM_MAP[phrase].items():
                if k not in merged:
                    merged[k] = v
                elif isinstance(v, list) and isinstance(merged[k], list):
                    existing = set(merged[k])
                    for item in v:
                        if item not in existing:
                            merged[k].append(item)
                            existing.add(item)
                elif isinstance(v, bool):
                    merged[k] = v
            return True
    return False


def _scan_query_for_details(query_lower: str, merged: dict):
    """Scan raw query text for detail terms the planner may have missed."""
    remaining = query_lower
    for phrase in _SORTED_TERMS:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        if re.search(pattern, remaining):
            for k, v in DETAIL_TERM_MAP[phrase].items():
                if k not in merged:
                    merged[k] = v
                elif isinstance(v, list) and isinstance(merged[k], list):
                    existing = set(merged[k])
                    for item in v:
                        if item not in existing:
                            merged[k].append(item)
                            existing.add(item)
                elif isinstance(v, bool):
                    merged[k] = v
            remaining = re.sub(pattern, " ", remaining).strip()


# ---------------------------------------------------------------------------
# Search result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single search result with product + attribute data."""
    product_id: str
    name: str
    brand: str
    price: float
    original_price: float | None
    image_url: str
    gallery_images: list[str] | None
    in_stock: bool
    similarity: float
    # Key attributes for display
    category_l1: str | None = None
    category_l2: str | None = None
    body_cling: str | None = None
    structure: str | None = None
    drape: str | None = None
    arm_coverage: str | None = None
    shoulder_coverage: str | None = None
    back_openness: str | None = None
    midriff_exposure: str | None = None
    neckline_depth: str | None = None
    sheerness: str | None = None
    cropped: str | None = None
    waist_definition: str | None = None
    has_pockets: bool = False
    slit_presence: bool = False
    slit_height: str | None = None
    detail_tags: list[str] = field(default_factory=list)
    vibe_tags: list[str] = field(default_factory=list)
    appearance_tags: list[str] = field(default_factory=list)
    occasions: list[str] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)
    formality: str | None = None


# ---------------------------------------------------------------------------
# Semantic + Attribute Search Engine
# ---------------------------------------------------------------------------

class AttributeSearchEngine:
    """
    Search engine combining FashionCLIP semantic similarity with
    v1.0.0.2 attribute filtering.

    Uses the Supabase RPC `search_semantic_with_attributes` which
    runs pgvector similarity + attribute WHERE clauses in one query.

    Workflow:
      1. Encode query with FashionCLIP -> 512-dim embedding
      2. Build RPC params from AttributeFilters
      3. Call RPC -> results ranked by cosine similarity, filtered by attributes
    """

    def __init__(self):
        self._sb = get_supabase_client()
        self._semantic_engine = None  # lazy-loaded WomenSearchEngine

    def _get_semantic_engine(self):
        """Lazy-load WomenSearchEngine for FashionCLIP encoding."""
        if self._semantic_engine is None:
            from women_search_engine import WomenSearchEngine
            self._semantic_engine = WomenSearchEngine()
        return self._semantic_engine

    def encode_text(self, query: str) -> np.ndarray:
        """Encode text query with FashionCLIP. Returns 512-dim L2-normalized vector."""
        return self._get_semantic_engine().encode_text(query)

    def search(
        self,
        query: str,
        filters: AttributeFilters | None = None,
        semantic_query: str | None = None,
        query_embedding: "np.ndarray | None" = None,
        exclude_product_ids: list[str] | None = None,
    ) -> tuple[list[SearchResult], dict]:
        """
        Run semantic search with optional attribute filters.

        Args:
            query: The user's raw query
            filters: Attribute filters (from plan_to_attribute_filters)
            semantic_query: Override semantic query (from planner's semantic_query).
                           If None, uses the raw query.
            query_embedding: Pre-computed FashionCLIP embedding (skips encode).
                           Used by extend-search pagination to reuse cached
                           embeddings from page 1.
            exclude_product_ids: Product IDs to exclude from results (seen items).
                               Passed through to the SQL RPC's exclude_product_ids
                               parameter for efficient server-side exclusion.

        Returns:
            (results, meta_dict)
        """
        t0 = time.time()
        filters = filters or AttributeFilters()

        # 1. Encode query with FashionCLIP (or reuse precomputed embedding)
        if query_embedding is not None:
            embedding = query_embedding
        else:
            encode_query = semantic_query or query
            embedding = self.encode_text(encode_query)

        # 2. Format embedding for pgvector
        embedding_list = embedding.astype("float32").tolist()
        vector_str = f"[{','.join(map(str, embedding_list))}]"

        # 3. Build RPC params
        rpc_params = self._build_rpc_params(
            vector_str, filters,
            exclude_product_ids=exclude_product_ids,
        )

        # 4. Call RPC (retry once on pgvector statement_timeout)
        rows = None
        for _attempt in range(2):
            try:
                resp = self._sb.rpc("search_semantic_with_attributes", rpc_params).execute()
                rows = resp.data or []
                break
            except Exception as e:
                err_str = str(e)
                is_timeout = "57014" in err_str or "statement timeout" in err_str
                if is_timeout and _attempt == 0:
                    logger.warning("RPC search_semantic_with_attributes timed out, retrying", query=query)
                    time.sleep(1.5)
                    continue
                logger.warning("RPC search_semantic_with_attributes failed: %s — falling back to search_multimodal", e)
                rows = self._fallback_search(vector_str, filters, exclude_product_ids=exclude_product_ids)
                break
        if rows is None:
            rows = self._fallback_search(vector_str, filters, exclude_product_ids=exclude_product_ids)

        # 5. Map to SearchResult objects
        results = [self._map_rpc_result(r) for r in rows]

        encode_query = semantic_query or query
        elapsed = time.time() - t0
        meta = {
            "query": query,
            "semantic_query": encode_query,
            "filters": filters.describe(),
            "has_attribute_filters": filters.has_attribute_filters(),
            "results": len(results),
            "elapsed_ms": round(elapsed * 1000),
            "top_similarity": round(results[0].similarity, 4) if results else 0,
        }
        logger.info(
            "attribute_search query=%r semantic=%r attr_filters=%s results=%d elapsed=%.0fms",
            query, encode_query, filters.has_attribute_filters(), len(results), elapsed * 1000,
        )
        return results, meta

    def search_multi_semantic(
        self,
        queries: list[str],
        filters: AttributeFilters | None = None,
        per_query_limit: int = 30,
        precomputed_embeddings: "list[np.ndarray] | None" = None,
        exclude_product_ids: list[str] | None = None,
    ) -> tuple[list[SearchResult], dict]:
        """
        Run multiple semantic queries and merge results (deduped, best score wins).

        This mirrors the multi-query approach in hybrid_search._search_semantic_multi().

        Args:
            queries: List of semantic query strings.
            filters: Attribute filters.
            per_query_limit: Max results per individual query.
            precomputed_embeddings: Pre-computed FashionCLIP embeddings (one per
                query). When provided, skips encode_text for each query.
            exclude_product_ids: Product IDs to exclude from all queries.
        """
        t0 = time.time()
        filters = filters or AttributeFilters()
        seen: dict[str, SearchResult] = {}

        for idx, q in enumerate(queries):
            emb = precomputed_embeddings[idx] if precomputed_embeddings and idx < len(precomputed_embeddings) else None
            results, _ = self.search(
                q, filters=filters,
                query_embedding=emb,
                exclude_product_ids=exclude_product_ids,
            )
            results = results[:per_query_limit]
            for r in results:
                if r.product_id not in seen or r.similarity > seen[r.product_id].similarity:
                    seen[r.product_id] = r

        merged = sorted(seen.values(), key=lambda r: -r.similarity)
        elapsed = time.time() - t0
        meta = {
            "queries": queries,
            "filters": filters.describe(),
            "results": len(merged),
            "elapsed_ms": round(elapsed * 1000),
        }
        return merged, meta

    # -----------------------------------------------------------------------
    # RPC parameter builder
    # -----------------------------------------------------------------------

    def _build_rpc_params(
        self,
        vector_str: str,
        f: AttributeFilters,
        exclude_product_ids: list[str] | None = None,
    ) -> dict:
        """Build the RPC parameter dict from AttributeFilters."""
        params: dict[str, Any] = {
            "query_embedding": vector_str,
            "match_count": f.limit,
            "match_offset": f.offset,
            "embedding_version": 1,
        }

        # Exclude seen products (for extend-search pagination)
        if exclude_product_ids:
            params["exclude_product_ids"] = exclude_product_ids

        # Basic
        if f.category_l1:
            params["filter_category_l1"] = f.category_l1
        if f.category_l2:
            params["filter_category_l2"] = f.category_l2
        if f.brands:
            params["filter_brands"] = f.brands
        if f.exclude_brands:
            params["exclude_brands"] = f.exclude_brands
        if f.min_price is not None:
            params["filter_min_price"] = f.min_price
        if f.max_price is not None:
            params["filter_max_price"] = f.max_price

        # Coverage
        if f.arm_coverage:
            params["filter_arm_coverage"] = f.arm_coverage
        if f.shoulder_coverage:
            params["filter_shoulder_coverage"] = f.shoulder_coverage
        if f.neckline_depth:
            params["filter_neckline_depth"] = f.neckline_depth
        if f.midriff_exposure:
            params["filter_midriff_exposure"] = f.midriff_exposure
        if f.back_openness:
            params["filter_back_openness"] = f.back_openness
        if f.sheerness_visual:
            params["filter_sheerness"] = f.sheerness_visual

        # Shape
        if f.body_cling_visual:
            params["filter_body_cling"] = f.body_cling_visual
        if f.structure_level:
            params["filter_structure_level"] = f.structure_level
        if f.drape_level:
            params["filter_drape_level"] = f.drape_level
        if f.bulk_visual:
            params["filter_bulk"] = f.bulk_visual
        if f.cropped_degree:
            params["filter_cropped_degree"] = f.cropped_degree
        if f.waist_definition_visual:
            params["filter_waist_definition"] = f.waist_definition_visual
        if f.leg_volume_visual:
            params["filter_leg_volume"] = f.leg_volume_visual

        # Details
        if f.has_pockets is not None:
            params["filter_has_pockets"] = f.has_pockets
        if f.pocket_types:
            params["filter_pocket_types"] = f.pocket_types
        if f.pocket_has_zip is not None:
            params["filter_pocket_has_zip"] = f.pocket_has_zip
        if f.slit_presence is not None:
            params["filter_slit_presence"] = f.slit_presence
        if f.slit_height:
            params["filter_slit_height"] = f.slit_height
        if f.detail_tags:
            params["filter_detail_tags"] = f.detail_tags
        if f.lining:
            params["filter_lining"] = f.lining

        # Occasions / style
        if f.occasions:
            params["filter_occasions"] = f.occasions
        if f.formality:
            params["filter_formality"] = f.formality
        if f.seasons:
            params["filter_seasons"] = f.seasons

        return params

    # -----------------------------------------------------------------------
    # Fallback: use existing search_multimodal when new RPC not deployed
    # -----------------------------------------------------------------------

    def _fallback_search(
        self,
        vector_str: str,
        f: AttributeFilters,
        exclude_product_ids: list[str] | None = None,
    ) -> list[dict]:
        """
        Fallback when search_semantic_with_attributes RPC is not yet deployed.
        Uses existing search_multimodal + PostgREST attribute post-filter.
        """
        # 1. Semantic search via existing RPC (no attribute filters)
        rpc_params: dict[str, Any] = {
            "query_embedding": vector_str,
            "match_count": f.limit * 5 if f.has_attribute_filters() else f.limit,
            "match_offset": 0,
            "embedding_version": 1,
        }
        if f.category_l1:
            rpc_params["filter_categories"] = [c.lower() for c in f.category_l1]
        if f.brands:
            rpc_params["include_brands"] = f.brands
        if f.exclude_brands:
            rpc_params["exclude_brands"] = f.exclude_brands
        if f.min_price is not None:
            rpc_params["min_price"] = f.min_price
        if f.max_price is not None:
            rpc_params["max_price"] = f.max_price
        if exclude_product_ids:
            rpc_params["exclude_product_ids"] = exclude_product_ids

        # Retry once on statement_timeout
        rows = []
        for _attempt in range(2):
            try:
                resp = self._sb.rpc("search_multimodal", rpc_params).execute()
                rows = resp.data or []
                break
            except Exception as e:
                err_str = str(e)
                is_timeout = "57014" in err_str or "statement timeout" in err_str
                if is_timeout and _attempt == 0:
                    logger.warning("Fallback search_multimodal timed out, retrying")
                    time.sleep(1.5)
                    continue
                logger.warning("Fallback search_multimodal failed: %s", e)
                break

        if not f.has_attribute_filters():
            return rows[:f.limit]

        # 2. Post-filter by fetching attributes for matched product IDs
        product_ids = [r["product_id"] for r in rows]
        if not product_ids:
            return []

        # Batch fetch attributes
        attr_resp = (
            self._sb.table("product_attributes")
            .select("sku_id, category_l1, category_l2, body_cling_visual, "
                    "structure_level, drape_level, arm_coverage, shoulder_coverage, "
                    "neckline_depth, midriff_exposure, back_openness, sheerness_visual, "
                    "cropped_degree, waist_definition_visual, leg_volume_visual, "
                    "bulk_visual, has_pockets_visible, pocket_details, slit_presence, "
                    "slit_height, detail_tags, lining_status_likely, vibe_tags, "
                    "appearance_top_tags, occasions, formality, style_tags")
            .eq("extractor_version", "v1.0.0.2")
            .in_("sku_id", product_ids)
            .execute()
        )
        attr_by_id = {r["sku_id"]: r for r in (attr_resp.data or [])}

        # 3. Filter and enrich
        filtered = []
        for row in rows:
            pid = row["product_id"]
            attrs = attr_by_id.get(pid)
            if not attrs:
                continue  # No v1.0.0.2 attributes = excluded when filters active

            if not self._passes_attribute_filter(attrs, f):
                continue

            # Enrich row with attribute fields
            row["pa_category_l1"] = attrs.get("category_l1")
            row["pa_category_l2"] = attrs.get("category_l2")
            row["pa_body_cling"] = attrs.get("body_cling_visual")
            row["pa_structure_level"] = attrs.get("structure_level")
            row["pa_drape_level"] = attrs.get("drape_level")
            row["pa_arm_coverage"] = attrs.get("arm_coverage")
            row["pa_shoulder_coverage"] = attrs.get("shoulder_coverage")
            row["pa_neckline_depth"] = attrs.get("neckline_depth")
            row["pa_midriff_exposure"] = attrs.get("midriff_exposure")
            row["pa_back_openness"] = attrs.get("back_openness")
            row["pa_sheerness"] = attrs.get("sheerness_visual")
            row["pa_cropped_degree"] = attrs.get("cropped_degree")
            row["pa_waist_definition"] = attrs.get("waist_definition_visual")
            row["pa_has_pockets"] = attrs.get("has_pockets_visible")
            row["pa_pocket_details"] = attrs.get("pocket_details") or {}
            row["pa_slit_presence"] = attrs.get("slit_presence")
            row["pa_slit_height"] = attrs.get("slit_height")
            row["pa_detail_tags"] = attrs.get("detail_tags") or []
            row["pa_vibe_tags"] = attrs.get("vibe_tags") or []
            row["pa_appearance_tags"] = attrs.get("appearance_top_tags") or []
            row["pa_occasions"] = attrs.get("occasions") or []
            row["pa_formality"] = attrs.get("formality")
            row["pa_style_tags"] = attrs.get("style_tags") or []
            filtered.append(row)

            if len(filtered) >= f.limit:
                break

        return filtered

    def _passes_attribute_filter(self, attrs: dict, f: AttributeFilters) -> bool:
        """Check if a product's attributes pass all active filters."""
        def _in(val, allowed):
            return val is not None and val in allowed

        def _overlap(arr, allowed):
            if not arr:
                return False
            return bool(set(arr) & set(allowed))

        def _contains(arr, required):
            if not arr:
                return False
            return all(r in arr for r in required)

        if f.arm_coverage and not _in(attrs.get("arm_coverage"), f.arm_coverage):
            return False
        if f.shoulder_coverage and not _in(attrs.get("shoulder_coverage"), f.shoulder_coverage):
            return False
        if f.neckline_depth and not _in(attrs.get("neckline_depth"), f.neckline_depth):
            return False
        if f.midriff_exposure and not _in(attrs.get("midriff_exposure"), f.midriff_exposure):
            return False
        if f.back_openness and not _in(attrs.get("back_openness"), f.back_openness):
            return False
        if f.sheerness_visual and not _in(attrs.get("sheerness_visual"), f.sheerness_visual):
            return False
        if f.body_cling_visual and not _in(attrs.get("body_cling_visual"), f.body_cling_visual):
            return False
        if f.structure_level and not _in(attrs.get("structure_level"), f.structure_level):
            return False
        if f.drape_level and not _in(attrs.get("drape_level"), f.drape_level):
            return False
        if f.bulk_visual and not _in(attrs.get("bulk_visual"), f.bulk_visual):
            return False
        if f.cropped_degree and not _in(attrs.get("cropped_degree"), f.cropped_degree):
            return False
        if f.waist_definition_visual and not _in(attrs.get("waist_definition_visual"), f.waist_definition_visual):
            return False
        if f.leg_volume_visual and not _in(attrs.get("leg_volume_visual"), f.leg_volume_visual):
            return False
        if f.has_pockets is not None and attrs.get("has_pockets_visible") != f.has_pockets:
            return False
        if f.pocket_types:
            pd = attrs.get("pocket_details") or {}
            db_types = [t.lower() for t in (pd.get("types") or [])]
            if not any(ft in db_types for ft in f.pocket_types):
                return False
        if f.pocket_has_zip is True:
            pd = attrs.get("pocket_details") or {}
            zip_count = pd.get("zip_count", 0) or 0
            db_types = [t.lower() for t in (pd.get("types") or [])]
            has_zip = zip_count > 0 or any("zip" in t for t in db_types)
            if not has_zip:
                return False
        if f.slit_presence is not None and attrs.get("slit_presence") != f.slit_presence:
            return False
        if f.slit_height and not _in(attrs.get("slit_height"), f.slit_height):
            return False
        if f.detail_tags and not _contains(attrs.get("detail_tags") or [], f.detail_tags):
            return False
        if f.lining and not _in(attrs.get("lining_status_likely"), f.lining):
            return False
        if f.occasions and not _overlap(attrs.get("occasions") or [], f.occasions):
            return False
        if f.formality and not _in(attrs.get("formality"), f.formality):
            return False
        if f.seasons and not _overlap(attrs.get("seasons") or [], f.seasons):
            return False

        return True

    # -----------------------------------------------------------------------
    # Result mapping
    # -----------------------------------------------------------------------

    def _map_rpc_result(self, row: dict) -> SearchResult:
        """Map an RPC result row to SearchResult."""
        return SearchResult(
            product_id=row.get("product_id", ""),
            name=row.get("name", ""),
            brand=row.get("brand", ""),
            price=row.get("price", 0),
            original_price=row.get("original_price"),
            image_url=row.get("primary_image_url", ""),
            gallery_images=row.get("gallery_images"),
            in_stock=row.get("in_stock", True),
            similarity=row.get("similarity", 0),
            category_l1=row.get("pa_category_l1"),
            category_l2=row.get("pa_category_l2"),
            body_cling=row.get("pa_body_cling"),
            structure=row.get("pa_structure_level"),
            drape=row.get("pa_drape_level"),
            arm_coverage=row.get("pa_arm_coverage"),
            shoulder_coverage=row.get("pa_shoulder_coverage"),
            back_openness=row.get("pa_back_openness"),
            midriff_exposure=row.get("pa_midriff_exposure"),
            neckline_depth=row.get("pa_neckline_depth"),
            sheerness=row.get("pa_sheerness"),
            cropped=row.get("pa_cropped_degree"),
            waist_definition=row.get("pa_waist_definition"),
            has_pockets=row.get("pa_has_pockets", False) or False,
            slit_presence=row.get("pa_slit_presence", False) or False,
            slit_height=row.get("pa_slit_height"),
            detail_tags=row.get("pa_detail_tags") or [],
            vibe_tags=row.get("pa_vibe_tags") or [],
            appearance_tags=row.get("pa_appearance_tags") or [],
            occasions=row.get("pa_occasions") or [],
            style_tags=row.get("pa_style_tags") or [],
            formality=row.get("pa_formality"),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine: AttributeSearchEngine | None = None


def get_attribute_search_engine() -> AttributeSearchEngine:
    global _engine
    if _engine is None:
        _engine = AttributeSearchEngine()
    return _engine
