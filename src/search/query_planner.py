"""
LLM-based Query Planner for agentic search.

Sits before the search pipeline and decomposes natural language queries into:
1. Structured filters (hard constraints for Algolia facets)
2. An optimized Algolia keyword query (what to text-search)
3. A semantic description for FashionCLIP (visual understanding)
4. Related pattern/category expansions (fuzzy matching)
5. Intent classification with confidence

This replaces the brittle regex-based attribute extraction with an LLM that
understands fashion vocabulary, composite concepts (e.g. "floral leaves" is
a botanical/leaf print pattern), and user intent.

Falls back to the regex-based QueryClassifier when:
- OpenAI API key is not configured
- LLM call times out (default 2s)
- LLM returns invalid/unparseable response
- Feature flag is disabled
"""

import json
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from core.logging import get_logger
from config.settings import get_settings
from search.models import QueryIntent

logger = get_logger(__name__)


# =============================================================================
# Planner Output Schema
# =============================================================================

class SearchPlan(BaseModel):
    """Structured output from the LLM query planner."""

    # Intent
    intent: str = Field(
        description="Query intent: 'exact' (brand search), 'specific' (category+attributes), 'vague' (mood/aesthetic)"
    )

    # Algolia keyword query - what to text-search in product names/descriptions.
    # Should be product-relevant terms only (e.g. "leaf print jacket", not "a with leaves").
    # Empty string means rely entirely on facet filters.
    algolia_query: str = Field(
        default="",
        description="Optimized keyword query for Algolia text search"
    )

    # Semantic description for FashionCLIP - a rich visual description.
    # E.g. "a jacket with a botanical leaf and floral print pattern"
    semantic_query: str = Field(
        default="",
        description="Rich visual description for FashionCLIP semantic search"
    )

    # Hard filters - exact Algolia facet values to filter on.
    # These are AND-ed together. Only include when the user clearly wants to constrain.
    filters: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Hard facet filters to apply"
    )

    # Expanded filter values - related values to include in post-filtering.
    # E.g. for "floral leaves": patterns=["Floral"] but expanded_patterns=["Floral", "Tropical", "Abstract"]
    # This prevents the strict post-filter from dropping good semantic results.
    expanded_filters: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Broader filter values for lenient post-filtering of semantic results"
    )

    # Brand detected (if any)
    brand: Optional[str] = Field(
        default=None,
        description="Detected brand name, if any"
    )

    # Terms that the LLM consumed into filters (for logging/debugging)
    matched_terms: List[str] = Field(
        default_factory=list,
        description="Query terms consumed into filters"
    )

    # Confidence 0.0-1.0 in the plan
    confidence: float = Field(
        default=0.8,
        description="Planner confidence in its interpretation"
    )


# =============================================================================
# System Prompt
# =============================================================================

_SYSTEM_PROMPT = """You are a fashion search query planner. Your job is to decompose a user's natural language search query into a structured search plan for a fashion e-commerce search engine.

You must return a JSON object with these fields:

1. **intent**: One of "exact", "specific", "vague"
   - "exact": Pure brand search (e.g. "Boohoo", "Zara")
   - "specific": Has a concrete product type and/or attributes (e.g. "floral midi dress", "black leather jacket")
   - "vague": Mood/aesthetic/vibe with no concrete product type (e.g. "quiet luxury", "date night outfit")

2. **algolia_query**: Optimized text query for keyword search in product names.
   - Include ONLY terms that would appear in product names/titles in a fashion store.
   - Product names look like: "Floral Print Bomber Jacket", "Leaf Print Midi Dress", "Striped Knit Cardigan"
   - Convert descriptive language to product-name language: "floral leaves" -> "leaf print" or "floral print"
   - Remove filler words ("a", "with", "for", "something"), intent words ("outfit", "look", "wear")
   - If all terms became filters, return empty string "" (filter-only search is fine)
   - IMPORTANT: Do NOT include terms that are better served as facet filters (colors, patterns, materials) unless they commonly appear in product names

3. **semantic_query**: A rich visual description for image-similarity search.
   - Keep the FULL meaning of the original query
   - Expand with visual details: "floral leaves jacket" -> "a jacket with botanical leaf and flower print pattern"
   - This powers FashionCLIP which understands visual concepts from text

4. **filters**: Hard facet filters. Only the following keys are valid, with their allowed values:

   - **category_l1**: ["Tops", "Bottoms", "Dresses", "Outerwear", "Activewear", "Swimwear", "Intimates", "Accessories"]
   - **category_l2**: Use the BROADEST matching value(s). For "jacket" use ["Jacket", "Jackets"]. For subtypes, include BOTH the subtype and generic: ["Bomber Jacket", "Jacket", "Jackets"]
   - **patterns**: ["Solid", "Floral", "Striped", "Plaid", "Polka Dot", "Animal Print", "Abstract", "Geometric", "Tie Dye", "Camo", "Colorblock", "Tropical"]
   - **colors**: Exact color values: ["Black", "White", "Red", "Blue", "Navy Blue", "Green", "Pink", "Yellow", "Purple", "Orange", "Brown", "Beige", "Cream", "Gray", "Burgundy", "Olive", "Taupe", "Off White", "Light Blue"]
   - **formality**: ["Formal", "Semi-Formal", "Business Casual", "Smart Casual", "Casual"]
   - **occasions**: ["Date Night", "Party", "Office", "Work", "Wedding Guest", "Vacation", "Workout", "Everyday", "Brunch", "Night Out", "Weekend", "Lounging", "Beach"]
   - **fit_type**: ["Slim", "Fitted", "Regular", "Relaxed", "Oversized", "Loose"]
   - **neckline**: ["V-Neck", "Crew", "Turtleneck", "Off-Shoulder", "Strapless", "Halter", "Scoop", "Square", "Sweetheart", "Cowl", "Boat", "One Shoulder", "Collared", "Hooded", "Mock"]
   - **sleeve_type**: ["Sleeveless", "Short", "Long", "Cap", "Puff", "3/4", "Flutter"]
   - **length**: ["Mini", "Midi", "Maxi", "Cropped", "Floor-length", "Ankle"]
   - **rise**: ["High", "Mid", "Low"]
   - **materials**: ["Cotton", "Linen", "Silk", "Satin", "Denim", "Faux Leather", "Wool", "Velvet", "Chiffon", "Lace", "Mesh", "Knit", "Jersey", "Fleece"]
   - **silhouette**: ["A-Line", "Bodycon", "Flared", "Straight", "Wide Leg"]
   - **seasons**: ["Summer", "Spring", "Fall", "Winter"]
   - **style_tags**: ["Bohemian", "Romantic", "Glamorous", "Edgy", "Vintage", "Sporty", "Classic", "Modern", "Minimalist", "Preppy", "Streetwear", "Sexy", "Western", "Utility"]
   - **brands**: Only if a specific brand is mentioned
   
   RULES:
   - Only include filters the user explicitly or strongly implies
   - For category_l2, include singular AND plural: ["Jacket", "Jackets"]
   - When the user mentions a specific type (e.g. "bomber jacket"), include BOTH the specific AND generic: ["Bomber Jacket", "Jacket", "Jackets"]

5. **expanded_filters**: Broader/related values for lenient post-filtering of semantic results.
   Same keys as filters, but with MORE values to avoid dropping good results.
   
   Examples:
   - "floral leaves" -> filters.patterns=["Floral"] but expanded_filters.patterns=["Floral", "Tropical", "Abstract"]
   - "jacket" -> filters.category_l2=["Jacket","Jackets"] but expanded_filters.category_l2=["Jacket","Jackets","Bomber Jacket","Denim Jacket","Leather Jacket","Puffer Jacket","Fleece Jacket"]
   
   The expanded_filters are used ONLY for post-filtering semantic (FashionCLIP) results. They let visually-matching products through even if their exact label differs from the strict filter.

6. **brand**: Detected brand name or null.

7. **matched_terms**: List of query words/phrases you consumed into filters.

8. **confidence**: 0.0-1.0 how confident you are in this plan.

IMPORTANT: Return ONLY valid JSON. No markdown, no explanation, no code blocks."""


# =============================================================================
# Query Planner
# =============================================================================

class QueryPlanner:
    """LLM-based query planner using OpenAI gpt-4o-mini."""

    def __init__(self):
        self._client = None
        self._client_lock = threading.Lock()
        settings = get_settings()
        self._api_key = settings.openai_api_key
        self._model = settings.query_planner_model
        self._timeout = settings.query_planner_timeout_seconds
        self._enabled = settings.query_planner_enabled and bool(self._api_key)

    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    from openai import OpenAI
                    self._client = OpenAI(
                        api_key=self._api_key,
                        timeout=self._timeout,
                    )
        return self._client

    @property
    def enabled(self) -> bool:
        return self._enabled

    def plan(self, query: str) -> Optional[SearchPlan]:
        """
        Generate a search plan for the given query.

        Returns None if the planner is disabled, times out, or fails.
        The caller should fall back to regex-based extraction.
        """
        if not self._enabled:
            logger.debug("Query planner disabled (no API key or feature flag off)")
            return None

        t_start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=800,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            if not raw:
                logger.warning("Query planner returned empty response")
                return None

            data = json.loads(raw)
            plan = SearchPlan(**data)

            latency_ms = int((time.time() - t_start) * 1000)
            logger.info(
                "Query planner generated search plan",
                query=query,
                intent=plan.intent,
                algolia_query=plan.algolia_query,
                semantic_query=plan.semantic_query,
                filters=plan.filters,
                expanded_filters=plan.expanded_filters,
                confidence=plan.confidence,
                latency_ms=latency_ms,
            )
            return plan

        except json.JSONDecodeError as e:
            logger.warning("Query planner returned invalid JSON", error=str(e))
            return None
        except Exception as e:
            latency_ms = int((time.time() - t_start) * 1000)
            logger.warning(
                "Query planner failed, falling back to regex",
                error=str(e),
                latency_ms=latency_ms,
            )
            return None

    def plan_to_request_updates(
        self, plan: SearchPlan
    ) -> Tuple[Dict[str, Any], Dict[str, List[str]], List[str], str, str, str]:
        """
        Convert a SearchPlan into values the hybrid search pipeline can use.

        Returns:
            Tuple of:
            - request_updates: Dict of HybridSearchRequest field updates (filters)
            - expanded_filters: Dict of expanded filter values for lenient post-filtering
            - matched_terms: List of consumed query terms
            - algolia_query: Optimized Algolia keyword query
            - semantic_query: Rich semantic query for FashionCLIP
            - intent_str: Intent string ("exact", "specific", "vague")
        """
        request_updates: Dict[str, Any] = {}
        expanded = dict(plan.expanded_filters)

        # Map plan filters to HybridSearchRequest fields
        _VALID_FILTER_FIELDS = {
            "category_l1", "category_l2", "patterns", "colors", "formality",
            "occasions", "fit_type", "neckline", "sleeve_type", "length",
            "rise", "materials", "silhouette", "seasons", "style_tags",
            "brands", "categories",
        }

        for field, values in plan.filters.items():
            if field in _VALID_FILTER_FIELDS and values:
                request_updates[field] = values

        # Brand injection
        if plan.brand and "brands" not in request_updates:
            request_updates["brands"] = [plan.brand]

        return (
            request_updates,
            expanded,
            plan.matched_terms,
            plan.algolia_query,
            plan.semantic_query or plan.algolia_query,
            plan.intent,
        )


# =============================================================================
# Singleton
# =============================================================================

_planner: Optional[QueryPlanner] = None
_planner_lock = threading.Lock()


def get_query_planner() -> QueryPlanner:
    """Get or create the QueryPlanner singleton (thread-safe)."""
    global _planner
    if _planner is None:
        with _planner_lock:
            if _planner is None:
                _planner = QueryPlanner()
    return _planner
