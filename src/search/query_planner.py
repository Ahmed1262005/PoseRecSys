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

    # Exclusion filters - attribute values to EXCLUDE from results.
    # Used for functional/coverage queries where certain attributes are
    # incompatible with what the user wants.
    # E.g. "top that doesn't show bra straps" -> exclude_filters={"neckline": ["Strapless", "Off-Shoulder", "Halter", "One Shoulder"]}
    # Same keys as filters, but these values are REMOVED from results.
    exclude_filters: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Attribute values to exclude from results (hard drop)"
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

    # Price constraints extracted from query
    max_price: Optional[float] = Field(
        default=None,
        description="Maximum price if user specified (e.g. 'under $50' -> 50.0)"
    )
    min_price: Optional[float] = Field(
        default=None,
        description="Minimum price if user specified"
    )
    on_sale_only: bool = Field(
        default=False,
        description="True if user wants sale/discounted items only"
    )

    # Confidence 0.0-1.0 in the plan
    confidence: float = Field(
        default=0.8,
        description="Planner confidence in its interpretation"
    )


# =============================================================================
# System Prompt
# =============================================================================

_SYSTEM_PROMPT = """You are a fashion search query planner for a women's fashion e-commerce store. Your job is to decompose a user's natural language search query into a structured search plan.

FASHION REASONING PRINCIPLES:

1. **Coverage means OPAQUE coverage.** Sheer, mesh, or lace fabric does not provide real coverage.
   If the user wants to "hide" or "cover" a body part, they need opaque fabric over it.
   A lace-sleeve top does NOT hide arms. A mesh panel does NOT cover the back.

2. **Vibe language maps to formality and occasion.** Users describe what they want with mood words.
   Translate these into the structured filters we have:
   - "effortless", "not too try-hard", "low-key", "chill" → formality: ["Casual"]
   - "put-together", "polished", "classy" → formality: ["Smart Casual", "Business Casual"]
   - "sexy but classy", "going out" → formality: ["Smart Casual"], occasions: ["Night Out", "Date Night"]
   - "first date", "date night", "date" → occasions: ["Date Night"]
   - "office", "work", "professional" → occasions: ["Office", "Work"]
   - "wedding", "formal event" → occasions: ["Wedding Guest"]
   - "brunch", "weekend plans" → occasions: ["Brunch", "Weekend"]

3. **Aspirational language is about aesthetic, not price.** When users say "looks expensive",
   "looks designer", "elevated", "luxe", "high-end looking" — they want items that LOOK
   premium, not items that ARE expensive. Do NOT set min_price. Instead use:
   - style_tags: ["Classic", "Minimalist", "Modern"] in filters
   - formality: ["Smart Casual", "Business Casual"]
   - Materials that read expensive in expanded_filters: ["Silk", "Satin", "Wool", "Linen"]

4. **"Outfit" queries are about the hero piece, not bottoms.** When someone asks for "an outfit
   for X" or "something for X", they're looking for tops, dresses, or outerwear — the pieces
   that define a look. Bottoms (pants, jeans, skirts) are supporting pieces. Only include
   "Bottoms" in category_l1 when the user explicitly mentions pants, jeans, shorts, skirts,
   trousers, or bottoms.

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
     IMPORTANT: For vague/outfit queries, default to ["Tops", "Dresses"] unless the user
     explicitly mentions bottoms/pants/jeans/skirts. Add "Outerwear" only when relevant
     (e.g. "jacket", "coat", "layering", cold weather).
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
   - Use formality and occasions filters actively — most vibe queries map to one of these
   - For category_l2, include singular AND plural: ["Jacket", "Jackets"]
   - When the user mentions a specific type (e.g. "bomber jacket"), include BOTH the specific AND generic: ["Bomber Jacket", "Jacket", "Jackets"]

5. **expanded_filters**: Broader/related values for lenient post-filtering of semantic results.
   Same keys as filters, but with MORE values to avoid dropping good results.
   
   Examples:
   - "floral leaves" -> filters.patterns=["Floral"] but expanded_filters.patterns=["Floral", "Tropical", "Abstract"]
   - "jacket" -> filters.category_l2=["Jacket","Jackets"] but expanded_filters.category_l2=["Jacket","Jackets","Bomber Jacket","Denim Jacket","Leather Jacket","Puffer Jacket","Fleece Jacket"]
   
   The expanded_filters are used ONLY for post-filtering semantic (FashionCLIP) results. They let visually-matching products through even if their exact label differs from the strict filter.

6. **exclude_filters**: Attribute values to EXCLUDE from results. Same valid keys as filters.
   These are HARD exclusions — any product matching an excluded value is removed.

   ONLY use exclude_filters when the user expresses a NEGATIVE constraint — wanting to
   avoid, hide, or cover something. Queries like "sexy", "classy", "elegant", "cute",
   "looks expensive" are POSITIVE preferences — handle them with filters and semantic_query,
   NOT with exclude_filters.

   EXCLUSION SETS — copy the EXACT values from the matching set(s).
   For compound queries, MERGE sets (union the values for each key).

   SET A — ARM/SHOULDER COVERAGE
   Triggers: "hides arms", "cover arms", "don't show arms", "long sleeves", "longer sleeves", "covers shoulders"
   When the user wants to hide/cover arms, they need OPAQUE coverage — so also exclude
   sheer materials that would show skin through the sleeves.
     sleeve_type: ["Short", "Cap", "Spaghetti", "Sleeveless"]
     neckline: ["Off-Shoulder", "One Shoulder", "Strapless"]
     materials: ["Mesh", "Lace", "Chiffon"]

   SET B — BASIC SLEEVES
   Triggers: "with sleeves", "not sleeveless", "has sleeves"
     sleeve_type: ["Spaghetti", "Sleeveless"]

   SET C — NECKLINE MODESTY
   Triggers: "no cleavage", "covers chest", "high neckline", "modest neckline"
     neckline: ["V-Neck", "Sweetheart", "Halter", "Off-Shoulder", "One Shoulder", "Strapless"]

   SET D — BRA STRAP COVERAGE
   Triggers: "covers bra straps", "doesn't show straps", "hides straps"
     neckline: ["Strapless", "Off-Shoulder", "Halter", "One Shoulder"]
     sleeve_type: ["Sleeveless", "Spaghetti"]
     style_tags: ["Backless", "Open Back", "Low Back"]

   SET E — MODEST / NOT REVEALING (combines A + C + length + back)
   Triggers: "modest", "conservative", "not revealing", "not too revealing", "covers up"
     neckline: ["V-Neck", "Sweetheart", "Halter", "Off-Shoulder", "One Shoulder", "Strapless"]
     sleeve_type: ["Short", "Cap", "Spaghetti", "Sleeveless"]
     length: ["Mini", "Cropped"]
     style_tags: ["Backless", "Open Back", "Low Back", "Cut-Out"]
     materials: ["Mesh", "Lace", "Chiffon"]

   SET F — BACK COVERAGE
   Triggers: "covers back", "no backless", "back covered"
     style_tags: ["Backless", "Open Back", "Low Back"]

   SET G — LENGTH
   Triggers: "not too short", "not mini", "longer"
     length: ["Mini", "Cropped"]

   SET H — NOT TIGHT
   Triggers: "not tight", "not clingy", "not body-hugging", "not form-fitting"
     fit_type: ["Fitted", "Slim"]
     silhouette: ["Bodycon"]

   SET I — NOT OVERSIZED
   Triggers: "not bulky", "not oversized", "structured", "flattering"
     fit_type: ["Oversized"]
   WARNING: "not bulky" (SET I) and "not tight" (SET H) are OPPOSITES. Never apply both.

   SET J — NOT SHEER
   Triggers: "not see-through", "not sheer", "opaque", "not transparent"
     materials: ["Mesh", "Lace", "Chiffon"]

   RULES:
   1. exclude_filters is a TOP-LEVEL field, NOT nested inside filters.
   2. exclude_filters and filters for the SAME key must NEVER overlap.
   3. Copy values EXACTLY from the sets above. Do not drop, abbreviate, or paraphrase values.
   4. When combining sets, MERGE values per key (union). Example: SET A + SET C →
      neckline gets the union of both: ["V-Neck","Sweetheart","Halter","Off-Shoulder","One Shoulder","Strapless"]
      sleeve_type: ["Short","Cap","Spaghetti","Sleeveless"]
   5. When the user says "with X" (e.g. "with sleeves"), exclude the opposite using the matching set.
   6. category_l1 ALWAYS goes in **filters**, NEVER in exclude_filters.
   7. Be generous with exclusions ONLY when the user asks to avoid something.

7. **max_price**: Float or null. Extract from "under $50" -> 50.0, "below $100" -> 100.0, etc.
   IMPORTANT: "Looks expensive" or "looks designer" is about aesthetic, NOT price. Do NOT set
   min_price for aspirational queries.

8. **min_price**: Float or null. Extract from "over $200" -> 200.0, etc.

9. **on_sale_only**: Boolean. True if user says "on sale", "discounted", "clearance", "sale items", "deals".
   Examples: "On sale coats" -> on_sale_only=true. "Discounted matching sets" -> on_sale_only=true.
   "Affordable" or "cheap" do NOT mean on_sale_only (those are just price preference).

10. **brand**: Detected brand name or null.

11. **matched_terms**: List of query words/phrases you consumed into filters.

12. **confidence**: 0.0-1.0 how confident you are in this plan.

VOCABULARY TRANSLATION (use these in algolia_query):
- "skirt with shorts underneath" / "skirt with built-in shorts" -> algolia_query="skort"
- "butter yellow" / "mustard" -> colors=["Yellow"]
- "chocolate brown" / "espresso" -> colors=["Brown"]
- "cherry red" / "crimson" -> colors=["Red", "Burgundy"]
- "navy" -> colors=["Navy Blue"]
- "nude" / "skin tone" -> colors=["Beige", "Taupe"]
- "no shoulder pads" / "unstructured" -> algolia_query should include "unstructured" or "soft shoulder"
- "elastic waistband" / "pull on" -> algolia_query should include "pull-on" or "elastic waist"
- "squat proof" -> algolia_query="squat proof leggings", materials=["Jersey"]

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

            # Fix common LLM mistake: nesting exclude_filters inside filters
            filters = data.get("filters", {})
            if isinstance(filters, dict) and "exclude_filters" in filters:
                nested_excl = filters.pop("exclude_filters")
                # Merge into top-level exclude_filters (don't overwrite)
                if isinstance(nested_excl, dict):
                    existing = data.get("exclude_filters", {})
                    if not isinstance(existing, dict):
                        existing = {}
                    for k, v in nested_excl.items():
                        if k not in existing and isinstance(v, list):
                            existing[k] = v
                    data["exclude_filters"] = existing
                logger.debug(
                    "Extracted misplaced exclude_filters from filters dict",
                    extracted=data.get("exclude_filters"),
                )

            plan = SearchPlan(**data)

            latency_ms = int((time.time() - t_start) * 1000)
            logger.info(
                "Query planner generated search plan",
                query=query,
                intent=plan.intent,
                algolia_query=plan.algolia_query,
                semantic_query=plan.semantic_query,
                filters=plan.filters,
                exclude_filters=plan.exclude_filters,
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
    ) -> Tuple[Dict[str, Any], Dict[str, List[str]], Dict[str, List[str]], List[str], str, str, str]:
        """
        Convert a SearchPlan into values the hybrid search pipeline can use.

        Returns:
            Tuple of:
            - request_updates: Dict of HybridSearchRequest field updates (filters)
            - expanded_filters: Dict of expanded filter values for lenient post-filtering
            - exclude_filters: Dict of attribute values to EXCLUDE (hard drops)
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

        # Price / sale injection
        if plan.max_price is not None:
            request_updates["max_price"] = plan.max_price
        if plan.min_price is not None:
            request_updates["min_price"] = plan.min_price
        if plan.on_sale_only:
            request_updates["on_sale_only"] = True

        # Exclusion filters - map to exclude_* request fields
        # First, correct common LLM typos in field names
        _TYPO_CORRECTIONS = {
            "necktine": "neckline",
            "necktline": "neckline",
            "neckelines": "neckline",
            "sleevetype": "sleeve_type",
            "sleeve_types": "sleeve_type",
            "sleev_type": "sleeve_type",
            "fit_types": "fit_type",
            "fittype": "fit_type",
            "silouette": "silhouette",
            "sillouette": "silhouette",
            "pattern": "patterns",
            "color": "colors",
            "material": "materials",
            "occasion": "occasions",
            "season": "seasons",
        }
        corrected_excludes = {}
        for field, values in plan.exclude_filters.items():
            corrected = _TYPO_CORRECTIONS.get(field, field)
            if corrected != field:
                logger.info(
                    "Corrected LLM typo in exclude_filters",
                    original=field, corrected=corrected,
                )
            if corrected in corrected_excludes:
                # Merge with existing values (deduplicate)
                existing = set(v.lower() for v in corrected_excludes[corrected])
                for v in values:
                    if v.lower() not in existing:
                        corrected_excludes[corrected].append(v)
                        existing.add(v.lower())
            else:
                corrected_excludes[corrected] = values

        exclude_updates: Dict[str, List[str]] = {}
        for field, values in corrected_excludes.items():
            if field in _VALID_FILTER_FIELDS and values:
                exclude_updates[field] = values

        if exclude_updates:
            # Map to exclude_* fields on HybridSearchRequest
            _EXCLUDE_FIELD_MAP = {
                "neckline": "exclude_neckline",
                "sleeve_type": "exclude_sleeve_type",
                "length": "exclude_length",
                "fit_type": "exclude_fit_type",
                "silhouette": "exclude_silhouette",
                "patterns": "exclude_patterns",
                "colors": "exclude_colors",
                "materials": "exclude_materials",
                "occasions": "exclude_occasions",
                "seasons": "exclude_seasons",
                "formality": "exclude_formality",
                "rise": "exclude_rise",
                "style_tags": "exclude_style_tags",
            }
            for field, values in exclude_updates.items():
                req_field = _EXCLUDE_FIELD_MAP.get(field)
                if req_field:
                    request_updates[req_field] = values

        return (
            request_updates,
            expanded,
            exclude_updates,
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
