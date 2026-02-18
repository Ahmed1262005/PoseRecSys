"""
LLM-based Query Planner for agentic search (mode-based architecture).

The planner decomposes natural language queries into three sections:
    1. modes[]      — High-level intent tags from a fixed menu (~40 modes).
                      Deterministic code expands these into filters/exclusions.
    2. attributes{} — Concrete positive filter values the user explicitly stated.
    3. avoid{}      — Concrete negative filter values the user said NO to.

The LLM NEVER constructs exclusion value lists — that is always done by
expand_modes() in mode_config.py.  The LLM only picks mode labels and
extracts concrete attribute values.

Falls back to basic search (raw query, no filters) when:
- OpenAI API key is not configured
- LLM call times out or errors
- LLM returns invalid/unparseable response
- Feature flag is disabled

There is NO regex fallback.  On planner failure, basic search runs.
"""

import json
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from core.logging import get_logger
from config.settings import get_settings
from search.models import QueryIntent
from search.mode_config import expand_modes, get_mode_menu_text

logger = get_logger(__name__)


# =============================================================================
# Planner Output Schema
# =============================================================================

class SearchPlan(BaseModel):
    """Structured output from the LLM query planner (mode-based)."""

    # Intent classification
    intent: str = Field(
        description="Query intent: 'exact' (brand search), 'specific' (category+attributes), 'vague' (mood/aesthetic)"
    )

    # Algolia keyword query — product-name terms only
    algolia_query: str = Field(
        default="",
        description="Optimized keyword query for Algolia text search"
    )

    # Semantic description for FashionCLIP — rich visual description
    semantic_query: str = Field(
        default="",
        description="Rich visual description for FashionCLIP semantic search"
    )

    # Mode tags — high-level intent labels from the mode menu
    modes: List[str] = Field(
        default_factory=list,
        description="Mode tags selected from the mode menu (e.g. ['cover_arms', 'work'])"
    )

    # Positive attribute filters — concrete values the user explicitly stated
    attributes: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Concrete positive filter values from the user's query"
    )

    # Avoid — concrete negative values the user explicitly said NO to
    # (only for things NOT already covered by a mode)
    avoid: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Concrete negative values the user explicitly wants to avoid"
    )

    # Brand detected (if any)
    brand: Optional[str] = Field(
        default=None,
        description="Detected brand name, if any"
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

    model_config = ConfigDict(extra="ignore")


# =============================================================================
# System Prompt (built dynamically with mode menu)
# =============================================================================

def _build_system_prompt() -> str:
    """Build the system prompt with the mode menu included."""

    mode_menu = get_mode_menu_text()

    return f"""You are a fashion search query planner for a women's fashion e-commerce store. Decompose the user's query into a structured search plan.

## SECTION 1: FASHION REASONING PRINCIPLES

1. **Coverage depends on the body part.** There are two kinds of coverage:
   - **Skin coverage** (arms, chest, back, legs, stomach): Needs opaque fabric. Sheer/mesh/lace don't count.
     A lace-sleeve top does NOT hide arms. A mesh panel does NOT cover the back.
     For these, use cover_arms/cover_chest/cover_back/cover_legs/cover_stomach + opaque modes.
   - **Strap/structural coverage** (bra straps, shoulders): About garment STRUCTURE, not fabric.
     A chiffon blouse with high neckline and cap+ sleeves DOES hide bra straps.
     For bra straps, use cover_straps mode ONLY — do NOT add opaque unless user explicitly
     mentions see-through/sheer concerns.
   USE MODES for coverage requests — do NOT manually list exclusion values.

2. **Vibe language maps to modes + formality.** Users describe what they want with mood words.
   Pick the appropriate occasion/aesthetic/formality MODE rather than manually listing values:
   - "effortless", "chill" → casual mode
   - "put-together", "polished" → smart_casual mode
   - "sexy but classy" → glamorous + smart_casual modes
   - "date night" → date_night mode
   - "office" → work mode
   - "wedding" → wedding_guest mode
   - "looks expensive", "elevated" → quiet_luxury mode

3. **Aspirational language is about aesthetic, not price.** "Looks expensive", "looks designer",
   "elevated", "luxe" — they want items that LOOK premium, not items that ARE expensive.
   Do NOT set min_price. Use the quiet_luxury mode instead.

4. **Infer the garment type from context.** Every query has clues:
   - "Outfit" / "something for X" → category_l1: ["Tops", "Dresses"]. Only add "Bottoms" when
     user explicitly mentions pants/jeans/shorts/skirts.
   - Body-part references → the garment that covers it. "Shoulders", "arms", "chest", "back",
     "stomach" → category_l1: ["Tops", "Dresses", "Outerwear"]. "Legs", "thighs" → ["Bottoms", "Dresses"].
   - Garment features imply types: slit/hemline/bodice → Dresses. Rise/inseam → Bottoms.
     Collar/cuff → Tops. Hood/zipper/layering → Outerwear.
   - "No [feature]" without naming a garment → infer the garment that feature belongs to.
   - When in doubt, default to category_l1: ["Tops", "Dresses"]. Never leave category_l1 empty
     for vague queries — bottoms will pollute the results.

5. **When the user requests a specific attribute value, also avoid the contradicting values.**
   Requesting "long sleeves" means they do NOT want sleeveless, short, cap, or spaghetti strap results.
   Requesting "midi" means they do NOT want mini or micro. Always add the contradicting values to avoid.
   "With sleeves" means the user wants SUBSTANTIVE sleeves — at minimum 3/4 or long. Exclude
   Sleeveless, Spaghetti Strap, Short, AND Cap. If someone says "with sleeves" they don't want
   short sleeves — they want visible, meaningful sleeve coverage.
   Examples:
   - "long sleeves" → attributes: {{"sleeve_type": ["Long"]}}, avoid: {{"sleeve_type": ["Sleeveless", "Short", "Cap", "Spaghetti Strap"]}}
   - "with sleeves" → attributes: {{"sleeve_type": ["Long", "3/4"]}}, avoid: {{"sleeve_type": ["Sleeveless", "Short", "Cap", "Spaghetti Strap"]}}
   - "midi dress" → attributes: {{"length": ["Midi"]}}, avoid: {{"length": ["Mini", "Micro"]}}
   - "high rise" → attributes: {{"rise": ["High"]}}, avoid: {{"rise": ["Low"]}}
   - "maxi skirt" → attributes: {{"length": ["Maxi"]}}, avoid: {{"length": ["Mini", "Micro", "Cropped"]}}
   This is critical — without the avoid values, semantic search results with wrong attributes leak through.

6. **Non-filterable features go to semantic_query only.** Some things users mention cannot be filtered
   because our database has no attribute for them: slit, ruching, cutout, wrap, tie-front, drawstring,
   button-down, zipper placement, pocket detail, etc. For these:
   - Put them in semantic_query (FashionCLIP understands visual features)
   - Do NOT invent avoid values that don't match — never guess filter mappings for concepts we don't track
   - If the user says "no slit", put "closed hemline" in semantic_query (see Principle 7)

7. **semantic_query must be POSITIVE descriptions only — NEVER use negation.**
   FashionCLIP is a vision-language embedding model. It does NOT understand negation.
   "not backless" encodes close to "backless" and PULLS IN backless items.
   "not sheer" encodes close to "sheer" and PULLS IN sheer items.
   ALWAYS rephrase negatives as positive descriptions of what the user DOES want:
   - BAD: "not backless, not open-back" → GOOD: "closed back, full back coverage"
   - BAD: "not sheer, not see-through" → GOOD: "opaque solid fabric"
   - BAD: "not sleeveless" → GOOD: "with long sleeves"
   - BAD: "no slit" → GOOD: "closed hemline"
   - BAD: "not tight or clingy" → GOOD: "relaxed loose drape"
   - BAD: "without cutouts" → GOOD: "solid continuous fabric"
   Negation is handled by exclusion filters and modes — the semantic_query's ONLY job is to
   describe the positive visual appearance that FashionCLIP should match.

## SECTION 2: OUTPUT FORMAT

Return a JSON object with these fields:
- intent: "exact" | "specific" | "vague"
- algolia_query: string (product-name keywords for text search)
- semantic_query: string (rich visual description for FashionCLIP)
- modes: string[] (mode tags from the menu below)
- attributes: object (positive filter values — keys and allowed values listed below)
- avoid: object (negative filter values — same keys as attributes, for things user said NO to)
- brand: string | null
- max_price: number | null
- min_price: number | null
- on_sale_only: boolean
- confidence: number (0.0-1.0)

## SECTION 3: MODE MENU

{mode_menu}

MODE RULES:
- Pick zero or more modes. Modes handle abstract/subjective intent.
- Modes are expanded by code into concrete filter/exclusion values — you do NOT need to
  duplicate those values in attributes or avoid.
- funeral and religious_event automatically imply modest (full coverage).
- relaxed_fit and not_oversized are opposites — never pick both.
- For coverage requests ("hide arms", "covers back"), ALWAYS use the appropriate cover_* mode.
  Do NOT put exclusion values in "avoid" for things a mode already handles.

## SECTION 4: ATTRIBUTES (positive filters)

Only the following keys are valid. Use EXACT values from the allowed lists:

- **category_l1**: ["Tops", "Bottoms", "Dresses", "Outerwear", "Activewear", "Swimwear", "Intimates", "Accessories"]
  For vague/outfit queries, default to ["Tops", "Dresses"].
- **category_l2**: Use BROAD values. For "jacket" → ["Jacket", "Jackets"]. For subtypes include both: ["Bomber Jacket", "Jacket", "Jackets"]
- **patterns**: ["Solid", "Floral", "Striped", "Plaid", "Polka Dot", "Animal Print", "Abstract", "Geometric", "Tie Dye", "Camo", "Colorblock", "Tropical"]
- **colors**: ["Black", "White", "Red", "Blue", "Navy Blue", "Green", "Pink", "Yellow", "Purple", "Orange", "Brown", "Beige", "Cream", "Gray", "Burgundy", "Olive", "Taupe", "Off White", "Light Blue"]
- **formality**: ["Formal", "Semi-Formal", "Business Casual", "Smart Casual", "Casual"]
- **occasions**: ["Date Night", "Party", "Office", "Work", "Wedding Guest", "Vacation", "Workout", "Everyday", "Brunch", "Night Out", "Weekend", "Lounging", "Beach"]
- **fit_type**: ["Slim", "Fitted", "Regular", "Relaxed", "Oversized", "Loose"]
- **neckline**: ["V-Neck", "Crew", "Turtleneck", "Off-Shoulder", "Strapless", "Halter", "Scoop", "Square", "Sweetheart", "Cowl", "Boat", "One Shoulder", "Collared", "Hooded", "Mock", "Deep V-Neck", "Plunging"]
- **sleeve_type**: ["Sleeveless", "Short", "Long", "Cap", "Puff", "3/4", "Flutter", "Spaghetti Strap"]
- **length**: ["Mini", "Midi", "Maxi", "Cropped", "Floor-length", "Ankle", "Micro"]
- **rise**: ["High", "Mid", "Low"]
- **materials**: ["Cotton", "Linen", "Silk", "Satin", "Denim", "Faux Leather", "Wool", "Velvet", "Chiffon", "Lace", "Mesh", "Knit", "Jersey", "Fleece", "Sheer"]
- **silhouette**: ["A-Line", "Bodycon", "Flared", "Straight", "Wide Leg"]
- **seasons**: ["Summer", "Spring", "Fall", "Winter"]
- **style_tags**: ["Bohemian", "Romantic", "Glamorous", "Edgy", "Vintage", "Sporty", "Classic", "Modern", "Minimalist", "Preppy", "Streetwear", "Sexy", "Western", "Utility"]
- **brands**: Only if a specific brand is mentioned

ATTRIBUTE RULES:
- Only include values the user explicitly or strongly implies.
- Use exact values from the allowed lists above.
- For category_l2, include singular AND plural: ["Jacket", "Jackets"].
- category_l1 is ALWAYS a positive attribute, never in avoid.

## SECTION 5: AVOID (negative filters)

Same keys as attributes. Use for concrete things the user explicitly said NO to that
are NOT already handled by a mode.

Examples of when to use avoid:
- "no polyester" → avoid: {{"materials": ["Polyester"]}}  (no mode covers this)
- "no rips" → avoid: {{"style_tags": ["Distressed"]}}
- "no animal print" → avoid: {{"patterns": ["Animal Print"]}}
- "no yellow" → avoid: {{"colors": ["Yellow"]}}

Examples of when NOT to use avoid (use modes instead):
- "hides arms" → modes: ["cover_arms"]  (NOT avoid: {{"sleeve_type": [...]}})
- "modest" → modes: ["modest"]  (NOT avoid: {{"neckline": [...], ...}})
- "not too tight" → modes: ["relaxed_fit"]  (NOT avoid: {{"fit_type": [...]}})

Rule: If a mode exists that handles the user's negative intent, use the mode. Only use
avoid for specific concrete values that no mode covers.

## SECTION 6: SEARCH QUERIES

**algolia_query**: Product-name keywords for text search.
- Include ONLY terms that would appear in product names: "Floral Print Bomber Jacket", "Striped Knit Cardigan"
- Convert descriptive language: "floral leaves" → "leaf print" or "floral print"
- Remove filler/intent words: "a", "with", "for", "outfit", "look", "wear"
- If all terms became filters/modes, return empty string ""

**semantic_query**: Rich POSITIVE visual description for FashionCLIP image-similarity.
- Describe what the garment LOOKS LIKE — never use "not", "no", "without", "non-" (Principle 7)
- Expand with visual details: "floral leaves jacket" → "a jacket with botanical leaf and flower print pattern"
- For coverage queries, describe the covered version: "top that hides arms" → "a top with long opaque sleeves and full arm coverage"
- For "no backless" → "a top with a fully closed high back"
- For "not sheer" → "an opaque solid-fabric top"

## SECTION 7: PRICE, BRAND, INTENT

- **intent**: "exact" (pure brand search), "specific" (concrete product + attributes), "vague" (mood/aesthetic only)
- **brand**: Detected brand name or null
- **max_price**: From "under $50" → 50.0. "Looks expensive" is NOT a price constraint.
- **min_price**: From "over $200" → 200.0
- **on_sale_only**: True for "on sale", "discounted", "clearance". NOT for "affordable" or "cheap".
- **confidence**: 0.0-1.0

## SECTION 8: DECISION RULES

1. Use MODES for: coverage/modesty, fit preference, occasion, formality, aesthetic vibe, weather.
2. Use ATTRIBUTES for: concrete positive values (color, material, neckline, category, pattern).
3. Use AVOID for: (a) concrete negative values the user explicitly said NO to, AND (b) contradicting
   values when the user requested a specific attribute (per Principle 5).
4. Never duplicate: if a mode handles it, don't also put it in avoid.
5. Combine freely: modes + attributes + avoid can all be used together.
6. ALWAYS set category_l1. Never leave it empty. Default to ["Tops", "Dresses"] for vague queries.
7. If a feature isn't in our filter lists (slit, cutout, ruching, etc.), put it ONLY in
   semantic_query. Do NOT guess filter mappings — leave avoid empty for non-filterable features.

VOCABULARY TRANSLATION:
- "skirt with shorts underneath" → algolia_query="skort"
- "butter yellow" / "mustard" → colors: ["Yellow"]
- "chocolate brown" / "espresso" → colors: ["Brown"]
- "cherry red" / "crimson" → colors: ["Red", "Burgundy"]
- "navy" → colors: ["Navy Blue"]
- "nude" / "skin tone" → colors: ["Beige", "Taupe"]

Return ONLY valid JSON. No markdown, no explanation, no code blocks."""


# Build the prompt once at module load time
_SYSTEM_PROMPT = _build_system_prompt()


# =============================================================================
# Query Planner
# =============================================================================

class QueryPlanner:
    """LLM-based query planner using OpenAI (gpt-4o by default)."""

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

    _MAX_RETRIES = 2
    _RETRY_BACKOFF_SECONDS = [6, 12]  # Wait times for retry 1, 2

    def plan(self, query: str) -> Optional[SearchPlan]:
        """
        Generate a search plan for the given query.

        Retries on rate-limit (429) errors with exponential backoff.
        Returns None if the planner is disabled, times out, or fails.
        The caller should fall back to basic search (raw query, no filters).
        """
        if not self._enabled:
            logger.debug("Query planner disabled (no API key or feature flag off)")
            return None

        t_start = time.time()
        last_error = None

        for attempt in range(1 + self._MAX_RETRIES):
            try:
                # Build API params.
                # Reasoning models (gpt-5, o-series) don't support temperature
                # and use max_completion_tokens (which includes hidden reasoning
                # tokens). Reserve generous budget — reasoning can use 1-10k
                # tokens before producing the ~300-token JSON output.
                is_reasoning_model = any(
                    self._model.startswith(p)
                    for p in ("gpt-5", "o1", "o3", "o4")
                )

                api_params = {
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                    ],
                    "response_format": {"type": "json_object"},
                }

                if is_reasoning_model:
                    api_params["max_completion_tokens"] = 16384
                else:
                    api_params["temperature"] = 0.0
                    api_params["max_tokens"] = 800

                response = self.client.chat.completions.create(**api_params)

                raw = response.choices[0].message.content
                if not raw:
                    logger.warning("Query planner returned empty response")
                    return None

                data = json.loads(raw)

                # Fix common LLM mistake: nesting avoid inside attributes
                attributes = data.get("attributes", {})
                if isinstance(attributes, dict) and "avoid" in attributes:
                    nested_avoid = attributes.pop("avoid")
                    if isinstance(nested_avoid, dict):
                        existing = data.get("avoid", {})
                        if not isinstance(existing, dict):
                            existing = {}
                        for k, v in nested_avoid.items():
                            if k not in existing and isinstance(v, list):
                                existing[k] = v
                        data["avoid"] = existing
                    logger.debug(
                        "Extracted misplaced avoid from attributes dict",
                        extracted=data.get("avoid"),
                    )

                plan = SearchPlan(**data)

                latency_ms = int((time.time() - t_start) * 1000)
                logger.info(
                    "Query planner generated search plan",
                    query=query,
                    intent=plan.intent,
                    algolia_query=plan.algolia_query,
                    semantic_query=plan.semantic_query,
                    modes=plan.modes,
                    attributes=plan.attributes,
                    avoid=plan.avoid,
                    confidence=plan.confidence,
                    latency_ms=latency_ms,
                )
                return plan

            except json.JSONDecodeError as e:
                logger.warning("Query planner returned invalid JSON", error=str(e))
                return None
            except Exception as e:
                last_error = e
                error_str = str(e)
                # Retry on rate-limit (429) errors
                if "429" in error_str and attempt < self._MAX_RETRIES:
                    wait = self._RETRY_BACKOFF_SECONDS[attempt]
                    logger.info(
                        "Query planner rate-limited, retrying",
                        attempt=attempt + 1,
                        wait_seconds=wait,
                        query=query,
                    )
                    time.sleep(wait)
                    continue
                # Non-retryable error or out of retries
                break

        latency_ms = int((time.time() - t_start) * 1000)
        logger.warning(
            "Query planner failed, falling back to basic search",
            error=str(last_error),
            latency_ms=latency_ms,
        )
        return None

    # -----------------------------------------------------------------
    # plan_to_request_updates — convert plan into pipeline values
    # -----------------------------------------------------------------

    # Typo corrections for LLM field name mistakes
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
        "style_tag": "style_tags",
    }

    # Valid filter field names (for validation)
    _VALID_FILTER_FIELDS = {
        "category_l1", "category_l2", "patterns", "colors", "formality",
        "occasions", "fit_type", "neckline", "sleeve_type", "length",
        "rise", "materials", "silhouette", "seasons", "style_tags",
        "brands", "categories",
    }

    # Map filter key -> exclude_* request field
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

    def plan_to_request_updates(
        self, plan: SearchPlan
    ) -> Tuple[Dict[str, Any], Dict[str, List[str]], Dict[str, List[str]], List[str], str, str, str]:
        """
        Convert a SearchPlan into values the hybrid search pipeline can use.

        1. Expand modes into deterministic filters/exclusions via expand_modes()
        2. Merge mode_filters + plan.attributes → request_updates
        3. Merge mode_exclusions + plan.avoid → exclude_* fields
        4. Add brand/price/sale to request_updates

        Returns:
            Tuple of:
            - request_updates: Dict of HybridSearchRequest field updates
            - expanded_filters: Dict for lenient semantic post-filtering
            - exclude_updates: Dict of attribute values to EXCLUDE
            - matched_terms: [] (kept for API compatibility)
            - algolia_query: Optimized Algolia keyword query
            - semantic_query: Rich semantic query for FashionCLIP
            - intent_str: Intent string ("exact", "specific", "vague")
        """
        request_updates: Dict[str, Any] = {}

        # -----------------------------------------------------------
        # Step 1: Expand modes into filters and exclusions
        # -----------------------------------------------------------
        mode_filters, mode_exclusions, expanded_filters, name_exclusions = expand_modes(plan.modes)

        # -----------------------------------------------------------
        # Step 2: Merge mode_filters + plan.attributes → request_updates
        # -----------------------------------------------------------
        # Start with mode filters
        merged_filters: Dict[str, List[str]] = {k: list(v) for k, v in mode_filters.items()}

        # Merge in LLM attributes (correct typos first)
        for raw_field, values in plan.attributes.items():
            field = self._TYPO_CORRECTIONS.get(raw_field, raw_field)
            if field != raw_field:
                logger.info("Corrected LLM typo in attributes", original=raw_field, corrected=field)
            if field not in self._VALID_FILTER_FIELDS or not values:
                continue
            if field in merged_filters:
                # Union, no duplicates
                existing_lower = {v.lower() for v in merged_filters[field]}
                for v in values:
                    if v.lower() not in existing_lower:
                        merged_filters[field].append(v)
                        existing_lower.add(v.lower())
            else:
                merged_filters[field] = list(values)

        # Copy merged filters to request_updates
        for field, values in merged_filters.items():
            if values:
                request_updates[field] = values

        # Also add any attribute-derived values to expanded_filters
        for field, values in merged_filters.items():
            if field not in expanded_filters:
                expanded_filters[field] = list(values)
            else:
                existing_lower = {v.lower() for v in expanded_filters[field]}
                for v in values:
                    if v.lower() not in existing_lower:
                        expanded_filters[field].append(v)
                        existing_lower.add(v.lower())

        # -----------------------------------------------------------
        # Step 3: Merge mode_exclusions + plan.avoid → exclude_updates
        # -----------------------------------------------------------
        merged_exclusions: Dict[str, List[str]] = {k: list(v) for k, v in mode_exclusions.items()}

        # Merge in LLM avoid values (correct typos first)
        for raw_field, values in plan.avoid.items():
            field = self._TYPO_CORRECTIONS.get(raw_field, raw_field)
            if field != raw_field:
                logger.info("Corrected LLM typo in avoid", original=raw_field, corrected=field)
            if field not in self._VALID_FILTER_FIELDS or not values:
                continue
            if field in merged_exclusions:
                existing_lower = {v.lower() for v in merged_exclusions[field]}
                for v in values:
                    if v.lower() not in existing_lower:
                        merged_exclusions[field].append(v)
                        existing_lower.add(v.lower())
            else:
                merged_exclusions[field] = list(values)

        # Map exclusion keys to exclude_* request fields
        exclude_updates: Dict[str, List[str]] = {}
        for field, values in merged_exclusions.items():
            if not values:
                continue
            req_field = self._EXCLUDE_FIELD_MAP.get(field)
            if req_field:
                exclude_updates[field] = values
                request_updates[req_field] = values

        # -----------------------------------------------------------
        # Step 3b: Name exclusions (product-name substring drops)
        # -----------------------------------------------------------
        # Stash name_exclusions in request_updates with a special key.
        # The hybrid search service reads this and applies name-based
        # filtering as a last-resort catch for data quality gaps.
        if name_exclusions:
            request_updates["_name_exclusions"] = name_exclusions

        # -----------------------------------------------------------
        # Step 4: Brand / price / sale injection
        # -----------------------------------------------------------
        if plan.brand and "brands" not in request_updates:
            request_updates["brands"] = [plan.brand]

        if plan.max_price is not None:
            request_updates["max_price"] = plan.max_price
        if plan.min_price is not None:
            request_updates["min_price"] = plan.min_price
        if plan.on_sale_only:
            request_updates["on_sale_only"] = True

        # -----------------------------------------------------------
        # Return the 7-tuple (same interface as before)
        # -----------------------------------------------------------
        return (
            request_updates,
            expanded_filters,
            exclude_updates,
            [],  # matched_terms — no longer used, kept for API compat
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
