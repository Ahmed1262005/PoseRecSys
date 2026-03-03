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
from search.models import QueryIntent, FollowUpQuestion, FollowUpOption
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

    # Multiple diverse semantic queries for wider result variety.
    # Each targets a different visual subcategory or style angle.
    # When present, these REPLACE semantic_query for the search.
    semantic_queries: List[str] = Field(
        default_factory=list,
        description="2-4 diverse FashionCLIP queries targeting different visual angles"
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

    # Follow-up questions for vague/ambiguous queries (Phase 1).
    # Raw dicts from LLM output; parsed into FollowUpQuestion models
    # by QueryPlanner.plan() and stored in parsed_follow_ups.
    follow_ups: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Raw follow-up questions from the LLM (unparsed dicts)"
    )

    # Parsed follow-ups (populated after plan() validates the raw dicts)
    parsed_follow_ups: List[Any] = Field(
        default_factory=list,
        description="Validated FollowUpQuestion objects (populated by QueryPlanner.plan())"
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
   - **Skin coverage** (arms, chest, back, legs, stomach, shoulders): Needs opaque fabric.
     Sheer/mesh/lace don't count — you can SEE THROUGH them, so the body part is NOT covered.
     A lace-sleeve top does NOT hide arms. A mesh panel does NOT cover the back.
     A mesh dress does NOT cover shoulders. If the user says "covers [body part]", they mean
     you cannot see skin through the fabric.
     For these, use cover_arms/cover_chest/cover_back/cover_legs/cover_stomach + opaque modes.
     "Covers shoulders" = skin coverage = cover_straps + opaque modes.
   - **Strap/structural coverage** (bra straps only): About garment STRUCTURE, not fabric.
     A chiffon blouse with high neckline and cap+ sleeves DOES hide bra straps.
     ONLY use cover_straps WITHOUT opaque for "hides bra straps" / "doesn't show bra straps".
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
   - "not too revealing", "not revealing", "not too sexy" → cover_chest + cover_stomach modes.
     This is a MODERATE coverage request — avoid crop tops, deep necklines, and backless, but
     do NOT use full modest mode. The user still wants to look good, just not overly exposed.

3. **Aspirational language means affordable luxury.** "Looks expensive", "looks designer",
   "elevated", "luxe" — they want items that LOOK premium but are NOT actually expensive.
   Do NOT set min_price. Use the quiet_luxury mode AND set max_price to 100.
   The whole point of "looks expensive" is getting the look without the price tag.

4. **Infer the garment type from context.** Every query has clues:
   - When user explicitly names a garment type, use ONLY that type:
     "top" / "blouse" / "shirt" → category_l1: ["Tops"] ONLY.
     "dress" → category_l1: ["Dresses"] ONLY.
     "pants" / "jeans" / "trousers" / "skirt" → category_l1: ["Bottoms"] ONLY.
     "jacket" / "coat" → category_l1: ["Outerwear"] ONLY.
     NEVER add extra categories when the user specified a garment type.
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

8. **We sell CLOTHING, not underwear.** This is a women's fashion clothing store.
   We do NOT sell intimates, underwear, bras, lingerie, or shapewear.
   When a user mentions undergarments, they are ALWAYS talking about clothing that
   HIDES or works well OVER those undergarments:
   - "doesn't show underwear lines" → thick/structured Bottoms or Dresses (NOT intimates)
   - "doesn't show bra straps" → Tops/Dresses with cover_straps mode
   - "no VPL" (visible panty lines) → Bottoms/Dresses with thicker/structured fabric
   - "works with a strapless bra" → Tops/Dresses with Strapless/Off-Shoulder neckline
   - "hides bra" → Tops/Dresses with opaque mode
   NEVER set category_l1 to "Intimates" — it does not exist in our catalog.

## SECTION 2: OUTPUT FORMAT

Return a JSON object with these fields:
- intent: "exact" | "specific" | "vague"
- algolia_query: string (product-name keywords for text search)
- semantic_query: string (rich visual description for FashionCLIP — used as fallback)
- semantic_queries: string[] (2-4 DIVERSE FashionCLIP queries — see Section 6b below)
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

- **category_l1**: ["Tops", "Bottoms", "Dresses", "Outerwear", "Activewear", "Swimwear", "Accessories"]
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

## SECTION 6b: DIVERSE SEMANTIC QUERIES (semantic_queries)

A single semantic query pulls visually-similar items that cluster together. To get DIVERSE results,
generate 2-4 semantic queries that each target a DIFFERENT visual angle of the user's intent.

RULES:
- Each query follows the same positive-only rules as semantic_query (Principle 7)
- Vary by: garment TYPE, STYLE angle, SILHOUETTE, COLOR mood, or FABRIC texture
- Do NOT repeat the same description with synonyms — each must pull a genuinely different cluster
- For exact brand queries: just 1 query is fine (set semantic_queries to [semantic_query])
- For specific queries: 2-3 queries exploring different interpretations
- For vague queries: 3-4 queries covering different garment types or aesthetic angles
- Keep each query under 77 tokens (FashionCLIP model limit)

Example for "work outfit":
```json
"semantic_queries": [
  "structured tailored blazer in solid neutral tones, professional office wear",
  "elegant silk button-up blouse with refined collar, polished workwear",
  "fitted knee-length pencil dress in dark fabric, clean professional silhouette"
]
```

Example for "cute summer dress":
```json
"semantic_queries": [
  "flowy floral print midi sundress in soft pastel colors, lightweight fabric",
  "fitted ribbed knit mini dress in bright solid color, casual summer style",
  "tiered ruffle cotton maxi dress with smocked bodice, bohemian summer"
]
```

Example for "red midi dress" (specific — fewer needed):
```json
"semantic_queries": [
  "a red satin midi dress with a fitted bodice and flowing skirt",
  "a red knit bodycon midi dress with long sleeves, elegant and fitted"
]
```

If you can only think of one meaningful angle, set semantic_queries to [semantic_query].

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

## SECTION 9: FOLLOW-UP QUESTIONS

**Goal:** For vague or under-specified queries, generate 1-3 contextual follow-up questions that
materially reduce ambiguity and improve retrieval quality. Only ask follow-ups when the missing
info would meaningfully change the results.

**Output:** Return a JSON array in the key "follow_ups".
Each follow-up object has:
- "dimension": one of ["garment_type", "fit", "vibe", "occasion", "formality", "coverage", "price", "color"]
- "question": natural, conversational question text specific to THIS query
- "options": 2-4 choices, each with:
  - "label": 2-5 words (symbols allowed like "$50-$100")
  - "filters": a filter PATCH to apply

If the query is specific enough that no follow-ups are needed, return: "follow_ups": []

**Allowed filter keys (STRICT):**
The "filters" object MUST only use these keys:
category_l1, colors, formality, min_price, max_price, occasions, fit_type, silhouette,
sleeve_type, length, neckline, materials, patterns, style_tags, modes.

**Canonical values (STRICT):**
Values must come from the taxonomy in Section 4 above. Do NOT invent new enum values.

**"modes" is COVERAGE-ONLY (IMPORTANT):**
modes is reserved strictly for coverage/modesty constraints (e.g. "cover_arms", "cover_chest",
"cover_legs", "cover_midriff", "fully_covered"). Do NOT put vibes or formality concepts in modes.

**Merge semantics (CRITICAL):**
Each option's "filters" is a PATCH applied on top of the existing search plan. Follow-ups NARROW,
not widen:

1) Overwrite keys (single-choice narrowing): category_l1, formality, min_price, max_price
2) Replace keys (do NOT union; choose one path): colors, occasions, style_tags, materials,
   patterns, modes, fit_type, silhouette, sleeve_type, length, neckline
3) Price tightening: If an existing min_price/max_price already exists, the new bounds MUST
   tighten or stay within it. Never widen.

### Dimension guide

**"fit"** — silhouette and proportion (HIGH LIFT for most queries).
"Fitted vs relaxed vs oversized" changes results more than almost any other dimension.
Use fit_type and/or silhouette filters. Ask for any broad or outfit query.

**"vibe"** — style direction and aesthetic (HIGH LIFT for vague queries).
Each vibe option should carry MULTIPLE filter keys to give the search real signal:
- "Romantic & feminine" → {{"style_tags": ["Romantic"], "patterns": ["Floral"], "materials": ["Lace", "Chiffon"]}}
- "Glamorous" → {{"style_tags": ["Glamorous"], "materials": ["Satin", "Silk"], "fit_type": ["Fitted", "Slim"]}}
- "Edgy" → {{"style_tags": ["Edgy"], "materials": ["Faux Leather"], "patterns": ["Solid"]}}
- "Minimal & clean" → {{"style_tags": ["Minimalist"], "patterns": ["Solid"], "fit_type": ["Slim", "Fitted"]}}
- "Relaxed & effortless" → {{"style_tags": ["Bohemian"], "fit_type": ["Relaxed", "Loose"]}}
- "Classic & polished" → {{"style_tags": ["Classic"], "fit_type": ["Fitted", "Regular"], "patterns": ["Solid"]}}
- "Sporty casual" → {{"style_tags": ["Sporty"], "fit_type": ["Regular", "Relaxed"]}}
- "Bold & trendy" → {{"style_tags": ["Streetwear", "Modern"], "patterns": ["Colorblock", "Abstract"]}}
Pick 3-4 vibes relevant to the query. Do NOT always use the same set.

**"garment_type"** — category anchor. Only ask when no garment is specified. Prefer single-item
anchors: Dresses / Tops / Bottoms / Outerwear / Jumpsuits.

**"occasion"** / **"formality"** — ask ONE of these, not both.
Formality is LOW-LIFT when the query already implies a dress code: "date" → semi-formal,
"work" → business casual, "wedding guest" → formal, "party" → semi-formal, "brunch" → casual.
If the occasion already signals formality, skip formality and ask a higher-lift dimension.

**"coverage"** — modesty. Only ask if the query mentions modesty, user context says covered,
or the category commonly needs it (tops, dresses) AND the query is broad.

**"price"** / **"color"** — low lift. Only ask if the user explicitly signals budget sensitivity
or color preference.

### Concrete attributes (category-specific)

When asking about fit or a concrete attribute, choose what matters most for the category:
- **tops** → neckline (highest lift), then sleeve_type, then fit
- **dresses** → length (mini/midi/maxi), then fit, then sleeve_type or neckline
- **bottoms/jeans** → fit (slim/relaxed/wide-leg), then length
- **outerwear** → length (cropped/mid/long), then fit

These can use dimension "fit" with the relevant filter keys.

### When to ask follow-ups (Ambiguity rubric)

Do NOT ask follow-ups if the query already includes enough constraints to narrow results.
Examples: "red midi dress", "black blazer for work", "long sleeve modest maxi dress under $120" → follow_ups: []

Ask follow-ups when the query is under-specified. Generate 1-3 follow-ups when ANY trigger
below applies, choosing the highest-lift missing dimensions.

**Trigger A — Broad category only (IMPORTANT):**
Query is ONLY a broad category with no other constraints ("tops", "dresses", "jeans", "skirts"):
- ALWAYS generate 2-3 follow-ups.
- Priority: 1) fit (silhouette/proportion), 2) vibe OR category-specific attribute (neckline
  for tops, length for dresses), 3) occasion (only if no lifestyle context)

**Trigger B — Occasion/outfit intent without an anchor:**
Query implies an occasion but no garment ("outfit for a date", "vacation outfits"):
- Ask garment_type first, then fit OR vibe, then occasion only if truly unclear.

**Trigger C — Vague descriptors:**
Broad adjectives without constraints ("cute", "nice", "hot", "elegant", "put together"):
- Ask vibe first (highest lift — maps directly to style+fabric+fit).
- Then garment_type or fit if still broad.

**Trigger D — Coverage/modesty ambiguity:**
Ask coverage ONLY if: the query mentions modesty/coverage, or user context indicates covered,
or the category commonly benefits AND the query is broad.

**Trigger E — Price ambiguity:**
Ask price ONLY if: the query mentions budget sensitivity ("cheap", "affordable", "designer"),
or user context provides budget AND the query is very broad.

**Trigger F — Color ambiguity:**
Ask color ONLY if: the user explicitly asked for a color, or color is central.

### How many follow-ups
- Max 3 total.
- Trigger A (broad category): return 2-3.
- Otherwise: return 1-3 depending on how many high-signal dimensions are missing.

### Priority order (pick highest expected lift first)
1) garment_type (only if no category anchor exists)
2) fit (silhouette/proportion — high lift for nearly all broad queries)
3) vibe (style direction — carries multiple filter signals)
4) concrete attribute (category-specific: neckline for tops, length for dresses, etc.)
5) occasion OR formality (choose ONE — only when genuinely ambiguous about context)
6) coverage (modes) when relevant (see Trigger D)
7) price (see Trigger E)
8) color (see Trigger F)

### Rules
- Do NOT ask about anything already specified in the query or the existing plan.
- For "occasion OR formality": pick ONE, and only if no higher-lift dimension remains.
  If the query implies an occasion, skip formality (it's already implied).
- Prefer fit and vibe over formality for most queries — they change results more.
- Garment type: prefer single-item anchors (Dresses / Tops / Bottoms / Outerwear).
- Questions must be specific to the query context (not generic).
  Bad: "What's your budget?" Good: "What price range works for your date-night look?"
- Options must be meaningfully distinct and map cleanly to filters.
- Avoid judgmental or stereotype-y labels. Use neutral language.

### Examples

Example 1: query = "tops" (Trigger A — broad category)
```json
"follow_ups": [
  {{
    "dimension": "fit",
    "question": "What fit do you prefer?",
    "options": [
      {{"label": "Fitted", "filters": {{"fit_type": ["Fitted", "Slim"]}}}},
      {{"label": "Relaxed", "filters": {{"fit_type": ["Relaxed", "Loose"]}}}},
      {{"label": "Oversized", "filters": {{"fit_type": ["Oversized"]}}}},
      {{"label": "Cropped", "filters": {{"length": ["Cropped"]}}}}
    ]
  }},
  {{
    "dimension": "vibe",
    "question": "What vibe are you going for?",
    "options": [
      {{"label": "Minimal & clean", "filters": {{"style_tags": ["Minimalist"], "patterns": ["Solid"]}}}},
      {{"label": "Romantic", "filters": {{"style_tags": ["Romantic"], "patterns": ["Floral"], "materials": ["Lace"]}}}},
      {{"label": "Edgy", "filters": {{"style_tags": ["Edgy"], "materials": ["Faux Leather"]}}}},
      {{"label": "Classic", "filters": {{"style_tags": ["Classic"], "patterns": ["Solid", "Striped"]}}}}
    ]
  }},
  {{
    "dimension": "occasion",
    "question": "Where will you wear these?",
    "options": [
      {{"label": "Everyday", "filters": {{"occasions": ["Everyday"]}}}},
      {{"label": "Work", "filters": {{"occasions": ["Work"]}}}},
      {{"label": "Going out", "filters": {{"occasions": ["Night Out"]}}}},
      {{"label": "Date", "filters": {{"occasions": ["Date Night"]}}}}
    ]
  }}
]
```

Example 2: query = "outfit for a first date" (Trigger B — occasion without anchor)
```json
"follow_ups": [
  {{
    "dimension": "garment_type",
    "question": "What are you looking for?",
    "options": [
      {{"label": "A dress", "filters": {{"category_l1": ["Dresses"]}}}},
      {{"label": "A top", "filters": {{"category_l1": ["Tops"]}}}},
      {{"label": "A jumpsuit", "filters": {{"category_l1": ["Dresses"]}}}},
      {{"label": "Outerwear", "filters": {{"category_l1": ["Outerwear"]}}}}
    ]
  }},
  {{
    "dimension": "vibe",
    "question": "What vibe for your date?",
    "options": [
      {{"label": "Romantic & feminine", "filters": {{"style_tags": ["Romantic"], "patterns": ["Floral"], "materials": ["Lace", "Chiffon"]}}}},
      {{"label": "Effortless & cool", "filters": {{"style_tags": ["Modern"], "fit_type": ["Relaxed"]}}}},
      {{"label": "Glamorous", "filters": {{"style_tags": ["Glamorous"], "materials": ["Satin", "Silk"], "fit_type": ["Fitted"]}}}},
      {{"label": "Minimal & chic", "filters": {{"style_tags": ["Minimalist"], "patterns": ["Solid"], "fit_type": ["Slim"]}}}}
    ]
  }}
]
```

Example 3: query = "something cute" (Trigger C — vague descriptor)
```json
"follow_ups": [
  {{
    "dimension": "vibe",
    "question": "What kind of cute?",
    "options": [
      {{"label": "Flirty & feminine", "filters": {{"style_tags": ["Romantic", "Sexy"], "materials": ["Lace", "Satin"]}}}},
      {{"label": "Casual & minimal", "filters": {{"style_tags": ["Minimalist"], "fit_type": ["Regular", "Relaxed"], "patterns": ["Solid"]}}}},
      {{"label": "Bold & trendy", "filters": {{"style_tags": ["Streetwear", "Modern"], "patterns": ["Colorblock"]}}}},
      {{"label": "Classic & polished", "filters": {{"style_tags": ["Classic"], "fit_type": ["Fitted"], "patterns": ["Solid"]}}}}
    ]
  }},
  {{
    "dimension": "garment_type",
    "question": "What are you looking for?",
    "options": [
      {{"label": "A dress", "filters": {{"category_l1": ["Dresses"]}}}},
      {{"label": "A top", "filters": {{"category_l1": ["Tops"]}}}},
      {{"label": "Bottoms", "filters": {{"category_l1": ["Bottoms"]}}}}
    ]
  }}
]
```

Example 4: query = "dresses" (Trigger A — broad category)
```json
"follow_ups": [
  {{
    "dimension": "fit",
    "question": "What length and shape?",
    "options": [
      {{"label": "Mini", "filters": {{"length": ["Mini"]}}}},
      {{"label": "Midi", "filters": {{"length": ["Midi"]}}}},
      {{"label": "Maxi", "filters": {{"length": ["Maxi"]}}}},
      {{"label": "Bodycon", "filters": {{"silhouette": ["Bodycon"], "fit_type": ["Fitted"]}}}}
    ]
  }},
  {{
    "dimension": "vibe",
    "question": "What's the vibe?",
    "options": [
      {{"label": "Romantic & flowy", "filters": {{"style_tags": ["Romantic"], "silhouette": ["A-Line"], "materials": ["Chiffon"]}}}},
      {{"label": "Glamorous", "filters": {{"style_tags": ["Glamorous"], "materials": ["Satin", "Silk"]}}}},
      {{"label": "Casual & easy", "filters": {{"style_tags": ["Modern"], "fit_type": ["Relaxed"], "materials": ["Cotton", "Jersey"]}}}},
      {{"label": "Edgy", "filters": {{"style_tags": ["Edgy"], "materials": ["Faux Leather"], "patterns": ["Solid"]}}}}
    ]
  }}
]
```

Example 5: query = "red midi dress" (specific enough — no follow-ups)
```json
"follow_ups": []
```

## SECTION 10: PERSONALIZED FOLLOW-UPS (when user context is provided)

When a "User context:" prefix is present, personalize follow-ups as follows:

1. **Reorder options** so the most likely choice for this user is FIRST.
   - Prefer behavior-based evidence (clicks/saves/history) over demographics.
   - If no strong evidence exists, use "most popular first" defaults.

2. **Calibrate price options** to the user's typical range (if budget is known).
   - Use 3 bands that partition their range.
     Example: user range $15-$80 → "Under $30" / "$30-$60" / "$60+"
     Example: user range $80-$300 → "Under $100" / "$100-$200" / "$200+"
   - Never suggest ranges far outside the user's range.

3. **Coverage ordering:**
   - If modesty=covered, put coverage-friendly options first.
   - If modesty=balanced, keep default ordering.

4. **Match vibe/style ordering** to user preferences.
   - If user commonly prefers minimalist, reorder vibe options to put "Minimal" first.
   - If user prefers boho, reorder to put "Bohemian" / "Romantic" first.
   - All values must be canonical (from Section 4).

5. **Match fit ordering** to user preferences.
   - If user prefers oversized/relaxed styles, reorder fit options accordingly.
   - If user prefers fitted, put "Fitted" / "Slim" first.

6. **Do NOT remove options** — only reorder them.
7. **Skip follow-up questions** already answered by user context or the query.
8. **If no user context is provided**, generate follow-ups in the default order (most popular first).

## SECTION 11: REFINEMENT MODE

When the user message starts with "[REFINEMENT]", you are refining an existing search — NOT starting
fresh. The user already searched, saw follow-up questions, and selected answers.

**RULES FOR REFINEMENT:**

1. **Selected filters are MANDATORY.** The selections listed in the user message MUST appear verbatim
   in your "attributes" output. Do NOT weaken, remove, reinterpret, or contradict them.
   If the user selected {{"fit_type": ["Fitted", "Slim"]}}, your attributes MUST include
   "fit_type": ["Fitted", "Slim"].

2. **Merge with original intent.** The original query provides the base context (occasion, garment
   type, vibe). The selections NARROW it. Combine them — don't replace the original intent.
   Example: query="outfit for a date" + selection {{"category_l1": ["Dresses"]}} →
   your plan should be about dresses for a date, not just generic dresses.

3. **Generate new semantic_queries** that describe what the user NOW wants — original intent +
   all selected filters woven into visual descriptions. Each query should paint a picture
   FashionCLIP can match.
   Example: query="tops" + fitted + edgy → semantic_queries: [
     "fitted black faux leather crop top with edgy streetwear style",
     "slim fitted ribbed tank top in dark solid color, modern edge",
     "structured fitted bodysuit in sleek fabric, edgy minimalist"
   ]

4. **Generate 1-2 NEW follow-up questions** about dimensions NOT yet answered. The user message
   lists "Answered dimensions" — do NOT re-ask those. Pick the next highest-lift unanswered
   dimensions from the priority list in Section 9.

5. **Keep the same intent classification** as the original query would get, unless selections
   make it more specific (e.g. vague + category selection → specific).

6. **algolia_query** should include product-name keywords that reflect the refinement. Remove
   filler words. If all terms are in filters, return empty string "".

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

    @staticmethod
    def _parse_follow_ups(raw: Any) -> List[FollowUpQuestion]:
        """Parse raw follow_ups dicts from LLM into validated FollowUpQuestion objects.

        Gracefully skips malformed entries. Returns empty list on any issues.
        """
        if not isinstance(raw, list):
            return []
        parsed: List[FollowUpQuestion] = []
        for raw_q in raw:
            if not isinstance(raw_q, dict):
                continue
            try:
                options = []
                for raw_opt in raw_q.get("options", []):
                    if isinstance(raw_opt, dict) and "label" in raw_opt:
                        options.append(FollowUpOption(
                            label=raw_opt["label"],
                            filters=raw_opt.get("filters", {}),
                        ))
                if options and "question" in raw_q:
                    parsed.append(FollowUpQuestion(
                        dimension=raw_q.get("dimension", "other"),
                        question=raw_q["question"],
                        options=options,
                    ))
            except Exception:
                continue  # Skip malformed follow-up, non-fatal
        return parsed

    @staticmethod
    def _format_context_line(user_context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Format the [User context: ...] line from a user_context dict.

        Returns None if user_context is empty or produces no parts.
        """
        if not user_context:
            return None

        parts = []

        # Age group
        age = user_context.get("age_group")
        if age:
            parts.append(f"age={age}")

        # Style persona
        style = user_context.get("style_persona")
        if style:
            if isinstance(style, list):
                parts.append(f"style={'/'.join(style)}")
            else:
                parts.append(f"style={style}")

        # Brand affinity — use cluster descriptions (natural language)
        descs = user_context.get("cluster_descriptions")
        if descs:
            if isinstance(descs, list):
                parts.append(f"brand_affinity={'/'.join(descs[:3])}")
            else:
                parts.append(f"brand_affinity={descs}")

        # Price range
        price_range = user_context.get("price_range")
        if price_range and isinstance(price_range, dict):
            pmin = price_range.get("min")
            pmax = price_range.get("max")
            if pmin is not None and pmax is not None:
                parts.append(f"price=${pmin}-${pmax}")
            elif pmax is not None:
                parts.append(f"price=up to ${pmax}")

        # Brand openness
        openness = user_context.get("brand_openness")
        if openness:
            parts.append(f"openness={openness}")

        # Modesty
        modesty = user_context.get("modesty")
        if modesty:
            parts.append(f"modesty={modesty}")

        if not parts:
            return None

        return f"[User context: {', '.join(parts)}]"

    @staticmethod
    def _build_user_message(
        query: str,
        user_context: Optional[Dict[str, Any]] = None,
        selected_filters: Optional[Dict[str, Any]] = None,
        selection_labels: Optional[List[str]] = None,
    ) -> str:
        """Build the user message with optional context prefix and refinement block.

        When user_context is provided, prepend a compact context line (~80-120
        tokens) so the LLM can personalize follow-up option ordering per
        Section 10.

        When selected_filters / selection_labels are provided, the message is
        formatted in REFINEMENT mode (Section 11 of the system prompt). The
        LLM sees the original query plus the user's follow-up selections and
        generates updated semantic queries, modes, avoids, and new follow-ups.

        Args:
            query: The raw search query.
            user_context: Optional dict with keys: age_group, brand_clusters,
                cluster_descriptions, price_range, style_persona,
                brand_openness, modesty.
            selected_filters: Optional dict of follow-up filter selections
                (e.g. {"fit_type": ["Fitted"], "modes": ["cover_arms"]}).
            selection_labels: Optional list of human-readable labels
                (e.g. ["Fitted", "Covered arms"]).

        Returns:
            User message string.
        """
        context_line = QueryPlanner._format_context_line(user_context)

        # ------------------------------------------------------------------
        # REFINEMENT mode — when follow-up selections are provided, format
        # the message so Section 11 of the system prompt activates.
        # ------------------------------------------------------------------
        if selected_filters:
            answered_dimensions = QueryPlanner._infer_answered_dimensions(selected_filters)

            parts = []
            if context_line:
                parts.append(context_line)

            parts.append("[REFINEMENT]")
            parts.append(f'Original query: "{query}"')

            if selection_labels:
                parts.append(f"User selected: {', '.join(selection_labels)}")

            filters_json = json.dumps(selected_filters, indent=2)
            parts.append(f"Selected filters:\n{filters_json}")

            answered = ", ".join(answered_dimensions) if answered_dimensions else "none"
            parts.append(f"Answered dimensions: {answered}")
            parts.append(f"Do NOT re-ask: {answered}")

            return "\n\n".join(parts)

        # ------------------------------------------------------------------
        # Normal mode — optional context prefix + raw query.
        # ------------------------------------------------------------------
        if context_line:
            return f"{context_line}\n\n{query}"

        return query

    def plan(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None,
        selected_filters: Optional[Dict[str, Any]] = None,
        selection_labels: Optional[List[str]] = None,
    ) -> Optional[SearchPlan]:
        """
        Generate a search plan for the given query.

        When selected_filters / selection_labels are provided, the planner
        runs in REFINEMENT mode (Section 11): the LLM sees the original
        query plus the user's follow-up selections and generates updated
        semantic queries, modes, avoids, and new follow-up questions.
        After the LLM responds, selected filters are force-injected into
        the plan so they cannot be dropped.

        Args:
            query: The user's search query.
            user_context: Optional compact dict with user profile info for
                personalized follow-ups. Keys: age_group, brand_clusters,
                cluster_descriptions, price_range, style_persona,
                brand_openness, modesty.
            selected_filters: Optional dict of follow-up filter selections.
            selection_labels: Optional list of human-readable option labels.

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

                # Build user message: optional context prefix + query.
                # When selected_filters are present, the message includes a
                # [REFINEMENT] block activating Section 11.
                user_message = self._build_user_message(
                    query, user_context, selected_filters, selection_labels,
                )

                api_params = {
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    "response_format": {"type": "json_object"},
                }

                if is_reasoning_model:
                    # Budget: ~560 output tokens + ~7,000 reasoning tokens.
                    # Reasoning models use hidden chain-of-thought that counts
                    # against this limit.  8192 is generous enough to avoid
                    # truncation while preventing the model from spending 16k
                    # tokens thinking (which was causing 40-66s latencies).
                    api_params["max_completion_tokens"] = 8192
                else:
                    api_params["temperature"] = 0.0
                    api_params["max_tokens"] = 1600

                response = self.client.chat.completions.create(**api_params)

                # Check for truncation — if finish_reason is "length", the
                # model hit max_completion_tokens and the JSON may be incomplete.
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "length":
                    logger.warning(
                        "Query planner output truncated (hit max_completion_tokens)",
                        model=self._model,
                        query=query,
                        max_tokens=api_params.get("max_completion_tokens") or api_params.get("max_tokens"),
                    )

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

                # REFINEMENT mode: force-inject selected filters into the
                # plan so they are ALWAYS present even if the LLM omitted
                # or weakened them.
                if selected_filters:
                    if "attributes" not in data or not isinstance(data["attributes"], dict):
                        data["attributes"] = {}
                    for key, values in selected_filters.items():
                        if key == "modes":
                            # Modes go to the modes list, not attributes
                            if isinstance(values, list):
                                existing_modes = data.get("modes") or []
                                for m in values:
                                    if m not in existing_modes:
                                        existing_modes.append(m)
                                data["modes"] = existing_modes
                            continue
                        if key in ("min_price", "max_price", "on_sale_only"):
                            # Price/sale go to top-level fields
                            data[key] = values
                            continue
                        if isinstance(values, list) and values:
                            data["attributes"][key] = values

                plan = SearchPlan(**data)

                # Parse follow_ups from raw LLM output into validated
                # FollowUpQuestion Pydantic models.
                plan.parsed_follow_ups = self._parse_follow_ups(
                    data.get("follow_ups", [])
                )

                latency_ms = int((time.time() - t_start) * 1000)
                is_refinement = bool(selected_filters)
                logger.info(
                    "Query planner generated search plan",
                    query=query,
                    refinement=is_refinement,
                    intent=plan.intent,
                    algolia_query=plan.algolia_query,
                    semantic_query=plan.semantic_query,
                    modes=plan.modes,
                    attributes=plan.attributes,
                    avoid=plan.avoid,
                    follow_ups_count=len(plan.parsed_follow_ups),
                    confidence=plan.confidence,
                    latency_ms=latency_ms,
                    **({"selection_labels": selection_labels} if is_refinement else {}),
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
    # Refinement planner — second LLM call after follow-up selections
    # -----------------------------------------------------------------

    # Map filter keys → follow-up dimension names for the LLM
    _FILTER_KEY_TO_DIMENSION = {
        "category_l1": "garment_type",
        "category_l2": "garment_type",
        "fit_type": "fit",
        "silhouette": "fit",
        "style_tags": "vibe",
        "patterns": "vibe",
        "materials": "vibe",
        "formality": "formality",
        "occasions": "occasion",
        "modes": "coverage",
        "min_price": "price",
        "max_price": "price",
        "colors": "color",
        "color_family": "color",
        "neckline": "fit",
        "sleeve_type": "fit",
        "length": "fit",
        "rise": "fit",
    }

    @staticmethod
    def _infer_answered_dimensions(selected_filters: Dict[str, Any]) -> List[str]:
        """Infer which follow-up dimensions the user already answered from filter keys."""
        dims: set = set()
        for key in selected_filters:
            dim = QueryPlanner._FILTER_KEY_TO_DIMENSION.get(key)
            if dim:
                dims.add(dim)
        return sorted(dims)

    @staticmethod
    def _build_refine_user_message(
        original_query: str,
        selected_filters: Dict[str, Any],
        selection_labels: Optional[List[str]] = None,
        answered_dimensions: Optional[List[str]] = None,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build user message for the refinement planner.

        Prefixed with [REFINEMENT] so the system prompt's Section 11 activates.
        Includes the original query, selected filters, and which dimensions are answered.
        """
        parts = []

        # User context (same format as initial planner)
        if user_context:
            ctx_parts = []
            age = user_context.get("age_group")
            if age:
                ctx_parts.append(f"age={age}")
            style = user_context.get("style_persona")
            if style:
                ctx_parts.append(f"style={'/'.join(style) if isinstance(style, list) else style}")
            descs = user_context.get("cluster_descriptions")
            if descs and isinstance(descs, list):
                ctx_parts.append(f"brand_affinity={'/'.join(descs[:3])}")
            price_range = user_context.get("price_range")
            if price_range and isinstance(price_range, dict):
                pmin, pmax = price_range.get("min"), price_range.get("max")
                if pmin is not None and pmax is not None:
                    ctx_parts.append(f"price=${pmin}-${pmax}")
                elif pmax is not None:
                    ctx_parts.append(f"price=up to ${pmax}")
            modesty = user_context.get("modesty")
            if modesty:
                ctx_parts.append(f"modesty={modesty}")
            if ctx_parts:
                parts.append(f"[User context: {', '.join(ctx_parts)}]")

        # Refinement header
        parts.append("[REFINEMENT]")
        parts.append(f'Original query: "{original_query}"')

        # Selected filters with labels for context
        if selection_labels:
            parts.append(f"User selected: {', '.join(selection_labels)}")

        filters_json = json.dumps(selected_filters, indent=2)
        parts.append(f"Selected filters:\n{filters_json}")

        answered = ", ".join(answered_dimensions) if answered_dimensions else "none"
        parts.append(f"Answered dimensions: {answered}")
        parts.append(f"Do NOT re-ask: {answered}")

        return "\n\n".join(parts)

    def refine(
        self,
        original_query: str,
        selected_filters: Dict[str, Any],
        selection_labels: Optional[List[str]] = None,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[SearchPlan]:
        """
        Generate a refined search plan incorporating user's follow-up selections.

        Uses the same system prompt as plan() but with a [REFINEMENT]-prefixed
        user message (activating Section 11 instructions). The LLM generates
        updated semantic queries, attributes, and new follow-ups.

        Returns None on failure (caller should fall back to skip_planner path).
        """
        if not self._enabled:
            logger.debug("Query planner disabled — cannot refine")
            return None

        answered_dimensions = self._infer_answered_dimensions(selected_filters)

        user_message = self._build_refine_user_message(
            original_query=original_query,
            selected_filters=selected_filters,
            selection_labels=selection_labels,
            answered_dimensions=answered_dimensions,
            user_context=user_context,
        )

        t_start = time.time()
        last_error = None

        for attempt in range(1 + self._MAX_RETRIES):
            try:
                is_reasoning_model = any(
                    self._model.startswith(p)
                    for p in ("gpt-5", "o1", "o3", "o4")
                )

                api_params = {
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    "response_format": {"type": "json_object"},
                }

                if is_reasoning_model:
                    api_params["max_completion_tokens"] = 8192
                else:
                    api_params["temperature"] = 0.0
                    api_params["max_tokens"] = 1600

                response = self.client.chat.completions.create(**api_params)

                finish_reason = response.choices[0].finish_reason
                if finish_reason == "length":
                    logger.warning(
                        "Refine planner output truncated",
                        model=self._model,
                        query=original_query,
                    )

                raw = response.choices[0].message.content
                if not raw:
                    logger.warning("Refine planner returned empty response")
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

                # Force-inject selected filters into attributes so they are
                # ALWAYS present even if the LLM omitted or weakened them.
                if "attributes" not in data or not isinstance(data["attributes"], dict):
                    data["attributes"] = {}
                for key, values in selected_filters.items():
                    if key == "modes":
                        # Modes go to the modes list, not attributes
                        if isinstance(values, list):
                            existing_modes = data.get("modes") or []
                            for m in values:
                                if m not in existing_modes:
                                    existing_modes.append(m)
                            data["modes"] = existing_modes
                        continue
                    if key in ("min_price", "max_price", "on_sale_only"):
                        # Price/sale go to top-level fields
                        data[key] = values
                        continue
                    if isinstance(values, list) and values:
                        data["attributes"][key] = values

                plan = SearchPlan(**data)

                # Parse follow_ups
                plan.parsed_follow_ups = self._parse_follow_ups(
                    data.get("follow_ups", [])
                )

                latency_ms = int((time.time() - t_start) * 1000)
                logger.info(
                    "Refine planner generated updated plan",
                    original_query=original_query,
                    intent=plan.intent,
                    algolia_query=plan.algolia_query,
                    semantic_queries=plan.semantic_queries,
                    modes=plan.modes,
                    attributes=plan.attributes,
                    avoid=plan.avoid,
                    follow_ups_count=len(plan.parsed_follow_ups),
                    selected_filters=selected_filters,
                    latency_ms=latency_ms,
                )
                return plan

            except json.JSONDecodeError as e:
                logger.warning("Refine planner returned invalid JSON", error=str(e))
                return None
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "429" in error_str and attempt < self._MAX_RETRIES:
                    wait = self._RETRY_BACKOFF_SECONDS[attempt]
                    logger.info(
                        "Refine planner rate-limited, retrying",
                        attempt=attempt + 1,
                        wait_seconds=wait,
                    )
                    time.sleep(wait)
                    continue
                break

        latency_ms = int((time.time() - t_start) * 1000)
        logger.warning(
            "Refine planner failed, will fall back to skip_planner",
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
        # Step 3b: Stash mode-only exclusion keys so the pipeline
        #          can strip them for SPECIFIC intent (Option A).
        # -----------------------------------------------------------
        if mode_exclusions:
            mode_excl_keys = set()
            for field in mode_exclusions:
                req_field = self._EXCLUDE_FIELD_MAP.get(field)
                if req_field:
                    mode_excl_keys.add(req_field)
            if mode_excl_keys:
                request_updates["_mode_excl_keys"] = list(mode_excl_keys)

        # -----------------------------------------------------------
        # Step 3c: Name exclusions (product-name substring drops)
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
        # Build the semantic queries list.
        # Prefer the new diverse semantic_queries if the LLM provided them.
        # Fall back to the single semantic_query (or algolia_query).
        # -----------------------------------------------------------
        fallback_semantic = plan.semantic_query or plan.algolia_query
        semantic_queries = plan.semantic_queries if plan.semantic_queries else [fallback_semantic]
        # Filter out empty strings
        semantic_queries = [q for q in semantic_queries if q.strip()]
        if not semantic_queries:
            semantic_queries = [fallback_semantic]

        # -----------------------------------------------------------
        # Return the 8-tuple (extended from 7-tuple)
        # -----------------------------------------------------------
        return (
            request_updates,
            expanded_filters,
            exclude_updates,
            [],  # matched_terms — no longer used, kept for API compat
            plan.algolia_query,
            fallback_semantic,  # single semantic query (backward compat)
            plan.intent,
            semantic_queries,  # NEW: diverse semantic queries list
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
