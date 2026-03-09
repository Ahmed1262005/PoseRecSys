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
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from core.logging import get_logger
from config.settings import get_settings
from search.models import QueryIntent, FollowUpQuestion, FollowUpOption
from search.mode_config import expand_modes, get_mode_menu_text
from search.follow_up_translator import translate_follow_up_filters

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
        description="2-6 diverse FashionCLIP queries targeting different visual angles (5-6 for vague, 2-3 for specific, 1 for exact)"
    )

    # Mode tags — high-level intent labels from the mode menu
    modes: List[str] = Field(
        default_factory=list,
        description="Mode tags selected from the mode menu (e.g. ['cover_arms', 'work'])"
    )

    # Positive attribute filters — concrete values the user explicitly stated.
    # Values are List[str] for most keys, but bool for has_pockets,
    # pocket_has_zip, slit_presence.
    attributes: Dict[str, Union[List[str], bool]] = Field(
        default_factory=dict,
        description="Concrete positive filter values from the user's query"
    )

    # Avoid — concrete negative values the user explicitly said NO to
    # (only for things NOT already covered by a mode)
    avoid: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Concrete negative values the user explicitly wants to avoid"
    )

    # Brand detected (if any) — hard filter to this brand
    brand: Optional[str] = Field(
        default=None,
        description="Detected brand name for exact brand search (hard filter)"
    )

    # Brand referenced as a style signal — NOT a purchase target.
    # Set for "like X", "X vibe", "X style", "similar to X" queries.
    # Mutually exclusive with `brand`.
    vibe_brand: Optional[str] = Field(
        default=None,
        description="Brand referenced as style signal ('like X' / 'X vibe'), NOT a hard filter"
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

    # Non-filterable product details from the query (e.g., "zipped pockets",
    # "pearl buttons", "ruched sides").  These are details that have NO
    # structured attribute in our v1.0.0.2 schema.  When present, they are
    # included in semantic_queries so FashionCLIP can attempt visual matching.
    # NOTE: Many details that were previously non-filterable (pockets, backless,
    # lace, slit, etc.) are NOW filterable via v1.0.0.2 attributes — put those
    # in the `attributes` dict instead.
    detail_terms: List[str] = Field(
        default_factory=list,
        description="Truly non-filterable product details (pearl buttons, zipper placement, hardware)"
    )

    # When True, the query's key feature is a non-filterable visual detail
    # that cannot be matched by structured attributes or FashionCLIP.
    # Only set this for details with NO attribute mapping (pearl buttons,
    # specific hardware, zipper placement).  For filterable details
    # (pockets, backless, lace, slit, etc.) use `attributes` instead.
    detail_mode: bool = Field(
        default=False,
        description="True when the query's key feature is a truly non-filterable visual detail"
    )

    # Keywords the LLM expects to find in product names/descriptions of items
    # that have the requested detail.  Used for text-based prefiltering in
    # Algolia to narrow candidates for detail-mode queries.
    prefilter_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords likely in product names/descriptions for items with this detail"
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

5. **Think critically about the occasion.** When a query mentions a specific occasion, apply
   real-world fashion knowledge about what is APPROPRIATE for that event:

   **Garment type defaults by occasion** — override the generic "Tops, Dresses" default:
   - Wedding / wedding guest / gala / cocktail party / formal event → category_l1: ["Dresses"] ONLY.
     These are dress-first occasions. Nobody searches "wedding outfit" expecting tank tops.
     Only include Tops/Outerwear if the user explicitly asks ("top for a wedding", "jacket for a gala").
   - Date night / dinner / evening out → category_l1: ["Dresses"] ONLY (same reasoning).
   - Job interview / business meeting → category_l1: ["Tops", "Dresses", "Bottoms"].
     Separates (blouse + pants) are equally appropriate as a dress.
   - Brunch / casual outing / vacation → category_l1: ["Tops", "Dresses"] (the default is fine).

   **Color/style etiquette by occasion** — add to avoid{{}} when socially appropriate:
   - Wedding guest → avoid: {{"colors": ["White", "Ivory", "Cream"]}}.
     White is reserved for the bride. This is a strong cultural norm.
   - Funeral / memorial → avoid: {{"colors": ["Red", "Hot Pink", "Neon"]}}.
     Bright/loud colors are inappropriate. Prefer dark, muted tones.
   - Job interview → avoid: {{"style_tags": ["Sexy", "Glamorous", "Edgy"]}}.
     Keep it professional and understated.

   **Formality calibration** — match the event's expected dress code:
   - Wedding / gala → formal or semi-formal mode. NOT smart_casual (too casual).
   - Cocktail party → semi-formal or glamorous mode.
   - Date night → smart_casual or glamorous (depends on vibe).
   - Office → work mode. NOT casual.

   The key insight: when someone searches an occasion, they want APPROPRIATE results for that
   specific event. Generic "Tops + Dresses" with no color avoids is lazy. Apply your fashion
   knowledge to every occasion query.

6. **When the user requests a specific attribute value, also avoid the contradicting values.**
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

7. **Distinguish filterable attributes from truly non-filterable details.**
    Many visual details are NOW filterable via our v1.0.0.2 product attribute schema.

    **NOW FILTERABLE — put in `attributes` dict (Section 4b):**
    Pockets (has_pockets), backless/open back (back_openness), lace/ruffle/crochet/embroidery
    (detail_tags), slit (slit_presence, slit_height), cutout/pleated/quilted/mesh (detail_tags),
    bodycon/loose/oversized (body_cling_visual), structured/flowy (structure_level, drape_level),
    wide-leg/flared/skinny (leg_volume_visual), cinched waist (waist_definition_visual),
    cropped (cropped_degree), sheer (sheerness_visual), low-cut/plunging/high-neck (neckline_depth),
    off-shoulder/strapless/one-shoulder (shoulder_coverage), sleeveless/long-sleeve (arm_coverage),
    midriff-baring (midriff_exposure), lined/unlined (lining_status_likely).

    **STILL NON-FILTERABLE — put in semantic_query AND detail_terms:**
    Pearl buttons, specific zipper placement, button style, hardware type, tie-front, drawstring,
    wrap construction, ruching placement (e.g., "ruched sides" — distinct from ruched_bodice tag),
    specific stitching, embellishment placement.

    For non-filterable details:
    - Put them in semantic_query (FashionCLIP understands visual features)
    - ALSO put them in detail_terms (e.g., detail_terms: ["pearl buttons"])
    - Do NOT invent filter values — never guess attribute mappings for concepts we don't track
    - If the user says "no slit", put "closed hemline" in semantic_query (see Principle 8) — do NOT
      add "no slit" to detail_terms (detail_terms is for POSITIVE features to verify, not negations)

8. **semantic_query must be POSITIVE descriptions only — NEVER use negation.**
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

9. **We sell CLOTHING, not underwear.** This is a women's fashion clothing store.
   We do NOT sell intimates, underwear, bras, lingerie, or shapewear.
   When a user mentions undergarments, they are ALWAYS talking about clothing that
   HIDES or works well OVER those undergarments:
   - "doesn't show underwear lines" → thick/structured Bottoms or Dresses (NOT intimates)
   - "doesn't show bra straps" → Tops/Dresses with cover_straps mode
   - "no VPL" (visible panty lines) → Bottoms/Dresses with thicker/structured fabric
   - "works with a strapless bra" → Tops/Dresses with Strapless/Off-Shoulder neckline
   - "hides bra" → Tops/Dresses with opaque mode
    NEVER set category_l1 to "Intimates" — it does not exist in our catalog.

10. **Decompose trend/aesthetic references into visual attributes.**
    Users often search with cultural trend terms instead of product keywords. These terms describe
    a visual AESTHETIC, not a product name. You MUST translate them into concrete visual attributes
    for filters AND rich descriptive semantic_queries. The algolia_query should contain ONLY the
    garment type keyword (e.g., "coat", "dress", "top").

    Common trend/aesthetic terms and their visual decomposition:

    - **"mob wife" / "mob boss wife" / "mafia wife"** → The iconic mob wife aesthetic:
      patterns: ["Animal Print"], materials: ["Faux Leather", "Wool", "Velvet"],
      fit_type: ["Oversized"], colors: ["Black", "Brown", "Cream", "Burgundy"],
      style_tags: ["Glamorous", "Edgy"]. Semantic queries should describe oversized faux fur coats,
      leopard print, dramatic silhouettes, dark luxurious fabrics, gold hardware, bold statement pieces.

    - **"old money" / "quiet luxury" / "stealth wealth"** → Understated elegance:
      materials: ["Wool", "Cotton", "Silk"], colors: ["Beige", "Navy Blue", "White", "Cream", "Gray"],
      style_tags: ["Classic", "Minimalist", "Preppy"], formality: ["Smart Casual", "Business Casual"].
      Semantic queries should describe tailored cashmere, neutral tones, structured blazers, clean lines.

    - **"clean girl" / "that girl"** → Effortless minimal style:
      patterns: ["Solid"], colors: ["White", "Beige", "Black", "Brown"],
      materials: ["Cotton", "Knit", "Jersey"], style_tags: ["Minimalist", "Modern"],
      fit_type: ["Fitted", "Slim"]. Semantic queries should describe sleek basics, neutral tones,
      minimal jewelry-friendly silhouettes.

    - **"coquette" / "feminine" / "balletcore"** → Delicate romantic style:
      patterns: ["Solid", "Floral"], colors: ["Pink", "White", "Cream", "Light Blue"],
      materials: ["Lace", "Satin", "Chiffon", "Silk"], style_tags: ["Romantic"],
      neckline: ["Sweetheart", "Square", "V-Neck"]. Semantic queries should describe bows, ribbons,
      soft pastels, delicate fabrics, ballet-inspired silhouettes.

    - **"dark feminine" / "dark academia"** → Moody intellectual aesthetic:
      colors: ["Black", "Burgundy", "Brown", "Navy Blue", "Olive"],
      materials: ["Wool", "Knit", "Faux Leather", "Velvet"], style_tags: ["Edgy", "Vintage"],
      patterns: ["Plaid", "Solid"]. Semantic queries should describe structured dark fabrics,
      layered academic looks, gothic romantic elements.

    - **"coastal grandmother" / "coastal chic"** → Relaxed elegant seaside style:
      colors: ["White", "Beige", "Light Blue", "Navy Blue", "Cream"],
      materials: ["Linen", "Cotton", "Knit"], style_tags: ["Classic", "Minimalist"],
      fit_type: ["Relaxed", "Regular"]. Semantic queries should describe linen trousers,
      cable-knit sweaters, breezy white shirts, neutral seaside elegance.

    For ANY trend term not listed above, apply the same principle: think about what that
    aesthetic LOOKS like visually, then decompose it into concrete patterns, materials, colors,
    fit_type, and style_tags. Put the rich visual description in semantic_queries.

## SECTION 2: OUTPUT FORMAT

Return a JSON object with these fields:
- intent: "exact" | "specific" | "vague"
- algolia_query: string (product-name keywords for text search)
- semantic_query: string (rich visual description for FashionCLIP — used as fallback)
- semantic_queries: string[] (2-6 DIVERSE FashionCLIP queries — see Section 6b below)
- modes: string[] (mode tags from the menu below)
- attributes: object (positive filter values — keys and allowed values listed below)
- avoid: object (negative filter values — same keys as attributes, for things user said NO to)
- detail_terms: string[] (TRULY non-filterable product details — pearl buttons, hardware, zipper placement. NOT for pockets/backless/lace/slit which are now filterable via Section 4b)
- detail_mode: boolean (true ONLY when the query's distinguishing feature has NO attribute mapping at all — see Section 8 rule 8)
- prefilter_keywords: string[] (keywords likely in product names/descriptions for text prefiltering — only used when detail_mode=true)
- brand: string | null (for EXACT brand purchase — hard filter. See Section 7.)
- vibe_brand: string | null (brand referenced as STYLE SIGNAL, not a purchase target — "like X", "X vibe". See Section 7. Mutually exclusive with brand.)
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

## SECTION 4b: PRODUCT DETAIL ATTRIBUTES (v1.0.0.2 schema)

These attributes capture fine-grained visual details from Gemini Vision extraction.
Use these for detail queries instead of detail_terms/detail_mode. Only the following
keys are valid. Use EXACT values from the allowed lists:

- **back_openness**: ["open", "partial", "closed"]
- **shoulder_coverage**: ["exposed", "strap_only", "off_shoulder", "one_shoulder", "covered"]
- **arm_coverage**: ["none", "short", "half", "three_quarter", "full"]
- **neckline_depth**: ["deep", "low", "moderate", "high"]
- **midriff_exposure**: ["exposed", "partial", "covered"]
- **sheerness_visual**: ["semi_sheer", "opaque"]
- **body_cling_visual**: ["bodycon", "skim", "slim", "regular", "relaxed", "loose"]
- **structure_level**: ["structured", "moderate", "soft", "unstructured"]
- **drape_level**: ["none", "low", "moderate", "high"]
- **cropped_degree**: ["very", "moderate", "slightly", "none"]
- **waist_definition_visual**: ["cinched", "defined", "natural", "undefined"]
- **leg_volume_visual**: ["skinny", "slim", "straight", "wide", "flared", "balloon"]
- **bulk_visual**: ["sleek", "low", "moderate", "bulky"]
- **has_pockets**: true/false (boolean — set true when user asks for pockets)
- **pocket_types**: array from vocabulary ["patch", "welt", "flap", "zip", "cargo", "seam", "kangaroo", "slash", "jetted", "inseam"] (lowercase — filter by specific pocket type in pocket_details.types)
- **pocket_has_zip**: true/false (boolean — set true when user asks for zippered/zipped pockets; checks zip_count > 0 and zip pocket types)
- **slit_presence**: true/false (boolean — set true when user asks for a slit)
- **slit_height**: ["low", "mid", "high"]
- **detail_tags**: array from vocabulary ["lace_trim", "ruffle_detail", "crochet_detail", "distressed_detail", "scalloped_hem", "ruched_bodice", "embroidery_detail", "ribbed_trim", "mesh_panels", "fringe_detail", "raw_hem", "frayed_edge", "quilted_texture", "pleated_detail", "cutout_detail"]
- **lining_status_likely**: ["lined", "partially_lined", "unlined"]

DETAIL ATTRIBUTE MAPPING EXAMPLES:
- "dress with pockets" → attributes: {{"has_pockets": true, "category_l1": ["Dresses"]}}
- "jacket with zipped pockets" → attributes: {{"has_pockets": true, "pocket_has_zip": true, "category_l1": ["Outerwear"]}}
- "cargo pants with pockets" → attributes: {{"has_pockets": true, "pocket_types": ["cargo"], "category_l1": ["Bottoms"]}}
- "blazer with welt pockets" → attributes: {{"has_pockets": true, "pocket_types": ["welt", "jetted"], "category_l1": ["Outerwear"]}}
- "hoodie with kangaroo pocket" → attributes: {{"has_pockets": true, "pocket_types": ["kangaroo"], "category_l1": ["Tops"]}}
- "backless dress" → attributes: {{"back_openness": ["open", "partial"], "category_l1": ["Dresses"]}}
- "lace midi dress" → attributes: {{"detail_tags": ["lace_trim"], "length": ["Midi"], "category_l1": ["Dresses"]}}
- "high slit evening dress" → attributes: {{"slit_presence": true, "slit_height": ["high"], "category_l1": ["Dresses"]}}
- "wide leg pants with pockets" → attributes: {{"leg_volume_visual": ["wide"], "has_pockets": true, "category_l1": ["Bottoms"]}}
- "off shoulder top" → attributes: {{"shoulder_coverage": ["off_shoulder"], "category_l1": ["Tops"]}}
- "sheer blouse" → attributes: {{"sheerness_visual": ["semi_sheer"], "category_l1": ["Tops"]}}
- "structured blazer" → attributes: {{"structure_level": ["structured"], "category_l1": ["Outerwear"]}}
- "flowy maxi dress" → attributes: {{"drape_level": ["high", "moderate"], "body_cling_visual": ["loose", "relaxed"], "length": ["Maxi"], "category_l1": ["Dresses"]}}
- "bodycon mini dress" → attributes: {{"body_cling_visual": ["bodycon"], "length": ["Mini"], "category_l1": ["Dresses"]}}
- "ruffle trim top" → attributes: {{"detail_tags": ["ruffle_detail"], "category_l1": ["Tops"]}}
- "pleated skirt" → attributes: {{"detail_tags": ["pleated_detail"], "category_l1": ["Bottoms"]}}
- "cutout dress" → attributes: {{"detail_tags": ["cutout_detail"], "category_l1": ["Dresses"]}}
- "lined winter coat" → attributes: {{"lining_status_likely": ["lined"], "category_l1": ["Outerwear"]}}

ATTRIBUTE RULES:
- Only include values the user explicitly or strongly implies.
- Use exact values from the allowed lists above.
- For category_l2, include singular AND plural: ["Jacket", "Jackets"].
- category_l1 is ALWAYS a positive attribute, never in avoid.
- For v1.0.0.2 detail attributes: use attributes dict, NOT detail_terms.
  detail_terms is ONLY for truly non-filterable details (pearl buttons, hardware, etc.).

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
- Describe what the garment LOOKS LIKE — never use "not", "no", "without", "non-" (Principle 8)
- Expand with visual details: "floral leaves jacket" → "a jacket with botanical leaf and flower print pattern"
- For coverage queries, describe the covered version: "top that hides arms" → "a top with long opaque sleeves and full arm coverage"
- For "no backless" → "a top with a fully closed high back"
- For "not sheer" → "an opaque solid-fabric top"

## SECTION 6b: DIVERSE SEMANTIC QUERIES (semantic_queries)

A single semantic query pulls visually-similar items that cluster together. To get DIVERSE results,
generate multiple semantic queries that each target a DIFFERENT visual angle of the user's intent.

RULES:
- Each query follows the same positive-only rules as semantic_query (Principle 8)
- Vary by: garment TYPE, STYLE angle, SILHOUETTE, COLOR mood, or FABRIC texture
- Do NOT repeat the same description with synonyms — each must pull a genuinely different cluster
- For exact brand queries: just 1 query is fine (set semantic_queries to [semantic_query])
- For specific queries: 2-3 queries exploring different interpretations
- For vague/open queries: 5-6 queries — each MUST target a DIFFERENT garment category or type.
  If the query is about "outfits" or a general occasion, cover: dresses, tops, bottoms (trousers/shorts/skirts),
  outerwear/layering, and co-ord sets or jumpsuits. Do NOT generate multiple queries that describe
  the same garment type with slightly different wording — that defeats the purpose.
- Keep each query under 77 tokens (FashionCLIP model limit)

Example for "work outfit" (vague — 5 queries, each a DIFFERENT garment):
```json
"semantic_queries": [
  "structured tailored blazer in solid neutral tones, professional office wear",
  "elegant silk button-up blouse with refined collar, polished workwear",
  "fitted knee-length pencil dress in dark fabric, clean professional silhouette",
  "high-waisted tailored wide-leg trousers in black or navy, polished office bottoms",
  "matching coordinated two-piece blazer and trouser set, professional power suiting"
]
```

Example for "vacation outfits for Europe" (vague — 6 queries spanning garment categories):
```json
"semantic_queries": [
  "lightweight floral midi sundress with strappy details, warm-weather resort wear",
  "relaxed linen blouse in soft neutral tones, breezy European summer top",
  "high-waisted wide-leg linen trousers in light earthy tones, comfortable travel bottoms",
  "lightweight knit cardigan or linen blazer for layering, versatile outerwear for cool evenings",
  "tailored high-waisted shorts in neutral cotton, casual chic summer bottoms",
  "matching two-piece co-ord set in breathable fabric, effortless vacation outfit"
]
```

Example for "cute summer dress" (specific — 3 queries, all dresses but different silhouettes):
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

Example for "mob wife coat" (trend/aesthetic — see Principle 10):
```json
"semantic_queries": [
  "oversized luxurious faux fur coat in leopard animal print, bold dramatic glamorous mob wife style",
  "long structured black wool coat with dramatic oversized silhouette, dark luxurious power dressing",
  "oversized shaggy fur coat in dark brown or cream, vintage Hollywood glamour statement outerwear"
]
```

**DETAIL-FOCUSED QUERIES — when the user mentions a non-filterable product detail:**
When the query's KEY distinguishing feature is a specific product detail that has NO structured
filter (e.g., "zipped pockets", "hidden buttons", "ruched seams", "tie waist", "drawstring hem",
"patch pockets", "pearl buttons"), the semantic queries must OVERLAP on that detail instead of
diversifying the garment type. The detail IS the search — varying jacket subtypes misses the point.

Rules for detail queries:
- ALL 3-4 semantic queries MUST mention the specific detail
- Vary the PHRASING and VISUAL ANGLE of the detail, NOT the garment type
- Describe exactly what the detail LOOKS like from different perspectives
- Include close-up descriptions ("visible zipper pulls on front pockets") alongside full-garment
  descriptions ("jacket with zippered cargo pockets")
- The more specific and overlapping the descriptions, the better — products matched by 3+ of
  these queries are far more likely to actually have the detail

Example for "jacket with zipped pockets":
```json
"semantic_queries": [
  "jacket with visible zippered pockets on the front, metal zipper pulls on pocket closures",
  "outerwear with zip-closure cargo pockets, functional zippered pocket flaps",
  "jacket featuring multiple zippered pockets with exposed zipper hardware, utility style",
  "coat with zipper pocket details, secure zip-up pockets on chest and sides"
]
```

Example for "dress with ruched sides":
```json
"semantic_queries": [
  "dress with gathered ruched fabric along the sides, textured side seam draping",
  "bodycon dress with side ruching creating figure-flattering gathered texture",
  "fitted dress featuring ruched side panels, cinched draped fabric at the waist"
]
```

Example for "blouse with pearl buttons":
```json
"semantic_queries": [
  "blouse with decorative pearl buttons down the front placket, elegant button detail",
  "feminine top featuring small pearl button closures, classic pearl-buttoned shirt",
  "button-up blouse with visible round pearl buttons, refined polished detailing"
]
```

If you can only think of one meaningful angle, set semantic_queries to [semantic_query].

**VIBE-BRAND QUERIES — when the user references a brand as a style signal:**

Example for "Like Zara but better quality":
```json
{{
  "intent": "vague",
  "vibe_brand": "Zara",
  "brand": null,
  "min_price": 40,
  "max_price": null,
  "algolia_query": "",
  "semantic_queries": [
    "polished clean-cut basics in neutral tones, tailored minimalist everyday wardrobe essentials",
    "elevated versatile knitwear and structured blazers with refined details, quality fabrics",
    "modern classic blouses and trousers with clean lines, sophisticated muted palette"
  ],
  "modes": ["smart_casual", "quiet_luxury"],
  "attributes": {{"category_l1": ["Tops", "Dresses", "Outerwear"]}},
  "confidence": 0.85
}}
```

Example for "Boho maxi dress like Anthropologie":
```json
{{
  "intent": "specific",
  "vibe_brand": "Anthropologie",
  "brand": null,
  "algolia_query": "boho maxi dress",
  "semantic_queries": [
    "flowing bohemian maxi dress with earthy layered textures and artisan details",
    "printed boho maxi dress with tiered skirt, vintage-inspired romantic patterns",
    "relaxed oversized maxi dress with embroidery or crochet details, festival style"
  ],
  "modes": ["boho"],
  "attributes": {{"category_l1": ["Dresses"], "length": ["Maxi"]}},
  "confidence": 0.9
}}
```

## SECTION 7: PRICE, BRAND, INTENT

- **intent**: "exact" (pure brand search), "specific" (concrete product + attributes), "vague" (mood/aesthetic only)
- **brand**: Detected brand name or null. For EXACT brand purchase — hard filter.
- **vibe_brand**: Brand referenced as a STYLE SIGNAL, not a purchase target. Mutually exclusive with `brand`.
- **max_price**: From "under $50" → 50.0. "Looks expensive" is NOT a price constraint.
- **min_price**: From "over $200" → 200.0
- **on_sale_only**: True for "on sale", "discounted", "clearance". NOT for "affordable" or "cheap".
- **confidence**: 0.0-1.0

### Brand-as-Vibe Queries

When a user references a brand as a STYLE REFERENCE (not wanting to buy that specific brand):

**Detection patterns** — set `vibe_brand`, do NOT set `brand`:
- "like X", "X vibe", "X style", "X aesthetic", "similar to X"
- "alternative to X", "X inspired", "X but ..."
- "better quality than X", "cheaper than X", "more affordable X"
- "X look for less", "dupes for X"

**Exact brand patterns** — set `brand`, do NOT set `vibe_brand`:
- "X dress", "X tops", just the brand name, "buy X", "shop X"
- Misspelled or partial brand names: "zar" → brand: "Zara", "bohoo" → brand: "Boohoo",
  "prin polly" → brand: "Princess Polly", "forver 21" → brand: "Forever 21"
- CRITICAL: A bare brand name (even misspelled/truncated) means the user wants to SHOP
  that brand. Set brand (not vibe_brand) and intent "exact". vibe_brand requires explicit
  comparative language ("like X", "X style", "X vibe", "similar to X", "X but...").

**How to decompose a brand's aesthetic:**
Use your fashion knowledge to identify the brand's style DNA, then translate it into
concrete search attributes:

1. Identify the brand's silhouette language → modes and attributes (e.g., Anthropologie = boho mode, Theory = smart_casual)
2. Identify the brand's color world → inform semantic queries (e.g., Zara = neutral/muted, Free People = earthy/warm)
3. Identify the brand's formality level → appropriate formality mode
4. Identify the brand's typical occasions → relevant occasion modes
5. Generate semantic_queries that capture the brand's visual DNA (e.g., for "like Anthropologie": "flowing bohemian maxi dress earthy layered textures")
6. Correct misspelled/partial brand names to the official spelling in BOTH brand and vibe_brand
   (e.g., "Antropologie" → "Anthropologie", "zar" → "Zara", "bohoo" → "Boohoo").
   Always set algolia_query to the corrected brand name for exact brand queries.

**Quality/price modifiers** — use the brand's price tier as reference:
- "better quality than X" / "elevated X" → same style DNA, set min_price ABOVE the brand's tier.
  Do NOT set max_price — the user wants to go UP in quality, so leave the price open-ended.
  (e.g., "better than Zara [$15-80]" → min_price: 40, max_price: null)
- "cheaper than X" / "affordable X" / "X for less" → same style DNA, set max_price BELOW the brand's tier.
  Do NOT set min_price — the user wants affordable options.
  (e.g., "cheaper Aritzia [$100-400]" → max_price: 80, min_price: null)
- "X dupe" → same style DNA, lower max_price, no brand filter

**Style cluster landscape** (for reasoning about brand tiers and aesthetic DNA):

VALUE ($8-80): basics/capsule | fast-trend/party | teen-casual | mall-trend
MID ($30-200): boho/indie/layered | trendy-feminine/going-out | mainstream | y2k/edgy
MID-PREMIUM ($60-400): modern-classic/tailored | feminine-eco-chic | resort-minimal | artsy-resort
PREMIUM ($100-600): premium-denim | athleisure/wellness | quiet-lux/minimal | contemporary-designer | outdoor
LUXURY ($150-3000): occasion/eventwear | quiet-luxury/investment

Brands sharing aesthetic DNA span tiers. Use your fashion knowledge to identify the right
tier and style family. For example: Zara (value/basics) → COS, Massimo Dutti, & Other Stories
(mid-premium, same clean/versatile DNA); Anthropologie (mid/boho) → Free People (same cluster),
Sezane (feminine eco-chic upgrade); Boohoo (value/party) → Princess Polly (mid, same going-out DNA).

Think of brand DNA as: silhouette language + color palette + formality + construction quality.
"Better quality" means same silhouette language and palette at a higher construction tier.

## SECTION 8: DECISION RULES

1. Use MODES for: coverage/modesty, fit preference, occasion, formality, aesthetic vibe, weather.
2. Use ATTRIBUTES for: concrete positive values (color, material, neckline, category, pattern).
3. Use AVOID for: (a) concrete negative values the user explicitly said NO to, AND (b) contradicting
   values when the user requested a specific attribute (per Principle 6).
4. Never duplicate: if a mode handles it, don't also put it in avoid.
5. Combine freely: modes + attributes + avoid can all be used together.
6. ALWAYS set category_l1. Never leave it empty. Default to ["Tops", "Dresses"] for vague queries.
7. **Filterable details go to attributes, non-filterable go to detail_terms.**
    Check Section 4b for the list of filterable v1.0.0.2 attributes.
    - If a detail IS in Section 4b (pockets, backless, lace, slit, cutout, bodycon, wide-leg, etc.):
      → Put in `attributes` dict using the exact keys/values from Section 4b.
      → Do NOT set detail_mode=true. Do NOT put in detail_terms.
    - If a detail is NOT in Section 4b (pearl buttons, specific hardware,
      tie-front, wrap construction, drawstring, button style, zipper placement):
      → Put in BOTH semantic_query AND detail_terms.
      → Do NOT guess filter mappings — leave attributes empty for non-filterable features.
8. DETAIL MODE: Set detail_mode=true ONLY when the query's KEY distinguishing feature has
    NO attribute mapping in Section 4b. This is rare — most visual details are now filterable.
    Only use for: pearl buttons, specific zipper placement, hardware type, tie-front mechanism,
    drawstring, wrap construction, specific stitching, embellishment placement.

    When detail_mode=true, populate ALL of:
    - detail_terms: the exact visual detail (e.g., ["pearl buttons"])
    - prefilter_keywords: 8-15 keywords likely in product NAMES or DESCRIPTIONS
    - algolia_query: STRIP the detail terms, keep only the base garment type

    Examples of detail_mode=true (truly non-filterable):
      "dress with pearl buttons" →
        detail_mode: true, detail_terms: ["pearl buttons"],
        prefilter_keywords: ["pearl", "button", "buttons", "embellished", "beaded",
                             "detail", "cardigan", "knit", "vintage", "classic", "ornate"],
        algolia_query: "dress"

    Examples of detail_mode=FALSE (use attributes instead):
      "dress with pockets" → detail_mode: false, attributes: {{"has_pockets": true, "category_l1": ["Dresses"]}}
      "backless dress" → detail_mode: false, attributes: {{"back_openness": ["open", "partial"], "category_l1": ["Dresses"]}}
      "lace midi dress" → detail_mode: false, attributes: {{"detail_tags": ["lace_trim"], "length": ["Midi"], "category_l1": ["Dresses"]}}
      "high slit evening dress" → detail_mode: false, attributes: {{"slit_presence": true, "slit_height": ["high"], "category_l1": ["Dresses"]}}
      "cutout dress" → detail_mode: false, attributes: {{"detail_tags": ["cutout_detail"], "category_l1": ["Dresses"]}}
      "wide leg pants" → detail_mode: false, attributes: {{"leg_volume_visual": ["wide"], "category_l1": ["Bottoms"]}}

VOCABULARY TRANSLATION:
- "skirt with shorts underneath" → algolia_query="skort"
- "butter yellow" / "mustard" → colors: ["Yellow"]
- "chocolate brown" / "espresso" → colors: ["Brown"]
- "cherry red" / "crimson" → colors: ["Red", "Burgundy"]
- "navy" → colors: ["Navy Blue"]
- "nude" / "skin tone" → colors: ["Beige", "Taupe"]

## SECTION 9: FOLLOW-UP QUESTIONS (Look-Focused)

Generate 1-3 follow-up questions that materially narrow results by clarifying the LOOK of what
should appear. These follow-ups must be multiple-choice (2-4 options each) and must be answerable
quickly.

### Critical rules
1) Never ask about price, budget, age, or size. We already have these.
2) Do not ask about weather directly in most cases. We know user location and can infer
   season/temperature bands internally.
   - Only ask weather-like questions if the query is explicitly TRAVEL-related
     (e.g., "vacation in Europe", "NYC winter outfits").
   - If you must ask, phrase it as "layering tolerance" / "warmth preference", not "what's the weather?"
3) Optimize for LOOK: what items should look like, not logistics.
   Primary look levers: category (dress vs separates), formality/dress code, vibe/aesthetic,
   coverage/modesty, silhouette/fit, color palette/pattern, fabric vibe, and occasion context.
4) Ask ONLY questions that change the result set meaningfully. Each question must map to
   concrete retrieval filters. Avoid "nice-to-know" questions.
5) Maximum 3 questions. Minimum 0. If the query is already specific enough
   (e.g., "black satin midi dress with long sleeves"), return 0 questions.
6) Questions must be mutually non-redundant.
7) MCQ options must be:
   - 2-4 options
   - Mutually exclusive (minimal overlap)
   - Include "No preference" only when it won't harm retrieval
8) Avoid sensitive or awkward personal questions (body type, religion, etc.).
   Coverage is allowed as a neutral preference.

### Output format

Each follow-up object has:
- "dimension": one of ["category", "setting", "formality", "vibe", "coverage", "silhouette",
  "color_palette", "fabric_vibe", "layering"]
- "question": natural, conversational question text
- "options": 2-4 choices, each with:
  - "label": 2-5 words
  - "filters": a filter PATCH object (see filter schema below)

If the query is specific enough that no follow-ups are needed, return: "follow_ups": []

### Filter patch schema

Use this vocabulary for filter patches. The system translates these into concrete retrieval
filters automatically.

- **product_types**: ["dress", "top", "bottom", "outerwear", "set", "jumpsuit"]
- **dress_code**: "casual" | "smart_casual" | "cocktail" | "formal" | "black_tie"
- **vibe**: ["classic", "romantic", "minimal", "trendy", "edgy", "boho", "sporty",
  "glamorous", "vintage", "preppy", "sexy", "western", "utility", "streetwear"]
- **coverage**:
    neckline: "high" | "mid" | "low" | "strapless_ok"
    sleeves: "long" | "short" | "sleeveless_ok"
    hem: "mini_ok" | "midi" | "maxi"
- **silhouette**:
    fit: "relaxed" | "regular" | "fitted"
    rise: "high" | "mid" (if relevant to bottoms)
- **color_palette**: ["neutrals", "pastels", "jewel_tones", "brights", "dark",
  "warm", "cool", "earth_tones"]
- **pattern**: ["solid", "floral", "stripe", "polka_dot", "abstract", "geometric",
  "animal_print", "plaid", "no_preference"]
- **fabric_vibe**: ["linen", "cotton", "knit", "satin", "silk", "denim", "leather",
  "wool", "velvet", "chiffon", "jersey", "no_preference"]
- **layering**: ["light_layering", "medium_layering", "warm_layering"]

If a user selects an option, the system applies these filters to narrow retrieval.

### Question selection heuristics

Decide which dimensions to ask based on query type:

**A) Occasion/outfit query** (e.g., "brunch outfit", "first date", "wedding guest outfit"):
Priority order:
1) category (dress vs separates vs set vs jumpsuit) — high impact
2) formality OR setting (pick one; setting often implies formality)
3) vibe OR coverage (pick whichever is more ambiguous for the occasion)

**B) Single-item query** (e.g., "a top for brunch", "black blazer"):
Priority order:
1) silhouette/fit (relaxed vs fitted; length)
2) vibe (minimal vs romantic vs edgy)
3) color_palette or fabric_vibe (only if needed)

**C) Travel query** (explicit travel):
Priority order:
1) destination context / activity setting
2) category
3) layering tolerance (only here is it appropriate)

**D) Already-specific query:**
If user already provided: category + key attributes (silhouette/coverage/color/fabric),
return: "follow_ups": []

### Quality guardrails
- Avoid options that our inventory can't satisfy. If uncertain, choose broader options.
- Avoid overly granular questions early (e.g., "square neck vs sweetheart") unless query is
  already narrow and user is close to decision.
- Keep questions short and human.

### Examples

Example 1: query = "Brunch outfit" (Occasion/outfit — Type A)
```json
"follow_ups": [
  {{
    "dimension": "category",
    "question": "What kind of brunch look are you going for?",
    "options": [
      {{"label": "Dress", "filters": {{"product_types": ["dress"]}}}},
      {{"label": "Top + bottom", "filters": {{"product_types": ["top", "bottom"]}}}},
      {{"label": "Matching set", "filters": {{"product_types": ["set"]}}}},
      {{"label": "Jumpsuit", "filters": {{"product_types": ["jumpsuit"]}}}}
    ]
  }},
  {{
    "dimension": "setting",
    "question": "What's the vibe of the place?",
    "options": [
      {{"label": "Casual cafe", "filters": {{"dress_code": "casual"}}}},
      {{"label": "Nice brunch spot", "filters": {{"dress_code": "smart_casual"}}}},
      {{"label": "Birthday / bottomless", "filters": {{"dress_code": "cocktail"}}}},
      {{"label": "Brunch date", "filters": {{"dress_code": "smart_casual", "vibe": ["romantic", "minimal"]}}}}
    ]
  }},
  {{
    "dimension": "vibe",
    "question": "What look do you want to give?",
    "options": [
      {{"label": "Effortless casual", "filters": {{"vibe": ["classic"], "silhouette": {{"fit": "relaxed"}}}}}},
      {{"label": "Polished chic", "filters": {{"vibe": ["minimal", "classic"], "silhouette": {{"fit": "regular"}}}}}},
      {{"label": "Feminine & cute", "filters": {{"vibe": ["romantic"], "pattern": ["floral"]}}}},
      {{"label": "Trendy statement", "filters": {{"vibe": ["trendy"], "color_palette": ["brights", "jewel_tones"]}}}}
    ]
  }}
]
```

Example 2: query = "Help me find an outfit for a first date" (Occasion — Type A)
```json
"follow_ups": [
  {{
    "dimension": "category",
    "question": "Do you want a dress look or separates?",
    "options": [
      {{"label": "Dress", "filters": {{"product_types": ["dress"]}}}},
      {{"label": "Top + bottom", "filters": {{"product_types": ["top", "bottom"]}}}},
      {{"label": "Matching set", "filters": {{"product_types": ["set"]}}}},
      {{"label": "Jumpsuit", "filters": {{"product_types": ["jumpsuit"]}}}}
    ]
  }},
  {{
    "dimension": "setting",
    "question": "What kind of date is it?",
    "options": [
      {{"label": "Coffee / daytime", "filters": {{"dress_code": "casual"}}}},
      {{"label": "Dinner (nice)", "filters": {{"dress_code": "smart_casual"}}}},
      {{"label": "Drinks (night)", "filters": {{"dress_code": "cocktail"}}}},
      {{"label": "Activity date", "filters": {{"dress_code": "casual", "silhouette": {{"fit": "relaxed"}}}}}}
    ]
  }},
  {{
    "dimension": "coverage",
    "question": "How bold do you want the look to feel?",
    "options": [
      {{"label": "More covered", "filters": {{"coverage": {{"neckline": "high", "sleeves": "long", "hem": "midi"}}}}}},
      {{"label": "Balanced", "filters": {{"coverage": {{"neckline": "mid", "sleeves": "short", "hem": "midi"}}}}}},
      {{"label": "More open", "filters": {{"coverage": {{"neckline": "low", "sleeves": "sleeveless_ok", "hem": "mini_ok"}}}}}},
      {{"label": "No preference", "filters": {{}}}}
    ]
  }}
]
```

Example 3: query = "wedding guest outfit" (Occasion — Type A)
```json
"follow_ups": [
  {{
    "dimension": "formality",
    "question": "What's the dress code?",
    "options": [
      {{"label": "Black tie", "filters": {{"dress_code": "black_tie"}}}},
      {{"label": "Formal", "filters": {{"dress_code": "formal"}}}},
      {{"label": "Cocktail", "filters": {{"dress_code": "cocktail"}}}},
      {{"label": "Semi-formal", "filters": {{"dress_code": "smart_casual"}}}}
    ]
  }},
  {{
    "dimension": "category",
    "question": "What kind of outfit do you want?",
    "options": [
      {{"label": "Dress", "filters": {{"product_types": ["dress"]}}}},
      {{"label": "Jumpsuit", "filters": {{"product_types": ["jumpsuit"]}}}},
      {{"label": "Set (top + skirt)", "filters": {{"product_types": ["set"]}}}},
      {{"label": "Suiting", "filters": {{"product_types": ["outerwear", "bottom"], "vibe": ["minimal", "classic"]}}}}
    ]
  }},
  {{
    "dimension": "vibe",
    "question": "Which look fits you most?",
    "options": [
      {{"label": "Classic elegant", "filters": {{"vibe": ["classic", "minimal"], "color_palette": ["neutrals", "dark"]}}}},
      {{"label": "Romantic feminine", "filters": {{"vibe": ["romantic"], "pattern": ["floral"], "color_palette": ["pastels"]}}}},
      {{"label": "Modern minimal", "filters": {{"vibe": ["minimal"], "pattern": ["solid"]}}}},
      {{"label": "Bold statement", "filters": {{"vibe": ["trendy"], "color_palette": ["jewel_tones", "brights"]}}}}
    ]
  }}
]
```

Example 4: query = "Vacation outfits for Europe" (Travel — Type C)
```json
"follow_ups": [
  {{
    "dimension": "setting",
    "question": "What's the main plan?",
    "options": [
      {{"label": "City walking", "filters": {{"dress_code": "casual", "silhouette": {{"fit": "relaxed"}}}}}},
      {{"label": "Nice dinners", "filters": {{"dress_code": "smart_casual"}}}},
      {{"label": "Beach / coastal", "filters": {{"dress_code": "casual", "fabric_vibe": ["linen", "cotton"]}}}},
      {{"label": "Mixed itinerary", "filters": {{"dress_code": "smart_casual"}}}}
    ]
  }},
  {{
    "dimension": "category",
    "question": "What do you want more of?",
    "options": [
      {{"label": "Dresses", "filters": {{"product_types": ["dress"]}}}},
      {{"label": "Sets", "filters": {{"product_types": ["set"]}}}},
      {{"label": "Tops + bottoms", "filters": {{"product_types": ["top", "bottom"]}}}},
      {{"label": "Outer layers", "filters": {{"product_types": ["outerwear"]}}}}
    ]
  }},
  {{
    "dimension": "layering",
    "question": "How much layering do you want?",
    "options": [
      {{"label": "Light layers", "filters": {{"layering": ["light_layering"]}}}},
      {{"label": "Medium layers", "filters": {{"layering": ["medium_layering"]}}}},
      {{"label": "Warm layers", "filters": {{"layering": ["warm_layering"]}}}},
      {{"label": "No preference", "filters": {{}}}}
    ]
  }}
]
```

Example 5: query = "black satin midi dress with long sleeves" (Already specific — Type D)
```json
"follow_ups": []
```

### Negative examples (DO NOT DO THIS)
- "What's your budget?" → we already have it
- "What size are you?" → we already have it
- "How old are you?" → we already have it, and it's sensitive
- "What's the weather like?" → we infer from location; only ask layering for travel
- Asking 4+ questions
- Options that overlap (e.g., "cute" vs "pretty")
- Options with no filter mapping (e.g., "surprise me" with no effect)

## SECTION 10: PERSONALIZED FOLLOW-UPS (when user context is provided)

When a "User context:" prefix is present, personalize follow-ups as follows:

1. **Reorder options** so the most likely choice for this user is FIRST.
   - Prefer behavior-based evidence (clicks/saves/history) over demographics.
   - If no strong evidence exists, use "most popular first" defaults.

2. **Coverage ordering:**
   - If modesty=covered, put coverage-friendly options first.
   - If modesty=balanced, keep default ordering.

3. **Match vibe/style ordering** to user preferences.
   - If user commonly prefers minimalist, reorder vibe options to put "Minimal" first.
   - If user prefers boho, reorder to put "Bohemian" / "Romantic" first.

4. **Match fit ordering** to user preferences.
   - If user prefers oversized/relaxed styles, reorder fit options accordingly.
   - If user prefers fitted, put "Fitted" / "Slim" first.

5. **Do NOT remove options** — only reorder them.
6. **Skip follow-up questions** already answered by user context or the query.
7. **If no user context is provided**, generate follow-ups in the default order (most popular first).

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
   FashionCLIP can match. Keep the SAME number of queries as the initial search would have
   generated — the filters narrow WHAT to search for; the diverse queries vary HOW it looks
   (different silhouettes, fabrics, color moods, or styling). Do NOT collapse to fewer queries
   just because the user picked a filter — that's when variety matters most.
   Example: query="tops" + fitted + edgy → semantic_queries: [
     "fitted black faux leather crop top with edgy streetwear style",
     "slim fitted ribbed tank top in dark solid color, modern edge",
     "structured fitted bodysuit in sleek fabric, edgy minimalist",
     "fitted cropped denim jacket with moto edge, hardware details",
     "slim stretch mesh top with bold graphic, dark edgy streetwear"
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
    """LLM-based query planner — supports OpenAI and Gemini (via OpenAI-compat endpoint)."""

    # Gemini OpenAI-compatible endpoint
    _GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(self):
        self._client = None
        self._client_lock = threading.Lock()
        settings = get_settings()
        self._provider = getattr(settings, "query_planner_provider", "openai").lower()
        self._model = settings.query_planner_model
        self._timeout = settings.query_planner_timeout_seconds

        # Pick API key based on provider
        if self._provider == "gemini":
            self._api_key = getattr(settings, "google_api_key", "")
            # Default model for gemini if user didn't override
            if self._model == "gpt-4.1-mini":
                self._model = "gemini-2.0-flash"
        else:
            self._api_key = settings.openai_api_key

        self._enabled = settings.query_planner_enabled and bool(self._api_key)

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def client(self):
        """Lazy-load OpenAI client (works for both OpenAI and Gemini endpoints)."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    from openai import OpenAI
                    client_kwargs = {
                        "api_key": self._api_key,
                        "timeout": self._timeout,
                    }
                    if self._provider == "gemini":
                        client_kwargs["base_url"] = self._GEMINI_BASE_URL
                    self._client = OpenAI(**client_kwargs)
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
                        # Translate LLM's look-focused keys to native pipeline keys
                        native_filters = translate_follow_up_filters(
                            raw_opt.get("filters", {})
                        )
                        options.append(FollowUpOption(
                            label=raw_opt["label"],
                            filters=native_filters,
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
            logger.debug("Query planner disabled (no API key or feature flag off)",
                         provider=self._provider, model=self._model)
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
                    api_params["temperature"] = 0.15
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
                    provider=self._provider,
                    model=self._model,
                    query=query,
                    refinement=is_refinement,
                    intent=plan.intent,
                    algolia_query=plan.algolia_query,
                    semantic_query=plan.semantic_query,
                    modes=plan.modes,
                    attributes=plan.attributes,
                    avoid=plan.avoid,
                    detail_terms=plan.detail_terms,
                    detail_mode=plan.detail_mode,
                    prefilter_keywords=plan.prefilter_keywords,
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
            provider=self._provider,
            model=self._model,
            error=str(last_error),
            latency_ms=latency_ms,
        )
        return None

    # -----------------------------------------------------------------
    # Refinement planner — second LLM call after follow-up selections
    # -----------------------------------------------------------------

    # Map filter keys → follow-up dimension names for the LLM
    _FILTER_KEY_TO_DIMENSION = {
        # Native pipeline keys
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
        # LLM look-focused prompt keys (pre-translation)
        "product_types": "garment_type",
        "dress_code": "formality",
        "vibe": "vibe",
        "coverage": "coverage",
        "color_palette": "color",
        "pattern": "vibe",
        "fabric_vibe": "vibe",
        "layering": "coverage",
        "setting": "occasion",
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
                    api_params["temperature"] = 0.15
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
                    provider=self._provider,
                    model=self._model,
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
            provider=self._provider,
            model=self._model,
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

    # Valid filter field names (for validation).
    # Includes both classic Algolia filters and v1.0.0.2 attribute keys.
    _VALID_FILTER_FIELDS = {
        "category_l1", "category_l2", "patterns", "colors", "formality",
        "occasions", "fit_type", "neckline", "sleeve_type", "length",
        "rise", "materials", "silhouette", "seasons", "style_tags",
        "brands", "categories",
    }

    # v1.0.0.2 detail attribute keys — these pass through to
    # plan_to_attribute_filters() and are NOT applied as Algolia filters.
    _V1002_ATTRIBUTE_KEYS = {
        "back_openness", "shoulder_coverage", "arm_coverage",
        "neckline_depth", "midriff_exposure", "sheerness_visual",
        "body_cling_visual", "structure_level", "drape_level",
        "cropped_degree", "waist_definition_visual", "leg_volume_visual",
        "bulk_visual", "has_pockets", "pocket_types", "pocket_has_zip",
        "slit_presence", "slit_height",
        "detail_tags", "lining_status_likely",
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

        # Merge in LLM attributes (correct typos first).
        # v1.0.0.2 detail attribute keys (has_pockets, back_openness, etc.)
        # are skipped here — they stay in plan.attributes and are consumed
        # by plan_to_attribute_filters() in the hybrid search pipeline.
        for raw_field, values in plan.attributes.items():
            field = self._TYPO_CORRECTIONS.get(raw_field, raw_field)
            if field != raw_field:
                logger.info("Corrected LLM typo in attributes", original=raw_field, corrected=field)
            # Skip v1.0.0.2 keys — they are handled by attribute_search.py
            if field in self._V1002_ATTRIBUTE_KEYS:
                continue
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
        if plan.vibe_brand:
            # Vibe brand → do NOT set hard brand filter.
            # Pass through as internal key for post-planner cluster boosting.
            request_updates["_vibe_brand"] = plan.vibe_brand
            logger.info(
                "Vibe brand detected — skipping hard brand filter",
                vibe_brand=plan.vibe_brand,
            )
        elif plan.brand and "brands" not in request_updates:
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
