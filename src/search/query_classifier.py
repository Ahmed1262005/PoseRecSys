"""
Query Intent Classifier.

Classifies search queries into three categories to determine the search strategy:
- exact: Brand/product name lookup -> Algolia dominates
- specific: Category + attribute ("blue midi dress") -> balanced
- vague: Style/occasion/vibe ("quiet luxury") -> FashionCLIP dominates
"""

import re
from typing import Dict, List, Optional, Set, Tuple

from core.logging import get_logger
from search.models import QueryIntent

logger = get_logger(__name__)


# ============================================================================
# Known brand names (loaded from DB at startup, fallback set here)
# ============================================================================

_BRAND_NAMES: Optional[Set[str]] = None

# Mapping from lowercase brand -> original casing (populated by load_brands)
_BRAND_ORIGINALS: Dict[str, str] = {}

# Fallback set of common fashion brands (lowercase).
# The full set is loaded from the DB via load_brands().
_FALLBACK_BRANDS: Set[str] = {
    "boohoo", "alo yoga", "alo", "missguided", "forever 21", "forever21",
    "free people", "reformation", "nasty gal", "prettylittlething",
    "asos", "zara", "h&m", "mango", "topshop", "urban outfitters",
    "anthropologie", "lululemon", "nike", "adidas", "gap", "uniqlo",
    "cos", "arket", "&other stories", "other stories", "everlane",
    "abercrombie", "abercrombie & fitch", "hollister", "american eagle",
    "princess polly", "showpo", "hello molly", "meshki", "white fox",
    "house of cb", "oh polly", "plt", "na-kd", "nakd", "shein",
    "revolve", "nordstrom", "net-a-porter", "ssense", "farfetch",
}

# Pre-compiled regex for brand matching (built on first use or after load_brands)
_BRAND_PATTERNS: Optional[List[Tuple[str, re.Pattern]]] = None


def _build_brand_patterns(brands: Set[str]) -> List[Tuple[str, re.Pattern]]:
    """Build regex patterns for brand matching with word boundaries.

    Sorts brands longest-first so "alo yoga" matches before "alo".
    Uses word boundaries to prevent "cos" matching "cosplay".
    Special handling for brands with & (e.g. "h&m", "ba&sh").
    """
    sorted_brands = sorted(brands, key=len, reverse=True)
    patterns = []
    for brand in sorted_brands:
        # Escape regex special chars but keep the brand readable
        escaped = re.escape(brand)
        # Use word boundaries; for brands starting/ending with &, use lookaround
        pattern = re.compile(r'(?:^|\b|(?<=\s))' + escaped + r'(?:\b|(?=\s)|$)', re.IGNORECASE)
        patterns.append((brand, pattern))
    return patterns


def _get_brand_patterns() -> List[Tuple[str, re.Pattern]]:
    """Get or build brand matching patterns."""
    global _BRAND_PATTERNS
    if _BRAND_PATTERNS is None:
        _BRAND_PATTERNS = _build_brand_patterns(_get_brands())
    return _BRAND_PATTERNS


def load_brands(supabase_client=None) -> Set[str]:
    """
    Load brand names from Algolia facets (fast) or the products table.

    Uses Algolia search_for_facet_values (up to 500 brands) and merges
    with the fallback set. This avoids scanning the entire products table.
    Call this once at startup to populate the brand set.
    Falls back to _FALLBACK_BRANDS on error.
    """
    global _BRAND_NAMES, _BRAND_PATTERNS, _BRAND_ORIGINALS
    brands = set(_FALLBACK_BRANDS)  # Start with fallback
    try:
        # Primary: Load from Algolia facets.
        # Algolia caps maxFacetHits at 100, so we query multiple prefixes
        # to cover the full brand list (131 brands).
        from search.algolia_client import get_algolia_client
        client = get_algolia_client()
        prefixes = [""] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
        for prefix in prefixes:
            try:
                result = client.search_for_facet_values("brand", prefix, max_facet_hits=100)
                for hit in result.get("facetHits", []):
                    b = hit.get("value")
                    if b:
                        lower = b.lower().strip()
                        brands.add(lower)
                        _BRAND_ORIGINALS[lower] = b.strip()
            except Exception:
                pass  # individual prefix failure is fine
    except Exception as e:
        logger.warning("Failed to load brands from Algolia facets", error=str(e))

    if len(brands) > len(_FALLBACK_BRANDS):
        _BRAND_NAMES = brands
        _BRAND_PATTERNS = _build_brand_patterns(brands)
        return brands

    # Fallback: Try Supabase if Algolia failed
    if supabase_client:
        try:
            last_brand = ""
            for _ in range(10):  # Max 10 cursor pages
                q = (
                    supabase_client.table("products")
                    .select("brand")
                    .order("brand")
                    .limit(1000)
                )
                if last_brand:
                    q = q.gt("brand", last_brand)
                result = q.execute()
                if not result.data:
                    break
                for row in result.data:
                    b = row.get("brand")
                    if b:
                        lower = b.lower().strip()
                        brands.add(lower)
                        _BRAND_ORIGINALS[lower] = b.strip()
                        last_brand = b
                if len(result.data) < 1000:
                    break
        except Exception as e:
            logger.warning("Failed to load brands from Supabase", error=str(e))

    _BRAND_NAMES = brands
    _BRAND_PATTERNS = _build_brand_patterns(brands)
    return brands


def _get_brands() -> Set[str]:
    """Get the current brand set."""
    if _BRAND_NAMES is not None:
        return _BRAND_NAMES
    return _FALLBACK_BRANDS


# ============================================================================
# Word-boundary keyword matching
# ============================================================================

def _build_keyword_pattern(keywords: Set[str]) -> re.Pattern:
    """Build a single compiled regex that matches any keyword at word boundaries.

    Multi-word phrases (e.g. "tank top", "crop top") are matched as literal
    phrases. Single words use \\b word boundaries to avoid substring matches
    (e.g. "tan" won't match "important").

    Sorts keywords longest-first so multi-word phrases match before their
    component words.
    """
    sorted_kws = sorted(keywords, key=len, reverse=True)
    parts = []
    for kw in sorted_kws:
        escaped = re.escape(kw)
        parts.append(r'\b' + escaped + r'\b')
    return re.compile('|'.join(parts), re.IGNORECASE)


# ============================================================================
# Category / attribute keywords
# ============================================================================

_CATEGORY_KEYWORDS = {
    "dress", "dresses", "top", "tops", "blouse", "blouses", "shirt", "shirts",
    "sweater", "sweaters", "pants", "trousers", "jeans", "skirt", "skirts",
    "shorts", "jacket", "jackets", "coat", "coats", "blazer", "blazers",
    "cardigan", "cardigans", "hoodie", "hoodies", "bodysuit", "bodysuits",
    "romper", "rompers", "jumpsuit", "jumpsuits", "leggings", "tights",
    "sweatshirt", "sweatshirts", "tank", "tank top", "camisole", "crop top",
    "vest", "parka", "trench", "windbreaker", "kimono",
    # Additional missing keywords
    "bralette", "bralettes", "corset", "corsets", "tunic", "tunics",
    "pullover", "pullovers", "henley", "polo", "polos", "joggers",
    "culottes", "poncho", "cape", "wrap", "wraps",
}

_ATTRIBUTE_KEYWORDS = {
    # Colors
    "black", "white", "red", "blue", "green", "pink", "yellow", "purple",
    "navy", "beige", "tan", "brown", "grey", "gray", "ivory", "cream",
    "burgundy", "wine", "orange", "coral", "teal", "olive", "mauve",
    # Patterns
    "striped", "floral", "plaid", "checkered", "polka dot", "animal print",
    "leopard", "geometric", "abstract", "solid", "tie dye", "camo",
    # Fit / length
    "midi", "mini", "maxi", "cropped", "oversized", "fitted", "slim",
    "relaxed", "skinny", "wide leg", "straight", "high waisted", "low rise",
    # Materials
    "cotton", "linen", "silk", "satin", "denim", "leather", "wool",
    "cashmere", "velvet", "chiffon", "lace", "knit", "jersey",
    # Necklines / sleeves
    "v-neck", "crew neck", "turtleneck", "off shoulder", "halter",
    "strapless", "sleeveless", "long sleeve", "puff sleeve",
    # Formality levels (searchable Algolia attribute)
    "formal", "semi-formal", "business casual", "smart casual", "cocktail",
}

_VAGUE_KEYWORDS = {
    # Pure style/vibe/aesthetic terms that have no direct Algolia facet
    "elegant", "cozy", "trendy", "chic", "minimalist", "quiet luxury",
    "old money", "clean girl", "streetwear", "preppy", "cottagecore",
    "dark academia", "y2k", "coastal", "mob wife", "coquette",
    "athleisure",
    "summer vibes", "fall outfit", "winter layers", "spring look",
    "something for", "outfit for", "what to wear",
    # Note: occasion terms (date night, office, party, etc.), formality terms
    # (formal, casual, etc.), and season terms (summer, winter) are handled
    # by extract_attributes() and mapped to Algolia facet filters.
}

# Pre-compiled patterns for keyword matching (word-boundary safe)
_CATEGORY_PATTERN = _build_keyword_pattern(_CATEGORY_KEYWORDS)
_ATTRIBUTE_PATTERN = _build_keyword_pattern(_ATTRIBUTE_KEYWORDS)
_VAGUE_PATTERN = _build_keyword_pattern(_VAGUE_KEYWORDS)


# ============================================================================
# Classifier
# ============================================================================

class QueryClassifier:
    """
    Classify search queries to determine the optimal search strategy.

    Uses word-boundary regex matching to avoid substring false positives
    (e.g. "tan" won't match "important", "cos" won't match "cosplay").
    """

    @staticmethod
    def extract_brand(query: str) -> Optional[str]:
        """Extract a brand name from the query, if present.

        Returns the brand name. Prefers original casing from Algolia
        when available, falls back to lowercase.
        """
        q = query.strip()
        for brand_lower, pattern in _get_brand_patterns():
            if pattern.search(q):
                # Return original casing if available, else lowercase
                return _BRAND_ORIGINALS.get(brand_lower, brand_lower)
        return None

    @staticmethod
    def _find_brand(q: str) -> Optional[Tuple[str, bool]]:
        """Find a brand in the query. Returns (brand_lower, is_exact) or None.

        is_exact is True if the query is *just* the brand name.
        """
        q_lower = q.lower().strip()
        for brand_lower, pattern in _get_brand_patterns():
            if pattern.search(q):
                # Check if query is just the brand
                remainder = pattern.sub("", q_lower).strip()
                is_exact = (remainder == "")
                return (brand_lower, is_exact)
        return None

    @staticmethod
    def classify(query: str) -> QueryIntent:
        """
        Classify a search query.

        Returns:
            QueryIntent.EXACT    - Brand/product name -> Algolia dominates
            QueryIntent.SPECIFIC - Category + attribute -> balanced merge
            QueryIntent.VAGUE    - Style/vibe -> FashionCLIP dominates
        """
        q = query.lower().strip()

        # 1. Check for brand names (word-boundary safe)
        brand_match = QueryClassifier._find_brand(q)
        if brand_match:
            brand_lower, is_exact = brand_match
            if is_exact:
                return QueryIntent.EXACT
            return QueryIntent.SPECIFIC

        # 2. Detect category, attribute, and vague keywords (word-boundary safe)
        has_category = bool(_CATEGORY_PATTERN.search(q))
        has_attribute = bool(_ATTRIBUTE_PATTERN.search(q))
        has_vague = bool(_VAGUE_PATTERN.search(q))

        # 3. Category + anything = SPECIFIC (even if vague keywords present)
        #    e.g. "office dress", "party dress", "casual top" -> SPECIFIC
        #    The user has a concrete category in mind.
        if has_category:
            return QueryIntent.SPECIFIC

        # 4. Attribute-only queries (color, material, fit) = SPECIFIC
        if has_attribute:
            return QueryIntent.SPECIFIC

        # 5. Purely vague / style-intent queries (no category keyword)
        #    e.g. "quiet luxury", "date night outfit", "dark academia"
        if has_vague:
            return QueryIntent.VAGUE

        # 6. Very short queries (1-2 words) that didn't match above
        word_count = len(q.split())
        if word_count <= 2:
            # Could be a product name or generic term
            return QueryIntent.SPECIFIC

        # 7. Longer queries without clear category terms are likely vague
        if word_count >= 4:
            return QueryIntent.VAGUE

        return QueryIntent.SPECIFIC

    # Default RRF weights per intent. Override via environment variables:
    #   RRF_WEIGHT_EXACT_ALGOLIA=0.85
    #   RRF_WEIGHT_SPECIFIC_ALGOLIA=0.60
    #   RRF_WEIGHT_VAGUE_ALGOLIA=0.35
    _rrf_weights: Optional[Dict[str, Tuple[float, float]]] = None

    @classmethod
    def _get_rrf_weights(cls) -> Dict[str, Tuple[float, float]]:
        """Get RRF weights (algolia, semantic) per intent. Loaded once from env."""
        if cls._rrf_weights is None:
            import os
            def _load(intent_name: str, default_algolia: float) -> Tuple[float, float]:
                env_key = f"RRF_WEIGHT_{intent_name.upper()}_ALGOLIA"
                try:
                    algolia_w = float(os.getenv(env_key, str(default_algolia)))
                except ValueError:
                    algolia_w = default_algolia
                return (algolia_w, 1.0 - algolia_w)

            cls._rrf_weights = {
                "exact": _load("exact", 0.85),
                "specific": _load("specific", 0.60),
                "vague": _load("vague", 0.35),
            }
        return cls._rrf_weights

    @staticmethod
    def extract_attributes(query: str) -> Tuple[Dict[str, list], List[str]]:
        """
        Extract structured attribute filters from a natural language query.

        Maps query terms to Algolia facet values so the search can apply
        them as hard filters instead of relying on keyword matching alone.

        Returns:
            Tuple of:
            - Dict with keys matching HybridSearchRequest filter fields.
              Only populated keys are included.
            - List of matched query terms that were converted to filters.
              These can be stripped from the Algolia text query since they're
              now handled by facet filters (e.g. "formal" in "formal shirt"
              becomes a formality filter, so Algolia only needs to search "shirt").
        """
        q = query.lower().strip()
        filters: Dict[str, list] = {}
        matched_terms: List[str] = []

        # --- Formality ---
        _FORMALITY_MAP = {
            # Cast a wider net for formality to avoid 0-result dead ends.
            # E.g. "formal shirt" should find Business Casual/Semi-Formal
            # shirts, not return empty because 0 shirts are tagged "Formal".
            "formal": ["Formal", "Semi-Formal", "Business Casual"],
            "semi-formal": ["Semi-Formal"],
            "semi formal": ["Semi-Formal"],
            "business casual": ["Business Casual"],
            "smart casual": ["Smart Casual"],
            "casual": ["Casual"],
            "cocktail": ["Formal", "Semi-Formal"],
            "dressy": ["Semi-Formal", "Formal"],
            "black tie": ["Formal"],
            "professional": ["Business Casual", "Smart Casual"],
        }
        for term, vals in _FORMALITY_MAP.items():
            if re.search(r'\b' + re.escape(term) + r'\b', q):
                filters["formality"] = vals
                matched_terms.append(term)
                break

        # --- Occasions ---
        _OCCASION_MAP = {
            "date night": ["Date Night"],
            "date": ["Date Night"],
            "party": ["Party"],
            "brunch": ["Brunch"],
            "office": ["Office", "Work"],
            "work": ["Work", "Office"],
            "wedding": ["Wedding Guest"],
            "wedding guest": ["Wedding Guest"],
            "vacation": ["Vacation"],
            "holiday": ["Vacation"],
            "beach": ["Vacation", "Beach"],
            "workout": ["Workout"],
            "gym": ["Workout"],
            "lounge": ["Lounging"],
            "lounging": ["Lounging"],
            "weekend": ["Weekend", "Everyday"],
            "everyday": ["Everyday"],
            "night out": ["Night Out", "Party"],
            "evening": ["Date Night", "Party"],
            "festival": ["Party", "Vacation"],
        }
        for term, vals in sorted(_OCCASION_MAP.items(), key=lambda x: -len(x[0])):
            if re.search(r'\b' + re.escape(term) + r'\b', q):
                filters.setdefault("occasions", []).extend(vals)
                filters["occasions"] = list(dict.fromkeys(filters["occasions"]))
                matched_terms.append(term)
                break

        # --- Colors ---
        _COLOR_MAP = {
            "black": "Black", "white": "White", "red": "Red", "blue": "Blue",
            "navy": "Navy Blue", "navy blue": "Navy Blue",
            "light blue": "Light Blue", "baby blue": "Light Blue",
            "green": "Green", "olive": "Olive", "olive green": "Olive Green",
            "pink": "Pink", "hot pink": "Pink", "blush": "Pink",
            "yellow": "Yellow", "mustard": "Yellow",
            "purple": "Purple", "lavender": "Purple",
            "orange": "Orange", "coral": "Orange",
            "brown": "Brown", "chocolate": "Chocolate Brown",
            "beige": "Beige", "tan": "Beige", "khaki": "Beige",
            "cream": "Cream", "ivory": "Off White", "off white": "Off White",
            "grey": "Gray", "gray": "Gray", "charcoal": "Gray",
            "burgundy": "Burgundy", "wine": "Burgundy", "maroon": "Burgundy",
            "taupe": "Taupe",
            "teal": "Green",
            "mauve": "Pink",
        }
        for term, val in sorted(_COLOR_MAP.items(), key=lambda x: -len(x[0])):
            if re.search(r'\b' + re.escape(term) + r'\b', q):
                filters["colors"] = [val]
                matched_terms.append(term)
                break

        # --- Patterns ---
        _PATTERN_MAP = {
            "floral": "Floral", "flower": "Floral",
            "striped": "Striped", "stripe": "Striped", "stripes": "Striped",
            "plaid": "Plaid", "checkered": "Plaid", "check": "Plaid",
            "polka dot": "Polka Dot", "polka": "Polka Dot", "dotted": "Polka Dot",
            "animal print": "Animal Print", "leopard": "Animal Print",
            "snake": "Animal Print", "zebra": "Animal Print",
            "abstract": "Abstract",
            "geometric": "Geometric",
            "solid": "Solid",
            "tie dye": "Tie Dye", "tie-dye": "Tie Dye",
            "camo": "Camo", "camouflage": "Camo",
            "colorblock": "Colorblock", "color block": "Colorblock",
        }
        for term, val in sorted(_PATTERN_MAP.items(), key=lambda x: -len(x[0])):
            if re.search(r'\b' + re.escape(term) + r'\b', q):
                filters["patterns"] = [val]
                matched_terms.append(term)
                break

        # --- Fit ---
        _FIT_MAP = {
            "slim": "Slim", "slim fit": "Slim",
            "fitted": "Fitted", "bodycon": "Fitted",
            "relaxed": "Relaxed", "relaxed fit": "Relaxed",
            "oversized": "Oversized", "oversize": "Oversized",
            "loose": "Loose", "baggy": "Loose",
            "regular fit": "Regular",
        }
        for term, val in sorted(_FIT_MAP.items(), key=lambda x: -len(x[0])):
            if re.search(r'\b' + re.escape(term) + r'\b', q):
                filters["fit_type"] = [val]
                matched_terms.append(term)
                break

        # --- Neckline ---
        _NECKLINE_MAP = {
            "v-neck": "V-Neck", "v neck": "V-Neck", "vneck": "V-Neck",
            "crew neck": "Crew", "crewneck": "Crew", "crew": "Crew",
            "turtleneck": "Turtleneck", "turtle neck": "Turtleneck",
            "mock neck": "Mock", "mock": "Mock",
            "off shoulder": "Off-Shoulder", "off-shoulder": "Off-Shoulder",
            "strapless": "Strapless",
            "halter": "Halter",
            "scoop": "Scoop", "scoop neck": "Scoop",
            "square neck": "Square", "square": "Square",
            "sweetheart": "Sweetheart",
            "cowl": "Cowl", "cowl neck": "Cowl",
            "boat neck": "Boat", "boatneck": "Boat",
            "one shoulder": "One Shoulder",
            "collared": "Collared",
            "hooded": "Hooded",
        }
        for term, val in sorted(_NECKLINE_MAP.items(), key=lambda x: -len(x[0])):
            if re.search(r'\b' + re.escape(term) + r'\b', q):
                filters["neckline"] = [val]
                matched_terms.append(term)
                break

        # --- Sleeve ---
        _SLEEVE_MAP = {
            "sleeveless": "Sleeveless",
            "long sleeve": "Long", "long-sleeve": "Long",
            "short sleeve": "Short", "short-sleeve": "Short",
            "cap sleeve": "Cap",
            "puff sleeve": "Puff", "puff": "Puff",
            "3/4 sleeve": "3/4", "three quarter": "3/4",
            "flutter sleeve": "Flutter",
        }
        for term, val in sorted(_SLEEVE_MAP.items(), key=lambda x: -len(x[0])):
            if re.search(r'\b' + re.escape(term) + r'\b', q):
                filters["sleeve_type"] = [val]
                matched_terms.append(term)
                break

        # --- Length ---
        _LENGTH_MAP = {
            "mini": "Mini",
            "midi": "Midi",
            "maxi": "Maxi",
            "cropped": "Cropped", "crop": "Cropped",
            "floor length": "Floor-length", "floor-length": "Floor-length",
            "ankle": "Ankle",
        }
        for term, val in sorted(_LENGTH_MAP.items(), key=lambda x: -len(x[0])):
            if re.search(r'\b' + re.escape(term) + r'\b', q):
                filters["length"] = [val]
                matched_terms.append(term)
                break

        # --- Rise ---
        _RISE_MAP = {
            "high waisted": "High", "high-waisted": "High", "high rise": "High",
            "high-rise": "High",
            "mid rise": "Mid", "mid-rise": "Mid",
            "low rise": "Low", "low-rise": "Low",
        }
        for term, val in sorted(_RISE_MAP.items(), key=lambda x: -len(x[0])):
            if re.search(r'\b' + re.escape(term) + r'\b', q):
                filters["rise"] = [val]
                matched_terms.append(term)
                break

        # --- Material ---
        _MATERIAL_MAP = {
            "cotton": "Cotton", "cotton blend": "Cotton Blend",
            "linen": "Linen", "linen blend": "Linen Blend",
            "silk": "Silk", "silky": "Silk",
            "satin": "Satin",
            "denim": "Denim",
            "leather": "Faux Leather", "faux leather": "Faux Leather",
            "wool": "Wool", "cashmere": "Wool",
            "velvet": "Velvet",
            "chiffon": "Chiffon",
            "lace": "Lace", "lacy": "Lace",
            "mesh": "Mesh", "sheer": "Mesh",
            "knit": "Knit", "knitted": "Knit",
            "jersey": "Jersey",
            "fleece": "Fleece",
        }
        for term, val in sorted(_MATERIAL_MAP.items(), key=lambda x: -len(x[0])):
            if re.search(r'\b' + re.escape(term) + r'\b', q):
                filters["materials"] = [val]
                matched_terms.append(term)
                break

        # --- Silhouette ---
        _SILHOUETTE_MAP = {
            "a-line": "A-Line", "a line": "A-Line",
            "bodycon": "Bodycon",
            "flared": "Flared", "flare": "Flared",
            "straight": "Straight",
            "wide leg": "Wide Leg", "wide-leg": "Wide Leg",
        }
        for term, val in sorted(_SILHOUETTE_MAP.items(), key=lambda x: -len(x[0])):
            if re.search(r'\b' + re.escape(term) + r'\b', q):
                filters["silhouette"] = [val]
                matched_terms.append(term)
                break

        # --- Seasons ---
        _SEASON_MAP = {
            "summer": "Summer", "spring": "Spring",
            "fall": "Fall", "autumn": "Fall",
            "winter": "Winter",
        }
        for term, val in sorted(_SEASON_MAP.items(), key=lambda x: -len(x[0])):
            if re.search(r'\b' + re.escape(term) + r'\b', q):
                filters["seasons"] = [val]
                matched_terms.append(term)
                break

        # --- Category L1 (broad) ---
        _CAT_L1_MAP = {
            "activewear": "Activewear", "athletic": "Activewear",
            "sportswear": "Activewear",
            "swimwear": "Swimwear", "swimsuit": "Swimwear",
            "bikini": "Swimwear", "bathing suit": "Swimwear",
            "outerwear": "Outerwear",
            # Generic broad terms — use category_l1 since they span many
            # category_l2 values (e.g. "dress" covers Mini Dress, Midi Dress,
            # Maxi Dress, Shirt Dress, Slip Dress, etc.)
            "dress": "Dresses", "dresses": "Dresses",
            "top": "Tops", "tops": "Tops",
            "bottom": "Bottoms", "bottoms": "Bottoms",
        }
        for term, val in sorted(_CAT_L1_MAP.items(), key=lambda x: -len(x[0])):
            if re.search(r'\b' + re.escape(term) + r'\b', q):
                filters["category_l1"] = [val]
                matched_terms.append(term)
                break

        # --- Category L2 (specific article type) ---
        # These map query terms to exact Algolia category_l2 values.
        # Multi-word first, then single-word (longest match wins).
        # Values are lists to handle singular/plural variants in Algolia
        # (e.g. Algolia has both "Blouse" (9707) and "Blouses" (142)).
        _CAT_L2_MAP = {
            # Tops
            "button up shirt": ["Shirts", "Shirt"],
            "button-up shirt": ["Shirts", "Shirt"],
            "button down": ["Shirts", "Shirt"],
            "button-down": ["Shirts", "Shirt"],
            "crop top": ["Crop Top", "Crop Tops"],
            "crop tops": ["Crop Top", "Crop Tops"],
            "tank top": ["Tank Top", "Tank Tops"],
            "tank tops": ["Tank Top", "Tank Tops"],
            "tube top": ["Tube Top", "Tube Tops"],
            "tube tops": ["Tube Top", "Tube Tops"],
            "halter top": ["Halter Top", "Halter Tops"],
            "long sleeve top": ["Long Sleeve Top", "Long Sleeve Tops"],
            "polo shirt": ["Polo Shirt", "Polo Shirts"],
            "polo": ["Polo Shirt", "Polo Shirts"],
            "t-shirt": ["T-Shirt", "T-Shirts"],
            "tee": ["T-Shirt", "T-Shirts"],
            "tees": ["T-Shirt", "T-Shirts"],
            "t shirt": ["T-Shirt", "T-Shirts"],
            "blouse": ["Blouse", "Blouses"],
            "blouses": ["Blouse", "Blouses"],
            "camisole": ["Camisole", "Camisoles"],
            "cami": ["Camisole", "Camisoles"],
            "bodysuit": ["Bodysuit", "Bodysuits"],
            "bodysuits": ["Bodysuit", "Bodysuits"],
            "bustier": ["Bustier", "Bustiers"],
            "corset": ["Corset", "Corsets"],
            "bralette": ["Bralette", "Bralettes"],
            "bandeau": ["Bandeau Top", "Bandeau Tops"],
            "tunic": ["Tunic", "Tunics"],
            "turtleneck": ["Turtleneck", "Turtlenecks"],
            "shirt": ["Shirt", "Shirts"],
            "shirts": ["Shirt", "Shirts"],
            # Knitwear
            "cardigan": ["Cardigan", "Cardigans"],
            "cardigans": ["Cardigan", "Cardigans"],
            "sweater": ["Sweater", "Sweaters"],
            "sweaters": ["Sweater", "Sweaters"],
            "pullover": ["Sweater", "Sweaters"],
            "sweater vest": ["Sweater Vest", "Sweater Vests"],
            "hoodie": ["Hoodie", "Hoodies"],
            "hoodies": ["Hoodie", "Hoodies"],
            "sweatshirt": ["Sweatshirt", "Sweatshirts"],
            "sweatshirts": ["Sweatshirt", "Sweatshirts"],
            # Dresses
            "mini dress": ["Mini Dress", "Mini Dresses"],
            "mini dresses": ["Mini Dress", "Mini Dresses"],
            "midi dress": ["Midi Dress", "Midi Dresses"],
            "midi dresses": ["Midi Dress", "Midi Dresses"],
            "maxi dress": ["Maxi Dress", "Maxi Dresses"],
            "maxi dresses": ["Maxi Dress", "Maxi Dresses"],
            "shirt dress": ["Shirt Dress", "Shirt Dresses"],
            "slip dress": ["Slip Dress", "Slip Dresses"],
            "wrap dress": ["Wrap Dress", "Wrap Dresses"],
            "sweater dress": ["Sweater Dress", "Sweater Dresses"],
            "t-shirt dress": ["T-Shirt Dress", "T-Shirt Dresses"],
            "tshirt dress": ["T-Shirt Dress", "T-Shirt Dresses"],
            "formal dress": ["Formal Dress", "Formal Dresses"],
            "casual dress": ["Casual Dress", "Casual Dresses"],
            "bodycon dress": ["Bodycon", "Bodycon Dress", "Bodycon Dresses"],
            "shift dress": ["Shift Dress", "Shift Dresses"],
            # Bottoms
            "jeans": ["Jeans"],
            "denim jeans": ["Jeans"],
            "pants": ["Pants"],
            "trousers": ["Pants", "Trousers"],
            "leggings": ["Leggings"],
            "shorts": ["Shorts"],
            "sweatpants": ["Sweatpants"],
            "joggers": ["Sweatpants", "Joggers"],
            "skirt": ["Skirt", "Skirts"],
            "skirts": ["Skirt", "Skirts"],
            # Outerwear
            "blazer": ["Blazer", "Blazers"],
            "blazers": ["Blazer", "Blazers"],
            "jacket": ["Jacket", "Jackets"],
            "jackets": ["Jacket", "Jackets"],
            "coat": ["Coat", "Coats"],
            "coats": ["Coat", "Coats"],
            "trench coat": ["Trench Coat", "Trench Coats"],
            "trench": ["Trench Coat", "Trench Coats"],
            "puffer jacket": ["Puffer Jacket", "Puffer Jackets"],
            "puffer": ["Puffer Jacket", "Puffer Jackets"],
            "leather jacket": ["Leather Jacket", "Leather Jackets"],
            "denim jacket": ["Denim Jacket", "Denim Jackets"],
            "jean jacket": ["Denim Jacket", "Denim Jackets"],
            "fleece jacket": ["Fleece Jacket", "Fleece Jackets"],
            "vest": ["Vest", "Vests"],
            "vests": ["Vest", "Vests"],
            "waistcoat": ["Waistcoat", "Waistcoats"],
            "parka": ["Parka", "Parkas"],
            "poncho": ["Poncho", "Ponchos"],
            # One-piece
            "jumpsuit": ["Jumpsuit", "Jumpsuits"],
            "jumpsuits": ["Jumpsuit", "Jumpsuits"],
            "romper": ["Romper", "Rompers"],
            "rompers": ["Romper", "Rompers"],
            # Activewear
            "sports bra": ["Sports Bra", "Sports Bras"],
            "tracksuit": ["Tracksuit", "Tracksuits"],
            # Swimwear
            "one-piece swimsuit": ["One-Piece Swimsuit", "One-Piece Swimsuits"],
            "one piece swimsuit": ["One-Piece Swimsuit", "One-Piece Swimsuits"],
            "bikini set": ["Bikini Set", "Bikini Sets"],
            "bikini top": ["Bikini Top", "Bikini Tops"],
            "bikini": ["Bikini", "Bikini Set", "Bikini Sets"],
            # Loungewear
            "pajama": ["Pajama Set", "Pajama Sets"],
            "pajamas": ["Pajama Set", "Pajama Sets"],
            "pjs": ["Pajama Set", "Pajama Sets"],
            "robe": ["Robe", "Robes"],
            # Generic broad terms -> use category_l1 or broad_category instead
            # "dress"/"dresses"/"top"/"tops"/"shirt"/"shirts" are too broad for
            # category_l2 and work fine with Algolia keyword matching.
            # Specific missing terms from _CATEGORY_KEYWORDS:
            "kimono": ["Kimono", "Kimonos"],
            "culottes": ["Culottes"],
            "tights": ["Tights"],
            "henley": ["Henley", "Henleys"],
            "cape": ["Cape", "Capes"],
            "windbreaker": ["Windbreaker", "Windbreakers"],
        }
        for term, vals in sorted(_CAT_L2_MAP.items(), key=lambda x: -len(x[0])):
            if re.search(r'\b' + re.escape(term) + r'\b', q):
                filters["category_l2"] = vals
                matched_terms.append(term)
                break

        # --- Style tags ---
        _STYLE_MAP = {
            "bohemian": "Bohemian", "boho": "Bohemian",
            "romantic": "Romantic",
            "glamorous": "Glamorous", "glam": "Glamorous",
            "edgy": "Edgy",
            "vintage": "Vintage", "retro": "Vintage",
            "sporty": "Sporty",
            "classic": "Classic",
            "modern": "Modern",
            "minimalist": "Minimalist", "minimal": "Minimalist",
            "preppy": "Preppy",
            "streetwear": "Streetwear", "street": "Streetwear",
            "sexy": "Sexy",
            "western": "Western",
            "utility": "Utility",
        }
        for term, val in sorted(_STYLE_MAP.items(), key=lambda x: -len(x[0])):
            if re.search(r'\b' + re.escape(term) + r'\b', q):
                filters["style_tags"] = [val]
                matched_terms.append(term)
                break

        return filters, matched_terms

    @classmethod
    def get_algolia_weight(cls, intent: QueryIntent) -> float:
        """Get the Algolia weight for RRF merge based on intent."""
        weights = cls._get_rrf_weights()
        return weights.get(intent.value, (0.60, 0.40))[0]

    @classmethod
    def get_semantic_weight(cls, intent: QueryIntent) -> float:
        """Get the semantic (FashionCLIP) weight for RRF merge based on intent."""
        weights = cls._get_rrf_weights()
        return weights.get(intent.value, (0.60, 0.40))[1]
