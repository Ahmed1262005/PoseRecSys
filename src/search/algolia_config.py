"""
Algolia Index Configuration.

Defines searchable attributes, facets, custom ranking, synonyms,
and the product-to-record mapping for the Algolia index.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime


# ============================================================================
# Index Settings
# ============================================================================

ALGOLIA_INDEX_SETTINGS: Dict[str, Any] = {
    # ===========================================
    # SEARCHABLE ATTRIBUTES (priority order)
    # ===========================================
    "searchableAttributes": [
        "name",                          # 1st - Product name (highest)
        "brand",                         # 2nd - Brand name
        "unordered(category_l2)",        # 3rd - Specific type (Blouse, Jeans)
        "unordered(article_type)",       # 4th - Article type from scraper
        "unordered(style_tags)",         # 5th - Style descriptors
        "unordered(apparent_fabric)",    # 6th - Material
        "unordered(primary_color)",      # 7th - Color
    ],

    # ===========================================
    # FACETS FOR FILTERING
    # ===========================================
    "attributesForFaceting": [
        # Searchable facets (can search within facet values)
        "searchable(brand)",

        # Filter-only facets
        "filterOnly(in_stock)",
        "filterOnly(is_on_sale)",

        # Category hierarchy
        "category_l1",
        "category_l2",
        "broad_category",
        "article_type",

        # Construction
        "silhouette",
        "length",
        "neckline",
        "sleeve_type",
        "closure_type",

        # Color & Pattern
        "primary_color",
        "color_family",
        "pattern",

        # Material
        "apparent_fabric",
        "texture",
        "sheen",

        # Style
        "style_tags",
        "occasions",
        "seasons",
        "formality",
        "trend_tags",

        # Fit
        "fit_type",
        "stretch",
        "rise",
        "leg_shape",

        # Price (numeric)
        "price",
    ],

    # ===========================================
    # CUSTOM RANKING (business tie-breakers)
    # ===========================================
    "customRanking": [
        "desc(trending_score)",
        "desc(popularity_score)",
        "desc(in_stock)",
    ],

    # ===========================================
    # TYPO TOLERANCE
    # ===========================================
    "typoTolerance": True,
    "minWordSizefor1Typo": 4,   # was 3 — prevents 3-char typo collisions (top→tee)
    "minWordSizefor2Typos": 7,  # was 6 — stricter on 2-typo matches
    # Disable typo tolerance on fashion terms that collide with each other
    # at edit distance 1-2 (shirt↔skirt, mini↔midi, slip↔slim, etc.)
    "disableTypoToleranceOnWords": [
        # These pairs are 1 edit apart and mean completely different things
        "shirt", "skirt", "short", "shorts",
        "mini", "midi", "maxi",
        "slip", "slim", "slit",
        "crop", "drop",
        "lace", "silk",
        "knit", "unit",
        "vest", "best",
        "cape", "tape",
        "wrap", "warp",
        "cuff", "puff",
        "tank",
        "coat",
        "belt", "felt",
    ],

    # ===========================================
    # LANGUAGE
    # ===========================================
    "removeStopWords": True,
    "ignorePlurals": True,

    # ===========================================
    # HIGHLIGHTING
    # ===========================================
    "attributesToHighlight": ["name", "brand"],
    "highlightPreTag": "<mark>",
    "highlightPostTag": "</mark>",

    # ===========================================
    # PAGINATION
    # ===========================================
    "hitsPerPage": 50,
    "paginationLimitedTo": 1000,

    # ===========================================
    # ATTRIBUTES TO RETRIEVE (performance)
    # ===========================================
    "attributesToRetrieve": [
        "objectID",
        "name",
        "brand",
        "category_l1",
        "category_l2",
        "broad_category",
        "article_type",
        "primary_color",
        "color_family",
        "pattern",
        "apparent_fabric",
        "style_tags",
        "occasions",
        "seasons",
        "formality",
        "fit_type",
        "silhouette",
        "length",
        "neckline",
        "sleeve_type",
        "rise",
        "price",
        "original_price",
        "is_on_sale",
        "in_stock",
        "trending_score",
        "image_url",
        "gallery_images",
    ],
}


# ============================================================================
# Synonyms
# ============================================================================

ALGOLIA_SYNONYMS: List[Dict[str, Any]] = [
    # ===================================================================
    # CLOTHING TERMS
    # Principle: only synonymize words that are genuinely interchangeable
    # names for the SAME garment. Do NOT conflate related but different
    # garments (shirt ≠ blouse ≠ top, romper ≠ jumpsuit, coat ≠ jacket).
    # ===================================================================

    # Pants/trousers are the same garment
    {"objectID": "pants_trousers", "type": "synonym",
     "synonyms": ["pants", "trousers", "slacks"]},

    # Sweater variants — all refer to the same knit pullover garment
    {"objectID": "sweater_jumper", "type": "synonym",
     "synonyms": ["sweater", "jumper", "pullover"]},

    # T-shirt spelling variants — same garment, different spellings
    {"objectID": "tee_tshirt", "type": "synonym",
     "synonyms": ["tee", "t-shirt", "tshirt", "t shirt"]},

    # Jeans/denim — same garment
    {"objectID": "jeans_denim", "type": "synonym",
     "synonyms": ["jeans", "denim jeans"]},

    # Cardigan spelling variants
    {"objectID": "cardigan_cardi", "type": "synonym",
     "synonyms": ["cardigan", "cardi"]},

    # Leggings/tights — genuinely interchangeable
    {"objectID": "leggings_tights", "type": "synonym",
     "synonyms": ["leggings", "tights"]},

    # Hoodie — spelling variant only
    {"objectID": "hoodie_hoody", "type": "synonym",
     "synonyms": ["hoodie", "hoody"]},

    # Romper/playsuit — same garment, different regional names
    {"objectID": "romper_playsuit", "type": "synonym",
     "synonyms": ["romper", "playsuit"]},

    # Blazer/sport coat — same garment
    {"objectID": "blazer_sportcoat", "type": "synonym",
     "synonyms": ["blazer", "sport coat", "sportcoat"]},

    # Swimsuit variants
    {"objectID": "swimsuit_bathers", "type": "synonym",
     "synonyms": ["swimsuit", "bathing suit", "swimming costume"]},

    # Camisole shorthand
    {"objectID": "camisole_cami", "type": "synonym",
     "synonyms": ["camisole", "cami"]},

    # Joggers/sweatpants — same garment
    {"objectID": "joggers_sweatpants", "type": "synonym",
     "synonyms": ["joggers", "sweatpants", "track pants"]},

    # ===================================================================
    # ONE-WAY SYNONYMS (alternatives → canonical)
    # Use oneWaySynonym so searching "gown" finds dresses, but searching
    # "dress" does NOT return gowns (gowns are a subset of dresses).
    # ===================================================================

    # "frock" and "gown" are types of dresses — one-way expansion
    {"objectID": "frock_to_dress", "type": "oneWaySynonym",
     "input": "frock", "synonyms": ["dress"]},
    {"objectID": "gown_to_dress", "type": "oneWaySynonym",
     "input": "gown", "synonyms": ["dress"]},

    # "parka" is a type of coat
    {"objectID": "parka_to_coat", "type": "oneWaySynonym",
     "input": "parka", "synonyms": ["coat"]},

    # "bodycon" is a fit, not a garment — but searching it should also find bodycon dresses
    {"objectID": "bodycon_to_dress", "type": "oneWaySynonym",
     "input": "bodycon", "synonyms": ["bodycon dress"]},

    # ===================================================================
    # FIT TERMS
    # Only synonymize words that mean the exact same fit description.
    # ===================================================================

    {"objectID": "oversized_baggy", "type": "synonym",
     "synonyms": ["oversized", "baggy"]},
    {"objectID": "cropped_crop", "type": "synonym",
     "synonyms": ["cropped", "crop"]},
    {"objectID": "slim_skinny", "type": "synonym",
     "synonyms": ["slim fit", "skinny fit"]},

    # ===================================================================
    # PATTERN TERMS — spelling variants only
    # ===================================================================

    {"objectID": "striped_stripes", "type": "synonym",
     "synonyms": ["striped", "stripes", "stripe"]},
    {"objectID": "floral_flower", "type": "synonym",
     "synonyms": ["floral", "flower", "flowery"]},
    {"objectID": "plaid_tartan", "type": "synonym",
     "synonyms": ["plaid", "tartan", "checkered"]},
    {"objectID": "polkadot_dots", "type": "synonym",
     "synonyms": ["polka dot", "polka dots", "dotted"]},
    {"objectID": "leopard_animal", "type": "synonym",
     "synonyms": ["leopard print", "animal print"]},
    {"objectID": "camo_camouflage", "type": "synonym",
     "synonyms": ["camo", "camouflage"]},

    # ===================================================================
    # MATERIAL TERMS — only true equivalents
    # Do NOT conflate cotton↔jersey, silk↔satin (different materials!)
    # ===================================================================

    {"objectID": "leather_faux", "type": "synonym",
     "synonyms": ["faux leather", "vegan leather", "pleather"]},
    {"objectID": "denim_chambray", "type": "synonym",
     "synonyms": ["denim", "chambray"]},
    {"objectID": "linen_flax", "type": "synonym",
     "synonyms": ["linen", "flax"]},

    # ===================================================================
    # COLOR TERMS — genuine name equivalents only
    # ===================================================================

    {"objectID": "navy_darkblue", "type": "synonym",
     "synonyms": ["navy", "navy blue"]},
    {"objectID": "burgundy_wine", "type": "synonym",
     "synonyms": ["burgundy", "wine", "maroon"]},
    {"objectID": "grey_gray", "type": "synonym",
     "synonyms": ["grey", "gray"]},
    {"objectID": "ivory_cream", "type": "synonym",
     "synonyms": ["ivory", "cream", "off-white", "offwhite"]},

    # ===================================================================
    # STYLE TERMS — only exact equivalents
    # Do NOT conflate casual↔everyday, formal↔elegant↔dressy
    # (these have different connotations; use extract_attributes() for
    # mapping vibe terms to structured facet filters instead).
    # ===================================================================

    {"objectID": "boho_bohemian", "type": "synonym",
     "synonyms": ["boho", "bohemian"]},
    {"objectID": "retro_vintage", "type": "synonym",
     "synonyms": ["retro", "vintage"]},
    {"objectID": "athleisure_sporty", "type": "synonym",
     "synonyms": ["athleisure", "activewear"]},
]


# ============================================================================
# Product-to-Algolia Record Mapping
# ============================================================================

def _to_timestamp(value) -> int:
    """Convert a datetime or string to unix timestamp."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, datetime):
        return int(value.timestamp())
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(dt.timestamp())
        except (ValueError, TypeError):
            return 0
    return 0


def product_to_algolia_record(
    product: dict,
    attributes: Optional[dict] = None,
) -> dict:
    """
    Convert a Supabase product + product_attributes row to an Algolia record.

    Args:
        product: Row from the products table.
        attributes: Row from product_attributes table (Gemini Vision).

    Returns:
        Dict suitable for Algolia indexing.
    """
    attrs = attributes or {}
    construction = attrs.get("construction", {}) or {}

    record = {
        # ==========================================
        # Required
        # ==========================================
        "objectID": str(product["id"]),

        # ==========================================
        # Searchable Text
        # ==========================================
        "name": product.get("name"),
        "brand": product.get("brand"),
        "article_type": product.get("article_type"),

        # ==========================================
        # Category (Gemini Vision)
        # ==========================================
        "category_l1": attrs.get("category_l1"),
        "category_l2": attrs.get("category_l2"),
        "broad_category": product.get("broad_category"),

        # ==========================================
        # Construction (Gemini Vision)
        # ==========================================
        "silhouette": attrs.get("silhouette") or construction.get("silhouette"),
        "length": construction.get("length"),
        "neckline": construction.get("neckline"),
        "sleeve_type": construction.get("sleeve_type"),
        "closure_type": construction.get("closure_type"),

        # ==========================================
        # Color & Pattern
        # ==========================================
        "primary_color": attrs.get("primary_color"),
        "color_family": attrs.get("color_family"),
        "secondary_colors": attrs.get("secondary_colors") or [],
        "pattern": attrs.get("pattern"),
        "colors": product.get("colors") or [],

        # ==========================================
        # Material
        # ==========================================
        "apparent_fabric": attrs.get("apparent_fabric"),
        "texture": attrs.get("texture"),
        "sheen": attrs.get("sheen"),
        "materials": product.get("materials") or [],

        # ==========================================
        # Style (Gemini Vision)
        # ==========================================
        "style_tags": attrs.get("style_tags") or [],
        "occasions": attrs.get("occasions") or [],
        "seasons": attrs.get("seasons") or [],
        "formality": attrs.get("formality"),
        "trend_tags": attrs.get("trend_tags") or [],

        # ==========================================
        # Fit (Gemini Vision)
        # ==========================================
        "fit_type": attrs.get("fit_type"),
        "fit": product.get("fit"),
        "stretch": attrs.get("stretch"),
        "rise": attrs.get("rise"),
        "leg_shape": attrs.get("leg_shape"),

        # ==========================================
        # Pricing
        # ==========================================
        "price": float(product.get("price", 0) or 0),
        "original_price": float(
            product.get("original_price") or product.get("price") or 0
        ),
        "is_on_sale": bool(
            product.get("original_price")
            and float(product.get("original_price") or 0)
            > float(product.get("price", 0) or 0)
        ),

        # ==========================================
        # Stock & Recency
        # ==========================================
        "in_stock": product.get("in_stock", True),
        "created_at_timestamp": _to_timestamp(product.get("created_at")),

        # ==========================================
        # Ranking Metrics
        # ==========================================
        "trending_score": float(product.get("trending_score", 0) or 0),
        "popularity_score": float(product.get("popularity_score", 0) or 0),

        # ==========================================
        # Images (non-indexed, for display)
        # ==========================================
        "image_url": product.get("primary_image_url"),
        "gallery_images": product.get("gallery_images") or [],
    }

    # Remove None values (Algolia doesn't like them)
    return {k: v for k, v in record.items() if v is not None}
