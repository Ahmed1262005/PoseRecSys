"""
Brand-Cluster Mapping for Fashion Persona Engine.

Maps 131+ brands to persona clusters (A-X) based on brand DNA:
- ICP (ideal customer profile)
- Style positioning
- Typical occasions
- Color palette tendency
- Revealing level

Each cluster has metadata traits used for scoring and matching.
Python dict for fast iteration. Can move to DB table later.
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass


# =============================================================================
# Cluster Traits
# =============================================================================

@dataclass(frozen=True)
class ClusterTraits:
    """Metadata for a fashion persona cluster."""
    name: str
    style_tags: Tuple[str, ...]
    occasions: Tuple[str, ...]
    palette: str          # neutral, broad, loud, muted, earth
    revealing: str        # modest, balanced, revealing, varies
    price_tier: str       # value, mid, premium, luxury
    formality: str        # casual, smart-casual, versatile, formal
    # --- Enriched fields for LLM planner context ---
    description: str = ""           # Natural-language style summary for LLM
    icp_age: str = ""               # Ideal customer age range (e.g. "25-45")
    typical_price_range: Tuple[int, int] = (0, 0)  # Actual $ range (min, max)


CLUSTER_TRAITS: Dict[str, ClusterTraits] = {
    "A": ClusterTraits(
        name="Modern Classics / Elevated Everyday",
        style_tags=("tailored", "clean", "knitwear", "blazers", "feminine"),
        occasions=("work", "smart-casual", "dates", "dinners", "weekends", "travel"),
        palette="neutral", revealing="balanced", price_tier="mid-premium",
        formality="smart-casual",
        description="Polished basics and put-together outfits. Tailored casual, clean silhouettes, knitwear, blazers, trousers, elevated denim, feminine-but-wearable dresses. Lots of neutrals with restrained prints.",
        icp_age="25-45",
        typical_price_range=(60, 400),
    ),
    "B": ClusterTraits(
        name="Premium Denim / Casual Americana",
        style_tags=("denim", "basics", "leather", "off-duty"),
        occasions=("everyday", "casual-work", "weekends", "nights-out"),
        palette="earth", revealing="balanced", price_tier="premium",
        formality="casual",
        description="Denim-first, quality/fit obsessed. Premium denim staples, leather jackets, tees, 'off-duty cool'. Strong jeans-plus-top formula. Denim washes, earth tones, limited brights.",
        icp_age="28-50",
        typical_price_range=(120, 350),
    ),
    "C": ClusterTraits(
        name="Mass Casual / Teen Mainstream",
        style_tags=("casual", "basics", "logo", "denim", "graphic"),
        occasions=("school", "campus", "errands", "casual"),
        palette="broad", revealing="balanced", price_tier="value",
        formality="casual",
        description="Value-conscious, trend-aware mall-core. Casual basics, logo/graphic tees, denim, hoodies, simple dresses. Neutrals plus seasonal brights, more logos than premium brands.",
        icp_age="14-28",
        typical_price_range=(15, 65),
    ),
    "D": ClusterTraits(
        name="Premium Athleisure / Wellness",
        style_tags=("leggings", "sets", "sports-bra", "lounge", "performance"),
        occasions=("gym", "athleisure", "travel", "casual-weekends"),
        palette="neutral", revealing="varies", price_tier="premium",
        formality="casual",
        description="Fitness-meets-lifestyle identity. Leggings, matching sets, sports bras, soft lounge, performance-meets-fashion. Heavy neutrals with muted pastels. Body-contouring but layerable.",
        icp_age="22-40",
        typical_price_range=(60, 150),
    ),
    "E": ClusterTraits(
        name="Athletic Heritage / Sportswear",
        style_tags=("sneakers", "joggers", "sporty", "logo", "heritage"),
        occasions=("workouts", "streetwear", "weekends"),
        palette="broad", revealing="modest", price_tier="mid",
        formality="casual",
        description="Athletic or athleisure buyer, brand-logo comfort. Sneakers, joggers, tees, outer layers, team/heritage sporty. Black/white/gray plus strong primary colors, visible logos.",
        icp_age="16-40",
        typical_price_range=(30, 120),
    ),
    "F": ClusterTraits(
        name="Comfort Footwear / Mass Sporty",
        style_tags=("comfort", "practical", "easy-basics"),
        occasions=("everyday", "errands", "activities"),
        palette="neutral", revealing="modest", price_tier="value",
        formality="casual",
        description="Practical comfort-first shopper. Casual shoes plus easy basics, comfort/performance over fashion edge. Neutrals dominate, some brights in sport lines.",
        icp_age="25-60",
        typical_price_range=(30, 100),
    ),
    "G": ClusterTraits(
        name="Affordable Essentials / Core Wardrobe",
        style_tags=("tees", "knits", "trousers", "capsule", "basics"),
        occasions=("everyday", "casual-work", "travel"),
        palette="muted", revealing="balanced", price_tier="value",
        formality="casual",
        description="Value-focused, dependable basics with minimal styling effort. Tees, knits, simple trousers, workhorse outerwear, capsule building blocks. Neutrals plus muted seasonal colors.",
        icp_age="18-45",
        typical_price_range=(15, 80),
    ),
    "H": ClusterTraits(
        name="Ultra-Fast Fashion / High-Trend",
        style_tags=("micro-trends", "party", "crop", "cutouts", "bodycon"),
        occasions=("going-out", "parties", "festivals", "vacation"),
        palette="loud", revealing="revealing", price_tier="value",
        formality="casual",
        description="Trend-chasing, social/going-out driven. Micro-trends, club/party looks, crop tops, minis, cutouts, bodycon, 'outfit for tonight'. Wide loud colors, high-contrast, trend palettes shift fast. Short hems, cutouts, low necklines, tight fits.",
        icp_age="16-28",
        typical_price_range=(8, 50),
    ),
    "I": ClusterTraits(
        name="Outdoor / Technical Performance",
        style_tags=("shells", "fleece", "puffers", "hiking", "technical"),
        occasions=("outdoors", "hiking", "travel", "gorpcore"),
        palette="earth", revealing="modest", price_tier="premium",
        formality="casual",
        description="Functionality-first outdoors buyer. Shells, fleeces, puffers, hiking pants, base layers, technical fabrics. Earth tones plus practical brights, lots of black/gray/navy. Full coverage utility fits.",
        icp_age="20-55",
        typical_price_range=(60, 400),
    ),
    "J": ClusterTraits(
        name="Youth Mall Trend",
        style_tags=("cute", "trend", "denim", "mini", "going-out"),
        occasions=("campus", "casual", "dates", "parties"),
        palette="broad", revealing="balanced", price_tier="value-mid",
        formality="casual",
        description="Trend-driven but more mall than ultra-fast. Cute trend tops, denim, mini skirts, going-out sets, seasonal basics. Playful broad palette with prints/graphics. Shorter/fitted silhouettes common.",
        icp_age="16-26",
        typical_price_range=(12, 60),
    ),
    "K": ClusterTraits(
        name="Premium Contemporary Staples / Quiet-Lux",
        style_tags=("blazers", "coats", "premium-knits", "elevated-tees", "minimal"),
        occasions=("work", "dinners", "travel", "capsule"),
        palette="neutral", revealing="balanced", price_tier="premium",
        formality="smart-casual",
        description="Style-literate, quality/fit focused, buys fewer-better pieces. Blazers, coats, premium knits, elevated tees, tailored pants, minimal dresses. Neutrals and restrained palettes, minimal prints. Sleek but rarely overtly sexy.",
        icp_age="25-45",
        typical_price_range=(100, 400),
    ),
    "L": ClusterTraits(
        name="Premium Contemporary Designer",
        style_tags=("tailored", "structured", "statement-coats", "fashion"),
        occasions=("office", "dinners", "events", "city"),
        palette="neutral", revealing="balanced", price_tier="premium",
        formality="smart-casual",
        description="Higher disposable income, likes designer signaling without full luxury pricing. Tailored, structured, statement coats, elevated denim, refined dresses. Sophisticated neutrals plus seasonal fashion colors. Polished over provocative.",
        icp_age="28-50",
        typical_price_range=(150, 600),
    ),
    "M": ClusterTraits(
        name="Boho / Indie / Festival",
        style_tags=("boho", "oversized", "western", "prints", "layered"),
        occasions=("weekends", "festivals", "travel", "dates"),
        palette="earth", revealing="balanced", price_tier="mid",
        formality="casual",
        description="Creative/alt vibe, likes textures, layers, and expressive styling. Boho dresses, oversized knits, denim shorts, western touches, playful prints. Earthy tones, vintage-wash denim, lots of patterns. Mix of flowy coverage and occasional short/fitted pieces.",
        icp_age="16-35",
        typical_price_range=(50, 200),
    ),
    "P": ClusterTraits(
        name="Department-Store Mainstream / Logo-Lifestyle",
        style_tags=("logo", "classic-denim", "casual-dresses", "safe"),
        occasions=("everyday", "casual-work", "family"),
        palette="neutral", revealing="modest", price_tier="mid",
        formality="casual",
        description="Mainstream shopper, prefers recognizable brands, classic fits. Logo tees, classic denim, casual dresses, work basics, safe styling. Classic neutrals plus preppy primaries (navy/red/white).",
        icp_age="20-55",
        typical_price_range=(40, 150),
    ),
    "Q": ClusterTraits(
        name="Modern Feminine Eco-Chic",
        style_tags=("dresses", "flattering", "romantic", "minimal-feminine"),
        occasions=("dates", "brunch", "weddings-guest", "vacation"),
        palette="muted", revealing="balanced", price_tier="mid-premium",
        formality="versatile",
        description="Feminine, values fit and silhouette, often sustainability-aware. Dresses, flattering tops, skirts, sets, romantic minimal rather than boho. Neutrals plus soft florals and warm tones. Slits, fitted bodices, occasional lower necklines.",
        icp_age="22-40",
        typical_price_range=(80, 300),
    ),
    "R": ClusterTraits(
        name="Trendy Feminine / Going-Out Elevated",
        style_tags=("mini-dresses", "corset", "sets", "prints", "photo-ready"),
        occasions=("nights-out", "parties", "vacation", "brunch"),
        palette="broad", revealing="revealing", price_tier="mid",
        formality="casual",
        description="Social/calendar-driven, wants cute flattering photo-ready looks. Mini dresses, corset tops, sets, prints, vacation/night-out pieces. More color/print than premium brands. Shorter hems, cutouts, bodycon show up regularly.",
        icp_age="18-35",
        typical_price_range=(30, 100),
    ),
    "S": ClusterTraits(
        name="Resort / Coastal Minimal",
        style_tags=("linen", "slip-dresses", "minimal", "resort"),
        occasions=("vacations", "summer", "beach-dinners", "casual"),
        palette="neutral", revealing="balanced", price_tier="mid-premium",
        formality="casual",
        description="'Clean sexy' vacation aesthetic. Linen sets, slip dresses, minimal silhouettes, curated resortwear. Whites/creams, tans, blacks, soft pastels, low-print. Open backs and slinky fits but not clubby.",
        icp_age="22-40",
        typical_price_range=(100, 400),
    ),
    "T": ClusterTraits(
        name="Designer Occasion / Eventwear",
        style_tags=("cocktail", "gowns", "statement", "structured"),
        occasions=("weddings", "galas", "formal-dinners", "events"),
        palette="neutral", revealing="varies", price_tier="premium-luxury",
        formality="formal",
        description="Event-driven buyer, wants standout silhouettes. Cocktail dresses, gowns, statement sets, structured pieces, bold details. Classic event colors (black, jewel tones) and bold prints. Often sexier (slits, open backs, sculpted fits) but can be elegant/moderate.",
        icp_age="25-55",
        typical_price_range=(150, 800),
    ),
    "U": ClusterTraits(
        name="Intimates / Shapewear / Lounge",
        style_tags=("bras", "shapewear", "underwear", "lounge"),
        occasions=("layering", "at-home", "everyday"),
        palette="neutral", revealing="n/a", price_tier="mid",
        formality="casual",
        description="Comfort, smoothing, under-outfit intent. Bras, bodysuits, shapewear, underwear, lounge basics. Heavy neutrals (nude range, black, cream). Function-first rather than 'sexy outfits'.",
        icp_age="18-50",
        typical_price_range=(20, 80),
    ),
    "V": ClusterTraits(
        name="Luxury Designer / Quiet Luxury",
        style_tags=("impeccable-basics", "tailoring", "investment-pieces"),
        occasions=("work", "travel", "city", "elevated-everyday"),
        palette="neutral", revealing="balanced", price_tier="luxury",
        formality="smart-casual",
        description="High-income, brand/fabric-driven, buys investment pieces. Impeccable basics, tailoring, premium outerwear, refined silhouettes. Neutrals and subdued tones, minimal prints/logos. Coverage-forward, shape via tailoring not skin.",
        icp_age="28-55",
        typical_price_range=(300, 3000),
    ),
    "W": ClusterTraits(
        name="Resort Statement / Artsy Vacation",
        style_tags=("printed-dresses", "sets", "airy", "artful"),
        occasions=("vacations", "beach-clubs", "destination-dinners"),
        palette="loud", revealing="balanced", price_tier="mid-premium",
        formality="casual",
        description="Vacation wardrobe builder, artful prints. Printed dresses/sets, airy silhouettes, resort pieces that photograph well. Bold saturated colors, lots of prints/patterns. Flowy with strategic skin (open backs, side cutouts sometimes).",
        icp_age="22-45",
        typical_price_range=(100, 400),
    ),
    "X": ClusterTraits(
        name="Y2K / Statement Denim / Edgy",
        style_tags=("statement-jeans", "distressed", "y2k", "graphic", "edgy"),
        occasions=("nights-out", "concerts", "streetwear"),
        palette="loud", revealing="balanced", price_tier="mid",
        formality="casual",
        description="Street/Y2K-leaning, likes loud denim, logos, attitude, nostalgia plus edge. Statement jeans, distressed/low-rise vibes, bold washes, graphic tops, club-casual. High-contrast, black-heavy, logo/graphic presence.",
        icp_age="18-35",
        typical_price_range=(40, 150),
    ),
}


# =============================================================================
# Brand -> Cluster Mapping (lowercase brand -> (cluster_id, confidence))
# =============================================================================

BRAND_CLUSTER_MAP: Dict[str, Tuple[str, float]] = {
    # --- Cluster A: Modern Classics / Elevated Everyday ---
    "reformation": ("A", 1.0),
    "ba&sh": ("A", 1.0),
    "club monaco": ("A", 1.0),
    "ann taylor": ("A", 0.9),
    "j.crew": ("A", 0.9),
    "rails": ("A", 0.9),
    "reiss": ("A", 1.0),
    "cos": ("A", 0.9),
    "arket": ("A", 0.9),
    "& other stories": ("A", 0.9),
    "other stories": ("A", 0.9),
    "massimo dutti": ("A", 0.9),
    "ted baker": ("A", 0.8),
    "karen millen": ("A", 0.8),
    "whistles": ("A", 1.0),
    "hobbs": ("A", 0.9),
    "white house black market": ("A", 0.9),
    "banana republic": ("A", 0.9),
    "mango": ("A", 0.7),
    # --- Cluster B: Premium Denim / Casual Americana ---
    "joe's jeans": ("B", 1.0),
    "re/done": ("B", 1.0),
    "rag & bone": ("B", 1.0),
    "citizens of humanity": ("B", 1.0),
    "agolde": ("B", 1.0),
    "dl1961": ("B", 1.0),
    "hudson jeans": ("B", 0.9),
    "paige": ("B", 1.0),
    "frame": ("B", 1.0),
    "mother": ("B", 1.0),
    "levi's": ("B", 0.8),
    "madewell": ("B", 0.8),
    "j brand": ("B", 0.9),
    "scotch & soda": ("B", 0.8),
    "7 for all mankind": ("B", 1.0),
    "ag jeans": ("B", 1.0),
    "lee": ("B", 0.8),
    "lucky brand": ("B", 0.8),
    "mavi jeans": ("B", 0.9),
    "guess": ("B", 0.7),
    # --- Cluster C: Mass Casual / Teen Mainstream ---
    "american eagle outfitters": ("C", 1.0),
    "american eagle": ("C", 1.0),
    "hollister": ("C", 1.0),
    "aeropostale": ("C", 1.0),
    "hot topic": ("C", 0.8),
    "abercrombie & fitch": ("C", 0.7),
    "abercrombie": ("C", 0.7),
    "pull&bear": ("C", 0.8),
    "brandy melville": ("C", 0.9),
    "garage": ("C", 0.9),
    "pacsun": ("C", 0.8),
    # --- Cluster D: Premium Athleisure / Wellness ---
    "alo yoga": ("D", 1.0),
    "alo": ("D", 1.0),
    "lululemon": ("D", 1.0),
    "vuori": ("D", 1.0),
    "beyond yoga": ("D", 1.0),
    "girlfriend collective": ("D", 0.9),
    "varley": ("D", 1.0),
    "free people movement": ("D", 0.9),
    "athleta": ("D", 0.9),
    "outdoor voices": ("D", 0.9),
    "tory sport": ("D", 0.8),
    # --- Cluster E: Athletic Heritage / Sportswear ---
    "nike": ("E", 1.0),
    "adidas": ("E", 1.0),
    "puma": ("E", 1.0),
    "under armour": ("E", 1.0),
    "new balance": ("E", 1.0),
    "reebok": ("E", 1.0),
    "asics": ("E", 0.9),
    "fila": ("E", 0.9),
    "champion": ("E", 0.9),
    "converse": ("E", 0.8),
    "skechers": ("E", 0.7),
    "reigning champ": ("E", 0.8),
    # --- Cluster F: Comfort Footwear / Mass Sporty ---
    "crocs": ("F", 1.0),
    "ugg": ("F", 0.8),
    "birkenstock": ("F", 0.9),
    # --- Cluster G: Affordable Essentials / Core Wardrobe ---
    "gap": ("G", 1.0),
    "old navy": ("G", 1.0),
    "uniqlo": ("G", 1.0),
    "h&m": ("G", 0.9),
    "everlane": ("G", 0.9),
    "universal standard": ("G", 0.9),
    "zara": ("G", 0.7),
    "quince": ("G", 0.9),
    "cotton on": ("G", 0.8),
    "splendid": ("G", 0.8),
    "lane bryant": ("G", 0.8),
    "torrid": ("G", 0.8),
    # --- Cluster H: Ultra-Fast Fashion / High-Trend ---
    "boohoo": ("H", 1.0),
    "missguided": ("H", 1.0),
    "prettylittlething": ("H", 1.0),
    "plt": ("H", 1.0),
    "shein": ("H", 1.0),
    "nasty gal": ("H", 0.9),
    "fashion nova": ("H", 1.0),
    "meshki": ("H", 0.9),
    "oh polly": ("H", 0.9),
    "white fox": ("H", 0.8),
    "cider": ("H", 0.8),
    "bershka": ("H", 0.8),
    # --- Cluster I: Outdoor / Technical ---
    "the north face": ("I", 1.0),
    "patagonia": ("I", 1.0),
    "columbia": ("I", 1.0),
    "arc'teryx": ("I", 1.0),
    "l.l.bean": ("I", 0.9),
    "carhartt": ("I", 0.8),
    # --- Cluster J: Youth Mall Trend ---
    "forever 21": ("J", 1.0),
    "forever21": ("J", 1.0),
    "charlotte russe": ("J", 1.0),
    "windsor": ("J", 0.9),
    "asos": ("J", 0.7),
    "express": ("J", 0.8),
    # --- Cluster K: Premium Contemporary Staples ---
    "the frankie shop": ("K", 1.0),
    "theory": ("K", 1.0),
    "vince": ("K", 1.0),
    "allsaints": ("K", 0.9),
    "aritzia": ("K", 0.9),
    "acne studios": ("K", 0.9),
    "a.p.c.": ("K", 0.9),
    "a.p.c": ("K", 0.9),
    "equipment": ("K", 0.8),
    # --- Cluster L: Premium Contemporary Designer ---
    "l'agence": ("L", 1.0),
    "alice + olivia": ("L", 1.0),
    "veronica beard": ("L", 1.0),
    "zimmermann": ("L", 0.9),
    "aje": ("L", 1.0),
    "acler": ("L", 1.0),
    "staud": ("L", 0.9),
    "a.l.c.": ("L", 1.0),
    "ganni": ("L", 0.9),
    "sandro": ("L", 0.8),
    "maje": ("L", 0.8),
    "tory burch": ("L", 0.8),
    "kate spade": ("L", 0.7),
    "coach": ("L", 0.7),
    # --- Cluster M: Boho / Indie / Festival ---
    "free people": ("M", 1.0),
    "anthropologie": ("M", 0.9),
    "spell": ("M", 1.0),
    "urban outfitters": ("M", 0.7),
    # --- Cluster P: Department-Store Mainstream ---
    "tommy hilfiger": ("P", 1.0),
    "ralph lauren": ("P", 0.9),
    "calvin klein": ("P", 0.9),
    "michael kors": ("P", 0.9),
    "dkny": ("P", 0.9),
    "nautica": ("P", 0.8),
    "vineyard vines": ("P", 0.8),
    "tommy bahama": ("P", 0.7),
    "brooks brothers": ("P", 0.8),
    "talbots": ("P", 0.8),
    "dickies": ("P", 0.7),
    # --- Cluster Q: Modern Feminine Eco-Chic ---
    "rouje": ("Q", 1.0),
    "faithfull the brand": ("Q", 1.0),
    "christy dawn": ("Q", 1.0),
    "doen": ("Q", 1.0),
    "sezane": ("Q", 1.0),
    "sézane": ("Q", 1.0),
    # --- Cluster R: Trendy Feminine / Going-Out ---
    "princess polly": ("R", 1.0),
    "hello molly": ("R", 1.0),
    "showpo": ("R", 0.9),
    "tiger mist": ("R", 1.0),
    # --- Cluster S: Resort / Coastal Minimal ---
    "sir the label": ("S", 1.0),
    "cult gaia": ("S", 1.0),
    # --- Cluster T: Designer Occasion / Eventwear ---
    "house of cb": ("T", 1.0),
    "nadine merabi": ("T", 1.0),
    "self-portrait": ("T", 1.0),
    "revolve": ("T", 0.7),
    # --- Cluster U: Intimates / Shapewear / Lounge ---
    "skims": ("U", 0.8),
    "spanx": ("U", 1.0),
    "savage x fenty": ("U", 1.0),
    # --- Cluster V: Luxury Designer / Quiet Luxury ---
    "the row": ("V", 1.0),
    "toteme": ("V", 1.0),
    "khaite": ("V", 1.0),
    "max mara": ("V", 1.0),
    # --- Cluster W: Resort Statement / Artsy ---
    "farm rio": ("W", 1.0),
    "johanna ortiz": ("W", 1.0),
    # --- Cluster X: Y2K / Statement Denim / Edgy ---
    "true religion": ("X", 1.0),
    "na-kd": ("X", 0.8),
    "nakd": ("X", 0.8),
    "good american": ("X", 0.8),
}

# Secondary cluster for multi-persona brands
BRAND_SECONDARY_CLUSTER: Dict[str, Tuple[str, float]] = {
    "zara": ("A", 0.5),
    "mango": ("G", 0.5),
    "asos": ("H", 0.5),
    "revolve": ("R", 0.5),
    "cos": ("K", 0.5),
    "skims": ("D", 0.5),
    "free people": ("Q", 0.4),
    "nasty gal": ("R", 0.4),
    "madewell": ("A", 0.4),
    "h&m": ("C", 0.4),
    "levi's": ("G", 0.4),
    # Sandro/Maje: primary L (designer), secondary A (elevated everyday)
    "sandro": ("A", 0.6),
    "maje": ("A", 0.6),
    # Abercrombie post-2020 rebrand: primary C (teen), secondary A (elevated)
    "abercrombie & fitch": ("A", 0.5),
    "abercrombie": ("A", 0.5),
    # Urban Outfitters: primary M (boho), secondary J (youth mall)
    "urban outfitters": ("J", 0.5),
    # Ganni: primary L (designer), secondary K (contemporary staples)
    "ganni": ("K", 0.5),
    # Paige: primary B (denim), secondary A (elevated everyday)
    "paige": ("A", 0.4),
    # Cotton On: primary G (essentials), secondary C (teen mainstream)
    "cotton on": ("C", 0.4),
    # Guess: primary B (denim), secondary X (Y2K/edgy)
    "guess": ("X", 0.4),
}

DEFAULT_CLUSTER = "G"


# =============================================================================
# Onboarding Brand Types → Cluster Mapping
# The 7 high-level brand types shown to users during onboarding, mapped to
# the internal cluster IDs. Used to provide context for the LLM planner.
# =============================================================================

BRAND_TYPE_TO_CLUSTERS: Dict[str, List[str]] = {
    "Everyday & Elevated":      ["A", "G", "K", "Q"],
    "Trendy & Affordable":      ["C", "H", "J", "R"],
    "Streetwear & Denim":       ["B", "X"],
    "Workout & Athleisure":     ["D", "E"],
    "Polished & Professional":  ["A", "K", "L"],
    "Designer & Luxury":        ["L", "T", "V"],
    "Outdoor & Adventure":      ["I"],
}

CLUSTER_TO_BRAND_TYPE: Dict[str, str] = {}
for _btype, _cids in BRAND_TYPE_TO_CLUSTERS.items():
    for _cid in _cids:
        if _cid not in CLUSTER_TO_BRAND_TYPE:
            CLUSTER_TO_BRAND_TYPE[_cid] = _btype


# =============================================================================
# Reverse Mapping: cluster_id -> set of lowercase brand names
# Built at import time from BRAND_CLUSTER_MAP + BRAND_SECONDARY_CLUSTER.
# =============================================================================

CLUSTER_TO_BRANDS: Dict[str, Set[str]] = {}
for _brand, (_cid, _conf) in BRAND_CLUSTER_MAP.items():
    CLUSTER_TO_BRANDS.setdefault(_cid, set()).add(_brand)
for _brand, (_cid, _conf) in BRAND_SECONDARY_CLUSTER.items():
    CLUSTER_TO_BRANDS.setdefault(_cid, set()).add(_brand)


# =============================================================================
# Lookup Functions
# =============================================================================

def get_brand_cluster(brand: str) -> Optional[Tuple[str, float]]:
    """Look up the primary cluster for a brand."""
    return BRAND_CLUSTER_MAP.get(brand.lower().strip())


def get_brand_clusters(brand: str) -> List[Tuple[str, float]]:
    """Get all clusters for a brand (primary + secondary)."""
    key = brand.lower().strip()
    result = []
    primary = BRAND_CLUSTER_MAP.get(key)
    if primary:
        result.append(primary)
    secondary = BRAND_SECONDARY_CLUSTER.get(key)
    if secondary:
        result.append(secondary)
    return result


def get_cluster_traits(cluster_id: str) -> Optional[ClusterTraits]:
    """Get traits for a cluster."""
    return CLUSTER_TRAITS.get(cluster_id)


def get_cluster_for_item(brand: str) -> Optional[str]:
    """Get cluster ID for an item by its brand. Returns None if unmapped."""
    entry = BRAND_CLUSTER_MAP.get(brand.lower().strip())
    return entry[0] if entry else None


def compute_cluster_scores_from_brands(
    preferred_brands: List[str],
) -> Dict[str, float]:
    """
    Compute initial cluster scores from preferred brands (cold-start).

    Returns:
        Dict[cluster_id, score] normalized to 0..1
    """
    scores: Dict[str, float] = {}
    for brand in preferred_brands:
        for cluster_id, confidence in get_brand_clusters(brand):
            scores[cluster_id] = scores.get(cluster_id, 0.0) + confidence

    if scores:
        max_score = max(scores.values())
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
    return scores


def get_cluster_adjacent_brands(preferred_brands: List[str]) -> Set[str]:
    """
    Find all brands in the same clusters as the preferred brands,
    excluding the preferred brands themselves.

    Used for 3-tier candidate bucketing:
      Tier 1 (60%): preferred brands (exact match)
      Tier 2 (30%): cluster-adjacent brands (same style cluster)
      Tier 3 (10%): everything else (discovery)

    Args:
        preferred_brands: User's preferred brand names from onboarding.

    Returns:
        Set of lowercase brand names that are cluster-adjacent but not preferred.
    """
    pref_lower = {b.lower().strip() for b in preferred_brands}

    # Find all clusters the preferred brands belong to (primary + secondary)
    pref_cluster_ids: Set[str] = set()
    for brand in pref_lower:
        for cluster_id, _confidence in get_brand_clusters(brand):
            pref_cluster_ids.add(cluster_id)

    if not pref_cluster_ids:
        return set()

    # Collect all brands in those clusters
    adjacent: Set[str] = set()
    for cid in pref_cluster_ids:
        adjacent.update(CLUSTER_TO_BRANDS.get(cid, set()))

    # Remove the preferred brands themselves
    adjacent -= pref_lower
    return adjacent
