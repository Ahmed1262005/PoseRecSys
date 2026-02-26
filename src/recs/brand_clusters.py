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
# Cluster Complement Prompts (FashionCLIP)
#
# Premade text prompts per cluster x target category for profile-aware
# complement retrieval.  Each prompt is a short visual description tuned
# for FashionCLIP's text-image embedding space (5-12 words, concrete
# garment terms + aesthetic modifiers).  Two prompts per slot — the first
# captures the cluster's core aesthetic, the second broadens it.
#
# Used by OutfitEngine._score_category() to inject 1-2 user-style prompts
# into the complement search alongside the 3 source-derived prompts.
# =============================================================================

CLUSTER_COMPLEMENT_PROMPTS: Dict[str, Dict[str, List[str]]] = {

    # A: Modern Classics / Elevated Everyday
    "A": {
        "tops": [
            "tailored feminine blouse clean lines neutral knitwear women",
            "polished elevated knit top cream navy smart casual women",
        ],
        "bottoms": [
            "tailored trousers clean silhouette neutral women elevated denim",
            "feminine high-waist pants smart casual cream navy classic women",
        ],
        "dresses": [
            "feminine wearable midi dress clean silhouette tailored neutral women",
            "elevated everyday wrap dress smart casual knit women",
        ],
        "outerwear": [
            "polished blazer tailored neutral clean silhouette women",
            "feminine cardigan knitwear elevated cream camel smart casual women",
        ],
    },

    # B: Premium Denim / Casual Americana
    "B": {
        "tops": [
            "off-duty cool tee women relaxed basics earth tones quality",
            "casual leather jacket friendly top women premium denim style",
        ],
        "bottoms": [
            "premium straight-leg jeans women dark wash quality denim",
            "relaxed wide-leg denim women earth tones off-duty cool",
        ],
        "dresses": [
            "casual shirt dress women denim-friendly earth tones relaxed",
            "off-duty cool midi dress women leather-trim casual americana",
        ],
        "outerwear": [
            "leather jacket women premium quality off-duty cool black",
            "denim jacket women relaxed fit earth tones casual americana",
        ],
    },

    # C: Mass Casual / Teen Mainstream
    "C": {
        "tops": [
            "casual graphic tee women logo basics relaxed everyday",
            "simple hoodie women campus casual comfortable basics",
        ],
        "bottoms": [
            "casual denim jeans women everyday basics comfortable fit",
            "simple joggers women campus casual relaxed everyday",
        ],
        "dresses": [
            "simple casual dress women everyday basics comfortable",
            "cute basic mini dress women campus casual relaxed",
        ],
        "outerwear": [
            "casual hoodie jacket women basics comfortable everyday",
            "simple denim jacket women relaxed casual basics",
        ],
    },

    # D: Premium Athleisure / Wellness
    "D": {
        "tops": [
            "premium sports bra women matching set muted neutral performance",
            "soft lounge top women athleisure neutral pastel comfortable",
        ],
        "bottoms": [
            "high-waist leggings women performance neutral body-contouring",
            "matching set joggers women premium athleisure muted pastel",
        ],
        "dresses": [
            "athletic midi dress women soft neutral athleisure comfortable",
            "sporty casual dress women performance fabric muted neutral",
        ],
        "outerwear": [
            "performance zip jacket women athleisure neutral premium",
            "soft cropped hoodie women lounge neutral pastel comfortable",
        ],
    },

    # E: Athletic Heritage / Sportswear
    "E": {
        "tops": [
            "sporty heritage tee women logo athletic black white",
            "athletic crop top women performance sporty bold primary colors",
        ],
        "bottoms": [
            "sporty joggers women athletic heritage black grey logo",
            "track pants women sporty performance bold primary colors",
        ],
        "dresses": [
            "sporty casual dress women athletic comfortable everyday",
            "athletic heritage dress women minimal sporty black white",
        ],
        "outerwear": [
            "sporty windbreaker women athletic heritage logo bold",
            "athletic zip-up jacket women sporty performance black grey",
        ],
    },

    # G: Affordable Essentials / Core Wardrobe
    "G": {
        "tops": [
            "simple tee women basics capsule wardrobe neutral clean",
            "everyday knit top women muted colors minimal effort basic",
        ],
        "bottoms": [
            "simple trousers women basics capsule neutral everyday",
            "basic denim women muted tones clean everyday wardrobe",
        ],
        "dresses": [
            "simple everyday dress women basics muted neutral capsule",
            "easy casual dress women minimal basics clean wardrobe",
        ],
        "outerwear": [
            "workhorse jacket women basics neutral capsule practical",
            "simple coat women muted tones everyday essential clean",
        ],
    },

    # H: Ultra-Fast Fashion / High-Trend
    "H": {
        "tops": [
            "trendy crop top women cutouts bodycon going-out bold colors",
            "party top women micro-trend loud print high-contrast sexy",
        ],
        "bottoms": [
            "trendy mini skirt women bodycon going-out loud bold colors",
            "low-rise pants women micro-trend party high-contrast fitted",
        ],
        "dresses": [
            "bodycon mini dress women party cutouts going-out bold",
            "trendy going-out dress women loud print sexy fitted",
        ],
        "outerwear": [
            "cropped jacket women trendy bold going-out statement",
            "faux fur coat women party loud high-contrast glamour",
        ],
    },

    # I: Outdoor / Technical Performance
    "I": {
        "tops": [
            "technical base layer women outdoor performance earth tones",
            "fleece pullover women hiking practical navy grey functional",
        ],
        "bottoms": [
            "hiking pants women technical outdoor earth tones practical",
            "performance cargo pants women outdoor functional durable",
        ],
        "dresses": [
            "technical outdoor dress women practical earth tones functional",
            "trail-friendly dress women performance durable comfortable",
        ],
        "outerwear": [
            "technical shell jacket women outdoor waterproof earth tones",
            "puffer vest women hiking practical performance black navy",
        ],
    },

    # J: Youth Mall Trend
    "J": {
        "tops": [
            "cute trend top women fitted going-out playful prints",
            "going-out crop top women denim-friendly mini cute seasonal",
        ],
        "bottoms": [
            "denim mini skirt women cute trend playful fitted",
            "trendy fitted jeans women going-out cute seasonal",
        ],
        "dresses": [
            "cute mini dress women trend going-out playful prints",
            "flirty casual dress women fitted seasonal fun young",
        ],
        "outerwear": [
            "cute cropped jacket women trend denim playful fun",
            "trendy oversized blazer women going-out seasonal cute",
        ],
    },

    # K: Premium Contemporary Staples / Quiet-Lux
    "K": {
        "tops": [
            "premium knit top women clean lines neutral minimal elevated tee",
            "sleek tailored blouse women understated cream black quiet luxury",
        ],
        "bottoms": [
            "tailored wide-leg trousers women premium neutral minimal clean",
            "sleek structured pants women black cream quiet luxury elevated",
        ],
        "dresses": [
            "minimal tailored dress women premium neutral clean silhouette",
            "sleek midi dress women elevated understated quiet luxury black",
        ],
        "outerwear": [
            "structured blazer women premium neutral minimal tailored coat",
            "oversized wool coat women quiet luxury elevated black camel",
        ],
    },

    # L: Premium Contemporary Designer
    "L": {
        "tops": [
            "structured designer blouse women tailored sophisticated neutral",
            "elevated fashion top women statement refined polished city",
        ],
        "bottoms": [
            "tailored designer trousers women structured sophisticated neutral",
            "elevated denim women refined polished premium dark wash",
        ],
        "dresses": [
            "refined designer dress women structured sophisticated neutral",
            "elevated midi dress women polished fashion statement city",
        ],
        "outerwear": [
            "statement coat women structured designer sophisticated neutral",
            "tailored designer blazer women elevated polished premium",
        ],
    },

    # M: Boho / Indie / Festival
    "M": {
        "tops": [
            "boho oversized knit women earthy prints layered textured",
            "flowy bohemian blouse women western festival vintage artsy",
        ],
        "bottoms": [
            "boho wide-leg pants women earthy prints flowy textured",
            "vintage denim shorts women festival western bohemian artsy",
        ],
        "dresses": [
            "boho maxi dress women flowy prints earthy vintage festival",
            "bohemian midi dress women western layered textured artsy",
        ],
        "outerwear": [
            "boho oversized cardigan women earthy knit layered textured",
            "vintage denim jacket women festival western bohemian artsy",
        ],
    },

    # P: Department-Store Mainstream / Logo-Lifestyle
    "P": {
        "tops": [
            "classic logo tee women recognizable brand casual neutral",
            "safe casual blouse women preppy navy red white basic",
        ],
        "bottoms": [
            "classic straight jeans women safe casual neutral everyday",
            "casual chinos women preppy neutral classic mainstream",
        ],
        "dresses": [
            "casual classic dress women safe styling neutral everyday",
            "simple wrap dress women mainstream preppy navy red basic",
        ],
        "outerwear": [
            "classic trench coat women mainstream safe neutral logo",
            "casual blazer women preppy neutral classic safe everyday",
        ],
    },

    # Q: Modern Feminine Eco-Chic
    "Q": {
        "tops": [
            "flattering feminine top women romantic minimal soft neutral",
            "fitted bodice top women sustainable silhouette warm floral",
        ],
        "bottoms": [
            "flattering midi skirt women feminine romantic neutral warm",
            "sustainable wide-leg pants women minimal feminine soft flowy",
        ],
        "dresses": [
            "flattering feminine dress women romantic minimal soft floral",
            "sustainable midi dress women fitted bodice silhouette warm",
        ],
        "outerwear": [
            "feminine structured blazer women romantic neutral warm tones",
            "soft knit cardigan women minimal sustainable eco-chic warm",
        ],
    },

    # R: Trendy Feminine / Going-Out Elevated
    "R": {
        "tops": [
            "corset top women cute photo-ready going-out prints fitted",
            "trendy feminine crop top women sets mini vacation night-out",
        ],
        "bottoms": [
            "trendy mini skirt women cute photo-ready prints fitted",
            "going-out pants women feminine sets night-out bodycon",
        ],
        "dresses": [
            "cute mini dress women photo-ready prints going-out fitted",
            "trendy feminine dress women vacation night-out sets corset",
        ],
        "outerwear": [
            "cropped blazer women trendy feminine going-out cute sets",
            "statement jacket women photo-ready night-out prints fitted",
        ],
    },

    # S: Resort / Coastal Minimal
    "S": {
        "tops": [
            "linen camisole women resort minimal clean white cream",
            "slip-style top women coastal curated slinky black pastel",
        ],
        "bottoms": [
            "linen wide-leg pants women resort minimal white cream tan",
            "clean tailored shorts women coastal curated neutral pastel",
        ],
        "dresses": [
            "slip dress women resort minimal slinky cream black pastel",
            "linen midi dress women coastal curated clean vacation white",
        ],
        "outerwear": [
            "linen blazer women resort minimal clean white cream neutral",
            "lightweight cover-up women coastal curated pastel vacation",
        ],
    },

    # T: Designer Occasion / Eventwear
    "T": {
        "tops": [
            "statement cocktail top women structured bold jewel tones",
            "sculpted occasion blouse women elegant evening black event",
        ],
        "bottoms": [
            "structured wide-leg trousers women event elegant jewel tones",
            "evening tailored pants women cocktail sophisticated black",
        ],
        "dresses": [
            "cocktail dress women statement structured bold jewel tones",
            "event gown women elegant sculpted evening sophisticated black",
        ],
        "outerwear": [
            "statement evening coat women structured bold jewel tones",
            "elegant tailored jacket women cocktail event sophisticated",
        ],
    },

    # V: Luxury Designer / Quiet Luxury
    "V": {
        "tops": [
            "impeccable cashmere top women luxury tailored neutral subdued",
            "investment blouse women refined silhouette quiet minimal cream",
        ],
        "bottoms": [
            "impeccable tailored trousers women luxury neutral subdued",
            "investment wide-leg pants women refined quiet luxury black",
        ],
        "dresses": [
            "impeccable tailored dress women luxury neutral refined subdued",
            "investment midi dress women quiet luxury silhouette cream black",
        ],
        "outerwear": [
            "impeccable wool coat women luxury tailored neutral investment",
            "refined cashmere jacket women quiet luxury subdued cream camel",
        ],
    },

    # W: Resort Statement / Artsy Vacation
    "W": {
        "tops": [
            "artful printed top women resort bold saturated airy vacation",
            "statement set top women vacation colorful flowy photograph-ready",
        ],
        "bottoms": [
            "artful printed pants women resort bold saturated airy flowy",
            "vacation wide-leg trousers women colorful statement prints",
        ],
        "dresses": [
            "artful printed dress women resort bold saturated airy vacation",
            "statement maxi dress women vacation colorful flowy photograph",
        ],
        "outerwear": [
            "artful printed kimono women resort bold saturated airy vacation",
            "statement lightweight jacket women vacation colorful prints",
        ],
    },

    # X: Y2K / Statement Denim / Edgy
    "X": {
        "tops": [
            "edgy graphic tee women y2k streetwear bold black statement",
            "distressed crop top women loud logo attitude high-contrast",
        ],
        "bottoms": [
            "statement jeans women distressed low-rise bold wash y2k edgy",
            "edgy cargo pants women streetwear loud black high-contrast",
        ],
        "dresses": [
            "edgy mini dress women y2k streetwear bold black statement",
            "distressed denim dress women loud attitude high-contrast",
        ],
        "outerwear": [
            "edgy leather jacket women y2k streetwear bold black distressed",
            "statement denim jacket women loud graphic y2k high-contrast",
        ],
    },
}


# =============================================================================
# Style Persona -> Cluster Fallback Mapping
#
# When a user has style_persona set but few/no preferred_brands, map their
# stated style to the most representative brand clusters.  Two clusters
# per persona — first is primary match, second broadens the aesthetic.
# =============================================================================

PERSONA_TO_CLUSTERS: Dict[str, List[str]] = {
    "classic":          ["A", "K"],     # Modern Classics, Quiet-Lux
    "minimal":          ["K", "G"],     # Quiet-Lux, Essentials
    "minimalist":       ["K", "G"],     # alias
    "elegant":          ["L", "T"],     # Premium Designer, Eventwear
    "trendy":           ["H", "J"],     # Ultra-Fast, Youth Mall
    "casual":           ["C", "G"],     # Mass Casual, Essentials
    "streetwear":       ["X", "E"],     # Y2K/Edgy, Athletic Heritage
    "sporty":           ["D", "E"],     # Premium Athleisure, Athletic
    "bohemian":         ["M", "W"],     # Boho/Indie, Resort Statement
    "boho":             ["M", "W"],     # alias
    "romantic":         ["Q", "R"],     # Eco-Chic, Going-Out Elevated
    "edgy":             ["X", "B"],     # Y2K/Edgy, Premium Denim
    "preppy":           ["A", "P"],     # Modern Classics, Mainstream
    "glamorous":        ["T", "R"],     # Eventwear, Going-Out
    "glam":             ["T", "R"],     # alias
    "athleisure":       ["D", "E"],     # Premium Athleisure, Athletic
    "vintage":          ["M", "C"],     # Boho/Indie, Mass Casual
    "business casual":  ["A", "K"],     # Modern Classics, Quiet-Lux
    "business-casual":  ["A", "K"],     # alias
}


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
