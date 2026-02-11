"""
Brand reference data: clusters, average prices, and reverse lookups.

Brand clusters group brands by style adjacency — brands in the same
cluster are considered "similar enough" for discovery recommendations.

Sourced from retailer pricing data and editorial clustering.
"""

from typing import Dict, List, Tuple

# ── Average price per brand (USD) ─────────────────────────────────
BRAND_AVG_PRICE: Dict[str, float] = {
    "& Other Stories": 50, "7 For All Mankind": 110, "A.P.C": 150,
    "Abercrombie & Fitch": 35, "Acne Studios": 450, "Adanola": 65,
    "Adidas": 35, "Aeropostale": 20, "AG Jeans": 90, "Agolde": 90,
    "Aje": 220, "ALC": 110, "Alemais": 330, "Alexis": 320,
    "AllSaints": 80, "Alo Yoga": 80, "American Eagle Outfitters": 15,
    "Andres Otalora": 300, "Anine Bing": 120, "Ann Taylor": 75,
    "Another Tomorrow": 500, "Anthropologie": 77, "Arc'teryx": 80,
    "Arcina Ori": 280, "Aritzia": 45, "Athleta": 75, "Ba&sh": 120,
    "Banana Republic": 55, "Bershka": 20, "Boohoo": 20,
    "Brandy Melville": 23, "Brooks Brothers": 60, "Calvin Klein": 40,
    "Carhartt": 32, "Champion": 22, "Cider": 16,
    "Citizens of Humanity": 150, "Club Monaco": 60, "Columbia": 28,
    "COS": 50, "Cotopaxi": 40, "Cotton On": 15, "Cult Gaia": 430,
    "Cuyana": 120, "Diesel": 105, "Dissh": 130, "DKNY": 40,
    "DL1961": 90, "Dynamite": 35, "Edikted": 15, "Equipment": 200,
    "Everlane": 50, "Express": 40, "Faithful": 180, "Farm Rio": 80,
    "Forever 21": 8, "Frame": 130, "Free People": 42, "Ganni": 200,
    "Gap": 15, "Garage": 33, "Good American": 50, "Guess": 40,
    "H&M": 18, "Hollister": 18, "Hudson": 115, "J.Crew": 50,
    "Jenni Kayne": 110, "Joanna Ortiz": 800, "Joe's Jeans": 100,
    "Kallmeyer": 500, "L'AGENCE": 150, "Lee": 20, "Levi's": 30,
    "Lucky Brand": 45, "Lululemon": 63, "Madewell": 100, "Maje": 160,
    "Mango": 20, "Massimo Dutti": 45, "Mavi Jeans": 120,
    "Missguided": 13, "Mother Denim": 100, "Moussy": 90,
    "Nanushka": 200, "Nasty Gal": 20, "New Balance": 40, "Nike": 35,
    "Oak + Fort": 35, "Old Navy": 12, "Outdoor Voices": 55,
    "PacSun": 11, "Paige": 92, "Patagonia": 50, "PatBo": 350,
    "Posse": 330, "PrettyLittleThing": 14, "Princess Polly": 45,
    "Pull&Bear": 30, "Puma": 40, "Quince": 25, "Rachel Gilbert": 300,
    "Rag & Bone": 115, "Rails": 160, "Ralph Lauren": 100,
    "Re/Done": 150, "Reformation": 90, "Reiss": 130, "Rihoas": 30,
    "Rouje": 85, "Sandro": 55, "Scotch & Soda": 50, "Sézane": 40,
    "Shona Joy": 240, "Silvia Tcherassi": 500, "Simkhai": 300,
    "Sir the label": 350, "Skechers": 30, "Skims": 60,
    "SLVRLAKE": 110, "Staud": 130, "Stradivarius": 18,
    "the frankie shop": 110, "The North Face": 36, "Theory": 135,
    "Tommy Hilfiger": 30, "Toteme": 800, "True Religion": 92,
    "Uniqlo": 17, "Urban Outfitters": 25, "Urban Planet": 13,
    "Vince": 180, "Vuori": 65, "White House Black Market": 50,
    "Zara": 30,
}

# Case-insensitive price lookup
_BRAND_PRICE_LOWER: Dict[str, float] = {
    k.lower(): v for k, v in BRAND_AVG_PRICE.items()
}


# ── Brand clusters (style-adjacent groups) ────────────────────────
BRAND_CLUSTERS: Dict[str, List[str]] = {
    # A: Contemporary classic / smart casual
    "A": ["Aritzia", "Anthropologie", "Banana Republic", "COS", "J.Crew",
          "Sandro", "Sézane", "Club Monaco", "Everlane", "Ann Taylor",
          "Brooks Brothers", "White House Black Market", "& Other Stories",
          "Massimo Dutti", "Ralph Lauren", "Madewell"],
    # B: Premium denim
    "B": ["7 For All Mankind", "AG Jeans", "Citizens of Humanity", "DL1961",
          "Joe's Jeans", "Mavi Jeans", "Paige", "Rag & Bone", "Re/Done",
          "Good American", "Hudson", "Agolde", "Mother Denim", "Moussy"],
    # C: Casual American heritage
    "C": ["Abercrombie & Fitch", "Aeropostale", "American Eagle Outfitters",
          "Levi's", "Gap", "Lee", "Hollister", "Lucky Brand"],
    # E: Athletic performance
    "E": ["Adidas", "New Balance", "Nike", "Puma"],
    # F: Casual athletic
    "F": ["Skechers", "Champion"],
    # G: Fast fashion / everyday
    "G": ["Zara", "Mango", "Express", "Uniqlo", "Oak + Fort", "Old Navy",
          "H&M", "Quince"],
    # I: Outdoor / technical
    "I": ["The North Face", "Patagonia", "Arc'teryx", "Columbia", "Carhartt",
          "Cotopaxi"],
    # J: Young / trend-forward
    "J": ["Garage", "Princess Polly", "Pull&Bear"],
    # K: Elevated contemporary
    "K": ["A.P.C", "Ba&sh", "Cuyana", "Equipment", "Jenni Kayne", "L'AGENCE",
          "Nanushka", "Rails", "Reiss", "the frankie shop", "Anine Bing",
          "AllSaints", "Theory", "Vince", "Dissh"],
    # M: Eclectic / bohemian
    "M": ["Brandy Melville", "Urban Outfitters", "Free People"],
    # P: Mainstream designer
    "P": ["Calvin Klein", "Tommy Hilfiger", "DKNY", "Guess"],
    # Q: Sustainable fashion
    "Q": ["Reformation", "Staud"],
    # S: Minimalist resort
    "S": ["Posse", "Sir the label"],
    # T: Luxury occasion
    "T": ["Alexis", "Andres Otalora", "PatBo", "Aje", "ALC", "Cult Gaia",
          "Joanna Ortiz", "Rachel Gilbert", "Shona Joy", "Simkhai",
          "Silvia Tcherassi"],
    # U: Body-focused
    "U": ["Skims"],
    # W: Statement / bold
    "W": ["Alemais", "Arcina Ori"],
    # X: Edgy denim
    "X": ["Diesel", "True Religion"],
}


# ── Derived lookups (built once at import time) ───────────────────

# brand (lower) -> cluster ID
BRAND_TO_CLUSTER: Dict[str, str] = {}
for _cid, _brands in BRAND_CLUSTERS.items():
    for _b in _brands:
        BRAND_TO_CLUSTER[_b.lower()] = _cid


def get_brand_avg_price(brand: str) -> float:
    """Return average price for a brand (case-insensitive), or 0.0 if unknown."""
    return _BRAND_PRICE_LOWER.get(brand.lower(), 0.0)


def get_brand_cluster(brand: str) -> str:
    """Return cluster ID for a brand (case-insensitive), or '' if unknown."""
    return BRAND_TO_CLUSTER.get(brand.lower(), "")


def derive_price_range(brands: List[str]) -> Tuple[float, float]:
    """
    Derive a price range from selected brands' average prices.

    Uses 0.4x of minimum brand avg as floor, 2.0x of max brand avg as
    ceiling.  Returns (0.0, 0.0) if no brand data found.
    """
    prices = [
        _BRAND_PRICE_LOWER[b.lower()]
        for b in brands
        if b.lower() in _BRAND_PRICE_LOWER
    ]
    if not prices:
        return 0.0, 0.0

    mean_price = sum(prices) / len(prices)
    return round(min(prices) * 0.4, 2), round(max(prices) * 2.0, 2)


def get_user_brand_clusters(preferred_brands: List[str]) -> set:
    """Return set of cluster IDs that the user's preferred brands belong to."""
    clusters = set()
    for b in preferred_brands:
        cid = BRAND_TO_CLUSTER.get(b.lower())
        if cid:
            clusters.add(cid)
    return clusters
