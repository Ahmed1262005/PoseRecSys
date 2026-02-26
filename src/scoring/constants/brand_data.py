"""
Brand reference data: clusters, average prices, and reverse lookups.

Brand clusters group brands by style adjacency — brands in the same
cluster are considered "similar enough" for discovery recommendations.

Synced with the authoritative 22-cluster system in recs/brand_clusters.py.
Sourced from retailer pricing data and editorial clustering.
"""

from typing import Dict, List, Tuple

# ── Average price per brand (USD) ─────────────────────────────────
BRAND_AVG_PRICE: Dict[str, float] = {
    "& Other Stories": 50, "7 For All Mankind": 110, "A.L.C.": 250,
    "A.P.C": 150, "A.P.C.": 150,
    "Abercrombie & Fitch": 35, "Abercrombie": 35,
    "Acler": 280, "Acne Studios": 450, "Adanola": 65,
    "Adidas": 35, "Aeropostale": 20, "AG Jeans": 90, "Agolde": 90,
    "Aje": 220, "ALC": 110, "Alemais": 330, "Alexis": 320,
    "Alice + Olivia": 280, "AllSaints": 80,
    "Alo": 80, "Alo Yoga": 80,
    "American Eagle": 15, "American Eagle Outfitters": 15,
    "Andres Otalora": 300, "Anine Bing": 120, "Ann Taylor": 75,
    "Another Tomorrow": 500, "Anthropologie": 77, "Arc'teryx": 80,
    "Arcina Ori": 280, "Aritzia": 45, "Arket": 55,
    "ASOS": 30, "Asics": 60, "Athleta": 75,
    "Ba&sh": 120, "Banana Republic": 55,
    "Bershka": 20, "Beyond Yoga": 80, "Birkenstock": 100,
    "Boohoo": 20, "Brandy Melville": 23, "Brooks Brothers": 60,
    "Calvin Klein": 40, "Carhartt": 32,
    "Champion": 22, "Charlotte Russe": 15, "Christy Dawn": 200,
    "Cider": 16, "Citizens of Humanity": 150,
    "Club Monaco": 60, "Coach": 200, "Columbia": 28,
    "Converse": 50, "COS": 50, "Cotopaxi": 40, "Cotton On": 15,
    "Crocs": 40, "Cult Gaia": 430, "Cuyana": 120,
    "Dickies": 25, "Diesel": 105, "Dissh": 130, "DKNY": 40,
    "DL1961": 90, "Doen": 250, "Dynamite": 35,
    "Edikted": 15, "Equipment": 200, "Everlane": 50, "Express": 40,
    "Faithful": 180, "Faithfull the Brand": 180,
    "Farm Rio": 80, "Fashion Nova": 25, "FILA": 40,
    "Forever 21": 8, "Forever21": 8, "Frame": 130,
    "Free People": 42, "Free People Movement": 70,
    "Ganni": 200, "Gap": 15, "Garage": 33,
    "Girlfriend Collective": 60, "Good American": 50,
    "Guess": 40,
    "H&M": 18, "Hello Molly": 45, "Hobbs": 120, "Hollister": 18,
    "Hot Topic": 20, "House of CB": 200,
    "Hudson": 115, "Hudson Jeans": 115,
    "J Brand": 120, "J.Crew": 50,
    "Jenni Kayne": 110, "Joanna Ortiz": 800, "Joe's Jeans": 100,
    "Johanna Ortiz": 800,
    "Kallmeyer": 500, "Karen Millen": 100, "Kate Spade": 150,
    "Khaite": 800,
    "L.L.Bean": 55, "L'AGENCE": 150, "Lane Bryant": 35,
    "Lee": 20, "Levi's": 30, "Lucky Brand": 45,
    "Lululemon": 63,
    "Madewell": 100, "Maje": 160, "Mango": 20,
    "Massimo Dutti": 45, "Mavi Jeans": 120, "Max Mara": 800,
    "Meshki": 50, "Michael Kors": 100,
    "Missguided": 13, "Mother": 100, "Mother Denim": 100, "Moussy": 90,
    "Na-Kd": 35, "Nadine Merabi": 350, "Nakd": 35,
    "Nanushka": 200, "Nasty Gal": 20, "Nautica": 40,
    "New Balance": 40, "Nike": 35,
    "Oak + Fort": 35, "Oh Polly": 55, "Old Navy": 12,
    "Other Stories": 50, "Outdoor Voices": 55,
    "PacSun": 11, "Paige": 92, "Patagonia": 50, "PatBo": 350,
    "PLT": 14, "Posse": 330, "PrettyLittleThing": 14,
    "Princess Polly": 45, "Pull&Bear": 30, "Puma": 40,
    "Quince": 25,
    "Rachel Gilbert": 300, "Rag & Bone": 115, "Rails": 160,
    "Ralph Lauren": 100, "Re/Done": 150, "Reebok": 50,
    "Reformation": 90, "Reigning Champ": 100, "Reiss": 130,
    "Revolve": 80, "Rihoas": 30, "Rouje": 85,
    "Sandro": 55, "Savage x Fenty": 40, "Scotch & Soda": 50,
    "Self-Portrait": 350, "Sézane": 40, "Sezane": 40,
    "Shein": 12, "Shona Joy": 240, "Showpo": 50,
    "Silvia Tcherassi": 500, "Simkhai": 300,
    "Sir the label": 350, "Skechers": 30, "Skims": 60,
    "SLVRLAKE": 110, "Spanx": 65, "Spell": 150, "Splendid": 50,
    "Staud": 130, "Stradivarius": 18,
    "Talbots": 60, "Ted Baker": 120, "The Frankie Shop": 110,
    "the frankie shop": 110, "The North Face": 36, "The Row": 1200,
    "Theory": 135, "Tiger Mist": 50,
    "Tommy Bahama": 70, "Tommy Hilfiger": 30,
    "Tory Burch": 250, "Tory Sport": 100, "Torrid": 35,
    "Toteme": 800, "True Religion": 92,
    "UGG": 120, "Under Armour": 35,
    "Uniqlo": 17, "Universal Standard": 65,
    "Urban Outfitters": 25, "Urban Planet": 13,
    "Varley": 85, "Veronica Beard": 350,
    "Vince": 180, "Vineyard Vines": 65, "Vuori": 65,
    "Whistles": 130, "White Fox": 50,
    "White House Black Market": 50, "Windsor": 40,
    "Zara": 30, "Zimmermann": 500,
}

# Case-insensitive price lookup
_BRAND_PRICE_LOWER: Dict[str, float] = {
    k.lower(): v for k, v in BRAND_AVG_PRICE.items()
}


# ── Brand clusters (22-cluster system, synced with recs/brand_clusters.py) ──
# Authoritative source: BRAND_CLUSTER_MAP in recs/brand_clusters.py
# This copy is used by the ProfileScorer for fast cluster lookups.
BRAND_CLUSTERS: Dict[str, List[str]] = {
    # A: Modern Classics / Elevated Everyday
    "A": ["& Other Stories", "Other Stories", "Ann Taylor", "Arket", "Ba&sh",
          "Banana Republic", "Club Monaco", "COS", "Hobbs", "J.Crew",
          "Karen Millen", "Mango", "Massimo Dutti", "Rails", "Reformation",
          "Reiss", "Ted Baker", "Whistles", "White House Black Market"],
    # B: Premium Denim / Casual Americana
    "B": ["7 For All Mankind", "AG Jeans", "Agolde", "Citizens of Humanity",
          "DL1961", "Frame", "Guess", "Hudson", "Hudson Jeans", "J Brand",
          "Joe's Jeans", "Lee", "Levi's", "Lucky Brand", "Madewell",
          "Mavi Jeans", "Mother", "Mother Denim", "Paige", "Rag & Bone",
          "Re/Done", "Scotch & Soda"],
    # C: Mass Casual / Teen Mainstream
    "C": ["Abercrombie", "Abercrombie & Fitch", "Aeropostale",
          "American Eagle", "American Eagle Outfitters", "Brandy Melville",
          "Garage", "Hollister", "Hot Topic", "PacSun", "Pull&Bear"],
    # D: Premium Athleisure / Wellness
    "D": ["Alo", "Alo Yoga", "Athleta", "Beyond Yoga",
          "Free People Movement", "Girlfriend Collective", "Lululemon",
          "Outdoor Voices", "Tory Sport", "Varley", "Vuori"],
    # E: Athletic Heritage / Sportswear
    "E": ["Adidas", "Asics", "Champion", "Converse", "FILA", "New Balance",
          "Nike", "Puma", "Reebok", "Reigning Champ", "Skechers",
          "Under Armour"],
    # F: Comfort Footwear / Mass Sporty
    "F": ["Birkenstock", "Crocs", "UGG"],
    # G: Affordable Essentials / Core Wardrobe
    "G": ["Cotton On", "Everlane", "Gap", "H&M", "Lane Bryant", "Old Navy",
          "Quince", "Splendid", "Torrid", "Uniqlo", "Universal Standard",
          "Zara"],
    # H: Ultra-Fast Fashion / High-Trend
    "H": ["Bershka", "Boohoo", "Cider", "Fashion Nova", "Meshki",
          "Missguided", "Nasty Gal", "Oh Polly", "PLT",
          "PrettyLittleThing", "Shein", "White Fox"],
    # I: Outdoor / Technical Performance
    "I": ["Arc'teryx", "Carhartt", "Columbia", "L.L.Bean", "Patagonia",
          "The North Face"],
    # J: Youth Mall Trend
    "J": ["ASOS", "Charlotte Russe", "Express", "Forever 21", "Forever21",
          "Windsor"],
    # K: Premium Contemporary Staples / Quiet-Lux
    "K": ["A.P.C", "A.P.C.", "Acne Studios", "AllSaints", "Aritzia",
          "Equipment", "the frankie shop", "The Frankie Shop", "Theory",
          "Vince"],
    # L: Premium Contemporary Designer
    "L": ["A.L.C.", "ALC", "Acler", "Aje", "Alice + Olivia", "Coach",
          "Ganni", "Kate Spade", "L'AGENCE", "Maje", "Sandro", "Staud",
          "Tory Burch", "Veronica Beard", "Zimmermann"],
    # M: Boho / Indie / Festival
    "M": ["Anthropologie", "Free People", "Spell", "Urban Outfitters"],
    # P: Department-Store Mainstream / Logo-Lifestyle
    "P": ["Brooks Brothers", "Calvin Klein", "Dickies", "DKNY",
          "Michael Kors", "Nautica", "Ralph Lauren", "Talbots",
          "Tommy Bahama", "Tommy Hilfiger", "Vineyard Vines"],
    # Q: Modern Feminine Eco-Chic
    "Q": ["Christy Dawn", "Doen", "Faithfull the Brand", "Rouje",
          "Sezane", "Sézane"],
    # R: Trendy Feminine / Going-Out Elevated
    "R": ["Hello Molly", "Princess Polly", "Showpo", "Tiger Mist"],
    # S: Resort / Coastal Minimal
    "S": ["Cult Gaia", "Posse", "Sir the label"],
    # T: Designer Occasion / Eventwear
    "T": ["House of CB", "Nadine Merabi", "Revolve", "Self-Portrait"],
    # U: Intimates / Shapewear / Lounge
    "U": ["Savage x Fenty", "Skims", "Spanx"],
    # V: Luxury Designer / Quiet Luxury
    "V": ["Khaite", "Max Mara", "The Row", "Toteme"],
    # W: Resort Statement / Artsy Vacation
    "W": ["Farm Rio", "Johanna Ortiz", "Joanna Ortiz"],
    # X: Y2K / Statement Denim / Edgy
    "X": ["Good American", "Na-Kd", "Nakd", "True Religion"],
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
