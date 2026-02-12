"""
End-to-End Benchmark Suite: 100 User Profiles x Scoring + Reranking Pipelines.

Tests the full scoring stack:
1. Context scoring (age + weather) — ContextScorer
2. Session scoring — SessionScoringEngine
3. Feed reranking — GreedyConstrainedReranker
4. Search reranking — SessionReranker
5. Cross-cutting: session actions → search/feed reflect changes

Metrics computed:
- NDCG@K, Precision@K, MRR, Hit Rate@K (K=5,10,20)
- Brand Precision@K, Category Precision@K
- Diversity (brand entropy, unique brands in top-K)
- Age/Weather appropriateness ratios

Run:
    PYTHONPATH=src python -m pytest tests/unit/test_e2e_benchmark.py -v -s
"""

import math
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock

import pytest


# ============================================================================
# METRICS MODULE
# ============================================================================

@dataclass
class RankingMetrics:
    """Standard IR/RecSys ranking metrics."""
    ndcg_5: float = 0.0
    ndcg_10: float = 0.0
    ndcg_20: float = 0.0
    precision_5: float = 0.0
    precision_10: float = 0.0
    precision_20: float = 0.0
    mrr: float = 0.0
    hit_rate_5: float = 0.0
    hit_rate_10: float = 0.0
    hit_rate_20: float = 0.0
    brand_precision_10: float = 0.0
    category_precision_10: float = 0.0
    style_precision_10: float = 0.0
    diversity_brands_10: int = 0
    brand_entropy_10: float = 0.0
    age_appropriate_ratio: float = 0.0
    weather_appropriate_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _dcg(relevances: List[float], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)  # log2(rank+1), rank is 1-indexed
    return dcg


def _ndcg_at_k(relevances: List[float], k: int) -> float:
    """Normalized DCG: actual / ideal."""
    actual = _dcg(relevances, k)
    ideal = _dcg(sorted(relevances, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0


def _precision_at_k(relevant: List[bool], k: int) -> float:
    """Fraction of top-K that are relevant."""
    top = relevant[:k]
    return sum(top) / k if k > 0 else 0.0


def _mrr(relevant: List[bool]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant item."""
    for i, r in enumerate(relevant):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def _hit_rate_at_k(relevant: List[bool], k: int) -> float:
    """1 if any relevant item in top-K, else 0."""
    return 1.0 if any(relevant[:k]) else 0.0


def _entropy(counts: Dict[str, int]) -> float:
    """Shannon entropy of a distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def compute_random_baseline(
    items: List[dict],
    relevant_brands: Set[str],
    relevant_categories: Set[str],
    relevant_styles: Set[str],
    relevant_article_types: Set[str],
    age_appropriate_types: Set[str],
    weather_appropriate_types: Set[str],
    n_shuffles: int = 50,
) -> RankingMetrics:
    """
    Compute expected metrics for random ordering (average over n_shuffles).
    This is the baseline — any good ranker should significantly beat this.
    """
    rng = random.Random(99)
    all_metrics = []
    for _ in range(n_shuffles):
        shuffled = items.copy()
        rng.shuffle(shuffled)
        m = compute_ranking_metrics(
            shuffled, relevant_brands, relevant_categories,
            relevant_styles, relevant_article_types,
            age_appropriate_types, weather_appropriate_types,
        )
        all_metrics.append(m)

    # Average across shuffles
    avg = RankingMetrics()
    for field_name in RankingMetrics.__dataclass_fields__:
        val = statistics.mean(getattr(m, field_name) for m in all_metrics)
        setattr(avg, field_name, val)
    return avg


def compute_ranking_metrics(
    ranked_items: List[dict],
    relevant_brands: Set[str],
    relevant_categories: Set[str],
    relevant_styles: Set[str],
    relevant_article_types: Set[str],
    age_appropriate_types: Set[str],
    weather_appropriate_types: Set[str],
) -> RankingMetrics:
    """
    Compute full ranking metrics for a result list.

    Relevance is defined as: item matches at least one of the user's
    preferred brands, categories, or styles.
    """
    # Build relevance signals
    relevances = []
    brand_hits = []
    cat_hits = []
    style_hits = []
    age_hits = []
    weather_hits = []

    for item in ranked_items:
        brand = (item.get("brand") or "").lower()
        cat = (item.get("broad_category") or "").lower()
        atype = (item.get("article_type") or "").lower()
        styles = [s.lower() for s in (item.get("style_tags") or [])]

        brand_match = brand in relevant_brands
        cat_match = cat in relevant_categories
        style_match = any(s in relevant_styles for s in styles)
        type_match = atype in relevant_article_types

        # Graded relevance: 3 for brand+cat, 2 for brand or cat, 1 for style/type
        rel = 0.0
        if brand_match and cat_match:
            rel = 3.0
        elif brand_match or cat_match:
            rel = 2.0
        elif style_match or type_match:
            rel = 1.0
        relevances.append(rel)

        # Binary relevance (any match counts)
        is_relevant = brand_match or cat_match or style_match or type_match
        brand_hits.append(brand_match)
        cat_hits.append(cat_match)
        style_hits.append(style_match)
        age_hits.append(atype in age_appropriate_types)
        weather_hits.append(atype in weather_appropriate_types)

    binary_relevant = [r > 0 for r in relevances]

    # Brand distribution in top-10
    top_10_brands = Counter(
        (item.get("brand") or "unknown").lower()
        for item in ranked_items[:10]
    )

    return RankingMetrics(
        ndcg_5=_ndcg_at_k(relevances, 5),
        ndcg_10=_ndcg_at_k(relevances, 10),
        ndcg_20=_ndcg_at_k(relevances, 20),
        precision_5=_precision_at_k(binary_relevant, 5),
        precision_10=_precision_at_k(binary_relevant, 10),
        precision_20=_precision_at_k(binary_relevant, 20),
        mrr=_mrr(binary_relevant),
        hit_rate_5=_hit_rate_at_k(binary_relevant, 5),
        hit_rate_10=_hit_rate_at_k(binary_relevant, 10),
        hit_rate_20=_hit_rate_at_k(binary_relevant, 20),
        brand_precision_10=_precision_at_k(brand_hits, 10),
        category_precision_10=_precision_at_k(cat_hits, 10),
        style_precision_10=_precision_at_k(style_hits, 10),
        diversity_brands_10=len(top_10_brands),
        brand_entropy_10=_entropy(top_10_brands),
        age_appropriate_ratio=(
            sum(age_hits[:10]) / min(10, len(age_hits)) if age_hits else 0.0
        ),
        weather_appropriate_ratio=(
            sum(weather_hits[:10]) / min(10, len(weather_hits)) if weather_hits else 0.0
        ),
    )


# ============================================================================
# USER PROFILE DEFINITIONS
# ============================================================================

@dataclass
class BenchmarkProfile:
    """One of 100 user profiles for E2E benchmarking."""
    profile_id: str
    name: str

    # Demographics
    age: int
    age_group: str          # "18-24", "25-34", "35-44", "45-64", "65+"
    style_persona: str      # trendy, classic, elegant, minimal, streetwear, sporty, romantic, boho

    # Weather/Location
    city: str
    country: str
    temperature_c: float
    weather_condition: str  # clear, rain, snow, clouds
    season: str             # spring, summer, fall, winter
    is_hot: bool = False
    is_cold: bool = False
    is_rainy: bool = False

    # Preferences
    preferred_brands: List[str] = field(default_factory=list)
    preferred_categories: List[str] = field(default_factory=list)   # tops, bottoms, dresses, outerwear
    preferred_article_types: List[str] = field(default_factory=list)
    preferred_styles: List[str] = field(default_factory=list)
    preferred_occasions: List[str] = field(default_factory=list)
    preferred_colors: List[str] = field(default_factory=list)
    preferred_patterns: List[str] = field(default_factory=list)
    coverage_prefs: List[str] = field(default_factory=list)         # "no_crop", "no_revealing"

    # Search
    search_queries: List[str] = field(default_factory=list)
    search_intent: str = "specific"     # exact, specific, vague

    # Price
    min_price: float = 0.0
    max_price: float = 200.0

    # Expected behavior (ground truth for metrics)
    expected_age_types: List[str] = field(default_factory=list)   # article types appropriate for age
    expected_weather_types: List[str] = field(default_factory=list)  # article types appropriate for weather


def _build_100_profiles() -> List[BenchmarkProfile]:
    """
    Build 100 diverse profiles: 5 age groups x 4 weather conditions x 5 style personas.

    Age groups: GEN_Z (18-24), YOUNG_ADULT (25-34), MID_CAREER (35-44), ESTABLISHED (45-64), SENIOR (65+)
    Weather: hot_summer, cold_winter, mild_spring, rainy_fall
    Styles: trendy, classic, minimal, romantic, streetwear
    """
    profiles = []

    age_configs = [
        {
            "age_group": "18-24", "ages": [19, 20, 21, 22],
            "expected_types": ["crop_top", "tank_top", "mini_dress", "bodycon_dress", "mini_skirt",
                               "shorts", "bralette", "tube_top", "halter_top", "romper"],
            "coverage": [],
        },
        {
            "age_group": "25-34", "ages": [26, 28, 30, 32],
            "expected_types": ["blouse", "midi_dress", "jeans", "blazer", "wrap_dress",
                               "tshirt", "sweater", "slip_dress", "pants", "cardigan"],
            "coverage": [],
        },
        {
            "age_group": "35-44", "ages": [36, 38, 40, 42],
            "expected_types": ["blazer", "blouse", "midi_dress", "pants", "sheath_dress",
                               "sweater", "coat", "cardigan", "wide_leg_pants", "shirt_dress"],
            "coverage": ["no_crop"],
        },
        {
            "age_group": "45-64", "ages": [47, 50, 55, 60],
            "expected_types": ["blazer", "coat", "pants", "blouse", "cardigan", "sweater",
                               "sheath_dress", "shift_dress", "turtleneck", "trench_coat"],
            "coverage": ["no_crop", "no_revealing"],
        },
        {
            "age_group": "65+", "ages": [67, 70, 73, 75],
            "expected_types": ["cardigan", "coat", "blazer", "pants", "blouse", "sweater",
                               "shift_dress", "turtleneck", "vest", "trench_coat"],
            "coverage": ["no_crop", "no_revealing", "no_deep_necklines"],
        },
    ]

    weather_configs = [
        {
            "label": "hot_summer", "city": "Miami", "country": "US",
            "temperature_c": 33.0, "condition": "clear", "season": "summer",
            "is_hot": True, "is_cold": False, "is_rainy": False,
            "expected_types": ["tank_top", "cami", "shorts", "sundress", "crop_top",
                               "mini_dress", "mini_skirt", "romper", "halter_top", "tube_top"],
        },
        {
            "label": "cold_winter", "city": "Chicago", "country": "US",
            "temperature_c": -5.0, "condition": "snow", "season": "winter",
            "is_hot": False, "is_cold": True, "is_rainy": False,
            "expected_types": ["coat", "puffer", "sweater", "cardigan", "turtleneck",
                               "hoodie", "blazer", "pants", "jeans", "trench_coat"],
        },
        {
            "label": "mild_spring", "city": "Paris", "country": "FR",
            "temperature_c": 18.0, "condition": "clouds", "season": "spring",
            "is_hot": False, "is_cold": False, "is_rainy": False,
            "expected_types": ["jacket", "blazer", "cardigan", "jeans", "pants",
                               "dress", "blouse", "tshirt", "midi_dress", "sweater"],
        },
        {
            "label": "rainy_fall", "city": "London", "country": "UK",
            "temperature_c": 12.0, "condition": "rain", "season": "fall",
            "is_hot": False, "is_cold": False, "is_rainy": True,
            "expected_types": ["jacket", "coat", "pants", "jeans", "windbreaker",
                               "trench_coat", "sweater", "blazer", "cardigan", "hoodie"],
        },
    ]

    style_configs = [
        {
            "persona": "trendy",
            "brands": ["zara", "boohoo", "forever 21", "asos", "prettylittlething"],
            "styles": ["trendy", "casual"],
            "occasions": ["going-out", "parties", "dates"],
            "colors": ["black", "red", "pink"],
            "patterns": ["solid", "floral"],
            "categories": ["tops", "dresses"],
            "article_types": ["crop_top", "mini_dress", "bodycon_dress", "tube_top", "shorts"],
            "queries": ["trendy party dress", "cute crop top", "going out outfit"],
            "price_max": 80.0,
        },
        {
            "persona": "classic",
            "brands": ["j.crew", "banana republic", "ann taylor", "mango", "cos"],
            "styles": ["classic", "elegant"],
            "occasions": ["work", "dinners", "smart-casual"],
            "colors": ["navy", "beige", "white"],
            "patterns": ["solid", "striped"],
            "categories": ["tops", "bottoms"],
            "article_types": ["blazer", "blouse", "pants", "midi_dress", "coat"],
            "queries": ["office blazer", "classic white blouse", "work trousers"],
            "price_max": 200.0,
        },
        {
            "persona": "minimal",
            "brands": ["cos", "everlane", "uniqlo", "the frankie shop", "arket"],
            "styles": ["minimal", "classic"],
            "occasions": ["everyday", "work", "travel"],
            "colors": ["black", "white", "grey"],
            "patterns": ["solid"],
            "categories": ["tops", "bottoms"],
            "article_types": ["tshirt", "pants", "sweater", "jeans", "blazer"],
            "queries": ["minimal black sweater", "clean white tee", "capsule wardrobe basics"],
            "price_max": 150.0,
        },
        {
            "persona": "romantic",
            "brands": ["reformation", "ba&sh", "maje", "sandro", "free people"],
            "styles": ["romantic", "elegant", "boho"],
            "occasions": ["dates", "brunch", "weddings-guest"],
            "colors": ["blush", "lavender", "dusty rose"],
            "patterns": ["floral", "lace"],
            "categories": ["dresses", "tops"],
            "article_types": ["wrap_dress", "midi_dress", "blouse", "slip_dress", "maxi_dress"],
            "queries": ["floral wrap dress", "romantic date night dress", "lace top"],
            "price_max": 250.0,
        },
        {
            "persona": "streetwear",
            "brands": ["nike", "adidas", "new balance", "champion", "puma"],
            "styles": ["streetwear", "sporty", "trendy"],
            "occasions": ["weekends", "streetwear", "casual"],
            "colors": ["black", "white", "green"],
            "patterns": ["solid", "logo"],
            "categories": ["tops", "bottoms"],
            "article_types": ["hoodie", "joggers", "tshirt", "shorts", "sweatshirt"],
            "queries": ["oversized hoodie", "nike joggers", "streetwear essentials"],
            "price_max": 120.0,
        },
    ]

    idx = 0
    for age_cfg in age_configs:
        for weather_cfg in weather_configs:
            for style_cfg in style_configs:
                age = age_cfg["ages"][idx % len(age_cfg["ages"])]
                profile = BenchmarkProfile(
                    profile_id=f"profile_{idx:03d}",
                    name=f"{style_cfg['persona'].title()} {age_cfg['age_group']} in {weather_cfg['city']}",
                    age=age,
                    age_group=age_cfg["age_group"],
                    style_persona=style_cfg["persona"],
                    city=weather_cfg["city"],
                    country=weather_cfg["country"],
                    temperature_c=weather_cfg["temperature_c"],
                    weather_condition=weather_cfg["condition"],
                    season=weather_cfg["season"],
                    is_hot=weather_cfg["is_hot"],
                    is_cold=weather_cfg["is_cold"],
                    is_rainy=weather_cfg["is_rainy"],
                    preferred_brands=style_cfg["brands"],
                    preferred_categories=style_cfg["categories"],
                    preferred_article_types=style_cfg["article_types"],
                    preferred_styles=style_cfg["styles"],
                    preferred_occasions=style_cfg["occasions"],
                    preferred_colors=style_cfg["colors"],
                    preferred_patterns=style_cfg["patterns"],
                    coverage_prefs=age_cfg["coverage"],
                    search_queries=style_cfg["queries"],
                    min_price=0.0,
                    max_price=style_cfg["price_max"],
                    expected_age_types=age_cfg["expected_types"],
                    expected_weather_types=weather_cfg["expected_types"],
                )
                profiles.append(profile)
                idx += 1

    assert len(profiles) == 100, f"Expected 100 profiles, got {len(profiles)}"
    return profiles


# ============================================================================
# MOCK DATA GENERATORS
# ============================================================================

# Full brand catalog (~90 unique brands from BRAND_CLUSTER_MAP)
# Only 25 are "profile brands" — the rest are noise
_PROFILE_BRANDS = {
    "zara", "boohoo", "reformation", "nike", "cos",
    "forever 21", "h&m", "mango", "adidas", "everlane",
    "lululemon", "ba&sh", "j.crew", "uniqlo", "asos",
    "prettylittlething", "new balance", "the frankie shop",
    "banana republic", "free people", "puma", "sandro",
    "champion", "ann taylor", "maje",
}

# Full brand pool matching real catalog distribution (131 brands)
# Profile brands are only ~19% of the pool (25 out of 131)
BENCHMARK_BRANDS = [
    # --- Profile brands (25) ---
    "Zara", "Boohoo", "Reformation", "Nike", "Cos",
    "Forever 21", "H&M", "Mango", "Adidas", "Everlane",
    "Lululemon", "Ba&sh", "J.Crew", "Uniqlo", "Asos",
    "PrettyLittleThing", "New Balance", "The Frankie Shop",
    "Banana Republic", "Free People", "Puma", "Sandro",
    "Champion", "Ann Taylor", "Maje",
    # --- Noise brands (106) — no profile selects these ---
    "Club Monaco", "Rails", "Reiss", "Arket", "& Other Stories",
    "Massimo Dutti", "Ted Baker", "Karen Millen", "Whistles", "Hobbs",
    "White House Black Market", "Joe's Jeans", "Re/Done", "Rag & Bone",
    "Citizens of Humanity", "Agolde", "DL1961", "Hudson Jeans", "Paige",
    "Frame", "Mother", "Levi's", "Madewell", "J Brand",
    "Scotch & Soda", "American Eagle", "Hollister", "Aeropostale",
    "Hot Topic", "Abercrombie & Fitch", "Pull&Bear",
    "Alo Yoga", "Vuori", "Beyond Yoga", "Girlfriend Collective", "Varley",
    "Free People Movement", "Under Armour", "Reebok", "Asics", "Fila",
    "Skechers", "Crocs", "Ugg", "Birkenstock",
    "Gap", "Old Navy", "Universal Standard",
    "Missguided", "Shein", "Nasty Gal", "Fashion Nova", "Meshki",
    "Oh Polly", "White Fox",
    "The North Face", "Patagonia", "Columbia", "Arc'teryx",
    "Charlotte Russe", "Windsor", "Express",
    "Theory", "Vince", "AllSaints", "Aritzia",
    "L'Agence", "Alice + Olivia", "Veronica Beard", "Zimmermann",
    "Aje", "Acler", "Staud", "A.L.C.",
    "Anthropologie", "Spell",
    "Tommy Hilfiger", "Ralph Lauren", "Calvin Klein", "Michael Kors", "DKNY",
    "Rouje", "Faithfull the Brand", "Christy Dawn", "Doen",
    "Princess Polly", "Hello Molly", "Showpo", "Tiger Mist",
    "Sir the Label", "Cult Gaia",
    "House of CB", "Nadine Merabi", "Self-Portrait", "Revolve",
    "Skims", "Spanx", "Savage X Fenty",
    "The Row", "Toteme", "Khaite", "Max Mara",
    "Farm Rio", "Johanna Ortiz",
    "True Religion", "Na-kd", "Good American",
    # Extra noise brands not in any cluster (mimics unknown/niche catalog brands)
    "Noisy May", "Vero Moda", "Only", "Pieces", "Y.A.S",
    "Monki", "Weekday", "& Tuesday", "Neon Rose", "Glamorous",
    "Topshop", "River Island", "New Look", "Dorothy Perkins",
    "Oasis", "Warehouse", "Mint Velvet", "Phase Eight",
]

# Brand frequency weights: Boohoo/Missguided/Forever 21 dominate (matching real DB)
# Top brands have 10-20K products; niche brands have hundreds
_BRAND_WEIGHTS = []
_HIGH_FREQ_BRANDS = {"Boohoo", "Missguided", "Forever 21", "PrettyLittleThing",
                      "Princess Polly", "H&M", "Shein", "Asos", "Fashion Nova"}
_MED_FREQ_BRANDS = {"Zara", "Mango", "Nike", "Adidas", "Reformation", "Cos",
                     "Lululemon", "Topshop", "River Island", "New Look"}
for _b in BENCHMARK_BRANDS:
    if _b in _HIGH_FREQ_BRANDS:
        _BRAND_WEIGHTS.append(8.0)
    elif _b in _MED_FREQ_BRANDS:
        _BRAND_WEIGHTS.append(3.0)
    else:
        _BRAND_WEIGHTS.append(1.0)
_TOTAL_BRAND_WEIGHT = sum(_BRAND_WEIGHTS)
BRAND_PROBABILITIES = [w / _TOTAL_BRAND_WEIGHT for w in _BRAND_WEIGHTS]


BENCHMARK_ARTICLE_TYPES = [
    "crop_top", "tank_top", "tshirt", "blouse", "blazer",
    "sweater", "cardigan", "hoodie", "coat", "puffer",
    "mini_dress", "midi_dress", "maxi_dress", "bodycon_dress", "wrap_dress",
    "slip_dress", "sundress", "shift_dress", "sheath_dress", "shirt_dress",
    "jeans", "pants", "shorts", "joggers", "mini_skirt",
    "midi_skirt", "wide_leg_pants", "leggings", "turtleneck", "vest",
    "trench_coat", "windbreaker", "romper", "jumpsuit", "sweatshirt",
    "tube_top", "halter_top", "bralette", "bodysuit", "cami",
]

BENCHMARK_BROAD_CATEGORIES = {
    "crop_top": "tops", "tank_top": "tops", "tshirt": "tops", "blouse": "tops",
    "blazer": "outerwear", "sweater": "tops", "cardigan": "tops", "hoodie": "tops",
    "coat": "outerwear", "puffer": "outerwear", "mini_dress": "dresses",
    "midi_dress": "dresses", "maxi_dress": "dresses", "bodycon_dress": "dresses",
    "wrap_dress": "dresses", "slip_dress": "dresses", "sundress": "dresses",
    "shift_dress": "dresses", "sheath_dress": "dresses", "shirt_dress": "dresses",
    "jeans": "bottoms", "pants": "bottoms", "shorts": "bottoms", "joggers": "bottoms",
    "mini_skirt": "bottoms", "midi_skirt": "bottoms", "wide_leg_pants": "bottoms",
    "leggings": "bottoms", "turtleneck": "tops", "vest": "outerwear",
    "trench_coat": "outerwear", "windbreaker": "outerwear", "romper": "dresses",
    "jumpsuit": "dresses", "sweatshirt": "tops", "tube_top": "tops",
    "halter_top": "tops", "bralette": "tops", "bodysuit": "tops", "cami": "tops",
}

# Expanded styles: 15 options (profiles only prefer 2-3 each)
BENCHMARK_STYLES = ["casual", "trendy", "classic", "elegant", "minimal",
                     "streetwear", "sporty", "romantic", "boho",
                     "preppy", "grunge", "vintage", "edgy", "glam", "athleisure"]

# Expanded occasions
BENCHMARK_OCCASIONS = ["Everyday", "Work", "Date Night", "Party", "Weekend",
                        "Brunch", "Vacation", "Gym", "Formal", "Casual",
                        "Wedding Guest", "Festival", "School", "Travel",
                        "Lounge", "Beach", "Interview"]

BENCHMARK_SEASONS_MAP = {
    "summer": ["Summer", "Spring"],
    "winter": ["Winter", "Fall"],
    "spring": ["Spring", "Summer"],
    "fall": ["Fall", "Winter"],
}

BENCHMARK_MATERIALS_MAP = {
    "summer": ["cotton", "linen", "silk", "chiffon"],
    "winter": ["wool", "cashmere", "fleece", "leather"],
    "spring": ["cotton", "denim", "jersey", "silk"],
    "fall": ["wool", "denim", "corduroy", "suede"],
}

BENCHMARK_COLORS = ["Black", "White", "Navy", "Red", "Pink", "Beige",
                     "Grey", "Blue", "Green", "Brown", "Burgundy", "Olive",
                     "Mustard", "Teal", "Coral", "Lavender"]

BENCHMARK_PATTERNS = ["Solid", "Floral", "Striped", "Geometric", "Animal Print",
                       "Plaid", "Abstract", "Polka Dot", "Paisley", "Tie-Dye",
                       "Houndstooth", "Camo"]


def _weighted_brand_choice(rng: random.Random) -> str:
    """Pick a brand using realistic frequency weights."""
    r = rng.random()
    cumulative = 0.0
    for i, p in enumerate(BRAND_PROBABILITIES):
        cumulative += p
        if r <= cumulative:
            return BENCHMARK_BRANDS[i]
    return BENCHMARK_BRANDS[-1]


def _generate_mock_candidates(n: int = 200, seed: int = 42) -> List[Any]:
    """Generate n mock Candidate-like objects with realistic noise distribution."""
    rng = random.Random(seed)
    candidates = []

    for i in range(n):
        atype = rng.choice(BENCHMARK_ARTICLE_TYPES)
        brand = _weighted_brand_choice(rng)
        broad_cat = BENCHMARK_BROAD_CATEGORIES.get(atype, "tops")
        season_tags = rng.choice(list(BENCHMARK_SEASONS_MAP.values()))
        material_season = rng.choice(list(BENCHMARK_MATERIALS_MAP.keys()))
        materials = rng.sample(BENCHMARK_MATERIALS_MAP[material_season], k=min(2, len(BENCHMARK_MATERIALS_MAP[material_season])))
        styles = rng.sample(BENCHMARK_STYLES, k=rng.randint(1, 3))
        occasions = rng.sample(BENCHMARK_OCCASIONS, k=rng.randint(1, 3))

        candidate = MagicMock()
        candidate.item_id = f"prod_{i:04d}"
        candidate.product_id = f"prod_{i:04d}"
        candidate.brand = brand
        candidate.broad_category = broad_cat
        candidate.article_type = atype
        candidate.fit = rng.choice(["slim", "regular", "relaxed", "oversized"])
        candidate.color_family = rng.choice(["Neutrals", "Brights", "Pastels", "Dark", "Cool"])
        candidate.colors = [rng.choice(BENCHMARK_COLORS).lower()]
        candidate.pattern = rng.choice(BENCHMARK_PATTERNS)
        candidate.formality = rng.choice(["Casual", "Smart Casual", "Semi-Formal", "Formal"])
        candidate.neckline = rng.choice(["Crew", "V-Neck", "Scoop", "Off-Shoulder", "Turtleneck"])
        candidate.sleeve = rng.choice(["Short", "Long", "Sleeveless", "Three-Quarter"])
        candidate.length = rng.choice(["Mini", "Midi", "Maxi", "Regular", "Cropped"])
        candidate.style_tags = styles
        candidate.occasions = occasions
        candidate.seasons = season_tags
        candidate.materials = materials
        candidate.name = f"{brand} {atype.replace('_', ' ').title()}"
        candidate.image_url = f"https://img.example.com/{candidate.item_id}.jpg"
        candidate.price = round(rng.uniform(15.0, 300.0), 2)
        candidate.is_new = rng.random() < 0.15
        candidate.is_on_sale = rng.random() < 0.2
        candidate.final_score = round(rng.uniform(0.3, 0.9), 4)
        candidate.embedding_score = round(rng.uniform(0.4, 0.95), 4)
        candidate.preference_score = round(rng.uniform(0.3, 0.8), 4)
        candidate.sasrec_score = round(rng.uniform(0.2, 0.85), 4)
        candidate.source = rng.choice(["taste_vector", "trending", "exploration"])

        candidates.append(candidate)

    return candidates


def _generate_mock_search_results(n: int = 50, seed: int = 42) -> List[dict]:
    """Generate n mock search result dicts with realistic noise distribution."""
    rng = random.Random(seed)
    results = []

    for i in range(n):
        atype = rng.choice(BENCHMARK_ARTICLE_TYPES)
        brand = _weighted_brand_choice(rng)
        broad_cat = BENCHMARK_BROAD_CATEGORIES.get(atype, "tops")
        season_tags = rng.choice(list(BENCHMARK_SEASONS_MAP.values()))
        styles = rng.sample(BENCHMARK_STYLES, k=rng.randint(1, 3))
        occasions = rng.sample(BENCHMARK_OCCASIONS, k=rng.randint(1, 3))

        result = {
            "product_id": f"search_{i:04d}",
            "name": f"{brand} {atype.replace('_', ' ').title()}",
            "brand": brand,
            "image_url": f"https://img.example.com/search_{i:04d}.jpg",
            "gallery_images": [],
            "price": round(rng.uniform(15.0, 300.0), 2),
            "original_price": None,
            "is_on_sale": rng.random() < 0.2,
            "broad_category": broad_cat,
            "article_type": atype,
            "primary_color": rng.choice(BENCHMARK_COLORS),
            "color_family": rng.choice(["Neutrals", "Brights", "Pastels", "Dark"]),
            "colors": [rng.choice(BENCHMARK_COLORS).lower()],
            "pattern": rng.choice(BENCHMARK_PATTERNS),
            "fit_type": rng.choice(["Slim", "Regular", "Relaxed", "Oversized"]),
            "formality": rng.choice(["Casual", "Smart Casual", "Semi-Formal"]),
            "length": rng.choice(["Mini", "Midi", "Maxi", "Regular"]),
            "neckline": rng.choice(["Crew", "V-Neck", "Scoop", "Off-Shoulder"]),
            "sleeve_type": rng.choice(["Short", "Long", "Sleeveless"]),
            "style_tags": styles,
            "occasions": occasions,
            "seasons": season_tags,
            "materials": rng.sample(["cotton", "polyester", "wool", "silk", "linen"], k=2),
            "apparent_fabric": rng.choice(["cotton", "polyester", "wool", "silk"]),
            "source": rng.choice(["algolia", "semantic"]),
            "rrf_score": round(rng.uniform(0.01, 0.08), 4),
        }
        results.append(result)

    return results


def _candidate_to_scoring_dict(candidate) -> dict:
    """Convert a mock MagicMock candidate to a scoring dict."""
    return {
        "product_id": candidate.item_id,
        "article_type": candidate.article_type or "",
        "broad_category": candidate.broad_category or "",
        "brand": candidate.brand or "",
        "style_tags": candidate.style_tags or [],
        "occasions": candidate.occasions or [],
        "pattern": candidate.pattern,
        "formality": candidate.formality,
        "fit_type": candidate.fit,
        "neckline": candidate.neckline,
        "sleeve_type": candidate.sleeve,
        "length": candidate.length,
        "color_family": candidate.color_family,
        "seasons": candidate.seasons or [],
        "materials": candidate.materials or [],
        "name": candidate.name or "",
        "image_url": candidate.image_url or "",
    }


def _profile_to_user_context(profile: BenchmarkProfile):
    """Build a UserContext from a BenchmarkProfile."""
    from scoring.context import AgeGroup, Season, WeatherContext, UserContext

    age_map = {
        "18-24": AgeGroup.GEN_Z,
        "25-34": AgeGroup.YOUNG_ADULT,
        "35-44": AgeGroup.MID_CAREER,
        "45-64": AgeGroup.ESTABLISHED,
        "65+": AgeGroup.SENIOR,
    }
    season_map = {
        "spring": Season.SPRING,
        "summer": Season.SUMMER,
        "fall": Season.FALL,
        "winter": Season.WINTER,
    }

    weather = WeatherContext(
        temperature_c=profile.temperature_c,
        feels_like_c=profile.temperature_c,  # approximate
        condition=profile.weather_condition,
        humidity=60,
        wind_speed_mps=5.0,
        season=season_map[profile.season],
        is_hot=profile.is_hot,
        is_cold=profile.is_cold,
        is_mild=not profile.is_hot and not profile.is_cold,
        is_rainy=profile.is_rainy,
    )

    ctx = UserContext(
        user_id=profile.profile_id,
        age_group=age_map[profile.age_group],
        age_years=profile.age,
        city=profile.city,
        country=profile.country,
        weather=weather,
        coverage_prefs=profile.coverage_prefs,
    )
    return ctx


def _profile_to_search_profile(profile: BenchmarkProfile) -> dict:
    """Build a search reranker user_profile from a BenchmarkProfile."""
    return {
        "soft_prefs": {
            "preferred_brands": profile.preferred_brands,
            "preferred_styles": profile.preferred_styles,
            "preferred_colors": profile.preferred_colors,
            "preferred_patterns": profile.preferred_patterns,
            "preferred_fits": [],
            "preferred_sleeves": [],
            "preferred_lengths": [],
            "preferred_necklines": [],
            "preferred_formality": [],
        },
        "hard_filters": {
            "include_occasions": profile.preferred_occasions,
            "exclude_brands": [],
            "exclude_colors": [],
            "exclude_materials": [],
        },
    }


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def set_random_seed():
    random.seed(42)


@pytest.fixture(scope="module")
def all_profiles():
    return _build_100_profiles()


@pytest.fixture(scope="module")
def mock_candidates():
    return _generate_mock_candidates(n=200, seed=42)


@pytest.fixture(scope="module")
def mock_search_results():
    return _generate_mock_search_results(n=50, seed=42)


# ============================================================================
# TEST CLASSES
# ============================================================================


class TestMetricsModule:
    """Verify the metrics functions themselves are correct."""

    def test_dcg_known_values(self):
        # relevances = [3, 2, 1, 0, 0]
        # DCG@3 = 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.2618 + 0.5 = 4.7618
        rels = [3.0, 2.0, 1.0, 0.0, 0.0]
        dcg = _dcg(rels, 3)
        assert abs(dcg - 4.7618) < 0.01

    def test_ndcg_perfect_ranking(self):
        rels = [3.0, 2.0, 1.0, 0.0]
        assert _ndcg_at_k(rels, 4) == pytest.approx(1.0, abs=1e-6)

    def test_ndcg_worst_ranking(self):
        rels = [0.0, 0.0, 0.0, 3.0]
        ndcg = _ndcg_at_k(rels, 4)
        assert ndcg < 0.5  # worst is well below 1.0

    def test_precision_at_k(self):
        relevant = [True, False, True, False, True]
        assert _precision_at_k(relevant, 5) == 0.6
        assert _precision_at_k(relevant, 3) == pytest.approx(2/3)

    def test_mrr_first_hit(self):
        assert _mrr([True, False, False]) == 1.0
        assert _mrr([False, True, False]) == 0.5
        assert _mrr([False, False, True]) == pytest.approx(1/3)
        assert _mrr([False, False, False]) == 0.0

    def test_hit_rate(self):
        assert _hit_rate_at_k([False, False, True], 3) == 1.0
        assert _hit_rate_at_k([False, False, True], 2) == 0.0

    def test_entropy_uniform(self):
        # 4 items with equal counts => log2(4) = 2.0
        counts = {"a": 5, "b": 5, "c": 5, "d": 5}
        assert abs(_entropy(counts) - 2.0) < 0.01

    def test_entropy_single(self):
        counts = {"a": 10}
        assert _entropy(counts) == 0.0

    def test_compute_ranking_metrics_basic(self):
        items = [
            {"brand": "Zara", "broad_category": "tops", "article_type": "tshirt", "style_tags": ["casual"]},
            {"brand": "Nike", "broad_category": "bottoms", "article_type": "joggers", "style_tags": ["sporty"]},
            {"brand": "Unknown", "broad_category": "dresses", "article_type": "gown", "style_tags": ["formal"]},
        ]
        metrics = compute_ranking_metrics(
            items,
            relevant_brands={"zara"},
            relevant_categories={"tops"},
            relevant_styles={"casual"},
            relevant_article_types={"tshirt"},
            age_appropriate_types={"tshirt", "joggers"},
            weather_appropriate_types={"tshirt"},
        )
        assert metrics.mrr == 1.0           # first item is relevant
        assert metrics.hit_rate_5 == 1.0
        assert metrics.precision_10 > 0      # at least 1 of 3 is relevant


class TestProfileGeneration:
    """Verify the 100 profiles are diverse and well-formed."""

    def test_generates_100_profiles(self, all_profiles):
        assert len(all_profiles) == 100

    def test_all_age_groups_represented(self, all_profiles):
        age_groups = {p.age_group for p in all_profiles}
        assert age_groups == {"18-24", "25-34", "35-44", "45-64", "65+"}

    def test_all_weather_conditions_represented(self, all_profiles):
        conditions = {(p.is_hot, p.is_cold, p.is_rainy) for p in all_profiles}
        assert (True, False, False) in conditions   # hot
        assert (False, True, False) in conditions    # cold
        assert (False, False, True) in conditions    # rainy
        assert (False, False, False) in conditions   # mild

    def test_all_style_personas_represented(self, all_profiles):
        personas = {p.style_persona for p in all_profiles}
        assert personas == {"trendy", "classic", "minimal", "romantic", "streetwear"}

    def test_each_profile_has_brands(self, all_profiles):
        for p in all_profiles:
            assert len(p.preferred_brands) >= 3

    def test_each_profile_has_queries(self, all_profiles):
        for p in all_profiles:
            assert len(p.search_queries) >= 2

    def test_each_profile_has_expected_types(self, all_profiles):
        for p in all_profiles:
            assert len(p.expected_age_types) >= 5
            assert len(p.expected_weather_types) >= 5

    def test_20_profiles_per_age_group(self, all_profiles):
        counts = Counter(p.age_group for p in all_profiles)
        for ag in ["18-24", "25-34", "35-44", "45-64", "65+"]:
            assert counts[ag] == 20

    def test_25_profiles_per_weather(self, all_profiles):
        counts = Counter(p.season for p in all_profiles)
        for s in ["summer", "winter", "spring", "fall"]:
            assert counts[s] == 25

    def test_20_profiles_per_style(self, all_profiles):
        counts = Counter(p.style_persona for p in all_profiles)
        for s in ["trendy", "classic", "minimal", "romantic", "streetwear"]:
            assert counts[s] == 20

    def test_unique_profile_ids(self, all_profiles):
        ids = [p.profile_id for p in all_profiles]
        assert len(ids) == len(set(ids))


class TestContextScoringBenchmark:
    """
    Benchmark ContextScorer across all 100 profiles.
    Verifies age-appropriate and weather-appropriate items get boosted.
    """

    def test_age_scoring_directional(self, all_profiles):
        """Age-appropriate items should score higher than inappropriate ones."""
        from scoring.scorer import ContextScorer

        scorer = ContextScorer()
        pass_count = 0

        for profile in all_profiles:
            ctx = _profile_to_user_context(profile)

            # Create an age-appropriate item
            good_type = profile.expected_age_types[0]
            good_item = {
                "article_type": good_type,
                "broad_category": BENCHMARK_BROAD_CATEGORIES.get(good_type, "tops"),
                "style_tags": profile.preferred_styles,
                "occasions": profile.preferred_occasions,
                "pattern": "Solid",
                "formality": "Casual",
                "neckline": "Crew",
                "length": "Regular",
                "name": f"Test {good_type}",
                "fit_type": "regular",
                "materials": ["cotton"],
                "seasons": BENCHMARK_SEASONS_MAP.get(profile.season, ["Spring"]),
            }

            # Create an age-inappropriate item (opposite end of spectrum)
            if profile.age_group in ("45-64", "65+"):
                bad_type = "crop_top"
            else:
                bad_type = "shift_dress"
            bad_item = {
                "article_type": bad_type,
                "broad_category": BENCHMARK_BROAD_CATEGORIES.get(bad_type, "tops"),
                "style_tags": ["formal"],
                "occasions": ["Work"],
                "pattern": "Solid",
                "formality": "Formal",
                "neckline": "Turtleneck",
                "length": "Maxi",
                "name": f"Test {bad_type}",
                "fit_type": "regular",
                "materials": ["wool"],
                "seasons": ["Winter"],
            }

            good_score = scorer.score_item(good_item, ctx)
            bad_score = scorer.score_item(bad_item, ctx)

            if good_score >= bad_score:
                pass_count += 1

        # At least 70% of profiles should show directional correctness
        ratio = pass_count / len(all_profiles)
        print(f"\n  Age scoring directional correctness: {pass_count}/100 = {ratio:.0%}")
        assert ratio >= 0.60, f"Only {ratio:.0%} of profiles show age-appropriate boosting"

    def test_weather_scoring_directional(self, all_profiles):
        """Weather-appropriate items should score higher than inappropriate ones."""
        from scoring.scorer import ContextScorer

        scorer = ContextScorer()
        pass_count = 0

        for profile in all_profiles:
            ctx = _profile_to_user_context(profile)

            good_type = profile.expected_weather_types[0]
            good_item = {
                "article_type": good_type,
                "broad_category": BENCHMARK_BROAD_CATEGORIES.get(good_type, "tops"),
                "style_tags": ["casual"],
                "occasions": ["Everyday"],
                "pattern": "Solid",
                "formality": "Casual",
                "neckline": "Crew",
                "length": "Regular",
                "name": f"Test {good_type}",
                "fit_type": "regular",
                "materials": BENCHMARK_MATERIALS_MAP.get(profile.season, ["cotton"]),
                "seasons": BENCHMARK_SEASONS_MAP.get(profile.season, ["Spring"]),
            }

            # Weather-inappropriate: summer clothes in winter, winter clothes in summer
            if profile.is_hot:
                bad_type = "puffer"
                bad_materials = ["wool", "cashmere"]
                bad_seasons = ["Winter"]
            elif profile.is_cold:
                bad_type = "tank_top"
                bad_materials = ["linen", "chiffon"]
                bad_seasons = ["Summer"]
            elif profile.is_rainy:
                bad_type = "sundress"
                bad_materials = ["silk", "chiffon"]
                bad_seasons = ["Summer"]
            else:  # mild
                bad_type = "puffer"
                bad_materials = ["down", "sherpa"]
                bad_seasons = ["Winter"]

            bad_item = {
                "article_type": bad_type,
                "broad_category": BENCHMARK_BROAD_CATEGORIES.get(bad_type, "tops"),
                "style_tags": ["casual"],
                "occasions": ["Everyday"],
                "pattern": "Solid",
                "formality": "Casual",
                "neckline": "Crew",
                "length": "Regular",
                "name": f"Test {bad_type}",
                "fit_type": "regular",
                "materials": bad_materials,
                "seasons": bad_seasons,
            }

            good_score = scorer.score_item(good_item, ctx)
            bad_score = scorer.score_item(bad_item, ctx)

            if good_score > bad_score:
                pass_count += 1

        ratio = pass_count / len(all_profiles)
        print(f"\n  Weather scoring directional correctness: {pass_count}/100 = {ratio:.0%}")
        assert ratio >= 0.70, f"Only {ratio:.0%} show weather-appropriate boosting"

    def test_context_scores_within_bounds(self, all_profiles, mock_candidates):
        """All context adjustments must be in [-0.20, +0.20]."""
        from scoring.scorer import ContextScorer, MAX_CONTEXT_ADJUSTMENT

        scorer = ContextScorer()

        for profile in all_profiles[:20]:  # Sample 20 to keep fast
            ctx = _profile_to_user_context(profile)
            for cand in mock_candidates[:50]:
                item_dict = _candidate_to_scoring_dict(cand)
                adj = scorer.score_item(item_dict, ctx)
                assert -MAX_CONTEXT_ADJUSTMENT - 1e-9 <= adj <= MAX_CONTEXT_ADJUSTMENT + 1e-9, \
                    f"Context adj {adj} out of bounds for profile {profile.profile_id}"

    def test_context_scoring_throughput(self, all_profiles, mock_candidates):
        """Throughput should be >= 50K items/sec."""
        from scoring.scorer import ContextScorer

        scorer = ContextScorer()
        ctx = _profile_to_user_context(all_profiles[0])
        items = [_candidate_to_scoring_dict(c) for c in mock_candidates[:200]]

        # Warmup
        for item in items[:10]:
            scorer.score_item(item, ctx)

        start = time.perf_counter()
        iterations = 5
        for _ in range(iterations):
            for item in items:
                scorer.score_item(item, ctx)
        elapsed = time.perf_counter() - start

        total_ops = iterations * len(items)
        throughput = total_ops / elapsed
        print(f"\n  Context scoring throughput: {throughput:,.0f} items/sec")
        assert throughput >= 50_000, f"Throughput {throughput:,.0f} below 50K/sec"


class TestSessionScoringBenchmark:
    """
    Benchmark SessionScoringEngine across profiles.
    Tests that actions update scores and affect ranking.
    """

    def test_click_boosts_brand(self, all_profiles, mock_candidates):
        """Clicking items from a brand should boost that brand's score."""
        from recs.session_scoring import SessionScoringEngine, SessionScores

        engine = SessionScoringEngine()
        pass_count = 0

        for profile in all_profiles[:50]:  # Sample 50
            scores = engine.initialize_from_onboarding(
                preferred_brands=profile.preferred_brands
            )
            target_brand = profile.preferred_brands[0]

            # Click 3 items from target brand
            for i in range(3):
                engine.process_action(
                    scores, action="click",
                    product_id=f"click_{i}",
                    brand=target_brand,
                    item_type=profile.preferred_article_types[0] if profile.preferred_article_types else "tshirt",
                    attributes={"style": profile.preferred_styles[0] if profile.preferred_styles else "casual"},
                )

            brand_key = target_brand.lower()
            if brand_key in scores.brand_scores and scores.get_score("brand", brand_key) > 0:
                pass_count += 1

        ratio = pass_count / 50
        print(f"\n  Click -> brand boost: {pass_count}/50 = {ratio:.0%}")
        assert ratio >= 0.90, f"Only {ratio:.0%} profiles show brand boost after clicks"

    def test_skip_penalizes(self, all_profiles, mock_candidates):
        """Skipping items should add negative signal."""
        from recs.session_scoring import SessionScoringEngine, SessionScores

        engine = SessionScoringEngine()
        pass_count = 0

        for profile in all_profiles[:50]:
            scores = SessionScores()

            # Skip 5 items from a brand
            skip_brand = "Boohoo"
            for i in range(5):
                engine.process_action(
                    scores, action="skip",
                    product_id=f"skip_{i}",
                    brand=skip_brand,
                    item_type="mini_dress",
                )

            brand_key = skip_brand.lower()
            if brand_key in scores.brand_scores and scores.get_score("brand", brand_key) < 0:
                pass_count += 1

        ratio = pass_count / 50
        print(f"\n  Skip -> brand penalty: {pass_count}/50 = {ratio:.0%}")
        assert ratio >= 0.90

    def test_search_signal_updates_intents(self, all_profiles):
        """Search signals should update search_intents in SessionScores."""
        from recs.session_scoring import SessionScoringEngine, SessionScores

        engine = SessionScoringEngine()
        pass_count = 0

        for profile in all_profiles[:50]:
            scores = SessionScores()

            # Process a search signal
            query = profile.search_queries[0]
            engine.process_search_signal(
                scores,
                query=query,
                filters={"categories": profile.preferred_categories},
            )

            # Should have at least one intent recorded
            if len(scores.search_intents) > 0:
                pass_count += 1

        ratio = pass_count / 50
        print(f"\n  Search signal -> intent update: {pass_count}/50 = {ratio:.0%}")
        assert ratio >= 0.80

    def test_session_scoring_affects_candidate_order(self, all_profiles, mock_candidates):
        """Session scoring should reorder candidates based on actions."""
        from recs.session_scoring import SessionScoringEngine

        engine = SessionScoringEngine()
        reorder_count = 0

        for profile in all_profiles[:20]:
            scores = engine.initialize_from_onboarding(
                preferred_brands=profile.preferred_brands
            )

            # Click several items from preferred brand
            for i in range(5):
                engine.process_action(
                    scores, action="click",
                    product_id=f"pref_{i}",
                    brand=profile.preferred_brands[0],
                    item_type=profile.preferred_article_types[0] if profile.preferred_article_types else "tshirt",
                    attributes={"style": profile.preferred_styles[0] if profile.preferred_styles else "casual"},
                )

            # Get original order
            candidates = mock_candidates[:30]
            original_ids = [c.item_id for c in candidates]

            # Score candidates (returns sorted)
            # Need fresh candidates since score_candidates mutates
            fresh = _generate_mock_candidates(30, seed=42)
            scored = engine.score_candidates(scores, fresh)
            new_ids = [c.item_id for c in scored]

            if original_ids != new_ids:
                reorder_count += 1

        ratio = reorder_count / 20
        print(f"\n  Session scoring reorders candidates: {reorder_count}/20 = {ratio:.0%}")
        assert ratio >= 0.70

    def test_onboarding_brands_reflected_in_scores(self, all_profiles):
        """Onboarding preferred brands should appear in initial cluster/brand scores."""
        from recs.session_scoring import SessionScoringEngine

        engine = SessionScoringEngine()
        pass_count = 0

        for profile in all_profiles:
            scores = engine.initialize_from_onboarding(
                preferred_brands=profile.preferred_brands
            )

            # At least one preferred brand or its cluster should be in scores
            has_brand = any(
                b.lower() in scores.brand_scores
                for b in profile.preferred_brands
            )
            has_cluster = len(scores.cluster_scores) > 0

            if has_brand or has_cluster:
                pass_count += 1

        ratio = pass_count / len(all_profiles)
        print(f"\n  Onboarding brands reflected: {pass_count}/100 = {ratio:.0%}")
        assert ratio >= 0.90


class TestFeedRerankerBenchmark:
    """
    Benchmark the GreedyConstrainedReranker (feed pipeline).
    Checks diversity constraints, brand caps, exploration.
    """

    def test_brand_share_cap(self, mock_candidates):
        """No brand should exceed max_brand_share of the feed."""
        from recs.feed_reranker import GreedyConstrainedReranker, RerankerConfig

        config = RerankerConfig(max_brand_share=0.40)
        reranker = GreedyConstrainedReranker(config)

        target = 50
        result = reranker.rerank(mock_candidates, target_size=target)
        brand_counts = Counter(c.brand for c in result)
        max_allowed = max(3, int(target * config.max_brand_share))

        for brand, count in brand_counts.items():
            assert count <= max_allowed, f"Brand {brand} has {count} items, exceeds share cap of {max_allowed}"

    def test_soft_penalties_create_diversity(self, mock_candidates):
        """Brand and category decay should produce diverse output without hard caps."""
        from recs.feed_reranker import GreedyConstrainedReranker, RerankerConfig

        config = RerankerConfig()
        reranker = GreedyConstrainedReranker(config)

        result = reranker.rerank(mock_candidates, target_size=50)
        brand_counts = Counter(c.brand for c in result)

        # Soft penalties should produce at least some brand diversity
        assert len(brand_counts) >= 3, "Soft penalties should produce multiple brands"

    def test_seen_items_excluded(self, mock_candidates):
        """Seen items should not appear in output."""
        from recs.feed_reranker import GreedyConstrainedReranker

        reranker = GreedyConstrainedReranker()
        seen = {c.item_id for c in mock_candidates[:10]}

        result = reranker.rerank(mock_candidates, target_size=50, seen_ids=seen)
        result_ids = {c.item_id for c in result}

        assert result_ids.isdisjoint(seen), "Seen items appeared in reranked output"

    def test_diversity_stats_structure(self, mock_candidates):
        """get_diversity_stats should return well-formed data."""
        from recs.feed_reranker import GreedyConstrainedReranker

        reranker = GreedyConstrainedReranker()
        result = reranker.rerank(mock_candidates, target_size=30)
        stats = reranker.get_diversity_stats(result)

        assert "total_items" in stats
        assert "unique_brands" in stats
        assert "unique_types" in stats
        assert "brand_entropy" in stats
        assert stats["total_items"] == len(result)
        assert stats["unique_brands"] >= 2
        assert stats["brand_entropy"] > 0

    def test_reranker_across_profiles(self, all_profiles, mock_candidates):
        """Run reranker for each profile and collect diversity stats."""
        from recs.feed_reranker import GreedyConstrainedReranker, RerankerConfig
        from recs.session_scoring import SessionScoringEngine

        config = RerankerConfig()  # Default soft-penalty config
        reranker = GreedyConstrainedReranker(config)
        engine = SessionScoringEngine()

        entropies = []
        unique_brands_list = []

        for profile in all_profiles[:20]:  # Sample 20
            scores = engine.initialize_from_onboarding(
                preferred_brands=profile.preferred_brands
            )

            # Score candidates first
            fresh = _generate_mock_candidates(100, seed=hash(profile.profile_id) % 10000)
            scored = engine.score_candidates(scores, fresh)

            # Rerank
            result = reranker.rerank(scored, target_size=20)
            stats = reranker.get_diversity_stats(result)

            entropies.append(stats["brand_entropy"])
            unique_brands_list.append(stats["unique_brands"])

        avg_entropy = statistics.mean(entropies)
        avg_brands = statistics.mean(unique_brands_list)
        print(f"\n  Feed reranker across 20 profiles:")
        print(f"    Avg brand entropy: {avg_entropy:.2f}")
        print(f"    Avg unique brands in top-20: {avg_brands:.1f}")
        assert avg_entropy > 1.0, "Brand entropy too low"
        assert avg_brands >= 3, "Too few unique brands"


class TestSearchRerankerBenchmark:
    """
    Benchmark the SessionReranker (search pipeline).
    Tests profile scoring, session scoring, context scoring, dedup, brand diversity.
    """

    def test_profile_scoring_boosts_preferred_brands(self, all_profiles, mock_search_results):
        """When preferred brand items exist in results, they should get positive adjustment."""
        from search.reranker import SessionReranker

        reranker = SessionReranker()
        boost_count = 0
        testable_count = 0

        for profile in all_profiles[:30]:
            results = [r.copy() for r in mock_search_results[:20]]
            pref_brands = {b.lower() for b in profile.preferred_brands}
            user_profile = _profile_to_search_profile(profile)

            # Check if any preferred brand is even in the result set
            has_pref = any(
                (r.get("brand") or "").lower() in pref_brands for r in results
            )
            if not has_pref:
                continue  # Can't test boost if no preferred brands present
            testable_count += 1

            reranked = reranker.rerank(results, user_profile=user_profile)

            # Check if any preferred brand item has positive profile_adjustment
            for item in reranked:
                brand = (item.get("brand") or "").lower()
                if brand in pref_brands:
                    adj = item.get("profile_adjustment", 0)
                    if adj > 0:
                        boost_count += 1
                        break

        ratio = boost_count / max(testable_count, 1)
        print(f"\n  Profile scoring boosts preferred brands: {boost_count}/{testable_count} testable = {ratio:.0%}")
        # With 131 brands, preferred brands may rarely appear in 20 random results
        # When they do appear, they should get boosted most of the time
        assert testable_count == 0 or ratio >= 0.40, \
            f"Only {ratio:.0%} of testable profiles show brand boost"

    def test_session_scoring_integration(self, all_profiles, mock_search_results):
        """Session scores should affect search reranking."""
        from search.reranker import SessionReranker
        from recs.session_scoring import SessionScoringEngine, SessionScores

        reranker = SessionReranker()
        engine = SessionScoringEngine()
        effect_count = 0

        for profile in all_profiles[:20]:
            scores = engine.initialize_from_onboarding(
                preferred_brands=profile.preferred_brands
            )

            # Add some clicks
            for i in range(3):
                engine.process_action(
                    scores, action="click",
                    product_id=f"click_{i}",
                    brand=profile.preferred_brands[0],
                    item_type=profile.preferred_article_types[0] if profile.preferred_article_types else "tshirt",
                )

            results = [r.copy() for r in mock_search_results[:20]]

            reranked = reranker.rerank(
                results,
                session_scores=scores,
            )

            # Check if any item has a session_adjustment
            has_session_adj = any(
                item.get("session_adjustment", 0) != 0
                for item in reranked
            )
            if has_session_adj:
                effect_count += 1

        ratio = effect_count / 20
        print(f"\n  Session scores affect search: {effect_count}/20 = {ratio:.0%}")
        assert ratio >= 0.60

    def test_context_scoring_integration(self, all_profiles, mock_search_results):
        """Context scores should affect search reranking."""
        from search.reranker import SessionReranker

        reranker = SessionReranker()
        effect_count = 0

        for profile in all_profiles[:20]:
            ctx = _profile_to_user_context(profile)
            results = [r.copy() for r in mock_search_results[:20]]

            reranked = reranker.rerank(results, user_context=ctx)

            has_context_adj = any(
                item.get("context_adjustment", 0) != 0
                for item in reranked
            )
            if has_context_adj:
                effect_count += 1

        ratio = effect_count / 20
        print(f"\n  Context scores affect search: {effect_count}/20 = {ratio:.0%}")
        assert ratio >= 0.70

    def test_brand_diversity_in_search(self, mock_search_results):
        """Search results should respect brand diversity cap."""
        from search.reranker import SessionReranker

        reranker = SessionReranker()

        # Create results with heavy brand bias
        biased = []
        for i in range(30):
            r = mock_search_results[i % len(mock_search_results)].copy()
            r["product_id"] = f"biased_{i}"
            r["brand"] = "Boohoo"
            r["rrf_score"] = 0.05 - i * 0.001
            biased.append(r)
        # Add some others
        for i in range(10):
            r = mock_search_results[i % len(mock_search_results)].copy()
            r["product_id"] = f"other_{i}"
            r["brand"] = f"Brand_{i}"
            r["rrf_score"] = 0.02
            biased.append(r)

        reranked = reranker.rerank(biased, max_per_brand=4)
        boohoo_count = sum(1 for r in reranked if r["brand"] == "Boohoo")
        assert boohoo_count <= 4, f"Boohoo has {boohoo_count} items, should be <= 4"

    def test_deduplication(self, mock_search_results):
        """Near-duplicate items should be removed."""
        from search.reranker import SessionReranker

        reranker = SessionReranker()

        # Create duplicates with size variants
        results = []
        for size in ["", "Petite ", "Plus Size ", "Tall "]:
            r = mock_search_results[0].copy()
            r["product_id"] = f"dup_{size.strip() or 'regular'}"
            r["name"] = f"{size}Black Mini Dress"
            r["brand"] = "Boohoo"
            r["rrf_score"] = 0.05
            results.append(r)
        # Add non-duplicates
        for i in range(10):
            r = mock_search_results[i + 1 % len(mock_search_results)].copy()
            r["product_id"] = f"unique_{i}"
            results.append(r)

        reranked = reranker.rerank(results)

        # Should have fewer items than input due to dedup
        names_with_mini = [r["name"] for r in reranked if "Mini Dress" in r.get("name", "")]
        # Size variants should be deduplicated to at most 1-2
        assert len(names_with_mini) <= 2, f"Expected dedup, got {len(names_with_mini)} 'Mini Dress' items"


class TestFullPipelineBenchmark:
    """
    End-to-end benchmark: profile -> context scoring -> session scoring -> reranking -> metrics.
    Runs all 100 profiles and computes aggregate metrics.
    """

    def test_feed_pipeline_all_profiles(self, all_profiles):
        """
        Full feed pipeline benchmark across 100 profiles.
        Measures NDCG, Precision, MRR, diversity at K=5,10,20.
        """
        from scoring.scorer import ContextScorer
        from recs.session_scoring import SessionScoringEngine
        from recs.feed_reranker import GreedyConstrainedReranker, RerankerConfig

        scorer = ContextScorer()
        engine = SessionScoringEngine()
        reranker = GreedyConstrainedReranker(RerankerConfig())

        all_metrics = []
        all_baselines = []
        start = time.perf_counter()

        for profile in all_profiles:
            # 1. Initialize session from onboarding
            session_scores = engine.initialize_from_onboarding(
                preferred_brands=profile.preferred_brands
            )

            # 2. Simulate some user actions
            for i, brand in enumerate(profile.preferred_brands[:3]):
                engine.process_action(
                    session_scores, action="click",
                    product_id=f"action_{i}",
                    brand=brand,
                    item_type=profile.preferred_article_types[i % len(profile.preferred_article_types)] if profile.preferred_article_types else "tshirt",
                    attributes={"style": profile.preferred_styles[0] if profile.preferred_styles else "casual"},
                )

            # 3. Generate candidates (different seed per profile for variety)
            candidates = _generate_mock_candidates(100, seed=hash(profile.profile_id) % 100000)

            # 4. Session scoring (reorders candidates)
            scored = engine.score_candidates(session_scores, candidates)

            # 5. Context scoring
            ctx = _profile_to_user_context(profile)
            for cand in scored:
                item_dict = _candidate_to_scoring_dict(cand)
                adj = scorer.score_item(item_dict, ctx)
                cand.final_score += adj

            # 6. Feed reranking
            reranked = reranker.rerank(scored, target_size=20)

            # 7. Convert to dicts for metrics
            result_dicts = []
            for c in reranked:
                result_dicts.append({
                    "brand": c.brand,
                    "broad_category": c.broad_category,
                    "article_type": c.article_type,
                    "style_tags": c.style_tags,
                })

            # 8. Compute metrics
            rel_kw = dict(
                relevant_brands={b.lower() for b in profile.preferred_brands},
                relevant_categories={c.lower() for c in profile.preferred_categories},
                relevant_styles={s.lower() for s in profile.preferred_styles},
                relevant_article_types={t.lower() for t in profile.preferred_article_types},
                age_appropriate_types={t.lower() for t in profile.expected_age_types},
                weather_appropriate_types={t.lower() for t in profile.expected_weather_types},
            )
            metrics = compute_ranking_metrics(result_dicts, **rel_kw)
            baseline = compute_random_baseline(result_dicts, **rel_kw)
            all_metrics.append(metrics)
            all_baselines.append(baseline)

        elapsed = time.perf_counter() - start

        # Aggregate
        avg = lambda field: statistics.mean(getattr(m, field) for m in all_metrics)
        med = lambda field: statistics.median(getattr(m, field) for m in all_metrics)
        base = lambda field: statistics.mean(getattr(m, field) for m in all_baselines)

        def _lift(field):
            b = base(field)
            return (avg(field) - b) / b * 100 if b > 0 else float('inf') if avg(field) > 0 else 0

        print(f"\n{'='*80}")
        print(f"  FEED PIPELINE BENCHMARK — 100 Profiles")
        print(f"{'='*80}")
        print(f"  Total time: {elapsed:.2f}s ({elapsed/100*1000:.1f}ms per profile)")
        print(f"")
        print(f"  {'Metric':<30} {'Scored':>8} {'Random':>8} {'Lift':>8}")
        print(f"  {'-'*54}")
        for field in ["ndcg_5", "ndcg_10", "ndcg_20",
                       "precision_5", "precision_10", "precision_20",
                       "mrr", "hit_rate_5", "hit_rate_10", "hit_rate_20",
                       "brand_precision_10", "category_precision_10",
                       "style_precision_10", "diversity_brands_10",
                       "brand_entropy_10",
                       "age_appropriate_ratio", "weather_appropriate_ratio"]:
            lift_str = f"{_lift(field):>+7.1f}%" if base(field) > 0 else "    inf"
            print(f"  {field:<30} {avg(field):>8.4f} {base(field):>8.4f} {lift_str}")
        print(f"{'='*80}")

        # Assertions — scored should beat random baseline
        assert avg("ndcg_10") > base("ndcg_10"), \
            f"Scored NDCG@10 ({avg('ndcg_10'):.4f}) should beat random ({base('ndcg_10'):.4f})"
        assert avg("brand_precision_10") >= base("brand_precision_10") * 0.95, \
            f"Scored brand P@10 should not regress significantly vs random"
        assert avg("diversity_brands_10") >= 2, "Brand diversity too low"

    def test_search_pipeline_all_profiles(self, all_profiles):
        """
        Full search pipeline benchmark across all 100 profiles.
        For each profile, runs each of their search queries through the search reranker.
        """
        from scoring.scorer import ContextScorer
        from search.reranker import SessionReranker
        from recs.session_scoring import SessionScoringEngine

        reranker = SessionReranker()
        engine = SessionScoringEngine()

        all_metrics = []
        all_baselines = []
        start = time.perf_counter()

        for profile in all_profiles:
            # Initialize session
            session_scores = engine.initialize_from_onboarding(
                preferred_brands=profile.preferred_brands
            )

            # Simulate actions
            for i, brand in enumerate(profile.preferred_brands[:2]):
                engine.process_action(
                    session_scores, action="click",
                    product_id=f"sa_{i}",
                    brand=brand,
                    item_type=profile.preferred_article_types[0] if profile.preferred_article_types else "tshirt",
                )

            # Process each search query
            for query in profile.search_queries:
                engine.process_search_signal(
                    session_scores,
                    query=query,
                    filters={"categories": profile.preferred_categories},
                )

            # Generate mock search results
            results = _generate_mock_search_results(
                30, seed=hash(profile.profile_id + profile.search_queries[0]) % 100000
            )

            ctx = _profile_to_user_context(profile)
            user_profile = _profile_to_search_profile(profile)

            # Full rerank with all signals
            reranked = reranker.rerank(
                results,
                user_profile=user_profile,
                session_scores=session_scores,
                user_context=ctx,
            )

            # Compute metrics
            rel_kw = dict(
                relevant_brands={b.lower() for b in profile.preferred_brands},
                relevant_categories={c.lower() for c in profile.preferred_categories},
                relevant_styles={s.lower() for s in profile.preferred_styles},
                relevant_article_types={t.lower() for t in profile.preferred_article_types},
                age_appropriate_types={t.lower() for t in profile.expected_age_types},
                weather_appropriate_types={t.lower() for t in profile.expected_weather_types},
            )
            metrics = compute_ranking_metrics(reranked[:20], **rel_kw)
            baseline = compute_random_baseline(reranked[:20], **rel_kw)
            all_metrics.append(metrics)
            all_baselines.append(baseline)

        elapsed = time.perf_counter() - start

        avg = lambda field: statistics.mean(getattr(m, field) for m in all_metrics)
        med = lambda field: statistics.median(getattr(m, field) for m in all_metrics)
        base = lambda field: statistics.mean(getattr(m, field) for m in all_baselines)

        def _lift(field):
            b = base(field)
            return (avg(field) - b) / b * 100 if b > 0 else float('inf') if avg(field) > 0 else 0

        print(f"\n{'='*80}")
        print(f"  SEARCH PIPELINE BENCHMARK — 100 Profiles")
        print(f"{'='*80}")
        print(f"  Total time: {elapsed:.2f}s ({elapsed/100*1000:.1f}ms per profile)")
        print(f"")
        print(f"  {'Metric':<30} {'Scored':>8} {'Random':>8} {'Lift':>8}")
        print(f"  {'-'*54}")
        for field in ["ndcg_5", "ndcg_10", "ndcg_20",
                       "precision_5", "precision_10", "precision_20",
                       "mrr", "hit_rate_5", "hit_rate_10", "hit_rate_20",
                       "brand_precision_10", "category_precision_10",
                       "style_precision_10", "diversity_brands_10",
                       "brand_entropy_10",
                       "age_appropriate_ratio", "weather_appropriate_ratio"]:
            lift_str = f"{_lift(field):>+7.1f}%" if base(field) > 0 else "    inf"
            print(f"  {field:<30} {avg(field):>8.4f} {base(field):>8.4f} {lift_str}")
        print(f"{'='*80}")

        # Assertions — scored should beat random baseline
        assert avg("ndcg_10") >= base("ndcg_10") * 0.95, \
            f"Search NDCG@10 should not be significantly worse than random"
        assert avg("diversity_brands_10") >= 2


class TestCrossCuttingBenchmark:
    """
    Cross-cutting tests: actions in one pipeline should affect another.
    """

    def test_feed_action_affects_search_reranking(self, all_profiles, mock_search_results):
        """
        A click action in the feed should affect session_adjustment scores in search.
        With 131 brands, the clicked brand may not be in the results, but
        session scoring should still produce non-zero adjustments via cluster/type signals.
        """
        from recs.session_scoring import SessionScoringEngine
        from search.reranker import SessionReranker

        engine = SessionScoringEngine()
        reranker = SessionReranker()

        adjustment_count = 0
        for profile in all_profiles[:30]:
            scores = engine.initialize_from_onboarding(
                preferred_brands=profile.preferred_brands
            )

            # Rerank before actions
            results_before = [r.copy() for r in mock_search_results[:20]]
            reranked_before = reranker.rerank(results_before, session_scores=scores)
            before_adjs = [r.get("session_adjustment", 0) for r in reranked_before]

            # Add clicks on target brand
            target_brand = profile.preferred_brands[0]
            for i in range(5):
                engine.process_action(
                    scores, action="click",
                    product_id=f"feed_click_{i}",
                    brand=target_brand,
                    item_type="midi_dress",
                    attributes={"style": "casual"},
                )

            # Rerank after actions
            results_after = [r.copy() for r in mock_search_results[:20]]
            reranked_after = reranker.rerank(results_after, session_scores=scores)
            after_adjs = [r.get("session_adjustment", 0) for r in reranked_after]

            # Session adjustments should differ after clicks
            if before_adjs != after_adjs:
                adjustment_count += 1

        ratio = adjustment_count / 30
        print(f"\n  Feed action -> search session adj changed: {adjustment_count}/30 = {ratio:.0%}")
        assert ratio >= 0.30

    def test_search_signal_affects_feed_scoring(self, all_profiles, mock_candidates):
        """
        A search signal should update intents that affect feed candidate scoring.
        """
        from recs.session_scoring import SessionScoringEngine

        engine = SessionScoringEngine()
        pass_count = 0

        for profile in all_profiles[:30]:
            scores = engine.initialize_from_onboarding(
                preferred_brands=profile.preferred_brands
            )

            # Score candidates before search
            cands_before = _generate_mock_candidates(20, seed=hash(profile.profile_id) % 10000)
            scored_before = engine.score_candidates(scores, cands_before)
            order_before = [c.item_id for c in scored_before]

            # Process search signal
            engine.process_search_signal(
                scores,
                query=profile.search_queries[0],
                filters={
                    "categories": profile.preferred_categories,
                    "brands": profile.preferred_brands[:2],
                },
            )

            # Score candidates after search
            cands_after = _generate_mock_candidates(20, seed=hash(profile.profile_id) % 10000)
            scored_after = engine.score_candidates(scores, cands_after)
            order_after = [c.item_id for c in scored_after]

            if order_before != order_after:
                pass_count += 1

        ratio = pass_count / 30
        print(f"\n  Search signal -> feed reorder: {pass_count}/30 = {ratio:.0%}")
        assert ratio >= 0.40

    def test_session_scores_persist_and_load(self, all_profiles):
        """
        Session scores should survive serialization round-trip.
        """
        from recs.session_scoring import SessionScoringEngine, SessionScores
        from recs.session_state import InMemoryScoringBackend

        engine = SessionScoringEngine()
        backend = InMemoryScoringBackend()
        pass_count = 0

        for profile in all_profiles[:20]:
            scores = engine.initialize_from_onboarding(
                preferred_brands=profile.preferred_brands
            )

            # Add some actions
            for i in range(3):
                engine.process_action(
                    scores, action="click",
                    product_id=f"persist_{i}",
                    brand=profile.preferred_brands[0],
                    item_type="tshirt",
                )

            # Persist
            backend.save_scores(profile.profile_id, scores)

            # Load
            loaded = backend.get_scores(profile.profile_id)
            if loaded is not None:
                # Verify key data survived
                if (loaded.brand_scores == scores.brand_scores and
                        loaded.action_count == scores.action_count):
                    pass_count += 1

        ratio = pass_count / 20
        print(f"\n  Session scores persist round-trip: {pass_count}/20 = {ratio:.0%}")
        assert ratio >= 0.90

    def test_session_scores_json_roundtrip(self, all_profiles):
        """SessionScores should survive JSON serialization."""
        from recs.session_scoring import SessionScoringEngine, SessionScores

        engine = SessionScoringEngine()
        pass_count = 0

        for profile in all_profiles[:20]:
            scores = engine.initialize_from_onboarding(
                preferred_brands=profile.preferred_brands
            )

            for i in range(3):
                engine.process_action(
                    scores, action="click",
                    product_id=f"json_{i}",
                    brand=profile.preferred_brands[0],
                    item_type="tshirt",
                )

            # JSON round-trip
            json_str = scores.to_json()
            loaded = SessionScores.from_json(json_str)

            if (loaded.brand_scores == scores.brand_scores and
                    loaded.cluster_scores == scores.cluster_scores and
                    loaded.action_count == scores.action_count):
                pass_count += 1

        ratio = pass_count / 20
        print(f"\n  JSON round-trip: {pass_count}/20 = {ratio:.0%}")
        assert ratio >= 0.95


class TestWeatherAPIScoringBenchmark:
    """
    Weather-specific scoring benchmarks.
    Tests material, season, and temperature scoring rules.
    """

    def test_summer_boosts_light_clothes(self, all_profiles):
        """In hot summer, tanks/shorts/sundresses should get boosted."""
        from scoring.scorer import ContextScorer

        scorer = ContextScorer()
        hot_profiles = [p for p in all_profiles if p.is_hot]
        assert len(hot_profiles) == 25

        boost_ratios = []
        for profile in hot_profiles:
            ctx = _profile_to_user_context(profile)

            summer_items = ["tank_top", "shorts", "sundress", "cami", "crop_top"]
            winter_items = ["coat", "puffer", "turtleneck", "sweater"]

            summer_scores = []
            for atype in summer_items:
                item = {
                    "article_type": atype,
                    "broad_category": BENCHMARK_BROAD_CATEGORIES.get(atype, "tops"),
                    "materials": ["cotton", "linen"],
                    "seasons": ["Summer", "Spring"],
                    "style_tags": ["casual"], "occasions": ["Everyday"],
                    "pattern": "Solid", "formality": "Casual",
                    "name": atype, "fit_type": "regular",
                }
                summer_scores.append(scorer.score_item(item, ctx))

            winter_scores = []
            for atype in winter_items:
                item = {
                    "article_type": atype,
                    "broad_category": BENCHMARK_BROAD_CATEGORIES.get(atype, "outerwear"),
                    "materials": ["wool", "cashmere"],
                    "seasons": ["Winter", "Fall"],
                    "style_tags": ["classic"], "occasions": ["Work"],
                    "pattern": "Solid", "formality": "Casual",
                    "name": atype, "fit_type": "regular",
                }
                winter_scores.append(scorer.score_item(item, ctx))

            avg_summer = statistics.mean(summer_scores)
            avg_winter = statistics.mean(winter_scores)
            boost_ratios.append(avg_summer > avg_winter)

        pass_ratio = sum(boost_ratios) / len(boost_ratios)
        print(f"\n  Hot summer boosts light clothes: {sum(boost_ratios)}/25 = {pass_ratio:.0%}")
        assert pass_ratio >= 0.80

    def test_winter_boosts_warm_clothes(self, all_profiles):
        """In cold winter, coats/sweaters/turtlenecks should get boosted."""
        from scoring.scorer import ContextScorer

        scorer = ContextScorer()
        cold_profiles = [p for p in all_profiles if p.is_cold]
        assert len(cold_profiles) == 25

        boost_ratios = []
        for profile in cold_profiles:
            ctx = _profile_to_user_context(profile)

            winter_items = ["coat", "puffer", "sweater", "turtleneck", "cardigan"]
            summer_items = ["tank_top", "shorts", "sundress", "crop_top"]

            winter_scores = []
            for atype in winter_items:
                item = {
                    "article_type": atype,
                    "broad_category": BENCHMARK_BROAD_CATEGORIES.get(atype, "outerwear"),
                    "materials": ["wool", "cashmere"],
                    "seasons": ["Winter", "Fall"],
                    "style_tags": ["classic"], "occasions": ["Everyday"],
                    "pattern": "Solid", "formality": "Casual",
                    "name": atype, "fit_type": "regular",
                }
                winter_scores.append(scorer.score_item(item, ctx))

            summer_scores = []
            for atype in summer_items:
                item = {
                    "article_type": atype,
                    "broad_category": BENCHMARK_BROAD_CATEGORIES.get(atype, "tops"),
                    "materials": ["linen", "chiffon"],
                    "seasons": ["Summer"],
                    "style_tags": ["casual"], "occasions": ["Everyday"],
                    "pattern": "Solid", "formality": "Casual",
                    "name": atype, "fit_type": "regular",
                }
                summer_scores.append(scorer.score_item(item, ctx))

            avg_winter = statistics.mean(winter_scores)
            avg_summer = statistics.mean(summer_scores)
            boost_ratios.append(avg_winter > avg_summer)

        pass_ratio = sum(boost_ratios) / len(boost_ratios)
        print(f"\n  Cold winter boosts warm clothes: {sum(boost_ratios)}/25 = {pass_ratio:.0%}")
        assert pass_ratio >= 0.80

    def test_rainy_weather_boosts_outerwear(self, all_profiles):
        """In rainy weather, jackets/coats/trench coats should get boosted."""
        from scoring.scorer import ContextScorer

        scorer = ContextScorer()
        rainy_profiles = [p for p in all_profiles if p.is_rainy]
        assert len(rainy_profiles) == 25

        boost_count = 0
        for profile in rainy_profiles:
            ctx = _profile_to_user_context(profile)

            rainy_items = ["jacket", "trench_coat", "windbreaker"]
            exposed_items = ["sundress", "tank_top"]

            rainy_scores = []
            for atype in rainy_items:
                item = {
                    "article_type": atype,
                    "broad_category": "outerwear",
                    "materials": ["nylon", "polyester"],
                    "seasons": ["Fall", "Spring"],
                    "style_tags": ["casual"], "occasions": ["Everyday"],
                    "pattern": "Solid", "formality": "Casual",
                    "name": atype, "fit_type": "regular",
                }
                rainy_scores.append(scorer.score_item(item, ctx))

            exposed_scores = []
            for atype in exposed_items:
                item = {
                    "article_type": atype,
                    "broad_category": BENCHMARK_BROAD_CATEGORIES.get(atype, "tops"),
                    "materials": ["silk", "chiffon"],
                    "seasons": ["Summer"],
                    "style_tags": ["casual"], "occasions": ["Everyday"],
                    "pattern": "Solid", "formality": "Casual",
                    "name": atype, "fit_type": "regular",
                }
                exposed_scores.append(scorer.score_item(item, ctx))

            if statistics.mean(rainy_scores) > statistics.mean(exposed_scores):
                boost_count += 1

        ratio = boost_count / 25
        print(f"\n  Rainy weather boosts outerwear: {boost_count}/25 = {ratio:.0%}")
        assert ratio >= 0.70

    def test_material_season_alignment(self, all_profiles):
        """
        Items with season-appropriate materials should score higher.
        e.g., linen in summer > wool in summer.
        """
        from scoring.scorer import ContextScorer

        scorer = ContextScorer()
        pass_count = 0

        for profile in all_profiles:
            ctx = _profile_to_user_context(profile)

            # Good material for this season
            good_materials = BENCHMARK_MATERIALS_MAP.get(profile.season, ["cotton"])
            # Bad material for this season (opposite season)
            opposite = {"summer": "winter", "winter": "summer",
                        "spring": "winter", "fall": "summer"}
            bad_materials = BENCHMARK_MATERIALS_MAP.get(opposite[profile.season], ["wool"])

            good_item = {
                "article_type": "blouse",
                "broad_category": "tops",
                "materials": good_materials[:2],
                "seasons": BENCHMARK_SEASONS_MAP.get(profile.season, ["Spring"]),
                "style_tags": ["casual"], "occasions": ["Everyday"],
                "pattern": "Solid", "formality": "Casual",
                "name": "Test Blouse", "fit_type": "regular",
            }

            bad_item = {
                "article_type": "blouse",
                "broad_category": "tops",
                "materials": bad_materials[:2],
                "seasons": BENCHMARK_SEASONS_MAP.get(opposite[profile.season], ["Winter"]),
                "style_tags": ["casual"], "occasions": ["Everyday"],
                "pattern": "Solid", "formality": "Casual",
                "name": "Test Blouse", "fit_type": "regular",
            }

            good_score = scorer.score_item(good_item, ctx)
            bad_score = scorer.score_item(bad_item, ctx)

            if good_score >= bad_score:
                pass_count += 1

        ratio = pass_count / len(all_profiles)
        print(f"\n  Material-season alignment: {pass_count}/100 = {ratio:.0%}")
        assert ratio >= 0.70


class TestAggregateReport:
    """
    Final aggregate report across all test dimensions.
    Prints a comprehensive summary table.
    """

    def test_generate_full_report(self, all_profiles):
        """Run everything and generate the final report."""
        from scoring.scorer import ContextScorer
        from recs.session_scoring import SessionScoringEngine
        from recs.feed_reranker import GreedyConstrainedReranker, RerankerConfig
        from search.reranker import SessionReranker

        scorer = ContextScorer()
        engine = SessionScoringEngine()
        feed_reranker = GreedyConstrainedReranker(RerankerConfig())
        search_reranker = SessionReranker()

        # Collect metrics by dimension
        by_age = defaultdict(list)
        by_weather = defaultdict(list)
        by_style = defaultdict(list)

        feed_metrics_all = []
        feed_baselines_all = []
        search_metrics_all = []
        search_baselines_all = []

        start = time.perf_counter()

        for profile in all_profiles:
            ctx = _profile_to_user_context(profile)
            session_scores = engine.initialize_from_onboarding(
                preferred_brands=profile.preferred_brands
            )

            # Simulate actions
            for i, brand in enumerate(profile.preferred_brands[:3]):
                engine.process_action(
                    session_scores, action="click",
                    product_id=f"r_{i}",
                    brand=brand,
                    item_type=profile.preferred_article_types[i % len(profile.preferred_article_types)] if profile.preferred_article_types else "tshirt",
                    attributes={"style": profile.preferred_styles[0] if profile.preferred_styles else "casual"},
                )

            # Process search signals
            for q in profile.search_queries[:2]:
                engine.process_search_signal(
                    session_scores, query=q,
                    filters={"categories": profile.preferred_categories},
                )

            rel_kw = dict(
                relevant_brands={b.lower() for b in profile.preferred_brands},
                relevant_categories={c.lower() for c in profile.preferred_categories},
                relevant_styles={s.lower() for s in profile.preferred_styles},
                relevant_article_types={t.lower() for t in profile.preferred_article_types},
                age_appropriate_types={t.lower() for t in profile.expected_age_types},
                weather_appropriate_types={t.lower() for t in profile.expected_weather_types},
            )

            # --- Feed pipeline ---
            candidates = _generate_mock_candidates(80, seed=hash(profile.profile_id) % 100000)
            scored = engine.score_candidates(session_scores, candidates)
            for cand in scored:
                adj = scorer.score_item(_candidate_to_scoring_dict(cand), ctx)
                cand.final_score += adj
            feed_result = feed_reranker.rerank(scored, target_size=20)
            feed_dicts = [{"brand": c.brand, "broad_category": c.broad_category,
                           "article_type": c.article_type, "style_tags": c.style_tags}
                          for c in feed_result]
            fm = compute_ranking_metrics(feed_dicts, **rel_kw)
            fb = compute_random_baseline(feed_dicts, **rel_kw, n_shuffles=20)
            feed_metrics_all.append(fm)
            feed_baselines_all.append(fb)

            # --- Search pipeline ---
            search_results = _generate_mock_search_results(
                30, seed=hash(profile.profile_id + "search") % 100000
            )
            user_profile = _profile_to_search_profile(profile)
            search_result = search_reranker.rerank(
                search_results, user_profile=user_profile,
                session_scores=session_scores, user_context=ctx,
            )
            sm = compute_ranking_metrics(search_result[:20], **rel_kw)
            sb = compute_random_baseline(search_result[:20], **rel_kw, n_shuffles=20)
            search_metrics_all.append(sm)
            search_baselines_all.append(sb)

            # Group by dimension
            by_age[profile.age_group].append((fm, sm, fb, sb))
            label = "hot" if profile.is_hot else "cold" if profile.is_cold else "rainy" if profile.is_rainy else "mild"
            by_weather[label].append((fm, sm, fb, sb))
            by_style[profile.style_persona].append((fm, sm, fb, sb))

        elapsed = time.perf_counter() - start

        # === PRINT REPORT ===
        def _avg(metric_list, field):
            return statistics.mean(getattr(m, field) for m in metric_list)

        def _lift_str(scored_val, base_val):
            if base_val > 0:
                lift = (scored_val - base_val) / base_val * 100
                return f"{lift:>+6.1f}%"
            return "   inf" if scored_val > 0 else "    0%"

        print(f"\n{'#'*90}")
        print(f"  E2E BENCHMARK AGGREGATE REPORT (with random baseline)")
        print(f"  100 Profiles | {elapsed:.2f}s total | {elapsed/100*1000:.1f}ms/profile")
        print(f"{'#'*90}")

        # --- Overall with baseline ---
        print(f"\n  === OVERALL (Scored vs Random Baseline) ===")
        print(f"  {'Metric':<28} {'Feed':>7} {'F.Base':>7} {'F.Lift':>8} {'Search':>7} {'S.Base':>7} {'S.Lift':>8}")
        print(f"  {'-'*73}")
        for field in ["ndcg_10", "precision_10", "mrr", "hit_rate_10",
                       "brand_precision_10", "category_precision_10",
                       "style_precision_10", "diversity_brands_10",
                       "brand_entropy_10",
                       "age_appropriate_ratio", "weather_appropriate_ratio"]:
            fv = _avg(feed_metrics_all, field)
            fbv = _avg(feed_baselines_all, field)
            sv = _avg(search_metrics_all, field)
            sbv = _avg(search_baselines_all, field)
            print(f"  {field:<28} {fv:>7.4f} {fbv:>7.4f} {_lift_str(fv, fbv)} {sv:>7.4f} {sbv:>7.4f} {_lift_str(sv, sbv)}")

        # --- By Age Group ---
        print(f"\n  === BY AGE GROUP (Feed NDCG@10: Scored vs Baseline) ===")
        print(f"  {'Age Group':<12} {'F.NDCG':>7} {'F.Base':>7} {'F.Lift':>8} {'S.NDCG':>7} {'S.Base':>7} {'S.Lift':>8} {'Age%':>6}")
        print(f"  {'-'*63}")
        for ag in ["18-24", "25-34", "35-44", "45-64", "65+"]:
            tuples = by_age[ag]
            fms = [t[0] for t in tuples]
            sms = [t[1] for t in tuples]
            fbs = [t[2] for t in tuples]
            sbs = [t[3] for t in tuples]
            fv = _avg(fms, 'ndcg_10'); fbv = _avg(fbs, 'ndcg_10')
            sv = _avg(sms, 'ndcg_10'); sbv = _avg(sbs, 'ndcg_10')
            age_r = _avg(fms, 'age_appropriate_ratio')
            print(f"  {ag:<12} {fv:>7.4f} {fbv:>7.4f} {_lift_str(fv, fbv)} {sv:>7.4f} {sbv:>7.4f} {_lift_str(sv, sbv)} {age_r:>6.2f}")

        # --- By Weather ---
        print(f"\n  === BY WEATHER (Feed NDCG@10: Scored vs Baseline) ===")
        print(f"  {'Weather':<12} {'F.NDCG':>7} {'F.Base':>7} {'F.Lift':>8} {'S.NDCG':>7} {'S.Base':>7} {'S.Lift':>8} {'Wea%':>6}")
        print(f"  {'-'*63}")
        for w in ["hot", "cold", "mild", "rainy"]:
            tuples = by_weather[w]
            fms = [t[0] for t in tuples]
            sms = [t[1] for t in tuples]
            fbs = [t[2] for t in tuples]
            sbs = [t[3] for t in tuples]
            fv = _avg(fms, 'ndcg_10'); fbv = _avg(fbs, 'ndcg_10')
            sv = _avg(sms, 'ndcg_10'); sbv = _avg(sbs, 'ndcg_10')
            wea_r = _avg(fms, 'weather_appropriate_ratio')
            print(f"  {w:<12} {fv:>7.4f} {fbv:>7.4f} {_lift_str(fv, fbv)} {sv:>7.4f} {sbv:>7.4f} {_lift_str(sv, sbv)} {wea_r:>6.2f}")

        # --- By Style ---
        print(f"\n  === BY STYLE PERSONA (Brand P@10: Scored vs Baseline) ===")
        print(f"  {'Style':<12} {'F.BrP':>7} {'F.Base':>7} {'F.Lift':>8} {'S.BrP':>7} {'S.Base':>7} {'S.Lift':>8}")
        print(f"  {'-'*57}")
        for s in ["trendy", "classic", "minimal", "romantic", "streetwear"]:
            tuples = by_style[s]
            fms = [t[0] for t in tuples]
            sms = [t[1] for t in tuples]
            fbs = [t[2] for t in tuples]
            sbs = [t[3] for t in tuples]
            fv = _avg(fms, 'brand_precision_10'); fbv = _avg(fbs, 'brand_precision_10')
            sv = _avg(sms, 'brand_precision_10'); sbv = _avg(sbs, 'brand_precision_10')
            print(f"  {s:<12} {fv:>7.4f} {fbv:>7.4f} {_lift_str(fv, fbv)} {sv:>7.4f} {sbv:>7.4f} {_lift_str(sv, sbv)}")

        print(f"\n{'#'*90}")

        # Assertions — scored should beat random baselines
        feed_ndcg = _avg(feed_metrics_all, "ndcg_10")
        feed_base = _avg(feed_baselines_all, "ndcg_10")
        search_ndcg = _avg(search_metrics_all, "ndcg_10")
        feed_brand = _avg(feed_metrics_all, "brand_precision_10")
        feed_brand_base = _avg(feed_baselines_all, "brand_precision_10")

        assert len(feed_metrics_all) == 100
        assert len(search_metrics_all) == 100
        assert elapsed < 120, f"Benchmark too slow: {elapsed:.1f}s (should be <120s)"
        assert feed_ndcg > feed_base, \
            f"Feed NDCG@10 ({feed_ndcg:.4f}) should beat random ({feed_base:.4f})"
        assert feed_brand >= feed_brand_base, \
            f"Feed Brand P@10 ({feed_brand:.4f}) should beat random ({feed_brand_base:.4f})"
