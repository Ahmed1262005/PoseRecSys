"""
Recommendation System Demo — Interactive Gradio UI.

5-tab interface showing how the scoring pipeline works:
  Tab 1: Premade Profiles — pick a profile, see feed before/after actions
  Tab 2: Interactive Playground — live actions + feed regeneration
  Tab 3: Manual Profile Builder — build a profile from scratch
  Tab 4: Action Deep Dive — action sequences + score evolution
  Tab 5: Search Results — how search reranking changes with actions

Uses REAL product data from Supabase (images, attributes, prices, brands).
Default pool: 5,000 products (override with DEMO_POOL_SIZE env var).

Usage:
    cd /mnt/d/ecommerce/recommendationSystem
    source .venv/bin/activate
    PYTHONPATH=src python scripts/rec_demo_gradio.py
    # -> http://localhost:7861
"""

import copy
import html as _html
import json
import math
import os
import random
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# -- path setup ---------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import gradio as gr

# -- real pipeline imports -----------------------------------------------------
from scoring.scorer import ContextScorer
from scoring.context import AgeGroup, Season, WeatherContext, UserContext
from scoring.profile_scorer import ProfileScorer
from scoring.constants.brand_data import (
    BRAND_TO_CLUSTER as SHARED_BRAND_TO_CLUSTER,
    derive_price_range as shared_derive_price_range,
)
from recs.session_scoring import (
    SessionScoringEngine, SessionScores,
    extract_intent_filters, extract_search_signals,
)
from recs.feed_reranker import GreedyConstrainedReranker, RerankerConfig
from search.reranker import SessionReranker
import traceback
import functools
import logging

logger = logging.getLogger("rec_demo")


# =============================================================================
# BRAND PRICING & CLUSTERS
# =============================================================================

# Average price per brand (USD) — sourced from retailer data
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

# Case-insensitive lookup
_BRAND_PRICE_LOWER = {k.lower(): v for k, v in BRAND_AVG_PRICE.items()}

# Brand clusters — brands in the same cluster are style-adjacent
BRAND_CLUSTERS: Dict[str, List[str]] = {
    "A": ["Aritzia", "Anthropologie", "Banana Republic", "COS", "J.Crew",
          "Sandro", "Sézane", "Club Monaco", "Everlane", "Ann Taylor",
          "Brooks Brothers", "White House Black Market", "& Other Stories",
          "Massimo Dutti", "Ralph Lauren", "Madewell"],
    "B": ["7 For All Mankind", "AG Jeans", "Citizens of Humanity", "DL1961",
          "Joe's Jeans", "Mavi Jeans", "Paige", "Rag & Bone", "Re/Done",
          "Good American", "Hudson", "Agolde", "Mother Denim", "Moussy"],
    "C": ["Abercrombie & Fitch", "Aeropostale", "American Eagle Outfitters",
          "Levi's", "Gap", "Lee", "Hollister", "Lucky Brand"],
    "E": ["Adidas", "New Balance", "Nike", "Puma"],
    "F": ["Skechers", "Champion"],
    "G": ["Zara", "Mango", "Express", "Uniqlo", "Oak + Fort", "Old Navy",
          "H&M", "Quince"],
    "I": ["The North Face", "Patagonia", "Arc'teryx", "Columbia", "Carhartt",
          "Cotopaxi"],
    "J": ["Garage", "Princess Polly", "Pull&Bear"],
    "K": ["A.P.C", "Ba&sh", "Cuyana", "Equipment", "Jenni Kayne", "L'AGENCE",
          "Nanushka", "Rails", "Reiss", "the frankie shop", "Anine Bing",
          "AllSaints", "Theory", "Vince", "Dissh"],
    "M": ["Brandy Melville", "Urban Outfitters", "Free People"],
    "P": ["Calvin Klein", "Tommy Hilfiger", "DKNY", "Guess"],
    "Q": ["Reformation", "Staud"],
    "S": ["Posse", "Sir the label"],
    "T": ["Alexis", "Andres Otalora", "PatBo", "Aje", "ALC", "Cult Gaia",
          "Joanna Ortiz", "Rachel Gilbert", "Shona Joy", "Simkhai",
          "Silvia Tcherassi"],
    "U": ["Skims"],
    "W": ["Alemais", "Arcina Ori"],
    "X": ["Diesel", "True Religion"],
}

# Reverse lookup: brand -> cluster ID
BRAND_TO_CLUSTER: Dict[str, str] = {}
for _cid, _brands in BRAND_CLUSTERS.items():
    for _b in _brands:
        BRAND_TO_CLUSTER[_b.lower()] = _cid


def derive_price_range(brands: List[str]) -> Tuple[float, float]:
    """Derive a price range from selected brands' average prices.

    Strategy:
      - Compute the mean avg price of selected brands
      - min_price = 0.4x of mean (floor $5)
      - max_price = 2.0x of mean (ceiling $1000)
      This gives a comfortable range that covers the brands' price tier
      while allowing discovery of similarly-priced items.

    If no brands are selected or none have pricing data, returns (0, 500).
    """
    prices = []
    for b in brands:
        p = _BRAND_PRICE_LOWER.get(b.lower())
        if p is not None:
            prices.append(p)

    if not prices:
        return (0.0, 500.0)

    mean_price = sum(prices) / len(prices)
    min_price = max(5.0, round(mean_price * 0.4, -1))   # Round to nearest 10
    max_price = min(1000.0, round(mean_price * 2.0, -1))

    # Ensure min < max
    if min_price >= max_price:
        max_price = min_price + 50

    return (min_price, max_price)


# =============================================================================
# REAL DATA FETCHER
# =============================================================================

class RealCandidate:
    """Lightweight candidate object mimicking the Candidate interface.
    Uses __slots__ for speed and to avoid MagicMock overhead."""

    __slots__ = [
        "item_id", "product_id", "brand", "broad_category", "article_type",
        "fit", "color_family", "colors", "pattern", "formality", "neckline",
        "sleeve", "length", "style_tags", "occasions", "seasons", "materials",
        "name", "image_url", "gallery_images", "price", "original_price",
        "is_new", "is_on_sale", "final_score", "embedding_score",
        "preference_score", "sasrec_score", "source",
        "coverage_level", "skin_exposure", "coverage_details",
        "model_body_type", "model_size_estimate",
    ]

    def __init__(self, **kwargs):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))


def _flatten_product(row: dict) -> dict:
    """Flatten a Supabase product row + nested product_attributes into a single dict."""
    attrs = row.pop("product_attributes", None) or {}

    # Extract construction sub-fields
    construction = attrs.get("construction") or {}

    return {
        "product_id": str(row.get("id", "")),
        "name": row.get("name") or "Unknown",
        "brand": row.get("brand") or "Unknown",
        "category": row.get("category") or "",
        "broad_category": (attrs.get("category_l1") or "").lower() if (attrs.get("category_l1") or "").lower() in ("tops", "bottoms", "dresses", "outerwear") else "",
        "article_type": attrs.get("category_l2") or row.get("article_type") or "",
        "price": float(row.get("price") or 0),
        "original_price": float(row.get("original_price") or 0),
        "is_on_sale": bool(row.get("original_price") and row["original_price"] > (row.get("price") or 0)),
        "is_new": False,
        "image_url": row.get("primary_image_url") or "",
        "gallery_images": row.get("gallery_images") or [],
        "colors": row.get("colors") or [],
        "materials": row.get("materials") or [],
        "fit": (row.get("fit") or attrs.get("fit_type") or "").title(),
        "neckline": row.get("neckline") or construction.get("neckline") or "",
        "sleeve": row.get("sleeve") or construction.get("sleeve_type") or "",
        "length": row.get("length") or construction.get("length") or "",
        "style_tags": attrs.get("style_tags") or row.get("style_tags") or [],
        "occasions": attrs.get("occasions") or [],
        "seasons": attrs.get("seasons") or [],
        "pattern": attrs.get("pattern") or "",
        "formality": attrs.get("formality") or "",
        "color_family": attrs.get("color_family") or "",
        "fit_type": (attrs.get("fit_type") or row.get("fit") or "").title(),
        "primary_color": attrs.get("primary_color") or "",
        "apparent_fabric": attrs.get("apparent_fabric") or "",
        "silhouette": attrs.get("silhouette") or construction.get("silhouette") or "",
        "sleeve_type": construction.get("sleeve_type") or row.get("sleeve") or "",
        "trending_score": float(row.get("trending_score") or 0),
        # Additional Gemini attributes for style mapping
        "sheen": attrs.get("sheen") or "",
        "texture": attrs.get("texture") or "",
        "pattern_scale": attrs.get("pattern_scale") or "",
        "trend_tags": attrs.get("trend_tags") or [],
        # Coverage & body type (from Gemini Vision)
        "coverage_level": attrs.get("coverage_level") or "",
        "skin_exposure": attrs.get("skin_exposure") or "",
        "coverage_details": attrs.get("coverage_details") or [],
        "model_body_type": attrs.get("model_body_type") or "",
        "model_size_estimate": attrs.get("model_size_estimate") or "",
    }


def _row_to_candidate(flat: dict, idx: int) -> RealCandidate:
    """Convert a flattened product dict into a RealCandidate."""
    rng = random.Random(hash(flat["product_id"]))
    return RealCandidate(
        item_id=flat["product_id"],
        product_id=flat["product_id"],
        brand=flat["brand"],
        broad_category=flat["broad_category"],
        article_type=flat["article_type"],
        fit=flat["fit"],
        color_family=flat["color_family"],
        colors=flat["colors"],
        pattern=flat["pattern"],
        formality=flat["formality"],
        neckline=flat["neckline"],
        sleeve=flat["sleeve"],
        length=flat["length"],
        style_tags=flat["style_tags"],
        occasions=flat["occasions"],
        seasons=flat["seasons"],
        materials=flat["materials"],
        name=flat["name"],
        image_url=flat["image_url"],
        gallery_images=flat["gallery_images"],
        price=flat["price"],
        original_price=flat["original_price"],
        is_new=flat["is_new"],
        is_on_sale=flat["is_on_sale"],
        coverage_level=flat.get("coverage_level", ""),
        skin_exposure=flat.get("skin_exposure", ""),
        coverage_details=flat.get("coverage_details", []),
        model_body_type=flat.get("model_body_type", ""),
        model_size_estimate=flat.get("model_size_estimate", ""),
        final_score=round(rng.uniform(0.4, 0.7), 4),
        embedding_score=round(rng.uniform(0.4, 0.95), 4),
        preference_score=round(rng.uniform(0.3, 0.8), 4),
        sasrec_score=round(rng.uniform(0.2, 0.85), 4),
        source=rng.choice(["taste_vector", "trending", "exploration"]),
    )


def _row_to_search_result(flat: dict) -> dict:
    """Convert a flattened product dict into a search-result dict."""
    rng = random.Random(hash(flat["product_id"]) + 999)
    d = dict(flat)
    d["rrf_score"] = round(rng.uniform(0.01, 0.08), 4)
    d["source"] = rng.choice(["algolia", "semantic"])
    return d


DEMO_POOL_SIZE = int(os.getenv("DEMO_POOL_SIZE", "5000"))


def fetch_product_pool() -> Tuple[List[dict], List[RealCandidate], List[dict]]:
    """Fetch a diverse subset of in-stock products from Supabase with attributes.

    Fetches DEMO_POOL_SIZE products (default 5000) by sampling evenly-spaced
    batches across the entire DB.  This gives good brand/category diversity
    while keeping startup under 10 seconds.

    Returns (flat_dicts, candidates, search_results).
    """
    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY env vars required")

    sb = create_client(url, key)

    SELECT_COLS = (
        "id, name, brand, category, broad_category, article_type, "
        "price, original_price, in_stock, fit, length, sleeve, "
        "neckline, rise, base_color, colors, materials, style_tags, "
        "primary_image_url, gallery_images, trending_score, "
        "gender, "
        "product_attributes!left("
        "  category_l1, category_l2, category_l3, "
        "  construction, primary_color, color_family, secondary_colors, "
        "  pattern, pattern_scale, apparent_fabric, texture, sheen, "
        "  style_tags, occasions, seasons, formality, trend_tags, "
        "  fit_type, stretch, rise, leg_shape, silhouette, "
        "  coverage_level, skin_exposure, coverage_details, "
        "  model_body_type, model_size_estimate"
        ")"
    )

    # Get total count
    count_result = sb.table("products").select("id", count="exact").eq(
        "in_stock", True
    ).not_.is_("primary_image_url", "null").execute()
    total = count_result.count or 0
    # total known, skip logging

    target = min(DEMO_POOL_SIZE, total)
    batch_size = 1000
    all_rows: list = []

    if total <= target:
        # Small enough to fetch everything
        offset = 0
        while True:
            t0 = time.time()
            result = sb.table("products").select(SELECT_COLS).eq(
                "in_stock", True
            ).not_.is_("primary_image_url", "null").range(
                offset, offset + batch_size - 1
            ).execute()
            if not result.data:
                break
            all_rows.extend(result.data)
            elapsed = time.time() - t0
            # batch progress (silent)
            offset += batch_size
            if len(result.data) < batch_size:
                break
    else:
        # Sample evenly-spaced batches across the DB for diversity
        num_batches = math.ceil(target / batch_size)
        stride = max(batch_size, total // num_batches)
        offsets = [i * stride for i in range(num_batches)]

        for idx, off in enumerate(offsets):
            if len(all_rows) >= target:
                break
            t0 = time.time()
            result = sb.table("products").select(SELECT_COLS).eq(
                "in_stock", True
            ).not_.is_("primary_image_url", "null").range(
                off, off + batch_size - 1
            ).execute()
            if not result.data:
                continue
            all_rows.extend(result.data)
            elapsed = time.time() - t0
            # batch progress (silent)
            if len(all_rows) >= target:
                break

    # Trim to target if we over-fetched
    if len(all_rows) > target:
        all_rows = all_rows[:target]

    # fetched, skip logging

    # Flatten and convert
    flats = [_flatten_product(row) for row in all_rows]
    flats = [f for f in flats if f["image_url"]]
    # filtered, skip logging

    candidates = [_row_to_candidate(f, i) for i, f in enumerate(flats)]
    search_results = [_row_to_search_result(f) for f in flats]

    return flats, candidates, search_results


# =============================================================================
# PROFILE DEFINITIONS
# =============================================================================

@dataclass
class DemoProfile:
    """A user profile for the demo."""
    profile_id: str
    name: str
    age: int
    age_group: str
    style_persona: str
    city: str
    country: str
    temperature_c: float
    weather_condition: str
    season: str
    is_hot: bool = False
    is_cold: bool = False
    is_rainy: bool = False
    preferred_brands: List[str] = field(default_factory=list)
    preferred_categories: List[str] = field(default_factory=list)
    preferred_article_types: List[str] = field(default_factory=list)
    preferred_styles: List[str] = field(default_factory=list)
    preferred_occasions: List[str] = field(default_factory=list)
    preferred_colors: List[str] = field(default_factory=list)
    preferred_patterns: List[str] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    min_price: float = 0.0
    max_price: float = 200.0


def _build_premade_profiles() -> Dict[str, DemoProfile]:
    """Build ~20 curated demo profiles."""
    profiles = {}

    defs = [
        ("Gen Z Trendy in Miami", 21, "18-24", "trendy", "Miami", "US", 33.0, "clear", "summer",
         True, False, False,
         ["Zara", "Boohoo", "Forever 21", "PrettyLittleThing", "Asos"],
         ["tops", "dresses"], ["crop_top", "mini_dress", "bodycon_dress", "tube_top", "shorts"],
         ["trendy", "casual"], ["Party", "Date Night", "Weekend"],
         ["Black", "Red", "Pink"], ["Solid", "Floral"],
         ["trendy party dress", "cute crop top"], 80.0),

        ("Gen Z Streetwear in Chicago", 20, "18-24", "streetwear", "Chicago", "US", -5.0, "snow", "winter",
         False, True, False,
         ["Nike", "Adidas", "New Balance", "Champion", "Puma"],
         ["tops", "bottoms"], ["hoodie", "joggers", "tshirt", "sweatshirt", "puffer"],
         ["streetwear", "sporty"], ["Weekend", "Casual", "School"],
         ["Black", "White", "Green"], ["Solid"],
         ["oversized hoodie", "nike joggers"], 120.0),

        ("Gen Z Romantic in Paris", 22, "18-24", "romantic", "Paris", "FR", 18.0, "clouds", "spring",
         False, False, False,
         ["Reformation", "Ba&sh", "Maje", "Sandro", "Free People"],
         ["dresses", "tops"], ["wrap_dress", "midi_dress", "blouse", "slip_dress"],
         ["romantic", "elegant", "boho"], ["Date Night", "Brunch"],
         ["Pink", "Lavender", "White"], ["Floral"],
         ["floral wrap dress", "romantic date outfit"], 250.0),

        ("Young Pro Classic in NYC", 29, "25-34", "classic", "New York", "US", 5.0, "clouds", "winter",
         False, True, False,
         ["J.Crew", "Banana Republic", "Ann Taylor", "Mango", "Cos"],
         ["tops", "bottoms"], ["blazer", "blouse", "pants", "midi_dress", "coat"],
         ["classic", "elegant"], ["Work", "Formal", "Interview"],
         ["Navy", "Beige", "White"], ["Solid", "Striped"],
         ["office blazer", "classic white blouse"], 200.0),

        ("Young Adult Minimal in Copenhagen", 27, "25-34", "minimal", "Copenhagen", "DK", 12.0, "rain", "fall",
         False, False, True,
         ["Cos", "Everlane", "Uniqlo", "The Frankie Shop", "Arket"],
         ["tops", "bottoms"], ["tshirt", "pants", "sweater", "jeans", "blazer"],
         ["minimal", "classic"], ["Everyday", "Work", "Travel"],
         ["Black", "White", "Grey"], ["Solid"],
         ["minimal black sweater", "capsule wardrobe basics"], 150.0),

        ("Young Adult Boho in LA", 26, "25-34", "boho", "Los Angeles", "US", 28.0, "clear", "summer",
         True, False, False,
         ["Free People", "Anthropologie", "Spell", "Reformation", "Faithfull the Brand"],
         ["dresses", "tops"], ["maxi_dress", "sundress", "blouse", "romper", "cami"],
         ["boho", "romantic"], ["Vacation", "Beach", "Festival"],
         ["Beige", "White", "Coral"], ["Floral", "Paisley"],
         ["boho maxi dress", "festival outfit"], 180.0),

        ("Mid-Career Elegant in Milan", 40, "35-44", "classic", "Milan", "IT", 22.0, "clear", "spring",
         False, False, False,
         ["Mango", "Massimo Dutti", "Reiss", "Theory", "Vince"],
         ["tops", "bottoms"], ["blazer", "blouse", "midi_dress", "pants", "coat"],
         ["elegant", "classic"], ["Work", "Formal", "Date Night"],
         ["Navy", "Beige", "Burgundy"], ["Solid"],
         ["elegant office blazer", "tailored trousers"], 300.0),

        ("Mid-Career Sporty in Denver", 38, "35-44", "sporty", "Denver", "US", 15.0, "clear", "spring",
         False, False, False,
         ["Lululemon", "Alo Yoga", "Nike", "Vuori", "Beyond Yoga"],
         ["tops", "bottoms"], ["leggings", "hoodie", "tshirt", "joggers", "tank_top"],
         ["sporty", "athleisure", "casual"], ["Gym", "Weekend", "Everyday"],
         ["Black", "Grey", "Navy"], ["Solid"],
         ["workout leggings", "athleisure outfit"], 150.0),

        ("Mid-Career Trendy in London", 36, "35-44", "trendy", "London", "UK", 10.0, "rain", "fall",
         False, False, True,
         ["AllSaints", "Aritzia", "Whistles", "& Other Stories", "Reiss"],
         ["tops", "outerwear"], ["blazer", "sweater", "coat", "jeans", "midi_dress"],
         ["trendy", "classic"], ["Work", "Weekend", "Date Night"],
         ["Black", "Olive", "Burgundy"], ["Solid", "Plaid"],
         ["trench coat outfit", "fall layering ideas"], 250.0),

        ("Established Classic in Boston", 52, "45-64", "classic", "Boston", "US", -2.0, "snow", "winter",
         False, True, False,
         ["J.Crew", "Ralph Lauren", "Ann Taylor", "Banana Republic", "Tommy Hilfiger"],
         ["tops", "bottoms"], ["blazer", "coat", "pants", "blouse", "cardigan"],
         ["classic", "elegant"], ["Work", "Formal"],
         ["Navy", "Beige", "Grey"], ["Solid", "Striped"],
         ["classic wool coat", "business casual outfit"], 250.0),

        ("Established Elegant in Dubai", 48, "45-64", "elegant", "Dubai", "AE", 38.0, "clear", "summer",
         True, False, False,
         ["Max Mara", "Zimmermann", "L'Agence", "Veronica Beard", "Theory"],
         ["dresses", "tops"], ["midi_dress", "blouse", "blazer", "maxi_dress"],
         ["elegant", "classic"], ["Formal", "Date Night", "Vacation"],
         ["White", "Beige", "Navy"], ["Solid"],
         ["elegant summer dress", "resort wear"], 400.0),

        ("Senior Classic in Savannah", 70, "65+", "classic", "Savannah", "US", 25.0, "clear", "spring",
         False, False, False,
         ["Ralph Lauren", "Ann Taylor", "J.Crew", "Talbots", "Banana Republic"],
         ["tops", "bottoms"], ["cardigan", "blazer", "pants", "blouse", "shift_dress"],
         ["classic"], ["Everyday", "Work"],
         ["Navy", "Beige", "White"], ["Solid"],
         ["comfortable classic cardigan", "elegant blouse"], 200.0),

        ("Senior Elegant in Vienna", 68, "65+", "elegant", "Vienna", "AT", 2.0, "snow", "winter",
         False, True, False,
         ["Max Mara", "Hobbs", "Phase Eight", "Mint Velvet", "Reiss"],
         ["tops", "outerwear"], ["coat", "cardigan", "sweater", "blazer", "turtleneck"],
         ["elegant", "classic"], ["Formal", "Everyday"],
         ["Grey", "Burgundy", "Navy"], ["Solid", "Houndstooth"],
         ["wool coat", "cashmere sweater"], 350.0),

        ("Budget Shopper in Manila", 23, "18-24", "trendy", "Manila", "PH", 32.0, "rain", "summer",
         True, False, True,
         ["Shein", "Fashion Nova", "Boohoo", "Forever 21", "Nasty Gal"],
         ["tops", "dresses"], ["crop_top", "mini_dress", "tank_top", "sundress"],
         ["trendy", "glam"], ["Party", "Date Night"],
         ["Pink", "Red", "Black"], ["Solid", "Floral"],
         ["cheap party dress", "affordable outfit"], 50.0),

        ("Luxury Minimal in Tokyo", 35, "35-44", "minimal", "Tokyo", "JP", 8.0, "clouds", "fall",
         False, False, False,
         ["The Row", "Toteme", "Khaite", "Cos", "The Frankie Shop"],
         ["tops", "bottoms"], ["blazer", "pants", "sweater", "coat", "tshirt"],
         ["minimal", "elegant"], ["Work", "Everyday"],
         ["Black", "White", "Grey"], ["Solid"],
         ["quiet luxury blazer", "minimalist wardrobe"], 500.0),

        ("Festival Lover in Austin", 24, "18-24", "boho", "Austin", "US", 35.0, "clear", "summer",
         True, False, False,
         ["Free People", "Spell", "Farm Rio", "Anthropologie", "Princess Polly"],
         ["dresses", "tops"], ["maxi_dress", "romper", "crop_top", "cami", "sundress"],
         ["boho", "romantic", "trendy"], ["Festival", "Beach", "Vacation"],
         ["Coral", "Mustard", "Teal"], ["Floral", "Tie-Dye", "Paisley"],
         ["festival outfit ideas", "boho maxi dress"], 120.0),

        ("Curvy Confident in Atlanta", 30, "25-34", "trendy", "Atlanta", "US", 22.0, "clear", "spring",
         False, False, False,
         ["Good American", "Universal Standard", "Skims", "Fashion Nova", "Boohoo"],
         ["dresses", "bottoms"], ["bodycon_dress", "jeans", "wrap_dress", "midi_dress"],
         ["trendy", "glam"], ["Date Night", "Party", "Weekend"],
         ["Black", "Red", "Pink"], ["Solid"],
         ["curvy friendly dress", "high waist jeans"], 100.0),

        ("Eco-Conscious in Portland", 31, "25-34", "minimal", "Portland", "US", 14.0, "rain", "fall",
         False, False, True,
         ["Reformation", "Everlane", "Christy Dawn", "Girlfriend Collective", "Doen"],
         ["tops", "bottoms"], ["tshirt", "jeans", "sweater", "blazer", "midi_dress"],
         ["minimal", "classic", "boho"], ["Everyday", "Work"],
         ["Beige", "Olive", "White"], ["Solid"],
         ["sustainable fashion", "ethical wardrobe"], 200.0),

        ("Night Out Queen in Vegas", 25, "25-34", "glam", "Las Vegas", "US", 30.0, "clear", "summer",
         True, False, False,
         ["Oh Polly", "House of CB", "Meshki", "Revolve", "Tiger Mist"],
         ["dresses", "tops"], ["bodycon_dress", "mini_dress", "bodysuit", "crop_top"],
         ["glam", "trendy", "edgy"], ["Party", "Date Night"],
         ["Black", "Red", "White"], ["Solid"],
         ["going out dress", "party outfit"], 150.0),

        ("Work From Home in Seattle", 33, "25-34", "casual", "Seattle", "US", 11.0, "rain", "fall",
         False, False, True,
         ["Everlane", "Gap", "Uniqlo", "Madewell", "Aritzia"],
         ["tops", "bottoms"], ["sweater", "joggers", "tshirt", "cardigan", "leggings"],
         ["casual", "minimal"], ["Lounge", "Everyday", "Travel"],
         ["Grey", "Beige", "Navy"], ["Solid"],
         ["cozy work from home outfit", "comfortable loungewear"], 100.0),
    ]

    for d in defs:
        pid = d[0].lower().replace(" ", "_").replace("'", "")
        p = DemoProfile(
            profile_id=pid, name=d[0], age=d[1], age_group=d[2],
            style_persona=d[3], city=d[4], country=d[5],
            temperature_c=d[6], weather_condition=d[7], season=d[8],
            is_hot=d[9], is_cold=d[10], is_rainy=d[11],
            preferred_brands=d[12], preferred_categories=d[13],
            preferred_article_types=d[14], preferred_styles=d[15],
            preferred_occasions=d[16], preferred_colors=d[17],
            preferred_patterns=d[18],
            search_queries=d[19], max_price=d[20],
        )
        profiles[p.name] = p

    return profiles


# Premade action scenarios
ACTION_SCENARIOS = {
    "Clicked 5 Zara dresses": [
        ("click", "Zara", "midi_dress", {"fit": "regular", "pattern": "Floral"}),
        ("click", "Zara", "wrap_dress", {"fit": "regular", "pattern": "Solid"}),
        ("click", "Zara", "mini_dress", {"fit": "slim", "neckline": "V-Neck"}),
        ("click", "Zara", "slip_dress", {"fit": "slim", "pattern": "Solid"}),
        ("click", "Zara", "midi_dress", {"fit": "regular", "neckline": "Scoop"}),
    ],
    "Skipped all fast-fashion, searched 'quiet luxury blazer'": [
        ("skip", "Boohoo", "bodycon_dress", {}),
        ("skip", "Shein", "crop_top", {}),
        ("skip", "Fashion Nova", "mini_dress", {}),
        ("skip", "PrettyLittleThing", "tube_top", {}),
        ("skip", "Forever 21", "tank_top", {}),
        ("search", "", "", {"query": "quiet luxury blazer", "filters": {"styles": ["minimal", "elegant"]}}),
    ],
    "Added Nike hoodie to cart, searched 'streetwear'": [
        ("add_to_cart", "Nike", "hoodie", {"fit": "oversized", "pattern": "Solid"}),
        ("click", "Nike", "joggers", {"fit": "relaxed"}),
        ("click", "Adidas", "sweatshirt", {"fit": "oversized"}),
        ("search", "", "", {"query": "streetwear essentials", "filters": {"styles": ["streetwear"]}}),
    ],
    "Browsed work wear, added blazer to wishlist": [
        ("click", "Theory", "blazer", {"fit": "regular", "formality": "Smart Casual"}),
        ("click", "Reiss", "pants", {"fit": "regular", "formality": "Smart Casual"}),
        ("add_to_wishlist", "J.Crew", "blazer", {"fit": "regular"}),
        ("click", "Ann Taylor", "blouse", {"fit": "regular", "neckline": "Crew"}),
        ("search", "", "", {"query": "office outfit", "filters": {"occasions": ["Work"]}}),
    ],
    "Summer vacation shopping -- bright dresses": [
        ("click", "Reformation", "sundress", {"pattern": "Floral"}),
        ("click", "Farm Rio", "maxi_dress", {"pattern": "Abstract"}),
        ("add_to_cart", "Free People", "romper", {"pattern": "Floral"}),
        ("click", "Spell", "midi_dress", {"pattern": "Paisley"}),
        ("search", "", "", {"query": "summer vacation dress", "filters": {"colors": ["Coral", "Teal"]}}),
    ],
    "Heavy skipper -- very picky": [
        ("skip", "Boohoo", "crop_top", {}),
        ("skip", "Shein", "bodycon_dress", {}),
        ("skip", "Forever 21", "mini_skirt", {}),
        ("skip", "Nasty Gal", "tube_top", {}),
        ("skip", "Fashion Nova", "bralette", {}),
        ("skip", "Missguided", "bodysuit", {}),
        ("click", "Reformation", "midi_dress", {"fit": "regular", "pattern": "Solid"}),
    ],
    "Purchased a coat, looking for winter layers": [
        ("purchase", "Max Mara", "coat", {"fit": "regular", "pattern": "Solid"}),
        ("click", "Cos", "turtleneck", {"fit": "regular"}),
        ("click", "& Other Stories", "sweater", {"fit": "regular"}),
        ("add_to_wishlist", "Arket", "cardigan", {"fit": "relaxed"}),
        ("search", "", "", {"query": "winter layering", "filters": {"seasons": ["Winter"]}}),
    ],
}


# =============================================================================
# CANDIDATE / DICT CONVERSION HELPERS
# =============================================================================

def candidate_to_dict(c: RealCandidate) -> dict:
    """Convert a RealCandidate to a dict for ContextScorer / ProfileScorer."""
    return {
        "product_id": c.item_id, "article_type": c.article_type or "",
        "broad_category": c.broad_category or "", "brand": c.brand or "",
        "price": float(c.price or 0),
        "style_tags": c.style_tags or [], "occasions": c.occasions or [],
        "pattern": c.pattern or "", "formality": c.formality or "",
        "fit_type": c.fit or "", "neckline": c.neckline or "",
        "sleeve_type": c.sleeve or "", "length": c.length or "",
        "color_family": c.color_family or "", "seasons": c.seasons or [],
        "materials": c.materials or [], "name": c.name or "",
        "image_url": c.image_url or "",
        "coverage_level": c.coverage_level or "",
        "skin_exposure": c.skin_exposure or "",
        "coverage_details": c.coverage_details or [],
        "model_body_type": c.model_body_type or "",
        "model_size_estimate": c.model_size_estimate or "",
    }


def candidate_to_feed_dict(c: RealCandidate, flat_idx: Optional[int] = None) -> dict:
    """Convert candidate to a displayable dict with scores."""
    gen_styles = []
    if flat_idx is not None and flat_idx < len(ALL_FLATS):
        gen_styles = ALL_FLATS[flat_idx].get("general_styles") or []
    return {
        "product_id": c.item_id, "name": c.name, "brand": c.brand,
        "article_type": c.article_type, "broad_category": c.broad_category,
        "price": c.price, "image_url": c.image_url,
        "gallery_images": getattr(c, "gallery_images", []) or [],
        "style_tags": c.style_tags, "occasions": c.occasions,
        "general_styles": gen_styles,
        "pattern": c.pattern, "formality": c.formality,
        "fit_type": c.fit, "color_family": c.color_family,
        "neckline": c.neckline, "sleeve_type": c.sleeve,
        "length": c.length, "seasons": c.seasons, "materials": c.materials,
        "is_on_sale": c.is_on_sale, "is_new": c.is_new,
        "final_score": c.final_score,
        "source": c.source,
        "coverage_level": c.coverage_level, "skin_exposure": c.skin_exposure,
        "coverage_details": c.coverage_details,
        "model_body_type": c.model_body_type, "model_size_estimate": c.model_size_estimate,
    }


def clone_candidate(c: RealCandidate) -> RealCandidate:
    """Deep-clone a RealCandidate."""
    return RealCandidate(**{slot: getattr(c, slot) for slot in RealCandidate.__slots__})


_INAPPROPRIATE_KEYWORDS = {
    "girls'", "boys'", "kids", "toddler", "infant", "baby", "pajama",
    "pyjama", "sleepwear", "nightgown", "maternity", "nursing",
}


def _is_inappropriate_item(c: RealCandidate) -> bool:
    """Filter out kids, maternity, pajamas, and other irrelevant items."""
    name_lower = (c.name or "").lower()
    return any(kw in name_lower for kw in _INAPPROPRIATE_KEYWORDS)


def get_candidates_for_profile(profile: DemoProfile, all_candidates: List[RealCandidate],
                                n: int = 200) -> List[RealCandidate]:
    """Sample n candidates with 3-tier brand priority.

    Tier 1 (60%): Preferred brands — user explicitly chose these
    Tier 2 (30%): Cluster-adjacent brands — same style cluster, not explicitly chosen
    Tier 3 (10%): Random — diversity / discovery from outside the cluster
    Filters out kids, pajamas, maternity items.
    """
    rng = random.Random(hash(profile.profile_id))
    pref_brands = {b.lower() for b in (profile.preferred_brands or [])}

    # Find cluster-adjacent brands (same clusters as preferred, but not preferred themselves)
    pref_clusters: Set[str] = set()
    for b in pref_brands:
        cid = BRAND_TO_CLUSTER.get(b)
        if cid:
            pref_clusters.add(cid)

    cluster_brands: Set[str] = set()
    for cid in pref_clusters:
        for b in BRAND_CLUSTERS.get(cid, []):
            bl = b.lower()
            if bl not in pref_brands:
                cluster_brands.add(bl)

    # Bucket items into 3 tiers
    tier1: List[RealCandidate] = []  # preferred brands
    tier2: List[RealCandidate] = []  # cluster-adjacent brands
    tier3: List[RealCandidate] = []  # everything else
    for c in all_candidates:
        if _is_inappropriate_item(c):
            continue
        bl = (c.brand or "").lower()
        if pref_brands and bl in pref_brands:
            tier1.append(c)
        elif cluster_brands and bl in cluster_brands:
            tier2.append(c)
        else:
            tier3.append(c)

    rng.shuffle(tier1)
    rng.shuffle(tier2)
    rng.shuffle(tier3)

    if not pref_brands:
        return tier3[:n]

    # Allocate: 60% preferred, 30% cluster-adjacent, 10% random
    budget1 = min(len(tier1), int(n * 0.60))
    budget2 = min(len(tier2), int(n * 0.30))
    budget3 = n - budget1 - budget2

    selected = tier1[:budget1] + tier2[:budget2] + tier3[:max(0, budget3)]

    # If we're short (e.g. not enough cluster brands), backfill from remaining tiers
    if len(selected) < n:
        remaining = (tier1[budget1:] + tier2[budget2:] + tier3[max(0, budget3):])
        rng.shuffle(remaining)
        selected.extend(remaining[:n - len(selected)])

    rng.shuffle(selected)
    return selected[:n]


def inject_session_intent_candidates(
    base_candidates: List[RealCandidate],
    all_candidates: List[RealCandidate],
    session_scores: SessionScores,
    max_inject: int = 50,
) -> List[RealCandidate]:
    """Replace tail candidates with items matching session intent signals.

    When a user searches for "alo yoga leggings" repeatedly, the session EMA
    builds up brand/type affinity.  This function ensures the 200-candidate
    pool actually *contains* items matching those signals, so session scoring
    can meaningfully reorder them.

    Args:
        base_candidates: Current candidate pool (e.g. 200 items).
        all_candidates: Full startup pool (e.g. 5000 items).
        session_scores: Live session scores with EMA signals.
        max_inject: Maximum number of intent candidates to inject.

    Returns:
        Updated candidate list (same length or longer if pool was short).
    """
    intent = extract_intent_filters(session_scores)
    if not intent["has_intent"]:
        return base_candidates

    intent_brands = {b.lower() for b in intent["brands"]}
    intent_types = {t.lower() for t in intent["types"]}

    # IDs already in the base pool
    existing_ids = {c.item_id for c in base_candidates}

    # Find matching items in the full pool that aren't already present
    matching = []
    for c in all_candidates:
        if c.item_id in existing_ids:
            continue
        brand_match = c.brand and c.brand.lower() in intent_brands
        type_match = c.article_type and c.article_type.lower() in intent_types
        if brand_match or type_match:
            matching.append(c)
        if len(matching) >= max_inject:
            break

    if not matching:
        return base_candidates

    # Replace the LAST N items (lowest priority) with intent candidates
    n_replace = min(len(matching), max_inject, len(base_candidates))
    result = base_candidates[:-n_replace] + matching[:n_replace]

    print(f"[GradioDemo] Injected {n_replace} session-intent candidates "
          f"(brands={intent['brands']}, types={intent['types']})")

    return result


def get_search_results_for_profile(profile: DemoProfile, all_results: List[dict],
                                    n: int = 60) -> List[dict]:
    """Sample n search results from the pool."""
    rng = random.Random(hash(profile.profile_id) + 7)
    pool = list(all_results)
    rng.shuffle(pool)
    return pool[:n]


# =============================================================================
# PROFILE -> CONTEXT HELPERS
# =============================================================================

def profile_to_user_context(p: DemoProfile) -> UserContext:
    age_map = {
        "18-24": AgeGroup.GEN_Z, "25-34": AgeGroup.YOUNG_ADULT,
        "35-44": AgeGroup.MID_CAREER, "45-64": AgeGroup.ESTABLISHED,
        "65+": AgeGroup.SENIOR,
    }
    season_map = {
        "spring": Season.SPRING, "summer": Season.SUMMER,
        "fall": Season.FALL, "winter": Season.WINTER,
    }
    weather = WeatherContext(
        temperature_c=p.temperature_c, feels_like_c=p.temperature_c,
        condition=p.weather_condition, humidity=60, wind_speed_mps=5.0,
        season=season_map[p.season], is_hot=p.is_hot, is_cold=p.is_cold,
        is_mild=not p.is_hot and not p.is_cold, is_rainy=p.is_rainy,
    )
    return UserContext(
        user_id=p.profile_id, age_group=age_map[p.age_group],
        age_years=p.age, city=p.city, country=p.country,
        weather=weather,
    )


def profile_to_search_profile(p: DemoProfile) -> dict:
    """Convert a DemoProfile to search profile dict.

    Uses flat field names that ProfileScorer expects (not nested soft_prefs/hard_filters).
    """
    return {
        "preferred_brands": p.preferred_brands or [],
        "style_persona": [p.style_persona] if p.style_persona else [],
        "preferred_styles": p.preferred_styles or [],
        "preferred_colors": p.preferred_colors or [],
        "preferred_patterns": p.preferred_patterns or [],
        "preferred_fits": [],
        "preferred_sleeves": [],
        "preferred_lengths": [],
        "preferred_necklines": [],
        "preferred_formality": [],
        "occasions": p.preferred_occasions or [],
        "exclude_brands": [],
        "colors_to_avoid": [],
        "styles_to_avoid": [],
        "patterns_liked": p.preferred_patterns or [],
        "patterns_avoided": [],
        "global_min_price": p.min_price if p.min_price > 0 else None,
        "global_max_price": p.max_price if p.max_price > 0 else None,
    }


# =============================================================================
# SCORING PIPELINE
# =============================================================================

context_scorer = ContextScorer()
profile_scorer = ProfileScorer()
session_engine = SessionScoringEngine()
feed_reranker = GreedyConstrainedReranker(RerankerConfig())
search_reranker = SessionReranker()


def demo_profile_to_scorer_dict(p: "DemoProfile") -> dict:
    """Convert a DemoProfile to a dict that ProfileScorer understands.

    Maps DemoProfile fields to the OnboardingProfile-compatible field names
    expected by ProfileScorer.
    """
    return {
        "preferred_brands": p.preferred_brands or [],
        "brand_openness": None,
        "style_persona": [p.style_persona] if p.style_persona else [],
        "style_directions": [],
        "preferred_fits": [],
        "fit_category_mapping": [],
        "preferred_sleeves": [],
        "sleeve_category_mapping": [],
        "preferred_lengths": [],
        "length_category_mapping": [],
        "preferred_lengths_dresses": [],
        "length_dresses_category_mapping": [],
        "preferred_necklines": [],
        "preferred_rises": [],
        "top_types": [t for t in (p.preferred_article_types or [])
                      if any(k in t for k in ("top", "tshirt", "blouse", "sweater", "cardigan", "hoodie", "cami", "tank", "bodysuit"))],
        "bottom_types": [t for t in (p.preferred_article_types or [])
                         if any(k in t for k in ("jeans", "pants", "shorts", "skirt", "jogger", "legging", "trouser"))],
        "dress_types": [t for t in (p.preferred_article_types or [])
                        if any(k in t for k in ("dress", "jumpsuit", "romper"))],
        "outerwear_types": [t for t in (p.preferred_article_types or [])
                            if any(k in t for k in ("jacket", "coat", "blazer", "puffer", "vest"))],
        "patterns_liked": p.preferred_patterns or [],
        "patterns_avoided": [],
        "occasions": p.preferred_occasions or [],
        "colors_to_avoid": [],
        "global_min_price": p.min_price if p.min_price > 0 else None,
        "global_max_price": p.max_price if p.max_price > 0 else None,
        "no_crop": False,
        "no_revealing": False,
        "no_deep_necklines": False,
        "no_sleeveless": False,
        "no_tanks": False,
        "styles_to_avoid": [],
    }


# User-facing persona choices for Tab 3 dropdown
# These are matched directly against item style_tags by ProfileScorer
# (no PERSONA_STYLE_MAP expansion needed -- the structured data IS the persona)
PERSONA_CHOICES = [
    "Business Casual", "Casual", "Trendy", "Classic", "Elegant",
    "Minimal", "Boho", "Romantic", "Streetwear", "Sporty",
    "Athleisure", "Glam", "Edgy", "Preppy", "Vintage",
]


def run_feed_pipeline(profile: DemoProfile, session_scores: Optional[SessionScores] = None,
                      candidates: Optional[List[RealCandidate]] = None, n: int = 20,
                      seen_ids: Optional[set] = None) -> List[dict]:
    """Run the full feed pipeline: profile + context scoring -> session scoring -> reranking."""
    if candidates is None:
        candidates = get_cached_candidates(profile, 200)

    ctx = profile_to_user_context(profile)

    # Deep copy candidates so we don't mutate originals
    working = [clone_candidate(c) for c in candidates]

    # Step 1: Profile scoring via shared ProfileScorer (attribute-driven)
    scorer_dict = demo_profile_to_scorer_dict(profile)
    profile_adjs = {}
    for w in working:
        item_dict = candidate_to_dict(w)
        padj = profile_scorer.score_item(item_dict, scorer_dict)
        w.final_score = w.final_score + padj
        profile_adjs[w.item_id] = padj

    # Step 2: Context scoring (age + weather)
    context_adjs = {}
    for w in working:
        item_dict = candidate_to_dict(w)
        adj = context_scorer.score_item(item_dict, ctx)
        w.final_score = w.final_score + adj
        context_adjs[w.item_id] = adj

    # Step 3: Session scoring
    session_adjs = {}
    if session_scores:
        blended = session_engine.compute_blended_scores(session_scores)
        for w in working:
            sess_score = session_engine.score_item(session_scores, w, blended)
            session_adjs[w.item_id] = sess_score
        scored = session_engine.score_candidates(session_scores, working)
        working = scored

    # Step 4: Feed reranking (diversity constraints + seen_ids exclusion)
    reranked = feed_reranker.rerank(working, target_size=n, seen_ids=seen_ids)

    # Convert to display dicts with score breakdowns
    result = []
    for i, c in enumerate(reranked):
        d = candidate_to_feed_dict(c)
        d["rank"] = i + 1
        d["profile_adjustment"] = round(profile_adjs.get(c.item_id, 0.0), 4)
        d["context_adj"] = round(context_adjs.get(c.item_id, 0.0), 4)
        d["session_adj"] = round(session_adjs.get(c.item_id, 0.0), 4)
        result.append(d)
    return result


# Category display names and mapping from broad_category values
FEED_CATEGORIES = [
    ("Tops", {"tops", "knitwear", "shirts", "blouses", "sweaters"}),
    ("Bottoms", {"bottoms", "jeans", "pants", "trousers", "shorts", "skirts"}),
    ("Dresses", {"dresses", "dress", "jumpsuits", "rompers"}),
    ("Outerwear", {"outerwear", "jackets", "coats"}),
]


# Material group mapping — maps apparent_fabric keywords to shopper-friendly groups
MATERIAL_GROUPS: dict[str, list[str]] = {
    "Cotton": ["cotton", "cotton blend", "organic cotton"],
    "Polyester": ["polyester", "polyester blend", "recycled polyester"],
    "Denim": ["denim"],
    "Knit": ["knit", "ribbed knit", "knitted fabric", "jersey"],
    "Silk & Satin": ["silk", "silk blend", "satin", "charmeuse"],
    "Linen": ["linen", "linen blend"],
    "Wool & Cashmere": ["wool", "wool blend", "cashmere", "merino"],
    "Leather & Suede": ["leather", "faux leather", "suede", "faux suede"],
    "Lace & Mesh": ["lace", "mesh", "crochet", "tulle"],
    "Chiffon & Sheer": ["chiffon", "crepe", "georgette"],
    "Velvet & Plush": ["velvet", "velour", "fleece", "faux fur", "terry", "french terry"],
    "Nylon & Spandex": ["nylon", "nylon blend", "polyamide", "spandex", "lycra"],
    "Viscose & Rayon": ["viscose", "viscose blend", "rayon", "rayon blend", "modal", "modal blend", "lyocell", "lyocell blend"],
}

# Build reverse lookup: fabric keyword -> group name
_FABRIC_TO_GROUP: dict[str, str] = {}
for _group, _fabrics in MATERIAL_GROUPS.items():
    for _fab in _fabrics:
        _FABRIC_TO_GROUP[_fab] = _group


def _map_to_material_group(flat: dict) -> str:
    """Map a product's apparent_fabric to a material group. Returns '' if no match."""
    fabric = (flat.get("apparent_fabric") or "").strip().lower()
    return _FABRIC_TO_GROUP.get(fabric, "")


def _classify_category(item: dict) -> str:
    """Map an item's broad_category/article_type to a display category."""
    bcat = (item.get("broad_category") or "").lower()
    atype = (item.get("article_type") or "").lower()
    for label, keywords in FEED_CATEGORIES:
        if bcat in keywords:
            return label
        if any(kw in atype for kw in keywords):
            return label
    return "Other"


def run_categorized_feed(profile: DemoProfile, session_scores: Optional[SessionScores] = None,
                         per_category: int = 20) -> Dict[str, List[dict]]:
    """Run the feed pipeline and return results grouped by clothing category.

    Uses a larger candidate pool (500) to ensure enough items per category,
    runs the scoring pipeline once, then splits into category buckets.
    """
    candidates = get_cached_candidates(profile, 500)
    # Run full scoring pipeline on all 500
    all_items = run_feed_pipeline(profile, session_scores=session_scores,
                                  candidates=candidates, n=200)

    # Filter out coverage-killed items (score < 0 means coverage violation)
    all_items = [item for item in all_items if float(item.get("final_score", 0) or 0) > 0]

    # Bucket into categories
    buckets: Dict[str, List[dict]] = {}
    for item in all_items:
        cat = _classify_category(item)
        if cat not in buckets:
            buckets[cat] = []
        buckets[cat].append(item)

    # Trim each bucket and re-rank within category
    result: Dict[str, List[dict]] = {}
    for label, _ in FEED_CATEGORIES:
        items = buckets.get(label, [])[:per_category]
        for i, item in enumerate(items):
            item["rank"] = i + 1
        if items:
            result[label] = items

    # Add "Other" if any
    other = buckets.get("Other", [])[:10]
    for i, item in enumerate(other):
        item["rank"] = i + 1
    if other:
        result["Other"] = other

    return result


def render_categorized_feed(categorized: Dict[str, List[dict]], title: str = "Feed") -> str:
    """Render a categorized feed as HTML with sections."""
    parts = [f"<h2>{_html.escape(title)}</h2>"]
    for cat_label in [l for l, _ in FEED_CATEGORIES] + ["Other"]:
        items = categorized.get(cat_label)
        if not items:
            continue
        parts.append(f'<div class="category-section"><h3>{_html.escape(cat_label)}</h3>')
        for item in items:
            parts.append(render_product_card(item))
        parts.append('</div>')
    return "\n".join(parts)


def run_search_pipeline(profile: DemoProfile, session_scores: Optional[SessionScores] = None,
                        results: Optional[List[dict]] = None, n: int = 20) -> List[dict]:
    """Run search reranking pipeline."""
    if results is None:
        results = get_search_results_for_profile(profile, ALL_SEARCH_RESULTS, 60)

    ctx = profile_to_user_context(profile)
    search_prof = profile_to_search_profile(profile)

    # Deep copy
    working = [dict(r) for r in results]

    # Run search reranker
    reranked = search_reranker.rerank(
        working, user_profile=search_prof, max_per_brand=4,
        user_context=ctx, session_scores=session_scores,
    )

    for i, r in enumerate(reranked[:n]):
        r["rank"] = i + 1
    return reranked[:n]


def apply_actions(session_scores: SessionScores, actions: list) -> SessionScores:
    """Apply a list of actions to session scores."""
    for action_tuple in actions:
        action_type = action_tuple[0]
        if action_type == "search":
            info = action_tuple[3] if len(action_tuple) > 3 else {}
            session_engine.process_search_signal(
                session_scores,
                query=info.get("query", ""),
                filters=info.get("filters", {}),
            )
        else:
            brand = action_tuple[1] if len(action_tuple) > 1 else ""
            item_type = action_tuple[2] if len(action_tuple) > 2 else ""
            attributes = action_tuple[3] if len(action_tuple) > 3 else {}
            pid = f"action_{brand}_{item_type}".lower().replace(" ", "_")
            session_engine.process_action(
                session_scores, action=action_type, product_id=pid,
                brand=brand, item_type=item_type, attributes=attributes,
            )
    return session_scores


# =============================================================================
# CSS
# =============================================================================

CUSTOM_CSS = """
.result-card {
    display: flex; gap: 14px; padding: 14px; margin: 8px 0;
    border: 1px solid #e0e0e0; border-radius: 12px; background: #fafafa;
    transition: box-shadow 0.2s;
}
.result-card:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
.card-images { display: flex; flex-direction: column; gap: 4px; flex-shrink: 0; }
.card-image { width: 100px; height: 135px; object-fit: cover; border-radius: 8px; }
.card-gallery { display: flex; gap: 3px; }
.card-thumb { width: 30px; height: 40px; object-fit: cover; border-radius: 4px; opacity: 0.85; }
.card-thumb:hover { opacity: 1; }
.card-image-placeholder {
    width: 100px; height: 135px; background: #eee; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: #999; font-size: 11px; text-align: center; padding: 4px;
}
.card-body { flex: 1; min-width: 0; }
.category-section { margin: 16px 0 8px 0; padding: 10px 0; border-top: 2px solid #e0e0e0; }
.category-section h3 { margin: 0 0 8px 0; color: #1a237e; font-size: 16px; }
.card-title { font-weight: 600; font-size: 13px; margin-bottom: 2px; line-height: 1.3; }
.card-brand-price { color: #555; font-size: 12px; margin-bottom: 3px; }
.card-rank { color: #888; font-size: 10px; font-family: monospace; margin-bottom: 3px; }
.card-attrs { color: #555; font-size: 10px; margin-top: 2px; line-height: 1.4; }
.card-tags { margin-top: 3px; display: flex; flex-wrap: wrap; gap: 3px; }
.card-tag {
    display: inline-block; padding: 1px 6px; border-radius: 8px;
    font-size: 9px; background: #e8eaf6; color: #3949ab;
}
.card-tag-occasion { background: #fff8e1; color: #f57f17; }

.score-breakdown {
    margin-top: 4px; padding: 4px 8px; background: #f5f5f5; border-radius: 6px;
    font-size: 10px; font-family: monospace; color: #333; line-height: 1.6;
}
.score-pos { color: #2e7d32; font-weight: 600; }
.score-neg { color: #c62828; font-weight: 600; }
.score-neutral { color: #888; }

.profile-card {
    padding: 16px; border: 1px solid #e0e0e0; border-radius: 12px;
    background: #f8f9ff; margin: 8px 0;
}
.profile-card h3 { margin: 0 0 8px 0; font-size: 16px; color: #1a237e; }
.profile-detail { font-size: 12px; color: #555; margin: 2px 0; }
.profile-badge {
    display: inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 11px; font-weight: 600; margin: 2px;
}
.badge-age { background: #e3f2fd; color: #1565c0; }
.badge-weather { background: #fff3e0; color: #e65100; }
.badge-style { background: #f3e5f5; color: #7b1fa2; }
.badge-brand { background: #e8f5e9; color: #2e7d32; }

.diff-card {
    display: flex; gap: 10px; padding: 10px; margin: 6px 0;
    border: 1px solid #e0e0e0; border-radius: 10px;
}
.diff-up { border-left: 4px solid #4caf50; background: #f1f8e9; }
.diff-down { border-left: 4px solid #ef5350; background: #fce4ec; }
.diff-new { border-left: 4px solid #2196f3; background: #e3f2fd; }
.diff-same { border-left: 4px solid #bdbdbd; background: #fafafa; }
.diff-arrow { font-size: 14px; font-weight: 700; min-width: 50px; text-align: center; }
.diff-arrow-up { color: #4caf50; }
.diff-arrow-down { color: #ef5350; }
.diff-arrow-new { color: #2196f3; }

.session-state {
    padding: 12px; background: #fffde7; border: 1px solid #fff9c4;
    border-radius: 10px; font-size: 11px; font-family: monospace;
}
.session-state h4 { margin: 0 0 6px 0; font-size: 13px; color: #f57f17; }

.evolution-step {
    padding: 10px; margin: 6px 0; border: 1px solid #e0e0e0;
    border-radius: 8px; background: #fafafa;
}
.evolution-step h4 { margin: 0 0 4px 0; font-size: 12px; color: #1565c0; }

.filter-hint { padding: 20px; text-align: center; color: #666; }
.filter-count { padding: 8px 12px; background: #f0f4ff; border-radius: 8px; margin-bottom: 8px; color: #222; }

/* Dark mode overrides */
.dark .result-card { background: #1e1e2e; border-color: #444; }
.dark .result-card:hover { box-shadow: 0 2px 12px rgba(255,255,255,0.06); }
.dark .card-title { color: #e0e0e0; }
.dark .card-brand-price { color: #aaa; }
.dark .card-rank { color: #888; }
.dark .card-attrs { color: #aaa; }
.dark .card-tag { background: #2a2d4a; color: #8e99f3; }
.dark .card-tag-occasion { background: #3e2f1a; color: #ffb74d; }
.dark .card-image-placeholder { background: #2a2a3a; color: #888; }
.dark .score-breakdown { background: #252535; color: #ccc; }
.dark .profile-card { background: #1e1e2e; border-color: #444; }
.dark .profile-card h3 { color: #8e99f3; }
.dark .profile-detail { color: #aaa; }
.dark .diff-card { border-color: #444; }
.dark .diff-up { background: #1b2e1b; }
.dark .diff-down { background: #2e1b1b; }
.dark .diff-new { background: #1b2333; }
.dark .diff-same { background: #1e1e2e; }
.dark .session-state { background: #2a2518; border-color: #444; color: #ccc; }
.dark .session-state h4 { color: #ffb74d; }
.dark .evolution-step { background: #1e1e2e; border-color: #444; }
.dark .evolution-step h4 { color: #8e99f3; }
.dark .category-section { border-color: #444; }
.dark .category-section h3 { color: #8e99f3; }
.dark .filter-hint { color: #aaa; }
.dark .filter-count { background: #1e2340; color: #e0e0e0; }
"""


# =============================================================================
# HTML RENDERING
# =============================================================================

def _score_span(val: float, label: str = "") -> str:
    if val > 0.001:
        cls, prefix = "score-pos", "+"
    elif val < -0.001:
        cls, prefix = "score-neg", ""
    else:
        cls, prefix = "score-neutral", ""
    text = f"{label}{prefix}{val:.4f}" if label else f"{prefix}{val:.4f}"
    return f'<span class="{cls}">{_html.escape(text)}</span>'


def render_product_card(item: dict, show_scores: bool = True) -> str:
    rank = item.get("rank", "?")
    name = _html.escape(str(item.get("name", "Unknown")))
    brand = _html.escape(str(item.get("brand", "Unknown")))
    price = float(item.get("price", 0) or 0)
    atype = str(item.get("article_type", ""))
    img = item.get("image_url", "")
    pid = str(item.get("product_id", ""))[:12]
    sale = item.get("is_on_sale", False)
    is_new = item.get("is_new", False)

    gallery = item.get("gallery_images") or []

    if img:
        img_html = f'<img class="card-image" src="{_html.escape(img)}" alt="{name}" loading="lazy">'
        # Add gallery thumbnails (up to 3 extra)
        if gallery:
            gallery_imgs = [g for g in gallery[:3] if g and g != img]
            if gallery_imgs:
                thumbs = "".join(
                    f'<img class="card-thumb" src="{_html.escape(g)}" loading="lazy">'
                    for g in gallery_imgs
                )
                img_html += f'<div class="card-gallery">{thumbs}</div>'
    else:
        img_html = f'<div class="card-image-placeholder">{_html.escape(atype)}</div>'

    price_str = f"${price:.2f}"
    if sale:
        orig = float(item.get("original_price", 0) or 0)
        if orig > price:
            price_str = f'<span style="text-decoration:line-through;color:#999">${orig:.2f}</span> <span style="color:#e74c3c;font-weight:bold">${price:.2f}</span>'
        else:
            price_str = f'<span style="color:#e74c3c;font-weight:bold">${price:.2f} SALE</span>'
    if is_new:
        price_str += ' <span style="color:#1565c0;font-weight:bold">NEW</span>'

    tags_html = ""
    # Show general style labels (pre-computed from style_tags + occasions + formality)
    gen_styles = item.get("general_styles") or []
    if not gen_styles:
        # Fallback to raw style_tags if general_styles not available
        gen_styles = (item.get("style_tags") or [])[:3]
    for s in gen_styles[:4]:
        tags_html += f'<span class="card-tag">{_html.escape(str(s))}</span>'

    score_html = ""
    if show_scores:
        final = float(item.get("final_score", 0) or item.get("rrf_score", 0) or 0)
        ctx_adj = float(item.get("context_adj", 0) or item.get("context_adjustment", 0) or 0)
        sess_adj = float(item.get("session_adj", 0) or item.get("session_adjustment", 0) or 0)
        prof_adj = float(item.get("profile_adjustment", 0) or 0)

        score_html = f'''<div class="score-breakdown">
            Final: <b>{final:.4f}</b>
            &nbsp;|&nbsp; Context: {_score_span(ctx_adj)}
            &nbsp;|&nbsp; Session: {_score_span(sess_adj)}
            &nbsp;|&nbsp; Profile: {_score_span(prof_adj)}
        </div>'''

    attrs = []
    for key in ["pattern", "fit_type", "neckline", "sleeve_type", "length", "color_family",
                 "coverage_level", "skin_exposure", "model_body_type", "model_size_estimate"]:
        v = item.get(key)
        if v and v not in ("Moderate", "Medium", "Unknown"):
            attrs.append(f"{key.replace('_', ' ').title()}: {v}")
    attrs_html = f'<div class="card-attrs">{" &middot; ".join(attrs[:7])}</div>' if attrs else ""

    return f'''<div class="result-card">
        <div class="card-images">{img_html}</div>
        <div class="card-body">
            <div class="card-rank">#{rank} &middot; {_html.escape(pid)} &middot; {_html.escape(str(item.get("source", "")))}</div>
            <div class="card-title">{name}</div>
            <div class="card-brand-price">{brand} &middot; {price_str}</div>
            {attrs_html}
            <div class="card-tags">{tags_html}</div>
            {score_html}
        </div>
    </div>'''


def render_feed(items: List[dict], title: str = "Feed") -> str:
    if not items:
        return f"<p><i>No items in {title}</i></p>"
    html_parts = [f"<h3>{_html.escape(title)}</h3>"]
    for item in items:
        html_parts.append(render_product_card(item))
    return "\n".join(html_parts)


def render_profile_card(p: DemoProfile) -> str:
    brands_html = "".join(f'<span class="profile-badge badge-brand">{_html.escape(b)}</span>' for b in p.preferred_brands[:5])
    styles_html = "".join(f'<span class="profile-badge badge-style">{_html.escape(s)}</span>' for s in p.preferred_styles)

    weather_icon = "Hot" if p.is_hot else "Cold" if p.is_cold else "Rainy" if p.is_rainy else "Mild"

    return f'''<div class="profile-card">
        <h3>{_html.escape(p.name)}</h3>
        <div class="profile-detail">
            <span class="profile-badge badge-age">Age: {p.age} ({p.age_group})</span>
            <span class="profile-badge badge-weather">{p.city}, {p.country} &middot; {p.temperature_c}C &middot; {weather_icon} &middot; {p.season.title()}</span>
        </div>
        <div class="profile-detail" style="margin-top:6px">Styles: {styles_html}</div>
        <div class="profile-detail">Brands: {brands_html}</div>
        <div class="profile-detail">Categories: {", ".join(p.preferred_categories)} &middot; Price range (from brands): ${p.min_price:.0f}-${p.max_price:.0f}</div>

    </div>'''


def render_session_state(scores: SessionScores) -> str:
    def _top_entries(d: dict, n: int = 5) -> str:
        """Render top-N entries from a Dict[str, PreferenceState] map."""
        if not d:
            return "<i>empty</i>"
        # PreferenceState values — sort by blended score magnitude
        sorted_items = sorted(
            d.items(),
            key=lambda x: abs(x[1].score() if hasattr(x[1], 'score') else x[1]),
            reverse=True,
        )[:n]
        parts = []
        for k, v in sorted_items:
            if hasattr(v, 'score'):
                # PreferenceState: show (fast, slow, n)
                s = v.score()
                cls = "score-pos" if s > 0 else "score-neg" if s < 0 else "score-neutral"
                parts.append(
                    f'<span class="{cls}">{_html.escape(k)}: '
                    f'f={v.fast:+.2f} s={v.slow:+.2f} n={v.count}</span>'
                )
            else:
                # Fallback for plain float (shouldn't happen but be safe)
                cls = "score-pos" if v > 0 else "score-neg" if v < 0 else "score-neutral"
                parts.append(f'<span class="{cls}">{_html.escape(k)}: {v:+.3f}</span>')
        return ", ".join(parts)

    skipped = ", ".join(list(scores.skipped_ids)[:10]) if scores.skipped_ids else "none"

    return f'''<div class="session-state">
        <h4>Session State ({scores.action_count} actions)</h4>
        <div><b>Brand scores:</b> {_top_entries(scores.brand_scores)}</div>
        <div><b>Type scores:</b> {_top_entries(scores.type_scores)}</div>
        <div><b>Attr scores:</b> {_top_entries(scores.attr_scores)}</div>
        <div><b>Search intents:</b> {_top_entries(scores.search_intents)}</div>
        <div><b>Cluster scores:</b> {_top_entries(scores.cluster_scores)}</div>
        <div><b>Skipped IDs:</b> {_html.escape(skipped)}</div>
    </div>'''


def render_diff(before: List[dict], after: List[dict]) -> str:
    before_map = {item["product_id"]: (i, item) for i, item in enumerate(before)}
    after_map = {item["product_id"]: (i, item) for i, item in enumerate(after)}

    html_parts = ["<h3>Feed Diff (Before -> After)</h3>"]

    for pid, (new_rank, item) in sorted(after_map.items(), key=lambda x: x[1][0]):
        if pid in before_map:
            old_rank = before_map[pid][0]
            delta = old_rank - new_rank
            if delta > 0:
                cls, arrow_cls = "diff-up", "diff-arrow-up"
                arrow = f"#{old_rank + 1} -> #{new_rank + 1} (+{delta})"
            elif delta < 0:
                cls, arrow_cls = "diff-down", "diff-arrow-down"
                arrow = f"#{old_rank + 1} -> #{new_rank + 1} ({delta})"
            else:
                cls, arrow_cls = "diff-same", ""
                arrow = f"#{new_rank + 1} (unchanged)"
        else:
            cls, arrow_cls = "diff-new", "diff-arrow-new"
            arrow = f"NEW at #{new_rank + 1}"

        name = _html.escape(str(item.get("name", "")))
        brand = _html.escape(str(item.get("brand", "")))

        # Read scores with fallback for search results (rrf_score / session_adjustment)
        after_score = float(item.get("final_score", 0) or item.get("rrf_score", 0) or 0)
        after_ctx = float(item.get("context_adj", 0) or item.get("context_adjustment", 0) or 0)
        after_sess = float(item.get("session_adj", 0) or item.get("session_adjustment", 0) or 0)

        # Compute deltas against "before" values when available
        if pid in before_map:
            b_item = before_map[pid][1]
            before_score = float(b_item.get("final_score", 0) or b_item.get("rrf_score", 0) or 0)
            before_ctx = float(b_item.get("context_adj", 0) or b_item.get("context_adjustment", 0) or 0)
            before_sess = float(b_item.get("session_adj", 0) or b_item.get("session_adjustment", 0) or 0)
            d_score = after_score - before_score
            d_ctx = after_ctx - before_ctx
            d_sess = after_sess - before_sess
        else:
            # New item — show absolute values
            d_score = after_score
            d_ctx = after_ctx
            d_sess = after_sess

        html_parts.append(f'''<div class="diff-card {cls}">
            <div class="diff-arrow {arrow_cls}">{arrow}</div>
            <div style="flex:1">
                <b>{name}</b> &middot; {brand} &middot;
                Score: {after_score:.4f} ({_score_span(d_score)})
                &nbsp;|&nbsp; ctx: {_score_span(d_ctx)} &nbsp;|&nbsp; sess: {_score_span(d_sess)}
            </div>
        </div>''')

    removed = set(before_map.keys()) - set(after_map.keys())
    if removed:
        html_parts.append("<h4>Removed from top-20:</h4>")
        for pid in list(removed)[:10]:
            old_rank, item = before_map[pid]
            name = _html.escape(str(item.get("name", "")))
            html_parts.append(f'<div class="diff-card diff-down"><div>#{old_rank + 1} {name} -- REMOVED</div></div>')

    return "\n".join(html_parts)


# =============================================================================
# GLOBAL STATE
# =============================================================================

print("Loading product data from Supabase...")
try:
    ALL_FLATS, ALL_CANDIDATES, ALL_SEARCH_RESULTS = fetch_product_pool()
except Exception as e:
    print(f"  ERROR: Failed to fetch products from Supabase: {e}")
    print("  Starting with empty product pool. Check SUPABASE_URL and SUPABASE_SERVICE_KEY.")
    ALL_FLATS, ALL_CANDIDATES, ALL_SEARCH_RESULTS = [], [], []

# Derive dropdown values from real data
DB_BRANDS = sorted(set(f["brand"] for f in ALL_FLATS if f.get("brand")))
_BRAND_DISPLAY_TO_NAME: dict[str, str] = {b: b for b in DB_BRANDS}  # identity until full pool overrides
DB_ARTICLE_TYPES = sorted(set(f["article_type"] for f in ALL_FLATS if f.get("article_type")))
DB_STYLES_RAW = sorted(set(s for f in ALL_FLATS for s in (f.get("style_tags") or [])))
DB_COLORS = sorted(set(c for f in ALL_FLATS for c in (f.get("colors") or [])))
DB_PATTERNS = sorted(set(f["pattern"] for f in ALL_FLATS if f.get("pattern")))
DB_OCCASIONS_RAW = sorted(set(o for f in ALL_FLATS for o in (f.get("occasions") or [])))

# --- General style options (replaces 100+ style_tags + occasions + formality) ---
GENERAL_STYLE_OPTIONS = [
    "Casual", "Classic", "Minimalist", "Preppy", "Office",
    "Activewear", "Streetwear", "Trendy", "Romantic", "Edgy",
    "Boho", "Night Out", "Occasionwear", "Statement", "Maternity", "Evening",
]

# Mapping from raw style_tags / occasions / formality values -> general style(s)
# Keys are lowercased.  Each raw value can map to one or more general styles.
_STYLE_TAG_MAP: Dict[str, List[str]] = {
    # --- Casual ---
    "casual": ["Casual"], "everyday": ["Casual"], "relaxed": ["Casual"],
    "comfortable": ["Casual"], "easy": ["Casual"], "laid-back": ["Casual"],
    "weekend": ["Casual"], "basic": ["Casual"], "cozy": ["Casual"],
    "lounge": ["Casual"], "loungewear": ["Casual"], "athleisure": ["Casual", "Activewear"],
    # --- Classic ---
    "classic": ["Classic"], "timeless": ["Classic"], "traditional": ["Classic"],
    "polished": ["Classic"], "refined": ["Classic"], "sophisticated": ["Classic"],
    "tailored": ["Classic"], "elegant": ["Classic", "Evening"],
    # --- Minimalist ---
    "minimal": ["Minimalist"], "minimalist": ["Minimalist"], "clean": ["Minimalist"],
    "simple": ["Minimalist"], "understated": ["Minimalist"], "modern": ["Minimalist"],
    "sleek": ["Minimalist"],
    # --- Preppy ---
    "preppy": ["Preppy"], "collegiate": ["Preppy"], "ivy": ["Preppy"],
    "heritage": ["Preppy"], "nautical": ["Preppy"], "sporty-chic": ["Preppy"],
    # --- Office ---
    "office": ["Office"], "work": ["Office"], "professional": ["Office"],
    "business": ["Office"], "business casual": ["Office"], "corporate": ["Office"],
    "workwear": ["Office"], "smart casual": ["Office"],
    # --- Activewear ---
    "activewear": ["Activewear"], "athletic": ["Activewear"], "sporty": ["Activewear"],
    "sportswear": ["Activewear"], "workout": ["Activewear"], "fitness": ["Activewear"],
    "performance": ["Activewear"], "gym": ["Activewear"], "running": ["Activewear"],
    "yoga": ["Activewear"], "training": ["Activewear"],
    # --- Streetwear ---
    "streetwear": ["Streetwear"], "street": ["Streetwear"], "urban": ["Streetwear"],
    "hip-hop": ["Streetwear"], "skate": ["Streetwear"], "graphic": ["Streetwear"],
    "oversized": ["Streetwear"],
    # --- Trendy ---
    "trendy": ["Trendy"], "fashion-forward": ["Trendy"], "on-trend": ["Trendy"],
    "contemporary": ["Trendy"], "modern chic": ["Trendy"], "y2k": ["Trendy"],
    "retro": ["Trendy"], "vintage-inspired": ["Trendy"], "cutout": ["Trendy"],
    "bold": ["Trendy", "Statement"],
    # --- Romantic ---
    "romantic": ["Romantic"], "feminine": ["Romantic"], "floral": ["Romantic"],
    "girly": ["Romantic"], "soft": ["Romantic"], "delicate": ["Romantic"],
    "whimsical": ["Romantic"], "dreamy": ["Romantic"], "pretty": ["Romantic"],
    "dainty": ["Romantic"], "cottagecore": ["Romantic"],
    # --- Edgy ---
    "edgy": ["Edgy"], "grunge": ["Edgy"], "punk": ["Edgy"], "goth": ["Edgy"],
    "rock": ["Edgy"], "rebellious": ["Edgy"], "alternative": ["Edgy"],
    "dark": ["Edgy"], "distressed": ["Edgy"], "leather": ["Edgy"],
    "moto": ["Edgy"],
    # --- Boho ---
    "boho": ["Boho"], "bohemian": ["Boho"], "hippie": ["Boho"],
    "free-spirited": ["Boho"], "earthy": ["Boho"], "artisan": ["Boho"],
    "crochet": ["Boho"], "festival": ["Boho"], "tribal": ["Boho"],
    "eclectic": ["Boho"],
    # --- Night Out ---
    "night out": ["Night Out"], "going out": ["Night Out"], "club": ["Night Out"],
    "party": ["Night Out"], "clubwear": ["Night Out"], "sexy": ["Night Out"],
    "date night": ["Night Out"], "date": ["Night Out"], "nightlife": ["Night Out"],
    # --- Occasionwear ---
    "occasion": ["Occasionwear"], "wedding": ["Occasionwear"], "wedding guest": ["Occasionwear"],
    "bridal": ["Occasionwear"], "formal": ["Occasionwear", "Evening"],
    "cocktail": ["Occasionwear"], "ceremony": ["Occasionwear"],
    "special occasion": ["Occasionwear"], "prom": ["Occasionwear"],
    "graduation": ["Occasionwear"], "bridesmaid": ["Occasionwear"],
    # --- Statement ---
    "statement": ["Statement"], "dramatic": ["Statement"], "maximalist": ["Statement"],
    "avant-garde": ["Statement"], "artistic": ["Statement"], "loud": ["Statement"],
    "show-stopping": ["Statement"], "eye-catching": ["Statement"],
    # --- Maternity ---
    "maternity": ["Maternity"], "pregnancy": ["Maternity"], "nursing": ["Maternity"],
    "bump-friendly": ["Maternity"],
    # --- Evening ---
    "evening": ["Evening"], "gala": ["Evening"], "black tie": ["Evening"],
    "dinner": ["Evening"], "upscale": ["Evening"], "luxe": ["Evening"],
    "glamorous": ["Evening"], "glam": ["Evening"],
}

# Occasion -> general style mapping
_OCCASION_MAP: Dict[str, List[str]] = {
    "everyday": ["Casual"], "casual": ["Casual"], "weekend": ["Casual"],
    "work": ["Office"], "office": ["Office"], "business": ["Office"],
    "professional": ["Office"],
    "party": ["Night Out"], "club": ["Night Out"], "night out": ["Night Out"],
    "nightlife": ["Night Out"], "date night": ["Night Out"], "date": ["Night Out"],
    "going out": ["Night Out"],
    "wedding": ["Occasionwear"], "wedding guest": ["Occasionwear"],
    "formal event": ["Occasionwear"], "ceremony": ["Occasionwear"],
    "cocktail": ["Occasionwear"], "prom": ["Occasionwear"],
    "graduation": ["Occasionwear"], "bridal": ["Occasionwear"],
    "special occasion": ["Occasionwear"],
    "evening": ["Evening"], "dinner": ["Evening"], "gala": ["Evening"],
    "black tie": ["Evening"],
    "gym": ["Activewear"], "workout": ["Activewear"], "sports": ["Activewear"],
    "yoga": ["Activewear"], "running": ["Activewear"], "training": ["Activewear"],
    "vacation": ["Casual", "Boho"], "travel": ["Casual"],
    "festival": ["Boho"], "beach": ["Casual", "Boho"],
}

# Formality -> general style mapping
_FORMALITY_MAP: Dict[str, List[str]] = {
    "very casual": ["Casual"], "casual": ["Casual"],
    "smart casual": ["Office", "Classic"], "business casual": ["Office"],
    "business": ["Office"], "business formal": ["Office", "Classic"],
    "semi-formal": ["Evening", "Occasionwear"],
    "formal": ["Evening", "Occasionwear"],
    "black tie": ["Evening"],
}


def _map_to_general_styles(flat: dict) -> List[str]:
    """Map a product's attributes to general style labels.

    Uses style_tags + occasions + formality keyword maps, PLUS attribute-based
    heuristics for Statement (and other styles that benefit from richer signals).
    """
    result: Set[str] = set()

    # --- 1. Keyword maps (style_tags, occasions, formality) ---
    for tag in (flat.get("style_tags") or []):
        key = tag.strip().lower()
        if key in _STYLE_TAG_MAP:
            result.update(_STYLE_TAG_MAP[key])

    for occ in (flat.get("occasions") or []):
        key = occ.strip().lower()
        if key in _OCCASION_MAP:
            result.update(_OCCASION_MAP[key])

    formality = (flat.get("formality") or "").strip().lower()
    if formality in _FORMALITY_MAP:
        result.update(_FORMALITY_MAP[formality])

    # Also check trend_tags through the style_tag map
    for tag in (flat.get("trend_tags") or []):
        key = tag.strip().lower()
        if key in _STYLE_TAG_MAP:
            result.update(_STYLE_TAG_MAP[key])

    # --- 2. Attribute-based Statement detection ---
    # Products with bold/dramatic attributes are Statement even if style_tags
    # don't explicitly say so.  We use a scoring approach: each signal adds
    # a point, and >= 2 points triggers Statement.
    if "Statement" not in result:
        statement_score = 0

        # Pattern signals (bold prints)
        pattern = (flat.get("pattern") or "").lower()
        _STATEMENT_PATTERNS = {
            "animal print", "animal", "leopard", "zebra", "snake",
            "abstract", "graphic", "color block", "colour block",
            "tie-dye", "tie dye", "tropical", "baroque",
            "geometric", "camouflage", "camo",
        }
        if pattern in _STATEMENT_PATTERNS:
            statement_score += 1

        # Large-scale pattern
        pattern_scale = (flat.get("pattern_scale") or "").lower()
        if pattern_scale == "large":
            statement_score += 1

        # Sheen signals (metallic, sequin, glitter)
        sheen = (flat.get("sheen") or "").lower()
        _STATEMENT_SHEENS = {"metallic", "sequin", "glitter", "holographic", "iridescent"}
        if sheen in _STATEMENT_SHEENS:
            statement_score += 2  # Strong signal

        # Texture signals (dramatic textures)
        texture = (flat.get("texture") or "").lower()
        _STATEMENT_TEXTURES = {
            "faux fur", "fur", "feathered", "feather", "velvet",
            "sequin", "beaded", "embossed", "quilted",
        }
        if texture in _STATEMENT_TEXTURES:
            statement_score += 1

        # Fabric signals (statement fabrics)
        fabric = (flat.get("apparent_fabric") or "").lower()
        _STATEMENT_FABRICS = {
            "sequin", "metallic", "leather", "faux leather", "patent",
            "vinyl", "pvc", "latex", "faux fur", "fur", "feather",
            "velvet", "brocade", "organza", "tulle",
        }
        if fabric in _STATEMENT_FABRICS:
            statement_score += 1

        # Neckline signals (dramatic necklines)
        neckline = (flat.get("neckline") or "").lower()
        _STATEMENT_NECKLINES = {
            "plunging", "deep v", "one shoulder", "one-shoulder",
            "asymmetric", "asymmetrical", "strapless", "halter",
            "off shoulder", "off-shoulder", "bardot",
        }
        if neckline in _STATEMENT_NECKLINES:
            statement_score += 1

        # Silhouette signals
        silhouette = (flat.get("silhouette") or "").lower()
        _STATEMENT_SILHOUETTES = {
            "asymmetric", "asymmetrical", "dramatic", "voluminous",
            "sculptural", "exaggerated", "balloon", "cocoon",
        }
        if silhouette in _STATEMENT_SILHOUETTES:
            statement_score += 1

        # Coverage details signals (bold/revealing design choices)
        coverage_details = flat.get("coverage_details") or []
        _STATEMENT_COVERAGE = {
            "cutouts", "sheer_panels", "sheer", "high_slit",
            "backless", "open_back", "midriff_exposed",
            "lace_up", "corset",
        }
        for cd in coverage_details:
            if cd.strip().lower() in _STATEMENT_COVERAGE:
                statement_score += 1
                break  # Count once

        # Primary color signals (bold, attention-grabbing colors)
        primary_color = (flat.get("primary_color") or "").lower()
        _STATEMENT_COLORS = {
            "red", "hot pink", "fuchsia", "magenta", "neon",
            "electric blue", "cobalt", "emerald", "gold", "silver",
            "orange", "yellow", "lime", "purple",
        }
        if primary_color in _STATEMENT_COLORS:
            statement_score += 1

        # Threshold: 2+ signals = Statement
        if statement_score >= 2:
            result.add("Statement")

    return sorted(result) if result else []


# Pre-compute general_styles and material_group for every product
for _flat in ALL_FLATS:
    _flat["general_styles"] = _map_to_general_styles(_flat)
    _flat["material_group"] = _map_to_material_group(_flat)

# Additional dropdown values for Filter Explorer (Tab 6)
DB_FORMALITY = sorted(set(f.get("formality") for f in ALL_FLATS if f.get("formality")))
DB_FIT_TYPES = sorted(set(f.get("fit") or f.get("fit_type") for f in ALL_FLATS if f.get("fit") or f.get("fit_type")))
DB_NECKLINES = sorted(set(f.get("neckline") for f in ALL_FLATS if f.get("neckline")))
DB_SLEEVE_TYPES = sorted(set(f.get("sleeve") or f.get("sleeve_type") for f in ALL_FLATS if f.get("sleeve") or f.get("sleeve_type")))
DB_LENGTHS = sorted(set(f.get("length") for f in ALL_FLATS if f.get("length")))
DB_SILHOUETTES = sorted(set(f.get("silhouette") for f in ALL_FLATS if f.get("silhouette")))
DB_RISES = sorted(set(f.get("rise") for f in ALL_FLATS if f.get("rise") and f.get("rise") not in ("", "N/A")))
DB_COLOR_FAMILIES = sorted(set(f.get("color_family") for f in ALL_FLATS if f.get("color_family")))
DB_PRIMARY_COLORS = sorted(set(f.get("primary_color") for f in ALL_FLATS if f.get("primary_color")))
DB_MATERIALS = sorted(set(m for f in ALL_FLATS for m in (f.get("materials") or []) if m))
DB_SEASONS = sorted(set(s for f in ALL_FLATS for s in (f.get("seasons") or []) if s))
DB_CATEGORIES = sorted(set(f.get("broad_category") for f in ALL_FLATS if f.get("broad_category") and f.get("broad_category").lower() != "accessories"))
DB_COVERAGE_LEVELS = sorted(set(f.get("coverage_level") for f in ALL_FLATS if f.get("coverage_level")))
DB_SKIN_EXPOSURE = sorted(set(f.get("skin_exposure") for f in ALL_FLATS if f.get("skin_exposure")))
DB_MODEL_BODY_TYPES = sorted(set(f.get("model_body_type") for f in ALL_FLATS if f.get("model_body_type")))
DB_MODEL_SIZE_ESTIMATES = sorted(set(f.get("model_size_estimate") for f in ALL_FLATS if f.get("model_size_estimate")))

    # globals ready, skip logging

PREMADE_PROFILES = _build_premade_profiles()
PROFILE_NAMES = list(PREMADE_PROFILES.keys())

_tab2_state: Dict[str, dict] = {}
_tab3_state: Dict[str, dict] = {}

# -- Candidate cache: avoids re-iterating 5K items on every button click ------
_candidate_cache: Dict[str, Tuple[List[RealCandidate], float]] = {}
_CACHE_TTL = 300  # 5 minutes


def get_cached_candidates(profile: DemoProfile, n: int = 200) -> List[RealCandidate]:
    """Get candidates for a profile, caching to avoid repeated 5K iterations."""
    key = f"{profile.profile_id}:{n}"
    if key in _candidate_cache:
        cached, ts = _candidate_cache[key]
        if time.time() - ts < _CACHE_TTL:
            return cached
    result = get_candidates_for_profile(profile, ALL_CANDIDATES, n)
    _candidate_cache[key] = (result, time.time())
    return result


def _safe_callback(n_outputs: int):
    """Decorator that wraps Gradio callbacks with error handling.

    On exception, returns an HTML error message in the first output slot
    and empty strings for the rest, preventing raw tracebacks in the UI.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {fn.__name__}: {e}", exc_info=True)
                err_html = (
                    f'<div style="padding:16px;background:#fce4ec;border:1px solid #ef9a9a;'
                    f'border-radius:8px;margin:8px 0;">'
                    f'<b style="color:#c62828;">Error in {fn.__name__}:</b><br>'
                    f'<pre style="color:#333;font-size:12px;white-space:pre-wrap;">'
                    f'{_html.escape(str(e))}</pre></div>'
                )
                return tuple([err_html] + [""] * (n_outputs - 1))
        return wrapper
    return decorator


def _check_data_loaded() -> None:
    """Raise if product data was not loaded at startup."""
    if not ALL_CANDIDATES:
        raise RuntimeError(
            "No product data available. Check that SUPABASE_URL and "
            "SUPABASE_SERVICE_KEY are set and Supabase is reachable."
        )


# =============================================================================
# TAB 1: PREMADE PROFILES
# =============================================================================

@_safe_callback(3)
def tab1_select_profile(profile_name: str):
    if not profile_name or profile_name not in PREMADE_PROFILES:
        return "Select a profile", "", ""
    _check_data_loaded()
    p = PREMADE_PROFILES[profile_name]
    profile_html = render_profile_card(p)
    feed = run_feed_pipeline(p, session_scores=None, n=20)
    feed_html = render_feed(feed, "Initial Feed (Onboarding Only)")
    return profile_html, feed_html, ""


@_safe_callback(4)
def tab1_apply_scenario(profile_name: str, scenario_name: str):
    if not profile_name or profile_name not in PREMADE_PROFILES:
        return "Select a profile first", "", "", ""
    if not scenario_name or scenario_name not in ACTION_SCENARIOS:
        return "Select a scenario", "", "", ""
    _check_data_loaded()

    p = PREMADE_PROFILES[profile_name]
    candidates = get_cached_candidates(p, 200)

    before_feed = run_feed_pipeline(p, session_scores=None, candidates=candidates, n=20)

    session = session_engine.initialize_from_onboarding(preferred_brands=p.preferred_brands)
    session = apply_actions(session, ACTION_SCENARIOS[scenario_name])

    # Refresh candidates with session-intent items after actions
    after_candidates = inject_session_intent_candidates(
        candidates, ALL_CANDIDATES, session, max_inject=50,
    )
    after_feed = run_feed_pipeline(p, session_scores=session, candidates=after_candidates, n=20)

    return (render_feed(before_feed, "Before Actions"),
            render_feed(after_feed, "After Actions"),
            render_diff(before_feed, after_feed),
            render_session_state(session))


# =============================================================================
# TAB 2: INTERACTIVE PLAYGROUND
# =============================================================================

@_safe_callback(4)
def tab2_init(profile_name: str):
    if not profile_name or profile_name not in PREMADE_PROFILES:
        return "Select a profile", "", "", "No session"
    _check_data_loaded()

    p = PREMADE_PROFILES[profile_name]
    candidates = get_cached_candidates(p, 200)
    session = session_engine.initialize_from_onboarding(preferred_brands=p.preferred_brands)

    _tab2_state[profile_name] = {
        "profile": p, "candidates": list(candidates), "session": session,
        "action_log": [], "seen_ids": set(),
    }

    feed = run_feed_pipeline(p, session_scores=session, candidates=candidates, n=20)
    return (render_profile_card(p), render_feed(feed, "Current Feed"),
            render_session_state(session),
            "Feed items: " + ", ".join(item["brand"] + " " + str(item["article_type"]) for item in feed[:10]))


@_safe_callback(3)
def tab2_do_action(profile_name: str, action_type: str, brand: str, item_type: str, attrs_json: str):
    if not profile_name or profile_name not in _tab2_state:
        return "Initialize a profile first", "", ""

    st = _tab2_state[profile_name]
    try:
        attributes = json.loads(attrs_json) if attrs_json and attrs_json.strip() else {}
    except json.JSONDecodeError:
        attributes = {}

    pid = f"action_{brand}_{item_type}".lower().replace(" ", "_")
    session_engine.process_action(
        st["session"], action=action_type, product_id=pid,
        brand=brand, item_type=item_type, attributes=attributes,
    )
    st["action_log"].append(f"{action_type}: {brand} {item_type}")

    # Refresh candidates with session-intent items
    st["candidates"] = inject_session_intent_candidates(
        get_candidates_for_profile(st["profile"], ALL_CANDIDATES, 200),
        ALL_CANDIDATES,
        st["session"],
        max_inject=50,
    )

    feed = run_feed_pipeline(st["profile"], session_scores=st["session"],
                              candidates=st["candidates"], n=20,
                              seen_ids=st["seen_ids"])
    # Track displayed items so they don't reappear on next action
    for item in feed:
        pid = item.get("product_id") or item.get("item_id", "")
        if pid:
            st["seen_ids"].add(pid)

    log_html = "<br>".join(f"{i+1}. {a}" for i, a in enumerate(st["action_log"]))
    seen_count = len(st["seen_ids"])
    return (render_feed(feed, f"Feed (after {len(st['action_log'])} actions, {seen_count} seen)"),
            render_session_state(st["session"]),
            f"<b>Action Log:</b><br>{log_html}")


def _search_local_pool(query: str, n: int = 20) -> List[dict]:
    """Simple text-match search against the local product pool.

    In production this would be an Algolia call. Here we simulate by
    matching query words against product name, brand, article_type,
    and style_tags — then return the top-n matches as dicts.
    """
    if not query or not query.strip():
        return []

    q_words = query.lower().split()
    scored: List[Tuple[float, dict]] = []

    for flat in ALL_FLATS:
        score = 0.0
        haystack = " ".join(
            str(flat.get(k) or "") for k in
            ("name", "brand", "article_type", "broad_category")
        ).lower()
        # Also include list fields
        for tag in (flat.get("style_tags") or []):
            haystack += " " + str(tag).lower()
        for occ in (flat.get("occasions") or []):
            haystack += " " + str(occ).lower()

        for w in q_words:
            if w in haystack:
                score += 1.0
        if score > 0:
            scored.append((score, flat))

    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored[:n]]


@_safe_callback(3)
def tab2_do_search(profile_name: str, query: str, filter_colors: list, filter_styles: list):
    if not profile_name or profile_name not in _tab2_state:
        return "Initialize a profile first", "", ""

    st = _tab2_state[profile_name]

    # Search the local product pool (simulates Algolia call)
    search_results = _search_local_pool(query, n=20)

    # Extract structured signals from what the search returned
    signals = extract_search_signals(search_results, top_n=10)

    # Merge in explicit UI filter selections
    if filter_colors:
        signals.setdefault("colors", []).extend(filter_colors)
    if filter_styles:
        signals.setdefault("styles", []).extend(filter_styles)

    session_engine.process_search_signal(st["session"], query=query, filters=signals)

    hit_summary = ""
    if search_results:
        brands_found = set(r.get("brand", "") for r in search_results[:5])
        types_found = set(r.get("article_type", "") for r in search_results[:5])
        hit_summary = f" | top hits: {', '.join(b for b in brands_found if b)} ({', '.join(t for t in types_found if t)})"

    st["action_log"].append(f"search: '{query}' -> {len(search_results)} results, signals={signals}{hit_summary}")

    # Refresh candidates with session-intent items
    st["candidates"] = inject_session_intent_candidates(
        get_candidates_for_profile(st["profile"], ALL_CANDIDATES, 200),
        ALL_CANDIDATES,
        st["session"],
        max_inject=50,
    )

    feed = run_feed_pipeline(st["profile"], session_scores=st["session"],
                              candidates=st["candidates"], n=20,
                              seen_ids=st["seen_ids"])
    # Track displayed items so they don't reappear on next search
    for item in feed:
        pid = item.get("product_id") or item.get("item_id", "")
        if pid:
            st["seen_ids"].add(pid)

    log_html = "<br>".join(f"{i+1}. {a}" for i, a in enumerate(st["action_log"]))
    seen_count = len(st["seen_ids"])
    return (render_feed(feed, f"Feed (after {len(st['action_log'])} actions, {seen_count} seen)"),
            render_session_state(st["session"]),
            f"<b>Action Log:</b><br>{log_html}")


def tab2_reset_seen(profile_name: str):
    """Reset seen_ids so the feed can show previously-displayed items again."""
    if profile_name in _tab2_state:
        count = len(_tab2_state[profile_name]["seen_ids"])
        _tab2_state[profile_name]["seen_ids"] = set()
        return f"Reset {count} seen items. Next feed will include all candidates."
    return "No active profile"


# =============================================================================
# TAB 3: MANUAL PROFILE BUILDER
# =============================================================================

def _age_to_group(age: int) -> str:
    """Derive age group from numeric age."""
    if age < 25:
        return "18-24"
    elif age < 35:
        return "25-34"
    elif age < 45:
        return "35-44"
    elif age < 65:
        return "45-64"
    else:
        return "65+"


@_safe_callback(3)
def tab3_build_and_run(age, style, city, country,
                       temp, weather, season,
                       brands, categories):
    _check_data_loaded()
    age = int(age)
    age_group = _age_to_group(age)
    brand_list = brands or []
    min_price, max_price = derive_price_range(brand_list)
    p = DemoProfile(
        profile_id="custom_profile",
        name=f"Custom {style.title()} {age_group}",
        age=age, age_group=age_group, style_persona=style,
        city=city, country=country, temperature_c=temp,
        weather_condition=weather, season=season,
        is_hot=temp > 28, is_cold=temp < 5, is_rainy=weather == "rain",
        preferred_brands=brand_list,
        preferred_categories=categories or [],
        preferred_styles=[style] if style else [],
        preferred_colors=[],
        preferred_patterns=[],
        min_price=min_price,
        max_price=max_price,
    )
    session = session_engine.initialize_from_onboarding(preferred_brands=p.preferred_brands)

    # Run categorized feed (20 per category from a larger 500-candidate pool)
    categorized = run_categorized_feed(p, session_scores=session, per_category=20)

    _tab3_state["custom"] = {
        "profile": p, "session": session,
        "candidates": get_candidates_for_profile(p, ALL_CANDIDATES, 500),
        "action_log": [],
    }

    return render_profile_card(p), render_categorized_feed(categorized, "Generated Feed"), render_session_state(session)


@_safe_callback(3)
def tab3_do_action(action_type, brand, item_type):
    if "custom" not in _tab3_state:
        return "Build a profile first", "", ""

    st = _tab3_state["custom"]
    pid = f"action_{brand}_{item_type}".lower().replace(" ", "_")
    session_engine.process_action(st["session"], action=action_type, product_id=pid,
                                  brand=brand, item_type=item_type, attributes={})
    st["action_log"].append(f"{action_type}: {brand} {item_type}")

    categorized = run_categorized_feed(st["profile"], session_scores=st["session"], per_category=20)
    return (render_categorized_feed(categorized, f"Feed (after {len(st['action_log'])} actions)"),
            render_session_state(st["session"]),
            "<br>".join(f"{i+1}. {a}" for i, a in enumerate(st["action_log"])))


# =============================================================================
# TAB 4: ACTION DEEP DIVE
# =============================================================================

@_safe_callback(3)
def tab4_run_sequence(profile_name: str, actions_text: str):
    if not profile_name or profile_name not in PREMADE_PROFILES:
        return "Select a profile", "", ""
    _check_data_loaded()

    p = PREMADE_PROFILES[profile_name]
    candidates = get_candidates_for_profile(p, ALL_CANDIDATES, 200)

    actions = []
    for line in actions_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3:
            atype = parts[0]
            b = parts[1]
            it = parts[2]
            attrs = {}
            if len(parts) >= 4:
                try:
                    attrs = json.loads(parts[3])
                except json.JSONDecodeError:
                    pass
            if atype == "search":
                attrs = {"query": b, "filters": {}}
            actions.append((atype, b, it, attrs))

    if not actions:
        return "No valid actions found", "", ""

    session = session_engine.initialize_from_onboarding(preferred_brands=p.preferred_brands)

    evolution_parts = []
    checkpoints = [0] + list(range(5, len(actions) + 1, 5))
    if len(actions) not in checkpoints:
        checkpoints.append(len(actions))

    feed_0 = run_feed_pipeline(p, session_scores=session, candidates=candidates, n=10)
    evolution_parts.append(f'<div class="evolution-step"><h4>Step 0: Initial (onboarding only)</h4>')
    evolution_parts.append(render_session_state(session))
    evolution_parts.append("<p><b>Top 5:</b> " + " | ".join(
        f"#{i+1} {item['brand']} {item['article_type']}" for i, item in enumerate(feed_0[:5])
    ) + "</p></div>")

    for i, action_tuple in enumerate(actions):
        atype = action_tuple[0]
        if atype == "search":
            info = action_tuple[3] if len(action_tuple) > 3 else {}
            session_engine.process_search_signal(session, query=info.get("query", ""), filters=info.get("filters", {}))
        else:
            b = action_tuple[1] if len(action_tuple) > 1 else ""
            it = action_tuple[2] if len(action_tuple) > 2 else ""
            attrs = action_tuple[3] if len(action_tuple) > 3 else {}
            pid = f"action_{i}_{b}_{it}".lower().replace(" ", "_")
            session_engine.process_action(session, action=atype, product_id=pid,
                                          brand=b, item_type=it, attributes=attrs)

        step = i + 1
        if step in checkpoints:
            # Refresh candidates with session-intent items at each checkpoint
            step_candidates = inject_session_intent_candidates(
                get_candidates_for_profile(p, ALL_CANDIDATES, 200),
                ALL_CANDIDATES, session, max_inject=50,
            )
            feed = run_feed_pipeline(p, session_scores=session, candidates=step_candidates, n=10)
            desc = f"{atype}: {action_tuple[1]} {action_tuple[2]}" if len(action_tuple) >= 3 else str(action_tuple)
            evolution_parts.append(
                f'<div class="evolution-step"><h4>Step {step}: After "{_html.escape(desc)}"</h4>')
            evolution_parts.append(render_session_state(session))
            evolution_parts.append("<p><b>Top 5:</b> " + " | ".join(
                f"#{j+1} {item['brand']} {item['article_type']}" for j, item in enumerate(feed[:5])
            ) + "</p></div>")

    # Final feed with session-intent candidates
    final_candidates = inject_session_intent_candidates(
        get_candidates_for_profile(p, ALL_CANDIDATES, 200),
        ALL_CANDIDATES, session, max_inject=50,
    )
    final_feed = run_feed_pipeline(p, session_scores=session, candidates=final_candidates, n=20)
    return ("\n".join(evolution_parts),
            render_feed(final_feed, "Final Feed (after all actions)"),
            render_session_state(session))


# =============================================================================
# TAB 5: SEARCH RESULTS
# =============================================================================

@_safe_callback(4)
def tab5_compare(profile_name: str, scenario_name: str):
    if not profile_name or profile_name not in PREMADE_PROFILES:
        return "Select a profile", "", "", ""
    if not scenario_name or scenario_name not in ACTION_SCENARIOS:
        return "Select a scenario", "", "", ""
    _check_data_loaded()

    p = PREMADE_PROFILES[profile_name]
    results = get_search_results_for_profile(p, ALL_SEARCH_RESULTS, 60)

    before = run_search_pipeline(p, session_scores=None, results=results, n=20)

    session = session_engine.initialize_from_onboarding(preferred_brands=p.preferred_brands)
    session = apply_actions(session, ACTION_SCENARIOS[scenario_name])

    after = run_search_pipeline(p, session_scores=session, results=results, n=20)

    return (render_feed(before, "Search Results: Before Actions"),
            render_feed(after, "Search Results: After Actions"),
            render_diff(before, after),
            render_session_state(session))


# =============================================================================
# TAB 6 – FILTER EXPLORER  (uses full product pool from Supabase, loaded at startup)
# =============================================================================

# -- Full product pool for filter explorer (loaded at startup in background) ---
_full_filter_pool: Optional[List[dict]] = None
_full_filter_pool_ready = threading.Event()
_full_filter_pool_error: Optional[str] = None


def _load_full_filter_pool() -> None:
    """Fetch ALL in-stock products from Supabase into _full_filter_pool.

    Called once in a background thread at startup.  Sets
    _full_filter_pool_ready when done (or on error).
    """
    global _full_filter_pool, _full_filter_pool_error

    try:
        from supabase import create_client

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY env vars required")

        sb = create_client(url, key)

        SELECT_COLS = (
            "id, name, brand, category, broad_category, article_type, "
            "price, original_price, in_stock, fit, length, sleeve, "
            "neckline, rise, base_color, colors, materials, style_tags, "
            "primary_image_url, gallery_images, trending_score, "
            "gender, "
            "product_attributes!left("
            "  category_l1, category_l2, category_l3, "
            "  construction, primary_color, color_family, secondary_colors, "
            "  pattern, pattern_scale, apparent_fabric, texture, sheen, "
            "  style_tags, occasions, seasons, formality, trend_tags, "
            "  fit_type, stretch, rise, leg_shape, silhouette, "
            "  coverage_level, skin_exposure, coverage_details, "
            "  model_body_type, model_size_estimate"
            ")"
        )

        batch_size = 1000
        all_rows: list = []
        offset = 0

        # Get total count
        count_result = sb.table("products").select("id", count="exact").eq(
            "in_stock", True
        ).not_.is_("primary_image_url", "null").execute()
        total = count_result.count or 0
        # loading full pool

        while True:
            t0 = time.time()
            result = sb.table("products").select(SELECT_COLS).eq(
                "in_stock", True
            ).not_.is_("primary_image_url", "null").range(
                offset, offset + batch_size - 1
            ).execute()
            if not result.data:
                break
            all_rows.extend(result.data)
            elapsed = time.time() - t0
            if len(all_rows) % 10000 < batch_size:
                pass  # silent progress
            offset += batch_size
            if len(result.data) < batch_size:
                break

        # fetched, flattening

        flats = [_flatten_product(row) for row in all_rows]
        flats = [f for f in flats if f.get("image_url")]

        # Pre-compute general_styles and material_group for every product
        for f in flats:
            f["general_styles"] = _map_to_general_styles(f)
            f["material_group"] = _map_to_material_group(f)

        print("[FilterPool] Ready")
        _full_filter_pool = flats

    except Exception as exc:
        _full_filter_pool_error = str(exc)
        print(f"[FilterPool] ERROR loading full pool: {exc}")

    finally:
        _full_filter_pool_ready.set()


def _get_full_filter_pool() -> List[dict]:
    """Return the full product pool, waiting for the background load if needed."""
    global _full_filter_pool

    if _full_filter_pool is not None:
        return _full_filter_pool

    # Wait for background thread (with timeout so we don't hang forever)
    if not _full_filter_pool_ready.wait(timeout=120):
        raise RuntimeError("Full product pool load timed out after 120s.")

    if _full_filter_pool_error:
        raise RuntimeError(f"Full product pool failed to load: {_full_filter_pool_error}")

    if _full_filter_pool is None:
        raise RuntimeError("Full product pool is empty after loading.")

    return _full_filter_pool


# Load the full filter pool at startup (blocking) so Tab 6 is ready immediately
print("[FilterPool] Starting full pool load (this runs before the UI opens)...")
_load_full_filter_pool()

# Re-derive all dropdown values from the full pool so every brand/value is included
if _full_filter_pool:
    _all = _full_filter_pool
    from collections import Counter as _Counter
    _brand_counts = _Counter(f["brand"] for f in _all if f.get("brand"))
    DB_BRANDS = [f"{b} ({_brand_counts[b]:,})" for b in sorted(_brand_counts.keys())]
    _BRAND_DISPLAY_TO_NAME = {f"{b} ({_brand_counts[b]:,})": b for b in _brand_counts}
    DB_ARTICLE_TYPES = sorted(set(f["article_type"] for f in _all if f.get("article_type")))
    DB_STYLES_RAW = sorted(set(s for f in _all for s in (f.get("style_tags") or [])))
    DB_COLORS = sorted(set(c for f in _all for c in (f.get("colors") or [])))
    DB_PATTERNS = sorted(set(f["pattern"] for f in _all if f.get("pattern")))
    DB_OCCASIONS_RAW = sorted(set(o for f in _all for o in (f.get("occasions") or [])))
    DB_FORMALITY = sorted(set(f.get("formality") for f in _all if f.get("formality")))
    DB_FIT_TYPES = sorted(set(f.get("fit") or f.get("fit_type") for f in _all if f.get("fit") or f.get("fit_type")))
    DB_NECKLINES = sorted(set(f.get("neckline") for f in _all if f.get("neckline")))
    DB_SLEEVE_TYPES = sorted(set(f.get("sleeve") or f.get("sleeve_type") for f in _all if f.get("sleeve") or f.get("sleeve_type")))
    DB_LENGTHS = sorted(set(f.get("length") for f in _all if f.get("length")))
    DB_SILHOUETTES = sorted(set(f.get("silhouette") for f in _all if f.get("silhouette")))
    DB_RISES = sorted(set(f.get("rise") for f in _all if f.get("rise") and f.get("rise") not in ("", "N/A")))
    DB_COLOR_FAMILIES = sorted(set(f.get("color_family") for f in _all if f.get("color_family")))
    DB_MATERIALS = sorted(set(m for f in _all for m in (f.get("materials") or []) if m))
    DB_SEASONS = sorted(set(s for f in _all for s in (f.get("seasons") or []) if s))
    DB_CATEGORIES = sorted(set(f.get("broad_category") for f in _all if f.get("broad_category") and f.get("broad_category").lower() != "accessories"))
    DB_COVERAGE_LEVELS = sorted(set(f.get("coverage_level") for f in _all if f.get("coverage_level")))
    DB_SKIN_EXPOSURE = sorted(set(f.get("skin_exposure") for f in _all if f.get("skin_exposure")))
    DB_MODEL_BODY_TYPES = sorted(set(f.get("model_body_type") for f in _all if f.get("model_body_type")))
    DB_MODEL_SIZE_ESTIMATES = sorted(set(f.get("model_size_estimate") for f in _all if f.get("model_size_estimate")))
    print(f"[FilterPool] Dropdown values derived from full pool ({len(DB_BRANDS)} brands)")


def _flat_matches_filters(f: dict, filters: dict) -> bool:
    """Check if a flat product dict passes ALL filter groups.

    Works directly on flat dicts (no RealCandidate needed).
    AND across filter groups, OR within a group.
    """

    # -- Category & Type --
    if filters.get("categories"):
        val = (f.get("broad_category") or "").lower()
        if not any(v.lower() == val for v in filters["categories"]):
            return False

    if filters.get("article_types"):
        val = (f.get("article_type") or "").lower()
        if not any(v.lower() == val for v in filters["article_types"]):
            return False

    # -- Style (general style labels, pre-computed) --
    if filters.get("styles"):
        item_gen = set(f.get("general_styles") or [])
        if not item_gen.intersection(filters["styles"]):
            return False

    if filters.get("seasons"):
        item_seasons = set(s.lower() for s in (f.get("seasons") or []))
        if not item_seasons.intersection(s.lower() for s in filters["seasons"]):
            return False

    # -- Color & Pattern --
    if filters.get("color_families"):
        val = (f.get("color_family") or "").lower()
        if not any(v.lower() == val for v in filters["color_families"]):
            return False


    if filters.get("patterns"):
        val = (f.get("pattern") or "").lower()
        if not any(v.lower() == val for v in filters["patterns"]):
            return False

    # -- Construction --
    if filters.get("fit_types"):
        val = (f.get("fit") or f.get("fit_type") or "").lower()
        if not any(v.lower() == val for v in filters["fit_types"]):
            return False

    if filters.get("necklines"):
        val = (f.get("neckline") or "").lower()
        if not any(v.lower() == val for v in filters["necklines"]):
            return False

    if filters.get("sleeve_types"):
        val = (f.get("sleeve") or f.get("sleeve_type") or "").lower()
        if not any(v.lower() == val for v in filters["sleeve_types"]):
            return False

    if filters.get("lengths"):
        val = (f.get("length") or "").lower()
        if not any(v.lower() == val for v in filters["lengths"]):
            return False

    if filters.get("silhouettes"):
        val = (f.get("silhouette") or "").lower()
        if not any(v.lower() == val for v in filters["silhouettes"]):
            return False

    # -- Brand & Price --
    if filters.get("brands"):
        val = (f.get("brand") or "").lower()
        if not any(v.lower() == val for v in filters["brands"]):
            return False

    if filters.get("exclude_brands"):
        val = (f.get("brand") or "").lower()
        if any(v.lower() == val for v in filters["exclude_brands"]):
            return False

    price = float(f.get("price") or 0)
    if filters.get("min_price") is not None and filters["min_price"] > 0:
        if price < filters["min_price"]:
            return False
    if filters.get("max_price") is not None and filters["max_price"] > 0:
        if price > filters["max_price"]:
            return False

    if filters.get("on_sale") and not f.get("is_on_sale"):
        return False

    # -- Material group --
    if filters.get("materials"):
        item_group = f.get("material_group") or ""
        if item_group not in filters["materials"]:
            return False

    # -- Gemini coverage & body type --
    if filters.get("coverage_levels"):
        val = (f.get("coverage_level") or "").strip()
        if not val or not any(v == val for v in filters["coverage_levels"]):
            return False

    if filters.get("skin_exposure"):
        val = (f.get("skin_exposure") or "").strip()
        if not val or not any(v == val for v in filters["skin_exposure"]):
            return False

    if filters.get("model_body_types"):
        val = (f.get("model_body_type") or "").strip()
        if not val or not any(v == val for v in filters["model_body_types"]):
            return False

    if filters.get("model_size_estimates"):
        val = (f.get("model_size_estimate") or "").strip()
        if not val or not any(v == val for v in filters["model_size_estimates"]):
            return False

    return True


def _flat_to_card_dict(f: dict, rank: int = 0) -> dict:
    """Convert a flat product dict to a display dict for render_product_card."""
    return {
        "product_id": f.get("product_id", ""),
        "name": f.get("name", "Unknown"),
        "brand": f.get("brand", "Unknown"),
        "article_type": f.get("article_type", ""),
        "broad_category": f.get("broad_category", ""),
        "price": f.get("price", 0),
        "original_price": f.get("original_price", 0),
        "image_url": f.get("image_url", ""),
        "gallery_images": f.get("gallery_images") or [],
        "is_on_sale": f.get("is_on_sale", False),
        "is_new": False,
        "general_styles": f.get("general_styles") or [],
        "style_tags": f.get("style_tags") or [],
        "pattern": f.get("pattern", ""),
        "fit_type": f.get("fit") or f.get("fit_type", ""),
        "neckline": f.get("neckline", ""),
        "sleeve_type": f.get("sleeve") or f.get("sleeve_type", ""),
        "length": f.get("length", ""),
        "color_family": f.get("color_family", ""),
        "coverage_level": f.get("coverage_level", ""),
        "skin_exposure": f.get("skin_exposure", ""),
        "model_body_type": f.get("model_body_type", ""),
        "model_size_estimate": f.get("model_size_estimate", ""),
        "rank": rank,
        "source": "",
    }


def _build_facet_summary(matched_flats: List[dict], total: int) -> str:
    """Build HTML summary with facet distributions."""
    n = len(matched_flats)
    pct = (n / total * 100) if total else 0

    html = f'<div style="padding:12px;background:#f0f4ff;border-radius:10px;margin-bottom:12px;">'
    html += f'<h3 style="margin:0 0 8px 0;color:#1a237e;">Filter Results: {n:,} / {total:,} products ({pct:.1f}%)</h3>'

    if not matched_flats:
        html += '<p style="color:#c62828;"><b>No products match these filters.</b> Try broadening your criteria.</p>'
        html += '</div>'
        return html

    # Build facet counters
    facets = {
        "Brand": Counter(),
        "Category": Counter(),
        "Article Type": Counter(),
        "General Style": Counter(),
        "Pattern": Counter(),
        "Color Family": Counter(),
        "Fit Type": Counter(),
        "Neckline": Counter(),
        "Sleeve Type": Counter(),
        "Length": Counter(),
        "Seasons": Counter(),
        "Materials": Counter(),
        "Silhouette": Counter(),
        "On Sale": Counter(),
    }

    for f in matched_flats:
        if f.get("brand"): facets["Brand"][f["brand"]] += 1
        if f.get("broad_category"): facets["Category"][f["broad_category"]] += 1
        if f.get("article_type"): facets["Article Type"][f["article_type"]] += 1
        if f.get("pattern"): facets["Pattern"][f["pattern"]] += 1
        if f.get("color_family"): facets["Color Family"][f["color_family"]] += 1
        if f.get("fit") or f.get("fit_type"): facets["Fit Type"][f.get("fit") or f.get("fit_type")] += 1
        if f.get("neckline"): facets["Neckline"][f["neckline"]] += 1
        if f.get("sleeve") or f.get("sleeve_type"): facets["Sleeve Type"][f.get("sleeve") or f.get("sleeve_type")] += 1
        if f.get("length"): facets["Length"][f["length"]] += 1
        if f.get("silhouette"): facets["Silhouette"][f["silhouette"]] += 1
        facets["On Sale"]["Yes" if f.get("is_on_sale") else "No"] += 1
        for gs in (f.get("general_styles") or []):
            facets["General Style"][gs] += 1
        for s in (f.get("seasons") or []):
            facets["Seasons"][s] += 1
        for m in (f.get("materials") or []):
            facets["Materials"][m] += 1

    # Price stats
    prices = [float(f.get("price") or 0) for f in matched_flats if f.get("price")]
    if prices:
        html += f'<div style="margin:4px 0;font-size:12px;color:#555;">'
        html += f'Price range: <b>${min(prices):.2f}</b> — <b>${max(prices):.2f}</b> '
        html += f'(avg ${sum(prices)/len(prices):.2f}, median ${sorted(prices)[len(prices)//2]:.2f})'
        html += '</div>'

    # Render facets as collapsible sections
    html += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:8px;margin-top:8px;">'

    for facet_name, counter in facets.items():
        if not counter or len(counter) < 1:
            continue
        top_items = counter.most_common(10)
        total_in_facet = sum(counter.values())

        html += f'<div style="background:#fff;padding:8px;border-radius:8px;border:1px solid #e0e0e0;">'
        html += f'<div style="font-weight:600;font-size:11px;color:#1a237e;margin-bottom:4px;">{facet_name} ({len(counter)} values)</div>'
        for val, cnt in top_items:
            bar_pct = (cnt / n * 100) if n else 0
            html += f'<div style="display:flex;align-items:center;gap:4px;margin:1px 0;font-size:10px;">'
            html += f'<div style="min-width:80px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{_html.escape(str(val))}">{_html.escape(str(val))}</div>'
            html += f'<div style="flex:1;background:#e8eaf6;border-radius:3px;height:10px;overflow:hidden;">'
            html += f'<div style="width:{bar_pct:.1f}%;background:#5c6bc0;height:100%;border-radius:3px;"></div>'
            html += f'</div>'
            html += f'<div style="min-width:40px;text-align:right;color:#666;">{cnt}</div>'
            html += '</div>'
        if len(counter) > 10:
            html += f'<div style="font-size:9px;color:#999;margin-top:2px;">... +{len(counter)-10} more</div>'
        html += '</div>'

    html += '</div></div>'
    return html


def _build_filter_data_table(matched_flats: List[dict], max_rows: int = 50) -> List[List]:
    """Build a data table (list of lists) from matched flat dicts."""
    cols = ["name", "brand", "category", "article_type", "general_styles",
            "pattern", "color_family", "fit", "neckline", "sleeve",
            "length", "silhouette", "price", "materials", "seasons",
            "coverage_level", "skin_exposure", "model_body_type", "model_size_estimate"]
    rows = []
    for f in matched_flats[:max_rows]:
        row = []
        for col in cols:
            val = f.get(col, "")
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val[:5])
            elif val is None:
                val = ""
            elif col == "price":
                val = f"${float(val or 0):.2f}"
            else:
                val = str(val)
            row.append(val)
        rows.append(row)
    return rows


FILTER_TABLE_HEADERS = ["Name", "Brand", "Category", "Article Type", "General Style",
                        "Pattern", "Color Family", "Fit", "Neckline", "Sleeve",
                        "Length", "Silhouette", "Price", "Materials", "Seasons",
                        "Coverage", "Skin Exp.", "Model Body Type", "Model Size"]


@_safe_callback(3)
def tab6_apply_filters(
    categories, article_types, styles,
    seasons,
    color_families, patterns,
    fit_types, necklines, sleeve_types, lengths, silhouettes,
    brands, exclude_brands, min_price, max_price, on_sale,
    materials,
    coverage_levels, skin_exposure, model_body_types, model_size_estimates,
):
    """Apply filters to the FULL product pool (~96K items) and return results.

    The full pool is lazy-loaded from Supabase on first use.
    """
    t0 = time.time()

    filters = {
        "categories": categories or [],
        "article_types": article_types or [],
        "styles": styles or [],
        "seasons": seasons or [],
        "color_families": color_families or [],
        "patterns": patterns or [],
        "fit_types": fit_types or [],
        "necklines": necklines or [],
        "sleeve_types": sleeve_types or [],
        "lengths": lengths or [],
        "silhouettes": silhouettes or [],
        "brands": [_BRAND_DISPLAY_TO_NAME.get(b, b) for b in (brands or [])],
        "exclude_brands": [_BRAND_DISPLAY_TO_NAME.get(b, b) for b in (exclude_brands or [])],
        "min_price": float(min_price) if min_price else 0,
        "max_price": float(max_price) if max_price else 0,
        "on_sale": bool(on_sale),
        "materials": materials or [],
        "coverage_levels": coverage_levels or [],
        "skin_exposure": skin_exposure or [],
        "model_body_types": model_body_types or [],
        "model_size_estimates": model_size_estimates or [],
    }

    # Check if ANY filter is active
    has_any_filter = any([
        filters["categories"], filters["article_types"], filters["styles"],
        filters["seasons"],
        filters["color_families"], filters["patterns"],
        filters["fit_types"], filters["necklines"], filters["sleeve_types"],
        filters["lengths"], filters["silhouettes"],
        filters["brands"], filters["exclude_brands"],
        filters["min_price"] > 0, filters["max_price"] > 0, filters["on_sale"],
        filters["materials"],
        filters["coverage_levels"], filters["skin_exposure"],
        filters["model_body_types"], filters["model_size_estimates"],
    ])

    if not has_any_filter:
        return (
            "",
            '<div class="filter-hint">'
            '<h3>Select at least one filter to explore the product pool.</h3></div>',
            [[""] * len(FILTER_TABLE_HEADERS)],
        )

    # Load the full product pool (lazy -- first call fetches from Supabase)
    pool = _get_full_filter_pool()
    total = len(pool)

    # Filter directly on flat dicts (no RealCandidate needed)
    matched_flats = [f for f in pool if _flat_matches_filters(f, filters)]

    # Deduplicate: remove items with same name or same image URL
    _seen_names: set[str] = set()
    _seen_images: set[str] = set()
    deduped: list[dict] = []
    for f in matched_flats:
        name_key = (f.get("name") or "").strip().lower()
        img_key = (f.get("image_url") or "").strip()
        if name_key and name_key in _seen_names:
            continue
        if img_key and img_key in _seen_images:
            continue
        if name_key:
            _seen_names.add(name_key)
        if img_key:
            _seen_images.add(img_key)
        deduped.append(f)
    matched_flats = deduped

    n = len(matched_flats)
    elapsed = time.time() - t0

    summary_html = ""

    # Product card grid (up to 80 products)
    GRID_SIZE = 80
    sample = matched_flats[:GRID_SIZE] if n <= GRID_SIZE else random.sample(matched_flats, GRID_SIZE)

    if matched_flats:
        grid_html = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:4px;">'
        for i, f in enumerate(sample):
            d = _flat_to_card_dict(f, rank=i + 1)
            grid_html += render_product_card(d, show_scores=False)
        grid_html += '</div>'
    else:
        grid_html = '<p style="color:#c62828;"><i>No matching products.</i></p>'

    # Data table
    table_data = _build_filter_data_table(matched_flats)
    if not table_data:
        table_data = [[""] * len(FILTER_TABLE_HEADERS)]

    return summary_html, grid_html, table_data


# =============================================================================
# GRADIO APP
# =============================================================================

def build_app():
    with gr.Blocks(title="Recommendation System Demo") as app:

        gr.Markdown("# Recommendation System Demo\nReal product data from Supabase. Scoring: Context (age+weather) + Session (actions) + Reranking (diversity)")

        with gr.Tabs():

            # TAB 1
            with gr.TabItem("1. Premade Profiles"):
                gr.Markdown("Pick a curated profile, see the initial feed, then apply an action scenario.")
                with gr.Row():
                    with gr.Column(scale=1):
                        t1_profile = gr.Dropdown(choices=PROFILE_NAMES, label="Select Profile")
                        t1_btn_select = gr.Button("Load Profile", variant="primary")
                        t1_scenario = gr.Dropdown(choices=list(ACTION_SCENARIOS.keys()), label="Action Scenario")
                        t1_btn_scenario = gr.Button("Apply Scenario", variant="secondary")
                    with gr.Column(scale=1):
                        t1_profile_html = gr.HTML(label="Profile")

                with gr.Row():
                    with gr.Column(scale=1):
                        t1_feed_html = gr.HTML(label="Initial Feed")
                    with gr.Column(scale=1):
                        t1_after_html = gr.HTML(label="After Actions")

                with gr.Row():
                    with gr.Column(scale=1):
                        t1_diff_html = gr.HTML(label="Diff")
                    with gr.Column(scale=1):
                        t1_state_html = gr.HTML(label="Session State")

                t1_btn_select.click(tab1_select_profile, [t1_profile],
                                    [t1_profile_html, t1_feed_html, t1_diff_html])
                t1_btn_scenario.click(tab1_apply_scenario, [t1_profile, t1_scenario],
                                      [t1_feed_html, t1_after_html, t1_diff_html, t1_state_html])

            # TAB 2
            with gr.TabItem("2. Interactive Playground"):
                gr.Markdown("Pick a profile, take actions, watch the feed update in real time.")
                with gr.Row():
                    with gr.Column(scale=1):
                        t2_profile = gr.Dropdown(choices=PROFILE_NAMES, label="Select Profile")
                        t2_btn_init = gr.Button("Initialize", variant="primary")
                        gr.Markdown("### Actions")
                        t2_action = gr.Dropdown(choices=["click", "skip", "add_to_cart", "add_to_wishlist", "purchase"],
                                                label="Action Type", value="click")
                        t2_brand = gr.Dropdown(choices=DB_BRANDS, label="Brand", allow_custom_value=True)
                        t2_type = gr.Dropdown(choices=DB_ARTICLE_TYPES, label="Article Type")
                        t2_attrs = gr.Textbox(label='Attributes JSON (optional)', value='{}', lines=1)
                        t2_btn_action = gr.Button("Do Action", variant="secondary")
                        gr.Markdown("### Search Signal")
                        t2_query = gr.Textbox(label="Search Query", placeholder="e.g. floral summer dress")
                        t2_colors = gr.Dropdown(choices=DB_COLORS, label="Color Filter", multiselect=True)
                        t2_styles = gr.Dropdown(choices=GENERAL_STYLE_OPTIONS, label="Style Filter", multiselect=True)
                        t2_btn_search = gr.Button("Send Search Signal", variant="secondary")
                        gr.Markdown("---")
                        t2_btn_reset_seen = gr.Button("Reset Seen Items", variant="stop")
                        t2_reset_msg = gr.Textbox(label="Reset Status", interactive=False)
                    with gr.Column(scale=2):
                        t2_profile_html = gr.HTML(label="Profile")
                        t2_feed_html = gr.HTML(label="Current Feed")

                with gr.Row():
                    with gr.Column(scale=1):
                        t2_state_html = gr.HTML(label="Session State")
                    with gr.Column(scale=1):
                        t2_log_html = gr.HTML(label="Action Log")
                    with gr.Column(scale=1):
                        t2_info = gr.HTML(label="Info")

                t2_btn_init.click(tab2_init, [t2_profile], [t2_profile_html, t2_feed_html, t2_state_html, t2_info])
                t2_btn_action.click(tab2_do_action, [t2_profile, t2_action, t2_brand, t2_type, t2_attrs],
                                    [t2_feed_html, t2_state_html, t2_log_html])
                t2_btn_search.click(tab2_do_search, [t2_profile, t2_query, t2_colors, t2_styles],
                                    [t2_feed_html, t2_state_html, t2_log_html])
                t2_btn_reset_seen.click(tab2_reset_seen, [t2_profile], [t2_reset_msg])

            # TAB 3
            with gr.TabItem("3. Manual Profile Builder"):
                gr.Markdown("Build a custom profile from scratch.")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Demographics")
                        t3_age = gr.Slider(18, 80, value=28, step=1, label="Age")
                        t3_style = gr.Dropdown(choices=PERSONA_CHOICES,
                                               label="Style Persona", value="Casual")
                        gr.Markdown("### Location & Weather")
                        t3_city = gr.Textbox(label="City", value="New York")
                        t3_country = gr.Textbox(label="Country", value="US")
                        t3_temp = gr.Slider(-10, 45, value=20, step=1, label="Temperature (C)")
                        t3_weather = gr.Dropdown(choices=["clear", "clouds", "rain", "snow"],
                                                  label="Weather", value="clear")
                        t3_season = gr.Dropdown(choices=["spring", "summer", "fall", "winter"],
                                                 label="Season", value="spring")
                    with gr.Column(scale=1):
                        gr.Markdown("### Preferences")
                        t3_brands = gr.Dropdown(choices=DB_BRANDS, label="Preferred Brands", multiselect=True)
                        t3_cats = gr.Dropdown(choices=["tops", "bottoms", "dresses", "outerwear"],
                                              label="Categories", multiselect=True)
                        t3_btn_build = gr.Button("Build Profile & Generate Feed", variant="primary")

                with gr.Row():
                    t3_profile_html = gr.HTML(label="Profile")
                with gr.Row():
                    with gr.Column(scale=2):
                        t3_feed_html = gr.HTML(label="Feed")
                    with gr.Column(scale=1):
                        t3_state_html = gr.HTML(label="Session State")

                with gr.Accordion("Quick Actions (after building)", open=False):
                    with gr.Row():
                        t3_action = gr.Dropdown(choices=["click", "skip", "add_to_cart", "purchase"],
                                                label="Action", value="click")
                        t3_abrand = gr.Dropdown(choices=DB_BRANDS, label="Brand", allow_custom_value=True)
                        t3_atype = gr.Dropdown(choices=DB_ARTICLE_TYPES, label="Type")
                        t3_btn_act = gr.Button("Apply Action")
                    t3_log = gr.HTML(label="Action Log")

                t3_btn_build.click(tab3_build_and_run,
                    [t3_age, t3_style, t3_city, t3_country,
                     t3_temp, t3_weather, t3_season,
                     t3_brands, t3_cats],
                    [t3_profile_html, t3_feed_html, t3_state_html])
                t3_btn_act.click(tab3_do_action, [t3_action, t3_abrand, t3_atype],
                                 [t3_feed_html, t3_state_html, t3_log])

            # TAB 4
            with gr.TabItem("4. Action Deep Dive"):
                gr.Markdown("""Define a sequence of actions and see how the feed evolves step by step.

**Format:** One action per line: `action_type, brand, item_type`
- Actions: `click`, `skip`, `add_to_cart`, `add_to_wishlist`, `purchase`
- For search: `search, query_text, _`
- Lines starting with `#` are comments""")

                with gr.Row():
                    with gr.Column(scale=1):
                        t4_profile = gr.Dropdown(choices=PROFILE_NAMES, label="Select Profile")
                        t4_actions = gr.Textbox(label="Action Sequence", lines=15, value="""# Example: browsing dresses then going sporty
click, Reformation, midi_dress
click, Reformation, wrap_dress
click, Ba&sh, slip_dress
skip, Boohoo, bodycon_dress
skip, Shein, crop_top
add_to_cart, Reformation, midi_dress
click, Nike, hoodie
click, Adidas, joggers
search, streetwear essentials, _
purchase, Nike, hoodie""")
                        t4_btn = gr.Button("Run Sequence", variant="primary")
                    with gr.Column(scale=2):
                        t4_evolution = gr.HTML(label="Score Evolution")

                with gr.Row():
                    with gr.Column(scale=2):
                        t4_final = gr.HTML(label="Final Feed")
                    with gr.Column(scale=1):
                        t4_state = gr.HTML(label="Final Session State")

                t4_btn.click(tab4_run_sequence, [t4_profile, t4_actions],
                             [t4_evolution, t4_final, t4_state])

            # TAB 5
            with gr.TabItem("5. Search Results"):
                gr.Markdown("See how search reranking changes after user actions.")
                with gr.Row():
                    with gr.Column(scale=1):
                        t5_profile = gr.Dropdown(choices=PROFILE_NAMES, label="Select Profile")
                        t5_scenario = gr.Dropdown(choices=list(ACTION_SCENARIOS.keys()), label="Action Scenario")
                        t5_btn = gr.Button("Compare Search Results", variant="primary")
                with gr.Row():
                    with gr.Column(scale=1):
                        t5_before = gr.HTML(label="Search Before")
                    with gr.Column(scale=1):
                        t5_after = gr.HTML(label="Search After")
                with gr.Row():
                    with gr.Column(scale=1):
                        t5_diff = gr.HTML(label="Diff")
                    with gr.Column(scale=1):
                        t5_state = gr.HTML(label="Session State")

                t5_btn.click(tab5_compare, [t5_profile, t5_scenario],
                             [t5_before, t5_after, t5_diff, t5_state])

            # TAB 6
            with gr.TabItem("6. Filter Explorer"):
                gr.Markdown(
                    "Explore **all in-stock products** with every available filter dimension. "
                    "AND across groups, OR within a group. The full pool loads from Supabase "
                    "in the background at startup."
                )

                with gr.Row():
                    # Left column: filters
                    with gr.Column(scale=1):
                        # -- Category & Type --
                        with gr.Accordion("Category & Type", open=True):
                            t6_categories = gr.Dropdown(
                                choices=DB_CATEGORIES, label="Broad Category",
                                multiselect=True, allow_custom_value=False,
                            )
                            t6_article_types = gr.Dropdown(
                                choices=DB_ARTICLE_TYPES, label="Article Type",
                                multiselect=True, allow_custom_value=False,
                            )

                        # -- Style --
                        with gr.Accordion("Style", open=False):
                            t6_styles = gr.Dropdown(
                                choices=GENERAL_STYLE_OPTIONS, label="General Style",
                                multiselect=True, allow_custom_value=False,
                            )
                            t6_seasons = gr.Dropdown(
                                choices=DB_SEASONS, label="Seasons",
                                multiselect=True, allow_custom_value=False,
                            )

                        # -- Color & Pattern --
                        with gr.Accordion("Color & Pattern", open=False):
                            t6_color_families = gr.Dropdown(
                                choices=DB_COLOR_FAMILIES, label="Color Family",
                                multiselect=True, allow_custom_value=False,
                            )
                            t6_patterns = gr.Dropdown(
                                choices=DB_PATTERNS, label="Pattern",
                                multiselect=True, allow_custom_value=False,
                            )

                        # -- Construction --
                        with gr.Accordion("Construction (Fit, Neckline, Sleeve, Length)", open=False):
                            t6_fit_types = gr.Dropdown(
                                choices=DB_FIT_TYPES, label="Fit Type",
                                multiselect=True, allow_custom_value=False,
                            )
                            t6_necklines = gr.Dropdown(
                                choices=DB_NECKLINES, label="Neckline",
                                multiselect=True, allow_custom_value=False,
                            )
                            t6_sleeve_types = gr.Dropdown(
                                choices=DB_SLEEVE_TYPES, label="Sleeve Type",
                                multiselect=True, allow_custom_value=False,
                            )
                            t6_lengths = gr.Dropdown(
                                choices=DB_LENGTHS, label="Length",
                                multiselect=True, allow_custom_value=False,
                            )
                            t6_silhouettes = gr.Dropdown(
                                choices=DB_SILHOUETTES, label="Silhouette",
                                multiselect=True, allow_custom_value=False,
                            )

                        # -- Brand & Price --
                        with gr.Accordion("Brand & Price", open=False):
                            t6_brands = gr.Dropdown(
                                choices=DB_BRANDS, label="Brands (include)",
                                multiselect=True, allow_custom_value=False,
                            )
                            t6_exclude_brands = gr.Dropdown(
                                choices=DB_BRANDS, label="Brands (exclude)",
                                multiselect=True, allow_custom_value=False,
                            )
                            with gr.Row():
                                t6_min_price = gr.Number(label="Min Price ($)", value=0, precision=2)
                                t6_max_price = gr.Number(label="Max Price ($)", value=0, precision=2)
                            t6_on_sale = gr.Checkbox(label="On Sale Only", value=False)

                        # -- Material --
                        with gr.Accordion("Material", open=False):
                            t6_materials = gr.Dropdown(
                                choices=sorted(MATERIAL_GROUPS.keys()), label="Material",
                                multiselect=True, allow_custom_value=False,
                            )

                        # -- Coverage & Body Type (Gemini Vision) --
                        with gr.Accordion("Coverage & Model (Gemini Vision)", open=False):
                            t6_coverage_levels = gr.Dropdown(
                                choices=DB_COVERAGE_LEVELS, label="Coverage Level",
                                multiselect=True, allow_custom_value=False,
                            )
                            t6_skin_exposure = gr.Dropdown(
                                choices=DB_SKIN_EXPOSURE, label="Skin Exposure",
                                multiselect=True, allow_custom_value=False,
                            )
                            t6_model_body_types = gr.Dropdown(
                                choices=DB_MODEL_BODY_TYPES, label="Model Body Type",
                                multiselect=True, allow_custom_value=False,
                            )
                            t6_model_size_estimates = gr.Dropdown(
                                choices=DB_MODEL_SIZE_ESTIMATES, label="Model Size Estimate",
                                multiselect=True, allow_custom_value=False,
                            )

                        with gr.Row():
                            t6_btn = gr.Button("Apply Filters", variant="primary", size="lg")
                            t6_btn_clear = gr.Button("Clear All", variant="stop", size="sm")

                    # Right column: results
                    with gr.Column(scale=2):
                        t6_summary = gr.HTML(label="Results Summary", value="")
                        t6_grid = gr.HTML(label="Product Grid")
                        t6_table = gr.Dataframe(
                            headers=FILTER_TABLE_HEADERS,
                            label="Data Table (up to 50 rows)",
                            wrap=True,
                            interactive=False,
                        )

                # All filter inputs for the click handler
                _t6_filter_inputs = [
                    t6_categories, t6_article_types, t6_styles,
                    t6_seasons,
                    t6_color_families, t6_patterns,
                    t6_fit_types, t6_necklines, t6_sleeve_types, t6_lengths, t6_silhouettes,
                    t6_brands, t6_exclude_brands, t6_min_price, t6_max_price, t6_on_sale,
                    t6_materials,
                    t6_coverage_levels, t6_skin_exposure, t6_model_body_types, t6_model_size_estimates,
                ]

                t6_btn.click(
                    tab6_apply_filters,
                    inputs=_t6_filter_inputs,
                    outputs=[t6_summary, t6_grid, t6_table],
                )

                # Clear all filters
                def _t6_clear():
                    """Reset all filter widgets to defaults."""
                    return (
                        [], [], [],                    # categories, article_types, styles
                        [],                            # seasons
                        [], [],                        # color_families, patterns
                        [], [], [], [], [],            # fit thru silhouettes
                        [], [], 0, 0, False,           # brands, exclude, prices, sale
                        [],                            # materials
                        [], [], [], [],                # gemini coverage/body type
                        "",                            # summary
                        "",                            # grid
                        [[""] * len(FILTER_TABLE_HEADERS)],  # table
                    )

                t6_btn_clear.click(
                    _t6_clear,
                    inputs=[],
                    outputs=_t6_filter_inputs + [t6_summary, t6_grid, t6_table],
                )

    return app


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Building Recommendation Demo UI...")
    print(f"  {len(PREMADE_PROFILES)} premade profiles")
    print(f"  {len(ACTION_SCENARIOS)} action scenarios")
    print("  Products loaded from DB")
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7861, share=False, css=CUSTOM_CSS)
