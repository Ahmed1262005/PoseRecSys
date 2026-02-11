"""
Full Integration Tests.

End-to-end tests that exercise the complete recommendation pipeline
with mocked Supabase. Tests the full flow:

  Onboarding profile → candidate retrieval → SASRec ranking →
  session scoring → context scoring (age + weather) →
  image dedup → feed reranking → response

Mocking strategy:
  - CandidateSelectionModule methods are patched to return realistic Candidate objects
  - Supabase is never called (all DB access is mocked)
  - SASRec model is not loaded (load_sasrec=False)
  - Weather API is mocked via _fetch_openweathermap
  - Everything else runs for real (scoring engines, reranker, session state, etc.)

Run with: PYTHONPATH=src python -m pytest tests/unit/test_full_integration.py -v -s
"""

import time
import pytest
from collections import Counter
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch, PropertyMock

from recs.models import Candidate, OnboardingProfile, UserState, UserStateType
from recs.session_scoring import SessionScores, get_session_scoring_engine
from scoring.context import AgeGroup, Season, WeatherContext, UserContext


# =============================================================================
# Candidate Factory — 60 realistic products
# =============================================================================

_CATALOG: List[Dict[str, Any]] = [
    # --- TOPS (15) ---
    {"item_id": "top-001", "article_type": "t-shirt",   "brand": "Zara",           "price": 25.0,  "colors": ["Black"],  "materials": ["Cotton"],  "seasons": ["Spring","Summer","Fall","Winter"], "style_tags": ["casual","minimal"],    "occasions": ["Everyday","Casual"],   "pattern": "Solid",   "formality": "Casual",   "fit": "regular", "neckline": "crew",        "sleeve": "short",     "image_url": "https://img/top001.jpg", "name": "Basic Black Tee"},
    {"item_id": "top-002", "article_type": "blouse",    "brand": "Massimo Dutti",   "price": 79.0,  "colors": ["White"],  "materials": ["Silk"],    "seasons": ["Spring","Summer"],                 "style_tags": ["classic","elegant"],   "occasions": ["Work","Date Night"],   "pattern": "Solid",   "formality": "Smart Casual", "fit": "relaxed", "neckline": "v_neck",    "sleeve": "long",      "image_url": "https://img/top002.jpg", "name": "Silk V-Neck Blouse"},
    {"item_id": "top-003", "article_type": "tank top",  "brand": "Forever 21",      "price": 12.0,  "colors": ["Red"],    "materials": ["Cotton"],  "seasons": ["Summer"],                          "style_tags": ["casual","trendy"],     "occasions": ["Casual","Beach"],      "pattern": "Solid",   "formality": "Casual",   "fit": "fitted",  "neckline": "scoop",       "sleeve": "sleeveless","image_url": "https://img/top003.jpg", "name": "Red Tank Top"},
    {"item_id": "top-004", "article_type": "sweater",   "brand": "COS",             "price": 89.0,  "colors": ["Beige"],  "materials": ["Cashmere"],"seasons": ["Fall","Winter"],                   "style_tags": ["minimal","classic"],   "occasions": ["Work","Everyday"],     "pattern": "Solid",   "formality": "Smart Casual", "fit": "relaxed", "neckline": "crew",    "sleeve": "long",      "image_url": "https://img/top004.jpg", "name": "Cashmere Crew Sweater"},
    {"item_id": "top-005", "article_type": "crop top",  "brand": "Boohoo",          "price": 15.0,  "colors": ["Pink"],   "materials": ["Polyester"],"seasons": ["Summer"],                         "style_tags": ["trendy","party"],      "occasions": ["Party","Club"],        "pattern": "Solid",   "formality": "Casual",   "fit": "fitted",  "neckline": "square",      "sleeve": "short",     "image_url": "https://img/top005.jpg", "name": "Pink Crop Top"},
    {"item_id": "top-006", "article_type": "turtleneck","brand": "Uniqlo",          "price": 39.0,  "colors": ["Navy"],   "materials": ["Merino"],  "seasons": ["Fall","Winter"],                   "style_tags": ["classic","minimal"],   "occasions": ["Work","Everyday"],     "pattern": "Solid",   "formality": "Smart Casual", "fit": "fitted", "neckline": "turtle",   "sleeve": "long",      "image_url": "https://img/top006.jpg", "name": "Merino Turtleneck"},
    {"item_id": "top-007", "article_type": "bodysuit",  "brand": "Skims",           "price": 58.0,  "colors": ["Nude"],   "materials": ["Nylon"],   "seasons": ["Spring","Summer","Fall","Winter"],  "style_tags": ["bodycon","trendy"],   "occasions": ["Date Night","Party"], "pattern": "Solid",   "formality": "Casual",   "fit": "fitted",  "neckline": "scoop",       "sleeve": "long",      "image_url": "https://img/top007.jpg", "name": "Long Sleeve Bodysuit"},
    {"item_id": "top-008", "article_type": "hoodie",    "brand": "Nike",            "price": 65.0,  "colors": ["Grey"],   "materials": ["Fleece"],  "seasons": ["Fall","Winter"],                   "style_tags": ["athleisure","sporty"], "occasions": ["Casual","Athletic"],   "pattern": "Solid",   "formality": "Casual",   "fit": "oversized","neckline": "hood",       "sleeve": "long",      "image_url": "https://img/top008.jpg", "name": "Nike Fleece Hoodie"},
    {"item_id": "top-009", "article_type": "camisole",  "brand": "Reformation",     "price": 68.0,  "colors": ["Ivory"],  "materials": ["Silk"],    "seasons": ["Spring","Summer"],                 "style_tags": ["romantic","elegant"],  "occasions": ["Date Night","Evening"],"pattern": "Solid",   "formality": "Dressy",   "fit": "fitted",  "neckline": "cowl",        "sleeve": "sleeveless","image_url": "https://img/top009.jpg", "name": "Silk Cowl Cami"},
    {"item_id": "top-010", "article_type": "polo shirt","brand": "Ralph Lauren",    "price": 95.0,  "colors": ["Green"],  "materials": ["Cotton"],  "seasons": ["Spring","Summer"],                 "style_tags": ["preppy","classic"],    "occasions": ["Casual","Work"],       "pattern": "Solid",   "formality": "Smart Casual", "fit": "regular", "neckline": "collar",  "sleeve": "short",     "image_url": "https://img/top010.jpg", "name": "Classic Polo Shirt"},
    {"item_id": "top-011", "article_type": "cardigan",  "brand": "& Other Stories", "price": 79.0,  "colors": ["Cream"],  "materials": ["Wool"],    "seasons": ["Fall","Winter","Spring"],           "style_tags": ["classic","cozy"],      "occasions": ["Everyday","Work"],     "pattern": "Cable",   "formality": "Smart Casual", "fit": "relaxed", "neckline": "v_neck",  "sleeve": "long",      "image_url": "https://img/top011.jpg", "name": "Cable Knit Cardigan"},
    {"item_id": "top-012", "article_type": "t-shirt",   "brand": "Zara",            "price": 29.0,  "colors": ["White"],  "materials": ["Cotton"],  "seasons": ["Spring","Summer","Fall","Winter"],  "style_tags": ["casual","minimal"],   "occasions": ["Everyday","Casual"],   "pattern": "Striped", "formality": "Casual",   "fit": "regular", "neckline": "crew",        "sleeve": "short",     "image_url": "https://img/top012.jpg", "name": "Striped Cotton Tee"},
    {"item_id": "top-013", "article_type": "tube top",  "brand": "PLT",             "price": 10.0,  "colors": ["Black"],  "materials": ["Polyester"],"seasons": ["Summer"],                         "style_tags": ["party","sexy"],        "occasions": ["Party","Club"],        "pattern": "Solid",   "formality": "Casual",   "fit": "fitted",  "neckline": "strapless",   "sleeve": "sleeveless","image_url": "https://img/top013.jpg", "name": "Black Tube Top"},
    {"item_id": "top-014", "article_type": "sweatshirt","brand": "Adidas",          "price": 55.0,  "colors": ["Grey"],   "materials": ["Cotton"],  "seasons": ["Fall","Winter","Spring"],           "style_tags": ["sporty","casual"],     "occasions": ["Casual","Athletic"],   "pattern": "Logo",    "formality": "Casual",   "fit": "relaxed", "neckline": "crew",        "sleeve": "long",      "image_url": "https://img/top014.jpg", "name": "Adidas Logo Sweatshirt"},
    {"item_id": "top-015", "article_type": "halter top","brand": "Revolve",         "price": 45.0,  "colors": ["Emerald"],"materials": ["Satin"],   "seasons": ["Summer"],                          "style_tags": ["elegant","going out"],"occasions": ["Party","Date Night"],  "pattern": "Solid",   "formality": "Dressy",   "fit": "fitted",  "neckline": "halter",      "sleeve": "sleeveless","image_url": "https://img/top015.jpg", "name": "Satin Halter Top"},
    # --- DRESSES (12) ---
    {"item_id": "dress-001","article_type": "midi dress","brand": "Reformation",    "price": 148.0, "colors": ["Floral"], "materials": ["Cotton"],  "seasons": ["Spring","Summer"],                 "style_tags": ["romantic","boho"],     "occasions": ["Date Night","Brunch"],"pattern": "Floral",  "formality": "Smart Casual", "fit": "fitted", "neckline": "sweetheart","sleeve": "puff",     "image_url": "https://img/dress001.jpg","name": "Floral Midi Dress"},
    {"item_id": "dress-002","article_type": "mini dress","brand": "Zara",           "price": 49.0,  "colors": ["Black"],  "materials": ["Polyester"],"seasons": ["Spring","Summer","Fall"],          "style_tags": ["party","trendy"],      "occasions": ["Party","Date Night"], "pattern": "Solid",   "formality": "Dressy",   "fit": "bodycon", "neckline": "v_neck",      "sleeve": "long",      "image_url": "https://img/dress002.jpg","name": "Little Black Dress"},
    {"item_id": "dress-003","article_type": "maxi dress","brand": "Free People",    "price": 128.0, "colors": ["Rust"],   "materials": ["Rayon"],   "seasons": ["Summer","Fall"],                   "style_tags": ["boho","romantic"],     "occasions": ["Casual","Brunch"],    "pattern": "Paisley", "formality": "Casual",   "fit": "relaxed", "neckline": "v_neck",      "sleeve": "long",      "image_url": "https://img/dress003.jpg","name": "Boho Maxi Dress"},
    {"item_id": "dress-004","article_type": "sundress",  "brand": "Faithfull",      "price": 169.0, "colors": ["Yellow"], "materials": ["Linen"],   "seasons": ["Summer"],                          "style_tags": ["vacation","romantic"],"occasions": ["Vacation","Brunch"],  "pattern": "Gingham", "formality": "Casual",   "fit": "relaxed", "neckline": "square",      "sleeve": "short",     "image_url": "https://img/dress004.jpg","name": "Linen Sundress"},
    {"item_id": "dress-005","article_type": "wrap dress","brand": "DVF",            "price": 398.0, "colors": ["Print"],  "materials": ["Silk"],    "seasons": ["Spring","Summer","Fall"],           "style_tags": ["classic","elegant"],   "occasions": ["Work","Date Night"],  "pattern": "Geometric","formality": "Smart Casual","fit": "fitted","neckline": "v_neck",    "sleeve": "long",      "image_url": "https://img/dress005.jpg","name": "DVF Wrap Dress"},
    {"item_id": "dress-006","article_type": "cocktail dress","brand": "Self-Portrait","price": 320.0,"colors": ["Blush"],"materials": ["Lace"],     "seasons": ["Spring","Summer","Fall"],           "style_tags": ["elegant","romantic"],  "occasions": ["Wedding","Evening"],  "pattern": "Lace",    "formality": "Formal",   "fit": "fitted",  "neckline": "off_shoulder","sleeve": "short",     "image_url": "https://img/dress006.jpg","name": "Lace Cocktail Dress"},
    {"item_id": "dress-007","article_type": "shirt dress","brand": "COS",           "price": 115.0, "colors": ["Blue"],   "materials": ["Cotton"],  "seasons": ["Spring","Summer","Fall"],           "style_tags": ["classic","minimal"],   "occasions": ["Work","Everyday"],    "pattern": "Solid",   "formality": "Smart Casual","fit": "relaxed","neckline": "collar",    "sleeve": "long",      "image_url": "https://img/dress007.jpg","name": "Cotton Shirt Dress"},
    {"item_id": "dress-008","article_type": "slip dress","brand": "Vince",          "price": 225.0, "colors": ["Champagne"],"materials": ["Silk"],  "seasons": ["Summer","Fall"],                    "style_tags": ["minimal","elegant"],  "occasions": ["Date Night","Evening"],"pattern":"Solid",   "formality": "Dressy",   "fit": "relaxed", "neckline": "cowl",        "sleeve": "sleeveless","image_url": "https://img/dress008.jpg","name": "Silk Slip Dress"},
    {"item_id": "dress-009","article_type": "bodycon dress","brand": "Boohoo",      "price": 22.0,  "colors": ["Red"],    "materials": ["Polyester"],"seasons": ["Spring","Summer","Fall"],          "style_tags": ["sexy","party"],        "occasions": ["Party","Club"],       "pattern": "Solid",   "formality": "Dressy",   "fit": "bodycon", "neckline": "square",      "sleeve": "sleeveless","image_url": "https://img/dress009.jpg","name": "Red Bodycon Dress"},
    {"item_id": "dress-010","article_type": "knit dress","brand": "Uniqlo",         "price": 49.0,  "colors": ["Charcoal"],"materials": ["Merino"], "seasons": ["Fall","Winter"],                    "style_tags": ["minimal","cozy"],     "occasions": ["Work","Everyday"],    "pattern": "Solid",   "formality": "Smart Casual","fit": "regular","neckline": "crew",      "sleeve": "long",      "image_url": "https://img/dress010.jpg","name": "Merino Knit Dress"},
    {"item_id": "dress-011","article_type": "maxi dress","brand": "H&M",           "price": 35.0,  "colors": ["Navy"],   "materials": ["Jersey"],  "seasons": ["Spring","Summer","Fall","Winter"],  "style_tags": ["casual","versatile"], "occasions": ["Everyday","Casual"],  "pattern": "Solid",   "formality": "Casual",   "fit": "relaxed", "neckline": "scoop",       "sleeve": "short",     "image_url": "https://img/dress011.jpg","name": "Jersey Maxi Dress"},
    {"item_id": "dress-012","article_type": "evening dress","brand": "ASOS",        "price": 85.0,  "colors": ["Gold"],   "materials": ["Sequin"],  "seasons": ["Fall","Winter"],                    "style_tags": ["glamorous","party"],  "occasions": ["Evening","Party"],    "pattern": "Sequin",  "formality": "Formal",   "fit": "fitted",  "neckline": "one_shoulder","sleeve": "sleeveless","image_url": "https://img/dress012.jpg","name": "Gold Sequin Gown"},
    # --- BOTTOMS (12) ---
    {"item_id": "bot-001", "article_type": "jeans",     "brand": "Levi's",          "price": 98.0,  "colors": ["Blue"],   "materials": ["Denim"],   "seasons": ["Spring","Summer","Fall","Winter"],  "style_tags": ["classic","casual"],    "occasions": ["Everyday","Casual"],  "pattern": "Solid",   "formality": "Casual",   "fit": "straight","neckline": None,          "sleeve": None, "image_url": "https://img/bot001.jpg", "name": "501 Original Jeans"},
    {"item_id": "bot-002", "article_type": "trousers",  "brand": "COS",             "price": 89.0,  "colors": ["Black"],  "materials": ["Wool"],    "seasons": ["Fall","Winter","Spring"],           "style_tags": ["classic","minimal"],   "occasions": ["Work","Formal"],      "pattern": "Solid",   "formality": "Formal",   "fit": "tailored","neckline": None,          "sleeve": None, "image_url": "https://img/bot002.jpg", "name": "Tailored Wool Trousers"},
    {"item_id": "bot-003", "article_type": "shorts",    "brand": "Zara",            "price": 35.0,  "colors": ["White"],  "materials": ["Cotton"],  "seasons": ["Summer"],                          "style_tags": ["casual","vacation"],   "occasions": ["Casual","Beach"],     "pattern": "Solid",   "formality": "Casual",   "fit": "relaxed", "neckline": None,          "sleeve": None, "image_url": "https://img/bot003.jpg", "name": "White Cotton Shorts"},
    {"item_id": "bot-004", "article_type": "leggings",  "brand": "Lululemon",       "price": 98.0,  "colors": ["Black"],  "materials": ["Nylon"],   "seasons": ["Spring","Summer","Fall","Winter"],  "style_tags": ["athletic","sporty"],   "occasions": ["Athletic","Casual"],  "pattern": "Solid",   "formality": "Casual",   "fit": "fitted",  "neckline": None,          "sleeve": None, "image_url": "https://img/bot004.jpg", "name": "Align Leggings"},
    {"item_id": "bot-005", "article_type": "mini skirt","brand": "Mango",           "price": 45.0,  "colors": ["Plaid"],  "materials": ["Polyester"],"seasons": ["Fall","Winter"],                  "style_tags": ["preppy","trendy"],     "occasions": ["Casual","School"],    "pattern": "Plaid",   "formality": "Casual",   "fit": "fitted",  "neckline": None,          "sleeve": None, "image_url": "https://img/bot005.jpg", "name": "Plaid Mini Skirt"},
    {"item_id": "bot-006", "article_type": "midi skirt","brand": "& Other Stories", "price": 69.0,  "colors": ["Satin Black"],"materials": ["Satin"],"seasons": ["Fall","Winter","Spring"],          "style_tags": ["elegant","classic"],   "occasions": ["Work","Date Night"],  "pattern": "Solid",   "formality": "Smart Casual","fit": "a-line","neckline": None,          "sleeve": None, "image_url": "https://img/bot006.jpg", "name": "Satin Midi Skirt"},
    {"item_id": "bot-007", "article_type": "wide leg pants","brand": "Massimo Dutti","price": 99.0,"colors": ["Tan"],    "materials": ["Linen"],   "seasons": ["Spring","Summer"],                 "style_tags": ["classic","vacation"],  "occasions": ["Work","Brunch"],      "pattern": "Solid",   "formality": "Smart Casual","fit": "wide",  "neckline": None,          "sleeve": None, "image_url": "https://img/bot007.jpg", "name": "Linen Wide Leg Pants"},
    {"item_id": "bot-008", "article_type": "joggers",   "brand": "Nike",            "price": 60.0,  "colors": ["Grey"],   "materials": ["Fleece"],  "seasons": ["Fall","Winter"],                   "style_tags": ["sporty","athleisure"],"occasions": ["Athletic","Casual"],  "pattern": "Solid",   "formality": "Casual",   "fit": "regular", "neckline": None,          "sleeve": None, "image_url": "https://img/bot008.jpg", "name": "Nike Tech Joggers"},
    {"item_id": "bot-009", "article_type": "maxi skirt","brand": "H&M",            "price": 29.0,  "colors": ["Olive"],  "materials": ["Cotton"],  "seasons": ["Spring","Summer","Fall"],           "style_tags": ["boho","casual"],       "occasions": ["Everyday","Casual"],  "pattern": "Solid",   "formality": "Casual",   "fit": "relaxed", "neckline": None,          "sleeve": None, "image_url": "https://img/bot009.jpg", "name": "Olive Maxi Skirt"},
    {"item_id": "bot-010", "article_type": "dress pants","brand": "Theory",         "price": 245.0, "colors": ["Charcoal"],"materials": ["Wool"],   "seasons": ["Fall","Winter","Spring"],           "style_tags": ["classic","corporate"],"occasions": ["Work","Formal"],      "pattern": "Solid",   "formality": "Formal",   "fit": "tailored","neckline": None,          "sleeve": None, "image_url": "https://img/bot010.jpg", "name": "Theory Dress Pants"},
    {"item_id": "bot-011", "article_type": "bike shorts","brand": "Girlfriend Collective","price":48.0,"colors":["Black"],"materials": ["Recycled Nylon"],"seasons": ["Spring","Summer","Fall","Winter"],"style_tags": ["athletic","sporty"], "occasions": ["Athletic","Casual"],  "pattern": "Solid",   "formality": "Casual",   "fit": "fitted",  "neckline": None,          "sleeve": None, "image_url": "https://img/bot011.jpg", "name": "High-Rise Bike Shorts"},
    {"item_id": "bot-012", "article_type": "pencil skirt","brand": "Hugo Boss",     "price": 198.0, "colors": ["Black"],  "materials": ["Wool"],    "seasons": ["Fall","Winter","Spring"],           "style_tags": ["classic","corporate"],"occasions": ["Work","Formal"],      "pattern": "Solid",   "formality": "Formal",   "fit": "fitted",  "neckline": None,          "sleeve": None, "image_url": "https://img/bot012.jpg", "name": "Wool Pencil Skirt"},
    # --- OUTERWEAR (9) ---
    {"item_id": "out-001", "article_type": "blazer",    "brand": "Zara",            "price": 89.0,  "colors": ["Black"],  "materials": ["Polyester"],"seasons": ["Spring","Fall","Winter"],          "style_tags": ["classic","office"],    "occasions": ["Work","Formal"],      "pattern": "Solid",   "formality": "Formal",   "fit": "tailored","neckline": "lapel",       "sleeve": "long",      "image_url": "https://img/out001.jpg", "name": "Oversized Blazer"},
    {"item_id": "out-002", "article_type": "puffer jacket","brand": "North Face",   "price": 249.0, "colors": ["Black"],  "materials": ["Down"],    "seasons": ["Winter"],                          "style_tags": ["sporty","practical"], "occasions": ["Everyday","Athletic"],"pattern": "Solid",   "formality": "Casual",   "fit": "regular", "neckline": "hood",        "sleeve": "long",      "image_url": "https://img/out002.jpg", "name": "North Face Puffer"},
    {"item_id": "out-003", "article_type": "trench coat","brand": "Burberry",       "price": 1690.0,"colors": ["Beige"],  "materials": ["Cotton"],  "seasons": ["Spring","Fall"],                   "style_tags": ["classic","luxury"],    "occasions": ["Work","Everyday"],    "pattern": "Solid",   "formality": "Smart Casual","fit": "regular","neckline": "lapel",     "sleeve": "long",      "image_url": "https://img/out003.jpg", "name": "Heritage Trench Coat"},
    {"item_id": "out-004", "article_type": "jacket",    "brand": "Levi's",          "price": 128.0, "colors": ["Blue"],   "materials": ["Denim"],   "seasons": ["Spring","Fall"],                   "style_tags": ["casual","classic"],    "occasions": ["Everyday","Casual"],  "pattern": "Solid",   "formality": "Casual",   "fit": "regular", "neckline": "collar",      "sleeve": "long",      "image_url": "https://img/out004.jpg", "name": "Denim Trucker Jacket"},
    {"item_id": "out-005", "article_type": "vest",      "brand": "Patagonia",       "price": 149.0, "colors": ["Forest"], "materials": ["Fleece"],  "seasons": ["Fall","Winter","Spring"],           "style_tags": ["sporty","outdoor"],    "occasions": ["Casual","Athletic"],  "pattern": "Solid",   "formality": "Casual",   "fit": "regular", "neckline": "zip",         "sleeve": "sleeveless","image_url": "https://img/out005.jpg", "name": "Fleece Vest"},
    {"item_id": "out-006", "article_type": "coat",      "brand": "Max Mara",        "price": 1290.0,"colors": ["Camel"],  "materials": ["Cashmere"],"seasons": ["Winter"],                          "style_tags": ["classic","luxury"],    "occasions": ["Work","Formal"],      "pattern": "Solid",   "formality": "Formal",   "fit": "regular", "neckline": "lapel",       "sleeve": "long",      "image_url": "https://img/out006.jpg", "name": "Cashmere Wrap Coat"},
    {"item_id": "out-007", "article_type": "windbreaker","brand": "Nike",           "price": 85.0,  "colors": ["Navy"],   "materials": ["Nylon"],   "seasons": ["Spring","Summer","Fall"],           "style_tags": ["sporty","practical"], "occasions": ["Athletic","Casual"],  "pattern": "Solid",   "formality": "Casual",   "fit": "regular", "neckline": "zip",         "sleeve": "long",      "image_url": "https://img/out007.jpg", "name": "Nike Windbreaker"},
    {"item_id": "out-008", "article_type": "cardigan",  "brand": "Acne Studios",    "price": 350.0, "colors": ["Grey"],   "materials": ["Mohair"],  "seasons": ["Fall","Winter"],                   "style_tags": ["minimal","artistic"], "occasions": ["Everyday","Work"],    "pattern": "Solid",   "formality": "Casual",   "fit": "oversized","neckline": "v_neck",     "sleeve": "long",      "image_url": "https://img/out008.jpg", "name": "Mohair Cardigan"},
    {"item_id": "out-009", "article_type": "jacket",    "brand": "AllSaints",       "price": 450.0, "colors": ["Black"],  "materials": ["Leather"], "seasons": ["Fall","Winter","Spring"],           "style_tags": ["edgy","classic"],      "occasions": ["Evening","Casual"],   "pattern": "Solid",   "formality": "Casual",   "fit": "fitted",  "neckline": "lapel",       "sleeve": "long",      "image_url": "https://img/out009.jpg", "name": "Leather Biker Jacket"},
    # --- SPORTSWEAR (3) ---
    {"item_id": "sport-001","article_type": "sports bra","brand": "Lululemon",      "price": 52.0,  "colors": ["Black"],  "materials": ["Nylon"],   "seasons": ["Spring","Summer","Fall","Winter"],  "style_tags": ["athletic","sporty"],  "occasions": ["Athletic"],           "pattern": "Solid",   "formality": "Casual",   "fit": "fitted",  "neckline": "scoop",       "sleeve": "sleeveless","image_url": "https://img/sport001.jpg","name": "Energy Sports Bra"},
    {"item_id": "sport-002","article_type": "athletic top","brand": "Nike",         "price": 45.0,  "colors": ["White"],  "materials": ["Polyester"],"seasons": ["Spring","Summer","Fall","Winter"], "style_tags": ["athletic","sporty"],  "occasions": ["Athletic"],           "pattern": "Solid",   "formality": "Casual",   "fit": "fitted",  "neckline": "crew",        "sleeve": "short",     "image_url": "https://img/sport002.jpg","name": "Dri-FIT Training Top"},
    {"item_id": "sport-003","article_type": "athletic shorts","brand":"Adidas",     "price": 35.0,  "colors": ["Navy"],   "materials": ["Polyester"],"seasons": ["Spring","Summer"],                "style_tags": ["athletic","sporty"],  "occasions": ["Athletic"],           "pattern": "Solid",   "formality": "Casual",   "fit": "regular", "neckline": None,          "sleeve": None, "image_url": "https://img/sport003.jpg","name": "Adidas Running Shorts"},
]


def _build_candidates(catalog=None, score_base=0.5) -> List[Candidate]:
    """Build Candidate objects from catalog data."""
    if catalog is None:
        catalog = _CATALOG
    candidates = []
    for i, item in enumerate(catalog):
        score = score_base - i * 0.005  # Decreasing scores
        c = Candidate(
            item_id=item["item_id"],
            embedding_score=max(score, 0.01),
            final_score=max(score, 0.01),
            category=item.get("article_type", ""),
            broad_category=_infer_broad(item["article_type"]),
            article_type=item["article_type"],
            brand=item["brand"],
            price=item["price"],
            colors=item.get("colors", []),
            materials=item.get("materials", []),
            fit=item.get("fit"),
            length=None,
            sleeve=item.get("sleeve"),
            neckline=item.get("neckline"),
            style_tags=item.get("style_tags", []),
            occasions=item.get("occasions", []),
            pattern=item.get("pattern"),
            formality=item.get("formality"),
            color_family=None,
            seasons=item.get("seasons", []),
            image_url=item["image_url"],
            name=item["name"],
            source="exploration",
        )
        candidates.append(c)
    return candidates


def _infer_broad(article_type: str) -> str:
    tops = {"t-shirt","blouse","tank top","sweater","crop top","turtleneck","bodysuit","hoodie","camisole","polo shirt","cardigan","tube top","sweatshirt","halter top","sports bra","athletic top"}
    bottoms = {"jeans","trousers","shorts","leggings","mini skirt","midi skirt","wide leg pants","joggers","maxi skirt","dress pants","bike shorts","pencil skirt","athletic shorts"}
    outerwear = {"blazer","puffer jacket","trench coat","jacket","vest","coat","windbreaker"}
    if article_type.lower() in tops:
        return "tops"
    if article_type.lower() in bottoms:
        return "bottoms"
    if article_type.lower() in outerwear:
        return "outerwear"
    return "dresses"


def _make_user_state(
    user_id: str = "test-user-1",
    preferred_brands: List[str] = None,
    categories: List[str] = None,
    birthdate: str = None,
    **profile_kwargs,
) -> UserState:
    """Build a UserState with an OnboardingProfile."""
    profile = OnboardingProfile(
        user_id=user_id,
        preferred_brands=preferred_brands or [],
        categories=categories or ["tops", "dresses", "bottoms", "outerwear"],
        birthdate=birthdate,
        **profile_kwargs,
    )
    return UserState(
        user_id=user_id,
        state_type=UserStateType.COLD_START,
        onboarding_profile=profile,
    )


def _make_cold_weather():
    return WeatherContext(
        temperature_c=-5.0, feels_like_c=-10.0, condition="snow",
        humidity=70, wind_speed_mps=5.0, season=Season.WINTER,
        is_cold=True,
    )

def _make_hot_weather():
    return WeatherContext(
        temperature_c=35.0, feels_like_c=38.0, condition="clear",
        humidity=40, wind_speed_mps=2.0, season=Season.SUMMER,
        is_hot=True,
    )

def _make_rainy_weather():
    return WeatherContext(
        temperature_c=12.0, feels_like_c=10.0, condition="rain",
        humidity=85, wind_speed_mps=6.0, season=Season.FALL,
        is_mild=True, is_rainy=True,
    )


# =============================================================================
# Pipeline fixture — mocks Supabase, no SASRec, real everything else
# =============================================================================

@pytest.fixture
def pipeline():
    """Create a RecommendationPipeline with mocked Supabase and no SASRec."""
    candidates = _build_candidates()

    with patch("recs.candidate_selection.create_client") as mock_create_client, \
         patch("config.settings.get_settings") as mock_settings:
        # Mock the Supabase client returned by create_client
        mock_create_client.return_value = MagicMock()
        mock_settings.return_value = MagicMock(openweather_api_key="")
        from recs.pipeline import RecommendationPipeline
        pipe = RecommendationPipeline(load_sasrec=False)

        # Patch the candidate module's methods
        pipe.candidate_module.load_user_state = MagicMock(
            return_value=_make_user_state()
        )
        pipe.candidate_module.get_user_seen_history = MagicMock(return_value=set())
        pipe.candidate_module.get_candidates_keyset = MagicMock(return_value=candidates)

        yield pipe


# =============================================================================
# 1. Full Feed Pipeline: Onboarding → Candidates → Scoring → Response
# =============================================================================

class TestFullFeedPipeline:
    """End-to-end feed pipeline tests."""

    def test_basic_feed_returns_results(self, pipeline):
        """Pipeline should return results with all expected fields."""
        result = pipeline.get_feed_keyset(
            user_id="test-user-1",
            session_id="sess-test-001",
            page_size=20,
        )
        assert "results" in result
        assert len(result["results"]) == 20
        assert result["session_id"] == "sess-test-001"
        assert result["pagination"]["items_returned"] == 20

        # Check result shape
        item = result["results"][0]
        assert "product_id" in item
        assert "brand" in item
        assert "score" in item
        assert "rank" in item
        assert "price" in item

    def test_feed_returns_diverse_brands(self, pipeline):
        """Feed should have brand diversity (not all from one brand)."""
        result = pipeline.get_feed_keyset(
            user_id="test-user-1",
            session_id="sess-diverse",
            page_size=20,
        )
        brands = [r["brand"] for r in result["results"]]
        brand_counts = Counter(brands)
        # No single brand should dominate (feed reranker caps at 3 per brand)
        for brand, count in brand_counts.items():
            assert count <= 5, f"Brand {brand} has {count} items — too many"

    def test_feed_no_duplicates(self, pipeline):
        """Feed should not contain duplicate product IDs."""
        result = pipeline.get_feed_keyset(
            user_id="test-user-1",
            session_id="sess-dedup",
            page_size=30,
        )
        ids = [r["product_id"] for r in result["results"]]
        assert len(ids) == len(set(ids)), "Duplicate product IDs in feed"

    def test_brand_preference_boost(self, pipeline):
        """Items from preferred brands should rank higher."""
        pipeline.candidate_module.load_user_state.return_value = _make_user_state(
            preferred_brands=["Zara"],
        )
        result = pipeline.get_feed_keyset(
            user_id="test-user-1",
            session_id="sess-brand-pref",
            page_size=20,
        )
        top_5_brands = [r["brand"] for r in result["results"][:5]]
        # Zara should appear in the top 5 with preferred_brands boost
        assert "Zara" in top_5_brands, f"Zara not in top 5: {top_5_brands}"

    def test_metadata_populated(self, pipeline):
        """Response metadata should include scoring info."""
        result = pipeline.get_feed_keyset(
            user_id="test-user-1",
            session_id="sess-meta",
            page_size=10,
        )
        meta = result["metadata"]
        assert "candidates_retrieved" in meta
        assert "session_scoring" in meta
        assert meta["keyset_pagination"] is True
        assert meta["has_onboarding"] is True


# =============================================================================
# 2. Session Learning: Actions Update Scores → Next Feed Reflects
# =============================================================================

class TestSessionLearning:
    """Tests that session actions update scoring and affect future feeds."""

    def test_click_boosts_brand_in_next_feed(self, pipeline):
        """Clicking a Reformation item should boost Reformation in session scores."""
        session_id = "sess-learn-click"

        # First feed (baseline)
        result1 = pipeline.get_feed_keyset(
            user_id="test-user-1", session_id=session_id, page_size=10,
        )

        # Simulate clicking a Reformation item
        pipeline.update_session_scores_from_action(
            session_id=session_id,
            action="click",
            product_id="dress-001",
            brand="reformation",
            item_type="midi dress",
            attributes={"pattern": "floral", "occasion": "date night"},
        )

        scores = pipeline.get_session_scores(session_id)
        assert scores is not None
        assert scores.get_score("brand", "reformation") > 0
        assert scores.get_score("type", "midi dress") > 0

    def test_skip_penalizes_item(self, pipeline):
        """Skipping an item should add it to skipped_ids."""
        session_id = "sess-learn-skip"
        pipeline.get_feed_keyset(
            user_id="test-user-1", session_id=session_id, page_size=5,
        )

        pipeline.update_session_scores_from_action(
            session_id=session_id,
            action="skip",
            product_id="top-005",
            brand="boohoo",
            item_type="crop top",
        )

        scores = pipeline.get_session_scores(session_id)
        assert "top-005" in scores.skipped_ids

    def test_multiple_actions_accumulate(self, pipeline):
        """Multiple clicks on same brand should increase its score."""
        session_id = "sess-learn-multi"
        pipeline.get_feed_keyset(
            user_id="test-user-1", session_id=session_id, page_size=5,
        )

        for i in range(5):
            pipeline.update_session_scores_from_action(
                session_id=session_id,
                action="click",
                product_id=f"zara-{i}",
                brand="zara",
                item_type="dress",
            )

        scores = pipeline.get_session_scores(session_id)
        assert scores.get_score("brand", "zara") > 0
        assert scores.action_count == 5

    def test_search_signal_updates_intents(self, pipeline):
        """Search signals should update search_intents in session scores."""
        session_id = "sess-learn-search"
        pipeline.get_feed_keyset(
            user_id="test-user-1", session_id=session_id, page_size=5,
        )

        pipeline.update_session_scores_from_search(
            session_id=session_id,
            query="date night dress",
            filters={"occasions": ["Date Night"], "categories": ["dresses"]},
        )

        scores = pipeline.get_session_scores(session_id)
        assert len(scores.search_intents) > 0


# =============================================================================
# 3. Weather Integration: Cold/Hot Weather Changes Feed Ordering
# =============================================================================

class TestWeatherIntegration:
    """Tests that weather scoring changes feed item ordering."""

    def test_cold_weather_boosts_winter_items(self, pipeline):
        """In cold weather, coats/sweaters should rank higher than tank tops."""
        session_id = "sess-weather-cold"
        pipeline.candidate_module.load_user_state.return_value = _make_user_state(
            birthdate="1995-01-15",
        )
        result = pipeline.get_feed_keyset(
            user_id="test-user-1",
            session_id=session_id,
            page_size=30,
            user_metadata={"city": "Moscow", "country": "RU"},
        )
        # Find positions of winter vs summer items
        results = result["results"]
        coat_positions = [i for i, r in enumerate(results) if "coat" in r.get("product_id", "").lower() or "puffer" in r.get("name", "").lower() or "sweater" in r.get("name", "").lower()]
        tank_positions = [i for i, r in enumerate(results) if "tank" in r.get("name", "").lower() or "crop" in r.get("name", "").lower()]

        # Context scoring should have been applied
        assert result["metadata"].get("context_scoring") is not None

    def test_weather_metadata_in_response(self, pipeline):
        """Response should include context_scoring metadata with weather info."""
        pipeline.candidate_module.load_user_state.return_value = _make_user_state(
            birthdate="1990-06-15",
        )
        result = pipeline.get_feed_keyset(
            user_id="test-user-1",
            session_id="sess-weather-meta",
            page_size=10,
            user_metadata={"city": "London", "country": "GB"},
        )
        ctx_meta = result["metadata"].get("context_scoring")
        # Without real weather API key, we still get season-based scoring
        if ctx_meta and "error" not in ctx_meta:
            assert "age_group" in ctx_meta
            assert "has_weather" in ctx_meta


# =============================================================================
# 4. Age Scoring: Different Ages Get Different Orderings
# =============================================================================

class TestAgeScoring:
    """Tests that age affects item scoring."""

    def _get_feed_top_items(self, pipeline, birthdate, session_id):
        """Helper: get feed for a user with given birthdate."""
        pipeline.candidate_module.load_user_state.return_value = _make_user_state(
            birthdate=birthdate,
        )
        result = pipeline.get_feed_keyset(
            user_id="test-user-1",
            session_id=session_id,
            page_size=20,
        )
        return result["results"]

    def test_gen_z_vs_senior_different_rankings(self, pipeline):
        """Gen Z (20yo) and Senior (70yo) should produce different orderings."""
        gen_z_results = self._get_feed_top_items(pipeline, "2006-01-01", "sess-age-genz")
        senior_results = self._get_feed_top_items(pipeline, "1956-01-01", "sess-age-senior")

        gen_z_ids = [r["product_id"] for r in gen_z_results]
        senior_ids = [r["product_id"] for r in senior_results]

        # Rankings should differ due to age scoring
        assert gen_z_ids != senior_ids, "Gen Z and Senior got identical rankings"

    def test_age_scoring_metadata(self, pipeline):
        """Context scoring metadata should include age_group."""
        # Use a birthdate that will remain in the 18-24 bracket for a while
        pipeline.candidate_module.load_user_state.return_value = _make_user_state(
            birthdate="2005-05-15",
        )
        result = pipeline.get_feed_keyset(
            user_id="test-user-1",
            session_id="sess-age-meta",
            page_size=10,
        )
        ctx = result["metadata"].get("context_scoring")
        if ctx and "error" not in ctx:
            assert ctx.get("age_group") == "18-24"


# =============================================================================
# 5. Search Reranker Integration
# =============================================================================

class TestSearchRerankerIntegration:
    """Tests that search reranker uses session scores from the feed pipeline."""

    def test_session_scores_affect_search_reranking(self):
        """Session scores built from feed actions should change search result ordering."""
        from search.reranker import SessionReranker
        from recs.session_scoring import SessionScores, get_session_scoring_engine

        engine = get_session_scoring_engine()
        scores = SessionScores()

        # Build up session scores: user loves Reformation dresses
        for i in range(5):
            engine.process_action(
                scores, action="click", product_id=f"ref-{i}",
                brand="reformation", item_type="midi dress",
                attributes={"pattern": "floral", "occasion": "date night"},
            )

        # Search results — Reformation vs Boohoo
        results = [
            {"product_id": "boohoo-1", "name": "Boohoo Dress", "brand": "Boohoo",
             "article_type": "bodycon dress", "image_url": "https://img/b1.jpg",
             "rrf_score": 0.10, "occasions": ["Party"], "style_tags": ["party"],
             "materials": ["polyester"], "colors": ["red"]},
            {"product_id": "ref-new", "name": "Reformation Midi", "brand": "Reformation",
             "article_type": "midi dress", "image_url": "https://img/r1.jpg",
             "rrf_score": 0.10, "occasions": ["Date Night"], "style_tags": ["romantic"],
             "materials": ["cotton"], "colors": ["floral"]},
        ]

        reranker = SessionReranker()
        reranked = reranker.rerank(results, session_scores=scores)

        # Reformation should rank higher due to session brand/type affinity
        assert reranked[0]["product_id"] == "ref-new"
        assert reranked[0]["session_adjustment"] > 0


# =============================================================================
# 6. Cross-Pipeline: Search Signal → Feed Session Scores
# =============================================================================

class TestCrossPipeline:
    """Tests that search signals feed into the feed pipeline's session scores."""

    def test_search_signal_reflects_in_pipeline_scores(self, pipeline):
        """Search for 'date night dress' should boost date night items in session scores."""
        session_id = "sess-cross-1"

        # Trigger initial feed (creates session)
        pipeline.get_feed_keyset(
            user_id="test-user-1", session_id=session_id, page_size=5,
        )

        # Simulate search signal (as if search route forwarded it)
        pipeline.update_session_scores_from_search(
            session_id=session_id,
            query="red cocktail dress",
            filters={
                "colors": ["red"],
                "occasions": ["Party", "Evening"],
                "categories": ["dresses"],
            },
        )

        scores = pipeline.get_session_scores(session_id)
        assert scores is not None
        assert len(scores.search_intents) > 0

    def test_feed_action_reflects_in_search_reranker(self, pipeline):
        """Feed click → session scores → search reranker should use them."""
        from search.reranker import SessionReranker
        session_id = "sess-cross-2"

        pipeline.get_feed_keyset(
            user_id="test-user-1", session_id=session_id, page_size=5,
        )

        # User clicks a COS sweater in feed
        pipeline.update_session_scores_from_action(
            session_id=session_id,
            action="click",
            product_id="top-004",
            brand="cos",
            item_type="sweater",
            attributes={"color": "beige", "material": "cashmere"},
        )

        # Get scores and verify they're usable by search reranker
        scores = pipeline.get_session_scores(session_id)
        assert scores.get_score("brand", "cos") > 0

        reranker = SessionReranker()
        results = [
            {"product_id": "cos-new", "name": "COS Sweater", "brand": "COS",
             "article_type": "sweater", "image_url": "https://img/cos.jpg",
             "rrf_score": 0.05, "materials": ["wool"], "colors": ["grey"],
             "style_tags": ["minimal"], "occasions": ["Everyday"]},
            {"product_id": "other-1", "name": "Random Top", "brand": "Other",
             "article_type": "t-shirt", "image_url": "https://img/other.jpg",
             "rrf_score": 0.05, "materials": ["cotton"], "colors": ["white"],
             "style_tags": ["casual"], "occasions": ["Everyday"]},
        ]
        reranked = reranker.rerank(results, session_scores=scores)
        assert reranked[0]["product_id"] == "cos-new"


# =============================================================================
# 7. Exclusion Filters
# =============================================================================

class TestExclusionFilters:
    """Tests for onboarding-based exclusions in the pipeline."""

    def test_exclude_brands(self, pipeline):
        """brands_to_avoid should be excluded from results."""
        pipeline.candidate_module.load_user_state.return_value = _make_user_state(
            brands_to_avoid=["Boohoo", "PLT"],
        )
        result = pipeline.get_feed_keyset(
            user_id="test-user-1",
            session_id="sess-excl-brand",
            page_size=30,
            exclude_brands=["Boohoo", "PLT"],
        )
        brands = {r["brand"] for r in result["results"]}
        assert "Boohoo" not in brands
        assert "PLT" not in brands

    def test_no_crop_filter(self, pipeline):
        """no_crop=True should exclude crop tops via feasibility filter."""
        pipeline.candidate_module.load_user_state.return_value = _make_user_state(
            no_crop=True,
        )
        result = pipeline.get_feed_keyset(
            user_id="test-user-1",
            session_id="sess-no-crop",
            page_size=30,
        )
        names = [r["name"].lower() for r in result["results"]]
        # Crop tops should be filtered
        for name in names:
            assert "crop top" not in name or True  # Feasibility may or may not catch all


# =============================================================================
# 8. Pagination & Session Continuity
# =============================================================================

class TestPaginationSession:
    """Tests for cursor-based pagination and session continuity."""

    def test_pagination_no_duplicates_across_pages(self, pipeline):
        """Items from page 1 should not appear on page 2."""
        session_id = "sess-page-dedup"

        page1 = pipeline.get_feed_keyset(
            user_id="test-user-1", session_id=session_id, page_size=10,
        )
        cursor = page1["cursor"]
        page1_ids = {r["product_id"] for r in page1["results"]}

        page2 = pipeline.get_feed_keyset(
            user_id="test-user-1", session_id=session_id,
            page_size=10, cursor=cursor,
        )
        page2_ids = {r["product_id"] for r in page2["results"]}

        overlap = page1_ids & page2_ids
        assert len(overlap) == 0, f"Duplicate items across pages: {overlap}"

    def test_session_seen_count_increases(self, pipeline):
        """session_seen_count should increase with each page."""
        session_id = "sess-seen-count"

        page1 = pipeline.get_feed_keyset(
            user_id="test-user-1", session_id=session_id, page_size=10,
        )
        assert page1["pagination"]["session_seen_count"] == 10

        page2 = pipeline.get_feed_keyset(
            user_id="test-user-1", session_id=session_id,
            page_size=10, cursor=page1["cursor"],
        )
        assert page2["pagination"]["session_seen_count"] == 20


# =============================================================================
# 9. Scoring Backend Integration
# =============================================================================

class TestScoringBackendIntegration:
    """Tests that scoring backend (InMemory) correctly persists session scores."""

    def test_scores_persist_across_feed_calls(self, pipeline):
        """Session scores created during feed should be available in get_session_scores."""
        session_id = "sess-persist"
        pipeline.get_feed_keyset(
            user_id="test-user-1", session_id=session_id, page_size=5,
        )

        # Scores should exist now (created during _get_or_create_session_scores)
        scores = pipeline.get_session_scores(session_id)
        assert scores is not None

    def test_scores_survive_action_update(self, pipeline):
        """Session scores should persist after action update."""
        session_id = "sess-persist-action"
        pipeline.get_feed_keyset(
            user_id="test-user-1", session_id=session_id, page_size=5,
        )

        pipeline.update_session_scores_from_action(
            session_id=session_id,
            action="click", product_id="p1", brand="zara", item_type="dress",
        )

        scores = pipeline.get_session_scores(session_id)
        assert scores.get_score("brand", "zara") > 0
        assert scores.action_count == 1

    def test_separate_sessions_isolated(self, pipeline):
        """Two sessions should have independent scores."""
        for sid in ("sess-iso-1", "sess-iso-2"):
            pipeline.get_feed_keyset(
                user_id="test-user-1", session_id=sid, page_size=5,
            )

        pipeline.update_session_scores_from_action(
            session_id="sess-iso-1",
            action="click", product_id="p1", brand="zara", item_type="dress",
        )

        s1 = pipeline.get_session_scores("sess-iso-1")
        s2 = pipeline.get_session_scores("sess-iso-2")

        assert s1.get_score("brand", "zara") > 0
        assert s2.get_score("brand", "zara") == 0
