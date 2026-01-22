"""
Swipe Style Learning Server (v5 - Semantic)

FastAPI server with beautiful frontend for style preference learning.
Uses the SwipeEngine with taxonomy-based insights.

API Base URL: http://ecommerce.api.outrove.ai:8080/

Women's Fashion Endpoints:
- GET  /api/women/options         - Get available categories and attributes
- POST /api/women/session/start   - Start a new style learning session
- POST /api/women/session/choose  - Record user's choice (1 of 4)
- POST /api/women/session/skip    - Skip all 4 items
- GET  /api/women/session/{user_id}/summary - Get learned preferences
- GET  /api/women/feed/{user_id}  - Get personalized feed based on preferences
- GET  /women-images/{path}       - Serve women's product images
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import uvicorn
import numpy as np

from engines import (
    SwipeEngine, UserPreferences, SwipeAction,
    FourChoiceEngine, FourChoicePreferences,
    RankingEngine, RankingPreferences,
    AttributeTestEngine, AttributeTestPreferences, ATTRIBUTE_TEST_PHASES,
    PredictiveFourEngine, PredictivePreferences
)
from gender_config import (
    get_config, get_category_order, MENS_CONFIG, WOMENS_CONFIG
)

# API metadata
API_TITLE = "Outrove Fashion Style Learning API"
API_DESCRIPTION = """
## Fashion Style Learning API

Learn user fashion preferences through a 4-choice swipe interface.

### How It Works
1. **Start Session** - Initialize with optional category/color preferences
2. **Choose Items** - User picks favorite from 4 items shown
3. **System Learns** - Bayesian preference learning on attributes
4. **Get Feed** - Personalized recommendations based on learned taste

### Women's Fashion Categories
- `tops_knitwear` - Sweaters, cardigans, knit tops
- `tops_woven` - Blouses, shirts, woven tops
- `tops_sleeveless` - Tank tops, camisoles
- `tops_special` - Bodysuits, special tops
- `dresses` - All dress types
- `bottoms_trousers` - Pants, jeans, trousers
- `bottoms_skorts` - Skirts, shorts
- `outerwear` - Jackets, coats
- `sportswear` - Athletic wear, leggings

### Attribute Dimensions Learned
| Attribute | Example Values |
|-----------|---------------|
| pattern | solid, striped, floral, geometric |
| style | casual, office, evening, bohemian |
| color_family | neutral, bright, cool, pastel, dark |
| fit_vibe | fitted, relaxed, oversized, cropped |
| neckline | crew, v_neck, off_shoulder, sweetheart |
| occasion | everyday, work, date_night, party |
| sleeve_type | sleeveless, short, long, puff |

### Base URL
`http://ecommerce.api.outrove.ai:8080/`
"""

tags_metadata = [
    {
        "name": "Women's Fashion",
        "description": "Style learning endpoints for women's fashion. Start a session, make choices, get personalized recommendations.",
    },
    {
        "name": "Men's Fashion",
        "description": "Style learning endpoints for men's fashion (T-shirts).",
    },
    {
        "name": "Unified",
        "description": "Gender-aware endpoints that work for both men's and women's fashion.",
    },
]

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version="5.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
# Note: allow_credentials=True requires explicit origins (not "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ecommerce.outrove.ai",
        "http://ecommerce.outrove.ai",
        "https://ecommerce.api.outrove.ai",
        "http://ecommerce.api.outrove.ai",
        "https://outrove.ai",
        "http://outrove.ai",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Mount static files for images - Men's fashion
IMAGES_DIR = Path("/home/ubuntu/recSys/outfitTransformer/HPImages")
if IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

# Mount static files for images - Women's fashion
WOMEN_IMAGES_DIR = Path("/home/ubuntu/recSys/outfitTransformer/data/women_fashion/images_webp")
if WOMEN_IMAGES_DIR.exists():
    app.mount("/women-images", StaticFiles(directory=str(WOMEN_IMAGES_DIR)), name="women-images")

# Mount redesign assets
REDESIGN_DIR = Path("/home/ubuntu/recSys/outfitTransformer/interface/redesign")
if REDESIGN_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(REDESIGN_DIR)), name="assets")

# Initialize engines - Men's (default)
engine = SwipeEngine()
four_engine = FourChoiceEngine()
ranking_engine = RankingEngine()
attribute_engine = AttributeTestEngine()
predictive_engine = PredictiveFourEngine()  # Predictive category-focused engine (men's)

# Integrate Supabase recommendation endpoints
try:
    from recs.api_endpoints import integrate_with_app
    integrate_with_app(app)
except Exception as e:
    print(f"Warning: Could not load recommendation endpoints: {e}")

# Engine registry for gender-aware unified API
# Engines are created on-demand to save memory when not used
_unified_engines: Dict[str, PredictiveFourEngine] = {}


def get_unified_engine(gender: str) -> PredictiveFourEngine:
    """Get or create a PredictiveFourEngine for the specified gender."""
    gender_key = "female" if gender.lower() in ("female", "women", "woman", "f", "w") else "male"

    if gender_key not in _unified_engines:
        print(f"Creating unified engine for gender: {gender_key}")
        _unified_engines[gender_key] = PredictiveFourEngine(gender=gender_key)

    return _unified_engines[gender_key]


def get_image_url_for_gender(item_id: str, gender: str) -> str:
    """Generate image URL based on gender."""
    gender_key = "female" if gender.lower() in ("female", "women", "woman", "f", "w") else "male"

    if gender_key == "female":
        # Women's images are organized as: /women-images/{category}/{subcategory}/{id}.webp
        # The item_id format is: {category}/{subcategory}/{image_index}
        return f"/women-images/{item_id.replace(' ', '%20')}.webp"
    else:
        # Men's images: /images/{item_id}.webp
        return f"/images/{item_id.replace(' ', '%20')}.webp"

# Session storage (in production, use Redis/database)
sessions: Dict[str, UserPreferences] = {}
current_items: Dict[str, str] = {}
preloaded_items: Dict[str, str] = {}  # Store next item for preloading

# Four-choice session storage (OLD - kept for backwards compatibility)
four_sessions: Dict[str, FourChoicePreferences] = {}
four_current_items: Dict[str, List[str]] = {}  # Store current 4 items

# Predictive four-choice session storage (NEW - category-focused)
predictive_sessions: Dict[str, PredictivePreferences] = {}
predictive_current_items: Dict[str, List[str]] = {}
predictive_test_info: Dict[str, Dict] = {}

# Ranking session storage
ranking_sessions: Dict[str, RankingPreferences] = {}
ranking_current_items: Dict[str, List[str]] = {}  # Store current items to rank

# Attribute test session storage
attr_sessions: Dict[str, AttributeTestPreferences] = {}
attr_current_items: Dict[str, List[str]] = {}
attr_test_info: Dict[str, Dict] = {}  # Store current test metadata

# Unified gender-aware session storage
# Keys are "{user_id}:{gender}" for isolation
unified_sessions: Dict[str, PredictivePreferences] = {}
unified_current_items: Dict[str, List[str]] = {}
unified_test_info: Dict[str, Dict] = {}


def get_unified_session_key(user_id: str, gender: str) -> str:
    """Generate session key combining user_id and gender."""
    gender_key = "female" if gender.lower() in ("female", "women", "woman", "f", "w") else "male"
    return f"{user_id}:{gender_key}"


def convert_numpy(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Convert both keys and values
        return {convert_numpy(k): convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(v) for v in obj)
    elif isinstance(obj, set):
        return list(convert_numpy(v) for v in obj)
    return obj


class StartSessionRequest(BaseModel):
    user_id: str = "default_user"
    colors_to_avoid: List[str] = []
    materials_to_avoid: List[str] = []  # Fabrics to avoid: polyester, wool, etc.
    selected_styles: List[str] = []  # Style selections from setup: plain, graphics, smalllogos, athletic, pocket
    selected_necklines: List[str] = []  # Neckline selections: crew, vneck, henley
    selected_sleeves: List[str] = []  # Sleeve selections: short, long, nosleeve


# Mapping from frontend style names to category names in metadata
STYLE_TO_CATEGORY = {
    'plain': 'Plain T-shirts',
    'graphics': 'Graphics T-shirts',
    'smalllogos': 'Small logos',
    'athletic': 'Athletic',
    'pocket': 'Plain T-shirts',  # Pocket tees are subset of plain, use same category
}

# Mapping from frontend neckline names to neckline values in metadata
NECKLINE_TO_METADATA = {
    'crew': 'crewneck',
    'vneck': 'v neck',
    'henley': 'henley',
}

# Mapping from frontend sleeve names to sleeve values in metadata
SLEEVE_TO_METADATA = {
    'short': 'short sleeve',
    'long': 'long sleeve',
    'nosleeve': 'no sleeve',
}

# Mapping from frontend material names to fabric values in metadata (for avoidance)
MATERIAL_TO_METADATA = {
    'polyester': 'polyester',
    'wool': 'wool',
    'linen': 'linen',
    'silk': 'silk',
    'synthetics': 'synthetic',
    'leather': 'leather',
}


class SwipeRequest(BaseModel):
    user_id: str = "default_user"
    action: str  # "like", "dislike", "skip"


@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the main frontend."""
    return get_html_template()


@app.post("/api/start")
async def start_session(request: StartSessionRequest):
    """Start a new swipe session."""
    user_id = request.user_id

    # Map selected styles to category names
    selected_categories = set()
    for style in request.selected_styles:
        style_lower = style.lower().strip()
        if style_lower in STYLE_TO_CATEGORY:
            selected_categories.add(STYLE_TO_CATEGORY[style_lower])

    # Map selected necklines to metadata values
    selected_necklines = set()
    for neckline in request.selected_necklines:
        neckline_lower = neckline.lower().strip()
        if neckline_lower in NECKLINE_TO_METADATA:
            selected_necklines.add(NECKLINE_TO_METADATA[neckline_lower])

    # Map selected sleeves to metadata values
    selected_sleeves = set()
    for sleeve in request.selected_sleeves:
        sleeve_lower = sleeve.lower().strip()
        if sleeve_lower in SLEEVE_TO_METADATA:
            selected_sleeves.add(SLEEVE_TO_METADATA[sleeve_lower])

    # Map materials to avoid to metadata values
    materials_to_avoid = set()
    for material in request.materials_to_avoid:
        material_lower = material.lower().strip()
        if material_lower in MATERIAL_TO_METADATA:
            materials_to_avoid.add(MATERIAL_TO_METADATA[material_lower])

    # Log filtering info
    if selected_categories:
        print(f"[Session {user_id}] Prefiltering to categories: {selected_categories}")
    else:
        print(f"[Session {user_id}] No style filter - showing all categories")

    if selected_necklines:
        print(f"[Session {user_id}] Prefiltering to necklines: {selected_necklines}")

    if selected_sleeves:
        print(f"[Session {user_id}] Prefiltering to sleeves: {selected_sleeves}")

    if materials_to_avoid:
        print(f"[Session {user_id}] Avoiding materials: {materials_to_avoid}")

    # Create new preferences
    prefs = UserPreferences(
        user_id=user_id,
        colors_to_avoid=set(c.lower().strip() for c in request.colors_to_avoid if c),
        materials_to_avoid=materials_to_avoid,
        selected_categories=selected_categories,
        selected_necklines=selected_necklines,
        selected_sleeves=selected_sleeves
    )

    sessions[user_id] = prefs

    # Get candidate count for stats
    all_candidates = engine.get_candidates(prefs)
    print(f"[Session {user_id}] {len(all_candidates)} items match filters (out of {len(engine.item_ids)} total)")

    # Get first item
    result = engine.get_next_item(prefs)
    item_id, session_complete = result

    if not item_id or session_complete:
        return {"status": "no_items", "message": "No items available", "filter_stats": {
            "total_items": len(engine.item_ids),
            "filtered_items": len(all_candidates),
            "selected_categories": list(selected_categories) if selected_categories else ["all"]
        }}

    current_items[user_id] = item_id
    item_info = engine.get_item_info(item_id)

    # Preload second item (peek without recording)
    # Temporarily add to seen to get a different item
    prefs.skipped_ids.append(item_id)
    next_result = engine.get_next_item(prefs)
    prefs.skipped_ids.pop()  # Remove the temp skip

    next_item_id, _ = next_result
    preloaded_items[user_id] = next_item_id

    next_item_response = None
    if next_item_id:
        next_item_info = engine.get_item_info(next_item_id)
        next_item_response = format_item_response(next_item_id, next_item_info)

    summary = engine.get_preference_summary(prefs)

    return convert_numpy({
        "status": "started",
        "item": format_item_response(item_id, item_info),
        "next_item": next_item_response,  # Preloaded for smooth animation
        "stats": summary,
        "session_complete": False,
        "filter_stats": {
            "total_items": len(engine.item_ids),
            "filtered_items": len(all_candidates),
            "selected_categories": list(selected_categories) if selected_categories else ["all"],
            "selected_necklines": list(selected_necklines) if selected_necklines else ["all"]
        }
    })


@app.post("/api/swipe")
async def record_swipe(request: SwipeRequest):
    """Record a swipe and get next item."""
    user_id = request.user_id

    if user_id not in sessions:
        raise HTTPException(status_code=400, detail="No active session. Call /api/start first.")

    prefs = sessions[user_id]
    item_id = current_items.get(user_id)

    if not item_id:
        raise HTTPException(status_code=400, detail="No current item")

    # Map action string to enum
    action_map = {
        "like": SwipeAction.LIKE,
        "dislike": SwipeAction.DISLIKE,
        "skip": SwipeAction.SKIP
    }
    action = action_map.get(request.action.lower(), SwipeAction.SKIP)

    # Record the swipe
    prefs = engine.record_swipe(prefs, item_id, action)
    sessions[user_id] = prefs

    # Use preloaded item as the next item
    next_item_id = preloaded_items.get(user_id)

    # Check if session should end
    summary = engine.get_preference_summary(prefs)
    if summary.get('session_complete') or not next_item_id:
        current_items[user_id] = None
        preloaded_items[user_id] = None
        return convert_numpy({
            "status": "complete",
            "item": None,
            "next_item": None,
            "stats": summary,
            "session_complete": True
        })

    # Update current item to the preloaded one
    current_items[user_id] = next_item_id
    item_info = engine.get_item_info(next_item_id)

    # Preload the NEXT item (for smooth animation)
    # IMPORTANT: Temporarily add current item to seen to prevent duplicates
    prefs.skipped_ids.append(next_item_id)
    future_result = engine.get_next_item(prefs)
    prefs.skipped_ids.pop()  # Remove temp skip - will be properly recorded on next swipe

    future_item_id, session_complete = future_result

    # Store for next swipe
    preloaded_items[user_id] = future_item_id if not session_complete else None

    future_item_response = None
    if future_item_id and not session_complete:
        future_item_info = engine.get_item_info(future_item_id)
        future_item_response = format_item_response(future_item_id, future_item_info)

    return convert_numpy({
        "status": "continue",
        "item": format_item_response(next_item_id, item_info),
        "next_item": future_item_response,  # Preloaded for back card
        "stats": summary,
        "session_complete": False
    })


@app.get("/api/stats/{user_id}")
async def get_stats(user_id: str):
    """Get current preference stats."""
    if user_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    prefs = sessions[user_id]
    summary = engine.get_preference_summary(prefs)
    return convert_numpy(summary)


# ============================================
# FOUR-CHOICE API ENDPOINTS
# ============================================

class FourChoiceStartRequest(BaseModel):
    user_id: str = "default_user"
    colors_to_avoid: List[str] = []
    materials_to_avoid: List[str] = []
    selected_styles: List[str] = []
    selected_necklines: List[str] = []
    selected_sleeves: List[str] = []


class FourChoiceRequest(BaseModel):
    user_id: str = "default_user"
    winner_id: str  # The item user chose


# === Unified Gender-Aware Request Models ===

class UnifiedStartRequest(BaseModel):
    """Request model for unified gender-aware session start."""
    user_id: str = "default_user"
    gender: str = "male"  # "male", "female", "men", "women", etc.
    colors_to_avoid: List[str] = []
    materials_to_avoid: List[str] = []
    selected_categories: List[str] = []  # Category names from gender config


class UnifiedChoiceRequest(BaseModel):
    """Request model for unified gender-aware choice recording."""
    user_id: str = "default_user"
    gender: str = "male"
    winner_id: str


class UnifiedSkipRequest(BaseModel):
    """Request model for unified gender-aware skip recording."""
    user_id: str = "default_user"
    gender: str = "male"


# === Unified Gender-Aware API Endpoints ===

@app.get("/api/unified/options/{gender}")
async def get_unified_options(gender: str):
    """Get available categories and options for the specified gender."""
    config = get_config(gender)

    return convert_numpy({
        "gender": "female" if gender.lower() in ("female", "women", "woman", "f", "w") else "male",
        "categories": config['category_order'],
        "attribute_prompts": {
            attr: [label for label, _ in prompts]
            for attr, prompts in config['attribute_prompts'].items()
        },
        "attribute_weights": config['attribute_weights']
    })


@app.post("/api/unified/four/start")
async def start_unified_session(request: UnifiedStartRequest):
    """Start a new gender-aware four-choice session."""
    user_id = request.user_id
    gender = request.gender
    session_key = get_unified_session_key(user_id, gender)

    # Get gender-specific engine
    engine = get_unified_engine(gender)
    gender_key = "female" if gender.lower() in ("female", "women", "woman", "f", "w") else "male"

    # Create preferences
    prefs = PredictivePreferences(
        user_id=user_id,
        gender=gender_key,
        colors_to_avoid=set(c.lower().strip() for c in request.colors_to_avoid if c),
        materials_to_avoid=set(m.lower().strip() for m in request.materials_to_avoid if m),
    )

    # Initialize session with category order
    selected_cats = request.selected_categories if request.selected_categories else None
    prefs = engine.initialize_session(prefs, selected_cats)
    unified_sessions[session_key] = prefs

    # Get first 4 items
    four_items, test_info = engine.get_four_items(prefs)

    if len(four_items) < 4:
        return {"status": "no_items", "message": "Not enough items available for this gender"}

    unified_current_items[session_key] = four_items
    unified_test_info[session_key] = test_info

    # Format items for response
    items_response = []
    for item_id in four_items:
        info = engine.get_item_info(item_id)
        items_response.append({
            "id": item_id,
            "image_url": get_image_url_for_gender(item_id, gender_key),
            "category": info.get("category", ""),
            "brand": info.get("brand", ""),
            "color": info.get("color", ""),
            "cluster": engine.item_to_cluster.get(item_id),
        })

    print(f"[Unified {gender_key}:{user_id}] Started - Category: {test_info.get('category')}")

    return convert_numpy({
        "status": "started",
        "gender": gender_key,
        "items": items_response,
        "round": 1,
        "test_info": {
            "category": test_info.get('category'),
            "category_index": test_info.get('category_index'),
            "total_categories": test_info.get('total_categories'),
            "round_in_category": test_info.get('round_in_category'),
            "clusters_shown": test_info.get('clusters_shown'),
            "prediction": test_info.get('prediction', {}),
            "categories_completed": test_info.get('categories_completed', []),
        },
        "session_complete": False
    })


@app.post("/api/unified/four/choose")
async def record_unified_choice(request: UnifiedChoiceRequest):
    """Record user's choice in a unified gender-aware session."""
    user_id = request.user_id
    gender = request.gender
    session_key = get_unified_session_key(user_id, gender)
    gender_key = "female" if gender.lower() in ("female", "women", "woman", "f", "w") else "male"

    if session_key not in unified_sessions:
        raise HTTPException(
            status_code=400,
            detail=f"No active session for {gender_key}. Call /api/unified/four/start first."
        )

    prefs = unified_sessions[session_key]
    current_four = unified_current_items.get(session_key, [])
    engine = get_unified_engine(gender)

    if not current_four:
        raise HTTPException(status_code=400, detail="No current items")

    if request.winner_id not in current_four:
        raise HTTPException(status_code=400, detail="Winner must be one of the shown items")

    # Record the choice
    prefs, result_info = engine.record_choice(prefs, request.winner_id)
    unified_sessions[session_key] = prefs

    # Log prediction result
    prediction = unified_test_info.get(session_key, {}).get('prediction', {})
    if result_info.get('had_prediction'):
        status = "CORRECT" if result_info['prediction_correct'] else "WRONG"
        print(f"[Unified {gender_key}:{user_id}] Prediction {status}")

    # Check if all categories complete
    if result_info.get('all_complete') or engine.is_session_complete(prefs):
        summary = engine.get_session_summary(prefs)
        feed_preview = engine.get_feed_preview(prefs, items_per_category=8)
        unified_current_items[session_key] = []
        return convert_numpy({
            "status": "complete",
            "gender": gender_key,
            "items": [],
            "round": prefs.rounds_completed,
            "stats": summary,
            "result_info": result_info,
            "feed_preview": feed_preview,
            "session_complete": True
        })

    # Get next 4 items
    four_items, test_info = engine.get_four_items(prefs)

    if len(four_items) < 4:
        summary = engine.get_session_summary(prefs)
        feed_preview = engine.get_feed_preview(prefs, items_per_category=8)
        return convert_numpy({
            "status": "complete",
            "gender": gender_key,
            "items": [],
            "round": prefs.rounds_completed,
            "stats": summary,
            "result_info": result_info,
            "feed_preview": feed_preview,
            "session_complete": True
        })

    unified_current_items[session_key] = four_items
    unified_test_info[session_key] = test_info

    # Format items for response
    items_response = []
    for item_id in four_items:
        info = engine.get_item_info(item_id)
        items_response.append({
            "id": item_id,
            "image_url": get_image_url_for_gender(item_id, gender_key),
            "category": info.get("category", ""),
            "brand": info.get("brand", ""),
            "color": info.get("color", ""),
            "cluster": engine.item_to_cluster.get(item_id),
        })

    summary = engine.get_session_summary(prefs)

    return convert_numpy({
        "status": "ok",
        "gender": gender_key,
        "items": items_response,
        "round": prefs.rounds_completed,
        "test_info": {
            "category": test_info.get('category'),
            "category_index": test_info.get('category_index'),
            "total_categories": test_info.get('total_categories'),
            "round_in_category": test_info.get('round_in_category'),
            "clusters_shown": test_info.get('clusters_shown'),
            "prediction": test_info.get('prediction', {}),
            "categories_completed": test_info.get('categories_completed', []),
        },
        "result_info": result_info,
        "stats": summary,
        "session_complete": False
    })


@app.post("/api/unified/four/skip")
async def record_unified_skip(request: UnifiedSkipRequest):
    """Record when user skips all 4 items in unified session."""
    user_id = request.user_id
    gender = request.gender
    session_key = get_unified_session_key(user_id, gender)
    gender_key = "female" if gender.lower() in ("female", "women", "woman", "f", "w") else "male"

    if session_key not in unified_sessions:
        raise HTTPException(status_code=400, detail="No active session")

    prefs = unified_sessions[session_key]
    engine = get_unified_engine(gender)

    # Record skip
    prefs = engine.record_skip_all(prefs)
    unified_sessions[session_key] = prefs

    # Get next items
    four_items, test_info = engine.get_four_items(prefs)

    if len(four_items) < 4 or engine.is_session_complete(prefs):
        summary = engine.get_session_summary(prefs)
        feed_preview = engine.get_feed_preview(prefs, items_per_category=8)
        return convert_numpy({
            "status": "complete",
            "gender": gender_key,
            "items": [],
            "round": prefs.rounds_completed,
            "stats": summary,
            "feed_preview": feed_preview,
            "session_complete": True
        })

    unified_current_items[session_key] = four_items
    unified_test_info[session_key] = test_info

    items_response = []
    for item_id in four_items:
        info = engine.get_item_info(item_id)
        items_response.append({
            "id": item_id,
            "image_url": get_image_url_for_gender(item_id, gender_key),
            "category": info.get("category", ""),
            "brand": info.get("brand", ""),
            "color": info.get("color", ""),
            "cluster": engine.item_to_cluster.get(item_id),
        })

    return convert_numpy({
        "status": "ok",
        "gender": gender_key,
        "items": items_response,
        "round": prefs.rounds_completed,
        "test_info": {
            "category": test_info.get('category'),
            "category_index": test_info.get('category_index'),
            "total_categories": test_info.get('total_categories'),
            "round_in_category": test_info.get('round_in_category'),
        },
        "session_complete": False
    })


@app.get("/api/unified/four/summary/{gender}/{user_id}")
async def get_unified_summary(gender: str, user_id: str):
    """Get session summary for a unified session."""
    session_key = get_unified_session_key(user_id, gender)
    gender_key = "female" if gender.lower() in ("female", "women", "woman", "f", "w") else "male"

    if session_key not in unified_sessions:
        raise HTTPException(status_code=404, detail="No session found")

    prefs = unified_sessions[session_key]
    engine = get_unified_engine(gender)

    summary = engine.get_session_summary(prefs)
    feed_preview = engine.get_feed_preview(prefs, items_per_category=8)

    return convert_numpy({
        "gender": gender_key,
        "user_id": user_id,
        "summary": summary,
        "feed_preview": feed_preview
    })


# =============================================================================
# WOMEN'S FASHION API ENDPOINTS
# Base URL: http://ecommerce.api.outrove.ai:8080/api/women/
# =============================================================================

# --- Request/Response Models for Women's API ---

class WomenSessionStartRequest(BaseModel):
    """Request body for starting a women's style learning session."""
    user_id: str = Field(
        default="default_user",
        description="Unique identifier for the user",
        example="user_12345"
    )
    colors_to_avoid: List[str] = Field(
        default=[],
        description="List of colors to exclude from recommendations",
        example=["pink", "yellow", "orange"]
    )
    materials_to_avoid: List[str] = Field(
        default=[],
        description="List of materials to exclude",
        example=["polyester", "silk"]
    )
    selected_categories: List[str] = Field(
        default=[],
        description="Categories to focus on (empty = all categories). Options: tops_knitwear, tops_woven, tops_sleeveless, tops_special, dresses, bottoms_trousers, bottoms_skorts, outerwear, sportswear",
        example=["tops_knitwear", "dresses", "bottoms_trousers"]
    )

class WomenChoiceRequest(BaseModel):
    """Request body for recording a user's choice."""
    user_id: str = Field(
        description="User ID from session start",
        example="user_12345"
    )
    winner_id: str = Field(
        description="ID of the chosen item (one of the 4 shown items)",
        example="tops_knitwear/sweaters/123"
    )

class WomenSkipRequest(BaseModel):
    """Request body for skipping all 4 items."""
    user_id: str = Field(
        description="User ID from session start",
        example="user_12345"
    )

class ItemResponse(BaseModel):
    """Response model for a fashion item."""
    id: str = Field(description="Unique item identifier", example="tops_knitwear/sweaters/123")
    image_url: str = Field(description="URL to item image", example="/women-images/tops_knitwear/sweaters/123.webp")
    category: str = Field(description="Item category", example="tops_knitwear")
    brand: str = Field(description="Brand name", example="")
    color: str = Field(description="Primary color", example="")
    cluster: int = Field(description="Visual cluster ID (0-11)", example=5)

class PredictionInfo(BaseModel):
    """Information about the system's prediction."""
    predicted_cluster: Optional[int] = Field(description="Predicted winning cluster")
    confidence: float = Field(description="Prediction confidence (0-1)")
    has_prediction: bool = Field(description="Whether a prediction was made")

class TestInfo(BaseModel):
    """Information about current test state."""
    category: str = Field(description="Current category being tested", example="tops_knitwear")
    category_index: int = Field(description="Current category index (1-based)", example=1)
    total_categories: int = Field(description="Total number of categories", example=9)
    round_in_category: int = Field(description="Round number within current category", example=1)
    clusters_shown: List[int] = Field(description="Cluster IDs of the 4 items shown")
    prediction: Optional[Dict] = Field(description="System's prediction for this round")
    categories_completed: List[str] = Field(description="List of completed categories")

# --- Session storage for women's API ---
women_sessions: Dict[str, PredictivePreferences] = {}
women_current_items: Dict[str, List[str]] = {}
women_test_info: Dict[str, Dict] = {}


@app.get(
    "/api/women/options",
    tags=["Women's Fashion"],
    summary="Get available options for women's fashion",
    description="""
    Returns all available categories, attribute values, and their weights for women's fashion.

    Use this to build filter UIs and understand what the system can learn.

    **Categories:** 9 categories from tops to outerwear
    **Attributes:** 7 attribute dimensions (pattern, style, color, etc.)
    """,
    response_description="Available options for women's fashion"
)
async def get_women_options():
    """Get available categories and attributes for women's fashion."""
    config = WOMENS_CONFIG

    return {
        "gender": "female",
        "categories": [
            {"id": "tops_knitwear", "label": "Sweaters & Knits", "description": "Sweaters, cardigans, knit tops"},
            {"id": "tops_woven", "label": "Blouses & Shirts", "description": "Woven blouses, shirts, button-ups"},
            {"id": "tops_sleeveless", "label": "Tank Tops & Camis", "description": "Sleeveless tops, camisoles"},
            {"id": "tops_special", "label": "Bodysuits", "description": "Bodysuits and special tops"},
            {"id": "dresses", "label": "Dresses", "description": "All dress styles"},
            {"id": "bottoms_trousers", "label": "Pants & Trousers", "description": "Pants, jeans, trousers"},
            {"id": "bottoms_skorts", "label": "Skirts & Shorts", "description": "Skirts, shorts, skorts"},
            {"id": "outerwear", "label": "Outerwear", "description": "Jackets, coats, blazers"},
            {"id": "sportswear", "label": "Sportswear", "description": "Athletic wear, leggings"},
        ],
        "attributes": {
            "pattern": {
                "weight": 0.25,
                "values": ["solid", "striped", "floral", "geometric", "animal_print", "plaid", "polka_dots", "lace"]
            },
            "style": {
                "weight": 0.20,
                "values": ["casual", "office", "evening", "bohemian", "minimalist", "romantic", "athletic"]
            },
            "color_family": {
                "weight": 0.15,
                "values": ["neutral", "bright", "cool", "pastel", "dark"]
            },
            "fit_vibe": {
                "weight": 0.15,
                "values": ["fitted", "relaxed", "oversized", "cropped", "flowy"]
            },
            "neckline": {
                "weight": 0.10,
                "values": ["crew", "v_neck", "scoop", "off_shoulder", "sweetheart", "halter", "square", "turtleneck", "cowl"]
            },
            "occasion": {
                "weight": 0.10,
                "values": ["everyday", "work", "date_night", "party", "beach"]
            },
            "sleeve_type": {
                "weight": 0.05,
                "values": ["sleeveless", "short_sleeve", "long_sleeve", "puff_sleeve", "bell_sleeve", "flutter_sleeve"]
            }
        },
        "colors_available": ["black", "white", "gray", "navy", "blue", "red", "green", "brown", "pink", "yellow", "orange", "purple", "cream", "beige"],
        "total_items": len(list(WOMEN_IMAGES_DIR.rglob("*.webp"))),
        "image_base_url": "http://ecommerce.api.outrove.ai:8080/women-images/"
    }


# Mapping from broad onboarding categories to specific Tinder test categories
ONBOARDING_TO_TINDER_CATEGORIES = {
    "tops": ["tops_knitwear", "tops_woven", "tops_sleeveless", "tops_special"],
    "bottoms": ["bottoms_trousers", "bottoms_skorts"],
    "dresses": ["dresses"],
    "skirts": ["bottoms_skorts"],
    "outerwear": ["outerwear"],
    "one-piece": ["dresses"],
    "sportswear": ["sportswear"],
}


def save_tinder_results_to_supabase(
    user_id: str,
    prefs: PredictivePreferences,
    summary: dict
) -> bool:
    """Save tinder test results (taste_vector) to Supabase for use in feed.

    This enables the recommendation pipeline to use the learned taste_vector
    for personalized pgvector similarity search.
    """
    import os
    from supabase import create_client

    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            print(f"[SaveTinder] Missing Supabase credentials")
            return False

        sb = create_client(url, key)

        # Extract taste_vector from prefs
        taste_vector = None
        if prefs.taste_vector is not None:
            # Convert numpy array to list for JSON serialization
            taste_vector = prefs.taste_vector.tolist() if hasattr(prefs.taste_vector, 'tolist') else list(prefs.taste_vector)

        if not taste_vector or len(taste_vector) != 512:
            print(f"[SaveTinder] Invalid taste_vector: len={len(taste_vector) if taste_vector else 0}")
            return False

        # Determine if user_id is UUID or anon_id
        user_uuid = None
        anon_id = None
        if user_id and len(user_id) == 36 and '-' in user_id:
            user_uuid = user_id
        else:
            anon_id = user_id

        # Save to user_seed_preferences table via existing RPC
        result = sb.rpc('save_tinder_preferences', {
            'p_user_id': user_uuid,
            'p_anon_id': anon_id,
            'p_taste_vector': taste_vector,
            'p_rounds_completed': prefs.rounds_completed,
            'p_categories_tested': list(prefs.completed_categories),
            'p_attribute_preferences': summary.get('attribute_preferences', {}),
            'p_prediction_accuracy': summary.get('prediction_accuracy', 0),
            'p_gender': prefs.gender or 'female'
        }).execute()

        print(f"[SaveTinder] Saved taste_vector for {user_id}: {len(taste_vector)} dims, {prefs.rounds_completed} rounds")
        return True

    except Exception as e:
        print(f"[SaveTinder] Error saving to Supabase: {e}")
        # Try fallback direct insert
        try:
            # Fallback: direct upsert to user_seed_preferences
            data = {
                'anon_id': anon_id or user_uuid,
                'taste_vector': taste_vector,
                'rounds_completed': prefs.rounds_completed,
                'categories_tested': list(prefs.completed_categories),
                'gender': prefs.gender or 'female',
            }
            if user_uuid:
                data['user_id'] = user_uuid

            sb.table('user_seed_preferences').upsert(
                data,
                on_conflict='anon_id'
            ).execute()
            print(f"[SaveTinder] Fallback save succeeded for {user_id}")
            return True
        except Exception as e2:
            print(f"[SaveTinder] Fallback also failed: {e2}")
            return False


def get_user_onboarding_preferences(user_id: str) -> dict:
    """Fetch user's full onboarding preferences from Supabase.

    Returns a dict with all onboarding fields:
    - categories: list of broad categories
    - colors_to_avoid: list of colors to exclude
    - materials_to_avoid: list of fabrics to exclude
    - preferred_brands: list of preferred brands
    - brands_to_avoid: list of brands to exclude
    - tops_prefs: dict with fits, sleeves, necklines, types for tops
    - bottoms_prefs: dict with fits, rise, lengths, types for bottoms
    - dresses_prefs: dict with fits, sleeves, lengths, types for dresses
    - skirts_prefs: dict with fits, lengths, types for skirts
    - one_piece_prefs: dict with fits, sleeves, types for one-pieces
    - outerwear_prefs: dict with fits, types for outerwear
    """
    import os
    from supabase import create_client

    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            return None

        sb = create_client(url, key)
        result = sb.table('user_onboarding_profiles').select(
            'categories, colors_to_avoid, materials_to_avoid, '
            'preferred_brands, brands_to_avoid, '
            'tops_prefs, bottoms_prefs, dresses_prefs, skirts_prefs, '
            'one_piece_prefs, outerwear_prefs'
        ).eq('anon_id', user_id).execute()

        if result.data and len(result.data) > 0:
            return result.data[0]
    except Exception as e:
        print(f"[Session] Error fetching onboarding: {e}")

    return None


def get_user_onboarding_categories(user_id: str) -> tuple:
    """Fetch user's onboarding categories from Supabase.

    DEPRECATED: Use get_user_onboarding_preferences() for full data.
    Kept for backward compatibility.
    """
    onboarding = get_user_onboarding_preferences(user_id)
    if onboarding:
        return (
            onboarding.get('categories', []),
            onboarding.get('colors_to_avoid', []),
            onboarding.get('materials_to_avoid', [])
        )
    return None, None, None


def map_onboarding_to_tinder_categories(onboarding_categories: list) -> list:
    """Map broad onboarding categories to specific Tinder test categories."""
    if not onboarding_categories:
        return None

    tinder_cats = set()
    for cat in onboarding_categories:
        cat_lower = cat.lower().replace("-", "_").replace(" ", "_")
        if cat_lower in ONBOARDING_TO_TINDER_CATEGORIES:
            tinder_cats.update(ONBOARDING_TO_TINDER_CATEGORIES[cat_lower])

    return list(tinder_cats) if tinder_cats else None


@app.post(
    "/api/women/session/start",
    tags=["Women's Fashion"],
    summary="Start a new style learning session",
    description="""
    Initialize a new style learning session for women's fashion.

    **Flow:**
    1. Call this endpoint with user_id and optional preferences
    2. Receive 4 items to display to the user
    3. User picks their favorite → call `/api/women/session/choose`
    4. Repeat until session_complete=true
    5. Get personalized feed with `/api/women/feed/{user_id}`

    **Optional Filters:**
    - `colors_to_avoid`: Exclude specific colors
    - `materials_to_avoid`: Exclude specific materials
    - `selected_categories`: Focus on specific categories only

    **Auto-loading from Onboarding:**
    If no filters are provided, the system will automatically load the user's
    preferences from their onboarding profile (if one exists).
    """,
    response_description="Session started with 4 items to display"
)
async def start_women_session(request: WomenSessionStartRequest):
    """Start a new women's fashion style learning session."""
    user_id = request.user_id

    # Get women's engine
    engine = get_unified_engine("female")

    # Try to load user's onboarding preferences if not explicitly provided
    selected_cats = request.selected_categories
    colors_to_avoid = request.colors_to_avoid
    materials_to_avoid = request.materials_to_avoid
    brands_to_avoid = None
    preferred_brands = None
    onboarding_prefs = {}
    categories_source = "explicit"

    if not selected_cats:
        # Auto-load from full onboarding profile
        onboarding = get_user_onboarding_preferences(user_id)

        if onboarding:
            onboard_cats = onboarding.get('categories', [])
            if onboard_cats:
                # Map broad categories to Tinder categories
                selected_cats = map_onboarding_to_tinder_categories(onboard_cats)
                categories_source = "onboarding"
                print(f"[Session {user_id}] Auto-loaded categories from onboarding: {onboard_cats} -> {selected_cats}")

            # Also use onboarding colors/materials if not explicitly provided
            onboard_colors = onboarding.get('colors_to_avoid', [])
            if not colors_to_avoid and onboard_colors:
                colors_to_avoid = onboard_colors
                print(f"[Session {user_id}] Auto-loaded colors_to_avoid: {colors_to_avoid}")

            onboard_materials = onboarding.get('materials_to_avoid', [])
            if not materials_to_avoid and onboard_materials:
                materials_to_avoid = onboard_materials
                print(f"[Session {user_id}] Auto-loaded materials_to_avoid: {materials_to_avoid}")

            # NEW: Load brand preferences
            brands_to_avoid = onboarding.get('brands_to_avoid', [])
            if brands_to_avoid:
                print(f"[Session {user_id}] Auto-loaded brands_to_avoid: {brands_to_avoid}")

            preferred_brands = onboarding.get('preferred_brands', [])
            if preferred_brands:
                print(f"[Session {user_id}] Auto-loaded preferred_brands: {preferred_brands}")

            # NEW: Load per-category preferences for soft scoring in feed
            onboarding_prefs = {
                'tops': onboarding.get('tops_prefs', {}),
                'bottoms': onboarding.get('bottoms_prefs', {}),
                'dresses': onboarding.get('dresses_prefs', {}),
                'skirts': onboarding.get('skirts_prefs', {}),
                'one_piece': onboarding.get('one_piece_prefs', {}),
                'outerwear': onboarding.get('outerwear_prefs', {}),
            }
            # Filter out empty prefs
            onboarding_prefs = {k: v for k, v in onboarding_prefs.items() if v}
            if onboarding_prefs:
                print(f"[Session {user_id}] Auto-loaded per-category prefs for: {list(onboarding_prefs.keys())}")

    # Create preferences with all onboarding data
    prefs = PredictivePreferences(
        user_id=user_id,
        gender="female",
        colors_to_avoid=set(c.lower().strip() for c in (colors_to_avoid or []) if c),
        materials_to_avoid=set(m.lower().strip() for m in (materials_to_avoid or []) if m),
        brands_to_avoid=set(b.lower().strip() for b in (brands_to_avoid or []) if b),
        preferred_brands=set(b.lower().strip() for b in (preferred_brands or []) if b),
        onboarding_prefs=onboarding_prefs,
    )

    # Initialize session with selected categories
    prefs = engine.initialize_session(prefs, selected_cats)
    women_sessions[user_id] = prefs

    # Get first 4 items
    four_items, test_info = engine.get_four_items(prefs)

    if len(four_items) < 4:
        raise HTTPException(status_code=500, detail="Not enough items available. Try different categories.")

    women_current_items[user_id] = four_items
    women_test_info[user_id] = test_info

    # Format response
    items_response = []
    for item_id in four_items:
        info = engine.get_item_info(item_id)
        items_response.append({
            "id": item_id,
            "image_url": f"/women-images/{item_id.replace(' ', '%20')}.webp",
            "category": info.get("category", ""),
            "brand": info.get("brand", ""),
            "color": info.get("color", ""),
            "cluster": engine.item_to_cluster.get(item_id),
        })

    return convert_numpy({
        "status": "started",
        "user_id": user_id,
        "items": items_response,
        "round": 1,
        "test_info": {
            "category": test_info.get('category'),
            "category_label": {
                'tops_knitwear': 'Sweaters & Knits',
                'tops_woven': 'Blouses & Shirts',
                'tops_sleeveless': 'Tank Tops & Camis',
                'tops_special': 'Bodysuits',
                'dresses': 'Dresses',
                'bottoms_trousers': 'Pants & Trousers',
                'bottoms_skorts': 'Skirts & Shorts',
                'outerwear': 'Outerwear',
                'sportswear': 'Sportswear'
            }.get(test_info.get('category'), test_info.get('category')),
            "category_index": test_info.get('category_index'),
            "total_categories": test_info.get('total_categories'),
            "round_in_category": test_info.get('round_in_category'),
            "clusters_shown": test_info.get('clusters_shown'),
            "prediction": test_info.get('prediction', {}),
        },
        "session_complete": False,
        "message": "Pick your favorite from the 4 items shown"
    })


@app.post(
    "/api/women/session/choose",
    tags=["Women's Fashion"],
    summary="Record user's choice",
    description="""
    Record which item the user chose from the 4 displayed items.

    The system learns from this choice:
    - Winner's attributes get positive signal
    - Losers' attributes get negative signal
    - Taste vector is updated via contrastive learning

    **Response:** Next 4 items to display, or session_complete=true if done.
    """,
    response_description="Next items to display or completion status"
)
async def record_women_choice(request: WomenChoiceRequest):
    """Record user's choice and get next items."""
    user_id = request.user_id
    winner_id = request.winner_id

    if user_id not in women_sessions:
        raise HTTPException(status_code=400, detail="No active session. Call /api/women/session/start first.")

    prefs = women_sessions[user_id]
    current_four = women_current_items.get(user_id, [])

    if not current_four:
        raise HTTPException(status_code=400, detail="No current items in session")

    if winner_id not in current_four:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid winner_id. Must be one of: {current_four}"
        )

    engine = get_unified_engine("female")

    # Record choice
    prefs, result_info = engine.record_choice(prefs, winner_id)
    women_sessions[user_id] = prefs

    # Check if complete
    if result_info.get('all_complete') or engine.is_session_complete(prefs):
        summary = engine.get_session_summary(prefs)
        feed_preview = engine.get_feed_preview(prefs, items_per_category=8)
        women_current_items[user_id] = []

        # Save tinder test results to Supabase for feed personalization
        save_success = save_tinder_results_to_supabase(user_id, prefs, summary)
        print(f"[Session {user_id}] Tinder test complete. Save to Supabase: {'success' if save_success else 'FAILED'}")

        return convert_numpy({
            "status": "complete",
            "user_id": user_id,
            "items": [],
            "round": prefs.rounds_completed,
            "session_complete": True,
            "summary": summary,
            "feed_preview": feed_preview,
            "message": "Style profile complete! Use /api/women/feed/{user_id} for personalized recommendations."
        })

    # Get next 4 items
    four_items, test_info = engine.get_four_items(prefs)

    if len(four_items) < 4:
        summary = engine.get_session_summary(prefs)
        feed_preview = engine.get_feed_preview(prefs, items_per_category=8)
        return convert_numpy({
            "status": "complete",
            "user_id": user_id,
            "items": [],
            "round": prefs.rounds_completed,
            "session_complete": True,
            "summary": summary,
            "feed_preview": feed_preview,
            "message": "Style profile complete!"
        })

    women_current_items[user_id] = four_items
    women_test_info[user_id] = test_info

    # Format response
    items_response = []
    for item_id in four_items:
        info = engine.get_item_info(item_id)
        items_response.append({
            "id": item_id,
            "image_url": f"/women-images/{item_id.replace(' ', '%20')}.webp",
            "category": info.get("category", ""),
            "brand": info.get("brand", ""),
            "color": info.get("color", ""),
            "cluster": engine.item_to_cluster.get(item_id),
        })

    return convert_numpy({
        "status": "continue",
        "user_id": user_id,
        "items": items_response,
        "round": prefs.rounds_completed + 1,
        "test_info": {
            "category": test_info.get('category'),
            "category_label": {
                'tops_knitwear': 'Sweaters & Knits',
                'tops_woven': 'Blouses & Shirts',
                'tops_sleeveless': 'Tank Tops & Camis',
                'tops_special': 'Bodysuits',
                'dresses': 'Dresses',
                'bottoms_trousers': 'Pants & Trousers',
                'bottoms_skorts': 'Skirts & Shorts',
                'outerwear': 'Outerwear',
                'sportswear': 'Sportswear'
            }.get(test_info.get('category'), test_info.get('category')),
            "category_index": test_info.get('category_index'),
            "total_categories": test_info.get('total_categories'),
            "round_in_category": test_info.get('round_in_category'),
            "clusters_shown": test_info.get('clusters_shown'),
            "prediction": test_info.get('prediction', {}),
            "categories_completed": test_info.get('categories_completed', []),
        },
        "result_info": {
            "prediction_correct": result_info.get('prediction_correct'),
            "consecutive_correct": result_info.get('consecutive_correct'),
            "category_complete": result_info.get('category_complete'),
        },
        "session_complete": False
    })


@app.post(
    "/api/women/session/skip",
    tags=["Women's Fashion"],
    summary="Skip all 4 items",
    description="""
    Skip all 4 currently displayed items (none appealing).

    This counts as 4 dislikes for the shown clusters/attributes.
    Use sparingly - prefer choosing the "least bad" option for better learning.
    """,
    response_description="Next items to display"
)
async def skip_women_items(request: WomenSkipRequest):
    """Skip all 4 items and get new ones."""
    user_id = request.user_id

    if user_id not in women_sessions:
        raise HTTPException(status_code=400, detail="No active session")

    prefs = women_sessions[user_id]
    engine = get_unified_engine("female")

    # Record skip
    prefs = engine.record_skip_all(prefs)
    women_sessions[user_id] = prefs

    # Get next 4
    four_items, test_info = engine.get_four_items(prefs)

    if len(four_items) < 4 or engine.is_session_complete(prefs):
        summary = engine.get_session_summary(prefs)
        feed_preview = engine.get_feed_preview(prefs, items_per_category=8)
        return convert_numpy({
            "status": "complete",
            "user_id": user_id,
            "items": [],
            "session_complete": True,
            "summary": summary,
            "feed_preview": feed_preview
        })

    women_current_items[user_id] = four_items
    women_test_info[user_id] = test_info

    items_response = []
    for item_id in four_items:
        info = engine.get_item_info(item_id)
        items_response.append({
            "id": item_id,
            "image_url": f"/women-images/{item_id.replace(' ', '%20')}.webp",
            "category": info.get("category", ""),
            "brand": info.get("brand", ""),
            "color": info.get("color", ""),
            "cluster": engine.item_to_cluster.get(item_id),
        })

    return convert_numpy({
        "status": "continue",
        "user_id": user_id,
        "items": items_response,
        "round": prefs.rounds_completed + 1,
        "test_info": {
            "category": test_info.get('category'),
            "category_index": test_info.get('category_index'),
            "total_categories": test_info.get('total_categories'),
        },
        "session_complete": False
    })


@app.get(
    "/api/women/session/{user_id}/summary",
    tags=["Women's Fashion"],
    summary="Get session summary and learned preferences",
    description="""
    Get the current state of a user's style learning session.

    Returns:
    - **summary**: Learned preferences, rounds completed, attribute scores
    - **feed_preview**: Sample personalized recommendations per category

    Can be called during or after a session.
    """,
    response_description="Session summary with learned preferences"
)
async def get_women_session_summary(user_id: str):
    """Get session summary for a women's fashion session."""
    if user_id not in women_sessions:
        raise HTTPException(status_code=404, detail=f"No session found for user: {user_id}")

    prefs = women_sessions[user_id]
    engine = get_unified_engine("female")

    summary = engine.get_session_summary(prefs)
    feed_preview = engine.get_feed_preview(prefs, items_per_category=8)

    return convert_numpy({
        "user_id": user_id,
        "gender": "female",
        "summary": summary,
        "feed_preview": feed_preview
    })


@app.get(
    "/api/women/feed/{user_id}",
    tags=["Women's Fashion"],
    summary="Get personalized feed based on learned preferences",
    description="""
    Get personalized fashion recommendations based on learned preferences.

    **Parameters:**
    - `user_id`: User ID from session
    - `items_per_category`: Number of items per category (default: 20)
    - `categories`: Comma-separated list of categories to include (default: all)

    **Ranking Formula:**
    Items are ranked by: `0.4 * taste_similarity + 0.35 * attr_match + 0.25 * cluster_match`

    **Response includes:**
    - Similarity score (how well item matches taste vector)
    - Attribute match score (how well attributes match preferences)
    - Cluster match score (how well visual cluster matches preferences)
    """,
    response_description="Personalized recommendations per category"
)
async def get_women_feed(
    user_id: str,
    items_per_category: int = Query(default=20, ge=1, le=100, description="Items per category"),
    categories: Optional[str] = Query(default=None, description="Comma-separated category IDs to include")
):
    """Get personalized feed for women's fashion."""
    if user_id not in women_sessions:
        raise HTTPException(status_code=404, detail=f"No session found for user: {user_id}")

    prefs = women_sessions[user_id]
    engine = get_unified_engine("female")

    feed = engine.get_feed_preview(prefs, items_per_category=items_per_category)

    # Filter categories if specified
    if categories:
        cat_list = [c.strip() for c in categories.split(",")]
        feed = {k: v for k, v in feed.items() if k in cat_list}

    # Add human-readable labels
    category_labels = {
        'tops_knitwear': 'Sweaters & Knits',
        'tops_woven': 'Blouses & Shirts',
        'tops_sleeveless': 'Tank Tops & Camis',
        'tops_special': 'Bodysuits',
        'dresses': 'Dresses',
        'bottoms_trousers': 'Pants & Trousers',
        'bottoms_skorts': 'Skirts & Shorts',
        'outerwear': 'Outerwear',
        'sportswear': 'Sportswear'
    }

    formatted_feed = {}
    for cat, items in feed.items():
        formatted_feed[cat] = {
            "label": category_labels.get(cat, cat),
            "items": items,
            "count": len(items)
        }

    return convert_numpy({
        "user_id": user_id,
        "gender": "female",
        "feed": formatted_feed,
        "total_items": sum(len(items) for items in feed.values()),
        "image_base_url": "http://ecommerce.api.outrove.ai:8080/women-images/"
    })


@app.get(
    "/api/women/health",
    tags=["Women's Fashion"],
    summary="Health check for women's fashion API",
    description="Check if the women's fashion API is healthy and get stats."
)
async def women_health_check():
    """Health check for women's fashion system."""
    try:
        engine = get_unified_engine("female")
        return {
            "status": "healthy",
            "gender": "female",
            "total_items": len(engine.item_ids),
            "total_categories": len(engine.category_clusters),
            "categories": list(engine.category_clusters.keys()),
            "clusters_per_category": {
                cat: len(clusters) for cat, clusters in engine.category_clusters.items()
            },
            "attributes_loaded": len(engine.item_attributes) if engine.item_attributes else 0,
            "image_base_url": "http://ecommerce.api.outrove.ai:8080/women-images/",
            "api_docs": "http://ecommerce.api.outrove.ai:8080/docs"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# Lazy-loaded women's search engine (Supabase-based)
_women_search_engine = None


def get_women_search_engine():
    """Get or create WomenSearchEngine singleton (uses Supabase pgvector)."""
    global _women_search_engine
    if _women_search_engine is None:
        from women_search_engine import WomenSearchEngine
        _women_search_engine = WomenSearchEngine()
    return _women_search_engine


@app.post(
    "/api/women/search",
    tags=["Women's Fashion"],
    summary="Text search for women's fashion",
    description="""
Search for women's fashion items using natural language queries.

Uses FashionCLIP to encode text queries and Supabase pgvector to find
visually similar items in the products database.

**Example queries:**
- "flowy blue dress"
- "black blazer"
- "something elegant for a wedding"
- "cozy stay at home vibes"

**Pagination:**
- `page`: Page number (1-indexed, default: 1)
- `page_size`: Results per page (default: 50)

**Optional filters:**
- `categories`: Filter by product categories
- `exclude_colors`: Colors to exclude from results
- `exclude_materials`: Materials to exclude
- `exclude_brands`: Brands to exclude
- `min_price` / `max_price`: Price range filter
"""
)
async def search_women_fashion(request: dict):
    """
    Text search for women's fashion using FashionCLIP + Supabase pgvector.
    """
    query = request.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    page = max(1, request.get("page", 1))
    page_size = min(max(1, request.get("page_size", 50)), 200)

    # Optional filters
    categories = request.get("categories")
    exclude_colors = request.get("exclude_colors")
    exclude_materials = request.get("exclude_materials")
    exclude_brands = request.get("exclude_brands")
    min_price = request.get("min_price")
    max_price = request.get("max_price")
    exclude_product_ids = request.get("exclude_product_ids")

    try:
        engine = get_women_search_engine()

        # Use filtered search if any filters provided
        if any([exclude_colors, exclude_materials, exclude_brands, min_price, max_price, exclude_product_ids]):
            return engine.search_with_filters(
                query=query,
                page=page,
                page_size=page_size,
                categories=categories,
                exclude_colors=exclude_colors,
                exclude_materials=exclude_materials,
                exclude_brands=exclude_brands,
                min_price=min_price,
                max_price=max_price,
                exclude_product_ids=exclude_product_ids,
            )
        else:
            return engine.search(
                query=query,
                page=page,
                page_size=page_size,
                categories=categories
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post(
    "/api/women/similar",
    tags=["Women's Fashion"],
    summary="Find similar products",
    description="""
Find visually similar products to a given product.

Uses FashionCLIP embeddings and Supabase pgvector for similarity search.

**Parameters:**
- `product_id`: UUID of the source product (required)
- `page`: Page number (1-indexed, default: 1)
- `page_size`: Results per page (default: 50)
- `same_category`: Only return items from the same category (default: false)
"""
)
async def get_similar_products(request: dict):
    """
    Find visually similar products using FashionCLIP embeddings.
    """
    product_id = request.get("product_id", "").strip()
    if not product_id:
        raise HTTPException(status_code=400, detail="product_id is required")

    page = max(1, request.get("page", 1))
    page_size = min(max(1, request.get("page_size", 50)), 200)
    same_category = request.get("same_category", False)

    try:
        engine = get_women_search_engine()
        return engine.get_similar(
            product_id=product_id,
            page=page,
            page_size=page_size,
            same_category=same_category
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similar search failed: {str(e)}")


@app.post(
    "/api/women/complete-fit",
    tags=["Women's Fashion"],
    summary="Complete the fit - find complementary items using CLIP",
    description="""
Find complementary items to complete an outfit using CLIP semantic search.

**How it works:**
1. Gets source product details (color, category, occasion, usage)
2. For each complementary category, generates a semantic query like:
   "elegant jacket to wear with red dress"
3. Uses FashionCLIP to find items that semantically match the query
4. Returns top items per category

**Example queries generated:**
- Source: Red evening dress → "evening jacket to wear with red dress"
- Source: Casual blue blouse → "casual pants to wear with blue blouse"

**Category complements:**
- Tops → Bottoms, Outerwear
- Bottoms → Tops, Outerwear
- Dresses → Outerwear
- Outerwear → Tops, Bottoms, Dresses
"""
)
async def complete_the_fit(request: dict):
    """
    Find complementary items using CLIP semantic search.
    """
    product_id = request.get("product_id", "").strip()
    if not product_id:
        raise HTTPException(status_code=400, detail="product_id is required")

    items_per_category = min(max(1, request.get("items_per_category", 4)), 20)

    try:
        engine = get_women_search_engine()
        return engine.complete_the_fit(
            product_id=product_id,
            items_per_category=items_per_category
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Complete fit failed: {str(e)}")


@app.get(
    "/api/women/stats",
    tags=["Women's Fashion"],
    summary="Get search engine statistics",
)
async def get_women_search_stats():
    """Get statistics about the women's fashion search engine."""
    try:
        engine = get_women_search_engine()
        return engine.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# =============================================================================
# END WOMEN'S FASHION API ENDPOINTS
# =============================================================================


@app.get("/four", response_class=HTMLResponse)
async def get_four_frontend():
    """Serve the four-choice frontend."""
    return get_four_choice_html()


@app.get("/women", response_class=HTMLResponse)
async def get_women_frontend():
    """Serve the women's fashion frontend."""
    return get_women_html()


@app.post("/api/four/start")
async def start_four_session(request: FourChoiceStartRequest):
    """Start a new PREDICTIVE four-choice session (category-focused)."""
    user_id = request.user_id

    # Map selected styles to category names for category order
    selected_categories = []
    for style in request.selected_styles:
        style_lower = style.lower().strip()
        if style_lower in STYLE_TO_CATEGORY:
            selected_categories.append(STYLE_TO_CATEGORY[style_lower])

    # Map filters
    selected_necklines = set()
    for neckline in request.selected_necklines:
        neckline_lower = neckline.lower().strip()
        if neckline_lower in NECKLINE_TO_METADATA:
            selected_necklines.add(NECKLINE_TO_METADATA[neckline_lower])

    selected_sleeves = set()
    for sleeve in request.selected_sleeves:
        sleeve_lower = sleeve.lower().strip()
        if sleeve_lower in SLEEVE_TO_METADATA:
            selected_sleeves.add(SLEEVE_TO_METADATA[sleeve_lower])

    materials_to_avoid = set()
    for material in request.materials_to_avoid:
        material_lower = material.lower().strip()
        if material_lower in MATERIAL_TO_METADATA:
            materials_to_avoid.add(MATERIAL_TO_METADATA[material_lower])

    # Create PREDICTIVE preferences
    prefs = PredictivePreferences(
        user_id=user_id,
        colors_to_avoid=set(c.lower().strip() for c in request.colors_to_avoid if c),
        materials_to_avoid=materials_to_avoid,
        selected_necklines=selected_necklines,
        selected_sleeves=selected_sleeves
    )

    # Initialize session with category order
    prefs = predictive_engine.initialize_session(prefs, selected_categories if selected_categories else None)
    predictive_sessions[user_id] = prefs

    # Get first 4 items (from first category)
    four_items, test_info = predictive_engine.get_four_items(prefs)

    if len(four_items) < 4:
        return {"status": "no_items", "message": "Not enough items available"}

    predictive_current_items[user_id] = four_items
    predictive_test_info[user_id] = test_info

    # Format items for response
    items_response = []
    for item_id in four_items:
        info = predictive_engine.get_item_info(item_id)
        item_resp = format_item_response(item_id, info)
        # Add cluster info for this item
        item_resp['cluster'] = predictive_engine.item_to_cluster.get(item_id)
        items_response.append(item_resp)

    print(f"[Predictive {user_id}] Started - Category: {test_info.get('category')}")
    print(f"[Predictive {user_id}] Clusters shown: {test_info.get('clusters_shown')}")

    return convert_numpy({
        "status": "started",
        "items": items_response,
        "round": 1,
        "test_info": {
            "category": test_info.get('category'),
            "category_index": test_info.get('category_index'),
            "total_categories": test_info.get('total_categories'),
            "round_in_category": test_info.get('round_in_category'),
            "clusters_shown": test_info.get('clusters_shown'),
            "prediction": test_info.get('prediction', {}),
            "categories_completed": test_info.get('categories_completed', []),
        },
        "session_complete": False
    })


@app.post("/api/four/choose")
async def record_four_choice(request: FourChoiceRequest):
    """Record user's choice and validate prediction."""
    user_id = request.user_id

    if user_id not in predictive_sessions:
        raise HTTPException(status_code=400, detail="No active session. Call /api/four/start first.")

    prefs = predictive_sessions[user_id]
    current_four = predictive_current_items.get(user_id, [])

    if not current_four:
        raise HTTPException(status_code=400, detail="No current items")

    if request.winner_id not in current_four:
        raise HTTPException(status_code=400, detail="Winner must be one of the shown items")

    # Record the choice and validate prediction
    prefs, result_info = predictive_engine.record_choice(prefs, request.winner_id)
    predictive_sessions[user_id] = prefs

    # Log prediction result
    prediction = predictive_test_info.get(user_id, {}).get('prediction', {})
    if result_info.get('had_prediction'):
        status = "CORRECT" if result_info['prediction_correct'] else "WRONG"
        print(f"[Predictive {user_id}] Prediction {status}: predicted={prediction.get('predicted_cluster')}, actual={result_info['winner_cluster']}, conf={prediction.get('confidence', 0):.3f}")

    if result_info.get('category_complete'):
        print(f"[Predictive {user_id}] Category '{prefs.category_order[prefs.current_category_index - 1]}' COMPLETE after {result_info.get('consecutive_correct')} correct predictions")

    # Check if all categories complete
    if result_info.get('all_complete') or predictive_engine.is_session_complete(prefs):
        summary = predictive_engine.get_session_summary(prefs)
        feed_preview = predictive_engine.get_feed_preview(prefs, items_per_category=8)
        predictive_current_items[user_id] = []
        return convert_numpy({
            "status": "complete",
            "items": [],
            "round": prefs.rounds_completed,
            "stats": summary,
            "result_info": result_info,
            "feed_preview": feed_preview,
            "session_complete": True
        })

    # Get next 4 items
    four_items, test_info = predictive_engine.get_four_items(prefs)

    if len(four_items) < 4:
        summary = predictive_engine.get_session_summary(prefs)
        feed_preview = predictive_engine.get_feed_preview(prefs, items_per_category=8)
        return convert_numpy({
            "status": "complete",
            "items": [],
            "round": prefs.rounds_completed,
            "stats": summary,
            "result_info": result_info,
            "feed_preview": feed_preview,
            "session_complete": True
        })

    predictive_current_items[user_id] = four_items
    predictive_test_info[user_id] = test_info

    # Format items
    items_response = []
    for item_id in four_items:
        info = predictive_engine.get_item_info(item_id)
        item_resp = format_item_response(item_id, info)
        item_resp['cluster'] = predictive_engine.item_to_cluster.get(item_id)
        items_response.append(item_resp)

    summary = predictive_engine.get_session_summary(prefs)

    return convert_numpy({
        "status": "continue",
        "items": items_response,
        "round": prefs.rounds_completed + 1,
        "test_info": {
            "category": test_info.get('category'),
            "category_index": test_info.get('category_index'),
            "total_categories": test_info.get('total_categories'),
            "round_in_category": test_info.get('round_in_category'),
            "clusters_shown": test_info.get('clusters_shown'),
            "prediction": test_info.get('prediction', {}),
            "categories_completed": test_info.get('categories_completed', []),
        },
        "result_info": result_info,
        "stats": summary,
        "session_complete": False
    })


@app.post("/api/four/skip")
async def skip_four_choice(request: BaseModel):
    """Skip all 4 items (none appealing)."""
    user_id = getattr(request, 'user_id', 'default_user')

    if user_id not in predictive_sessions:
        raise HTTPException(status_code=400, detail="No active session")

    prefs = predictive_sessions[user_id]

    # Record skip
    prefs = predictive_engine.record_skip_all(prefs)
    predictive_sessions[user_id] = prefs

    # Get next 4
    four_items, test_info = predictive_engine.get_four_items(prefs)

    if len(four_items) < 4 or predictive_engine.is_session_complete(prefs):
        summary = predictive_engine.get_session_summary(prefs)
        feed_preview = predictive_engine.get_feed_preview(prefs, items_per_category=8)
        return convert_numpy({
            "status": "complete",
            "items": [],
            "round": prefs.rounds_completed,
            "stats": summary,
            "feed_preview": feed_preview,
            "session_complete": True
        })

    predictive_current_items[user_id] = four_items
    predictive_test_info[user_id] = test_info

    items_response = []
    for item_id in four_items:
        info = predictive_engine.get_item_info(item_id)
        item_resp = format_item_response(item_id, info)
        item_resp['cluster'] = predictive_engine.item_to_cluster.get(item_id)
        items_response.append(item_resp)

    return convert_numpy({
        "status": "continue",
        "items": items_response,
        "round": prefs.rounds_completed + 1,
        "test_info": {
            "category": test_info.get('category'),
            "category_index": test_info.get('category_index'),
            "total_categories": test_info.get('total_categories'),
            "round_in_category": test_info.get('round_in_category'),
            "clusters_shown": test_info.get('clusters_shown'),
            "prediction": test_info.get('prediction', {}),
            "categories_completed": test_info.get('categories_completed', []),
        },
        "session_complete": False
    })


# ============ RANKING ENDPOINTS ============

class RankingStartRequest(BaseModel):
    user_id: str = "default_user"
    colors_to_avoid: List[str] = []
    materials_to_avoid: List[str] = []
    selected_styles: List[str] = []
    selected_necklines: List[str] = []
    selected_sleeves: List[str] = []


class RankingSubmitRequest(BaseModel):
    user_id: str = "default_user"
    ranked_ids: List[str]  # Item IDs in order: best first, worst last


@app.get("/rank", response_class=HTMLResponse)
async def get_ranking_frontend():
    """Serve the ranking frontend."""
    return get_ranking_html()


@app.post("/api/rank/start")
async def start_ranking_session(request: RankingStartRequest):
    """Start a new ranking session."""
    user_id = request.user_id

    # Map selected styles to category names
    selected_categories = set()
    for style in request.selected_styles:
        style_lower = style.lower().strip()
        if style_lower in STYLE_TO_CATEGORY:
            selected_categories.add(STYLE_TO_CATEGORY[style_lower])

    # Map selected necklines
    selected_necklines = set()
    for neckline in request.selected_necklines:
        neckline_lower = neckline.lower().strip()
        if neckline_lower in NECKLINE_TO_METADATA:
            selected_necklines.add(NECKLINE_TO_METADATA[neckline_lower])

    # Map selected sleeves
    selected_sleeves = set()
    for sleeve in request.selected_sleeves:
        sleeve_lower = sleeve.lower().strip()
        if sleeve_lower in SLEEVE_TO_METADATA:
            selected_sleeves.add(SLEEVE_TO_METADATA[sleeve_lower])

    # Map materials to avoid
    materials_to_avoid = set()
    for material in request.materials_to_avoid:
        material_lower = material.lower().strip()
        if material_lower in MATERIAL_TO_METADATA:
            materials_to_avoid.add(MATERIAL_TO_METADATA[material_lower])

    # Create new preferences
    prefs = RankingPreferences(
        user_id=user_id,
        colors_to_avoid=set(c.lower().strip() for c in request.colors_to_avoid if c),
        materials_to_avoid=materials_to_avoid,
        selected_categories=selected_categories,
        selected_necklines=selected_necklines,
        selected_sleeves=selected_sleeves
    )

    ranking_sessions[user_id] = prefs

    # Get candidates and items to rank
    candidates = ranking_engine.get_candidates(prefs)
    print(f"[Ranking {user_id}] Started with {len(candidates)} candidates")

    items_to_rank = ranking_engine.get_items_to_rank(prefs, candidates)

    if len(items_to_rank) < ranking_engine.ITEMS_PER_ROUND:
        return {"status": "no_items", "message": "Not enough items available"}

    ranking_current_items[user_id] = items_to_rank

    # Format items
    items_response = []
    for item_id in items_to_rank:
        info = ranking_engine.get_item_info(item_id)
        items_response.append(format_item_response(item_id, info))

    return convert_numpy({
        "status": "started",
        "items": items_response,
        "round": 1,
        "total_rounds": ranking_engine.MAX_ROUNDS,
        "items_per_round": ranking_engine.ITEMS_PER_ROUND,
        "session_complete": False
    })


@app.post("/api/rank/submit")
async def submit_ranking(request: RankingSubmitRequest):
    """Submit ranking and get next items."""
    user_id = request.user_id

    if user_id not in ranking_sessions:
        raise HTTPException(status_code=400, detail="No active session. Call /api/rank/start first.")

    prefs = ranking_sessions[user_id]
    current_items = ranking_current_items.get(user_id, [])

    if not current_items:
        raise HTTPException(status_code=400, detail="No current items")

    # Validate that all ranked IDs are from current items
    current_set = set(current_items)
    ranked_set = set(request.ranked_ids)
    if not ranked_set.issubset(current_set):
        raise HTTPException(status_code=400, detail="Ranked items must be from the shown items")

    if len(request.ranked_ids) != len(current_items):
        raise HTTPException(status_code=400, detail=f"Must rank all {len(current_items)} items")

    # Record the ranking
    prefs = ranking_engine.record_ranking(prefs, request.ranked_ids)
    ranking_sessions[user_id] = prefs

    # Check if session complete
    if ranking_engine.is_session_complete(prefs):
        summary = ranking_engine.get_session_summary(prefs)
        ranking_current_items[user_id] = []
        return convert_numpy({
            "status": "complete",
            "items": [],
            "round": prefs.rounds_completed,
            "stats": summary,
            "session_complete": True
        })

    # Get next items to rank
    candidates = ranking_engine.get_candidates(prefs)
    items_to_rank = ranking_engine.get_items_to_rank(prefs, candidates)

    if len(items_to_rank) < ranking_engine.ITEMS_PER_ROUND:
        # Not enough items, end session
        summary = ranking_engine.get_session_summary(prefs)
        return convert_numpy({
            "status": "complete",
            "items": [],
            "round": prefs.rounds_completed,
            "stats": summary,
            "session_complete": True
        })

    ranking_current_items[user_id] = items_to_rank

    # Format items
    items_response = []
    for item_id in items_to_rank:
        info = ranking_engine.get_item_info(item_id)
        items_response.append(format_item_response(item_id, info))

    summary = ranking_engine.get_session_summary(prefs)

    return convert_numpy({
        "status": "continue",
        "items": items_response,
        "round": prefs.rounds_completed + 1,
        "stats": summary,
        "session_complete": False
    })


@app.post("/api/rank/skip")
async def skip_ranking(request: BaseModel):
    """Skip all items (none appealing)."""
    user_id = getattr(request, 'user_id', 'default_user')

    if user_id not in ranking_sessions:
        raise HTTPException(status_code=400, detail="No active session")

    prefs = ranking_sessions[user_id]
    current_items = ranking_current_items.get(user_id, [])

    if current_items:
        prefs = ranking_engine.record_skip_all(prefs, current_items)
        ranking_sessions[user_id] = prefs

    # Get next items
    candidates = ranking_engine.get_candidates(prefs)
    items_to_rank = ranking_engine.get_items_to_rank(prefs, candidates)

    if len(items_to_rank) < ranking_engine.ITEMS_PER_ROUND or ranking_engine.is_session_complete(prefs):
        summary = ranking_engine.get_session_summary(prefs)
        return convert_numpy({
            "status": "complete",
            "items": [],
            "round": prefs.rounds_completed,
            "stats": summary,
            "session_complete": True
        })

    ranking_current_items[user_id] = items_to_rank

    items_response = []
    for item_id in items_to_rank:
        info = ranking_engine.get_item_info(item_id)
        items_response.append(format_item_response(item_id, info))

    return convert_numpy({
        "status": "continue",
        "items": items_response,
        "round": prefs.rounds_completed + 1,
        "session_complete": False
    })


# ============ ATTRIBUTE TEST ENDPOINTS ============

class AttrTestStartRequest(BaseModel):
    user_id: str = "default_user"
    colors_to_avoid: List[str] = []
    materials_to_avoid: List[str] = []
    selected_styles: List[str] = []
    selected_necklines: List[str] = []
    selected_sleeves: List[str] = []


class AttrTestChooseRequest(BaseModel):
    user_id: str = "default_user"
    winner_id: str


@app.get("/attr", response_class=HTMLResponse)
async def get_attr_frontend():
    """Serve the attribute test frontend."""
    return get_attr_test_html()


@app.post("/api/attr/start")
async def start_attr_session(request: AttrTestStartRequest):
    """Start a new attribute isolation test session."""
    user_id = request.user_id

    # Map filters (reuse same logic as other endpoints)
    selected_categories = set()
    for style in request.selected_styles:
        style_lower = style.lower().strip()
        if style_lower in STYLE_TO_CATEGORY:
            selected_categories.add(STYLE_TO_CATEGORY[style_lower])

    selected_necklines = set()
    for neckline in request.selected_necklines:
        neckline_lower = neckline.lower().strip()
        if neckline_lower in NECKLINE_TO_METADATA:
            selected_necklines.add(NECKLINE_TO_METADATA[neckline_lower])

    selected_sleeves = set()
    for sleeve in request.selected_sleeves:
        sleeve_lower = sleeve.lower().strip()
        if sleeve_lower in SLEEVE_TO_METADATA:
            selected_sleeves.add(SLEEVE_TO_METADATA[sleeve_lower])

    materials_to_avoid = set()
    for material in request.materials_to_avoid:
        material_lower = material.lower().strip()
        if material_lower in MATERIAL_TO_METADATA:
            materials_to_avoid.add(MATERIAL_TO_METADATA[material_lower])

    # Create preferences
    prefs = AttributeTestPreferences(
        user_id=user_id,
        colors_to_avoid=set(c.lower().strip() for c in request.colors_to_avoid if c),
        materials_to_avoid=materials_to_avoid,
        selected_categories=selected_categories,
        selected_necklines=selected_necklines,
        selected_sleeves=selected_sleeves
    )

    attr_sessions[user_id] = prefs

    # Get first 4 items with test info
    candidates = attribute_engine.get_candidates(prefs)
    four_items, test_info = attribute_engine.get_four_items(prefs, candidates)

    if len(four_items) < 4:
        return {"status": "no_items", "message": "Not enough items available"}

    attr_current_items[user_id] = four_items
    attr_test_info[user_id] = test_info

    # Format items for response
    items_response = []
    for item_id in four_items:
        info = attribute_engine.get_item_info(item_id)
        item_resp = format_item_response(item_id, info)
        # Add the test attribute value for this item
        item_resp['test_attribute_value'] = test_info.get('items_attributes', {}).get(item_id, 'unknown')
        items_response.append(item_resp)

    print(f"[Attr-Test {user_id}] Started with {len(candidates)} candidates")
    print(f"[Attr-Test {user_id}] Phase {test_info.get('phase')}: {test_info.get('phase_name')} - Testing: {test_info.get('attribute')}")

    # Get progress info
    progress = attribute_engine.get_progress(prefs)

    return convert_numpy({
        "status": "started",
        "items": items_response,
        "round": 1,
        "total_rounds": attribute_engine.MAX_ROUNDS,
        "test_info": {
            "phase": test_info.get('phase', 1),
            "phase_name": test_info.get('phase_name', 'Style Foundation'),
            "phase_icon": test_info.get('phase_icon', '🎨'),
            "phase_description": test_info.get('phase_description', ''),
            "attribute": test_info.get('attribute', 'archetype'),
            "total_phases": test_info.get('total_phases', 6),
        },
        "progress": progress,
        "session_complete": False
    })


@app.post("/api/attr/choose")
async def record_attr_choice(request: AttrTestChooseRequest):
    """Record user's choice and get next items with new test info."""
    user_id = request.user_id

    if user_id not in attr_sessions:
        raise HTTPException(status_code=400, detail="No active session. Call /api/attr/start first.")

    prefs = attr_sessions[user_id]
    current_four = attr_current_items.get(user_id, [])

    if not current_four:
        raise HTTPException(status_code=400, detail="No current items")

    if request.winner_id not in current_four:
        raise HTTPException(status_code=400, detail="Winner must be one of the shown items")

    # Record the choice
    prefs = attribute_engine.record_choice(prefs, request.winner_id, current_four)
    attr_sessions[user_id] = prefs

    # Check if session complete
    if attribute_engine.is_session_complete(prefs):
        summary = attribute_engine.get_session_summary(prefs)
        attr_current_items[user_id] = []
        return convert_numpy({
            "status": "complete",
            "items": [],
            "round": prefs.rounds_completed,
            "stats": summary,
            "session_complete": True
        })

    # Get next 4 items with test info
    candidates = attribute_engine.get_candidates(prefs)
    four_items, test_info = attribute_engine.get_four_items(prefs, candidates)

    if len(four_items) < 4 or test_info.get('complete'):
        # Not enough items or all tests complete
        summary = attribute_engine.get_session_summary(prefs)
        return convert_numpy({
            "status": "complete",
            "items": [],
            "round": prefs.rounds_completed,
            "stats": summary,
            "session_complete": True
        })

    attr_current_items[user_id] = four_items
    attr_test_info[user_id] = test_info

    # Format items
    items_response = []
    for item_id in four_items:
        info = attribute_engine.get_item_info(item_id)
        item_resp = format_item_response(item_id, info)
        item_resp['test_attribute_value'] = test_info.get('items_attributes', {}).get(item_id, 'unknown')
        items_response.append(item_resp)

    summary = attribute_engine.get_session_summary(prefs)
    progress = attribute_engine.get_progress(prefs)

    print(f"[Attr-Test {user_id}] Round {prefs.rounds_completed + 1} - Testing: {test_info.get('attribute')}")

    return convert_numpy({
        "status": "continue",
        "items": items_response,
        "round": prefs.rounds_completed + 1,
        "test_info": {
            "phase": test_info.get('phase', 1),
            "phase_name": test_info.get('phase_name', 'Style Foundation'),
            "phase_icon": test_info.get('phase_icon', '🎨'),
            "phase_description": test_info.get('phase_description', ''),
            "attribute": test_info.get('attribute', 'archetype'),
            "total_phases": test_info.get('total_phases', 6),
        },
        "progress": progress,
        "stats": summary,
        "session_complete": False
    })


@app.post("/api/attr/skip")
async def skip_attr_choice(request: BaseModel):
    """Skip all 4 items (none appealing)."""
    user_id = getattr(request, 'user_id', 'default_user')

    if user_id not in attr_sessions:
        raise HTTPException(status_code=400, detail="No active session")

    prefs = attr_sessions[user_id]
    current_four = attr_current_items.get(user_id, [])

    if current_four:
        prefs = attribute_engine.record_skip_all(prefs, current_four)
        attr_sessions[user_id] = prefs

    # Get next 4 items
    candidates = attribute_engine.get_candidates(prefs)
    four_items, test_info = attribute_engine.get_four_items(prefs, candidates)

    if len(four_items) < 4 or attribute_engine.is_session_complete(prefs) or test_info.get('complete'):
        summary = attribute_engine.get_session_summary(prefs)
        return convert_numpy({
            "status": "complete",
            "items": [],
            "round": prefs.rounds_completed,
            "stats": summary,
            "session_complete": True
        })

    attr_current_items[user_id] = four_items
    attr_test_info[user_id] = test_info

    items_response = []
    for item_id in four_items:
        info = attribute_engine.get_item_info(item_id)
        item_resp = format_item_response(item_id, info)
        item_resp['test_attribute_value'] = test_info.get('items_attributes', {}).get(item_id, 'unknown')
        items_response.append(item_resp)

    progress = attribute_engine.get_progress(prefs)

    return convert_numpy({
        "status": "continue",
        "items": items_response,
        "round": prefs.rounds_completed + 1,
        "test_info": {
            "phase": test_info.get('phase', 1),
            "phase_name": test_info.get('phase_name', 'Style Foundation'),
            "phase_icon": test_info.get('phase_icon', '🎨'),
            "phase_description": test_info.get('phase_description', ''),
            "attribute": test_info.get('attribute', 'archetype'),
            "total_phases": test_info.get('total_phases', 6),
        },
        "progress": progress,
        "session_complete": False
    })


def format_item_response(item_id: str, info: Dict) -> Dict:
    """Format item info for frontend."""
    # Build image path - use URL encoding for spaces
    category = info.get('category', 'Plain T-shirts')
    # Extract just the number from item_id like "Plain T-shirts/123"
    item_num = item_id.split('/')[-1] if '/' in item_id else item_id

    # URL encode the category for spaces
    category_encoded = category.replace(' ', '%20')
    image_url = f"/images/{category_encoded}/{item_num}.webp"

    # Get cluster profile from engine
    cluster = info.get('cluster', 0)
    cluster_profile = engine._get_cluster_profile(cluster)

    return {
        "id": item_id,
        "image_url": image_url,
        "category": category,
        "brand": info.get('brand', 'Unknown'),
        "color": info.get('color', 'Unknown'),
        "fit": info.get('fit', 'Regular'),
        "fabric": info.get('fabric', 'Cotton'),
        "archetype": info.get('archetype', 'classic'),
        "cluster": int(cluster) if isinstance(cluster, (np.integer, int)) else 0,
        "cluster_profile": cluster_profile.get('description', 'exploring...'),
        "dominant_archetype": cluster_profile.get('dominant_archetype', 'classic'),
        "top_anchors": cluster_profile.get('top_anchors', [])
    }


def get_html_template() -> str:
    """Return the full HTML template with setup screen and swipe interface."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
    <title>Outrove - Style Discovery</title>
    <script src="https://cdn.tailwindcss.com?plugins=forms,typography"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@200;300;400;500;600;700;800&family=Noto+Sans:wght@400;500;700&display=swap" rel="stylesheet"/>
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
    <script>
        tailwind.config = {
            darkMode: "class",
            theme: {
                extend: {
                    colors: {
                        "primary": "#1A1A1A",
                        "primary-dark": "#0A0A0A",
                        "background-light": "#f9fafb",
                        "background-dark": "#121212",
                        "surface-light": "#ffffff",
                        "surface-dark": "#1E1E1E",
                        "text-main": "#131811",
                        "text-muted": "#6c7275",
                    },
                    fontFamily: {
                        display: ["Plus Jakarta Sans", "sans-serif"],
                        body: ["Noto Sans", "sans-serif"],
                    },
                }
            }
        };
    </script>
    <style>
        .card-stack-container { perspective: 1000px; }
        body::-webkit-scrollbar { display: none; }
        .no-scrollbar::-webkit-scrollbar { display: none; }
        .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }

        @keyframes swipeRight {
            0% { transform: translateX(0) rotate(0deg); opacity: 1; }
            100% { transform: translateX(150%) rotate(20deg); opacity: 0; }
        }
        @keyframes swipeLeft {
            0% { transform: translateX(0) rotate(0deg); opacity: 1; }
            100% { transform: translateX(-150%) rotate(-20deg); opacity: 0; }
        }
        @keyframes cardEnter {
            0% { transform: scale(0.92) translateY(24px); opacity: 0.4; }
            100% { transform: scale(1) translateY(0); opacity: 1; }
        }
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(10px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .swipe-right { animation: swipeRight 0.4s ease-out forwards; }
        .swipe-left { animation: swipeLeft 0.4s ease-out forwards; }
        .card-enter { animation: cardEnter 0.3s ease-out forwards; }
        .fade-in { animation: fadeIn 0.5s ease-out forwards; }

        .stamp {
            position: absolute;
            top: 2rem;
            padding: 0.5rem 1rem;
            border: 4px solid;
            border-radius: 0.5rem;
            font-size: 1.5rem;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 30;
            background: rgba(255,255,255,0.9);
        }
        .stamp-like { left: 2rem; border-color: #22c55e; color: #22c55e; transform: rotate(-12deg); }
        .stamp-dislike { right: 2rem; border-color: #ef4444; color: #ef4444; transform: rotate(12deg); }
        .show-stamp { opacity: 1; }
    </style>
</head>
<body class="bg-background-light text-text-main font-display antialiased min-h-screen flex flex-col">

<!-- Header -->
<header class="sticky top-0 z-50 bg-surface-light border-b border-gray-100 px-6 py-3 shadow-sm">
    <div class="max-w-[1100px] mx-auto flex items-center justify-between">
        <div class="flex items-center gap-3">
            <h2 class="text-xl font-bold tracking-tight">Style Discovery</h2>
        </div>
        <button onclick="showSetup()" class="text-sm font-medium text-text-muted hover:text-text-main transition-colors">
            Restart
        </button>
    </div>
</header>

<!-- Setup Screen - Core Selection -->
<div id="setup-screen" class="flex-grow flex flex-col items-center py-8 px-4 sm:px-6">
    <div class="w-full max-w-[960px] flex flex-col gap-10">

        <!-- Core Selection Section -->
        <div class="text-center">
            <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">CORE SELECTION</h2>
            <h3 class="text-2xl font-bold text-text-main">Which tops are you looking for?</h3>
        </div>

        <div class="flex justify-center" id="core-selection">
            <button onclick="toggleCoreType('tshirts')" data-type="tshirts" class="core-type-btn flex flex-col items-center justify-center p-4 rounded-xl border transition-all h-[180px] w-[200px] border-gray-200 bg-white hover:border-gray-300">
                <div class="w-16 h-16 mb-4 flex items-center justify-center">
                    <img src="/assets/Initial%20Screen%20Icons/TShirt%20Icons.png" alt="T-Shirts" class="object-contain h-full w-full"/>
                </div>
                <span class="font-bold text-sm mb-1">T-Shirts</span>
                <span class="text-[10px] text-gray-500">Casual everyday</span>
            </button>
        </div>

        <div class="border-t border-gray-100 my-2"></div>

        <!-- Size Section -->
        <div class="text-center">
            <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">MEASUREMENTS</h2>
            <h3 class="text-2xl font-bold text-text-main mb-8">What is your typical size?</h3>
            <div class="max-w-2xl mx-auto">
                <div class="flex items-center justify-between bg-white border border-gray-200 rounded-full p-1.5 h-14 shadow-sm" id="size-selector">
                    <button onclick="selectSize('XS')" data-size="XS" class="size-btn flex-1 h-full rounded-full text-xs font-bold transition-all flex items-center justify-center text-text-main hover:bg-gray-50">XS</button>
                    <button onclick="selectSize('S')" data-size="S" class="size-btn flex-1 h-full rounded-full text-xs font-bold transition-all flex items-center justify-center text-text-main hover:bg-gray-50">S</button>
                    <button onclick="selectSize('M')" data-size="M" class="size-btn flex-1 h-full rounded-full text-xs font-bold transition-all flex items-center justify-center text-white bg-black shadow-md">M</button>
                    <button onclick="selectSize('L')" data-size="L" class="size-btn flex-1 h-full rounded-full text-xs font-bold transition-all flex items-center justify-center text-text-main hover:bg-gray-50">L</button>
                    <button onclick="selectSize('XL')" data-size="XL" class="size-btn flex-1 h-full rounded-full text-xs font-bold transition-all flex items-center justify-center text-text-main hover:bg-gray-50">XL</button>
                    <button onclick="selectSize('2XL')" data-size="2XL" class="size-btn flex-1 h-full rounded-full text-xs font-bold transition-all flex items-center justify-center text-text-main hover:bg-gray-50">2XL</button>
                </div>
            </div>
        </div>

        <div class="border-t border-gray-100 my-2"></div>

        <!-- T-Shirt Style Section -->
        <div id="tshirt-styles-section" class="hidden">
            <div class="text-center mb-6">
                <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">T-SHIRT STYLES</h2>
                <h3 class="text-xl font-bold text-text-main">Select styles you wear</h3>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-5 gap-4" id="tshirt-styles">
                <div onclick="toggleStyle('plain')" data-style="plain" class="style-btn cursor-pointer rounded-xl border overflow-hidden transition-all border-gray-200 hover:border-gray-300">
                    <div class="w-full aspect-square bg-gray-50 flex items-center justify-center overflow-hidden">
                        <img src="/assets/Tees%20styles/Plain.png" alt="Plain" class="w-full h-full object-cover"/>
                    </div>
                    <div class="p-3 text-center border-t border-gray-100 bg-white">
                        <span class="text-sm font-bold text-gray-900">Plain</span>
                    </div>
                </div>
                <div onclick="toggleStyle('graphics')" data-style="graphics" class="style-btn cursor-pointer rounded-xl border overflow-hidden transition-all border-gray-200 hover:border-gray-300">
                    <div class="w-full aspect-square bg-gray-50 flex items-center justify-center overflow-hidden">
                        <img src="/assets/Tees%20styles/Graphic%20Tshirt.png" alt="Graphics" class="w-full h-full object-cover"/>
                    </div>
                    <div class="p-3 text-center border-t border-gray-100 bg-white">
                        <span class="text-sm font-bold text-gray-900">Graphics</span>
                    </div>
                </div>
                <div onclick="toggleStyle('smalllogos')" data-style="smalllogos" class="style-btn cursor-pointer rounded-xl border overflow-hidden transition-all border-gray-200 hover:border-gray-300">
                    <div class="w-full aspect-square bg-gray-50 flex items-center justify-center overflow-hidden">
                        <img src="/assets/Tees%20styles/Small%20Logos.png" alt="Small Logos" class="w-full h-full object-cover"/>
                    </div>
                    <div class="p-3 text-center border-t border-gray-100 bg-white">
                        <span class="text-sm font-bold text-gray-900">Small Logos</span>
                    </div>
                </div>
                <div onclick="toggleStyle('athletic')" data-style="athletic" class="style-btn cursor-pointer rounded-xl border overflow-hidden transition-all border-gray-200 hover:border-gray-300">
                    <div class="w-full aspect-square bg-gray-50 flex items-center justify-center overflow-hidden">
                        <img src="/assets/Tees%20styles/Atheletic%20Tshirt.png" alt="Athletic" class="w-full h-full object-cover"/>
                    </div>
                    <div class="p-3 text-center border-t border-gray-100 bg-white">
                        <span class="text-sm font-bold text-gray-900">Athletic</span>
                    </div>
                </div>
                <div onclick="toggleStyle('pocket')" data-style="pocket" class="style-btn cursor-pointer rounded-xl border overflow-hidden transition-all border-gray-200 hover:border-gray-300">
                    <div class="w-full aspect-square bg-gray-50 flex items-center justify-center overflow-hidden">
                        <img src="/assets/Tees%20styles/Pocket%20Tshirt.png" alt="Pocket" class="w-full h-full object-cover"/>
                    </div>
                    <div class="p-3 text-center border-t border-gray-100 bg-white">
                        <span class="text-sm font-bold text-gray-900">Pocket</span>
                    </div>
                </div>
            </div>

            <div class="border-t border-gray-100 my-6"></div>

            <!-- Neckline Section -->
            <div class="text-center mb-6">
                <h3 class="text-lg font-bold text-text-main">Neckline Preference</h3>
            </div>
            <div class="grid grid-cols-3 gap-4 max-w-md mx-auto" id="neckline-selector">
                <div onclick="toggleNeckline('crew')" data-neckline="crew" class="neckline-btn cursor-pointer flex flex-col items-center">
                    <div class="w-full aspect-[4/3] rounded-xl bg-gray-100 overflow-hidden flex items-center justify-center transition-all hover:bg-gray-200">
                        <img src="/assets/neck%20Styles/Crew.png" alt="Crew" class="w-full h-full object-cover"/>
                    </div>
                    <span class="mt-2 text-sm font-semibold text-gray-900">Crew</span>
                </div>
                <div onclick="toggleNeckline('vneck')" data-neckline="vneck" class="neckline-btn cursor-pointer flex flex-col items-center">
                    <div class="w-full aspect-[4/3] rounded-xl bg-gray-100 overflow-hidden flex items-center justify-center transition-all hover:bg-gray-200">
                        <img src="/assets/neck%20Styles/Vneck.png" alt="V-Neck" class="w-full h-full object-cover"/>
                    </div>
                    <span class="mt-2 text-sm font-semibold text-gray-900">V-Neck</span>
                </div>
                <div onclick="toggleNeckline('henley')" data-neckline="henley" class="neckline-btn cursor-pointer flex flex-col items-center">
                    <div class="w-full aspect-[4/3] rounded-xl bg-gray-100 overflow-hidden flex items-center justify-center transition-all hover:bg-gray-200">
                        <img src="/assets/neck%20Styles/henley.png" alt="Henley" class="w-full h-full object-cover"/>
                    </div>
                    <span class="mt-2 text-sm font-semibold text-gray-900">Henley</span>
                </div>
            </div>
        </div>

        <div class="border-t border-gray-100 my-2"></div>

        <!-- Colors to Avoid -->
        <div class="text-center">
            <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">PREFERENCES</h2>
            <h3 class="text-2xl font-bold text-text-main mb-2">Colors to Avoid</h3>
            <p class="text-gray-500 text-sm mb-8">Tap any color you dislike to cross it out.</p>
            <div class="flex justify-center">
                <div id="color-selector" class="flex flex-wrap justify-center gap-4 max-w-[700px]"></div>
            </div>
        </div>

        <div class="border-t border-gray-100 my-2"></div>

        <!-- Materials to Avoid -->
        <div class="text-center">
            <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">FABRICS</h2>
            <h3 class="text-2xl font-bold text-text-main mb-8">Materials to Avoid</h3>
            <div class="grid grid-cols-2 md:grid-cols-3 gap-6" id="materials-selector">
                <button onclick="toggleMaterial('polyester')" data-material="polyester" class="material-btn relative group overflow-hidden rounded-xl aspect-[16/9] transition-transform hover:scale-[1.02]">
                    <img src="/assets/Materials/Polyster.png" alt="Polyester" class="w-full h-full object-cover"/>
                    <div class="absolute inset-0 flex items-center justify-center">
                        <span class="material-label px-4 py-1.5 rounded-full text-xs font-bold shadow-sm bg-white text-black">Polyester</span>
                    </div>
                </button>
                <button onclick="toggleMaterial('wool')" data-material="wool" class="material-btn relative group overflow-hidden rounded-xl aspect-[16/9] transition-transform hover:scale-[1.02]">
                    <img src="/assets/Materials/Wool.png" alt="Wool" class="w-full h-full object-cover"/>
                    <div class="absolute inset-0 flex items-center justify-center">
                        <span class="material-label px-4 py-1.5 rounded-full text-xs font-bold shadow-sm bg-white text-black">Wool</span>
                    </div>
                </button>
                <button onclick="toggleMaterial('linen')" data-material="linen" class="material-btn relative group overflow-hidden rounded-xl aspect-[16/9] transition-transform hover:scale-[1.02]">
                    <img src="/assets/Materials/Linen%20rectangle.png" alt="Linen" class="w-full h-full object-cover"/>
                    <div class="absolute inset-0 flex items-center justify-center">
                        <span class="material-label px-4 py-1.5 rounded-full text-xs font-bold shadow-sm bg-white text-black">Linen</span>
                    </div>
                </button>
                <button onclick="toggleMaterial('silk')" data-material="silk" class="material-btn relative group overflow-hidden rounded-xl aspect-[16/9] transition-transform hover:scale-[1.02]">
                    <img src="/assets/Materials/Silk.png" alt="Silk" class="w-full h-full object-cover"/>
                    <div class="absolute inset-0 flex items-center justify-center">
                        <span class="material-label px-4 py-1.5 rounded-full text-xs font-bold shadow-sm bg-white text-black">Silk</span>
                    </div>
                </button>
                <button onclick="toggleMaterial('synthetics')" data-material="synthetics" class="material-btn relative group overflow-hidden rounded-xl aspect-[16/9] transition-transform hover:scale-[1.02]">
                    <img src="/assets/Materials/Synthetics.png" alt="Synthetics" class="w-full h-full object-cover"/>
                    <div class="absolute inset-0 flex items-center justify-center">
                        <span class="material-label px-4 py-1.5 rounded-full text-xs font-bold shadow-sm bg-white text-black">Synthetics</span>
                    </div>
                </button>
                <button onclick="toggleMaterial('leather')" data-material="leather" class="material-btn relative group overflow-hidden rounded-xl aspect-[16/9] transition-transform hover:scale-[1.02]">
                    <img src="/assets/Materials/Leather.png" alt="Leather" class="w-full h-full object-cover"/>
                    <div class="absolute inset-0 flex items-center justify-center">
                        <span class="material-label px-4 py-1.5 rounded-full text-xs font-bold shadow-sm bg-white text-black">Leather</span>
                    </div>
                </button>
            </div>
        </div>

        <div class="h-24"></div>
    </div>
</div>

<!-- Sticky Bottom Button -->
<div id="setup-footer" class="sticky bottom-0 w-full bg-white/95 backdrop-blur-md border-t border-gray-200 py-4 px-6 shadow-lg z-40">
    <div class="max-w-[960px] mx-auto flex justify-center">
        <button onclick="beginSession()" class="bg-primary hover:bg-primary-dark text-white font-bold py-4 px-12 rounded-xl shadow-md transition-all transform active:scale-95 flex items-center gap-2 text-lg">
            Start Style Discovery
            <span class="material-symbols-outlined">arrow_forward</span>
        </button>
    </div>
</div>

<!-- Main Swipe Screen (hidden initially) -->
<main id="swipe-screen" class="hidden flex-grow flex flex-col items-center justify-center w-full max-w-6xl mx-auto px-4 py-6 relative">

    <!-- Header with Progress -->
    <div class="w-full max-w-md mb-4">
        <div class="flex justify-between items-end mb-2">
            <div>
                <span class="text-xs font-semibold uppercase tracking-wider text-subtext-light block mb-1">Style Profile</span>
                <h2 id="header-text" class="text-lg font-bold">Discovering your style...</h2>
            </div>
            <div class="text-right">
                <span id="swipe-count" class="text-2xl font-bold text-gray-900">0</span>
                <span class="text-sm text-subtext-light">/40</span>
            </div>
        </div>
        <div class="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div id="progress-bar" class="h-full bg-gradient-to-r from-gray-700 to-gray-500 rounded-full transition-all duration-500" style="width: 0%"></div>
        </div>
        <div class="flex justify-between mt-1 text-xs text-subtext-light">
            <span id="coverage-text">Coverage: 0%</span>
            <span id="stability-text">Stability: 0.00</span>
        </div>
    </div>

    <!-- Card Stack - BIGGER -->
    <div id="card-container" class="relative w-full max-w-md h-[560px] card-stack-container flex items-center justify-center">
        <!-- Back card placeholder (shows preloaded next image) -->
        <div id="back-card" class="absolute w-full h-full bg-card-light dark:bg-card-dark rounded-2xl shadow-soft border border-gray-100 dark:border-gray-700 transform scale-[0.92] translate-y-6 opacity-40 z-0 overflow-hidden">
            <div class="relative h-[400px] bg-gray-100 dark:bg-gray-800 w-full overflow-hidden">
                <img id="back-image" alt="Next Item" class="w-full h-full object-contain" src="" />
            </div>
        </div>

        <!-- Main card -->
        <div id="main-card" class="absolute w-full h-full bg-card-light dark:bg-card-dark rounded-2xl shadow-card border border-gray-100 dark:border-gray-700 z-10 overflow-hidden flex flex-col">
            <div id="stamp-like" class="stamp stamp-like">Like</div>
            <div id="stamp-dislike" class="stamp stamp-dislike">Nope</div>

            <div class="relative h-[400px] bg-gray-100 dark:bg-gray-800 w-full overflow-hidden">
                <img id="item-image" alt="Item" class="w-full h-full object-contain" src="" />
                <div id="archetype-badge" class="absolute top-3 right-3 bg-white/90 backdrop-blur-sm text-xs px-3 py-1.5 rounded-full font-semibold shadow-sm">
                    classic
                </div>
            </div>

            <div class="p-4 flex-grow flex flex-col justify-between">
                <div>
                    <div class="flex justify-between items-start mb-1">
                        <div>
                            <h3 id="item-brand" class="text-lg font-bold">Loading...</h3>
                            <p id="item-category" class="text-xs text-gray-500 font-medium uppercase tracking-wide">Category</p>
                        </div>
                        <span id="item-fit" class="bg-gray-100 dark:bg-gray-700 text-xs px-2 py-1 rounded-lg font-medium">Fit</span>
                    </div>
                    <p id="cluster-info" class="text-sm text-subtext-light mt-2">Exploring visual clusters...</p>
                </div>
                <div class="flex gap-2 mt-2 flex-wrap">
                    <span id="tag-color" class="text-xs border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 rounded-md px-2 py-1"></span>
                    <span id="tag-fabric" class="text-xs border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 rounded-md px-2 py-1"></span>
                </div>
            </div>
        </div>

        <!-- Action Buttons -->
        <div class="absolute -bottom-16 flex items-center justify-center gap-6 w-full z-30">
            <button onclick="swipe('dislike')" class="group bg-white p-4 rounded-full shadow-lg border border-gray-100 hover:scale-110 hover:border-red-200 active:scale-95 transition-all">
                <span class="material-symbols-outlined text-red-500 text-3xl">close</span>
            </button>
            <button onclick="swipe('skip')" class="bg-white p-3 rounded-full shadow-md border border-gray-100 hover:scale-110 active:scale-95 transition-all">
                <span class="material-symbols-outlined text-gray-400 text-xl">undo</span>
            </button>
            <button onclick="swipe('like')" class="group bg-white p-4 rounded-full shadow-lg border border-gray-100 hover:scale-110 hover:border-green-200 active:scale-95 transition-all">
                <span class="material-symbols-outlined text-green-500 text-3xl">favorite</span>
            </button>
        </div>
    </div>

    <div class="mt-20 text-center opacity-60">
        <p class="text-sm text-subtext-light flex items-center justify-center gap-2">
            <span class="border border-gray-300 rounded px-2 py-0.5 text-xs font-mono">&#8592;</span>
            Dislike | Like
            <span class="border border-gray-300 rounded px-2 py-0.5 text-xs font-mono">&#8594;</span>
        </p>
    </div>
</main>

<!-- Stats Section -->
<section id="stats-section" class="hidden w-full bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800 py-5 px-4 mt-auto">
    <div class="max-w-4xl mx-auto">
        <h4 class="text-xs font-bold text-subtext-light uppercase tracking-widest mb-4 text-center">Live Style Insights</h4>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div class="bg-gradient-to-br from-gray-50 to-slate-100 rounded-xl p-3 border border-gray-200">
                <div class="flex items-center gap-1 mb-1">
                    <span class="material-symbols-outlined text-gray-600 text-sm">auto_awesome</span>
                    <span class="text-xs font-semibold text-gray-700 uppercase">Archetype</span>
                </div>
                <p id="stat-archetype" class="text-base font-bold capitalize">classic</p>
                <p id="stat-archetype-score" class="text-xs text-subtext-light">+0.000</p>
            </div>
            <div class="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-3 border border-green-100">
                <div class="flex items-center gap-1 mb-1">
                    <span class="material-symbols-outlined text-green-500 text-sm">favorite</span>
                    <span class="text-xs font-semibold text-green-600 uppercase">Likes</span>
                </div>
                <p id="stat-anchor-likes" class="text-base font-bold">solids</p>
                <p id="stat-anchor-likes-count" class="text-xs text-subtext-light">0 items</p>
            </div>
            <div class="bg-gradient-to-br from-red-50 to-orange-50 rounded-xl p-3 border border-red-100">
                <div class="flex items-center gap-1 mb-1">
                    <span class="material-symbols-outlined text-red-500 text-sm">block</span>
                    <span class="text-xs font-semibold text-red-600 uppercase">Avoids</span>
                </div>
                <p id="stat-anchor-dislikes" class="text-base font-bold">-</p>
                <p id="stat-anchor-dislikes-count" class="text-xs text-subtext-light">0 items</p>
            </div>
            <div class="bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-3 border border-purple-100">
                <div class="flex items-center gap-1 mb-1">
                    <span class="material-symbols-outlined text-purple-500 text-sm">category</span>
                    <span class="text-xs font-semibold text-purple-600 uppercase">Category</span>
                </div>
                <p id="stat-category" class="text-base font-bold">-</p>
                <p id="stat-category-score" class="text-xs text-subtext-light">0/0</p>
            </div>
        </div>
    </div>
</section>

<!-- Complete Modal -->
<div id="complete-modal" class="hidden fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 overflow-y-auto">
    <div class="bg-white rounded-2xl shadow-2xl max-w-lg w-full p-6 my-8">
        <div class="text-center mb-4">
            <div class="w-14 h-14 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <span class="material-symbols-outlined text-green-500 text-2xl">check_circle</span>
            </div>
            <h3 class="text-xl font-bold">Style Profile Complete!</h3>
            <p class="text-subtext-light text-sm">Here's what we learned about your preferences</p>
        </div>
        <div id="final-stats" class="text-left max-h-[60vh] overflow-y-auto pr-2"></div>
        <button onclick="showSetup()" class="w-full bg-gray-900 text-white py-3 rounded-xl font-semibold hover:bg-gray-800 transition-colors mt-4">
            Start New Session
        </button>
    </div>
</div>

<script>
const COLORS = [
    { name: 'black', bg: 'bg-black' },
    { name: 'white', bg: 'bg-white border border-gray-200' },
    { name: 'gray', bg: 'bg-gray-400' },
    { name: 'navy', bg: 'bg-blue-900' },
    { name: 'blue', bg: 'bg-blue-500' },
    { name: 'red', bg: 'bg-red-500' },
    { name: 'green', bg: 'bg-green-600' },
    { name: 'brown', bg: 'bg-amber-700' },
    { name: 'pink', bg: 'bg-pink-400' },
    { name: 'yellow', bg: 'bg-yellow-400' },
    { name: 'orange', bg: 'bg-orange-400' },
    { name: 'purple', bg: 'bg-purple-500' },
    { name: 'neon', bg: 'bg-lime-400' },
];

// State
let selectedColors = new Set();
let selectedCoreTypes = new Set();
let selectedSize = 'M';
let selectedStyles = new Set();
let selectedNecklines = new Set();
let selectedMaterials = new Set();
let currentItem = null;
let nextItem = null;
let isAnimating = false;
let stats = {};

document.addEventListener('DOMContentLoaded', () => {
    initColorSelector();
    setupKeyboardControls();
});

function initColorSelector() {
    const container = document.getElementById('color-selector');
    COLORS.forEach(color => {
        const chip = document.createElement('button');
        chip.className = `group relative size-12 rounded-full ${color.bg} shadow-sm hover:scale-110 transition-transform`;
        chip.title = color.name;
        chip.innerHTML = `<span class="hidden group-hover:flex absolute inset-0 items-center justify-center ${color.name === 'white' || color.name === 'yellow' || color.name === 'neon' ? 'text-gray-800' : 'text-white'} material-symbols-outlined text-lg">close</span>`;
        chip.onclick = () => toggleColor(color.name, chip);
        container.appendChild(chip);
    });
}

function toggleColor(name, chip) {
    if (selectedColors.has(name)) {
        selectedColors.delete(name);
        chip.classList.remove('ring-2', 'ring-primary', 'ring-offset-2', 'opacity-50', 'grayscale');
        chip.querySelector('span').classList.add('hidden');
        chip.querySelector('span').classList.remove('flex');
    } else {
        selectedColors.add(name);
        chip.classList.add('ring-2', 'ring-primary', 'ring-offset-2', 'opacity-50', 'grayscale');
        chip.querySelector('span').classList.remove('hidden');
        chip.querySelector('span').classList.add('flex');
    }
}

function toggleCoreType(type) {
    const btn = document.querySelector(`[data-type="${type}"]`);
    if (selectedCoreTypes.has(type)) {
        selectedCoreTypes.delete(type);
        btn.classList.remove('border-2', 'border-black', 'bg-black', 'text-white');
        btn.classList.add('border-gray-200', 'bg-white');
        btn.querySelector('img').classList.remove('invert', 'brightness-0');
        btn.querySelectorAll('span').forEach(s => s.classList.remove('text-white', 'text-gray-300'));
    } else {
        selectedCoreTypes.add(type);
        btn.classList.add('border-2', 'border-black', 'bg-black', 'text-white');
        btn.classList.remove('border-gray-200', 'bg-white');
        btn.querySelector('img').classList.add('invert', 'brightness-0');
        btn.querySelectorAll('span')[0].classList.add('text-white');
        btn.querySelectorAll('span')[1].classList.add('text-gray-300');
    }
    // Show/hide T-shirt styles section
    if (selectedCoreTypes.has('tshirts')) {
        document.getElementById('tshirt-styles-section').classList.remove('hidden');
    } else {
        document.getElementById('tshirt-styles-section').classList.add('hidden');
    }
}

function selectSize(size) {
    selectedSize = size;
    document.querySelectorAll('.size-btn').forEach(btn => {
        if (btn.dataset.size === size) {
            btn.classList.add('text-white', 'bg-black', 'shadow-md');
            btn.classList.remove('text-text-main', 'hover:bg-gray-50');
        } else {
            btn.classList.remove('text-white', 'bg-black', 'shadow-md');
            btn.classList.add('text-text-main', 'hover:bg-gray-50');
        }
    });
}

function toggleStyle(style) {
    const btn = document.querySelector(`[data-style="${style}"]`);
    if (selectedStyles.has(style)) {
        selectedStyles.delete(style);
        btn.classList.remove('border-2', 'border-black');
        btn.classList.add('border-gray-200');
    } else {
        selectedStyles.add(style);
        btn.classList.add('border-2', 'border-black');
        btn.classList.remove('border-gray-200');
    }
}

function toggleNeckline(neckline) {
    console.log('toggleNeckline called:', neckline);
    const btn = document.querySelector(`[data-neckline="${neckline}"]`);
    const imgDiv = btn.querySelector('div');
    if (selectedNecklines.has(neckline)) {
        selectedNecklines.delete(neckline);
        imgDiv.classList.remove('ring-2', 'ring-black', 'ring-offset-2');
    } else {
        selectedNecklines.add(neckline);
        imgDiv.classList.add('ring-2', 'ring-black', 'ring-offset-2');
    }
    console.log('selectedNecklines now:', Array.from(selectedNecklines));
}

function toggleMaterial(material) {
    const btn = document.querySelector(`[data-material="${material}"]`);
    const img = btn.querySelector('img');
    const label = btn.querySelector('.material-label');
    if (selectedMaterials.has(material)) {
        selectedMaterials.delete(material);
        img.classList.remove('grayscale', 'opacity-50');
        label.classList.remove('bg-gray-200', 'text-gray-500', 'line-through');
        label.classList.add('bg-white', 'text-black');
    } else {
        selectedMaterials.add(material);
        img.classList.add('grayscale', 'opacity-50');
        label.classList.add('bg-gray-200', 'text-gray-500', 'line-through');
        label.classList.remove('bg-white', 'text-black');
    }
}

function setupKeyboardControls() {
    document.addEventListener('keydown', (e) => {
        if (isAnimating) return;
        if (e.key === 'ArrowLeft') swipe('dislike');
        else if (e.key === 'ArrowRight') swipe('like');
        else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') swipe('skip');
    });
}

async function beginSession() {
    try {
        const response = await fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: 'default_user',
                colors_to_avoid: Array.from(selectedColors),
                materials_to_avoid: Array.from(selectedMaterials),  // Prefilter: avoid these fabrics
                selected_styles: Array.from(selectedStyles),  // Prefilter: only show these styles
                selected_necklines: Array.from(selectedNecklines),  // Prefilter: only show these necklines
                // Include full setup preferences (for future use)
                preferences: {
                    coreTypes: Array.from(selectedCoreTypes),
                    size: selectedSize,
                    styles: Array.from(selectedStyles),
                    necklines: Array.from(selectedNecklines),
                    materialsToAvoid: Array.from(selectedMaterials)
                }
            })
        });
        const data = await response.json();

        if (data.status === 'started') {
            currentItem = data.item;
            nextItem = data.next_item;
            stats = data.stats;

            document.getElementById('setup-screen').classList.add('hidden');
            document.getElementById('setup-footer').classList.add('hidden');
            document.getElementById('swipe-screen').classList.remove('hidden');
            document.getElementById('stats-section').classList.remove('hidden');

            updateCard(data.item);
            updateStats(data.stats);

            if (data.next_item) {
                document.getElementById('back-image').src = data.next_item.image_url;
            }
        }
    } catch (error) {
        console.error('Failed to start:', error);
        alert('Failed to start session. Check console.');
    }
}

async function swipe(action) {
    if (isAnimating || !currentItem) return;
    isAnimating = true;

    const card = document.getElementById('main-card');
    const stampLike = document.getElementById('stamp-like');
    const stampDislike = document.getElementById('stamp-dislike');

    if (action === 'like') {
        stampLike.classList.add('show-stamp');
        card.classList.add('swipe-right');
    } else if (action === 'dislike') {
        stampDislike.classList.add('show-stamp');
        card.classList.add('swipe-left');
    }

    try {
        const response = await fetch('/api/swipe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: 'default_user', action })
        });
        const data = await response.json();
        stats = data.stats;

        setTimeout(() => {
            card.classList.remove('swipe-right', 'swipe-left');
            stampLike.classList.remove('show-stamp');
            stampDislike.classList.remove('show-stamp');

            if (data.session_complete) {
                showCompleteModal(data.stats);
            } else if (data.item) {
                // The back card already has the next image preloaded
                // Now update front card to show what was in back card
                if (nextItem) {
                    updateCard(nextItem);
                } else {
                    updateCard(data.item);
                }

                // Update state
                currentItem = nextItem || data.item;
                nextItem = data.next_item;

                // Preload the NEXT next image into back card
                if (data.next_item) {
                    document.getElementById('back-image').src = data.next_item.image_url;
                }

                card.classList.add('card-enter');
                updateStats(data.stats);
                setTimeout(() => {
                    card.classList.remove('card-enter');
                    isAnimating = false;
                }, 300);
            }
        }, 400);
    } catch (error) {
        console.error('Swipe failed:', error);
        isAnimating = false;
    }
}

function updateCard(item) {
    document.getElementById('item-image').src = item.image_url;
    document.getElementById('item-brand').textContent = item.brand || 'Unknown';
    document.getElementById('item-category').textContent = item.category;
    document.getElementById('item-fit').textContent = item.fit || 'Regular';
    document.getElementById('archetype-badge').textContent = item.dominant_archetype || 'classic';
    document.getElementById('tag-color').textContent = item.color || '';
    document.getElementById('tag-fabric').textContent = item.fabric || '';
    document.getElementById('cluster-info').textContent = `Cluster: ${item.cluster_profile}`;
}

function updateStats(data) {
    document.getElementById('swipe-count').textContent = data.total_swipes || 0;
    document.getElementById('coverage-text').textContent = `Coverage: ${data.coverage || '0%'}`;
    document.getElementById('stability-text').textContent = `Stability: ${(data.taste_stability || 0).toFixed(2)}`;

    const progress = Math.min((data.total_swipes || 0) / 40 * 100, 100);
    document.getElementById('progress-bar').style.width = `${progress}%`;

    const headerText = document.getElementById('header-text');
    if (data.total_swipes < 10) headerText.textContent = "Let's explore some styles...";
    else if (data.total_swipes < 20) headerText.textContent = "Getting to know your preferences...";
    else if (data.total_swipes < 30) headerText.textContent = "Your style is taking shape!";
    else headerText.textContent = "Fine-tuning your profile...";

    const styleProfile = data.style_profile || {};
    const archetypes = styleProfile.archetypes || [];
    const anchors = styleProfile.visual_anchors || [];

    if (archetypes.length > 0) {
        const top = archetypes[0];
        document.getElementById('stat-archetype').textContent = top.archetype;
        document.getElementById('stat-archetype-score').textContent = `${top.preference >= 0 ? '+' : ''}${top.preference.toFixed(3)}`;
    }

    const likedAnchors = anchors.filter(a => a.direction === 'likes').slice(0, 2);
    if (likedAnchors.length > 0) {
        document.getElementById('stat-anchor-likes').textContent = likedAnchors.map(a => a.anchor).join(', ');
    }
    document.getElementById('stat-anchor-likes-count').textContent = `${data.likes || 0} items`;

    const dislikedAnchors = anchors.filter(a => a.direction === 'dislikes').slice(0, 2);
    if (dislikedAnchors.length > 0) {
        document.getElementById('stat-anchor-dislikes').textContent = dislikedAnchors.map(a => a.anchor).join(', ');
    }
    document.getElementById('stat-anchor-dislikes-count').textContent = `${data.dislikes || 0} items`;

    const categories = data.attribute_preferences?.category || [];
    if (categories.length > 0) {
        const topCat = categories[0];
        document.getElementById('stat-category').textContent = topCat.value.replace(' T-shirts', '');
        document.getElementById('stat-category-score').textContent = `${topCat.likes}/${topCat.total} (${topCat.score.toFixed(2)})`;
    }
}

function showCompleteModal(data) {
    // Helper to format numbers cleanly (avoid floating point errors like 0.19999999)
    const fmt = (n) => Number.isInteger(n) ? n : Math.round(n * 10) / 10;

    const finalStats = document.getElementById('final-stats');
    const styleProfile = data.style_profile || {};
    const archetypes = styleProfile.archetypes || [];
    const anchors = styleProfile.visual_anchors || [];
    const brands = data.brand_preferences || {};
    const attrs = data.attribute_preferences || {};

    const likeRate = data.total_swipes > 0 ? Math.round(data.likes / data.total_swipes * 100) : 0;

    // Build archetypes section
    let archetypeHtml = archetypes.slice(0, 4).map(a => {
        const pref = a.preference;
        const barWidth = Math.min(Math.abs(pref) * 200, 100);
        const isPositive = pref >= 0;
        return `<div class="flex items-center justify-between py-1">
            <span class="text-sm capitalize w-32">${a.archetype.replace('_', ' ')}</span>
            <div class="flex-1 mx-2 h-2 bg-gray-200 rounded overflow-hidden">
                <div class="h-full ${isPositive ? 'bg-green-400' : 'bg-red-400'}" style="width: ${barWidth}%"></div>
            </div>
            <span class="text-xs font-mono w-16 text-right ${isPositive ? 'text-green-600' : 'text-red-600'}">${pref >= 0 ? '+' : ''}${pref.toFixed(3)}</span>
        </div>`;
    }).join('');

    // Build anchors section (likes vs dislikes)
    const likedAnchors = anchors.filter(a => a.direction === 'likes');
    const dislikedAnchors = anchors.filter(a => a.direction === 'dislikes');

    // Build brands section
    const brandEntries = Object.entries(brands).sort((a, b) => b[1].score - a[1].score);
    let brandsHtml = brandEntries.slice(0, 5).map(([name, info]) => {
        const pct = Math.round(info.score * 100);
        return `<span class="inline-flex items-center bg-gray-100 text-gray-700 text-xs px-2 py-1 rounded-full mr-1 mb-1">
            ${name} <span class="ml-1 text-gray-500">${fmt(info.likes)}/${fmt(info.total)}</span>
        </span>`;
    }).join('');

    // Build colors section
    const colors = attrs.color || [];
    let colorsHtml = colors.slice(0, 6).map(c => {
        const pct = Math.round(c.score * 100);
        const isLiked = c.score >= 0.5;
        return `<span class="inline-flex items-center ${isLiked ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'} text-xs px-2 py-1 rounded-full mr-1 mb-1">
            ${c.value} <span class="ml-1 opacity-70">${fmt(c.likes)}/${fmt(c.total)}</span>
        </span>`;
    }).join('');

    // Build fits section
    const fits = attrs.fit || [];
    let fitsHtml = fits.slice(0, 4).map(f => {
        const isLiked = f.score >= 0.5;
        return `<span class="inline-flex items-center ${isLiked ? 'bg-purple-50 text-purple-700' : 'bg-gray-100 text-gray-600'} text-xs px-2 py-1 rounded-full mr-1 mb-1">
            ${f.value} <span class="ml-1 opacity-70">${Math.round(f.score * 100)}%</span>
        </span>`;
    }).join('');

    // Build fabrics section
    const fabrics = attrs.fabric || [];
    let fabricsHtml = fabrics.slice(0, 4).map(f => {
        const isLiked = f.score >= 0.5;
        return `<span class="inline-flex items-center ${isLiked ? 'bg-amber-50 text-amber-700' : 'bg-gray-100 text-gray-600'} text-xs px-2 py-1 rounded-full mr-1 mb-1">
            ${f.value} <span class="ml-1 opacity-70">${Math.round(f.score * 100)}%</span>
        </span>`;
    }).join('');

    // Build categories section
    const categories = attrs.category || [];
    let categoriesHtml = categories.map(c => {
        const pct = Math.round(c.score * 100);
        const barColor = c.score >= 0.5 ? 'bg-green-400' : 'bg-red-400';
        return `<div class="flex items-center justify-between py-1">
            <span class="text-sm">${c.value.replace(' T-shirts', '')}</span>
            <div class="flex items-center gap-2">
                <div class="w-20 h-2 bg-gray-200 rounded overflow-hidden">
                    <div class="${barColor} h-full" style="width: ${pct}%"></div>
                </div>
                <span class="text-xs font-mono w-12">${fmt(c.likes)}/${fmt(c.total)}</span>
            </div>
        </div>`;
    }).join('');

    finalStats.innerHTML = `
        <!-- Overview -->
        <div class="grid grid-cols-4 gap-2 mb-4 text-center">
            <div class="bg-gray-50 rounded-lg p-2">
                <p class="text-lg font-bold text-primary">${data.total_swipes}</p>
                <p class="text-[10px] text-gray-500 uppercase">Swipes</p>
            </div>
            <div class="bg-green-50 rounded-lg p-2">
                <p class="text-lg font-bold text-green-600">${data.likes}</p>
                <p class="text-[10px] text-gray-500 uppercase">Likes</p>
            </div>
            <div class="bg-red-50 rounded-lg p-2">
                <p class="text-lg font-bold text-red-500">${data.dislikes}</p>
                <p class="text-[10px] text-gray-500 uppercase">Dislikes</p>
            </div>
            <div class="bg-purple-50 rounded-lg p-2">
                <p class="text-lg font-bold text-purple-600">${likeRate}%</p>
                <p class="text-[10px] text-gray-500 uppercase">Like Rate</p>
            </div>
        </div>

        <!-- Style Archetypes -->
        <div class="mb-4">
            <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                <span class="w-2 h-2 bg-gray-500 rounded-full"></span> Style Archetypes
            </h4>
            <div class="bg-gray-50 rounded-lg p-3">
                ${archetypeHtml || '<p class="text-xs text-gray-400">Not enough data</p>'}
            </div>
        </div>

        <!-- Visual Anchors -->
        <div class="mb-4">
            <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                <span class="w-2 h-2 bg-green-500 rounded-full"></span> Visual Preferences
            </h4>
            <div class="grid grid-cols-2 gap-2">
                <div class="bg-green-50 rounded-lg p-2">
                    <p class="text-[10px] text-green-600 font-semibold mb-1">DRAWN TO</p>
                    <p class="text-sm font-medium">${likedAnchors.map(a => a.anchor).join(', ') || '-'}</p>
                </div>
                <div class="bg-red-50 rounded-lg p-2">
                    <p class="text-[10px] text-red-600 font-semibold mb-1">AVOIDS</p>
                    <p class="text-sm font-medium">${dislikedAnchors.map(a => a.anchor).join(', ') || '-'}</p>
                </div>
            </div>
        </div>

        <!-- Categories -->
        <div class="mb-4">
            <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                <span class="w-2 h-2 bg-purple-500 rounded-full"></span> Category Breakdown
            </h4>
            <div class="bg-gray-50 rounded-lg p-3">
                ${categoriesHtml || '<p class="text-xs text-gray-400">Not enough data</p>'}
            </div>
        </div>

        <!-- Brands -->
        <div class="mb-4">
            <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                <span class="w-2 h-2 bg-gray-500 rounded-full"></span> Top Brands
            </h4>
            <div class="flex flex-wrap">
                ${brandsHtml || '<span class="text-xs text-gray-400">Not enough data</span>'}
            </div>
        </div>

        <!-- Colors -->
        <div class="mb-4">
            <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                <span class="w-2 h-2 bg-pink-500 rounded-full"></span> Color Preferences
            </h4>
            <div class="flex flex-wrap">
                ${colorsHtml || '<span class="text-xs text-gray-400">Not enough data</span>'}
            </div>
        </div>

        <!-- Fit & Fabric -->
        <div class="grid grid-cols-2 gap-3 mb-4">
            <div>
                <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2">Fit</h4>
                <div class="flex flex-wrap">${fitsHtml || '<span class="text-xs text-gray-400">-</span>'}</div>
            </div>
            <div>
                <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2">Fabric</h4>
                <div class="flex flex-wrap">${fabricsHtml || '<span class="text-xs text-gray-400">-</span>'}</div>
            </div>
        </div>

        <!-- Technical -->
        <div class="border-t border-gray-200 pt-3 mt-3">
            <div class="flex justify-between text-xs text-gray-500">
                <span>Coverage: ${data.coverage}</span>
                <span>Stability: ${(data.taste_stability || 0).toFixed(3)}</span>
                <span>Clusters: ${data.clusters_rejected || 0} rejected</span>
            </div>
        </div>
    `;
    document.getElementById('complete-modal').classList.remove('hidden');
    isAnimating = false;
}

function showSetup() {
    document.getElementById('complete-modal').classList.add('hidden');
    document.getElementById('swipe-screen').classList.add('hidden');
    document.getElementById('stats-section').classList.add('hidden');
    document.getElementById('setup-screen').classList.remove('hidden');
    document.getElementById('setup-footer').classList.remove('hidden');
    currentItem = null;
    nextItem = null;
    stats = {};
    // Reset selections
    selectedColors = new Set();
    selectedCoreTypes = new Set();
    selectedSize = 'M';
    selectedStyles = new Set();
    selectedNecklines = new Set();
    selectedMaterials = new Set();
    // Reset UI
    document.querySelectorAll('.core-type-btn').forEach(btn => {
        btn.classList.remove('border-2', 'border-black', 'bg-black', 'text-white');
        btn.classList.add('border-gray-200', 'bg-white');
        btn.querySelector('img')?.classList.remove('invert', 'brightness-0');
    });
    document.querySelectorAll('.size-btn').forEach(btn => {
        btn.classList.remove('text-white', 'bg-black', 'shadow-md');
        btn.classList.add('text-text-main', 'hover:bg-gray-50');
    });
    document.querySelector('[data-size="M"]')?.classList.add('text-white', 'bg-black', 'shadow-md');
    document.getElementById('tshirt-styles-section')?.classList.add('hidden');
    // Reinit color selector
    const colorContainer = document.getElementById('color-selector');
    colorContainer.innerHTML = '';
    initColorSelector();
}
</script>

</body>
</html>'''


def get_four_choice_html() -> str:
    """Return the HTML template for the four-choice interface."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
    <title>Style Discovery - Pick Your Favorite</title>
    <script src="https://cdn.tailwindcss.com?plugins=forms,typography"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@200;300;400;500;600;700;800&family=Noto+Sans:wght@400;500;700&display=swap" rel="stylesheet"/>
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        "primary": "#1A1A1A",
                        "primary-dark": "#0A0A0A",
                        "background-light": "#f9fafb",
                        "surface-light": "#ffffff",
                        "text-main": "#131811",
                        "text-muted": "#6c7275",
                    },
                    fontFamily: {
                        display: ["Plus Jakarta Sans", "sans-serif"],
                        body: ["Noto Sans", "sans-serif"],
                    },
                }
            }
        };
    </script>
    <style>
        body::-webkit-scrollbar { display: none; }
        .no-scrollbar::-webkit-scrollbar { display: none; }
        .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }

        @keyframes cardFlyIn0 {
            0% { transform: translateX(-100%) translateY(-100%) scale(0.5); opacity: 0; }
            100% { transform: translateX(0) translateY(0) scale(1); opacity: 1; }
        }
        @keyframes cardFlyIn1 {
            0% { transform: translateX(100%) translateY(-100%) scale(0.5); opacity: 0; }
            100% { transform: translateX(0) translateY(0) scale(1); opacity: 1; }
        }
        @keyframes cardFlyIn2 {
            0% { transform: translateX(-100%) translateY(100%) scale(0.5); opacity: 0; }
            100% { transform: translateX(0) translateY(0) scale(1); opacity: 1; }
        }
        @keyframes cardFlyIn3 {
            0% { transform: translateX(100%) translateY(100%) scale(0.5); opacity: 0; }
            100% { transform: translateX(0) translateY(0) scale(1); opacity: 1; }
        }
        @keyframes cardSelect {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); box-shadow: 0 0 30px rgba(34, 197, 94, 0.5); }
            100% { transform: scale(0); opacity: 0; }
        }
        @keyframes cardFade {
            0% { opacity: 1; transform: scale(1); }
            100% { opacity: 0; transform: scale(0.8); }
        }

        .card-fly-0 { animation: cardFlyIn0 0.4s ease-out forwards; }
        .card-fly-1 { animation: cardFlyIn1 0.4s ease-out 0.05s forwards; opacity: 0; }
        .card-fly-2 { animation: cardFlyIn2 0.4s ease-out 0.1s forwards; opacity: 0; }
        .card-fly-3 { animation: cardFlyIn3 0.4s ease-out 0.15s forwards; opacity: 0; }
        .card-selected { animation: cardSelect 0.5s ease-out forwards; }
        .card-fade { animation: cardFade 0.3s ease-out forwards; }

        .choice-card {
            transition: all 0.2s ease;
            cursor: pointer;
        }
        .choice-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.15);
        }
    </style>
</head>
<body class="bg-background-light text-text-main font-display antialiased min-h-screen flex flex-col">

<!-- Header -->
<header class="sticky top-0 z-50 bg-surface-light border-b border-gray-100 px-6 py-3 shadow-sm">
    <div class="max-w-[1100px] mx-auto flex items-center justify-between">
        <h2 class="text-xl font-bold tracking-tight">Style Discovery</h2>
        <button onclick="showSetup()" class="text-sm font-medium text-text-muted hover:text-text-main transition-colors">
            Restart
        </button>
    </div>
</header>

<!-- Setup Screen -->
<div id="setup-screen" class="flex-grow flex flex-col items-center py-8 px-4 sm:px-6">
    <div class="w-full max-w-[960px] flex flex-col gap-10">

        <!-- Core Selection Section -->
        <div class="text-center">
            <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">CORE SELECTION</h2>
            <h3 class="text-2xl font-bold text-text-main">Which tops are you looking for?</h3>
        </div>

        <div class="flex justify-center" id="core-selection">
            <button onclick="toggleCoreType('tshirts')" data-type="tshirts" class="core-type-btn flex flex-col items-center justify-center p-4 rounded-xl border transition-all h-[180px] w-[200px] border-gray-200 bg-white hover:border-gray-300">
                <div class="w-16 h-16 mb-4 flex items-center justify-center">
                    <img src="/assets/Initial%20Screen%20Icons/TShirt%20Icons.png" alt="T-Shirts" class="object-contain h-full w-full"/>
                </div>
                <span class="font-bold text-sm mb-1">T-Shirts</span>
                <span class="text-[10px] text-gray-500">Casual everyday</span>
            </button>
        </div>

        <div class="border-t border-gray-100 my-2"></div>

        <!-- Size Section -->
        <div class="text-center">
            <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">MEASUREMENTS</h2>
            <h3 class="text-2xl font-bold text-text-main mb-8">What is your typical size?</h3>
            <div class="max-w-2xl mx-auto">
                <div class="flex items-center justify-between bg-white border border-gray-200 rounded-full p-1.5 h-14 shadow-sm" id="size-selector">
                    <button onclick="selectSize('XS')" data-size="XS" class="size-btn flex-1 h-full rounded-full text-xs font-bold transition-all flex items-center justify-center text-text-main hover:bg-gray-50">XS</button>
                    <button onclick="selectSize('S')" data-size="S" class="size-btn flex-1 h-full rounded-full text-xs font-bold transition-all flex items-center justify-center text-text-main hover:bg-gray-50">S</button>
                    <button onclick="selectSize('M')" data-size="M" class="size-btn flex-1 h-full rounded-full text-xs font-bold transition-all flex items-center justify-center text-white bg-black shadow-md">M</button>
                    <button onclick="selectSize('L')" data-size="L" class="size-btn flex-1 h-full rounded-full text-xs font-bold transition-all flex items-center justify-center text-text-main hover:bg-gray-50">L</button>
                    <button onclick="selectSize('XL')" data-size="XL" class="size-btn flex-1 h-full rounded-full text-xs font-bold transition-all flex items-center justify-center text-text-main hover:bg-gray-50">XL</button>
                    <button onclick="selectSize('2XL')" data-size="2XL" class="size-btn flex-1 h-full rounded-full text-xs font-bold transition-all flex items-center justify-center text-text-main hover:bg-gray-50">2XL</button>
                </div>
            </div>
        </div>

        <div class="border-t border-gray-100 my-2"></div>

        <!-- T-Shirt Style Section -->
        <div id="tshirt-styles-section" class="hidden">
            <div class="text-center mb-6">
                <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">T-SHIRT STYLES</h2>
                <h3 class="text-xl font-bold text-text-main">Select styles you wear</h3>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-5 gap-4" id="tshirt-styles">
                <div onclick="toggleStyle('plain')" data-style="plain" class="style-btn cursor-pointer rounded-xl border overflow-hidden transition-all border-gray-200 hover:border-gray-300">
                    <div class="w-full aspect-square bg-gray-50 flex items-center justify-center overflow-hidden">
                        <img src="/assets/Tees%20styles/Plain.png" alt="Plain" class="w-full h-full object-cover"/>
                    </div>
                    <div class="p-3 text-center border-t border-gray-100 bg-white">
                        <span class="text-sm font-bold text-gray-900">Plain</span>
                    </div>
                </div>
                <div onclick="toggleStyle('graphics')" data-style="graphics" class="style-btn cursor-pointer rounded-xl border overflow-hidden transition-all border-gray-200 hover:border-gray-300">
                    <div class="w-full aspect-square bg-gray-50 flex items-center justify-center overflow-hidden">
                        <img src="/assets/Tees%20styles/Graphic%20Tshirt.png" alt="Graphics" class="w-full h-full object-cover"/>
                    </div>
                    <div class="p-3 text-center border-t border-gray-100 bg-white">
                        <span class="text-sm font-bold text-gray-900">Graphics</span>
                    </div>
                </div>
                <div onclick="toggleStyle('smalllogos')" data-style="smalllogos" class="style-btn cursor-pointer rounded-xl border overflow-hidden transition-all border-gray-200 hover:border-gray-300">
                    <div class="w-full aspect-square bg-gray-50 flex items-center justify-center overflow-hidden">
                        <img src="/assets/Tees%20styles/Small%20Logos.png" alt="Small Logos" class="w-full h-full object-cover"/>
                    </div>
                    <div class="p-3 text-center border-t border-gray-100 bg-white">
                        <span class="text-sm font-bold text-gray-900">Small Logos</span>
                    </div>
                </div>
                <div onclick="toggleStyle('athletic')" data-style="athletic" class="style-btn cursor-pointer rounded-xl border overflow-hidden transition-all border-gray-200 hover:border-gray-300">
                    <div class="w-full aspect-square bg-gray-50 flex items-center justify-center overflow-hidden">
                        <img src="/assets/Tees%20styles/Atheletic%20Tshirt.png" alt="Athletic" class="w-full h-full object-cover"/>
                    </div>
                    <div class="p-3 text-center border-t border-gray-100 bg-white">
                        <span class="text-sm font-bold text-gray-900">Athletic</span>
                    </div>
                </div>
                <div onclick="toggleStyle('pocket')" data-style="pocket" class="style-btn cursor-pointer rounded-xl border overflow-hidden transition-all border-gray-200 hover:border-gray-300">
                    <div class="w-full aspect-square bg-gray-50 flex items-center justify-center overflow-hidden">
                        <img src="/assets/Tees%20styles/Pocket%20Tshirt.png" alt="Pocket" class="w-full h-full object-cover"/>
                    </div>
                    <div class="p-3 text-center border-t border-gray-100 bg-white">
                        <span class="text-sm font-bold text-gray-900">Pocket</span>
                    </div>
                </div>
            </div>

            <div class="border-t border-gray-100 my-6"></div>

            <!-- Neckline Section -->
            <div class="text-center mb-6">
                <h3 class="text-lg font-bold text-text-main">Neckline Preference</h3>
            </div>
            <div class="grid grid-cols-3 gap-4 max-w-md mx-auto" id="neckline-selector">
                <div onclick="toggleNeckline('crew')" data-neckline="crew" class="neckline-btn cursor-pointer flex flex-col items-center">
                    <div class="w-full aspect-[4/3] rounded-xl bg-gray-100 overflow-hidden flex items-center justify-center transition-all hover:bg-gray-200">
                        <img src="/assets/neck%20Styles/Crew.png" alt="Crew" class="w-full h-full object-cover"/>
                    </div>
                    <span class="mt-2 text-sm font-semibold text-gray-900">Crew</span>
                </div>
                <div onclick="toggleNeckline('vneck')" data-neckline="vneck" class="neckline-btn cursor-pointer flex flex-col items-center">
                    <div class="w-full aspect-[4/3] rounded-xl bg-gray-100 overflow-hidden flex items-center justify-center transition-all hover:bg-gray-200">
                        <img src="/assets/neck%20Styles/Vneck.png" alt="V-Neck" class="w-full h-full object-cover"/>
                    </div>
                    <span class="mt-2 text-sm font-semibold text-gray-900">V-Neck</span>
                </div>
                <div onclick="toggleNeckline('henley')" data-neckline="henley" class="neckline-btn cursor-pointer flex flex-col items-center">
                    <div class="w-full aspect-[4/3] rounded-xl bg-gray-100 overflow-hidden flex items-center justify-center transition-all hover:bg-gray-200">
                        <img src="/assets/neck%20Styles/henley.png" alt="Henley" class="w-full h-full object-cover"/>
                    </div>
                    <span class="mt-2 text-sm font-semibold text-gray-900">Henley</span>
                </div>
            </div>

            <div class="border-t border-gray-100 my-6"></div>

            <!-- Sleeve Length Section -->
            <div class="text-center mb-6">
                <h3 class="text-lg font-bold text-text-main">Sleeve Length</h3>
            </div>
            <div class="grid grid-cols-2 gap-4 max-w-md mx-auto" id="sleeve-selector">
                <div onclick="toggleSleeve('short')" data-sleeve="short" class="sleeve-btn cursor-pointer flex flex-col items-center">
                    <div class="w-full aspect-[4/3] rounded-xl bg-gray-100 overflow-hidden flex items-center justify-center transition-all hover:bg-gray-200">
                        <img src="/assets/Tshirt%20Style/Short%20Sleeves.png" alt="Short Sleeve" class="w-full h-full object-cover"/>
                    </div>
                    <span class="mt-2 text-sm font-semibold text-gray-900">Short Sleeve</span>
                </div>
                <div onclick="toggleSleeve('long')" data-sleeve="long" class="sleeve-btn cursor-pointer flex flex-col items-center">
                    <div class="w-full aspect-[4/3] rounded-xl bg-gray-100 overflow-hidden flex items-center justify-center transition-all hover:bg-gray-200">
                        <img src="/assets/Tshirt%20Style/Long%20sleeves.png" alt="Long Sleeve" class="w-full h-full object-cover"/>
                    </div>
                    <span class="mt-2 text-sm font-semibold text-gray-900">Long Sleeve</span>
                </div>
            </div>
        </div>

        <div class="border-t border-gray-100 my-2"></div>

        <!-- Colors to Avoid -->
        <div class="text-center">
            <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">PREFERENCES</h2>
            <h3 class="text-2xl font-bold text-text-main mb-2">Colors to Avoid</h3>
            <p class="text-gray-500 text-sm mb-8">Tap any color you dislike to cross it out.</p>
            <div class="flex justify-center">
                <div id="color-selector" class="flex flex-wrap justify-center gap-4 max-w-[700px]"></div>
            </div>
        </div>

        <div class="border-t border-gray-100 my-2"></div>

        <!-- Materials to Avoid -->
        <div class="text-center">
            <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">FABRICS</h2>
            <h3 class="text-2xl font-bold text-text-main mb-8">Materials to Avoid</h3>
            <div class="grid grid-cols-2 md:grid-cols-3 gap-6" id="materials-selector">
                <button onclick="toggleMaterial('polyester')" data-material="polyester" class="material-btn relative group overflow-hidden rounded-xl aspect-[16/9] transition-transform hover:scale-[1.02]">
                    <img src="/assets/Materials/Polyster.png" alt="Polyester" class="w-full h-full object-cover"/>
                    <div class="absolute inset-0 flex items-center justify-center">
                        <span class="material-label px-4 py-1.5 rounded-full text-xs font-bold shadow-sm bg-white text-black">Polyester</span>
                    </div>
                </button>
                <button onclick="toggleMaterial('wool')" data-material="wool" class="material-btn relative group overflow-hidden rounded-xl aspect-[16/9] transition-transform hover:scale-[1.02]">
                    <img src="/assets/Materials/Wool.png" alt="Wool" class="w-full h-full object-cover"/>
                    <div class="absolute inset-0 flex items-center justify-center">
                        <span class="material-label px-4 py-1.5 rounded-full text-xs font-bold shadow-sm bg-white text-black">Wool</span>
                    </div>
                </button>
                <button onclick="toggleMaterial('linen')" data-material="linen" class="material-btn relative group overflow-hidden rounded-xl aspect-[16/9] transition-transform hover:scale-[1.02]">
                    <img src="/assets/Materials/Linen%20rectangle.png" alt="Linen" class="w-full h-full object-cover"/>
                    <div class="absolute inset-0 flex items-center justify-center">
                        <span class="material-label px-4 py-1.5 rounded-full text-xs font-bold shadow-sm bg-white text-black">Linen</span>
                    </div>
                </button>
                <button onclick="toggleMaterial('silk')" data-material="silk" class="material-btn relative group overflow-hidden rounded-xl aspect-[16/9] transition-transform hover:scale-[1.02]">
                    <img src="/assets/Materials/Silk.png" alt="Silk" class="w-full h-full object-cover"/>
                    <div class="absolute inset-0 flex items-center justify-center">
                        <span class="material-label px-4 py-1.5 rounded-full text-xs font-bold shadow-sm bg-white text-black">Silk</span>
                    </div>
                </button>
                <button onclick="toggleMaterial('synthetics')" data-material="synthetics" class="material-btn relative group overflow-hidden rounded-xl aspect-[16/9] transition-transform hover:scale-[1.02]">
                    <img src="/assets/Materials/Synthetics.png" alt="Synthetics" class="w-full h-full object-cover"/>
                    <div class="absolute inset-0 flex items-center justify-center">
                        <span class="material-label px-4 py-1.5 rounded-full text-xs font-bold shadow-sm bg-white text-black">Synthetics</span>
                    </div>
                </button>
                <button onclick="toggleMaterial('leather')" data-material="leather" class="material-btn relative group overflow-hidden rounded-xl aspect-[16/9] transition-transform hover:scale-[1.02]">
                    <img src="/assets/Materials/Leather.png" alt="Leather" class="w-full h-full object-cover"/>
                    <div class="absolute inset-0 flex items-center justify-center">
                        <span class="material-label px-4 py-1.5 rounded-full text-xs font-bold shadow-sm bg-white text-black">Leather</span>
                    </div>
                </button>
            </div>
        </div>

        <div class="h-24"></div>
    </div>
</div>

<!-- Sticky Bottom Button -->
<div id="setup-footer" class="sticky bottom-0 w-full bg-white/95 backdrop-blur-md border-t border-gray-200 py-4 px-6 shadow-lg z-40">
    <div class="max-w-[960px] mx-auto flex justify-center">
        <button onclick="startSession()" class="bg-primary hover:bg-primary-dark text-white font-bold py-4 px-12 rounded-xl shadow-md transition-all transform active:scale-95 flex items-center gap-2 text-lg">
            Start Style Quiz
            <span class="material-symbols-outlined">arrow_forward</span>
        </button>
    </div>
</div>

<!-- Main Quiz Screen -->
<main id="quiz-screen" class="hidden flex-grow flex flex-col items-center px-4 py-6">
    <!-- Category Header -->
    <div class="w-full max-w-2xl mb-6">
        <div class="text-center">
            <span class="text-xs font-medium text-gray-400 uppercase tracking-wider">Category <span id="cat-index">1</span> of <span id="cat-total">4</span></span>
            <h2 id="current-category" class="text-2xl font-bold text-gray-900 mt-1">Graphics T-shirts</h2>
        </div>
        <!-- Progress bar -->
        <div class="mt-4 flex items-center gap-3">
            <div class="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                <div id="progress-bar" class="h-full bg-gradient-to-r from-gray-700 to-gray-900 rounded-full transition-all duration-500" style="width: 0%"></div>
            </div>
            <span class="text-xs text-gray-500 min-w-[60px]">Round <span id="round-num">1</span></span>
        </div>
    </div>

    <!-- Instruction -->
    <h3 class="text-lg font-medium text-gray-600 text-center mb-6">Which do you like most?</h3>

    <!-- 2x2 Grid -->
    <div id="cards-grid" class="grid grid-cols-2 gap-4 w-full max-w-2xl">
        <!-- Cards will be inserted here -->
    </div>

    <!-- Skip Button -->
    <button onclick="skipAll()" class="mt-6 text-gray-400 hover:text-gray-600 text-sm font-medium flex items-center gap-1">
        <span class="material-symbols-outlined text-base">refresh</span>
        None of these
    </button>
</main>

<!-- Complete Modal -->
<div id="complete-modal" class="hidden fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 overflow-y-auto">
    <div class="bg-white rounded-2xl shadow-2xl max-w-lg w-full p-6 my-8">
        <div class="text-center mb-4">
            <div class="w-14 h-14 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <span class="material-symbols-outlined text-green-500 text-2xl">check_circle</span>
            </div>
            <h3 class="text-xl font-bold">Style Profile Complete!</h3>
            <p class="text-gray-500 text-sm">Completed in just <span id="final-rounds">0</span> rounds</p>
        </div>
        <div id="final-stats" class="text-left max-h-[60vh] overflow-y-auto pr-2"></div>
        <button onclick="showSetup()" class="w-full bg-gray-900 text-white py-3 rounded-xl font-semibold hover:bg-gray-800 mt-4">
            Start New Quiz
        </button>
    </div>
</div>

<script>
const COLORS = [
    { name: 'black', bg: 'bg-black' },
    { name: 'white', bg: 'bg-white border border-gray-200' },
    { name: 'gray', bg: 'bg-gray-400' },
    { name: 'navy', bg: 'bg-blue-900' },
    { name: 'blue', bg: 'bg-blue-500' },
    { name: 'red', bg: 'bg-red-500' },
    { name: 'green', bg: 'bg-green-600' },
    { name: 'brown', bg: 'bg-amber-700' },
    { name: 'pink', bg: 'bg-pink-400' },
    { name: 'yellow', bg: 'bg-yellow-400' },
    { name: 'orange', bg: 'bg-orange-400' },
    { name: 'purple', bg: 'bg-purple-500' },
    { name: 'neon', bg: 'bg-lime-400' },
];

// State
let selectedColors = new Set();
let selectedCoreTypes = new Set();
let selectedSize = 'M';
let selectedStyles = new Set();
let selectedNecklines = new Set();
let selectedSleeves = new Set();
let selectedMaterials = new Set();
let currentItems = [];
let isAnimating = false;
let currentTestInfo = null;
let predictionStats = { correct: 0, total: 0 };

document.addEventListener('DOMContentLoaded', () => {
    initColorSelector();
});

function initColorSelector() {
    const container = document.getElementById('color-selector');
    COLORS.forEach(color => {
        const chip = document.createElement('button');
        chip.className = `group relative size-12 rounded-full ${color.bg} shadow-sm hover:scale-110 transition-transform`;
        chip.title = color.name;
        chip.innerHTML = `<span class="hidden group-hover:flex absolute inset-0 items-center justify-center ${color.name === 'white' || color.name === 'yellow' || color.name === 'neon' ? 'text-gray-800' : 'text-white'} material-symbols-outlined text-lg">close</span>`;
        chip.onclick = () => toggleColor(color.name, chip);
        container.appendChild(chip);
    });
}

function toggleColor(name, chip) {
    if (selectedColors.has(name)) {
        selectedColors.delete(name);
        chip.classList.remove('ring-2', 'ring-primary', 'ring-offset-2', 'opacity-50', 'grayscale');
        chip.querySelector('span').classList.add('hidden');
        chip.querySelector('span').classList.remove('flex');
    } else {
        selectedColors.add(name);
        chip.classList.add('ring-2', 'ring-primary', 'ring-offset-2', 'opacity-50', 'grayscale');
        chip.querySelector('span').classList.remove('hidden');
        chip.querySelector('span').classList.add('flex');
    }
}

function toggleCoreType(type) {
    const btn = document.querySelector(`[data-type="${type}"]`);
    if (selectedCoreTypes.has(type)) {
        selectedCoreTypes.delete(type);
        btn.classList.remove('border-2', 'border-black', 'bg-black', 'text-white');
        btn.classList.add('border-gray-200', 'bg-white');
        btn.querySelector('img').classList.remove('invert', 'brightness-0');
        btn.querySelectorAll('span').forEach(s => s.classList.remove('text-white', 'text-gray-300'));
    } else {
        selectedCoreTypes.add(type);
        btn.classList.add('border-2', 'border-black', 'bg-black', 'text-white');
        btn.classList.remove('border-gray-200', 'bg-white');
        btn.querySelector('img').classList.add('invert', 'brightness-0');
        btn.querySelectorAll('span')[0].classList.add('text-white');
        btn.querySelectorAll('span')[1].classList.add('text-gray-300');
    }
    // Show/hide T-shirt styles section
    if (selectedCoreTypes.has('tshirts')) {
        document.getElementById('tshirt-styles-section').classList.remove('hidden');
    } else {
        document.getElementById('tshirt-styles-section').classList.add('hidden');
    }
}

function selectSize(size) {
    selectedSize = size;
    document.querySelectorAll('.size-btn').forEach(btn => {
        if (btn.dataset.size === size) {
            btn.classList.add('text-white', 'bg-black', 'shadow-md');
            btn.classList.remove('text-text-main', 'hover:bg-gray-50');
        } else {
            btn.classList.remove('text-white', 'bg-black', 'shadow-md');
            btn.classList.add('text-text-main', 'hover:bg-gray-50');
        }
    });
}

function toggleStyle(style) {
    const btn = document.querySelector(`[data-style="${style}"]`);
    if (selectedStyles.has(style)) {
        selectedStyles.delete(style);
        btn.classList.remove('border-2', 'border-black');
        btn.classList.add('border-gray-200');
    } else {
        selectedStyles.add(style);
        btn.classList.add('border-2', 'border-black');
        btn.classList.remove('border-gray-200');
    }
}

function toggleNeckline(neckline) {
    console.log('toggleNeckline called:', neckline);
    const btn = document.querySelector(`[data-neckline="${neckline}"]`);
    const imgDiv = btn.querySelector('div');
    if (selectedNecklines.has(neckline)) {
        selectedNecklines.delete(neckline);
        imgDiv.classList.remove('ring-2', 'ring-black', 'ring-offset-2');
    } else {
        selectedNecklines.add(neckline);
        imgDiv.classList.add('ring-2', 'ring-black', 'ring-offset-2');
    }
    console.log('selectedNecklines now:', Array.from(selectedNecklines));
}

function toggleSleeve(sleeve) {
    console.log('toggleSleeve called:', sleeve);
    const btn = document.querySelector(`[data-sleeve="${sleeve}"]`);
    const imgDiv = btn.querySelector('div');
    if (selectedSleeves.has(sleeve)) {
        selectedSleeves.delete(sleeve);
        imgDiv.classList.remove('ring-2', 'ring-black', 'ring-offset-2');
    } else {
        selectedSleeves.add(sleeve);
        imgDiv.classList.add('ring-2', 'ring-black', 'ring-offset-2');
    }
    console.log('selectedSleeves now:', Array.from(selectedSleeves));
}

function toggleMaterial(material) {
    const btn = document.querySelector(`[data-material="${material}"]`);
    const img = btn.querySelector('img');
    const label = btn.querySelector('.material-label');
    if (selectedMaterials.has(material)) {
        selectedMaterials.delete(material);
        img.classList.remove('grayscale', 'opacity-50');
        label.classList.remove('bg-gray-200', 'text-gray-500', 'line-through');
        label.classList.add('bg-white', 'text-black');
    } else {
        selectedMaterials.add(material);
        img.classList.add('grayscale', 'opacity-50');
        label.classList.add('bg-gray-200', 'text-gray-500', 'line-through');
        label.classList.remove('bg-white', 'text-black');
    }
}

async function startSession() {
    try {
        const requestBody = {
            user_id: 'default_user',
            colors_to_avoid: Array.from(selectedColors),
            materials_to_avoid: Array.from(selectedMaterials),
            selected_styles: Array.from(selectedStyles),
            selected_necklines: Array.from(selectedNecklines),
            selected_sleeves: Array.from(selectedSleeves)
        };
        console.log('Starting session with filters:', requestBody);

        const response = await fetch('/api/four/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        const data = await response.json();
        console.log('Server response:', data);

        if (data.status === 'started') {
            currentItems = data.items;
            currentTestInfo = data.test_info;
            predictionStats = { correct: 0, total: 0 };

            document.getElementById('setup-screen').classList.add('hidden');
            document.getElementById('setup-footer').classList.add('hidden');
            document.getElementById('quiz-screen').classList.remove('hidden');

            renderCards(data.items);
            updateCategoryInfo(data.test_info);
            updateProgress(1);
        }
    } catch (error) {
        console.error('Failed to start:', error);
    }
}

function updateCategoryInfo(testInfo) {
    if (!testInfo) return;

    document.getElementById('current-category').textContent = testInfo.category || 'Testing...';
    document.getElementById('cat-index').textContent = testInfo.category_index || 1;
    document.getElementById('cat-total').textContent = testInfo.total_categories || 4;
}

// Internal prediction tracking (not shown to user)
function logPredictionResult(resultInfo) {
    if (!resultInfo || !resultInfo.had_prediction) return;

    // Log to console for debugging/internal metrics
    const status = resultInfo.prediction_correct ? 'CORRECT' : 'WRONG';
    console.log(`[Prediction ${status}] consecutive: ${resultInfo.consecutive_correct}, category_complete: ${resultInfo.category_complete}`);
}

function renderCards(items) {
    const grid = document.getElementById('cards-grid');
    grid.innerHTML = '';

    items.forEach((item, idx) => {
        const card = document.createElement('div');
        card.className = `choice-card bg-white rounded-2xl shadow-lg overflow-hidden card-fly-${idx}`;
        card.onclick = () => selectCard(item.id, idx);
        card.innerHTML = `
            <div class="aspect-square bg-gray-100 overflow-hidden">
                <img src="${item.image_url}" alt="${item.category}" class="w-full h-full object-contain"/>
            </div>
            <div class="p-3">
                <p class="font-bold text-sm truncate">${item.brand}</p>
                <p class="text-xs text-gray-500">${item.color} • ${item.category.replace(' T-shirts', '')}</p>
            </div>
        `;
        grid.appendChild(card);
    });
}

async function selectCard(itemId, idx) {
    if (isAnimating) return;
    isAnimating = true;

    const cards = document.querySelectorAll('.choice-card');

    // Animate selection
    cards.forEach((card, i) => {
        if (i === idx) {
            card.classList.add('card-selected');
        } else {
            card.classList.add('card-fade');
        }
    });

    // Wait for animation
    await new Promise(r => setTimeout(r, 400));

    try {
        const response = await fetch('/api/four/choose', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: 'default_user',
                winner_id: itemId
            })
        });
        const data = await response.json();

        // Track prediction internally (not shown to user)
        if (data.result_info && data.result_info.had_prediction) {
            predictionStats.total++;
            if (data.result_info.prediction_correct) {
                predictionStats.correct++;
            }
            logPredictionResult(data.result_info);
        }

        if (data.session_complete) {
            showComplete(data.stats, data.feed_preview);
        } else {
            currentItems = data.items;
            currentTestInfo = data.test_info;

            updateCategoryInfo(data.test_info);
            updateProgress(data.round);
            renderCards(data.items);
        }
    } catch (error) {
        console.error('Failed to record choice:', error);
    }

    isAnimating = false;
}

async function skipAll() {
    if (isAnimating) return;
    isAnimating = true;

    const cards = document.querySelectorAll('.choice-card');
    cards.forEach(card => card.classList.add('card-fade'));

    await new Promise(r => setTimeout(r, 300));

    try {
        const response = await fetch('/api/four/skip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: 'default_user' })
        });
        const data = await response.json();

        if (data.session_complete) {
            showComplete(data.stats, data.feed_preview);
        } else {
            currentItems = data.items;
            currentTestInfo = data.test_info;
            updateCategoryInfo(data.test_info);
            updateProgress(data.round);
            renderCards(data.items);
        }
    } catch (error) {
        console.error('Failed to skip:', error);
    }

    isAnimating = false;
}

function updateProgress(round) {
    document.getElementById('round-num').textContent = round;

    // Overall progress based on categories completed
    const totalCats = currentTestInfo?.total_categories || 4;
    const catIndex = currentTestInfo?.category_index || 1;
    const roundInCat = currentTestInfo?.round_in_category || 1;

    // Estimate progress: each category is a chunk, plus current progress within category
    const catProgress = ((catIndex - 1) + (roundInCat / 8)) / totalCats * 100;
    document.getElementById('progress-bar').style.width = `${Math.min(catProgress, 100)}%`;
}

function showComplete(stats, feedPreview = []) {
    document.getElementById('final-rounds').textContent = stats.rounds_completed || 0;

    const finalStats = document.getElementById('final-stats');
    const styleProfile = stats.style_profile || {};
    const archetypes = styleProfile.archetypes || [];
    const anchors = styleProfile.visual_anchors || [];
    const brands = stats.brand_preferences || {};

    // Count stats
    const catsCompleted = stats.categories_completed?.length || 0;
    const totalCats = stats.total_categories || 4;
    const likes = stats.likes || 0;
    const dislikes = stats.dislikes || 0;

    // Build archetypes section
    let archetypeHtml = archetypes.slice(0, 4).map(a => {
        const pref = a.preference;
        const barWidth = Math.min(Math.abs(pref) * 200, 100);
        const isPositive = pref >= 0;
        return `<div class="flex items-center justify-between py-1">
            <span class="text-sm capitalize w-28">${a.archetype.replace('_', ' ')}</span>
            <div class="flex-1 mx-2 h-2 bg-gray-200 rounded overflow-hidden">
                <div class="h-full ${isPositive ? 'bg-green-400' : 'bg-red-400'}" style="width: ${barWidth}%"></div>
            </div>
            <span class="text-xs font-mono w-14 text-right ${isPositive ? 'text-green-600' : 'text-red-600'}">${pref >= 0 ? '+' : ''}${pref.toFixed(3)}</span>
        </div>`;
    }).join('');

    // Visual anchors
    const likedAnchors = anchors.filter(a => a.direction === 'likes');
    const dislikedAnchors = anchors.filter(a => a.direction === 'dislikes');

    // Brands
    const brandEntries = Object.entries(brands).sort((a, b) => (b[1].score || 0) - (a[1].score || 0));
    let brandsHtml = brandEntries.slice(0, 6).map(([name, info]) => {
        const isLiked = (info.score || 0) >= 0.5;
        return `<span class="inline-flex items-center ${isLiked ? 'bg-green-50 text-green-700' : 'bg-gray-100 text-gray-600'} text-xs px-2 py-1 rounded-full mr-1 mb-1">
            ${name} <span class="ml-1 opacity-70">${info.likes || 0}/${info.total || info.likes || 0}</span>
        </span>`;
    }).join('');

    // Build category stats HTML with learned style preferences
    const categoryStats = stats.category_stats || {};
    let categoryStatsHtml = Object.entries(categoryStats).map(([cat, catData]) => {
        const status = catData.complete ? '✓' : '...';
        const statusColor = catData.complete ? 'text-green-600' : 'text-gray-400';

        // Get top cluster preference for this category
        const topPref = (catData.cluster_preferences || [])[0];
        const styleHint = topPref?.profile ? topPref.profile.slice(0, 30) : '';

        return `<div class="py-2 border-b border-gray-100 last:border-0">
            <div class="flex justify-between items-center">
                <div class="flex items-center gap-2">
                    <span class="${statusColor} font-bold">${status}</span>
                    <span class="text-sm font-medium">${cat}</span>
                </div>
                <span class="text-xs text-gray-400">${catData.rounds} rounds</span>
            </div>
            ${styleHint ? `<p class="text-xs text-gray-500 ml-6 mt-1">${styleHint}</p>` : ''}
        </div>`;
    }).join('');

    // Build cluster preferences summary (what visual styles the user likes)
    let clusterPrefsHtml = '';
    const allClusterPrefs = [];
    Object.entries(categoryStats).forEach(([cat, catData]) => {
        (catData.cluster_preferences || []).forEach(cp => {
            if (cp.score > 0.5) {
                allClusterPrefs.push({ category: cat, ...cp });
            }
        });
    });
    if (allClusterPrefs.length > 0) {
        clusterPrefsHtml = allClusterPrefs.slice(0, 4).map(cp => {
            const desc = cp.profile || `Cluster ${cp.cluster}`;
            return `<div class="bg-gray-50 rounded-lg p-2 mb-2">
                <p class="text-xs text-gray-500">${cp.category}</p>
                <p class="text-sm font-medium">${desc}</p>
                <p class="text-xs text-green-600">${Math.round(cp.score * 100)}% match</p>
            </div>`;
        }).join('');
    }

    finalStats.innerHTML = `
        <!-- Overview -->
        <div class="grid grid-cols-3 gap-2 mb-4 text-center">
            <div class="bg-gray-50 rounded-lg p-3">
                <p class="text-xl font-bold">${stats.rounds_completed}</p>
                <p class="text-[10px] text-gray-500 uppercase">Rounds</p>
            </div>
            <div class="bg-green-50 rounded-lg p-3">
                <p class="text-xl font-bold text-green-600">${likes}</p>
                <p class="text-[10px] text-gray-500 uppercase">Chosen</p>
            </div>
            <div class="bg-purple-50 rounded-lg p-3">
                <p class="text-xl font-bold text-purple-600">${catsCompleted}/${totalCats}</p>
                <p class="text-[10px] text-gray-500 uppercase">Categories</p>
            </div>
        </div>

        <!-- Categories Tested -->
        <div class="mb-4">
            <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                <span class="w-2 h-2 bg-indigo-500 rounded-full"></span> Categories Tested
            </h4>
            <div class="bg-gray-50 rounded-lg p-3">
                ${categoryStatsHtml || '<p class="text-xs text-gray-400">No category data</p>'}
            </div>
        </div>

        <!-- Learned Style Preferences -->
        ${clusterPrefsHtml ? `
        <div class="mb-4">
            <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                <span class="w-2 h-2 bg-green-500 rounded-full"></span> Your Style Preferences
            </h4>
            ${clusterPrefsHtml}
        </div>
        ` : ''}

        <!-- Style Archetypes -->
        ${archetypeHtml ? `
        <div class="mb-4">
            <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                <span class="w-2 h-2 bg-gray-500 rounded-full"></span> Style Archetypes
            </h4>
            <div class="bg-gray-50 rounded-lg p-3">
                ${archetypeHtml}
            </div>
        </div>
        ` : ''}

        <!-- Visual Preferences -->
        ${(likedAnchors.length > 0 || dislikedAnchors.length > 0) ? `
        <div class="mb-4">
            <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                <span class="w-2 h-2 bg-green-500 rounded-full"></span> Visual Preferences
            </h4>
            <div class="grid grid-cols-2 gap-2">
                <div class="bg-green-50 rounded-lg p-2">
                    <p class="text-[10px] text-green-600 font-semibold mb-1">DRAWN TO</p>
                    <p class="text-sm font-medium">${likedAnchors.map(a => a.anchor).join(', ') || '-'}</p>
                </div>
                <div class="bg-red-50 rounded-lg p-2">
                    <p class="text-[10px] text-red-600 font-semibold mb-1">AVOIDS</p>
                    <p class="text-sm font-medium">${dislikedAnchors.map(a => a.anchor).join(', ') || '-'}</p>
                </div>
            </div>
        </div>
        ` : ''}

        <!-- Top Brands -->
        ${brandsHtml ? `
        <div class="mb-4">
            <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                <span class="w-2 h-2 bg-gray-500 rounded-full"></span> Top Brands
            </h4>
            <div class="flex flex-wrap">
                ${brandsHtml}
            </div>
        </div>
        ` : ''}

        <!-- Feed Preview by Category -->
        ${feedPreview && Object.keys(feedPreview).length > 0 ? `
        <div class="mb-4">
            <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                <span class="w-2 h-2 bg-blue-500 rounded-full"></span> Predicted For You
            </h4>
            <p class="text-xs text-gray-400 mb-3">Based on your taste profile</p>
            ${Object.entries(feedPreview).map(([category, items]) => `
                <div class="mb-4">
                    <h5 class="text-sm font-semibold text-gray-700 mb-2">${category}</h5>
                    <div class="grid grid-cols-4 gap-2">
                        ${items.slice(0, 8).map(item => `
                            <div class="relative group">
                                <div class="aspect-square bg-gray-100 rounded-lg overflow-hidden">
                                    <img src="${item.image_url}"
                                         alt="${item.brand || 'Item'}"
                                         class="w-full h-full object-cover"
                                         onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><rect fill=%22%23f3f4f6%22 width=%22100%22 height=%22100%22/><text x=%2250%22 y=%2250%22 dominant-baseline=%22middle%22 text-anchor=%22middle%22 fill=%22%239ca3af%22 font-size=%2212%22>No img</text></svg>'">
                                </div>
                                <div class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent p-1 rounded-b-lg">
                                    <p class="text-[8px] text-white truncate">${item.brand || ''}</p>
                                </div>
                                ${item.similarity ? `<div class="absolute top-1 right-1 bg-green-500 text-white text-[8px] px-1 rounded">${Math.round(item.similarity * 100)}%</div>` : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>
            `).join('')}
        </div>
        ` : ''}

        <!-- Technical (internal) -->
        <div class="border-t border-gray-200 pt-3 mt-3">
            <div class="flex justify-between text-xs text-gray-400">
                <span>Taste stability: ${(stats.taste_stability || 0).toFixed(3)}</span>
                <span>Items seen: ${likes + dislikes}</span>
            </div>
        </div>
    `;

    document.getElementById('complete-modal').classList.remove('hidden');
}

function showSetup() {
    document.getElementById('complete-modal').classList.add('hidden');
    document.getElementById('quiz-screen').classList.add('hidden');
    document.getElementById('setup-screen').classList.remove('hidden');
    document.getElementById('setup-footer').classList.remove('hidden');

    // Reset all state
    selectedColors = new Set();
    selectedCoreTypes = new Set();
    selectedSize = 'M';
    selectedStyles = new Set();
    selectedNecklines = new Set();
    selectedSleeves = new Set();
    selectedMaterials = new Set();
    currentItems = [];
    currentTestInfo = null;
    predictionStats = { correct: 0, total: 0 };

    // Reset core type buttons
    document.querySelectorAll('.core-type-btn').forEach(btn => {
        btn.classList.remove('border-2', 'border-black', 'bg-black', 'text-white');
        btn.classList.add('border-gray-200', 'bg-white');
        btn.querySelector('img').classList.remove('invert', 'brightness-0');
        btn.querySelectorAll('span').forEach(s => s.classList.remove('text-white', 'text-gray-300'));
    });

    // Hide t-shirt styles section
    document.getElementById('tshirt-styles-section').classList.add('hidden');

    // Reset style buttons
    document.querySelectorAll('.style-btn').forEach(btn => {
        btn.classList.remove('border-2', 'border-black');
        btn.classList.add('border-gray-200');
    });

    // Reset neckline buttons
    document.querySelectorAll('.neckline-btn').forEach(btn => {
        btn.querySelector('div').classList.remove('ring-2', 'ring-black', 'ring-offset-2');
    });

    // Reset sleeve buttons
    document.querySelectorAll('.sleeve-btn').forEach(btn => {
        const imgDiv = btn.querySelector('div');
        imgDiv.classList.remove('ring-2', 'ring-black', 'ring-offset-2');
    });

    // Reset size selector
    document.querySelectorAll('.size-btn').forEach(btn => {
        if (btn.dataset.size === 'M') {
            btn.classList.add('text-white', 'bg-black', 'shadow-md');
            btn.classList.remove('text-text-main', 'hover:bg-gray-50');
        } else {
            btn.classList.remove('text-white', 'bg-black', 'shadow-md');
            btn.classList.add('text-text-main', 'hover:bg-gray-50');
        }
    });

    // Reset material buttons
    document.querySelectorAll('.material-btn').forEach(btn => {
        const img = btn.querySelector('img');
        const label = btn.querySelector('.material-label');
        img.classList.remove('grayscale', 'opacity-50');
        label.classList.remove('bg-gray-200', 'text-gray-500', 'line-through');
        label.classList.add('bg-white', 'text-black');
    });

    // Reset color selector
    document.getElementById('color-selector').innerHTML = '';
    initColorSelector();
}
</script>

</body>
</html>'''


def get_ranking_html() -> str:
    """Return the HTML template for the drag-and-drop ranking interface."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
    <title>Style Discovery - Rank Your Favorites</title>
    <script src="https://cdn.tailwindcss.com?plugins=forms,typography"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@200;300;400;500;600;700;800&family=Noto+Sans:wght@400;500;700&display=swap" rel="stylesheet"/>
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        "primary": "#1A1A1A",
                        "primary-dark": "#0A0A0A",
                        "background-light": "#f9fafb",
                        "surface-light": "#ffffff",
                        "text-main": "#131811",
                        "text-muted": "#6c7275",
                    },
                    fontFamily: {
                        display: ["Plus Jakarta Sans", "sans-serif"],
                        body: ["Noto Sans", "sans-serif"],
                    },
                }
            }
        };
    </script>
    <style>
        body::-webkit-scrollbar { display: none; }
        .no-scrollbar::-webkit-scrollbar { display: none; }
        .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }

        .rank-item {
            transition: all 0.2s ease;
            cursor: grab;
            touch-action: none;
        }
        .rank-item:active {
            cursor: grabbing;
        }
        .rank-item.dragging {
            opacity: 0.9;
            transform: scale(1.05);
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            z-index: 1000;
        }
        .rank-item.drag-over {
            border-color: #10B981 !important;
            background: #ECFDF5;
        }
        .rank-slot {
            transition: all 0.2s ease;
        }
        .rank-slot.highlight {
            background: #F0FDF4;
            border-color: #10B981;
        }

        @keyframes slideIn {
            0% { transform: translateY(20px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
        }
        .slide-in { animation: slideIn 0.3s ease-out forwards; }
        .slide-in-1 { animation-delay: 0.05s; opacity: 0; }
        .slide-in-2 { animation-delay: 0.1s; opacity: 0; }
        .slide-in-3 { animation-delay: 0.15s; opacity: 0; }
        .slide-in-4 { animation-delay: 0.2s; opacity: 0; }
        .slide-in-5 { animation-delay: 0.25s; opacity: 0; }

        @keyframes fadeOut {
            0% { opacity: 1; transform: scale(1); }
            100% { opacity: 0; transform: scale(0.95); }
        }
        .fade-out { animation: fadeOut 0.3s ease-out forwards; }

        .rank-badge {
            transition: all 0.2s ease;
        }
    </style>
</head>
<body class="bg-background-light text-text-main font-display antialiased min-h-screen flex flex-col">

<!-- Header -->
<header class="sticky top-0 z-50 bg-surface-light border-b border-gray-100 px-6 py-3 shadow-sm">
    <div class="max-w-[1100px] mx-auto flex items-center justify-between">
        <h2 class="text-xl font-bold tracking-tight">Style Discovery</h2>
        <button onclick="showSetup()" class="text-sm font-medium text-text-muted hover:text-text-main transition-colors">
            Restart
        </button>
    </div>
</header>

<!-- Setup Screen -->
<div id="setup-screen" class="flex-grow flex flex-col items-center py-8 px-4 sm:px-6">
    <div class="w-full max-w-[960px] flex flex-col gap-10">

        <!-- Core Selection Section -->
        <div class="text-center">
            <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">CORE SELECTION</h2>
            <h3 class="text-2xl font-bold text-text-main">Which tops are you looking for?</h3>
        </div>

        <div class="flex justify-center" id="core-selection">
            <button onclick="toggleCoreType('tshirts')" data-type="tshirts" class="core-type-btn flex flex-col items-center justify-center p-4 rounded-xl border transition-all h-[180px] w-[200px] border-gray-200 bg-white hover:border-gray-300">
                <div class="w-16 h-16 mb-4 flex items-center justify-center">
                    <img src="/assets/Initial%20Screen%20Icons/TShirt%20Icons.png" alt="T-Shirts" class="object-contain h-full w-full"/>
                </div>
                <span class="font-bold text-sm mb-1">T-Shirts</span>
                <span class="text-[10px] text-gray-500">Casual everyday</span>
            </button>
        </div>

        <div class="border-t border-gray-100 my-2"></div>

        <!-- Size Section -->
        <div class="text-center">
            <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">MEASUREMENTS</h2>
            <h3 class="text-2xl font-bold text-text-main mb-8">What is your typical size?</h3>
            <div class="max-w-2xl mx-auto">
                <div class="flex items-center justify-between bg-white border border-gray-200 rounded-full p-1.5 h-14 shadow-sm" id="size-selector">
                    <button onclick="selectSize('XS')" data-size="XS" class="size-btn flex-1 h-full rounded-full text-xs font-bold transition-all flex items-center justify-center text-text-main hover:bg-gray-50">XS</button>
                    <button onclick="selectSize('S')" data-size="S" class="size-btn flex-1 h-full rounded-full text-xs font-bold transition-all flex items-center justify-center text-text-main hover:bg-gray-50">S</button>
                    <button onclick="selectSize('M')" data-size="M" class="size-btn flex-1 h-full rounded-full text-xs font-bold transition-all flex items-center justify-center text-white bg-black shadow-md">M</button>
                    <button onclick="selectSize('L')" data-size="L" class="size-btn flex-1 h-full rounded-full text-xs font-bold transition-all flex items-center justify-center text-text-main hover:bg-gray-50">L</button>
                    <button onclick="selectSize('XL')" data-size="XL" class="size-btn flex-1 h-full rounded-full text-xs font-bold transition-all flex items-center justify-center text-text-main hover:bg-gray-50">XL</button>
                    <button onclick="selectSize('2XL')" data-size="2XL" class="size-btn flex-1 h-full rounded-full text-xs font-bold transition-all flex items-center justify-center text-text-main hover:bg-gray-50">2XL</button>
                </div>
            </div>
        </div>

        <div class="border-t border-gray-100 my-2"></div>

        <!-- T-Shirt Style Section -->
        <div id="tshirt-styles-section" class="hidden">
            <div class="text-center mb-6">
                <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">T-SHIRT STYLES</h2>
                <h3 class="text-xl font-bold text-text-main">Select styles you wear</h3>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-5 gap-4" id="tshirt-styles">
                <div onclick="toggleStyle('plain')" data-style="plain" class="style-btn cursor-pointer rounded-xl border overflow-hidden transition-all border-gray-200 hover:border-gray-300">
                    <div class="w-full aspect-square bg-gray-50 flex items-center justify-center overflow-hidden">
                        <img src="/assets/Tees%20styles/Plain.png" alt="Plain" class="w-full h-full object-cover"/>
                    </div>
                    <div class="p-3 text-center border-t border-gray-100 bg-white">
                        <span class="text-sm font-bold text-gray-900">Plain</span>
                    </div>
                </div>
                <div onclick="toggleStyle('graphics')" data-style="graphics" class="style-btn cursor-pointer rounded-xl border overflow-hidden transition-all border-gray-200 hover:border-gray-300">
                    <div class="w-full aspect-square bg-gray-50 flex items-center justify-center overflow-hidden">
                        <img src="/assets/Tees%20styles/Graphic%20Tshirt.png" alt="Graphics" class="w-full h-full object-cover"/>
                    </div>
                    <div class="p-3 text-center border-t border-gray-100 bg-white">
                        <span class="text-sm font-bold text-gray-900">Graphics</span>
                    </div>
                </div>
                <div onclick="toggleStyle('smalllogos')" data-style="smalllogos" class="style-btn cursor-pointer rounded-xl border overflow-hidden transition-all border-gray-200 hover:border-gray-300">
                    <div class="w-full aspect-square bg-gray-50 flex items-center justify-center overflow-hidden">
                        <img src="/assets/Tees%20styles/Small%20Logos.png" alt="Small Logos" class="w-full h-full object-cover"/>
                    </div>
                    <div class="p-3 text-center border-t border-gray-100 bg-white">
                        <span class="text-sm font-bold text-gray-900">Small Logos</span>
                    </div>
                </div>
                <div onclick="toggleStyle('athletic')" data-style="athletic" class="style-btn cursor-pointer rounded-xl border overflow-hidden transition-all border-gray-200 hover:border-gray-300">
                    <div class="w-full aspect-square bg-gray-50 flex items-center justify-center overflow-hidden">
                        <img src="/assets/Tees%20styles/Atheletic%20Tshirt.png" alt="Athletic" class="w-full h-full object-cover"/>
                    </div>
                    <div class="p-3 text-center border-t border-gray-100 bg-white">
                        <span class="text-sm font-bold text-gray-900">Athletic</span>
                    </div>
                </div>
                <div onclick="toggleStyle('pocket')" data-style="pocket" class="style-btn cursor-pointer rounded-xl border overflow-hidden transition-all border-gray-200 hover:border-gray-300">
                    <div class="w-full aspect-square bg-gray-50 flex items-center justify-center overflow-hidden">
                        <img src="/assets/Tees%20styles/Pocket%20Tshirt.png" alt="Pocket" class="w-full h-full object-cover"/>
                    </div>
                    <div class="p-3 text-center border-t border-gray-100 bg-white">
                        <span class="text-sm font-bold text-gray-900">Pocket</span>
                    </div>
                </div>
            </div>

            <div class="border-t border-gray-100 my-6"></div>

            <!-- Neckline Section -->
            <div class="text-center mb-6">
                <h3 class="text-lg font-bold text-text-main">Neckline Preference</h3>
            </div>
            <div class="grid grid-cols-3 gap-4 max-w-md mx-auto" id="neckline-selector">
                <div onclick="toggleNeckline('crew')" data-neckline="crew" class="neckline-btn cursor-pointer flex flex-col items-center">
                    <div class="w-full aspect-[4/3] rounded-xl bg-gray-100 overflow-hidden flex items-center justify-center transition-all hover:bg-gray-200">
                        <img src="/assets/neck%20Styles/Crew.png" alt="Crew" class="w-full h-full object-cover"/>
                    </div>
                    <span class="mt-2 text-sm font-semibold text-gray-900">Crew</span>
                </div>
                <div onclick="toggleNeckline('vneck')" data-neckline="vneck" class="neckline-btn cursor-pointer flex flex-col items-center">
                    <div class="w-full aspect-[4/3] rounded-xl bg-gray-100 overflow-hidden flex items-center justify-center transition-all hover:bg-gray-200">
                        <img src="/assets/neck%20Styles/Vneck.png" alt="V-Neck" class="w-full h-full object-cover"/>
                    </div>
                    <span class="mt-2 text-sm font-semibold text-gray-900">V-Neck</span>
                </div>
                <div onclick="toggleNeckline('henley')" data-neckline="henley" class="neckline-btn cursor-pointer flex flex-col items-center">
                    <div class="w-full aspect-[4/3] rounded-xl bg-gray-100 overflow-hidden flex items-center justify-center transition-all hover:bg-gray-200">
                        <img src="/assets/neck%20Styles/henley.png" alt="Henley" class="w-full h-full object-cover"/>
                    </div>
                    <span class="mt-2 text-sm font-semibold text-gray-900">Henley</span>
                </div>
            </div>

            <div class="border-t border-gray-100 my-6"></div>

            <!-- Sleeve Length Section -->
            <div class="text-center mb-6">
                <h3 class="text-lg font-bold text-text-main">Sleeve Length</h3>
            </div>
            <div class="grid grid-cols-2 gap-4 max-w-md mx-auto" id="sleeve-selector">
                <div onclick="toggleSleeve('short')" data-sleeve="short" class="sleeve-btn cursor-pointer flex flex-col items-center">
                    <div class="w-full aspect-[4/3] rounded-xl bg-gray-100 overflow-hidden flex items-center justify-center transition-all hover:bg-gray-200">
                        <img src="/assets/Tshirt%20Style/Short%20Sleeves.png" alt="Short Sleeve" class="w-full h-full object-cover"/>
                    </div>
                    <span class="mt-2 text-sm font-semibold text-gray-900">Short Sleeve</span>
                </div>
                <div onclick="toggleSleeve('long')" data-sleeve="long" class="sleeve-btn cursor-pointer flex flex-col items-center">
                    <div class="w-full aspect-[4/3] rounded-xl bg-gray-100 overflow-hidden flex items-center justify-center transition-all hover:bg-gray-200">
                        <img src="/assets/Tshirt%20Style/Long%20sleeves.png" alt="Long Sleeve" class="w-full h-full object-cover"/>
                    </div>
                    <span class="mt-2 text-sm font-semibold text-gray-900">Long Sleeve</span>
                </div>
            </div>
        </div>

        <div class="border-t border-gray-100 my-2"></div>

        <!-- Colors to Avoid -->
        <div class="text-center">
            <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">PREFERENCES</h2>
            <h3 class="text-2xl font-bold text-text-main mb-2">Colors to Avoid</h3>
            <p class="text-gray-500 text-sm mb-8">Tap any color you dislike to cross it out.</p>
            <div class="flex justify-center">
                <div id="color-selector" class="flex flex-wrap justify-center gap-4 max-w-[700px]"></div>
            </div>
        </div>

        <div class="border-t border-gray-100 my-2"></div>

        <!-- Materials to Avoid -->
        <div class="text-center">
            <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">FABRICS</h2>
            <h3 class="text-2xl font-bold text-text-main mb-8">Materials to Avoid</h3>
            <div class="grid grid-cols-2 md:grid-cols-3 gap-6" id="materials-selector">
                <button onclick="toggleMaterial('polyester')" data-material="polyester" class="material-btn relative group overflow-hidden rounded-xl aspect-[16/9] transition-transform hover:scale-[1.02]">
                    <img src="/assets/Materials/Polyster.png" alt="Polyester" class="w-full h-full object-cover"/>
                    <div class="absolute inset-0 flex items-center justify-center">
                        <span class="material-label px-4 py-1.5 rounded-full text-xs font-bold shadow-sm bg-white text-black">Polyester</span>
                    </div>
                </button>
                <button onclick="toggleMaterial('wool')" data-material="wool" class="material-btn relative group overflow-hidden rounded-xl aspect-[16/9] transition-transform hover:scale-[1.02]">
                    <img src="/assets/Materials/Wool.png" alt="Wool" class="w-full h-full object-cover"/>
                    <div class="absolute inset-0 flex items-center justify-center">
                        <span class="material-label px-4 py-1.5 rounded-full text-xs font-bold shadow-sm bg-white text-black">Wool</span>
                    </div>
                </button>
                <button onclick="toggleMaterial('linen')" data-material="linen" class="material-btn relative group overflow-hidden rounded-xl aspect-[16/9] transition-transform hover:scale-[1.02]">
                    <img src="/assets/Materials/Linen%20rectangle.png" alt="Linen" class="w-full h-full object-cover"/>
                    <div class="absolute inset-0 flex items-center justify-center">
                        <span class="material-label px-4 py-1.5 rounded-full text-xs font-bold shadow-sm bg-white text-black">Linen</span>
                    </div>
                </button>
                <button onclick="toggleMaterial('silk')" data-material="silk" class="material-btn relative group overflow-hidden rounded-xl aspect-[16/9] transition-transform hover:scale-[1.02]">
                    <img src="/assets/Materials/Silk.png" alt="Silk" class="w-full h-full object-cover"/>
                    <div class="absolute inset-0 flex items-center justify-center">
                        <span class="material-label px-4 py-1.5 rounded-full text-xs font-bold shadow-sm bg-white text-black">Silk</span>
                    </div>
                </button>
                <button onclick="toggleMaterial('synthetics')" data-material="synthetics" class="material-btn relative group overflow-hidden rounded-xl aspect-[16/9] transition-transform hover:scale-[1.02]">
                    <img src="/assets/Materials/Synthetics.png" alt="Synthetics" class="w-full h-full object-cover"/>
                    <div class="absolute inset-0 flex items-center justify-center">
                        <span class="material-label px-4 py-1.5 rounded-full text-xs font-bold shadow-sm bg-white text-black">Synthetics</span>
                    </div>
                </button>
                <button onclick="toggleMaterial('leather')" data-material="leather" class="material-btn relative group overflow-hidden rounded-xl aspect-[16/9] transition-transform hover:scale-[1.02]">
                    <img src="/assets/Materials/Leather.png" alt="Leather" class="w-full h-full object-cover"/>
                    <div class="absolute inset-0 flex items-center justify-center">
                        <span class="material-label px-4 py-1.5 rounded-full text-xs font-bold shadow-sm bg-white text-black">Leather</span>
                    </div>
                </button>
            </div>
        </div>

        <div class="h-24"></div>
    </div>
</div>

<!-- Sticky Bottom Button -->
<div id="setup-footer" class="sticky bottom-0 w-full bg-white/95 backdrop-blur-md border-t border-gray-200 py-4 px-6 shadow-lg z-40">
    <div class="max-w-[960px] mx-auto flex justify-center">
        <button onclick="startSession()" class="bg-primary hover:bg-primary-dark text-white font-bold py-4 px-12 rounded-xl shadow-md transition-all transform active:scale-95 flex items-center gap-2 text-lg">
            Start Ranking Quiz
            <span class="material-symbols-outlined">arrow_forward</span>
        </button>
    </div>
</div>

<!-- Main Ranking Screen -->
<main id="ranking-screen" class="hidden flex-grow flex flex-col items-center px-4 py-6">
    <!-- Progress -->
    <div class="w-full max-w-3xl mb-4">
        <div class="flex justify-between items-center mb-2">
            <span class="text-sm font-medium text-gray-600">Round <span id="round-num">1</span> of ~8</span>
            <span class="text-sm text-gray-500">Stability: <span id="stability">0.00</span></span>
        </div>
        <div class="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
            <div id="progress-bar" class="h-full bg-gradient-to-r from-purple-400 to-indigo-500 rounded-full transition-all duration-500" style="width: 0%"></div>
        </div>
    </div>

    <!-- Instruction -->
    <div class="text-center mb-4">
        <h2 class="text-xl font-bold">Drag to rank from favorite to least</h2>
        <p class="text-sm text-gray-500 mt-1">Best at top (#1), worst at bottom (#6)</p>
    </div>

    <!-- Ranking Area -->
    <div id="ranking-area" class="w-full max-w-3xl">
        <div class="grid grid-cols-2 md:grid-cols-3 gap-4" id="items-grid">
            <!-- Items will be inserted here -->
        </div>
    </div>

    <!-- Action Buttons -->
    <div class="flex flex-col items-center gap-3 mt-6">
        <button onclick="submitRanking()" id="submit-btn" class="bg-gradient-to-r from-purple-500 to-indigo-600 hover:from-purple-600 hover:to-indigo-700 text-white font-bold py-3 px-10 rounded-xl shadow-md transition-all transform active:scale-95 flex items-center gap-2">
            <span class="material-symbols-outlined">check</span>
            Submit Ranking
        </button>
        <button onclick="skipAll()" class="text-gray-400 hover:text-gray-600 text-sm font-medium flex items-center gap-1">
            <span class="material-symbols-outlined text-base">refresh</span>
            None of these
        </button>
    </div>
</main>

<!-- Complete Modal -->
<div id="complete-modal" class="hidden fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 overflow-y-auto">
    <div class="bg-white rounded-2xl shadow-2xl max-w-lg w-full p-6 my-8">
        <div class="text-center mb-4">
            <div class="w-14 h-14 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <span class="material-symbols-outlined text-purple-500 text-2xl">check_circle</span>
            </div>
            <h3 class="text-xl font-bold">Style Profile Complete!</h3>
            <p class="text-gray-500 text-sm">Completed in just <span id="final-rounds">0</span> rounds</p>
        </div>
        <div id="final-stats" class="text-left max-h-[60vh] overflow-y-auto pr-2"></div>
        <button onclick="showSetup()" class="w-full bg-gray-900 text-white py-3 rounded-xl font-semibold hover:bg-gray-800 mt-4">
            Start New Quiz
        </button>
    </div>
</div>

<script>
const COLORS = [
    { name: 'black', bg: 'bg-black' },
    { name: 'white', bg: 'bg-white border border-gray-200' },
    { name: 'gray', bg: 'bg-gray-400' },
    { name: 'navy', bg: 'bg-blue-900' },
    { name: 'blue', bg: 'bg-blue-500' },
    { name: 'red', bg: 'bg-red-500' },
    { name: 'green', bg: 'bg-green-600' },
    { name: 'brown', bg: 'bg-amber-700' },
    { name: 'pink', bg: 'bg-pink-400' },
    { name: 'yellow', bg: 'bg-yellow-400' },
    { name: 'orange', bg: 'bg-orange-400' },
    { name: 'purple', bg: 'bg-purple-500' },
    { name: 'neon', bg: 'bg-lime-400' },
];

// State
let selectedColors = new Set();
let selectedCoreTypes = new Set();
let selectedSize = 'M';
let selectedStyles = new Set();
let selectedNecklines = new Set();
let selectedSleeves = new Set();
let selectedMaterials = new Set();
let currentItems = [];
let rankedOrder = [];
let isAnimating = false;
let draggedElement = null;
let draggedIndex = -1;

document.addEventListener('DOMContentLoaded', () => {
    initColorSelector();
});

function initColorSelector() {
    const container = document.getElementById('color-selector');
    COLORS.forEach(color => {
        const chip = document.createElement('button');
        chip.className = `group relative size-12 rounded-full ${color.bg} shadow-sm hover:scale-110 transition-transform`;
        chip.title = color.name;
        chip.innerHTML = `<span class="hidden group-hover:flex absolute inset-0 items-center justify-center ${color.name === 'white' || color.name === 'yellow' || color.name === 'neon' ? 'text-gray-800' : 'text-white'} material-symbols-outlined text-lg">close</span>`;
        chip.onclick = () => toggleColor(color.name, chip);
        container.appendChild(chip);
    });
}

function toggleColor(name, chip) {
    if (selectedColors.has(name)) {
        selectedColors.delete(name);
        chip.classList.remove('ring-2', 'ring-primary', 'ring-offset-2', 'opacity-50', 'grayscale');
        chip.querySelector('span').classList.add('hidden');
        chip.querySelector('span').classList.remove('flex');
    } else {
        selectedColors.add(name);
        chip.classList.add('ring-2', 'ring-primary', 'ring-offset-2', 'opacity-50', 'grayscale');
        chip.querySelector('span').classList.remove('hidden');
        chip.querySelector('span').classList.add('flex');
    }
}

function toggleCoreType(type) {
    const btn = document.querySelector(`[data-type="${type}"]`);
    if (selectedCoreTypes.has(type)) {
        selectedCoreTypes.delete(type);
        btn.classList.remove('border-2', 'border-black', 'bg-black', 'text-white');
        btn.classList.add('border-gray-200', 'bg-white');
        btn.querySelector('img').classList.remove('invert', 'brightness-0');
        btn.querySelectorAll('span').forEach(s => s.classList.remove('text-white', 'text-gray-300'));
    } else {
        selectedCoreTypes.add(type);
        btn.classList.add('border-2', 'border-black', 'bg-black', 'text-white');
        btn.classList.remove('border-gray-200', 'bg-white');
        btn.querySelector('img').classList.add('invert', 'brightness-0');
        btn.querySelectorAll('span')[0].classList.add('text-white');
        btn.querySelectorAll('span')[1].classList.add('text-gray-300');
    }
    if (selectedCoreTypes.has('tshirts')) {
        document.getElementById('tshirt-styles-section').classList.remove('hidden');
    } else {
        document.getElementById('tshirt-styles-section').classList.add('hidden');
    }
}

function selectSize(size) {
    selectedSize = size;
    document.querySelectorAll('.size-btn').forEach(btn => {
        if (btn.dataset.size === size) {
            btn.classList.add('text-white', 'bg-black', 'shadow-md');
            btn.classList.remove('text-text-main', 'hover:bg-gray-50');
        } else {
            btn.classList.remove('text-white', 'bg-black', 'shadow-md');
            btn.classList.add('text-text-main', 'hover:bg-gray-50');
        }
    });
}

function toggleStyle(style) {
    const btn = document.querySelector(`[data-style="${style}"]`);
    if (selectedStyles.has(style)) {
        selectedStyles.delete(style);
        btn.classList.remove('border-2', 'border-black');
        btn.classList.add('border-gray-200');
    } else {
        selectedStyles.add(style);
        btn.classList.add('border-2', 'border-black');
        btn.classList.remove('border-gray-200');
    }
}

function toggleNeckline(neckline) {
    console.log('toggleNeckline called:', neckline);
    const btn = document.querySelector(`[data-neckline="${neckline}"]`);
    const imgDiv = btn.querySelector('div');
    if (selectedNecklines.has(neckline)) {
        selectedNecklines.delete(neckline);
        imgDiv.classList.remove('ring-2', 'ring-black', 'ring-offset-2');
    } else {
        selectedNecklines.add(neckline);
        imgDiv.classList.add('ring-2', 'ring-black', 'ring-offset-2');
    }
    console.log('selectedNecklines now:', Array.from(selectedNecklines));
}

function toggleSleeve(sleeve) {
    console.log('toggleSleeve called:', sleeve);
    const btn = document.querySelector(`[data-sleeve="${sleeve}"]`);
    const imgDiv = btn.querySelector('div');
    if (selectedSleeves.has(sleeve)) {
        selectedSleeves.delete(sleeve);
        imgDiv.classList.remove('ring-2', 'ring-black', 'ring-offset-2');
    } else {
        selectedSleeves.add(sleeve);
        imgDiv.classList.add('ring-2', 'ring-black', 'ring-offset-2');
    }
    console.log('selectedSleeves now:', Array.from(selectedSleeves));
}

function toggleMaterial(material) {
    const btn = document.querySelector(`[data-material="${material}"]`);
    const img = btn.querySelector('img');
    const label = btn.querySelector('.material-label');
    if (selectedMaterials.has(material)) {
        selectedMaterials.delete(material);
        img.classList.remove('grayscale', 'opacity-50');
        label.classList.remove('bg-gray-200', 'text-gray-500', 'line-through');
        label.classList.add('bg-white', 'text-black');
    } else {
        selectedMaterials.add(material);
        img.classList.add('grayscale', 'opacity-50');
        label.classList.add('bg-gray-200', 'text-gray-500', 'line-through');
        label.classList.remove('bg-white', 'text-black');
    }
}

async function startSession() {
    try {
        const response = await fetch('/api/rank/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: 'default_user',
                colors_to_avoid: Array.from(selectedColors),
                materials_to_avoid: Array.from(selectedMaterials),
                selected_styles: Array.from(selectedStyles),
                selected_necklines: Array.from(selectedNecklines),
                selected_sleeves: Array.from(selectedSleeves)
            })
        });
        const data = await response.json();

        if (data.status === 'started') {
            currentItems = data.items;
            rankedOrder = data.items.map(item => item.id);
            document.getElementById('setup-screen').classList.add('hidden');
            document.getElementById('setup-footer').classList.add('hidden');
            document.getElementById('ranking-screen').classList.remove('hidden');
            renderItems(data.items);
            updateProgress(1, 0);
        }
    } catch (error) {
        console.error('Failed to start:', error);
    }
}

function renderItems(items) {
    const grid = document.getElementById('items-grid');
    grid.innerHTML = '';

    items.forEach((item, idx) => {
        const card = document.createElement('div');
        card.className = `rank-item bg-white rounded-2xl shadow-lg overflow-hidden slide-in slide-in-${idx} relative`;
        card.draggable = true;
        card.dataset.id = item.id;
        card.dataset.index = idx;

        const rankBadge = getRankBadge(idx);

        card.innerHTML = `
            <div class="absolute top-2 left-2 z-10 rank-badge ${rankBadge.bg} ${rankBadge.text} w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm shadow-md">
                ${idx + 1}
            </div>
            <div class="absolute top-2 right-2 z-10 text-gray-400">
                <span class="material-symbols-outlined">drag_indicator</span>
            </div>
            <div class="aspect-square bg-gray-100 overflow-hidden">
                <img src="${item.image_url}" alt="${item.category}" class="w-full h-full object-contain pointer-events-none"/>
            </div>
            <div class="p-3">
                <p class="font-bold text-sm truncate">${item.brand}</p>
                <p class="text-xs text-gray-500">${item.color} • ${item.category.replace(' T-shirts', '')}</p>
            </div>
        `;

        // Drag events
        card.addEventListener('dragstart', handleDragStart);
        card.addEventListener('dragend', handleDragEnd);
        card.addEventListener('dragover', handleDragOver);
        card.addEventListener('drop', handleDrop);
        card.addEventListener('dragenter', handleDragEnter);
        card.addEventListener('dragleave', handleDragLeave);

        // Touch events for mobile
        card.addEventListener('touchstart', handleTouchStart, { passive: false });
        card.addEventListener('touchmove', handleTouchMove, { passive: false });
        card.addEventListener('touchend', handleTouchEnd);

        grid.appendChild(card);
    });
}

function getRankBadge(rank) {
    if (rank === 0) return { bg: 'bg-yellow-400', text: 'text-yellow-900' };
    if (rank === 1) return { bg: 'bg-gray-300', text: 'text-gray-700' };
    if (rank === 2) return { bg: 'bg-amber-600', text: 'text-white' };
    return { bg: 'bg-gray-100', text: 'text-gray-600' };
}

function handleDragStart(e) {
    draggedElement = e.target.closest('.rank-item');
    draggedIndex = parseInt(draggedElement.dataset.index);
    draggedElement.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
}

function handleDragEnd(e) {
    draggedElement.classList.remove('dragging');
    document.querySelectorAll('.rank-item').forEach(item => {
        item.classList.remove('drag-over');
    });
    draggedElement = null;
    draggedIndex = -1;
}

function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
}

function handleDragEnter(e) {
    const target = e.target.closest('.rank-item');
    if (target && target !== draggedElement) {
        target.classList.add('drag-over');
    }
}

function handleDragLeave(e) {
    const target = e.target.closest('.rank-item');
    if (target) {
        target.classList.remove('drag-over');
    }
}

function handleDrop(e) {
    e.preventDefault();
    const target = e.target.closest('.rank-item');
    if (target && target !== draggedElement) {
        const targetIndex = parseInt(target.dataset.index);
        moveItem(draggedIndex, targetIndex);
    }
}

// Touch handling for mobile
let touchStartY = 0;
let touchElement = null;

function handleTouchStart(e) {
    touchElement = e.target.closest('.rank-item');
    if (touchElement) {
        touchStartY = e.touches[0].clientY;
        draggedIndex = parseInt(touchElement.dataset.index);
        setTimeout(() => {
            if (touchElement) touchElement.classList.add('dragging');
        }, 150);
    }
}

function handleTouchMove(e) {
    if (!touchElement) return;
    e.preventDefault();

    const touch = e.touches[0];
    const elementBelow = document.elementFromPoint(touch.clientX, touch.clientY);
    const targetCard = elementBelow?.closest('.rank-item');

    document.querySelectorAll('.rank-item').forEach(item => {
        item.classList.remove('drag-over');
    });

    if (targetCard && targetCard !== touchElement) {
        targetCard.classList.add('drag-over');
    }
}

function handleTouchEnd(e) {
    if (!touchElement) return;

    touchElement.classList.remove('dragging');

    const touch = e.changedTouches[0];
    const elementBelow = document.elementFromPoint(touch.clientX, touch.clientY);
    const targetCard = elementBelow?.closest('.rank-item');

    if (targetCard && targetCard !== touchElement) {
        const targetIndex = parseInt(targetCard.dataset.index);
        moveItem(draggedIndex, targetIndex);
    }

    document.querySelectorAll('.rank-item').forEach(item => {
        item.classList.remove('drag-over');
    });

    touchElement = null;
    draggedIndex = -1;
}

function moveItem(fromIndex, toIndex) {
    // Move item from fromIndex to toIndex (reorder, not swap)
    // This shifts all items in between

    const grid = document.getElementById('items-grid');
    const items = Array.from(grid.children);
    const movedItem = items[fromIndex];

    // Remove from rankedOrder and insert at new position
    const [movedId] = rankedOrder.splice(fromIndex, 1);
    rankedOrder.splice(toIndex, 0, movedId);

    // Update DOM: remove and reinsert at correct position
    grid.removeChild(movedItem);

    if (toIndex >= items.length - 1) {
        // Insert at end
        grid.appendChild(movedItem);
    } else if (toIndex <= fromIndex) {
        // Moving up: insert before the item currently at toIndex
        grid.insertBefore(movedItem, grid.children[toIndex]);
    } else {
        // Moving down: insert before the item currently at toIndex (which shifted up after removal)
        grid.insertBefore(movedItem, grid.children[toIndex]);
    }

    // Update all rank badges
    updateRankBadges();
}

function updateRankBadges() {
    const items = document.querySelectorAll('.rank-item');
    items.forEach((item, idx) => {
        item.dataset.index = idx;
        const badge = item.querySelector('.rank-badge');
        const rankStyle = getRankBadge(idx);
        badge.className = `absolute top-2 left-2 z-10 rank-badge ${rankStyle.bg} ${rankStyle.text} w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm shadow-md`;
        badge.textContent = idx + 1;
    });

    // Update rankedOrder based on current DOM order
    rankedOrder = Array.from(items).map(item => item.dataset.id);
}

async function submitRanking() {
    if (isAnimating) return;
    isAnimating = true;

    // Animate out
    document.querySelectorAll('.rank-item').forEach(item => {
        item.classList.add('fade-out');
    });

    await new Promise(r => setTimeout(r, 300));

    try {
        const response = await fetch('/api/rank/submit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: 'default_user',
                ranked_ids: rankedOrder
            })
        });
        const data = await response.json();

        if (data.session_complete) {
            showComplete(data.stats);
        } else {
            currentItems = data.items;
            rankedOrder = data.items.map(item => item.id);
            updateProgress(data.round, data.stats?.taste_stability || 0);
            renderItems(data.items);
        }
    } catch (error) {
        console.error('Failed to submit ranking:', error);
    }

    isAnimating = false;
}

async function skipAll() {
    if (isAnimating) return;
    isAnimating = true;

    document.querySelectorAll('.rank-item').forEach(item => {
        item.classList.add('fade-out');
    });

    await new Promise(r => setTimeout(r, 300));

    try {
        const response = await fetch('/api/rank/skip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: 'default_user' })
        });
        const data = await response.json();

        if (data.session_complete) {
            showComplete(data.stats);
        } else {
            currentItems = data.items;
            rankedOrder = data.items.map(item => item.id);
            updateProgress(data.round, 0);
            renderItems(data.items);
        }
    } catch (error) {
        console.error('Failed to skip:', error);
    }

    isAnimating = false;
}

function updateProgress(round, stability) {
    document.getElementById('round-num').textContent = round;
    document.getElementById('stability').textContent = stability.toFixed(2);
    const progress = Math.min((round / 8) * 100, 100);
    document.getElementById('progress-bar').style.width = `${progress}%`;
}

function showComplete(stats) {
    document.getElementById('final-rounds').textContent = stats.rounds_completed || 0;

    // Helper to format numbers cleanly (avoid floating point errors like 0.19999999)
    const fmt = (n) => Number.isInteger(n) ? n : Math.round(n * 10) / 10;

    const finalStats = document.getElementById('final-stats');
    const styleProfile = stats.style_profile || {};
    const archetypes = styleProfile.archetypes || [];
    const anchors = styleProfile.visual_anchors || [];
    const categories = stats.attribute_preferences?.category || [];
    const colors = stats.attribute_preferences?.color || [];
    const fits = stats.attribute_preferences?.fit || [];
    const fabrics = stats.attribute_preferences?.fabric || [];
    const brands = stats.brand_preferences || {};

    // Build archetypes section
    let archetypeHtml = archetypes.slice(0, 4).map(a => {
        const pref = a.preference;
        const barWidth = Math.min(Math.abs(pref) * 200, 100);
        const isPositive = pref >= 0;
        return `<div class="flex items-center justify-between py-1">
            <span class="text-sm capitalize w-28">${a.archetype.replace('_', ' ')}</span>
            <div class="flex-1 mx-2 h-2 bg-gray-200 rounded overflow-hidden">
                <div class="h-full ${isPositive ? 'bg-purple-400' : 'bg-red-400'}" style="width: ${barWidth}%"></div>
            </div>
            <span class="text-xs font-mono w-14 text-right ${isPositive ? 'text-purple-600' : 'text-red-600'}">${pref >= 0 ? '+' : ''}${pref.toFixed(3)}</span>
        </div>`;
    }).join('');

    // Visual anchors
    const likedAnchors = anchors.filter(a => a.direction === 'likes');
    const dislikedAnchors = anchors.filter(a => a.direction === 'dislikes');

    // Categories
    let categoriesHtml = categories.map(c => {
        const pct = Math.round(c.score * 100);
        const barColor = c.score >= 0.5 ? 'bg-purple-400' : 'bg-red-400';
        return `<div class="flex items-center justify-between py-1">
            <span class="text-sm">${c.value.replace(' T-shirts', '')}</span>
            <div class="flex items-center gap-2">
                <div class="w-16 h-2 bg-gray-200 rounded overflow-hidden">
                    <div class="${barColor} h-full" style="width: ${pct}%"></div>
                </div>
                <span class="text-xs font-mono w-12">${fmt(c.likes)}/${fmt(c.total)}</span>
            </div>
        </div>`;
    }).join('');

    // Colors
    let colorsHtml = colors.slice(0, 6).map(c => {
        const isGood = c.score >= 0.5;
        return `<span class="inline-flex items-center ${isGood ? 'bg-purple-50 text-purple-700' : 'bg-red-50 text-red-700'} text-xs px-2 py-1 rounded-full mr-1 mb-1">
            ${c.value} <span class="ml-1 opacity-70">${fmt(c.likes)}/${fmt(c.total)}</span>
        </span>`;
    }).join('');

    // Brands
    const brandEntries = Object.entries(brands).sort((a, b) => (b[1].score || 0) - (a[1].score || 0));
    let brandsHtml = brandEntries.slice(0, 5).map(([name, info]) => {
        return `<span class="inline-flex items-center bg-gray-100 text-gray-700 text-xs px-2 py-1 rounded-full mr-1 mb-1">
            ${name} <span class="ml-1 text-gray-500">${fmt(info.likes)}/${fmt(info.total || info.likes)}</span>
        </span>`;
    }).join('');

    // Fits
    let fitsHtml = fits.slice(0, 4).map(f => {
        const isLiked = f.score >= 0.5;
        return `<span class="inline-flex items-center ${isLiked ? 'bg-purple-50 text-purple-700' : 'bg-gray-100 text-gray-600'} text-xs px-2 py-1 rounded-full mr-1 mb-1">
            ${f.value} <span class="ml-1 opacity-70">${Math.round(f.score * 100)}%</span>
        </span>`;
    }).join('');

    // Fabrics
    let fabricsHtml = fabrics.slice(0, 4).map(f => {
        const isLiked = f.score >= 0.5;
        return `<span class="inline-flex items-center ${isLiked ? 'bg-amber-50 text-amber-700' : 'bg-gray-100 text-gray-600'} text-xs px-2 py-1 rounded-full mr-1 mb-1">
            ${f.value} <span class="ml-1 opacity-70">${Math.round(f.score * 100)}%</span>
        </span>`;
    }).join('');

    // Ranking-specific stats
    const pairwiseComparisons = stats.pairwise_comparisons || 0;
    const informationGain = stats.information_bits || 0;

    finalStats.innerHTML = `
        <!-- Overview -->
        <div class="grid grid-cols-4 gap-2 mb-4 text-center">
            <div class="bg-gray-50 rounded-lg p-2">
                <p class="text-lg font-bold">${stats.rounds_completed}</p>
                <p class="text-[10px] text-gray-500 uppercase">Rounds</p>
            </div>
            <div class="bg-purple-50 rounded-lg p-2">
                <p class="text-lg font-bold text-purple-600">${pairwiseComparisons}</p>
                <p class="text-[10px] text-gray-500 uppercase">Comparisons</p>
            </div>
            <div class="bg-indigo-50 rounded-lg p-2">
                <p class="text-lg font-bold text-indigo-600">${informationGain.toFixed(1)}</p>
                <p class="text-[10px] text-gray-500 uppercase">Info Bits</p>
            </div>
            <div class="bg-green-50 rounded-lg p-2">
                <p class="text-lg font-bold text-green-600">${(stats.taste_stability * 100).toFixed(0)}%</p>
                <p class="text-[10px] text-gray-500 uppercase">Stable</p>
            </div>
        </div>

        <!-- Style Archetypes -->
        <div class="mb-4">
            <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                <span class="w-2 h-2 bg-purple-500 rounded-full"></span> Style Archetypes
            </h4>
            <div class="bg-gray-50 rounded-lg p-3">
                ${archetypeHtml || '<p class="text-xs text-gray-400">Not enough data</p>'}
            </div>
        </div>

        <!-- Visual Preferences -->
        <div class="mb-4">
            <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                <span class="w-2 h-2 bg-green-500 rounded-full"></span> Visual Preferences
            </h4>
            <div class="grid grid-cols-2 gap-2">
                <div class="bg-green-50 rounded-lg p-2">
                    <p class="text-[10px] text-green-600 font-semibold mb-1">DRAWN TO</p>
                    <p class="text-sm font-medium">${likedAnchors.map(a => a.anchor).join(', ') || '-'}</p>
                </div>
                <div class="bg-red-50 rounded-lg p-2">
                    <p class="text-[10px] text-red-600 font-semibold mb-1">AVOIDS</p>
                    <p class="text-sm font-medium">${dislikedAnchors.map(a => a.anchor).join(', ') || '-'}</p>
                </div>
            </div>
        </div>

        <!-- Categories -->
        <div class="mb-4">
            <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                <span class="w-2 h-2 bg-indigo-500 rounded-full"></span> Category Breakdown
            </h4>
            <div class="bg-gray-50 rounded-lg p-3">
                ${categoriesHtml || '<p class="text-xs text-gray-400">Not enough data</p>'}
            </div>
        </div>

        <!-- Brands -->
        <div class="mb-4">
            <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                <span class="w-2 h-2 bg-gray-500 rounded-full"></span> Top Brands
            </h4>
            <div class="flex flex-wrap">
                ${brandsHtml || '<span class="text-xs text-gray-400">Not enough data</span>'}
            </div>
        </div>

        <!-- Colors -->
        <div class="mb-4">
            <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                <span class="w-2 h-2 bg-pink-500 rounded-full"></span> Color Preferences
            </h4>
            <div class="flex flex-wrap">
                ${colorsHtml || '<span class="text-xs text-gray-400">Not enough data</span>'}
            </div>
        </div>

        <!-- Fit & Fabric -->
        <div class="grid grid-cols-2 gap-3 mb-4">
            <div>
                <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2">Fit</h4>
                <div class="flex flex-wrap">${fitsHtml || '<span class="text-xs text-gray-400">-</span>'}</div>
            </div>
            <div>
                <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2">Fabric</h4>
                <div class="flex flex-wrap">${fabricsHtml || '<span class="text-xs text-gray-400">-</span>'}</div>
            </div>
        </div>

        <!-- Technical -->
        <div class="border-t border-gray-200 pt-3 mt-3">
            <div class="flex justify-between text-xs text-gray-500">
                <span>Coverage: ${stats.coverage || '-'}</span>
                <span>Stability: ${(stats.taste_stability || 0).toFixed(3)}</span>
            </div>
        </div>
    `;

    document.getElementById('complete-modal').classList.remove('hidden');
}

function showSetup() {
    document.getElementById('complete-modal').classList.add('hidden');
    document.getElementById('ranking-screen').classList.add('hidden');
    document.getElementById('setup-screen').classList.remove('hidden');
    document.getElementById('setup-footer').classList.remove('hidden');

    // Reset all state
    selectedColors = new Set();
    selectedCoreTypes = new Set();
    selectedSize = 'M';
    selectedStyles = new Set();
    selectedNecklines = new Set();
    selectedSleeves = new Set();
    selectedMaterials = new Set();
    currentItems = [];
    rankedOrder = [];

    // Reset core type buttons
    document.querySelectorAll('.core-type-btn').forEach(btn => {
        btn.classList.remove('border-2', 'border-black', 'bg-black', 'text-white');
        btn.classList.add('border-gray-200', 'bg-white');
        btn.querySelector('img').classList.remove('invert', 'brightness-0');
        btn.querySelectorAll('span').forEach(s => s.classList.remove('text-white', 'text-gray-300'));
    });

    // Hide t-shirt styles section
    document.getElementById('tshirt-styles-section').classList.add('hidden');

    // Reset style buttons
    document.querySelectorAll('.style-btn').forEach(btn => {
        btn.classList.remove('border-2', 'border-black');
        btn.classList.add('border-gray-200');
    });

    // Reset neckline buttons
    document.querySelectorAll('.neckline-btn').forEach(btn => {
        btn.querySelector('div').classList.remove('ring-2', 'ring-black', 'ring-offset-2');
    });

    // Reset sleeve buttons
    document.querySelectorAll('.sleeve-btn').forEach(btn => {
        const imgDiv = btn.querySelector('div');
        imgDiv.classList.remove('ring-2', 'ring-black', 'ring-offset-2');
    });

    // Reset size selector
    document.querySelectorAll('.size-btn').forEach(btn => {
        if (btn.dataset.size === 'M') {
            btn.classList.add('text-white', 'bg-black', 'shadow-md');
            btn.classList.remove('text-text-main', 'hover:bg-gray-50');
        } else {
            btn.classList.remove('text-white', 'bg-black', 'shadow-md');
            btn.classList.add('text-text-main', 'hover:bg-gray-50');
        }
    });

    // Reset material buttons
    document.querySelectorAll('.material-btn').forEach(btn => {
        const img = btn.querySelector('img');
        const label = btn.querySelector('.material-label');
        img.classList.remove('grayscale', 'opacity-50');
        label.classList.remove('bg-gray-200', 'text-gray-500', 'line-through');
        label.classList.add('bg-white', 'text-black');
    });

    // Reset color selector
    document.getElementById('color-selector').innerHTML = '';
    initColorSelector();
}
</script>

</body>
</html>'''


def get_attr_test_html() -> str:
    """Return the HTML template for the attribute isolation test interface."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
    <title>Style Discovery - Attribute Test</title>
    <script src="https://cdn.tailwindcss.com?plugins=forms,typography"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@200;300;400;500;600;700;800&family=Noto+Sans:wght@400;500;700&display=swap" rel="stylesheet"/>
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        "primary": "#1A1A1A",
                        "primary-dark": "#0A0A0A",
                        "background-light": "#f9fafb",
                        "surface-light": "#ffffff",
                        "text-main": "#131811",
                        "text-muted": "#6c7275",
                    },
                    fontFamily: {
                        display: ["Plus Jakarta Sans", "sans-serif"],
                        body: ["Noto Sans", "sans-serif"],
                    },
                }
            }
        };
    </script>
    <style>
        body::-webkit-scrollbar { display: none; }
        .no-scrollbar::-webkit-scrollbar { display: none; }
        .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }

        @keyframes cardFlyIn0 {
            0% { transform: translateX(-100%) translateY(-100%) scale(0.5); opacity: 0; }
            100% { transform: translateX(0) translateY(0) scale(1); opacity: 1; }
        }
        @keyframes cardFlyIn1 {
            0% { transform: translateX(100%) translateY(-100%) scale(0.5); opacity: 0; }
            100% { transform: translateX(0) translateY(0) scale(1); opacity: 1; }
        }
        @keyframes cardFlyIn2 {
            0% { transform: translateX(-100%) translateY(100%) scale(0.5); opacity: 0; }
            100% { transform: translateX(0) translateY(0) scale(1); opacity: 1; }
        }
        @keyframes cardFlyIn3 {
            0% { transform: translateX(100%) translateY(100%) scale(0.5); opacity: 0; }
            100% { transform: translateX(0) translateY(0) scale(1); opacity: 1; }
        }
        @keyframes cardSelect {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); box-shadow: 0 0 30px rgba(34, 197, 94, 0.5); }
            100% { transform: scale(0); opacity: 0; }
        }
        @keyframes cardFade {
            0% { opacity: 1; transform: scale(1); }
            100% { opacity: 0; transform: scale(0.8); }
        }
        @keyframes phaseTransition {
            0% { transform: scale(0.9); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

        .card-fly-0 { animation: cardFlyIn0 0.4s ease-out forwards; }
        .card-fly-1 { animation: cardFlyIn1 0.4s ease-out 0.05s forwards; opacity: 0; }
        .card-fly-2 { animation: cardFlyIn2 0.4s ease-out 0.1s forwards; opacity: 0; }
        .card-fly-3 { animation: cardFlyIn3 0.4s ease-out 0.15s forwards; opacity: 0; }
        .card-selected { animation: cardSelect 0.5s ease-out forwards; }
        .card-fade { animation: cardFade 0.3s ease-out forwards; }
        .phase-transition { animation: phaseTransition 0.5s ease-out forwards; }

        .choice-card {
            transition: all 0.2s ease;
            cursor: pointer;
        }
        .choice-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.15);
        }

        .phase-dot {
            transition: all 0.3s ease;
        }
        .phase-dot.active {
            transform: scale(1.3);
        }
        .phase-dot.complete {
            background: linear-gradient(135deg, #10b981, #059669);
        }
    </style>
</head>
<body class="bg-background-light text-text-main font-display antialiased min-h-screen flex flex-col">

<!-- Header -->
<header class="sticky top-0 z-50 bg-surface-light border-b border-gray-100 px-6 py-3 shadow-sm">
    <div class="max-w-[1100px] mx-auto flex items-center justify-between">
        <h2 class="text-xl font-bold tracking-tight">Attribute Discovery</h2>
        <button onclick="showSetup()" class="text-sm font-medium text-text-muted hover:text-text-main transition-colors">
            Restart
        </button>
    </div>
</header>

<!-- Setup Screen (simplified - skip straight to quiz for now) -->
<div id="setup-screen" class="flex-grow flex flex-col items-center justify-center py-8 px-4">
    <div class="text-center max-w-md">
        <div class="text-6xl mb-6">🎨</div>
        <h1 class="text-3xl font-bold mb-4">Discover Your Style DNA</h1>
        <p class="text-gray-600 mb-8">
            We'll test one attribute at a time - fit, color, pattern, and more.
            Each set of 4 items is similar, except for one key difference.
        </p>
        <div class="space-y-3 text-left bg-gray-50 rounded-xl p-6 mb-8">
            <div class="flex items-center gap-3">
                <span class="text-2xl">1️⃣</span>
                <span class="text-sm">Style Foundation - discover your vibe</span>
            </div>
            <div class="flex items-center gap-3">
                <span class="text-2xl">2️⃣</span>
                <span class="text-sm">Category & Logo - what type speaks to you</span>
            </div>
            <div class="flex items-center gap-3">
                <span class="text-2xl">3️⃣</span>
                <span class="text-sm">Fit & Form - find your ideal shape</span>
            </div>
            <div class="flex items-center gap-3">
                <span class="text-2xl">4️⃣</span>
                <span class="text-sm">Visual Elements - patterns & graphics</span>
            </div>
            <div class="flex items-center gap-3">
                <span class="text-2xl">5️⃣</span>
                <span class="text-sm">Color Palette - build your color profile</span>
            </div>
            <div class="flex items-center gap-3">
                <span class="text-2xl">6️⃣</span>
                <span class="text-sm">Material Feel - texture preferences</span>
            </div>
        </div>
        <button onclick="startSession()" class="bg-primary hover:bg-primary-dark text-white font-bold py-4 px-12 rounded-xl shadow-md transition-all transform active:scale-95 flex items-center gap-2 text-lg mx-auto">
            Start Discovery
            <span class="material-symbols-outlined">arrow_forward</span>
        </button>
    </div>
</div>

<!-- Main Quiz Screen -->
<main id="quiz-screen" class="hidden flex-grow flex flex-col items-center px-4 py-6">
    <!-- Phase Header -->
    <div id="phase-header" class="w-full max-w-2xl mb-4 phase-transition">
        <div class="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-4 border border-indigo-100">
            <div class="flex items-center justify-between mb-2">
                <div class="flex items-center gap-2">
                    <span id="phase-icon" class="text-2xl">🎨</span>
                    <div>
                        <span id="phase-name" class="font-bold text-indigo-900">Style Foundation</span>
                        <p id="phase-desc" class="text-xs text-indigo-600">Discover your overall style vibe</p>
                    </div>
                </div>
                <div class="text-right">
                    <span class="text-xs text-gray-500">Phase</span>
                    <div class="font-bold text-indigo-900"><span id="phase-num">1</span>/6</div>
                </div>
            </div>
            <!-- Phase dots -->
            <div class="flex justify-center gap-2 mt-2">
                <div class="phase-dot w-3 h-3 rounded-full bg-indigo-500 active" data-phase="1"></div>
                <div class="phase-dot w-3 h-3 rounded-full bg-gray-300" data-phase="2"></div>
                <div class="phase-dot w-3 h-3 rounded-full bg-gray-300" data-phase="3"></div>
                <div class="phase-dot w-3 h-3 rounded-full bg-gray-300" data-phase="4"></div>
                <div class="phase-dot w-3 h-3 rounded-full bg-gray-300" data-phase="5"></div>
                <div class="phase-dot w-3 h-3 rounded-full bg-gray-300" data-phase="6"></div>
            </div>
        </div>
    </div>

    <!-- Progress -->
    <div class="w-full max-w-2xl mb-4">
        <div class="flex justify-between items-center mb-2">
            <span class="text-sm font-medium text-gray-600">
                Testing: <span id="test-attr" class="font-bold text-indigo-600">archetype</span>
            </span>
            <span class="text-sm text-gray-500">Round <span id="round-num">1</span></span>
        </div>
        <div class="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
            <div id="progress-bar" class="h-full bg-gradient-to-r from-indigo-400 to-purple-500 rounded-full transition-all duration-500" style="width: 0%"></div>
        </div>
    </div>

    <!-- Instruction -->
    <div class="text-center mb-4">
        <h2 class="text-xl font-bold">Which <span id="attr-label" class="text-indigo-600">style</span> do you prefer?</h2>
        <p class="text-sm text-gray-500 mt-1">These items are similar - pick the one that speaks to you most</p>
    </div>

    <!-- 2x2 Grid -->
    <div id="cards-grid" class="grid grid-cols-2 gap-4 w-full max-w-2xl">
        <!-- Cards will be inserted here -->
    </div>

    <!-- Skip Button -->
    <button onclick="skipAll()" class="mt-6 text-gray-400 hover:text-gray-600 text-sm font-medium flex items-center gap-1">
        <span class="material-symbols-outlined text-base">refresh</span>
        None of these
    </button>
</main>

<!-- Complete Modal -->
<div id="complete-modal" class="hidden fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 overflow-y-auto">
    <div class="bg-white rounded-2xl shadow-2xl max-w-lg w-full p-6 my-8">
        <div class="text-center mb-4">
            <div class="w-14 h-14 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <span class="material-symbols-outlined text-green-500 text-2xl">check_circle</span>
            </div>
            <h3 class="text-xl font-bold">Style DNA Complete!</h3>
            <p class="text-gray-500 text-sm">Completed in <span id="final-rounds">0</span> rounds</p>
        </div>
        <div id="final-stats" class="text-left max-h-[60vh] overflow-y-auto pr-2"></div>
        <button onclick="showSetup()" class="w-full bg-gray-900 text-white py-3 rounded-xl font-semibold hover:bg-gray-800 mt-4">
            Start New Discovery
        </button>
    </div>
</div>

<script>
// State
let currentItems = [];
let isAnimating = false;
let currentTestInfo = {};
let currentProgress = {};

async function startSession() {
    try {
        const response = await fetch('/api/attr/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: 'default_user' })
        });
        const data = await response.json();

        if (data.status === 'started') {
            currentItems = data.items;
            currentTestInfo = data.test_info;
            currentProgress = data.progress;
            document.getElementById('setup-screen').classList.add('hidden');
            document.getElementById('quiz-screen').classList.remove('hidden');
            updatePhaseDisplay(data.test_info);
            updateProgress(1, data.progress);
            renderCards(data.items);
        }
    } catch (error) {
        console.error('Failed to start:', error);
    }
}

function updatePhaseDisplay(testInfo) {
    document.getElementById('phase-icon').textContent = testInfo.phase_icon || '🎨';
    document.getElementById('phase-name').textContent = testInfo.phase_name || 'Style Foundation';
    document.getElementById('phase-desc').textContent = testInfo.phase_description || '';
    document.getElementById('phase-num').textContent = testInfo.phase || 1;
    document.getElementById('test-attr').textContent = formatAttrName(testInfo.attribute);
    document.getElementById('attr-label').textContent = formatAttrName(testInfo.attribute);

    // Update phase dots
    document.querySelectorAll('.phase-dot').forEach((dot, idx) => {
        const phaseNum = idx + 1;
        dot.classList.remove('active', 'complete', 'bg-indigo-500', 'bg-gray-300');
        if (phaseNum < testInfo.phase) {
            dot.classList.add('complete');
        } else if (phaseNum === testInfo.phase) {
            dot.classList.add('active', 'bg-indigo-500');
        } else {
            dot.classList.add('bg-gray-300');
        }
    });
}

function formatAttrName(attr) {
    const names = {
        'archetype': 'style',
        'category': 'type',
        'logo_style': 'logo placement',
        'fit': 'fit',
        'neckline': 'neckline',
        'pattern_density': 'pattern',
        'color_family': 'color palette',
        'color_tone': 'color tone',
        'material': 'material'
    };
    return names[attr] || attr.replace('_', ' ');
}

function renderCards(items) {
    const grid = document.getElementById('cards-grid');
    grid.innerHTML = '';

    items.forEach((item, idx) => {
        const attrValue = item.test_attribute_value || 'unknown';
        const card = document.createElement('div');
        card.className = `choice-card bg-white rounded-2xl shadow-lg overflow-hidden card-fly-${idx}`;
        card.onclick = () => selectCard(item.id, idx);
        card.innerHTML = `
            <div class="aspect-square bg-gray-100 overflow-hidden relative">
                <img src="${item.image_url}" alt="${item.category}" class="w-full h-full object-contain"/>
                <div class="absolute bottom-2 left-2 right-2">
                    <span class="inline-block bg-indigo-600 text-white text-xs font-bold px-2 py-1 rounded-full">
                        ${formatAttrValue(attrValue)}
                    </span>
                </div>
            </div>
            <div class="p-3">
                <p class="font-bold text-sm truncate">${item.brand}</p>
                <p class="text-xs text-gray-500">${item.color} • ${item.category.replace(' T-shirts', '')}</p>
            </div>
        `;
        grid.appendChild(card);
    });
}

function formatAttrValue(value) {
    if (!value || value === 'unknown') return '?';
    return value.replace('_', ' ').replace(/\\b\\w/g, l => l.toUpperCase());
}

async function selectCard(itemId, idx) {
    if (isAnimating) return;
    isAnimating = true;

    const cards = document.querySelectorAll('.choice-card');
    cards.forEach((card, i) => {
        if (i === idx) {
            card.classList.add('card-selected');
        } else {
            card.classList.add('card-fade');
        }
    });

    await new Promise(r => setTimeout(r, 400));

    try {
        const response = await fetch('/api/attr/choose', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: 'default_user',
                winner_id: itemId
            })
        });
        const data = await response.json();

        if (data.session_complete) {
            showComplete(data.stats);
        } else {
            currentItems = data.items;
            currentTestInfo = data.test_info;
            currentProgress = data.progress;
            updatePhaseDisplay(data.test_info);
            updateProgress(data.round, data.progress);
            renderCards(data.items);
        }
    } catch (error) {
        console.error('Failed to record choice:', error);
    }

    isAnimating = false;
}

async function skipAll() {
    if (isAnimating) return;
    isAnimating = true;

    const cards = document.querySelectorAll('.choice-card');
    cards.forEach(card => card.classList.add('card-fade'));

    await new Promise(r => setTimeout(r, 300));

    try {
        const response = await fetch('/api/attr/skip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: 'default_user' })
        });
        const data = await response.json();

        if (data.session_complete) {
            showComplete(data.stats);
        } else {
            currentItems = data.items;
            currentTestInfo = data.test_info;
            currentProgress = data.progress;
            updatePhaseDisplay(data.test_info);
            updateProgress(data.round, data.progress);
            renderCards(data.items);
        }
    } catch (error) {
        console.error('Failed to skip:', error);
    }

    isAnimating = false;
}

function updateProgress(round, progress) {
    document.getElementById('round-num').textContent = round;
    const pct = progress?.percent_complete || (round / 20 * 100);
    document.getElementById('progress-bar').style.width = `${Math.min(pct, 100)}%`;
}

function showComplete(stats) {
    document.getElementById('final-rounds').textContent = stats.rounds_completed || 0;

    const finalStats = document.getElementById('final-stats');
    const learnedPrefs = stats.learned_preferences || {};
    const phases = stats.phases || [];

    // Build learned preferences section
    let learnedHtml = '<h4 class="font-bold text-gray-900 mb-3">Your Style DNA</h4>';
    learnedHtml += '<div class="space-y-2">';

    for (const [attr, info] of Object.entries(learnedPrefs)) {
        const conf = info.confidence || 0;
        const confPct = Math.round(conf * 100);
        const confColor = conf > 0.7 ? 'text-green-600' : conf > 0.4 ? 'text-yellow-600' : 'text-gray-500';
        learnedHtml += `
            <div class="flex items-center justify-between py-2 border-b border-gray-100">
                <span class="text-sm capitalize">${formatAttrName(attr)}</span>
                <div class="flex items-center gap-2">
                    <span class="font-bold text-indigo-600">${formatAttrValue(info.value)}</span>
                    <span class="text-xs ${confColor}">${confPct}%</span>
                </div>
            </div>
        `;
    }
    learnedHtml += '</div>';

    // Phases completed
    let phasesHtml = '<h4 class="font-bold text-gray-900 mb-3 mt-6">Phases Completed</h4>';
    phasesHtml += '<div class="grid grid-cols-2 gap-2">';
    for (const phase of phases) {
        const statusIcon = phase.complete ? '✅' : '⏳';
        phasesHtml += `
            <div class="flex items-center gap-2 text-sm ${phase.complete ? 'text-green-700' : 'text-gray-500'}">
                <span>${statusIcon}</span>
                <span>${phase.icon || ''} ${phase.name}</span>
            </div>
        `;
    }
    phasesHtml += '</div>';

    // Stats summary
    const fmt = (n) => Number.isInteger(n) ? n : Math.round(n * 10) / 10;
    let statsHtml = '<h4 class="font-bold text-gray-900 mb-3 mt-6">Session Stats</h4>';
    statsHtml += `
        <div class="grid grid-cols-2 gap-3 text-sm">
            <div class="bg-gray-50 rounded-lg p-3 text-center">
                <div class="font-bold text-2xl text-indigo-600">${stats.rounds_completed || 0}</div>
                <div class="text-gray-500 text-xs">Rounds</div>
            </div>
            <div class="bg-gray-50 rounded-lg p-3 text-center">
                <div class="font-bold text-2xl text-purple-600">${stats.phases_completed || 0}/${stats.total_phases || 6}</div>
                <div class="text-gray-500 text-xs">Phases</div>
            </div>
            <div class="bg-gray-50 rounded-lg p-3 text-center">
                <div class="font-bold text-2xl text-green-600">${Object.keys(learnedPrefs).length}</div>
                <div class="text-gray-500 text-xs">Preferences Learned</div>
            </div>
            <div class="bg-gray-50 rounded-lg p-3 text-center">
                <div class="font-bold text-2xl text-orange-600">${fmt(stats.information_bits || 0)}</div>
                <div class="text-gray-500 text-xs">Info Bits</div>
            </div>
        </div>
    `;

    finalStats.innerHTML = learnedHtml + phasesHtml + statsHtml;
    document.getElementById('complete-modal').classList.remove('hidden');
}

function showSetup() {
    document.getElementById('complete-modal').classList.add('hidden');
    document.getElementById('quiz-screen').classList.add('hidden');
    document.getElementById('setup-screen').classList.remove('hidden');
}
</script>

</body>
</html>'''


def get_women_html() -> str:
    """Return the HTML template for the women's fashion interface."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
    <title>Women's Style Discovery</title>
    <script src="https://cdn.tailwindcss.com?plugins=forms,typography"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@200;300;400;500;600;700;800&family=Noto+Sans:wght@400;500;700&display=swap" rel="stylesheet"/>
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        "primary": "#1A1A1A",
                        "background-light": "#fdf2f8",
                        "surface-light": "#ffffff",
                        "text-main": "#131811",
                        "text-muted": "#6c7275",
                    },
                    fontFamily: {
                        display: ["Plus Jakarta Sans", "sans-serif"],
                    },
                }
            }
        };
    </script>
    <style>
        @keyframes cardFlyIn0 { 0% { transform: translateX(-100%) translateY(-100%) scale(0.5); opacity: 0; } 100% { transform: translateX(0) translateY(0) scale(1); opacity: 1; } }
        @keyframes cardFlyIn1 { 0% { transform: translateX(100%) translateY(-100%) scale(0.5); opacity: 0; } 100% { transform: translateX(0) translateY(0) scale(1); opacity: 1; } }
        @keyframes cardFlyIn2 { 0% { transform: translateX(-100%) translateY(100%) scale(0.5); opacity: 0; } 100% { transform: translateX(0) translateY(0) scale(1); opacity: 1; } }
        @keyframes cardFlyIn3 { 0% { transform: translateX(100%) translateY(100%) scale(0.5); opacity: 0; } 100% { transform: translateX(0) translateY(0) scale(1); opacity: 1; } }
        @keyframes cardSelect { 0% { transform: scale(1); } 50% { transform: scale(1.05); box-shadow: 0 0 30px rgba(236, 72, 153, 0.5); } 100% { transform: scale(0); opacity: 0; } }
        @keyframes cardFade { 0% { opacity: 1; transform: scale(1); } 100% { opacity: 0; transform: scale(0.8); } }
        .card-fly-0 { animation: cardFlyIn0 0.4s ease-out forwards; }
        .card-fly-1 { animation: cardFlyIn1 0.4s ease-out 0.05s forwards; opacity: 0; }
        .card-fly-2 { animation: cardFlyIn2 0.4s ease-out 0.1s forwards; opacity: 0; }
        .card-fly-3 { animation: cardFlyIn3 0.4s ease-out 0.15s forwards; opacity: 0; }
        .card-selected { animation: cardSelect 0.5s ease-out forwards; }
        .card-fade { animation: cardFade 0.3s ease-out forwards; }
        .choice-card { transition: all 0.2s ease; cursor: pointer; }
        .choice-card:hover { transform: translateY(-4px); box-shadow: 0 12px 24px rgba(0,0,0,0.15); }
        .category-chip.selected { background: #1A1A1A; color: white; }
    </style>
</head>
<body class="bg-background-light text-text-main font-display antialiased min-h-screen flex flex-col">

<header class="sticky top-0 z-50 bg-surface-light border-b border-pink-100 px-6 py-3 shadow-sm">
    <div class="max-w-[1100px] mx-auto flex items-center justify-between">
        <h2 class="text-xl font-bold tracking-tight text-pink-600">Women's Style Discovery</h2>
        <button onclick="showSetup()" class="text-sm font-medium text-text-muted hover:text-text-main">Restart</button>
    </div>
</header>

<div id="setup-screen" class="flex-grow flex flex-col items-center py-8 px-4 sm:px-6">
    <div class="w-full max-w-[960px] flex flex-col gap-10">
        <div class="text-center">
            <h2 class="text-sm font-bold text-pink-400 uppercase tracking-wider mb-2">CATEGORIES</h2>
            <h3 class="text-2xl font-bold text-text-main">What are you looking for?</h3>
            <p class="text-gray-500 text-sm mt-2">Select categories to focus on (optional)</p>
        </div>
        <div class="flex flex-wrap justify-center gap-3" id="category-selection">
            <button onclick="toggleCategory('tops_knitwear')" data-cat="tops_knitwear" class="category-chip px-4 py-2 rounded-full border border-gray-300 text-sm font-medium hover:border-gray-400">Sweaters & Knits</button>
            <button onclick="toggleCategory('tops_woven')" data-cat="tops_woven" class="category-chip px-4 py-2 rounded-full border border-gray-300 text-sm font-medium hover:border-gray-400">Blouses & Shirts</button>
            <button onclick="toggleCategory('tops_sleeveless')" data-cat="tops_sleeveless" class="category-chip px-4 py-2 rounded-full border border-gray-300 text-sm font-medium hover:border-gray-400">Tank Tops & Camis</button>
            <button onclick="toggleCategory('tops_special')" data-cat="tops_special" class="category-chip px-4 py-2 rounded-full border border-gray-300 text-sm font-medium hover:border-gray-400">Bodysuits</button>
            <button onclick="toggleCategory('dresses')" data-cat="dresses" class="category-chip px-4 py-2 rounded-full border border-gray-300 text-sm font-medium hover:border-gray-400">Dresses</button>
            <button onclick="toggleCategory('bottoms_trousers')" data-cat="bottoms_trousers" class="category-chip px-4 py-2 rounded-full border border-gray-300 text-sm font-medium hover:border-gray-400">Pants & Trousers</button>
            <button onclick="toggleCategory('bottoms_skorts')" data-cat="bottoms_skorts" class="category-chip px-4 py-2 rounded-full border border-gray-300 text-sm font-medium hover:border-gray-400">Skirts & Shorts</button>
            <button onclick="toggleCategory('outerwear')" data-cat="outerwear" class="category-chip px-4 py-2 rounded-full border border-gray-300 text-sm font-medium hover:border-gray-400">Outerwear</button>
            <button onclick="toggleCategory('sportswear')" data-cat="sportswear" class="category-chip px-4 py-2 rounded-full border border-gray-300 text-sm font-medium hover:border-gray-400">Sportswear</button>
        </div>
        <div class="border-t border-pink-100 my-2"></div>
        <div class="text-center">
            <h2 class="text-sm font-bold text-pink-400 uppercase tracking-wider mb-2">PREFERENCES</h2>
            <h3 class="text-2xl font-bold text-text-main mb-2">Colors to Avoid</h3>
            <p class="text-gray-500 text-sm mb-8">Tap any color you dislike.</p>
            <div class="flex justify-center"><div id="color-selector" class="flex flex-wrap justify-center gap-4 max-w-[700px]"></div></div>
        </div>
        <div class="h-24"></div>
    </div>
</div>

<div id="setup-footer" class="sticky bottom-0 w-full bg-white/95 backdrop-blur-md border-t border-pink-200 py-4 px-6 shadow-lg z-40">
    <div class="max-w-[960px] mx-auto flex justify-center">
        <button onclick="startSession()" class="bg-pink-600 hover:bg-pink-700 text-white font-bold py-4 px-12 rounded-xl shadow-md flex items-center gap-2 text-lg">
            Start Style Quiz <span class="material-symbols-outlined">arrow_forward</span>
        </button>
    </div>
</div>

<main id="quiz-screen" class="hidden flex-grow flex flex-col items-center px-4 py-6">
    <div class="w-full max-w-2xl mb-6">
        <div class="text-center">
            <span class="text-xs font-medium text-pink-400 uppercase tracking-wider">Category <span id="cat-index">1</span> of <span id="cat-total">9</span></span>
            <h2 id="current-category" class="text-2xl font-bold text-gray-900 mt-1">Sweaters & Knits</h2>
        </div>
        <div class="mt-4 flex items-center gap-3">
            <div class="flex-1 h-2 bg-pink-100 rounded-full overflow-hidden">
                <div id="progress-bar" class="h-full bg-gradient-to-r from-pink-500 to-pink-600 rounded-full transition-all" style="width: 0%"></div>
            </div>
            <span class="text-xs text-gray-500 min-w-[60px]">Round <span id="round-num">1</span></span>
        </div>
    </div>
    <h3 class="text-lg font-medium text-gray-600 text-center mb-6">Which do you like most?</h3>
    <div id="cards-grid" class="grid grid-cols-2 gap-4 w-full max-w-2xl"></div>
    <button onclick="skipAll()" class="mt-6 text-gray-400 hover:text-gray-600 text-sm font-medium flex items-center gap-1">
        <span class="material-symbols-outlined text-base">refresh</span> None of these
    </button>
</main>

<div id="complete-modal" class="hidden fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 overflow-y-auto">
    <div class="bg-white rounded-2xl shadow-2xl max-w-lg w-full p-6 my-8">
        <div class="text-center mb-4">
            <div class="w-14 h-14 bg-pink-100 rounded-full flex items-center justify-center mx-auto mb-3">
                <span class="material-symbols-outlined text-pink-500 text-2xl">check_circle</span>
            </div>
            <h3 class="text-xl font-bold">Style Profile Complete!</h3>
            <p class="text-gray-500 text-sm">Completed in <span id="final-rounds">0</span> rounds</p>
        </div>
        <div id="final-stats" class="text-left max-h-[60vh] overflow-y-auto pr-2"></div>
        <button onclick="showSetup()" class="w-full bg-pink-600 text-white py-3 rounded-xl font-semibold hover:bg-pink-700 mt-4">Start New Quiz</button>
    </div>
</div>

<script>
const COLORS = [
    { name: 'black', bg: 'bg-black' }, { name: 'white', bg: 'bg-white border border-gray-200' },
    { name: 'gray', bg: 'bg-gray-400' }, { name: 'navy', bg: 'bg-blue-900' }, { name: 'blue', bg: 'bg-blue-500' },
    { name: 'red', bg: 'bg-red-500' }, { name: 'green', bg: 'bg-green-600' }, { name: 'brown', bg: 'bg-amber-700' },
    { name: 'pink', bg: 'bg-pink-400' }, { name: 'yellow', bg: 'bg-yellow-400' }, { name: 'orange', bg: 'bg-orange-400' },
    { name: 'purple', bg: 'bg-purple-500' }, { name: 'cream', bg: 'bg-amber-50 border border-gray-200' },
];
const CATEGORY_LABELS = {
    'tops_knitwear': 'Sweaters & Knits', 'tops_woven': 'Blouses & Shirts', 'tops_sleeveless': 'Tank Tops & Camis',
    'tops_special': 'Bodysuits', 'dresses': 'Dresses', 'bottoms_trousers': 'Pants & Trousers',
    'bottoms_skorts': 'Skirts & Shorts', 'outerwear': 'Outerwear', 'sportswear': 'Sportswear'
};
let selectedColors = new Set(), selectedCategories = new Set(), currentItems = [], isAnimating = false, currentTestInfo = null, roundNum = 1;

document.addEventListener('DOMContentLoaded', () => { initColorSelector(); });

function initColorSelector() {
    const container = document.getElementById('color-selector');
    COLORS.forEach(color => {
        const chip = document.createElement('button');
        chip.className = `group relative size-12 rounded-full ${color.bg} shadow-sm hover:scale-110 transition-transform`;
        chip.innerHTML = `<span class="hidden group-hover:flex absolute inset-0 items-center justify-center ${['white','yellow','cream'].includes(color.name)?'text-gray-800':'text-white'} material-symbols-outlined text-lg">close</span>`;
        chip.onclick = () => toggleColor(color.name, chip);
        container.appendChild(chip);
    });
}

function toggleColor(name, chip) {
    if (selectedColors.has(name)) { selectedColors.delete(name); chip.classList.remove('ring-2','ring-primary','ring-offset-2','opacity-50','grayscale'); chip.querySelector('span').classList.add('hidden'); }
    else { selectedColors.add(name); chip.classList.add('ring-2','ring-primary','ring-offset-2','opacity-50','grayscale'); chip.querySelector('span').classList.remove('hidden'); chip.querySelector('span').classList.add('flex'); }
}

function toggleCategory(cat) {
    const btn = document.querySelector(`[data-cat="${cat}"]`);
    if (selectedCategories.has(cat)) { selectedCategories.delete(cat); btn.classList.remove('selected'); }
    else { selectedCategories.add(cat); btn.classList.add('selected'); }
}

async function startSession() {
    try {
        const requestBody = { user_id: 'women_user_' + Date.now(), gender: 'female', colors_to_avoid: Array.from(selectedColors), selected_categories: Array.from(selectedCategories) };
        const response = await fetch('/api/unified/four/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(requestBody) });
        const data = await response.json();
        if (data.status === 'started') {
            currentItems = data.items; currentTestInfo = data.test_info; roundNum = 1; window.userId = requestBody.user_id;
            document.getElementById('setup-screen').classList.add('hidden'); document.getElementById('setup-footer').classList.add('hidden'); document.getElementById('quiz-screen').classList.remove('hidden');
            renderCards(data.items); updateCategoryInfo(data.test_info); updateProgress(1);
        }
    } catch (error) { console.error('Failed to start:', error); alert('Failed to start: ' + error.message); }
}

function updateCategoryInfo(testInfo) { if (!testInfo) return; document.getElementById('current-category').textContent = CATEGORY_LABELS[testInfo.category] || testInfo.category; document.getElementById('cat-index').textContent = testInfo.category_index || 1; document.getElementById('cat-total').textContent = testInfo.total_categories || 9; }
function updateProgress(round) { document.getElementById('round-num').textContent = round; document.getElementById('progress-bar').style.width = Math.min((round/20)*100, 100) + '%'; }

function renderCards(items) {
    const grid = document.getElementById('cards-grid'); grid.innerHTML = '';
    items.forEach((item, idx) => {
        const card = document.createElement('div');
        card.className = `choice-card bg-white rounded-2xl shadow-lg overflow-hidden card-fly-${idx}`;
        card.onclick = () => selectCard(item.id, idx);
        const parts = item.id.split('/'), subcategory = parts[1] || '', displayCat = CATEGORY_LABELS[item.category||parts[0]] || subcategory;
        card.innerHTML = `<div class="aspect-square bg-gray-50 overflow-hidden"><img src="${item.image_url}" alt="${displayCat}" class="w-full h-full object-contain"/></div><div class="p-3"><p class="font-medium text-sm truncate text-gray-700">${subcategory.replace(/_/g,' ')}</p><p class="text-xs text-gray-400">${displayCat}</p></div>`;
        grid.appendChild(card);
    });
}

async function selectCard(itemId, idx) {
    if (isAnimating) return; isAnimating = true;
    document.querySelectorAll('.choice-card').forEach((card, i) => { card.classList.add(i === idx ? 'card-selected' : 'card-fade'); });
    await new Promise(r => setTimeout(r, 400));
    try {
        const response = await fetch('/api/unified/four/choose', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ user_id: window.userId, gender: 'female', winner_id: itemId }) });
        const data = await response.json();
        if (data.session_complete || data.status === 'complete') { showComplete(data); }
        else { currentItems = data.items; currentTestInfo = data.test_info; roundNum++; renderCards(data.items); updateCategoryInfo(data.test_info); updateProgress(roundNum); }
    } catch (error) { console.error('Failed:', error); }
    isAnimating = false;
}

async function skipAll() {
    if (isAnimating) return; isAnimating = true;
    try {
        const response = await fetch('/api/unified/four/skip', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ user_id: window.userId, gender: 'female' }) });
        const data = await response.json();
        if (data.session_complete || data.status === 'complete') { showComplete(data); }
        else { currentItems = data.items; currentTestInfo = data.test_info; roundNum++; renderCards(data.items); updateCategoryInfo(data.test_info); updateProgress(roundNum); }
    } catch (error) { console.error('Failed:', error); }
    isAnimating = false;
}

function showComplete(data) {
    document.getElementById('final-rounds').textContent = roundNum;
    const statsDiv = document.getElementById('final-stats');
    const stats = data.stats || {};
    const feedPreview = data.feed_preview || {};
    const attrPrefs = stats.attribute_preferences || {};
    const categoryStats = stats.category_stats || {};
    const likes = stats.likes || 0;
    const dislikes = stats.dislikes || 0;
    const catsCompleted = (stats.categories_completed || []).length;
    const totalCats = stats.total_categories || 9;

    // Build category stats HTML
    let categoryStatsHtml = '';
    Object.entries(categoryStats).forEach(([cat, catData]) => {
        if (catData.rounds > 0) {
            const label = CATEGORY_LABELS[cat] || cat;
            const status = catData.complete ? '✓' : `${catData.rounds} rounds`;
            categoryStatsHtml += `<div class="flex justify-between text-xs py-1 border-b border-gray-100 last:border-0">
                <span class="font-medium">${label}</span>
                <span class="${catData.complete ? 'text-green-600' : 'text-gray-500'}">${status}</span>
            </div>`;
        }
    });

    // Build attribute preferences (top 3 preferred for key attributes)
    let stylePrefsHtml = '';
    const keyAttrs = ['pattern', 'style', 'color_family', 'fit_vibe', 'neckline'];
    keyAttrs.forEach(attr => {
        const data = attrPrefs[attr];
        if (data && data.preferred && data.preferred.length > 0) {
            const prefs = data.preferred.slice(0, 2).map(p => p[0]).join(', ');
            stylePrefsHtml += `<div class="bg-pink-50 rounded-lg p-2 text-center">
                <p class="text-[10px] text-pink-600 font-semibold uppercase">${attr.replace('_', ' ')}</p>
                <p class="text-sm font-medium">${prefs}</p>
            </div>`;
        }
    });

    statsDiv.innerHTML = `
        <!-- Overview -->
        <div class="grid grid-cols-3 gap-2 mb-4 text-center">
            <div class="bg-gray-50 rounded-lg p-3">
                <p class="text-xl font-bold">${stats.rounds_completed || roundNum}</p>
                <p class="text-[10px] text-gray-500 uppercase">Rounds</p>
            </div>
            <div class="bg-pink-50 rounded-lg p-3">
                <p class="text-xl font-bold text-pink-600">${likes}</p>
                <p class="text-[10px] text-gray-500 uppercase">Chosen</p>
            </div>
            <div class="bg-purple-50 rounded-lg p-3">
                <p class="text-xl font-bold text-purple-600">${catsCompleted}/${totalCats}</p>
                <p class="text-[10px] text-gray-500 uppercase">Categories</p>
            </div>
        </div>

        <!-- Style Preferences -->
        ${stylePrefsHtml ? `
        <div class="mb-4">
            <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                <span class="w-2 h-2 bg-pink-500 rounded-full"></span> Your Style Preferences
            </h4>
            <div class="grid grid-cols-2 gap-2">${stylePrefsHtml}</div>
        </div>
        ` : ''}

        <!-- Categories Tested -->
        ${categoryStatsHtml ? `
        <div class="mb-4">
            <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                <span class="w-2 h-2 bg-indigo-500 rounded-full"></span> Categories Tested
            </h4>
            <div class="bg-gray-50 rounded-lg p-3">${categoryStatsHtml}</div>
        </div>
        ` : ''}

        <!-- Feed Preview with Images -->
        ${feedPreview && Object.keys(feedPreview).length > 0 ? `
        <div class="mb-4">
            <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                <span class="w-2 h-2 bg-pink-500 rounded-full"></span> Recommended For You
            </h4>
            <p class="text-xs text-gray-400 mb-3">Based on your style preferences</p>
            ${Object.entries(feedPreview).map(([category, items]) => `
                <div class="mb-4">
                    <h5 class="text-sm font-semibold text-gray-700 mb-2">${CATEGORY_LABELS[category] || category}</h5>
                    <div class="grid grid-cols-4 gap-2">
                        ${items.slice(0, 8).map(item => `
                            <div class="relative group">
                                <div class="aspect-square bg-gray-100 rounded-lg overflow-hidden">
                                    <img src="${item.image_url}"
                                         alt="Recommendation"
                                         class="w-full h-full object-cover"
                                         onerror="this.parentElement.innerHTML='<div class=\\'flex items-center justify-center h-full text-gray-300 text-xs\\'>No img</div>'">
                                </div>
                                ${item.similarity ? `<div class="absolute top-1 right-1 bg-pink-500 text-white text-[8px] px-1 rounded">${Math.round(item.similarity * 100)}%</div>` : ''}
                                ${item.attributes ? `<div class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-1.5 rounded-b-lg">
                                    <p class="text-[8px] text-white truncate">${item.attributes.pattern || ''} ${item.attributes.style || ''}</p>
                                </div>` : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>
            `).join('')}
        </div>
        ` : ''}

        <!-- Technical -->
        <div class="border-t border-gray-200 pt-3 mt-3">
            <div class="flex justify-between text-xs text-gray-400">
                <span>Taste stability: ${(stats.taste_stability || 0).toFixed(3)}</span>
                <span>Items seen: ${likes + dislikes}</span>
            </div>
        </div>
    `;

    document.getElementById('quiz-screen').classList.add('hidden');
    document.getElementById('complete-modal').classList.remove('hidden');
}

function showSetup() {
    document.getElementById('complete-modal').classList.add('hidden'); document.getElementById('quiz-screen').classList.add('hidden');
    document.getElementById('setup-screen').classList.remove('hidden'); document.getElementById('setup-footer').classList.remove('hidden');
    selectedColors.clear(); selectedCategories.clear();
    document.querySelectorAll('.category-chip').forEach(b => b.classList.remove('selected'));
    document.querySelectorAll('#color-selector button').forEach(b => { b.classList.remove('ring-2','ring-primary','ring-offset-2','opacity-50','grayscale'); b.querySelector('span').classList.add('hidden'); });
}
</script>
</body>
</html>'''


if __name__ == "__main__":
    print("Starting Outrove Style Swipe Server...")
    print("Access at: http://localhost:8080")
    print("Four-choice demo at: http://localhost:8080/four")
    print("Women's fashion at: http://localhost:8080/women")
    print("Ranking demo at: http://localhost:8080/rank")
    print("Attribute test at: http://localhost:8080/attr")
    uvicorn.run(app, host="0.0.0.0", port=8080)
