"""
FastAPI REST API for Fashion Personalized Feed
Includes OutfitTransformer compatibility inference
"""
import os
import pickle
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from collections import defaultdict

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import random


# Pagination Models
class PaginationParams(BaseModel):
    """Common pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")


class PaginationMeta(BaseModel):
    """Pagination metadata for responses"""
    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_prev: bool


def paginate(items: list, page: int, page_size: int) -> tuple:
    """Helper function to paginate a list of items"""
    total_items = len(items)
    total_pages = max(1, (total_items + page_size - 1) // page_size)

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    paginated_items = items[start_idx:end_idx]

    meta = PaginationMeta(
        page=page,
        page_size=page_size,
        total_items=total_items,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1
    )

    return paginated_items, meta


# Request/Response Models
class FeedRequest(BaseModel):
    """Request model for feed generation"""
    user_id: str = Field(..., description="User identifier")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Weight configuration: visual, cf, diversity"
    )


class FeedItem(BaseModel):
    """Single item in feed response"""
    item_id: str
    score: float
    sources: List[str] = []


class FeedResponse(BaseModel):
    """Response model for feed"""
    user_id: str
    items: List[FeedItem]
    pagination: PaginationMeta


class SimilarRequest(BaseModel):
    """Request model for similar items"""
    item_id: str = Field(..., description="Item ID to find similar items for")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")


class SimilarResponse(BaseModel):
    """Response model for similar items"""
    item_id: str
    similar: List[FeedItem]
    pagination: PaginationMeta


class FeedbackRequest(BaseModel):
    """Request model for recording feedback"""
    user_id: str = Field(..., description="User identifier")
    item_id: str = Field(..., description="Item identifier")
    action: str = Field(..., description="Action type: view, like, purchase, etc.")


class FeedbackResponse(BaseModel):
    """Response model for feedback"""
    status: str
    user_id: str
    item_id: str


class ItemResponse(BaseModel):
    """Response model for item details"""
    item_id: str
    category: Optional[str] = None
    name: Optional[str] = None
    has_embedding: bool = False
    metadata: Dict = {}


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    items_count: int
    categories_count: int
    recbole_loaded: bool
    outfit_transformer_loaded: bool = False


# === OutfitTransformer Compatibility Models ===

class StyleThisItemRequest(BaseModel):
    """Request model for generating outfits around a single item"""
    item_id: str = Field(..., description="The item to build outfits around")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=3, ge=1, le=10, description="Outfits per page")
    items_per_outfit: int = Field(default=4, ge=2, le=6, description="Items per outfit (including the input item)")


class OutfitSet(BaseModel):
    """A complete outfit set"""
    items: List[Dict] = Field(..., description="Items in the outfit with metadata")
    compatibility_score: float
    style_description: Optional[str] = None


class StyleThisItemResponse(BaseModel):
    """Response with multiple outfit options"""
    anchor_item: Dict
    outfits: List[OutfitSet]
    pagination: PaginationMeta


class CompatibilityRequest(BaseModel):
    """Request model for outfit compatibility scoring"""
    item_ids: List[str] = Field(..., description="List of item IDs in the outfit", min_length=2)


class CompatibilityResponse(BaseModel):
    """Response model for compatibility score"""
    item_ids: List[str]
    compatibility_score: float = Field(..., description="Compatibility score (0-1)")
    item_count: int


class FITBRequest(BaseModel):
    """Request model for Fill-In-The-Blank prediction"""
    context_items: List[str] = Field(..., description="Context item IDs (outfit with one missing)", min_length=1)
    candidate_items: List[str] = Field(..., description="Candidate item IDs to score", min_length=1, max_length=10)


class FITBResponse(BaseModel):
    """Response model for FITB prediction"""
    context_items: List[str]
    ranked_candidates: List[Dict] = Field(..., description="Candidates ranked by compatibility score")
    best_match: str


class OutfitCompleteRequest(BaseModel):
    """Request model for outfit completion with category diversity"""
    item_ids: List[str] = Field(..., description="Current outfit items", min_length=1)
    target_categories: Optional[List[str]] = Field(
        default=None,
        description="Categories to complete outfit with (e.g., ['bottom', 'shoe']). If not specified, auto-completes to full outfit (top + bottom + shoe)"
    )
    gender: Optional[str] = Field(
        default=None,
        description="Gender filter for suggestions: 'women' or 'men'. Women see women's + unisex items, men see men's + unisex items."
    )
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=5, ge=1, le=20, description="Suggestions per category")
    category: Optional[str] = Field(default=None, description="DEPRECATED: Use target_categories instead")


class OutfitCompleteResponse(BaseModel):
    """Response model for outfit completion with category diversity"""
    current_items: List[str]
    current_categories: List[str] = Field(default=[], description="Categories of current items")
    target_categories: List[str] = Field(default=[], description="Categories being suggested")
    suggestions: Dict[str, List[Dict]] = Field(
        default={},
        description="Suggestions grouped by category: {'bottom': [...], 'shoe': [...]}"
    )


class SimilarItemsRequest(BaseModel):
    """Request model for finding visually similar items"""
    item_id: str = Field(..., description="The item to find similar items for")
    k: int = Field(default=10, ge=1, le=50, description="Number of similar items to return")
    same_category: bool = Field(
        default=True,
        description="If True, only return items from same category (e.g., top -> tops). If False, return similar items from any category."
    )
    gender: Optional[str] = Field(
        default=None,
        description="Gender filter for suggestions: 'women' or 'men'. Women see women's + unisex items, men see men's + unisex items."
    )


class SimilarItemsResponse(BaseModel):
    """Response model for similar items"""
    item_id: str
    category: Optional[str] = None
    similar_items: List[Dict] = Field(default=[], description="List of similar items with scores")


# === Polyvore-U Feed Models ===

class ProductFilter(BaseModel):
    """Filter model for product queries"""
    categories: Optional[List[str]] = Field(
        default=None,
        description="Filter by fine-grained category (e.g., 'women's t-shirt', 'skirt')"
    )
    broad_categories: Optional[List[str]] = Field(
        default=None,
        description="Filter by broad category: 'top', 'bottom', 'shoe'"
    )
    genders: Optional[List[str]] = Field(
        default=None,
        description="Filter by gender: 'women', 'men', 'unisex'"
    )
    min_price: Optional[float] = Field(default=None, ge=0, description="Minimum price")
    max_price: Optional[float] = Field(default=None, ge=0, description="Maximum price")
    search: Optional[str] = Field(default=None, description="Search in product name")


class PolyvoreUFeedRequest(BaseModel):
    """Request model for Polyvore-U feed generation"""
    user_id: str = Field(..., description="User identifier")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    shuffle: bool = Field(default=True, description="Shuffle results for variety")
    filters: Optional[ProductFilter] = Field(default=None, description="Optional filters")


class PolyvoreUFeedItem(BaseModel):
    """Single item in Polyvore-U feed response"""
    item_id: int
    name: str = Field(..., description="Product name")
    price: float = Field(..., description="Product price")
    image_url: str = Field(..., description="Image URL path")
    category: str = Field(..., description="Fine-grained category (50 types)")
    broad_category: Optional[str] = Field(default=None, description="Broad category: top, bottom, shoe")
    gender: Optional[str] = Field(default=None, description="Gender: women, men, unisex")
    score: Optional[float] = Field(default=None, description="Relevance score")
    rating: Optional[float] = Field(default=None, description="Average rating (3.5-5.0)")
    review_count: Optional[int] = Field(default=None, description="Number of reviews")
    description: Optional[str] = Field(default=None, description="Product description")


class PolyvoreUFeedResponse(BaseModel):
    """Response model for Polyvore-U feed"""
    user_id: str
    items: List[PolyvoreUFeedItem]
    pagination: PaginationMeta
    liked_count: int = 0


class PolyvoreULikeRequest(BaseModel):
    """Request model for like/dislike actions"""
    user_id: str = Field(..., description="User identifier")
    item_id: int = Field(..., description="Item ID to like/dislike")


class PolyvoreULikeResponse(BaseModel):
    """Response model for like/dislike actions"""
    status: str
    user_id: str
    item_id: int
    action: str
    liked_count: int
    disliked_count: int


class PolyvoreUSimilarRequest(BaseModel):
    """Request model for similar items"""
    item_id: int = Field(..., description="Item ID to find similar items for")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    filters: Optional[ProductFilter] = Field(default=None, description="Optional filters")


class PolyvoreUSimilarResponse(BaseModel):
    """Response model for similar items"""
    item_id: int
    similar: List[PolyvoreUFeedItem]
    pagination: PaginationMeta


class PolyvoreUUserResponse(BaseModel):
    """Response model for user profile"""
    user_id: str
    liked_items: List[int]
    disliked_count: int
    liked_count: int


class PolyvoreUHealthResponse(BaseModel):
    """Response model for Polyvore-U health check"""
    status: str
    items_count: int
    categories: Dict[str, int]
    faiss_index_loaded: bool


# === Style Quiz Models ===

class StyleQuizStep2Request(BaseModel):
    """Request for step 2: get items from selected styles"""
    gender: str = Field(..., description="Gender: 'women' or 'men'")
    selected_styles: List[str] = Field(..., description="List of style names selected in step 1")


class SetupUserRequest(BaseModel):
    """Request to complete user setup with selected items"""
    user_id: str = Field(..., description="User identifier")
    gender: str = Field(..., description="Gender: 'women' or 'men'")
    selected_item_ids: List[int] = Field(..., description="List of item IDs user selected (minimum 15)")


class ExclusionsDict(BaseModel):
    """Exclusions preferences by category"""
    colors: Optional[List[str]] = Field(default=None, description="Colors to avoid")
    patterns: Optional[List[str]] = Field(default=None, description="Patterns to avoid")
    item_types: Optional[List[str]] = Field(default=None, description="Item types to avoid")
    shoes: Optional[List[str]] = Field(default=None, description="Shoe types to avoid")


class UserExclusionsRequest(BaseModel):
    """Request to set user exclusion preferences"""
    user_id: str = Field(..., description="User identifier")
    gender: str = Field(..., description="Gender: 'women' or 'men'")
    exclusions: ExclusionsDict = Field(..., description="Exclusion preferences by category")


class UserExclusionsResponse(BaseModel):
    """Response after setting exclusions"""
    user_id: str
    items_excluded: int
    message: str


class StyleCardResponse(BaseModel):
    """Single style card for step 1"""
    name: str
    description: str
    sample_images: List[str]
    preview_item_ids: List[int]


class StyleQuizStep1Response(BaseModel):
    """Response for step 1: style cards"""
    gender: str
    styles: List[StyleCardResponse]


class StyleQuizItemResponse(BaseModel):
    """Single item in style quiz with full product details"""
    item_id: int
    name: str = Field(..., description="Product name")
    price: float = Field(..., description="Product price")
    image_url: str = Field(..., description="Image URL path")
    category: str = Field(..., description="Fine-grained category")
    broad_category: Optional[str] = Field(default=None, description="Broad category: top, bottom, shoe")
    gender: Optional[str] = Field(default=None, description="Gender: women, men, unisex")
    style: str = Field(..., description="Style category (Casual, Formal, etc.)")
    description: Optional[str] = Field(default=None, description="Product description")


class StyleQuizStep2Response(BaseModel):
    """Response for step 2: items from selected styles"""
    gender: str
    selected_styles: List[str]
    items: List[StyleQuizItemResponse]
    minimum_selection: int = 5


class StyleQuizItemsResponse(BaseModel):
    """Response for simplified single-step style quiz"""
    gender: str
    items: List[StyleQuizItemResponse]
    total_items: int
    minimum_selection: int = 5


class SetupUserResponse(BaseModel):
    """Response after completing user setup"""
    user_id: str
    gender: str
    items_selected: int
    setup_complete: bool
    message: str


# Style Quiz Configuration
STYLE_DESCRIPTIONS = {
    'Casual': 'Relaxed, everyday clothing for comfort',
    'Formal': 'Professional and elegant attire',
    'Streetwear': 'Urban, trendy street fashion',
    'Minimalist': 'Clean lines, neutral colors, simple',
    'Bohemian': 'Free-spirited, artistic, natural',
    'Athleisure': 'Sporty, comfortable, active',
    'Vintage': '70s/80s/90s inspired retro looks',
    'Preppy': 'Classic, polished, collegiate style',
    'Romantic': 'Soft, feminine, floral patterns'
}

# Style quiz items (loaded from pickle file or generated candidates)
style_quiz_items: Dict[str, Dict[str, List[int]]] = {}

# User profiles with gender and setup status
user_setup_profiles: Dict[str, Dict] = {}


# Global state
generator = None
user_profiles: Dict[str, Dict] = defaultdict(lambda: {'history': [], 'feedback': []})

# Polyvore-U state
polyvore_u_generator = None
polyvore_u_image_paths = None
polyvore_u_image_dir = None
polyvore_u_user_history: Dict[str, List[int]] = defaultdict(list)  # user_id -> liked item_ids
polyvore_u_user_dislikes: Dict[str, set] = defaultdict(set)  # user_id -> disliked item_ids
polyvore_u_category_map = None  # category_id -> category_name (e.g., {0: "skirt", 1: "women's canvas shoe", ...})
polyvore_u_iid_cate_map = None  # item_id -> category_id (e.g., {67: 46, 103724: 47, ...})
polyvore_u_cate_iid_map = None  # category_id -> list of item_ids
fashion_items = None  # image_filename -> {name, text, price, class, categories}

# Broad category groupings for Polyvore-U (50 fine-grained -> 5 broad)
POLYVORE_U_BROAD_CATEGORIES = {
    "top": [
        "women's chiffon top", "women's sweater", "women's t-shirt", "women's sleeveless top",
        "women's shirt", "women's sweatshirt", "vest", "men's polo shirt", "men's sweater",
        "men's t-shirt", "men's shirt", "men's sweatshirt"
    ],
    "bottom": [
        "skirt", "women's casual pants", "women's jeans", "legging",
        "men's jeans", "men's casual pants"
    ],
    "shoe": [
        "women's canvas shoe", "women's slipper", "women's sandal", "women's casual shoe",
        "women's boot", "women's shoe", "ankle boot", "men's high-top shoe", "men's shoe", "canvas shoe"
    ],
    "outerwear": [
        "women's wool coat", "women's leather jacket", "women's winter jacket",
        "women's suit jacket", "women's casual coat", "trench coat", "men's jacket",
        "men's winter jacket", "men's leather jacket", "men's coat"
    ],
    "accessory": [
        "earrings", "bracelet", "belt", "ring", "necklace", "bangle",
        "crossbody bag", "travel bag", "pendant", "watch", "hat"
    ],
    "full_body": ["dress"]  # Can't be combined with tops or bottoms
}

# OutfitTransformer state
outfit_transformer = None
outfit_embeddings = None
outfit_metadata = None
outfit_category_to_idx = None
outfit_device = None
category_exclusions = None  # dress_ids and sock_ids to filter from outfit suggestions
gender_mapping = None  # Gender mapping for items (women_ids, men_ids, unisex_ids)
polyvore_item_to_image = None  # Reverse mapping: item_id -> image_key (e.g., "194508109" -> "214181831_1")
style_exclusions = None  # Style exclusion items: {category: {option: {'women': [...], 'men': [...]}}}

# Amazon Fashion state
amazon_feed = None
amazon_metadata = None
amazon_user_likes: Dict[str, set] = defaultdict(set)  # user_id -> liked item_ids
amazon_user_dislikes: Dict[str, set] = defaultdict(set)  # user_id -> disliked item_ids


def load_amazon_fashion():
    """Load Amazon Fashion feed system."""
    global amazon_feed, amazon_metadata

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check if required files exist
    embeddings_path = os.path.join(project_root, "data/amazon_fashion/processed/amazon_mens_embeddings.pkl")
    faiss_index_path = os.path.join(project_root, "models/amazon_mens_faiss.index")
    faiss_ids_path = os.path.join(project_root, "models/amazon_mens_faiss_ids.npy")
    sasrec_checkpoint = os.path.join(project_root, "models/sasrec_amazon/SASRec-Dec-12-2025_01-35-54.pth")
    metadata_path = os.path.join(project_root, "data/amazon_fashion/processed/item_metadata.pkl")
    interactions_path = os.path.join(project_root, "data/amazon_fashion/recbole/amazon_mens/amazon_mens.inter")

    if not os.path.exists(embeddings_path) or not os.path.exists(faiss_index_path):
        print("Warning: Amazon Fashion embeddings or Faiss index not found")
        return False

    try:
        from amazon_feed import AmazonFashionFeed

        amazon_feed = AmazonFashionFeed(
            embeddings_path=embeddings_path,
            faiss_index_path=faiss_index_path,
            faiss_ids_path=faiss_ids_path,
            duorec_checkpoint=sasrec_checkpoint if os.path.exists(sasrec_checkpoint) else None,
            user_interactions_path=interactions_path,
            use_gpu=True
        )

        # Load metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                amazon_metadata = pickle.load(f)
            print(f"  Loaded Amazon metadata for {len(amazon_metadata)} items")

        print(f"Amazon Fashion Feed loaded: {len(amazon_feed.item_ids)} items, {len(amazon_feed.user_history)} users")
        return True

    except Exception as e:
        print(f"Warning: Could not load Amazon Fashion Feed: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_outfit_transformer(checkpoint_path: str, embeddings_path: str, metadata_path: str, use_gpu: bool):
    """Load OutfitTransformer model for inference"""
    global outfit_transformer, outfit_embeddings, outfit_metadata, outfit_category_to_idx, outfit_device, polyvore_item_to_image

    import json
    from outfit_transformer import OutfitTransformer

    outfit_device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=outfit_device, weights_only=False)
    config = checkpoint['config']

    # Load embeddings
    with open(embeddings_path, 'rb') as f:
        emb_data = pickle.load(f)
        outfit_embeddings = emb_data['embeddings']

    # Load metadata for categories
    with open(metadata_path, 'r') as f:
        outfit_metadata = json.load(f)

    # Build category vocabulary (same as training)
    outfit_category_to_idx = {}
    for item_id, meta in outfit_metadata.items():
        cat = meta.get('category_name', 'unknown')
        if cat not in outfit_category_to_idx:
            outfit_category_to_idx[cat] = len(outfit_category_to_idx)
    if 'unknown' not in outfit_category_to_idx:
        outfit_category_to_idx['unknown'] = len(outfit_category_to_idx)

    # Create model
    outfit_transformer = OutfitTransformer(
        visual_dim=512,
        num_categories=config['num_categories'],
        num_items=config['num_items'],
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads']
    ).to(outfit_device)

    outfit_transformer.load_state_dict(checkpoint['model_state_dict'])
    outfit_transformer.eval()

    print(f"OutfitTransformer loaded from {checkpoint_path}")
    print(f"  Device: {outfit_device}")
    print(f"  Items with embeddings: {len(outfit_embeddings)}")
    print(f"  Categories: {len(outfit_category_to_idx)}")
    print(f"  FITB accuracy at checkpoint: {checkpoint.get('fitb_accuracy', 'N/A')}")

    # Load image_to_tid mapping and create reverse mapping (item_id -> image_key)
    image_mapping_path = os.path.join(os.path.dirname(metadata_path), "image_to_tid_mapping.json")
    if os.path.exists(image_mapping_path):
        with open(image_mapping_path, 'r') as f:
            img_to_tid = json.load(f)
        # Reverse mapping: item_id -> image_key
        polyvore_item_to_image = {}
        for img_key, tid in img_to_tid.items():
            if tid not in polyvore_item_to_image:
                polyvore_item_to_image[tid] = img_key
        print(f"  Loaded item-to-image mapping for {len(polyvore_item_to_image)} items")


def load_polyvore_u():
    """Load Polyvore-U candidate generator and data."""
    global polyvore_u_generator, polyvore_u_image_paths, polyvore_u_image_dir, polyvore_u_category_map
    global polyvore_u_iid_cate_map, polyvore_u_cate_iid_map, category_exclusions, gender_mapping, fashion_items, style_exclusions
    from pathlib import Path

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    polyvore_u_embeddings = os.path.join(project_root, "data/polyvore_u/polyvore_u_clip_embeddings.npy")
    polyvore_u_faiss = os.path.join(project_root, "models/polyvore_u_faiss.index")
    polyvore_u_images_npy = os.path.join(project_root, "data/polyvore_u/all_item_image_paths.npy")
    polyvore_u_category_npy = os.path.join(project_root, "data/polyvore_u/id_cate_dict.npy")
    polyvore_u_iid_cate_npy = os.path.join(project_root, "data/polyvore_u/map/iid_cate_dict.npy")
    polyvore_u_cate_iid_npy = os.path.join(project_root, "data/polyvore_u/map/cate_iid_dict.npy")

    if not os.path.exists(polyvore_u_embeddings) or not os.path.exists(polyvore_u_faiss):
        print("Warning: Polyvore-U embeddings or Faiss index not found")
        print(f"  Embeddings: {polyvore_u_embeddings} - exists: {os.path.exists(polyvore_u_embeddings)}")
        print(f"  Faiss index: {polyvore_u_faiss} - exists: {os.path.exists(polyvore_u_faiss)}")
        return False

    try:
        from candidate_generator import CandidateGenerator

        polyvore_u_generator = CandidateGenerator(
            embeddings_path=polyvore_u_embeddings,
            faiss_index_path=polyvore_u_faiss
        )

        # Load image paths
        if os.path.exists(polyvore_u_images_npy):
            polyvore_u_image_paths = np.load(polyvore_u_images_npy, allow_pickle=True)
            polyvore_u_image_dir = Path(os.path.join(project_root, "data/polyvore_u/291x291"))
            print(f"  Loaded {len(polyvore_u_image_paths)} image paths")
        else:
            print(f"  Warning: Image paths not found at {polyvore_u_images_npy}")

        # Load category mapping (category_id -> category_name)
        if os.path.exists(polyvore_u_category_npy):
            category_data = np.load(polyvore_u_category_npy, allow_pickle=True)
            # Handle 0-dim numpy array containing a dict
            if category_data.ndim == 0:
                polyvore_u_category_map = category_data.item()
            else:
                polyvore_u_category_map = category_data
            print(f"  Loaded category map with {len(polyvore_u_category_map)} entries")
        else:
            print(f"  Warning: Category map not found at {polyvore_u_category_npy}")

        # Load item -> category mapping (item_id -> category_id)
        if os.path.exists(polyvore_u_iid_cate_npy):
            iid_data = np.load(polyvore_u_iid_cate_npy, allow_pickle=True)
            if iid_data.ndim == 0:
                polyvore_u_iid_cate_map = iid_data.item()
            else:
                polyvore_u_iid_cate_map = iid_data
            print(f"  Loaded item->category map with {len(polyvore_u_iid_cate_map)} items")
        else:
            print(f"  Warning: Item->category map not found at {polyvore_u_iid_cate_npy}")

        # Load category -> items mapping (category_id -> list of item_ids)
        if os.path.exists(polyvore_u_cate_iid_npy):
            cate_data = np.load(polyvore_u_cate_iid_npy, allow_pickle=True)
            if cate_data.ndim == 0:
                polyvore_u_cate_iid_map = cate_data.item()
            else:
                polyvore_u_cate_iid_map = cate_data
            print(f"  Loaded category->items map with {len(polyvore_u_cate_iid_map)} categories")
        else:
            print(f"  Warning: Category->items map not found at {polyvore_u_cate_iid_npy}")

        # Load category exclusions (dresses and socks to filter from outfit suggestions)
        category_exclusions_path = os.path.join(project_root, "data/polyvore_u/category_exclusions.pkl")
        if os.path.exists(category_exclusions_path):
            with open(category_exclusions_path, 'rb') as f:
                category_exclusions = pickle.load(f)
            dress_count = len(category_exclusions.get('dress_ids', set()))
            sock_count = len(category_exclusions.get('sock_ids', set()))
            print(f"  Loaded category exclusions: {dress_count} dresses, {sock_count} socks")
        else:
            category_exclusions = {'dress_ids': set(), 'sock_ids': set()}
            print(f"  Warning: Category exclusions not found at {category_exclusions_path}")

        # Load gender mapping (women_ids, men_ids, unisex_ids for gender filtering)
        gender_mapping_path = os.path.join(project_root, "data/polyvore_u/gender_mapping.pkl")
        if os.path.exists(gender_mapping_path):
            with open(gender_mapping_path, 'rb') as f:
                gender_mapping = pickle.load(f)
            women_count = len(gender_mapping.get('women_ids', set()))
            men_count = len(gender_mapping.get('men_ids', set()))
            unisex_count = len(gender_mapping.get('unisex_ids', set()))
            print(f"  Loaded gender mapping: {women_count} women, {men_count} men, {unisex_count} unisex items")
        else:
            gender_mapping = {'women_ids': set(), 'men_ids': set(), 'unisex_ids': set(), 'item_to_gender': {}}
            print(f"  Warning: Gender mapping not found at {gender_mapping_path}")

        # Load style exclusions (items to auto-dislike for each exclusion option)
        style_exclusions_path = os.path.join(project_root, "data/polyvore_u/style_exclusions.pkl")
        if os.path.exists(style_exclusions_path):
            with open(style_exclusions_path, 'rb') as f:
                style_exclusions = pickle.load(f)
            total_options = sum(len(opts) for opts in style_exclusions.values())
            print(f"  Loaded style exclusions: {total_options} options across {len(style_exclusions)} categories")
        else:
            style_exclusions = None
            print(f"  Warning: Style exclusions not found at {style_exclusions_path}")

        # Load fashion items (real names and descriptions keyed by image filename)
        fashion_items_path = os.path.join(project_root, "data/polyvore_u/fashion_items.pickle")
        if os.path.exists(fashion_items_path):
            with open(fashion_items_path, 'rb') as f:
                fashion_items = pickle.load(f)
            print(f"  Loaded fashion items with {len(fashion_items)} entries (real names & descriptions)")
        else:
            fashion_items = {}
            print(f"  Warning: Fashion items not found at {fashion_items_path}")

        print(f"Polyvore-U generator loaded with {polyvore_u_generator.embeddings.shape[0]} items")
        return True

    except Exception as e:
        print(f"Warning: Could not load Polyvore-U generator: {e}")
        import traceback
        traceback.print_exc()
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown"""
    global generator

    # Startup
    print("Starting Fashion Feed API...")

    # Get project root (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check if embeddings exist - use Polyvore-U by default
    embeddings_path = os.environ.get(
        "EMBEDDINGS_PATH",
        os.path.join(project_root, "models/polyvore_u_embeddings.pkl")
    )
    faiss_path = os.environ.get(
        "FAISS_PATH",
        os.path.join(project_root, "models/polyvore_u_faiss.index")
    )
    metadata_path = os.environ.get(
        "METADATA_PATH",
        os.path.join(project_root, "data/polyvore_u/polyvore_u_item_metadata.json")
    )
    # Auto-detect BERT4Rec checkpoint if not specified
    recbole_checkpoint = os.environ.get("RECBOLE_CHECKPOINT", None)
    if recbole_checkpoint is None:
        # Find latest BERT4Rec checkpoint in models directory
        import glob
        bert4rec_checkpoints = glob.glob(os.path.join(project_root, "models/BERT4Rec*.pth"))
        if bert4rec_checkpoints:
            # Sort by modification time, get latest
            recbole_checkpoint = max(bert4rec_checkpoints, key=os.path.getmtime)
            print(f"Auto-detected BERT4Rec checkpoint: {recbole_checkpoint}")
    outfit_checkpoint = os.environ.get(
        "OUTFIT_TRANSFORMER_CHECKPOINT",
        os.path.join(project_root, "models/outfit_transformer_best.pth")
    )
    use_gpu = os.environ.get("USE_GPU", "true").lower() == "true"

    if os.path.exists(embeddings_path):
        from feed_generator import HybridFeedGenerator

        generator = HybridFeedGenerator(
            embeddings_path=embeddings_path,
            faiss_path=faiss_path if os.path.exists(faiss_path) else None,
            recbole_checkpoint=recbole_checkpoint,
            item_metadata_path=metadata_path if os.path.exists(metadata_path) else None,
            use_gpu=use_gpu
        )
        print(f"Feed generator loaded with {len(generator.item_ids)} items")
    else:
        print(f"Warning: Embeddings not found at {embeddings_path}")
        print("API will start but feed endpoints will not work")

    # Load OutfitTransformer for compatibility scoring (uses original Polyvore dataset)
    outfit_embeddings_path = os.path.join(project_root, "models/polyvore_embeddings.pkl")
    outfit_metadata_path = os.path.join(project_root, "data/polyvore/polyvore_item_metadata.json")
    if os.path.exists(outfit_checkpoint) and os.path.exists(outfit_embeddings_path) and os.path.exists(outfit_metadata_path):
        try:
            load_outfit_transformer(outfit_checkpoint, outfit_embeddings_path, outfit_metadata_path, use_gpu)
        except Exception as e:
            print(f"Warning: Could not load OutfitTransformer: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Warning: OutfitTransformer checkpoint not found at {outfit_checkpoint}")
        print("Compatibility endpoints will not work")

    # Load Polyvore-U generator for /polyvore-u/* endpoints
    print("\nLoading Polyvore-U generator...")
    if load_polyvore_u():
        print("Polyvore-U endpoints ready")
    else:
        print("Warning: Polyvore-U endpoints will not work")

    # Load Amazon Fashion feed for /amazon/* endpoints
    print("\nLoading Amazon Fashion feed...")
    if load_amazon_fashion():
        print("Amazon Fashion endpoints ready")
    else:
        print("Warning: Amazon Fashion endpoints will not work")

    # Load style quiz items
    global style_quiz_items
    style_quiz_path = os.path.join(project_root, "generated_outfits/style_quiz/style_quiz_candidates.pkl")
    curated_path = os.path.join(project_root, "data/polyvore_u/style_quiz_items.pkl")

    # Try curated items first, then fall back to candidates
    if os.path.exists(curated_path):
        with open(curated_path, 'rb') as f:
            raw_data = pickle.load(f)
            # Convert to int item_ids if needed
            for style, genders in raw_data.items():
                style_quiz_items[style] = {}
                for gender, items in genders.items():
                    style_quiz_items[style][gender] = [int(item_id) for item_id in items]
        print(f"Loaded curated style quiz items from {curated_path}")
    elif os.path.exists(style_quiz_path):
        with open(style_quiz_path, 'rb') as f:
            raw_data = pickle.load(f)
            # The candidates file has (item_id, score) tuples
            for style, genders in raw_data.items():
                style_quiz_items[style] = {}
                for gender, items in genders.items():
                    # Extract just the item_ids, taking top 20
                    style_quiz_items[style][gender] = [int(item_id) for item_id, _ in items[:20]]
        print(f"Loaded style quiz candidates from {style_quiz_path}")
    else:
        print("Warning: No style quiz items found. Style quiz endpoints will return empty results.")

    if style_quiz_items:
        total_styles = len(style_quiz_items)
        total_items = sum(
            len(items)
            for genders in style_quiz_items.values()
            for items in genders.values()
        )
        print(f"Style quiz ready: {total_styles} styles, {total_items} total items")

    yield

    # Shutdown
    print("Shutting down Fashion Feed API...")


# Create FastAPI app
app = FastAPI(
    title="Fashion Personalized Feed API",
    description="REST API for personalized fashion recommendations using FashionCLIP + RecBole",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for item images
# Get project root for image path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_images_dir = os.path.join(_project_root, "data/polyvore_u/291x291")
if os.path.exists(_images_dir):
    app.mount("/images", StaticFiles(directory=_images_dir), name="images")
    print(f"Mounted images directory: {_images_dir}")

# Mount original Polyvore images for /style-this-item endpoint
_polyvore_images_dir = os.path.join(_project_root, "data/polyvore/images/images")
if os.path.exists(_polyvore_images_dir):
    app.mount("/polyvore-images", StaticFiles(directory=_polyvore_images_dir), name="polyvore-images")
    print(f"Mounted Polyvore images directory: {_polyvore_images_dir}")

# Mount Amazon Fashion images
_amazon_images_dir = os.path.join(_project_root, "data/amazon_fashion/images")
if os.path.exists(_amazon_images_dir):
    app.mount("/amazon-images", StaticFiles(directory=_amazon_images_dir), name="amazon-images")
    print(f"Mounted Amazon Fashion images directory: {_amazon_images_dir}")


def get_style_quiz_image_url(item_id: int) -> str:
    """Get image URL for a style quiz item."""
    if polyvore_u_image_paths is not None and item_id < len(polyvore_u_image_paths):
        return f"/images/{polyvore_u_image_paths[item_id]}"
    return f"/images/empty_image.png"


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if generator is None:
        return HealthResponse(
            status="degraded",
            items_count=0,
            categories_count=0,
            recbole_loaded=False,
            outfit_transformer_loaded=outfit_transformer is not None
        )

    return HealthResponse(
        status="healthy",
        items_count=len(generator.item_ids),
        categories_count=len(generator.category_to_items),
        recbole_loaded=generator.bert4rec_model is not None,
        outfit_transformer_loaded=outfit_transformer is not None
    )


@app.post("/feed", response_model=FeedResponse)
async def get_feed(request: FeedRequest):
    """
    Get personalized feed for a user

    Uses user's interaction history to generate recommendations
    combining visual similarity and collaborative filtering.
    """
    if generator is None:
        raise HTTPException(status_code=503, detail="Feed generator not initialized")

    # Get user history
    history = user_profiles[request.user_id]['history']

    # Generate all candidate items (get more than we need for pagination)
    max_items = request.page * request.page_size + 100  # Buffer for pagination
    feed = generator.generate_feed(
        user_id=request.user_id,
        history=history,
        k=max_items,
        weights=request.weights
    )

    items = [
        FeedItem(
            item_id=f['item_id'],
            score=f['score'],
            sources=f.get('sources', [])
        )
        for f in feed
    ]

    # Apply pagination
    paginated_items, pagination_meta = paginate(items, request.page, request.page_size)

    return FeedResponse(
        user_id=request.user_id,
        items=paginated_items,
        pagination=pagination_meta
    )


@app.post("/similar", response_model=SimilarResponse)
async def get_similar(request: SimilarRequest):
    """
    Get visually similar items

    Uses FashionCLIP embeddings to find items that look similar.
    """
    if generator is None:
        raise HTTPException(status_code=503, detail="Feed generator not initialized")

    # Check if item exists
    if request.item_id not in generator.embeddings:
        raise HTTPException(status_code=404, detail=f"Item {request.item_id} not found")

    # Get more items than needed for pagination
    max_items = request.page * request.page_size + 100
    similar = generator.get_visual_similar(request.item_id, k=max_items)

    items = [
        FeedItem(item_id=s['item_id'], score=s['score'], sources=['visual'])
        for s in similar
    ]

    # Apply pagination
    paginated_items, pagination_meta = paginate(items, request.page, request.page_size)

    return SimilarResponse(
        item_id=request.item_id,
        similar=paginated_items,
        pagination=pagination_meta
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def record_feedback(request: FeedbackRequest):
    """
    Record user feedback/interaction

    Supported actions: view, like, dislike, purchase, save
    """
    if generator is not None:
        # Verify item exists
        if request.item_id not in generator.embeddings:
            raise HTTPException(status_code=404, detail=f"Item {request.item_id} not found")

    # Record feedback
    user_profiles[request.user_id]['feedback'].append({
        'item_id': request.item_id,
        'action': request.action
    })

    # Add to history for positive actions
    if request.action in ['view', 'like', 'purchase', 'save']:
        if request.item_id not in user_profiles[request.user_id]['history']:
            user_profiles[request.user_id]['history'].append(request.item_id)

    return FeedbackResponse(
        status="success",
        user_id=request.user_id,
        item_id=request.item_id
    )


@app.get("/item/{item_id}", response_model=ItemResponse)
async def get_item(item_id: str):
    """Get item details"""
    if generator is None:
        raise HTTPException(status_code=503, detail="Feed generator not initialized")

    details = generator.get_item_details(item_id)

    if details is None:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    return ItemResponse(
        item_id=item_id,
        category=details.get('semantic_category') or details.get('category_id'),
        name=details.get('name'),
        has_embedding=item_id in generator.embeddings,
        metadata=details
    )


@app.get("/user/{user_id}/history")
async def get_user_history(user_id: str):
    """Get user's interaction history"""
    return {
        "user_id": user_id,
        "history": user_profiles[user_id]['history'],
        "feedback_count": len(user_profiles[user_id]['feedback'])
    }


@app.delete("/user/{user_id}/history")
async def clear_user_history(user_id: str):
    """Clear user's interaction history"""
    user_profiles[user_id]['history'] = []
    user_profiles[user_id]['feedback'] = []

    return {"status": "success", "user_id": user_id}


@app.get("/categories")
async def list_categories():
    """List all available categories"""
    if generator is None:
        raise HTTPException(status_code=503, detail="Feed generator not initialized")

    return {
        "categories": [
            {"name": cat, "count": len(items)}
            for cat, items in generator.category_to_items.items()
        ]
    }


class CategoryItemsResponse(BaseModel):
    """Response model for category items"""
    category: str
    items: List[str]
    pagination: PaginationMeta


@app.get("/category/{category}/items", response_model=CategoryItemsResponse)
async def get_category_items(
    category: str,
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page")
):
    """Get items from a specific category with pagination"""
    if generator is None:
        raise HTTPException(status_code=503, detail="Feed generator not initialized")

    if category not in generator.category_to_items:
        raise HTTPException(status_code=404, detail=f"Category {category} not found")

    # Get all items in category
    all_items = generator.category_to_items[category]

    # Apply pagination
    paginated_items, pagination_meta = paginate(all_items, page, page_size)

    return CategoryItemsResponse(
        category=category,
        items=paginated_items,
        pagination=pagination_meta
    )


@app.get("/stats")
async def get_stats():
    """Get API statistics"""
    return {
        "total_users": len(user_profiles),
        "total_items": len(generator.item_ids) if generator else 0,
        "total_categories": len(generator.category_to_items) if generator else 0,
        "generator_ready": generator is not None,
        "outfit_transformer_ready": outfit_transformer is not None
    }


# === OutfitTransformer Compatibility Endpoints ===

def get_item_features(item_ids: List[str]):
    """Get visual features and category IDs for items"""
    visual_features = []
    category_ids = []
    valid_items = []

    for item_id in item_ids:
        if item_id not in outfit_embeddings:
            continue

        visual_features.append(outfit_embeddings[item_id])

        # Get category
        cat_name = outfit_metadata.get(item_id, {}).get('category_name', 'unknown')
        cat_id = outfit_category_to_idx.get(cat_name, outfit_category_to_idx.get('unknown', 0))
        category_ids.append(cat_id)
        valid_items.append(item_id)

    if not visual_features:
        return None, None, []

    visual_tensor = torch.tensor(np.stack(visual_features), dtype=torch.float32, device=outfit_device)
    category_tensor = torch.tensor(category_ids, dtype=torch.long, device=outfit_device)

    return visual_tensor, category_tensor, valid_items


def get_complementary_candidates(
    anchor_item_ids: List[str],
    target_categories: List[str],
    k_per_category: int = 100,
    exclude_ids: set = None,
    gender: str = None
) -> Dict[str, List[str]]:
    """
    Find complementary items from specified categories using hybrid approach:
    1. Get all items in target category using actual category mappings
    2. Compute visual similarity to anchor items
    3. Return top-k visually similar items PER target category

    This ensures outfit completion returns items from DIFFERENT categories
    that still have complementary style/colors.

    Gender filtering:
    - gender="women" -> returns women's items + unisex items
    - gender="men" -> returns men's items + unisex items
    - gender=None -> returns all items (no filter)
    """
    import random

    exclude_ids = exclude_ids or set()
    candidates_by_category = {}

    # Build gender-allowed item IDs set
    gender_allowed_ids = None
    if gender and gender_mapping:
        if gender.lower() == "women":
            gender_allowed_ids = gender_mapping.get('women_ids', set()) | gender_mapping.get('unisex_ids', set())
        elif gender.lower() == "men":
            gender_allowed_ids = gender_mapping.get('men_ids', set()) | gender_mapping.get('unisex_ids', set())

    # Get anchor embeddings for similarity computation
    anchor_embeddings = []
    for item_id in anchor_item_ids:
        if item_id in outfit_embeddings:
            anchor_embeddings.append(outfit_embeddings[item_id])

    if not anchor_embeddings:
        return {}

    # Compute mean anchor embedding (represents the style to match)
    anchor_mean = np.mean(anchor_embeddings, axis=0)
    anchor_norm = np.linalg.norm(anchor_mean)
    if anchor_norm == 0:
        return {}

    for target_cat in target_categories:
        # Get items in this broad category using actual category mappings
        category_items = get_polyvore_u_items_by_broad_category(target_cat)

        if not category_items:
            continue

        # Sample to avoid iterating over all items (performance)
        sample_size = min(5000, len(category_items))
        sampled_ids = random.sample(category_items, sample_size)

        # Compute similarity to anchor for each candidate
        similarities = []
        for item_int_id in sampled_ids:
            item_id = str(item_int_id)
            if item_id in exclude_ids:
                continue
            if item_id not in outfit_embeddings:
                continue

            # Filter out dresses from bottoms and socks from shoes
            if category_exclusions:
                if target_cat == "bottom" and item_int_id in category_exclusions.get('dress_ids', set()):
                    continue
                if target_cat == "shoe" and item_int_id in category_exclusions.get('sock_ids', set()):
                    continue

            # Gender filter: only include items that match the specified gender
            if gender_allowed_ids is not None and item_int_id not in gender_allowed_ids:
                continue

            emb = outfit_embeddings[item_id]
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                continue

            # Cosine similarity
            sim = np.dot(anchor_mean, emb) / (anchor_norm * emb_norm)
            similarities.append((item_id, float(sim)))

        # Sort by similarity and take top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        candidates_by_category[target_cat] = [item_id for item_id, _ in similarities[:k_per_category]]

    return candidates_by_category


@app.post("/compatibility", response_model=CompatibilityResponse)
async def score_compatibility(request: CompatibilityRequest):
    """
    Score outfit compatibility using OutfitTransformer

    Returns a compatibility score (0-1) for how well the items go together.
    Higher scores indicate better outfit coherence.
    """
    if outfit_transformer is None:
        raise HTTPException(status_code=503, detail="OutfitTransformer not initialized")

    visual_features, category_ids, valid_items = get_item_features(request.item_ids)

    if visual_features is None or len(valid_items) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 2 valid items. Found {len(valid_items)} items with embeddings."
        )

    # Add batch dimension
    visual_features = visual_features.unsqueeze(0)
    category_ids = category_ids.unsqueeze(0)

    with torch.no_grad():
        score = outfit_transformer.predict_compatibility(visual_features, category_ids)
        compatibility_score = float(score.cpu().item())

    return CompatibilityResponse(
        item_ids=valid_items,
        compatibility_score=compatibility_score,
        item_count=len(valid_items)
    )


@app.post("/fitb", response_model=FITBResponse)
async def fill_in_the_blank(request: FITBRequest):
    """
    Fill-In-The-Blank prediction

    Given context items (partial outfit) and candidate items,
    rank candidates by how well they complete the outfit.
    """
    if outfit_transformer is None:
        raise HTTPException(status_code=503, detail="OutfitTransformer not initialized")

    # Get context features
    context_visual, context_cats, valid_context = get_item_features(request.context_items)

    if context_visual is None or len(valid_context) < 1:
        raise HTTPException(
            status_code=400,
            detail="Need at least 1 valid context item with embeddings."
        )

    # Get candidate features
    candidate_visual, candidate_cats, valid_candidates = get_item_features(request.candidate_items)

    if candidate_visual is None or len(valid_candidates) < 1:
        raise HTTPException(
            status_code=400,
            detail="Need at least 1 valid candidate item with embeddings."
        )

    # Score each candidate
    scores = []
    with torch.no_grad():
        candidate_scores = outfit_transformer.score_fitb_answers(
            context_visual, context_cats,
            candidate_visual, candidate_cats
        )

        for i, item_id in enumerate(valid_candidates):
            scores.append({
                "item_id": item_id,
                "score": float(candidate_scores[i].cpu().item())
            })

    # Sort by score descending
    scores.sort(key=lambda x: x['score'], reverse=True)

    return FITBResponse(
        context_items=valid_context,
        ranked_candidates=scores,
        best_match=scores[0]['item_id'] if scores else ""
    )


@app.post("/outfit/complete", response_model=OutfitCompleteResponse)
async def complete_outfit(request: OutfitCompleteRequest):
    """
    Suggest items to complete an outfit with CATEGORY DIVERSITY.

    Given current outfit items, finds complementary items from DIFFERENT categories.
    For example: Input a pink top -> Get matching bottoms and shoes (not more tops).

    - If no target_categories specified, auto-completes to full outfit (top + bottom + shoe)
    - Uses hybrid approach: visual similarity within target categories + OutfitTransformer scoring
    """
    if outfit_transformer is None:
        raise HTTPException(status_code=503, detail="OutfitTransformer not initialized")

    # Get context features
    context_visual, context_cats, valid_context = get_item_features(request.item_ids)

    if context_visual is None or len(valid_context) < 1:
        raise HTTPException(
            status_code=400,
            detail="Need at least 1 valid item with embeddings."
        )

    # Determine current outfit BROAD categories (top, bottom, shoe, etc.)
    current_categories = set()
    for item_id in valid_context:
        try:
            cat = get_polyvore_u_broad_category(int(item_id))
            if cat:
                current_categories.add(cat)
        except (ValueError, TypeError):
            pass

    # Determine target categories
    if request.target_categories:
        # User specified which categories to add
        target_cats = [c for c in request.target_categories if c not in current_categories]
    elif request.category:
        # Single category filter (backwards compatible - DEPRECATED)
        target_cats = [request.category] if request.category not in current_categories else []
    else:
        # Default: complete to full outfit (top + bottom + shoe)
        all_cats = {"top", "bottom", "shoe"}
        target_cats = list(all_cats - current_categories)

    if not target_cats:
        return OutfitCompleteResponse(
            current_items=valid_context,
            current_categories=list(current_categories),
            target_categories=[],
            suggestions={}
        )

    # Get candidate items from target categories (exclude items already in outfit)
    exclude_set = set(request.item_ids)

    # HYBRID APPROACH: Get visually similar candidates from target categories
    candidates_by_category = get_complementary_candidates(
        anchor_item_ids=valid_context,
        target_categories=target_cats,
        k_per_category=100,
        exclude_ids=exclude_set,
        gender=request.gender
    )

    if not candidates_by_category:
        return OutfitCompleteResponse(
            current_items=valid_context,
            current_categories=list(current_categories),
            target_categories=target_cats,
            suggestions={}
        )

    # Score candidates with OutfitTransformer per category
    suggestions_by_category = {}

    for target_cat, candidate_ids in candidates_by_category.items():
        if not candidate_ids:
            suggestions_by_category[target_cat] = []
            continue

        # Get candidate features
        candidate_visual, candidate_cats, valid_candidates = get_item_features(candidate_ids)

        if candidate_visual is None or len(valid_candidates) == 0:
            suggestions_by_category[target_cat] = []
            continue

        # Score candidates
        scores = []
        with torch.no_grad():
            batch_size = min(len(valid_candidates), 10)
            for i in range(0, len(valid_candidates), batch_size):
                batch_visual = candidate_visual[i:i+batch_size]
                batch_cats = candidate_cats[i:i+batch_size]
                batch_items = valid_candidates[i:i+batch_size]

                batch_scores = outfit_transformer.score_fitb_answers(
                    context_visual, context_cats,
                    batch_visual, batch_cats
                )

                for j, item_id in enumerate(batch_items):
                    scores.append({
                        "item_id": item_id,
                        "score": float(batch_scores[j].cpu().item()),
                        "category": target_cat
                    })

        # Sort by score and take top page_size per category
        scores.sort(key=lambda x: x['score'], reverse=True)
        suggestions_by_category[target_cat] = scores[:request.page_size]

    return OutfitCompleteResponse(
        current_items=valid_context,
        current_categories=list(current_categories),
        target_categories=target_cats,
        suggestions=suggestions_by_category
    )


@app.post("/items/similar", response_model=SimilarItemsResponse)
async def get_similar_items(request: SimilarItemsRequest):
    """
    Find visually similar items (for "more like this" functionality).

    By default, returns items from the SAME category (e.g., pink top -> other tops).
    Set same_category=false to get similar items from ANY category.

    Use cases:
    - User sees a pink top, wants to see other similar tops -> same_category=true (default)
    - User wants visually similar items regardless of type -> same_category=false

    NOTE: This is DIFFERENT from /outfit/complete which returns COMPLEMENTARY items
    from DIFFERENT categories (top -> matching pants, shoes).
    """
    item_id = request.item_id

    # Validate item exists in embeddings
    if item_id not in outfit_embeddings:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found in embeddings")

    # Get item's category
    try:
        item_category = get_polyvore_u_category(int(item_id))
    except (ValueError, TypeError):
        item_category = None

    # Compute visual similarity using embeddings
    anchor_emb = outfit_embeddings[item_id]
    anchor_norm = np.linalg.norm(anchor_emb)

    if anchor_norm == 0:
        return SimilarItemsResponse(
            item_id=item_id,
            category=item_category,
            similar_items=[]
        )

    # Build gender-allowed item IDs set for filtering
    gender_allowed_ids = None
    if request.gender and gender_mapping:
        if request.gender.lower() == "women":
            gender_allowed_ids = gender_mapping.get('women_ids', set()) | gender_mapping.get('unisex_ids', set())
        elif request.gender.lower() == "men":
            gender_allowed_ids = gender_mapping.get('men_ids', set()) | gender_mapping.get('unisex_ids', set())

    # Compute similarity to all items
    similarities = []
    for other_id, other_emb in outfit_embeddings.items():
        if other_id == item_id:
            continue

        # Gender filter: only include items that match the specified gender
        if gender_allowed_ids is not None:
            try:
                other_int_id = int(other_id)
                if other_int_id not in gender_allowed_ids:
                    continue
            except (ValueError, TypeError):
                continue

        # Filter by category if requested
        if request.same_category and item_category:
            try:
                other_category = get_polyvore_u_category(int(other_id))
                if other_category != item_category:
                    continue
            except (ValueError, TypeError):
                continue

        other_norm = np.linalg.norm(other_emb)
        if other_norm == 0:
            continue

        sim = np.dot(anchor_emb, other_emb) / (anchor_norm * other_norm)
        similarities.append((other_id, float(sim)))

    # Sort by similarity and take top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar = similarities[:request.k]

    # Format response
    similar_items = []
    for sim_id, sim_score in top_similar:
        try:
            sim_category = get_polyvore_u_category(int(sim_id))
        except (ValueError, TypeError):
            sim_category = None

        similar_items.append({
            "item_id": sim_id,
            "score": round(sim_score, 4),
            "category": sim_category
        })

    return SimilarItemsResponse(
        item_id=item_id,
        category=item_category,
        similar_items=similar_items
    )


@app.post("/outfit/validate")
async def validate_outfit(request: CompatibilityRequest):
    """
    Validate an outfit and get detailed feedback

    Returns compatibility score with interpretation
    and identifies which items may not fit well.
    """
    if outfit_transformer is None:
        raise HTTPException(status_code=503, detail="OutfitTransformer not initialized")

    visual_features, category_ids, valid_items = get_item_features(request.item_ids)

    if visual_features is None or len(valid_items) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 2 valid items. Found {len(valid_items)} items with embeddings."
        )

    # Get overall compatibility
    visual_features_batch = visual_features.unsqueeze(0)
    category_ids_batch = category_ids.unsqueeze(0)

    with torch.no_grad():
        overall_score = float(outfit_transformer.predict_compatibility(
            visual_features_batch, category_ids_batch
        ).cpu().item())

    # Check each item's contribution by computing leave-one-out scores
    item_contributions = []
    with torch.no_grad():
        for i in range(len(valid_items)):
            # Create outfit without this item
            mask = torch.ones(len(valid_items), dtype=torch.bool)
            mask[i] = False

            if mask.sum() < 2:
                item_contributions.append({
                    "item_id": valid_items[i],
                    "category": outfit_metadata.get(valid_items[i], {}).get('category_name', 'unknown'),
                    "contribution": 0.0,
                    "fits_well": True
                })
                continue

            subset_visual = visual_features[mask].unsqueeze(0)
            subset_cats = category_ids[mask].unsqueeze(0)

            subset_score = float(outfit_transformer.predict_compatibility(
                subset_visual, subset_cats
            ).cpu().item())

            # If removing this item increases score, it doesn't fit well
            contribution = overall_score - subset_score
            item_contributions.append({
                "item_id": valid_items[i],
                "category": outfit_metadata.get(valid_items[i], {}).get('category_name', 'unknown'),
                "contribution": contribution,
                "fits_well": contribution >= -0.05  # Threshold for "fits well"
            })

    # Interpret score
    if overall_score >= 0.7:
        interpretation = "Excellent outfit! Items complement each other very well."
    elif overall_score >= 0.5:
        interpretation = "Good outfit. Consider swapping items with negative contribution."
    elif overall_score >= 0.3:
        interpretation = "Average outfit. Some items may not match well."
    else:
        interpretation = "Items don't seem to go well together. Consider different combinations."

    return {
        "item_ids": valid_items,
        "overall_score": overall_score,
        "interpretation": interpretation,
        "item_analysis": item_contributions,
        "items_that_dont_fit": [
            item["item_id"] for item in item_contributions if not item["fits_well"]
        ]
    }


@app.post("/style-this-item", response_model=StyleThisItemResponse)
async def style_this_item(request: StyleThisItemRequest):
    """
    Generate multiple complete outfit sets featuring a single item.

    Use case: User views a t-shirt → show them 3-5 complete outfit ideas
    that include that t-shirt with different pants, shoes, accessories.

    The algorithm:
    1. Get the anchor item's category and visual features
    2. Determine complementary categories needed for a complete outfit
    3. Find diverse candidate items from each category
    4. Build and score multiple outfit combinations
    5. Return top-scoring diverse outfits
    """
    import random

    if outfit_transformer is None:
        raise HTTPException(status_code=503, detail="OutfitTransformer not initialized")

    if generator is None:
        raise HTTPException(status_code=503, detail="Feed generator not initialized")

    anchor_id = request.item_id

    # Verify anchor item exists
    if anchor_id not in outfit_embeddings:
        raise HTTPException(status_code=404, detail=f"Item {anchor_id} not found in embeddings")

    # Get anchor item info
    anchor_meta = outfit_metadata.get(anchor_id, {})
    anchor_category = anchor_meta.get('category_name', 'unknown')
    anchor_local_img = get_polyvore_local_image_url(anchor_id)
    anchor_image_url = anchor_local_img if anchor_local_img else anchor_meta.get('image_url', '')
    anchor_reviews = get_mock_reviews(anchor_id)

    anchor_info = {
        "item_id": anchor_id,
        "category": anchor_category,
        "name": anchor_meta.get('name', ''),
        "price": anchor_meta.get('price', 0.0),
        "image_url": anchor_image_url,
        "rating": anchor_reviews["rating"],
        "review_count": anchor_reviews["review_count"]
    }

    # Define outfit category templates based on anchor category
    # These define what categories make a complete outfit
    outfit_templates = {
        # Tops
        'Blouses': ['Jeans', 'Pants', 'Skirts', 'Shorts', 'Flats', 'Heels', 'Sneakers', 'Bags'],
        'Tees': ['Jeans', 'Shorts', 'Skirts', 'Sneakers', 'Sandals', 'Bags', 'Jackets'],
        'Sweaters': ['Jeans', 'Pants', 'Skirts', 'Boots', 'Sneakers', 'Bags', 'Coats'],
        'Sweatshirts': ['Jeans', 'Joggers', 'Sneakers', 'Boots', 'Bags'],
        'Tanks': ['Shorts', 'Skirts', 'Jeans', 'Sandals', 'Sneakers'],

        # Bottoms
        'Jeans': ['Tees', 'Blouses', 'Sweaters', 'Sneakers', 'Boots', 'Heels', 'Bags'],
        'Pants': ['Blouses', 'Sweaters', 'Tees', 'Heels', 'Flats', 'Loafers', 'Bags'],
        'Skirts': ['Blouses', 'Tees', 'Sweaters', 'Heels', 'Flats', 'Sandals', 'Bags'],
        'Shorts': ['Tees', 'Tanks', 'Blouses', 'Sandals', 'Sneakers', 'Bags'],

        # Outerwear
        'Jackets': ['Tees', 'Jeans', 'Pants', 'Sneakers', 'Boots', 'Bags'],
        'Coats': ['Sweaters', 'Jeans', 'Pants', 'Boots', 'Bags', 'Scarves'],
        'Blazers': ['Blouses', 'Pants', 'Jeans', 'Heels', 'Loafers', 'Bags'],

        # Shoes
        'Sneakers': ['Jeans', 'Shorts', 'Tees', 'Sweatshirts', 'Jackets'],
        'Boots': ['Jeans', 'Skirts', 'Sweaters', 'Coats', 'Jackets'],
        'Heels': ['Dresses', 'Skirts', 'Pants', 'Blouses', 'Blazers'],
        'Sandals': ['Shorts', 'Skirts', 'Dresses', 'Tees', 'Tanks'],
        'Flats': ['Jeans', 'Pants', 'Skirts', 'Blouses', 'Dresses'],

        # Dresses
        'Dresses': ['Heels', 'Sandals', 'Flats', 'Bags', 'Jackets', 'Cardigans'],

        # Default template
        'default': ['Tops', 'Bottoms', 'Shoes', 'Bags', 'Outerwear']
    }

    # Get complementary categories for this item
    complementary_cats = outfit_templates.get(anchor_category, outfit_templates['default'])

    # Find candidate items from visually similar items (diverse pool)
    similar_items = generator.get_visual_similar(anchor_id, k=200, exclude_ids={anchor_id})

    # Group candidates by category
    candidates_by_category = {}
    for item in similar_items:
        item_id = item['item_id']
        if item_id in outfit_metadata:
            cat = outfit_metadata[item_id].get('category_name', 'unknown')
            if cat not in candidates_by_category:
                candidates_by_category[cat] = []
            candidates_by_category[cat].append(item_id)

    # Also add random items from complementary categories for diversity
    all_item_ids = list(outfit_embeddings.keys())
    for cat in complementary_cats:
        if cat not in candidates_by_category:
            candidates_by_category[cat] = []
        # Find items of this category
        for item_id in random.sample(all_item_ids, min(500, len(all_item_ids))):
            if item_id in outfit_metadata:
                item_cat = outfit_metadata[item_id].get('category_name', '')
                if cat.lower() in item_cat.lower() or item_cat.lower() in cat.lower():
                    if item_id not in candidates_by_category[cat]:
                        candidates_by_category[cat].append(item_id)
                        if len(candidates_by_category[cat]) >= 30:
                            break

    # Build outfit combinations
    generated_outfits = []
    items_per_outfit = request.items_per_outfit - 1  # Minus anchor item

    # Try to generate diverse outfits
    # Generate more than needed to support pagination (up to page * page_size + buffer)
    target_outfits = request.page * request.page_size + request.page_size * 2  # Buffer for extra pages
    used_items = {anchor_id}
    attempts = 0
    max_attempts = target_outfits * 20

    while len(generated_outfits) < target_outfits and attempts < max_attempts:
        attempts += 1

        # Select categories for this outfit
        available_cats = [c for c in candidates_by_category.keys()
                         if c != anchor_category and len(candidates_by_category[c]) > 0]

        if len(available_cats) < items_per_outfit:
            # Not enough categories, use what we have
            selected_cats = available_cats[:items_per_outfit]
        else:
            selected_cats = random.sample(available_cats, items_per_outfit)

        if not selected_cats:
            continue

        # Pick one item from each category
        outfit_items = [anchor_id]
        outfit_valid = True

        for cat in selected_cats:
            # Get candidates not yet used
            available = [i for i in candidates_by_category[cat] if i not in used_items]
            if not available:
                available = candidates_by_category[cat]  # Reuse if needed

            if available:
                selected = random.choice(available)
                outfit_items.append(selected)
            else:
                outfit_valid = False
                break

        if not outfit_valid or len(outfit_items) < 2:
            continue

        # Score this outfit
        visual_features, category_ids, valid_items = get_item_features(outfit_items)

        if visual_features is None or len(valid_items) < 2:
            continue

        visual_batch = visual_features.unsqueeze(0)
        category_batch = category_ids.unsqueeze(0)

        with torch.no_grad():
            score = float(outfit_transformer.predict_compatibility(
                visual_batch, category_batch
            ).cpu().item())

        # Only keep outfits with decent compatibility
        if score < 0.3:
            continue

        # Build outfit info
        outfit_info = {
            "items": [],
            "compatibility_score": score
        }

        for item_id in valid_items:
            meta = outfit_metadata.get(item_id, {})
            # Prefer local image URL, fall back to metadata URL
            local_img = get_polyvore_local_image_url(item_id)
            image_url = local_img if local_img else meta.get('image_url', '')
            reviews = get_mock_reviews(item_id)
            outfit_info["items"].append({
                "item_id": item_id,
                "name": meta.get('name', ''),
                "price": meta.get('price', 0.0),
                "image_url": image_url,
                "category": meta.get('category_name', 'unknown'),
                "is_anchor": item_id == anchor_id,
                "rating": reviews["rating"],
                "review_count": reviews["review_count"]
            })

        # Add style description based on categories
        categories_in_outfit = [outfit_metadata.get(i, {}).get('category_name', '') for i in valid_items]
        if any('casual' in c.lower() or 'tee' in c.lower() or 'jeans' in c.lower() or 'sneaker' in c.lower() for c in categories_in_outfit):
            outfit_info["style_description"] = "Casual Look"
        elif any('blazer' in c.lower() or 'heel' in c.lower() or 'dress' in c.lower() for c in categories_in_outfit):
            outfit_info["style_description"] = "Dressy Look"
        elif any('boot' in c.lower() or 'jacket' in c.lower() or 'coat' in c.lower() for c in categories_in_outfit):
            outfit_info["style_description"] = "Street Style"
        else:
            outfit_info["style_description"] = "Everyday Look"

        generated_outfits.append(outfit_info)

        # Mark items as used for diversity
        for item_id in valid_items:
            if item_id != anchor_id:
                used_items.add(item_id)

    # Sort by compatibility score
    generated_outfits.sort(key=lambda x: x['compatibility_score'], reverse=True)

    # Apply pagination
    paginated_outfits, pagination_meta = paginate(generated_outfits, request.page, request.page_size)

    return StyleThisItemResponse(
        anchor_item=anchor_info,
        outfits=[
            OutfitSet(
                items=o["items"],
                compatibility_score=o["compatibility_score"],
                style_description=o.get("style_description")
            )
            for o in paginated_outfits
        ],
        pagination=pagination_meta
    )


# === Polyvore-U Feed Endpoints ===

def get_polyvore_u_image_path(item_id: int) -> Optional[str]:
    """Get full image path for a Polyvore-U item."""
    if polyvore_u_image_paths is None or item_id < 0 or item_id >= len(polyvore_u_image_paths):
        return None
    filename = polyvore_u_image_paths[item_id]
    if filename == 'empty_image.png':
        return None
    if polyvore_u_image_dir is None:
        return None
    full_path = polyvore_u_image_dir / filename
    if full_path.exists():
        return str(full_path)
    return None


def get_polyvore_u_category(item_id: int) -> Optional[str]:
    """
    Get the fine-grained category name for a Polyvore-U item.

    Returns category names like "women's shoe", "women's shirt", "skirt", etc.
    """
    if polyvore_u_iid_cate_map is None or polyvore_u_category_map is None:
        return None

    # Get category ID for this item
    cat_id = polyvore_u_iid_cate_map.get(item_id)
    if cat_id is None:
        return None

    # Get category name
    return polyvore_u_category_map.get(cat_id, "unknown")


def clean_category_name(category: Optional[str]) -> Optional[str]:
    """
    Remove gender prefix from category names for display.

    Converts "women's t-shirt" -> "t-shirt", "men's jeans" -> "jeans"
    Keeps non-gendered categories like "skirt", "dress" unchanged.
    """
    if category is None:
        return None

    # Remove gender prefixes
    if category.startswith("women's "):
        return category[8:]  # len("women's ") = 8
    if category.startswith("men's "):
        return category[6:]  # len("men's ") = 6

    return category


def get_mock_price(item_id: int) -> float:
    """Deterministic mock price based on item_id (same ID = same price)"""
    rng = random.Random(item_id)  # Don't affect global state
    return round(rng.uniform(15, 200), 2)


# Pool of generic fashion review texts
MOCK_REVIEW_POOL = [
    # 5-star reviews
    {"rating": 5, "title": "Absolutely love it!", "text": "Perfect fit and amazing quality. Exactly as pictured. Will definitely buy again!", "reviewer": "FashionLover23"},
    {"rating": 5, "title": "Best purchase ever", "text": "This exceeded my expectations! The material is so soft and comfortable. Highly recommend.", "reviewer": "StyleQueen"},
    {"rating": 5, "title": "Stunning!", "text": "Got so many compliments wearing this. The color is beautiful and true to the photos.", "reviewer": "ChicShopper"},
    {"rating": 5, "title": "Worth every penny", "text": "High quality material that looks expensive. Fast shipping too!", "reviewer": "TrendyTina"},
    {"rating": 5, "title": "My new favorite", "text": "I've already worn this multiple times. It goes with everything in my closet!", "reviewer": "ClassicStyle"},
    {"rating": 5, "title": "Perfect!", "text": "Fits like a glove. The stitching is impeccable and the fabric is premium quality.", "reviewer": "QualityFirst"},
    {"rating": 5, "title": "Obsessed!", "text": "This is exactly what I was looking for. The design is so elegant and versatile.", "reviewer": "ModernChic"},
    {"rating": 5, "title": "Amazing find", "text": "Can't believe the quality at this price point. Looks way more expensive than it is.", "reviewer": "BargainHunter"},

    # 4-star reviews
    {"rating": 4, "title": "Great quality", "text": "Really nice piece. Fits well and looks great. Minor thread issue but overall happy.", "reviewer": "HappyCustomer"},
    {"rating": 4, "title": "Very pleased", "text": "Good purchase. Material is nice and the color is accurate. Runs slightly small.", "reviewer": "SizeSmart"},
    {"rating": 4, "title": "Love the style", "text": "Beautiful design and good quality. Would have given 5 stars but shipping took a while.", "reviewer": "PatientShopper"},
    {"rating": 4, "title": "Nice addition to my wardrobe", "text": "Solid purchase. Comfortable and stylish. Washes well too!", "reviewer": "PracticalPete"},
    {"rating": 4, "title": "Good value", "text": "Happy with this purchase. The fit is good and it's versatile for many occasions.", "reviewer": "VersatileVal"},
    {"rating": 4, "title": "Impressed", "text": "Better than expected for the price. The details are nice and well-made.", "reviewer": "DetailDiva"},
    {"rating": 4, "title": "Would recommend", "text": "Great everyday piece. Comfortable and looks put-together. A staple item.", "reviewer": "EverydayElla"},
    {"rating": 4, "title": "Solid choice", "text": "Good quality and true to size. The material is breathable and comfortable.", "reviewer": "ComfortKing"},

    # 3-star reviews (occasional)
    {"rating": 3, "title": "Decent", "text": "It's okay. Not bad but not amazing either. Material is thinner than expected.", "reviewer": "HonestReview"},
    {"rating": 3, "title": "Average", "text": "Looks nice but runs a bit large. Had to size down. Color was slightly different.", "reviewer": "TruthfulTom"},
]

# Mock reviewer names for generating additional reviews
MOCK_REVIEWER_NAMES = [
    "StyleSeeker", "FashionForward", "TrendyTraveler", "ChicChaser", "ModeMaven",
    "WardrobeWizard", "LookLover", "OutfitObsessed", "GlamGuru", "ClosetCurator",
    "DressedUp", "SartorialSam", "PolishedPenny", "CasualCathy", "ElegantEve",
    "BoldBella", "ClassyCarl", "VintageVibes", "MinimalistMia", "MaximalistMax",
    "ColorCrazy", "NeutralNancy", "PrintPrincess", "SolidSophie", "TextureTed",
    "LayerLover", "AccessoryAddict", "ShoeSnob", "BagBoss", "JewelJunkie"
]

def get_mock_reviews(item_id, num_reviews: int = 3) -> dict:
    """
    Generate deterministic mock reviews based on item_id.

    Returns rating, review_count, and sample review texts.
    Same item_id always returns the same reviews for consistency.
    """
    # Handle both int and string item_ids
    if isinstance(item_id, str):
        seed = hash(item_id) % (2**31)
    else:
        seed = item_id
    rng = random.Random(seed)

    # Generate overall rating (weighted towards positive)
    rating = round(rng.uniform(3.5, 5.0), 1)
    review_count = rng.randint(5, 500)

    # Select sample reviews deterministically
    reviews = []
    selected_indices = rng.sample(range(len(MOCK_REVIEW_POOL)), min(num_reviews, len(MOCK_REVIEW_POOL)))

    for i, idx in enumerate(selected_indices):
        base_review = MOCK_REVIEW_POOL[idx].copy()
        # Vary the reviewer name deterministically
        reviewer_idx = (seed + i) % len(MOCK_REVIEWER_NAMES)
        base_review["reviewer"] = MOCK_REVIEWER_NAMES[reviewer_idx]
        # Add a mock date (deterministic based on item + review index)
        days_ago = rng.randint(1, 180)
        base_review["days_ago"] = days_ago
        reviews.append(base_review)

    return {
        "rating": rating,
        "review_count": review_count,
        "reviews": reviews
    }


def get_polyvore_local_image_url(item_id: str) -> str:
    """Get local image URL for original Polyvore item (string ID like '194508109')"""
    if polyvore_item_to_image is None:
        return None

    img_key = polyvore_item_to_image.get(item_id)
    if img_key:
        # img_key is like "214181831_1" -> folder/file = "214181831/1.jpg"
        parts = img_key.split('_')
        if len(parts) == 2:
            outfit_id, position = parts
            return f"/polyvore-images/{outfit_id}/{position}.jpg"
    return None


def get_polyvore_u_image_url(item_id: int) -> str:
    """Get image URL from item_id"""
    if polyvore_u_image_paths is not None and 0 <= item_id < len(polyvore_u_image_paths):
        filename = polyvore_u_image_paths[item_id]
        if filename != 'empty_image.png':
            return f"/images/{filename}"
    return "/images/empty_image.png"


def get_polyvore_u_gender(item_id: int) -> str:
    """Get gender for item"""
    if gender_mapping:
        return gender_mapping.get('item_to_gender', {}).get(item_id, "unisex")
    return "unisex"


def get_fashion_item_data(item_id: int) -> dict:
    """Get real name and description from fashion_items via image filename"""
    if fashion_items is None or polyvore_u_image_paths is None:
        return None
    if item_id < 0 or item_id >= len(polyvore_u_image_paths):
        return None

    img_filename = polyvore_u_image_paths[item_id]
    if img_filename == 'empty_image.png':
        return None

    return fashion_items.get(img_filename)


def get_polyvore_u_real_name(item_id: int) -> str:
    """Get real product name from fashion_items, fallback to category"""
    item_data = get_fashion_item_data(item_id)
    if item_data and 'name' in item_data:
        return item_data['name']
    # Fallback to category name
    category = get_polyvore_u_category(item_id)
    return category.title() if category else "Unknown"


def get_polyvore_u_description(item_id: int) -> str:
    """Get real product description from fashion_items"""
    item_data = get_fashion_item_data(item_id)
    if item_data and 'text' in item_data:
        return item_data['text']
    return None


def enrich_polyvore_u_item(item_id: int, score: float = None, include_reviews: bool = False) -> dict:
    """Enrich item_id with full metadata including real name and description"""
    category = get_polyvore_u_category(item_id) or "unknown"
    broad_cat = get_polyvore_u_broad_category(item_id)
    item_gender = get_polyvore_u_gender(item_id)
    real_name = get_polyvore_u_real_name(item_id)
    description = get_polyvore_u_description(item_id)
    reviews_data = get_mock_reviews(item_id)

    result = {
        "item_id": item_id,
        "name": real_name,
        "price": get_mock_price(item_id),
        "image_url": get_polyvore_u_image_url(item_id),
        "category": category,
        "broad_category": broad_cat,
        "gender": item_gender,
        "score": score,
        "rating": reviews_data["rating"],
        "review_count": reviews_data["review_count"]
    }
    if description:
        result["description"] = description
    if include_reviews:
        result["reviews"] = reviews_data["reviews"]
    return result


def apply_polyvore_u_filters(items: List[dict], filters: ProductFilter, use_or_logic: bool = True) -> List[dict]:
    """
    Apply filters to list of enriched items.

    Args:
        items: List of enriched item dicts
        filters: ProductFilter with filter criteria
        use_or_logic: If True, use OR logic for positive filters (categories, broad_categories, genders)
                      and rank by match count. If False, use AND logic (all must match).

    OR Logic (default):
        - Positive filters (categories, broad_categories, genders): match ANY, boost score by match count
        - Hard filters (min_price, max_price, search): still AND logic (must pass)

    AND Logic:
        - All filters must match exactly (original behavior)
    """
    if not filters:
        return items

    # Hard filters always use AND logic
    result = items
    if filters.min_price is not None:
        result = [i for i in result if i['price'] >= filters.min_price]
    if filters.max_price is not None:
        result = [i for i in result if i['price'] <= filters.max_price]
    if filters.search:
        search_lower = filters.search.lower()
        result = [i for i in result if search_lower in i['name'].lower()]

    # Check if any positive filters are specified
    has_positive_filters = bool(filters.categories or filters.broad_categories or filters.genders)

    if not has_positive_filters:
        return result

    if use_or_logic:
        # OR logic: match ANY filter, boost by match count
        scored_items = []
        for item in result:
            match_count = 0

            # Check categories (OR - match any)
            if filters.categories:
                if item.get('category') in filters.categories:
                    match_count += 1

            # Check broad_categories (OR - match any)
            if filters.broad_categories:
                if item.get('broad_category') in filters.broad_categories:
                    match_count += 1

            # Check genders (OR - match any)
            if filters.genders:
                if item.get('gender') in filters.genders:
                    match_count += 1

            # Only include items that match at least one filter
            if match_count > 0:
                # Add match_count to existing score for ranking
                item_with_boost = item.copy()
                original_score = item.get('score', 1.0)
                # Boost score by match count (each match adds 0.1 to score)
                item_with_boost['score'] = original_score + (match_count * 0.1)
                item_with_boost['filter_matches'] = match_count
                scored_items.append(item_with_boost)

        # Sort by boosted score (higher is better)
        scored_items.sort(key=lambda x: x.get('score', 0), reverse=True)
        return scored_items
    else:
        # AND logic: all specified filters must match (original behavior)
        if filters.categories:
            result = [i for i in result if i['category'] in filters.categories]
        if filters.broad_categories:
            result = [i for i in result if i.get('broad_category') in filters.broad_categories]
        if filters.genders:
            result = [i for i in result if i.get('gender') in filters.genders]
        return result


def get_polyvore_u_broad_category(item_id: int) -> Optional[str]:
    """
    Get the broad category for a Polyvore-U item based on ID ranges.

    The embedding system uses these ranges (from category_offsets.pkl):
    - Top:    IDs 1 to 85,389
    - Bottom: IDs 85,390 to 145,591
    - Shoe:   IDs 145,592 to 212,085

    Returns: "top", "bottom", or "shoe"
    """
    # Use simple ID-based ranges from embedding generation
    # These ranges are fixed based on how embeddings were generated
    if 1 <= item_id <= 85389:
        return "top"
    elif 85390 <= item_id <= 145591:
        return "bottom"
    elif 145592 <= item_id <= 212085:
        return "shoe"
    else:
        return None  # Invalid item ID


def get_polyvore_u_items_by_broad_category(broad_category: str) -> List[int]:
    """
    Get all item IDs that belong to a broad category based on ID ranges.

    The embedding system uses these ranges (from category_offsets.pkl):
    - Top:    IDs 1 to 85,389
    - Bottom: IDs 85,390 to 145,591
    - Shoe:   IDs 145,592 to 212,085

    Args:
        broad_category: One of "top", "bottom", "shoe"

    Returns:
        List of item IDs in that category
    """
    # Use simple ID-based ranges from embedding generation
    if broad_category == "top":
        return list(range(1, 85390))
    elif broad_category == "bottom":
        return list(range(85390, 145592))
    elif broad_category == "shoe":
        return list(range(145592, 212086))
    else:
        return []


def generate_polyvore_u_feed(
    user_id: str,
    k: int = 20,
    shuffle: bool = True
) -> List[tuple]:
    """
    Generate personalized feed for Polyvore-U user.

    Returns list of (item_id, image_path, score, category) tuples.
    """
    import random

    history = polyvore_u_user_history.get(user_id, [])
    dislikes = polyvore_u_user_dislikes.get(user_id, set())

    # Get candidates with scores
    candidates = polyvore_u_generator.generate_candidates_with_scores(
        user_history=history,
        k=k * 3 + len(dislikes) + 50,
        exclude_seen=True
    )

    # Filter dislikes and get images
    valid_items = []
    for item_id, score in candidates:
        if item_id in dislikes:
            continue
        if item_id in history:
            continue
        img_path = get_polyvore_u_image_path(item_id)
        category = get_polyvore_u_category(item_id)
        if img_path:
            valid_items.append((item_id, img_path, score, category))
        if len(valid_items) >= k * 2:
            break

    # Shuffle with score-weighted sampling for variety
    if shuffle and len(valid_items) > k:
        scores = np.array([s for _, _, s, _ in valid_items])
        scores_shifted = scores - scores.min() + 0.1
        probs = scores_shifted / scores_shifted.sum()

        indices = np.random.choice(
            len(valid_items),
            size=min(k, len(valid_items)),
            replace=False,
            p=probs
        )
        feed = [valid_items[i] for i in indices]
        feed.sort(key=lambda x: x[2], reverse=True)
    else:
        feed = valid_items[:k]

    return feed


@app.get("/polyvore-u/health", response_model=PolyvoreUHealthResponse)
async def polyvore_u_health():
    """Health check for Polyvore-U endpoints."""
    if polyvore_u_generator is None:
        return PolyvoreUHealthResponse(
            status="not_initialized",
            items_count=0,
            categories={"top": 0, "bottom": 0, "shoe": 0},
            faiss_index_loaded=False
        )

    return PolyvoreUHealthResponse(
        status="healthy",
        items_count=polyvore_u_generator.embeddings.shape[0],
        categories={
            "top": 85389,
            "bottom": 60202,
            "shoe": 66494
        },
        faiss_index_loaded=polyvore_u_generator.index is not None
    )


@app.post("/polyvore-u/feed", response_model=PolyvoreUFeedResponse)
async def get_polyvore_u_feed(request: PolyvoreUFeedRequest):
    """
    Get personalized feed for a Polyvore-U user.

    Uses FashionCLIP + Faiss for visual similarity candidate generation.
    Feed adapts based on user's liked items.

    Optional filters:
    - categories: List of fine-grained categories (e.g., "women's t-shirt", "skirt")
    - broad_categories: List of "top", "bottom", "shoe"
    - genders: List of "women", "men", "unisex"
    - min_price, max_price: Price range filter
    - search: Search in product name
    """
    import random as rnd

    if polyvore_u_generator is None:
        raise HTTPException(status_code=503, detail="Polyvore-U generator not initialized")

    history = polyvore_u_user_history.get(request.user_id, [])
    dislikes = polyvore_u_user_dislikes.get(request.user_id, set())

    # Check if broad_categories filter is specified - handle it specially
    if request.filters and request.filters.broad_categories:
        # Generate items directly from the specified broad category ID ranges
        # This ensures we get items from the requested categories
        candidate_ids = []
        for broad_cat in request.filters.broad_categories:
            if broad_cat == "top":
                candidate_ids.extend(range(1, 85390))
            elif broad_cat == "bottom":
                candidate_ids.extend(range(85390, 145592))
            elif broad_cat == "shoe":
                candidate_ids.extend(range(145592, 212086))

        # Filter out history and dislikes
        candidate_ids = [i for i in candidate_ids if i not in dislikes and i not in history]

        # Sample a subset for efficiency (we need more than page_size for other filters)
        max_sample = request.page * request.page_size + 1000
        if len(candidate_ids) > max_sample:
            # If user has history, use similarity scoring to rank
            if history and polyvore_u_generator is not None:
                # Get embeddings for history items
                history_embeds = polyvore_u_generator.embeddings[history]
                avg_embed = history_embeds.mean(axis=0)

                # Score candidate items by similarity to user's taste
                candidate_embeds = polyvore_u_generator.embeddings[candidate_ids]
                similarities = np.dot(candidate_embeds, avg_embed) / (
                    np.linalg.norm(candidate_embeds, axis=1) * np.linalg.norm(avg_embed) + 1e-8
                )

                # Get top items by similarity
                top_indices = np.argsort(similarities)[::-1][:max_sample]
                candidate_ids = [candidate_ids[i] for i in top_indices]
                scores = [float(similarities[i]) for i in top_indices]
            else:
                # Random sampling for cold start users
                rnd.shuffle(candidate_ids)
                candidate_ids = candidate_ids[:max_sample]
                scores = [1.0] * len(candidate_ids)
        else:
            scores = [1.0] * len(candidate_ids)

        # Build feed items
        enriched_items = []
        for i, item_id in enumerate(candidate_ids):
            score = scores[i] if i < len(scores) else 1.0
            img_path = get_polyvore_u_image_path(item_id)
            if img_path and img_path != 'empty_image.png':
                enriched_items.append(enrich_polyvore_u_item(item_id, score))

        # Apply remaining filters (categories, genders, price, search) - but not broad_categories again
        if request.filters:
            filters_without_broad = ProductFilter(
                categories=request.filters.categories,
                genders=request.filters.genders,
                min_price=request.filters.min_price,
                max_price=request.filters.max_price,
                search=request.filters.search
            )
            enriched_items = apply_polyvore_u_filters(enriched_items, filters_without_broad)
    else:
        # Standard flow: use the generator
        extra_for_filters = 500 if request.filters else 100
        max_items = request.page * request.page_size + extra_for_filters
        feed = generate_polyvore_u_feed(
            user_id=request.user_id,
            k=max_items,
            shuffle=request.shuffle
        )

        # Enrich items with full metadata
        enriched_items = [
            enrich_polyvore_u_item(item_id, score)
            for item_id, img_path, score, category in feed
        ]

        # Apply filters if provided
        if request.filters:
            enriched_items = apply_polyvore_u_filters(enriched_items, request.filters)

    # Convert to Pydantic models
    items = [PolyvoreUFeedItem(**item) for item in enriched_items]

    # Apply pagination
    paginated_items, pagination_meta = paginate(items, request.page, request.page_size)

    # Get liked count
    liked_count = len(polyvore_u_user_history.get(request.user_id, []))

    return PolyvoreUFeedResponse(
        user_id=request.user_id,
        items=paginated_items,
        pagination=pagination_meta,
        liked_count=liked_count
    )


@app.post("/polyvore-u/like", response_model=PolyvoreULikeResponse)
async def polyvore_u_like(request: PolyvoreULikeRequest):
    """
    Like an item in Polyvore-U.

    Adds item to user's history, removes from dislikes.
    """
    if polyvore_u_generator is None:
        raise HTTPException(status_code=503, detail="Polyvore-U generator not initialized")

    # Validate item_id
    if request.item_id < 0 or request.item_id >= polyvore_u_generator.embeddings.shape[0]:
        raise HTTPException(status_code=404, detail=f"Item {request.item_id} not found")

    # Add to history
    if request.item_id not in polyvore_u_user_history[request.user_id]:
        polyvore_u_user_history[request.user_id].append(request.item_id)

    # Remove from dislikes
    polyvore_u_user_dislikes[request.user_id].discard(request.item_id)

    return PolyvoreULikeResponse(
        status="success",
        user_id=request.user_id,
        item_id=request.item_id,
        action="like",
        liked_count=len(polyvore_u_user_history[request.user_id]),
        disliked_count=len(polyvore_u_user_dislikes[request.user_id])
    )


@app.post("/polyvore-u/dislike", response_model=PolyvoreULikeResponse)
async def polyvore_u_dislike(request: PolyvoreULikeRequest):
    """
    Dislike an item in Polyvore-U.

    Adds item to dislikes, removes from history.
    """
    if polyvore_u_generator is None:
        raise HTTPException(status_code=503, detail="Polyvore-U generator not initialized")

    # Validate item_id
    if request.item_id < 0 or request.item_id >= polyvore_u_generator.embeddings.shape[0]:
        raise HTTPException(status_code=404, detail=f"Item {request.item_id} not found")

    # Add to dislikes
    polyvore_u_user_dislikes[request.user_id].add(request.item_id)

    # Remove from history
    if request.item_id in polyvore_u_user_history[request.user_id]:
        polyvore_u_user_history[request.user_id].remove(request.item_id)

    return PolyvoreULikeResponse(
        status="success",
        user_id=request.user_id,
        item_id=request.item_id,
        action="dislike",
        liked_count=len(polyvore_u_user_history[request.user_id]),
        disliked_count=len(polyvore_u_user_dislikes[request.user_id])
    )


@app.post("/polyvore-u/similar", response_model=PolyvoreUSimilarResponse)
async def get_polyvore_u_similar(request: PolyvoreUSimilarRequest):
    """
    Get items similar to a specific Polyvore-U item.

    Uses FashionCLIP + Faiss for visual similarity search.

    Optional filters:
    - categories: List of fine-grained categories (e.g., "women's t-shirt", "skirt")
    - broad_categories: List of "top", "bottom", "shoe"
    - genders: List of "women", "men", "unisex"
    - min_price, max_price: Price range filter
    - search: Search in product name
    """
    if polyvore_u_generator is None:
        raise HTTPException(status_code=503, detail="Polyvore-U generator not initialized")

    # Validate item_id
    if request.item_id < 0 or request.item_id >= polyvore_u_generator.embeddings.shape[0]:
        raise HTTPException(status_code=404, detail=f"Item {request.item_id} not found")

    # Get more items to account for filtering
    extra_for_filters = 200 if request.filters else 50
    max_items = request.page * request.page_size + extra_for_filters
    candidates = polyvore_u_generator.generate_candidates_with_scores(
        user_history=[request.item_id],
        k=max_items,
        exclude_seen=False
    )

    # Enrich items with full metadata
    enriched_items = []
    for item_id, score in candidates:
        if item_id == request.item_id:
            continue
        enriched_items.append(enrich_polyvore_u_item(item_id, score))

    # Apply filters if provided
    if request.filters:
        enriched_items = apply_polyvore_u_filters(enriched_items, request.filters)

    # Convert to Pydantic models
    items = [PolyvoreUFeedItem(**item) for item in enriched_items]

    # Apply pagination
    paginated_items, pagination_meta = paginate(items, request.page, request.page_size)

    return PolyvoreUSimilarResponse(
        item_id=request.item_id,
        similar=paginated_items,
        pagination=pagination_meta
    )


@app.get("/polyvore-u/user/{user_id}")
async def get_polyvore_u_user(user_id: str):
    """Get Polyvore-U user profile with enriched liked items."""
    liked_item_ids = polyvore_u_user_history.get(user_id, [])

    # Enrich liked items with full metadata
    liked_items_enriched = [
        enrich_polyvore_u_item(item_id)
        for item_id in liked_item_ids
    ]

    return {
        "user_id": user_id,
        "liked_items": liked_items_enriched,
        "liked_count": len(liked_item_ids),
        "disliked_count": len(polyvore_u_user_dislikes.get(user_id, set()))
    }


@app.delete("/polyvore-u/user/{user_id}")
async def clear_polyvore_u_user(user_id: str):
    """Clear Polyvore-U user history."""
    polyvore_u_user_history[user_id] = []
    polyvore_u_user_dislikes[user_id] = set()

    return {"status": "success", "user_id": user_id, "message": "History cleared"}


@app.get("/polyvore-u/item/{item_id}")
async def get_polyvore_u_item(item_id: int):
    """Get details for a specific Polyvore-U item with full metadata including reviews."""
    if polyvore_u_generator is None:
        raise HTTPException(status_code=503, detail="Polyvore-U generator not initialized")

    if item_id < 0 or item_id >= polyvore_u_generator.embeddings.shape[0]:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    # Get enriched item with all metadata including reviews
    enriched = enrich_polyvore_u_item(item_id, include_reviews=True)
    enriched["has_embedding"] = True

    return enriched


# === Style Quiz Endpoints ===

@app.get("/style-quiz/step1", response_model=StyleQuizStep1Response)
async def get_style_cards(gender: str = "women"):
    """
    Step 1: Get style cards with sample images.

    Returns 9 style categories with sample images for user to select preferences.
    """
    if gender not in ["women", "men"]:
        raise HTTPException(status_code=400, detail="Gender must be 'women' or 'men'")

    styles = []
    for style_name, desc in STYLE_DESCRIPTIONS.items():
        item_ids = style_quiz_items.get(style_name, {}).get(gender, [])

        if item_ids:
            # Get 4 sample images for the card
            sample_images = []
            preview_ids = []
            for item_id in item_ids[:4]:
                sample_images.append(get_style_quiz_image_url(item_id))
                preview_ids.append(item_id)

            styles.append(StyleCardResponse(
                name=style_name,
                description=desc,
                sample_images=sample_images,
                preview_item_ids=preview_ids
            ))

    return StyleQuizStep1Response(gender=gender, styles=styles)


@app.post("/style-quiz/step2", response_model=StyleQuizStep2Response)
async def get_style_items(request: StyleQuizStep2Request):
    """
    Step 2: Get all items from selected styles for user to pick.

    User must select minimum 15 items from these to complete setup.
    """
    if request.gender not in ["women", "men"]:
        raise HTTPException(status_code=400, detail="Gender must be 'women' or 'men'")

    items = []
    for style_name in request.selected_styles:
        if style_name not in STYLE_DESCRIPTIONS:
            continue

        style_items_list = style_quiz_items.get(style_name, {}).get(request.gender, [])
        for item_id in style_items_list:
            raw_category = get_polyvore_u_category(item_id) if polyvore_u_iid_cate_map else "unknown"
            category = clean_category_name(raw_category) if raw_category else "unknown"
            broad_cat = get_polyvore_u_broad_category(item_id)
            item_gender = get_polyvore_u_gender(item_id)
            real_name = get_polyvore_u_real_name(item_id)
            description = get_polyvore_u_description(item_id)
            items.append(StyleQuizItemResponse(
                item_id=item_id,
                name=real_name,
                price=get_mock_price(item_id),
                image_url=get_style_quiz_image_url(item_id),
                category=category,
                broad_category=broad_cat,
                gender=item_gender,
                style=style_name,
                description=description
            ))

    # Shuffle items to mix styles together
    random.shuffle(items)

    return StyleQuizStep2Response(
        gender=request.gender,
        selected_styles=request.selected_styles,
        items=items,
        minimum_selection=15
    )


@app.post("/user/setup", response_model=SetupUserResponse)
async def setup_user(request: SetupUserRequest):
    """
    Complete user setup with selected items.

    Validates minimum 15 items selected.
    Creates user history for feed personalization.
    """
    if len(request.selected_item_ids) < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Minimum 5 items required. You selected {len(request.selected_item_ids)}."
        )

    if request.gender not in ["women", "men"]:
        raise HTTPException(status_code=400, detail="Gender must be 'women' or 'men'")

    user_id = request.user_id

    # Store user profile with gender
    user_setup_profiles[user_id] = {
        'gender': request.gender,
        'setup_complete': True,
        'setup_items': request.selected_item_ids
    }

    # Set initial history for Polyvore-U feed generation
    polyvore_u_user_history[user_id] = list(request.selected_item_ids)

    # Also update the general user_profiles for original feed endpoint
    user_profiles[user_id]['history'] = [
        {'item_id': str(item_id), 'action': 'setup_selection'}
        for item_id in request.selected_item_ids
    ]

    return SetupUserResponse(
        user_id=user_id,
        gender=request.gender,
        items_selected=len(request.selected_item_ids),
        setup_complete=True,
        message=f"Setup complete! Your feed will be personalized based on {len(request.selected_item_ids)} items."
    )


@app.post("/user/exclusions", response_model=UserExclusionsResponse)
async def set_user_exclusions(request: UserExclusionsRequest):
    """
    Set user exclusion preferences (things to avoid).

    Auto-dislikes representative items for each selected exclusion option.
    This affects feed filtering - disliked items won't appear in recommendations.

    Categories:
    - colors: red, pink, khaki, yellow, green, blue, navy, purple, brown, gray, white, black
    - patterns: plaids, stripes, dots, floral, geometric, novelty
    - item_types: neckwear, graphic_tshirts, short_sleeves, button_down, vneck, pants, denim, coats_jackets, blazers, tank_tops, belts, shorts, socks, shoes
    - shoes: boat_shoes, boots, laceup_dress, performance_sneakers, driver, loafer, casual_sneaker, sandal
    """
    global style_exclusions

    if style_exclusions is None:
        raise HTTPException(
            status_code=503,
            detail="Style exclusions data not loaded. Please restart the server."
        )

    if request.gender not in ["women", "men"]:
        raise HTTPException(status_code=400, detail="Gender must be 'women' or 'men'")

    user_id = request.user_id
    gender = request.gender

    # Initialize user dislikes if not exists
    if user_id not in polyvore_u_user_dislikes:
        polyvore_u_user_dislikes[user_id] = set()

    items_added = 0

    # Process each exclusion category
    exclusion_categories = {
        'colors': request.exclusions.colors,
        'patterns': request.exclusions.patterns,
        'item_types': request.exclusions.item_types,
        'shoes': request.exclusions.shoes
    }

    for category, options in exclusion_categories.items():
        if options is None:
            continue

        category_data = style_exclusions.get(category, {})
        for option in options:
            option_data = category_data.get(option, {})
            item_ids = option_data.get(gender, [])

            for item_id in item_ids:
                if item_id not in polyvore_u_user_dislikes[user_id]:
                    polyvore_u_user_dislikes[user_id].add(item_id)
                    items_added += 1

    return UserExclusionsResponse(
        user_id=user_id,
        items_excluded=items_added,
        message=f"Exclusions applied. {items_added} items added to dislikes."
    )


@app.get("/user/{user_id}/profile")
async def get_user_profile(user_id: str):
    """Get user profile including setup status."""
    profile = user_setup_profiles.get(user_id)
    if not profile:
        return {
            "user_id": user_id,
            "setup_complete": False,
            "gender": None,
            "liked_count": len(polyvore_u_user_history.get(user_id, []))
        }

    return {
        "user_id": user_id,
        "setup_complete": profile.get('setup_complete', False),
        "gender": profile.get('gender'),
        "setup_items_count": len(profile.get('setup_items', [])),
        "liked_count": len(polyvore_u_user_history.get(user_id, []))
    }


@app.get("/style-quiz/styles")
async def get_available_styles():
    """Get list of available style categories with descriptions."""
    return {
        "styles": [
            {"name": name, "description": desc}
            for name, desc in STYLE_DESCRIPTIONS.items()
        ],
        "total_styles": len(STYLE_DESCRIPTIONS)
    }


@app.get("/style-quiz/items", response_model=StyleQuizItemsResponse)
async def get_style_quiz_items(gender: str = "women", items_per_style: int = 3):
    """
    Simplified single-step style quiz: Get diverse items across all styles.

    Returns 27 items by default (3 per style × 9 styles), shuffled together.
    User selects items they like (minimum 5) to seed recommendations.

    Args:
        gender: "women" or "men"
        items_per_style: Number of items to include per style (default 3)
    """
    if gender not in ["women", "men"]:
        raise HTTPException(status_code=400, detail="Gender must be 'women' or 'men'")

    if items_per_style < 1 or items_per_style > 10:
        raise HTTPException(status_code=400, detail="items_per_style must be between 1 and 10")

    items = []

    # Get items from each style category
    for style_name in STYLE_DESCRIPTIONS.keys():
        style_items_list = style_quiz_items.get(style_name, {}).get(gender, [])

        # Take up to items_per_style from each style
        for item_id in style_items_list[:items_per_style]:
            raw_category = get_polyvore_u_category(item_id) if polyvore_u_iid_cate_map else "unknown"
            # Clean category name by removing gender prefix for display
            category = clean_category_name(raw_category) if raw_category else "unknown"
            broad_cat = get_polyvore_u_broad_category(item_id)
            item_gender = get_polyvore_u_gender(item_id)
            real_name = get_polyvore_u_real_name(item_id)
            description = get_polyvore_u_description(item_id)
            items.append(StyleQuizItemResponse(
                item_id=item_id,
                name=real_name,
                price=get_mock_price(item_id),
                image_url=get_style_quiz_image_url(item_id),
                category=category,
                broad_category=broad_cat,
                gender=item_gender,
                style=style_name,
                description=description
            ))

    # Shuffle to mix styles together for variety
    random.shuffle(items)

    return StyleQuizItemsResponse(
        gender=gender,
        items=items,
        total_items=len(items),
        minimum_selection=5
    )


# === Amazon Fashion Models ===

class AmazonFeedRequest(BaseModel):
    """Request model for Amazon Fashion feed generation"""
    user_id: str = Field(..., description="User identifier (Amazon user ID or new user)")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")


class AmazonFeedItem(BaseModel):
    """Single item in Amazon Fashion feed response"""
    item_id: str
    title: str = ""
    brand: str = ""
    category: List[str] = []
    price: str = ""
    image_url: str
    score: float
    source: str  # 'sasrec' or 'clip'


class AmazonFeedResponse(BaseModel):
    """Response model for Amazon Fashion feed"""
    user_id: str
    items: List[AmazonFeedItem]
    pagination: PaginationMeta
    history_count: int = 0


class AmazonSimilarRequest(BaseModel):
    """Request model for Amazon similar items"""
    item_id: str = Field(..., description="Item ID (ASIN) to find similar items for")
    k: int = Field(default=10, ge=1, le=50, description="Number of similar items to return")


class AmazonSimilarResponse(BaseModel):
    """Response model for Amazon similar items"""
    item_id: str
    item_info: Optional[Dict] = None
    similar_items: List[AmazonFeedItem]


class AmazonLikeRequest(BaseModel):
    """Request model for like/dislike actions"""
    user_id: str = Field(..., description="User identifier")
    item_id: str = Field(..., description="Item ID (ASIN)")


class AmazonLikeResponse(BaseModel):
    """Response model for like/dislike actions"""
    status: str
    user_id: str
    item_id: str
    action: str
    liked_count: int
    disliked_count: int


class AmazonStyleQuizRequest(BaseModel):
    """Request for cold-start style quiz"""
    liked_items: List[str] = Field(..., description="List of item IDs user liked in quiz", min_length=1)
    disliked_items: Optional[List[str]] = Field(default=None, description="List of item IDs user disliked")
    num_recommendations: int = Field(default=20, ge=5, le=100, description="Number of recommendations")


class AmazonStyleQuizResponse(BaseModel):
    """Response for style quiz recommendations"""
    recommendations: List[AmazonFeedItem]
    based_on_items: int


class AmazonHealthResponse(BaseModel):
    """Response model for Amazon Fashion health check"""
    status: str
    items_count: int
    users_count: int
    sasrec_loaded: bool
    metadata_loaded: bool


# === Amazon Fashion Endpoints ===

def get_amazon_item_info(item_id: str) -> Dict:
    """Get metadata for an Amazon item."""
    if amazon_metadata and item_id in amazon_metadata:
        meta = amazon_metadata[item_id]
        return {
            'title': meta.get('title', ''),
            'brand': meta.get('brand', ''),
            'category': meta.get('category', []),
            'price': meta.get('price', '')
        }
    return {'title': '', 'brand': '', 'category': [], 'price': ''}


def get_amazon_image_url(item_id: str) -> str:
    """Get image URL for an Amazon item."""
    return f"/amazon-images/{item_id}.jpg"


@app.get("/amazon/health", response_model=AmazonHealthResponse)
async def amazon_health_check():
    """Health check for Amazon Fashion endpoints"""
    if amazon_feed is None:
        return AmazonHealthResponse(
            status="unavailable",
            items_count=0,
            users_count=0,
            sasrec_loaded=False,
            metadata_loaded=False
        )

    return AmazonHealthResponse(
        status="healthy",
        items_count=len(amazon_feed.item_ids),
        users_count=len(amazon_feed.user_history),
        sasrec_loaded=amazon_feed.seq_model is not None,
        metadata_loaded=amazon_metadata is not None
    )


@app.post("/amazon/feed", response_model=AmazonFeedResponse)
async def amazon_feed_endpoint(request: AmazonFeedRequest):
    """
    Get personalized feed for an Amazon Fashion user.

    Uses hybrid approach:
    1. FashionCLIP to find visually similar items to user's history
    2. SASRec to re-rank candidates based on sequential patterns
    """
    if amazon_feed is None:
        raise HTTPException(status_code=503, detail="Amazon Fashion feed not available")

    user_id = request.user_id

    # Check if user exists in original dataset
    history_count = 0
    if user_id in amazon_feed.user_history:
        history_count = len(amazon_feed.user_history[user_id])

    # Get additional dislikes from API interactions
    user_dislikes = amazon_user_dislikes.get(user_id, set())

    # Get recommendations (larger pool for filtering and pagination)
    pool_size = request.page * request.page_size + 100
    recommendations = amazon_feed.get_feed(
        user_id,
        limit=pool_size,
        exclude_seen=True
    )

    # Filter out disliked items
    filtered_recs = [r for r in recommendations if r['item_id'] not in user_dislikes]

    # Paginate
    paginated_recs, pagination = paginate(filtered_recs, request.page, request.page_size)

    # Build response items with metadata
    items = []
    for rec in paginated_recs:
        item_id = rec['item_id']
        info = get_amazon_item_info(item_id)
        items.append(AmazonFeedItem(
            item_id=item_id,
            title=info['title'],
            brand=info['brand'],
            category=info['category'],
            price=info['price'],
            image_url=get_amazon_image_url(item_id),
            score=rec['score'],
            source=rec['source']
        ))

    return AmazonFeedResponse(
        user_id=user_id,
        items=items,
        pagination=pagination,
        history_count=history_count
    )


@app.post("/amazon/similar", response_model=AmazonSimilarResponse)
async def amazon_similar_items(request: AmazonSimilarRequest):
    """Find visually similar items using FashionCLIP embeddings."""
    if amazon_feed is None:
        raise HTTPException(status_code=503, detail="Amazon Fashion feed not available")

    item_id = request.item_id

    if item_id not in amazon_feed.embeddings_dict:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    # Get similar items
    similar = amazon_feed.get_similar_items(item_id, k=request.k)

    # Build response
    similar_items = []
    for item in similar:
        sim_id = item['item_id']
        info = get_amazon_item_info(sim_id)
        similar_items.append(AmazonFeedItem(
            item_id=sim_id,
            title=info['title'],
            brand=info['brand'],
            category=info['category'],
            price=info['price'],
            image_url=get_amazon_image_url(sim_id),
            score=item['similarity'],
            source='clip'
        ))

    return AmazonSimilarResponse(
        item_id=item_id,
        item_info=get_amazon_item_info(item_id),
        similar_items=similar_items
    )


@app.post("/amazon/like", response_model=AmazonLikeResponse)
async def amazon_like_item(request: AmazonLikeRequest):
    """Record a like for an item (adds to user's positive signal)."""
    if amazon_feed is None:
        raise HTTPException(status_code=503, detail="Amazon Fashion feed not available")

    user_id = request.user_id
    item_id = request.item_id

    # Add to likes, remove from dislikes if present
    amazon_user_likes[user_id].add(item_id)
    amazon_user_dislikes[user_id].discard(item_id)

    return AmazonLikeResponse(
        status="success",
        user_id=user_id,
        item_id=item_id,
        action="like",
        liked_count=len(amazon_user_likes[user_id]),
        disliked_count=len(amazon_user_dislikes[user_id])
    )


@app.post("/amazon/dislike", response_model=AmazonLikeResponse)
async def amazon_dislike_item(request: AmazonLikeRequest):
    """Record a dislike for an item (filters from future recommendations)."""
    if amazon_feed is None:
        raise HTTPException(status_code=503, detail="Amazon Fashion feed not available")

    user_id = request.user_id
    item_id = request.item_id

    # Add to dislikes, remove from likes if present
    amazon_user_dislikes[user_id].add(item_id)
    amazon_user_likes[user_id].discard(item_id)

    return AmazonLikeResponse(
        status="success",
        user_id=user_id,
        item_id=item_id,
        action="dislike",
        liked_count=len(amazon_user_likes[user_id]),
        disliked_count=len(amazon_user_dislikes[user_id])
    )


@app.post("/amazon/style-quiz", response_model=AmazonStyleQuizResponse)
async def amazon_style_quiz(request: AmazonStyleQuizRequest):
    """
    Cold-start style quiz: get recommendations based on liked items.

    For new users without purchase history, show them items and let them
    like/dislike. Then use FashionCLIP to build a style vector and
    recommend similar items.
    """
    if amazon_feed is None:
        raise HTTPException(status_code=503, detail="Amazon Fashion feed not available")

    # Validate items exist
    valid_liked = [i for i in request.liked_items if i in amazon_feed.embeddings_dict]
    valid_disliked = [i for i in (request.disliked_items or []) if i in amazon_feed.embeddings_dict]

    if not valid_liked:
        raise HTTPException(status_code=400, detail="No valid liked items found")

    # Build user vector from style quiz
    user_vec = amazon_feed.style_quiz_init(valid_liked, valid_disliked)

    if user_vec is None:
        raise HTTPException(status_code=400, detail="Could not build style profile")

    # Get candidates
    exclude = set(valid_liked + valid_disliked)
    candidates = amazon_feed.get_clip_candidates(
        user_vec,
        k=request.num_recommendations,
        exclude_items=list(exclude)
    )

    # Build response
    items = []
    for item_id, score in candidates:
        info = get_amazon_item_info(item_id)
        items.append(AmazonFeedItem(
            item_id=item_id,
            title=info['title'],
            brand=info['brand'],
            category=info['category'],
            price=info['price'],
            image_url=get_amazon_image_url(item_id),
            score=score,
            source='clip'
        ))

    return AmazonStyleQuizResponse(
        recommendations=items,
        based_on_items=len(valid_liked)
    )


@app.get("/amazon/item/{item_id}")
async def amazon_get_item(item_id: str):
    """Get details for a specific Amazon item."""
    if amazon_feed is None:
        raise HTTPException(status_code=503, detail="Amazon Fashion feed not available")

    if item_id not in amazon_feed.embeddings_dict:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    info = get_amazon_item_info(item_id)

    return {
        "item_id": item_id,
        "title": info['title'],
        "brand": info['brand'],
        "category": info['category'],
        "price": info['price'],
        "image_url": get_amazon_image_url(item_id),
        "has_embedding": True
    }


@app.get("/amazon/user/{user_id}")
async def amazon_get_user(user_id: str):
    """Get user profile including history and preferences."""
    if amazon_feed is None:
        raise HTTPException(status_code=503, detail="Amazon Fashion feed not available")

    # Get original purchase history
    history = amazon_feed.user_history.get(user_id, [])

    # Get API-recorded likes/dislikes
    likes = list(amazon_user_likes.get(user_id, set()))
    dislikes = list(amazon_user_dislikes.get(user_id, set()))

    return {
        "user_id": user_id,
        "purchase_history_count": len(history),
        "recent_purchases": history[-10:] if history else [],
        "api_likes": likes,
        "api_dislikes_count": len(dislikes)
    }


@app.get("/amazon/random-items")
async def amazon_random_items(count: int = Query(default=20, ge=1, le=100)):
    """Get random items for style quiz or browsing."""
    if amazon_feed is None:
        raise HTTPException(status_code=503, detail="Amazon Fashion feed not available")

    # Get random items that have images
    import os as _os
    project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    images_dir = _os.path.join(project_root, "data/amazon_fashion/images")

    items_with_images = [
        item_id for item_id in amazon_feed.item_ids
        if _os.path.exists(_os.path.join(images_dir, f"{item_id}.jpg"))
    ]

    # Sample random items
    sample_size = min(count, len(items_with_images))
    sampled = random.sample(items_with_images, sample_size)

    # Build response
    items = []
    for item_id in sampled:
        info = get_amazon_item_info(item_id)
        items.append({
            "item_id": item_id,
            "title": info['title'],
            "brand": info['brand'],
            "category": info['category'],
            "price": info['price'],
            "image_url": get_amazon_image_url(item_id)
        })

    return {"items": items, "count": len(items)}


@app.get("/amazon/random-user")
async def amazon_random_user(min_history: int = Query(default=5, ge=1)):
    """Get a random user ID with sufficient purchase history."""
    if amazon_feed is None:
        raise HTTPException(status_code=503, detail="Amazon Fashion feed not available")

    # Find users with enough history
    eligible_users = [
        uid for uid, items in amazon_feed.user_history.items()
        if len(items) >= min_history
    ]

    if not eligible_users:
        raise HTTPException(status_code=404, detail=f"No users found with >= {min_history} items")

    selected = random.choice(eligible_users)
    history = amazon_feed.user_history[selected]

    return {
        "user_id": selected,
        "history_count": len(history),
        "recent_items": history[-5:]
    }


# ============================================================================
# OUTROVE ONBOARDING API ENDPOINTS
# ============================================================================

# Outrove filter state (lazy loaded)
outrove_filter = None


def load_outrove_filter():
    """Load Outrove candidate filter."""
    global outrove_filter
    if outrove_filter is not None:
        return True

    try:
        from outrove_filter import OutroveCandidateFilter
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tops_path = os.path.join(project_root, "data/amazon_fashion/processed/tops_enriched.pkl")

        if not os.path.exists(tops_path):
            print("Warning: tops_enriched.pkl not found. Run precompute_tops_attributes.py first.")
            return False

        outrove_filter = OutroveCandidateFilter(tops_path)
        print(f"Outrove filter loaded: {outrove_filter.get_stats()['total_items']} tops items")
        return True

    except Exception as e:
        print(f"Warning: Could not load Outrove filter: {e}")
        import traceback
        traceback.print_exc()
        return False


# Pydantic models for Outrove API

class OutroveGlobalPreferences(BaseModel):
    """Global preferences from onboarding."""
    selectedCoreTypes: List[str] = Field(default_factory=list, description="e.g., ['t-shirts', 'hoodies']")
    typicalSize: List[str] = Field(default_factory=list, description="e.g., ['L', 'XL']")
    colorsToAvoid: List[str] = Field(default_factory=list, description="e.g., ['yellow', 'pink']")
    materialsToAvoid: List[str] = Field(default_factory=list, description="e.g., ['polyester']")


class OutroveTShirtsPrefs(BaseModel):
    """T-shirts module preferences."""
    size: Optional[List[str]] = None
    fit: Optional[str] = None
    sleeveLength: Optional[str] = None
    necklines: Optional[List[str]] = None
    styleVariants: Optional[List[str]] = None
    graphicsTolerance: Optional[str] = None
    priceRange: Optional[List[float]] = None


class OutrovePolosPrefs(BaseModel):
    """Polos module preferences."""
    size: Optional[List[str]] = None
    fit: Optional[str] = None
    styleVariants: Optional[List[str]] = None
    patternLogoTolerance: Optional[str] = None
    pocketPreference: Optional[str] = None
    priceRange: Optional[List[float]] = None


class OutroveSweaterPrefs(BaseModel):
    """Sweaters module preferences."""
    size: Optional[List[str]] = None
    fit: Optional[str] = None
    necklines: Optional[List[str]] = None
    materials: Optional[List[str]] = None
    weight: Optional[str] = None
    priceRange: Optional[List[float]] = None


class OutroveHoodiesPrefs(BaseModel):
    """Hoodies module preferences."""
    size: Optional[List[str]] = None
    fit: Optional[str] = None
    stylePreference: Optional[List[str]] = None
    brandingTolerance: Optional[str] = None
    priceRange: Optional[List[float]] = None


class OutroveShirtsPrefs(BaseModel):
    """Button-down shirts module preferences."""
    size: Optional[List[str]] = None
    fit: Optional[str] = None
    tuckPreference: Optional[str] = None
    fabrics: Optional[List[str]] = None
    patterns: Optional[List[str]] = None
    priceRange: Optional[List[float]] = None


class OutroveOnboardingRequest(BaseModel):
    """Complete Outrove onboarding profile."""
    user_id: Optional[str] = "anonymous"
    selectedCoreTypes: List[str] = Field(..., description="Required: product categories of interest")
    typicalSize: List[str] = Field(default_factory=list)
    colorsToAvoid: List[str] = Field(default_factory=list)
    materialsToAvoid: List[str] = Field(default_factory=list)
    tshirts: Optional[OutroveTShirtsPrefs] = None
    polos: Optional[OutrovePolosPrefs] = None
    sweaters: Optional[OutroveSweaterPrefs] = None
    hoodies: Optional[OutroveHoodiesPrefs] = None
    shirts: Optional[OutroveShirtsPrefs] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


class OutroveItemResponse(BaseModel):
    """Item in Outrove response."""
    item_id: str
    score: float
    category: str
    title: str
    brand: str
    price: Optional[float]
    colors: List[str]
    materials: List[str]
    fit: Optional[str]
    visual_fit: Optional[str]
    pattern: Optional[str]
    visual_pattern: Optional[str]
    image_url: str


class OutroveFeedResponse(BaseModel):
    """Outrove feed response."""
    user_id: str
    items: List[OutroveItemResponse]
    by_category: Dict[str, int]
    pagination: PaginationMeta


@app.get("/outrove/health")
async def outrove_health():
    """Check Outrove service health."""
    if not load_outrove_filter():
        raise HTTPException(status_code=503, detail="Outrove filter not available")

    stats = outrove_filter.get_stats()
    return {
        "status": "healthy",
        "total_items": stats['total_items'],
        "categories": stats['items_per_category'],
        "has_visual_features": stats['has_embeddings'],
    }


@app.get("/outrove/options")
async def outrove_options():
    """Get available options for onboarding form."""
    return {
        "core_types": ["t-shirts", "polos", "sweaters", "hoodies", "shirts"],
        "sizes": ["XS", "S", "M", "L", "XL", "XXL", "3XL"],
        "colors": [
            "white", "beige", "tan", "yellow", "mustard", "orange", "coral",
            "red", "burgundy", "pink", "lavender", "purple", "light-blue",
            "blue", "navy", "teal", "mint", "green", "olive", "brown",
            "gray", "charcoal", "black"
        ],
        "materials": ["polyester", "wool", "linen", "silk", "leather", "synthetics"],
        "fits": ["slim", "regular", "relaxed", "oversized"],
        "tshirt_necklines": ["crew", "v-neck", "henley"],
        "tshirt_styles": ["plain", "small-graphics", "graphic-tees", "pocket-tees", "athletic"],
        "graphics_tolerance": ["no-graphics", "small-logos", "graphics-ok", "bold-graphics"],
        "hoodie_styles": ["zip-up", "pullover"],
        "shirt_fabrics": ["oxford", "poplin", "flannel", "denim", "linen", "overshirts"],
        "shirt_patterns": ["solids", "subtle", "stripes", "plaid", "bold"],
    }


@app.post("/outrove/feed", response_model=OutroveFeedResponse)
async def outrove_feed(request: OutroveOnboardingRequest):
    """
    Generate personalized feed based on Outrove onboarding profile.

    Pipeline:
    1. Strict filtering (Outrove): category, colors to avoid, materials to avoid, price
    2. Soft preference scoring (OR logic): necklines, fit, materials, patterns - boost by match count
    3. Ranking (SASRec): sequential model scores candidates based on user history

    Items matching more preferences rank higher.
    If user has no history, items are ranked by preference match count.
    """
    if not load_outrove_filter():
        raise HTTPException(status_code=503, detail="Outrove filter not available")

    from outrove_filter import OutroveUserProfile

    # Parse profile from request
    profile = OutroveUserProfile.from_dict(request.model_dump())

    if not profile.global_prefs.selected_core_types:
        raise HTTPException(status_code=400, detail="selectedCoreTypes is required")

    # Stage 1: Get filtered candidates WITH preference scores (OR logic for soft preferences)
    scored_candidates = outrove_filter.get_candidates_with_scores(profile)
    candidates_by_cat = outrove_filter.get_candidates_by_category(profile)

    # Count by category
    by_category = {cat: len(items) for cat, items in candidates_by_cat.items()}

    # Stage 2: Rank with SASRec (if amazon_feed is loaded and has the model)
    ranked_items = []

    if amazon_feed is not None and amazon_feed.seq_model is not None:
        # Use SASRec to rank candidates, combining with preference scores
        user_id = request.user_id or "anonymous"
        candidate_list = [item_id for item_id, _ in scored_candidates]
        preference_scores = {item_id: score for item_id, score in scored_candidates}

        # Get SASRec scores
        ranked = amazon_feed.rank_with_sasrec(
            user_id,
            candidate_list,
            topk=len(candidate_list)  # Rank all candidates
        )

        # Combine SASRec scores with preference scores
        for item_id, sasrec_score in ranked:
            item_data = outrove_filter.get_item(item_id)
            if item_data:
                # Combine scores: SASRec + preference boost
                pref_score = preference_scores.get(item_id, 1.0)
                combined_score = sasrec_score + (pref_score - 1.0)  # Add preference bonus
                ranked_items.append((item_id, combined_score, item_data.get('outrove_type', 'unknown')))

        # Re-sort by combined score
        ranked_items.sort(key=lambda x: x[1], reverse=True)
    else:
        # No SASRec available - use preference scores for ranking
        for item_id, pref_score in scored_candidates:
            item_data = outrove_filter.get_item(item_id)
            if item_data:
                ranked_items.append((item_id, pref_score, item_data.get('outrove_type', 'unknown')))

    # Paginate
    paginated, meta = paginate(ranked_items, request.page, request.page_size)

    # Build response items
    items = []
    for item_id, score, category in paginated:
        item_data = outrove_filter.get_item(item_id)
        if item_data:
            items.append(OutroveItemResponse(
                item_id=item_id,
                score=round(score, 3),
                category=category,
                title=item_data.get('title', 'Unknown'),
                brand=item_data.get('brand', ''),
                price=item_data.get('price'),
                colors=item_data.get('colors', []),
                materials=item_data.get('materials', []),
                fit=item_data.get('fit'),
                visual_fit=item_data.get('visual_fit'),
                pattern=item_data.get('pattern'),
                visual_pattern=item_data.get('visual_pattern'),
                image_url=get_amazon_image_url(item_id),
            ))

    return OutroveFeedResponse(
        user_id=request.user_id or "anonymous",
        items=items,
        by_category=dict(by_category),
        pagination=meta,
    )


@app.post("/outrove/filter-stats")
async def outrove_filter_stats(request: OutroveOnboardingRequest):
    """
    Get statistics about filtering without returning items.

    Useful for showing user how many items match their preferences.
    """
    if not load_outrove_filter():
        raise HTTPException(status_code=503, detail="Outrove filter not available")

    from outrove_filter import OutroveUserProfile

    profile = OutroveUserProfile.from_dict(request.model_dump())

    if not profile.global_prefs.selected_core_types:
        raise HTTPException(status_code=400, detail="selectedCoreTypes is required")

    # Get filter statistics
    stats = outrove_filter.get_filter_stats(profile)

    total_before = stats['items_before_filtering']
    total_after = stats['items_after_filtering']

    return {
        "user_id": request.user_id,
        "selected_categories": profile.global_prefs.selected_core_types,
        "filters_applied": stats['filters_applied'],
        "items_before_filtering": total_before,
        "items_after_filtering": total_after,
        "items_filtered_out": total_before - total_after,
        "filter_rate": stats['filter_rate_percent'],
        "by_category": stats['by_category'],
    }


@app.get("/outrove/item/{item_id}")
async def outrove_item(item_id: str):
    """Get detailed item information including all extracted attributes."""
    if not load_outrove_filter():
        raise HTTPException(status_code=503, detail="Outrove filter not available")

    item = outrove_filter.get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found in tops catalog")

    # Add image URL
    item_response = dict(item)
    item_response['image_url'] = get_amazon_image_url(item_id)

    return item_response


@app.get("/outrove/category/{category}/items")
async def outrove_category_items(
    category: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """Get all items in a specific Outrove category."""
    if not load_outrove_filter():
        raise HTTPException(status_code=503, detail="Outrove filter not available")

    valid_categories = ["t-shirts", "polos", "sweaters", "hoodies", "shirts", "henleys"]
    if category not in valid_categories:
        raise HTTPException(status_code=400, detail=f"Invalid category. Must be one of: {valid_categories}")

    item_ids = list(outrove_filter.by_category.get(category, set()))
    paginated_ids, meta = paginate(item_ids, page, page_size)

    items = []
    for item_id in paginated_ids:
        item_data = outrove_filter.get_item(item_id)
        if item_data:
            items.append({
                "item_id": item_id,
                "title": item_data.get('title', 'Unknown'),
                "brand": item_data.get('brand', ''),
                "price": item_data.get('price'),
                "image_url": get_amazon_image_url(item_id),
            })

    return {
        "category": category,
        "items": items,
        "pagination": meta,
    }



# Run with: uvicorn api:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    uvicorn.run(app, host=host, port=port)
