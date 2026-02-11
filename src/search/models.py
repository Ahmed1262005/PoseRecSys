"""
Pydantic models for the search API.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


# ============================================================================
# Enums
# ============================================================================

class QueryIntent(str, Enum):
    """Classification of search query intent."""
    EXACT = "exact"        # Brand name, product name, SKU
    SPECIFIC = "specific"  # Category + attribute ("blue midi dress")
    VAGUE = "vague"        # Style/occasion/vibe ("quiet luxury", "date night")


# ============================================================================
# Request Models
# ============================================================================

class HybridSearchRequest(BaseModel):
    """Request body for hybrid search."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")

    # Filters
    categories: Optional[List[str]] = Field(None, description="Broad categories (tops, bottoms, dresses, outerwear)")
    category_l1: Optional[List[str]] = Field(None, description="Gemini L1 categories (Tops, Bottoms, Dresses)")
    category_l2: Optional[List[str]] = Field(None, description="Gemini L2 categories (Blouse, Jeans)")
    brands: Optional[List[str]] = Field(None, description="Filter by brand names")
    exclude_brands: Optional[List[str]] = Field(None, description="Brands to exclude")
    colors: Optional[List[str]] = Field(None, description="Filter by colors")
    color_family: Optional[List[str]] = Field(None, description="Filter by color family")
    patterns: Optional[List[str]] = Field(None, description="Filter by patterns (Solid, Floral, etc.)")
    materials: Optional[List[str]] = Field(None, description="Filter by materials")
    occasions: Optional[List[str]] = Field(None, description="Filter by occasions")
    seasons: Optional[List[str]] = Field(None, description="Filter by seasons")
    formality: Optional[List[str]] = Field(None, description="Filter by formality level")
    fit_type: Optional[List[str]] = Field(None, description="Filter by fit type")
    neckline: Optional[List[str]] = Field(None, description="Filter by neckline")
    sleeve_type: Optional[List[str]] = Field(None, description="Filter by sleeve type")
    length: Optional[List[str]] = Field(None, description="Filter by length")
    rise: Optional[List[str]] = Field(None, description="Filter by rise (bottoms)")
    silhouette: Optional[List[str]] = Field(None, description="Filter by silhouette (Fitted, A-Line, Straight)")
    article_type: Optional[List[str]] = Field(None, description="Filter by article type (jeans, t-shirt, midi dress)")
    style_tags: Optional[List[str]] = Field(None, description="Filter by style tags (boho, minimalist, preppy)")
    min_price: Optional[float] = Field(None, ge=0, description="Minimum price")
    max_price: Optional[float] = Field(None, ge=0, description="Maximum price")
    on_sale_only: bool = Field(False, description="Only show sale items")

    # Pagination
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(50, ge=1, le=100, description="Results per page")

    # Session
    session_id: Optional[str] = Field(None, description="Session ID for deduplication")

    # Search options
    semantic_boost: float = Field(0.4, ge=0.0, le=1.0, description="Weight for semantic results in RRF")

    @model_validator(mode="after")
    def validate_price_range(self):
        """Ensure min_price <= max_price when both are set."""
        if self.min_price is not None and self.max_price is not None:
            if self.min_price > self.max_price:
                raise ValueError(f"min_price ({self.min_price}) must be <= max_price ({self.max_price})")
        return self


class AutocompleteRequest(BaseModel):
    """Request for autocomplete."""
    query: str = Field(..., min_length=1, max_length=200)
    limit: int = Field(10, ge=1, le=20)


class SearchClickRequest(BaseModel):
    """Record a search result click."""
    query: str
    product_id: str
    position: int


class SearchConversionRequest(BaseModel):
    """Record a search conversion."""
    query: str
    product_id: str


# ============================================================================
# Response Models
# ============================================================================

class ProductResult(BaseModel):
    """A single product in search results."""
    product_id: str
    name: str
    brand: str
    image_url: Optional[str] = None
    gallery_images: Optional[List[str]] = None
    price: float = 0
    original_price: Optional[float] = None
    is_on_sale: bool = False

    # Category
    category_l1: Optional[str] = None
    category_l2: Optional[str] = None
    broad_category: Optional[str] = None
    article_type: Optional[str] = None

    # Attributes
    primary_color: Optional[str] = None
    color_family: Optional[str] = None
    pattern: Optional[str] = None
    apparent_fabric: Optional[str] = None
    fit_type: Optional[str] = None
    formality: Optional[str] = None
    silhouette: Optional[str] = None
    length: Optional[str] = None
    neckline: Optional[str] = None
    sleeve_type: Optional[str] = None
    rise: Optional[str] = None
    style_tags: Optional[List[str]] = None
    occasions: Optional[List[str]] = None
    seasons: Optional[List[str]] = None

    # Ranking info
    algolia_rank: Optional[int] = None
    semantic_rank: Optional[int] = None
    semantic_score: Optional[float] = None
    rrf_score: Optional[float] = None


class PaginationInfo(BaseModel):
    """Pagination metadata."""
    page: int
    page_size: int
    has_more: bool
    total_results: Optional[int] = Field(None, description="Total merged results available (before pagination)")


class FacetValue(BaseModel):
    """A single facet value with its count."""
    value: str
    count: int


class HybridSearchResponse(BaseModel):
    """Response from hybrid search."""
    query: str
    intent: str
    results: List[ProductResult]
    pagination: PaginationInfo
    timing: Dict[str, int] = Field(default_factory=dict, description="Timing breakdown in ms")
    facets: Optional[Dict[str, List[FacetValue]]] = Field(
        None,
        description="Available filter options with counts (only values with count > 1). "
        "Keys: brand, category_l1, formality, primary_color, color_family, pattern, "
        "fit_type, neckline, sleeve_type, length, silhouette, rise, occasions, "
        "seasons, style_tags, article_type, broad_category, is_on_sale, materials",
    )


class AutocompleteProductSuggestion(BaseModel):
    """A product suggestion in autocomplete."""
    id: str
    name: str
    brand: str
    image_url: Optional[str] = None
    price: Optional[float] = None
    highlighted_name: Optional[str] = None


class AutocompleteBrandSuggestion(BaseModel):
    """A brand suggestion in autocomplete."""
    name: str
    highlighted: Optional[str] = None


class AutocompleteResponse(BaseModel):
    """Response from autocomplete."""
    products: List[AutocompleteProductSuggestion]
    brands: List[AutocompleteBrandSuggestion]
    query: str
