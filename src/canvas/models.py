"""Pydantic models for the Canvas API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InspirationSource(str, Enum):
    upload = "upload"
    url = "url"
    camera = "camera"
    pinterest = "pinterest"


# ---------------------------------------------------------------------------
# Inspiration CRUD
# ---------------------------------------------------------------------------

class InspirationResponse(BaseModel):
    """Single inspiration item returned to the client."""

    id: str
    source: InspirationSource
    image_url: str
    original_url: Optional[str] = None
    title: Optional[str] = None
    style_label: Optional[str] = None
    style_confidence: Optional[float] = None
    style_attributes: Dict[str, Any] = Field(default_factory=dict)
    pinterest_pin_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class InspirationListResponse(BaseModel):
    inspirations: List[InspirationResponse]
    count: int


class UrlInspirationRequest(BaseModel):
    """Request body for adding an inspiration from a URL."""

    url: str = Field(..., description="Public image URL")
    title: Optional[str] = Field(None, max_length=500)


class PinterestSyncRequest(BaseModel):
    """Request body for syncing Pinterest pins as inspirations."""

    pin_ids: Optional[List[str]] = Field(
        None,
        description="Specific Pinterest pin IDs to sync. "
        "If omitted, uses the user's saved board/section selection.",
    )
    max_pins: Optional[int] = Field(
        None, ge=1, le=200,
        description="Max pins to sync (defaults to server setting).",
    )


class DeleteInspirationResponse(BaseModel):
    deleted: bool
    taste_vector_updated: bool
    remaining_count: int
    inspirations: List["InspirationResponse"] = Field(
        default_factory=list,
        description="Surviving inspirations after deletion (authoritative list).",
    )


# ---------------------------------------------------------------------------
# Style elements (feed-param-ready)
# ---------------------------------------------------------------------------

class AttributeScore(BaseModel):
    """A single attribute value with its aggregation score."""

    value: str
    count: int = Field(description="How many inspirations contributed this value")
    confidence: float = Field(
        description="Normalised confidence (0-1) across inspirations",
    )


class StyleElementsResponse(BaseModel):
    """
    Extracted style attributes mapped to feed query-param names.

    ``suggested_filters`` contains keys that match the feed endpoint's
    ``include_*`` query parameters so the frontend can pass them directly.

    ``raw_attributes`` provides the full distribution per attribute category
    for UI display (toggle chips, confidence bars, etc.).
    """

    suggested_filters: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Pre-built filter dict keyed by feed param names "
            "(e.g. include_style_tags, include_patterns)."
        ),
    )
    raw_attributes: Dict[str, List[AttributeScore]] = Field(
        default_factory=dict,
        description="Full attribute distributions per category.",
    )
    inspiration_count: int = 0


# ---------------------------------------------------------------------------
# Complete-the-fit from an inspiration
# ---------------------------------------------------------------------------

class SimilarProductsResponse(BaseModel):
    """Products visually similar to an inspiration image."""

    products: List[Dict[str, Any]] = Field(
        description="Deduplicated list of similar products, ordered by similarity",
    )
    count: int = Field(description="Number of products returned")
    inspiration_id: str


class CompleteFitFromInspirationRequest(BaseModel):
    """Optional overrides when building an outfit from an inspiration."""

    items_per_category: int = Field(
        default=4, ge=1, le=20,
        description="Items per category in carousel mode",
    )
    category: Optional[str] = Field(
        None,
        description="Target category for feed mode (e.g. 'tops', 'outerwear'). "
        "Omit for carousel mode.",
    )
    offset: int = Field(default=0, ge=0, description="Skip first N items (feed mode)")
    limit: Optional[int] = Field(
        None, ge=1, le=100,
        description="Max items to return (feed mode)",
    )


class CompleteFitFromInspirationResponse(BaseModel):
    """
    Response for complete-the-fit from an inspiration image.

    ``matched_product`` is the closest real product to the inspiration.
    ``outfit`` is the full outfit-builder result keyed by category.
    """

    matched_product: Dict[str, Any] = Field(
        description="The closest real product used as the outfit seed",
    )
    outfit: Dict[str, Any] = Field(
        description="Full outfit builder response (recommendations, scoring_info, etc.)",
    )
