"""
FastAPI routes for the POSE Canvas module.

All endpoints live under ``/api/canvas`` and require JWT authentication.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from supabase import Client

from canvas.models import (
    CompleteFitFromInspirationRequest,
    CompleteFitFromInspirationResponse,
    DeleteInspirationResponse,
    InspirationListResponse,
    InspirationResponse,
    PinterestSyncRequest,
    SimilarProductsResponse,
    StyleElementsResponse,
    UrlInspirationRequest,
)
from canvas.service import get_canvas_service
from config.database import get_db
from core.auth import SupabaseUser, require_auth
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/canvas", tags=["Canvas"])


# ---------------------------------------------------------------------------
# Inspirations — list
# ---------------------------------------------------------------------------

@router.get(
    "/inspirations",
    response_model=InspirationListResponse,
    summary="List inspirations",
    description="Return all inspiration images for the authenticated user.",
)
def list_inspirations(
    user: SupabaseUser = Depends(require_auth),
    db: Client = Depends(get_db),
) -> JSONResponse:
    svc = get_canvas_service()
    items = svc.list_inspirations(user.id, db)
    body = InspirationListResponse(inspirations=items, count=len(items))
    return JSONResponse(
        content=body.model_dump(mode="json"),
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# Inspirations — upload
# ---------------------------------------------------------------------------

@router.post(
    "/inspirations/upload",
    response_model=InspirationResponse,
    summary="Upload inspiration image",
    description=(
        "Upload a local image file, encode it with FashionCLIP, classify its "
        "style, and add it to the user's canvas."
    ),
)
async def upload_inspiration(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    user: SupabaseUser = Depends(require_auth),
    db: Client = Depends(get_db),
) -> InspirationResponse:
    _validate_image_upload(file)
    contents = await file.read()
    svc = get_canvas_service()
    try:
        return svc.add_inspiration_upload(
            user_id=user.id,
            image_bytes=contents,
            filename=file.filename or "upload.jpg",
            supabase=db,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


# ---------------------------------------------------------------------------
# Inspirations — URL
# ---------------------------------------------------------------------------

@router.post(
    "/inspirations/url",
    response_model=InspirationResponse,
    summary="Add inspiration from URL",
    description="Fetch an image from a public URL, encode, classify, and add.",
)
def add_inspiration_url(
    body: UrlInspirationRequest,
    user: SupabaseUser = Depends(require_auth),
    db: Client = Depends(get_db),
) -> InspirationResponse:
    svc = get_canvas_service()
    try:
        return svc.add_inspiration_url(
            user_id=user.id,
            url=body.url,
            title=body.title,
            supabase=db,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to add URL inspiration", url=body.url)
        raise HTTPException(status_code=502, detail=f"Could not fetch image: {exc}")


# ---------------------------------------------------------------------------
# Inspirations — Pinterest sync
# ---------------------------------------------------------------------------

@router.post(
    "/inspirations/pinterest",
    response_model=List[InspirationResponse],
    summary="Sync Pinterest pins as inspirations",
    description=(
        "Import Pinterest pins as inspiration images. Provide specific pin IDs "
        "or let the server use the user's saved board/section selection."
    ),
)
def sync_pinterest(
    body: PinterestSyncRequest,
    user: SupabaseUser = Depends(require_auth),
    db: Client = Depends(get_db),
) -> List[InspirationResponse]:
    svc = get_canvas_service()
    try:
        return svc.add_inspiration_pinterest(
            user_id=user.id,
            supabase=db,
            pin_ids=body.pin_ids,
            max_pins=body.max_pins,
        )
    except Exception as exc:
        logger.exception("Pinterest sync failed")
        raise HTTPException(status_code=502, detail=f"Pinterest sync error: {exc}")


# ---------------------------------------------------------------------------
# Inspirations — delete
# ---------------------------------------------------------------------------

@router.delete(
    "/inspirations/{inspiration_id}",
    response_model=DeleteInspirationResponse,
    summary="Remove an inspiration",
    description=(
        "Delete an inspiration image and recompute the taste vector in the "
        "background.  Idempotent: returns 200 even if the item was already "
        "deleted (the desired state is achieved).  The response includes the "
        "authoritative ``inspirations`` list so the frontend can replace its "
        "cache without a separate GET."
    ),
)
def delete_inspiration(
    inspiration_id: str,
    background_tasks: BackgroundTasks,
    user: SupabaseUser = Depends(require_auth),
    db: Client = Depends(get_db),
) -> JSONResponse:
    svc = get_canvas_service()
    result = svc.remove_inspiration(user.id, inspiration_id, db)

    # Schedule taste-vector recomputation in the background so the HTTP
    # response returns immediately (~200ms instead of ~14s).
    background_tasks.add_task(svc.recompute_taste_vector, user.id, db)

    return JSONResponse(
        content=result.model_dump(mode="json"),
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# Style elements
# ---------------------------------------------------------------------------

@router.get(
    "/style-elements",
    response_model=StyleElementsResponse,
    summary="Get extracted style elements",
    description=(
        "Aggregate style attributes from all inspirations and return them "
        "mapped to the feed endpoint's ``include_*`` query parameter names. "
        "The frontend can pass ``suggested_filters`` directly to "
        "``GET /api/recs/v2/feed``."
    ),
)
def get_style_elements(
    user: SupabaseUser = Depends(require_auth),
    db: Client = Depends(get_db),
) -> StyleElementsResponse:
    svc = get_canvas_service()
    return svc.get_style_elements(user.id, db)


# ---------------------------------------------------------------------------
# Similar products
# ---------------------------------------------------------------------------

@router.get(
    "/inspirations/{inspiration_id}/similar",
    response_model=SimilarProductsResponse,
    summary="Find similar products",
    description=(
        "Find products visually similar to an inspiration image using "
        "FashionCLIP embedding similarity.  Results are deduplicated "
        "(product ID, image hash, sister-brand, size-variant, fuzzy name) "
        "and brand-diversity-capped."
    ),
)
def similar_products(
    inspiration_id: str,
    count: int = 12,
    user: SupabaseUser = Depends(require_auth),
    db: Client = Depends(get_db),
) -> SimilarProductsResponse:
    svc = get_canvas_service()
    products = svc.find_similar_products(
        inspiration_id=inspiration_id,
        user_id=user.id,
        supabase=db,
        count=min(count, 30),
    )
    if not products:
        raise HTTPException(
            status_code=404,
            detail="Inspiration not found or no similar products",
        )
    return SimilarProductsResponse(
        products=products,
        count=len(products),
        inspiration_id=inspiration_id,
    )


# ---------------------------------------------------------------------------
# Complete-the-fit from an inspiration
# ---------------------------------------------------------------------------

@router.post(
    "/inspirations/{inspiration_id}/complete-fit",
    response_model=CompleteFitFromInspirationResponse,
    summary="Complete the fit from an inspiration",
    description=(
        "Find the closest real product to the inspiration image and run the "
        "outfit builder (TATTOO scoring) to suggest complementary pieces.\n\n"
        "The response includes the matched product and the full outfit "
        "recommendation payload."
    ),
)
def complete_fit_from_inspiration(
    inspiration_id: str,
    body: CompleteFitFromInspirationRequest = CompleteFitFromInspirationRequest(),
    user: SupabaseUser = Depends(require_auth),
    db: Client = Depends(get_db),
) -> CompleteFitFromInspirationResponse:
    svc = get_canvas_service()

    # 1. Find closest real product
    matched = svc.find_closest_product(inspiration_id, user.id, db)
    if not matched:
        raise HTTPException(
            status_code=404,
            detail="Inspiration not found or no matching products",
        )

    product_id = str(matched["product_id"])

    # 2. Build outfit using the outfit engine
    try:
        from services.outfit_engine import get_outfit_engine
        engine = get_outfit_engine()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Outfit engine not available: {exc}",
        )

    outfit_result = engine.build_outfit(
        product_id=product_id,
        items_per_category=body.items_per_category,
        target_category=body.category,
        offset=body.offset,
        limit=body.limit,
        user_id=user.id,
    )

    if outfit_result.get("error") and outfit_result.get("source_product") is None:
        detail = outfit_result["error"]
        if "not found" in detail.lower():
            raise HTTPException(status_code=404, detail=detail)
        raise HTTPException(status_code=500, detail=detail)

    return CompleteFitFromInspirationResponse(
        matched_product=matched,
        outfit=outfit_result,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_MAX_UPLOAD_MB = 10


def _validate_image_upload(file: UploadFile) -> None:
    if file.content_type and file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
            f"Allowed: {', '.join(sorted(_ALLOWED_TYPES))}",
        )
    if file.size and file.size > _MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {_MAX_UPLOAD_MB} MB",
        )
