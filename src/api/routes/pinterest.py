"""Pinterest OAuth + style sync endpoints."""

import secrets
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from config.database import get_supabase_client
from config.settings import get_settings
from core.auth import SupabaseUser, require_auth
from core.logging import get_logger
from integrations.oauth_state import OAuthStateError, sign_state, verify_state
from integrations.pinterest_client import PinterestApiError, PinterestClient
from integrations.pinterest_signals import score_collection_intent
from integrations.pinterest_style import extract_pin_image_urls, extract_pin_preview, get_pinterest_style_extractor

logger = get_logger(__name__)

router = APIRouter(prefix="/api/integrations/pinterest", tags=["Integrations"])


class PinterestAuthorizeResponse(BaseModel):
    auth_url: str
    state: str
    scopes: List[str]
    redirect_uri: str


class PinterestStatusResponse(BaseModel):
    connected: bool
    scope: Optional[str] = None
    expires_at: Optional[str] = None
    last_sync_at: Optional[str] = None
    last_sync_count: Optional[int] = None
    token_source: Optional[str] = None


class PinterestTokenRequest(BaseModel):
    access_token: str = Field(..., min_length=10, description="Pinterest access token")
    refresh_token: Optional[str] = Field(None, description="Optional refresh token")
    expires_in: Optional[int] = Field(None, ge=60, description="Token lifetime in seconds")
    refresh_token_expires_in: Optional[int] = Field(None, ge=60, description="Refresh token lifetime in seconds")
    scope: Optional[str] = Field(None, description="Space or comma separated scopes")
    token_type: Optional[str] = Field(None, description="Token type, usually 'bearer'")


class PinterestTokenHealthResponse(BaseModel):
    status: str
    expires_at: Optional[str] = None
    seconds_until_expiry: Optional[int] = None
    has_refresh_token: bool
    refresh_available: bool
    should_reconnect: bool
    checked_at: str
    token_source: Optional[str] = None


class PinterestBoardResponse(BaseModel):
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    privacy: Optional[str] = None
    pin_count: Optional[int] = None
    cover_image_url: Optional[str] = None
    shopping_intent_score: float = 0.0
    shopping_signals: List[str] = Field(default_factory=list)


class PinterestSectionResponse(BaseModel):
    id: str
    name: Optional[str] = None
    shopping_intent_score: float = 0.0
    shopping_signals: List[str] = Field(default_factory=list)


class PinterestPinPreview(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    link: Optional[str] = None
    board_id: Optional[str] = None
    board_section_id: Optional[str] = None
    image_url: Optional[str] = None
    merchant_domain: Optional[str] = None
    shopping_intent_score: float = 0.0
    shopping_signals: List[str] = Field(default_factory=list)


class PinterestPreviewResponse(BaseModel):
    board_id: Optional[str] = None
    section_id: Optional[str] = None
    pins: List[PinterestPinPreview]


class PinterestSectionSelection(BaseModel):
    board_id: str
    section_id: str


class PinterestSelectionRequest(BaseModel):
    boards: List[str] = Field(default_factory=list, description="Board IDs to include")
    sections: List[PinterestSectionSelection] = Field(
        default_factory=list,
        description="Board sections to include",
    )
    auto_include_new_boards: bool = Field(
        default=True,
        description="Automatically include newly created boards on sync",
    )


class PinterestSelectionResponse(BaseModel):
    boards: List[str]
    sections: List[PinterestSectionSelection]
    auto_include_new_boards: bool = True


class PinterestSyncRequest(BaseModel):
    max_pins: Optional[int] = Field(None, ge=1, le=500, description="Max pins to fetch")
    max_images: Optional[int] = Field(None, ge=1, le=200, description="Max images to embed")


class PinterestSyncResponse(BaseModel):
    status: str
    pins_fetched: int
    image_urls_found: int
    images_used: int
    images_failed: int
    taste_vector_saved: bool


@router.get(
    "/authorize",
    response_model=PinterestAuthorizeResponse,
    summary="Start Pinterest OAuth flow",
)
def pinterest_authorize(user: SupabaseUser = Depends(require_auth)) -> PinterestAuthorizeResponse:
    settings = get_settings()
    client = PinterestClient()

    if not client.is_configured():
        raise HTTPException(status_code=500, detail="Pinterest integration is not configured")

    secret = settings.pinterest_oauth_state_secret or settings.supabase_jwt_secret
    payload = {
        "uid": user.id,
        "ts": time.time(),
        "nonce": secrets.token_urlsafe(12),
    }
    state = sign_state(payload, secret)
    auth_url = client.build_authorization_url(state)

    return PinterestAuthorizeResponse(
        auth_url=auth_url,
        state=state,
        scopes=settings.pinterest_scopes,
        redirect_uri=settings.pinterest_redirect_uri,
    )


@router.get(
    "/callback",
    summary="Pinterest OAuth callback",
)
def pinterest_callback(
    code: Optional[str] = Query(None, description="Authorization code"),
    state: Optional[str] = Query(None, description="OAuth state"),
    error: Optional[str] = Query(None, description="Pinterest error"),
) -> Dict[str, Any]:
    settings = get_settings()
    client = PinterestClient()

    if error:
        raise HTTPException(status_code=400, detail=f"Pinterest authorization failed: {error}")
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing OAuth code or state")

    secret = settings.pinterest_oauth_state_secret or settings.supabase_jwt_secret
    try:
        payload = verify_state(state, secret, settings.pinterest_oauth_state_ttl_seconds)
    except OAuthStateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    user_id = payload.get("uid")
    if not user_id:
        raise HTTPException(status_code=400, detail="Invalid OAuth state payload")

    if not client.is_configured():
        raise HTTPException(status_code=500, detail="Pinterest integration is not configured")

    try:
        token_data = client.exchange_code_for_token(code)
        client.upsert_tokens(user_id, token_data)
    except PinterestApiError as exc:
        logger.error("Pinterest token exchange failed", error=str(exc))
        raise HTTPException(status_code=502, detail="Pinterest token exchange failed") from exc

    return {"status": "connected", "user_id": user_id}


@router.get(
    "/status",
    response_model=PinterestStatusResponse,
    summary="Check Pinterest connection status",
)
def pinterest_status(user: SupabaseUser = Depends(require_auth)) -> PinterestStatusResponse:
    client = PinterestClient()
    client.ensure_env_token(user.id)
    row = client.get_token_row(user.id)
    if not row:
        return PinterestStatusResponse(connected=False)

    metadata = client.get_metadata(user.id)
    return PinterestStatusResponse(
        connected=True,
        scope=row.get("scope"),
        expires_at=row.get("expires_at"),
        last_sync_at=row.get("last_sync_at"),
        last_sync_count=row.get("last_sync_count"),
        token_source=metadata.get("token_source"),
    )


@router.get(
    "/token/health",
    response_model=PinterestTokenHealthResponse,
    summary="Check Pinterest token expiry and reconnect status",
)
def pinterest_token_health(user: SupabaseUser = Depends(require_auth)) -> PinterestTokenHealthResponse:
    settings = get_settings()
    client = PinterestClient()
    health = client.get_token_health(user.id, settings.pinterest_token_expiry_grace_seconds)
    return PinterestTokenHealthResponse(**health)


@router.post(
    "/token",
    summary="Connect Pinterest using a pre-generated access token",
)
def pinterest_connect_token(
    request: PinterestTokenRequest,
    user: SupabaseUser = Depends(require_auth),
) -> Dict[str, str]:
    client = PinterestClient()
    token_data = {
        "access_token": request.access_token,
        "refresh_token": request.refresh_token,
        "expires_in": request.expires_in,
        "refresh_token_expires_in": request.refresh_token_expires_in,
        "scope": request.scope,
        "token_type": request.token_type,
    }
    client.upsert_tokens(user.id, token_data)
    client.set_token_source(user.id, "manual")
    return {"status": "connected"}


@router.get(
    "/boards",
    response_model=List[PinterestBoardResponse],
    summary="List Pinterest boards for preview",
)
def pinterest_boards(
    limit: int = Query(50, ge=1, le=200),
    user: SupabaseUser = Depends(require_auth),
) -> List[PinterestBoardResponse]:
    client = PinterestClient()
    client.ensure_env_token(user.id)
    try:
        access_token = client.get_access_token(user.id)
    except PinterestApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    boards: List[PinterestBoardResponse] = []
    bookmark: Optional[str] = None
    page_size = min(100, limit)

    while len(boards) < limit:
        response = client.list_boards(access_token, page_size=page_size, bookmark=bookmark)
        items = response.get("items") or []
        for item in items:
            media = item.get("media") or {}
            image_cover_url = None
            if isinstance(media, dict):
                image_cover_url = media.get("image_cover_url") or media.get("image_cover") or None
            score, signals = score_collection_intent(item.get("name") or "")
            boards.append(PinterestBoardResponse(
                id=item.get("id"),
                name=item.get("name"),
                description=item.get("description"),
                privacy=item.get("privacy"),
                pin_count=item.get("pin_count"),
                cover_image_url=image_cover_url,
                shopping_intent_score=score,
                shopping_signals=signals,
            ))
            if len(boards) >= limit:
                break
        bookmark = response.get("bookmark")
        if not bookmark or not items:
            break

    return boards[:limit]


@router.get(
    "/boards/{board_id}/sections",
    response_model=List[PinterestSectionResponse],
    summary="List sections for a board",
)
def pinterest_board_sections(
    board_id: str,
    limit: int = Query(50, ge=1, le=200),
    user: SupabaseUser = Depends(require_auth),
) -> List[PinterestSectionResponse]:
    client = PinterestClient()
    client.ensure_env_token(user.id)
    try:
        access_token = client.get_access_token(user.id)
    except PinterestApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    sections: List[PinterestSectionResponse] = []
    bookmark: Optional[str] = None
    page_size = min(100, limit)

    while len(sections) < limit:
        response = client.list_board_sections(access_token, board_id=board_id, page_size=page_size, bookmark=bookmark)
        items = response.get("items") or []
        for item in items:
            score, signals = score_collection_intent(item.get("name") or "")
            sections.append(PinterestSectionResponse(
                id=item.get("id"),
                name=item.get("name"),
                shopping_intent_score=score,
                shopping_signals=signals,
            ))
            if len(sections) >= limit:
                break
        bookmark = response.get("bookmark")
        if not bookmark or not items:
            break

    return sections[:limit]


@router.get(
    "/preview",
    response_model=PinterestPreviewResponse,
    summary="Preview pins for a board or section",
)
def pinterest_preview(
    board_id: Optional[str] = Query(None, description="Board ID"),
    section_id: Optional[str] = Query(None, description="Section ID"),
    limit: int = Query(24, ge=1, le=100),
    user: SupabaseUser = Depends(require_auth),
) -> PinterestPreviewResponse:
    if section_id and not board_id:
        raise HTTPException(status_code=400, detail="board_id is required when section_id is provided")

    client = PinterestClient()
    client.ensure_env_token(user.id)
    try:
        access_token = client.get_access_token(user.id)
    except PinterestApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if section_id:
        response = client.list_section_pins(access_token, board_id=board_id, section_id=section_id, page_size=limit)
    elif board_id:
        response = client.list_board_pins(access_token, board_id=board_id, page_size=limit)
    else:
        response = client.list_pins(access_token, page_size=limit)

    pins = [PinterestPinPreview(**extract_pin_preview(pin)) for pin in (response.get("items") or [])]
    return PinterestPreviewResponse(board_id=board_id, section_id=section_id, pins=pins)


@router.get(
    "/selection",
    response_model=PinterestSelectionResponse,
    summary="Get saved Pinterest selection",
)
def pinterest_get_selection(user: SupabaseUser = Depends(require_auth)) -> PinterestSelectionResponse:
    client = PinterestClient()
    client.ensure_env_token(user.id)
    if not client.get_token_row(user.id):
        raise HTTPException(status_code=409, detail="Pinterest account not connected")
    selection = client.get_selection(user.id)
    boards = selection.get("boards") or []
    sections = selection.get("sections") or []
    auto_include = selection.get("auto_include_new_boards", True)
    return PinterestSelectionResponse(boards=boards, sections=sections, auto_include_new_boards=auto_include)


@router.post(
    "/selection",
    response_model=PinterestSelectionResponse,
    summary="Save Pinterest selection",
)
def pinterest_save_selection(
    request: PinterestSelectionRequest,
    user: SupabaseUser = Depends(require_auth),
) -> PinterestSelectionResponse:
    client = PinterestClient()
    client.ensure_env_token(user.id)
    if not client.get_token_row(user.id):
        raise HTTPException(status_code=409, detail="Pinterest account not connected")
    selection = {
        "boards": request.boards,
        "sections": [section.model_dump() for section in request.sections],
        "auto_include_new_boards": request.auto_include_new_boards,
        "updated_at": time.time(),
    }
    client.save_selection(user.id, selection)
    return PinterestSelectionResponse(
        boards=request.boards,
        sections=request.sections,
        auto_include_new_boards=request.auto_include_new_boards,
    )


@router.post(
    "/selection/refresh",
    response_model=PinterestSelectionResponse,
    summary="Refresh saved selection from Pinterest",
)
def pinterest_refresh_selection(user: SupabaseUser = Depends(require_auth)) -> PinterestSelectionResponse:
    client = PinterestClient()
    client.ensure_env_token(user.id)
    if not client.get_token_row(user.id):
        raise HTTPException(status_code=409, detail="Pinterest account not connected")
    selection = client.sync_selection(user.id)
    boards = selection.get("boards") or []
    sections_raw = selection.get("sections") or []
    sections = [PinterestSectionSelection(**section) for section in sections_raw if isinstance(section, dict)]
    auto_include = selection.get("auto_include_new_boards", True)
    return PinterestSelectionResponse(
        boards=boards,
        sections=sections,
        auto_include_new_boards=auto_include,
    )


@router.post(
    "/sync",
    response_model=PinterestSyncResponse,
    summary="Sync Pinterest pins and compute taste vector",
)
def pinterest_sync(
    request: PinterestSyncRequest,
    user: SupabaseUser = Depends(require_auth),
) -> PinterestSyncResponse:
    settings = get_settings()
    client = PinterestClient()
    client.ensure_env_token(user.id)

    max_pins = request.max_pins or settings.pinterest_default_max_pins
    max_images = request.max_images or settings.pinterest_default_max_images

    try:
        selection = client.sync_selection(user.id)
        pins = client.fetch_pins(user.id, max_pins=max_pins, selection=selection)
    except PinterestApiError as exc:
        client.update_sync_status(user.id, count=0, error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    image_urls = extract_pin_image_urls(pins)
    extractor = get_pinterest_style_extractor()
    taste_vector, stats = extractor.compute_taste_vector(
        image_urls=image_urls,
        max_images=max_images,
        timeout_seconds=settings.pinterest_request_timeout_seconds,
    )

    if not taste_vector:
        client.update_sync_status(user.id, count=stats.get("images_used", 0), error="no_images")
        raise HTTPException(status_code=422, detail="No usable Pinterest images found")

    if len(taste_vector) != 512:
        client.update_sync_status(user.id, count=stats.get("images_used", 0), error="invalid_vector")
        raise HTTPException(status_code=500, detail="Computed taste vector has invalid length")

    try:
        supabase = get_supabase_client()
        supabase.rpc(
            "update_user_taste_vector",
            {
                "p_user_id": user.id,
                "p_taste_vector": taste_vector,
            },
        ).execute()
    except Exception as exc:
        client.update_sync_status(user.id, count=stats.get("images_used", 0), error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to save taste vector") from exc

    client.update_sync_status(user.id, count=stats.get("images_used", 0), error=None)

    return PinterestSyncResponse(
        status="ok",
        pins_fetched=len(pins),
        image_urls_found=len(image_urls),
        images_used=stats.get("images_used", 0),
        images_failed=stats.get("images_failed", 0),
        taste_vector_saved=True,
    )


@router.delete(
    "/disconnect",
    summary="Disconnect Pinterest account",
)
def pinterest_disconnect(user: SupabaseUser = Depends(require_auth)) -> Dict[str, str]:
    client = PinterestClient()
    client.delete_tokens(user.id)
    return {"status": "disconnected"}
