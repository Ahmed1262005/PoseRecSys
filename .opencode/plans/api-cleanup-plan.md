# API Cleanup Implementation Plan

## Goal
1. Make `/feed/action` instant (background Supabase writes)
2. Auto-persist seen_ids on feed requests (background)
3. Deprecate `/session/sync` and `/feed/endless`
4. Update API docs

## Changes

### File 1: `src/recs/api_endpoints.py`

#### 1a. Add BackgroundTasks import (line 23)
```python
# Change:
from fastapi import APIRouter, HTTPException, Query, Depends
# To:
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Depends
```

#### 1b. Add background helper function (after line ~165)
```python
def _bg_persist_interaction(
    user_id: str,
    session_id: str,
    product_id: str,
    action: str,
    source: str,
    position: Optional[int],
):
    """Background task: persist interaction to Supabase (non-blocking)."""
    try:
        service = get_service()
        service.record_user_interaction(
            anon_id=None,
            user_id=user_id,
            session_id=session_id,
            product_id=product_id,
            action=action,
            source=source,
            position=position,
        )
    except Exception as e:
        print(f"[BG] Failed to persist interaction: {e}")


def _bg_persist_seen_ids(
    user_id: str,
    session_id: str,
    seen_ids: List[str],
):
    """Background task: persist seen_ids to Supabase for ML training (non-blocking)."""
    if not seen_ids:
        return
    try:
        service = get_service()
        service.sync_session_seen_ids(
            anon_id=None,
            user_id=user_id,
            session_id=session_id,
            seen_ids=seen_ids,
        )
    except Exception as e:
        print(f"[BG] Failed to persist seen_ids: {e}")
```

#### 1c. Rewrite `/feed/action` endpoint (~line 1858)
The key change: session scoring first (instant), then enqueue Supabase write as background task.

```python
@v2_router.post(
    "/feed/action",
    response_model=RecordActionResponse,
    summary="Record user interaction",
    description="""
    Record an explicit user interaction with a product.

    **Response is instant** (~1ms) — session scoring updates in-memory,
    Supabase persistence happens in background (non-blocking).

    **Valid Actions:**
    - `click`: User tapped to view product details (signal: 0.5)
    - `hover`: User swiped through photo gallery (signal: 0.1)
    - `add_to_wishlist`: User saved/liked the item (signal: 2.0, strong positive)
    - `add_to_cart`: User added to cart (signal: 0.8)
    - `purchase`: User completed purchase (signal: 1.0)
    - `skip`: User explicitly dismissed (signal: -0.5)

    **Session Scoring:**
    Send `brand`, `item_type`, and `attributes` to enable multi-dimensional
    preference learning. The next feed request will reflect these preferences.

    Call this endpoint immediately when a user takes an action.
    """
)
async def record_action(
    request: RecordActionRequest,
    background_tasks: BackgroundTasks,
    user: SupabaseUser = Depends(require_auth),
):
    """Record a user interaction (instant response, background persistence)."""

    # Validate action type
    if request.action not in VALID_INTERACTION_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{request.action}'. Must be one of: {VALID_INTERACTION_ACTIONS}"
        )

    # Step 1: Update session scores FIRST (instant, in-memory/Redis ~1ms)
    try:
        pipeline = get_pipeline()
        pipeline.update_session_scores_from_action(
            session_id=request.session_id,
            action=request.action,
            product_id=request.product_id,
            brand=request.brand or "",
            item_type=request.item_type or "",
            attributes=request.attributes or {},
            source=request.source or "feed",
        )
    except Exception:
        pass  # Session scoring failure is non-fatal

    # Step 2: Enqueue Supabase write as background task (non-blocking)
    background_tasks.add_task(
        _bg_persist_interaction,
        user_id=user.id,
        session_id=request.session_id,
        product_id=request.product_id,
        action=request.action,
        source=request.source or "feed",
        position=request.position,
    )

    # Step 3: Return immediately (~1ms total)
    return RecordActionResponse(status="success", interaction_id=None)
```

#### 1d. Add BackgroundTasks to feed endpoint + auto-persist seen_ids (~line 1348)
Add `background_tasks: BackgroundTasks` param to `get_pipeline_feed()`, and after getting response, enqueue seen_ids persistence.

```python
async def get_pipeline_feed(
    background_tasks: BackgroundTasks,  # <-- ADD THIS
    user: SupabaseUser = Depends(require_auth),
    session_id: Optional[str] = Query(None, ...),
    # ... rest of params unchanged ...
):
    # ... existing code ...

    response = pipeline.get_feed_keyset(...)

    # Auto-persist seen_ids to Supabase in background
    shown_ids = [item["product_id"] for item in response.get("results", [])]
    if shown_ids:
        background_tasks.add_task(
            _bg_persist_seen_ids,
            user_id=user.id,
            session_id=response.get("session_id", session_id or ""),
            seen_ids=shown_ids,
        )

    return response
```

Do the same for `get_sale_items()` and `get_new_arrivals()`.

#### 1e. Deprecate `/session/sync` endpoint
Add deprecation notice to the endpoint:

```python
@v2_router.post(
    "/session/sync",
    response_model=SyncSessionResponse,
    summary="[DEPRECATED] Sync session seen_ids",
    description="""
    **DEPRECATED**: Seen_ids are now auto-persisted by the server on each feed request.
    This endpoint is kept for backward compatibility but will be removed in a future version.
    
    Previously used to sync seen_ids for ML training data.
    """,
    deprecated=True,  # <-- FastAPI shows this in OpenAPI/Swagger UI
)
```

#### 1f. Deprecate `/feed/endless` endpoint
```python
@v2_router.get(
    "/feed/endless",
    summary="[DEPRECATED] Use /feed instead",
    description="""
    **DEPRECATED**: Use GET /api/recs/v2/feed instead, which provides the same
    functionality with keyset cursor pagination, 23+ filters, and session scoring.
    
    This endpoint is kept for backward compatibility.
    """,
    deprecated=True,
)
```

### File 2: `docs/API_COMPLETE_REFERENCE.md`
Update to reflect:
- `/feed/action` response is now instant
- `/session/sync` marked deprecated
- `/feed/endless` marked deprecated
- Data flow diagrams showing background task architecture
- Remove Tinder/women endpoints from the reference

## Test Impact
- Existing tests should still pass (API contract unchanged, just timing)
- May need to verify BackgroundTasks don't interfere with test assertions
- `interaction_id` in response will now always be `None` (was previously the DB UUID)
  - If frontend needs the interaction_id, we could change this, but it's unlikely needed

## Verification
```bash
PYTHONPATH=src python -m pytest tests/unit/ -q --no-header
```
