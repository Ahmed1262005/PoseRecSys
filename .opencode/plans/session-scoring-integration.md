# Session Scoring Integration Plan

## Overview

Three tasks to fully integrate session scoring across the system:
1. Wire search signals into session scoring
2. Redis backend for session scores (required, no fallback)
3. Update search reranker with live session scores

---

## Task 1: Wire Search Signals into Session Scoring

**File: `src/api/routes/search.py`**

### Changes

**1a. Add helper function `_extract_search_filters()`** (after `_load_user_profile`):

```python
def _extract_search_filters(request: HybridSearchRequest) -> Dict[str, Any]:
    """Extract active filters from a search request for session scoring."""
    filters = {}
    # Map request fields to scoring filter keys
    filter_fields = [
        ("brands", "brands"),
        ("colors", "colors"),
        ("color_family", "color_family"),
        ("patterns", "patterns"),
        ("materials", "materials"),
        ("occasions", "occasions"),
        ("seasons", "seasons"),
        ("formality", "formality"),
        ("fit_type", "fit_types"),
        ("neckline", "necklines"),
        ("sleeve_type", "sleeve_types"),
        ("length", "lengths"),
        ("categories", "categories"),
        ("category_l1", "category_l1"),
        ("category_l2", "category_l2"),
        ("style_tags", "styles"),
        ("article_type", "article_type"),
    ]
    for req_field, score_key in filter_fields:
        val = getattr(request, req_field, None)
        if val:
            filters[score_key] = val
    return filters
```

**1b. Add search signal propagation** in `hybrid_search()` endpoint, after `service.search()` returns:

```python
    # Propagate search signals into session scoring (best-effort)
    if request.session_id:
        try:
            from recs.api_endpoints import get_pipeline
            pipeline = get_pipeline()
            filters = _extract_search_filters(request)
            pipeline.update_session_scores_from_search(
                session_id=request.session_id,
                query=request.query,
                filters=filters,
            )
        except Exception as e:
            logger.debug(
                "Search signal propagation failed (non-blocking)",
                session_id=request.session_id,
                error=str(e),
            )
```

**Why lazy import?** The `get_pipeline()` import is inside the try block to avoid circular imports and because the pipeline is heavy (loads SASRec model). If the recs module isn't available, search still works.

---

## Task 2: Redis Integration for Session Scores

### Task 2a: New backend class in `src/recs/session_state.py`

Add a `ScoringRedisBackend` class after the existing `RedisSessionBackend` class (~line 340):

```python
class ScoringRedisBackend:
    """
    Redis backend for SessionScores persistence.

    Key pattern: scoring:{session_id}
    Storage: JSON blob via SessionScores.to_json() / from_json()
    TTL: 24 hours (refreshed on access)
    """

    def __init__(self, redis_url: str, ttl: int = 86400):
        import redis as redis_lib
        self._redis = redis_lib.from_url(redis_url, decode_responses=True)
        self._ttl = ttl
        # Verify connection
        self._redis.ping()

    def _key(self, session_id: str) -> str:
        return f"scoring:{session_id}"

    def get_scores(self, session_id: str) -> Optional["SessionScores"]:
        """Load SessionScores from Redis. Returns None if not found."""
        from recs.session_scoring import SessionScores
        data = self._redis.get(self._key(session_id))
        if data is None:
            return None
        # Refresh TTL on access
        self._redis.expire(self._key(session_id), self._ttl)
        return SessionScores.from_json(data)

    def save_scores(self, session_id: str, scores: "SessionScores") -> None:
        """Persist SessionScores to Redis with TTL."""
        self._redis.setex(
            self._key(session_id),
            self._ttl,
            scores.to_json(),
        )

    def delete_scores(self, session_id: str) -> None:
        """Delete SessionScores from Redis."""
        self._redis.delete(self._key(session_id))

    def get_stats(self) -> Dict[str, Any]:
        """Get scoring backend stats."""
        count = 0
        for _ in self._redis.scan_iter(match="scoring:*", count=100):
            count += 1
        return {"active_scoring_sessions": count, "backend": "redis"}
```

### Task 2b: Update `src/recs/pipeline.py` to use Redis backend

**Replace `_session_scores_cache` class variable** with a proper backend + L1 cache:

In `__init__`:
```python
# Session scoring persistence
self._scoring_backend = self._init_scoring_backend()
self._scores_l1_cache: Dict[str, SessionScores] = {}  # Small L1 cache
self._SCORES_L1_MAX = 200  # Max L1 cache entries
```

Add `_init_scoring_backend()`:
```python
def _init_scoring_backend(self):
    """Initialize Redis backend for session scores (required)."""
    import os
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        raise RuntimeError(
            "REDIS_URL environment variable is required for session scoring. "
            "Set REDIS_URL to a valid Redis connection string."
        )
    from recs.session_state import ScoringRedisBackend
    backend = ScoringRedisBackend(redis_url=redis_url)
    logger.info("Session scoring Redis backend initialized", redis_url=redis_url[:20] + "...")
    return backend
```

**Update `_get_or_create_session_scores()`:**
```python
def _get_or_create_session_scores(self, session_id: str, user_state) -> SessionScores:
    # L1 cache check
    if session_id in self._scores_l1_cache:
        return self._scores_l1_cache[session_id]

    # Redis check
    scores = self._scoring_backend.get_scores(session_id)
    if scores is not None:
        self._scores_l1_cache[session_id] = scores
        return scores

    # Cold start - create new
    scores = self.scoring_engine.initialize_from_onboarding(
        preferred_brands=user_state.preferred_brands or [],
        profile=user_state.profile or {},
    )
    self._scoring_backend.save_scores(session_id, scores)
    self._scores_l1_cache[session_id] = scores

    # Evict L1 if too large
    if len(self._scores_l1_cache) > self._SCORES_L1_MAX:
        # Remove oldest half by last_updated
        sorted_keys = sorted(
            self._scores_l1_cache.keys(),
            key=lambda k: self._scores_l1_cache[k].last_updated,
        )
        for k in sorted_keys[:len(sorted_keys) // 2]:
            del self._scores_l1_cache[k]

    return scores
```

**Update `update_session_scores_from_action()`:**
```python
def update_session_scores_from_action(self, session_id, action, product_id, brand, item_type, attributes, source):
    # Load from L1 cache or Redis
    scores = self._scores_l1_cache.get(session_id)
    if scores is None:
        scores = self._scoring_backend.get_scores(session_id)
    if scores is None:
        scores = SessionScores()

    self.scoring_engine.process_action(
        scores, action=action, product_id=product_id,
        brand=brand, item_type=item_type,
        attributes=attributes or {}, source=source,
    )

    # Persist back
    self._scoring_backend.save_scores(session_id, scores)
    self._scores_l1_cache[session_id] = scores
```

**Update `update_session_scores_from_search()`:**
```python
def update_session_scores_from_search(self, session_id, query="", filters=None):
    scores = self._scores_l1_cache.get(session_id)
    if scores is None:
        scores = self._scoring_backend.get_scores(session_id)
    if scores is None:
        scores = SessionScores()

    self.scoring_engine.process_search_signal(
        scores, query=query, filters=filters or {},
    )

    # Persist back
    self._scoring_backend.save_scores(session_id, scores)
    self._scores_l1_cache[session_id] = scores
```

**Update `get_session_scores()`:**
```python
def get_session_scores(self, session_id: str) -> Optional[SessionScores]:
    scores = self._scores_l1_cache.get(session_id)
    if scores is None:
        scores = self._scoring_backend.get_scores(session_id)
        if scores is not None:
            self._scores_l1_cache[session_id] = scores
    return scores
```

---

## Task 3: Update Search Reranker with Live Session Scores

### Task 3a: `src/search/reranker.py` — Add session scoring step

**Add new constants:**
```python
# Session-based scoring weights (separate cap from profile scoring)
SESSION_BRAND_BOOST = 0.06
SESSION_TYPE_BOOST = 0.04
SESSION_ATTR_BOOST = 0.03
SESSION_INTENT_BOOST = 0.05
SESSION_SKIP_PENALTY = -0.08
MAX_SESSION_BOOST = 0.12  # Independent cap from MAX_BOOST
```

**Add `session_scores` parameter to `rerank()`:**
```python
def rerank(
    self,
    results: List[dict],
    user_profile: Optional[Dict[str, Any]] = None,
    seen_ids: Optional[Set[str]] = None,
    max_per_brand: int = 4,
    session_scores: Optional[Any] = None,  # SessionScores from recs.session_scoring
) -> List[dict]:
    if not results:
        return results

    # Step 1: Remove already-seen items
    if seen_ids:
        results = [r for r in results if r.get("product_id") not in seen_ids]

    # Step 2: Remove near-duplicates
    results = self._deduplicate(results)

    # Step 3: Apply soft scoring from user profile
    if user_profile:
        results = self._apply_profile_scoring(results, user_profile)

    # Step 3.5: Apply live session scoring (NEW)
    if session_scores:
        results = self._apply_session_scoring(results, session_scores)

    # Step 4: Apply brand diversity cap
    if max_per_brand > 0:
        results = self._apply_brand_diversity(results, max_per_brand)

    return results
```

**Add `_apply_session_scoring()` method:**
```python
def _apply_session_scoring(self, results: List[dict], session_scores) -> List[dict]:
    """Apply live session-learned preferences to search results.

    Uses brand_scores, type_scores, attr_scores, and search_intents
    from the SessionScores object to boost/demote results based on
    real-time user behavior during this session.
    """
    skipped = session_scores.skipped_ids or set()

    for item in results:
        adjustment = 0.0
        product_id = item.get("product_id", "")
        brand = (item.get("brand") or "").lower()
        item_type = (item.get("category") or item.get("article_type") or "").lower()

        # Skip penalty
        if product_id in skipped:
            adjustment += SESSION_SKIP_PENALTY

        # Brand affinity from session
        if brand and brand in session_scores.brand_scores:
            brand_val = session_scores.brand_scores[brand]
            if brand_val > 0:
                adjustment += min(brand_val * 0.1, SESSION_BRAND_BOOST)
            elif brand_val < 0:
                adjustment += max(brand_val * 0.1, -SESSION_BRAND_BOOST)

        # Type affinity from session
        if item_type and item_type in session_scores.type_scores:
            type_val = session_scores.type_scores[item_type]
            if type_val > 0:
                adjustment += min(type_val * 0.08, SESSION_TYPE_BOOST)

        # Attribute match from session
        attr_boost = 0.0
        for attr_key, attr_val in session_scores.attr_scores.items():
            if attr_val > 0:
                # Check if item has this attribute
                prefix, value = attr_key.split(":", 1) if ":" in attr_key else ("", attr_key)
                item_vals = _get_item_attr_values(item, prefix)
                if value.lower() in item_vals:
                    attr_boost += attr_val * 0.05
        adjustment += min(attr_boost, SESSION_ATTR_BOOST)

        # Search intent match
        intent_boost = 0.0
        for intent_key, intent_val in session_scores.search_intents.items():
            if intent_val > 0:
                prefix, value = intent_key.split(":", 1) if ":" in intent_key else ("", intent_key)
                item_vals = _get_item_attr_values(item, prefix)
                if value.lower() in item_vals:
                    intent_boost += intent_val * 0.06
        adjustment += min(intent_boost, SESSION_INTENT_BOOST)

        # Cap total session adjustment
        adjustment = max(-MAX_SESSION_BOOST, min(MAX_SESSION_BOOST, adjustment))

        item["rrf_score"] = item.get("rrf_score", 0) + adjustment
        item["session_adjustment"] = adjustment

    # Re-sort
    results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
    return results
```

**Add helper function** (module-level):
```python
def _get_item_attr_values(item: dict, prefix: str) -> Set[str]:
    """Extract attribute values from a search result item by prefix type."""
    prefix_to_fields = {
        "color": ["primary_color", "colors"],
        "fit": ["fit_type", "fit"],
        "sleeve": ["sleeve_type"],
        "length": ["length"],
        "neckline": ["neckline"],
        "pattern": ["pattern"],
        "occasion": ["occasions"],
        "style": ["style_tags"],
        "material": ["materials"],
        "formality": ["formality"],
        "category": ["category", "article_type"],
    }
    fields = prefix_to_fields.get(prefix, [])
    values = set()
    for field in fields:
        val = item.get(field)
        if isinstance(val, str) and val:
            values.add(val.lower())
        elif isinstance(val, list):
            values.update(v.lower() for v in val if v)
    return values
```

### Task 3b: `src/search/hybrid_search.py` — Accept and forward session_scores

**Update `search()` method signature** (add `session_scores` parameter):
```python
def search(
    self,
    request: HybridSearchRequest,
    user_id: Optional[str] = None,
    user_profile: Optional[Dict[str, Any]] = None,
    seen_ids: Optional[Set[str]] = None,
    session_scores: Optional[Any] = None,  # SessionScores from recs
) -> HybridSearchResponse:
```

**Update the reranker call** inside `search()` to pass session_scores:
```python
merged = self._reranker.rerank(
    results=merged,
    user_profile=user_profile,
    seen_ids=seen_ids,
    session_scores=session_scores,
)
```

### Task 3c: `src/api/routes/search.py` — Load session scores and pass to search

**Update `hybrid_search()` endpoint** to load session scores and pass them:

```python
def hybrid_search(
    request: HybridSearchRequest,
    user: SupabaseUser = Depends(require_auth),
) -> HybridSearchResponse:
    service = get_hybrid_search_service()
    user_profile = _load_user_profile(user.id)

    # Load live session scores if session_id provided
    session_scores = None
    if request.session_id:
        try:
            from recs.api_endpoints import get_pipeline
            pipeline = get_pipeline()
            session_scores = pipeline.get_session_scores(request.session_id)
        except Exception:
            pass  # Session scores are optional for search

    result = service.search(
        request=request,
        user_id=user.id,
        user_profile=user_profile,
        session_scores=session_scores,
    )

    # Propagate search signals into session scoring (best-effort)
    if request.session_id:
        try:
            from recs.api_endpoints import get_pipeline
            pipeline = get_pipeline()
            filters = _extract_search_filters(request)
            pipeline.update_session_scores_from_search(
                session_id=request.session_id,
                query=request.query,
                filters=filters,
            )
        except Exception as e:
            logger.debug(
                "Search signal propagation failed (non-blocking)",
                session_id=request.session_id,
                error=str(e),
            )

    return result
```

---

## Task 4: Tests

### New test file: `tests/unit/test_session_integration.py`

**Test categories:**

1. **Search signal extraction** (3-4 tests):
   - `_extract_search_filters()` with various filter combos
   - Empty request returns empty filters
   - All filter types mapped correctly

2. **ScoringRedisBackend** (5-6 tests, mocked Redis):
   - `save_scores()` + `get_scores()` round-trip
   - `get_scores()` returns None for missing key
   - `delete_scores()` removes key
   - TTL refresh on get
   - `get_stats()` counts keys

3. **Search reranker session scoring** (6-7 tests):
   - Items with preferred brand get boosted
   - Skipped items get penalized
   - Session scoring cap enforced
   - No session scores = no change
   - Results re-sorted after scoring
   - Search intents boost matching items
   - `_get_item_attr_values()` helper works correctly

4. **End-to-end integration** (2-3 tests):
   - Search → session signal → feed reflects search intent
   - Action → score update → Redis persist → reload matches

---

## Environment Requirements

### New requirement: Redis

```bash
# Add to requirements.txt
redis>=5.0.0

# Environment variable (required)
REDIS_URL=redis://localhost:6379/0
```

### For tests (mock Redis):
```bash
# Already available via unittest.mock, no new deps needed
```

---

## File Change Summary

| File | Action | Lines Changed (est.) |
|------|--------|---------------------|
| `src/api/routes/search.py` | MODIFY | +40 lines |
| `src/recs/session_state.py` | MODIFY | +55 lines (new class) |
| `src/recs/pipeline.py` | MODIFY | ~60 lines changed |
| `src/search/reranker.py` | MODIFY | +90 lines |
| `src/search/hybrid_search.py` | MODIFY | +3 lines |
| `tests/unit/test_session_integration.py` | NEW | ~300 lines |
| `requirements.txt` | MODIFY | +1 line |

---

## Execution Order

1. Task 2a: Create `ScoringRedisBackend` in `session_state.py`
2. Task 2b: Update `pipeline.py` to use Redis backend
3. Task 1: Wire search signals in `search.py`
4. Task 3a: Update `reranker.py` with session scoring
5. Task 3b: Update `hybrid_search.py` to forward session_scores
6. Task 3c: Update `search.py` to load + pass session scores
7. Write tests
8. Run full test suite

This order ensures Redis backend exists before pipeline uses it, and pipeline is updated before search route calls it.
