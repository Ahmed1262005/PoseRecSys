# OutfitTransformer API - Complete Reference

> **Base URL:** `https://your-server.com`
> **Auth:** All endpoints (except health/info) require `Authorization: Bearer <supabase_jwt_token>`

---

## Table of Contents

1. [Authentication](#authentication)
2. [Recommendation Feed](#recommendation-feed) (3 active + 2 deprecated)
   - [GET /api/recs/v2/feed](#get-apirecsv2feed) -- Primary feed endpoint
   - [GET /api/recs/v2/sale](#get-apirecsv2sale) -- Sale items
   - [GET /api/recs/v2/new-arrivals](#get-apirecsv2new-arrivals) -- New arrivals
   - [GET /api/recs/v2/feed/keyset](#get-apirecsv2feedkeyset) -- Keyset pagination (full filter parity)
   - ~~GET /api/recs/v2/feed/endless~~ (deprecated)
3. [User Actions](#user-actions) (1 endpoint, instant response)
   - [POST /api/recs/v2/feed/action](#post-apirecsv2feedaction) -- Record interaction
4. [Session Management](#session-management) (2 active + 1 deprecated)
   - [GET /api/recs/v2/feed/session/{session_id}](#get-apirecsv2feedsessionsession_id) -- Debug info
   - [DELETE /api/recs/v2/feed/session/{session_id}](#delete-apirecsv2feedsessionsession_id) -- Clear session
   - ~~POST /api/recs/v2/session/sync~~ (deprecated, server auto-persists)
5. [Onboarding](#onboarding) (3 endpoints)
   - [POST /api/recs/v2/onboarding](#post-apirecsv2onboarding) -- Save full profile
   - [POST /api/recs/v2/onboarding/core-setup](#post-apirecsv2onboardingcore-setup) -- Save core + get categories
   - [POST /api/recs/v2/onboarding/v3](#post-apirecsv2onboardingv3) -- V3 flat format
6. [Hybrid Search](#hybrid-search) (4 endpoints)
   - [POST /api/search/hybrid](#post-apisearchhybrid) -- Search products
   - [GET /api/search/autocomplete](#get-apisearchautocomplete) -- Autocomplete
   - [POST /api/search/click](#post-apisearchclick) -- Track click
   - [POST /api/search/conversion](#post-apisearchconversion) -- Track conversion
7. [Info & Health](#info--health) (5 endpoints, all public)
   - [GET /api/recs/v2/info](#get-apirecsv2info)
   - [GET /api/recs/v2/health](#get-apirecsv2health)
   - [GET /api/recs/v2/categories/mapping](#get-apirecsv2categoriesmapping)
   - [GET /api/search/health](#get-apisearchhealth)
   - [GET /health](#get-health)
8. [Architecture: Background Tasks](#architecture-background-tasks)

---

## Authentication

All endpoints (except those marked **Public**) require a Supabase JWT token passed via the `Authorization` header:

```
Authorization: Bearer <supabase_jwt_token>
```

### Getting a Token

Tokens are issued by Supabase Auth after login. Use the Supabase client SDK:

```javascript
// JavaScript (Supabase JS SDK)
const { data, error } = await supabase.auth.signInWithPassword({
  email: 'user@example.com',
  password: 'password',
});
const token = data.session.access_token;

// Use in API calls
fetch('/api/recs/v2/feed', {
  headers: { 'Authorization': `Bearer ${token}` }
});
```

```swift
// Swift (Supabase Swift SDK)
let session = try await supabase.auth.signIn(email: "user@example.com", password: "password")
let token = session.accessToken
```

### JWT Payload Structure

The server decodes the JWT and extracts:

```json
{
  "sub": "550e8400-e29b-41d4-a716-446655440000",   // User UUID -> user.id
  "email": "user@example.com",                       // -> user.email
  "phone": "+1234567890",                             // -> user.phone
  "role": "authenticated",                            // Postgres role
  "aal": "aal1",                                      // Auth assurance level
  "session_id": "abc-def-123",                        // Supabase session
  "is_anonymous": false,                              // Anonymous auth flag
  "app_metadata": {                                   // -> user.app_metadata
    "provider": "email",
    "providers": ["email"]
  },
  "user_metadata": {                                  // -> user.user_metadata
    "birthdate": "1995-03-15",                        // Used for age-based scoring
    "city": "New York",                               // Used for weather scoring
    "country": "US",
    "name": "Jane Doe"
  },
  "exp": 1707500000,                                  // Expiry timestamp
  "aud": "authenticated"                              // Must be "authenticated"
}
```

### user_metadata for Context Scoring

The `user_metadata` object in the JWT drives **context-aware scoring** (age affinity + weather/season):

| Field | Type | Used For | Example |
|-------|------|----------|---------|
| `birthdate` | string | Age group detection -> age-appropriate item scoring | `"1995-03-15"` |
| `city` | string | Weather API lookup -> season-appropriate item scoring | `"New York"` |
| `country` | string | Fallback location for weather | `"US"` |

If these fields are absent, context scoring is skipped (feed still works, just without age/weather personalization).

**Age groups derived from birthdate:**

| Age | Group | Effect |
|-----|-------|--------|
| 18-24 | `gen_z` | Boosts crop tops, bold colors, trendy items |
| 25-34 | `young_adult` | Balanced -- slight boost for modern fits |
| 35-44 | `mid_career` | Boosts structured pieces, smart-casual |
| 45-59 | `established` | Boosts classic styles, jewel tones, longer hemlines |
| 60+ | `senior` | Penalizes very revealing items, boosts elegant fits |

### Error Responses

| Status | Error | Meaning |
|--------|-------|---------|
| `401` | `"Authorization header required"` | No `Authorization` header sent |
| `401` | `"Token required"` | Header present but empty |
| `401` | `"Token has expired"` | JWT `exp` claim is in the past. Refresh the token. |
| `401` | `"Invalid token audience"` | JWT `aud` claim is not `"authenticated"` |
| `401` | `"Invalid token: ..."` | Signature verification failed, malformed token, etc. |

All 401 responses include `WWW-Authenticate: Bearer` header.

### Public Endpoints (No Auth Required)

| Endpoint | Description |
|----------|-------------|
| `GET /health` | App health check |
| `GET /ready` | Readiness probe |
| `GET /live` | Liveness probe |
| `GET /health/detailed` | Detailed health with dependency status |
| `GET /api/recs/v2/info` | Pipeline configuration |
| `GET /api/recs/v2/health` | Pipeline health |
| `GET /api/recs/v2/categories/mapping` | Category mapping reference |
| `GET /api/search/health` | Search service health |

---

## Recommendation Feed

### GET /api/recs/v2/feed

**The primary feed endpoint.** Returns personalized product recommendations with keyset cursor pagination.
Supports 40+ filters (21 attribute dimensions with include/exclude), session-aware scoring, age/weather context scoring, and diversity constraints.

#### Pagination Flow (Cursor-Based)

```
1. First request:  GET /api/recs/v2/feed?page_size=50
   Response: { session_id: "sess_abc123", cursor: "eyJz...", results: [...], pagination: { has_more: true } }

2. Next page:      GET /api/recs/v2/feed?session_id=sess_abc123&cursor=eyJz...&page_size=50
   Response: { session_id: "sess_abc123", cursor: "eyJz..NEW..", results: [...], pagination: { has_more: true } }

3. Repeat until:   pagination.has_more == false
```

**Key rules:**
- First request: Do NOT send `cursor` or `session_id` (both auto-generated)
- Subsequent requests: Send BOTH `session_id` and `cursor` from the previous response
- The cursor is an opaque base64 string -- do not parse or modify it
- No duplicates within a session (server tracks seen items)
- O(1) performance regardless of page depth (page 100 same speed as page 1)

#### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | No | auto-generated | Session ID from previous response. Send back for pagination continuity. |
| `cursor` | string | No | null | Opaque cursor from previous response. null = first page. |
| `page_size` | int | No | 50 | Items per page (1-200) |
| `gender` | string | No | "female" | Gender filter |

##### Category Filters

| Parameter | Type | Description |
|-----------|------|-------------|
| `categories` | string | Comma-separated broad categories: `tops`, `bottoms`, `dresses`, `outerwear`, `skirts`, `one-piece`, `sportswear` |
| `article_types` | string | Comma-separated specific types: `jeans`, `t-shirts`, `tank tops`, `sweaters`, `knitwear`, `midi dress`, `mini dress`, `blazer`, etc. Supports fuzzy matching (e.g., `jeans` matches `skinny jeans`, `straight jeans`). |

##### Color Filters

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_colors` | string | Comma-separated colors to include (hard filter -- item must match at least one). Example: `black,white,navy` |
| `exclude_colors` | string | Comma-separated colors to exclude. Example: `neon,orange` |

##### Brand Filters

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_brands` | string | Comma-separated brands to include (hard filter -- ONLY these brands). Example: `Zara,H&M` |
| `exclude_brands` | string | Comma-separated brands to exclude. Example: `Shein,Boohoo` |

##### Lifestyle Filters

| Parameter | Type | Description |
|-----------|------|-------------|
| `exclude_styles` | string | Comma-separated coverage styles to avoid: `deep-necklines`, `sheer`, `cutouts`, `backless`, `strapless` |

##### Price Filters

| Parameter | Type | Description |
|-----------|------|-------------|
| `min_price` | float | Minimum price (>= 0) |
| `max_price` | float | Maximum price (>= 0) |
| `on_sale_only` | bool | Only return items on sale. Default: `false` |

##### Attribute Filters (Hard Include/Exclude)

All attribute filters are **hard filters** -- items that don't match are strictly excluded. Every attribute supports both `include_` (whitelist) and `exclude_` (blacklist) variants. All values are **comma-separated** and **case-insensitive**.

**Include behavior:**
- Single-value attributes: item's value must be in the include list. Items with `null` values are **excluded**.
- Multi-value attributes: item must have **at least one** value matching the include list. Items with empty/null lists are **excluded**.

**Exclude behavior:**
- Single-value attributes: item's value must NOT be in the exclude list. Items with `null` values **pass through**.
- Multi-value attributes: item must have **no** values matching the exclude list. Items with empty/null lists **pass through**.

You can combine `include_` and `exclude_` on the same attribute (e.g., include Casual + Smart Casual formality, but exclude any that also match a specific exclusion).

###### Formality

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_formality` | string | Formality levels to include. Values: `Casual`, `Smart Casual`, `Semi-Formal`, `Formal` |
| `exclude_formality` | string | Formality levels to exclude |

###### Occasions

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_occasions` | string | Occasions to include. Values: `casual`, `office`, `evening`, `beach`, `active`, `date_night`, `party`, `work`, `everyday`, `vacation` |
| `exclude_occasions` | string | Occasions to exclude |

###### Seasons

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_seasons` | string | Seasons to include. Values: `Spring`, `Summer`, `Fall`, `Winter` |
| `exclude_seasons` | string | Seasons to exclude |

###### Style Tags

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_style_tags` | string | Style tags to include. Values: `Classic`, `Trendy`, `Bold`, `Minimal`, `Street`, `Boho`, `Romantic`, `Edgy`, `Preppy` |
| `exclude_style_tags` | string | Style tags to exclude |

###### Color Family

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_color_family` | string | Color families to include. Values: `Neutrals`, `Blues`, `Browns`, `Greens`, `Pinks`, `Reds`, `Purples`, `Yellows`, `Oranges`, `Whites`, `Blacks`, `Grays` |
| `exclude_color_family` | string | Color families to exclude |

###### Patterns

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_patterns` | string | Patterns to include. Values: `solid`, `stripes`, `floral`, `geometric`, `animal-print`, `plaid`, `polka-dot`, `abstract`, `checkered` |
| `exclude_patterns` | string | Patterns to exclude |

###### Silhouette

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_silhouette` | string | Silhouettes to include. Values: `Fitted`, `A-Line`, `Straight`, `Wide Leg`, `Skinny`, `Relaxed`, `Bodycon`, `Oversized`, `Flared` |
| `exclude_silhouette` | string | Silhouettes to exclude |

###### Fit

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_fit` | string | Fits to include. Values: `slim`, `regular`, `relaxed`, `oversized` |
| `exclude_fit` | string | Fits to exclude |

###### Length

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_length` | string | Lengths to include. Values: `cropped`, `standard`, `long`, `mini`, `midi`, `maxi` |
| `exclude_length` | string | Lengths to exclude |

###### Sleeves

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_sleeves` | string | Sleeve types to include. Values: `short`, `long`, `sleeveless`, `3/4`, `cap`, `puff` |
| `exclude_sleeves` | string | Sleeve types to exclude |

###### Neckline

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_neckline` | string | Necklines to include. Values: `crew`, `v-neck`, `scoop`, `turtleneck`, `mock`, `boat`, `square`, `halter`, `off-shoulder`, `sweetheart` |
| `exclude_neckline` | string | Necklines to exclude |

###### Rise

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_rise` | string | Rise to include (bottoms). Values: `high`, `mid`, `low` |
| `exclude_rise` | string | Rise to exclude |

###### Coverage

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_coverage` | string | Coverage levels to include. Values: `Full`, `Moderate`, `Partial`, `Minimal` |
| `exclude_coverage` | string | Coverage levels to exclude |

###### Materials

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_materials` | string | Materials to include. Values: `cotton`, `linen`, `silk`, `polyester`, `wool`, `denim`, `leather`, `satin`, `chiffon`, `knit`, `velvet`, `rayon`, `nylon` |
| `exclude_materials` | string | Materials to exclude |

##### Deprecated Parameters

These still work but map to the new `include_` equivalents internally. Prefer the new params.

| Parameter | Maps to | Description |
|-----------|---------|-------------|
| `fit` | `include_fit` | Comma-separated fits |
| `length` | `include_length` | Comma-separated lengths |
| `sleeves` | `include_sleeves` | Comma-separated sleeve types |
| `neckline` | `include_neckline` | Comma-separated necklines |
| `rise` | `include_rise` | Comma-separated rises |
| `preferred_brands` | `include_brands` | Comma-separated brands |

#### Response

```json
{
  "user_id": "uuid-string",
  "session_id": "sess_abc123def456",
  "cursor": "eyJzY29yZSI6IDAuNTIsICJpdGVtX2lkIjogInh4eCIsICJwYWdlIjogMH0=",
  "strategy": "exploration",
  "results": [
    {
      "product_id": "550e8400-e29b-41d4-a716-446655440000",
      "rank": 1,
      "score": 1.234,
      "reason": "personalized",
      "category": "Dresses",
      "broad_category": "dresses",
      "brand": "Reformation",
      "name": "Juliette Dress",
      "price": 218.00,
      "image_url": "https://...",
      "gallery_images": ["https://...", "https://..."],
      "colors": ["black", "white"],
      "source": "taste_vector",
      "original_price": 278.00,
      "is_on_sale": true,
      "discount_percent": 22,
      "is_new": false
    }
  ],
  "pagination": {
    "page": 0,
    "page_size": 50,
    "items_returned": 50,
    "session_seen_count": 50,
    "has_more": true
  },
  "metadata": {
    "candidates_retrieved": 500,
    "candidates_after_python_filters": 480,
    "candidates_after_scoring": 480,
    "candidates_after_dedup": 50,
    "sasrec_available": true,
    "seed_vector_available": true,
    "has_onboarding": true,
    "user_state_type": "tinder_complete",
    "by_source": {"taste_vector": 40, "exploration": 10},
    "keyset_pagination": true,
    "session_scoring": {
      "action_count": 3,
      "active_clusters": ["A", "M"],
      "active_brands": 2,
      "search_intents": 1,
      "session_intent_candidates": 15
    },
    "context_scoring": {
      "age_group": "young_adult",
      "has_weather": true,
      "season": "summer",
      "city": "New York"
    },
    "db_seen_history_count": 0,
    "feed_version": "v_abc123"
  }
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session identifier. **Send back on subsequent requests.** |
| `cursor` | string | Opaque pagination cursor. **Send back to get the next page.** |
| `strategy` | string | Ranking strategy used: `exploration`, `seed_vector`, `sasrec` |
| `results` | array | Array of product items (see below) |
| `pagination.page` | int | Current page number (0-indexed) |
| `pagination.page_size` | int | Requested page size |
| `pagination.items_returned` | int | Actual items returned (may be < page_size on last page) |
| `pagination.session_seen_count` | int | Total items shown in this session so far |
| `pagination.has_more` | bool | `true` if more pages available, `false` if catalog exhausted |
| `metadata` | object | Pipeline debug info (candidates retrieved, filters applied, scoring details) |

#### Result Item Fields

| Field | Type | Description |
|-------|------|-------------|
| `product_id` | string | Product UUID |
| `rank` | int | Global rank within the session (1-indexed, continuous across pages) |
| `score` | float | Final combined score (base + session + context scoring) |
| `reason` | string | Why this item was recommended: `personalized`, `style_matched`, `trending`, `explore` |
| `category` | string | Product category (e.g., "Dresses") |
| `broad_category` | string | Broad category (e.g., "dresses") |
| `brand` | string | Brand name |
| `name` | string | Product name |
| `price` | float | Current price |
| `image_url` | string | Primary image URL |
| `gallery_images` | string[] | Additional image URLs |
| `colors` | string[] | Product colors |
| `source` | string | Candidate source: `taste_vector`, `trending`, `exploration`, `session_intent` |
| `original_price` | float\|null | Original price before discount (null if not on sale) |
| `is_on_sale` | bool | Whether item is on sale |
| `discount_percent` | int\|null | Discount as integer percentage (e.g., 25 = 25% off) |
| `is_new` | bool | Whether item was added in the last 7 days |

---

### GET /api/recs/v2/sale

**Personalized sale items feed.** Same pipeline and filters as `/feed`, but only returns items where `original_price > price`.

#### Query Parameters

Same as [GET /api/recs/v2/feed](#query-parameters) (all filters, cursor, pagination supported).

#### Response

Same format as `/feed`. All items will have:
- `is_on_sale: true`
- `original_price`: filled
- `discount_percent`: filled (integer, e.g., 25)

#### Example

```
GET /api/recs/v2/sale?categories=dresses&max_price=100&page_size=20
```

---

### GET /api/recs/v2/new-arrivals

**Personalized new arrivals feed.** Same pipeline and filters as `/feed`, but only returns items added in the last 7 days.

#### Query Parameters

Same as [GET /api/recs/v2/feed](#query-parameters) (all filters, cursor, pagination supported).

#### Response

Same format as `/feed`. All items will have `is_new: true`.

#### Example

```
GET /api/recs/v2/new-arrivals?categories=tops&include_brands=Zara,H%26M&page_size=30
```

---

### GET /api/recs/v2/feed/keyset

**Keyset pagination endpoint with full filter parity.** Functionally identical to `/feed` with the same 40+ filter parameters. Does not auto-persist seen_ids in the background (caller manages session).

#### Query Parameters

Same as [GET /api/recs/v2/feed](#query-parameters) -- all filters, cursor, pagination, and attribute include/exclude params are supported.

The only differences from `/feed`:
- No background auto-persist of seen_ids (caller manages session lifecycle)
- Does not accept the deprecated legacy params (`fit`, `length`, `sleeves`, `neckline`, `rise`, `preferred_brands`) -- use the `include_` variants directly

#### Response

Same format as `/feed`.

---

### ~~GET /api/recs/v2/feed/endless~~ (DEPRECATED)

**Deprecated.** Use `GET /api/recs/v2/feed` instead, which provides keyset cursor pagination (O(1)), 40+ filters (21 attribute dimensions with include/exclude), session scoring, and context-aware scoring.

---

## User Actions

### POST /api/recs/v2/feed/action

Record an explicit user interaction with a product. **Response is instant (~1ms)** -- session scoring updates in-memory, Supabase persistence happens in a background task (non-blocking).

```
Request -> Validate (0ms) -> Session scoring update (1ms) -> Return 200 -> [Background: Supabase INSERT]
```

#### Request Body

```json
{
  "session_id": "sess_abc123def456",
  "product_id": "550e8400-e29b-41d4-a716-446655440000",
  "action": "click",
  "source": "feed",
  "position": 3,
  "brand": "Reformation",
  "item_type": "midi_dress",
  "attributes": {
    "fit": "fitted",
    "color_family": "black",
    "pattern": "solid",
    "formality": "smart_casual",
    "neckline": "v_neck",
    "sleeve": "short",
    "length": "midi"
  }
}
```

#### Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | string | **Yes** | Session ID from the feed response |
| `product_id` | string | **Yes** | Product UUID that was interacted with |
| `action` | string | **Yes** | Action type (see valid actions below) |
| `source` | string | No | Where the action happened: `feed`, `search`, `similar`, `style-this`. Default: `feed` |
| `position` | int | No | Position in feed/list when the action occurred (1-indexed) |
| `brand` | string | No | Product brand name (enables brand affinity learning) |
| `item_type` | string | No | Product type/article_type (enables type preference learning) |
| `attributes` | object | No | Product attributes for multi-dimensional preference learning |

#### Valid Actions

| Action | Signal Strength | Description |
|--------|----------------|-------------|
| `click` | 0.5 | User tapped to view product details |
| `hover` | 0.1 | User swiped through photo gallery |
| `add_to_wishlist` | 2.0 | User saved/liked the item (strong positive) |
| `add_to_cart` | 0.8 | User added to cart (conversion intent) |
| `purchase` | 1.0 | User completed purchase |
| `skip` | -0.5 | User explicitly skipped/dismissed |

**How actions affect the feed:**
- Each action updates the session's multi-dimensional EMA (Exponential Moving Average) preference model
- Dimensions tracked: brand affinity, type preference, style cluster, attribute patterns, search intent
- Higher-signal actions (wishlist, search) spike the "fast track" immediately
- The next feed request will reflect these learned preferences via session-aware scoring
- **Mismatch penalty**: After signaling interest in a type/brand, non-matching items are actively pushed down

#### Attributes Object

Send as many attributes as available from the product data. Each key updates a specific preference dimension:

| Key | Example Values | Effect |
|-----|---------------|--------|
| `fit` | `fitted`, `relaxed`, `oversized`, `slim` | Learns fit preference |
| `color_family` | `black`, `navy`, `red`, `pastels` | Learns color preference |
| `pattern` | `solid`, `floral`, `striped`, `plaid` | Learns pattern preference |
| `formality` | `casual`, `smart_casual`, `formal` | Learns formality level |
| `neckline` | `crew`, `v_neck`, `scoop`, `turtleneck` | Learns neckline preference |
| `sleeve` | `sleeveless`, `short`, `long`, `3/4` | Learns sleeve preference |
| `length` | `cropped`, `standard`, `long`, `midi`, `maxi` | Learns length preference |
| `style` | `bohemian`, `minimalist`, `preppy`, `streetwear` | Learns style preference |
| `occasion` | `casual`, `office`, `evening`, `party` | Learns occasion preference |

#### Response

```json
{
  "status": "success",
  "interaction_id": null
}
```

> **Note:** `interaction_id` is always `null` because the response returns before the background Supabase write completes. The interaction is still persisted reliably.

#### Example: Full Click Tracking Flow

```
1. User sees feed:
   GET /api/recs/v2/feed?session_id=sess_abc&page_size=50

2. User clicks product #3 (a Reformation midi dress):
   POST /api/recs/v2/feed/action
   {
     "session_id": "sess_abc",
     "product_id": "prod_123",
     "action": "click",
     "position": 3,
     "brand": "Reformation",
     "item_type": "midi_dress",
     "attributes": {"pattern": "floral", "color_family": "blue", "fit": "fitted"}
   }

3. User adds to wishlist:
   POST /api/recs/v2/feed/action
   {
     "session_id": "sess_abc",
     "product_id": "prod_123",
     "action": "add_to_wishlist",
     "brand": "Reformation",
     "item_type": "midi_dress"
   }

4. Feed refreshes (next page or pull-to-refresh):
   GET /api/recs/v2/feed?session_id=sess_abc&cursor=eyJz...
   -> Reformation and midi dresses are now boosted
   -> Non-dress items are penalized
```

---

## Session Management

> **Seen_ids are auto-persisted.** Each feed request (`/feed`, `/sale`, `/new-arrivals`) automatically
> persists the shown product IDs to Supabase in a background task. No manual sync needed.

### ~~POST /api/recs/v2/session/sync~~ (DEPRECATED)

**This endpoint is deprecated.** Seen_ids are now auto-persisted by the server on each feed request via background tasks. Kept for backward compatibility only.

---

### GET /api/recs/v2/feed/session/{session_id}

Get debug information about a session. Useful for debugging pagination issues or inspecting session state.

**No auth required for this endpoint (debug only).**

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Session ID to inspect |

#### Response

```json
{
  "session_id": "sess_abc123def456",
  "seen_count": 150,
  "current_offset": 150,
  "signals": {
    "views": 150,
    "clicks": 5,
    "skips": 2
  },
  "created_at": "2026-02-09T10:30:00",
  "last_access": "2026-02-09T11:45:00",
  "age_seconds": 4500,
  "cursor": {
    "score": 0.52,
    "item_id": "550e8400-e29b-41d4-a716-446655440099",
    "page": 2
  },
  "feed_version": {
    "version_id": "v_abc123",
    "created_at": "2026-02-09T10:30:00"
  }
}
```

#### Error

Returns `404` if session not found.

---

### DELETE /api/recs/v2/feed/session/{session_id}

Clear a session to start fresh. Removes all seen items, cursor position, and feed version. The next feed request will show items from the beginning.

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | string | Session ID to clear |

#### Response

```json
{
  "status": "success",
  "session_id": "sess_abc123def456",
  "message": "Session cleared. Next feed request will show fresh items."
}
```

---

## Onboarding

### POST /api/recs/v2/onboarding

Save the user's complete 10-module onboarding profile. This is called after the user finishes the full onboarding flow (including the Tinder-style style discovery).

#### Request Body

The request body mirrors the frontend's onboarding state. All modules are optional except `core_setup`:

```json
{
  "gender": "female",
  "core_setup": {
    "enabled": true,
    "selectedCategories": ["tops", "dresses", "bottoms"],
    "sizes": ["S", "M"],
    "birthdate": "1995-03-15",
    "colorsToAvoid": ["neon", "orange"],
    "materialsToAvoid": ["polyester"]
  },
  "lifestyle": {
    "enabled": true,
    "occupation": "office",
    "weekdayRoutine": ["work", "gym"],
    "weekendRoutine": ["brunch", "shopping"],
    "budgetRange": "mid"
  },
  "tops": {
    "enabled": true,
    "topTypes": ["blouse", "tee", "sweater"],
    "fits": ["regular", "relaxed"],
    "sleeves": ["short", "long"],
    "necklines": ["crew", "v-neck"],
    "priceComfort": { "min": 20, "max": 80 }
  },
  "bottoms": {
    "enabled": true,
    "bottomTypes": ["jeans", "trousers"],
    "fits": ["slim", "regular"],
    "rises": ["high", "mid"],
    "lengths": ["standard", "cropped"],
    "priceComfort": { "min": 30, "max": 120 }
  },
  "dresses": {
    "enabled": true,
    "dressTypes": ["midi", "wrap"],
    "fits": ["fitted", "a-line"],
    "lengths": ["midi", "maxi"],
    "sleeves": ["short", "sleeveless"],
    "priceComfort": { "min": 40, "max": 200 }
  },
  "one_piece": { "enabled": false },
  "outerwear": {
    "enabled": true,
    "outerwearTypes": ["blazer", "leather-jacket"],
    "fits": ["regular"],
    "sleeves": ["long"],
    "priceComfort": { "min": 50, "max": 300 }
  },
  "style": {
    "enabled": true,
    "styleDirections": ["classic", "minimalist"],
    "modestyPreference": "moderate"
  },
  "brands": {
    "enabled": true,
    "preferredBrands": ["Reformation", "Zara", "COS"],
    "brandsToAvoid": ["Shein"],
    "brandOpenness": "mix-favorites-new"
  },
  "style_discovery": {
    "enabled": true,
    "selections": [
      { "round": 1, "chosen_id": "prod_001", "all_ids": ["prod_001", "prod_002", "prod_003", "prod_004"] }
    ],
    "summary": {
      "taste_vector": [0.123, -0.456, ...],
      "top_categories": ["dresses", "tops"],
      "preferred_styles": ["minimalist", "classic"]
    }
  }
}
```

#### Response

```json
{
  "status": "success",
  "user_id": "uuid-string",
  "modules_saved": 8,
  "categories_selected": ["tops", "dresses", "bottoms"],
  "has_taste_vector": true
}
```

---

### POST /api/recs/v2/onboarding/core-setup

Save just the core-setup module and get the Tinder test categories. Call this **before** starting the style discovery (Tinder) test so it only shows items from the user's selected categories.

#### Integration Flow

```
1. User completes core-setup in onboarding
2. Frontend calls POST /api/recs/v2/onboarding/core-setup
3. Backend returns tinder_categories
4. Frontend uses tinder_categories when calling POST /api/women/session/start
5. User completes Tinder test
6. Frontend calls POST /api/recs/v2/onboarding with complete 10-module data
```

#### Request Body

```json
{
  "gender": "female",
  "core_setup": {
    "selectedCategories": ["tops", "dresses", "bottoms", "outerwear"],
    "sizes": ["S", "M"],
    "birthdate": "1995-03-15",
    "colorsToAvoid": ["neon"],
    "materialsToAvoid": []
  }
}
```

#### Response

```json
{
  "status": "success",
  "user_id": "uuid-string",
  "categories_selected": ["tops", "dresses", "bottoms", "outerwear"],
  "tinder_categories": [
    "tops_knitwear",
    "tops_woven",
    "tops_sleeveless",
    "tops_special",
    "dresses",
    "bottoms_trousers",
    "bottoms_skorts",
    "outerwear"
  ],
  "colors_to_avoid": ["neon"],
  "materials_to_avoid": []
}
```

**Category Mapping:**

| Onboarding Category | Tinder Categories |
|---------------------|-------------------|
| `tops` | `tops_knitwear`, `tops_woven`, `tops_sleeveless`, `tops_special` |
| `bottoms` | `bottoms_trousers`, `bottoms_skorts` |
| `dresses` | `dresses` |
| `skirts` | `bottoms_skorts` |
| `outerwear` | `outerwear` |
| `one-piece` | `dresses` |

---

### POST /api/recs/v2/onboarding/v3

Save V3 onboarding profile (new frontend spec). Uses flat attribute preferences with category mappings instead of per-module structure.

#### Request Body

```json
{
  "gender": "female",
  "core_setup": {
    "categories": ["tops", "bottoms", "dresses"],
    "topSize": ["S", "M"],
    "bottomSize": ["4", "6"],
    "outerwearSize": ["S"],
    "birthdate": "1995-03-15",
    "colorsToAvoid": ["neon"],
    "materialsToAvoid": []
  },
  "attribute_preferences": {
    "fits": [
      { "fitId": "regular", "categories": ["tops", "bottoms"] },
      { "fitId": "fitted", "categories": ["dresses"] }
    ],
    "sleeves": [
      { "sleeveId": "short", "categories": ["tops", "dresses"] }
    ],
    "lengths": [
      { "lengthId": "standard", "categories": ["tops", "bottoms"] }
    ],
    "lengthsDresses": [
      { "lengthId": "midi", "categories": ["dresses"] }
    ],
    "rises": ["high", "mid"],
    "necklines": ["crew", "v-neck"]
  },
  "type_preferences": {
    "tops": ["tee", "blouse", "sweater"],
    "bottoms": ["jeans", "trousers"],
    "dresses": ["midi-dress", "wrap-dress"],
    "outerwear": ["blazer"]
  },
  "lifestyle": {
    "occasions": ["casual", "office"],
    "stylesToAvoid": ["deep-necklines", "sheer"],
    "patternsLiked": ["solid", "stripes"],
    "patternsAvoided": ["animal-print"],
    "stylePersona": ["classic", "minimalist"]
  },
  "brands": {
    "preferred": ["Reformation", "COS"],
    "toAvoid": ["Shein"],
    "openness": "mix-favorites-new"
  },
  "style_discovery": {
    "completed": true,
    "swipedItems": ["prod_001", "prod_002"],
    "taste_vector": [0.123, -0.456, ...]
  }
}
```

#### Response

```json
{
  "status": "success",
  "user_id": "uuid-string",
  "categories_selected": ["tops", "bottoms", "dresses"],
  "has_taste_vector": true,
  "has_attribute_preferences": true,
  "has_type_preferences": true
}
```

---

## Hybrid Search

### POST /api/search/hybrid

Search products using the hybrid Algolia (lexical) + FashionCLIP (semantic) pipeline. Supports typo tolerance, synonyms, semantic understanding, and 23+ filters.

#### How It Works

1. **Query Classification** -- auto-detects intent:
   - `exact`: Brand/product name (e.g., "Ba&sh") -- Algolia dominates
   - `specific`: Category + attribute (e.g., "blue midi dress") -- balanced merge
   - `vague`: Style/occasion/vibe (e.g., "quiet luxury blazer") -- FashionCLIP dominates
2. **Algolia Search** -- lexical search with facets and typo tolerance
3. **FashionCLIP Semantic Search** -- visual/semantic similarity via pgvector
4. **RRF Merge** -- Reciprocal Rank Fusion combining both result sets
5. **Post-Filtering** -- strict attribute enforcement
6. **Reranking** -- session dedup, near-duplicate removal, profile boosts, brand diversity
7. **Facets** -- filterable attribute counts for UI filter chips

#### Request Body

```json
{
  "query": "summer vacation dress",
  "categories": ["dresses"],
  "brands": ["Reformation", "Free People"],
  "colors": ["blue", "white"],
  "min_price": 50,
  "max_price": 200,
  "occasions": ["vacation", "casual"],
  "patterns": ["floral", "solid"],
  "on_sale_only": false,
  "page": 1,
  "page_size": 50,
  "session_id": "sess_abc123",
  "semantic_boost": 0.4
}
```

#### Request Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | **Yes** | -- | Search query (1-500 chars) |
| `page` | int | No | 1 | Page number (1-indexed) |
| `page_size` | int | No | 50 | Results per page (1-100) |
| `session_id` | string | No | null | Session ID for dedup + feed signal wiring |
| `semantic_boost` | float | No | 0.4 | Weight for semantic results in RRF (0.0-1.0) |
| `on_sale_only` | bool | No | false | Only show sale items |

**Filter fields (all optional, all accept arrays):**

| Field | Type | Example |
|-------|------|---------|
| `categories` | string[] | `["tops", "dresses"]` |
| `category_l1` | string[] | `["Tops", "Dresses"]` |
| `category_l2` | string[] | `["Blouse", "Midi Dress"]` |
| `brands` | string[] | `["Zara", "H&M"]` |
| `exclude_brands` | string[] | `["Shein"]` |
| `colors` | string[] | `["black", "navy"]` |
| `color_family` | string[] | `["Neutrals", "Blues"]` |
| `patterns` | string[] | `["Solid", "Floral"]` |
| `materials` | string[] | `["Cotton", "Silk"]` |
| `occasions` | string[] | `["Office", "Casual"]` |
| `seasons` | string[] | `["Spring", "Summer"]` |
| `formality` | string[] | `["Casual", "Smart Casual"]` |
| `fit_type` | string[] | `["Slim", "Regular"]` |
| `neckline` | string[] | `["V-Neck", "Crew"]` |
| `sleeve_type` | string[] | `["Short", "Long"]` |
| `length` | string[] | `["Midi", "Maxi"]` |
| `rise` | string[] | `["High", "Mid"]` |
| `silhouette` | string[] | `["A-Line", "Fitted"]` |
| `article_type` | string[] | `["jeans", "midi dress"]` |
| `style_tags` | string[] | `["boho", "minimalist"]` |
| `min_price` | float | `29.99` |
| `max_price` | float | `199.99` |

#### Response

```json
{
  "query": "summer vacation dress",
  "intent": "specific",
  "results": [
    {
      "product_id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "Winslow Midi Dress",
      "brand": "Reformation",
      "image_url": "https://...",
      "gallery_images": ["https://...", "https://..."],
      "price": 178.00,
      "original_price": 248.00,
      "is_on_sale": true,
      "category_l1": "Dresses",
      "category_l2": "Midi Dress",
      "broad_category": "dresses",
      "article_type": "midi dress",
      "primary_color": "Blue",
      "color_family": "Blues",
      "pattern": "Floral",
      "apparent_fabric": "Linen Blend",
      "fit_type": "Regular",
      "formality": "Casual",
      "silhouette": "A-Line",
      "length": "Midi",
      "neckline": "V-Neck",
      "sleeve_type": "Short",
      "rise": null,
      "style_tags": ["boho", "romantic"],
      "occasions": ["Vacation", "Casual", "Date Night"],
      "seasons": ["Spring", "Summer"],
      "algolia_rank": 2,
      "semantic_rank": 1,
      "semantic_score": 0.87,
      "rrf_score": 0.045
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 50,
    "has_more": true,
    "total_results": 245
  },
  "timing": {
    "algolia_ms": 45,
    "semantic_ms": 120,
    "merge_ms": 5,
    "rerank_ms": 12,
    "total_ms": 190
  },
  "facets": {
    "brand": [
      { "value": "Reformation", "count": 23 },
      { "value": "Free People", "count": 18 }
    ],
    "color_family": [
      { "value": "Blues", "count": 45 },
      { "value": "Neutrals", "count": 30 }
    ],
    "formality": [
      { "value": "Casual", "count": 120 },
      { "value": "Smart Casual", "count": 45 }
    ],
    "pattern": [
      { "value": "Solid", "count": 89 },
      { "value": "Floral", "count": 34 }
    ]
  }
}
```

#### Facets

The `facets` field returns available filter options with counts. Use these to populate filter chips in the UI:

- Only values with `count > 1` are included
- Null/N/A values are excluded
- Only dimensions with 2+ distinct values are returned
- Available facet keys: `brand`, `category_l1`, `formality`, `primary_color`, `color_family`, `pattern`, `fit_type`, `neckline`, `sleeve_type`, `length`, `silhouette`, `rise`, `occasions`, `seasons`, `style_tags`, `article_type`, `broad_category`, `is_on_sale`, `materials`

#### Search + Feed Integration

When `session_id` is provided, the search query and filters are automatically forwarded to the recommendation pipeline's session scoring engine. This means:
- Searching "floral midi dress" teaches the feed you want floral midi dresses
- The next feed request (with the same `session_id`) will boost floral midi dresses and penalize non-matching items

---

### GET /api/search/autocomplete

Fast autocomplete powered by Algolia. Returns product name suggestions first, then brand suggestions.

#### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `q` | string | **Yes** | -- | Search query (1-200 chars) |
| `limit` | int | No | 10 | Max product suggestions (1-20) |

#### Response

```json
{
  "query": "ref",
  "products": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "Reformation Juliette Dress",
      "brand": "Reformation",
      "image_url": "https://...",
      "price": 218.00,
      "highlighted_name": "<em>Ref</em>ormation Juliette Dress"
    }
  ],
  "brands": [
    {
      "name": "Reformation",
      "highlighted": "<em>Ref</em>ormation"
    }
  ]
}
```

---

### POST /api/search/click

Record when a user clicks a search result. Used for search analytics and ranking improvement.

#### Request Body

```json
{
  "query": "summer dress",
  "product_id": "550e8400-e29b-41d4-a716-446655440000",
  "position": 3
}
```

#### Response

```json
{ "status": "ok" }
```

---

### POST /api/search/conversion

Record when a user converts (add to cart / purchase) from a search result.

#### Request Body

```json
{
  "query": "summer dress",
  "product_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### Response

```json
{ "status": "ok" }
```

---

## Info & Health

### GET /api/recs/v2/info

**Public.** Returns pipeline configuration metadata.

#### Response

```json
{
  "pipeline_config": {
    "max_per_category": 8,
    "exploration_rate": 0.1,
    "default_limit": 50,
    "max_limit": 200
  },
  "candidate_selection": {
    "primary_candidates": 300,
    "contextual_candidates": 100,
    "exploration_candidates": 50,
    "soft_weights": { "embedding": 0.6, "preference": 0.3, "novelty": 0.1 }
  },
  "sasrec_ranker": {
    "model_loaded": true,
    "vocab_size": 45000
  }
}
```

---

### GET /api/recs/v2/health

**Public.** Pipeline health check.

#### Response

```json
{
  "status": "healthy",
  "sasrec_loaded": true,
  "sasrec_vocab_size": 45000,
  "pipeline_ready": true
}
```

---

### GET /api/recs/v2/categories/mapping

**Public.** Category mapping reference for onboarding -> Tinder test integration.

#### Response

```json
{
  "onboarding_to_tinder": {
    "tops": ["tops_knitwear", "tops_woven", "tops_sleeveless", "tops_special"],
    "bottoms": ["bottoms_trousers", "bottoms_skorts"],
    "dresses": ["dresses"],
    "skirts": ["bottoms_skorts"],
    "outerwear": ["outerwear"],
    "one-piece": ["dresses"]
  },
  "tinder_categories": [
    { "id": "tops_knitwear", "label": "Sweaters & Knits", "broad": "tops" },
    { "id": "tops_woven", "label": "Blouses & Shirts", "broad": "tops" },
    { "id": "tops_sleeveless", "label": "Tank Tops & Camis", "broad": "tops" },
    { "id": "tops_special", "label": "Bodysuits", "broad": "tops" },
    { "id": "dresses", "label": "Dresses", "broad": "dresses" },
    { "id": "bottoms_trousers", "label": "Pants & Trousers", "broad": "bottoms" },
    { "id": "bottoms_skorts", "label": "Skirts & Shorts", "broad": "bottoms,skirts" },
    { "id": "outerwear", "label": "Outerwear", "broad": "outerwear" },
    { "id": "sportswear", "label": "Sportswear", "broad": "sportswear" }
  ],
  "onboarding_categories": [
    { "id": "tops", "label": "Tops", "tinder_maps_to": ["tops_knitwear", "tops_woven", "tops_sleeveless", "tops_special"] },
    { "id": "bottoms", "label": "Bottoms", "tinder_maps_to": ["bottoms_trousers", "bottoms_skorts"] },
    { "id": "dresses", "label": "Dresses", "tinder_maps_to": ["dresses"] },
    { "id": "skirts", "label": "Skirts", "tinder_maps_to": ["bottoms_skorts"] },
    { "id": "outerwear", "label": "Outerwear", "tinder_maps_to": ["outerwear"] },
    { "id": "one-piece", "label": "One-Piece", "tinder_maps_to": ["dresses"] }
  ]
}
```

---

### GET /api/search/health

**Public.** Search service health check (Algolia + semantic engine).

#### Response

```json
{
  "status": "healthy",
  "service": "search",
  "algolia": "healthy",
  "index_records": 85558,
  "semantic": "healthy"
}
```

---

### GET /health

**Public.** Basic application health check.

#### Response

```json
{
  "status": "healthy"
}
```

---

## Complete Integration Flow

### 1. New User Onboarding

```
POST /api/recs/v2/onboarding/core-setup   -> Save core prefs, get category mappings
POST /api/recs/v2/onboarding              -> Save full profile (all modules)
   OR
POST /api/recs/v2/onboarding/v3           -> Save V3 flat-format profile
```

### 2. Feed Consumption (Infinite Scroll)

```
GET  /api/recs/v2/feed?page_size=50         -> First page (get session_id + cursor)
                                               [BG: seen_ids auto-persisted to Supabase]
POST /api/recs/v2/feed/action                -> Track click (~1ms response)
                                               [BG: interaction persisted to Supabase]
POST /api/recs/v2/feed/action                -> Track wishlist (~1ms response)
GET  /api/recs/v2/feed?session_id=X&cursor=Y -> Next page (adapted to actions)
                                               [BG: seen_ids auto-persisted]
GET  /api/recs/v2/feed?session_id=X&cursor=Z -> Next page (further adapted)
```

No manual sync needed -- seen_ids are automatically persisted on each feed request.

### 3. Search + Feed Cross-Pollination

```
GET  /api/recs/v2/feed?session_id=X          -> Browse feed
POST /api/search/hybrid                       -> Search "floral midi dress" (with session_id)
                                                 [Auto-wires search signals to session]
POST /api/search/click                        -> Click a search result
GET  /api/recs/v2/feed?session_id=X&cursor=Y  -> Feed now boosts floral midi dresses
```

### 4. Filtered Feeds

```
# Basic category + color + price
GET /api/recs/v2/feed?categories=dresses&include_colors=black,navy&min_price=50&max_price=200

# Formal office wear, no bold patterns
GET /api/recs/v2/feed?include_formality=Formal,Semi-Formal&include_occasions=office&exclude_patterns=animal-print,geometric

# Summer casual tops in neutral tones, excluding polyester
GET /api/recs/v2/feed?categories=tops&include_seasons=Summer&include_formality=Casual&include_color_family=Neutrals&exclude_materials=polyester

# Fitted silhouette dresses, midi/maxi only, no strapless
GET /api/recs/v2/feed?categories=dresses&include_silhouette=Fitted,Bodycon&include_length=midi,maxi&exclude_neckline=off-shoulder

# Sale items with brand + style filter
GET /api/recs/v2/sale?include_brands=Zara,H%26M&include_style_tags=Classic,Minimal&on_sale_only=true

# New arrivals in specific color families
GET /api/recs/v2/new-arrivals?include_color_family=Blues,Greens&include_seasons=Spring

# Combine include + exclude on same dimension
GET /api/recs/v2/feed?include_formality=Casual,Smart Casual&exclude_coverage=Minimal

# Keyset endpoint (same filters)
GET /api/recs/v2/feed/keyset?categories=bottoms&include_fit=slim,regular&include_rise=high&exclude_materials=leather
```

---

## Architecture: Background Tasks

The API uses FastAPI `BackgroundTasks` to decouple fast responses from slow database writes.

### Action Recording

```
Frontend                    Server                           Supabase
   |                          |                                 |
   |-- POST /feed/action ---->|                                 |
   |                          |-- Update session scores (1ms) --|
   |<---- 200 { ok } --------|                                 |
   |                          |-- [Background] INSERT --------->|
   |                          |                                 |-- Persisted
```

### Feed + Auto-Persist

```
Frontend                    Server                           Supabase
   |                          |                                 |
   |-- GET /feed ------------>|                                 |
   |                          |-- Generate 50 items (200ms) ----|
   |<---- 200 { results } ----|                                 |
   |                          |-- [Background] INSERT seen_ids->|
   |                          |                                 |-- Persisted
```

### What runs in background (non-blocking):
- `user_interactions` table INSERT (from `/feed/action`)
- `session_seen_ids` table INSERT (from `/feed`, `/sale`, `/new-arrivals`)

### What runs inline (blocking, but fast):
- Session scoring update (in-memory/Redis, ~1ms)
- Feed generation (candidate retrieval + scoring + reranking)

---

## Error Responses

All endpoints return standard HTTP error codes:

| Code | Description |
|------|-------------|
| `400` | Invalid request (missing required fields, invalid action type, price validation) |
| `401` | Missing or invalid JWT token |
| `404` | Resource not found (session, product) |
| `500` | Internal server error |

Error body format:

```json
{
  "detail": "Invalid action 'like'. Must be one of: {'click', 'hover', 'add_to_wishlist', 'add_to_cart', 'purchase', 'skip'}"
}
```
