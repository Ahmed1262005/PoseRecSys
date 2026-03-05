# POSE Canvas API Reference

Base URL: `https://<host>/api/canvas`

All endpoints require a Supabase JWT in the `Authorization` header:

```
Authorization: Bearer <supabase_jwt_token>
```

---

## Overview

The Canvas API lets users build a visual inspiration board — upload images, paste URLs, or sync Pinterest pins. Each inspiration image is encoded with FashionCLIP (512-dim embedding), auto-classified by style via nearest-neighbor product attributes, and stored in the user's canvas.

The frontend uses these endpoints to:

1. **Build the canvas** — add/remove inspiration images
2. **Extract style** — get aggregated style attributes mapped to feed filter params
3. **Find similar products** — visually similar items from the catalog
4. **Complete the fit** — closest real product + outfit recommendations

### Typical Frontend Flow

```
Upload/URL/Pinterest  →  Canvas gallery  →  Style Elements (toggle filters)
                                          →  Similar Items (product grid)
                                          →  Complete the Fit (outfit builder)
                                          →  Feed with suggested_filters → GET /api/recs/v2/feed?include_style_tags=Sporty,Trendy&...
```

---

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/canvas/inspirations` | GET | List all inspirations |
| `/api/canvas/inspirations/upload` | POST | Upload an image file |
| `/api/canvas/inspirations/url` | POST | Add from a public URL |
| `/api/canvas/inspirations/pinterest` | POST | Bulk-import Pinterest pins |
| `/api/canvas/inspirations/{id}` | DELETE | Remove an inspiration |
| `/api/canvas/style-elements` | GET | Aggregated style profile |
| `/api/canvas/inspirations/{id}/similar` | GET | Find similar products |
| `/api/canvas/inspirations/{id}/complete-fit` | POST | Closest product + outfit |

---

## GET `/api/canvas/inspirations`

Returns all inspiration images for the authenticated user, ordered by creation date (newest first).

### Request

```
GET /api/canvas/inspirations
Authorization: Bearer <token>
```

No query parameters.

### Response `200`

```json
{
  "inspirations": [
    {
      "id": "2863221a-fdd6-43a1-bd98-3921afbffb71",
      "source": "upload",
      "image_url": "https://<supabase>/storage/v1/object/public/inspirations/<user_id>/8ae15d3b.jpg",
      "original_url": null,
      "title": "my_outfit.jpg",
      "style_label": "Sporty",
      "style_confidence": 0.28,
      "style_attributes": {
        "style_tags": { "Sporty": 0.28, "Trendy": 0.26, "Casual": 0.23 },
        "pattern": { "Solid": 0.82 },
        "color_family": { "Greens": 0.64, "Blues": 0.27 },
        "formality": { "Casual": 1.0 },
        "occasions": { "Workout": 0.35, "Everyday": 0.32 },
        "silhouette": { "Fitted": 0.55, "Slim": 0.27 },
        "fit_type": { "Slim": 0.64 },
        "sleeve_type": { "Sleeveless": 1.0 },
        "neckline": { "Scoop": 1.0 }
      },
      "pinterest_pin_id": null,
      "created_at": "2026-03-05T15:07:31.149100Z",
      "updated_at": "2026-03-05T15:07:31.149100Z"
    }
  ],
  "count": 1
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `inspirations` | array | List of inspiration objects |
| `count` | int | Total number of inspirations |

### Inspiration Object

| Field | Type | Description |
|-------|------|-------------|
| `id` | string (UUID) | Unique inspiration ID — used in `/similar`, `/complete-fit`, and `DELETE` |
| `source` | string | `"upload"`, `"url"`, `"camera"`, or `"pinterest"` |
| `image_url` | string | Public URL of the stored/original image |
| `original_url` | string \| null | Original URL before any re-hosting (URL source only) |
| `title` | string \| null | User-provided or auto-detected title |
| `style_label` | string \| null | Dominant style tag (e.g. `"Sporty"`, `"Boho"`, `"Classic"`). Null if classification failed. |
| `style_confidence` | float \| null | Confidence of the dominant label (0.0–1.0) |
| `style_attributes` | object | Per-dimension attribute distributions `{ attr_key: { value: score } }`. Keys: `style_tags`, `pattern`, `color_family`, `formality`, `occasions`, `silhouette`, `fit_type`, `sleeve_type`, `neckline` |
| `pinterest_pin_id` | string \| null | Pinterest pin ID (pinterest source only) |
| `created_at` | string (ISO 8601) | Creation timestamp |
| `updated_at` | string (ISO 8601) | Last update timestamp |

---

## POST `/api/canvas/inspirations/upload`

Upload an image file from the user's device or camera. The image is stored in Supabase Storage, encoded with FashionCLIP, and style-classified via K-nearest product attributes.

### Request

```
POST /api/canvas/inspirations/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Image file (JPEG, PNG, WebP, GIF). Max 10 MB. |

### cURL Example

```bash
curl -X POST https://<host>/api/canvas/inspirations/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@/path/to/outfit.jpg;type=image/jpeg"
```

### Response `200`

Returns a single `InspirationResponse` object (same shape as items in the list endpoint).

```json
{
  "id": "cc7ccb50-3f39-486e-913e-df017b2df1be",
  "source": "upload",
  "image_url": "https://<supabase>/storage/v1/object/public/inspirations/<user_id>/a1b2c3d4.jpg",
  "original_url": null,
  "title": "outfit.jpg",
  "style_label": "Casual",
  "style_confidence": 0.29,
  "style_attributes": { ... },
  "pinterest_pin_id": null,
  "created_at": "2026-03-05T19:50:00.000000Z",
  "updated_at": "2026-03-05T19:50:00.000000Z"
}
```

### Errors

| Code | Detail | Cause |
|------|--------|-------|
| 409 | `"Inspiration quota exceeded (max 50)"` | User has too many inspirations |
| 413 | `"File too large. Maximum size: 10 MB"` | File exceeds 10 MB |
| 415 | `"Unsupported file type: ..."` | Not JPEG/PNG/WebP/GIF |

---

## POST `/api/canvas/inspirations/url`

Add an inspiration from a public image URL. The server fetches the image, encodes it, and classifies the style.

### Request

```
POST /api/canvas/inspirations/url
Authorization: Bearer <token>
Content-Type: application/json
```

```json
{
  "url": "https://example.com/outfit.jpg",
  "title": "Summer look"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | Yes | Public image URL |
| `title` | string | No | Optional title (max 500 chars) |

### Response `200`

Returns a single `InspirationResponse` object.

```json
{
  "id": "0a7f8e34-dba7-47cd-94ae-a330283f9112",
  "source": "url",
  "image_url": "https://example.com/outfit.jpg",
  "original_url": "https://example.com/outfit.jpg",
  "title": "Summer look",
  "style_label": "Trendy",
  "style_confidence": 0.32,
  "style_attributes": { ... },
  "pinterest_pin_id": null,
  "created_at": "2026-03-05T19:52:00.000000Z",
  "updated_at": "2026-03-05T19:52:00.000000Z"
}
```

### Errors

| Code | Detail | Cause |
|------|--------|-------|
| 409 | `"Inspiration quota exceeded (max 50)"` | User has too many inspirations |
| 502 | `"Could not fetch image: ..."` | URL unreachable or not a valid image |

---

## POST `/api/canvas/inspirations/pinterest`

Bulk-import Pinterest pins as inspiration images. The server fetches each pin's image, encodes it, classifies its style, and skips any pins already imported.

### Request

```
POST /api/canvas/inspirations/pinterest
Authorization: Bearer <token>
Content-Type: application/json
```

```json
{
  "pin_ids": ["1234567890", "0987654321"],
  "max_pins": 20
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pin_ids` | string[] | No | Specific Pinterest pin IDs to import. If omitted, uses the user's saved board selection. |
| `max_pins` | int | No | Max pins to import (1–200). Defaults to server setting (60). |

### Response `200`

Returns an array of `InspirationResponse` objects (one per successfully imported pin).

```json
[
  {
    "id": "...",
    "source": "pinterest",
    "image_url": "https://i.pinimg.com/originals/...",
    "original_url": "https://www.pinterest.com/pin/...",
    "title": "Pin title",
    "style_label": "Boho",
    "style_confidence": 0.35,
    "style_attributes": { ... },
    "pinterest_pin_id": "1234567890",
    "created_at": "...",
    "updated_at": "..."
  }
]
```

### Errors

| Code | Detail | Cause |
|------|--------|-------|
| 502 | `"Pinterest sync error: ..."` | Pinterest API failure or token issue |

---

## DELETE `/api/canvas/inspirations/{inspiration_id}`

Remove an inspiration image. Also recomputes the user's taste vector (average of remaining embeddings).

### Request

```
DELETE /api/canvas/inspirations/{inspiration_id}
Authorization: Bearer <token>
```

| Parameter | Type | In | Description |
|-----------|------|-----|-------------|
| `inspiration_id` | string (UUID) | path | The inspiration to delete |

### Response `200`

```json
{
  "deleted": true,
  "taste_vector_updated": true,
  "remaining_count": 4
}
```

| Field | Type | Description |
|-------|------|-------------|
| `deleted` | bool | Whether the inspiration was found and removed |
| `taste_vector_updated` | bool | Whether the taste vector was recomputed (false if 0 remaining) |
| `remaining_count` | int | How many inspirations the user has left |

### Errors

| Code | Detail | Cause |
|------|--------|-------|
| 404 | `"Inspiration not found"` | Wrong ID or doesn't belong to this user |

---

## GET `/api/canvas/style-elements`

Aggregates style attributes across **all** of the user's inspirations and returns them mapped to the feed endpoint's `include_*` query parameter names. The frontend can pass `suggested_filters` directly to `GET /api/recs/v2/feed` as query params.

### Request

```
GET /api/canvas/style-elements
Authorization: Bearer <token>
```

No query parameters.

### Response `200`

```json
{
  "suggested_filters": {
    "include_style_tags": ["Sporty", "Trendy"],
    "include_patterns": ["Solid"],
    "include_color_family": ["Greens", "Blues"],
    "include_formality": ["Casual"],
    "include_occasions": ["Workout", "Everyday", "Lounging"],
    "include_silhouette": ["Fitted", "Slim"],
    "include_fit": ["Slim", "Fitted"],
    "include_sleeves": ["Sleeveless"],
    "include_neckline": ["Scoop"]
  },
  "raw_attributes": {
    "style_tags": [
      { "value": "Sporty", "count": 2, "confidence": 0.282 },
      { "value": "Trendy", "count": 1, "confidence": 0.256 },
      { "value": "Casual", "count": 2, "confidence": 0.231 }
    ],
    "pattern": [
      { "value": "Solid", "count": 3, "confidence": 0.818 }
    ]
  },
  "inspiration_count": 5
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `suggested_filters` | object | Pre-built filter dict. Keys are feed query-param names (e.g. `include_style_tags`). Values are string arrays of the top attribute values above a confidence threshold. **Pass these directly as query params to `GET /api/recs/v2/feed`.** |
| `raw_attributes` | object | Full distributions per attribute category. Each key maps to an array of `{ value, count, confidence }` objects sorted by confidence descending. Use this for rendering toggle chips in the UI. |
| `inspiration_count` | int | How many inspirations were aggregated |

### Suggested Filters → Feed Query Params

The keys in `suggested_filters` map directly to the feed endpoint:

```
GET /api/recs/v2/feed?include_style_tags=Sporty,Trendy&include_patterns=Solid&include_color_family=Greens,Blues&include_formality=Casual&include_occasions=Workout,Everyday,Lounging&include_silhouette=Fitted,Slim&include_fit=Slim,Fitted&include_sleeves=Sleeveless&include_neckline=Scoop
```

Users can toggle individual attribute chips on/off in the UI. The frontend removes toggled-off values from the query params before calling the feed.

### Attribute Key → Feed Param Mapping

| Attribute Key | Feed Query Param |
|---------------|-----------------|
| `style_tags` | `include_style_tags` |
| `pattern` | `include_patterns` |
| `color_family` | `include_color_family` |
| `formality` | `include_formality` |
| `occasions` | `include_occasions` |
| `silhouette` | `include_silhouette` |
| `fit_type` | `include_fit` |
| `sleeve_type` | `include_sleeves` |
| `neckline` | `include_neckline` |

---

## GET `/api/canvas/inspirations/{inspiration_id}/similar`

Find products visually similar to an inspiration image. Uses FashionCLIP embedding cosine similarity via pgvector. Results are fully deduplicated (5-pass pipeline) and brand-diversity-capped.

### Request

```
GET /api/canvas/inspirations/{inspiration_id}/similar?count=12
Authorization: Bearer <token>
```

| Parameter | Type | In | Default | Description |
|-----------|------|-----|---------|-------------|
| `inspiration_id` | string (UUID) | path | — | The inspiration to find similar products for |
| `count` | int | query | 12 | Number of results (1–30) |

### Response `200`

```json
{
  "products": [
    {
      "product_id": "5d0c738c-94d1-4831-9e67-a49f424ca5b4",
      "name": "Stripe Mid Length Bike Short 601",
      "brand": "Aje",
      "category": "bottoms",
      "broad_category": null,
      "colors": ["Velocity Green"],
      "materials": ["77% Nylon", "23% Elastane"],
      "price": 70.00,
      "fit": null,
      "length": null,
      "sleeve": null,
      "neckline": null,
      "style_tags": [],
      "primary_image_url": "https://usepose.s3.us-east-1.amazonaws.com/products/aje/.../gallery_1.jpg",
      "hero_image_url": null,
      "similarity": 0.859,
      "image_url": "https://usepose.s3.us-east-1.amazonaws.com/products/aje/.../gallery_1.jpg"
    },
    {
      "product_id": "3067635a-18ca-4b26-a9a2-15a3549d6e84",
      "name": "SoftActive Flare Leggings",
      "brand": "Garage",
      "category": "bottoms",
      "colors": ["Black", "Lush Green", "Martini Green"],
      "materials": ["75% nylon", "25% spandex"],
      "price": 40.00,
      "primary_image_url": "https://usepose.s3.us-east-1.amazonaws.com/products/garage/.../primary.jpg",
      "hero_image_url": null,
      "similarity": 0.800
    }
  ],
  "count": 2,
  "inspiration_id": "2863221a-fdd6-43a1-bd98-3921afbffb71"
}
```

### Product Fields

| Field | Type | Description |
|-------|------|-------------|
| `product_id` | string (UUID) | Product ID — use for `GET /api/recs/similar/{product_id}` or product detail |
| `name` | string | Product name |
| `brand` | string | Brand name |
| `category` | string | Product category (`"tops"`, `"bottoms"`, `"dresses"`, etc.) |
| `colors` | string[] | Available colors |
| `materials` | string[] | Material composition |
| `price` | float | Price in USD |
| `primary_image_url` | string | Main product image |
| `hero_image_url` | string \| null | Hero/editorial image if available |
| `similarity` | float | Cosine similarity to the inspiration (0.0–1.0). Higher = more similar. |

### Deduplication

Results go through a 5-pass dedup pipeline:

1. **Product ID** — removes duplicate rows from multi-image products
2. **Sister-brand + size-variant + image URL** — catches cross-brand clones (Boohoo/NastyGal/PLT), size variants ("Petite X" vs "X"), and shared product photos
3. **Image hash** — catches cross-brand products sharing the same photo URL pattern
4. **Fuzzy name** — catches near-duplicate names within the same brand group (e.g. "SoftActive Flare Leggings" vs "Active Flare Leggings")
5. **Brand cap** — max 3 items per brand to ensure variety

### Errors

| Code | Detail | Cause |
|------|--------|-------|
| 404 | `"Inspiration not found or no similar products"` | Invalid ID, doesn't belong to user, or no matches |

---

## POST `/api/canvas/inspirations/{inspiration_id}/complete-fit`

Find the closest real product to an inspiration image, then build a complete outfit around it using the TATTOO scoring engine.

### Request

```
POST /api/canvas/inspirations/{inspiration_id}/complete-fit
Authorization: Bearer <token>
Content-Type: application/json
```

| Parameter | Type | In | Description |
|-----------|------|-----|-------------|
| `inspiration_id` | string (UUID) | path | The inspiration to build an outfit from |

```json
{
  "items_per_category": 4,
  "category": null,
  "offset": 0,
  "limit": null
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `items_per_category` | int | 4 | Items per category in carousel mode (1–20) |
| `category` | string \| null | null | Target category for feed mode (e.g. `"tops"`, `"outerwear"`). Omit for carousel mode (all categories). |
| `offset` | int | 0 | Skip first N items (feed mode) |
| `limit` | int \| null | null | Max items to return (feed mode, 1–100) |

All fields are optional. Send `{}` for defaults.

### Response `200`

```json
{
  "matched_product": {
    "product_id": "5d0c738c-94d1-4831-9e67-a49f424ca5b4",
    "name": "Stripe Mid Length Bike Short 601",
    "brand": "Aje",
    "category": "bottoms",
    "price": 70.00,
    "primary_image_url": "https://...",
    "similarity": 0.859
  },
  "outfit": {
    "status": "ok",
    "source_product": { ... },
    "recommendations": {
      "tops": {
        "items": [
          {
            "product_id": "...",
            "name": "Ribbed Crop Tank",
            "brand": "Alo Yoga",
            "price": 48.00,
            "primary_image_url": "https://...",
            "hero_image_url": null,
            "tattoo_score": 0.82,
            "compatibility_score": 0.78,
            "final_score": 0.80
          }
        ],
        "count": 4
      },
      "outerwear": {
        "items": [ ... ],
        "count": 4
      }
    },
    "complete_outfit": {
      "item_count": 3,
      "total_price": 206.00
    },
    "scoring_info": { ... }
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `matched_product` | object | The closest real product to the inspiration image (same fields as similar products) |
| `outfit` | object | Full outfit builder response from the TATTOO engine |
| `outfit.status` | string | `"ok"` or error status |
| `outfit.recommendations` | object | Keyed by category (`"tops"`, `"outerwear"`, etc.), each with an `items` array |
| `outfit.complete_outfit` | object | Summary: `item_count`, `total_price` |

### Errors

| Code | Detail | Cause |
|------|--------|-------|
| 404 | `"Inspiration not found or no matching products"` | Invalid ID or no embedding match |
| 503 | `"Outfit engine not available: ..."` | Outfit engine failed to load |

---

## Error Responses

All errors return JSON with a `detail` field:

```json
{
  "detail": "Authorization header required"
}
```

### Common Errors

| Code | Detail | Cause |
|------|--------|-------|
| 401 | `"Authorization header required"` | Missing `Authorization` header |
| 401 | `"Invalid token: ..."` | Expired, malformed, or invalid JWT |
| 404 | Various | Resource not found |
| 409 | `"Inspiration quota exceeded (max 50)"` | Too many inspirations |
| 413 | `"File too large. Maximum size: 10 MB"` | Upload too big |
| 415 | `"Unsupported file type: ..."` | Wrong image format |

---

## Integration Recipes

### Recipe 1: Upload + Show Style

```
1. POST /api/canvas/inspirations/upload  (file)
   → Get back InspirationResponse with style_label, style_attributes

2. Show the image + style chips from style_attributes in the UI
```

### Recipe 2: Canvas → Personalized Feed

```
1. GET /api/canvas/style-elements
   → Get suggested_filters

2. Render attribute chips (toggle on/off)

3. Build query string from active chips:
   GET /api/recs/v2/feed?include_style_tags=Sporty,Trendy&include_patterns=Solid&...

4. User toggles a chip off → remove that param → re-fetch feed
```

### Recipe 3: Inspiration → Similar Products Grid

```
1. GET /api/canvas/inspirations/{id}/similar?count=12
   → Get 12 deduplicated similar products

2. Render as a product grid (use primary_image_url, name, brand, price, similarity)

3. User taps a product → navigate to product detail (product_id)
```

### Recipe 4: Inspiration → Complete Outfit

```
1. POST /api/canvas/inspirations/{id}/complete-fit
   Body: {}
   → Get matched_product + outfit.recommendations

2. Show matched_product as the "hero" item

3. Show outfit.recommendations by category as horizontal carousels:
   - tops: [item, item, item, item]
   - outerwear: [item, item, item, item]

4. Show outfit.complete_outfit.total_price as the outfit total
```

### Recipe 5: Pinterest Import → Full Experience

```
1. POST /api/canvas/inspirations/pinterest
   Body: { "max_pins": 30 }
   → Array of InspirationResponse objects

2. GET /api/canvas/inspirations
   → Full canvas with all inspirations

3. GET /api/canvas/style-elements
   → Aggregated style profile from all pins

4. For each inspiration the user taps:
   GET /api/canvas/inspirations/{id}/similar?count=12
```

---

## Rate Limits & Constraints

| Constraint | Value |
|------------|-------|
| Max inspirations per user | 50 |
| Max upload file size | 10 MB |
| Allowed image types | JPEG, PNG, WebP, GIF |
| Max similar products per request | 30 |
| Max Pinterest pins per sync | 200 |
| Similar items brand cap | 3 per brand |
| Embedding model | FashionCLIP (512-dim) |
| Style classification | K=20 nearest-neighbor via pgvector |
