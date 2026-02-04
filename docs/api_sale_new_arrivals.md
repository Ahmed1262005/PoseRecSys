# Sale & New Arrivals API Documentation

## Overview

Two personalized feed endpoints for discovering sale items and new arrivals:

| Endpoint | Description |
|----------|-------------|
| `GET /api/recs/v2/sale` | Picks on Sale - Personalized sale items |
| `GET /api/recs/v2/new-arrivals` | Just In - Personalized new arrivals |

Both endpoints use the same recommendation pipeline as `/api/recs/v2/feed`, with all filters and personalization features available.

---

## Authentication

All endpoints require either `user_id` or `anon_id` parameter.

---

## GET /api/recs/v2/sale

Returns personalized sale items (products where `original_price > price`).

### Parameters

#### Required (one of)
| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | string | UUID of registered user |
| `anon_id` | string | Anonymous user identifier |

#### Pagination
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cursor` | string | null | Cursor from previous response for next page |
| `page_size` | int | 50 | Items per page (1-200) |
| `session_id` | string | auto | Session ID for tracking seen items |

#### Filters
| Parameter | Type | Example | Description |
|-----------|------|---------|-------------|
| `gender` | string | "female" | Gender filter (default: "female") |
| `categories` | string | "tops,dresses" | Comma-separated broad categories |
| `article_types` | string | "t-shirts,blouses" | Comma-separated article types |
| `include_occasions` | string | "office,date-night" | Filter by occasion |
| `exclude_styles` | string | "bohemian,edgy" | Exclude style categories |
| `min_price` | float | 20.00 | Minimum price |
| `max_price` | float | 100.00 | Maximum price |
| `include_brands` | string | "Zara,H&M" | Only include these brands |
| `exclude_brands` | string | "Shein" | Exclude these brands |
| `include_colors` | string | "black,white" | Filter by colors |
| `exclude_colors` | string | "orange,yellow" | Exclude colors |
| `include_patterns` | string | "solid,striped" | Filter by patterns |
| `exclude_patterns` | string | "floral,animal" | Exclude patterns |

### Response

```json
{
  "user_id": "anon_abc123",
  "session_id": "sess_xyz789",
  "cursor": "eyJzY29yZSI6IDAuNzUsICJpdGVtX2lkIjogIjEyMzQifQ==",
  "strategy": "exploration",
  "results": [
    {
      "product_id": "f2165d58-7ce1-4664-af34-1d8b5962939b",
      "rank": 1,
      "score": 0.75,
      "reason": "explore",
      "category": "dresses",
      "broad_category": "dresses",
      "brand": "Banana Republic",
      "name": "Silk Midi Dress",
      "price": 89.99,
      "original_price": 149.99,
      "is_on_sale": true,
      "discount_percent": 40,
      "is_new": false,
      "image_url": "https://...",
      "gallery_images": ["https://...", "https://..."],
      "colors": ["navy", "black"],
      "source": "exploration"
    }
  ],
  "pagination": {
    "page": 0,
    "page_size": 50,
    "items_returned": 50,
    "has_more": true
  },
  "metadata": {
    "candidates_retrieved": 105,
    "keyset_pagination": true,
    "feed_version": "v_1234567890_abc123"
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `price` | float | Current sale price |
| `original_price` | float | Original price before discount |
| `is_on_sale` | boolean | Always `true` for this endpoint |
| `discount_percent` | int | Discount percentage (e.g., 40 for 40% off) |
| `is_new` | boolean | True if added in last 7 days |

### Examples

```bash
# Basic sale items
curl 'http://localhost:8000/api/recs/v2/sale?anon_id=user123&page_size=20'

# Sale dresses under $100
curl 'http://localhost:8000/api/recs/v2/sale?anon_id=user123&categories=dresses&max_price=100'

# Sale tops for office occasions
curl 'http://localhost:8000/api/recs/v2/sale?anon_id=user123&categories=tops&include_occasions=office'

# Sale items from specific brands
curl 'http://localhost:8000/api/recs/v2/sale?anon_id=user123&include_brands=Zara,Mango'

# Pagination - get next page
curl 'http://localhost:8000/api/recs/v2/sale?anon_id=user123&cursor=eyJzY29yZSI6...'
```

---

## GET /api/recs/v2/new-arrivals

Returns personalized new arrivals (products added in the last 7 days).

### Parameters

Same parameters as `/api/recs/v2/sale` (see above).

### Response

Same response format as `/api/recs/v2/sale`, but:
- `is_new` is always `true`
- Items ordered by most recent first

### Examples

```bash
# Basic new arrivals
curl 'http://localhost:8000/api/recs/v2/new-arrivals?anon_id=user123&page_size=20'

# New dresses
curl 'http://localhost:8000/api/recs/v2/new-arrivals?anon_id=user123&categories=dresses'

# New arrivals in price range
curl 'http://localhost:8000/api/recs/v2/new-arrivals?anon_id=user123&min_price=50&max_price=150'

# New arrivals excluding certain styles
curl 'http://localhost:8000/api/recs/v2/new-arrivals?anon_id=user123&exclude_styles=bohemian,sporty'
```

---

## Pagination

Both endpoints use **cursor-based pagination** for efficient infinite scroll.

### How It Works

1. **First request**: Don't include `cursor`
2. **Response**: Contains `cursor` string and `has_more` boolean
3. **Next page**: Include the `cursor` from previous response
4. **Continue**: Until `has_more` is `false`

### Example Flow

```bash
# Page 1
curl 'http://localhost:8000/api/recs/v2/sale?anon_id=user123&page_size=20'
# Response: { "cursor": "abc123...", "has_more": true, "results": [...] }

# Page 2
curl 'http://localhost:8000/api/recs/v2/sale?anon_id=user123&page_size=20&cursor=abc123...'
# Response: { "cursor": "def456...", "has_more": true, "results": [...] }

# Continue until has_more=false
```

### Session Tracking

Include `session_id` from the first response to maintain session state:

```bash
# First request - session_id auto-generated
curl '...?anon_id=user123'
# Response: { "session_id": "sess_xyz789", ... }

# Subsequent requests - include session_id
curl '...?anon_id=user123&session_id=sess_xyz789&cursor=...'
```

---

## Filtering

### Categories

Filter by broad product categories:

| Category | Description |
|----------|-------------|
| `tops` | All tops (t-shirts, blouses, sweaters) |
| `dresses` | Dresses and jumpsuits |
| `bottoms` | Pants, jeans, shorts |
| `skirts` | Skirts and skorts |
| `outerwear` | Jackets and coats |

```bash
# Single category
?categories=dresses

# Multiple categories
?categories=tops,dresses,skirts
```

### Article Types

Filter by specific article types:

```bash
# Specific types
?article_types=t-shirts,blouses

# Combined with category
?categories=tops&article_types=sweaters,cardigans
```

### Occasions

Filter by occasion suitability:

| Occasion | Description |
|----------|-------------|
| `everyday` | Casual daily wear |
| `office` | Work-appropriate |
| `date-night` | Evening/romantic |
| `party` | Going out |
| `vacation` | Resort/travel |

```bash
?include_occasions=office,everyday
```

### Styles

Exclude style categories you want to avoid:

| Style | Description |
|-------|-------------|
| `bohemian` | Boho/free-spirited |
| `edgy` | Bold/alternative |
| `romantic` | Feminine/soft |
| `minimal` | Clean/simple |
| `sporty` | Athletic-inspired |

```bash
?exclude_styles=bohemian,sporty
```

### Price Range

```bash
# Minimum price
?min_price=50

# Maximum price
?max_price=200

# Price range
?min_price=50&max_price=200
```

### Brands

```bash
# Only specific brands
?include_brands=Zara,Mango,H%26M

# Exclude brands
?exclude_brands=Shein,Forever21
```

### Colors

```bash
# Only these colors
?include_colors=black,white,navy

# Exclude colors
?exclude_colors=orange,neon
```

### Patterns

```bash
# Only these patterns
?include_patterns=solid,striped

# Exclude patterns
?exclude_patterns=floral,animal,geometric
```

---

## Combining Filters

All filters can be combined:

```bash
curl 'http://localhost:8000/api/recs/v2/sale?\
anon_id=user123&\
categories=tops,dresses&\
include_occasions=office&\
exclude_styles=bohemian&\
min_price=30&\
max_price=150&\
include_colors=black,white,navy&\
exclude_patterns=floral&\
page_size=20'
```

---

## Personalization

Both endpoints automatically apply personalization based on user profile:

- **Preferred brands**: Boosted in results
- **Style preferences**: Matched items ranked higher
- **Occasion preferences**: Filtered and scored
- **Fit/length preferences**: Soft scoring applied

For registered users (`user_id`), personalization uses saved onboarding profile.
For anonymous users (`anon_id`), personalization builds over session interactions.

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Either user_id or anon_id must be provided"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Internal server error"
}
```

---

## Rate Limits

- No explicit rate limits currently enforced
- Recommended: Max 10 requests/second per user

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-26 | Initial release of sale and new-arrivals endpoints |
