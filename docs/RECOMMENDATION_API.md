# Recommendation API Documentation

## Overview

The Recommendation API provides personalized fashion recommendations using:
- **FashionCLIP embeddings** (512-dim vectors) for visual similarity
- **pgvector** (Supabase) for fast similarity search
- **Tinder-style preference learning** for cold-start personalization

**Base URL:** `http://localhost:8080/api/recs`

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Authentication](#authentication)
3. [Endpoints](#endpoints)
   - [Health Check](#health-check)
   - [Save Preferences](#save-preferences)
   - [Get Personalized Feed](#get-personalized-feed)
   - [Get Similar Products](#get-similar-products)
   - [Get Trending Products](#get-trending-products)
   - [Get Categories](#get-categories)
   - [Get Product Details](#get-product-details)
4. [Integration Flow](#integration-flow)
5. [Response Codes](#response-codes)
6. [Data Models](#data-models)

---

## Quick Start

```bash
# 1. Get trending products (no user required)
curl "http://localhost:8080/api/recs/trending?gender=female&limit=10"

# 2. Save user preferences after Tinder test
curl -X POST "http://localhost:8080/api/recs/save-preferences" \
  -H "Content-Type: application/json" \
  -d '{
    "anon_id": "user_123",
    "gender": "female",
    "rounds_completed": 12,
    "taste_vector": [0.123, -0.456, ...]
  }'

# 3. Get personalized recommendations
curl "http://localhost:8080/api/recs/feed/user_123?gender=female&limit=20"
```

---

## Authentication

Currently, the API uses anonymous user IDs (`anon_id`) or authenticated user UUIDs (`user_id`). No API key is required for development.

| Parameter | Description |
|-----------|-------------|
| `user_id` | UUID for logged-in users (from auth system) |
| `anon_id` | String identifier for anonymous users |

**Note:** At least one of `user_id` or `anon_id` must be provided for user-specific endpoints.

---

## Endpoints

### Health Check

Check if the recommendation service is healthy and connected to Supabase.

```
GET /api/recs/health
```

**Response:**
```json
{
  "status": "healthy",
  "supabase_connected": true,
  "categories_count": 4,
  "total_products": 52692
}
```

---

### Save Preferences

Save user preferences from the Tinder-style test. This enables personalized recommendations.

```
POST /api/recs/save-preferences
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string (UUID) | No* | Logged-in user ID |
| `anon_id` | string | No* | Anonymous user ID |
| `gender` | string | No | `"female"` or `"male"` (default: `"female"`) |
| `rounds_completed` | integer | No | Number of Tinder rounds completed |
| `categories_tested` | array[string] | No | Categories shown during test |
| `attribute_preferences` | object | No | Learned attribute preferences |
| `prediction_accuracy` | float | No | Model prediction accuracy (0-1) |
| `taste_vector` | array[float] | No | 512-dim FashionCLIP embedding |

*At least one of `user_id` or `anon_id` is required.

**Example Request:**
```json
{
  "anon_id": "user_abc123",
  "gender": "female",
  "rounds_completed": 12,
  "categories_tested": ["tops_knitwear", "dresses", "bottoms_trousers"],
  "attribute_preferences": {
    "pattern": {
      "preferred": [["solid", 0.85], ["floral", 0.72]],
      "avoided": [["animal_print", 0.2]]
    },
    "style": {
      "preferred": [["casual", 0.9], ["minimalist", 0.75]],
      "avoided": []
    },
    "fit_vibe": {
      "preferred": [["relaxed", 0.8]],
      "avoided": [["oversized", 0.3]]
    }
  },
  "prediction_accuracy": 0.667,
  "taste_vector": [0.0234, -0.1567, 0.0891, ...]
}
```

**Response:**
```json
{
  "status": "success",
  "preference_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "user_id": "user_abc123",
  "seed_source": "tinder"
}
```

**`seed_source` Values:**

| Value | Description |
|-------|-------------|
| `tinder` | Taste vector provided (personalized recommendations enabled) |
| `attributes` | Only attribute preferences saved (limited personalization) |

---

### Get Personalized Feed

Get personalized product recommendations based on user's taste vector.

```
GET /api/recs/feed/{user_id}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | string | User ID (UUID or anon_id) |

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gender` | string | `"female"` | Filter by gender |
| `categories` | string | None | Comma-separated category filter |
| `limit` | integer | 50 | Number of results (1-200) |
| `offset` | integer | 0 | Number of results to skip (pagination) |

**Example Request:**
```bash
# First page
curl "http://localhost:8080/api/recs/feed/user_abc123?gender=female&limit=20&offset=0"

# Second page
curl "http://localhost:8080/api/recs/feed/user_abc123?gender=female&limit=20&offset=20"
```

**Response:**
```json
{
  "user_id": "user_abc123",
  "strategy": "seed_vector",
  "results": [
    {
      "product_id": "5847ba3a-1234-5678-90ab-cdef12345678",
      "name": "Oversized Knit Sweater",
      "brand": "Zara",
      "category": "tops",
      "gender": ["female"],
      "price": 49.99,
      "primary_image_url": "https://...",
      "hero_image_url": "https://...",
      "similarity": 0.892,
      "reason": "style_matched"
    },
    {
      "product_id": "6958cb4b-2345-6789-01bc-def123456789",
      "name": "Wide Leg Trousers",
      "brand": "H&M",
      "category": "bottoms",
      "gender": ["female"],
      "price": 34.99,
      "primary_image_url": "https://...",
      "similarity": 0.856,
      "reason": "style_matched"
    }
  ],
  "metadata": {
    "candidates_retrieved": 20,
    "seed_vector_available": true,
    "seed_source": "tinder",
    "offset": 0,
    "has_more": true
  }
}
```

**`strategy` Values:**

| Strategy | Description |
|----------|-------------|
| `seed_vector` | Personalized using user's taste vector (best) |
| `trending` | Fallback to trending products (cold start) |

**`reason` Values:**

| Reason | Description |
|--------|-------------|
| `style_matched` | Matched via taste vector similarity |
| `trending` | Popular/trending product |

---

### Get Similar Products

Find products visually similar to a given product.

```
GET /api/recs/similar/{product_id}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `product_id` | string (UUID) | Source product ID |

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gender` | string | `"female"` | Filter by gender |
| `category` | string | None | Filter by category |
| `limit` | integer | 20 | Number of results (1-100) |

**Example Request:**
```bash
curl "http://localhost:8080/api/recs/similar/5847ba3a-1234-5678-90ab-cdef12345678?limit=10"
```

**Response:**
```json
{
  "product_id": "5847ba3a-1234-5678-90ab-cdef12345678",
  "similar": [
    {
      "product_id": "7069dc5c-3456-7890-12cd-ef1234567890",
      "name": "Cable Knit Pullover",
      "brand": "Mango",
      "category": "tops",
      "gender": ["female"],
      "price": 59.99,
      "primary_image_url": "https://...",
      "similarity": 0.912
    },
    {
      "product_id": "8170ed6d-4567-8901-23de-f12345678901",
      "name": "Chunky Wool Sweater",
      "brand": "COS",
      "category": "tops",
      "gender": ["female"],
      "price": 89.99,
      "primary_image_url": "https://...",
      "similarity": 0.887
    }
  ]
}
```

**Error Response (Product Not Found):**
```json
{
  "detail": "Product not found: invalid-uuid"
}
```

---

### Get Trending Products

Get trending/popular products. Used as fallback for cold-start users.

```
GET /api/recs/trending
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gender` | string | `"female"` | Filter by gender |
| `category` | string | None | Filter by category |
| `limit` | integer | 50 | Number of results (1-200) |
| `offset` | integer | 0 | Number of results to skip (pagination) |

**Example Request:**
```bash
# First page
curl "http://localhost:8080/api/recs/trending?gender=female&limit=20&offset=0"

# Second page
curl "http://localhost:8080/api/recs/trending?gender=female&limit=20&offset=20"
```

**Response:**
```json
{
  "gender": "female",
  "category": "dresses",
  "results": [
    {
      "product_id": "9281fe7e-5678-9012-34ef-012345678901",
      "name": "Floral Midi Dress",
      "brand": "Reformation",
      "category": "dresses",
      "gender": ["female"],
      "price": 128.00,
      "primary_image_url": "https://...",
      "trending_score": 0.95
    }
  ],
  "count": 20,
  "offset": 0,
  "has_more": true
}
```

---

### Get Categories

Get available product categories with counts.

```
GET /api/recs/categories
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gender` | string | `"female"` | Filter by gender |

**Example Request:**
```bash
curl "http://localhost:8080/api/recs/categories?gender=female"
```

**Response:**
```json
[
  {"category": "tops", "product_count": 19978},
  {"category": "bottoms", "product_count": 15946},
  {"category": "dresses", "product_count": 12850},
  {"category": "outerwear", "product_count": 3918}
]
```

---

### Get Product Details

Get details for a single product.

```
GET /api/recs/product/{product_id}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `product_id` | string (UUID) | Product ID |

**Example Request:**
```bash
curl "http://localhost:8080/api/recs/product/5847ba3a-1234-5678-90ab-cdef12345678"
```

**Response:**
```json
{
  "id": "5847ba3a-1234-5678-90ab-cdef12345678",
  "name": "Oversized Knit Sweater",
  "brand": "Zara",
  "category": "tops",
  "gender": ["female"],
  "price": 49.99,
  "primary_image_url": "https://...",
  "hero_image_url": "https://...",
  "in_stock": true
}
```

---

## Integration Flow

### Complete User Journey

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER ONBOARDING FLOW                        │
└─────────────────────────────────────────────────────────────────┘

1. NEW USER ARRIVES
   │
   ▼
2. START TINDER-STYLE TEST
   POST /api/women/session/start
   │
   ▼
3. USER MAKES CHOICES (8-15 rounds)
   POST /api/women/session/choose
   │
   ▼
4. GET SESSION SUMMARY (includes taste_vector)
   GET /api/women/session/{user_id}/summary
   │
   ▼
5. SAVE TO RECOMMENDATION SYSTEM
   POST /api/recs/save-preferences
   {
     "anon_id": "user_123",
     "taste_vector": [...512 floats...],
     "attribute_preferences": {...}
   }
   │
   ▼
6. GET PERSONALIZED FEED
   GET /api/recs/feed/user_123
   → Returns personalized recommendations (strategy: "seed_vector")


┌─────────────────────────────────────────────────────────────────┐
│                    RETURNING USER FLOW                          │
└─────────────────────────────────────────────────────────────────┘

1. USER RETURNS
   │
   ▼
2. GET PERSONALIZED FEED (preferences already saved)
   GET /api/recs/feed/user_123
   │
   ▼
3. USER VIEWS PRODUCT
   │
   ▼
4. GET SIMILAR PRODUCTS
   GET /api/recs/similar/{product_id}


┌─────────────────────────────────────────────────────────────────┐
│                    COLD START FLOW                              │
└─────────────────────────────────────────────────────────────────┘

1. NEW USER (no preferences)
   │
   ▼
2. GET FEED → Returns trending products
   GET /api/recs/feed/new_user
   → strategy: "trending"
   │
   ▼
3. PROMPT USER TO TAKE STYLE TEST
```

### Code Example (JavaScript)

```javascript
// 1. After Tinder test completion, save preferences
async function saveUserPreferences(userId, testSummary) {
  const response = await fetch('/api/recs/save-preferences', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      anon_id: userId,
      gender: 'female',
      rounds_completed: testSummary.rounds_completed,
      categories_tested: testSummary.categories_tested,
      attribute_preferences: testSummary.attribute_preferences,
      prediction_accuracy: testSummary.prediction_accuracy,
      taste_vector: testSummary.taste_vector  // 512-dim array
    })
  });

  return response.json();
}

// 2. Get personalized feed
async function getPersonalizedFeed(userId, options = {}) {
  const params = new URLSearchParams({
    gender: options.gender || 'female',
    limit: options.limit || 50
  });

  if (options.categories) {
    params.append('categories', options.categories.join(','));
  }

  const response = await fetch(`/api/recs/feed/${userId}?${params}`);
  const data = await response.json();

  // Check if personalized
  if (data.strategy === 'seed_vector') {
    console.log('Personalized recommendations!');
  } else {
    console.log('Showing trending (user needs to complete style test)');
  }

  return data.results;
}

// 3. Get similar products when user views a product
async function getSimilarProducts(productId) {
  const response = await fetch(`/api/recs/similar/${productId}?limit=10`);
  const data = await response.json();
  return data.similar;
}
```

### Code Example (Python)

```python
import requests

BASE_URL = "http://localhost:8080/api/recs"

# 1. Save preferences after Tinder test
def save_preferences(user_id: str, test_summary: dict):
    response = requests.post(f"{BASE_URL}/save-preferences", json={
        "anon_id": user_id,
        "gender": "female",
        "rounds_completed": test_summary.get("rounds_completed"),
        "categories_tested": test_summary.get("categories_tested"),
        "attribute_preferences": test_summary.get("attribute_preferences"),
        "prediction_accuracy": test_summary.get("prediction_accuracy"),
        "taste_vector": test_summary.get("taste_vector")  # 512-dim list
    })
    return response.json()

# 2. Get personalized feed
def get_feed(user_id: str, gender: str = "female", limit: int = 50):
    response = requests.get(
        f"{BASE_URL}/feed/{user_id}",
        params={"gender": gender, "limit": limit}
    )
    data = response.json()

    print(f"Strategy: {data['strategy']}")
    print(f"Results: {len(data['results'])} products")

    return data["results"]

# 3. Get similar products
def get_similar(product_id: str, limit: int = 10):
    response = requests.get(
        f"{BASE_URL}/similar/{product_id}",
        params={"limit": limit}
    )
    return response.json()["similar"]

# Example usage
if __name__ == "__main__":
    # After user completes Tinder test
    test_summary = {
        "rounds_completed": 12,
        "categories_tested": ["tops_knitwear", "dresses"],
        "attribute_preferences": {"style": {"preferred": [["casual", 0.9]]}},
        "prediction_accuracy": 0.75,
        "taste_vector": [0.1, -0.2, ...]  # 512 floats
    }

    save_preferences("user_123", test_summary)

    # Get personalized recommendations
    products = get_feed("user_123", limit=20)
    for p in products[:5]:
        print(f"{p['name']} - ${p['price']} (sim: {p.get('similarity', 'N/A')})")
```

---

## Response Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `400` | Bad Request - Missing required parameters |
| `404` | Not Found - Product or user not found |
| `500` | Internal Server Error |

**Error Response Format:**
```json
{
  "detail": "Error message describing the issue"
}
```

---

## Data Models

### Product

```typescript
interface Product {
  product_id: string;      // UUID
  name: string;
  brand: string | null;
  category: string;
  gender: string[];        // ["female"] or ["male"] or ["female", "male"]
  price: number;
  primary_image_url: string;
  hero_image_url: string | null;
  similarity?: number;     // 0-1, present in feed/similar responses
  trending_score?: number; // 0-1, present in trending responses
  reason?: string;         // "style_matched" | "trending"
}
```

### FeedResponse

```typescript
interface FeedResponse {
  user_id: string;
  strategy: "seed_vector" | "trending";
  results: Product[];
  metadata: {
    candidates_retrieved: number;
    seed_vector_available: boolean;
    seed_source: "tinder" | "attributes" | "none";
  };
}
```

### AttributePreferences

```typescript
interface AttributePreferences {
  [attribute: string]: {
    preferred: [string, number][];  // [["solid", 0.85], ["floral", 0.72]]
    avoided: [string, number][];    // [["animal_print", 0.2]]
  };
}

// Available attributes:
// - pattern: solid, striped, floral, geometric, animal_print, plaid, etc.
// - style: casual, elegant, minimalist, bohemian, sporty, etc.
// - color_family: neutral, bright, cool, pastel, dark
// - fit_vibe: fitted, relaxed, oversized, cropped
// - neckline: crew, v_neck, off_shoulder, sweetheart, halter
// - sleeve_type: sleeveless, short, long, puff
// - occasion: everyday, work, date_night, party, beach
```

---

## Rate Limits

Currently no rate limits are enforced. For production:
- Recommended: 100 requests/minute per user
- Batch operations: Use pagination with reasonable limits

---

## Changelog

### v1.0.0 (2026-01-10)
- Initial release
- Endpoints: save-preferences, feed, similar, trending, categories, product
- Supabase pgvector integration
- Tinder-style preference learning support
