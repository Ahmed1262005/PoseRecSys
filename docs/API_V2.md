# Recommendation API V2 Documentation

Base URL: `/api/recs/v2`

## Overview

The V2 API provides personalized product recommendations with:
- **Keyset pagination** for O(1) infinite scroll performance
- **Session-based dedup** - no duplicates within a session
- **User interaction tracking** for ML training
- **Multiple personalization strategies** (cold start → warm user)

---

## Authentication

All endpoints accept user identification via:
- `anon_id` - Anonymous user identifier (string)
- `user_id` - UUID for authenticated users

At least one must be provided.

---

## Endpoints

### Feed Endpoints

#### GET /feed
Get personalized product recommendations with infinite scroll support.

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `anon_id` | string | Yes* | - | Anonymous user ID |
| `user_id` | string | Yes* | - | UUID user ID |
| `session_id` | string | No | auto | Session ID (return this in subsequent requests) |
| `gender` | string | No | "female" | Gender filter: "female" or "male" |
| `categories` | string | No | - | Comma-separated category filter |
| `cursor` | string | No | - | Pagination cursor from previous response |
| `page_size` | int | No | 50 | Items per page (1-200) |

*At least one of `anon_id` or `user_id` required.

**Response:**
```json
{
  "session_id": "sess_abc123",
  "results": [
    {
      "product_id": "uuid-xxx",
      "name": "Product Name",
      "brand": "Brand",
      "category": "dresses",
      "broad_category": "dresses",
      "price": 99.00,
      "primary_image_url": "https://...",
      "hero_image_url": "https://...",
      "colors": ["black", "white"],
      "materials": ["cotton"],
      "similarity": 0.85,
      "final_score": 0.82,
      "ranking_reason": "embedding_similarity"
    }
  ],
  "cursor": "eyJzY29yZSI6MC44NSwiaXRlbV9pZCI6Inh4eCJ9",
  "has_more": true,
  "metadata": {
    "user_state": "tinder_complete",
    "strategy": "embedding_similarity",
    "page_size": 50,
    "results_count": 50,
    "seen_count": 50
  }
}
```

**Pagination Flow:**
```
1. First request:
   GET /feed?anon_id=user123&page_size=50
   → Returns items + session_id + cursor

2. Next page:
   GET /feed?anon_id=user123&session_id=sess_xxx&cursor=eyJ...
   → Returns next items + new cursor

3. Repeat until has_more=false
```

**User States:**
| State | Description | Strategy |
|-------|-------------|----------|
| `cold_start` | No Tinder test completed | Trending items |
| `tinder_complete` | Has taste_vector from onboarding | Embedding similarity |
| `warm_user` | 5+ interactions | SASRec + embedding + preferences |

---

### User Interaction Tracking

#### POST /feed/action
Record an explicit user interaction with a product.

**Request Body:**
```json
{
  "anon_id": "user123",
  "user_id": null,
  "session_id": "sess_abc123",
  "product_id": "uuid-xxx",
  "action": "add_to_wishlist",
  "source": "feed",
  "position": 5
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `anon_id` | string | Yes* | Anonymous user ID |
| `user_id` | string | Yes* | UUID user ID |
| `session_id` | string | Yes | Session ID from feed response |
| `product_id` | string | Yes | Product UUID |
| `action` | string | Yes | Action type (see below) |
| `source` | string | No | Where interaction happened |
| `position` | int | No | Position in feed |

*At least one of `anon_id` or `user_id` required.

**Valid Actions:**
| Action | Signal Strength | Description |
|--------|-----------------|-------------|
| `click` | Medium | User tapped to view product details |
| `hover` | Light | User swiped through photo gallery |
| `add_to_wishlist` | Strong | User saved/liked the item |
| `add_to_cart` | Strong | User added to shopping cart |
| `purchase` | Strongest | User completed purchase |

**Valid Sources:**
- `feed` - Main recommendation feed
- `search` - Search results
- `similar` - Similar products section
- `style-this` - Style this item feature

**Response:**
```json
{
  "status": "success",
  "interaction_id": "uuid-interaction"
}
```

**Error Response (invalid action):**
```json
{
  "detail": "Invalid action 'like'. Must be one of: {'click', 'hover', 'add_to_wishlist', 'add_to_cart', 'purchase'}"
}
```

---

#### POST /session/sync
Sync session seen_ids for ML training data.

**Purpose:** Batch-persist which products were shown to the user. Used for negative sampling in ML training (items shown but not interacted with = implicit negatives).

**When to Call:**
- Every N pages (recommended: every 5 pages)
- On app close/background (`window.onbeforeunload`)
- On explicit session end

**Request Body:**
```json
{
  "anon_id": "user123",
  "user_id": null,
  "session_id": "sess_abc123",
  "seen_ids": [
    "uuid-product-1",
    "uuid-product-2",
    "uuid-product-3"
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `anon_id` | string | Yes* | Anonymous user ID |
| `user_id` | string | Yes* | UUID user ID |
| `session_id` | string | Yes | Session ID from feed response |
| `seen_ids` | string[] | Yes | Array of product UUIDs shown |

**Response:**
```json
{
  "status": "success",
  "synced_count": 150
}
```

---

### Session Management

#### GET /feed/session/{session_id}
Get information about a specific session (for debugging).

**Response:**
```json
{
  "session_id": "sess_abc123",
  "seen_count": 150,
  "user_signals": {
    "likes": ["uuid1", "uuid2"],
    "views": 150
  },
  "created_at": "2024-01-15T10:30:00Z",
  "last_access": "2024-01-15T11:45:00Z"
}
```

#### DELETE /feed/session/{session_id}
Clear a session to start fresh.

**Response:**
```json
{
  "status": "success",
  "session_id": "sess_abc123",
  "message": "Session cleared. Next feed request will show fresh items."
}
```

---

### Onboarding

#### POST /onboarding
Save complete 10-module onboarding profile.

**Request Body:**
```json
{
  "userId": "user123",
  "anonId": "anon456",
  "gender": "female",
  "core-setup": {
    "selectedCategories": ["tops", "dresses", "outerwear"],
    "sizes": ["S", "M"],
    "colorsToAvoid": ["orange", "yellow"],
    "materialsToAvoid": ["polyester"],
    "enabled": true
  },
  "tops": {
    "topTypes": ["blouses", "t-shirts"],
    "fits": ["regular", "relaxed"],
    "sleeves": ["short", "long"],
    "priceComfort": 75,
    "enabled": true
  },
  "style-discovery": {
    "roundsCompleted": 20,
    "sessionComplete": true,
    "summary": {
      "taste_vector": [0.1, 0.2, ...],
      "attribute_preferences": {}
    },
    "enabled": true
  }
}
```

**Response:**
```json
{
  "status": "success",
  "user_id": "user123",
  "modules_saved": 10,
  "categories_selected": ["tops", "dresses", "outerwear"],
  "has_taste_vector": true
}
```

#### POST /onboarding/core-setup
Save just core-setup and get Tinder test categories.

**Request Body:**
```json
{
  "anonId": "anon456",
  "gender": "female",
  "core-setup": {
    "selectedCategories": ["tops", "dresses"],
    "sizes": ["S", "M"],
    "colorsToAvoid": ["orange"],
    "materialsToAvoid": [],
    "enabled": true
  }
}
```

**Response:**
```json
{
  "status": "success",
  "user_id": "anon456",
  "categories_selected": ["tops", "dresses"],
  "tinder_categories": ["tops_knitwear", "tops_woven", "dresses"],
  "colors_to_avoid": ["orange"],
  "materials_to_avoid": []
}
```

---

### Utility Endpoints

#### GET /health
Check if the recommendation pipeline is healthy.

**Response:**
```json
{
  "status": "healthy",
  "sasrec_loaded": true,
  "sasrec_vocab_size": 76793,
  "pipeline_ready": true
}
```

#### GET /info
Get pipeline configuration information.

**Response:**
```json
{
  "candidate_selection": {
    "default_limit": 200,
    "exploration_rate": 0.1
  },
  "sasrec_ranker": {
    "model_loaded": true,
    "vocab_size": 76793,
    "model_path": "models/..."
  },
  "preference_scorer": {
    "weights": {
      "fit": 0.3,
      "style": 0.25,
      "length": 0.2,
      "material": 0.15,
      "color": 0.1
    }
  }
}
```

#### GET /categories/mapping
Get category mapping for Tinder test.

**Response:**
```json
{
  "onboarding_to_tinder": {
    "tops": ["tops_knitwear", "tops_woven", "tops_sleeveless"],
    "dresses": ["dresses"],
    "bottoms": ["bottoms_trousers", "bottoms_skorts"]
  },
  "tinder_categories": [
    {"id": "tops_knitwear", "label": "Sweaters & Knits", "broad": "tops"},
    {"id": "dresses", "label": "Dresses", "broad": "dresses"}
  ]
}
```

---

## Error Handling

All endpoints return errors in this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**HTTP Status Codes:**
| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid parameters) |
| 404 | Not Found (session/product not found) |
| 500 | Internal Server Error |

---

## Frontend Integration Example

```javascript
class RecommendationAPI {
  constructor(baseUrl = '/api/recs/v2') {
    this.baseUrl = baseUrl;
    this.sessionId = null;
    this.cursor = null;
    this.seenIds = [];
  }

  // Get feed page
  async getFeed(anonId, pageSize = 50) {
    const params = new URLSearchParams({
      anon_id: anonId,
      page_size: pageSize
    });

    if (this.sessionId) params.set('session_id', this.sessionId);
    if (this.cursor) params.set('cursor', this.cursor);

    const response = await fetch(`${this.baseUrl}/feed?${params}`);
    const data = await response.json();

    // Store for next request
    this.sessionId = data.session_id;
    this.cursor = data.cursor;

    // Accumulate seen_ids for sync
    data.results.forEach(r => this.seenIds.push(r.product_id));

    return data;
  }

  // Track user action
  async trackAction(productId, action, position) {
    await fetch(`${this.baseUrl}/feed/action`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        anon_id: this.anonId,
        session_id: this.sessionId,
        product_id: productId,
        action: action,
        position: position
      })
    });
  }

  // Sync seen_ids (call every 5 pages or on app close)
  async syncSeenIds() {
    if (this.seenIds.length === 0) return;

    await fetch(`${this.baseUrl}/session/sync`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        anon_id: this.anonId,
        session_id: this.sessionId,
        seen_ids: this.seenIds
      })
    });

    this.seenIds = []; // Clear after sync
  }
}

// Usage
const api = new RecommendationAPI();

// Load feed
const feed = await api.getFeed('user123');

// User likes item
await api.trackAction(feed.results[0].product_id, 'add_to_wishlist', 0);

// Sync on app close
window.addEventListener('beforeunload', () => api.syncSeenIds());
```

---

## Rate Limits

No rate limits currently enforced. For production, consider:
- 100 requests/minute per user for feed endpoints
- 1000 requests/minute per user for action tracking

---

## Changelog

**v2.1 (2024-01-14)**
- Added `/feed/action` endpoint for interaction tracking
- Added `/session/sync` endpoint for ML training data
- Actions: click, hover, add_to_wishlist, add_to_cart, purchase

**v2.0 (2024-01-10)**
- Keyset pagination for O(1) infinite scroll
- Session-based dedup (no duplicates within session)
- Feed versioning for stable ordering
