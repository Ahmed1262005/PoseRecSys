# Women's Fashion Style Learning API

**Base URL:** `http://ecommerce.api.outrove.ai:8080`

**Interactive Docs:** `http://ecommerce.api.outrove.ai:8080/docs`

**ReDoc:** `http://ecommerce.api.outrove.ai:8080/redoc`

---

## Overview

This API learns user fashion preferences through a 4-choice swipe interface. Users pick their favorite from 4 items, and the system learns their style preferences using Bayesian preference learning.

### How It Works

1. **Start Session** → Get 4 items to display
2. **User Chooses** → System learns from the choice
3. **Repeat** → Until preferences are learned (typically 10-20 rounds)
4. **Get Feed** → Personalized recommendations based on learned taste

---

## Quick Start

### 1. Check API Health

```bash
curl http://ecommerce.api.outrove.ai:8080/api/women/health
```

**Response:**
```json
{
  "status": "healthy",
  "gender": "female",
  "total_items": 4399,
  "total_categories": 9,
  "categories": ["tops_knitwear", "tops_woven", "tops_sleeveless", "tops_special", "dresses", "bottoms_trousers", "bottoms_skorts", "outerwear", "sportswear"],
  "attributes_loaded": 4399,
  "image_base_url": "http://ecommerce.api.outrove.ai:8080/women-images/"
}
```

### 2. Get Available Options

```bash
curl http://ecommerce.api.outrove.ai:8080/api/women/options
```

**Response:**
```json
{
  "gender": "female",
  "categories": [
    {"id": "tops_knitwear", "label": "Sweaters & Knits", "description": "Sweaters, cardigans, knit tops"},
    {"id": "tops_woven", "label": "Blouses & Shirts", "description": "Woven blouses, shirts, button-ups"},
    {"id": "tops_sleeveless", "label": "Tank Tops & Camis", "description": "Sleeveless tops, camisoles"},
    {"id": "tops_special", "label": "Bodysuits", "description": "Bodysuits and special tops"},
    {"id": "dresses", "label": "Dresses", "description": "All dress styles"},
    {"id": "bottoms_trousers", "label": "Pants & Trousers", "description": "Pants, jeans, trousers"},
    {"id": "bottoms_skorts", "label": "Skirts & Shorts", "description": "Skirts, shorts, skorts"},
    {"id": "outerwear", "label": "Outerwear", "description": "Jackets, coats, blazers"},
    {"id": "sportswear", "label": "Sportswear", "description": "Athletic wear, leggings"}
  ],
  "attributes": {
    "pattern": {
      "weight": 0.25,
      "values": ["solid", "striped", "floral", "geometric", "animal_print", "plaid", "polka_dots", "lace"]
    },
    "style": {
      "weight": 0.20,
      "values": ["casual", "office", "evening", "bohemian", "minimalist", "romantic", "athletic"]
    },
    "color_family": {
      "weight": 0.15,
      "values": ["neutral", "bright", "cool", "pastel", "dark"]
    },
    "fit_vibe": {
      "weight": 0.15,
      "values": ["fitted", "relaxed", "oversized", "cropped", "flowy"]
    },
    "neckline": {
      "weight": 0.10,
      "values": ["crew", "v_neck", "scoop", "off_shoulder", "sweetheart", "halter", "square", "turtleneck", "cowl"]
    },
    "occasion": {
      "weight": 0.10,
      "values": ["everyday", "work", "date_night", "party", "beach"]
    },
    "sleeve_type": {
      "weight": 0.05,
      "values": ["sleeveless", "short_sleeve", "long_sleeve", "puff_sleeve", "bell_sleeve", "flutter_sleeve"]
    }
  },
  "colors_available": ["black", "white", "gray", "navy", "blue", "red", "green", "brown", "pink", "yellow", "orange", "purple", "cream", "beige"],
  "total_items": 4399,
  "image_base_url": "http://ecommerce.api.outrove.ai:8080/women-images/"
}
```

---

## API Endpoints

### POST /api/women/session/start

Start a new style learning session.

**Request:**
```bash
curl -X POST http://ecommerce.api.outrove.ai:8080/api/women/session/start \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_12345",
    "colors_to_avoid": ["pink", "yellow"],
    "materials_to_avoid": ["polyester"],
    "selected_categories": ["dresses", "tops_knitwear"]
  }'
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | Unique user identifier |
| `colors_to_avoid` | string[] | No | Colors to exclude from items shown |
| `materials_to_avoid` | string[] | No | Materials to exclude |
| `selected_categories` | string[] | No | Categories to focus on (empty = all) |

**Response:**
```json
{
  "status": "started",
  "user_id": "user_12345",
  "items": [
    {
      "id": "dresses/dresses/405",
      "image_url": "/women-images/dresses/dresses/405.webp",
      "category": "dresses",
      "brand": "",
      "color": "",
      "cluster": 0
    },
    {
      "id": "dresses/dresses/621",
      "image_url": "/women-images/dresses/dresses/621.webp",
      "category": "dresses",
      "brand": "",
      "color": "",
      "cluster": 3
    },
    {
      "id": "dresses/dresses/155",
      "image_url": "/women-images/dresses/dresses/155.webp",
      "category": "dresses",
      "brand": "",
      "color": "",
      "cluster": 4
    },
    {
      "id": "dresses/dresses/754",
      "image_url": "/women-images/dresses/dresses/754.webp",
      "category": "dresses",
      "brand": "",
      "color": "",
      "cluster": 1
    }
  ],
  "round": 1,
  "test_info": {
    "category": "dresses",
    "category_label": "Dresses",
    "category_index": 1,
    "total_categories": 2,
    "round_in_category": 1,
    "clusters_shown": [0, 3, 4, 1],
    "prediction": {
      "predicted_cluster": 0,
      "confidence": 0.0,
      "has_prediction": false
    }
  },
  "session_complete": false,
  "message": "Pick your favorite from the 4 items shown"
}
```

---

### POST /api/women/session/choose

Record the user's choice from the 4 displayed items.

**Request:**
```bash
curl -X POST http://ecommerce.api.outrove.ai:8080/api/women/session/choose \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_12345",
    "winner_id": "dresses/dresses/405"
  }'
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | User ID from session start |
| `winner_id` | string | Yes | ID of the chosen item (must be one of the 4 shown) |

**Response (continuing):**
```json
{
  "status": "continue",
  "user_id": "user_12345",
  "items": [
    {"id": "dresses/dresses/355", "image_url": "/women-images/dresses/dresses/355.webp", "cluster": 5},
    {"id": "dresses/dresses/221", "image_url": "/women-images/dresses/dresses/221.webp", "cluster": 4},
    {"id": "dresses/dresses/168", "image_url": "/women-images/dresses/dresses/168.webp", "cluster": 11},
    {"id": "dresses/dresses/651", "image_url": "/women-images/dresses/dresses/651.webp", "cluster": 2}
  ],
  "round": 2,
  "test_info": {
    "category": "dresses",
    "category_label": "Dresses",
    "category_index": 1,
    "total_categories": 2,
    "round_in_category": 2,
    "prediction": {
      "predicted_cluster": 0,
      "confidence": 0.17,
      "has_prediction": true
    }
  },
  "result_info": {
    "prediction_correct": true,
    "consecutive_correct": 1,
    "category_complete": false
  },
  "session_complete": false
}
```

**Response (session complete):**
```json
{
  "status": "complete",
  "user_id": "user_12345",
  "items": [],
  "round": 15,
  "session_complete": true,
  "summary": {
    "likes": 15,
    "dislikes": 45,
    "attribute_preferences": {
      "pattern": {
        "preferred": [["solid", 0.72], ["geometric", 0.65]],
        "avoided": [["floral", 0.28], ["animal_print", 0.25]]
      },
      "style": {
        "preferred": [["minimalist", 0.78], ["casual", 0.67]],
        "avoided": [["romantic", 0.30]]
      }
    }
  },
  "feed_preview": {
    "dresses": [...],
    "tops_knitwear": [...]
  },
  "message": "Style profile complete! Use /api/women/feed/{user_id} for personalized recommendations."
}
```

---

### POST /api/women/session/skip

Skip all 4 items when none are appealing.

**Request:**
```bash
curl -X POST http://ecommerce.api.outrove.ai:8080/api/women/session/skip \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_12345"
  }'
```

**Note:** Use sparingly. Skipping counts as disliking all 4 items. Better to pick the "least bad" option.

---

### GET /api/women/session/{user_id}/summary

Get current session state and learned preferences.

**Request:**
```bash
curl http://ecommerce.api.outrove.ai:8080/api/women/session/user_12345/summary
```

**Response:**
```json
{
  "user_id": "user_12345",
  "gender": "female",
  "summary": {
    "session_complete": false,
    "total_swipes": 8,
    "likes": 2,
    "dislikes": 6,
    "taste_stability": 0.45,
    "attribute_preferences": {
      "pattern": {
        "preferred": [["geometric", 0.667], ["solid", 0.571]],
        "avoided": [["floral", 0.25]],
        "all_scores": [["geometric", 0.667], ["solid", 0.571], ["striped", 0.5], ["floral", 0.25]]
      },
      "style": {
        "preferred": [["evening", 0.667], ["minimalist", 0.6]],
        "avoided": [["bohemian", 0.333]],
        "all_scores": [["evening", 0.667], ["minimalist", 0.6], ["casual", 0.5], ["bohemian", 0.333]]
      },
      "color_family": {
        "preferred": [["dark", 0.714]],
        "avoided": [["bright", 0.286]],
        "all_scores": [["dark", 0.714], ["neutral", 0.5], ["bright", 0.286]]
      }
    },
    "category_stats": {
      "dresses": {"rounds": 5, "complete": true},
      "tops_knitwear": {"rounds": 3, "complete": false}
    }
  },
  "feed_preview": {
    "dresses": [...],
    "tops_knitwear": [...]
  }
}
```

---

### GET /api/women/feed/{user_id}

Get personalized recommendations based on learned preferences.

**Request:**
```bash
curl "http://ecommerce.api.outrove.ai:8080/api/women/feed/user_12345?items_per_category=20&categories=dresses,tops_knitwear"
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `items_per_category` | int | 20 | Number of items per category (1-100) |
| `categories` | string | all | Comma-separated category IDs |

**Response:**
```json
{
  "user_id": "user_12345",
  "gender": "female",
  "feed": {
    "dresses": {
      "label": "Dresses",
      "count": 20,
      "items": [
        {
          "id": "dresses/dresses/624",
          "category": "dresses",
          "image_url": "/women-images/dresses/dresses/624.webp",
          "brand": "",
          "color": "",
          "similarity": 0.826,
          "cluster_match": 0.67,
          "attr_match": 0.85,
          "attributes": {
            "pattern": "geometric",
            "style": "evening",
            "neckline": "v_neck",
            "fit_vibe": "fitted",
            "occasion": "date_night",
            "color_family": "dark",
            "dress_length": "midi",
            "dress_silhouette": "a_line"
          }
        }
      ]
    },
    "tops_knitwear": {
      "label": "Sweaters & Knits",
      "count": 20,
      "items": [...]
    }
  },
  "total_items": 40,
  "image_base_url": "http://ecommerce.api.outrove.ai:8080/women-images/"
}
```

---

## Image URLs

Images are served from: `http://ecommerce.api.outrove.ai:8080/women-images/`

**URL Pattern:** `/women-images/{category}/{subcategory}/{id}.webp`

**Example:**
```
http://ecommerce.api.outrove.ai:8080/women-images/dresses/dresses/405.webp
http://ecommerce.api.outrove.ai:8080/women-images/tops_knitwear/sweaters/123.webp
```

**To get full URL:** Prepend `http://ecommerce.api.outrove.ai:8080` to the `image_url` field.

---

## Categories Reference

| ID | Label | Items | Description |
|----|-------|-------|-------------|
| `tops_knitwear` | Sweaters & Knits | ~600 | Sweaters, cardigans, knit tops |
| `tops_woven` | Blouses & Shirts | ~800 | Woven blouses, shirts |
| `tops_sleeveless` | Tank Tops & Camis | ~400 | Sleeveless tops |
| `tops_special` | Bodysuits | ~200 | Bodysuits, special tops |
| `dresses` | Dresses | ~900 | All dress styles |
| `bottoms_trousers` | Pants & Trousers | ~700 | Pants, jeans, trousers |
| `bottoms_skorts` | Skirts & Shorts | ~400 | Skirts, shorts |
| `outerwear` | Outerwear | ~300 | Jackets, coats, blazers |
| `sportswear` | Sportswear | ~200 | Athletic wear, leggings |

**Total:** 4,399 items

---

## Attributes Reference

### Pattern (25% weight)
- `solid` - Single color, no pattern
- `striped` - Horizontal or vertical stripes
- `floral` - Flower patterns
- `geometric` - Shapes, abstract patterns
- `animal_print` - Leopard, zebra, snake
- `plaid` - Checkered, tartan
- `polka_dots` - Round dots
- `lace` - Lace fabric pattern

### Style (20% weight)
- `casual` - Everyday, relaxed
- `office` - Professional, business
- `evening` - Dressy, sophisticated
- `bohemian` - Free-spirited, artistic
- `minimalist` - Clean, simple
- `romantic` - Feminine, soft
- `athletic` - Sporty, performance

### Color Family (15% weight)
- `neutral` - White, black, gray, beige
- `bright` - Red, yellow, orange
- `cool` - Blue, green, purple
- `pastel` - Light pink, baby blue, lavender
- `dark` - Black, navy, charcoal

### Fit Vibe (15% weight)
- `fitted` - Form-fitting, slim
- `relaxed` - Loose, comfortable
- `oversized` - Extra loose, baggy
- `cropped` - Above waist length
- `flowy` - Drapey, soft movement

### Neckline (10% weight)
- `crew` - Round neckline
- `v_neck` - V-shaped neckline
- `scoop` - Wide, rounded
- `off_shoulder` - Bare shoulders
- `sweetheart` - Heart-shaped curve
- `halter` - Ties at neck
- `square` - Straight across
- `turtleneck` - High neck
- `cowl` - Draped neckline

### Occasion (10% weight)
- `everyday` - Daily casual wear
- `work` - Office appropriate
- `date_night` - Romantic evening
- `party` - Celebration, festive
- `beach` - Vacation, resort

### Sleeve Type (5% weight)
- `sleeveless` - No sleeves
- `short_sleeve` - Short sleeves
- `long_sleeve` - Full length
- `puff_sleeve` - Voluminous
- `bell_sleeve` - Flared, wide
- `flutter_sleeve` - Ruffled, flowing

---

## Complete Test Flow

```bash
# 1. Start a session
curl -X POST http://ecommerce.api.outrove.ai:8080/api/women/session/start \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user_001", "selected_categories": ["dresses"]}'

# Save one of the item IDs from the response, e.g., "dresses/dresses/405"

# 2. Make choices (repeat 5-10 times)
curl -X POST http://ecommerce.api.outrove.ai:8080/api/women/session/choose \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user_001", "winner_id": "dresses/dresses/405"}'

# 3. Check learned preferences
curl http://ecommerce.api.outrove.ai:8080/api/women/session/test_user_001/summary

# 4. Get personalized feed
curl "http://ecommerce.api.outrove.ai:8080/api/women/feed/test_user_001?items_per_category=10"
```

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "No active session. Call /api/women/session/start first."
}
```

### 400 Bad Request (invalid winner)
```json
{
  "detail": "Invalid winner_id. Must be one of: ['dresses/dresses/405', 'dresses/dresses/621', ...]"
}
```

### 404 Not Found
```json
{
  "detail": "No session found for user: unknown_user"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Not enough items available. Try different categories."
}
```

---

## Algorithm Overview

The system uses **Bayesian Preference Learning** with **Contrastive Taste Vectors**:

1. **4 items from different visual clusters** are shown each round
2. User picks 1 → **Winner attributes get +1 win**, losers get +1 count
3. **Bayesian score** = (wins + 1) / (count + 2) for each attribute value
4. **Taste vector** updated via contrastive learning in 512-dim FashionCLIP space
5. **Prediction** made before each round using attribute + cluster + taste scores
6. Category is "learned" when predictions are accurate or attribute confidence is high

### Ranking Formula (Feed)
```
score = 0.40 * taste_similarity
      + 0.35 * attribute_match
      + 0.25 * cluster_match
```

---

## Frontend Integration Example (JavaScript)

```javascript
const API_BASE = 'http://ecommerce.api.outrove.ai:8080';

// Start session
async function startSession(userId, categories = []) {
  const response = await fetch(`${API_BASE}/api/women/session/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      selected_categories: categories
    })
  });
  return response.json();
}

// Record choice
async function recordChoice(userId, winnerId) {
  const response = await fetch(`${API_BASE}/api/women/session/choose`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      winner_id: winnerId
    })
  });
  return response.json();
}

// Get feed
async function getFeed(userId, itemsPerCategory = 20) {
  const response = await fetch(
    `${API_BASE}/api/women/feed/${userId}?items_per_category=${itemsPerCategory}`
  );
  return response.json();
}

// Get full image URL
function getImageUrl(imageUrl) {
  return `${API_BASE}${imageUrl}`;
}

// Usage
async function main() {
  const userId = 'user_' + Date.now();

  // Start
  let data = await startSession(userId, ['dresses', 'tops_knitwear']);
  console.log('Round 1:', data.items.map(i => i.id));

  // Choose first item
  data = await recordChoice(userId, data.items[0].id);

  // Continue until complete
  while (!data.session_complete) {
    data = await recordChoice(userId, data.items[0].id);
    console.log(`Round ${data.round}:`, data.test_info.category);
  }

  // Get personalized feed
  const feed = await getFeed(userId);
  console.log('Recommendations:', feed.total_items, 'items');
}
```

---

## CORS

CORS is enabled for all origins. You can call this API from any frontend domain.

---

## Rate Limits

No rate limits currently. For production use, implement client-side throttling.

---

## Support

- **API Docs:** http://ecommerce.api.outrove.ai:8080/docs
- **Frontend Demo:** http://ecommerce.api.outrove.ai:8080/women
