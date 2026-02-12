# Recommendation API V2 Documentation

Base URL: `https://your-server.com/api/recs/v2`

---

## Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/onboarding` | Save complete 10-module onboarding profile |
| `POST` | `/onboarding/core-setup` | Save core-setup & get Tinder categories |
| `POST` | `/onboarding/v3` | Save V3 flat-format onboarding profile |
| `GET` | `/categories/mapping` | Get category mapping reference |
| `GET` | `/feed` | Get personalized product feed (40+ filters, cursor pagination) |
| `GET` | `/feed/keyset` | Keyset pagination feed (full filter parity with /feed) |
| `GET` | `/sale` | Sale items feed (same filters as /feed) |
| `GET` | `/new-arrivals` | New arrivals feed (same filters as /feed) |
| `POST` | `/feed/action` | Record user interaction (click, wishlist, skip) |
| `GET` | `/feed/session/{id}` | Get session debug info |
| `DELETE` | `/feed/session/{id}` | Delete/reset a session |
| `GET` | `/health` | Check API health status |
| `GET` | `/info` | Get pipeline configuration |

---

## Integration Flow

The correct integration flow for onboarding + Tinder test:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: User completes Core Setup (Module 1)                       │
│  - Selects broad categories: ["tops", "dresses", "bottoms"]         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: Call POST /api/recs/v2/onboarding/core-setup               │
│  - Saves partial onboarding                                         │
│  - Returns mapped tinder_categories for Tinder test                 │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: Start Tinder Test with POST /api/women/session/start       │
│  - Pass tinder_categories from Step 2                               │
│  - User swipes through items in those categories only               │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4: User completes remaining onboarding modules (2-9)          │
│  - Category preferences, style, brands, etc.                        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 5: Call POST /api/recs/v2/onboarding                          │
│  - Saves complete 10-module profile                                 │
│  - Includes style-discovery with taste_vector from Tinder test      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 6: Fetch personalized feed with GET /api/recs/v2/feed         │
│  - Uses taste_vector for similarity ranking                         │
│  - Applies onboarding preferences as filters/boosts                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Save Onboarding Profile

Save the complete 10-module user onboarding data including style discovery (Tinder test) results.

### Request

```
POST /api/recs/v2/onboarding
Content-Type: application/json
```

### Request Body

```json
{
  "userId": "uuid-string",
  "anonId": "anonymous_user_id",
  "gender": "female",

  "core-setup": {
    "selectedCategories": ["tops", "bottoms", "dresses"],
    "sizes": ["S", "M"],
    "birthdate": "1995-06-15",
    "colorsToAvoid": ["red", "orange"],
    "materialsToAvoid": ["wool", "polyester"],
    "enabled": true
  },

  "tops": {
    "topTypes": ["tee", "blouse", "sweater"],
    "fits": ["regular", "relaxed"],
    "sleeves": ["short-sleeve", "long-sleeve"],
    "priceComfort": 50,
    "enabled": true
  },

  "bottoms": {
    "bottomTypes": ["jeans", "pants"],
    "fits": ["straight", "relaxed"],
    "rises": ["high-rise", "mid-rise"],
    "lengths": ["full-length", "ankle"],
    "numericWaist": 28,
    "numericHip": 38,
    "priceComfort": 60,
    "enabled": true
  },

  "skirts": {
    "skirtTypes": ["a-line", "midi"],
    "lengths": ["midi", "maxi"],
    "fits": ["regular"],
    "numericWaist": 28,
    "priceComfort": 45,
    "enabled": true
  },

  "dresses": {
    "dressTypes": ["wrap", "a-line"],
    "fits": ["fitted", "regular"],
    "lengths": ["midi"],
    "sleeves": ["sleeveless"],
    "priceComfort": 55,
    "enabled": true
  },

  "one-piece": {
    "onePieceTypes": ["jumpsuit"],
    "fits": ["regular"],
    "lengths": ["regular"],
    "numericWaist": 28,
    "priceComfort": 65,
    "enabled": true
  },

  "outerwear": {
    "outerwearTypes": ["coat", "puffer"],
    "fits": ["regular", "oversized"],
    "sleeves": ["long-sleeve"],
    "priceComfort": 70,
    "enabled": true
  },

  "style": {
    "styleDirections": ["classic", "minimal"],
    "modestyPreference": "balanced",
    "enabled": true
  },

  "brands": {
    "preferredBrands": ["Everlane", "Madewell"],
    "brandsToAvoid": ["Forever 21"],
    "brandOpenness": "mix-favorites-new",
    "enabled": true
  },

  "style-discovery": {
    "userId": "same_as_above",
    "selections": [
      {
        "round": 1,
        "category": "tops",
        "winnerId": "product_id_123",
        "loserId": "product_id_456",
        "timestamp": "2026-01-12T15:00:00.000Z"
      }
    ],
    "roundsCompleted": 15,
    "sessionComplete": true,
    "summary": {
      "attribute_preferences": {
        "pattern": {
          "preferred": [["solid", 0.78], ["floral", 0.65]],
          "avoided": [["animal_print", 0.22]]
        },
        "style": {
          "preferred": [["casual", 0.82]]
        },
        "color_family": {
          "preferred": [["neutral", 0.75]]
        }
      },
      "taste_stability": 0.85,
      "taste_vector": [0.123, -0.456, ...]
    },
    "enabled": true
  },

  "completedAt": "2026-01-12T15:30:00.000Z"
}
```

### Field Reference

#### Root Level

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `userId` | string | One required | UUID for logged-in users |
| `anonId` | string | One required | ID for anonymous users |
| `gender` | string | No | `"female"` or `"male"`. Default: `"female"` |
| `completedAt` | string | No | ISO 8601 timestamp |

#### Module 1: core-setup (Hard Filters)

| Field | Type | Description |
|-------|------|-------------|
| `selectedCategories` | string[] | Categories to show: `tops`, `bottoms`, `dresses`, `skirts`, `one-piece`, `outerwear` |
| `sizes` | string[] | User sizes: `XS`, `S`, `M`, `L`, `XL`, `XXL` |
| `birthdate` | string | Format: `YYYY-MM-DD` |
| `colorsToAvoid` | string[] | Colors to exclude from recommendations |
| `materialsToAvoid` | string[] | Materials to exclude |
| `enabled` | boolean | Whether module data should be used |

#### Modules 2-7: Category Preferences (Soft Scoring)

**tops**
| Field | Type | Description |
|-------|------|-------------|
| `topTypes` | string[] | `tee`, `blouse`, `sweater`, `tank`, `crop`, `bodysuit` |
| `fits` | string[] | `slim`, `regular`, `relaxed`, `oversized` |
| `sleeves` | string[] | `sleeveless`, `short-sleeve`, `long-sleeve`, `3/4-sleeve` |
| `priceComfort` | number | Max price willing to pay |

**bottoms**
| Field | Type | Description |
|-------|------|-------------|
| `bottomTypes` | string[] | `jeans`, `pants`, `shorts`, `leggings` |
| `fits` | string[] | `skinny`, `straight`, `wide-leg`, `relaxed` |
| `rises` | string[] | `low-rise`, `mid-rise`, `high-rise` |
| `lengths` | string[] | `cropped`, `ankle`, `full-length` |
| `numericWaist` | number | Waist measurement in inches |
| `numericHip` | number | Hip measurement in inches |
| `priceComfort` | number | Max price willing to pay |

**skirts**
| Field | Type | Description |
|-------|------|-------------|
| `skirtTypes` | string[] | `mini`, `midi`, `maxi`, `a-line`, `pencil` |
| `lengths` | string[] | `mini`, `midi`, `maxi` |
| `fits` | string[] | `fitted`, `regular`, `relaxed` |
| `numericWaist` | number | Waist measurement |
| `priceComfort` | number | Max price |

**dresses**
| Field | Type | Description |
|-------|------|-------------|
| `dressTypes` | string[] | `wrap`, `a-line`, `shift`, `bodycon`, `maxi` |
| `fits` | string[] | `fitted`, `regular`, `relaxed` |
| `lengths` | string[] | `mini`, `midi`, `maxi` |
| `sleeves` | string[] | `sleeveless`, `short-sleeve`, `long-sleeve` |
| `priceComfort` | number | Max price |

**one-piece**
| Field | Type | Description |
|-------|------|-------------|
| `onePieceTypes` | string[] | `jumpsuit`, `romper`, `overalls` |
| `fits` | string[] | `fitted`, `regular`, `relaxed` |
| `lengths` | string[] | `short`, `regular`, `long` |
| `numericWaist` | number | Waist measurement |
| `priceComfort` | number | Max price |

**outerwear**
| Field | Type | Description |
|-------|------|-------------|
| `outerwearTypes` | string[] | `coat`, `puffer`, `blazer`, `jacket`, `cardigan` |
| `fits` | string[] | `fitted`, `regular`, `oversized` |
| `sleeves` | string[] | `long-sleeve`, `3/4-sleeve` |
| `priceComfort` | number | Max price |

#### Module 8: style

| Field | Type | Description |
|-------|------|-------------|
| `styleDirections` | string[] | `minimal`, `classic`, `trendy`, `statement` |
| `modestyPreference` | string | `modest`, `balanced`, `revealing` |

#### Module 9: brands

| Field | Type | Description |
|-------|------|-------------|
| `preferredBrands` | string[] | Brands to boost in ranking |
| `brandsToAvoid` | string[] | Brands to exclude |
| `brandOpenness` | string | `stick-to-favorites`, `mix`, `mix-favorites-new`, `discover-new` |

#### Module 10: style-discovery (Tinder Test)

| Field | Type | Description |
|-------|------|-------------|
| `userId` | string | Same as root userId/anonId |
| `selections` | array | Array of selection objects |
| `roundsCompleted` | number | Total rounds completed |
| `sessionComplete` | boolean | Whether test was fully completed |
| `summary` | object | Computed preferences summary |
| `summary.attribute_preferences` | object | Learned attribute preferences |
| `summary.taste_stability` | number | 0-1 score of preference consistency |
| `summary.taste_vector` | number[] | 512-dimensional FashionCLIP embedding |

### Response

```json
{
  "status": "success",
  "user_id": "test_frontend_user_123",
  "modules_saved": 8,
  "categories_selected": ["tops", "bottoms"],
  "has_taste_vector": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"success"` or `"error"` |
| `user_id` | string | The user identifier used |
| `modules_saved` | number | Count of enabled modules saved |
| `categories_selected` | string[] | Categories from core-setup |
| `has_taste_vector` | boolean | Whether taste_vector was saved |

### Example cURL

```bash
curl -X POST "https://your-server.com/api/recs/v2/onboarding" \
  -H "Content-Type: application/json" \
  -d '{
    "anonId": "user_abc123",
    "gender": "female",
    "core-setup": {
      "selectedCategories": ["tops", "dresses"],
      "sizes": ["M"],
      "colorsToAvoid": ["yellow"],
      "enabled": true
    },
    "style": {
      "styleDirections": ["minimal", "classic"],
      "modestyPreference": "balanced",
      "enabled": true
    },
    "style-discovery": {
      "roundsCompleted": 15,
      "sessionComplete": true,
      "summary": {
        "taste_vector": [0.1, -0.2, ...]
      },
      "enabled": true
    }
  }'
```

---

## 2. Save Core Setup (Partial Onboarding)

Save the core-setup module and get mapped categories for the Tinder test. This endpoint should be called **before** starting the Tinder test.

### Why This Endpoint?

The onboarding uses **broad categories** (e.g., `tops`, `bottoms`, `dresses`) but the Tinder test catalog uses **specific categories** (e.g., `tops_knitwear`, `tops_woven`, `bottoms_trousers`). This endpoint:

1. Saves the user's core-setup preferences
2. Returns the mapped `tinder_categories` to filter the Tinder test items

### Category Mapping

| Onboarding Category | Tinder Test Categories |
|---------------------|------------------------|
| `tops` | `tops_knitwear`, `tops_woven`, `tops_sleeveless`, `tops_special` |
| `bottoms` | `bottoms_trousers`, `bottoms_skorts` |
| `dresses` | `dresses` |
| `skirts` | `bottoms_skorts` |
| `outerwear` | `outerwear` |
| `one-piece` | `dresses` |
| `sportswear` | `sportswear` |

### Request

```
POST /api/recs/v2/onboarding/core-setup
Content-Type: application/json
```

### Request Body

```json
{
  "anonId": "user_abc123",
  "gender": "female",
  "core-setup": {
    "selectedCategories": ["tops", "dresses", "bottoms"],
    "sizes": ["S", "M"],
    "birthdate": "1995-06-15",
    "colorsToAvoid": ["red", "orange"],
    "materialsToAvoid": ["polyester"],
    "enabled": true
  }
}
```

### Response

```json
{
  "status": "success",
  "user_id": "user_abc123",
  "categories_selected": ["tops", "dresses", "bottoms"],
  "tinder_categories": [
    "tops_knitwear",
    "tops_woven",
    "tops_sleeveless",
    "tops_special",
    "dresses",
    "bottoms_trousers",
    "bottoms_skorts"
  ],
  "colors_to_avoid": ["red", "orange"],
  "materials_to_avoid": ["polyester"]
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"success"` or `"error"` |
| `user_id` | string | The user identifier |
| `categories_selected` | string[] | Original broad categories from core-setup |
| `tinder_categories` | string[] | Mapped specific categories for Tinder test |
| `colors_to_avoid` | string[] | Colors saved from core-setup |
| `materials_to_avoid` | string[] | Materials saved from core-setup |

### Example cURL

```bash
curl -X POST "https://your-server.com/api/recs/v2/onboarding/core-setup" \
  -H "Content-Type: application/json" \
  -d '{
    "anonId": "user_abc123",
    "gender": "female",
    "core-setup": {
      "selectedCategories": ["tops", "dresses"],
      "sizes": ["M"],
      "colorsToAvoid": ["yellow"],
      "enabled": true
    }
  }'
```

### Example Response

```json
{
  "status": "success",
  "user_id": "user_abc123",
  "categories_selected": ["tops", "dresses"],
  "tinder_categories": [
    "tops_knitwear",
    "tops_special",
    "dresses",
    "tops_woven",
    "tops_sleeveless"
  ],
  "colors_to_avoid": ["yellow"],
  "materials_to_avoid": []
}
```

---

## 3. Get Category Mapping

Get the complete category mapping reference.

### Request

```
GET /api/recs/v2/categories/mapping
```

### Response

```json
{
  "onboarding_to_tinder": {
    "tops": ["tops_knitwear", "tops_woven", "tops_sleeveless", "tops_special"],
    "bottoms": ["bottoms_trousers", "bottoms_skorts"],
    "dresses": ["dresses"],
    "skirts": ["bottoms_skorts"],
    "outerwear": ["outerwear"],
    "one-piece": ["dresses"],
    "sportswear": ["sportswear"]
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

### Example cURL

```bash
curl "https://your-server.com/api/recs/v2/categories/mapping"
```

---

## 4. Get Personalized Feed

Retrieve personalized product recommendations based on user's onboarding profile and taste vector.

### Request

```
GET /api/recs/v2/feed
```

### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `user_id` | string | One required | - | UUID for logged-in users |
| `anon_id` | string | One required | - | ID for anonymous users |
| `gender` | string | No | `"female"` | `"female"` or `"male"` |
| `categories` | string | No | - | Comma-separated category filter |
| `limit` | integer | No | 50 | Results per page (1-200) |
| `offset` | integer | No | 0 | Pagination offset |

### Response

```json
{
  "user_id": "user_abc123",
  "strategy": "seed_vector",
  "results": [
    {
      "product_id": "uuid-product-1",
      "rank": 1,
      "score": 0.92,
      "reason": "style_matched",
      "category": "tops",
      "brand": "Everlane",
      "name": "The Organic Cotton Crew",
      "price": 35.00,
      "image_url": "https://...",
      "colors": ["White", "Black"]
    }
  ],
  "metadata": {
    "candidates_retrieved": 168,
    "candidates_after_diversity": 50,
    "sasrec_available": true,
    "seed_vector_available": true,
    "has_onboarding": true,
    "user_state_type": "tinder_complete",
    "exploration_count": 5,
    "by_source": {
      "taste_vector": 45,
      "trending": 3,
      "exploration": 5
    },
    "offset": 0,
    "has_more": true
  },
  "pagination": {
    "limit": 50,
    "offset": 0,
    "total": 168
  }
}
```

### Response Fields

#### Root

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | User identifier |
| `strategy` | string | Ranking strategy used (see below) |
| `results` | array | Array of product items |
| `metadata` | object | Debug/analytics information |
| `pagination` | object | Pagination details |

#### Strategy Values

| Strategy | Description |
|----------|-------------|
| `sasrec` | Warm user (5+ interactions) - uses sequential model |
| `seed_vector` | Has taste_vector - uses embedding similarity |
| `trending` | Cold start - uses trending/popular items |

#### Result Item

| Field | Type | Description |
|-------|------|-------------|
| `product_id` | string | Unique product UUID |
| `rank` | number | Position in results (1-based) |
| `score` | number | Combined relevance score (0-1) |
| `reason` | string | Why this item was recommended |
| `category` | string | Product category |
| `brand` | string | Brand name |
| `name` | string | Product name |
| `price` | number | Price in USD |
| `image_url` | string | Primary product image URL |
| `colors` | string[] | Available colors |

#### Reason Values

| Reason | Description |
|--------|-------------|
| `personalized` | SASRec sequential prediction |
| `style_matched` | High embedding similarity (>0.7) |
| `preference_matched` | Matches onboarding preferences |
| `trending` | Popular/trending item |
| `explore` | Exploration/discovery item |

#### Metadata

| Field | Type | Description |
|-------|------|-------------|
| `candidates_retrieved` | number | Initial candidates before filtering |
| `candidates_after_diversity` | number | After diversity constraints |
| `sasrec_available` | boolean | Whether SASRec model is loaded |
| `seed_vector_available` | boolean | Whether user has taste_vector |
| `has_onboarding` | boolean | Whether user has onboarding profile |
| `user_state_type` | string | `cold_start`, `tinder_complete`, or `warm_user` |
| `exploration_count` | number | Number of exploration items |
| `by_source` | object | Breakdown by candidate source |
| `has_more` | boolean | More results available |

### Example cURL

```bash
# Basic request
curl "https://your-server.com/api/recs/v2/feed?anon_id=user_abc123&gender=female&limit=20"

# With category filter
curl "https://your-server.com/api/recs/v2/feed?anon_id=user_abc123&categories=tops,dresses&limit=20"

# Pagination
curl "https://your-server.com/api/recs/v2/feed?anon_id=user_abc123&limit=20&offset=20"
```

---

## 5. Health Check

Check if the recommendation service is healthy and ready.

### Request

```
GET /api/recs/v2/health
```

### Response

```json
{
  "status": "healthy",
  "sasrec_loaded": true,
  "sasrec_vocab_size": 76793,
  "pipeline_ready": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"healthy"` or `"unhealthy"` |
| `sasrec_loaded` | boolean | Whether SASRec model is loaded |
| `sasrec_vocab_size` | number | Number of items in SASRec vocabulary |
| `pipeline_ready` | boolean | Whether full pipeline is ready |

---

## 6. Pipeline Info

Get detailed information about pipeline configuration.

### Request

```
GET /api/recs/v2/info
```

### Response

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
    "soft_weights": {
      "fit": 0.20,
      "style": 0.25,
      "length": 0.15,
      "neckline": 0.15,
      "sleeve": 0.10,
      "brand": 0.15
    }
  },
  "sasrec_ranker": {
    "model_loaded": true,
    "vocab_size": 76793,
    "min_sequence_for_sasrec": 5,
    "warm_weights": {
      "sasrec": 0.40,
      "embedding": 0.35,
      "preference": 0.25
    },
    "cold_weights": {
      "embedding": 0.60,
      "preference": 0.40
    }
  }
}
```

---

## User State Types

The recommendation strategy depends on user state:

| State | Condition | Strategy | Weights |
|-------|-----------|----------|---------|
| `cold_start` | No taste_vector, no history | `trending` | N/A (trending only) |
| `tinder_complete` | Has taste_vector, <5 interactions | `seed_vector` | embedding: 0.60, preference: 0.40 |
| `warm_user` | Has taste_vector, 5+ interactions | `sasrec` | sasrec: 0.40, embedding: 0.35, preference: 0.25 |

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Either user_id/userId or anon_id/anonId must be provided"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Error message describing the issue"
}
```

---

## TypeScript Types

```typescript
// Onboarding Request
interface OnboardingRequest {
  userId?: string;
  anonId?: string;
  gender?: 'female' | 'male';

  'core-setup'?: {
    selectedCategories: string[];
    sizes: string[];
    birthdate?: string;
    colorsToAvoid: string[];
    materialsToAvoid: string[];
    enabled: boolean;
  };

  tops?: CategoryPrefs & { topTypes: string[]; sleeves: string[] };
  bottoms?: CategoryPrefs & {
    bottomTypes: string[];
    rises: string[];
    lengths: string[];
    numericWaist?: number;
    numericHip?: number;
  };
  skirts?: CategoryPrefs & { skirtTypes: string[]; lengths: string[] };
  dresses?: CategoryPrefs & { dressTypes: string[]; lengths: string[]; sleeves: string[] };
  'one-piece'?: CategoryPrefs & { onePieceTypes: string[]; lengths: string[] };
  outerwear?: CategoryPrefs & { outerwearTypes: string[]; sleeves: string[] };

  style?: {
    styleDirections: string[];
    modestyPreference: 'modest' | 'balanced' | 'revealing';
    enabled: boolean;
  };

  brands?: {
    preferredBrands: string[];
    brandsToAvoid: string[];
    brandOpenness: string;
    enabled: boolean;
  };

  'style-discovery'?: {
    userId?: string;
    selections: StyleSelection[];
    roundsCompleted: number;
    sessionComplete: boolean;
    summary?: {
      attribute_preferences: Record<string, any>;
      taste_stability?: number;
      taste_vector?: number[];
    };
    enabled: boolean;
  };

  completedAt?: string;
}

interface CategoryPrefs {
  fits: string[];
  priceComfort?: number;
  enabled: boolean;
}

interface StyleSelection {
  round: number;
  category?: string;
  winnerId: string;
  loserId: string;
  timestamp?: string;
}

// Onboarding Response
interface OnboardingResponse {
  status: 'success' | 'error';
  user_id: string;
  modules_saved: number;
  categories_selected: string[];
  has_taste_vector: boolean;
}

// Feed Response
interface FeedResponse {
  user_id: string;
  strategy: 'sasrec' | 'seed_vector' | 'trending';
  results: FeedItem[];
  metadata: FeedMetadata;
  pagination: Pagination;
}

interface FeedItem {
  product_id: string;
  rank: number;
  score: number;
  reason: 'personalized' | 'style_matched' | 'preference_matched' | 'trending' | 'explore';
  category: string;
  brand: string;
  name: string;
  price: number;
  image_url: string;
  colors: string[];
}

interface FeedMetadata {
  candidates_retrieved: number;
  candidates_after_diversity: number;
  sasrec_available: boolean;
  seed_vector_available: boolean;
  has_onboarding: boolean;
  user_state_type: 'cold_start' | 'tinder_complete' | 'warm_user';
  exploration_count: number;
  by_source: Record<string, number>;
  offset: number;
  has_more: boolean;
}

interface Pagination {
  limit: number;
  offset: number;
  total: number;
}

// Core Setup Request (for partial onboarding)
interface CoreSetupRequest {
  userId?: string;
  anonId?: string;
  gender?: 'female' | 'male';
  'core-setup': {
    selectedCategories: string[];
    sizes: string[];
    birthdate?: string;
    colorsToAvoid: string[];
    materialsToAvoid: string[];
    enabled: boolean;
  };
}

// Core Setup Response
interface CoreSetupResponse {
  status: 'success' | 'error';
  user_id: string;
  categories_selected: string[];
  tinder_categories: string[];
  colors_to_avoid: string[];
  materials_to_avoid: string[];
}

// Category Mapping Response
interface TinderCategory {
  id: string;
  label: string;
  broad: string;
}

interface OnboardingCategory {
  id: string;
  label: string;
  tinder_maps_to: string[];
}

interface CategoryMappingResponse {
  onboarding_to_tinder: Record<string, string[]>;
  tinder_categories: TinderCategory[];
  onboarding_categories: OnboardingCategory[];
}
```

---

## Integration Example (React)

```typescript
// api/recommendations.ts
const API_BASE = 'https://your-server.com/api/recs/v2';

// Step 1: Save core-setup and get Tinder categories
export async function saveCoreSetup(data: CoreSetupRequest): Promise<CoreSetupResponse> {
  const response = await fetch(`${API_BASE}/onboarding/core-setup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error(`Failed to save core setup: ${response.statusText}`);
  }

  return response.json();
}

// Step 2: Save complete onboarding (after Tinder test)
export async function saveOnboarding(data: OnboardingRequest): Promise<OnboardingResponse> {
  const response = await fetch(`${API_BASE}/onboarding`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error(`Failed to save onboarding: ${response.statusText}`);
  }

  return response.json();
}

// Step 3: Get personalized feed
export async function getFeed(params: {
  anonId?: string;
  userId?: string;
  gender?: string;
  categories?: string[];
  limit?: number;
  offset?: number;
}): Promise<FeedResponse> {
  const searchParams = new URLSearchParams();

  if (params.anonId) searchParams.set('anon_id', params.anonId);
  if (params.userId) searchParams.set('user_id', params.userId);
  if (params.gender) searchParams.set('gender', params.gender);
  if (params.categories) searchParams.set('categories', params.categories.join(','));
  if (params.limit) searchParams.set('limit', params.limit.toString());
  if (params.offset) searchParams.set('offset', params.offset.toString());

  const response = await fetch(`${API_BASE}/feed?${searchParams}`);

  if (!response.ok) {
    throw new Error(`Failed to get feed: ${response.statusText}`);
  }

  return response.json();
}

// Get category mapping (optional - for reference)
export async function getCategoryMapping(): Promise<CategoryMappingResponse> {
  const response = await fetch(`${API_BASE}/categories/mapping`);
  return response.json();
}

// =============================================================================
// Complete Integration Flow Example
// =============================================================================

// 1. After core-setup module completion (BEFORE Tinder test)
const handleCoreSetupComplete = async (coreSetupData: CoreSetupData) => {
  try {
    // Save core-setup and get mapped categories for Tinder test
    const result = await saveCoreSetup({
      anonId: getUserId(),
      gender: 'female',
      'core-setup': {
        selectedCategories: coreSetupData.selectedCategories, // ["tops", "dresses"]
        sizes: coreSetupData.sizes,
        colorsToAvoid: coreSetupData.colorsToAvoid,
        materialsToAvoid: coreSetupData.materialsToAvoid,
        enabled: true,
      },
    });

    console.log('Categories selected:', result.categories_selected);
    console.log('Tinder categories:', result.tinder_categories);

    // IMPORTANT: Use tinder_categories when starting the Tinder test
    // These are the specific categories that will filter the swipe items
    startTinderTest({
      userId: getUserId(),
      categories: result.tinder_categories, // Pass to Tinder test!
    });

  } catch (error) {
    console.error('Core setup failed:', error);
  }
};

// 2. After completing ALL onboarding modules + Tinder test
const handleOnboardingComplete = async (onboardingData: OnboardingData) => {
  try {
    const result = await saveOnboarding({
      anonId: getUserId(),
      gender: 'female',
      'core-setup': onboardingData.coreSetup,
      tops: onboardingData.tops,
      bottoms: onboardingData.bottoms,
      skirts: onboardingData.skirts,
      dresses: onboardingData.dresses,
      'one-piece': onboardingData.onePiece,
      outerwear: onboardingData.outerwear,
      style: onboardingData.style,
      brands: onboardingData.brands,
      'style-discovery': {
        roundsCompleted: onboardingData.tinderTest.roundsCompleted,
        sessionComplete: onboardingData.tinderTest.sessionComplete,
        selections: onboardingData.tinderTest.selections,
        summary: {
          attribute_preferences: onboardingData.tinderTest.attributePreferences,
          taste_stability: onboardingData.tinderTest.tasteStability,
          taste_vector: onboardingData.tinderTest.tasteVector, // 512-dim array
        },
        enabled: true,
      },
      completedAt: new Date().toISOString(),
    });

    console.log(`Saved ${result.modules_saved} modules`);
    console.log(`Has taste vector: ${result.has_taste_vector}`);

    // 3. Now fetch personalized feed
    const feed = await getFeed({ anonId: getUserId(), limit: 20 });
    setProducts(feed.results);

    console.log(`Feed strategy: ${feed.strategy}`); // 'seed_vector' if taste_vector saved

  } catch (error) {
    console.error('Onboarding failed:', error);
  }
};
```
