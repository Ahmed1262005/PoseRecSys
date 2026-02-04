# Women's Fashion Search API

## Endpoint

```
POST /api/women/search
```

Text search for women's fashion items using FashionCLIP with comprehensive filtering, hybrid search (semantic + keyword matching), and user profile integration.

---

## Request

### Headers

| Header | Value | Required |
|--------|-------|----------|
| `Content-Type` | `application/json` | Yes |

### Body Parameters

#### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | *required* | Natural language search query (e.g., "blue floral dress", "Zara midi dress") |
| `page` | integer | `1` | Page number (1-indexed) |
| `page_size` | integer | `50` | Results per page (max: 200) |

#### User Personalization (NEW)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_id` | string | `null` | User UUID for loading onboarding profile |
| `anon_id` | string | `null` | Anonymous ID for loading onboarding profile |
| `session_id` | string | `null` | Session ID for deduplication across pages |
| `apply_user_prefs` | boolean | `true` | Apply user profile preferences (colors/styles to avoid, preferred fits, etc.) |

#### Search Options (NEW)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_hybrid_search` | boolean | `true` | Enable keyword matching for brand/name (boosts "Zara dress" to show Zara items first) |
| `apply_diversity` | boolean | `false` | Limit items per category for variety |
| `max_per_category` | integer | `15` | Max items per category when diversity enabled |

#### Category Filters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `categories` | string[] | `null` | Broad categories: `tops`, `bottoms`, `dresses`, `outerwear` |
| `article_types` | string[] | `null` | Specific types: `jeans`, `t-shirts`, `midi dresses`, `blouses`, `jackets`, etc. |

#### Style Filters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fits` | string[] | `null` | Fit types: `slim`, `regular`, `relaxed`, `oversized` |
| `occasions` | string[] | `null` | Occasion types: `casual`, `office`, `evening`, `beach` |
| `exclude_styles` | string[] | `null` | Styles to exclude: `sheer`, `cutouts`, `backless`, `deep-necklines`, `strapless` |
| `patterns` | string[] | `null` | Patterns to include: `solid`, `stripes`, `floral`, `plaid`, `polka-dots` |
| `exclude_patterns` | string[] | `null` | Patterns to exclude |

#### Color Filters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_colors` | string[] | `null` | Colors to include (positive filter) |
| `exclude_colors` | string[] | `null` | Colors to exclude |

#### Material Filters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_materials` | string[] | `null` | Materials to include: `Cotton`, `Silk`, `Linen`, `Polyester`, etc. |
| `exclude_materials` | string[] | `null` | Materials to exclude |

#### Brand Filters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_brands` | string[] | `null` | Brands to include (case-insensitive) |
| `exclude_brands` | string[] | `null` | Brands to exclude (case-insensitive) |

#### Price Filters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_price` | number | `null` | Minimum price in USD |
| `max_price` | number | `null` | Maximum price in USD |

#### Other Filters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exclude_product_ids` | string[] | `null` | Product UUIDs to exclude from results |

---

## Response

### Success Response (200 OK)

```json
{
  "query": "Zara midi dress",
  "results": [
    {
      "product_id": "3ad05a7d-f12f-487a-97a8-eb6d3f6050df",
      "similarity": 0.92,
      "keyword_match": true,
      "name": "Zara Satin Midi Dress",
      "brand": "Zara",
      "category": "dresses",
      "broad_category": "dresses",
      "article_type": "midi dresses",
      "price": 79.99,
      "image_url": "https://...",
      "gallery_images": ["https://...", "https://..."],
      "colors": ["Black"],
      "materials": ["Satin", "Polyester"],
      "fit": "regular",
      "length": "midi",
      "sleeve": "short-sleeve",
      "preference_boost": 0.08,
      "preference_matches": ["brand"]
    }
  ],
  "count": 50,
  "filters_applied": {
    "filter_categories": ["dresses"],
    "filter_article_types": ["midi dresses"],
    "exclude_colors": ["Orange", "Neon"],
    "include_colors": null,
    "exclude_materials": null,
    "include_materials": null,
    "exclude_brands": null,
    "include_brands": null,
    "include_fits": ["regular"],
    "include_occasions": ["evening"],
    "exclude_styles": ["sheer", "backless"],
    "include_patterns": null,
    "exclude_patterns": null,
    "min_price": 30,
    "max_price": 150
  },
  "user_prefs_applied": true,
  "hybrid_search": true,
  "session_id": "sess-abc123",
  "pagination": {
    "page": 1,
    "page_size": 50,
    "has_more": true
  }
}
```

### Response Fields

#### Root Object

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | The search query that was executed |
| `results` | array | Array of matching products |
| `count` | integer | Number of results in current page |
| `filters_applied` | object | All filters that were applied (null if not set) |
| `user_prefs_applied` | boolean | Whether user profile preferences were applied |
| `hybrid_search` | boolean | Whether hybrid search (semantic + keyword) was used |
| `session_id` | string | Session ID for tracking (if provided) |
| `pagination` | object | Pagination information |
| `error` | string | Error message (only present if error occurred) |

#### Result Object

| Field | Type | Description |
|-------|------|-------------|
| `product_id` | string | Unique product UUID |
| `similarity` | number | Combined score (0-1, includes keyword boost and preference boost) |
| `keyword_match` | boolean | TRUE if query matched name/brand (NEW) |
| `name` | string | Product name |
| `brand` | string | Brand name |
| `category` | string | Product category |
| `broad_category` | string | Broad category (tops, bottoms, dresses, outerwear) |
| `article_type` | string | Specific article type (jeans, midi dresses, etc.) |
| `price` | number | Price in USD |
| `image_url` | string | Primary product image URL |
| `gallery_images` | string[] | Additional product images |
| `colors` | string[] | Product colors |
| `materials` | string[] | Product materials |
| `fit` | string | Fit type (slim, regular, relaxed, oversized) |
| `length` | string | Length (mini, midi, maxi, etc.) |
| `sleeve` | string | Sleeve type (sleeveless, short-sleeve, long-sleeve) |
| `preference_boost` | number | Score boost from user preferences (NEW) |
| `preference_matches` | string[] | Which preferences matched: "fit", "brand", "type", etc. (NEW) |

#### Pagination Object

| Field | Type | Description |
|-------|------|-------------|
| `page` | integer | Current page number |
| `page_size` | integer | Number of results per page |
| `has_more` | boolean | Whether more results are available |

### Error Response (400/500)

```json
{
  "detail": "Query is required"
}
```

---

## Examples

### Basic Search

```bash
curl -X POST http://localhost:8080/api/women/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "blue floral dress",
    "page_size": 20
  }'
```

### Search with Color Filters

```bash
curl -X POST http://localhost:8080/api/women/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "summer dress",
    "include_colors": ["White", "Beige", "Light Blue"],
    "exclude_colors": ["Black", "Gray"]
  }'
```

### Search with Price Range

```bash
curl -X POST http://localhost:8080/api/women/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "evening gown",
    "min_price": 100,
    "max_price": 300
  }'
```

### Search with Article Type

```bash
curl -X POST http://localhost:8080/api/women/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "comfortable pants",
    "article_types": ["jeans", "trousers"],
    "fits": ["relaxed", "regular"]
  }'
```

### Search with Occasion and Style Filters

```bash
curl -X POST http://localhost:8080/api/women/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "office appropriate dress",
    "occasions": ["office"],
    "exclude_styles": ["sheer", "cutouts", "backless", "deep-necklines"]
  }'
```

### Search with Brand Filters

```bash
curl -X POST http://localhost:8080/api/women/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "designer dress",
    "include_brands": ["Reformation", "Free People", "Rails"],
    "exclude_brands": ["Shein", "Fashion Nova"]
  }'
```

### Complex Multi-Filter Search

```bash
curl -X POST http://localhost:8080/api/women/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "elegant dinner dress",
    "page": 1,
    "page_size": 20,
    "categories": ["dresses"],
    "article_types": ["midi dresses", "maxi dresses"],
    "fits": ["regular", "slim"],
    "occasions": ["evening", "office"],
    "exclude_styles": ["sheer", "cutouts", "backless"],
    "include_colors": ["Black", "Navy", "Burgundy"],
    "exclude_colors": ["Neon", "Orange"],
    "include_materials": ["Silk", "Satin"],
    "min_price": 50,
    "max_price": 200
  }'
```

### Hybrid Search (Brand/Name Matching)

When searching for specific brands, items with matching brand/name are boosted:

```bash
# Search for Zara items - items with "Zara" in brand/name appear first
curl -X POST http://localhost:8080/api/women/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Zara midi dress",
    "use_hybrid_search": true
  }'

# Search for Reformation blouse
curl -X POST http://localhost:8080/api/women/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "reformation blouse",
    "page_size": 10
  }'
```

### With User Profile Integration

Load user's onboarding preferences and apply them:

```bash
# Search with user profile (applies colors_to_avoid, preferred_fits, etc.)
curl -X POST http://localhost:8080/api/women/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "elegant dress",
    "user_id": "user-uuid-123",
    "apply_user_prefs": true
  }'

# Anonymous user with session tracking
curl -X POST http://localhost:8080/api/women/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "summer dress",
    "anon_id": "anon-456",
    "session_id": "sess-789"
  }'
```

### Session Tracking (No Duplicates Across Pages)

Use session_id to prevent duplicates when paginating:

```bash
SESSION="sess-$(date +%s)"

# Page 1
curl -X POST http://localhost:8080/api/women/search \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"dress\", \"session_id\": \"$SESSION\", \"page\": 1, \"page_size\": 20}"

# Page 2 - will NOT show items from page 1
curl -X POST http://localhost:8080/api/women/search \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"dress\", \"session_id\": \"$SESSION\", \"page\": 2, \"page_size\": 20}"
```

### Diversity Constraints

Limit items per category for varied results:

```bash
curl -X POST http://localhost:8080/api/women/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "fashion",
    "apply_diversity": true,
    "max_per_category": 5
  }'
```

### Pagination Example (Legacy)

```bash
# Page 1
curl -X POST http://localhost:8080/api/women/search \
  -H "Content-Type: application/json" \
  -d '{"query": "dress", "page": 1, "page_size": 50}'

# Page 2
curl -X POST http://localhost:8080/api/women/search \
  -H "Content-Type: application/json" \
  -d '{"query": "dress", "page": 2, "page_size": 50}'
```

---

## Filter Behavior

### Positive vs Negative Filters

| Filter Type | Behavior |
|-------------|----------|
| `include_*` | Only return items that match at least one value |
| `exclude_*` | Exclude items that match any value |

### Case Sensitivity

| Filter | Case Sensitive |
|--------|----------------|
| `include_brands` / `exclude_brands` | No (case-insensitive) |
| `include_colors` / `exclude_colors` | Yes (match exact case) |
| `article_types` | No (case-insensitive) |
| `fits` | No (case-insensitive) |

### Null Filters

When a filter is `null` or not provided, it is not applied (all items pass through).

### Filter Combination

All filters are combined with AND logic:
- Item must pass ALL provided filters to be included
- Within array filters (like `include_colors: ["Black", "Navy"]`), items match if they contain ANY of the values

---

## Available Filter Values

### Categories
- `tops`
- `bottoms`
- `dresses`
- `outerwear`

### Article Types (partial list)
- Dresses: `mini dresses`, `midi dresses`, `maxi dresses`, `bodycon dresses`
- Tops: `t-shirts`, `blouses`, `sweaters`, `crop tops`, `tank tops`
- Bottoms: `jeans`, `trousers`, `skirts`, `shorts`, `leggings`
- Outerwear: `jackets`, `coats`, `blazers`, `cardigans`

### Fits
- `slim`
- `regular`
- `relaxed`
- `oversized`

### Occasions
- `casual`
- `office`
- `evening`
- `beach`

### Exclude Styles
- `sheer`
- `cutouts`
- `backless`
- `deep-necklines`
- `strapless`

### Common Colors
- `Black`, `White`, `Gray`, `Navy`, `Blue`, `Red`, `Pink`, `Green`, `Beige`, `Brown`, `Burgundy`, `Purple`, `Yellow`, `Orange`, `Cream`, `Gold`, `Silver`

### Common Materials
- `Cotton`, `Polyester`, `Silk`, `Linen`, `Wool`, `Rayon`, `Nylon`, `Spandex`, `Velvet`, `Satin`, `Denim`, `Leather`

---

## Rate Limits

No rate limits are currently enforced.

---

## Notes

1. **Hybrid Search**: By default, the search combines CLIP semantic similarity with keyword matching on brand/name. When you search "Zara dress", items with "Zara" in the brand get boosted. Disable with `use_hybrid_search: false` for pure semantic search.

2. **Similarity Scores**: Results are ordered by combined score (descending) which includes:
   - CLIP semantic similarity (0.7 weight by default)
   - Keyword match boost (0.3 weight if brand/name matches)
   - Preference boost (if user profile is applied)

3. **User Profile Integration**: When `user_id` or `anon_id` is provided:
   - Hard filters (colors_to_avoid, styles_to_avoid) are applied in SQL
   - Soft preferences (preferred_fits, preferred_brands) boost matching items
   - Request filters override profile filters if both are provided

4. **Session Tracking**: Use `session_id` to prevent duplicates across pages. Items shown on page 1 won't appear on page 2, even if they would normally rank higher.

5. **Soft Preference Boosts**:
   - Fit match: +0.05
   - Sleeve match: +0.03
   - Length match: +0.03
   - Preferred brand: +0.08
   - Type match: +0.04

6. **Occasion & Style Filters**: These use pre-computed scores with thresholds:
   - Occasion threshold: 0.20 (items with score >= 0.20 for any requested occasion pass)
   - Style threshold: 0.25 (items with score >= 0.25 for any excluded style are filtered out)

7. **Empty Results**: If no items match all filters, an empty results array is returned (not an error).

8. **Deduplication**: Results are automatically deduplicated by image hash and name+brand combination.

9. **Diversity**: Enable `apply_diversity: true` to limit items per broad_category (default 15), ensuring varied results across tops, bottoms, dresses, and outerwear.
