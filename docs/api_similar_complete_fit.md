# Similar Items & Complete the Look API Documentation

## Overview

Two endpoints for product-based recommendations:

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/recs/similar/{product_id}` | GET | No | Similar Items - Visually similar products via FashionCLIP embeddings |
| `/api/women/complete-fit` | POST | Yes (JWT) | Complete the Look - Complementary items to complete an outfit |

---

## GET /api/recs/similar/{product_id}

Returns products visually similar to a given product using FashionCLIP embeddings + pgvector cosine similarity.

**Auth:** None required (public endpoint).

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `product_id` | string (UUID) | The product to find similar items for |

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gender` | string | `"female"` | Gender filter |
| `category` | string | null | Filter by category (e.g., `"tops"`, `"dresses"`) |
| `limit` | int | 20 | Items per page (1-100) |
| `offset` | int | 0 | Skip first N items for pagination |

### Response

```json
{
  "product_id": "e8e5e4c1-d674-4555-af2f-605e152668af",
  "results": [
    {
      "product_id": "8f39f0f2-6254-4b1f-b836-5ef2b411fdc1",
      "name": "THE FAYE SWEATER",
      "brand": "Joe's Jeans",
      "category": "tops",
      "gender": ["female"],
      "price": 59.0,
      "primary_image_url": "https://usepose.s3.us-east-1.amazonaws.com/products/.../original_0.jpg",
      "hero_image_url": null,
      "similarity": 0.786,
      "rank": 1
    },
    {
      "product_id": "d91234d3-4b40-4793-9d4e-1517ff50f406",
      "name": "Cable Knit Crew Neck Sweater",
      "brand": "Alo Yoga",
      "category": "tops",
      "gender": ["female"],
      "price": 228.0,
      "primary_image_url": "https://usepose.s3.us-east-1.amazonaws.com/products/.../gallery_2.jpg",
      "hero_image_url": null,
      "similarity": 0.773,
      "rank": 2
    }
  ],
  "pagination": {
    "offset": 0,
    "limit": 2,
    "returned": 2,
    "has_more": true
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `product_id` | string | The source product ID |
| `results` | array | List of similar products |
| `results[].product_id` | string | Product UUID |
| `results[].name` | string | Product name |
| `results[].brand` | string | Brand name |
| `results[].category` | string | Product category |
| `results[].gender` | array | Gender tags |
| `results[].price` | float | Current price |
| `results[].primary_image_url` | string | Main product image URL |
| `results[].hero_image_url` | string/null | Hero image URL (if available) |
| `results[].similarity` | float | Cosine similarity score (0-1, higher = more similar) |
| `results[].rank` | int | 1-indexed rank (continuous across pages) |
| `pagination.offset` | int | Current offset |
| `pagination.limit` | int | Requested limit |
| `pagination.returned` | int | Actual items returned |
| `pagination.has_more` | boolean | Whether more pages are available |

### Pagination

Use `offset` and `limit` for infinite scroll:

```bash
# Page 1: items 1-20
curl '/api/recs/similar/{id}?limit=20&offset=0'
# Response: rank 1-20, has_more: true

# Page 2: items 21-40
curl '/api/recs/similar/{id}?limit=20&offset=20'
# Response: rank 21-40, has_more: true

# Continue until has_more=false
```

### Special Responses

When the product exists but has no embedding:

```json
{
  "product_id": "...",
  "results": [],
  "pagination": { "offset": 0, "limit": 20, "returned": 0, "has_more": false },
  "message": "Product has no embedding for similarity search"
}
```

### Examples

```bash
# Basic similar items (first 20)
curl 'http://localhost:8000/api/recs/similar/e8e5e4c1-d674-4555-af2f-605e152668af'

# Carousel - top 10
curl 'http://localhost:8000/api/recs/similar/e8e5e4c1-d674-4555-af2f-605e152668af?limit=10'

# Page 2
curl 'http://localhost:8000/api/recs/similar/e8e5e4c1-d674-4555-af2f-605e152668af?limit=20&offset=20'

# Filter to dresses only
curl 'http://localhost:8000/api/recs/similar/e8e5e4c1-d674-4555-af2f-605e152668af?category=dresses'
```

---

## POST /api/women/complete-fit

Returns complementary items to complete an outfit with the given product. Uses FashionCLIP semantic search to find items from complementary categories.

**Auth:** JWT required (`Authorization: Bearer <token>`).

### How It Works

1. Fetches source product details from Supabase
2. Determines complementary categories based on source category
3. Builds a FashionCLIP text query per category describing the ideal complement
   (e.g., *"jacket or coat to wear with Off White top, blouse, or sweater"*)
4. Runs pgvector similarity search per category
5. Returns ranked results with pagination per category

### Two Modes

- **Carousel mode** (default): Omit `category`. Returns top N items from **all** complementary categories. Use for product detail page carousels.
- **Feed mode**: Set `category` to a specific value. Returns paginated items for **one** category only. Use for "See All" infinite scroll.

### Request Body

```json
{
  "product_id": "e8e5e4c1-d674-4555-af2f-605e152668af",
  "items_per_category": 4,
  "category": null,
  "offset": 0,
  "limit": null
}
```

### Parameters

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `product_id` | string (UUID) | - | Yes | The product to complete the outfit for |
| `items_per_category` | int (1-20) | 4 | No | Items per category in carousel mode |
| `category` | string | null | No | Target category for feed mode (e.g., `"tops"`, `"outerwear"`) |
| `offset` | int | 0 | No | Skip first N items (feed mode only) |
| `limit` | int (1-100) | null | No | Max items to return (feed mode, overrides items_per_category) |

### Complementary Categories

Based on the source product's category:

| Source Category | Complementary Categories |
|-----------------|--------------------------|
| `tops` | bottoms, outerwear |
| `bottoms` | tops, outerwear |
| `dresses` | outerwear |
| `outerwear` | tops, bottoms, dresses |

### Response (Carousel Mode)

```json
{
  "source_product": {
    "product_id": "e8e5e4c1-d674-4555-af2f-605e152668af",
    "name": "The Chunky Jumper",
    "brand": "Mother Denim",
    "category": "tops",
    "price": 297.50,
    "base_color": "Off White",
    "colors": ["Off White"],
    "occasions": null,
    "usage": null,
    "image_url": "https://usepose.s3.us-east-1.amazonaws.com/products/.../primary.jpg"
  },
  "recommendations": {
    "bottoms": {
      "items": [
        {
          "product_id": "a1b2c3d4-...",
          "name": "Madelyn Shorts",
          "brand": "Brandy Melville",
          "category": "bottoms",
          "broad_category": null,
          "price": 14.00,
          "image_url": "https://...",
          "gallery_images": ["https://..."],
          "colors": ["Cream"],
          "materials": ["Cotton"],
          "similarity": 0.351,
          "rank": 1
        },
        {
          "product_id": "e5f6g7h8-...",
          "similarity": 0.346,
          "name": "Lounge Henley Short",
          "brand": "Nasty Gal",
          "price": 8.00,
          "rank": 2
        }
      ],
      "pagination": {
        "offset": 0,
        "limit": 4,
        "returned": 4,
        "has_more": true
      }
    },
    "outerwear": {
      "items": [
        {
          "product_id": "i9j0k1l2-...",
          "name": "Wool Look Boucle Belted Jacket",
          "brand": "Boohoo",
          "price": 39.00,
          "similarity": 0.356,
          "rank": 1
        }
      ],
      "pagination": {
        "offset": 0,
        "limit": 4,
        "returned": 4,
        "has_more": true
      }
    }
  },
  "queries_used": {
    "bottoms": "pants, skirt, or shorts to wear with Off White top, blouse, or sweater",
    "outerwear": "jacket or coat to wear with Off White top, blouse, or sweater"
  },
  "complete_outfit": {
    "items": ["e8e5e4c1-...", "a1b2c3d4-...", "e5f6g7h8-...", "i9j0k1l2-..."],
    "total_price": 1159.00,
    "item_count": 7
  }
}
```

### Response (Feed Mode)

When `category` is specified, returns only that category with pagination:

```json
{
  "source_product": { "..." : "..." },
  "recommendations": {
    "bottoms": {
      "items": [
        { "rank": 1, "name": "Madelyn Shorts", "brand": "Brandy Melville", "..." : "..." },
        { "rank": 2, "name": "Lounge Henley Short", "brand": "Nasty Gal", "..." : "..." },
        { "rank": 3, "..." : "..." },
        { "rank": 4, "..." : "..." },
        { "rank": 5, "..." : "..." }
      ],
      "pagination": {
        "offset": 0,
        "limit": 5,
        "returned": 5,
        "has_more": true
      }
    }
  },
  "queries_used": { "bottoms": "pants, skirt, or shorts to wear with ..." },
  "complete_outfit": { "items": ["..."], "total_price": 120.50, "item_count": 6 }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `source_product` | object | Details of the input product |
| `source_product.product_id` | string | Source product UUID |
| `source_product.name` | string | Product name |
| `source_product.brand` | string | Brand name |
| `source_product.category` | string | Product category |
| `source_product.price` | float | Current price |
| `source_product.base_color` | string | Primary color |
| `source_product.colors` | array | All colors |
| `source_product.occasions` | array/null | Occasion tags |
| `source_product.image_url` | string | Main image URL |
| `recommendations` | object | Map of category name to items + pagination |
| `recommendations[cat].items` | array | Complementary items for this category |
| `recommendations[cat].items[].product_id` | string | Product UUID |
| `recommendations[cat].items[].name` | string | Product name |
| `recommendations[cat].items[].brand` | string | Brand |
| `recommendations[cat].items[].price` | float | Price |
| `recommendations[cat].items[].similarity` | float | FashionCLIP similarity score (0-1) |
| `recommendations[cat].items[].rank` | int | 1-indexed rank within category (continuous across pages) |
| `recommendations[cat].items[].image_url` | string | Main image |
| `recommendations[cat].items[].gallery_images` | array | Gallery image URLs |
| `recommendations[cat].items[].colors` | array | Color tags |
| `recommendations[cat].items[].materials` | array | Material tags |
| `recommendations[cat].pagination.offset` | int | Current offset |
| `recommendations[cat].pagination.limit` | int | Items requested |
| `recommendations[cat].pagination.returned` | int | Items actually returned |
| `recommendations[cat].pagination.has_more` | boolean | More items available for pagination |
| `queries_used` | object | FashionCLIP text queries generated per category |
| `complete_outfit.items` | array | All product IDs (source + recommended) |
| `complete_outfit.total_price` | float | Sum of all item prices |
| `complete_outfit.item_count` | int | Total items in outfit |

### Examples

```bash
# Carousel mode - top 4 items per complementary category
curl -X POST 'http://localhost:8000/api/women/complete-fit' \
  -H 'Authorization: Bearer <JWT_TOKEN>' \
  -H 'Content-Type: application/json' \
  -d '{
    "product_id": "e8e5e4c1-d674-4555-af2f-605e152668af",
    "items_per_category": 4
  }'

# Carousel mode - top 6 items per category
curl -X POST 'http://localhost:8000/api/women/complete-fit' \
  -H 'Authorization: Bearer <JWT_TOKEN>' \
  -H 'Content-Type: application/json' \
  -d '{
    "product_id": "e8e5e4c1-d674-4555-af2f-605e152668af",
    "items_per_category": 6
  }'

# Feed mode - first page of bottoms (infinite scroll)
curl -X POST 'http://localhost:8000/api/women/complete-fit' \
  -H 'Authorization: Bearer <JWT_TOKEN>' \
  -H 'Content-Type: application/json' \
  -d '{
    "product_id": "e8e5e4c1-d674-4555-af2f-605e152668af",
    "category": "bottoms",
    "offset": 0,
    "limit": 20
  }'

# Feed mode - page 2 of outerwear
curl -X POST 'http://localhost:8000/api/women/complete-fit' \
  -H 'Authorization: Bearer <JWT_TOKEN>' \
  -H 'Content-Type: application/json' \
  -d '{
    "product_id": "e8e5e4c1-d674-4555-af2f-605e152668af",
    "category": "outerwear",
    "offset": 20,
    "limit": 20
  }'
```

### Pagination (Feed Mode)

```bash
# Page 1: items 1-10
curl -X POST '/api/women/complete-fit' \
  -H 'Authorization: Bearer <JWT_TOKEN>' \
  -H 'Content-Type: application/json' \
  -d '{"product_id": "xxx", "category": "tops", "offset": 0, "limit": 10}'
# Response: ranks 1-10, has_more: true

# Page 2: items 11-20
curl -X POST '/api/women/complete-fit' \
  -H 'Authorization: Bearer <JWT_TOKEN>' \
  -H 'Content-Type: application/json' \
  -d '{"product_id": "xxx", "category": "tops", "offset": 10, "limit": 10}'
# Response: ranks 11-20, has_more: true

# Continue until has_more: false
```

---

## Use Cases

### Similar Items Carousel

Show horizontal carousel of similar products on product detail page:

```javascript
// Initial load - get first 10 similar items (no auth needed)
const response = await fetch(`/api/recs/similar/${productId}?limit=10`);
const { results, pagination } = await response.json();

// User scrolls carousel - load more
if (pagination.has_more) {
  const nextPage = await fetch(`/api/recs/similar/${productId}?limit=10&offset=10`);
}
```

### Complete the Look: Carousel + Feed

1. Show carousels per category (carousel mode)
2. User taps "See All" on a category
3. Load infinite scroll for that category (feed mode)

```javascript
const headers = {
  'Authorization': `Bearer ${jwtToken}`,
  'Content-Type': 'application/json'
};

// 1. Initial carousel view
const carouselResponse = await fetch('/api/women/complete-fit', {
  method: 'POST',
  headers,
  body: JSON.stringify({
    product_id: productId,
    items_per_category: 4
  })
});
const { source_product, recommendations } = await carouselResponse.json();

// Display carousels for each category
Object.entries(recommendations).forEach(([category, data]) => {
  renderCarousel(category, data.items, data.pagination.has_more);
});

// 2. User taps "See All" on tops
const feedResponse = await fetch('/api/women/complete-fit', {
  method: 'POST',
  headers,
  body: JSON.stringify({
    product_id: productId,
    category: 'tops',
    offset: 0,
    limit: 20
  })
});

// 3. Infinite scroll - load more
const loadMore = async (currentOffset) => {
  const response = await fetch('/api/women/complete-fit', {
    method: 'POST',
    headers,
    body: JSON.stringify({
      product_id: productId,
      category: 'tops',
      offset: currentOffset,
      limit: 20
    })
  });
  const { recommendations } = await response.json();
  const { items, pagination } = recommendations.tops;
  appendItems(items);
  return pagination.has_more;
};
```

---

## Error Responses

### 401 Unauthorized (complete-fit only)

```json
{
  "detail": "Authorization header required"
}
```

### 404 Not Found

```json
{
  "detail": "Product not found: <product_id>"
}
```

### 503 Service Unavailable

```json
{
  "detail": "Search engine not available: <error>"
}
```

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.2 | 2026-02-11 | Wired `complete-fit` route to FastAPI (was logic-only, no endpoint). Added JWT auth requirement. Updated docs with real response examples. |
| 1.1 | 2025-01-26 | Added pagination to complete-fit (carousel + feed modes) |
| 1.0 | 2025-01-26 | Added pagination and rank to similar items endpoint |
