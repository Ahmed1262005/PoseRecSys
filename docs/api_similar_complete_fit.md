# Similar Items & Complete the Look API Documentation

## Overview

Two endpoints for product-based recommendations:

| Endpoint | Description |
|----------|-------------|
| `GET /api/recs/similar/{product_id}` | Similar Items - Products similar to a given item |
| `POST /api/women/complete-fit` | Complete the Look - Complementary items to complete an outfit |

---

## GET /api/recs/similar/{product_id}

Returns products similar to a given product, based on visual and semantic similarity.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `product_id` | string (UUID) | The product to find similar items for |

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gender` | string | "female" | Gender filter |
| `category` | string | null | Filter by category (e.g., "tops", "dresses") |
| `limit` | int | 20 | Items per page (1-100) |
| `offset` | int | 0 | Skip first N items for pagination |

### Response

```json
{
  "product_id": "f2165d58-7ce1-4664-af34-1d8b5962939b",
  "similar_products": [
    {
      "product_id": "a1234567-89ab-cdef-0123-456789abcdef",
      "name": "Silk Midi Dress",
      "brand": "Banana Republic",
      "category": "dresses",
      "price": 149.99,
      "image_url": "https://...",
      "gallery_images": ["https://...", "https://..."],
      "colors": ["navy", "black"],
      "materials": ["100% Silk"],
      "similarity": 0.892,
      "rank": 1
    },
    {
      "product_id": "b2345678-90bc-def0-1234-567890abcdef",
      "name": "Pleated Midi Dress",
      "brand": "Mango",
      "category": "dresses",
      "price": 89.99,
      "image_url": "https://...",
      "gallery_images": ["https://...", "https://..."],
      "colors": ["burgundy"],
      "materials": ["Polyester"],
      "similarity": 0.845,
      "rank": 2
    }
  ],
  "total_similar": 156,
  "has_more": true
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `product_id` | string | The source product ID |
| `similar_products` | array | List of similar products |
| `similar_products[].similarity` | float | Similarity score (0-1, higher = more similar) |
| `similar_products[].rank` | int | 1-indexed rank (continuous across pages) |
| `total_similar` | int | Total number of similar products available |
| `has_more` | boolean | Whether more pages are available |

### Pagination

Use `offset` and `limit` for pagination:

```bash
# Page 1: items 1-20
curl '/api/recs/similar/{id}?limit=20&offset=0'
# Response: rank 1-20, has_more: true

# Page 2: items 21-40
curl '/api/recs/similar/{id}?limit=20&offset=20'
# Response: rank 21-40, has_more: true

# Continue until has_more=false
```

### Examples

```bash
# Basic similar items
curl 'http://localhost:8000/api/recs/similar/f2165d58-7ce1-4664-af34-1d8b5962939b'

# Similar items - first 10
curl 'http://localhost:8000/api/recs/similar/f2165d58-7ce1-4664-af34-1d8b5962939b?limit=10'

# Similar items - page 2
curl 'http://localhost:8000/api/recs/similar/f2165d58-7ce1-4664-af34-1d8b5962939b?limit=20&offset=20'

# Similar items filtered by category
curl 'http://localhost:8000/api/recs/similar/f2165d58-7ce1-4664-af34-1d8b5962939b?category=dresses'
```

---

## POST /api/women/complete-fit

Returns complementary items to complete an outfit with the given product. Supports two modes:
1. **Carousel mode** (default): Returns top N items from each complementary category
2. **Feed mode**: Returns paginated items from a single category for infinite scroll

### Request Body

```json
{
  "product_id": "f2165d58-7ce1-4664-af34-1d8b5962939b",
  "items_per_category": 4,
  "category": null,
  "offset": 0,
  "limit": null
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `product_id` | string (UUID) | **required** | The product to complete the outfit for |
| `items_per_category` | int | 4 | Items per category in carousel mode |
| `category` | string | null | Target category for feed mode (e.g., "tops", "outerwear") |
| `offset` | int | 0 | Skip first N items (feed mode only) |
| `limit` | int | null | Max items to return (feed mode only) |

### Mode Selection

- **Carousel mode**: Omit `category` or set to `null`. Returns all complementary categories.
- **Feed mode**: Set `category` to a specific category. Returns paginated items for that category only.

### Response (Carousel Mode)

```json
{
  "source_product": {
    "product_id": "f2165d58-7ce1-4664-af34-1d8b5962939b",
    "name": "Corduroy Trousers",
    "brand": "7 For All Mankind",
    "category": "bottoms",
    "price": 224.00,
    "base_color": "Oxnard",
    "colors": ["Oxnard"],
    "occasions": ["everyday", "office"],
    "usage": null,
    "image_url": "https://..."
  },
  "recommendations": {
    "tops": {
      "items": [
        {
          "product_id": "a1234567-89ab-cdef-0123-456789abcdef",
          "similarity": 0.346,
          "name": "Silk Blouse",
          "brand": "Free People",
          "category": "tops",
          "broad_category": null,
          "price": 89.99,
          "image_url": "https://...",
          "gallery_images": ["https://...", "https://..."],
          "colors": ["Red"],
          "materials": ["Polyester", "Viscose"],
          "rank": 1
        },
        {
          "product_id": "b2345678-90bc-def0-1234-567890abcdef",
          "similarity": 0.338,
          "name": "Knit Sweater",
          "brand": "Boohoo",
          "category": "tops",
          "broad_category": null,
          "price": 45.00,
          "image_url": "https://...",
          "gallery_images": ["https://..."],
          "colors": ["Stone"],
          "materials": ["Soft knit fabric"],
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
          "product_id": "c3456789-0abc-def1-2345-6789abcdef01",
          "similarity": 0.329,
          "name": "Double Breasted Coat",
          "brand": "Ann Taylor",
          "category": "outerwear",
          "price": 204.99,
          "image_url": "https://...",
          "gallery_images": ["https://..."],
          "colors": ["Jewel Red"],
          "materials": ["52% Polyester", "48% Wool"],
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
    "tops": "top, blouse, or sweater to wear with Oxnard pants",
    "outerwear": "jacket or coat to wear with Oxnard pants"
  },
  "complete_outfit": {
    "items": ["f2165d58-...", "a1234567-...", "c3456789-..."],
    "total_price": 518.98,
    "item_count": 3
  }
}
```

### Response (Feed Mode)

When `category` is specified, returns only that category with pagination:

```json
{
  "source_product": { ... },
  "recommendations": {
    "tops": {
      "items": [
        { "rank": 1, ... },
        { "rank": 2, ... },
        { "rank": 3, ... },
        { "rank": 4, ... },
        { "rank": 5, ... }
      ],
      "pagination": {
        "offset": 0,
        "limit": 5,
        "returned": 5,
        "has_more": true
      }
    }
  },
  "queries_used": { "tops": "..." },
  "complete_outfit": { ... }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `source_product` | object | Details of the input product |
| `recommendations` | object | Map of category -> items + pagination |
| `recommendations[category].items` | array | Complementary items for this category |
| `recommendations[category].items[].similarity` | float | Semantic similarity score |
| `recommendations[category].items[].rank` | int | 1-indexed rank within category |
| `recommendations[category].pagination` | object | Pagination info for this category |
| `recommendations[category].pagination.has_more` | boolean | Whether more items available |
| `queries_used` | object | CLIP queries used for each category |
| `complete_outfit` | object | Summary of all recommended items |

### Complementary Categories

Based on the source product category:

| Source Category | Complementary Categories |
|-----------------|-------------------------|
| `tops` | bottoms, outerwear, dresses |
| `bottoms` | tops, outerwear |
| `dresses` | outerwear |
| `outerwear` | tops, bottoms, dresses |

### Pagination (Feed Mode)

For infinite scroll on a specific category:

```bash
# Page 1: tops items 1-10
curl -X POST '/api/women/complete-fit' \
  -d '{"product_id": "xxx", "category": "tops", "offset": 0, "limit": 10}'
# Response: ranks 1-10, has_more: true

# Page 2: tops items 11-20
curl -X POST '/api/women/complete-fit' \
  -d '{"product_id": "xxx", "category": "tops", "offset": 10, "limit": 10}'
# Response: ranks 11-20, has_more: true

# Continue until has_more: false
```

### Examples

```bash
# Carousel mode - get top 4 items per category
curl -X POST 'http://localhost:8000/api/women/complete-fit' \
  -H 'Content-Type: application/json' \
  -d '{
    "product_id": "f2165d58-7ce1-4664-af34-1d8b5962939b",
    "items_per_category": 4
  }'

# Carousel mode - get top 6 items per category
curl -X POST 'http://localhost:8000/api/women/complete-fit' \
  -H 'Content-Type: application/json' \
  -d '{
    "product_id": "f2165d58-7ce1-4664-af34-1d8b5962939b",
    "items_per_category": 6
  }'

# Feed mode - first page of tops
curl -X POST 'http://localhost:8000/api/women/complete-fit' \
  -H 'Content-Type: application/json' \
  -d '{
    "product_id": "f2165d58-7ce1-4664-af34-1d8b5962939b",
    "category": "tops",
    "offset": 0,
    "limit": 20
  }'

# Feed mode - second page of outerwear
curl -X POST 'http://localhost:8000/api/women/complete-fit' \
  -H 'Content-Type: application/json' \
  -d '{
    "product_id": "f2165d58-7ce1-4664-af34-1d8b5962939b",
    "category": "outerwear",
    "offset": 20,
    "limit": 20
  }'
```

---

## Use Cases

### Similar Items Carousel

Show horizontal carousel of similar products:

```javascript
// Initial load - get first 10 similar items
const response = await fetch(`/api/recs/similar/${productId}?limit=10`);
const { similar_products, has_more } = await response.json();

// User scrolls carousel - load more
if (has_more) {
  const nextPage = await fetch(`/api/recs/similar/${productId}?limit=10&offset=10`);
}
```

### Complete the Look Carousel + Feed

1. Show carousels per category (carousel mode)
2. User taps "See All" on a category
3. Load infinite scroll for that category (feed mode)

```javascript
// 1. Initial carousel view
const carouselResponse = await fetch('/api/women/complete-fit', {
  method: 'POST',
  body: JSON.stringify({
    product_id: productId,
    items_per_category: 4
  })
});
const { recommendations } = await carouselResponse.json();

// Display carousels for each category
Object.entries(recommendations).forEach(([category, data]) => {
  renderCarousel(category, data.items, data.pagination.has_more);
});

// 2. User taps "See All" on tops
const feedResponse = await fetch('/api/women/complete-fit', {
  method: 'POST',
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
    body: JSON.stringify({
      product_id: productId,
      category: 'tops',
      offset: currentOffset,
      limit: 20
    })
  });
  const { recommendations } = await response.json();
  const { items, pagination } = recommendations.tops;
  // Append items to list
  // Check pagination.has_more for more pages
};
```

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "product_id is required"
}
```

### 404 Not Found

```json
{
  "detail": "Product not found"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Internal server error"
}
```

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.1 | 2025-01-26 | Added pagination to complete-fit (carousel + feed modes) |
| 1.0 | 2025-01-26 | Added pagination and rank to similar items endpoint |
