# Recommendation API - Quick Reference

## Base URL
```
http://localhost:8080/api/recs
```

---

## Endpoints at a Glance

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/save-preferences` | Save user preferences |
| `GET` | `/feed/{user_id}` | Get personalized feed |
| `GET` | `/similar/{product_id}` | Get similar products |
| `GET` | `/trending` | Get trending products |
| `GET` | `/categories` | Get product categories |
| `GET` | `/product/{product_id}` | Get product details |

---

## Common Requests

### 1. Save Preferences (After Tinder Test)
```bash
curl -X POST "http://localhost:8080/api/recs/save-preferences" \
  -H "Content-Type: application/json" \
  -d '{
    "anon_id": "user_123",
    "gender": "female",
    "rounds_completed": 12,
    "taste_vector": [0.1, -0.2, ...]
  }'
```

### 2. Get Personalized Feed
```bash
curl "http://localhost:8080/api/recs/feed/user_123?gender=female&limit=20"
```

### 3. Get Similar Products
```bash
curl "http://localhost:8080/api/recs/similar/PRODUCT_UUID?limit=10"
```

### 4. Get Trending (Cold Start)
```bash
curl "http://localhost:8080/api/recs/trending?gender=female&limit=20"
```

### 5. Get Categories
```bash
curl "http://localhost:8080/api/recs/categories?gender=female"
```

---

## Response Strategy Values

| Strategy | Meaning | User State |
|----------|---------|------------|
| `seed_vector` | Personalized recommendations | Has taste vector |
| `trending` | Popular products fallback | New user / no preferences |

---

## Key Response Fields

### Feed Response
```json
{
  "strategy": "seed_vector",    // or "trending"
  "results": [...],
  "metadata": {
    "seed_vector_available": true,
    "seed_source": "tinder"     // or "attributes" or "none"
  }
}
```

### Product in Results
```json
{
  "product_id": "uuid",
  "name": "Product Name",
  "brand": "Brand",
  "category": "tops",
  "price": 49.99,
  "similarity": 0.89,           // 0-1 (higher = better match)
  "reason": "style_matched"     // or "trending"
}
```

---

## Integration Flow

```
User Takes Style Test
        │
        ▼
POST /save-preferences  ──►  Store taste_vector
        │
        ▼
GET /feed/{user_id}     ──►  Personalized results (strategy: "seed_vector")
        │
        ▼
User Views Product
        │
        ▼
GET /similar/{id}       ──►  Visual similarity results
```

---

## Query Parameters

| Parameter | Endpoints | Values | Default |
|-----------|-----------|--------|---------|
| `gender` | feed, similar, trending, categories | `female`, `male` | `female` |
| `limit` | feed, similar, trending | 1-200 | 50 |
| `offset` | feed, trending | 0+ | 0 |
| `category` | similar, trending | category name | None |
| `categories` | feed | comma-separated | None |

---

## Pagination

Both `/feed` and `/trending` support offset-based pagination:

```bash
# Page 1
curl "http://localhost:8080/api/recs/feed/user_123?limit=20&offset=0"

# Page 2
curl "http://localhost:8080/api/recs/feed/user_123?limit=20&offset=20"

# Page 3
curl "http://localhost:8080/api/recs/feed/user_123?limit=20&offset=40"
```

**Response includes:**
```json
{
  "metadata": {
    "offset": 20,
    "has_more": true
  }
}
```

---

## Error Responses

```json
{"detail": "Error message"}
```

| Code | Meaning |
|------|---------|
| 400 | Missing required parameter |
| 404 | Product/user not found |
| 500 | Server error |

---

## Taste Vector

- **Dimensions:** 512 (FashionCLIP embedding)
- **Source:** Averaged embeddings from liked items in Tinder test
- **Required for:** Personalized recommendations (`seed_vector` strategy)

---

## Categories (Female)

| Category | Example Products |
|----------|-----------------|
| `tops` | Sweaters, blouses, t-shirts |
| `bottoms` | Pants, jeans, trousers |
| `dresses` | All dress styles |
| `outerwear` | Jackets, coats |

---

## Test with cURL

```bash
# Health check
curl http://localhost:8080/api/recs/health

# Full flow test
USER_ID="test_$(date +%s)"

# Save dummy preferences
curl -X POST "http://localhost:8080/api/recs/save-preferences" \
  -H "Content-Type: application/json" \
  -d "{\"anon_id\": \"$USER_ID\", \"gender\": \"female\"}"

# Get feed (will be trending since no taste_vector)
curl "http://localhost:8080/api/recs/feed/$USER_ID?limit=5"
```
