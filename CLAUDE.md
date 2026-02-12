# OutfitTransformer - Fashion Recommendation System

## Project Overview

A production fashion recommendation system with style preference learning, personalized recommendations, and hybrid search.

**Core Features:**
- **Style Learning**: Tinder-style 4-choice interface to learn user preferences
- **Personalized Feed**: Recommendations based on learned taste vectors
- **Hybrid Search**: Algolia (lexical) + FashionCLIP (semantic) with RRF merging
- **Supabase Integration**: pgvector for similarity search, persistent storage
- **JWT Authentication**: All endpoints require Supabase JWT auth

---

## Project Structure

```
src/
├── api/
│   ├── app.py                          # FastAPI application factory
│   └── routes/
│       ├── health.py                   # Health check endpoints
│       ├── women.py                    # Women's fashion style learning
│       ├── unified.py                  # Gender-aware style learning
│       └── search.py                   # Hybrid search endpoints
│
├── config/
│   ├── settings.py                     # App settings (env vars)
│   ├── database.py                     # Supabase client setup
│   └── constants.py                    # Shared constants
│
├── core/
│   ├── auth.py                         # JWT authentication (Supabase)
│   ├── logging.py                      # Structured logging
│   ├── middleware.py                    # Request tracing middleware
│   └── utils.py                        # Utilities (convert_numpy, etc.)
│
├── engines/
│   ├── factory.py                      # Engine factory (get_engine, get_search_engine)
│   ├── swipe_engine.py                 # Base Tinder-style engine
│   └── predictive_four_engine.py       # Predictive four-choice with learning
│
├── search/                             # Hybrid search module
│   ├── __init__.py                     # Module exports
│   ├── algolia_config.py              # Index settings, 30 synonyms, record mapping, replica config
│   ├── algolia_client.py              # Algolia v4 SearchClientSync wrapper (singleton)
│   ├── models.py                      # Pydantic request/response models (23 filters, SortBy enum)
│   ├── query_classifier.py            # Intent classification (exact/specific/vague)
│   ├── hybrid_search.py               # Main service: Algolia + FashionCLIP + RRF merge
│   ├── reranker.py                    # Session-aware reranking with dedup & diversity
│   ├── autocomplete.py                # Product + brand autocomplete
│   └── analytics.py                   # Search event tracking to Supabase
│
├── recs/                               # Recommendation pipeline
│   ├── api_endpoints.py                # FastAPI routes (/api/recs/*)
│   ├── candidate_selection.py          # Candidate retrieval from pgvector
│   ├── candidate_factory.py            # Candidate generation
│   ├── feasibility_filter.py           # Feasibility filtering
│   ├── filter_utils.py                 # Filter utilities
│   ├── models.py                       # Pydantic models
│   ├── pipeline.py                     # Main recommendation pipeline
│   ├── recommendation_service.py       # Legacy recommendation service
│   ├── sasrec_ranker.py                # SASRec model ranking
│   └── session_state.py                # Feed session management
│
├── services/
│   └── session_manager.py              # Style learning session state
│
├── gender_config.py                    # Gender-specific configuration
└── women_search_engine.py              # Women's fashion search (Supabase + FashionCLIP)

scripts/
├── index_to_algolia.py                 # Bulk indexing from Supabase -> Algolia
├── setup_algolia_replicas.py           # One-time: create virtual replicas for sort-by
└── test_search_gradio.py               # Gradio test UI (5 tabs, 23 filters)

sql/                                    # Database migrations
tests/                                  # Pytest test suite (144 unit tests passing)
├── unit/                               # Unit tests
└── integration/                        # Integration tests (need running server)
docs/                                   # API documentation
```

---

## API Server

### Start Server

```bash
cd /mnt/d/ecommerce/recommendationSystem
source .venv/bin/activate
PYTHONPATH=src uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**Production:**
```bash
PYTHONPATH=src uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

**Gradio Test UI:**
```bash
PYTHONPATH=src python scripts/test_search_gradio.py
# -> http://localhost:7860
```

---

## Authentication

All endpoints (except health checks and public info) require JWT authentication.

**Header:** `Authorization: Bearer <supabase_jwt_token>`

Public endpoints (no auth required):
- `/health`, `/ready`, `/live`, `/health/detailed`
- `/api/recs/v2/info`, `/api/recs/v2/health`
- `/api/recs/v2/categories/mapping`
- `/api/search/health`

---

## API Endpoints

### Health Checks

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Basic health check |
| `/health/detailed` | GET | No | Detailed health with dependency status |
| `/ready` | GET | No | Readiness probe |
| `/live` | GET | No | Liveness probe |

### Hybrid Search

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/search/hybrid` | POST | Yes | Hybrid search (Algolia + FashionCLIP), supports `sort_by` |
| `/api/search/autocomplete` | GET | Yes | Product + brand autocomplete |
| `/api/search/click` | POST | Yes | Track click event |
| `/api/search/conversion` | POST | Yes | Track conversion event |
| `/api/search/health` | GET | No | Search module health check |

### Women's Fashion Style Learning

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/women/session/start` | POST | Yes | Start a new style learning session |
| `/api/women/session/choose` | POST | Yes | Record user's choice (1 of 4) |
| `/api/women/session/skip` | POST | Yes | Skip all 4 items |
| `/api/women/session/summary` | GET | Yes | Get learned preferences |
| `/api/women/feed` | GET | Yes | Get personalized feed |
| `/api/women/search` | POST | Yes | Search women's fashion |

### Unified Gender-Aware Style Learning

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/unified/four/start` | POST | Yes | Start a gender-aware session |
| `/api/unified/four/choose` | POST | Yes | Record choice |
| `/api/unified/four/skip` | POST | Yes | Skip all 4 items |
| `/api/unified/four/summary/{gender}` | GET | Yes | Get session summary |

### Recommendation Pipeline (v2)

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/recs/v2/onboarding` | POST | Yes | Save onboarding profile |
| `/api/recs/v2/onboarding/core-setup` | POST | Yes | Save core-setup, get Tinder categories |
| `/api/recs/v2/onboarding/v3` | POST | Yes | Save V3 onboarding profile |
| `/api/recs/v2/feed` | GET | Yes | Get feed using full pipeline |
| `/api/recs/v2/sale` | GET | Yes | Get sale items feed |
| `/api/recs/v2/new-arrivals` | GET | Yes | Get new arrivals feed |
| `/api/recs/v2/feed/endless` | GET | Yes | Endless scroll feed |
| `/api/recs/v2/feed/keyset` | GET | Yes | Keyset pagination feed |
| `/api/recs/v2/feed/action` | POST | Yes | Record user interaction |
| `/api/recs/v2/session/sync` | POST | Yes | Sync session seen_ids |
| `/api/recs/v2/feed/session/{session_id}` | GET | Yes | Get session info |
| `/api/recs/v2/feed/session/{session_id}` | DELETE | Yes | Delete session |
| `/api/recs/v2/info` | GET | No | Pipeline configuration info |
| `/api/recs/v2/health` | GET | No | Pipeline health check |
| `/api/recs/v2/categories/mapping` | GET | No | Category mappings |

### Legacy Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/recs/save-preferences` | POST | No | Save Tinder test results |
| `/api/recs/feed/{user_id}` | GET | No | Get personalized feed |
| `/api/recs/similar/{product_id}` | GET | Yes | Get similar products |
| `/api/recs/trending` | GET | No | Get trending products |
| `/api/recs/categories` | GET | No | Get product categories |
| `/api/recs/product/{product_id}` | GET | No | Get product details |
| `/api/recs/health` | GET | No | Legacy health check |

---

## Hybrid Search Architecture

### Pipeline Flow (sort_by=relevance, default)
1. **Query Classification** - Classify intent as EXACT (brand), SPECIFIC (category+filters), or VAGUE
2. **Algolia Search** - Lexical/keyword search with filters and facets (19 facet fields)
3. **FashionCLIP Semantic Search** - Visual/semantic similarity via pgvector embeddings
4. **RRF Merge** - Reciprocal Rank Fusion combining both result sets
5. **Post-Filtering** - Strict attribute filtering on semantic results (None = excluded)
6. **Enrichment** - Batch-fetch Gemini attributes from Algolia for semantic results
7. **Reranking** - Session dedup, near-duplicate removal, profile boosts, brand diversity
8. **Facets** - Return filterable facet counts (>1 count, excludes null/N/A, 2+ distinct values)

### Sort-by (search)

The `sort_by` parameter controls how results are ranked. Two modes exist:

| `sort_by` value | Pipeline | Speed |
|-----------------|----------|-------|
| `relevance` (default) | Full hybrid: Algolia + FashionCLIP + RRF merge + reranker | ~15s |
| `price_asc` | Algolia-only via virtual replica `products_price_asc` | ~200ms |
| `price_desc` | Algolia-only via virtual replica `products_price_desc` | ~200ms |
| `trending` | Algolia-only via virtual replica `products_trending` | ~200ms |

**When `sort_by != relevance`:** Algolia-only fast path. Skips semantic search, RRF merge, and reranker
to preserve deterministic sort order. Uses Algolia-native pagination. All 23 filters still apply.

**Virtual replicas** share primary index data (no extra storage). Each overrides `customRanking`:
- `products_price_asc` → `asc(price)`
- `products_price_desc` → `desc(price)`
- `products_trending` → `desc(trending_score), desc(popularity_score)`

Setup: `PYTHONPATH=src python scripts/setup_algolia_replicas.py` (one-time, needs `ALGOLIA_WRITE_KEY`)

**Example request:**
```json
POST /api/search/hybrid
{
  "query": "dress",
  "sort_by": "price_asc",
  "brands": ["Boohoo"],
  "min_price": 20,
  "max_price": 100
}
```

**Response includes:** `"sort_by": "price_asc"` in the top-level response object.

### Search Filters (23 total)
Query, page, per_page, brands, exclude_brands, categories, colors, min_price, max_price,
patterns, occasions, styles, fit_types, necklines, sleeve_types, materials, lengths,
age_groups, aesthetics, body_types, versatility_scores, care_instructions, sustainability_ratings

### Query Classifier Intent Rules
- **EXACT**: Pure brand query (e.g., "Ba&sh", "Boohoo")
- **SPECIFIC**: Category keyword present (e.g., "office dress", "black tops")
- **VAGUE**: No category keywords, possibly just mood/occasion words

### Algolia v4 API Notes (CRITICAL)
- `SearchClientSync(app_id, api_key)` -- NOT `SearchClient.create()`
- No `init_index()` -- all methods take `index_name` as first param
- `search_single_index(index_name, search_params={...})` -- returns pydantic model, use `.to_dict()`
- `get_objects(get_objects_params={"requests": [{"objectID": id, "indexName": name}, ...]})` -- batch fetch

### Reranker (Current State)
1. Session dedup (remove seen_ids)
2. Near-duplicate removal (size-variant name normalization, sister-brand mapping, same-image detection)
3. Profile-based soft scoring (10 boost types + 3 penalty types, capped at +/-0.15)
4. Brand diversity cap (max 4 per brand)

---

## Style Learning Flow

1. **Start Session** - Initialize with optional category/color preferences
2. **Choose Items** - User picks favorite from 4 items shown
3. **System Learns** - Bayesian preference learning on attributes
4. **Get Feed** - Personalized recommendations based on learned taste

### Attribute Dimensions Learned

| Attribute | Example Values |
|-----------|---------------|
| pattern | solid, striped, floral, geometric |
| style | casual, office, evening, bohemian |
| color_family | neutral, bright, cool, pastel, dark |
| fit_vibe | fitted, relaxed, oversized, cropped |
| neckline | crew, v_neck, off_shoulder, sweetheart |
| occasion | everyday, work, date_night, party |
| sleeve_type | sleeveless, short, long, puff |

### Women's Fashion Categories

- `tops_knitwear` - Sweaters, cardigans, knit tops
- `tops_woven` - Blouses, shirts, woven tops
- `tops_sleeveless` - Tank tops, camisoles
- `tops_special` - Bodysuits, special tops
- `dresses` - All dress types
- `bottoms_trousers` - Pants, jeans, trousers
- `bottoms_skorts` - Skirts, shorts
- `outerwear` - Jackets, coats
- `sportswear` - Athletic wear, leggings

---

## Database Inventory

```
products table total:       118,792
products (in_stock):         96,558
image_embeddings rows:      170,174  (multiple per product, pgvector)
product_attributes (Gemini): 115,874
Algolia index records:       94,000  (full coverage of in-stock products)
Distinct brands:             131
Top brands: Boohoo (20K), Missguided (9K), Forever 21 (7K), Princess Polly (7K)
```

- pgvector RPC `text_search_products` uses `DISTINCT ON (p.id)` to deduplicate multi-image products
- Embedding table: `image_embeddings` with `sku_id` FK to `products.id`
- Gemini attributes table: `product_attributes` with `sku_id` FK to `products.id`

---

## Dependencies

```
# API
fastapi
uvicorn[standard]

# Database
supabase

# ML
recbole
faiss-cpu
transformers

# Search
algoliasearch>=4.36.0

# Data Processing
numpy<2.0
pillow

# Auth
PyJWT

# Testing
pytest
pytest-asyncio
httpx

# UI
gradio>=6.5.1

# Utilities
pydantic
python-dotenv
```

Install:
```bash
pip install -r requirements.txt
```

---

## Testing

Run all tests:
```bash
PYTHONPATH=src python -m pytest tests/ -v
```

Run unit tests only (144 passing):
```bash
PYTHONPATH=src python -m pytest tests/unit/ -v
```

Run search tests:
```bash
PYTHONPATH=src python -m pytest tests/unit/test_search.py -v
```

Run integration tests (requires running server):
```bash
TEST_SERVER_URL=http://localhost:8000 PYTHONPATH=src python -m pytest tests/integration/ -v
```

---

## Environment Variables

```bash
# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_JWT_SECRET=your_jwt_secret

# Algolia
ALGOLIA_APP_ID=your_algolia_app_id
ALGOLIA_SEARCH_KEY=your_algolia_search_key
ALGOLIA_WRITE_KEY=your_algolia_write_key
ALGOLIA_INDEX_NAME=products

# API (optional)
HOST=0.0.0.0
PORT=8000
WORKERS=4
```

---

## Development Session History

### Phase 1: Core Search Module (Complete)
Built entire `src/search/` module from scratch:
- Algolia v4 client with singleton pattern, batch `get_objects()`, facets
- Hybrid search service (Algolia + FashionCLIP + RRF merge)
- Query classifier with intent detection and brand extraction
- 23-filter Pydantic models with price validation
- Session-aware reranker with dedup and brand diversity
- Autocomplete (products first, then brands)
- Analytics tracking (queries, clicks, conversions) to Supabase
- FastAPI routes for all search endpoints
- Bulk indexing script (Supabase -> Algolia)
- Gradio test UI with 5 tabs

### Phase 2: Critical Bug Fixes (Complete)
1. **FashionCLIP encode_text constant vector** - transformers 5.x returns `BaseModelOutputWithPooling`; fixed to use `pooler_output` instead of CLS token
2. **Query classifier occasion+category -> VAGUE** - reordered intent logic: category keywords checked before vague
3. **Semantic results missing Gemini attributes** - added batch enrichment via `AlgoliaClient.get_objects()`
4. **Strict post-filtering** - None values now excluded (not passed through); all 23 filter types enforced
5. **Near-duplicate removal** - size-variant normalization, sister-brand mapping, same-image detection
6. **Brand search special characters** - `extract_brand()` + `html.unescape()` for `&amp;` encoding
7. **Algolia re-index** - 85,558 of 96,558 in-stock products indexed
8. **Faceted search** - 19 facet fields, filtered to count>1, excludes null/N/A

### Phase 3: Gradio Test UI Enhancement (Complete)
5 tabs: Hybrid Search (all 23 filters), Compare Queries, Autocomplete, Click Analytics, Quick Tests (28 automated)

### Next: Reranker Overhaul (Pending)
Discussion started on rebuilding reranker with more capabilities.

---

## License

MIT
