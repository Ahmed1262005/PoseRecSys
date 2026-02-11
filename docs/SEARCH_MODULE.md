# Hybrid Search Module - Development History & Architecture

## Overview

This document captures the complete development history of the hybrid search module (`src/search/`), including architecture decisions, bug fixes, and the current state of the system.

---

## Phase 1: Core Search Module (Complete)

Built the entire `src/search/` module from scratch.

### Files Created

| File | Purpose | Status |
|------|---------|--------|
| `src/search/__init__.py` | Module exports | Done |
| `src/search/algolia_config.py` | Index settings, 30 synonyms, record mapping | Done |
| `src/search/algolia_client.py` | Algolia v4 `SearchClientSync` wrapper (singleton), `get_objects()` batch lookup, `facets` param | Done |
| `src/search/models.py` | Pydantic request/response models with 23 filters, `FacetValue`, price validation | Done |
| `src/search/query_classifier.py` | Intent classification (exact/specific/vague), brand loading from Algolia facets, `extract_brand()` | Done |
| `src/search/hybrid_search.py` | Main service: Algolia + FashionCLIP merge via RRF, semantic post-filter, enrichment, facets | Done |
| `src/search/reranker.py` | Session-aware reranking with dedup, brand diversity, profile boosts | Done |
| `src/search/autocomplete.py` | Products first, then brands, with error handling + in_stock filter | Done |
| `src/search/analytics.py` | Search event tracking (queries, clicks, conversions) to Supabase | Done |
| `src/api/routes/search.py` | FastAPI endpoints: `/api/search/hybrid`, `/autocomplete`, `/click`, `/conversion`, `/health` | Done |
| `scripts/index_to_algolia.py` | Bulk indexing from Supabase -> Algolia | Done |
| `scripts/test_search_gradio.py` | Gradio test UI with 5 tabs, all 23 filters, comparison mode | Done |

---

## Phase 2: Critical Bug Fixes (Complete)

### Bug 1: FashionCLIP encode_text Returning Constant Vector (CRITICAL)

**File:** `src/women_search_engine.py:313-343`

**Symptoms:** Every query returned the same results. Cosine similarity between any two query embeddings was 1.0.

**Root Cause:** transformers 5.x changed the return type of `get_text_features()`. It now returns `BaseModelOutputWithPooling` (a named tuple) instead of a raw tensor. The old code fell through to `last_hidden_state[:, 0, :]` (CLS token extraction), which produced **identical embeddings for all queries**.

**Fix:** Use `pooler_output` (the projected embedding after the pooling layer) instead of CLS token:
```python
# Before (broken):
output = model.get_text_features(**inputs)
# Falls through to: output.last_hidden_state[:, 0, :]  # CLS token - SAME for all queries

# After (fixed):
output = model.get_text_features(**inputs)
if hasattr(output, 'pooler_output'):
    embedding = output.pooler_output  # Correct projected embedding
```

**Verification:** "office dress" vs "party dress" now gives 0.84 cosine similarity (correct differentiation).

---

### Bug 2: Query Classifier - occasion+category Classified as VAGUE

**File:** `src/search/query_classifier.py:180-210`

**Symptoms:** "office dress" classified as VAGUE instead of SPECIFIC, resulting in no Algolia filters being applied.

**Root Cause:** The classifier checked for vague keywords (mood/occasion words like "office", "party") *before* checking for category keywords ("dress", "top"). Since "office" matched the vague list first, it short-circuited.

**Fix:** Detect all keyword types first, then apply priority logic:
```python
# Detect all keyword types
has_category = check_category_keywords(query)
has_vague = check_vague_keywords(query)

# Priority: category presence always wins
if has_category:
    return Intent.SPECIFIC  # "office dress" -> SPECIFIC
elif has_vague:
    return Intent.VAGUE     # "office vibes" -> VAGUE
```

---

### Bug 3: Semantic Results Missing All Gemini Attributes

**Files:** `src/search/hybrid_search.py` + `src/search/algolia_client.py`

**Symptoms:** Semantic (FashionCLIP) results had no `occasion`, `style`, `pattern`, etc. attributes. Only Algolia results had these fields.

**Root Cause:** pgvector only stores embeddings and basic product fields. The Gemini-generated attributes (occasion, style, pattern, etc.) are stored in Algolia but not in pgvector.

**Fix:** Added an enrichment step after semantic search:
1. Added `AlgoliaClient.get_objects()` method for batch lookups by objectID
2. After semantic results are retrieved, batch-fetch all their Gemini attributes from Algolia
3. Merge attributes into semantic results (best-effort: products not in Algolia stay un-enriched)

---

### Bug 4: Strict Post-Filtering on Semantic Results

**File:** `src/search/hybrid_search.py` `_post_filter_semantic()`

**Symptoms:** Filtering by "occasion=work" returned products with no occasion attribute at all. Users expected only explicitly work-tagged products.

**Root Cause:** The old logic gave "benefit of the doubt" to products with `None` attributes — if a product had no `occasion` field, it passed through occasion filters.

**Fix:** Strict exclusion — if a filter is active and a product lacks that attribute (`None`), the product is **removed**:
```python
# Before: None passed through (false positives)
if product.occasion and product.occasion not in filter_values:
    exclude = True

# After: None excluded (strict)
if product.occasion is None or product.occasion not in filter_values:
    exclude = True
```

Applied to all 23 filter types. Added dynamic fetch multiplier (5x when filters active, 3x otherwise) to compensate for stricter filtering reducing result counts.

---

### Bug 5: Near-Duplicate Removal

**File:** `src/search/reranker.py`

**Symptoms:** Search results contained obvious duplicates: same dress in different sizes (e.g., "Floral Dress - Petite" and "Floral Dress"), same product from sister brands (Boohoo vs Nasty Gal), products with identical images.

**Fix:** Added `_deduplicate()` with 3 mechanisms:
1. **Size-variant name normalization** - Strips "Petite", "Plus", "Tall", "Regular" from product names and deduplicates on normalized name
2. **Sister-brand mapping** - Maps known sister brands to canonical names (Boohoo/Nasty Gal/Missguided -> canonical) and deduplicates across brand families
3. **Same-image detection** - Products sharing the exact same primary image URL are deduplicated

Also reduced brand diversity cap from 10 to 4 per brand.

---

### Bug 6: Brand Search with Special Characters

**Files:** `src/search/query_classifier.py` + `src/search/hybrid_search.py`

**Symptoms:** Searching "Ba&sh" returned wrong results because Algolia split the query on `&`.

**Fix:**
1. Added `extract_brand()` method to `QueryClassifier` that returns the matched brand name
2. When intent=EXACT, auto-injects the brand as an Algolia filter instead of relying on text matching
3. Added `html.unescape()` at pipeline start to handle `&amp;` encoding from frontends

---

### Bug 7: Algolia Re-index

Ran `scripts/index_to_algolia.py --clear-first`. Results:
- **Before:** 58,420 records (stale, many out-of-stock)
- **After:** 85,558 records (11K skipped due to timeouts on first batches)
- Source: 96,558 in-stock products

---

### Bug 8: Faceted Search Response

**Files:** `src/search/models.py` + `src/search/hybrid_search.py` + `src/search/algolia_client.py`

**Feature:** Added faceted search so the frontend can render filter dropdowns dynamically.

- Added `FacetValue` model (`value: str`, `count: int`)
- Added `facets: Dict[str, List[FacetValue]]` field to `HybridSearchResponse`
- Algolia search now requests 19 facet fields
- Facet post-processing: count > 1, excludes null/N/A, requires 2+ distinct values per facet (single-value facets omitted)

---

## Phase 3: Gradio Test UI Enhancement (Complete)

**File:** `scripts/test_search_gradio.py`

5 tabs:
1. **Hybrid Search** - All 23 filters as multi-select dropdowns, pagination, session dedup, gallery thumbnails
2. **Compare Queries** - Side-by-side comparison with overlap analysis
3. **Autocomplete** - Live suggestions
4. **Click Analytics** - Event logging
5. **Quick Tests** - 28 automated tests with PASS/FAIL results

Brand dropdown updated to actual database brands (131 brands, no H&M/Zara — they don't exist in the DB).

---

## Architecture Details

### Hybrid Search Pipeline Flow

```
User Query
    |
    v
[HTML Unescape] -- handles &amp; from frontends
    |
    v
[Query Classifier] -- EXACT / SPECIFIC / VAGUE
    |
    +---> [Algolia Search] -- lexical/keyword, 19 facet fields
    |         |
    |         v
    |     Algolia Results (with Gemini attributes)
    |
    +---> [FashionCLIP Semantic Search] -- pgvector embeddings
    |         |
    |         v
    |     [Post-Filter] -- strict filtering, None = excluded
    |         |
    |         v
    |     [Enrichment] -- batch fetch Gemini attributes from Algolia
    |         |
    |         v
    |     Semantic Results (enriched)
    |
    v
[RRF Merge] -- Reciprocal Rank Fusion (k=60)
    |
    v
[Reranker]
    |-- Session dedup (remove seen_ids)
    |-- Near-duplicate removal
    |-- Profile-based soft scoring
    |-- Brand diversity cap (max 4)
    |
    v
Final Results + Facets
```

### Search Filters (23 total)

| Filter | Type | Description |
|--------|------|-------------|
| `query` | string | Search query text |
| `page` | int | Page number |
| `per_page` | int | Results per page |
| `brands` | list[str] | Include only these brands |
| `exclude_brands` | list[str] | Exclude these brands |
| `categories` | list[str] | Product categories |
| `colors` | list[str] | Color filter |
| `min_price` | float | Minimum price |
| `max_price` | float | Maximum price |
| `patterns` | list[str] | Pattern types (floral, striped, etc.) |
| `occasions` | list[str] | Occasion types (work, party, etc.) |
| `styles` | list[str] | Style types (casual, formal, etc.) |
| `fit_types` | list[str] | Fit types (slim, relaxed, etc.) |
| `necklines` | list[str] | Neckline types |
| `sleeve_types` | list[str] | Sleeve types |
| `materials` | list[str] | Material types |
| `lengths` | list[str] | Length types (mini, midi, maxi) |
| `age_groups` | list[str] | Target age groups |
| `aesthetics` | list[str] | Aesthetic styles |
| `body_types` | list[str] | Body type recommendations |
| `versatility_scores` | list[str] | Versatility ratings |
| `care_instructions` | list[str] | Care instruction types |
| `sustainability_ratings` | list[str] | Sustainability ratings |

### Query Classifier Intent Rules

| Intent | Trigger | Example | Behavior |
|--------|---------|---------|----------|
| EXACT | Pure brand query | "Ba&sh", "Boohoo" | Brand injected as Algolia filter |
| SPECIFIC | Category keyword present | "office dress", "black tops" | Normal hybrid search with filters |
| VAGUE | No category keywords | "summer vibes", "date night" | Semantic search weighted higher |

### Reranker Current State

**Dedup mechanisms:**
1. Session dedup (remove `seen_ids` from session)
2. Size-variant name normalization (strips Petite/Plus/Tall)
3. Sister-brand mapping (Boohoo/Nasty Gal/Missguided -> canonical)
4. Same-image-URL detection

**Profile-based scoring (10 boost types, 3 penalty types):**
- Boosts: preferred brands, colors, patterns, styles, occasions, fit types, necklines, materials, price range, categories
- Penalties: avoided brands, disliked patterns, disliked colors
- All capped at +/-0.15

**Brand diversity:** Max 4 results per brand.

---

## Database Schema

### Products Table
- `id` (PK) - Product SKU ID
- `name` - Product name
- `brand` - Brand name
- `price` - Current price
- `original_price` - Original price
- `image_url` - Primary image URL
- `product_url` - Link to product page
- `in_stock` - Boolean
- `category` - Product category
- `color` - Product color

### Image Embeddings Table
- `sku_id` (FK -> products.id)
- `embedding` (pgvector) - FashionCLIP image embedding
- Multiple rows per product (one per image)

### Product Attributes Table (Gemini-generated)
- `sku_id` (FK -> products.id)
- `occasion`, `style`, `pattern`, `fit_type`, `neckline`, `sleeve_type`, `material`, `length`
- `age_group`, `aesthetic`, `body_type`, `versatility_score`, `care_instructions`, `sustainability_rating`

### Key Stats
```
products table total:       118,792
products (in_stock):         96,558
image_embeddings rows:      170,174
product_attributes (Gemini): 115,874
Algolia index records:       85,558
Distinct brands:             131
```

---

## Algolia Configuration

### Index Settings
- **Searchable attributes:** name, brand, category, color, occasion, style, pattern
- **Custom ranking:** desc(in_stock), desc(price)
- **Faceting:** 19 facet fields for dynamic filter dropdowns
- **Synonyms:** 30 custom synonyms (e.g., "tee" = "t-shirt", "pants" = "trousers")

### Algolia v4 API Notes (CRITICAL)
```python
# Client initialization (v4 - NOT v3)
from algoliasearch.search.client import SearchClientSync
client = SearchClientSync(app_id, api_key)  # NOT SearchClient.create()

# Search (no init_index)
response = client.search_single_index(
    index_name="products",
    search_params={"query": "dress", "hitsPerPage": 20}
)
result = response.to_dict()  # Returns pydantic model, must convert

# Batch fetch objects
client.get_objects(get_objects_params={
    "requests": [
        {"objectID": "123", "indexName": "products"},
        {"objectID": "456", "indexName": "products"}
    ]
})
```

---

## Files Modified During Bug Fix Phase

| File | What Changed |
|------|-------------|
| `src/women_search_engine.py:313-343` | Fixed `encode_text` - `pooler_output` instead of CLS token |
| `src/search/query_classifier.py` | Reordered intent logic (category before vague), added `extract_brand()` |
| `src/search/hybrid_search.py` | HTML unescape, brand filter injection, enrichment step, strict post-filter, facets, dynamic fetch size |
| `src/search/algolia_client.py` | Added `get_objects()` batch method, `facets` param on `search()` |
| `src/search/reranker.py` | Added `_deduplicate()`, sister brands, size-variant normalization, brand cap 10->4 |
| `src/search/models.py` | Added `FacetValue` model, `facets` field on `HybridSearchResponse` |
| `tests/unit/test_search.py` | Updated `test_fit_type_none_passes` -> `test_fit_type_none_excluded` |
| `scripts/test_search_gradio.py` | Enhanced UI: all 23 filters, compare tab, real brands, Ba&sh quick test |

---

## Testing

**144 unit tests passing** (all search module tests green).

```bash
# Run all unit tests
PYTHONPATH=src python -m pytest tests/unit/ -v

# Run search tests only
PYTHONPATH=src python -m pytest tests/unit/test_search.py -v

# Run integration tests (requires running server)
TEST_SERVER_URL=http://localhost:8000 PYTHONPATH=src python -m pytest tests/integration/ -v
```

---

## Known Gaps / Next Steps: Reranker Overhaul

The reranker (`src/search/reranker.py`) needs significant enhancements:

1. **Hard filters from profile** - Avoided brands are only soft penalties; they still appear in results
2. **No price-range personalization** - User's price range preferences aren't used
3. **No trending/recency boost** - New arrivals and trending items aren't prioritized
4. **No "similar to recently viewed" signal** - Can't boost items similar to what user recently viewed
5. **No category diversity** - Could get 20 dresses and no tops in results
6. **No position-based diversity** - No interleaving of categories/brands across positions
7. **Hand-tuned weights** - Boost weights are guesses, not data-driven
8. **No A/B testing framework** - Can't test weight variations
9. **No query context** - Reranker doesn't receive the search query, can't boost query-relevant items
10. **Profile schema alignment** - Need to verify `user_profile` structure matches what Supabase stores

---

## Environment

```
Python 3.12, venv at .venv/
numpy<2.0 (recbole compatibility)
algoliasearch==4.36.0 (v4 client)
gradio==6.5.1
transformers==5.1.0 (requires pooler_output fix)
```
