# Search API Reference

Base URL: `https://<host>/api/search`

All endpoints (except `/health`) require a Supabase JWT in the `Authorization` header:

```
Authorization: Bearer <supabase_jwt_token>
```

---

## Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/search/hybrid` | POST | Yes | Hybrid search with LLM query planning, follow-up questions, and extend-search pagination |
| `/api/search/autocomplete` | GET | Yes | Product + brand autocomplete |
| `/api/search/click` | POST | Yes | Track click event |
| `/api/search/conversion` | POST | Yes | Track conversion event |
| `/api/search/health` | GET | No | Search module health check |

---

## POST `/api/search/hybrid`

Main search endpoint. Runs the full pipeline: LLM query planner → Algolia + FashionCLIP semantic search → RRF merge → reranker → post-sort (if applicable) → follow-up questions.

### Request Body

```jsonc
{
  // Required
  "query": "something cute for a night out",

  // Pagination
  "page": 1,                    // 1-indexed, default 1
  "page_size": 50,              // 1-100, default 50

  // Extend-search pagination (page 2+)
  // Pass both back from a previous response to get the next page
  // without re-running the LLM planner (~2-3s instead of ~12-15s)
  "search_session_id": "ss_abc123",  // from previous response
  "cursor": "eyJwIjoyfQ==",         // from previous response

  // Sort order
  "sort_by": "relevance",       // "relevance" | "price_asc" | "price_desc" | "trending"

  // Session (for dedup across pages)
  "session_id": "sess_abc123",  // optional, for session-level seen-item dedup

  // Semantic weight (advanced)
  "semantic_boost": 0.4,        // 0.0-1.0, weight of semantic results in RRF merge

  // ---- Include filters (all optional, null = no filter) ----
  "categories": ["dresses"],
  "category_l1": ["Dresses"],
  "category_l2": ["Midi Dress"],
  "brands": ["Boohoo"],
  "colors": ["Red", "Burgundy"],
  "color_family": ["Warm"],
  "patterns": ["Solid"],
  "materials": ["Silk", "Satin"],
  "occasions": ["Date Night"],
  "seasons": ["Summer"],
  "formality": ["Semi-Formal"],
  "fit_type": ["Fitted"],
  "neckline": ["V-Neck"],
  "sleeve_type": ["Long"],
  "length": ["Midi"],
  "rise": ["High"],
  "silhouette": ["A-Line"],
  "article_type": ["midi dress"],
  "style_tags": ["Romantic"],
  "min_price": 20.0,
  "max_price": 100.0,
  "on_sale_only": false,

  // ---- Exclude filters (all optional) ----
  "exclude_brands": ["Shein"],
  "exclude_colors": ["Yellow"],
  "exclude_patterns": ["Animal Print"],
  "exclude_materials": ["Polyester"],
  "exclude_occasions": ["Workout"],
  "exclude_seasons": ["Winter"],
  "exclude_formality": ["Formal"],
  "exclude_neckline": ["Strapless"],
  "exclude_sleeve_type": ["Sleeveless"],
  "exclude_length": ["Mini"],
  "exclude_fit_type": ["Oversized"],
  "exclude_silhouette": ["Bodycon"],
  "exclude_rise": ["Low"],
  "exclude_style_tags": ["Sporty"],

  // ---- Follow-up refinement (pass back from follow-up interactions) ----
  "selected_filters": {                    // accumulated follow-up selections
    "formality": ["Casual", "Smart Casual"],
    "style_tags": ["Glamorous"]
  },
  "selection_labels": ["Casual", "Smart Casual", "Glamorous"]  // human-readable labels
}
```

### Response

```jsonc
{
  "query": "something cute for a night out",
  "intent": "vague",                       // "exact" | "specific" | "vague"
  "sort_by": "relevance",

  "results": [
    {
      "product_id": "prod_123",
      "name": "Satin Cowl Neck Midi Dress",
      "brand": "Princess Polly",
      "image_url": "https://...",
      "gallery_images": ["https://...", "https://..."],
      "price": 68.0,
      "original_price": 89.0,
      "is_on_sale": true,
      "category_l1": "Dresses",
      "category_l2": "Midi Dress",
      "broad_category": "dresses",
      "article_type": "midi dress",
      "primary_color": "Red",
      "color_family": "Warm",
      "pattern": "Solid",
      "apparent_fabric": "Satin",
      "fit_type": "Fitted",
      "formality": "Semi-Formal",
      "silhouette": "A-Line",
      "length": "Midi",
      "neckline": "Cowl",
      "sleeve_type": "Spaghetti Strap",
      "rise": null,
      "style_tags": ["Romantic", "Glamorous"],
      "occasions": ["Date Night", "Party"],
      "seasons": ["Summer", "Spring"],
      "algolia_rank": 5,
      "semantic_rank": 2,
      "semantic_score": 0.847,
      "rrf_score": 0.031
    }
    // ... more results
  ],

  "pagination": {
    "page": 1,
    "page_size": 50,
    "has_more": true,
    "total_results": 237
  },

  // Extend-search pagination — pass both back for page 2+
  "search_session_id": "ss_abc123",
  "cursor": "eyJwIjoyfQ==",

  "facets": {
    "brand": [
      {"value": "Boohoo", "count": 42},
      {"value": "Princess Polly", "count": 28}
    ],
    "category_l1": [
      {"value": "Dresses", "count": 85},
      {"value": "Tops", "count": 64}
    ],
    "primary_color": [
      {"value": "Black", "count": 55},
      {"value": "Red", "count": 23}
    ]
    // ... more facet keys: color_family, pattern, fit_type, neckline,
    //     sleeve_type, length, silhouette, rise, occasions, seasons,
    //     style_tags, article_type, formality, is_on_sale, materials
  },

  // Follow-up questions (only present for vague/ambiguous queries)
  "follow_ups": [
    {
      "dimension": "formality",
      "question": "How dressed up do you want to be?",
      "options": [
        { "label": "Casual", "filters": {"formality": ["Casual"]} },
        { "label": "Smart casual", "filters": {"formality": ["Smart Casual"]} },
        { "label": "Dressy", "filters": {"formality": ["Semi-Formal"]} },
        { "label": "Formal", "filters": {"formality": ["Formal"]} }
      ]
    }
  ],

  // Refinement state (present after follow-up selections)
  "applied_filters": { "formality": ["Casual"] },
  "answered_dimensions": ["formality"],

  "timing": {
    "planner_ms": 3200,
    "algolia_ms": 180,
    "semantic_ms": 520,
    "semantic_query_count": 5,
    "total_ms": 4100
  }
}
```

---

## Sort Modes

| Value | Pipeline | Speed | Description |
|-------|----------|-------|-------------|
| `relevance` | Full hybrid: LLM planner → Algolia + FashionCLIP → RRF → reranker | ~12-15s (page 1), ~2-3s (page 2+) | Default. Best result quality and diversity. |
| `price_asc` | Full hybrid pipeline → **post-sort by price ascending** | ~12-15s (page 1), ~2-3s (page 2+) | Cheapest first among relevant results. |
| `price_desc` | Full hybrid pipeline → **post-sort by price descending** | ~12-15s (page 1), ~2-3s (page 2+) | Most expensive first among relevant results. |
| `trending` | Full hybrid pipeline → **post-sort by trending score** | ~12-15s (page 1), ~2-3s (page 2+) | Trending items first among relevant results. |

All sort modes run the full hybrid pipeline (Algolia + semantic + RRF merge + reranker), then sort the merged results before pagination. This ensures rich, diverse candidates regardless of sort order.

All filters work with all sort modes. Follow-up questions are generated for all sort modes.

---

## Extend-Search Pagination

Page 1 runs the full pipeline (LLM planner + Algolia + semantic + RRF + rerank). The plan state is cached server-side.

Page 2+ reuses the cached plan: skips the LLM planner, runs Algolia (native page=N) + semantic (reuses cached FashionCLIP embeddings, excludes already-seen product IDs) + RRF merge + rerank on fresh candidates.

### Client flow

```
Page 1:
  POST /api/search/hybrid
  {"query": "vacation outfits"}
  → Response includes search_session_id + cursor + has_more=true

Page 2:
  POST /api/search/hybrid
  {"query": "vacation outfits",
   "search_session_id": "ss_abc123",
   "cursor": "eyJwIjoyfQ=="}
  → Fresh candidates, zero overlap with page 1

Page 3:
  POST /api/search/hybrid
  {"query": "vacation outfits",
   "search_session_id": "ss_abc123",
   "cursor": "eyJwIjozfQ=="}
  → Fresh candidates, zero overlap with pages 1-2

End of results:
  → has_more=false, cursor=null → stop requesting
```

### Key rules

- Always pass back both `search_session_id` and `cursor` from the previous response.
- `query` must still be provided (validated) but the cached plan is used.
- Sessions expire after 10 minutes of inactivity.
- When `has_more=false` and `cursor=null`, there are no more results.

---

## Filter Reference

All filters are optional. When not provided (null), no filtering is applied for that dimension.

List filters (string arrays) use **OR** logic within the filter — e.g., `brands: ["Boohoo", "Princess Polly"]` returns products from either brand.

Multiple filters across different fields use **AND** logic — e.g., `brands: ["Boohoo"]` + `colors: ["Red"]` returns red Boohoo products.

### How Filters Apply in the Hybrid Pipeline

Filters are enforced at multiple stages:

| Stage | What happens |
|-------|-------------|
| **Algolia search** | Include filters → facet filters (AND). Exclude filters → NOT clauses. Price → numeric filter. On-sale → `is_on_sale:true`. |
| **Semantic post-filter** | After pgvector results are enriched with Algolia attributes, the same filters are enforced. Products with `null` attribute values are excluded by default (strict mode). |
| **RRF merge** | Both Algolia and semantic results have already been filtered. Merge produces a union. |
| **Reranker** | Session dedup, near-duplicate removal, brand diversity cap. Does not drop filtered results. |

### Include Filters (22)

| Field | Type | Algolia Facet | Example Values | Notes |
|-------|------|---------------|---------------|-------|
| `categories` | `string[]` | `broad_category` | `tops`, `bottoms`, `dresses`, `outerwear` | Legacy broad categories |
| `category_l1` | `string[]` | `category_l1` | `Tops`, `Bottoms`, `Dresses`, `Outerwear`, `Activewear`, `Swimwear`, `Accessories` | Gemini L1 categories (recommended) |
| `category_l2` | `string[]` | `category_l2` | `Blouse`, `Jeans`, `Midi Dress`, `Bomber Jacket` | Gemini L2 subcategories |
| `brands` | `string[]` | `brand` | `Boohoo`, `Princess Polly`, `Forever 21`, `A.P.C`, `Ba&sh` | Case-sensitive exact match |
| `colors` | `string[]` | `primary_color` | `Black`, `White`, `Red`, `Blue`, `Navy Blue`, `Green`, `Pink`, `Yellow`, `Purple`, `Orange`, `Brown`, `Beige`, `Cream`, `Gray`, `Burgundy`, `Olive`, `Taupe`, `Off White`, `Light Blue` | Primary product color |
| `color_family` | `string[]` | `color_family` | `Warm`, `Cool`, `Neutral` | Color temperature grouping |
| `patterns` | `string[]` | `pattern` | `Solid`, `Floral`, `Striped`, `Plaid`, `Polka Dot`, `Animal Print`, `Abstract`, `Geometric`, `Tie Dye`, `Camo`, `Colorblock`, `Tropical` | |
| `materials` | `string[]` | `apparent_fabric` | `Cotton`, `Linen`, `Silk`, `Satin`, `Denim`, `Faux Leather`, `Wool`, `Velvet`, `Chiffon`, `Lace`, `Mesh`, `Knit`, `Jersey`, `Fleece`, `Sheer` | Gemini-detected fabric |
| `occasions` | `string[]` | `occasions` | `Date Night`, `Party`, `Office`, `Work`, `Wedding Guest`, `Vacation`, `Workout`, `Everyday`, `Brunch`, `Night Out`, `Weekend`, `Lounging`, `Beach` | Multi-valued (product can have multiple) |
| `seasons` | `string[]` | `seasons` | `Summer`, `Spring`, `Fall`, `Winter` | Multi-valued |
| `formality` | `string[]` | `formality` | `Formal`, `Semi-Formal`, `Business Casual`, `Smart Casual`, `Casual`, `Loungewear` | |
| `fit_type` | `string[]` | `fit_type` | `Slim`, `Fitted`, `Regular`, `Relaxed`, `Oversized`, `Loose` | |
| `neckline` | `string[]` | `neckline` | `V-Neck`, `Crew`, `Turtleneck`, `Off-Shoulder`, `Strapless`, `Halter`, `Scoop`, `Square`, `Sweetheart`, `Cowl`, `Boat`, `One Shoulder`, `Collared`, `Hooded`, `Mock`, `Deep V-Neck`, `Plunging` | |
| `sleeve_type` | `string[]` | `sleeve_type` | `Sleeveless`, `Short`, `Long`, `Cap`, `Puff`, `3/4`, `Flutter`, `Spaghetti Strap` | |
| `length` | `string[]` | `length` | `Mini`, `Midi`, `Maxi`, `Cropped`, `Floor-length`, `Ankle`, `Micro` | |
| `rise` | `string[]` | `rise` | `High`, `Mid`, `Low` | Bottoms only |
| `silhouette` | `string[]` | `silhouette` | `A-Line`, `Bodycon`, `Flared`, `Straight`, `Wide Leg` | |
| `article_type` | `string[]` | `article_type` | `midi dress`, `jeans`, `t-shirt`, `blazer`, `bodysuit` | Specific garment type |
| `style_tags` | `string[]` | `style_tags` | `Bohemian`, `Romantic`, `Glamorous`, `Edgy`, `Vintage`, `Sporty`, `Classic`, `Modern`, `Minimalist`, `Preppy`, `Streetwear`, `Sexy`, `Western`, `Utility` | Multi-valued |
| `min_price` | `float` | numeric | `20.0` | Minimum price (inclusive). Must be >= 0. |
| `max_price` | `float` | numeric | `100.0` | Maximum price (inclusive). Must be >= min_price. |
| `on_sale_only` | `bool` | `is_on_sale` | `true` | Only return products where original_price > price. Default: `false`. |

### Exclude Filters (13)

Exclude filters remove products that match ANY of the specified values. They use `NOT` clauses in Algolia and strict exclusion in post-filtering.

| Field | Type | Excludes from | Example |
|-------|------|--------------|---------|
| `exclude_brands` | `string[]` | `brand` | `["Shein", "Temu"]` |
| `exclude_colors` | `string[]` | `primary_color` | `["Yellow", "Orange"]` |
| `exclude_patterns` | `string[]` | `pattern` | `["Animal Print", "Camo"]` |
| `exclude_materials` | `string[]` | `apparent_fabric` | `["Polyester", "Sheer"]` |
| `exclude_occasions` | `string[]` | `occasions` | `["Workout", "Lounging"]` |
| `exclude_seasons` | `string[]` | `seasons` | `["Winter"]` |
| `exclude_formality` | `string[]` | `formality` | `["Formal", "Loungewear"]` |
| `exclude_neckline` | `string[]` | `neckline` | `["Strapless", "Plunging"]` |
| `exclude_sleeve_type` | `string[]` | `sleeve_type` | `["Sleeveless", "Spaghetti Strap"]` |
| `exclude_length` | `string[]` | `length` | `["Mini", "Micro"]` |
| `exclude_fit_type` | `string[]` | `fit_type` | `["Oversized", "Loose"]` |
| `exclude_silhouette` | `string[]` | `silhouette` | `["Bodycon"]` |
| `exclude_rise` | `string[]` | `rise` | `["Low"]` |
| `exclude_style_tags` | `string[]` | `style_tags` | `["Sporty", "Utility"]` |

---

## Filter Examples

### Basic: Sale dresses under $50

```json
{
  "query": "dress",
  "category_l1": ["Dresses"],
  "max_price": 50,
  "on_sale_only": true
}
```

### Brand + color + price sort

```json
{
  "query": "tops",
  "brands": ["Boohoo", "Forever 21"],
  "colors": ["Black", "White"],
  "sort_by": "price_asc"
}
```

### Modest evening wear (using exclude filters)

```json
{
  "query": "evening outfit",
  "formality": ["Semi-Formal", "Formal"],
  "exclude_neckline": ["Strapless", "Plunging", "Deep V-Neck"],
  "exclude_sleeve_type": ["Sleeveless", "Spaghetti Strap"],
  "exclude_length": ["Mini", "Micro"]
}
```

### Vacation outfits sorted by price

```json
{
  "query": "vacation outfits for Europe",
  "sort_by": "price_desc",
  "page_size": 20
}
```

### Multi-filter with material and occasion

```json
{
  "query": "summer top",
  "materials": ["Cotton", "Linen"],
  "occasions": ["Vacation", "Weekend"],
  "seasons": ["Summer"],
  "max_price": 80
}
```

---

## Follow-up Questions

Follow-ups are generated by the LLM planner when the query is vague or under-specified. They help narrow down results interactively.

### When follow-ups appear

- Query is vague (e.g., "outfit for this weekend")
- Query is broad enough that filtering would help

### When follow-ups do NOT appear

- Query is already specific (e.g., "red midi dress")
- Query is an exact brand search (e.g., "Boohoo")

### Dimensions (7 possible)

| Dimension | What it refines | Filter keys used |
|-----------|----------------|------------------|
| `garment_type` | What category of clothing | `category_l1` |
| `occasion` | When/where they'll wear it | `occasions` |
| `formality` | How dressed up | `formality` |
| `vibe` | Style aesthetic | `style_tags` |
| `coverage` | Modesty/skin coverage | `modes` (expanded server-side) |
| `price` | Budget | `min_price`, `max_price` |
| `color` | Color preference | `colors` |

### Selection model: multi-select

Each follow-up question allows the user to pick **one or more** options. Selected options are merged and sent back via the `selected_filters` field on the next `/hybrid` request.

### How to apply follow-up selections

#### Step 1: Track selections per question

```javascript
const selections = {
  0: [  // garment_type question
    { label: "Dresses", filters: { category_l1: ["Dresses"] } },
    { label: "Tops",    filters: { category_l1: ["Tops"] } },
  ],
  1: [  // formality question
    { label: "Casual",       filters: { formality: ["Casual"] } },
    { label: "Smart Casual", filters: { formality: ["Smart Casual"] } },
  ],
};
```

#### Step 2: Merge selections

| Filter type | Merge rule | Example |
|-------------|-----------|---------|
| List values | **Union** (deduplicated) | `["Casual"] + ["Smart Casual"]` → `["Casual", "Smart Casual"]` |
| `modes` list | **Union** (additive coverage) | `["cover_arms"] + ["cover_chest"]` → `["cover_arms", "cover_chest"]` |
| `min_price` | **Min** of all values | `min(30, 60)` → `30` |
| `max_price` | **Max** of all values | `max(30, 60)` → `60` |

#### Step 3: Send back via `/hybrid`

```json
POST /api/search/hybrid
{
  "query": "outfit for this weekend",
  "selected_filters": {
    "category_l1": ["Dresses", "Tops"],
    "formality": ["Casual", "Smart Casual"]
  },
  "selection_labels": ["Dresses", "Tops", "Casual", "Smart Casual"]
}
```

The server runs the LLM planner in **refinement mode** — it regenerates semantic queries incorporating the selected filters and produces 1-2 new follow-up questions about unanswered dimensions.

#### Reference: merge function (JavaScript)

```javascript
function mergeSelections(selections) {
  const merged = {};

  for (const questionIdx of Object.keys(selections).sort()) {
    const selectedOptions = selections[questionIdx];
    if (!selectedOptions.length) continue;

    const questionMerged = {};
    for (const { filters } of selectedOptions) {
      for (const [key, value] of Object.entries(filters)) {
        if (key === "min_price") {
          const cur = questionMerged.min_price;
          questionMerged.min_price = cur != null ? Math.min(cur, value) : value;
        } else if (key === "max_price") {
          const cur = questionMerged.max_price;
          questionMerged.max_price = cur != null ? Math.max(cur, value) : value;
        } else if (Array.isArray(value)) {
          const existing = questionMerged[key] || [];
          const seen = new Set(existing);
          for (const v of value) {
            if (!seen.has(v)) { existing.push(v); seen.add(v); }
          }
          questionMerged[key] = existing;
        } else {
          questionMerged[key] = value;
        }
      }
    }

    Object.assign(merged, questionMerged);
  }

  return merged;
}
```

#### Special handling: `modes`

If `selected_filters` contains a `"modes"` key, the server expands those modes into concrete include/exclude filters:

- `{"modes": ["cover_arms"]}` → adds `exclude_sleeve_type: ["Sleeveless", "Spaghetti Strap"]`
- `{"modes": ["cover_arms", "cover_chest"]}` → adds exclusions for both (additive)

The client does not need to expand modes — just pass the mode names through.

---

## Follow-up `filters` Keys

The `filters` object inside each follow-up option uses a subset of the main filter keys:

| Key | Type | Used by dimension | Multi-select merge |
|-----|------|-------------------|--------------------|
| `category_l1` | `string[]` | `garment_type` | Union |
| `formality` | `string[]` | `formality` | Union |
| `occasions` | `string[]` | `occasion` | Union |
| `style_tags` | `string[]` | `vibe` | Union |
| `colors` | `string[]` | `color` | Union |
| `min_price` | `float` | `price` | Min (widest floor) |
| `max_price` | `float` | `price` | Max (widest ceiling) |
| `modes` | `string[]` | `coverage` | Union (additive) |
| `fit_type` | `string[]` | concrete attribute | Union |
| `sleeve_type` | `string[]` | concrete attribute | Union |
| `length` | `string[]` | concrete attribute | Union |
| `neckline` | `string[]` | concrete attribute | Union |
| `materials` | `string[]` | concrete attribute | Union |
| `patterns` | `string[]` | concrete attribute | Union |

---

## Client Integration Flow

### Basic search

```
Client                          Server
  |                               |
  |  POST /api/search/hybrid      |
  |  {"query": "red midi dress"}  |
  |------------------------------>|
  |                               |
  |  200 OK                       |
  |  { results: [...],            |
  |    follow_ups: [],            |
  |    search_session_id: "ss_x", |
  |    cursor: "eyJwIjoyfQ==" }   |
  |<------------------------------|
  |                               |
  |  Render product grid          |
  |                               |
  |  (User scrolls down)          |
  |                               |
  |  POST /api/search/hybrid      |
  |  {"query": "red midi dress",  |
  |   "search_session_id":"ss_x", |
  |   "cursor":"eyJwIjoyfQ=="}    |
  |------------------------------>|
  |                               |
  |  200 OK (fresh page 2 ~2-3s)  |
  |<------------------------------|
```

### Search with follow-ups + refinement

```
Client                                    Server
  |                                         |
  |  POST /api/search/hybrid                |
  |  {"query": "outfit for this weekend"}   |
  |---------------------------------------->|
  |                                         |
  |  200 OK                                 |
  |  { results: [...],                      |  ← show results immediately
  |    follow_ups: [                        |
  |      {question: "What are you           |
  |        looking for?", ...},             |
  |      {question: "How dressed up?", ...} |
  |    ] }                                  |
  |<----------------------------------------|
  |                                         |
  |  User taps: [Dresses] [Tops]            |
  |  User taps: [Casual]                    |
  |  User taps "Apply"                      |
  |                                         |
  |  POST /api/search/hybrid                |
  |  {"query": "outfit for this weekend",   |
  |   "selected_filters": {                 |
  |     "category_l1":["Dresses","Tops"],   |
  |     "formality": ["Casual"]             |
  |   },                                    |
  |   "selection_labels":                   |
  |     ["Dresses","Tops","Casual"]}        |
  |---------------------------------------->|
  |                                         |
  |  200 OK                                 |
  |  { results: [...],                      |  ← refined results
  |    follow_ups: [new questions...],      |  ← 1-2 new follow-ups
  |    applied_filters: {...},              |
  |    answered_dimensions: ["garment_type",|
  |      "formality"] }                     |
  |<----------------------------------------|
```

### Key implementation notes

1. **Show results and follow-ups together.** The initial response contains both. Display the product grid immediately with follow-up pills alongside or above.

2. **Follow-ups are optional.** If `follow_ups` is `null` or `[]`, don't render any question UI.

3. **Multi-select within each question.** Clicking a pill toggles it on/off. Multiple pills per question can be active simultaneously.

4. **Partial selection is fine.** The user doesn't need to answer all questions.

5. **Refinement generates new follow-ups.** After applying selections, the server returns 1-2 new questions about unanswered dimensions. The `answered_dimensions` field tracks what's been answered.

6. **Session continuity.** Pass the same `session_id` for seen-item dedup. Use `search_session_id` + `cursor` for extend-search pagination.

7. **Server handles list filters as OR.** `formality: ["Casual", "Smart Casual"]` returns products matching either value.

---

## GET `/api/search/autocomplete`

Returns product and brand suggestions as the user types.

### Query Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `q` | string | Yes | — | Search prefix (1-200 chars) |
| `limit` | int | No | 10 | Max suggestions (1-20) |

### Response

```jsonc
{
  "query": "flo",
  "products": [
    {
      "id": "prod_456",
      "name": "Floral Print Midi Dress",
      "brand": "Boohoo",
      "image_url": "https://...",
      "price": 42.0,
      "highlighted_name": "<em>Flo</em>ral Print Midi Dress"
    }
  ],
  "brands": [
    {
      "name": "Forever 21",
      "highlighted": "<em>Fo</em>rever 21"
    }
  ]
}
```

---

## POST `/api/search/click`

Track when a user clicks a product from search results.

### Request Body

```json
{
  "query": "red dress",
  "product_id": "prod_123",
  "position": 3
}
```

### Response

```json
{"status": "ok"}
```

Status code: `201 Created`

---

## POST `/api/search/conversion`

Track when a user converts (add to cart / purchase) from search.

### Request Body

```json
{
  "query": "red dress",
  "product_id": "prod_123"
}
```

### Response

```json
{"status": "ok"}
```

Status code: `201 Created`

---

## GET `/api/search/health`

No authentication required.

### Response

```jsonc
{
  "service": "search",
  "status": "healthy",        // "healthy" | "degraded" | "unhealthy"
  "algolia": "healthy",       // "healthy" | "unhealthy"
  "semantic": "healthy",      // "healthy" | "degraded" | "unhealthy"
  "index_records": 132711     // only when algolia is healthy
}
```
