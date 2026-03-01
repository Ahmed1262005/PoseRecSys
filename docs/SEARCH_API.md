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
| `/api/search/hybrid` | POST | Yes | Hybrid search with LLM query planning and follow-up questions |
| `/api/search/refine` | POST | Yes | Re-search with follow-up selections (skips LLM planner) |
| `/api/search/autocomplete` | GET | Yes | Product + brand autocomplete |
| `/api/search/click` | POST | Yes | Track click event |
| `/api/search/conversion` | POST | Yes | Track conversion event |
| `/api/search/health` | GET | No | Search module health check |

---

## POST `/api/search/hybrid`

Main search endpoint. Runs the full pipeline: LLM query planner → Algolia + FashionCLIP semantic search → RRF merge → reranker → follow-up questions.

### Request Body

```jsonc
{
  // Required
  "query": "something cute for a night out",

  // Pagination
  "page": 1,                    // 1-indexed, default 1
  "page_size": 50,              // 1-100, default 50

  // Sort order
  "sort_by": "relevance",       // "relevance" | "price_asc" | "price_desc" | "trending"

  // Session (for dedup across pages)
  "session_id": "sess_abc123",  // optional, server generates one if omitted

  // Semantic weight (advanced)
  "semantic_boost": 0.4,        // 0.0-1.0, weight of semantic results in RRF merge

  // ---- Include filters (all optional, null = no filter) ----
  "categories": ["dresses"],              // broad categories
  "category_l1": ["Dresses"],             // Gemini L1 (Tops, Bottoms, Dresses, Outerwear, Activewear, Swimwear, Accessories)
  "category_l2": ["Midi Dress"],          // Gemini L2 (Blouse, Jeans, Midi Dress, ...)
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
  "exclude_style_tags": ["Sporty"]
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
        {
          "label": "Casual",
          "filters": {"formality": ["Casual"]}
        },
        {
          "label": "Smart casual",
          "filters": {"formality": ["Smart Casual"]}
        },
        {
          "label": "Dressy",
          "filters": {"formality": ["Semi-Formal"]}
        },
        {
          "label": "Formal",
          "filters": {"formality": ["Formal"]}
        }
      ]
    },
    {
      "dimension": "vibe",
      "question": "What vibe are you going for?",
      "options": [
        {
          "label": "Glamorous",
          "filters": {"style_tags": ["Glamorous"]}
        },
        {
          "label": "Edgy",
          "filters": {"style_tags": ["Edgy"]}
        },
        {
          "label": "Romantic",
          "filters": {"style_tags": ["Romantic"]}
        },
        {
          "label": "Minimal",
          "filters": {"style_tags": ["Minimalist"]}
        }
      ]
    }
  ],

  "timing": {
    "planner_ms": 3200,
    "algolia_ms": 180,
    "semantic_ms": 520,
    "semantic_query_count": 3,
    "total_ms": 4100
  }
}
```

### Follow-up Questions

Follow-ups are generated by the LLM planner when the query is vague or under-specified. They are **not** generated when:

- The query is already specific (e.g. "red midi dress")
- `sort_by` is not `"relevance"` (sorted modes skip the full pipeline)
- The planner is skipped (e.g. via the `/refine` endpoint)

**Dimensions** (7 possible):

| Dimension | What it refines | Filter keys used |
|-----------|----------------|------------------|
| `garment_type` | What category of clothing | `category_l1` |
| `occasion` | When/where they'll wear it | `occasions` |
| `formality` | How dressed up | `formality` |
| `vibe` | Style aesthetic | `style_tags` |
| `coverage` | Modesty/skin coverage | `modes` (coverage-only: `cover_arms`, `cover_chest`, etc.) |
| `price` | Budget | `min_price`, `max_price` |
| `color` | Color preference | `colors` |

**Selection model: multi-select.** Each follow-up question allows the user to pick **one or more** options. For example, the user can select both "Casual" and "Smart Casual" from a formality question, or both "Dresses" and "Tops" from a garment type question. Selected options are combined using the merge rules described under the `/refine` endpoint below.

Each option's `filters` object is a **PATCH** — a partial filter update that the client merges into `selected_filters` for the `/refine` endpoint.

---

## POST `/api/search/refine`

Apply follow-up question selections to re-run the search **without calling the LLM planner again**. This is faster than `/hybrid` (~200-500ms savings by skipping the LLM call).

### Request Body

```jsonc
{
  "original_query": "something cute for a night out",

  // Merged filters from all selected follow-up options
  "selected_filters": {
    "formality": ["Casual", "Smart Casual"],
    "style_tags": ["Glamorous"],
    "category_l1": ["Dresses"]
  },

  // Optional (same as /hybrid)
  "page": 1,
  "page_size": 50,
  "session_id": "sess_abc123",
  "sort_by": "relevance",
  "semantic_boost": 0.4
}
```

### How to build `selected_filters`

Each follow-up question supports **multi-select**. The client tracks which options the user has toggled on per question, then merges everything into a single flat `selected_filters` dict.

#### Step 1: Track selections per question

Maintain a map of `questionIndex -> selectedOptions[]`:

```javascript
// Internal state (not sent to server)
const selections = {
  0: [  // garment_type question
    { label: "Dresses", filters: { category_l1: ["Dresses"] } },
    { label: "Tops",    filters: { category_l1: ["Tops"] } },
  ],
  1: [  // formality question
    { label: "Casual",       filters: { formality: ["Casual"] } },
    { label: "Smart Casual", filters: { formality: ["Smart Casual"] } },
  ],
  2: [  // vibe question (single selection)
    { label: "Glamorous", filters: { style_tags: ["Glamorous"] } },
  ],
};
```

Clicking a pill toggles it on/off. Multiple pills per question can be active simultaneously.

#### Step 2: Merge within each question (union)

For each question, combine the `filters` from all selected options:

| Filter type | Merge rule | Example |
|-------------|-----------|---------|
| List values (`formality`, `colors`, `category_l1`, `style_tags`, `occasions`, etc.) | **Union** (deduplicated) | `["Casual"] + ["Smart Casual"]` → `["Casual", "Smart Casual"]` |
| `modes` list | **Union** (additive coverage) | `["cover_arms"] + ["cover_chest"]` → `["cover_arms", "cover_chest"]` |
| `min_price` | **Min** of all values (widest floor) | `min(30, 60)` → `30` |
| `max_price` | **Max** of all values (widest ceiling) | `max(30, 60)` → `60` |

```javascript
// Question 0 merged: { category_l1: ["Dresses", "Tops"] }
// Question 1 merged: { formality: ["Casual", "Smart Casual"] }
// Question 2 merged: { style_tags: ["Glamorous"] }
```

#### Step 3: Merge across questions (overwrite by key)

Spread all per-question merged dicts into one. Since each question targets a different dimension, keys rarely collide. If they do, the later question's value wins.

```javascript
const selected_filters = {
  category_l1: ["Dresses", "Tops"],          // from question 0
  formality: ["Casual", "Smart Casual"],     // from question 1
  style_tags: ["Glamorous"],                 // from question 2
};
```

#### Step 4: Send to `/refine`

```javascript
POST /api/search/refine
{
  "original_query": "something cute for a night out",
  "selected_filters": {
    "category_l1": ["Dresses", "Tops"],
    "formality": ["Casual", "Smart Casual"],
    "style_tags": ["Glamorous"]
  },
  "session_id": "sess_abc123"
}
```

#### Reference: merge function (JavaScript)

```javascript
function mergeSelections(selections) {
  const merged = {};

  for (const questionIdx of Object.keys(selections).sort()) {
    const selectedOptions = selections[questionIdx];
    if (!selectedOptions.length) continue;

    // Merge all options within this question
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

    // Spread into overall merged (across questions)
    Object.assign(merged, questionMerged);
  }

  return merged;
}
```

#### Special handling: `modes`

If `selected_filters` contains a `"modes"` key, the `/refine` endpoint expands those modes server-side into concrete include/exclude filters. For example:

- `{"modes": ["cover_arms"]}` → adds `exclude_sleeve_type: ["Sleeveless", "Spaghetti Strap"]`
- `{"modes": ["cover_arms", "cover_chest"]}` → adds exclusions for both (additive)

The client does not need to expand modes — just pass the mode names through.

### Response

Same shape as `/api/search/hybrid` (returns `HybridSearchResponse`), but `follow_ups` will be `null` since the planner is skipped.

---

## Client Integration Flow

### Basic search (no follow-ups)

```
Client                          Server
  |                               |
  |  POST /api/search/hybrid      |
  |  {"query": "red midi dress"}  |
  |------------------------------>|
  |                               |
  |  200 OK                       |
  |  { results: [...],            |
  |    follow_ups: [] }           |  <-- specific query, no follow-ups
  |<------------------------------|
  |                               |
  |  Render product grid          |
```

### Search with follow-ups (multi-select)

```
Client                                    Server
  |                                         |
  |  POST /api/search/hybrid                |
  |  {"query": "outfit for this weekend"}   |
  |---------------------------------------->|
  |                                         |
  |  200 OK                                 |
  |  { results: [...],                      |  <-- show results immediately
  |    follow_ups: [                        |
  |      {dimension: "garment_type",        |
  |       question: "What are you           |
  |         looking for?",                  |
  |       options: [                        |
  |         {label: "Dresses",              |
  |          filters: {category_l1:         |
  |            ["Dresses"]}},               |
  |         {label: "Tops",                 |
  |          filters: {category_l1:         |
  |            ["Tops"]}},                  |
  |         {label: "Outerwear",            |
  |          filters: {category_l1:         |
  |            ["Outerwear"]}}              |
  |       ]},                               |
  |      {dimension: "formality", ...},     |
  |      {dimension: "vibe", ...}           |
  |    ] }                                  |
  |<----------------------------------------|
  |                                         |
  |  Render follow-up pills (multi-select)  |
  |                                         |
  |  User taps: [Dresses] [Tops]            |
  |  User taps: [Casual] [Smart Casual]     |
  |  User taps: [Glamorous]                 |
  |  User taps "Apply" button               |
  |                                         |
  |  Client merges selections:              |
  |  selected_filters = {                   |
  |    category_l1: ["Dresses", "Tops"],    |
  |    formality: ["Casual",                |
  |                "Smart Casual"],          |
  |    style_tags: ["Glamorous"]            |
  |  }                                      |
  |                                         |
  |  POST /api/search/refine                |
  |  { original_query: "outfit for this     |
  |      weekend",                          |
  |    selected_filters: {...},             |
  |    session_id: "sess_abc123" }          |
  |---------------------------------------->|
  |                                         |
  |  200 OK                                 |
  |  { results: [...],                      |  <-- refined results
  |    follow_ups: null }                   |
  |<----------------------------------------|
  |                                         |
  |  Replace product grid with new results  |
```

### Key implementation notes

1. **Show results and follow-ups together.** The initial `/hybrid` response contains both `results` and `follow_ups`. Display the product grid immediately and show follow-up questions as pills/chips alongside or above the grid.

2. **Follow-ups are optional.** If `follow_ups` is `null` or `[]`, don't render any question UI.

3. **Users can select from multiple questions.** Each question is independent. Merge all selected `filters` into one `selected_filters` dict before calling `/refine`.

4. **Multiple selections per question (multi-select).** Each question allows the user to toggle multiple options on/off. Display selected pills with a distinct visual state (e.g. filled/primary color). Merge rules:
   - List values: **union** (deduplicated, order-preserved)
   - `min_price`: **min** of all selected values (widest floor)
   - `max_price`: **max** of all selected values (widest ceiling)
   - `modes`: **union** (additive coverage constraints)

5. **Toggle behavior.** Clicking a selected pill deselects it (removes from the list). Clicking an unselected pill adds it. This gives the user full control over combinations.

6. **Partial selection is fine.** The user doesn't have to answer all questions. Only include filters from questions they actually interacted with.

7. **Session continuity.** Pass the same `session_id` to both `/hybrid` and `/refine` so seen-item dedup carries across.

8. **Refine is idempotent.** Calling `/refine` multiple times with different selections is safe. Each call runs a fresh search with the provided filters.

9. **Server handles list filters as OR.** When `selected_filters` contains `formality: ["Casual", "Smart Casual"]`, the server returns products matching **either** value. The multi-select widens within that dimension while the overall follow-up narrows the search by adding constraints.

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
  "index_records": 94000      // only when algolia is healthy
}
```

---

## Sort Modes

| Value | Pipeline | Speed | Follow-ups |
|-------|----------|-------|------------|
| `relevance` | Full hybrid: LLM planner → Algolia + FashionCLIP → RRF → reranker | ~3-15s | Yes |
| `price_asc` | LLM planner → Algolia virtual replica (cheapest first) | ~1-3s | No |
| `price_desc` | LLM planner → Algolia virtual replica (most expensive first) | ~1-3s | No |
| `trending` | LLM planner → Algolia virtual replica (trending score) | ~1-3s | No |

When `sort_by != "relevance"`:
- Semantic search, RRF merge, and reranker are skipped
- Algolia handles pagination natively
- All filters still apply
- `follow_ups` will be `null`

---

## Filter Reference

### Include Filters (21)

| Field | Type | Example Values |
|-------|------|---------------|
| `categories` | `string[]` | `tops`, `bottoms`, `dresses`, `outerwear` |
| `category_l1` | `string[]` | `Tops`, `Bottoms`, `Dresses`, `Outerwear`, `Activewear`, `Swimwear`, `Accessories` |
| `category_l2` | `string[]` | `Blouse`, `Jeans`, `Midi Dress`, `Bomber Jacket` |
| `brands` | `string[]` | `Boohoo`, `Princess Polly`, `Forever 21` |
| `colors` | `string[]` | `Black`, `White`, `Red`, `Blue`, `Navy Blue`, `Green`, `Pink`, `Yellow`, `Purple`, `Orange`, `Brown`, `Beige`, `Cream`, `Gray`, `Burgundy`, `Olive`, `Taupe`, `Off White`, `Light Blue` |
| `color_family` | `string[]` | `Warm`, `Cool`, `Neutral` |
| `patterns` | `string[]` | `Solid`, `Floral`, `Striped`, `Plaid`, `Polka Dot`, `Animal Print`, `Abstract`, `Geometric`, `Tie Dye`, `Camo`, `Colorblock`, `Tropical` |
| `materials` | `string[]` | `Cotton`, `Linen`, `Silk`, `Satin`, `Denim`, `Faux Leather`, `Wool`, `Velvet`, `Chiffon`, `Lace`, `Mesh`, `Knit`, `Jersey`, `Fleece`, `Sheer` |
| `occasions` | `string[]` | `Date Night`, `Party`, `Office`, `Work`, `Wedding Guest`, `Vacation`, `Workout`, `Everyday`, `Brunch`, `Night Out`, `Weekend`, `Lounging`, `Beach` |
| `seasons` | `string[]` | `Summer`, `Spring`, `Fall`, `Winter` |
| `formality` | `string[]` | `Formal`, `Semi-Formal`, `Business Casual`, `Smart Casual`, `Casual` |
| `fit_type` | `string[]` | `Slim`, `Fitted`, `Regular`, `Relaxed`, `Oversized`, `Loose` |
| `neckline` | `string[]` | `V-Neck`, `Crew`, `Turtleneck`, `Off-Shoulder`, `Strapless`, `Halter`, `Scoop`, `Square`, `Sweetheart`, `Cowl`, `Boat`, `One Shoulder`, `Collared`, `Hooded`, `Mock`, `Deep V-Neck`, `Plunging` |
| `sleeve_type` | `string[]` | `Sleeveless`, `Short`, `Long`, `Cap`, `Puff`, `3/4`, `Flutter`, `Spaghetti Strap` |
| `length` | `string[]` | `Mini`, `Midi`, `Maxi`, `Cropped`, `Floor-length`, `Ankle`, `Micro` |
| `rise` | `string[]` | `High`, `Mid`, `Low` |
| `silhouette` | `string[]` | `A-Line`, `Bodycon`, `Flared`, `Straight`, `Wide Leg` |
| `article_type` | `string[]` | `midi dress`, `jeans`, `t-shirt`, `blazer` |
| `style_tags` | `string[]` | `Bohemian`, `Romantic`, `Glamorous`, `Edgy`, `Vintage`, `Sporty`, `Classic`, `Modern`, `Minimalist`, `Preppy`, `Streetwear`, `Sexy`, `Western`, `Utility` |
| `min_price` | `float` | `20.0` |
| `max_price` | `float` | `100.0` |
| `on_sale_only` | `bool` | `true` |

### Exclude Filters (13)

All are `string[]` and use the same allowed values as their include counterparts:

`exclude_brands`, `exclude_colors`, `exclude_patterns`, `exclude_materials`, `exclude_occasions`, `exclude_seasons`, `exclude_formality`, `exclude_neckline`, `exclude_sleeve_type`, `exclude_length`, `exclude_fit_type`, `exclude_silhouette`, `exclude_rise`, `exclude_style_tags`

---

## Follow-up `filters` Keys

The `filters` object inside each follow-up option uses a **subset** of the main filter keys:

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

**Multi-select merge** column shows how multiple selected options within the same question are combined:
- **Union**: Deduplicated concatenation. `["Casual"] + ["Smart Casual"]` → `["Casual", "Smart Casual"]`. The server treats list values as OR (matches either).
- **Min/Max**: For price, multiple selections widen to the broadest range. `max_price: 30` + `max_price: 60` → `max_price: 60`.
- **Union (additive)**: For `modes`, multiple coverage modes are all applied. `["cover_arms", "cover_chest"]` excludes sleeveless AND low-neckline items.

The `modes` key is special — it is expanded server-side by `/refine` into concrete exclude filters (e.g. `cover_arms` → `exclude_sleeve_type: ["Sleeveless", "Spaghetti Strap"]`).
