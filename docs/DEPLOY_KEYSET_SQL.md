# Deploying Keyset Pagination SQL Functions to Supabase

## Status: Ready for Deployment

The keyset pagination functions are ready to deploy. After deployment, the system will automatically use O(1) keyset pagination instead of the fallback offset-based pagination.

## SQL File Location

```
sql/008_keyset_pagination.sql
```

## How to Deploy

### Option 1: Supabase Dashboard (Recommended)

1. Go to https://supabase.com/dashboard
2. Select project: `taqfbdeoesobozibcwmj`
3. Navigate to **SQL Editor**
4. Click **New Query**
5. Copy contents of `sql/008_keyset_pagination.sql`
6. Click **Run**
7. Verify: Check that 4 functions are created:
   - `match_products_keyset`
   - `get_trending_keyset`
   - `get_exploration_keyset`
   - `count_available_products_keyset`

### Option 2: Supabase CLI

```bash
# Install CLI if needed
npm install -g supabase

# Login
supabase login

# Link to project
supabase link --project-ref taqfbdeoesobozibcwmj

# Push schema
supabase db push
```

### Option 3: psql Direct Connection

```bash
# Get connection string from Supabase Dashboard > Settings > Database
psql "postgresql://postgres:[PASSWORD]@db.taqfbdeoesobozibcwmj.supabase.co:5432/postgres" \
  -f sql/008_keyset_pagination.sql
```

## Verification After Deployment

### 1. Test function exists

```sql
SELECT routine_name
FROM information_schema.routines
WHERE routine_schema = 'public'
AND routine_name LIKE '%keyset%';
```

Expected: 4 rows

### 2. Test API endpoint

```bash
curl "http://localhost:8080/api/recs/v2/feed/keyset?anon_id=test&page_size=10"
```

Check response contains:
- `"keyset_pagination": true` in metadata
- Proper cursor encoding

### 3. Run performance test

```bash
pytest tests/test_endless_scroll_v2.py::TestPerformance::test_constant_time_pagination -v
```

This test measures:
- Page 1 latency vs Page 50 latency
- Should be within 2x of each other (O(1) property)

## Functions Overview

| Function | Purpose | Pagination Key |
|----------|---------|----------------|
| `match_products_keyset` | pgvector similarity search | (similarity, id) |
| `get_trending_keyset` | Trending products | (trending_score, id) |
| `get_exploration_keyset` | Exploration items | (hash_score, id) |
| `count_available_products_keyset` | Total count (no cursor) | N/A |

## Rollback

If issues occur, the system will automatically fall back to the V1 endless scroll functions. No explicit rollback needed.

To explicitly disable keyset functions:

```sql
DROP FUNCTION IF EXISTS match_products_keyset;
DROP FUNCTION IF EXISTS get_trending_keyset;
DROP FUNCTION IF EXISTS get_exploration_keyset;
DROP FUNCTION IF EXISTS count_available_products_keyset;
```

## Current Test Results

```
19/20 tests PASSED
1/20 tests SKIPPED (test_constant_time_pagination - waiting for SQL deployment)

Correctness Guarantees:
- No duplicates within session: VERIFIED
- Stable ordering: VERIFIED
- Graceful degradation: VERIFIED
- Session isolation: VERIFIED
```
