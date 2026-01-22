# Endless Scroll V2 - Production Architecture

## Current State (V1) - NOT Production Ready

The current implementation uses:
- In-memory session state (lost on restart)
- SQL-level exclusion with arrays (O(n) per row)
- No horizontal scaling support

## Production Architecture (V2)

### 1. Redis Session State (Required)

```bash
# Deploy Redis
pip install redis
export REDIS_URL="redis://your-redis-host:6379/0"
```

Redis provides:
- Persistence across server restarts
- Shared state for horizontal scaling
- TTL-based automatic cleanup
- O(1) lookups for seen items (using Redis SETs) 

### 2. Cursor-Based Pagination (Replace Offset)

Instead of tracking `exclude_ids`, use a **cursor** based on the last item's score:

```python
# Current (O(n) exclusion):
WHERE p.id != ALL(exclude_product_ids)  # Slow with 1000+ IDs

# Better (O(1) cursor):
WHERE (similarity, p.id) < (last_similarity, last_item_id)
ORDER BY similarity DESC, p.id DESC
LIMIT 50
```

**Benefits:**
- Constant time regardless of page number
- No growing exclude arrays
- Works with any ordering (similarity, trending_score, etc.)

### 3. Hybrid Approach for True Endless Scroll

```
┌─────────────────────────────────────────────────────────────────┐
│                    PAGE REQUEST                                  │
│  {session_id, cursor, page_size}                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REDIS LOOKUP                                  │
│  1. Get cursor position from session                            │
│  2. Get bloom filter of seen items (for dedup)                  │
│  3. Get user's taste vector (cached)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CANDIDATE RETRIEVAL                           │
│                                                                  │
│  Strategy 1: Cursor-based (pages 0-20)                          │
│    SELECT * FROM products                                        │
│    WHERE (score, id) < (cursor_score, cursor_id)                │
│    ORDER BY score DESC, id DESC                                  │
│    LIMIT 100                                                     │
│                                                                  │
│  Strategy 2: Offset-based (pages 20+)                           │
│    Use SQL OFFSET for deep pagination                           │
│    (OK because items are pre-ranked)                            │
│                                                                  │
│  Strategy 3: Exploration (every 5th page)                       │
│    Random sampling for diversity                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BLOOM FILTER DEDUP                            │
│  Check each candidate against bloom filter                       │
│  - O(1) per item                                                 │
│  - 1% false positive rate acceptable                            │
│  - Memory: ~1.2KB per 1000 items                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    UPDATE SESSION                                │
│  1. Update cursor position                                       │
│  2. Add items to bloom filter                                    │
│  3. Set TTL (24 hours)                                          │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Bloom Filter for Seen Items

Replace the growing `exclude_ids` array with a bloom filter:

```python
from pybloom_live import BloomFilter

class SessionState:
    def __init__(self):
        # 10K capacity, 1% false positive rate
        self.seen_filter = BloomFilter(capacity=10000, error_rate=0.01)

    def add_seen(self, item_id: str):
        self.seen_filter.add(item_id)

    def might_have_seen(self, item_id: str) -> bool:
        return item_id in self.seen_filter  # O(1)
```

**Memory comparison:**
- Current (10K UUIDs): ~360KB per session
- Bloom filter (10K items): ~1.2KB per session

### 5. Pre-Computed Ranking Tables

For truly endless scroll, pre-compute rankings:

```sql
-- Materialized view updated hourly
CREATE MATERIALIZED VIEW ranked_products AS
SELECT
    p.id,
    p.*,
    ROW_NUMBER() OVER (
        PARTITION BY p.broad_category
        ORDER BY p.trending_score DESC
    ) as trending_rank,
    ROW_NUMBER() OVER (
        PARTITION BY p.broad_category
        ORDER BY p.created_at DESC
    ) as recency_rank
FROM products p
WHERE p.in_stock = true;

-- Index for fast cursor queries
CREATE INDEX idx_ranked_trending ON ranked_products (broad_category, trending_rank);
```

Then paginate by rank:

```sql
SELECT * FROM ranked_products
WHERE broad_category = 'bottoms'
  AND trending_rank > :cursor_rank
ORDER BY trending_rank
LIMIT 50;
```

### 6. Database Schema Changes

```sql
-- Session state table (if not using Redis)
CREATE TABLE user_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT,
    seen_bloom BYTEA,  -- Serialized bloom filter
    cursor_position JSONB,  -- {similarity: 0.95, item_id: 'xxx'}
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_access TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '24 hours'
);

-- Index for cleanup
CREATE INDEX idx_sessions_expires ON user_sessions (expires_at);

-- Cleanup job
DELETE FROM user_sessions WHERE expires_at < NOW();
```

### 7. Implementation Phases

#### Phase 1: Redis Integration (1 week)
- [ ] Deploy Redis instance
- [ ] Configure REDIS_URL environment variable
- [ ] Test session persistence across restarts
- [ ] Add Redis health check to /health endpoint

#### Phase 2: Cursor-Based Pagination (1 week)
- [ ] Add cursor fields to API response
- [ ] Modify SQL functions to use cursor WHERE clause
- [ ] Update frontend to pass cursor instead of page number
- [ ] Benchmark performance improvement

#### Phase 3: Bloom Filter (3 days)
- [ ] Install pybloom-live
- [ ] Replace Set[str] with BloomFilter in SessionState
- [ ] Serialize/deserialize bloom filter to Redis
- [ ] Monitor false positive rate

#### Phase 4: Pre-Computed Rankings (1 week)
- [ ] Create materialized views for common orderings
- [ ] Set up hourly refresh job
- [ ] Add fallback to real-time query if view stale
- [ ] Monitor view freshness

### 8. Scaling Considerations

| Users | Sessions | Current Memory | With Bloom Filter |
|-------|----------|----------------|-------------------|
| 1K | 1K | 360MB | 1.2MB |
| 10K | 10K | 3.6GB | 12MB |
| 100K | 100K | 36GB ❌ | 120MB ✅ |

### 9. Monitoring Metrics

```python
# Add to health endpoint
{
    "endless_scroll": {
        "active_sessions": 1234,
        "avg_seen_per_session": 150,
        "redis_connected": true,
        "bloom_filter_size_mb": 0.12,
        "cursor_cache_hit_rate": 0.95
    }
}
```

## Migration Path

1. **Week 1**: Deploy Redis, flip `SESSION_BACKEND=redis`
2. **Week 2**: Implement cursor-based pagination
3. **Week 3**: Add bloom filter deduplication
4. **Week 4**: Pre-computed ranking tables
5. **Week 5**: Load testing at 10K concurrent users

## Summary

| Aspect | V1 (Current) | V2 (Production) |
|--------|--------------|-----------------|
| Session storage | In-memory | Redis |
| Deduplication | O(n) array | O(1) bloom filter |
| Pagination | Offset + exclude | Cursor-based |
| Memory per session | ~36KB | ~1.2KB |
| Horizontal scaling | ❌ | ✅ |
| Survives restart | ❌ | ✅ |
| Max pages | ~20 (quality degrades) | Unlimited |
