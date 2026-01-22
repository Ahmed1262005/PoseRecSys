-- =====================================================
-- Keyset Pagination Functions for Endless Scroll V2
-- Run this after 007_endless_scroll.sql
--
-- V2 Improvements:
-- 1. Use keyset cursor (score, id) instead of OFFSET
-- 2. O(1) pagination regardless of page depth
-- 3. No growing exclude_ids arrays
-- 4. Stable ordering with tie-breaker on id
--
-- Key Insight:
-- V1: WHERE p.id != ALL(exclude_product_ids)  -- O(n) per row
-- V2: WHERE (score, p.id) < (cursor_score, cursor_id)  -- O(1) with index
-- =====================================================

-- =====================================================
-- STEP 1: match_products_keyset
-- pgvector search with keyset cursor for O(1) pagination
-- =====================================================
CREATE OR REPLACE FUNCTION match_products_keyset(
    query_embedding vector(512),
    filter_gender text DEFAULT NULL,
    filter_categories text[] DEFAULT NULL,
    exclude_colors text[] DEFAULT NULL,
    exclude_materials text[] DEFAULT NULL,
    exclude_brands text[] DEFAULT NULL,
    min_price numeric DEFAULT NULL,
    max_price numeric DEFAULT NULL,
    -- Keyset cursor parameters (NULL for first page)
    cursor_score float DEFAULT NULL,
    cursor_id uuid DEFAULT NULL,
    p_limit int DEFAULT 50
)
RETURNS TABLE(
    product_id uuid,
    name text,
    brand text,
    category text,
    broad_category text,
    colors text[],
    materials text[],
    price numeric,
    fit text,
    length text,
    sleeve text,
    neckline text,
    style_tags text[],
    primary_image_url text,
    hero_image_url text,
    similarity float
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id as product_id,
        p.name,
        p.brand,
        p.category,
        p.broad_category,
        p.colors,
        p.materials,
        p.price,
        p.fit,
        p.length,
        p.sleeve,
        p.neckline,
        p.style_tags,
        p.primary_image_url,
        p.hero_image_url,
        1 - (ie.embedding <=> query_embedding) as similarity
    FROM products p
    JOIN image_embeddings ie ON ie.sku_id = p.id
    WHERE p.in_stock = true
      -- Gender filter
      AND (filter_gender IS NULL OR filter_gender = ANY(p.gender))
      -- Category filter (broad_category or category)
      AND (filter_categories IS NULL
           OR p.broad_category = ANY(filter_categories)
           OR p.category = ANY(filter_categories))
      -- Price range
      AND (min_price IS NULL OR p.price >= min_price)
      AND (max_price IS NULL OR p.price <= max_price)
      -- Exclude colors (array overlap check)
      AND (exclude_colors IS NULL OR NOT (p.colors && exclude_colors))
      -- Exclude materials
      AND (exclude_materials IS NULL OR NOT (p.materials && exclude_materials))
      -- Exclude brands
      AND (exclude_brands IS NULL OR p.brand != ALL(exclude_brands))
      -- Ensure embedding exists
      AND ie.sku_id IS NOT NULL
      AND ie.embedding IS NOT NULL
      -- KEYSET CURSOR: O(1) pagination
      -- For first page (cursor_score IS NULL): return all
      -- For subsequent pages: return items with (score, id) < (cursor_score, cursor_id)
      AND (
          cursor_score IS NULL  -- First page
          OR (
              -- Composite comparison: (score DESC, id DESC)
              -- Items with lower score come after (we want descending order)
              (1 - (ie.embedding <=> query_embedding)) < cursor_score
              OR (
                  -- Tie-breaker: same score, use id for deterministic order
                  (1 - (ie.embedding <=> query_embedding)) = cursor_score
                  AND p.id < cursor_id
              )
          )
      )
    ORDER BY (1 - (ie.embedding <=> query_embedding)) DESC, p.id DESC
    LIMIT p_limit;
END;
$$;

-- =====================================================
-- STEP 2: get_trending_keyset
-- Trending products with keyset cursor
-- Uses trending_score as primary sort, id as tie-breaker
-- =====================================================
CREATE OR REPLACE FUNCTION get_trending_keyset(
    filter_gender text DEFAULT NULL,
    filter_categories text[] DEFAULT NULL,
    exclude_colors text[] DEFAULT NULL,
    exclude_materials text[] DEFAULT NULL,
    exclude_brands text[] DEFAULT NULL,
    min_price numeric DEFAULT NULL,
    max_price numeric DEFAULT NULL,
    -- Keyset cursor parameters
    cursor_score float DEFAULT NULL,
    cursor_id uuid DEFAULT NULL,
    p_limit int DEFAULT 50
)
RETURNS TABLE(
    product_id uuid,
    name text,
    brand text,
    category text,
    broad_category text,
    colors text[],
    materials text[],
    price numeric,
    fit text,
    length text,
    sleeve text,
    neckline text,
    style_tags text[],
    primary_image_url text,
    hero_image_url text,
    trending_score float,
    similarity float  -- Placeholder for compatibility
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id as product_id,
        p.name,
        p.brand,
        p.category,
        p.broad_category,
        p.colors,
        p.materials,
        p.price,
        p.fit,
        p.length,
        p.sleeve,
        p.neckline,
        p.style_tags,
        p.primary_image_url,
        p.hero_image_url,
        COALESCE(p.trending_score, 0.5)::float as trending_score,
        COALESCE(p.trending_score, 0.5)::float as similarity  -- Use trending as score
    FROM products p
    WHERE p.in_stock = true
      -- Gender filter
      AND (filter_gender IS NULL OR filter_gender = ANY(p.gender))
      -- Category filter
      AND (filter_categories IS NULL
           OR p.broad_category = ANY(filter_categories)
           OR p.category = ANY(filter_categories))
      -- Price range
      AND (min_price IS NULL OR p.price >= min_price)
      AND (max_price IS NULL OR p.price <= max_price)
      -- Exclude colors
      AND (exclude_colors IS NULL OR NOT (p.colors && exclude_colors))
      -- Exclude materials
      AND (exclude_materials IS NULL OR NOT (p.materials && exclude_materials))
      -- Exclude brands
      AND (exclude_brands IS NULL OR p.brand != ALL(exclude_brands))
      -- KEYSET CURSOR
      AND (
          cursor_score IS NULL  -- First page
          OR (
              COALESCE(p.trending_score, 0.5) < cursor_score
              OR (
                  COALESCE(p.trending_score, 0.5) = cursor_score
                  AND p.id < cursor_id
              )
          )
      )
    ORDER BY COALESCE(p.trending_score, 0.5) DESC, p.id DESC
    LIMIT p_limit;
END;
$$;

-- =====================================================
-- STEP 3: get_exploration_keyset
-- Exploration items with deterministic "random" ordering
-- Uses md5 hash of id for consistent but diverse results
-- =====================================================
CREATE OR REPLACE FUNCTION get_exploration_keyset(
    filter_gender text DEFAULT NULL,
    filter_categories text[] DEFAULT NULL,
    exclude_colors text[] DEFAULT NULL,
    exclude_materials text[] DEFAULT NULL,
    exclude_brands text[] DEFAULT NULL,
    min_price numeric DEFAULT NULL,
    max_price numeric DEFAULT NULL,
    -- Seed for deterministic randomness
    random_seed text DEFAULT NULL,
    -- Keyset cursor parameters (based on hash score)
    cursor_score float DEFAULT NULL,
    cursor_id uuid DEFAULT NULL,
    p_limit int DEFAULT 50
)
RETURNS TABLE(
    product_id uuid,
    name text,
    brand text,
    category text,
    broad_category text,
    colors text[],
    materials text[],
    price numeric,
    fit text,
    length text,
    sleeve text,
    neckline text,
    style_tags text[],
    primary_image_url text,
    hero_image_url text,
    exploration_score float,
    similarity float
)
LANGUAGE plpgsql AS $$
DECLARE
    effective_seed text;
BEGIN
    -- Use provided seed or generate one
    effective_seed := COALESCE(random_seed, md5(now()::text));

    RETURN QUERY
    SELECT
        p.id as product_id,
        p.name,
        p.brand,
        p.category,
        p.broad_category,
        p.colors,
        p.materials,
        p.price,
        p.fit,
        p.length,
        p.sleeve,
        p.neckline,
        p.style_tags,
        p.primary_image_url,
        p.hero_image_url,
        -- Deterministic "random" score based on id and seed
        (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float as exploration_score,
        (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float as similarity
    FROM products p
    WHERE p.in_stock = true
      -- Gender filter
      AND (filter_gender IS NULL OR filter_gender = ANY(p.gender))
      -- Category filter
      AND (filter_categories IS NULL
           OR p.broad_category = ANY(filter_categories)
           OR p.category = ANY(filter_categories))
      -- Price range
      AND (min_price IS NULL OR p.price >= min_price)
      AND (max_price IS NULL OR p.price <= max_price)
      -- Exclude colors
      AND (exclude_colors IS NULL OR NOT (p.colors && exclude_colors))
      -- Exclude materials
      AND (exclude_materials IS NULL OR NOT (p.materials && exclude_materials))
      -- Exclude brands
      AND (exclude_brands IS NULL OR p.brand != ALL(exclude_brands))
      -- KEYSET CURSOR (based on hash score)
      AND (
          cursor_score IS NULL
          OR (
              (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0) < cursor_score
              OR (
                  (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0) = cursor_score
                  AND p.id < cursor_id
              )
          )
      )
    ORDER BY (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0) DESC, p.id DESC
    LIMIT p_limit;
END;
$$;

-- =====================================================
-- STEP 4: count_available_products_keyset
-- Count products available for a given set of filters
-- (No cursor needed - just count total matching)
-- =====================================================
CREATE OR REPLACE FUNCTION count_available_products_keyset(
    filter_gender text DEFAULT NULL,
    filter_categories text[] DEFAULT NULL,
    exclude_colors text[] DEFAULT NULL,
    exclude_materials text[] DEFAULT NULL,
    exclude_brands text[] DEFAULT NULL,
    min_price numeric DEFAULT NULL,
    max_price numeric DEFAULT NULL
)
RETURNS int
LANGUAGE plpgsql AS $$
DECLARE
    product_count int;
BEGIN
    SELECT COUNT(*)
    INTO product_count
    FROM products p
    WHERE p.in_stock = true
      AND (filter_gender IS NULL OR filter_gender = ANY(p.gender))
      AND (filter_categories IS NULL
           OR p.broad_category = ANY(filter_categories)
           OR p.category = ANY(filter_categories))
      AND (min_price IS NULL OR p.price >= min_price)
      AND (max_price IS NULL OR p.price <= max_price)
      AND (exclude_colors IS NULL OR NOT (p.colors && exclude_colors))
      AND (exclude_materials IS NULL OR NOT (p.materials && exclude_materials))
      AND (exclude_brands IS NULL OR p.brand != ALL(exclude_brands));

    RETURN product_count;
END;
$$;

-- =====================================================
-- STEP 5: Create supporting indexes for keyset queries
-- These indexes make keyset pagination O(1)
-- =====================================================

-- Index for trending keyset pagination
-- DROP INDEX IF EXISTS idx_products_trending_keyset;
-- CREATE INDEX idx_products_trending_keyset
--     ON products (trending_score DESC NULLS LAST, id DESC)
--     WHERE in_stock = true;

-- Note: The pgvector index on embedding already supports
-- similarity search. The keyset cursor works with the
-- existing (embedding <=> query) ORDER BY.

-- =====================================================
-- Usage Examples:
-- =====================================================
--
-- First page (no cursor):
-- SELECT * FROM match_products_keyset(
--     query_embedding := '[0.1, 0.2, ...]'::vector(512),
--     filter_gender := 'female',
--     filter_categories := ARRAY['tops', 'dresses'],
--     cursor_score := NULL,
--     cursor_id := NULL,
--     p_limit := 50
-- );
--
-- Second page (with cursor from last item of first page):
-- SELECT * FROM match_products_keyset(
--     query_embedding := '[0.1, 0.2, ...]'::vector(512),
--     filter_gender := 'female',
--     filter_categories := ARRAY['tops', 'dresses'],
--     cursor_score := 0.847,  -- score of last item
--     cursor_id := 'uuid-of-last-item',
--     p_limit := 50
-- );
--
-- =====================================================

-- Grant permissions
GRANT EXECUTE ON FUNCTION match_products_keyset TO anon, authenticated;
GRANT EXECUTE ON FUNCTION get_trending_keyset TO anon, authenticated;
GRANT EXECUTE ON FUNCTION get_exploration_keyset TO anon, authenticated;
GRANT EXECUTE ON FUNCTION count_available_products_keyset TO anon, authenticated;
