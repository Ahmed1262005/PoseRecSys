-- =====================================================
-- Endless Scroll Functions for Recommendation Pipeline
-- Run this after 006_onboarding_v2.sql
--
-- Changes from 005_candidate_selection_functions.sql:
-- 1. Add p_offset parameter for true SQL-level pagination
-- 2. Functions return fresh candidates per page (not slicing fixed pool)
-- 3. Optimized for large exclude_ids arrays (session seen items)
--
-- Key Insight:
-- Current system: 450 candidates generated ONCE, sliced for pagination
-- Endless scroll: Fresh candidates PER PAGE with SQL OFFSET + exclusion
-- =====================================================

-- =====================================================
-- STEP 1: match_products_endless
-- pgvector search with OFFSET for true pagination
-- =====================================================
CREATE OR REPLACE FUNCTION match_products_endless(
    query_embedding vector(512),
    filter_gender text DEFAULT NULL,
    filter_categories text[] DEFAULT NULL,
    exclude_colors text[] DEFAULT NULL,
    exclude_materials text[] DEFAULT NULL,
    exclude_brands text[] DEFAULT NULL,
    min_price numeric DEFAULT NULL,
    max_price numeric DEFAULT NULL,
    exclude_product_ids text[] DEFAULT NULL,    -- Session seen items
    p_offset int DEFAULT 0,                     -- SQL-level pagination
    p_limit int DEFAULT 200                     -- Items per retrieval batch
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
      -- Exclude specific products (session seen items)
      -- Using NOT IN with subquery for better performance with large arrays
      AND (exclude_product_ids IS NULL OR p.id::text != ALL(exclude_product_ids))
      -- Ensure embedding exists
      AND ie.embedding IS NOT NULL
    ORDER BY ie.embedding <=> query_embedding
    OFFSET p_offset
    LIMIT p_limit;
END;
$$;

-- =====================================================
-- STEP 2: get_trending_endless
-- Trending products with OFFSET for true pagination
-- =====================================================
CREATE OR REPLACE FUNCTION get_trending_endless(
    filter_gender text DEFAULT NULL,
    filter_categories text[] DEFAULT NULL,
    exclude_colors text[] DEFAULT NULL,
    exclude_materials text[] DEFAULT NULL,
    exclude_brands text[] DEFAULT NULL,
    min_price numeric DEFAULT NULL,
    max_price numeric DEFAULT NULL,
    exclude_product_ids text[] DEFAULT NULL,
    p_offset int DEFAULT 0,
    p_limit int DEFAULT 100
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
        COALESCE(p.trending_score, 0.0)::float as trending_score,
        0.0::float as similarity
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
      AND (exclude_brands IS NULL OR p.brand != ALL(exclude_brands))
      AND (exclude_product_ids IS NULL OR p.id::text != ALL(exclude_product_ids))
    ORDER BY COALESCE(p.trending_score, 0) DESC, p.created_at DESC
    OFFSET p_offset
    LIMIT p_limit;
END;
$$;

-- =====================================================
-- STEP 3: get_exploration_endless
-- Random diverse items with exclusion (no offset - random each time)
-- =====================================================
CREATE OR REPLACE FUNCTION get_exploration_endless(
    filter_gender text DEFAULT NULL,
    filter_categories text[] DEFAULT NULL,
    exclude_colors text[] DEFAULT NULL,
    exclude_materials text[] DEFAULT NULL,
    exclude_brands text[] DEFAULT NULL,
    exclude_product_ids text[] DEFAULT NULL,
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
    hero_image_url text
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
        p.hero_image_url
    FROM products p
    WHERE p.in_stock = true
      AND (filter_gender IS NULL OR filter_gender = ANY(p.gender))
      AND (filter_categories IS NULL
           OR p.broad_category = ANY(filter_categories)
           OR p.category = ANY(filter_categories))
      AND (exclude_colors IS NULL OR NOT (p.colors && exclude_colors))
      AND (exclude_materials IS NULL OR NOT (p.materials && exclude_materials))
      AND (exclude_brands IS NULL OR p.brand != ALL(exclude_brands))
      AND (exclude_product_ids IS NULL OR p.id::text != ALL(exclude_product_ids))
    ORDER BY RANDOM()
    LIMIT p_limit;
END;
$$;

-- =====================================================
-- STEP 4: count_available_products
-- Count remaining products for has_more calculation
-- =====================================================
CREATE OR REPLACE FUNCTION count_available_products(
    filter_gender text DEFAULT NULL,
    filter_categories text[] DEFAULT NULL,
    exclude_colors text[] DEFAULT NULL,
    exclude_materials text[] DEFAULT NULL,
    exclude_brands text[] DEFAULT NULL,
    exclude_product_ids text[] DEFAULT NULL
)
RETURNS int
LANGUAGE plpgsql AS $$
DECLARE
    total_count int;
BEGIN
    SELECT COUNT(*)
    INTO total_count
    FROM products p
    WHERE p.in_stock = true
      AND (filter_gender IS NULL OR filter_gender = ANY(p.gender))
      AND (filter_categories IS NULL
           OR p.broad_category = ANY(filter_categories)
           OR p.category = ANY(filter_categories))
      AND (exclude_colors IS NULL OR NOT (p.colors && exclude_colors))
      AND (exclude_materials IS NULL OR NOT (p.materials && exclude_materials))
      AND (exclude_brands IS NULL OR p.brand != ALL(exclude_brands))
      AND (exclude_product_ids IS NULL OR p.id::text != ALL(exclude_product_ids));

    RETURN total_count;
END;
$$;

-- =====================================================
-- STEP 5: Create index for better exclusion performance
-- This helps when exclude_product_ids has many items
-- =====================================================
CREATE INDEX IF NOT EXISTS idx_products_id_text ON products ((id::text));

-- =====================================================
-- Usage Notes:
-- =====================================================
--
-- Page 1 (no seen items yet):
-- SELECT * FROM match_products_endless(
--     taste_vector,
--     'female',
--     ARRAY['bottoms'],
--     NULL,  -- exclude_colors
--     NULL,  -- exclude_materials
--     NULL,  -- exclude_brands
--     NULL,  -- min_price
--     NULL,  -- max_price
--     NULL,  -- exclude_product_ids (empty for first page)
--     0,     -- offset = 0
--     200    -- limit
-- );
--
-- Page 2 (after 50 items seen):
-- SELECT * FROM match_products_endless(
--     taste_vector,
--     'female',
--     ARRAY['bottoms'],
--     NULL, NULL, NULL, NULL, NULL,
--     ARRAY['id1', 'id2', ...],  -- 50 seen items
--     0,     -- offset = 0 (exclusion handles pagination)
--     200
-- );
--
-- Note: We pass exclude_product_ids instead of using offset
-- because vector similarity ordering changes as items are excluded.
-- This ensures we always get the NEXT BEST items, not skip fixed positions.
-- =====================================================
