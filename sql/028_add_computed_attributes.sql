-- =====================================================
-- Pattern Classification & Attribute Support
-- Run after 027_onboarding_v3.sql
--
-- This migration adds:
-- 1. computed_pattern_scores - JSONB with similarity scores for patterns
-- 2. Ensure fit, length, sleeve columns exist with indexes
-- 3. Helper functions for pattern matching
-- 4. Update keyset functions to support pattern filtering
-- =====================================================

-- =====================================================
-- STEP 1: Add computed_pattern_scores column
-- =====================================================
ALTER TABLE products ADD COLUMN IF NOT EXISTS computed_pattern_scores jsonb DEFAULT '{}';

COMMENT ON COLUMN products.computed_pattern_scores IS
    'Pre-computed pattern scores from FashionCLIP: {"solid": 0.80, "stripes": 0.25, "floral": 0.15, "plaid": 0.10, "animal-print": 0.05}';

-- =====================================================
-- STEP 2: Ensure fit, length, sleeve, rise columns exist
-- (These may already exist from earlier migrations)
-- =====================================================
ALTER TABLE products ADD COLUMN IF NOT EXISTS fit text;
ALTER TABLE products ADD COLUMN IF NOT EXISTS length text;
ALTER TABLE products ADD COLUMN IF NOT EXISTS sleeve text;
ALTER TABLE products ADD COLUMN IF NOT EXISTS rise text;

COMMENT ON COLUMN products.fit IS
    'Computed fit classification: slim, fitted, regular, relaxed, oversized';
COMMENT ON COLUMN products.length IS
    'Computed length classification: cropped, standard, long (tops/bottoms) or mini, midi, maxi (dresses/skirts)';
COMMENT ON COLUMN products.sleeve IS
    'Computed sleeve classification: sleeveless, short-sleeve, 3-4-sleeve, long-sleeve';
COMMENT ON COLUMN products.rise IS
    'Computed rise classification for bottoms: high-rise, mid-rise, low-rise';

-- =====================================================
-- STEP 3: Create indexes for efficient filtering
-- =====================================================
CREATE INDEX IF NOT EXISTS idx_products_pattern_scores ON products USING GIN (computed_pattern_scores);
CREATE INDEX IF NOT EXISTS idx_products_fit ON products(fit);
CREATE INDEX IF NOT EXISTS idx_products_length ON products(length);
CREATE INDEX IF NOT EXISTS idx_products_sleeve ON products(sleeve);
CREATE INDEX IF NOT EXISTS idx_products_rise ON products(rise);

-- =====================================================
-- STEP 4: Create helper function to check pattern inclusion
-- Returns true if ANY pattern in include_patterns exceeds threshold
-- =====================================================
CREATE OR REPLACE FUNCTION matches_pattern(
    pattern_scores jsonb,
    include_patterns text[],
    threshold float DEFAULT 0.30
)
RETURNS boolean
LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    pattern_name text;
    score float;
BEGIN
    IF include_patterns IS NULL OR array_length(include_patterns, 1) IS NULL THEN
        RETURN true;  -- No filter = match all
    END IF;

    IF pattern_scores IS NULL OR pattern_scores = '{}' THEN
        RETURN true;  -- No pattern data = don't filter out
    END IF;

    FOREACH pattern_name IN ARRAY include_patterns LOOP
        score := (pattern_scores ->> pattern_name)::float;
        IF score IS NOT NULL AND score >= threshold THEN
            RETURN true;  -- Found matching pattern above threshold
        END IF;
    END LOOP;

    RETURN false;  -- No matching patterns found
END;
$$;

-- =====================================================
-- STEP 5: Create helper function to check pattern exclusion
-- Returns true if ANY pattern in exclude_patterns exceeds threshold
-- =====================================================
CREATE OR REPLACE FUNCTION has_excluded_pattern(
    pattern_scores jsonb,
    exclude_patterns text[],
    threshold float DEFAULT 0.30
)
RETURNS boolean
LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    pattern_name text;
    score float;
BEGIN
    IF exclude_patterns IS NULL OR array_length(exclude_patterns, 1) IS NULL THEN
        RETURN false;  -- No exclusions = don't exclude
    END IF;

    IF pattern_scores IS NULL OR pattern_scores = '{}' THEN
        RETURN false;  -- No pattern data = don't exclude
    END IF;

    FOREACH pattern_name IN ARRAY exclude_patterns LOOP
        score := (pattern_scores ->> pattern_name)::float;
        IF score IS NOT NULL AND score >= threshold THEN
            RETURN true;  -- Found excluded pattern above threshold
        END IF;
    END LOOP;

    RETURN false;
END;
$$;

-- Grant execute on helper functions
GRANT EXECUTE ON FUNCTION matches_pattern TO anon, authenticated;
GRANT EXECUTE ON FUNCTION has_excluded_pattern TO anon, authenticated;

-- =====================================================
-- STEP 6: Create helper function for batch classification
-- Returns embeddings with category for attribute classification
-- =====================================================
CREATE OR REPLACE FUNCTION get_embeddings_with_category(
    p_offset int DEFAULT 0,
    p_limit int DEFAULT 100
)
RETURNS TABLE(sku_id uuid, embedding vector(512), broad_category text)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT ie.sku_id, ie.embedding, p.broad_category
    FROM image_embeddings ie
    JOIN products p ON p.id = ie.sku_id
    WHERE ie.embedding IS NOT NULL
    ORDER BY ie.sku_id
    OFFSET p_offset
    LIMIT p_limit;
END;
$$;

GRANT EXECUTE ON FUNCTION get_embeddings_with_category TO anon, authenticated;

-- =====================================================
-- STEP 7: Update match_products_keyset with pattern filters
-- =====================================================
DROP FUNCTION IF EXISTS match_products_keyset(vector(512), text, text[], text[], text[], text[], numeric, numeric, float, uuid, int, text[], text[], text[], float, float);

CREATE OR REPLACE FUNCTION match_products_keyset(
    query_embedding vector(512),
    filter_gender text DEFAULT NULL,
    filter_categories text[] DEFAULT NULL,
    exclude_colors text[] DEFAULT NULL,
    exclude_materials text[] DEFAULT NULL,
    exclude_brands text[] DEFAULT NULL,
    min_price numeric DEFAULT NULL,
    max_price numeric DEFAULT NULL,
    cursor_score float DEFAULT NULL,
    cursor_id uuid DEFAULT NULL,
    p_limit int DEFAULT 50,
    -- Lifestyle filters
    exclude_styles text[] DEFAULT NULL,
    include_occasions text[] DEFAULT NULL,
    include_article_types text[] DEFAULT NULL,
    style_threshold float DEFAULT 0.25,
    occasion_threshold float DEFAULT 0.20,
    -- NEW: Pattern filters
    include_patterns text[] DEFAULT NULL,
    exclude_patterns text[] DEFAULT NULL,
    pattern_threshold float DEFAULT 0.30
)
RETURNS TABLE(
    product_id uuid,
    name text,
    brand text,
    category text,
    broad_category text,
    article_type text,
    colors text[],
    materials text[],
    price numeric,
    fit text,
    length text,
    sleeve text,
    neckline text,
    rise text,
    style_tags text[],
    primary_image_url text,
    hero_image_url text,
    gallery_images text[],
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
        p.article_type,
        p.colors,
        p.materials,
        p.price,
        p.fit,
        p.length,
        p.sleeve,
        p.neckline,
        p.rise,
        p.style_tags,
        p.primary_image_url,
        p.hero_image_url,
        p.gallery_images,
        1 - (ie.embedding <=> query_embedding) as similarity
    FROM products p
    JOIN image_embeddings ie ON ie.sku_id = p.id
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
      AND ie.sku_id IS NOT NULL
      AND ie.embedding IS NOT NULL
      -- Lifestyle filters
      AND NOT has_excluded_style(p.computed_style_scores, exclude_styles, style_threshold)
      AND matches_occasion(p.computed_occasion_scores, include_occasions, occasion_threshold)
      AND matches_article_type(p.article_type, include_article_types)
      -- NEW: Pattern filters
      AND matches_pattern(p.computed_pattern_scores, include_patterns, pattern_threshold)
      AND NOT has_excluded_pattern(p.computed_pattern_scores, exclude_patterns, pattern_threshold)
      -- KEYSET CURSOR
      AND (
          cursor_score IS NULL
          OR (1 - (ie.embedding <=> query_embedding)) < cursor_score
          OR (
              (1 - (ie.embedding <=> query_embedding)) = cursor_score
              AND p.id < cursor_id
          )
      )
    ORDER BY (1 - (ie.embedding <=> query_embedding)) DESC, p.id DESC
    LIMIT p_limit;
END;
$$;

GRANT EXECUTE ON FUNCTION match_products_keyset TO anon, authenticated;

-- =====================================================
-- STEP 8: Update get_trending_keyset with pattern filters
-- =====================================================
DROP FUNCTION IF EXISTS get_trending_keyset(text, text[], text[], text[], text[], numeric, numeric, float, uuid, int, text[], text[], text[], float, float);

CREATE OR REPLACE FUNCTION get_trending_keyset(
    filter_gender text DEFAULT NULL,
    filter_categories text[] DEFAULT NULL,
    exclude_colors text[] DEFAULT NULL,
    exclude_materials text[] DEFAULT NULL,
    exclude_brands text[] DEFAULT NULL,
    min_price numeric DEFAULT NULL,
    max_price numeric DEFAULT NULL,
    cursor_score float DEFAULT NULL,
    cursor_id uuid DEFAULT NULL,
    p_limit int DEFAULT 50,
    -- Lifestyle filters
    exclude_styles text[] DEFAULT NULL,
    include_occasions text[] DEFAULT NULL,
    include_article_types text[] DEFAULT NULL,
    style_threshold float DEFAULT 0.25,
    occasion_threshold float DEFAULT 0.20,
    -- NEW: Pattern filters
    include_patterns text[] DEFAULT NULL,
    exclude_patterns text[] DEFAULT NULL,
    pattern_threshold float DEFAULT 0.30
)
RETURNS TABLE(
    product_id uuid,
    name text,
    brand text,
    category text,
    broad_category text,
    article_type text,
    colors text[],
    materials text[],
    price numeric,
    fit text,
    length text,
    sleeve text,
    neckline text,
    rise text,
    style_tags text[],
    primary_image_url text,
    hero_image_url text,
    gallery_images text[],
    trending_score float,
    similarity float
)
LANGUAGE plpgsql AS $$
DECLARE
    default_score CONSTANT float := 0.5;
BEGIN
    RETURN QUERY
    SELECT
        p.id as product_id,
        p.name,
        p.brand,
        p.category,
        p.broad_category,
        p.article_type,
        p.colors,
        p.materials,
        p.price,
        p.fit,
        p.length,
        p.sleeve,
        p.neckline,
        p.rise,
        p.style_tags,
        p.primary_image_url,
        p.hero_image_url,
        p.gallery_images,
        COALESCE(NULLIF(p.trending_score, 0), default_score)::float as trending_score,
        COALESCE(NULLIF(p.trending_score, 0), default_score)::float as similarity
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
      -- Lifestyle filters
      AND NOT has_excluded_style(p.computed_style_scores, exclude_styles, style_threshold)
      AND matches_occasion(p.computed_occasion_scores, include_occasions, occasion_threshold)
      AND matches_article_type(p.article_type, include_article_types)
      -- NEW: Pattern filters
      AND matches_pattern(p.computed_pattern_scores, include_patterns, pattern_threshold)
      AND NOT has_excluded_pattern(p.computed_pattern_scores, exclude_patterns, pattern_threshold)
      -- KEYSET CURSOR
      AND (
          cursor_score IS NULL
          OR COALESCE(NULLIF(p.trending_score, 0), default_score) < cursor_score
          OR (
              COALESCE(NULLIF(p.trending_score, 0), default_score) = cursor_score
              AND p.id < cursor_id
          )
      )
    ORDER BY COALESCE(NULLIF(p.trending_score, 0), default_score) DESC, p.id DESC
    LIMIT p_limit;
END;
$$;

GRANT EXECUTE ON FUNCTION get_trending_keyset TO anon, authenticated;

-- =====================================================
-- STEP 9: Update get_exploration_keyset with pattern filters
-- =====================================================
DROP FUNCTION IF EXISTS get_exploration_keyset(text, text[], text[], text[], text[], numeric, numeric, text, float, uuid, int, text[], text[], text[], float, float);

CREATE OR REPLACE FUNCTION get_exploration_keyset(
    filter_gender text DEFAULT NULL,
    filter_categories text[] DEFAULT NULL,
    exclude_colors text[] DEFAULT NULL,
    exclude_materials text[] DEFAULT NULL,
    exclude_brands text[] DEFAULT NULL,
    min_price numeric DEFAULT NULL,
    max_price numeric DEFAULT NULL,
    random_seed text DEFAULT NULL,
    cursor_score float DEFAULT NULL,
    cursor_id uuid DEFAULT NULL,
    p_limit int DEFAULT 50,
    -- Lifestyle filters
    exclude_styles text[] DEFAULT NULL,
    include_occasions text[] DEFAULT NULL,
    include_article_types text[] DEFAULT NULL,
    style_threshold float DEFAULT 0.25,
    occasion_threshold float DEFAULT 0.20,
    -- NEW: Pattern filters
    include_patterns text[] DEFAULT NULL,
    exclude_patterns text[] DEFAULT NULL,
    pattern_threshold float DEFAULT 0.30
)
RETURNS TABLE(
    product_id uuid,
    name text,
    brand text,
    category text,
    broad_category text,
    article_type text,
    colors text[],
    materials text[],
    price numeric,
    fit text,
    length text,
    sleeve text,
    neckline text,
    rise text,
    style_tags text[],
    primary_image_url text,
    hero_image_url text,
    gallery_images text[],
    exploration_score float,
    similarity float
)
LANGUAGE plpgsql AS $$
DECLARE
    effective_seed text;
BEGIN
    effective_seed := COALESCE(random_seed, md5(now()::text));

    RETURN QUERY
    SELECT
        p.id as product_id,
        p.name,
        p.brand,
        p.category,
        p.broad_category,
        p.article_type,
        p.colors,
        p.materials,
        p.price,
        p.fit,
        p.length,
        p.sleeve,
        p.neckline,
        p.rise,
        p.style_tags,
        p.primary_image_url,
        p.hero_image_url,
        p.gallery_images,
        (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float as exploration_score,
        (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float as similarity
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
      -- Lifestyle filters
      AND NOT has_excluded_style(p.computed_style_scores, exclude_styles, style_threshold)
      AND matches_occasion(p.computed_occasion_scores, include_occasions, occasion_threshold)
      AND matches_article_type(p.article_type, include_article_types)
      -- NEW: Pattern filters
      AND matches_pattern(p.computed_pattern_scores, include_patterns, pattern_threshold)
      AND NOT has_excluded_pattern(p.computed_pattern_scores, exclude_patterns, pattern_threshold)
      -- KEYSET CURSOR
      AND (
          cursor_score IS NULL
          OR (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0) < cursor_score
          OR (
              (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0) = cursor_score
              AND p.id < cursor_id
          )
      )
    ORDER BY (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0) DESC, p.id DESC
    LIMIT p_limit;
END;
$$;

GRANT EXECUTE ON FUNCTION get_exploration_keyset TO anon, authenticated;

-- =====================================================
-- Verification
-- =====================================================
SELECT 'Pattern and attribute migration complete' as status;
SELECT
    (SELECT count(*) FROM products WHERE computed_pattern_scores != '{}') as products_with_pattern_scores,
    (SELECT count(*) FROM products WHERE fit IS NOT NULL) as products_with_fit,
    (SELECT count(*) FROM products WHERE length IS NOT NULL) as products_with_length,
    (SELECT count(*) FROM products WHERE sleeve IS NOT NULL) as products_with_sleeve,
    (SELECT count(*) FROM products WHERE rise IS NOT NULL) as products_with_rise;

-- Notify PostgREST to reload schema
NOTIFY pgrst, 'reload schema';
