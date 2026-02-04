-- =====================================================
-- Lifestyle Filtering: Style and Occasion Classification
-- Run after 016_add_article_type.sql
--
-- This migration adds:
-- 1. computed_style_scores - JSONB with similarity scores for coverage styles
-- 2. computed_occasion_scores - JSONB with similarity scores for occasions
-- 3. GIN indexes for efficient filtering
-- 4. Updated keyset functions with lifestyle filtering
--
-- Score format: {"deep-necklines": 0.35, "sheer": 0.12, ...}
-- Storing scores (not just tags) allows threshold adjustment without backfill
-- =====================================================

-- =====================================================
-- STEP 1: Add computed score columns to products table
-- =====================================================
ALTER TABLE products ADD COLUMN IF NOT EXISTS computed_style_scores jsonb DEFAULT '{}';
ALTER TABLE products ADD COLUMN IF NOT EXISTS computed_occasion_scores jsonb DEFAULT '{}';

COMMENT ON COLUMN products.computed_style_scores IS
    'Pre-computed style scores from FashionCLIP: {"deep-necklines": 0.35, "sheer": 0.12, "cutouts": 0.08, "backless": 0.05, "strapless": 0.22}';
COMMENT ON COLUMN products.computed_occasion_scores IS
    'Pre-computed occasion scores from FashionCLIP: {"casual": 0.45, "office": 0.32, "evening": 0.18, "beach": 0.05}';

-- =====================================================
-- STEP 2: Create helper function to check style thresholds
-- Returns true if ANY style in exclude_styles exceeds its threshold
-- =====================================================
CREATE OR REPLACE FUNCTION has_excluded_style(
    style_scores jsonb,
    exclude_styles text[],
    threshold float DEFAULT 0.25
)
RETURNS boolean
LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    style_name text;
    score float;
BEGIN
    IF exclude_styles IS NULL OR array_length(exclude_styles, 1) IS NULL THEN
        RETURN false;
    END IF;

    IF style_scores IS NULL OR style_scores = '{}' THEN
        RETURN false;  -- No style data = don't exclude
    END IF;

    FOREACH style_name IN ARRAY exclude_styles LOOP
        score := (style_scores ->> style_name)::float;
        IF score IS NOT NULL AND score >= threshold THEN
            RETURN true;  -- Found excluded style above threshold
        END IF;
    END LOOP;

    RETURN false;
END;
$$;

-- =====================================================
-- STEP 3: Create helper function to check occasion match
-- Returns true if ANY occasion in include_occasions exceeds its threshold
-- =====================================================
CREATE OR REPLACE FUNCTION matches_occasion(
    occasion_scores jsonb,
    include_occasions text[],
    threshold float DEFAULT 0.20
)
RETURNS boolean
LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    occasion_name text;
    score float;
BEGIN
    IF include_occasions IS NULL OR array_length(include_occasions, 1) IS NULL THEN
        RETURN true;  -- No filter = match all
    END IF;

    IF occasion_scores IS NULL OR occasion_scores = '{}' THEN
        RETURN true;  -- No occasion data = don't filter out
    END IF;

    FOREACH occasion_name IN ARRAY include_occasions LOOP
        score := (occasion_scores ->> occasion_name)::float;
        IF score IS NOT NULL AND score >= threshold THEN
            RETURN true;  -- Found matching occasion above threshold
        END IF;
    END LOOP;

    RETURN false;  -- No matching occasions found
END;
$$;

-- =====================================================
-- STEP 4: Create helper function to check article type inclusion
-- Returns true if article_type matches any in include list
-- =====================================================
CREATE OR REPLACE FUNCTION matches_article_type(
    product_article_type text,
    include_article_types text[]
)
RETURNS boolean
LANGUAGE plpgsql IMMUTABLE AS $$
BEGIN
    IF include_article_types IS NULL OR array_length(include_article_types, 1) IS NULL THEN
        RETURN true;  -- No filter = match all
    END IF;

    IF product_article_type IS NULL OR product_article_type = '' THEN
        RETURN false;  -- No article type on product = don't match
    END IF;

    -- Case-insensitive comparison
    RETURN lower(product_article_type) = ANY(
        SELECT lower(unnest(include_article_types))
    );
END;
$$;

-- Grant execute on helper functions
GRANT EXECUTE ON FUNCTION has_excluded_style TO anon, authenticated;
GRANT EXECUTE ON FUNCTION matches_occasion TO anon, authenticated;
GRANT EXECUTE ON FUNCTION matches_article_type TO anon, authenticated;

-- =====================================================
-- STEP 5: Update match_products_keyset with lifestyle filters
-- =====================================================
DROP FUNCTION IF EXISTS match_products_keyset(vector(512), text, text[], text[], text[], text[], numeric, numeric, float, uuid, int);

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
    -- NEW: Lifestyle filters
    exclude_styles text[] DEFAULT NULL,
    include_occasions text[] DEFAULT NULL,
    include_article_types text[] DEFAULT NULL,
    style_threshold float DEFAULT 0.25,
    occasion_threshold float DEFAULT 0.20
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
      -- NEW: Lifestyle filters
      AND NOT has_excluded_style(p.computed_style_scores, exclude_styles, style_threshold)
      AND matches_occasion(p.computed_occasion_scores, include_occasions, occasion_threshold)
      AND matches_article_type(p.article_type, include_article_types)
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
-- STEP 6: Update get_trending_keyset with lifestyle filters
-- =====================================================
DROP FUNCTION IF EXISTS get_trending_keyset(text, text[], text[], text[], text[], numeric, numeric, float, uuid, int);

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
    -- NEW: Lifestyle filters
    exclude_styles text[] DEFAULT NULL,
    include_occasions text[] DEFAULT NULL,
    include_article_types text[] DEFAULT NULL,
    style_threshold float DEFAULT 0.25,
    occasion_threshold float DEFAULT 0.20
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
      -- NEW: Lifestyle filters
      AND NOT has_excluded_style(p.computed_style_scores, exclude_styles, style_threshold)
      AND matches_occasion(p.computed_occasion_scores, include_occasions, occasion_threshold)
      AND matches_article_type(p.article_type, include_article_types)
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
-- STEP 7: Update get_exploration_keyset with lifestyle filters
-- =====================================================
DROP FUNCTION IF EXISTS get_exploration_keyset(text, text[], text[], text[], text[], numeric, numeric, text, float, uuid, int);

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
    -- NEW: Lifestyle filters
    exclude_styles text[] DEFAULT NULL,
    include_occasions text[] DEFAULT NULL,
    include_article_types text[] DEFAULT NULL,
    style_threshold float DEFAULT 0.25,
    occasion_threshold float DEFAULT 0.20
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
      -- NEW: Lifestyle filters
      AND NOT has_excluded_style(p.computed_style_scores, exclude_styles, style_threshold)
      AND matches_occasion(p.computed_occasion_scores, include_occasions, occasion_threshold)
      AND matches_article_type(p.article_type, include_article_types)
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
-- STEP 8: Create index for JSONB queries (optional optimization)
-- GIN indexes help with JSONB key existence checks
-- =====================================================
CREATE INDEX IF NOT EXISTS idx_products_style_scores
    ON products USING GIN (computed_style_scores);
CREATE INDEX IF NOT EXISTS idx_products_occasion_scores
    ON products USING GIN (computed_occasion_scores);

-- =====================================================
-- Verification
-- =====================================================
SELECT 'Lifestyle filtering migration complete' as status;
SELECT
    (SELECT count(*) FROM products WHERE computed_style_scores != '{}') as products_with_style_scores,
    (SELECT count(*) FROM products WHERE computed_occasion_scores != '{}') as products_with_occasion_scores;
