-- =====================================================
-- Add computed_occasion_scores and computed_style_scores to keyset functions
-- Run after 030_brand_priority_exploration.sql
--
-- This migration updates the keyset functions to return
-- computed_occasion_scores and computed_style_scores for soft scoring.
-- =====================================================

-- Update get_exploration_keyset_with_brands to return scores
DROP FUNCTION IF EXISTS get_exploration_keyset_with_brands(text, text[], text[], text[], text[], numeric, numeric, text, float, uuid, int, text[], text[], text[], float, float, text[], text[], float, text[]);

CREATE OR REPLACE FUNCTION get_exploration_keyset_with_brands(
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
    occasion_threshold float DEFAULT 0.30,
    -- Pattern filters
    include_patterns text[] DEFAULT NULL,
    exclude_patterns text[] DEFAULT NULL,
    pattern_threshold float DEFAULT 0.30,
    -- Brand priority
    preferred_brands text[] DEFAULT NULL
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
    similarity float,
    is_preferred_brand boolean,
    -- NEW: Return computed scores for soft scoring
    computed_occasion_scores jsonb,
    computed_style_scores jsonb
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
        -- Brand-boosted exploration score
        CASE
            WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
            THEN (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float + 0.5
            ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
        END as exploration_score,
        CASE
            WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
            THEN (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float + 0.5
            ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
        END as similarity,
        CASE
            WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
            THEN true
            ELSE false
        END as is_preferred_brand,
        -- NEW: Return computed scores
        COALESCE(p.computed_occasion_scores, '{}'::jsonb) as computed_occasion_scores,
        COALESCE(p.computed_style_scores, '{}'::jsonb) as computed_style_scores
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
      -- Pattern filters
      AND matches_pattern(p.computed_pattern_scores, include_patterns, pattern_threshold)
      AND NOT has_excluded_pattern(p.computed_pattern_scores, exclude_patterns, pattern_threshold)
      -- KEYSET CURSOR
      AND (
          cursor_score IS NULL
          OR (
              CASE
                  WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                  THEN (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float + 0.5
                  ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
              END
          ) < cursor_score
          OR (
              (
                  CASE
                      WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                      THEN (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float + 0.5
                      ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
                  END
              ) = cursor_score
              AND p.id < cursor_id
          )
      )
    ORDER BY (
        CASE
            WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
            THEN (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float + 0.5
            ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
        END
    ) DESC, p.id DESC
    LIMIT p_limit;
END;
$$;

GRANT EXECUTE ON FUNCTION get_exploration_keyset_with_brands TO anon, authenticated;

-- Update get_exploration_keyset (without brand priority) to also return scores
DROP FUNCTION IF EXISTS get_exploration_keyset(text, text[], text[], text[], text[], numeric, numeric, text, float, uuid, int, text[], text[], text[], float, float, text[], text[], float);

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
    occasion_threshold float DEFAULT 0.30,
    -- Pattern filters
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
    similarity float,
    -- NEW: Return computed scores for soft scoring
    computed_occasion_scores jsonb,
    computed_style_scores jsonb
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
        (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float as similarity,
        -- NEW: Return computed scores
        COALESCE(p.computed_occasion_scores, '{}'::jsonb) as computed_occasion_scores,
        COALESCE(p.computed_style_scores, '{}'::jsonb) as computed_style_scores
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
      -- Pattern filters
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

-- Verification
SELECT 'Keyset functions updated with computed scores' as status;

-- Notify PostgREST to reload schema
NOTIFY pgrst, 'reload schema';
