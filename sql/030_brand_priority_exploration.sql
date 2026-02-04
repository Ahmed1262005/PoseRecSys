-- =====================================================
-- Brand-Priority Exploration Function
-- Run after 029_increase_occasion_threshold.sql
--
-- This migration adds a new exploration function that
-- prioritizes items from preferred brands in the ORDER BY.
-- This ensures preferred brands appear at the top of results.
-- =====================================================

-- Create new exploration function with brand priority
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
    -- NEW: Brand priority
    preferred_brands text[] DEFAULT NULL  -- Brands to prioritize in results
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
    is_preferred_brand boolean
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
        -- Brand-boosted exploration score: preferred brands get +0.5 boost
        CASE
            WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
            THEN (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float + 0.5
            ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
        END as exploration_score,
        -- Same score for similarity (used by ranker)
        CASE
            WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
            THEN (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float + 0.5
            ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
        END as similarity,
        -- Flag for preferred brand
        CASE
            WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
            THEN true
            ELSE false
        END as is_preferred_brand
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
      -- KEYSET CURSOR (brand-boosted score)
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
    -- ORDER BY brand-boosted score (preferred brands naturally sort higher)
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

-- Verification
SELECT 'Brand-priority exploration function created' as status;

-- Notify PostgREST to reload schema
NOTIFY pgrst, 'reload schema';
