-- Simple test function to verify cursor logic works
-- Run this in Supabase SQL Editor

-- First, drop existing function to ensure clean slate
DROP FUNCTION IF EXISTS get_trending_keyset(text, text[], text[], text[], text[], numeric, numeric, float, uuid, int);

-- Recreate with explicit cursor logic
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
    similarity float
)
LANGUAGE plpgsql AS $$
DECLARE
    computed_score float;
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
        COALESCE(p.trending_score, 0.5)::float as similarity
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
      -- KEYSET CURSOR - simplified and explicit
      AND (
          cursor_score IS NULL
          OR COALESCE(p.trending_score, 0.5) < cursor_score
          OR (
              COALESCE(p.trending_score, 0.5) = cursor_score
              AND p.id < cursor_id
          )
      )
    ORDER BY COALESCE(p.trending_score, 0.5) DESC, p.id DESC
    LIMIT p_limit;
END;
$$;

-- Grant permissions
GRANT EXECUTE ON FUNCTION get_trending_keyset TO anon, authenticated;

-- Test query - should return 0 rows with min UUID
SELECT COUNT(*) as should_be_zero FROM get_trending_keyset(
    filter_gender := 'female',
    cursor_score := 0.5,
    cursor_id := '00000000-0000-0000-0000-000000000000'::uuid,
    p_limit := 5
);
