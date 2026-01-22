-- =====================================================
-- Fix product functions for dual schema
-- (products table + review image_embeddings)
-- Run this in Supabase SQL Editor
-- =====================================================

-- =====================================================
-- STEP 1: Fix get_trending_products to work with products table
-- The products table has trending_score column directly
-- =====================================================
CREATE OR REPLACE FUNCTION get_trending_products(
    filter_gender text DEFAULT NULL,
    filter_category text DEFAULT NULL,
    result_limit int DEFAULT 50
)
RETURNS TABLE (
    product_id uuid,
    name text,
    brand text,
    category text,
    gender text[],
    price numeric,
    primary_image_url text,
    trending_score float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id as product_id,
        p.name,
        p.brand,
        p.category,
        p.gender,
        p.price,
        p.primary_image_url,
        COALESCE(p.trending_score, 0.0)::float as trending_score
    FROM products p
    WHERE
        (filter_gender IS NULL OR filter_gender = ANY(p.gender))
        AND (filter_category IS NULL OR p.category = filter_category)
        AND p.in_stock = true
    ORDER BY COALESCE(p.trending_score, 0) DESC, p.created_at DESC
    LIMIT result_limit;
END;
$$;

-- =====================================================
-- STEP 2: Create get_popular_products for general popularity
-- Uses view_count, cart_add_count etc from products table
-- =====================================================
CREATE OR REPLACE FUNCTION get_popular_products(
    filter_gender text DEFAULT NULL,
    filter_category text DEFAULT NULL,
    result_limit int DEFAULT 50
)
RETURNS TABLE (
    product_id uuid,
    name text,
    brand text,
    category text,
    gender text[],
    price numeric,
    primary_image_url text,
    popularity_score float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id as product_id,
        p.name,
        p.brand,
        p.category,
        p.gender,
        p.price,
        p.primary_image_url,
        (COALESCE(p.view_count, 0) * 0.1 +
         COALESCE(p.cart_add_count, 0) * 0.5 +
         COALESCE(p.trending_score, 0) * 100)::float as popularity_score
    FROM products p
    WHERE
        (filter_gender IS NULL OR filter_gender = ANY(p.gender))
        AND (filter_category IS NULL OR p.category = filter_category)
        AND p.in_stock = true
    ORDER BY popularity_score DESC, p.created_at DESC
    LIMIT result_limit;
END;
$$;

-- =====================================================
-- STEP 3: Create search_products_by_category for category browsing
-- =====================================================
CREATE OR REPLACE FUNCTION search_products_by_category(
    filter_gender text DEFAULT 'female',
    filter_categories text[] DEFAULT NULL,
    exclude_ids uuid[] DEFAULT NULL,
    result_limit int DEFAULT 50
)
RETURNS TABLE (
    product_id uuid,
    name text,
    brand text,
    category text,
    sub_category text,
    gender text[],
    price numeric,
    primary_image_url text,
    hero_image_url text
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id as product_id,
        p.name,
        p.brand,
        p.category,
        p.sub_category,
        p.gender,
        p.price,
        p.primary_image_url,
        p.hero_image_url
    FROM products p
    WHERE
        filter_gender = ANY(p.gender)
        AND p.in_stock = true
        AND (filter_categories IS NULL OR p.category = ANY(filter_categories))
        AND (exclude_ids IS NULL OR NOT (p.id = ANY(exclude_ids)))
    ORDER BY COALESCE(p.trending_score, 0) DESC, RANDOM()
    LIMIT result_limit;
END;
$$;

-- =====================================================
-- STEP 4: Create get_product_categories for UI
-- =====================================================
CREATE OR REPLACE FUNCTION get_product_categories(
    filter_gender text DEFAULT 'female'
)
RETURNS TABLE (
    category text,
    product_count bigint
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.category,
        COUNT(*)::bigint as product_count
    FROM products p
    WHERE
        filter_gender = ANY(p.gender)
        AND p.in_stock = true
    GROUP BY p.category
    ORDER BY product_count DESC;
END;
$$;

-- =====================================================
-- STEP 5: match_embeddings stays as is (works with image_embeddings table)
-- It returns review_sku_id for the review system
-- Already created in 001 migration
-- =====================================================

-- =====================================================
-- STEP 6: Create indexes for performance on products table
-- =====================================================
CREATE INDEX IF NOT EXISTS idx_products_gender_stock ON products USING GIN (gender) WHERE in_stock = true;
CREATE INDEX IF NOT EXISTS idx_products_category_gender ON products (category) WHERE in_stock = true;
CREATE INDEX IF NOT EXISTS idx_products_trending ON products (trending_score DESC NULLS LAST) WHERE in_stock = true;

-- =====================================================
-- Verify functions work
-- =====================================================
-- SELECT * FROM get_trending_products('female', NULL, 5);
-- SELECT * FROM get_popular_products('female', NULL, 5);
-- SELECT * FROM search_products_by_category('female', ARRAY['dresses', 'tops'], NULL, 5);
-- SELECT * FROM get_product_categories('female');
