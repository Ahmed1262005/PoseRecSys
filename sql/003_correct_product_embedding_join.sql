-- =====================================================
-- Correct SQL Functions for Product Recommendations
--
-- KEY INSIGHT:
-- image_embeddings table has TWO types of records:
-- 1. Product embeddings: sku_id IS NOT NULL (59,761) -> links to products.id
-- 2. Review embeddings: review_sku_id IS NOT NULL (50,892) -> separate system
--
-- For recommendations, use: image_embeddings.sku_id = products.id
-- Always filter: WHERE ie.sku_id IS NOT NULL
--
-- Run this in Supabase SQL Editor
-- =====================================================

-- =====================================================
-- STEP 1: Drop ALL existing functions first
-- =====================================================
DROP FUNCTION IF EXISTS match_products_by_embedding(vector(512), int, text, text);
DROP FUNCTION IF EXISTS match_embeddings(vector(512), int, text);
DROP FUNCTION IF EXISTS match_embeddings(vector(512), int);
DROP FUNCTION IF EXISTS get_trending_products(text, text, int);
DROP FUNCTION IF EXISTS get_popular_products(text, text, int);
DROP FUNCTION IF EXISTS get_similar_products(uuid, int, text, text);
DROP FUNCTION IF EXISTS get_product_embedding(uuid);
DROP FUNCTION IF EXISTS get_product_categories(text);
DROP FUNCTION IF EXISTS search_products_by_category(text, text[], uuid[], int);

-- =====================================================
-- STEP 2: Create match_embeddings for raw vector search
-- Returns embedding info only (fast, no product join)
-- =====================================================
CREATE OR REPLACE FUNCTION match_embeddings(
    query_embedding vector(512),
    match_count int DEFAULT 10,
    filter_gender text DEFAULT NULL
)
RETURNS TABLE (
    id uuid,
    sku_id uuid,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        ie.id,
        ie.sku_id,
        1 - (ie.embedding <=> query_embedding) as similarity
    FROM image_embeddings ie
    WHERE ie.sku_id IS NOT NULL  -- Only product embeddings, not review
    ORDER BY ie.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- =====================================================
-- STEP 3: Create match_products_by_embedding
-- Full product info with similarity scores
-- CORRECT JOIN: image_embeddings.sku_id = products.id
-- =====================================================
CREATE OR REPLACE FUNCTION match_products_by_embedding(
    query_embedding vector(512),
    match_count int DEFAULT 20,
    filter_gender text DEFAULT NULL,
    filter_category text DEFAULT NULL
)
RETURNS TABLE (
    product_id uuid,
    name text,
    brand text,
    category text,
    gender text[],
    price numeric,
    primary_image_url text,
    hero_image_url text,
    similarity float
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
        p.hero_image_url,
        1 - (ie.embedding <=> query_embedding) as similarity
    FROM image_embeddings ie
    INNER JOIN products p ON ie.sku_id = p.id  -- CORRECT JOIN
    WHERE
        ie.sku_id IS NOT NULL  -- Only product embeddings
        AND (filter_gender IS NULL OR filter_gender = ANY(p.gender))
        AND (filter_category IS NULL OR p.category = filter_category)
        AND p.in_stock = true
    ORDER BY ie.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- =====================================================
-- STEP 4: Create get_trending_products (uses products table directly)
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
    hero_image_url text,
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
        p.hero_image_url,
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
-- STEP 5: Create get_similar_products
-- Find products similar to a given product ID
-- =====================================================
CREATE OR REPLACE FUNCTION get_similar_products(
    source_product_id uuid,
    match_count int DEFAULT 20,
    filter_gender text DEFAULT NULL,
    filter_category text DEFAULT NULL
)
RETURNS TABLE (
    product_id uuid,
    name text,
    brand text,
    category text,
    gender text[],
    price numeric,
    primary_image_url text,
    hero_image_url text,
    similarity float
)
LANGUAGE plpgsql
AS $$
DECLARE
    source_embedding vector(512);
BEGIN
    -- Get the embedding for the source product
    SELECT ie.embedding INTO source_embedding
    FROM image_embeddings ie
    WHERE ie.sku_id = source_product_id
    LIMIT 1;

    -- If no embedding found, return empty
    IF source_embedding IS NULL THEN
        RETURN;
    END IF;

    -- Find similar products
    RETURN QUERY
    SELECT
        p.id as product_id,
        p.name,
        p.brand,
        p.category,
        p.gender,
        p.price,
        p.primary_image_url,
        p.hero_image_url,
        1 - (ie.embedding <=> source_embedding) as similarity
    FROM image_embeddings ie
    INNER JOIN products p ON ie.sku_id = p.id
    WHERE
        ie.sku_id IS NOT NULL
        AND ie.sku_id != source_product_id  -- Exclude source product
        AND (filter_gender IS NULL OR filter_gender = ANY(p.gender))
        AND (filter_category IS NULL OR p.category = filter_category)
        AND p.in_stock = true
    ORDER BY ie.embedding <=> source_embedding
    LIMIT match_count;
END;
$$;

-- =====================================================
-- STEP 6: Create get_product_embedding
-- Get the embedding vector for a product (for client-side use)
-- =====================================================
CREATE OR REPLACE FUNCTION get_product_embedding(
    p_product_id uuid
)
RETURNS vector(512)
LANGUAGE plpgsql
AS $$
DECLARE
    result_embedding vector(512);
BEGIN
    SELECT ie.embedding INTO result_embedding
    FROM image_embeddings ie
    WHERE ie.sku_id = p_product_id
    LIMIT 1;

    RETURN result_embedding;
END;
$$;

-- =====================================================
-- STEP 7: Create get_product_categories
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
-- STEP 8: Create index on sku_id for faster joins
-- =====================================================
CREATE INDEX IF NOT EXISTS idx_embeddings_sku_id ON image_embeddings (sku_id)
    WHERE sku_id IS NOT NULL;

-- =====================================================
-- VERIFICATION: Run these to test
-- =====================================================
-- SELECT * FROM get_trending_products('female', NULL, 3);
-- SELECT * FROM get_product_categories('female');
