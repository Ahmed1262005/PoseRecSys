-- =====================================================
-- Create recommendation functions
-- Run this AFTER 003a_drop_functions.sql
--
-- KEY: image_embeddings.sku_id = products.id
-- =====================================================

-- =====================================================
-- 1. match_embeddings - raw vector search
-- =====================================================
CREATE FUNCTION match_embeddings(
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
    WHERE ie.sku_id IS NOT NULL
    ORDER BY ie.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- =====================================================
-- 2. match_products_by_embedding - with product details
-- =====================================================
CREATE FUNCTION match_products_by_embedding(
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
    INNER JOIN products p ON ie.sku_id = p.id
    WHERE
        ie.sku_id IS NOT NULL
        AND (filter_gender IS NULL OR filter_gender = ANY(p.gender))
        AND (filter_category IS NULL OR p.category = filter_category)
        AND p.in_stock = true
    ORDER BY ie.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- =====================================================
-- 3. get_trending_products
-- =====================================================
CREATE FUNCTION get_trending_products(
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
-- 4. get_similar_products
-- =====================================================
CREATE FUNCTION get_similar_products(
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
    SELECT ie.embedding INTO source_embedding
    FROM image_embeddings ie
    WHERE ie.sku_id = source_product_id
    LIMIT 1;

    IF source_embedding IS NULL THEN
        RETURN;
    END IF;

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
        AND ie.sku_id != source_product_id
        AND (filter_gender IS NULL OR filter_gender = ANY(p.gender))
        AND (filter_category IS NULL OR p.category = filter_category)
        AND p.in_stock = true
    ORDER BY ie.embedding <=> source_embedding
    LIMIT match_count;
END;
$$;

-- =====================================================
-- 5. get_product_categories
-- =====================================================
CREATE FUNCTION get_product_categories(
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
-- 6. get_product_embedding
-- =====================================================
CREATE FUNCTION get_product_embedding(
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
-- 7. save_tinder_preferences
-- =====================================================
CREATE FUNCTION save_tinder_preferences(
    p_user_id uuid DEFAULT NULL,
    p_anon_id varchar DEFAULT NULL,
    p_gender varchar DEFAULT 'female',
    p_rounds_completed int DEFAULT 0,
    p_categories_tested text[] DEFAULT '{}',
    p_attribute_preferences jsonb DEFAULT '{}',
    p_cluster_preferences jsonb DEFAULT '{}',
    p_prediction_accuracy float DEFAULT NULL,
    p_taste_vector float[] DEFAULT NULL
)
RETURNS uuid
LANGUAGE plpgsql
AS $$
DECLARE
    result_id uuid;
    taste_vec vector(512);
BEGIN
    IF p_taste_vector IS NOT NULL AND array_length(p_taste_vector, 1) = 512 THEN
        taste_vec := p_taste_vector::vector(512);
    END IF;

    IF p_user_id IS NOT NULL THEN
        INSERT INTO user_seed_preferences (
            user_id, gender, rounds_completed, categories_tested,
            attribute_preferences, cluster_preferences, prediction_accuracy,
            taste_vector, completed_at, updated_at
        )
        VALUES (
            p_user_id, p_gender, p_rounds_completed, p_categories_tested,
            p_attribute_preferences, p_cluster_preferences, p_prediction_accuracy,
            taste_vec, NOW(), NOW()
        )
        ON CONFLICT (user_id) DO UPDATE SET
            gender = EXCLUDED.gender,
            rounds_completed = EXCLUDED.rounds_completed,
            categories_tested = EXCLUDED.categories_tested,
            attribute_preferences = EXCLUDED.attribute_preferences,
            cluster_preferences = EXCLUDED.cluster_preferences,
            prediction_accuracy = EXCLUDED.prediction_accuracy,
            taste_vector = EXCLUDED.taste_vector,
            completed_at = NOW(),
            updated_at = NOW()
        RETURNING id INTO result_id;
    ELSIF p_anon_id IS NOT NULL THEN
        INSERT INTO user_seed_preferences (
            anon_id, gender, rounds_completed, categories_tested,
            attribute_preferences, cluster_preferences, prediction_accuracy,
            taste_vector, completed_at, updated_at
        )
        VALUES (
            p_anon_id, p_gender, p_rounds_completed, p_categories_tested,
            p_attribute_preferences, p_cluster_preferences, p_prediction_accuracy,
            taste_vec, NOW(), NOW()
        )
        ON CONFLICT (anon_id) DO UPDATE SET
            gender = EXCLUDED.gender,
            rounds_completed = EXCLUDED.rounds_completed,
            categories_tested = EXCLUDED.categories_tested,
            attribute_preferences = EXCLUDED.attribute_preferences,
            cluster_preferences = EXCLUDED.cluster_preferences,
            prediction_accuracy = EXCLUDED.prediction_accuracy,
            taste_vector = EXCLUDED.taste_vector,
            completed_at = NOW(),
            updated_at = NOW()
        RETURNING id INTO result_id;
    ELSE
        RAISE EXCEPTION 'Either user_id or anon_id must be provided';
    END IF;

    RETURN result_id;
END;
$$;

-- =====================================================
-- Create index for faster joins
-- =====================================================
CREATE INDEX IF NOT EXISTS idx_embeddings_sku_id ON image_embeddings (sku_id)
    WHERE sku_id IS NOT NULL;
