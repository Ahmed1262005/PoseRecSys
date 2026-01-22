-- =====================================================
-- Fix get_trending_products to only return products WITH embeddings
-- =====================================================

DROP FUNCTION IF EXISTS get_trending_products(text, text, int);

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
    INNER JOIN image_embeddings ie ON ie.sku_id = p.id  -- Only products with embeddings
    WHERE
        (filter_gender IS NULL OR filter_gender = ANY(p.gender))
        AND (filter_category IS NULL OR p.category = filter_category)
        AND p.in_stock = true
    ORDER BY COALESCE(p.trending_score, 0) DESC, p.created_at DESC
    LIMIT result_limit;
END;
$$;

-- Test it
-- SELECT * FROM get_trending_products('female', NULL, 5);
