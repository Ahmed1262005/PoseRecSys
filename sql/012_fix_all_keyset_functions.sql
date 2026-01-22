-- Fix all keyset functions to handle trending_score = 0

-- =====================================================
-- Fix match_products_keyset (for warm users with taste vector)
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
-- Fix get_exploration_keyset
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
-- Verify all functions exist
-- =====================================================
SELECT 'Functions deployed:' as status;
SELECT routine_name FROM information_schema.routines
WHERE routine_schema = 'public' AND routine_name LIKE '%keyset%';
