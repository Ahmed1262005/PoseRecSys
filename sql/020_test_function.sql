-- Test function with a unique name to verify lifestyle filters work
CREATE OR REPLACE FUNCTION get_exploration_keyset_v2(
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
    exploration_score float
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
        (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float as exploration_score
    FROM products p
    WHERE p.in_stock = true
      AND (filter_gender IS NULL OR filter_gender = ANY(p.gender))
      -- LIFESTYLE FILTERS - THE KEY PART
      AND NOT has_excluded_style(p.computed_style_scores, exclude_styles, style_threshold)
      AND matches_occasion(p.computed_occasion_scores, include_occasions, occasion_threshold)
    ORDER BY exploration_score DESC
    LIMIT p_limit;
END;
$$;

GRANT EXECUTE ON FUNCTION get_exploration_keyset_v2 TO anon, authenticated;

SELECT 'Test function get_exploration_keyset_v2 created' as status;
