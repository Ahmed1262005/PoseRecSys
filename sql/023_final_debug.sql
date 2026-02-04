-- Final debug: Test WHERE clause with parameters
CREATE OR REPLACE FUNCTION test_where_debug(
    filter_gender text DEFAULT NULL,
    exclude_styles text[] DEFAULT NULL,
    style_threshold float DEFAULT 0.25,
    p_limit int DEFAULT 20
)
RETURNS TABLE(
    product_id uuid,
    name text,
    sheer_score float,
    has_excluded_result boolean,
    should_exclude boolean
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id as product_id,
        p.name,
        (p.computed_style_scores->>'sheer')::float as sheer_score,
        has_excluded_style(p.computed_style_scores, exclude_styles, style_threshold) as has_excluded_result,
        ((p.computed_style_scores->>'sheer')::float >= style_threshold) as should_exclude
    FROM products p
    WHERE p.in_stock = true
      AND (filter_gender IS NULL OR filter_gender = ANY(p.gender))
      -- This is the filter we're testing
      AND NOT has_excluded_style(p.computed_style_scores, exclude_styles, style_threshold)
    ORDER BY (p.computed_style_scores->>'sheer')::float DESC NULLS LAST
    LIMIT p_limit;
END;
$$;

GRANT EXECUTE ON FUNCTION test_where_debug TO anon, authenticated;
