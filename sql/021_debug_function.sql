-- Debug: Hardcoded filter to verify WHERE clause works
CREATE OR REPLACE FUNCTION get_exploration_debug(
    filter_gender text DEFAULT NULL,
    p_limit int DEFAULT 50
)
RETURNS TABLE(
    product_id uuid,
    name text,
    sheer_score float,
    excluded boolean
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id as product_id,
        p.name,
        (p.computed_style_scores->>'sheer')::float as sheer_score,
        has_excluded_style(p.computed_style_scores, ARRAY['sheer'], 0.20) as excluded
    FROM products p
    WHERE p.in_stock = true
      AND (filter_gender IS NULL OR filter_gender = ANY(p.gender))
      -- Hardcoded filter: exclude sheer > 0.20
      AND NOT has_excluded_style(p.computed_style_scores, ARRAY['sheer'], 0.20)
    ORDER BY (p.computed_style_scores->>'sheer')::float DESC NULLS LAST
    LIMIT p_limit;
END;
$$;

GRANT EXECUTE ON FUNCTION get_exploration_debug TO anon, authenticated;

SELECT 'Debug function created' as status;
