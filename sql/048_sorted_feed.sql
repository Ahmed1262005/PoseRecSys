-- =====================================================
-- Sorted Feed: keyset pagination with deterministic sort
-- by price (ASC / DESC).
--
-- Reuses the same filter set as get_exploration_keyset (044).
-- Only the ORDER BY and cursor comparison change.
--
-- sort_mode values:
--   'price_asc'  -> ORDER BY price ASC,  id ASC   (cheapest first)
--   'price_desc' -> ORDER BY price DESC, id DESC   (most expensive first)
-- =====================================================

CREATE OR REPLACE FUNCTION get_feed_sorted_keyset(
    p_sort_mode text DEFAULT 'price_asc',
    -- standard filters (same as get_exploration_keyset)
    filter_gender text DEFAULT NULL,
    filter_categories text[] DEFAULT NULL,
    exclude_colors text[] DEFAULT NULL,
    exclude_materials text[] DEFAULT NULL,
    exclude_brands text[] DEFAULT NULL,
    min_price numeric DEFAULT NULL,
    max_price numeric DEFAULT NULL,
    cursor_value float DEFAULT NULL,
    cursor_id uuid DEFAULT NULL,
    p_limit int DEFAULT 50,
    exclude_styles text[] DEFAULT NULL,
    include_occasions text[] DEFAULT NULL,
    include_article_types text[] DEFAULT NULL,
    style_threshold float DEFAULT 0.25,
    occasion_threshold float DEFAULT 0.18,
    include_patterns text[] DEFAULT NULL,
    exclude_patterns text[] DEFAULT NULL,
    pattern_threshold float DEFAULT 0.30,
    on_sale_only boolean DEFAULT false,
    new_arrivals_only boolean DEFAULT false,
    new_arrivals_days int DEFAULT 7,
    exclude_product_ids uuid[] DEFAULT NULL,
    include_materials text[] DEFAULT NULL,
    -- Attribute filters via product_attributes JOIN
    attr_include_formality text[] DEFAULT NULL,
    attr_exclude_formality text[] DEFAULT NULL,
    attr_include_seasons text[] DEFAULT NULL,
    attr_exclude_seasons text[] DEFAULT NULL,
    attr_include_style_tags text[] DEFAULT NULL,
    attr_exclude_style_tags text[] DEFAULT NULL,
    attr_include_color_family text[] DEFAULT NULL,
    attr_exclude_color_family text[] DEFAULT NULL,
    attr_include_silhouette text[] DEFAULT NULL,
    attr_exclude_silhouette text[] DEFAULT NULL,
    attr_include_fit_type text[] DEFAULT NULL,
    attr_exclude_fit_type text[] DEFAULT NULL,
    attr_include_coverage text[] DEFAULT NULL,
    attr_exclude_coverage text[] DEFAULT NULL,
    attr_include_pattern text[] DEFAULT NULL,
    attr_exclude_pattern text[] DEFAULT NULL,
    attr_include_neckline text[] DEFAULT NULL,
    attr_exclude_neckline text[] DEFAULT NULL,
    attr_include_sleeve_type text[] DEFAULT NULL,
    attr_exclude_sleeve_type text[] DEFAULT NULL,
    attr_include_length text[] DEFAULT NULL,
    attr_exclude_length text[] DEFAULT NULL,
    attr_include_occasions text[] DEFAULT NULL,
    attr_exclude_occasions text[] DEFAULT NULL
)
RETURNS TABLE(
    product_id uuid, name text, brand text, category text, broad_category text,
    article_type text, colors text[], materials text[], price numeric,
    fit text, length text, sleeve text, neckline text, rise text, style_tags text[],
    primary_image_url text, hero_image_url text, gallery_images text[],
    exploration_score float, similarity float,
    computed_occasion_scores jsonb, computed_style_scores jsonb, computed_pattern_scores jsonb,
    original_price numeric, discount_percent float, is_on_sale boolean, is_new boolean, created_at timestamptz
)
LANGUAGE plpgsql AS $$
BEGIN
    -- ===================================================================
    -- PRICE ASC  (cheapest first)
    -- Cursor: WHERE (price, id) > (cursor_value, cursor_id)
    -- ===================================================================
    IF p_sort_mode = 'price_asc' THEN
        RETURN QUERY
        SELECT
            p.id as product_id, p.name, p.brand, p.category, p.broad_category, p.article_type,
            p.colors, p.materials, p.price, p.fit, p.length, p.sleeve, p.neckline, p.rise, p.style_tags,
            p.primary_image_url, p.hero_image_url, p.gallery_images,
            p.price::float as exploration_score,
            p.price::float as similarity,
            COALESCE(p.computed_occasion_scores, '{}'::jsonb),
            COALESCE(p.computed_style_scores, '{}'::jsonb),
            COALESCE(p.computed_pattern_scores, '{}'::jsonb),
            p.original_price,
            CASE WHEN p.original_price IS NOT NULL AND p.original_price > 0 AND p.price > 0 AND p.original_price > p.price
                THEN ROUND((1 - p.price / p.original_price) * 100)::float ELSE NULL END,
            CASE WHEN p.original_price IS NOT NULL AND p.original_price > p.price THEN true ELSE false END,
            CASE WHEN p.created_at >= NOW() - (new_arrivals_days || ' days')::interval THEN true ELSE false END,
            p.created_at
        FROM products p
        LEFT JOIN product_attributes pa ON p.id = pa.sku_id
        WHERE p.in_stock = true
          AND p.price > 0
          -- Standard filters (identical to get_exploration_keyset)
          AND (filter_gender IS NULL OR filter_gender = ANY(p.gender))
          AND (filter_categories IS NULL OR p.broad_category = ANY(filter_categories) OR p.category = ANY(filter_categories))
          AND (min_price IS NULL OR p.price >= min_price)
          AND (max_price IS NULL OR p.price <= max_price)
          AND (exclude_colors IS NULL OR NOT (p.colors && exclude_colors))
          AND (exclude_materials IS NULL OR NOT (p.materials && exclude_materials))
          AND (exclude_brands IS NULL OR p.brand != ALL(exclude_brands))
          AND NOT has_excluded_style(p.computed_style_scores, exclude_styles, style_threshold)
          AND matches_occasion(p.computed_occasion_scores, include_occasions, occasion_threshold)
          AND matches_article_type(p.article_type, include_article_types)
          AND matches_pattern(p.computed_pattern_scores, include_patterns, pattern_threshold)
          AND NOT has_excluded_pattern(p.computed_pattern_scores, exclude_patterns, pattern_threshold)
          AND (NOT on_sale_only OR (p.original_price IS NOT NULL AND p.original_price > p.price AND p.price > 0))
          AND (NOT new_arrivals_only OR p.created_at >= NOW() - (new_arrivals_days || ' days')::interval)
          AND (exclude_product_ids IS NULL OR NOT (p.id = ANY(exclude_product_ids)))
          AND (include_materials IS NULL OR (p.materials && include_materials))
          -- Attribute filters via product_attributes
          AND (attr_include_formality IS NULL OR pa.formality = ANY(attr_include_formality))
          AND (attr_exclude_formality IS NULL OR pa.formality IS NULL OR pa.formality != ALL(attr_exclude_formality))
          AND (attr_include_color_family IS NULL OR pa.color_family = ANY(attr_include_color_family))
          AND (attr_exclude_color_family IS NULL OR pa.color_family IS NULL OR pa.color_family != ALL(attr_exclude_color_family))
          AND (attr_include_silhouette IS NULL OR pa.silhouette = ANY(attr_include_silhouette))
          AND (attr_exclude_silhouette IS NULL OR pa.silhouette IS NULL OR pa.silhouette != ALL(attr_exclude_silhouette))
          AND (attr_include_fit_type IS NULL OR pa.fit_type = ANY(attr_include_fit_type))
          AND (attr_exclude_fit_type IS NULL OR pa.fit_type IS NULL OR pa.fit_type != ALL(attr_exclude_fit_type))
          AND (attr_include_coverage IS NULL OR pa.coverage_level = ANY(attr_include_coverage))
          AND (attr_exclude_coverage IS NULL OR pa.coverage_level IS NULL OR pa.coverage_level != ALL(attr_exclude_coverage))
          AND (attr_include_pattern IS NULL OR pa.pattern = ANY(attr_include_pattern))
          AND (attr_exclude_pattern IS NULL OR pa.pattern IS NULL OR pa.pattern != ALL(attr_exclude_pattern))
          AND (attr_include_seasons IS NULL OR pa.seasons && attr_include_seasons)
          AND (attr_exclude_seasons IS NULL OR pa.seasons IS NULL OR NOT (pa.seasons && attr_exclude_seasons))
          AND (attr_include_style_tags IS NULL OR pa.style_tags && attr_include_style_tags)
          AND (attr_exclude_style_tags IS NULL OR pa.style_tags IS NULL OR NOT (pa.style_tags && attr_exclude_style_tags))
          AND (attr_include_occasions IS NULL OR pa.occasions && attr_include_occasions)
          AND (attr_exclude_occasions IS NULL OR pa.occasions IS NULL OR NOT (pa.occasions && attr_exclude_occasions))
          AND (attr_include_neckline IS NULL OR (pa.construction->>'neckline') = ANY(attr_include_neckline))
          AND (attr_exclude_neckline IS NULL OR (pa.construction->>'neckline') IS NULL OR (pa.construction->>'neckline') != ALL(attr_exclude_neckline))
          AND (attr_include_sleeve_type IS NULL OR (pa.construction->>'sleeve_type') = ANY(attr_include_sleeve_type))
          AND (attr_exclude_sleeve_type IS NULL OR (pa.construction->>'sleeve_type') IS NULL OR (pa.construction->>'sleeve_type') != ALL(attr_exclude_sleeve_type))
          AND (attr_include_length IS NULL OR (pa.construction->>'length') = ANY(attr_include_length))
          AND (attr_exclude_length IS NULL OR (pa.construction->>'length') IS NULL OR (pa.construction->>'length') != ALL(attr_exclude_length))
          -- Keyset cursor for price_asc: (price, id) > (cursor, cursor_id)
          AND (
              cursor_value IS NULL
              OR p.price > cursor_value
              OR (p.price = cursor_value AND p.id > cursor_id)
          )
        ORDER BY p.price ASC, p.id ASC
        LIMIT p_limit;

    -- ===================================================================
    -- PRICE DESC  (most expensive first)
    -- Cursor: WHERE (price, id) < (cursor_value, cursor_id)
    -- ===================================================================
    ELSIF p_sort_mode = 'price_desc' THEN
        RETURN QUERY
        SELECT
            p.id as product_id, p.name, p.brand, p.category, p.broad_category, p.article_type,
            p.colors, p.materials, p.price, p.fit, p.length, p.sleeve, p.neckline, p.rise, p.style_tags,
            p.primary_image_url, p.hero_image_url, p.gallery_images,
            p.price::float as exploration_score,
            p.price::float as similarity,
            COALESCE(p.computed_occasion_scores, '{}'::jsonb),
            COALESCE(p.computed_style_scores, '{}'::jsonb),
            COALESCE(p.computed_pattern_scores, '{}'::jsonb),
            p.original_price,
            CASE WHEN p.original_price IS NOT NULL AND p.original_price > 0 AND p.price > 0 AND p.original_price > p.price
                THEN ROUND((1 - p.price / p.original_price) * 100)::float ELSE NULL END,
            CASE WHEN p.original_price IS NOT NULL AND p.original_price > p.price THEN true ELSE false END,
            CASE WHEN p.created_at >= NOW() - (new_arrivals_days || ' days')::interval THEN true ELSE false END,
            p.created_at
        FROM products p
        LEFT JOIN product_attributes pa ON p.id = pa.sku_id
        WHERE p.in_stock = true
          AND p.price > 0
          -- Standard filters (identical)
          AND (filter_gender IS NULL OR filter_gender = ANY(p.gender))
          AND (filter_categories IS NULL OR p.broad_category = ANY(filter_categories) OR p.category = ANY(filter_categories))
          AND (min_price IS NULL OR p.price >= min_price)
          AND (max_price IS NULL OR p.price <= max_price)
          AND (exclude_colors IS NULL OR NOT (p.colors && exclude_colors))
          AND (exclude_materials IS NULL OR NOT (p.materials && exclude_materials))
          AND (exclude_brands IS NULL OR p.brand != ALL(exclude_brands))
          AND NOT has_excluded_style(p.computed_style_scores, exclude_styles, style_threshold)
          AND matches_occasion(p.computed_occasion_scores, include_occasions, occasion_threshold)
          AND matches_article_type(p.article_type, include_article_types)
          AND matches_pattern(p.computed_pattern_scores, include_patterns, pattern_threshold)
          AND NOT has_excluded_pattern(p.computed_pattern_scores, exclude_patterns, pattern_threshold)
          AND (NOT on_sale_only OR (p.original_price IS NOT NULL AND p.original_price > p.price AND p.price > 0))
          AND (NOT new_arrivals_only OR p.created_at >= NOW() - (new_arrivals_days || ' days')::interval)
          AND (exclude_product_ids IS NULL OR NOT (p.id = ANY(exclude_product_ids)))
          AND (include_materials IS NULL OR (p.materials && include_materials))
          -- Attribute filters via product_attributes
          AND (attr_include_formality IS NULL OR pa.formality = ANY(attr_include_formality))
          AND (attr_exclude_formality IS NULL OR pa.formality IS NULL OR pa.formality != ALL(attr_exclude_formality))
          AND (attr_include_color_family IS NULL OR pa.color_family = ANY(attr_include_color_family))
          AND (attr_exclude_color_family IS NULL OR pa.color_family IS NULL OR pa.color_family != ALL(attr_exclude_color_family))
          AND (attr_include_silhouette IS NULL OR pa.silhouette = ANY(attr_include_silhouette))
          AND (attr_exclude_silhouette IS NULL OR pa.silhouette IS NULL OR pa.silhouette != ALL(attr_exclude_silhouette))
          AND (attr_include_fit_type IS NULL OR pa.fit_type = ANY(attr_include_fit_type))
          AND (attr_exclude_fit_type IS NULL OR pa.fit_type IS NULL OR pa.fit_type != ALL(attr_exclude_fit_type))
          AND (attr_include_coverage IS NULL OR pa.coverage_level = ANY(attr_include_coverage))
          AND (attr_exclude_coverage IS NULL OR pa.coverage_level IS NULL OR pa.coverage_level != ALL(attr_exclude_coverage))
          AND (attr_include_pattern IS NULL OR pa.pattern = ANY(attr_include_pattern))
          AND (attr_exclude_pattern IS NULL OR pa.pattern IS NULL OR pa.pattern != ALL(attr_exclude_pattern))
          AND (attr_include_seasons IS NULL OR pa.seasons && attr_include_seasons)
          AND (attr_exclude_seasons IS NULL OR pa.seasons IS NULL OR NOT (pa.seasons && attr_exclude_seasons))
          AND (attr_include_style_tags IS NULL OR pa.style_tags && attr_include_style_tags)
          AND (attr_exclude_style_tags IS NULL OR pa.style_tags IS NULL OR NOT (pa.style_tags && attr_exclude_style_tags))
          AND (attr_include_occasions IS NULL OR pa.occasions && attr_include_occasions)
          AND (attr_exclude_occasions IS NULL OR pa.occasions IS NULL OR NOT (pa.occasions && attr_exclude_occasions))
          AND (attr_include_neckline IS NULL OR (pa.construction->>'neckline') = ANY(attr_include_neckline))
          AND (attr_exclude_neckline IS NULL OR (pa.construction->>'neckline') IS NULL OR (pa.construction->>'neckline') != ALL(attr_exclude_neckline))
          AND (attr_include_sleeve_type IS NULL OR (pa.construction->>'sleeve_type') = ANY(attr_include_sleeve_type))
          AND (attr_exclude_sleeve_type IS NULL OR (pa.construction->>'sleeve_type') IS NULL OR (pa.construction->>'sleeve_type') != ALL(attr_exclude_sleeve_type))
          AND (attr_include_length IS NULL OR (pa.construction->>'length') = ANY(attr_include_length))
          AND (attr_exclude_length IS NULL OR (pa.construction->>'length') IS NULL OR (pa.construction->>'length') != ALL(attr_exclude_length))
          -- Keyset cursor for price_desc: (price, id) < (cursor, cursor_id)
          AND (
              cursor_value IS NULL
              OR p.price < cursor_value
              OR (p.price = cursor_value AND p.id < cursor_id)
          )
        ORDER BY p.price DESC, p.id DESC
        LIMIT p_limit;

    ELSE
        RAISE EXCEPTION 'Invalid sort_mode: %. Must be price_asc or price_desc.', p_sort_mode;
    END IF;
END;
$$;

GRANT EXECUTE ON FUNCTION get_feed_sorted_keyset TO anon, authenticated;

-- Create indexes to support price-sorted queries with in_stock filter
CREATE INDEX IF NOT EXISTS idx_products_price_asc_in_stock
    ON products (price ASC, id ASC)
    WHERE in_stock = true AND price > 0;

CREATE INDEX IF NOT EXISTS idx_products_price_desc_in_stock
    ON products (price DESC, id DESC)
    WHERE in_stock = true AND price > 0;
