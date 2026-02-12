-- =====================================================
-- Replace pre-filter architecture with direct JOIN to product_attributes
--
-- Problem: The pre-filter approach queries product_attributes separately,
-- passes IDs to the main RPC, but is capped at 1000 rows by PostgREST.
-- Attribute filters are also applied redundantly in Python.
--
-- Solution: JOIN product_attributes directly in the exploration keyset
-- functions. All attribute filtering happens in SQL — no pre-filter step,
-- no Python attribute filtering, no 1000-row cap.
--
-- New params: ~24 attribute filter parameters (include/exclude for 12 dimensions)
-- Removed: include_product_ids (no longer needed)
-- =====================================================

-- =============================================================
-- get_exploration_keyset: JOIN product_attributes for attribute filtering
-- =============================================================

-- Drop old signature (with include_product_ids as last param)
DROP FUNCTION IF EXISTS get_exploration_keyset(text, text[], text[], text[], text[], numeric, numeric, text, float, uuid, int, text[], text[], text[], float, float, text[], text[], float, boolean, boolean, int, uuid[], text[], uuid[]);

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
DECLARE
    effective_seed text;
BEGIN
    effective_seed := COALESCE(random_seed, md5(now()::text));
    RETURN QUERY
    SELECT
        p.id as product_id, p.name, p.brand, p.category, p.broad_category, p.article_type,
        p.colors, p.materials, p.price, p.fit, p.length, p.sleeve, p.neckline, p.rise, p.style_tags,
        p.primary_image_url, p.hero_image_url, p.gallery_images,
        CASE WHEN new_arrivals_only THEN
            EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
        ELSE
            (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
        END as exploration_score,
        CASE WHEN new_arrivals_only THEN
            EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
        ELSE
            (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
        END as similarity,
        COALESCE(p.computed_occasion_scores, '{}'::jsonb) as computed_occasion_scores,
        COALESCE(p.computed_style_scores, '{}'::jsonb) as computed_style_scores,
        COALESCE(p.computed_pattern_scores, '{}'::jsonb) as computed_pattern_scores,
        p.original_price,
        CASE WHEN p.original_price IS NOT NULL AND p.original_price > 0 AND p.price > 0 AND p.original_price > p.price
            THEN ROUND((1 - p.price / p.original_price) * 100)::float ELSE NULL END as discount_percent,
        CASE WHEN p.original_price IS NOT NULL AND p.original_price > p.price THEN true ELSE false END as is_on_sale,
        CASE WHEN p.created_at >= NOW() - (new_arrivals_days || ' days')::interval THEN true ELSE false END as is_new,
        p.created_at
    FROM products p
    LEFT JOIN product_attributes pa ON p.id = pa.sku_id
    WHERE p.in_stock = true
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
      -- Attribute filters via product_attributes JOIN
      -- Single-value columns: include = ANY match, exclude = no match (NULLs pass exclude)
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
      -- Multi-value array columns: include = overlap, exclude = no overlap (NULLs pass exclude)
      AND (attr_include_seasons IS NULL OR pa.seasons && attr_include_seasons)
      AND (attr_exclude_seasons IS NULL OR pa.seasons IS NULL OR NOT (pa.seasons && attr_exclude_seasons))
      AND (attr_include_style_tags IS NULL OR pa.style_tags && attr_include_style_tags)
      AND (attr_exclude_style_tags IS NULL OR pa.style_tags IS NULL OR NOT (pa.style_tags && attr_exclude_style_tags))
      AND (attr_include_occasions IS NULL OR pa.occasions && attr_include_occasions)
      AND (attr_exclude_occasions IS NULL OR pa.occasions IS NULL OR NOT (pa.occasions && attr_exclude_occasions))
      -- JSONB construction fields
      AND (attr_include_neckline IS NULL OR (pa.construction->>'neckline') = ANY(attr_include_neckline))
      AND (attr_exclude_neckline IS NULL OR (pa.construction->>'neckline') IS NULL OR (pa.construction->>'neckline') != ALL(attr_exclude_neckline))
      AND (attr_include_sleeve_type IS NULL OR (pa.construction->>'sleeve_type') = ANY(attr_include_sleeve_type))
      AND (attr_exclude_sleeve_type IS NULL OR (pa.construction->>'sleeve_type') IS NULL OR (pa.construction->>'sleeve_type') != ALL(attr_exclude_sleeve_type))
      AND (attr_include_length IS NULL OR (pa.construction->>'length') = ANY(attr_include_length))
      AND (attr_exclude_length IS NULL OR (pa.construction->>'length') IS NULL OR (pa.construction->>'length') != ALL(attr_exclude_length))
      AND (
          cursor_score IS NULL
          OR (CASE WHEN new_arrivals_only THEN
                  EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
              ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
              END) < cursor_score
          OR ((CASE WHEN new_arrivals_only THEN
                  EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
              ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
              END) = cursor_score AND p.id < cursor_id)
      )
    ORDER BY (CASE WHEN new_arrivals_only THEN
        EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
    ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
    END) DESC, p.id DESC
    LIMIT p_limit;
END;
$$;

GRANT EXECUTE ON FUNCTION get_exploration_keyset TO anon, authenticated;


-- =============================================================
-- get_exploration_keyset_with_brands: JOIN product_attributes
-- =============================================================

-- Drop old signature
DROP FUNCTION IF EXISTS get_exploration_keyset_with_brands(text, text[], text[], text[], text[], numeric, numeric, text, float, uuid, int, text[], text[], text[], float, float, text[], text[], float, text[], boolean, boolean, int, uuid[], text[], uuid[]);

CREATE OR REPLACE FUNCTION get_exploration_keyset_with_brands(
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
    occasion_threshold float DEFAULT 0.18,
    include_patterns text[] DEFAULT NULL,
    exclude_patterns text[] DEFAULT NULL,
    pattern_threshold float DEFAULT 0.30,
    preferred_brands text[] DEFAULT NULL,
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
    exploration_score float, similarity float, is_preferred_brand boolean,
    computed_occasion_scores jsonb, computed_style_scores jsonb, computed_pattern_scores jsonb,
    original_price numeric, discount_percent float, is_on_sale boolean, is_new boolean, created_at timestamptz
)
LANGUAGE plpgsql AS $$
DECLARE
    effective_seed text;
BEGIN
    effective_seed := COALESCE(random_seed, md5(now()::text));
    RETURN QUERY
    SELECT
        p.id as product_id, p.name, p.brand, p.category, p.broad_category, p.article_type,
        p.colors, p.materials, p.price, p.fit, p.length, p.sleeve, p.neckline, p.rise, p.style_tags,
        p.primary_image_url, p.hero_image_url, p.gallery_images,
        CASE WHEN new_arrivals_only THEN
            CASE WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                THEN (EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')) + 0.5
                ELSE EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
            END
        ELSE
            CASE WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                THEN (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float + 0.5
                ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
            END
        END as exploration_score,
        CASE WHEN new_arrivals_only THEN
            CASE WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                THEN (EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')) + 0.5
                ELSE EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
            END
        ELSE
            CASE WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                THEN (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float + 0.5
                ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
            END
        END as similarity,
        CASE WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
            THEN true ELSE false END as is_preferred_brand,
        COALESCE(p.computed_occasion_scores, '{}'::jsonb) as computed_occasion_scores,
        COALESCE(p.computed_style_scores, '{}'::jsonb) as computed_style_scores,
        COALESCE(p.computed_pattern_scores, '{}'::jsonb) as computed_pattern_scores,
        p.original_price,
        CASE WHEN p.original_price IS NOT NULL AND p.original_price > 0 AND p.price > 0 AND p.original_price > p.price
            THEN ROUND((1 - p.price / p.original_price) * 100)::float ELSE NULL END as discount_percent,
        CASE WHEN p.original_price IS NOT NULL AND p.original_price > p.price THEN true ELSE false END as is_on_sale,
        CASE WHEN p.created_at >= NOW() - (new_arrivals_days || ' days')::interval THEN true ELSE false END as is_new,
        p.created_at
    FROM products p
    LEFT JOIN product_attributes pa ON p.id = pa.sku_id
    WHERE p.in_stock = true
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
      -- Attribute filters via product_attributes JOIN
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
      AND (
          cursor_score IS NULL
          OR (CASE WHEN new_arrivals_only THEN
              CASE WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                  THEN (EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')) + 0.5
                  ELSE EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
              END
          ELSE
              CASE WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                  THEN (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float + 0.5
                  ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
              END
          END) < cursor_score
          OR ((CASE WHEN new_arrivals_only THEN
              CASE WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                  THEN (EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')) + 0.5
                  ELSE EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
              END
          ELSE
              CASE WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
                  THEN (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float + 0.5
                  ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
              END
          END) = cursor_score AND p.id < cursor_id)
      )
    ORDER BY (CASE WHEN new_arrivals_only THEN
        CASE WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
            THEN (EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')) + 0.5
            ELSE EXTRACT(EPOCH FROM (p.created_at - (NOW() - INTERVAL '30 days'))) / EXTRACT(EPOCH FROM INTERVAL '30 days')
        END
    ELSE
        CASE WHEN preferred_brands IS NOT NULL AND LOWER(p.brand) = ANY(SELECT LOWER(unnest(preferred_brands)))
            THEN (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float + 0.5
            ELSE (('x' || substr(md5(p.id::text || effective_seed), 1, 8))::bit(32)::bigint / 4294967295.0)::float
        END
    END) DESC, p.id DESC
    LIMIT p_limit;
END;
$$;

GRANT EXECUTE ON FUNCTION get_exploration_keyset_with_brands TO anon, authenticated;

SELECT 'Migration 044: Replaced pre-filter with direct JOIN to product_attributes for attribute filtering' as status;
NOTIFY pgrst, 'reload schema';
