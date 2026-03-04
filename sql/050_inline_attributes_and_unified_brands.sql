-- =====================================================
-- Performance: Inline product_attributes in SQL RETURN + merge brand logic
--
-- Problems solved:
-- 1. _enrich_with_attributes() makes 1-3 extra HTTP round-trips to Supabase
--    after the main RPC, adding ~300-800ms per request. The data is already
--    JOINed for filtering but not returned.
-- 2. hybrid_brand_fetch uses TWO sequential RPCs (brand-boosted + general),
--    doubling the main query latency (~400-700ms wasted).
--
-- Solution:
-- 1. Add pa.* columns to the RETURNS TABLE of both RPCs so Python can skip
--    _enrich_with_attributes() entirely.
-- 2. Merge get_exploration_keyset_with_brands into get_exploration_keyset
--    with an optional preferred_brands param. One RPC handles both cases.
-- 3. Apply same changes to get_feed_sorted_keyset.
--
-- Net effect: Feed request drops from 5-8 Supabase HTTP calls to 2-3.
-- =====================================================

-- =============================================================
-- UNIFIED get_exploration_keyset (with optional preferred_brands)
-- =============================================================

-- Drop ALL overloads of get_exploration_keyset (regardless of param signature)
DO $$
DECLARE r RECORD;
BEGIN
    FOR r IN
        SELECT oid::regprocedure::text AS sig
        FROM pg_proc
        WHERE proname = 'get_exploration_keyset'
          AND pronamespace = 'public'::regnamespace
    LOOP
        EXECUTE 'DROP FUNCTION ' || r.sig;
    END LOOP;
END $$;

-- Drop ALL overloads of get_exploration_keyset_with_brands (merged into unified function)
DO $$
DECLARE r RECORD;
BEGIN
    FOR r IN
        SELECT oid::regprocedure::text AS sig
        FROM pg_proc
        WHERE proname = 'get_exploration_keyset_with_brands'
          AND pronamespace = 'public'::regnamespace
    LOOP
        EXECUTE 'DROP FUNCTION ' || r.sig;
    END LOOP;
END $$;

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
    -- NEW: optional preferred_brands (merges with_brands variant)
    preferred_brands text[] DEFAULT NULL,
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
    original_price numeric, discount_percent float, is_on_sale boolean, is_new boolean, created_at timestamptz,
    -- NEW: inlined product_attributes columns (eliminates _enrich_with_attributes)
    pa_occasions text[], pa_style_tags text[], pa_pattern text, pa_formality text,
    pa_fit_type text, pa_color_family text, pa_seasons text[], pa_silhouette text,
    pa_construction jsonb, pa_coverage_level text, pa_skin_exposure text,
    pa_coverage_details jsonb, pa_model_body_type text, pa_model_size_estimate text
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
        -- exploration_score with optional brand boost
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
        -- similarity (same as exploration_score)
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
        -- is_preferred_brand flag
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
        p.created_at,
        -- Inlined product_attributes columns
        pa.occasions as pa_occasions,
        pa.style_tags as pa_style_tags,
        pa.pattern as pa_pattern,
        pa.formality as pa_formality,
        pa.fit_type as pa_fit_type,
        pa.color_family as pa_color_family,
        pa.seasons as pa_seasons,
        pa.silhouette as pa_silhouette,
        pa.construction as pa_construction,
        pa.coverage_level as pa_coverage_level,
        pa.skin_exposure as pa_skin_exposure,
        pa.coverage_details as pa_coverage_details,
        pa.model_body_type as pa_model_body_type,
        pa.model_size_estimate as pa_model_size_estimate
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

GRANT EXECUTE ON FUNCTION get_exploration_keyset TO anon, authenticated;


-- =============================================================
-- UPDATED get_feed_sorted_keyset (with inlined attributes)
-- =============================================================

-- Drop ALL overloads of get_feed_sorted_keyset (regardless of param signature)
DO $$
DECLARE r RECORD;
BEGIN
    FOR r IN
        SELECT oid::regprocedure::text AS sig
        FROM pg_proc
        WHERE proname = 'get_feed_sorted_keyset'
          AND pronamespace = 'public'::regnamespace
    LOOP
        EXECUTE 'DROP FUNCTION ' || r.sig;
    END LOOP;
END $$;

CREATE OR REPLACE FUNCTION get_feed_sorted_keyset(
    p_sort_mode text DEFAULT 'price_asc',
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
    original_price numeric, discount_percent float, is_on_sale boolean, is_new boolean, created_at timestamptz,
    -- NEW: inlined product_attributes columns
    pa_occasions text[], pa_style_tags text[], pa_pattern text, pa_formality text,
    pa_fit_type text, pa_color_family text, pa_seasons text[], pa_silhouette text,
    pa_construction jsonb, pa_coverage_level text, pa_skin_exposure text,
    pa_coverage_details jsonb, pa_model_body_type text, pa_model_size_estimate text
)
LANGUAGE plpgsql AS $$
BEGIN
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
            p.created_at,
            -- Inlined product_attributes
            pa.occasions, pa.style_tags, pa.pattern, pa.formality,
            pa.fit_type, pa.color_family, pa.seasons, pa.silhouette,
            pa.construction, pa.coverage_level, pa.skin_exposure,
            pa.coverage_details, pa.model_body_type, pa.model_size_estimate
        FROM products p
        LEFT JOIN product_attributes pa ON p.id = pa.sku_id
        WHERE p.in_stock = true
          AND p.price > 0
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
          AND (NOT on_sale_only OR (p.original_price IS NOT NULL AND p.original_price > p.price))
          AND (NOT new_arrivals_only OR p.created_at >= NOW() - (new_arrivals_days || ' days')::interval)
          AND (exclude_product_ids IS NULL OR NOT (p.id = ANY(exclude_product_ids)))
          AND (include_materials IS NULL OR (p.materials && include_materials))
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
              cursor_value IS NULL
              OR p.price > cursor_value
              OR (p.price = cursor_value AND p.id > cursor_id)
          )
        ORDER BY p.price ASC, p.id ASC
        LIMIT p_limit;

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
            p.created_at,
            -- Inlined product_attributes
            pa.occasions, pa.style_tags, pa.pattern, pa.formality,
            pa.fit_type, pa.color_family, pa.seasons, pa.silhouette,
            pa.construction, pa.coverage_level, pa.skin_exposure,
            pa.coverage_details, pa.model_body_type, pa.model_size_estimate
        FROM products p
        LEFT JOIN product_attributes pa ON p.id = pa.sku_id
        WHERE p.in_stock = true
          AND p.price > 0
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
          AND (NOT on_sale_only OR (p.original_price IS NOT NULL AND p.original_price > p.price))
          AND (NOT new_arrivals_only OR p.created_at >= NOW() - (new_arrivals_days || ' days')::interval)
          AND (exclude_product_ids IS NULL OR NOT (p.id = ANY(exclude_product_ids)))
          AND (include_materials IS NULL OR (p.materials && include_materials))
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
              cursor_value IS NULL
              OR p.price < cursor_value
              OR (p.price = cursor_value AND p.id < cursor_id)
          )
        ORDER BY p.price DESC, p.id DESC
        LIMIT p_limit;

    ELSE
        RAISE EXCEPTION 'Invalid sort_mode: %. Use price_asc or price_desc.', p_sort_mode;
    END IF;
END;
$$;

GRANT EXECUTE ON FUNCTION get_feed_sorted_keyset TO anon, authenticated;

SELECT 'Migration 050: Inlined product_attributes in SQL RETURN + unified brand logic in get_exploration_keyset' as status;
NOTIFY pgrst, 'reload schema';
